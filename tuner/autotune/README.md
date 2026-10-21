# autotune: fitting the MRC tuner cost model to nightly configurations

`autotune` refits the tuner plugin's **cost model** so that, with
`NCCL_TUNER_PLUGIN=libnccl-tuner-mrc.so`, every nightly AllReduce, AllGather
and ReduceScatter configuration picks the best measured (algorithm, protocol)
at every message size. No CSV rules are generated. Nightly tuner runs set
`NCCL_TUNER_FORCE_COST_MODEL=1`, so the cost model alone drives the choice.

The parameters it fits live in the **`rail_nop2p`** profile of
[`../plugin.c`](../plugin.c), between the `BEGIN/END AUTOTUNE PARAMS` markers.
This profile covers communicators with MNNVL disabled (each physical node is
its own NCCL node) and `NCCL_P2P_DISABLE=1`. When `NCCL_SHM_DISABLE=1` is also
set, as in the nightly, there is no intra-node transport and NCCL treats every
rank as its own node (`localRanks 1`, `nNodes == nRanks`); `topology.py` maps
those runs the same way. The GB200F `mnnvl` profile is not
touched.

Only AllReduce, AllGather and ReduceScatter are swept, fitted and reported.
Alltoall and sendrecv are out of scope: they are grouped point-to-point
operations that never go through the tuner. The nightly runs for them are
ingested but skipped by `report`/`dips`, and `--colls` rejects them.

```
nightly report ──► report (dips)
template run ───► sweep scripts ──► [cluster] ──► ingest ──► oracle ──► fit ──► codegen (+build, parity)
                                                                                    │
                       A/B scripts ◄── validate ab-scripts ◄── validate offline ◄───┘
                            │
                       [cluster] ──► ingest ──► validate compare (gates)
```

## Setup

```bash
cd tuner/autotune
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
. .venv/bin/activate
python -m autotune -h
python -m pytest -q          # or: make -C ../.. test-autotune
```

The tools only read and write local files. They never launch jobs: `sweep` and
`validate ab-scripts` write shell scripts, and you run them on the cluster.

## Workflow

### 1. Nightly dip report

```bash
python -m autotune report nightly-run-logs.txt --out report/
```

This parses the nightly report: the `RUN n:` blocks, the captured `mpirun`
line and env, and the selected cycle. Each run is mapped to the communicator
NCCL builds, and the report lists **notches**: points lower than both
neighbours by more than 10%, for sizes of at least 1 MiB. On the 2026-10-15
report it finds:
- AR 256 MB at split 0x1, R=0 and R=1;
- AR 128 MB with P2P enabled.

Once an oracle exists (step 3), pass `--oracle oracle.json` to also flag
**envelope** gaps, i.e. sizes below the best measured cell. That catches
plateaus such as AR 4M–32M.

### 2. Sweep: measure every valid cell

```bash
python -m autotune sweep --template nightly-run-logs.txt --out sweep/ \
    --scales 8,32,128 --smoke-scales 8 --hostfile-source /path/to/all-hosts
sweep/run_all.sh smoke-8n          # validate scripts and parsers first
sweep/run_all.sh 8n 32n 128n       # resumable: finished runs are skipped
```

Each script reuses a nightly command verbatim: launcher options, numa binding
wrapper and every `-x`. The only changes are `-np`, the hostfile, the
benchmark binary and arguments, and the managed variables (`NCCL_TUNER_*`,
`NCCL_ALGO`/`NCCL_PROTO`, `NCCL_NVLS_ENABLE`, `NCCL_TESTS_SPLIT_MASK`,
`NCCL_DEBUG*`).

For each (collective, split mask, R) the sweep has three kinds of run:
- **probe**: tuner off, `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING`,
  `-w 1 -n 1 -N 1`. It records NCCL's default cell per size, its channel range
  and NCCL's init latency/bandwidth table. It is a separate run because
  per-call logging would perturb the timed runs.
- **default**: tuner off; NCCL chooses.
- **forced**: one run per valid cell via `NCCL_ALGO`/`NCCL_PROTO`, tuner off,
  `-N 3`.

Cells are limited to what NCCL 2.30 can run (CollNet excluded: no SHARP on
MRC):

| Algo | AllReduce | AllGather / ReduceScatter |
|---|---|---|
| ring | LL, LL128, SIMPLE | LL, LL128, SIMPLE |
| tree | LL, LL128, SIMPLE | – |
| nvls_tree | SIMPLE, only with ≥ 2 local ranks, in an `NCCL_NVLS_ENABLE=2` variant (probe + default + forced) | – |
| pat | – (not supported for AR) | SIMPLE, one rank per NCCL node only |
| nvls | – (needs CollNet across nodes) | – |

With 2 local ranks for split 0x1 (P2P disabled, SHM enabled), this gives
AR 11 (0x1) / 8 (0x3) runs per R, and AG/RS 5 / 6, so 82 runs per scale. With
the nightly env (P2P and SHM disabled) every split has one rank per NCCL node:
AR 8 / 8 and AG/RS 6 / 6, so 80 runs per scale, with no nvls_tree variant.
`--hostfile-pattern '/share/hosts-{nodes}'` uses existing
hostfiles instead of taking the first N unique hosts from
`--hostfile-source`.

### 3. Ingest and build the oracle

```bash
python -m autotune ingest --manifest sweep/manifest.json --db autotune.db
python -m autotune oracle --db autotune.db --out oracle.json --csv oracle.csv
```

**Cycles.** nccl-tests prints the `-N` cycles back to back. A new cycle starts
whenever the size column does not increase. **Only the last cycle is used.**
Earlier cycles can be skewed by dead BT0–BT1 links. A run with fewer than N
complete cycles, correctness errors, or a forced cell that NCCL did not honour
(checked against its TUNING lines when present) is excluded.

**Oracle.** For each configuration (profile, collective, nNodes, nRanks, R,
NVLS) and size, the oracle is the cell with the best last-cycle out-of-place
busbw. Cells within the noise band `tol` of it are *equivalent* winners.

**Noise** comes from duplicates the sweep already contains: the default run
vs the forced run of the probe's default cell, and AR R=0 vs R=1 of the same
cell. From these, `sigma = 1.4826·median|d|/√2` and
`tol = max(3%, 2·sigma)`, per nNodes.

### 4. Fit

```bash
python -m autotune fit --db autotune.db --oracle oracle.json --out fit/
```

For a fixed topology and cell, the plugin model is affine in message size:
`T(s) = L + s / (1000·algoBw)`. The latency `L` is linear in the latency
parameters (base, intra-hop and inter-hop, with hop counts from
`algoLatency`). On rail topologies each bandwidth cell is set by one cap
parameter. That gives the stages:

1. **Per-cell decomposition.** A robust fit of `T = L + k·s` per (topology,
   cell). Points far off the line are tagged as *anomalies* and excluded from
   the model fit; examples are single-size notches that no affine model can
   represent.
2. **Closed-form start.** Latency parameters by bounded linear least squares
   over all (topology, cell) `L` values, using 3 scales × 2 split masks so
   base, intra-hop and inter-hop terms are identifiable. Bandwidth parameters
   by inverting each cell's cap chain.
3. **Joint refinement.** `scipy.optimize.least_squares` on log(time),
   soft_l1 loss, with bounds and a weak pull toward stage 2.
4. **Selection polish.** Minimises a smooth surrogate of the *regret*: the
   busbw lost when the model's pick (argmin over the cells measured for that
   configuration) is not oracle-equivalent. The nightly scale is weighted ×3,
   and the log-time loss may get at most 5% worse.
5. **Cross-scale validation.** Leave one node count out, fit the rest,
   evaluate the held-out scale. Train 8+32 → predict 128 is the extrapolation
   test.
6. **Final fit and gate.** At the nightly scale (default: the largest nNodes;
   `--gate-nodes`) and with production NVLS (`--gate-nvls 0`), every
   (configuration, size) pick must be oracle-equivalent. Violations explained
   by anomaly-tagged points are listed as *exempt* instead.

**regBuff.** If a cell's bandwidth differs between R=0 and R≥1 by more than
the noise band, `regBwScale[algo][proto]` (the busBw multiplier the plugin
applies when NCCL reports `regBuff=1`) is fitted too. Control this with
`--regbuff auto|on|off`.

The fit writes:
- `fit/fit_result.json`: parameters and metrics;
- `fit/fit_report.md`: regret of NCCL's default vs seed vs fitted, CV table,
  gate, parameter table, anomalies;
- `fit/picks.csv`: the model's pick per (configuration, size).

### 5. Codegen, build, C/Python parity

```bash
python -m autotune codegen --fit fit/fit_result.json --parity
```

This rewrites only the `rail_nop2p` block in `plugin.c`, with provenance
comments. `railNoP2pParamsValidated` is set to 1 only when the gate passed
(`--force-validated` / `--not-validated` override that). It then rebuilds
`libnccl-tuner-mrc.so` and the test drivers, and checks every cell of the
compiled `cost_model_main --all-cells` against `costmodel.py`.

While `railNoP2pParamsValidated` is 0, the plugin uses the profile only when
`NCCL_TUNER_COST_MODEL_PROFILE=rail_nop2p` is set. Once it is 1, the profile
is also auto-selected for jobs with `NCCL_MNNVL_ENABLE=0` and
`NCCL_P2P_DISABLE=1` on more than one node.

### 6. Validate

```bash
# Offline: the compiled model's picks vs the oracle.
python -m autotune validate offline --db autotune.db --oracle oracle.json --out validate/

# Online A/B on the cluster (A = tuner off, B = MRC tuner + FORCE_COST_MODEL, alternating),
# plus one B probe per configuration that records the cells NCCL actually ran.
python -m autotune validate ab-scripts --template nightly-run-logs.txt --out ab/ \
    --scales 128,32 --tuner-lib /opt/microsoft/mrc/mrc-nccl-plugin/libnccl-tuner-mrc.so \
    --hostfile-source /path/to/all-hosts
ab/run_all.sh
python -m autotune ingest --manifest ab/manifest.json --db autotune.db
python -m autotune validate compare --db autotune.db --oracle oracle.json --out validate/
```

The A/B gates, per configuration:
- B is within the oracle noise band at every size;
- B is never more than 3% below A;
- every A notch is resolved in B;
- the cells NCCL used under the tuner (B probe) are oracle-equivalent.

### Dry run without cluster time

```bash
python -m autotune simulate --db sim.db --notch     # synthetic sweep from a perturbed "truth"
python -m autotune oracle --db sim.db --out sim-oracle.json
python -m autotune fit --db sim.db --oracle sim-oracle.json --out sim-fit/
```

## Nightly integration

- Add a tuner-enabled variant: `NCCL_TUNER_PLUGIN=libnccl-tuner-mrc.so`,
  `NCCL_TUNER_FORCE_COST_MODEL=1`, plus one probe per configuration with
  `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING`. Keep `-N 3`; the last cycle
  is used.
- Add ReduceScatter to the nightly.
- Run `python -m autotune report <nightly> --oracle oracle.json` on each
  nightly. A configuration that drifts outside the oracle band is the trigger
  to re-sweep that scale and refit.

## Modules

| Module | Purpose |
|---|---|
| `params.py` | `CostModelParams` schema (field order of the C struct); parse/render the C initializers |
| `costmodel.py` | Python mirror of the plugin cost model (parity-tested against C) |
| `topology.py` | (physical nodes, ranks per node, split, MNNVL, P2P) → NCCL (nNodes, nRanks, ppn, profile) |
| `ingest.py` / `db.py` | parsers (nightly report, raw logs, TUNING lines, NCCL tuning table) and the SQLite store |
| `sweep.py` | probe/default/forced and A/B script generation from a nightly template |
| `oracle.py` | best cell per (configuration, size), noise band |
| `dips.py` | notch / envelope report and plots |
| `fit.py` | stages 1–6 |
| `codegen.py` | write the block, build, C/Python parity |
| `validate.py` | offline check through the compiled model; A/B gates |
| `synth.py` | synthetic sweep data (tests, `simulate`) |
