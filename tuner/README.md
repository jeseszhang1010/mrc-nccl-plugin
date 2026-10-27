# MRC NCCL Tuner Plugin

A tuner plugin for NCCL that customizes algorithm and protocol selection using
two complementary mechanisms:

1. **CSV configuration model** — static, explicit rules read from
   `nccl_tuner.conf`. You pin the `(algorithm, protocol, channels)` for a given
   collective, message-size range, and topology.
2. **Cost model** — an analytical performance model for the GB200F platform.
   When no CSV rule matches, the plugin predicts the runtime of every viable
   `(algorithm, protocol)` cell and lets NCCL pick the fastest. Only the
   `rail_nop2p` profile is validated and enabled by default (see
   [Profiles](#profiles)).

Based on the [NCCL ext-tuner example plugin](https://github.com/NVIDIA/nccl/tree/master/ext-tuner/example).

## How the two models interact

For every collective call NCCL asks the plugin to fill a cost table. The plugin
resolves it in this order:

```
                 ┌──────────────────────────────────────────────┐
   getCollInfo   │ 1. CSV lookup (nccl_tuner.conf)              │
      ─────────► │    first matching rule wins → pin that cell  │
                 └───────────────┬──────────────────────────────┘
                                 │ no match
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ 2. Cost-model fallback (if enabled)          │
                 │    fill AllReduce / AllGather / ReduceScatter│
                 │    cells with predicted time; NCCL picks min │
                 └───────────────┬──────────────────────────────┘
                                 │ nothing populated / disabled
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ 3. NCCL's internal tuner (table left as-is)  │
                 └──────────────────────────────────────────────┘
```

- The CSV path has priority: a matching rule always wins over the cost model.
- The cost model is enabled by default and acts as the fallback for AllReduce,
  AllGather, and ReduceScatter, but only when a validated profile applies to
  the job (see [Profiles](#profiles)); otherwise NCCL's internal tuner is used.
  Set `NCCL_TUNER_COST_MODEL=0` for strict CSV-only behavior.
- Set `NCCL_TUNER_FORCE_COST_MODEL=1` to skip the CSV lookup entirely and drive
  selection purely from the cost model.

## Building

The tuner has no MRC, CUDA, or NCCL build dependencies. From the repository
root, build only the tuner with:

```bash
make tuner
```

This produces `libnccl-tuner-mrc.so` in the repository root. Running `make`
(or `make all`) at the repository root builds both the NET and tuner plugins.
Override the output path with `TUNER_TARGET=/path/to/libnccl-tuner-mrc.so`.

You can also build in this directory with `make -C tuner`, which produces
`tuner/libnccl-tuner-mrc.so`.

## Usage

1. Set `NCCL_TUNER_PLUGIN` to the path of the built shared library:

   ```bash
   export NCCL_TUNER_PLUGIN=/path/to/libnccl-tuner-mrc.so
   ```

2. Optionally specify a custom config file. By default the plugin reads
   `nccl_tuner.conf` from the current working directory. `tuner/nccl_tuner.conf`
   is a template with no active rules:

   ```bash
   export NCCL_TUNER_CONFIG_FILE=/path/to/my_config.conf
   ```

3. Run your NCCL workload as usual.

---

## 1. CSV Configuration Model

Edit `nccl_tuner.conf` to add tuning rules. Each non-comment line is CSV with
the format:

```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks[,numPipeOps][,regBuff]
```

### Fields

| Field | Values |
|-------|--------|
| collective_type | `broadcast`, `reduce`, `allgather`, `reducescatter`, `allreduce` (names for collective_type, algorithm and protocol are case-insensitive) |
| min_bytes / max_bytes | Inclusive byte range for this rule |
| algorithm | `tree`, `ring`, `collnet_direct`, `collnet_chain`, `nvls`, `nvls_tree`, `pat` |
| protocol | `ll`, `ll128`, `simple` |
| channels | Number of channels, or `-1` to keep NCCL's default |
| nNodes | Node count to match, or `-1` for any |
| nRanks | Rank count to match, or `-1` for any |
| numPipeOps | (optional) Pipeline ops to match, or `-1` for any |
| regBuff | (optional) Buffer registration (`0`/`1`/`-1` for any) |

### Matching semantics

- Lines starting with `#` are comments; blank lines are ignored.
- A rule matches when the collective type is equal, `nBytes` falls in
  `[min_bytes, max_bytes]`, and every specified topology field
  (`nNodes`, `nRanks`, `numPipeOps`, `regBuff`) equals the runtime value or is
  `-1`.
- **The first matching rule wins** — order rules from most specific to most
  general.
- `channels = -1` leaves NCCL's size-adaptive channel selection untouched.

### GB200F topology note

On GB200F, NCCL collapses an entire NVLink domain (up to 16 physical nodes = 64
GPUs) into a single "node". The `(nNodes, nRanks)` values in a rule reflect what
NCCL reports, not the physical node count:

```
nNodes = max(1, ceil(physical_nodes / 16))
nRanks = physical_nodes * 4        # 4 GPUs per physical node
```

When `physical_nodes <= 16` the deployment fits in one NVL domain, so `nNodes=1`
and only `nRanks` distinguishes those entries. With `NCCL_MNNVL_ENABLE=0`,
`nNodes` is the physical node count, unless `NCCL_P2P_DISABLE=1` and
`NCCL_SHM_DISABLE=1` are also set: then every GPU is its own NCCL node
(`nNodes == nRanks`).

The shipped `nccl_tuner.conf` contains no active rules; add rules measured on
your own deployment.

### Example

```
# Use ring/simple with 4 channels for large allreduce on 4 "nodes" (NVL domains)
allreduce,1048577,4294967295,ring,simple,4,4,-1

# Small allreduce that fits one NVL domain (2 physical nodes → nNodes=1, nRanks=8)
allreduce,4096,2097152,ring,ll,-1,1,8
```

---

## 2. Cost Model

When no CSV rule matches (or when forced), the plugin predicts the runtime of
each `(algorithm, protocol)` cell and lets NCCL choose the minimum. It follows
NCCL's analytical performance model:

```
time(us) = latency_us * latCount + nBytes / (1000 * algoBw_GBps)
```

For each supported `(algorithm, protocol)` the model computes:

1. `busBw  = nChannels * perChannelBw` (with protocol/algo-specific caps)
2. `algoBw = busBw * ratio` (ratio depends on collective + algorithm)
3. `latency = baseLat + per-hop intra/inter latencies`
4. `time   = latency * latCount + nBytes / (1000 * algoBw)`

All bandwidths are GB/s; all latencies are microseconds. Parameters were fit
against measured `nccl-tests` data at physical 2–256 nodes (4 GPUs per physical
node).

### Coverage

| Collective | Algorithms modeled |
|------------|--------------------|
| AllReduce | `ring`, `tree`, `nvls` (single NVL domain), `nvls_tree` |
| AllGather | `ring`, `nvls` (single NVL domain) |
| ReduceScatter | `ring`, `nvls` (single NVL domain) |

Other collectives fall through to NCCL's internal tuner. Plain `nvls` is
intra-NVL-domain only (it cannot span multiple NCCL nodes); `nvls_tree` is the
high-bandwidth inter-domain path. The `rail_nop2p` profile (below) also models
`pat` for AllGather/ReduceScatter when every node has a single rank.

### Profiles

The cost model has two parameter sets in `plugin.c`:

| Profile | Used for | Parameters | Status |
|---------|----------|------------|--------|
| `mnnvl` | GB200F with MNNVL: one NCCL node = one NVLink domain | `gb200fParamsDefault` | **Experimental** — not validated (`mnnvlParamsValidated = 0`); explicit opt-in only |
| `rail_nop2p` | `NCCL_MNNVL_ENABLE=0` and `NCCL_P2P_DISABLE=1`: all traffic, including intra-node, goes over the MRC NET; with `NCCL_SHM_DISABLE=1` NCCL reports one node per GPU (`nNodes == nRanks`) | `railNoP2pParamsDefault` (calibrated against measured nccl-tests runs at nNodes 8–256; spot-checked at nNodes 2 and 4) | Validated (`railNoP2pParamsValidated = 1`) |

`NCCL_TUNER_COST_MODEL_PROFILE=mnnvl|rail_nop2p` (case-insensitive) forces a
profile, validated or not. Unset (or `auto`), the plugin picks the profile that matches the job:
`rail_nop2p` when the job runs with `NCCL_MNNVL_ENABLE=0` and
`NCCL_P2P_DISABLE=1` across more than one node, `mnnvl` otherwise. It uses that
profile only if its `*ParamsValidated` flag in `plugin.c` is `1`. If not, the
cost model is disabled for the communicator (including
`NCCL_TUNER_FORCE_COST_MODEL=1`), and NCCL's internal tuner handles any
collective without a matching CSV rule. In this release, MNNVL jobs therefore
use NCCL's defaults unless you set `NCCL_TUNER_COST_MODEL_PROFILE=mnnvl`.

The `rail_nop2p` profile differs from `mnnvl` in three ways:
- **PAT:** it prices `pat` (AllGather/ReduceScatter, SIMPLE, one rank per
  node) with NCCL's PAT model shape.
- **regBuff:** for registered buffers (`regBuff=1`, e.g. nccl-tests `-R 1`)
  it multiplies the latency term per (algo, proto), separately for AllReduce
  (`arRegLatScale`) and AllGather (`agRegLatScale`). Registered-buffer
  AllGather ring/SIMPLE pays a roughly constant extra cost per call, which a
  bandwidth factor cannot represent; AllReduce shows no measurable penalty.
- **Cell coverage:** it sets `disableUnmodeledCells`, so any valid cell it
  cannot price gets a large explicit cost instead of keeping NCCL's own
  estimate. That way NCCL never compares plugin times with its own.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `NCCL_TUNER_COST_MODEL` | `1` | `1`=cost model on (fallback for AR/AG/RS); `0`=strict CSV-only |
| `NCCL_TUNER_FORCE_COST_MODEL` | `0` | `1`=skip CSV lookup and use the cost model directly (implies the model is on) |
| `NCCL_TUNER_COST_MODEL_PROFILE` | `auto` | `mnnvl`, `rail_nop2p` or `auto` (validated profiles only; see Profiles) |
| `NCCL_TUNER_COST_MODEL_CHANNELS` | platform default (32) | Override the default channel count |
| `NCCL_TUNER_COST_MODEL_BW_INTRA` | platform default | Per-channel intra-domain (NVLink) BW in GB/s |
| `NCCL_TUNER_COST_MODEL_BW_INTER` | platform default | Per-channel inter-domain (backend MRC/NET) BW in GB/s |
| `NCCL_TUNER_COST_MODEL_NVLS_EFF` | platform default | NVLS aggregate efficiency factor (0..1) |
| `NCCL_TUNER_COST_MODEL_GRAPH_LAT` | platform default | Per-hop inter-domain graph latency in us |

The model uses `nChannels` only to rank algorithm/protocol choices; it leaves
the actual runtime channel count at NCCL's default (matching the `channels=-1`
CSV behavior), because forcing a fixed count degrades sizes where NCCL's
size-adaptive pick is better.

### Examples

```bash
# Default: CSV rules first, cost model as fallback.
export NCCL_TUNER_PLUGIN=/path/to/libnccl-tuner-mrc.so

# Strict CSV-only (disable the cost model entirely).
export NCCL_TUNER_COST_MODEL=0

# Ignore the CSV and let the cost model drive all selections.
export NCCL_TUNER_FORCE_COST_MODEL=1

# Opt in to the experimental (unvalidated) MNNVL profile.
export NCCL_TUNER_COST_MODEL_PROFILE=mnnvl

# Cost model with a different inter-domain bandwidth assumption.
export NCCL_TUNER_COST_MODEL_BW_INTER=30.0
```

---

## Testing and Calibration

The `test/` directory contains standalone host-side drivers that include
`plugin.c` directly, so they build without CUDA or NCCL.

Build the drivers from the repository root:

```bash
make test-tuner
```

| Tool | Purpose |
|------|---------|
| `test/csv_tuning_main` | Exercises the CSV path — prints the pinned `(algo, proto, channels)` per point, or `(no match)`. Pass a config path as the first argument (defaults to `nccl_tuner.conf`). |
| `test/cost_model_main` | Runs the cost model across a topology/size sweep and prints the winning `(algo, proto, time_us)`. Options: `--profile mnnvl\|rail_nop2p`, `--topo NRANKS:NNODES` (repeatable), `--coll NAME` (repeatable), `--regbuff 0\|1`, `--min-bytes/--max-bytes N`, and `--all-cells` for one CSV line per priced cell plus the winner. |


Example:

```bash
make test-tuner
cd tuner
./test/csv_tuning_main nccl_tuner.conf   # inspect CSV rule matches
./test/cost_model_main                   # inspect cost-model winners
./test/cost_model_main --profile rail_nop2p --topo 256:128 --coll allreduce
```

## Adding Configurations

Add tuner entries to `nccl_tuner.conf` (or a file pointed to by
`NCCL_TUNER_CONFIG_FILE`). Lines starting with `#` are comments, and the first
matching rule wins. Use the test drivers above to confirm your rules resolve as
intended before deploying.
