# MRC NCCL Tuner Plugin

A tuner plugin for NCCL that customizes algorithm and protocol selection using
two complementary mechanisms:

1. **CSV configuration model** — static, explicit rules read from
   `nccl_tuner.conf`. You pin the `(algorithm, protocol, channels)` for a given
   collective, message-size range, and topology.
2. **Cost model** — an analytical performance model calibrated for the GB200F
   platform. When no CSV rule matches, the plugin predicts the runtime of every
   viable `(algorithm, protocol)` cell and lets NCCL pick the fastest.

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
  AllGather, and ReduceScatter. Set `NCCL_TUNER_COST_MODEL=0` for strict
  CSV-only behavior.
- Set `NCCL_TUNER_FORCE_COST_MODEL=1` to skip the CSV lookup entirely and drive
  selection purely from the cost model.

## Building

```bash
make
```

This produces `libnccl-tuner-mrc.so`.

## Usage

1. Set `NCCL_TUNER_PLUGIN` to the path of the built shared library:

   ```bash
   export NCCL_TUNER_PLUGIN=/path/to/libnccl-tuner-mrc.so
   ```

2. Optionally specify a custom config file (defaults to `nccl_tuner.conf`):

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
| collective_type | `broadcast`, `reduce`, `allgather`, `reducescatter`, `allreduce` |
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
and only `nRanks` distinguishes those entries. See the header of
`nccl_tuner.conf` for the full table of shipped defaults.

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
high-bandwidth inter-domain path.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `NCCL_TUNER_COST_MODEL` | `1` | `1`=cost model on (fallback for AR/AG/RS); `0`=strict CSV-only |
| `NCCL_TUNER_FORCE_COST_MODEL` | `0` | `1`=skip CSV lookup and use the cost model directly (implies the model is on) |
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

# Cost model with a different inter-domain bandwidth assumption.
export NCCL_TUNER_COST_MODEL_BW_INTER=30.0
```

---

## Testing and Calibration

The `test/` directory contains standalone host-side drivers that include
`plugin.c` directly, so they build without CUDA or NCCL.

Build the drivers:

```bash
make test
```

| Tool | Purpose |
|------|---------|
| `test/csv_tuning_main` | Exercises the CSV path — prints the pinned `(algo, proto, channels)` per point, or `(no match)`. Pass a config path as the first argument (defaults to `nccl_tuner.conf`). |
| `test/cost_model_main` | Runs the cost model across a topology/size sweep and prints the winning `(algo, proto, time_us)`. |


Example:

```bash
make test
./test/csv_tuning_main nccl_tuner.conf   # inspect CSV rule matches
./test/cost_model_main                   # inspect cost-model winners
```

## Adding Configurations

Add tuner entries to `nccl_tuner.conf` (or a file pointed to by
`NCCL_TUNER_CONFIG_FILE`). Lines starting with `#` are comments, and the first
matching rule wins. Use the test drivers above to confirm your rules resolve as
intended before deploying.
