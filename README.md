# mrc-nccl-plugin

External NCCL plugins using MRC (Multi-Path Reliable Connection) as transport:

* **NET plugin** (`libnccl-net-mrc.so`) — host-initiated networking, NCCL NET plugin ABI v12. Reports
  the network name `MRC` and requires a NCCL runtime that supports v12; the legacy C sources are not
  part of this build.
* **GIN plugin** (`libnccl-gin-mrc.so`) — GPU-initiated networking, NCCL GIN plugin ABI v14. Provides the
  host half of NCCL's `NCCL_NET_DEVICE_GIN_MRC` backend; the device half ships inside NCCL as
  `nccl_device/gin/mrc/`. It supports two receive-queue modes, CPU-managed and GPU-managed (below).

## GIN receive-queue modes: CPU RQ and GPU RQ

MRC QPs have no RDMA atomics, so every GIN signal travels as the immediate of an RDMA
Write-with-Immediate. Each one consumes a receive WQE on the target and produces a receive completion.
The two modes differ in who drains those receive queues (RQs) and applies the signals:

| | **CPU RQ** (`GIN_MRC_RQ=cpu`, default) | **GPU RQ** (`GIN_MRC_RQ=gpu`) |
|---|---|---|
| Who drains the receive CQs | A progress thread of the GIN plugin, on the host | NCCL device code, whenever a kernel reads or waits on a signal or counter |
| How signals reach GPU memory | The progress thread adds each value to the signal slot through GDRCopy | The kernel applies it directly |
| Receive CQ memory | Host | GPU |
| Device code NCCL is built with | Open-source GPUNetIO (`doca_gpu_dev_verbs_put_immediate`), from `DOCA_HOME` | DOCA SDK GPUNetIO device functions, from `DOCA_SDK_HOME` |
| Extra runtime requirement | GDRCopy (with forced-PCIe support for Data Direct) | None |
| Host CPU cost | One busy-polling thread per GIN context set (per device communicator and NIC) | None |
| Progress | Signals are applied even if no kernel is running on the receiver | Signals are applied only while some kernel on the receiver drives GIN progress |
| Defaults (`NCCL_GIN_MRC_RQ_DEPTH`, `NCCL_GIN_MRC_MIN_RNR_TIMER`) | 1024, 1 (0.01 ms) | SQ depth, 12 (0.64 ms) |
| Backend version NCCL reports to the plugin | 9 | 10 |

The mode is fixed when NCCL is built. **One plugin binary serves both modes**: NCCL reports its mode
to the plugin as the backend version, and the plugin follows it per communicator. To switch modes at
run time, point `LD_LIBRARY_PATH` (and application builds) at the NCCL build of that mode.

Which to choose: CPU RQ gave higher throughput on combine-heavy all-to-all traffic in our measurements
(see [Performance of the two modes](#performance-of-the-two-modes)), at the cost of host cores. GPU RQ
needs no host thread and no GDRCopy. Prefer it when host CPU is scarce.

## Requirements

* MRC
* CUDA
* OFED
* NCCL with NET v12 support (runtime)
* A C++17 compiler, Make, and Python 3 (build)
* For the GIN plugin only: the DOCA SDK (verbs and GPUNetIO, for the RCX QP type; `MRC_HOME`, default
  `/opt/mellanox/doca`), an NCCL build with GIN MRC (`NCCL_HOME`), and GDRCopy for the CPU-managed
  receive-queue mode

## Build

### NET plugin

```bash
make net MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda
```
Debug build
```bash
make net MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda DEBUG=1
```

To supply `nccl.h` from a custom NCCL installation, add `NCCL_HOME=/path/to/nccl/build` to the make
command. `nccl.h` will be included from `$(NCCL_HOME)/include`. A plain `make` builds both plugins
(`net` and `gin`). For a parallel build, run `make -j net` and `make -j gin` one after the other: both
share one object directory, so a top-level `make -j` can link one plugin while the other is still
rewriting a shared object. The output libraries are `libnccl-net-mrc.so` and `libnccl-gin-mrc.so`, with
`mrc` as the suffix of each plugin library.

### GIN plugin and NCCL, for CPU RQ and GPU RQ

The device half of the MRC GIN backend lives in NCCL and is opt-in (`GIN_MRC=1`). `GIN_MRC_RQ=cpu|gpu`
selects the receive-queue mode. The two modes cannot coexist in one NCCL build: the open-source and SDK
device headers share include guards and type names with different layouts, and the device code is
inlined into every kernel that uses the device API. So build NCCL **once per mode**, then build the GIN
plugin **once** for both.

**Step 1: build NCCL once per mode.** You need:

* `DOCA_HOME`: an open-source GPUNetIO checkout ([NVIDIA-DOCA/gpunetio](https://github.com/NVIDIA-DOCA/gpunetio)
  `main`) with `doca_gpu_dev_verbs_put_immediate`. It is used in both modes. Its device headers are
  installed with NCCL's, and the GIN plugin's CPU RQ code compiles against them.
* `DOCA_SDK_HOME` (GPU RQ only, default `/opt/mellanox/doca`): the DOCA SDK. NCCL copies its GPUNetIO
  device headers into `<build>/include/nccl_device/gin/mrc/doca_sdk/`.

`build-nccl-gin-mrc.sh` in the NCCL tree does this, and also rebuilds `docs/examples` in place against
the new build. Copy the example binaries elsewhere before you build the other mode.

```bash
cd /path/to/nccl
# CPU RQ
GIN_MRC_RQ=cpu BUILD_DIR=$PWD/build-cpu DOCA_HOME=/path/to/gpunetio ./build-nccl-gin-mrc.sh
# GPU RQ
GIN_MRC_RQ=gpu BUILD_DIR=$PWD/build-gpu DOCA_HOME=/path/to/gpunetio \
  DOCA_SDK_HOME=/opt/mellanox/doca ./build-nccl-gin-mrc.sh
```

Without the script, build NCCL with the following. The script uses the same flags, with
`NVCC_GENCODE` fixed to GB200 (`sm_100`):

```bash
make -j src.build BUILDDIR=$PWD/build-cpu GIN_MRC=1 GIN_MRC_RQ=cpu DOCA_HOME=/path/to/gpunetio \
  NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"
make -j src.build BUILDDIR=$PWD/build-gpu GIN_MRC=1 GIN_MRC_RQ=gpu DOCA_HOME=/path/to/gpunetio \
  DOCA_SDK_HOME=/opt/mellanox/doca NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"
```

To check which mode an NCCL build has, look at `NCCL_GIN_MRC_GPU_RQ` (`0` = CPU RQ, `1` = GPU RQ):

```bash
grep -h "define NCCL_GIN_MRC_GPU_RQ" /path/to/nccl/build-*/include/nccl_device/gin/mrc/gin_mrc_config.h
```

**Step 2: build the GIN plugin once.** It is built on the DOCA SDK in `MRC_HOME`. `NCCL_HOME` can be
either NCCL build from step 1, because both install the device structs that either mode hands to
NCCL. The same `libnccl-gin-mrc.so` then runs with both NCCL builds.

```bash
cd mrc-nccl-plugin
make gin MRC_HOME=/opt/mellanox/doca CUDA_HOME=/usr/local/cuda NCCL_HOME=/path/to/nccl/build-cpu
```

If you change the signal immediate layout (see [Signals and receive queues](#signals-and-receive-queues)),
pass the same `-DDOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_SHIFT` to NCCL and, in `CXXFLAGS`, to the plugin.

**Step 3: build applications against the NCCL build of the mode you want.** Compile any code that
uses the device API with:

```bash
-DNCCL_GIN_MRC_ENABLE=1 -I/path/to/nccl/build-<mode>/include
```

This selects MRC and disables GDAKI. The receive-queue mode comes from the NCCL install
(`nccl_device/gin/mrc/gin_mrc_config.h`), which also carries the headers either mode needs, so the
application needs no DOCA include path or library beyond it. Device code is inlined into the
application's kernels, so a binary built against one mode must run with the NCCL build of that mode.

JIT-compiled kernels, such as NCCL EP's, are compiled at run time against the NCCL headers they find.
Set the following for every rank, and one `ep_bench` binary works with either mode:

```bash
export NCCL_EP_JIT_BUILD_INCLUDE_DIR=/path/to/nccl/build-<mode>/include
export NVCC_EXTRA_FLAGS="-DNCCL_GIN_MRC_ENABLE=1"
```

### Signals and receive queues

In both modes the plugin creates the RCX QPs with the DOCA SDK.

With `GIN_MRC_RQ=cpu`, the plugin lays the open-source device QP struct over the send queue of each
QP. A put with a signal is a single `doca_gpu_dev_verbs_put_immediate` whose immediate packs the
signal id and value. A progress thread of the GIN plugin drains the host-memory
receive CQs, adds each value to the GPU slot of its signal through a GDRCopy mapping (forced-PCIe with
Data Direct), and returns the receive credits, with the same semantics as the open-source GPUNetIO
`doca_gpu_verbs_rq_progress` and signal maps. Device code only reads signal slots.

With `GIN_MRC_RQ=gpu`, NCCL drives the DOCA SDK GPU handlers of the QPs (`doca_gpu_verbs_export_qp`),
whose receive CQs are in GPU memory. The signal immediate rides the last Write-with-Immediate of the put's data
(zero-length Write-with-Immediates carry signals without data and the further chunks of a large Add). The GPU drains the receive CQs whenever it reads or waits on a signal or counter,
applies the signal operations and reposts the receive WQEs (`NCCL_GIN_MRC_RQ_POST_MODE`). No CPU thread
is involved, but a signal is only applied while some kernel on the receiver drives GIN progress.

In CPU RQ mode, the 15-bit signal id of the default immediate layout bounds a context to 32768 signals
(NCCL signals plus 3 per rank for `get`). Larger devComms fail context creation; rebuild NCCL and the plugin
with `-DDOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_SHIFT=24` for 23-bit ids, at the cost of 8-bit signal values
(a larger `ncclGin_SignalAdd` is split into more immediates).

## Run (NCCL)

NCCL loads external plugins via the `NCCL_NET_PLUGIN` environment variable. If `NCCL_NET_PLUGIN` is set as `libnccl-net-mrc.so` or just as the suffix `mrc`, add the folder containing the plugin library to your `LD_LIBRARY_PATH`. Instead, `NCCL_NET_PLUGIN` can be set to the absolute path, for e.g., `/path/to/libnccl-net-mrc.so`.

With `NCCL_DEBUG=INFO`, verify `Loaded net plugin MRC (v12)`, `Using network MRC`. For the version of mrc-nccl-plugin being used, look for `Initializing MRC plugin version` string. The plugin version and the git version will follow it. A `NET/Plugin` load error followed by `Using network IB` means NCCL fell back to its built-in transport; subsequent IB CQE errors do not establish a failure in the MRC data path. Setting `NCCL_NET=MRC` (and forwarding it to every rank) makes such fallback fail explicitly instead. 

The GIN plugin is selected the same way through `NCCL_GIN_PLUGIN`:

```bash
export NCCL_GIN_PLUGIN=$PWD/libnccl-gin-mrc.so
```
or
```bash
export NCCL_GIN_PLUGIN=mrc
```

Add `NCCL_GIN_TYPE=6` to force NCCL to use the MRC GIN backend rather than picking another
available one. MRC GIN requires `NCCL_P2P_DISABLE=1`, since a GIN rank stride greater than one is
not supported.

**Selecting the RQ mode.** No environment variable selects the mode. It comes from the NCCL library
the ranks load, so put `/path/to/nccl/build-cpu/lib` or `/path/to/nccl/build-gpu/lib` first in
`LD_LIBRARY_PATH` on every rank, and run binaries built against that same NCCL build. With
`NCCL_DEBUG=INFO`, every rank logs the mode it runs in:

```
GIN/MRC : rank 0 created 2 contexts over 16 ranks on mlx5_1 (CPU RQ, RQ depth 1024, RDW get window 4, Data Direct on, ...)
```

CPU RQ needs GDRCopy (`libgdrapi.so`) on every node. If the plugin cannot map the signal table, context
creation fails with `CPU RQ mode needs GDRCopy`.

GIN connection *i* of every rank must reach connection *i* of every peer. Starting with NCCL 2.32,
NCCL orders each GPU's GIN devices by the rail and plane IDs the plugin reports, and both plugins take
those IDs from `NCCL_IB_HCA` entries of the form `<dev>:<port>:<rail>:<plane>`. On fabrics where NICs
sit on disjoint planes, give NICs that share a GPU group the same rail and give each NIC its plane, for
example `NCCL_IB_HCA==mlx5_0:1:0:0,mlx5_1:1:0:1,mlx5_2:1:1:0,mlx5_3:1:1:1`. Without them, NCCL 2.32
may pair NICs on different planes and QP setup fails with `failed to connect QP`.

Each GIN context needs one MRC QP per peer, so a device communicator uses about
`ginContextCount / connections × (nRanks + 1)` QPs per rank on each NIC. MRC NICs cap the number of
connected QPs, and every process on the NIC shares the cap: about 3K per NIC was observed on Pilot.
When the cap is hit, the INIT->RTR transition fails with a firmware `BAD_PARAM_ERR` and the plugin logs
`QP connect failed after N of M QPs`. In that case, request fewer GIN contexts. A failure at 0 QPs
usually means the peer NIC is on another plane.

With Data Direct, a NIC is reachable over a direct PCIe link from one GPU only, so NCCL would give each
GPU a single, GPU-specific NIC. On fabrics where the two NICs of a GPU pair sit on different planes,
ranks would then be paired across planes. The GIN plugin therefore keeps reporting each NIC at its
regular PCI location, so every GPU still uses one connection per plane. It enables Data Direct per
connection, only on the NIC whose Data Direct function sits in front of the GPU
(`NCCL_GIN_MRC_DATA_DIRECT`, on by default). `NCCL_IB_DATA_DIRECT=1` would also move the NET plugin
and NCCL's host-RMA proxy to Data Direct, where they hit that plane mismatch, so keep it at 0 on such
fabrics.

## Test

Run each multi-node test once per RQ mode. Between modes, change only the NCCL build that
`LD_LIBRARY_PATH`, the binaries and `NCCL_EP_JIT_BUILD_INCLUDE_DIR` point at; the plugin stays the same.

### Single-node loopback (CPU RQ data path)

`make gin-test` builds `gin_mrc_rcx_loopback`, a single-node check of the CPU RQ data path. It connects
two RCX QPs of one NIC to each other, posts `doca_gpu_dev_verbs_put_immediate` from a kernel and
delivers signals on the CPU. It reports latency and message rate, with a plain-write ping-pong as
reference.

```bash
make gin-test NCCL_HOME=/path/to/nccl/build-cpu
./gin_mrc_rcx_loopback -d mlx5_0 -g 0          # [-x gid] [-t rcx|rc] [-n iters] [-s bytes]
./gin_mrc_rcx_loopback -d mlx5_1 -g 0 -z       # Data Direct
```

The test prints `PASSED` on success. With `-z`, `-d` must be the NIC whose Data Direct function sits
in front of GPU `-g`; with any other NIC, the ping-pong fails with a receive CQE error. Override
`NVCC_GENCODE` if the GPU is not GB200 (`sm_100`).

### Multi-node runs

Every rank needs the plugins, the NCCL build of the mode under test and the DOCA SDK libraries in its
environment. A launch on 4 nodes × 4 GPUs with Open MPI, where `MODE` is `cpu` or `gpu`:

```bash
NCCL_BUILD=/path/to/nccl/build-$MODE
export LD_LIBRARY_PATH=/opt/mellanox/doca/lib/aarch64-linux-gnu:$NCCL_BUILD/lib:/path/to/nccl_ep/lib:$LD_LIBRARY_PATH
mpirun -np 16 --map-by ppr:4:node --bind-to none --hostfile hostfile \
  -x LD_LIBRARY_PATH -x NCCL_DEBUG=INFO \
  -x NCCL_NET_PLUGIN=/path/to/libnccl-net-mrc.so -x NCCL_GIN_PLUGIN=/path/to/libnccl-gin-mrc.so \
  -x NCCL_GIN_TYPE=6 -x NCCL_P2P_DISABLE=1 -x NCCL_SHM_DISABLE=1 -x NCCL_NVLS_ENABLE=0 \
  -x NCCL_MNNVL_ENABLE=0 -x NCCL_IB_HCA==mlx5_1:1:0:1,mlx5_0:1:0:0,mlx5_3:1:1:1,mlx5_2:1:1:0 \
  -x NCCL_IB_DATA_DIRECT=0 -x NCCL_GIN_MRC_DATA_DIRECT=1 \
  -x NCCL_EP_JIT_BUILD_INCLUDE_DIR=$NCCL_BUILD/include -x NVCC_EXTRA_FLAGS=-DNCCL_GIN_MRC_ENABLE=1 \
  <binary> <args>
```

Adapt `NCCL_IB_HCA` (rail and plane IDs) to the fabric, and add the site's usual socket, GID and
MPI transport settings. Check the log for `GIN/MRC : rank ... (CPU RQ` or `(GPU RQ` to confirm the mode.

| Test | Binary and arguments | Checks | Pass criterion |
|---|---|---|---|
| Pure GIN all-to-all | NCCL `docs/examples/06_device_api/02_alltoall_gin/c/alltoall_gin` | Puts with signals, counters, flush | `Pure GIN AlltoAll result: PASSED` |
| Hybrid all-to-all | NCCL `docs/examples/06_device_api/03_alltoall_hybrid/c/alltoall_hybrid` | GIN together with LSA | `Hybrid AlltoAll result: PASSED` |
| MoE low latency | NCCL EP `ep_bench -a ll -V` | LL dispatch/combine, latency | `Global validation: Dispatch=PASSED, Combine=PASSED` |
| MoE high throughput | NCCL EP `ep_bench -a ht -V` | HT dispatch/combine, bandwidth | `Global validation: Dispatch=PASSED, Combine=PASSED` |

The NCCL examples are built against one NCCL build, so build them once per mode (step 3 above).
`ep_bench` (`nccl-extensions`, `nccl_ep/ep_bench.cu`) JIT-compiles its kernels, so one binary serves both
modes. Its defaults are 128 tokens for LL and 4096 for HT, hidden 7168, top-k 8, 256 experts, 10
warm-up and 50 timed iterations.

For performance, compare `ep_bench`'s `--- Host-observed performance ---` averages and p99 (LL), and
the `--- BW based on total time ---` dispatch and combine totals (HT). Run-to-run variation on a shared
fabric is around 10 % for LL, so alternate the two modes over several rounds rather than running all
of one mode first.

### Performance of the two modes

Measured with `ep_bench` defaults on 4 nodes × 4 GB200 (16 ranks), DOCA SDK 3.2, CUDA 13.0, Data
Direct on, the same plugin binary in both modes. Each value is the mean ± standard deviation of 5
rounds that alternate the two modes. Lower times are better.

| Metric | CPU RQ | GPU RQ | GPU RQ vs CPU RQ |
|---|---|---|---|
| HT dispatch (ms) | 78.2 ± 2.2 | 77.1 ± 0.9 | −1.4 %, within noise |
| HT combine (ms) | 72.2 ± 2.4 | 76.4 ± 0.8 | **+5.8 %** (significant; CPU RQ faster in 4 of 5 rounds) |
| HT dispatch + combine (ms) | 150.5 ± 4.4 | 153.5 ± 1.6 | +2.0 % |
| LL dispatch avg (ms) | 6.78 ± 0.35 | 7.21 ± 0.41 | +6.2 %, within noise |
| LL combine avg (ms) | 8.50 ± 0.57 | 8.67 ± 0.49 | +2.0 %, within noise |
| Busy host threads per node (4 ranks) | 20–24 | 4–8 | CPU RQ uses about 4–5 extra cores per GPU |

* **Throughput:** HT dispatch is the same in both modes. CPU RQ is consistently faster at HT combine,
  which sends many small signals. The likely cause is that the host thread drains receive queues
  continuously, while GPU RQ applies signals only when the receiving kernel polls. Changing GPU RQ's
  RQ depth or RNR timer, or turning Data Direct off, did not close the gap.
* **Latency:** LL favours CPU RQ on average, but the differences are within the round-to-round spread.
* **Host CPU:** the GPU RQ busy threads are each rank's main thread waiting on the GPU. CPU RQ adds one
  polling thread per GIN context set; NCCL EP creates several device communicators per rank.

Choose CPU RQ when host cores are available. Choose GPU RQ when host CPU is scarce or must stay quiet,
or when GDRCopy is unavailable, at a cost of a few percent on signal-heavy traffic.
