# mrc-nccl-plugin

External NCCL plugins using MRC (Multi-Path Reliable Connection) as transport:

* **NET plugin** (`libnccl-net-mrc.so`) — host-initiated networking, NCCL NET plugin ABI v12. Reports
  the network name `MRC` and requires a NCCL runtime that supports v12; the legacy C sources are not
  part of this build.
* **GIN plugin** (`libnccl-gin-mrc.so`) — GPU-initiated networking, NCCL GIN plugin ABI v14. Provides the
  host half of NCCL's `NCCL_NET_DEVICE_GIN_MRC` backend; the device half ships inside NCCL as
  `nccl_device/gin/mrc/`.


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

```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda
```
Debug build
```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda DEBUG=1
```

To supply `nccl.h` from a custom NCCL installation, add `NCCL_HOME=/path/to/nccl/build` to the make command. `nccl.h` will be included from `$(NCCL_HOME)/include`.

`make net` and `make gin` build a single plugin. The GIN plugin is built on the DOCA SDK in `MRC_HOME`
and needs the `NCCL_HOME` of an NCCL built with GIN MRC, whose installed headers provide the device
structs both receive-queue modes (below) hand to NCCL. One plugin build serves NCCL builds of either
mode:

```bash
make gin NCCL_HOME=/path/to/nccl/build
```

`make gin-test` builds `gin_mrc_rcx_loopback`, a single-node check of the CPU RQ data path: two RCX QPs
of one NIC connected to each other, `doca_gpu_dev_verbs_put_immediate` from a kernel, and CPU signal
delivery. It reports latency and message rate, with a plain-write ping-pong as reference. Add `-z` for
Data Direct with a NIC whose Data Direct function sits in front of the GPU (`-d`, `-g`). Override
`NVCC_GENCODE` if the GPU is not GB200 (`sm_100`).

`libnccl-net-mrc.so` and `libnccl-gin-mrc.so` are generated after building, with `mrc` as the suffix
of each plugin library.

### NCCL and application builds

The device half of the MRC GIN backend lives in NCCL and is opt-in: build NCCL with `GIN_MRC=1`
(`build-nccl-gin-mrc.sh` in the NCCL tree). `GIN_MRC_RQ` selects who manages the receive queues:

* `GIN_MRC_RQ=cpu` (default): `doca_gpu_dev_verbs_put_immediate` and CPU-managed receive queues, with
  the open-source GPUNetIO device headers NCCL embeds (`DOCA_HOME`). The plugin creates the QPs with
  the DOCA SDK and lays out the open-source device QP struct over their send queues.
* `GIN_MRC_RQ=gpu`: the GPU posts and drains the receive queues with the DOCA SDK device functions,
  through the DOCA SDK GPU handlers of the plugin's QPs (`doca_gpu_verbs_export_qp`). NCCL takes the
  SDK device headers from the DOCA SDK install (`DOCA_SDK_HOME`, default `/opt/mellanox/doca`) and
  installs them with its own headers.

The two modes cannot coexist in one NCCL build: the open-source and SDK device headers share include
guards and type names with different layouts, and the device code is inlined into every kernel using
the device API. Build NCCL once per mode (for example with `BUILD_DIR=build-cpu` and `build-gpu`) and
switch by pointing `LD_LIBRARY_PATH` and application builds at the matching install. NCCL reports the
mode to the plugin as the backend version (7 for CPU, 8 for GPU RQ), and the plugin follows it per
communicator.

Build any application using the device API, including JIT-compiled kernels such as NCCL EP's
(`NVCC_EXTRA_FLAGS`), with:

```bash
-DNCCL_GIN_MRC_ENABLE=1
```

This selects MRC and disables GDAKI. The receive-queue mode comes from the NCCL install
(`nccl_device/gin/mrc/gin_mrc_config.h`), which also carries the headers either mode needs, so the
application needs no DOCA include path or library beyond it.

### Signals and receive queues

MRC QPs have no atomics, so a signal travels as the immediate of an RDMA Write-with-Immediate.

With `GIN_MRC_RQ=cpu`, a put with a signal is a single `doca_gpu_dev_verbs_put_immediate` whose
immediate packs the signal id and value. A progress thread of the GIN plugin drains the host-memory
receive CQs, adds each value to the GPU slot of its signal through a GDRCopy mapping (forced-PCIe with
Data Direct), and returns the receive credits, with the same semantics as the open-source GPUNetIO
`doca_gpu_verbs_rq_progress` and signal maps. Device code only reads signal slots.

With `GIN_MRC_RQ=gpu`, the signal immediate rides the last Write-with-Immediate of the put's data
(8-byte writes to a sink carry signals without data and the further chunks of a large Add). The GPU drains the receive CQs whenever it reads or waits on a signal or counter,
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
