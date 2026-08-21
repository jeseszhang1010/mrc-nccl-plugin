# mrc-nccl-plugin

External NCCL plugins using MRC (Multi-Path Reliable Connection) as transport:

* **NET plugin** (`libnccl-net-mrc.so`) — host-initiated networking, NCCL NET plugin ABI v6 through v11.
* **GIN plugin** (`libnccl-gin-mrc.so`) — GPU-initiated networking, NCCL GIN plugin ABI v14. Provides the
  host half of NCCL's `NCCL_NET_DEVICE_GIN_MRC` backend; the device half ships inside NCCL as
  `nccl_device/gin/mrc/`.


## Requirements

* MRC
* CUDA
* OFED
* NCCL
* DOCA (GPUNetIO + verbs), for the GIN plugin only

## Build

```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl/build
```
Debug build
```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl/build DEBUG=1
```

`make net` and `make gin` build a single plugin. The GIN plugin needs a CUDA architecture matching
the target GPU; override `NVCC_GENCODE` if it is not GB200 (`sm_100`).

`libnccl-net-mrc.so` and `libnccl-gin-mrc.so` are generated after building, with `mrc` as the suffix
of each plugin library.

### NCCL and application builds

The device half of the MRC GIN backend lives in NCCL and is opt-in, because it vendors a newer DOCA
device-header snapshot than the GDAKI backend and the two cannot coexist in one translation unit.
Build NCCL, and any application using the device API, with:

```bash
-DNCCL_GIN_MRC_ENABLE=1
```

This selects MRC and disables GDAKI. No DOCA include path or library is needed for either build --
only the MRC GIN plugin links DOCA.

## Run (NCCL)

NCCL loads external plugins via the `NCCL_NET_PLUGIN` environment variable. It can be set to either
a suffix string or to a library name.

Example:

```bash
export NCCL_NET_PLUGIN=$PWD/libnccl-net-mrc.so
```
or
```bash
export NCCL_NET_PLUGIN=mrc
```

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
