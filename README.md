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
* DOCA (GPUNetIO + verbs), for the GIN plugin only

## Build

```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda
```
Debug build
```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda DEBUG=1
```

To supply `nccl.h` from a custom NCCL installation, add `NCCL_HOME=/path/to/nccl/build` to the make command. `nccl.h` will be included from `$(NCCL_HOME)/include`.

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
