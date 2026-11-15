#!/bin/bash

# Point NCCL_HOME to your NCCL built with GIN MRC enabled, and DOCA_HOME to the open-source DOCA
# GPUNetIO tree (https://github.com/NVIDIA-DOCA/gpunetio). The GIN plugin supports both receive-queue
# modes NCCL can be built with (GIN_MRC_RQ=cpu|gpu) when DOCA_HOME provides doca_gpu_verbs_export_sdk_qp
# (feature/export_sdk_qp, the default), and only GIN_MRC_RQ=cpu otherwise (e.g. main).
NCCL_HOME=${NCCL_HOME:-$PWD/../nccl-RCX/build}
DOCA_HOME=${DOCA_HOME:-$PWD/../gpunetio-sdkqp}
make clean
NCCL_HOME=$NCCL_HOME DOCA_HOME=$(realpath "$DOCA_HOME") make all
