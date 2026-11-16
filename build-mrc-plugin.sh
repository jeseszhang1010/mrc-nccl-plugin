#!/bin/bash

# Point NCCL_HOME to your NCCL built with GIN MRC enabled (GIN_MRC=1, either GIN_MRC_RQ mode). Both
# plugins are built on the DOCA SDK in MRC_HOME (default /opt/mellanox/doca); the GIN plugin follows
# the receive-queue mode of the NCCL it runs with.
NCCL_HOME=${NCCL_HOME:-$PWD/../nccl-RCX/build}
make clean
NCCL_HOME=$(realpath "$NCCL_HOME") make all
