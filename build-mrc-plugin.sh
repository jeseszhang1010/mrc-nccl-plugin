#!/bin/bash

# Point NCCL_HOME to your NCCL built with GIN MRC enabled (GIN_MRC=1) from the same branch
# (mrc-gin-v2.32.6-cpu-rq). Both plugins are built on the DOCA SDK in MRC_HOME (default
# /opt/mellanox/doca).
NCCL_HOME=${NCCL_HOME:-$PWD/../nccl-RCX/build}
make clean
NCCL_HOME=$(realpath "$NCCL_HOME") make all
