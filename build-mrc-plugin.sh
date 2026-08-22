#!/bin/bash

# Point NCCL_HOME to your NCCL built with GIN MRC enabled
NCCL_HOME=${NCCL_HOME:-$PWD/../nccl/build}
NCCL_HOME=$NCCL_HOME make all