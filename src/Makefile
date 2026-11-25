# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Build the NCCL NET plugin from the imported include, misc, and transport trees.

DEBUG ?= 0
MRC_HOME ?= /opt/mellanox/doca
CUDA_HOME ?= /usr/local/cuda
MRC_LIBDIR ?= $(MRC_HOME)/lib/aarch64-linux-gnu
PYTHON ?= python3

CXX ?= g++
ifneq ($(NCCL_HOME),)
CPPFLAGS += -I$(NCCL_HOME)/include
endif
CPPFLAGS += -I. -Iinclude -Iinclude/plugin -I$(MRC_HOME)/include
CPPFLAGS += -I$(CUDA_HOME)/include -I$(CUDA_HOME)/include/cccl
CPPFLAGS += -DNCCL_OS_LINUX -DNCCL_BUILD_RDMA_CORE -DNCCL_COMM_H_ -DncclNetIb=ncclNetPlugin_v12
CXXFLAGS ?= -O3 -g -fPIC -Wall -Wno-unused-function -Wno-sign-compare -Wvla -std=c++17
ifneq ($(DEBUG),0)
CXXFLAGS := $(filter-out -O3,$(CXXFLAGS)) -O0 -ggdb3
endif

# An external plugin cannot resolve private symbols from libnccl. Fail the
# build instead of letting dlopen fail and NCCL silently fall back to IB.
LDFLAGS += -shared -Wl,-z,defs
LDFLAGS += -L$(MRC_LIBDIR) -Wl,-rpath,$(MRC_LIBDIR)
LDFLAGS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64
LDLIBS += -libverbs -lnv_mrc -lcudart -ldl -lpthread -lrt

OUTPUT ?= ../libnccl-net-mrc.so
OBJDIR ?= ../build/obj
MISC_SRCS := $(filter-out misc/argcheck.cc misc/shmutils.cc misc/strongstream.cc,$(wildcard misc/*.cc))
SRCS := $(MISC_SRCS) $(wildcard os/*.cc) $(wildcard transport/*.cc) $(wildcard transport/*/*.cc)
OBJS := $(SRCS:%.cc=$(OBJDIR)/%.o)
DEPS := $(OBJS:.o=.d)
GIT_VERSION_FILE := $(OBJDIR)/include/nccl_git_version.h

# GIN plugin: the net_ib device discovery above plus the MRC QP code, built on the DOCA SDK in
# MRC_HOME (verbs and GPUNetIO: the RCX QP type is a DOCA SDK feature). NCCL_HOME is an NCCL built
# with GIN_MRC=1 from the same (GPU RQ) branch. NCCL's device code uses the DOCA SDK device headers,
# as does the receive-ring fill kernel.
GIN_OUTPUT ?= ../libnccl-gin-mrc.so
NVCC ?= $(CUDA_HOME)/bin/nvcc
# GB200 by default; override for other architectures.
NVCC_GENCODE ?= -gencode=arch=compute_100,code=sm_100
GIN_DEFINES := -DDOCA_ALLOW_EXPERIMENTAL_API -DGPUNETIO_VERBS_RCX_SUPPORTED
GIN_SDK_SRCS := $(wildcard gin_mrc/*.cc)
GIN_SDK_OBJS := $(GIN_SDK_SRCS:%.cc=$(OBJDIR)/%.o)
GIN_CU_OBJS := $(OBJDIR)/gin_mrc/mrc_rq_init.o
GIN_OBJS := $(OBJS) $(GIN_SDK_OBJS) $(GIN_CU_OBJS)
DEPS += $(GIN_SDK_OBJS:.o=.d) $(GIN_CU_OBJS:.o=.d)
GIN_LDLIBS := $(LDLIBS) -lcuda -ldoca_verbs -ldoca_gpunetio -ldoca_common

# NCCL's bundled mlx5dvcore.h would clash with the system mlx5dv.h the DOCA SDK headers include.
$(GIN_SDK_OBJS): CPPFLAGS += -Igin_mrc -I$(NCCL_HOME)/include $(GIN_DEFINES) -DNCCL_BUILD_MLX5DV

all: $(OUTPUT)

gin: $(GIN_OUTPUT)

$(OUTPUT): $(OBJS) nccl-net-mrc.map Makefile
	@printf "Linking    %-35s > %s\n" $(notdir $@) $@
	$(CXX) $(LDFLAGS) -Wl,--version-script=nccl-net-mrc.map -o $@ $(OBJS) $(LDLIBS)

$(GIN_OUTPUT): $(GIN_OBJS) nccl-gin-mrc.map Makefile
	@printf "Linking    %-35s > %s\n" $(notdir $@) $@
	$(CXX) $(LDFLAGS) -Wl,--version-script=nccl-gin-mrc.map -o $@ $(GIN_OBJS) $(GIN_LDLIBS)

# Receive-ring fill: NCCL's device code over the DOCA SDK device headers.
$(OBJDIR)/gin_mrc/mrc_rq_init.o: gin_mrc/mrc_rq_init.cu gin_mrc/mrc_rq_init.h Makefile
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(NVCC) -Igin_mrc -I$(NCCL_HOME)/include -I$(MRC_HOME)/include -DNCCL_GIN_MRC_ENABLE=1 \
	  $(NVCC_GENCODE) -std=c++17 -Xcompiler -fPIC -MMD -MP -c $< -o $@

$(OBJDIR)/misc/git_version.o $(OBJDIR)/transport/net_ib/init.o: $(GIT_VERSION_FILE)

$(GIT_VERSION_FILE): FORCE
	@mkdir -p $(dir $@)
	@$(PYTHON) ./misc/generate_git_version.py $@

$(OBJDIR)/%.o: %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(OBJDIR)/include -MMD -MP -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(OUTPUT) $(GIN_OUTPUT)

FORCE:

-include $(DEPS)

.PHONY: all gin clean FORCE
