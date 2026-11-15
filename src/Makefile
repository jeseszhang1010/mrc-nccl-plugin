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

# GIN plugin: the net_ib device discovery above plus the DOCA GPUNetIO QP code, built on the
# open-source DOCA GPUNetIO library: DOCA_HOME is a checkout of https://github.com/NVIDIA-DOCA/gpunetio.
# It runs in SDK mode for RCX QPs, forwarding to the DOCA SDK libraries in DOCA_SDK_LIB_PATH (default
# MRC_LIBDIR). The plugin follows the receive-queue mode of the NCCL it runs with (GIN_MRC_RQ):
#   cpu: needs doca_gpu_dev_verbs_put_immediate and the CPU-managed RQ (main branch).
#   gpu: also needs doca_gpu_verbs_export_sdk_qp (feature/export_sdk_qp, the default DOCA_HOME), plus
#        the DOCA SDK device headers in MRC_HOME/include for the receive-ring fill kernel. Without
#        it, the plugin is built for CPU RQ only.
#
# Like NCCL, the plugin compiles the library sources in rather than linking its shared library:
# the version script then keeps the open-source doca_verbs_* symbols local, so they cannot
# satisfy the same-named imports of the DOCA SDK libraries (libnv_mrc) loaded next to them.
GIN_OUTPUT ?= ../libnccl-gin-mrc.so
DOCA_HOME ?= $(abspath ../../gpunetio-sdkqp)
DOCA_SRCS := $(wildcard $(DOCA_HOME)/src/*.cpp $(DOCA_HOME)/src/gpu_dev_compat/*.cpp)
# Same configuration as NCCL's build of these sources: CUDA and verbs are resolved at run time.
DOCA_DEFINES := -DDOCA_GPUNETIO_USE_CUDA_WRAPPER -DDOCA_VERBS_USE_NET_WRAPPER
DOCA_OBJS := $(DOCA_SRCS:$(DOCA_HOME)/src/%.cpp=$(OBJDIR)/doca_gpunetio/%.o)
NVCC ?= $(CUDA_HOME)/bin/nvcc
# GB200 by default; override for other architectures.
NVCC_GENCODE ?= -gencode=arch=compute_100,code=sm_100
GIN_DEFINES := -DMRC_DOCA_SDK_LIB_PATH=\"$(MRC_LIBDIR)\"
GIN_CC_OBJS := $(patsubst %.cc,$(OBJDIR)/%.o,$(wildcard gin_mrc/*.cc))
GIN_CU_OBJS :=
MRC_GPU_RQ := $(shell grep -qs doca_gpu_verbs_export_sdk_qp $(DOCA_HOME)/include/host/doca_gpunetio.h && echo 1)
ifeq ($(MRC_GPU_RQ),1)
GIN_DEFINES += -DMRC_HAVE_GPU_RQ
GIN_CU_OBJS := $(OBJDIR)/gin_mrc/mrc_rq_init.o
endif
GIN_OBJS := $(OBJS) $(GIN_CC_OBJS) $(GIN_CU_OBJS) $(DOCA_OBJS)
DEPS += $(GIN_CC_OBJS:.o=.d) $(GIN_CU_OBJS:.o=.d) $(DOCA_OBJS:.o=.d)
GIN_LDLIBS := $(LDLIBS) -lcuda -lmlx5

# NCCL's bundled mlx5dvcore.h would clash with the system mlx5dv.h the verbs wrappers need here.
$(GIN_CC_OBJS): CPPFLAGS += -Igin_mrc -I$(NCCL_HOME)/include -I$(DOCA_HOME)/include $(GIN_DEFINES) -DNCCL_BUILD_MLX5DV

all: $(OUTPUT)

gin: $(GIN_OUTPUT)

$(OUTPUT): $(OBJS) nccl-net-mrc.map Makefile
	@printf "Linking    %-35s > %s\n" $(notdir $@) $@
	$(CXX) $(LDFLAGS) -Wl,--version-script=nccl-net-mrc.map -o $@ $(OBJS) $(LDLIBS)

$(GIN_OUTPUT): $(GIN_OBJS) nccl-gin-mrc.map Makefile
	@printf "Linking    %-35s > %s\n" $(notdir $@) $@
	$(CXX) $(LDFLAGS) -Wl,--version-script=nccl-gin-mrc.map -o $@ $(GIN_OBJS) $(GIN_LDLIBS)

# GPU RQ receive-ring fill: NCCL's GPU RQ device code over the DOCA SDK device headers, whatever
# mode NCCL_HOME was built with.
$(OBJDIR)/gin_mrc/mrc_rq_init.o: gin_mrc/mrc_rq_init.cu gin_mrc/mrc_rq_init.h Makefile
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(NVCC) -Igin_mrc -I$(NCCL_HOME)/include -I$(MRC_HOME)/include -DNCCL_GIN_MRC_ENABLE=1 -DNCCL_GIN_MRC_GPU_RQ=1 \
	  $(NVCC_GENCODE) -std=c++17 -Xcompiler -fPIC -MMD -MP -c $< -o $@

$(OBJDIR)/doca_gpunetio/%.o: $(DOCA_HOME)/src/%.cpp
	@printf "Compiling  %-35s > %s\n" $(notdir $<) $@
	@mkdir -p $(dir $@)
	$(CXX) -O3 -fPIC -std=c++17 $(DOCA_DEFINES) -I$(DOCA_HOME)/include -I$(DOCA_HOME)/src -I$(CUDA_HOME)/include -MMD -MP -c $< -o $@

# Single-node RCX loopback check of put_immediate + CPU signal delivery (see the test source).
GIN_TEST ?= ../gin_mrc_rcx_loopback
gin-test: $(GIN_TEST)

$(GIN_TEST): gin_mrc/test/rcx_loopback.cu gin_mrc/mrc_qp.cc gin_mrc/mrc_qp.h Makefile
	@printf "Linking    %-35s > %s\n" $(notdir $@) $@
	$(NVCC) -std=c++17 -O2 $(NVCC_GENCODE) -I$(DOCA_HOME)/include -I$(DOCA_HOME)/src $(DOCA_DEFINES) -o $@ gin_mrc/test/rcx_loopback.cu \
	  gin_mrc/mrc_qp.cc $(DOCA_SRCS) -libverbs -lmlx5 -lcuda -ldl

$(OBJDIR)/misc/git_version.o $(OBJDIR)/transport/net_ib/init.o: $(GIT_VERSION_FILE)

$(GIT_VERSION_FILE): FORCE
	@mkdir -p $(dir $@)
	@$(PYTHON) ./misc/generate_git_version.py $@

$(OBJDIR)/%.o: %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(OBJDIR)/include -MMD -MP -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(OUTPUT) $(GIN_OUTPUT) $(GIN_TEST)

FORCE:

-include $(DEPS)

.PHONY: all gin gin-test clean FORCE
