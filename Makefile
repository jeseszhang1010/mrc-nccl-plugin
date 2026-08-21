# Minimal build for external NCCL NET and GIN plugins.
#

DEBUG ?= 0
MRC_HOME ?= /opt/mellanox/doca
CUDA_HOME ?= /usr/local/cuda
MRC_LIBDIR ?= $(MRC_HOME)/lib/aarch64-linux-gnu
NCCL_HOME ?= /path/to/nccl/build

CC ?= gcc
CXX ?= g++
NVCC ?= $(CUDA_HOME)/bin/nvcc
# GB200 by default; override for other architectures.
NVCC_GENCODE ?= -gencode=arch=compute_100,code=sm_100

CFLAGS ?= -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -std=gnu11
CXXFLAGS ?= -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -std=gnu++17
ifneq ($(DEBUG),0)
CFLAGS += -g
CXXFLAGS += -g
endif

CPPFLAGS += -Iinclude -I$(NCCL_HOME)/include -I$(MRC_HOME)/include -I$(CUDA_HOME)/include
LDFLAGS  += -shared
LDFLAGS  += -L$(MRC_LIBDIR) -Wl,-rpath,$(MRC_LIBDIR)
LDLIBS   += -libverbs -lnv_mrc -ldl -lpthread 

# Only needed if something in this plugin ends up referencing CUDA runtime
# symbols (most builds won't). Keeping it as an opt-in knob.
CUDA_LDFLAGS ?= -L$(CUDA_HOME)/lib64
CUDA_LDLIBS ?= -lcudart

TARGET = libnccl-net-mrc.so
SRCS   = \
	src/net_mrc_plugin.c \
	src/p2p_plugin.c \
	src/param.c \
	src/utils.c \
	src/socket.c \
	src/ibvwrap.c
OBJS   = $(SRCS:.c=.o)

# GIN plugin. Needs C++/CUDA and the DOCA GPUNetIO libraries, so it is built as a
# second shared object rather than folded into the NET plugin.
GIN_TARGET   = libnccl-gin-mrc.so
GIN_CSRCS    = \
	src/p2p_plugin.c \
	src/param.c \
	src/utils.c \
	src/socket.c \
	src/ibvwrap.c
GIN_CXXSRCS  = \
	src/gin_mrc_plugin.cc \
	src/mrc_doca_ext.cc
GIN_CUSRCS   = src/gin_mrc_rq_init.cu
# Separate object suffix so the shared C sources are not fought over by both targets.
GIN_OBJS     = $(GIN_CSRCS:.c=.gin.o) $(GIN_CXXSRCS:.cc=.o) $(GIN_CUSRCS:.cu=.o)
GIN_CPPFLAGS = -DDOCA_ALLOW_EXPERIMENTAL_API -DGPUNETIO_VERBS_RCX_SUPPORTED -DNCCL_PLUGIN_GIN_ONLY
# NCCL dlopens this with RTLD_NOW, so an unresolved symbol is a load failure at run time.
# Catch it here instead.
GIN_LDFLAGS  = -Wl,--no-undefined
GIN_LDLIBS   = -libverbs -lmlx5 -lnv_mrc -ldl -lpthread -lcudart -lcuda -ldoca_verbs -ldoca_gpunetio -ldoca_common

all: $(TARGET) $(GIN_TARGET)

net: $(TARGET)

gin: $(GIN_TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(GIN_TARGET): $(GIN_OBJS)
	$(CXX) $(LDFLAGS) $(GIN_LDFLAGS) $(CUDA_LDFLAGS) -o $@ $^ $(GIN_LDLIBS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

%.gin.o: %.c
	$(CC) $(CPPFLAGS) $(GIN_CPPFLAGS) $(CFLAGS) -c -o $@ $<

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(GIN_CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(GIN_CPPFLAGS) $(NVCC_GENCODE) -std=c++17 -Xcompiler -fPIC -c -o $@ $<

clean:
	rm -f $(OBJS) $(GIN_OBJS) $(TARGET) $(GIN_TARGET)

.PHONY: all net gin clean

