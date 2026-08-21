TARGET ?= libnccl-net-mrc.so
GIN_TARGET ?= libnccl-gin-mrc.so

all: net gin

net:
	$(MAKE) -C src OUTPUT=../$(TARGET)

# Needs DOCA GPUNetIO and an NCCL_HOME built with GIN_MRC=1.
gin:
	$(MAKE) -C src gin GIN_OUTPUT=../$(GIN_TARGET)

clean:
	$(MAKE) -C src OUTPUT=../$(TARGET) GIN_OUTPUT=../$(GIN_TARGET) clean

test: net
	$(MAKE) -C tests PLUGIN=$(abspath $(TARGET)) test

.PHONY: all net gin clean test
