TARGET ?= libnccl-net-mrc.so
GIN_TARGET ?= libnccl-gin-mrc.so

all: net gin

net:
	$(MAKE) -C src OUTPUT=../$(TARGET)

# Needs the open-source DOCA GPUNetIO (DOCA_HOME) and an NCCL_HOME built with GIN_MRC=1 against it.
gin:
	$(MAKE) -C src gin GIN_OUTPUT=../$(GIN_TARGET)

gin-test:
	$(MAKE) -C src gin-test

clean:
	$(MAKE) -C src OUTPUT=../$(TARGET) GIN_OUTPUT=../$(GIN_TARGET) clean

test: net
	$(MAKE) -C tests PLUGIN=$(abspath $(TARGET)) test

.PHONY: all net gin gin-test clean test
