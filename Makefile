TARGET ?= libnccl-net-mrc.so

all:
	$(MAKE) -C net OUTPUT=../$(TARGET)

clean:
	$(MAKE) -C net OUTPUT=../$(TARGET) clean

test: all
	$(MAKE) -C tests PLUGIN=$(abspath $(TARGET)) test

.PHONY: all clean test
