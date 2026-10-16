# Plugins built by `make` / `make all`. Each plugin can also be built on its
# own, e.g. `make net` or `make tuner`, or by overriding PLUGINS, e.g.
# `make PLUGINS=tuner`.
PLUGINS ?= net tuner

# TARGET is kept for backward compatibility with older build scripts.
TARGET ?= libnccl-net-mrc.so
NET_TARGET ?= $(TARGET)
TUNER_TARGET ?= libnccl-tuner-mrc.so

all: $(PLUGINS)

net:
	$(MAKE) -C net OUTPUT=$(abspath $(NET_TARGET))

tuner:
	$(MAKE) -C tuner OUTPUT=$(abspath $(TUNER_TARGET))

clean: clean-net clean-tuner

clean-net:
	$(MAKE) -C net OUTPUT=$(abspath $(NET_TARGET)) clean

clean-tuner:
	$(MAKE) -C tuner OUTPUT=$(abspath $(TUNER_TARGET)) clean

test: net
	$(MAKE) -C tests PLUGIN=$(abspath $(NET_TARGET)) test

# Build the host-side tuner test drivers (no CUDA/NCCL required).
test-tuner:
	$(MAKE) -C tuner test

.PHONY: all net tuner clean clean-net clean-tuner test test-tuner
