# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Build the NCCL tuner plugin. It only depends on libc and the bundled
# nccl/ headers, so it does not need MRC, CUDA, or NCCL to build.

CPPFLAGS += -Inccl

OUTPUT ?= libnccl-tuner-mrc.so
SRCS := plugin.c
HDRS := $(wildcard nccl/*.h)
TEST_BINS := test/csv_tuning_main test/cost_model_main

.DEFAULT_GOAL := build

all: build test

build: $(OUTPUT)

$(OUTPUT): $(SRCS) $(HDRS) Makefile
	@printf "Compiling  %-35s > %s\n" $< $@
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared -o $@ -Wl,-soname,$(notdir $@) $(SRCS) $(LDFLAGS)

# Host-side test drivers include plugin.c directly.
test/%: test/%.c $(SRCS) $(HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	$(CC) $(CPPFLAGS) -O2 -Wall -o $@ $<

test: $(TEST_BINS)

clean:
	rm -f $(OUTPUT) $(TEST_BINS)

.PHONY: all build clean test
