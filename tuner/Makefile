PLUGIN_SO := libnccl-tuner-mrc.so

SRC_FILES := plugin.c

.DEFAULT_GOAL := build

all: build test

build: $(PLUGIN_SO)

$(PLUGIN_SO): $(SRC_FILES)
	@printf "Compiling  %-35s > %s\n" $< $@
	$(CC) -Inccl -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^


TEST_BINS := test/csv_tuning_main test/cost_model_main

test/%: test/%.c plugin.c
	@printf "Compiling  %-35s > %s\n" $< $@
	$(CC) -Inccl -O2 -Wall -o $@ $<

test: $(TEST_BINS)

clean:
	rm -f $(PLUGIN_SO) $(TEST_BINS)

.PHONY: all build clean test
