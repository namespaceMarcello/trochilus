# Trochilus — build. The core needs nothing but libc and OS threads.
#   make check      the gate: lint, native build with 0 warnings, then in Linux (Docker image
#                   trochilus-dev) gcc and clang tests, ASan+UBSan tests and the oracles
#   make            build/trochilus
#   make test       C tests (every kernel variant against scalar, GGUF, pool, profiler)
#   make oracle     tiny models against transformers (needs tools/.venv or the Docker image)
#   make tier-check the engine under every kernel tier: model tests, and logits identical byte for byte
#   make oracle-tokenizer  a real tokenizer against transformers (ids, pieces, NFC, decoding)
#   make bench      kernel microbenchmark (median and noise), native
#   make bench-mem  what the RAM gives: plain reads, the engine's matmul, attention on two KV layouts
#   make bench-attn the attention of a whole prompt on one layer, taken apart, with the bits checked
#   make bench-expf tr_expf on all the 2^32 floats against the correctly rounded exp and the C library's
#   make lint       docs and tables against the lessons in docs/LESSONS.md
#   make WERROR=1   warnings are errors
# Extra flags without losing the defaults: EXTRA_CFLAGS, EXTRA_LDFLAGS.

ifeq ($(origin CC),default)
CC      := gcc
endif
# every test and oracle runs the engine in its own process: a `trochilus serve` left up on this
# machine would otherwise run their commands (src/app/serve.c); tests/test_serve.c sets it back
export TR_SERVER = 0
BUILD   := build
CFLAGS  ?= -O2 -g
# -ffp-contract=off: no silent multiply-add fusion, the scalar kernels define the numbers (kernels.h)
CFLAGS  += -std=c11 -Wall -Wextra -Wshadow -Wno-unused-parameter -ffp-contract=off -MMD -MP $(EXTRA_CFLAGS)
ifeq ($(WERROR),1)
CFLAGS  += -Werror
endif

# gcc stores AVX registers into stack arrays with aligned moves, trusting a 32/64-byte stack
# alignment that MinGW (gcc bug 54412) and ASan's fake stack do not give: the GNU assembler
# rewrites them as unaligned moves, same speed on current CPUs (docs/LESSONS.md #19).
# clang uses its own assembler and aligns correctly.
ifeq ($(findstring clang,$(shell $(CC) --version 2>/dev/null)),)
CFLAGS  += -Wa,-muse-unaligned-vector-move
endif

ifeq ($(OS),Windows_NT)
EXE     := .exe
# no link timestamp in the PE header: the same code links to the same bytes, so Smart App Control
# keeps its verdict on a rebuilt binary instead of blocking a new hash (docs/LESSONS.md #12, #103)
LDLIBS  := -Wl,--no-insert-timestamp $(EXTRA_LDFLAGS)
PY      ?= tools/.venv/Scripts/python.exe
else
EXE     :=
LDLIBS  := -lpthread -lm -ldl $(EXTRA_LDFLAGS)
PY      ?= tools/.venv/bin/python
endif

CORE_SRC := $(wildcard src/base/*.c src/format/*.c src/kernels/*.c src/backend/*.c \
                       src/memory/*.c src/kv/*.c src/tokenizer/*.c src/models/*.c src/gen/*.c)
CORE_OBJ := $(CORE_SRC:%.c=$(BUILD)/%.o)
APP_OBJ  := $(BUILD)/src/app/main.o $(BUILD)/src/app/serve.o $(BUILD)/src/app/bar.o
TEST_BIN := $(patsubst tests/%.c,$(BUILD)/tests/%$(EXE),$(wildcard tests/test_*.c))
BENCH_BIN := $(BUILD)/tests/bench_kernels$(EXE)
MEM_BIN := $(BUILD)/tests/bench_mem$(EXE)
ATTN_BIN := $(BUILD)/tests/bench_attn$(EXE)
EXPF_BIN := $(BUILD)/tests/bench_expf$(EXE)
DISK_BIN := $(BUILD)/tests/bench_disk$(EXE)
# premise benches of 2026-09-24 (docs/MEASUREMENTS.md): built by the gate so they do not rot, run by hand
RESEARCH_BIN := $(foreach b,bench_gpu_attn bench_gpu_q8 bench_attn_bw bench_kvpack bench_expf32 bench_peak,$(BUILD)/tests/$(b)$(EXE))

.PHONY: all attn-probe test check-gcc check-clang check-asan check-tsan check-tiny check-real check-cut oracle tier-check oracle-tokenizer chat-check oracle-real spec-check bench bench-mem bench-attn bench-expf bench-disk lint profile check check-linux clean-machine clean platform-guard quick
all: $(BUILD)/trochilus$(EXE)

# Objects of two platforms must never share a BUILD directory: a build in the container with
# the default BUILD silently overwrites the native ones and the next link mixes them
# (docs/LESSONS.md #52). The stamp says who owns the directory. It is written once, whole (a
# rename): the gate's three lanes run makes on build/linux-gcc at once, and one that truncated the
# stamp to rewrite it could make another read it empty and refuse the directory (docs/LESSONS.md #144).
PLATFORM_TAG := $(if $(filter Windows_NT,$(OS)),windows,$(shell uname -s 2>/dev/null))-$(CC)
platform-guard:
	@mkdir -p $(BUILD)
	@if [ -f $(BUILD)/.platform ] && [ "$$(cat $(BUILD)/.platform)" != "$(PLATFORM_TAG)" ]; then \
		echo "error: $(BUILD) holds objects built by '$$(cat $(BUILD)/.platform)', this is '$(PLATFORM_TAG)'."; \
		echo "       build elsewhere (make BUILD=build/linux-gcc ...) or run 'make clean'."; exit 1; fi
	@[ -f $(BUILD)/.platform ] || { echo "$(PLATFORM_TAG)" > $(BUILD)/.platform.$$$$ && mv -f $(BUILD)/.platform.$$$$ $(BUILD)/.platform; }

$(BUILD)/trochilus$(EXE): $(CORE_OBJ) $(APP_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

ifeq ($(OS),Windows_NT)
# wmain: Windows passes the command line as UTF-16, main.c converts it to UTF-8 (docs/LESSONS.md #13)
$(BUILD)/trochilus$(EXE): LDLIBS += -municode
endif

# Makefile as a prerequisite: a changed flag recompiles everything (docs/LESSONS.md #21)
$(BUILD)/%.o: %.c Makefile | platform-guard
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/tests/%$(EXE): tests/%.c $(CORE_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(CORE_OBJ) -o $@ $(LDLIBS)

# test_cli and test_serve run the command line built beside them (it is not linked into the test)
$(BUILD)/tests/test_cli$(EXE) $(BUILD)/tests/test_serve$(EXE): $(BUILD)/trochilus$(EXE)

# test_bar links the command line's progress bar (src/app/bar.c), which the core does not hold
$(BUILD)/tests/test_bar$(EXE): tests/test_bar.c $(CORE_OBJ) $(BUILD)/src/app/bar.o
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(CORE_OBJ) $(BUILD)/src/app/bar.o -o $@ $(LDLIBS)

# test_base covers src/base and links only it: its bytes then change only with src/base, and the
# native run in `make check` is not blocked anew by every change elsewhere (tools/native_tests.sh)
BASE_OBJ := $(filter $(BUILD)/src/base/%,$(CORE_OBJ))
$(BUILD)/tests/test_base$(EXE): tests/test_base.c $(BASE_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(BASE_OBJ) -o $@ $(LDLIBS)

# test_hot counts the engine's allocator calls by routing them through wrappers (GNU ld and
# lld --wrap; Apple's linker has none, the test then skips the count).
ifeq ($(OS),Windows_NT)
HOT_WRAP := malloc calloc realloc free _aligned_malloc _aligned_free
else ifeq ($(shell uname -s),Darwin)
HOT_WRAP :=
$(BUILD)/tests/test_hot$(EXE): CFLAGS += -DTR_NO_WRAP
else
HOT_WRAP := malloc calloc realloc free posix_memalign
endif
$(BUILD)/tests/test_hot$(EXE): LDLIBS += $(foreach s,$(HOT_WRAP),-Wl,--wrap=$(s))

# a test that loops forever fails instead of hanging the gate (docs/LESSONS.md #35); no timeout(1): no limit
TEST_RUN := $(shell command -v timeout >/dev/null 2>&1 && echo timeout 300)
# TEST_TMP=<dir>: the tests write their temporary files there, not next to their binaries
# (tests/test.h tr_test_tmpdir). The container's targets set it to its own disk: on the Windows
# bind mount the synthetic models cost 100 s of each build's 120 (docs/LESSONS.md #142).
TEST_TMP ?=
test: $(TEST_BIN)
	@set -e; $(if $(TEST_TMP),mkdir -p $(TEST_TMP); export TR_TEST_TMPDIR=$(TEST_TMP);) \
	for t in $(TEST_BIN); do echo "== $$t"; $(TEST_RUN) ./$$t; done; echo "== all C tests passed"

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

# Each group is one run, under 60 s (docs/ARCHITECTURE.md, safety). Native, on a still machine.
bench-mem: $(MEM_BIN)
	./$(MEM_BIN) ram
	./$(MEM_BIN) weights
	./$(MEM_BIN) kv 512
	./$(MEM_BIN) kv 2048
	./$(MEM_BIN) kv 4000

# What the disk gives to a reader of experts (docs/MEASUREMENTS.md question 16): blocks as big as an
# expert matrix at random positions of DISK_FILE, without the system's cache. Under 60 s, reads only.
DISK_FILE ?= models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
bench-disk: $(DISK_BIN)
	./$(DISK_BIN) $(DISK_FILE)

# The attention of a whole prompt on one layer, taken apart: one query at a time against several
# queries per block of keys, with and without the softmax. One run per length and thread count.
bench-attn: $(ATTN_BIN)
	for n in 512 2048 4000; do ./$(ATTN_BIN) $$n || exit 1; ./$(ATTN_BIN) $$n --threads 1 --heads 1 || exit 1; done

# tr_expf on every one of the 2^32 floats, against the correctly rounded value and against the C
# library's expf (what each costs, too): it fails unless tr_expf is correctly rounded everywhere.
# Native and in the container: the two libraries are not the same code, and MinGW-w64's is itself
# correctly rounded everywhere, so on Windows zero differences from the library are asked as well.
ifeq ($(OS),Windows_NT)
EXPF_CHECK := --check --like-library
else
EXPF_CHECK := --check
endif
bench-expf: $(EXPF_BIN)
	./$(EXPF_BIN) $(EXPF_CHECK)

# A diagnostic engine in its own directory that writes every decode token's queries, keys and
# values to files (tools/attn_probe.c, read by tools/attn_skip_report.py). Never the engine.
attn-probe:
	$(MAKE) BUILD=$(BUILD)/probe EXTRA_CFLAGS=-DTR_ATTN_PROBE EXTRA_LDFLAGS=tools/attn_probe.c all

# Scenarios with the engine profiler (bench/scenarios.json), compared with the
# previous run on this machine. See docs/ARCHITECTURE.md §Profiling.
# SCENARIOS=bench/scenarios-olmoe-1b-7b.json for the real model (skipped if not downloaded)
SCENARIOS ?= bench/scenarios.json
profile: $(BUILD)/trochilus$(EXE)
	$(PY) tools/profile_suite.py --binary $(BUILD)/trochilus$(EXE) --scenarios $(SCENARIOS)

lint:
	$(PY) tools/lint.py
	$(PY) tools/route_trace_report.py --check
	$(PY) tools/check_measurements.py

# Tiny OLMoE: transformers reference -> GGUF -> engine, greedy tokens must match exactly.
FIX := fixtures/tiny-olmoe
$(FIX)/ref.json: tools/make_tiny_olmoe.py
	$(PY) tools/make_tiny_olmoe.py --output $(FIX) --force
$(FIX)/model-f32.gguf: $(FIX)/ref.json tools/hf_to_gguf.py
	$(PY) tools/hf_to_gguf.py $(FIX) $@ --type f32
$(FIX)/model-f16.gguf: $(FIX)/ref.json tools/hf_to_gguf.py
	$(PY) tools/hf_to_gguf.py $(FIX) $@ --type f16
$(FIX)/model-q8_0.gguf: $(FIX)/ref.json tools/hf_to_gguf.py
	$(PY) tools/hf_to_gguf.py $(FIX) $@ --type q8_0
# The same model with norm_topk_prob, clip_qkv and rope_theta 500000: the branches the default
# leaves off (docs/LESSONS.md #112).
FIXO := fixtures/tiny-olmoe-opts
$(FIXO)/ref.json: tools/make_tiny_olmoe.py
	$(PY) tools/make_tiny_olmoe.py --output $(FIXO) --force --options
$(FIXO)/model-f32.gguf: $(FIXO)/ref.json tools/hf_to_gguf.py
	$(PY) tools/hf_to_gguf.py $(FIXO) $@ --type f32
oracle: $(BUILD)/trochilus$(EXE) $(FIX)/model-f32.gguf $(FIX)/model-f16.gguf $(FIX)/model-q8_0.gguf \
		$(FIXO)/model-f32.gguf
	$(PY) tools/check_gguf.py $(FIX)/model-f32.gguf --compare-hf $(FIX)
	$(PY) tools/oracle.py $(FIX) $(FIX)/model-f32.gguf --binary $(BUILD)/trochilus$(EXE) --expect exact
	$(PY) tools/oracle.py $(FIX) $(FIX)/model-f16.gguf --binary $(BUILD)/trochilus$(EXE) --expect exact
	$(PY) tools/oracle.py $(FIX) $(FIX)/model-q8_0.gguf --binary $(BUILD)/trochilus$(EXE) --expect report
	$(PY) tools/check_gguf.py $(FIXO)/model-f32.gguf --compare-hf $(FIXO)
	$(PY) tools/oracle.py $(FIXO) $(FIXO)/model-f32.gguf --binary $(BUILD)/trochilus$(EXE) --expect exact

# Every kernel tier end to end, not only the best one this CPU has (tools/tier_check.sh): the model
# tests under TR_CPU_MAX=scalar and avx2, and the logits of the tiny fixtures identical, byte for
# byte, across tiers, thread counts and -b (docs/LESSONS.md #64).
tier-check: $(BUILD)/trochilus$(EXE) $(TEST_BIN) $(FIX)/model-f32.gguf $(FIX)/model-f16.gguf $(FIX)/model-q8_0.gguf
	sh tools/tier_check.sh $(BUILD) $(FIX)

# A real tokenizer against transformers: the model's tokenizer files pinned to a Hub revision
# (downloaded once into fixtures/), a vocabulary-only GGUF written as llama.cpp's converter writes
# it, and, when the model file is there, its own tokenizer metadata must be the same.
TOKFIX := fixtures/olmoe-1b-7b-0125-instruct-tokenizer
$(TOKFIX)/tokenizer.json:
	$(PY) tools/fetch_hf_tokenizer.py --repo allenai/OLMoE-1B-7B-0125-Instruct \
		--revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e --output $(TOKFIX)
$(TOKFIX)/vocab.gguf: $(TOKFIX)/tokenizer.json tools/hf_to_gguf.py
	$(PY) tools/hf_to_gguf.py $(TOKFIX) $@ --vocab-only --tokenizer-pre olmo
oracle-tokenizer: $(BUILD)/trochilus$(EXE) $(TOKFIX)/vocab.gguf
	$(PY) tools/tokenizer_oracle.py --gguf $(TOKFIX)/vocab.gguf --hf $(TOKFIX) --binary $(BUILD)/trochilus$(EXE) \
		--compare-gguf models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf

# trochilus chat on the real model (skipped without it): reusing the cache between turns gives the
# same second reply as a fresh run on the whole conversation.
chat-check: $(BUILD)/trochilus$(EXE) $(TOKFIX)/tokenizer.json
	$(PY) tools/chat_check.py --binary $(BUILD)/trochilus$(EXE) --model models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf --hf $(TOKFIX)

# Real OLMoE weights cut to 2 layers (tools/make_olmoe_2layer_gguf.py, tensors copied
# unchanged from the real Q8_0 file) against transformers on those same weights,
# dequantized (tools/make_olmoe_2layer_ref.py): an exact-reference oracle that does not
# depend on the tiny random fixture. Skipped (message, exit 0) without the real model. The cut lives
# beside the real model: in the container that is the models volume, on ext4, where the run under
# the smallest expert store takes 57 s instead of 306 s over the Windows bind mount (LESSONS #109).
REALFIX := models/olmoe-2layer
REAL_MODEL := models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
# the same model quantized to Q4_K from the Q8_0 by llama-quantize (tools/quantize_q4k.sh)
REALFIX_Q4K := models/olmoe-2layer-q4k
REAL_MODEL_Q4K := models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf
# and in Q4_K_M, llama-quantize's own mix of Q4_K and Q6_K (tools/quantize_q4k.sh m)
REALFIX_Q4KM := models/olmoe-2layer-q4km
REAL_MODEL_Q4KM := models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf
$(REALFIX)/model.gguf: $(REAL_MODEL) tools/make_olmoe_2layer_gguf.py
	$(PY) tools/make_olmoe_2layer_gguf.py $(REAL_MODEL) $@
$(REALFIX)/ref.json: $(REALFIX)/model.gguf tools/make_olmoe_2layer_ref.py
	$(PY) tools/make_olmoe_2layer_ref.py --gguf $(REALFIX)/model.gguf --hf $(TOKFIX) --output $(REALFIX)
oracle-real: $(BUILD)/trochilus$(EXE)
	@if [ ! -f $(REAL_MODEL) ]; then echo "oracle-real: SKIPPED, $(REAL_MODEL) not found"; else \
		$(MAKE) $(REALFIX)/model.gguf $(REALFIX)/ref.json && \
		$(PY) tools/oracle.py $(REALFIX) $(REALFIX)/model.gguf --binary $(BUILD)/trochilus$(EXE) --expect exact --logit-tol 1e-3; \
	fi

# Speculation on the prompt must not move a single token: the same prompt with --spec 0 and with
# every draft size gives the same text. It runs on the whole real model: the 2-layer cut writes
# text that repeats nothing, so the drafter never fires there and the check would prove nothing
# (docs/LESSONS.md #54). The prompt holds the file the model is asked to write again, which is
# where the drafter hits. Skipped without the real model.
spec-check: $(BUILD)/trochilus$(EXE)
	@if [ ! -f $(REAL_MODEL) ]; then echo "spec-check: SKIPPED, $(REAL_MODEL) not found"; else \
		$(BUILD)/trochilus$(EXE) run -m $(REAL_MODEL) -f bench/prompts/code-edit.txt -n 64 --spec 0 \
			> $(BUILD)/spec-0.txt 2>$(BUILD)/spec-0.err || \
			{ echo "spec-check: --spec 0 failed:"; cat $(BUILD)/spec-0.err; exit 1; }; \
		for k in 1 4 8 15; do \
			$(BUILD)/trochilus$(EXE) run -m $(REAL_MODEL) -f bench/prompts/code-edit.txt -n 64 --spec $$k \
				> $(BUILD)/spec-$$k.txt 2>$(BUILD)/spec-$$k.err || \
				{ echo "spec-check: --spec $$k failed:"; cat $(BUILD)/spec-$$k.err; exit 1; }; \
			cmp $(BUILD)/spec-0.txt $(BUILD)/spec-$$k.txt || exit 1; \
			grep -qE "speculation: [0-9]+ passes, [1-9][0-9]*/[1-9][0-9]* drafted" $(BUILD)/spec-$$k.err || \
				{ echo "spec-check: --spec $$k accepted no draft, it proves nothing"; \
				  cat $(BUILD)/spec-$$k.err; exit 1; }; \
		done; \
		echo "== spec-check: same text with --spec 0, 1, 4, 8, 15, and drafts really accepted"; \
	fi

# The gate. Correctness runs in Linux (Docker image trochilus-dev, tools/docker/Dockerfile) so that
# Windows Smart App Control, which blocks freshly built executables for minutes (docs/LESSONS.md #12),
# cannot make it flaky; on Windows the native build is still compiled with 0 warnings.
DOCKER_IMG := trochilus-dev:local
# Before anything else: nothing this project started is still running, or the gate does not
# start (four forgotten load generators ran at 100% under two days of measurements, and a build
# beside a measurement spoils it: docs/LESSONS.md #84, #57); and a script that is told to stop
# takes its children with it (tools/cleanup.lib), seen failing without the trap at every run; and
# a measurement takes the machine's marker and gives back only its own (tools/measure_guard.lib);
# and a comparison stops on a run that measured nothing (tools/ab_modes.sh, docs/LESSONS.md #132).
clean-machine:
	@mkdir -p $(BUILD) && date +%s > $(BUILD)/.check-start
	sh tools/orphans.sh
	sh tools/test_cleanup.sh
	sh tools/test_beside.sh
	sh tools/test_marker.sh
	sh tools/test_ab_modes.sh
ifeq ($(OS),Windows_NT)
check: clean-machine lint
	$(MAKE) WERROR=1 all $(TEST_BIN) $(BENCH_BIN) $(MEM_BIN) $(ATTN_BIN) $(EXPF_BIN) $(DISK_BIN) $(BUILD)/tests/dump_rope$(EXE) $(BUILD)/tests/dump_dequant$(EXE) \
		$(RESEARCH_BIN)
	@# the diagnostic probe's branch of olmoe.c (make attn-probe) still compiles, with 0 warnings
	$(CC) $(filter-out -MMD -MP,$(CFLAGS)) -Werror -DTR_ATTN_PROBE -fsyntax-only src/models/olmoe.c tools/attn_probe.c
	@# the models volume, when it exists, replaces models/ read over the Windows bind mount:
	@# the real-model checks load the same file from ext4 instead of 9p (docs/LESSONS.md #41).
	@# Beside it (tools/beside.sh), the C tests once more on Windows itself: its branches of src/base
	@# never run in the container (docs/LESSONS.md #103, #106); a binary Smart App Control blocks is
	@# SKIPPED, not failed. They share nothing with the container but the processors (they write
	@# beside their binaries, the container in build/linux-* and its own /tmp); their output comes
	@# when the container's gate ends
	MSYS_NO_PATHCONV=1 sh tools/beside.sh $(BUILD)/native-tests.log \
		"env -u MSYS_NO_PATHCONV sh tools/native_tests.sh $(TEST_BIN)" \
		docker run --rm --security-opt seccomp=unconfined -v "$(CURDIR):/src" \
		$$(docker volume inspect trochilus-models > /dev/null 2>&1 && echo "-v trochilus-models:/src/models") \
		-w /src $(DOCKER_IMG) make check-linux
	@# give the Docker VM's file cache back to Windows (docs/LESSONS.md #38)
	MSYS_NO_PATHCONV=1 docker run --rm --privileged $(DOCKER_IMG) sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
	sh tools/check_argv_utf8.sh $(BUILD)/trochilus$(EXE) $(TOKFIX)/vocab.gguf
	@# the decode's attention on this machine's GPU gives the CPU's bytes through the whole engine
	@# (the real model's 2-layer cut; skipped without a GPU, the cut, or with the exe blocked)
	sh tools/gpu_exact.sh quick $(BUILD)/trochilus$(EXE)
	@echo "== check passed in $$(( $$(date +%s) - $$(cat $(BUILD)/.check-start) )) s (native Windows build: 0 warnings, C tests natively; tests, ASan and oracles: Linux)"
else
check: clean-machine lint check-linux
	@echo "== check passed in $$(( $$(date +%s) - $$(cat $(BUILD)/.check-start) )) s"
endif

# The four builds of the C tests own their BUILD directories and share nothing, so they run at
# once (their tests write in a directory of their own, TEST_TMP); -Orecurse keeps each one's
# output in one piece. Then, on build/linux-gcc, three lanes at once: the tiny models, the whole
# real model (chat and speculation, one run after the other: each loads 7 GB; then the tokenizer
# oracle, 4 GB, never beside them), and the real model's 2-layer cut (its two oracles in order,
# they build the cut; 2.3 GB). The peak stays near 10 of the Docker VM's 15 GB: a fourth lane, or
# the tokenizer beside the whole model, could make the engine's memory estimate refuse a run.
# What the lanes share is made first.
NPROC := $(shell nproc 2>/dev/null || echo 4)
check-linux:
	$(MAKE) -j$(NPROC) -Orecurse check-gcc check-clang check-asan check-tsan
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} build/linux-gcc/trochilus \
		$(FIX)/model-f32.gguf $(FIX)/model-f16.gguf $(FIX)/model-q8_0.gguf $(TOKFIX)/vocab.gguf
	$(MAKE) -j3 -Orecurse check-tiny check-real check-cut

check-tiny:
	@# every dequantization the engine has, bit for bit gguf-py's (the reader of our oracles)
	$(PY) tools/check_dequant.py --binary build/linux-gcc/tests/dump_dequant
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle
	@# the same tiny models again, with a store too small to hold every expert (Esperti M1):
	@# TR_EXPERT_BUDGET_MIB=min forces the smallest store that can run, so this exercises eviction
	TR_EXPERT_BUDGET_MIB=min $(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle
	@# proof it really ran non-resident, not just green by luck: the "hits, ... misses" form of the
	@# experts: line only prints when n_slots < n_units (src/app/main.c print_experts)
	TR_EXPERT_BUDGET_MIB=min build/linux-gcc/trochilus generate -m $(FIX)/model-f32.gguf -p 6 -n 4 2>&1 >/dev/null | \
		grep -q "^experts:.*hits,.*misses"
	@echo "== expert budget min evicts on the tiny model"
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} tier-check
	$(PY) tools/profile_suite.py --binary build/linux-gcc/trochilus$(EXE) --smoke
	@# -c is honoured: 6 + 4 tokens fit a context of 16, and do not fit one of 8
	build/linux-gcc/trochilus generate -m $(FIX)/model-f32.gguf -p 6 -n 4 -c 16 > /dev/null
	! build/linux-gcc/trochilus generate -m $(FIX)/model-f32.gguf -p 6 -n 4 -c 8 > /dev/null 2>&1
	@echo "== context flag ok"
	@# decode speed counts evaluations, not tokens: 4 tokens are 3 evaluations (docs/LESSONS.md #40;
	@# tools/speed_compare.py reads this line)
	build/linux-gcc/trochilus generate -m $(FIX)/model-f32.gguf -p 6 -n 4 2>&1 >/dev/null | grep -q "generate: 4 tokens, 3 evaluations in"
	@echo "== speed line ok"

check-real:
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} chat-check
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} spec-check
	@# after the whole model, never beside it: the tokenizer oracle holds about 4 GB
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle-tokenizer
check-cut:
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle-real
	TR_EXPERT_BUDGET_MIB=min $(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle-real
	@# the same cut of the real model quantized to Q4_K (tools/quantize_q4k.sh; skipped without it)
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle-real \
		REAL_MODEL=$(REAL_MODEL_Q4K) REALFIX=$(REALFIX_Q4K)
	@# and in Q4_K_M, where Q6_K meets Q4_K in one model (tools/quantize_q4k.sh m; skipped without it)
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle-real \
		REAL_MODEL=$(REAL_MODEL_Q4KM) REALFIX=$(REALFIX_Q4KM)

check-gcc:
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 TEST_TMP=/tmp/tr-test/gcc test
	@# a race shows up once in many runs: the threaded model test runs 20 times, four at once (the
	@# pools then share the cores, which moves the threads' timing more than runs one by one do),
	@# each run with a directory of its own: they write the same synthetic model (docs/LESSONS.md #143)
	@seq 20 | xargs -P 4 -I{} sh -c 'mkdir -p /tmp/tr-test/hot{} && TR_TEST_TMPDIR=/tmp/tr-test/hot{} \
		build/linux-gcc/tests/test_hot > /dev/null' && echo "== test_hot 20/20"
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 build/linux-gcc/tests/bench_kernels build/linux-gcc/tests/bench_mem \
		build/linux-gcc/tests/bench_attn build/linux-gcc/tests/bench_disk build/linux-gcc/tests/dump_rope \
		build/linux-gcc/tests/dump_dequant
	@# tr_expf is the correctly rounded exp on every float, as gcc and as clang compile it
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 bench-expf
check-clang:
	$(MAKE) BUILD=build/linux-clang CC=clang WERROR=1 TEST_TMP=/tmp/tr-test/clang test
	$(MAKE) BUILD=build/linux-clang CC=clang WERROR=1 bench-expf
check-asan:
	$(MAKE) BUILD=build/linux-asan CC=gcc WERROR=1 \
		EXTRA_CFLAGS="-O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all" \
		EXTRA_LDFLAGS="-fsanitize=address,undefined" TEST_TMP=/tmp/tr-test/asan test
check-tsan:
	@# ThreadSanitizer on the pool; setarch -R: TSan cannot map its shadow memory with full ASLR
	$(MAKE) BUILD=build/linux-tsan CC=gcc WERROR=1 EXTRA_CFLAGS="-O1 -g -fsanitize=thread" \
		EXTRA_LDFLAGS="-fsanitize=thread" build/linux-tsan/tests/test_base build/linux-tsan/tests/test_hot
	TSAN_OPTIONS=halt_on_error=1 setarch $$(uname -m) -R build/linux-tsan/tests/test_base
	TSAN_OPTIONS=halt_on_error=1 setarch $$(uname -m) -R build/linux-tsan/tests/test_hot

# Between edits, NOT the gate: lint, the native build with 0 warnings, and the C tests in the
# Linux gcc build (the container on Windows). About a minute; a step is closed by `make check`.
ifeq ($(OS),Windows_NT)
quick: lint
	$(MAKE) WERROR=1 all $(TEST_BIN)
	MSYS_NO_PATHCONV=1 docker run --rm --security-opt seccomp=unconfined -v "$(CURDIR):/src" \
		-w /src $(DOCKER_IMG) make -j$(NPROC) BUILD=build/linux-gcc CC=gcc WERROR=1 TEST_TMP=/tmp/tr-test/gcc test
	@echo "== quick passed (not the gate: make check)"
else
quick: lint
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 test
	@echo "== quick passed (not the gate: make check)"
endif

clean:
	rm -rf $(BUILD)

-include $(CORE_OBJ:.o=.d) $(APP_OBJ:.o=.d)
