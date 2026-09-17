# Trochilus — build. The core needs nothing but libc and OS threads.
#   make check      the gate: lint, native build with 0 warnings, then in Linux (Docker image
#                   trochilus-dev) gcc and clang tests, ASan+UBSan tests and the oracles
#   make            build/trochilus
#   make test       C tests (every kernel variant against scalar, GGUF, pool, profiler)
#   make oracle     tiny models against transformers (needs tools/.venv or the Docker image)
#   make oracle-tokenizer  a real tokenizer against transformers (ids, pieces, NFC, decoding)
#   make bench      kernel microbenchmark (median and noise), native
#   make lint       docs and tables against the lessons in docs/LEZIONI.md
#   make WERROR=1   warnings are errors
# Extra flags without losing the defaults: EXTRA_CFLAGS, EXTRA_LDFLAGS.

ifeq ($(origin CC),default)
CC      := gcc
endif
BUILD   := build
CFLAGS  ?= -O2 -g
# -ffp-contract=off: no silent multiply-add fusion, the scalar kernels define the numbers (kernels.h)
CFLAGS  += -std=c11 -Wall -Wextra -Wshadow -Wno-unused-parameter -ffp-contract=off -MMD -MP $(EXTRA_CFLAGS)
ifeq ($(WERROR),1)
CFLAGS  += -Werror
endif

# gcc stores AVX registers into stack arrays with aligned moves, trusting a 32/64-byte stack
# alignment that MinGW (gcc bug 54412) and ASan's fake stack do not give: the GNU assembler
# rewrites them as unaligned moves, same speed on current CPUs (docs/LEZIONI.md #19).
# clang uses its own assembler and aligns correctly.
ifeq ($(findstring clang,$(shell $(CC) --version 2>/dev/null)),)
CFLAGS  += -Wa,-muse-unaligned-vector-move
endif

ifeq ($(OS),Windows_NT)
EXE     := .exe
LDLIBS  := $(EXTRA_LDFLAGS)
PY      ?= tools/.venv/Scripts/python.exe
else
EXE     :=
LDLIBS  := -lpthread -lm $(EXTRA_LDFLAGS)
PY      ?= tools/.venv/bin/python
endif

CORE_SRC := $(wildcard src/base/*.c src/format/*.c src/kernels/*.c src/backend/*.c \
                       src/memory/*.c src/kv/*.c src/tokenizer/*.c src/models/*.c)
CORE_OBJ := $(CORE_SRC:%.c=$(BUILD)/%.o)
APP_OBJ  := $(BUILD)/src/app/main.o
TEST_BIN := $(patsubst tests/%.c,$(BUILD)/tests/%$(EXE),$(wildcard tests/test_*.c))
BENCH_BIN := $(BUILD)/tests/bench_kernels$(EXE)

.PHONY: all test oracle oracle-tokenizer chat-check bench lint profile check check-linux clean
all: $(BUILD)/trochilus$(EXE)

$(BUILD)/trochilus$(EXE): $(CORE_OBJ) $(APP_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

ifeq ($(OS),Windows_NT)
# wmain: Windows passes the command line as UTF-16, main.c converts it to UTF-8 (docs/LEZIONI.md #13)
$(BUILD)/trochilus$(EXE): LDLIBS += -municode
endif

# Makefile as a prerequisite: a changed flag recompiles everything (docs/LEZIONI.md #21)
$(BUILD)/%.o: %.c Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/tests/%$(EXE): tests/%.c $(CORE_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(CORE_OBJ) -o $@ $(LDLIBS)

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

# a test that loops forever fails instead of hanging the gate (docs/LEZIONI.md #35); no timeout(1): no limit
TEST_RUN := $(shell command -v timeout >/dev/null 2>&1 && echo timeout 300)
test: $(TEST_BIN)
	@set -e; for t in $(TEST_BIN); do echo "== $$t"; $(TEST_RUN) ./$$t; done; echo "== all C tests passed"

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

# Scenarios with the engine profiler (bench/scenarios.json), compared with the
# previous run on this machine. See docs/ARCHITETTURA.md §Profilazione.
# SCENARIOS=bench/scenarios-olmoe-1b-7b.json for the real model (skipped if not downloaded)
SCENARIOS ?= bench/scenarios.json
profile: $(BUILD)/trochilus$(EXE)
	$(PY) tools/profile_suite.py --binary $(BUILD)/trochilus$(EXE) --scenarios $(SCENARIOS)

lint:
	$(PY) tools/lint.py

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
oracle: $(BUILD)/trochilus$(EXE) $(FIX)/model-f32.gguf $(FIX)/model-f16.gguf $(FIX)/model-q8_0.gguf
	$(PY) tools/check_gguf.py $(FIX)/model-f32.gguf --compare-hf $(FIX)
	$(PY) tools/oracle.py $(FIX) $(FIX)/model-f32.gguf --binary $(BUILD)/trochilus$(EXE) --expect exact
	$(PY) tools/oracle.py $(FIX) $(FIX)/model-f16.gguf --binary $(BUILD)/trochilus$(EXE) --expect exact
	$(PY) tools/oracle.py $(FIX) $(FIX)/model-q8_0.gguf --binary $(BUILD)/trochilus$(EXE) --expect report

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

# The gate. Correctness runs in Linux (Docker image trochilus-dev, tools/docker/Dockerfile) so that
# Windows Smart App Control, which blocks freshly built executables for minutes (docs/LEZIONI.md #12),
# cannot make it flaky; on Windows the native build is still compiled with 0 warnings.
DOCKER_IMG := trochilus-dev:local
ifeq ($(OS),Windows_NT)
check: lint
	$(MAKE) WERROR=1 all $(TEST_BIN) $(BENCH_BIN)
	MSYS_NO_PATHCONV=1 docker run --rm --security-opt seccomp=unconfined -v "$(CURDIR):/src" -w /src $(DOCKER_IMG) make check-linux
	@# give the Docker VM's file cache back to Windows (docs/LEZIONI.md #38)
	MSYS_NO_PATHCONV=1 docker run --rm --privileged $(DOCKER_IMG) sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
	sh tools/check_argv_utf8.sh $(BUILD)/trochilus$(EXE) $(TOKFIX)/vocab.gguf
	@echo "== check passed (native Windows build: 0 warnings; tests, ASan and oracles: Linux)"
else
check: lint check-linux
	@echo "== check passed"
endif

check-linux:
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 test
	$(MAKE) BUILD=build/linux-clang CC=clang WERROR=1 test
	$(MAKE) BUILD=build/linux-asan CC=gcc WERROR=1 \
		EXTRA_CFLAGS="-O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all" \
		EXTRA_LDFLAGS="-fsanitize=address,undefined" test
	@# ThreadSanitizer on the pool; setarch -R: TSan cannot map its shadow memory with full ASLR
	$(MAKE) BUILD=build/linux-tsan CC=gcc WERROR=1 EXTRA_CFLAGS="-O1 -g -fsanitize=thread" \
		EXTRA_LDFLAGS="-fsanitize=thread" build/linux-tsan/tests/test_base build/linux-tsan/tests/test_hot
	TSAN_OPTIONS=halt_on_error=1 setarch $$(uname -m) -R build/linux-tsan/tests/test_base
	TSAN_OPTIONS=halt_on_error=1 setarch $$(uname -m) -R build/linux-tsan/tests/test_hot
	@# a race shows up once in many runs: the threaded model test runs 20 times
	@for i in $$(seq 20); do build/linux-gcc/tests/test_hot > /dev/null || exit 1; done; echo "== test_hot 20/20"
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 build/linux-gcc/tests/bench_kernels
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} oracle-tokenizer
	$(MAKE) BUILD=build/linux-gcc CC=gcc WERROR=1 PY=$${PY:-tools/.venv/bin/python} chat-check
	$(PY) tools/profile_suite.py --binary build/linux-gcc/trochilus$(EXE) --smoke
	@# -c is honoured: 6 + 4 tokens fit a context of 16, and do not fit one of 8
	build/linux-gcc/trochilus generate -m $(FIX)/model-f32.gguf -p 6 -n 4 -c 16 > /dev/null
	! build/linux-gcc/trochilus generate -m $(FIX)/model-f32.gguf -p 6 -n 4 -c 8 > /dev/null 2>&1
	@echo "== context flag ok"

clean:
	rm -rf $(BUILD)

-include $(CORE_OBJ:.o=.d) $(APP_OBJ:.o=.d)
