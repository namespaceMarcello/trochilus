# Commands

From `CLAUDE.md` (2026-09-21): the complete list of project commands. In the map remain
those used every day; here is everything, including measurements and rarely used tools.

```bash
make check               # the gate: lint, build 0 warning, test, ASan, TSan, repeated tests, oracles, C tests natively; prints its time
make quick               # between edits, NOT the gate: lint, native build with 0 warnings, the C tests in Linux gcc (~2 min)
make check 2>&1 | awk '{ print strftime("%H:%M:%S"), $0; fflush() }' > build/check.log   # the gate with a time on every line
tools/.venv/Scripts/python.exe tools/gate_times.py build/check.log   # seconds per step of that log (docs/MEASUREMENTS.md §The gate)
sh tools/native_tests.sh build/tests/test_*.exe   # the C tests on Windows itself; SKIPPED where Smart App Control blocks
make                     # build/trochilus (core, no dependencies)
make test                # C tests: kernel SIMD = scalar, malformed GGUF, pool, profiler
make oracle              # tiny models: generate, convert, compare with transformers
make tier-check          # the engine under every tier (TR_CPU_MAX): model test and logits byte-identical
make oracle-tokenizer    # OLMoE tokenizer against transformers: ids, pieces, NFC, decode
make oracle-real         # true OLMoE cut to 2 layers against transformers (skipped without model)
build/trochilus run -m <f.gguf> -f prompt.txt -n 200    # text input, greedy generation
build/trochilus chat -m <file.gguf>                      # conversation with model chat template
make chat-check          # chat on true model: second response = run from zero (skipped without model)
make spec-check          # `--spec 0/1/4/8/15` give same text on true 2-layer model
build/trochilus run ... --spec 8   # speculation from prompt (same tokens)
sh tools/ab_spec.sh <gguf> <binary> bench/prompts/code.txt 8   # how much --spec yields, alternating runs
sh tools/build_llamacpp.sh; tools/compare_llamacpp.py ...   # in container: llama.cpp, logits comparison
tools/speed_compare.py ...   # in container: speed against llama.cpp and colibri (trochilus-models volume)
sh tools/race_llama.sh [runs]   # by the native rules: Trochilus vs llama.cpp, prompts 512 and 2048, 16 and 8 threads, A B B A (~30 min)
sh tools/bench_ggml.sh   # in container: llama.cpp's own matmul kernels alone (ggml API, ref/llama.cpp/build-trochilus), bench_peak's shapes, Q8_0 and Q4_K, mul_mat and mul_mat_id, Q4_K also repacked; 1 and 16 threads, median of 15 (~1 min)
sh tools/bench_expf_refs.sh [threads]   # in container: llama.cpp's exp (ggml_v_expf) and the C library's against tr_expf on every float (how many differ, worst ulp), then ns a value on one core (~1 min)
sh tools/race_q4k.sh [runs] [m]   # by the native rules: the real model in Q4_K against Q8_0, and llama.cpp on the Q4_K; with m, Q4_K_M against Q4_K (~20 min)
MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/src" -v trochilus-models:/src/models -w /src trochilus-dev:local sh tools/quantize_q4k.sh [m]   # models/...-Q4_K.gguf from our Q8_0; with m, ...-Q4_K_M.gguf (Q4_K and Q6_K, llama-quantize's own mix)
tools/.venv/Scripts/python.exe tools/check_dequant.py --binary build/tests/dump_dequant.exe   # every dequantization bit for bit gguf-py's (in make check)
sh tools/ab_speed.sh <gguf> <binary A> <binary B>   # two binaries alternated run by run (LESSONS #46)
sh tools/ab_modes.sh <rounds> "a=<command>" "b=<command>"   # modes of a binary (env, flag), round-robin order, A/A (LESSONS #66); AB_WALL=1 adds the whole run's ms (wall_ms)
build/trochilus serve [-m <gguf>] [-t n] [--expert-budget <MiB|min>] [--idle <min>]   # the engine kept between commands (question 49): generate/logits/run/chat run in it when it answers; TR_SERVER=0: never
build/trochilus serve --status | --stop   # what it keeps, how many requests and loads; stop it
sh tools/serve_first_prompt.sh [rounds]   # question 49: first prompt, new process against a kept store, half and full budget, wall_ms and misses
build/trochilus run ... --decode-threads 8   # force decode threads (default: measured per session)
sh tools/threads_phase.sh sweep | widths | after <binary before>   # threads per phase: -t 4/8/12/16, forced widths
sh tools/decode_context.sh measure | widths | long | change <before> | change-short <before>   # decode at 32/512/2048/4000 context: A/A, forced widths, bytes per zone; change-short, THE DEFAULT for a change: exactness, forced 8 at 512/2048/4000, the profile after (~40 min)
tools/.venv/Scripts/python.exe tools/decode_context_report.py speed|model|zones <file>   # MEASUREMENTS tables from runs; speed counts choices and changes
TR_GPU=0 build/trochilus run ...   # the decode's attention stays on the CPU (default: on the NVIDIA GPU when there is one, same bits; TR_GPU=1: say why a GPU could not be opened)
sh tools/gpu_exact.sh [full|quick] [binary]   # native: GPU against CPU through the engine, byte for byte (full: real model, logits of 2000 positions, tokens after 4000, speculation; quick: the 2-layer cut, in make check); fails if nvidia-smi sees a GPU the engine did not use
build/tests/bench_gpu_attn.exe info | check <probe dir> [--zero-copy] | edge | expf | time --data <probe dir> --n 2048|4000 --only abcezdws   # the GPU attention premise: bits on the probe dumps, edge cases, tr_expf on all 2^32 floats, time per layer (d: the decode's bursts, w: keep-warm); --mutate fma|tree|sumtree|zero|expf must fail
build/tests/bench_gpu_q8.exe info | check [--mutate norn|tree|order] | time [--runs 200] [--gap-us 1000]   # the exact Q8_0 matrix-vector product on the GPU: bits against dot_row, GB/s
build/tests/bench_attn_bw.exe 2048 --runs 19 [variant...]   # why the CPU's decode attention reads below the RAM's speed: variants (engine, head, prefetch, read, read4, mutant) paired step by step, bits checked
build/tests/bench_expf32.exe --slow-all --error --threads 8   # the exact exp in float32 only (for the GPU) against tr_expf on all 2^32 floats, both variants and the AVX-512 / AVX2 tiers, ~40 s; --no-timing for the check alone; sh tools/bench_native.sh bench_expf32 --quick for the ns a value
build/tests/bench_kvpack.exe roundtrip | bits | time [--run <name>|all]   # the KV packed in 28 bits, lossless: round trip, the attention's bits on the probe dumps, time against F32 (KVPACK_PROBE_DIR)
sh tools/bench_native.sh <bench> [args]   # any premise bench of build/tests under the native rules (marker, still machine, load declared, a copy one byte longer); build/bench_native/<bench>/
sh tools/bench_native.sh bench_peak [--runs N] [--one-core] [--q64]   # the no-FMA and FMA peaks (inline asm), the prefill kernel's stream in asm with 4 and 8 tokens, dot_row2_x4 and dot_row2_x8, tr_matmul on OLMoE's shapes with and without x8 (-x8), 1 and 16 threads (question 55); the panel lines (question 63); the tiles 4 x 6 and 3 x 8 as asm streams and as kernels against tr_matmul, byte for byte first (question 64; --q64: only those, the peaks and mix x8, a few minutes)
MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/src" -v trochilus-models:/src/models -w /src trochilus-dev:local sh tools/draft_agreement.sh   # a Q4 draft against the exact model, logits of every position on three real texts (question 53); then tools/draft_agreement_report.py models/draft q4_k
tools/.venv/Scripts/python.exe tools/route_union_report.py build/route/*.bin   # experts of k consecutive tokens against chance, and co-activation per layer (questions 54, 62)
tools/.venv/Scripts/python.exe tools/weights_genome.py <model.gguf> [--tensors <regex>]   # every block once: code and scale entropy, zero blocks, repeated blocks and rows (questions 56, 62; ~25 min on 7 GB)
tools/.venv/Scripts/python.exe tools/kv_repeats_report.py <probe dir>   # which layers' keys and values repeat exactly across positions (question 62)
make attn-probe   # build/probe/trochilus: a diagnostic engine that writes each decode token's q, K, V and output (tools/attn_probe.c); never the engine
TR_PROBE_DIR=<dir> TR_PROBE_KV_AT=<prompt tokens + n> build/probe/trochilus.exe run -m <gguf> -f <prompt> -n <n> -t 8   # the dump: K and V at the last token, q and the output of every token
tools/.venv/Scripts/python.exe tools/attn_skip_report.py <dir> [<dir> ...] [--layers 0,1] [--json f]   # which positions an exact attention could skip (s - max, exact zeros, absorption, high/low halves); first checks its replay gives the engine's bits
sh tools/orphans.sh      # is something of ours still on? make check and measurements don't start
sh tools/test_marker.sh | sh tools/mutate_marker.sh   # the machine's marker (~/.claude/macchina-ferma) taken, waited for, given back; its 9 mutations all red (on a copy of the library)
sh tools/test_ab_modes.sh   # ab_modes stops on a run that measured nothing, with and without AB_WALL (in make check)
sh tools/mutate_bar.sh   # in container: the progress bar's mutations (physics, render, decision, clearing, the load's progress), about 6 min
sh tools/mutate_prefetch.sh   # in container: the read ahead's mutations (store, thread API, prompt), gcc and ASan
sh tools/mutate_q6k.sh   # in container: Q6_K's 14 mutations (scalar, unpack, AVX2, AVX-512, tables, engine, reader), all red; about 5 min
sh tools/mutate_row2.sh   # in container: the two-row kernels' 23 mutations (x4 and x8, the SIMD lane tree, SIMD scales, table, tr_matmul's roads), all red; about 10 min
SET=prefetch sh tools/prefill_overlap.sh [rounds]   # native: the prompt at half budget with and without reading the next layer ahead (TR_PREFETCH=0), 512 and 2048, A/A
TR_BAR=0 build/trochilus run ...   # no progress bar while the model loads (it is drawn only when stderr is a terminal, and not with NO_COLOR or TERM=dumb)
sh tools/busy_machine.sh <n> <command>   # the ONLY way to load the machine: generators die with the script
sh tools/machine_still.sh [limit] [wait] [window]   # occupied processors (who: tools/background_load.ps1): guard for every measurement
sh tools/test_cleanup.sh   # a stopped script loses its children; without cleanup.lib trap the child stays
sh tools/beside.sh <log> '<first>' <second...>   # the first command in the background (output held, shown at the end), the second in the foreground; either failing fails it; the gate runs the native C tests beside the container this way
sh tools/test_beside.sh   # beside.sh's exit codes and output, and no first command left when the second fails (red without the trap); in make check
make bench               # microbenchmark of kernels (median + noise)
sh tools/bench_kernels.sh   # the same by the native rules (marker, still machine, load declared), a one-byte-longer copy; CONTAINER=1 in trochilus-dev; build/bench_kernels/; lines above 10% spread named in noisy.txt; bench_kernels --matrix: the whole-matrix tables only (each type with an x8 kernel also without it, -x8, in turn)
make bench-mem           # RAM bandwidth (sequential, sparse), engine matmul, attention on both layouts
make bench-disk          # disk for who reads experts, no system cache (DISK_FILE=<file>)
build/trochilus run ... --route-trace <file>   # routing trace: experts picked and predicted
tools/.venv/Scripts/python.exe tools/route_trace_report.py <trace> | --check   # prediction, LRU cache, streaming
tools/.venv/Scripts/python.exe tools/route_graph_report.py <trace> | --compare | --mask-from   # question 44
sh tools/mask_quality.sh   # experts off: KL and tokens against whole model (measurement only)
sh tools/experts_budget.sh measure | misses | direct   # M1: tok/s at 4 budgets, cost of a token, cache yes/no
build/trochilus run ... --expert-budget <MiB|min>   # expert RAM (default: the plan); TR_EXPERT_BUDGET_MIB in tests
sh tools/mutate_{route,tune,experts,stream}.sh | tools/mutate_reports.py   # in container: the mutations, all red
python3 tools/mutate_auto.py <src/file.c> <test> [test...] [--lines A-B,C-D] [--changed REF] [--list] [--asan] [--cmd '<shell>'] [--no-coverage]   # in container: generated mutations, survivors, timeouts, memory refusals and mutants on lines no check runs listed (SURVIVED, TIMEOUT, PRESSURE, UNCOVERED: a gcov pass first); --asan: the sanitizers judge the plain build's survivors; --cmd adds a check per mutant (the oracle, for a model file); --changed HEAD: only the lines changed since HEAD; a progress line per mutant on stderr
sh tools/mutate_files.sh [olmoe kernels ... main serve prof gguf experts threads platform]   # in container: mutate_auto on each file with its tests and oracles, build/mutate/<name>.txt, time left in build/mutate/<name>.progress; CHANGED=HEAD sh tools/mutate_files.sh <name>: only the lines changed since HEAD
make bench-attn          # attention on prompt (512/2048/4000) on one layer, broken down by phases, with bit control
make bench-expf          # tr_expf on all 2^32 floats against rounded value and C library
tools/.venv/Scripts/python.exe tools/gen_expf_table.py [--check | --scan]   # tr_expf constants from mpmath (src/kernels/expf_table.h)
sh tools/expf_quality.sh | sh tools/mutate_expf.sh   # in container: tr_expf against old binary (KL) and emulation (byte)
sh tools/platform_bits.sh   # do Windows and Linux give same bytes? logits and RoPE tables of both platforms
sh tools/prefill_context.sh measure | change <binary before>   # prefill at 512/2048/4000 in one session, A/A, logits to byte, zones
tools/.venv/Scripts/python.exe tools/prefill_context_report.py attn|zones <file>   # MEASUREMENTS tables §Prefill on long prompts
make profile             # profiler scenarios, median of N, identical tokens, comparison with previous
make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json   # true model, 16/8/4/1 threads
make profile SCENARIOS=bench/scenarios-decode-context.json   # true model, 32/512/2048/4000 context: ms and bytes per zone
make profile SCENARIOS=bench/scenarios-prefill-context.json  # true model, 512/2048/4000 prompts at 16 threads and 512 at 1 thread
tools/.venv/Scripts/python.exe tools/<script>.py    # on Linux/macOS: tools/.venv/bin/python
```

Before delivering: `make check` green. `make profile` comes with the profiler hooked to the engine.
On Windows correctness runs in Docker (`trochilus-dev:local`, `tools/docker/Dockerfile`):
Smart App Control blocks freshly compiled binaries (LESSONS #12); the C tests run natively too, and
a binary it blocks is reported SKIPPED (LESSONS #103).
