# Commands

From `CLAUDE.md` (2026-09-21): the complete list of project commands. In the map remain
those used every day; here is everything, including measurements and rarely used tools.

```bash
make check               # the gate: lint, build 0 warning, test, ASan, TSan, repeated tests, oracles, C tests natively; prints its time
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
sh tools/ab_speed.sh <gguf> <binary A> <binary B>   # two binaries alternated run by run (LESSONS #46)
sh tools/ab_modes.sh <rounds> "a=<command>" "b=<command>"   # modes of a binary (env, flag), round-robin order, A/A (LESSONS #66)
build/trochilus run ... --decode-threads 8   # force decode threads (default: measured per session)
sh tools/threads_phase.sh sweep | widths | after <binary before>   # threads per phase: -t 4/8/12/16, forced widths
sh tools/decode_context.sh measure | widths | long | change <before>   # decode at 32/512/2048/4000 context: A/A, forced widths, bytes per zone
tools/.venv/Scripts/python.exe tools/decode_context_report.py speed|model|zones <file>   # MEASUREMENTS tables from runs; speed counts choices and changes
sh tools/orphans.sh      # is something of ours still on? make check and measurements don't start
sh tools/busy_machine.sh <n> <command>   # the ONLY way to load the machine: generators die with the script
sh tools/machine_still.sh [limit] [wait] [window]   # occupied processors (who: tools/background_load.ps1): guard for every measurement
sh tools/test_cleanup.sh   # a stopped script loses its children; without cleanup.lib trap the child stays
make bench               # microbenchmark of kernels (median + noise)
make bench-mem           # RAM bandwidth (sequential, sparse), engine matmul, attention on both layouts
make bench-disk          # disk for who reads experts, no system cache (DISK_FILE=<file>)
build/trochilus run ... --route-trace <file>   # routing trace: experts picked and predicted
tools/.venv/Scripts/python.exe tools/route_trace_report.py <trace> | --check   # prediction, LRU cache, streaming
tools/.venv/Scripts/python.exe tools/route_graph_report.py <trace> | --compare | --mask-from   # question 44
sh tools/mask_quality.sh   # experts off: KL and tokens against whole model (measurement only)
sh tools/experts_budget.sh measure | misses | direct   # M1: tok/s at 4 budgets, cost of a token, cache yes/no
build/trochilus run ... --expert-budget <MiB|min>   # expert RAM (default: the plan); TR_EXPERT_BUDGET_MIB in tests
sh tools/mutate_{route,tune,experts,stream}.sh | tools/mutate_reports.py   # in container: the mutations, all red
python3 tools/mutate_auto.py <src/file.c> <test> [test...] [--lines A-B] [--list] [--asan] [--cmd '<shell>']   # in container: generated mutations, survivors listed; --cmd adds a check per mutant (the oracle, for a model file)
sh tools/mutate_files.sh [olmoe kernels ... main prof]   # in container: mutate_auto on each file with its tests and oracles, build/mutate/<name>.txt
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
