# Done

Archive, newest last. One entry per commit or closed step: `### <date> — <title>`, what was
implemented and how to check it.

### 2026-09-17 — Repository birth
Local repo `Desktop\trochilus` (branch `main`, LF). Apache-2.0 license with `NOTICE`
crediting colibri (Apache-2.0) and ds4 (MIT, text in `licenses/`). Reference sources
pinned as read-only worktrees in `ref/`: colibri `a90bed9`, ds4 `8db1d1d`. Hook that
rejects code commits without an entry here.

### 2026-09-17 — M0 step 1: base, GGUF, converter
`src/base/` (files with pread, memory, time, log; thread pool without OpenMP; CPU
detection at runtime with XGETBV check), `src/format/gguf.c` (GGUF v2/v3 reader that
validates every dimension against the file), `tools/hf_to_gguf.py` and
`tools/check_gguf.py` (gguf-py 0.19.0, OLMoE), commands `trochilus cpu` and
`trochilus inspect`. Check: `make WERROR=1 test`, then `build/trochilus cpu` and
`build/trochilus inspect fixtures/tiny-olmoe/model-q8_0.gguf`.

### 2026-09-17 — GGUF reader safety tests
`tests/test_gguf.c`: valid file read field by field; same file truncated at every byte
(never opens); 22 targeted corruptions (magic, version, huge counts, strings with NUL,
wrong types and sizes, misaligned offsets, outside or overflowing file, duplicate names
and keys), all rejected with a message; 3000 randomly mutated files, all that open have
readable tensors within bounds. Clean under AddressSanitizer + UBSan + leak check (gcc,
Linux). Check: `make test`.

### 2026-09-17 — Kernel microbench and first measurement
`tests/bench_kernels.c`: for each available tier and each kernel (dot_f32, dot_row
f16/q8_0) on rows from 64 to 4096 elements, median of N runs and spread (noise); plus
`tr_matmul` of an OLMoE expert matrix with 1 thread and physical cores. Scalar baseline
recorded in `docs/MEASUREMENTS.md`. Check:
`gcc -std=c11 -O2 -ffp-contract=off tests/bench_kernels.c src/base/*.c src/format/*.c src/kernels/*.c -o build/bench/bench_kernels.exe && build/bench/bench_kernels.exe`.

### 2026-09-17 — Engine profiler
`src/base/prof.h/.c`: fixed zones for each phase (embedding, projections, norms, rope,
KV write, attention, router, experts, lm_head, sampling, disk reads, pool waits),
prefill and decode separate, weight bytes touched per token and effective memory bandwidth.
Lives in the session, costs a jump when off, counts with RDTSC (invariant TSC, calibrated
to system clock). Reports table and JSON. Check: `make test` (`tests/test_prof.c`).
Not yet wired to the OLMoE graph (the step 2 agent is writing it).

### 2026-09-17 — Control cycle
`CLAUDE.md` §control cycle (7 points: before, errors, every error becomes a check, every
scenario enters tests, `make check` before "done", verification after an agent, docs);
`docs/LESSONS.md` with first 13 lessons (cause, prevention, state, found by); commit hook
that rejects `src/` changes without test, oracle, or scenario (except `no-test: <reason>`);
`test_parallel_varying_chunks` in `tests/test_base.c` for two pool races (LESSONS #4).
Check: hook tested on 5 cases (2 rejections, 3 accepted); `make test`.

### 2026-09-17 — Tier 0: tiny OLMoE exact, and the `make check` gate
Scalar kernels (`src/kernels/kernels.c`, contracted to 16 lanes), OLMoE graph
(`src/models/olmoe.c`), architecture register and memory guard (`src/models/model.c`),
commands `generate` and `logits`, `tools/oracle.py`. Oracle against transformers: f32
16/16 tokens (logit error 2.1e-7), f16 16/16 (3.2e-4), q8_0 16/16 (8.2e-3). `make
check`: lint (`tools/lint.py`: control characters in docs, type table against ggml,
`tmpfile` in tests, doc caps, lesson table), Windows build at 0 warnings, then in Docker
`trochilus-dev` gcc and clang tests, tests under ASan+UBSan, oracle. Lint found wrong
iq1_s in ds4 table (fixed). Check: `make check`.

### 2026-09-17 — Profiler wired to OLMoE, scenario suite
`tr_session_prof(tr_session*)` (sole addition to `model.h`, with `prof` entry in
`tr_arch_vtable`): session owns a `tr_prof`, off by default. `src/models/olmoe.c`
times each zone of forward pass (embedding, norms, QKV projections, rope, KV write,
attention, router, four expert steps, final norm, lm_head) and counts weight bytes
touched for each matmul and embedding; POOL_WAIT and WEIGHT_READ not yet wired.
`trochilus generate` has `--profile` (table to stderr), `--profile-json <file>` and
`-p <n>` (synthetic deterministic prompt, no tokenizer); prefill and decode separate,
sampling in `sample` zone. `tools/profile_suite.py` (`make profile`) runs
`bench/scenarios.json` (tiny f32/q8_0 model) multiple times, median and spread per
phase and zone, comparison with previous measurement on same machine (improved /
degraded / noise); `--smoke` (in `make check-linux`, after oracle) verifies that
`--profile` does not change generated tokens. `tests/test_model_prof.c`: tiny OLMoE
hand-built (no external fixture) proves profiler is off by default and turning it on
does not change logits (bit for bit) versus twin session without profiler. Check:
`make check`, then `make profile`.

### 2026-09-17 — Flag -c, full context as error, OLMoE-1B-7B downloaded
`trochilus generate -c <token>` limits KV cache (to profile real model with little memory);
context that fills before the end now exits with code 3 instead of 0 (found by new gate
test). Hook against backslashes in shell writes. Downloaded
`models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf` (AllenAI, 7.36 GB, sha256 verified). Check:
`make check`; `build/trochilus inspect models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf`.

### 2026-09-17 — AVX2 and AVX-512 kernels for dot_f32 and dot_row q8_0
`src/kernels/kernels_x86.c`: AVX2 and AVX-512 variants of two hot kernels, chosen at
startup by CPU (AVX-512, then AVX2, then scalar; `TR_CPU_MAX` limits them), bit-identical
to scalar. Common parts in `kernels_internal.h`; `tr_rmsnorm` uses active tier.
`test_kernels` compares every tier with scalar on all queues and special values, and
verifies catching a deliberately wrong variant. q8_0 kernel 10× faster on one thread;
OLMoE-1B-7B Q8_0 from 6.9 to 21.1 tok/s (8 threads), same tokens as scalar
(`docs/MEASUREMENTS.md`). `Makefile`: with gcc aligned AVX instructions become
misaligned (crash under ASan and on MinGW), and a changed flag recompiles everything.
Check: `make check`, `make bench`,
`build/trochilus generate -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -p 32 -n 16 -c 128 -t 8 --profile`.

### 2026-09-17 — Hot path, pool with spinning, repeated measurements
Hot path marked in code (`/* hot: begin */`): `tools/lint.py` rejects allocations,
strings, I/O and per-element math, with self-test of the check; `tests/test_hot.c`
counts allocations while model generates (zero) and verifies identical logits with 1/2/3/8
threads. New thread pool: one slot per thread, spin for 2 ms then sleep (dispatch at 16
threads from 53 to 1.5 µs on Windows). Gate adds ThreadSanitizer, `test_hot` 20 times,
and benchmark compilation. `profile_suite.py`: `context` field, min and max, identical
tokens across runs and threads; real model scenarios in `bench/scenarios-olmoe-1b-7b.json`.
Check: `make check`, `make bench`, `make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json`.

### 2026-09-17 — Single-thread part: rope from table, activations and attention in parallel
Rope: cos/sin table per position built at session creation (`tr_rope_table`), forward no
longer does trigonometry. Experts staged (gate+up of all, then activations of all as one
parallel job `tr_swiglu(pool, ...)`, then down, then weighted sum). Attention: one head
at a time in `tr_attention_head` (weighted sum with new kernel `axpy_f32`, scalar + AVX2 +
AVX-512), heads in parallel with one score row per thread. Real model logits identical to
the bit before. Check: `make check`; `trochilus logits` on same tokens with binary before
and after, `cmp` the files; `make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json`.

### 2026-09-17 — BPE tokenizer from GGUF metadata, exact against transformers; `trochilus run` with text
`src/tokenizer/`: tokens, types, and merges read from GGUF (merge becomes id pair →
rank and result), added tokens sought leftmost-longest per first byte, NFC and classes
`\p{L}` `\p{N}` `\s` from tables generated by probing HF `tokenizers` 0.22.2
(`tools/gen_unicode_tables.py`), GPT-2 rules played on codepoints, BPE with priority
queue (n log n). Families allowed only with oracle: today `olmo`. Commands
`trochilus tokenize` (also batch, pieces, decode) and `trochilus run` (text in, greedy
generation out). Converter: vocabulary as `convert_hf_to_gguf.py` in llama.cpp and
`--vocab-only` mode. Windows: arguments in UTF-8 (`wmain`), stdout binary. Lint: no
mutable static variables without `global-ok`. Tests with timeout. Check: `make check`
(C tests `test_tokenizer` and `test_unicode`, `make oracle-tokenizer`: 20,745 tests
with sweep of all codepoints, 0 differences on ids, pieces, normalized text, and decode;
real GGUF metadata identical); `trochilus run -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -f prompt.txt -n 120`.

### 2026-09-17 — `trochilus chat`: conversation with model template
`src/tokenizer/chat.c`: GGUF chat template recognized by its exact bytes (length and
FNV-1a) and rendered in C; unknown template rejected (today: OLMoE-0125-Instruct).
`tr_session_rewind` in model: each turn chat renders entire conversation, keeps already-
cached tokens same, and computes only rest. Windows console read in UTF-16
(`tr_stdin_line`), characters split across two tokens printed whole. Commands
`trochilus chat` and `chat-template` (for oracle). Check: `make check` (`test_session`:
logits identical to the bit after rewind; `make oracle-tokenizer`: 600 conversations
equal to `apply_chat_template`, text and tokens; `make chat-check`: on real model second
chat reply identical to `run` on entire conversation); by hand:
`build/trochilus chat -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf`. Without `-c`, if
RAM is short for 4096 tokens chat halves the context and prints it (LESSONS #37).

### 2026-09-17 — Comparison with llama.cpp on real model (tools)
`ref/llama.cpp` (commit `b49650a`, reference only) compiled in container by
`tools/build_llamacpp.sh`, with `tools/llamacpp_logits.c` (logits per position in same
format as `trochilus logits`, greedy generation). `tools/compare_llamacpp.py` compares
tokenization, greedy, and logits per position (best word, KL, margins). Results in
`docs/MEASUREMENTS.md`. Check: `sh tools/build_llamacpp.sh` and
`tools/compare_llamacpp.py ... --prompt bench/prompts/dante.txt` in container. `make
check` on Windows now empties Docker VM cache at the end.

### 2026-09-17 — Transformers oracle on real model cut to 2 layers
`tools/make_olmoe_2layer_gguf.py` copies first 2 layers of real GGUF (tensors byte for
byte, only `block_count` changes); `tools/make_olmoe_2layer_ref.py` builds OLMoE in
transformers with GGUF config and dequantized weights, writes greedy tokens and logits
per position of two prompts (27 and 1024 tokens). `tools/oracle.py` also reads multi-
prompt format (`--logit-tol`). `make oracle-real` enters `make check` (skipped without
model): identical tokens, logits within 1e-3 (measured 2.4e-4). Check: `make check`, or
in container `make BUILD=build/linux-gcc CC=gcc oracle-real`. Numbers in
`docs/MEASUREMENTS.md`.

### 2026-09-17 — Speed comparison with llama.cpp and colibri
`tools/speed_compare.py`: same threads and same lengths for Trochilus (`generate`),
llama.cpp (`llama-bench`, decode at same context depth) and colibri (`olmoe` on its int8
conversion), median of N runs; with `--tok-file` also tokenizer speed (Trochilus,
`llama-tokenize`, HF `tokenizers`). `tools/build_llamacpp.sh` also compiles `llama-bench`.
`trochilus generate` now prints tokens and evals and divides by evals; `make check`
verifies it. Numbers in `docs/MEASUREMENTS.md`. Check, in container with models in
`trochilus-models` volume:
`tools/speed_compare.py --model /models/<gguf> --trochilus build/linux-gcc/trochilus --llama-bench
ref/llama.cpp/build-trochilus/bin/llama-bench --threads 16,8 --prompt 32 --gen 32`.

### 2026-09-17 — Batch prefill in exact C
`tr_session_eval` runs in passes of at most `n_batch` tokens (default 512;
`tr_session_create` takes `n_batch`, `-b` in `generate`/`run`/`chat`/`logits`). Per pass:
K/V of all tokens, attention per (head, token); router per token and counting sort of
(token, expert) pairs per expert; `tr_matmul_grouped` (new, `kernels.c`) for gate/up/down,
and `tr_matmul` that visits tokens in blocks of 16 against each weight row; sum of experts
per token in id order; logits of last token only. Decode is the pass from one token.
512-token prompt at 16 threads: 30 → 197 tok/s (with kernel below), decode unchanged,
logits identical to bit to binary before. Check: `make check` (`tests/test_prefill.c`: 90
cases f32/Q8_0, n_batch, calls, threads, rewind; `test_kernels`: empty groups and rows
split across threads; `tools/oracle.py`: `logits -b 3/64/all` to byte on tiny and OLMoE
at 2 layers); speed with `tools/speed_compare.py ... --prompt 512`.

### 2026-09-17 — Prefill profile, 4-token kernel, alternating comparisons
`tools/profile_suite.py` reports zones in prefill too and accepts `batch` in a scenario;
`bench/scenarios-olmoe-1b-7b.json` has scenarios `prefill512-t16/8/4/1`. Profile shows
prefill is 90% in multiplications (attention 2%, serial 8%), so kernel `dot_row_x4`
(scalar, AVX2, AVX-512): one weight row against 4 tokens in registers, one load and one
conversion per four products, each identical to the bit to its `dot_row`. `TR_MATMUL_TILE`
can be changed from compile line. `tests/bench_kernels.c` measures `dot_row q8_0 x4` and
`tr_matmul` with 64 tokens; `tests/test_kernels.c` compares x4 with `dot_row` on every
SIMD tier. New `tools/ab_speed.sh`: two binaries alternated run by run, because laptop
heating fakes sequential comparisons (LESSONS #46). Check: `make check`, `make bench`,
and `sh tools/ab_speed.sh models/<gguf> build/base/b/trochilus build/linux-gcc/trochilus`
in container.

### 2026-09-17 — Why prefill didn't scale: where Windows puts threads
Measured on native (in Docker topology is not real, LESSONS #47): clock under load from
Windows counter and prefill with process affinity pinned, cases alternated every round.
16-thread prefill was not limited by power (clock drops 6-8% from 1 to 16 threads) nor by
CCD (8 threads split 4+4 on two chiplets go 7-9% better than 8 on one alone): scheduler
was putting two of the 16 threads on the same physical core. One thread per physical core
is worth +30% (134 → 174 tok/s at 2048-token prompt) and 8 to 16 cores prefill yields
2.02×. Closed questions 20 and 2 of `docs/MEASUREMENTS.md`, half of 3; opened 22 (Linux,
decode, machine occupied). Check: `docs/MEASUREMENTS.md` §Where threads go has tables,
method, and numbers.

### 2026-09-17 — Speculative decoding from prompt, bit-exact
`tr_session_eval_rows` keeps logits of last n positions of a pass instead of only last
(`tr_session_logits_back`), and each row is identical to bit to logits that token gives
alone. Above are `src/gen/lookup.c` (n-gram of tail sought backward in context, proposes
continuation of its last occurrence; 4 to 2 tokens, no state, no allocations) and
`src/gen/greedy.c` (`tr_greedy_step`: emits already-chosen token, verifies 1 + k
positions in one pass, keeps tokens model would choose anyway and rewinds with
`tr_session_rewind` on rejected ones). `generate` and `run` have `--spec <draft>` and
print how many drafts were accepted. Sources and what was taken from each: `docs/ORIGINS.md`
§Speculative prompt. Check: `make check` (new `tests/test_spec.c`: same tokens with draft
1..15, on two vocabularies because model vocabulary would never reject anything, LESSONS #50;
new `make spec-check` on real model cut to 2 layers: same text with `--spec 0/1/4/8/15`),
and `sh tools/ab_spec.sh <gguf> <binary> bench/prompts/code.txt 8` for speed at alternating
runs. Measured on full model (`docs/MEASUREMENTS.md` §Speculative prompt): 1.42× rewriting
file already in prompt (64% drafts accepted), 0.62× writing new code (13%), break-even
around 15%. `--spec` stays off by default until adaptive draft. `make` now refuses to mix
objects of two platforms in same folder (LESSONS #52).

### 2026-09-18 — Pool threads pinned to physical cores
`src/base/cpu.c` builds slot list (`tr_cpu_info.slot`, `n_slots`): one logical processor
per physical core before each SMT sibling, and cores taken in round-robin across last-level
caches, because 4+4 on two chiplets beats 8 on one. On Windows topology comes from
`GetLogicalProcessorInformationEx` (cores and L3 cache), on Linux from `topology/` and
`cache/index3/id`, on macOS not available (thread cannot be pinned: `n_slots` stays 0 and
nothing pins). List respects mask already imposed on process (`taskset`, `start /affinity`),
and then core count drops to usable ones too, so pool does not size for cores it will never
run on. `src/base/platform.c` adds `tr_thread_pin` (Windows `SetThreadGroupAffinity`,
which reaches beyond first processor group — llama.cpp doesn't, UPSTREAM #3; Linux
`sched_setaffinity`) and `tr_thread_affinity_restore`. In `src/base/threads.c` each worker
pins itself to its slot as soon as it starts, calling thread takes slot 0 and returns where
it was when pool dies (a process creating pools of different sizes — benchmarks — must not
stay stuck on first core). `TR_POOL_PIN` chooses mode: 0 nothing, 1 one logical processor
per thread, **2 entire physical core (default)**. Mode matters: pinning thread to processor
costs 17% on decode, pinning to its core does not (LESSONS #58). Check: `make check`,
where `tests/test_base.c` asks OS (`GetCurrentProcessorNumber`, `sched_getcpu`) which
processor each chunk ran on and demands it is assigned slot — with `TR_POOL_PIN=0` that
test is red. Speed: `sh tools/ab_speed.sh <gguf> build/pin/on.sh build/pin/off.sh` and
`docs/MEASUREMENTS.md` §Thread pinning. Sources and what was taken: `docs/ORIGINS.md`
§Thread placement.

### 2026-09-18 — Adaptive draft for `--spec`
`src/gen/greedy.c` holds `k_cur`: starts at `n_draft`, after partial rejection drops to
what was actually accepted, after draft fully accepted rises by one up to `n_draft`, and
after draft **entirely** wrong speculation pauses for a few steps, with pause that doubles
while probe from one token keeps being wrong (1, 3, 7, 15, max 16). Pause is the important
half: on MoE one more row costs 15-27 ms vs ~35 of a pass, because draft token chooses
other experts and pass also reads their weights, so break-even is at 60-75% drafts accepted
(`docs/MEASUREMENTS.md` §Adaptive draft, LESSONS #59). Result: 1.01× when model invents
(was 0.60× with fixed draft) and 1.15× when recopy file already in prompt (1.34× with fixed
draft, which stays available with `--spec-fixed`). Empty draft (lookup found nothing to
propose) changes nothing. Policy chosen (`tr_draft_policy`): adaptive by default,
`--spec-fixed` returns to fixed draft for measurements. Stats line also prints average
draft per pass. Tokens stay identical to those without speculation, with every policy: it
is invariant of `tests/test_spec.c` and `make spec-check`, which now runs on full model
and demands accepted drafts (LESSONS #54). New tests: `k_cur` drops below `n_draft` on
every true rejection and, starting from 1, rises to `n_draft` on repeating context
(LESSONS #55), and after entirely wrong draft next step must propose nothing — checked
every step, and suite fails if no case ever fully rejected a draft, so test cannot pass
for wrong reason. Check: `make check`, and `sh tools/ab_spec.sh <gguf> <binary> bench/prompts/code.txt 8`
for speed; numbers in `docs/MEASUREMENTS.md` §Adaptive draft.

### 2026-09-18 — What would lever 2 give (int8 activations with VNNI): nothing
Question 21 closed with a measurement, without writing kernel. `tests/bench_kernels.c` now
has int8 × int8 candidate with `_mm256_dpbusd_epi32` (same structure as ggml) alongside
our dots: on same row does 11.4 G elements/s vs 11.3 of our float dot, while our `dot_row_x4`
(one weight row vs 4 tokens) does 30.6. Candidate stays in benchmark, does not become
kernel: 8-bit activations are not the same number. Check: `make bench`, rows `dot q8_0xq8_0`
and `quantize x q8_0`; numbers and reads in `docs/MEASUREMENTS.md` §Int8 activations with VNNI.

### 2026-09-17 — A measurement that doesn't measure stops
`tools/ab_speed.sh` and `tools/ab_spec.sh` stop at first run that doesn't produce tok/s
line and print last lines of that run: before they printed table of medians on empty file,
and silent measurement looked like successful measurement (LESSONS #56).

### 2026-09-18 — Adversarial review of entire repository
Full re-read by another model (Fable 5.1): kernels, prefill, speculation, tokenizer, pool,
GGUF, platform, command line, tools, and docs. Outcome: invariants hold; three errors fixed,
each with red test before and green after; one measurement conclusion reversed; two checks
and one tool new (LESSONS #60-#68, `docs/MEASUREMENTS.md` §Review).
- `src/gen/greedy.c`: adaptive draft pause cap used old value, so 15 doubled to 31. Check:
  `build/.../tests/test_spec` (`test_adaptive_pause_is_capped`).
- `src/base/threads.c`: caller's prior affinity lives in thread with count of live pools
  (two pools destroyed in birth order left it on core 0), and worker without slot resumes
  that affinity instead of inheriting pin (Linux). Check: `tests/test_base`
  (`test_pool_caller_affinity_any_order`, `test_pool_oversubscribed`), also under TSan.
- `src/base/cpu.c`: a `TR_CPU_MAX` that is not a tier is a log warning, no longer silent.
- `make tier-check` (`tools/tier_check.sh`, inside `check-linux`): engine under `scalar`
  and `avx2`, and tiny model logits identical to byte across tiers, threads, and `-b`.
- `tests/bench_kernels.c`, section "one row, by": int8 VNNI with 4-token kernel structure
  (1.8-2.3× our float) and float at 8 tokens (0.79× at n=2048), alternated. Check:
  `make bench`.
- `tools/ab_modes.sh`: compare modes of same binary with order rotation and A/A check.
  Check: `sh tools/ab_modes.sh 3 "a=<command>" "a2=<same command>"`.
- Fixed comments that said opposite of code: `lookup.h`, `threads.h`, `threads.c`.

### 2026-09-18 — Native remeasurement with A/A check
No code changed: `sh tools/remeasure.sh` (14 minutes, machine stopped, 8 rounds, order
rotation, A/A) on three conclusions that were within spread, plus profiler zones for
question 12. Worst case `--spec` **0.953×** (not 1.01×) and best case 1.175×; −17%
on decode of processor pin not reproduced (difference between two pins is in prefill,
+9%); decode at 8 threads 1.09-1.12× vs 16. One draft row costs 13.7-17.6 ms out of 31.3,
for 61-91% in experts. Numbers in `docs/MEASUREMENTS.md` §Review, decisions in
`docs/STATUS.md`, LESSONS #58, #59, #66, #67. Check: `sh tools/remeasure.sh`, then
`build/remeasure/` (every run and medians).

### 2026-09-18 — Threads per phase: prompt on full pool, decode on width session measures
Question 26 closed. Measure (native, machine stopped, 8 rounds, A/A, contexts 512 and 2048):
prefill wants 16 threads, decode 8 (4 wins slightly at 512 and loses at 2048, 12 does not
yield). Number not hardcoded in engine: each session measures it. After vs before: decode
**1.085×** at 512, **1.02-1.03×** at 2048, prefill unchanged, `--spec 8` worst case 1.064×;
tokens identical to bit. Numbers in `docs/MEASUREMENTS.md` §Threads per phase, decision in
`docs/STATUS.md`, LESSONS #69-#72.
- `src/base/threads.{h,c}`: `tr_pool_set_active(p, n)` and `tr_pool_active`: following
  `parallel_for` uses first n threads (first n slots: distinct cores on two chiplets), others
  sleep. Check: `tests/test_base` (`test_pool_active`: 5000 width changes, also under TSan).
- `src/models/model.{h,c}`: every eval passes through `session_eval` (hot path, now under
  `tools/lint.py`): long pass on full pool; short pass (up to `TR_DECODE_ROWS` = 4 rows) on
  decode width, which session measures on first 9 single-token passes (full pool, half, quarter;
  widest within 1% of fastest; again every 1024) or that `tr_model_set_decode_threads` forces.
  `TR_DECODE_ROWS` in environment shifts the boundary for measurements (0: old engine). Check:
  `tests/test_phase` (widths and choice on fake times, sequence of widths on real session,
  forced width 1..8 and beyond, every logit identical to single thread, f32 and Q8_0; red
  before, green after, remeasure tried by mutation).
- `src/app/main.c`: `--decode-threads <n>` in `generate`, `run`, `chat`, `logits`; after
  speed lines `threads: 16 prompt, 8 decode (measured)` (or `forced`); `decode_threads` in
  profile JSON. Check: `build/trochilus generate -m <gguf> -p 64 -n 24` and same with
  `--decode-threads 8`.
- `tools/tier_check.sh`: `test_phase` under `scalar` and `avx2`, and logits identical to byte
  also with measured width (5 threads) and with `--decode-threads 2`. Check: `make tier-check`.
- `tools/threads_phase.sh` (`sweep`, `change <binary before>`, `after <binary before>`): waits
  for exe blocked by Smart App Control instead of recompiling, stops and restarts containers,
  waits for 12 GiB free. `tools/ab_modes.sh` collects `width` column (width each run measured
  itself). Check: `sh tools/threads_phase.sh change build/trochilus-before.exe`.
- Every `tools/*.sh` has body inside `main()` called from last line: script modified while
  running does not break anymore (LESSONS #69); `tools/lint.py` demands it.
  `tools/remeasure.sh` also waits for free memory (LESSONS #72). Check: `make lint`.

### 2026-09-19 — Decode at long context: RAM bandwidth, and KV with head positions in a row

Point 6 of next steps and question 4. Measured first (RAM ~54 GB/s; decode lost with context
because KV read at 32-36 GB/s, in jumps), then exact lever: decode 1.06-1.10× at context
512, 1.12-1.14× at 2048, 1.15-1.18× at 4000, prefill 1.39-1.43× at 4000, logits identical
to byte. Numbers in `docs/MEASUREMENTS.md` §Decode at long context.

- `src/kv/kv.{h,c}` (new layer): `tr_kv`, cache `[layer][KV head][position][head_dim]`, K and V in
  two blocks; `tr_kv_bytes` for memory guard, `tr_kv_init`/`tr_kv_free`, `tr_kv_keys` and
  `tr_kv_values` (position 0 of a head; position t is `t * head_dim` floats further),
  `tr_kv_write` (one layer pass, head by head; rewrites after rewind). Hot path under
  `tools/lint.py`. Check: `tests/test_kv.c` (layout, writes across multiple passes,
  rewrite, last element; no other float touched), red before (not there) and green after,
  and by mutation (`tools/mutate_kv.sh` in container: 5 wrong indices out of 5 found,
  three from `test_kv`, all from oracle).
- `src/models/olmoe.c`: session holds a `tr_kv`; attention calls `tr_attention_head` with
  stride `head_dim` and offset 0 on keys and values of its KV head (`h / group`): same
  calls on same floats. Check: `make check` (test_prefill, test_spec, test_session,
  test_phase, oracles, `tier-check`, `spec-check`), and on real model the `exact` stage
  below.
- `src/base/prof.{h,c}`: bytes read **per zone** (`tr_prof_count(p, zone, weights, disk)`,
  `tr_prof_count_kv` for KV read by attention: every position, once per head and per token),
  `kv_bytes_read` per phase; table prints MiB per token and GB/s of zone, JSON `bytes` per
  zone and `kv_bytes` per phase. Experts count `gate_up` and `down` separate. Check:
  `tests/test_prof.c`, `tests/test_model_prof.c` (KV of 3-token pass and one token at
  position 3, `qkv_proj` weights, sum of zones = phase totals);
  `build/trochilus generate -m <gguf> -p 512 -n 48 --profile`.
- `tests/bench_mem.c`, `make bench-mem` (compiled at 0 warnings in `make check`, native
  and Linux): `ram` (sequential and scattered reads in 2 MiB, 256 KiB, 4 KiB blocks,
  1-16 threads), `weights` (engine matmul on random expert matrices), `kv <positions>`
  (one token of attention on both layouts, real kernel and read-only). Prints compile
  date and time (LESSONS #24). Each group under 60 s.
- `tools/profile_suite.py`: `decode_threads` key in scenarios; per zone ms per token, MiB
  per token and GB/s; KV read per token and weight + KV traffic. `bench/scenarios-decode-context.json`:
  context 32, 512, 2048, 4000, measured width and 8 forced (16 forced at two long contexts).
  Check: `make profile SCENARIOS=bench/scenarios-decode-context.json`.
- `tools/decode_context.sh` (`measure`, `change <binary before>`; `PROF_BEFORE=<binary>`
  also profiles it): stops and restarts containers, waits for blocked exe and 12 GiB free;
  `exact` stage (logits to byte before vs after on 600 positions, one token per pass and
  passes of 64, and token after 4000-token prompt: if they differ no measure); four
  contexts in **one** `ab_modes.sh` session, 16 modes in de Bruijn order, each mode with
  its A/A copy; then `bench_mem` and profiles. `tools/decode_context_report.py speed | model | zones`
  makes MEASUREMENTS tables from run files. Check: `sh tools/decode_context.sh change build/trochilus-before.exe`
  (105 minutes).
- `tools/ab_modes.sh`: `AB_GUARD`, a command executed before every run; if it fails
  measurement stops (exit 3). `decode_context.sh`, `threads_phase.sh` and `remeasure.sh`
  set it to "no containers on" (LESSONS #73). Check: `AB_GUARD=false sh tools/ab_modes.sh 1 "a=true" "b=true"`.

### 2026-09-19 — Prefill on long prompts: grouped attention, work per token on pool, F32 rows with tier kernel

Point 4 of next steps, questions 7 and 30. Measured first (softmax, i.e. C library `expf`,
is 51-72% of prompt attention; repeated read of keys and values between 1% and 21-39% at
4000 tokens, per run, nothing below; 8% of prefill ran on single thread), then three exact
levers. Numbers in `docs/MEASUREMENTS.md` §Prefill on long prompts. Prefill **1.05-1.08×
at 512, 1.07-1.08× at 2048, 1.11-1.14× at 4000** (A/A 2.1%), decode indistinguishable
(A/A 2.4%), logits identical to byte on real model (600 positions one token per pass,
passes of 64, 4000-token prompt at passes of 512 and 100). Our `expf`, the big lever
remaining, not written: Marcello decides (question 37).

- `src/kernels/kernels.{h,c}`, `kernels_x86.c`: `tr_attention_group` (attention of a group
  of consecutive tokens of a head: blocks of `TR_ATTN_BLOCK` = 64 positions against all
  group queries, then softmax of each row, then values in blocks; every output identical
  to bit to `tr_attention_head`, which stays as definition) and two new kernels in table,
  `dot_f32_x4` (one query against 4 keys, one accumulator per key in named register) and
  `axpy_f32_x4` (4 values summed to one output, in order, one load and one output write
  per 4): scalar (definition: 4 calls), AVX2, AVX-512 with masked queues. **F32** weight
  rows (router) use tier kernels (`dot_row` = `dot_f32`, `dot_row_x4` = `dot_f32_x4`)
  instead of scalar loop. Check: `tests/test_kernels.c` (x4 vs scalar and vs 4 calls of
  own tier, every length up to 200 and queues; 525 groups vs `tr_attention_head` query
  per query, scores included, with sentinels beyond output and beyond score row; every
  tier must have its own kernels), under every tier in `make tier-check`.
- `src/models/olmoe.c`: attention runs in groups of `OLMOE_ATTN_QUERIES` = 16 tokens per
  head (decode is group of one: one path only), with 16 score rows per worker at stride
  non-multiple of 4 KiB; work of one token alone (embedding, norms, q and k norms, RoPE,
  KV write, residual sum, router choice with one scratch per worker, rows copied for
  experts, expert sum) divides on pool per token, at least `OLMOE_TOKENS_PER_CHUNK` = 8
  per piece: short pass stays on calling thread. Profiler counts KV bytes per group (the
  positions seen by group's last token), no longer per token. Check: `tests/test_prefill.c`
  (141-token prompt, context 160, passes up to 141: 108 cases identical to one token per
  pass), `tests/test_hot.c` (passes of 1, 4, and 20 tokens, also under ThreadSanitizer),
  `tests/test_model_prof.c` (bytes of 3-token pass, 20-token, and 5-token at position 20),
  `make check`.
- `tools/mutate_prefill.sh` (in container; with one argument only mutations containing it):
  20 plausible errors (17 in three levers, 3 on tier used), 20 seen by tests; two critical
  races seen only by ThreadSanitizer. One mutation was not seen (score computed beyond
  query context): test now puts sentinels after the row (LESSONS #80).
- `tests/bench_attn.c`, `make bench-attn` (0 warnings in `make check`, native and Linux):
  attention of full prompt on one layer, one query at a time (all, no softmax, products
  only, softmax only, weighted sum only), in groups with today's kernels, in groups with
  x4, and engine kernel (`grp full`); hash of all outputs, groups must give bits of
  "one query at a time"; `--threads`, `--heads`, `--group`, `--block`, `--only`.
  `tests/bench_mem.c kv` calls `tr_attention_group` as engine. `tests/bench_expf.c`,
  `make bench-expf` (native and in container: two C libraries are not same code): cost
  of `expf` and comparison with correct rounding on all 2^32 floats, 13 s at 16 threads.
- `tools/prefill_context.sh` (`bench`, `measure`, `change <binary before>`;
  `TROCHILUS=<binary>` measures engine that is not `build/trochilus.exe`, LESSONS #81):
  like `decode_context.sh`, for prefill at 512, 2048, 4000 in **one** session (order
  puts each length after every other, A/A copies), `exact` stage extended to 4000-token
  prompt at passes of 512 and 100 on 16 threads, `bench_attn`, profile on
  `bench/scenarios-prefill-context.json` (512, 2048, 4000 at 16 threads and 512 at one
  thread: how much of a zone does not divide). `tools/prefill_context_report.py attn | zones`
  makes tables; `tools/decode_context_report.py speed` also reads `p512` labels. Check:
  `sh tools/prefill_context.sh change build/trochilus-before.exe`.
- **Tier is used** (LESSONS #78, fourth time of green that doesn't see branch):
  `tests/test_tier_used.c`, in `make test` and under `TR_CPU_MAX=scalar` and `avx2` in
  `make tier-check`. Table: every hot-path entry of every tier is its own function, for
  every weight type engine accepts. Engine: one model per type (F32, F16, Q8_0, router
  F32 beside) runs with active table wrapped in counters (`tr_kernels_set_active`, tests
  only, in `kernels_internal.h`) and products counted per type are exactly rows × tokens.
  Red before (F16 scalar in every tier), green after **F16 kernels** of AVX2 (F16C) and
  AVX-512 (`dot_row`, `dot_row_x4`: exact conversion, bit-identical, 49× over scalar in
  `make bench`); `tests/synth_olmoe.h` also writes F16 models and with F32 router
  (`synth_f32_router`). Three new mutations in `tools/mutate_prefill.sh`: two seen only
  by this test. General rule in `CLAUDE.md`. Check: `make test`, `make tier-check`,
  `make bench`.
- **Our `expf`: prepared, not written** (Marcello decides; MEASUREMENTS §Prefill on long
  prompts, point 7). `tests/bench_expf.c` tries a candidate given at compile
  (`-DTR_EXPF_CANDIDATE=<function>`) on all 2^32 floats against library and reference;
  without candidate tries a sketch scalar that lives only in bench (0 differences from
  MinGW, 3.7 ns vs 30, 8 arguments on slow path); `--hard <file>` writes boundary cases
  and `tools/expf_hard_cases.py` recalculates them at 200 bits with mpmath (369 cases,
  0 reference errors, on both platforms). `tools/expf_quality.sh` (container, models
  volume) compiles measurement-only engine with every `expf` correctly rounded
  (`tools/cr_expf_emul.h`) and compares it to normal binary on real model
  (`tools/expf_quality.py`): average KL 3.9e-13, 0 different tokens in 1000. Engine
  not touched. Check: `make bench-expf`; `sh tools/expf_quality.sh` in container.
- `tools/measure_guard.lib` and `tools/stay_awake.ps1`, used by `prefill_context.sh`,
  `decode_context.sh`, `threads_phase.sh` and `remeasure.sh` (LESSONS #82): one
  measurement at a time (`build/.measuring.lock` with pid; second exits with 5, dead
  script's lock is retaken) and machine kept awake while lock exists (power request, no
  settings changed). Check: two `sh tools/prefill_context.sh bench` together, second
  refuses.

### 2026-09-19 — Scalar `tr_expf`, cleanup in scripts, remeasurement at clean machine

- **`tr_expf`** (`src/kernels/expf.c`, `src/kernels/expf_table.h` generated by
  `tools/gen_expf_table.py` with mpmath): exp(x) correctly rounded to float on every
  float, scalar, no C library calls inside. Table of 64 values of 2^(j/64), ln2/64 in
  two pieces, degree-5 polynomial, rounding test at 2^-50, 8 exceptions calculated at
  200 and 400 bits (found by `gen_expf_table.py --scan`, numpy mirror of fast path), NaN
  for untested rounding. `tr_softmax` and `tr_swiglu` use it; `tr_expf_path` and
  `tr_expf_exception` tell tests which path an argument takes. Check: `make bench-expf`
  (all 2^32 floats vs reference; on Windows also vs MinGW's `expf`: 0 and 0), `make test`
  (`tests/test_expf.c`: edges, fast path sampled, every exception, softmax and SiLU vs
  definition), `sh tools/mutate_expf.sh` in container (17 mutations out of 17 found).
- **In gate**: `make check` runs `make bench-expf` with gcc and clang; `tools/lint.py`
  rejects `expf(` and `exp(` in hot path, compares `expf_table.h` to its generator,
  demands cleanup trap in every `tools/*.sh` and rejects pipes to `tee` of what can fail;
  `clean-machine` (`tools/orphans.sh`, `tools/test_cleanup.sh`) first of all.
- **Quality and platforms**: `sh tools/expf_quality.sh` in container (binary before,
  emulation build from before-commit sources pulled from git, binary after: KL and
  tokens, and same bytes as emulation or exits 1); `sh tools/platform_bits.sh` (logits
  of Windows and Linux to byte on fixture and 2-layer real model; RoPE tables with
  `tests/dump_rope.c` and `tools/rope_table_compare.py`, skipped if Smart App Control
  blocks tool).
- **Cleanup** (LESSONS #84): `tools/cleanup.lib` in every script (descendants killed by
  EXIT trap, also native tree on Windows; INT and TERM become `exit 130`;
  `cleanup_run` for long steps, because shell waiting for foreground child doesn't serve
  signals); `tools/orphans.sh` and `tools/orphans.ps1`; `tools/busy_machine.sh`;
  `tools/test_cleanup.sh`. Check: `sh tools/test_cleanup.sh`; leave `yes` on then
  `sh tools/orphans.sh` or `make check`.
- **Machine interrogated** (LESSONS #85): `tools/machine_still.sh` and `tools/cpu_busy.ps1`
  (busy processors, before session and before each run, in `MEASURE_AB_GUARD`),
  `tools/background_load.ps1` and `measure_declare` (background load and `System`'s share
  in every measurement log); `prefill_context.sh` writes `binaries.sha256` itself and
  accepts `GEN_EXTRA` (decode at forced width in before/after comparisons). Check:
  `sh tools/machine_still.sh 3.0 0 5` at quiet machine and with three `yes` on from
  `sh tools/busy_machine.sh 3 sh tools/machine_still.sh 3.0 0 5`.
- **Remeasurement**: `threads_phase.sh widths` and `decode_context.sh widths` (decode
  forced at 16 vs 8 threads on four contexts), and `decode_context_report.py speed`
  that counts choices and width changes. Check: commands in MEASUREMENTS §Remeasure at
  clean machine, at machine left alone.

### 2026-09-19 — Decode width estimator, rewritten

- **What**: width of short passes no longer chosen by "widest within 1%" (was random:
  LESSONS #88, question 31). `src/models/model.c`: `tr_decode_tune_widths` gives widths
  from narrowest and doesn't drop below 4 threads; `tr_decode_tune_stats` gives each width
  its center (fastest pass) and noise (second's distance); `tr_decode_tune_pick` keeps
  narrowest within noise of pair (it and fastest) and asks more passes on both, up to 6,
  when it decides margin; `tr_decode_tune_debounce` changes only with two agreeing
  measures. Session remeasures at every doubling of context (from 32, recalculated from
  position: a `rewind` lowers it) and after 128 passes if a change waits for second vote.
  `tr_session_decode_history` holds what each measure decided, and `threads:` line prints
  it: `threads: 8 prompt, 4 decode (measured), choices 26:4 38:4(8) 76:8` (position:width in
  use, in parens choice debounce held). `tools/decode_context.sh long`: 1500 tokens after
  1000-token prompt, each run's changes counted from that history (`tools/long_switches.awk`).
- **How to check it**: `tests/test_phase.c` (in `make check`): pure functions on hand-
  written times, session with fake clock (`tr_session_set_tune_clock`), every branch with
  its counter. Saw red: before rewrite flat case with wait "for narrowest" (`0 != 2`);
  after, ten mutations, in container: `MSYS_NO_PATHCONV=1 docker run --rm -v
  "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_tune.sh` (half hour;
  `-e ONLY=history` for two of history). Line: `build/trochilus generate -m fixtures/tiny-olmoe/model-f32.gguf
  -p 20 -n 100 -t 8`. **On real model not yet measured**: validation is point 0 of
  `docs/STATUS.md`, at night at quiet machine.

### 2026-09-20 — M1, before writing code: routing trace and disk bench

- **What**: `trochilus run --route-trace <file>` (`src/models/olmoe.c`, `model.h`:
  `tr_session_route_trace_begin`, `tr_session_route_trace`) records per token and layer
  the experts chosen and two predictions of layer after (`pred_in` before layer experts,
  `pred_out` after layer done); off changes not a byte and touches nothing hot. 
  `tools/route_trace_report.py` derives questions 13-15 (prediction, LRU cache and pin,
  streaming per layer). `tests/bench_disk.c` / `make bench-disk`: reads without system
  cache, in blocks as big as expert matrix, 1-16 readers (question 16). New prompt
  `bench/prompts/code-1000.txt` (904 tokens). Closed as "no" questions 5, 34, 39; opened
  41 (disk a third of card) and 42 (first layer).
- **How to check it**: `tests/test_route.c` in `make check` (predictions exact on two
  synthetic models made for purpose, trace same for every pass shape, logits same with
  trace on); five red mutations: `MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src"
  -w /src trochilus-dev:local sh tools/mutate_route.sh`; `tools/.venv/Scripts/python.exe
  tools/route_trace_report.py --check` (in `make lint`). Numbers: in container
  `build/linux-gcc/trochilus run -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -f
  bench/prompts/code-1000.txt -n 300 -t 8 --decode-threads 8 --route-trace build/route/code-1000.bin`,
  then report on that file; `make bench-disk` native at quiet machine. Results in
  `docs/MEASUREMENTS.md` §M1, before writing code.

### 2026-09-20 — M1, batches 1 and 2: experts pass through store with RAM budget

- **What**: `src/memory/experts.{h,c}`: units (layer, expert) in slots allocated at load,
  direct index and O(1) LRU list, reads on-demand on calling thread, part sizes per layer.
  `src/models/olmoe.c`: dense stays in RAM, experts asked to store after router
  (`olmoe_refresh_experts`, off hot path), GGUF stays open for model lifetime, read
  error fails evaluation and puts `pos` back where it was. Auto plan
  (`tr_expert_budget_plan`), `tr_model_load_budget`, `--expert-budget <MiB|min>` and
  `TR_EXPERT_BUDGET_MIB`, `experts:` line after `threads:`. One path only: full budget
  store fills at load. `tools/lint.py` now rejects CRLF line ends (LESSONS #95).
- **How to check it**: `tests/test_experts.c` (contents to byte, LRU hand-calculated,
  whole layer at minimum, resident, injected errors, 800 calls vs reference model) and
  `tests/test_stream.c` (logits identical to byte across resident, minimum, intermediate,
  with and without pool; store bytes vs `tr_gguf_read_range`; read error mid-prompt; plan
  on fake numbers), in `make check`, where oracles also run under `TR_EXPERT_BUDGET_MIB=min`;
  mutations `tools/mutate_experts.sh` (10) and `tools/mutate_stream.sh` (5), all red.
  Real model in container: `trochilus logits` on 32 positions, resident vs
  `--expert-budget min` (72 units of 1024): `cmp` identical. No speed measurement yet
  (batch 3: direct reads).

### 2026-09-20 — Question 44: is code a small, deterministic graph? Trace v2 and expert mask

- **What**: routing trace also carries id of each token and router margin;
  `tr_model_set_expert_mask` / `--expert-mask <file>` turns off experts (measurement only:
  output no longer of model, and `expert mask:` line says so); `tools/route_graph_report.py`
  (coverage, static graph, repetitions, per-id table, margins, `--compare`, `--mask-from`
  also random); `tools/mask_quality.sh` (KL and tokens vs whole model); `tools/mutate_reports.py`;
  prompts `bench/prompts/trace-{c2,py,sh,prose-it,prose-en}.txt`. Results in
  `docs/MEASUREMENTS.md`.
- **How to check it**: `tests/test_route.c` (token ids, margins exact vs same model with
  one more expert per token, mask never chosen and per-layer, empty and removed = model);
  `tools/.venv/Scripts/python.exe tools/route_graph_report.py --check` and
  `tools/mutate_reports.py` (12 red mutations). In container: `trochilus run ... --route-trace`,
  then report on file; `sh tools/mask_quality.sh` from Git Bash. **Not yet redone after
  these changes**: `make check` and `tools/mutate_route.sh` entirely (stopped for memory 20/09).

### 2026-09-20 — M1 batch 3: experts read without system cache

- **What**: `tr_file_open_direct` and `tr_file_alignment` in `src/base/platform.{h,c}`; store
  reads aligned to 4096 without copies (sector margin per side in slot, offset recorded
  on each fill) and `tr_experts_stats` says if direct; `olmoe_load` opens second handle,
  tries aligned probe read and falls back to normal if filesystem refuses; `TR_EXPERT_DIRECT=0`
  and `TR_MEM_AVAILABLE_MIB` for measurements; `experts:` line says `direct` or `buffered`.
  New for measurements: `tools/experts_budget.sh`, `tools/experts_steady.awk`, store
  counters in `tools/ab_modes.sh`.
- **How to check it**: `tests/test_experts.c` runs every case with both alignments and has
  case of two experts with different remainders in same slot; `tests/test_stream.c`
  compares direct and normal to byte (resident and minimum) and forces tight plan branch;
  `tests/test_base.c` tries short read in last sector. Fifteen red mutations and control line green:
  `MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh
  tools/mutate_experts.sh`. All in `make check`.

### 2026-09-21 — M1 measured (four sessions), question 44 closed, one check on MEASUREMENTS

M1 measurements on real model, machine stopped: `experts_budget.sh measure | misses | long | direct`
(`build/experts_budget/`) and `mask_quality.sh` on three missing texts (`build/mask/quality.txt`).
Numbers and conclusions in `docs/MEASUREMENTS.md` §M1 measured and §Behavior on code, decisions in
`docs/STATUS.md`. In short: M1's cost is the **prompt** (each session re-reads whole model, ~4.3 s disk),
one generated token costs 0.03-0.6 units, long-generation decode is 0.87-0.90× of resident model, and
system cache would bloat prefill 2.42×. Question 14's simulation (22.4 units per token) answered another
question: LESSONS #98.

New: `tools/experts_budget.sh long` (200 vs 1000 generated tokens, context 1600) with `experts_steady.awk`
parametrized (`-v short_n= -v gap=`); `tools/check_measurements.py`, hooked to `lint` so to `make check`:
a MEASUREMENTS line giving a number from simulation or model-to-time must carry tag "model, not measure"
(or "model beaten by measure" if history), and a question marked with first tag cannot be struck as closed.

- **How to check it**: `tools/.venv/Scripts/python.exe tools/check_measurements.py` (green; remove a tag
  from one MEASUREMENTS tagged line fails it, and striking question 43 fails it with other rule).
  Measurements repeat with same commands: take an hour or two of machine stopped, and each session
  declares background load in log.

### 2026-09-21 — Prefill under budget reads model once per pass (question 47)

New `tools/prefill_overlap.sh` and `tools/prefill_overlap_report.py`: prefill split into its
two halves (disk wait, from `weight_read` zone, and compute) at prompt 512 and 2048, with
resident model and 50% budget, plus mode with single pass (`-b 2048`). Median of 5 rounds
plus one warmup, round-robin order, machine stopped, guards of other native measurements.

The count found something else: at 2048 tokens store reads 22,880 MiB, 3.5× the expert
table, once per 512-token pass. With single pass: 6,273 MiB and 11.27 s vs 23.51 (2.09×),
compute unchanged, last logit row identical to byte. Numbers and levers in
`docs/MEASUREMENTS.md` §Prefill reads model once per pass, priority in `docs/STATUS.md`,
lesson #99. Engine code not touched.

- **How to check it**: `TROCHILUS=<binary> sh tools/prefill_overlap.sh 5` (~20 minutes,
  machine stopped; results and profiles in `build/prefill_overlap/`). Exactness between
  two forms: `trochilus logits ... -b 512` and `-b 2048` on same prompt, last line of
  both files compared with `cmp`.

### 2026-09-21 — Project map, glossary, and CLAUDE.md brought back into measure

`docs/status.json` holds project state as data: stages M0-M6, one block per piece with state
(done / in progress / next / open), what it means in simple words, measured numbers with date,
linked MEASUREMENTS questions, files, commands and dependencies. `tools/status_html.py` makes
one page (columns per stage, arrows between blocks, panel on click), published as artifact
`6mx3NS4KtQrBFPRLkAYupr`. `docs/glossary.json` explains from scratch 38 keywords (LRU, KV,
prefill, quantize, tier, oracle, mutation, KL...): in panel every recognized word becomes
clickable and opens bubble with related terms.

`CLAUDE.md` was over caps (169 lines, 12.1 KB): full commands passed to `docs/COMMANDS.md`
without deleting any, map keeps daily ones, and returned to 136 lines / 8.5 KB, stamped with
preferences' mark. In "before each commit" table is now the line that keeps map alive: closed
step or decision also written in `docs/status.json`, and artifact regenerates and republishes
at same URL.

- **How to check it**: `tools/.venv/Scripts/python.exe tools/status_html.py` writes
  `build/stato/index.html` (should say how many blocks are done and how many glossary entries);
  `node ~/.claude/hooks/misura-claude-md.cjs CLAUDE.md` for map caps.

### 2026-09-21 — Layer-major order in prefill: 1.89× under budget, experts read once per prompt

Layer body extracted into `forward_layer(m, s, L, tokens, n_tok, pos0, x)` and embedding
and logits in their `forward_embed` / `forward_logits`; `forward_pass` calls them in
same order as before, with `x = s->x`. New `forward_prompt_layer_major`: embedding of
each block in its slice of `s->x_all`, then for each layer all prompt passes, then
logits. `s->x_all` ([n_ctx][n_embd]) allocated at session creation and only when expert
store is partial (asked by `tr_experts_get_stats`), counted in memory guard: nothing
allocated on hot path. `olmoe_eval` takes new path with `x_all != NULL && n > n_batch && trace == NULL`;
full budget and with `--route-trace`, stays as before.

Measure (`sh tools/prefill_overlap.sh 5`, machine stopped): at 2048-token prompt and 50%
budget, **6,273 MiB read instead of 22,880 and 12.41 s instead of 23.51 (1.89×)**, compute
indistinguishable. At 512 nothing changes. Numbers in `docs/MEASUREMENTS.md` §Prefill reads
model once per pass.

- **How to check it**: `make check` (`once_per_prompt` case of `tests/test_stream.c`
  counts units read for 36-token prompt in 12-token passes and wants max `n_units + n_slots`;
  red before fix with 33 on 16). On real model: `sh tools/prefill_overlap.sh 5`, results in
  `build/prefill_overlap/report.txt`.

### 2026-09-21 — Generation's fixed cost is store, not warm-up (question 46)

`sh tools/experts_budget.sh long` with also **full budget** (200 vs 1000 generated tokens,
context 1600, two rounds, machine stopped after closing open chat): with resident model
time per token doesn't improve from 200 to 1000 (34.64 → 33.48 tok/s, spread 4-7%), under
budget yes (29.21 → 30.79 at 50%, 26.53 → 29.84 at 25%). Fixed cost of ~0.35 s and ~0.84 s
exists only when reading from disk: it is LRU resettling after prompt, not kernels warming
nor decode counter. Numbers in `docs/MEASUREMENTS.md` §M1 measured point 3, question 46
narrowed to "where the rest goes" (misses explain ~115 ms out of 840).

- **How to check it**: `sh tools/experts_budget.sh long` (~30 minutes, machine stopped;
  first `sh tools/orphans.sh` must be clean). Results in `build/experts_budget/steady-long.txt`.

### 2026-09-22 — `README.md` in English: where we are, what's missing, what we boast of

First document of repo for those not working here (docs/ stay in Italian). The axis is what
Marcello asked: **what we did, what we will, where we reach**, with goal up front —
*democratize local AI*: no minimum requirements page (scalar is definition, SIMD and GPU
are accelerators), nothing to install, machine decides itself, RAM doesn't decide which
models you use, and smaller machine doesn't give worse answer (same tokens changing threads,
`-b`, tier, budget).

Sections: why; what we did (working + measured numbers + what we boast of, five items each
with proof nearby); what we will (stages M0-M6 with true state and next steps in order);
where we want to reach (four things we don't have: size unbound from RAM, any GPU or none,
small weights without silent losses, always zero options); what's missing no excuses; build;
map of `docs/`; **what we read** (colibri, ds4, llama.cpp as primary sources with ideas
taken one by one, plus transformers/`tokenizers`/OLMoE; only two code files ported); license.

**Out of README, Marcello's choice** (2026-09-22): no comparison with other engines and
nothing that opens flank — out go reports with HF tokenizer and llama.cpp, out old speed
comparison, out lines on what three engines don't try and on llama.cpp's AI policy. Found
defects and PRs sent to colibri and ds4 stay out: they are in `docs/UPSTREAM.md`, README
doesn't name them.

Two DeepSeek documents in `docs/` (third-party reports) enter `.gitignore`: not ours and
must not land in repo that can become public.

- **How to check it**: `cat README.md`; numbers cited are in `docs/MEASUREMENTS.md`
  (§Prefill on long prompts, §Decode at long context, §M1 measured, §Speed — Trochilus vs
  llama.cpp) and in `docs/STATUS.md`; `git check-ignore -v docs/DeepSeek_V41_Tech_Report.md`
  must answer.

### 2026-09-22 — Round logo and README header

Logo (bird on crocodile's head, `Desktop\logo trochilus.jpg`) cropped in a disk: circle
sized to **maximum distance of ink from center**, not bounding box, else tail and snout stay
out (subject arranged diagonally). Rendered at 2048 px and reduced to 512 for clean edge,
transparent outside disk. In `assets/logo.png`; variants with ring and tight crop in
`build/logo/`. README title centered below logo, with note logo is on first iteration, and
at top note README too is and will be refined and thinned.

- **How to check it**: open `assets/logo.png`, and README on GitHub.

### 2026-09-22 — English file names across repository

Marcello: no file in repository carries Italian name, and contents follow in English. This
step is names and every reference to them; contents are next step.

Renamed with `git mv`: `docs/ARCHITETTURA.md` → `ARCHITECTURE.md`, `STATO` → `STATUS`,
`MISURE` → `MEASUREMENTS`, `LESSONS` → `LESSONS`, `ORIGINI` → `ORIGINS`, `COMANDI` → `COMMANDS`,
`docs/archivio/FATTO.md` → `docs/archive/DONE.md`, `stato.json` → `status.json`,
`glossario.json` → `glossary.json`, `tools/check_misure.py` → `check_measurements.py`,
`tools/stato_html.py` → `status_html.py`, and two hooks
(`documenta-prima-del-commit.cjs` → `document-before-commit.cjs`,
`niente-barre-in-scritture-shell.cjs` → `no-backslashes-in-shell-writes.cjs`).

97 tracked files rewritten to point at new names: sources, tests, tools, `Makefile`,
`CLAUDE.md`, `.claude/settings.json` and hooks themselves. **`bench/prompts/` and
`bench/results/` deliberately left alone**: those bytes are input and output of a
measurement, not a reference, and editing them would invalidate every comparison made with them.

Two things the sweep taught us, both now in LESSONS: `pathlib.write_text` on Windows turns
every `\n` into `\r\n` (97 files came back with CRLF, caught by lint), and rename alone pushed
`docs/STATUS.md` past its 40 KB cap, because English names are longer than Italian ones.

- **How to check it**: `tools/.venv/Scripts/python.exe tools/lint.py` is green, and
  `git grep -l 'STATO.md|LESSONS.md|MISURE.md|docs/archivio'` matches only `bench/prompts/`.

### 2026-09-22 — The documents move to English (four done, two half done)

Marcello: everything in the repository is written in English; with him the conversation stays in
Italian. `CLAUDE.md` now carries that rule, in place of "documents in `docs/` in Italian".

**Finished and verified**: `CLAUDE.md` (rewritten by hand), `ARCHITECTURE.md`, `ORIGINS.md`,
`COMMANDS.md`, `STATUS.md`, `archive/DONE.md`, `UPSTREAM.md`. **Half done**: `LESSONS.md` and
`MEASUREMENTS.md` — the headings of both are English, and in MEASUREMENTS these sections still
have an Italian body: Open questions, Speculation from prompt, Adaptive draft, Kernel: one weight
row vs 4 tokens, Speed levers, Adversarial review, Threads per phase, Decode at long context,
Prefill on long prompts, Clean machine remeasure, `tr_expf`, M1 before writing code, M1 measured,
Prefill reads model once per pass, Is code behavior a small graph, Attempts. Stopped there on
purpose: the two Haiku agents were eating the usage.

**A translation is verified, not believed** (LESSONS #100): the first agent reported "complete"
and had translated the headings and one third of the bodies. The check compares the translated
file with the one in git and fails on anything that is not prose: numbers (6770 → 6770 in
MEASUREMENTS), table rows (618 → 618), fenced blocks, heading count, and it counts the Italian
words still in the file. Everything committed here passed it; one wrong term found that way,
`evicted hits` → `decayed hits` in ARCHITECTURE (it is a decaying score, not an eviction).

Cross-references follow the headings: `§Zona calda` → `§Hot path`, `§Profilazione` → `§Profiling`,
`§Esecuzione` → `§Execution` and the ORIGINS sections, rewritten in the Makefile, in the sources,
in the tests and in the tools. `LEZIONI #n` → `LESSONS #n` everywhere. The `§` references that
point at MEASUREMENTS sections are already English, because that file's headings were translated
first.

- **How to check it**: `tools/.venv/Scripts/python.exe tools/lint.py` is green;
  `git grep -n 'LEZIONI'` matches only `bench/prompts/`, which is measurement input and is never
  edited.
- **What is left**: the Italian bodies of `LESSONS.md` and `MEASUREMENTS.md`, one section at a
  time, and then the line in README that still calls the documents Italian.

### 2026-09-22 — Review of the reader, the expert store, the pool and the platform; a faster, native gate
A review of `src/format/gguf.c`, `src/memory/experts.c`, `src/base/threads.c`, `src/base/platform.c`
(Opus 5.5) and of the gate itself. Seven errors fixed and two discoveries, each with a check (LESSONS #102-#110):
- `tr_file_pread` on a direct handle failed at the end of the file on NTFS (error 87 on the retry
  from an unaligned offset): an expert ending the file could not be read on Windows (#103);
- generating a token allocated on Windows: MinGW's emulated TLS mallocs on a thread's first touch
  (#106); the pool's slots shared cache lines (#104); the GGUF arena gave 8-byte alignment, not 16
  (#105); the profiler's clock was calibrated on one unguarded pair of reads (#108);
- the gate was red on `main` since the translation (#102), and the C tests had never run natively.

The gate: the C tests run natively on Windows too (`tools/native_tests.sh`, SKIPPED where Smart App
Control blocks), links are reproducible (`-Wl,--no-insert-timestamp`) so a rebuilt binary keeps its
verdict, the four compiler builds run at once, the tiny-model steps beside the real model's, the
2-layer cut lives on the models volume, `test_cleanup.sh` no longer waits out its orphan, and the
gate prints its time. `tools/mutate_auto.py` generates mutations for any file; `test_gguf`,
`test_experts`, `test_base`, `test_prof` gained the cases its survivors asked for.

- **How to check it**: `make check` (green, "check passed in N s", "native tests: 18 passed");
  `python3 tools/mutate_auto.py src/format/gguf.c test_gguf --asan` in the container.

### 2026-09-23 — Review of the model, the kernels, the tokenizer and the command line
A review of `src/models/olmoe.c`, `src/kernels/`, `src/tokenizer/`, `src/app/main.c` and the printed
report of `src/base/prof.c` (Opus 5.5), every file through `tools/mutate_auto.py` (LESSONS #111-#116):
- the command line sized its buffers from `-n` and `-p`: near 2^62 the size wrapped to a few bytes
  and the tokens ran past them in `generate`, `run` and `chat`; `--tokens` cast its ids to int32
  (`4294967297` became token 1); `--profile-json` wrote control characters raw (#111).
  `tests/test_cli.c` runs the binary on a synthetic model that carries a tokenizer and OLMoE's chat
  template (`synth_with_tokenizer`), on Linux, under ASan and natively on Windows;
- three OLMoE options had never met transformers: `norm_topk_prob`, `clip_qkv` and a `rope_theta`
  read from the file. `fixtures/tiny-olmoe-opts` (`make_tiny_olmoe.py --options`) is in `make
  oracle`: 16/16 tokens, logits within 2.1e-7 (#112). `tests/test_model_load.c` covers malformed
  files one fault at a time, the bounds of `eval` and `eval_rows`, the expert mask, the router's tie
  rule, an expert part in another type than its gate, and the edges of the route trace; a failed
  eval keeps the logits the session had (`test_stream`);
- the profiler's printed table is pinned on four profiles with known numbers (`test_prof`); the
  kernel dispatch too: a tier comes under its own name, only when the CPU has it, the fastest is
  chosen, and the norm and the attention go through the active table (`test_tier_used`);
- `mutate_auto.py` runs extra checks per mutant (`--cmd`: the oracles) and counts a kill only when
  the check fails twice (`FLAKY` otherwise: memory pressure in the Docker VM had killed mutants no
  test could see, #115); `tools/mutate_files.sh` knows the checks of every file and sizes its jobs to
  the VM's memory.

- **How to check it**: `make check`; in the container `sh tools/mutate_files.sh olmoe` (or any other
  name, docs/COMMANDS.md), survivors in `build/mutate/<name>.txt`, named in `docs/MEASUREMENTS.md`
  §Generated mutations.
- **What is left**: the survivors of `tokenizer.c` (75) and `main.c` (334), to read one by one; the
  numbers of the command line are parsed with `atoi`/`atoll` (`-n abc` is 0, not an error).

### 2026-09-23 — The tokenizer's survivors read, and a mutation tool that stops hiding them

- `tests/test_tokenizer.c` refuses every malformed tokenizer by its own message (22 files: tokens
  missing, not strings or empty, types not int32 or out of range at both ends, merges not strings,
  a merge missing one side or making an unused token, add_eos without its id, a caller with no
  error buffer) and accepts the edges (token types 0 and 6, id 0 as a merge side and result, bos
  and eos 0, a merge making a user-defined token); new cases for byte 0, a code point one past the
  byte alphabet, `'re` and its rule at a segment's end, a 64-byte piece, an empty control token
  against every byte value. The synthetic file counts its keys as it writes them (#113).
- `src/tokenizer/tokenizer.c`: the added-token predicate is one helper, `is_added` (#118). No change
  in behaviour.
- `tools/mutate_auto.py`: a timeout repeats with three times the budget and is listed as `TIMEOUT`
  (#117); a failure that is the machine's memory is judged again alone at the end, `PRESSURE` if
  refused again (#119).
- Survivors of `tokenizer.c` 75 → 43, each named in `docs/MEASUREMENTS.md` §Generated mutations.

- **How to check it**: `make check`; in the container `sh tools/mutate_files.sh tokenizer`
  (~13 min), report in `build/mutate/tokenizer.txt`.
- **What is left**: the survivors of `main.c` (334); strict number parsing on the command line; the
  other files' mutation runs again with the corrected tool.

### 2026-09-23 — The command line's survivors read, and one strict option parser

- `src/app/main.c`: every command reads its arguments through one table and one parser
  (`parse_opts`): a number is decimal digits inside the option's own range, anything else (`abc`,
  `5x`, `" 5"`, `+5`, `-1`, a value past int64) is exit 2 with the option named; an option given
  last without its value is named too; `--expert-budget` takes 1 to 2^43 - 1 MiB or `min` (2^44
  wrapped to 0, which means automatic: #121). `cpu` and `inspect` refuse extra arguments,
  `tokenize` refuses `--pieces` with `--decode`, `-p 0` is a usage error. The commands are one
  table; `tokenize --batch` and `chat-template` read their records with one function; the route
  trace is one chain of writes; `step[]` starts zeroed.
- `src/tokenizer/unicode.c`: `tr_utf8_whole_prefix`, the chat's cut before a split UTF-8
  character, moved from `main.c` so a test can reach it (`test_unicode`, 13 cases).
- `tests/test_cli.c` rewritten: stdout and stderr apart, every command and every usage error, both
  ends of every range, `inspect`'s listing, generate against `logits`, `--spec` with drafts
  accepted stopping at `-n`, the threads line, `--expert-mask`, `tokenize`'s modes and records,
  the route trace's bytes, the chat against `run`, `/reset`, `chat-template`'s refusals.
- Survivors of `main.c` 334 → 43, each named in `docs/MEASUREMENTS.md` §Generated mutations
  (#120).
- `tools/mutate_auto.py`: a check out of time is killed with its whole session, and a run that
  leaves a process behind says so and fails (#122); a mutant refused for memory alone, when the
  unmutated tree passes alone, is killed (#123); each check gets ten times its own time, fastest
  first, with `TR_TEST_FAILFAST=1` (`tests/test.h`: a C test stops at its first failure); a
  progress line per mutant; `--lines` takes several ranges and `--changed REF` the lines that
  differ from a git ref (#124). `tools/mutate_files.sh`: tests without pinning or spinning,
  progress in `build/mutate/<name>.progress`, `CHANGED=<ref>`. Every file of it ran again: the
  whole set in 42 minutes plus 31 for `olmoe.c`; every verdict is logged with the check that gave it
  (#125: five kills of `olmoe.c` did not repeat).

- **How to check it**: `make check`; in the container `sh tools/mutate_files.sh main` (~23 min;
  `CHANGED=HEAD` for only the lines changed since the last commit), report in
  `build/mutate/main.txt`, time left in `build/mutate/main.progress`. By hand: `build/trochilus generate -m <f> -p 4 -n abc` is exit 2,
  "generate: -n takes a whole number from 0 to 9223372036854775807, not 'abc'".

### 2026-09-23 — The machine's marker, and the engine kept between commands (`trochilus serve`)
The measuring scripts follow the machine's rule: `tools/measure_guard.lib` takes
`~/.claude/macchina-ferma` for the duration of a timing measurement (noclobber, waits up to 6 h for
another window's, refreshed before every run, given back only by its writer) and no longer stops
or waits for other projects' containers; `tools/test_marker.sh` in `make check`,
`tools/mutate_marker.sh` 9 of 9 red. Question 49: `trochilus serve` keeps pool, model and expert
store; `generate`, `logits`, `run` and `chat` run in it when its endpoint answers, byte for byte the
same (the client's directory and streams: fd passing on POSIX, pipes on Windows), idle exit, a
rebuilt client declined by build stamp and run locally; `tools/serve_first_prompt.sh` measures it,
`AB_WALL=1` in `ab_modes.sh`. Native `test_cli` green on Windows (the `\r\n` fix of #126).
Check: `make check`; by hand, `build/trochilus serve -m <gguf> &`, then `build/trochilus run -m
<gguf> -p hello` twice and `build/trochilus serve --status` (1 load, 2 requests), `serve --stop`.

### 2026-09-23 — Question 49 measured; a rainbow water bar while the model loads
`sh tools/serve_first_prompt.sh 5`: the first 2048-token prompt through `trochilus serve` takes
7.81 s instead of 12.79 at full budget (the load gone) and 12.04 instead of 13.00 at half budget
(docs/MEASUREMENTS.md §The engine kept between commands). While a command loads the model, stderr
shows variant A of the bar Marcello chose (a tilted tube filling with rainbow water, "reading the
model X/Y GiB"): the core reports bytes through `tr_model_load_progress` (after every dense
tensor and every expert unit), `src/app/bar.c` draws at most 30 frames a second only on a
terminal, and clears its line when the load ends or fails. `tests/test_bar.c` (golden frame from
`build/progress-bar/preview.py`), `test_stream` checks the reported bytes; `tools/mutate_bar.sh`.
`ab_modes.sh` with `AB_WALL=1` no longer records a failed run as a fast one
(`tools/test_ab_modes.sh`); the measuring scripts wait for Smart App Control before taking the
machine's marker. Check: `make check`; by hand, in Windows Terminal (not Git Bash):
`build\trochilus.exe run -m models\OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -p Hello -n 16
--expert-budget 8192`; `TR_BAR=0` turns it off.

### 2026-09-23 — The prompt reads the next layer while this one computes
Under a partial expert budget the layer-major prompt hands layer L+1's missing units to an I/O
thread (`tr_experts_prefetch`, a small thread and monitor API in `src/base/threads.h`) while layer
L computes; the store's bookkeeping stays on the calling thread, a unit still in flight is waited
for, a failed read ahead fails the eval and leaves nothing in flight. Same bytes out at every
budget and thread count (`tests/test_prefetch.c`, TSan in `make check`); `tools/mutate_prefetch.sh`.
2048 tokens at half budget: 11.45 → 9.39 s (docs/MEASUREMENTS.md §Reading the next layer).
`orphans.ps1` no longer takes Git's launcher of the asking script for an orphan (LESSONS #136).
Check: `make check`; `SET=prefetch sh tools/prefill_overlap.sh 5` (native, ~20 min);
`TR_PREFETCH=0` turns it off.

### 2026-09-24 — Three server bugs, and `make quick`
`tests/test_serve.c` grew the cases a mutation triage asked for (malformed requests, descriptors,
a working directory the server cannot enter, a reply of another protocol, a second model); two of
them found real bugs in `src/app/serve.c`: a descriptor leaked per request that sent 4 descriptors
(LESSONS #137), a double free in the client on a reply of another protocol (#138), and on
Windows a reload of the model for the same file named with another case (#140, seen only
natively). Mutants of serve.c alive: 128 -> 73 (205 killed, no timeout). `make
quick`: lint, the native build and the C tests in Linux gcc, about 2 minutes, for use between
edits (the gate stays `make check`). Check: `make quick`, `make check`.

### 2026-09-24 — The gate in 238 s instead of 428
The C tests write their temporary files where `TR_TEST_TMPDIR` says (`tests/test.h`
`tr_test_tmpdir`), and the gate sets it to the container's own disk: on the Windows bind mount the
synthetic models cost 100 s of each build's 120. Every synthetic model carries the pid in its name
(`synth_write`), the 20 runs of `test_hot` go four at a time, the model steps run in three lanes
(tiny; the whole real model then the tokenizer oracle; the 2-layer cut), `test_marker.sh` waits by
polling (10 → 2.7 s), and the platform stamp is written atomically. `tools/speed_compare.py` forces
the decode width (`--decode-threads same`) and `tools/race_llama.sh` runs the race against
llama.cpp by the rules of a native measurement. Table in MEASUREMENTS §The gate. Check: `make
check` ("check passed in ... s").

### 2026-09-24 — Q4_K on the CPU, and the race against llama.cpp again
The engine reads Q4_K weights (M2, first step): scalar dequantization and dot row in
`src/kernels/kernels.c` (the definition: gguf-py's order of operations, the dequantized weight in
every product), AVX2 and AVX-512 bit for bit the same in `kernels_x86.c`, where AVX-512 computes
the 16 values a sub-block can take once and picks each weight with `vpermps` (our own, from reading
llama.cpp and ik_llama.cpp: ORIGINS §Q4_K on the CPU). Tests: a block packed as ggml packs it,
dot = dot of the dequantized row, every tier against scalar with a wrong kernel that must be seen,
the reader's Q4_K sizes, a synthetic Q4_K model through the active table in every tier, and
`tools/check_dequant.py` (every float bit for bit gguf-py's, Q8_0 too, in `make check`). The real
OLMoE in Q4_K comes from our Q8_0 by `tools/quantize_q4k.sh` (llama-quantize, no download), and its
2-layer cut meets transformers in `make check` (32/32 tokens, logits within 2e-4). The race against
llama.cpp redone on the current binary: prefill 1.8× theirs at 16 threads (was 13×), decode 1.04× at
context 512, 1.16× at 2048 (MEASUREMENTS §Speed — again). Check: `make check`; `make bench`;
`sh tools/race_q4k.sh`; `build/trochilus run -m models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf -f
prompt.txt`.

### 2026-09-24 — Q6_K on the CPU, and the Q4_K_M model
The engine reads Q6_K weights (M2, second step), so Q4_K_M, the file people download, runs: scalar
dequantization and dot row in `src/kernels/kernels.c` (gguf-py's order, the dequantized weight in
every product; the block in `kernels_internal.h`), AVX2 and AVX-512 bit for bit the same in
`kernels_x86.c`: a block's 256 quants assembled once on bytes into q − 32 (ik_llama.cpp's way), then
Q8_0's path per element (ORIGINS §Q6_K on the CPU). Tests: a block packed as ggml packs it, dot =
dot of the dequantized row, every tier against scalar with a wrong Q6_K kernel that must be seen,
the reader's Q6_K sizes (`test_gguf`, one test for both k-quants), a synthetic Q6_K model through
the active table in every tier, `tools/check_dequant.py` with Q6_K (1M floats bit for bit gguf-py's);
`tools/mutate_q6k.sh`, 14 mutations, all red. `tools/quantize_q4k.sh m` makes the real OLMoE in
Q4_K_M from our Q8_0 (17 tensors Q6_K: output, and attn_v and ffn_down_exps in 8 layers); its
2-layer cut meets transformers in `make check` (`REAL_MODEL_Q4KM`: 32/32 tokens, logits within
2e-4); the whole model writes what the Q4_K writes on "The capital of France is". Measured:
MEASUREMENTS §Q6_K on the CPU. Check: `make check`; `sh tools/mutate_q6k.sh` (container);
`sh tools/bench_kernels.sh`; `sh tools/race_q4k.sh 5 m`; `build/trochilus run -m
models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf -f prompt.txt`.

### 2026-09-24 — Two weight rows at a time in the prefill, measured again
The kernels measured again (the AVX2 ratios of the first series were noise; `tools/bench_kernels.sh`
now names every line above 10% spread) and the whole matrix measured per type: Q8_0, Q4_K and Q6_K
cost the same through `tr_matmul`, bound by loading the input rows, not by decoding the weights.
So `dot_row2_x4` (`src/kernels/kernels.h`): two weight rows against the same four input rows, each
input vector loaded once for eight products, AVX-512 for Q8_0, Q4_K and Q6_K, used by `tr_matmul`
on pairs of rows; every sum still its own `dot_row` bit for bit. The sixteen Q6_K scales now in SIMD.
Tests: every tier's two-row kernel against two scalar x4, a wrong one per type that must be seen,
`test_tier_used` counting its products and requiring every pair of rows through it;
`tools/mutate_row2.sh`, 11 mutations, all red. Prefill 1.20–1.26× on Q8_0, 1.11–1.12× on Q4_K_M,
decode unchanged (MEASUREMENTS §Two weight rows). Check: `make check`; `sh tools/bench_kernels.sh`;
`sh tools/mutate_row2.sh` (container); `build/trochilus generate -m <gguf> -p 512 -n 16`.

### 2026-09-24 — The race with llama.cpp again
`sh tools/race_llama.sh 5` on commit 2d709e0: llama.cpp's prefill is now 1.41× ours at 16 threads
and 1.13× at 8 (was 1.8× and 1.5×), decode 1.00–1.06× at context 512 and 1.12–1.15× at 2048
(MEASUREMENTS §Two weight rows). Check: `sh tools/race_llama.sh 5`.

### 2026-09-24 — Skipping cached positions exactly: the premise measured, and closed
Marcello's decision: the long-context decode goes faster only with logits identical to the byte.
Before any kernel, the premise of the exact skips (positions whose exp is exactly 0, or whose
contribution the rounding absorbs, found with cheap bounds): `make attn-probe` builds a diagnostic
engine (`build/probe/`, `-DTR_ATTN_PROBE`, `tools/attn_probe.c`) that writes each decode token's
queries, keys, values and outputs; `tools/attn_skip_report.py` replays the attention in float32,
first proves its replay gives the engine's bits (8704 head outputs; two mutations red), then counts
what each criterion would skip. On six real runs (prose, code, synthetic; ~2000 and ~4000 tokens;
65 M positions) no score falls 30 below its max and an oracle skips at most 0.2% of the bytes: the
whole family closed (MEASUREMENTS §Skipping cached positions exactly, LESSONS #152). The gate
compiles the probe's branch of `olmoe.c` (0 warnings). Check: `make attn-probe`, then the two
commands in `docs/COMMANDS.md`.

### 2026-09-24 — The decode's attention one position at a time
A decode token's attention read its KV at 46-48 GB/s where a plain read of the same bytes gets 52-54:
the x4 kernels read four rows 512 bytes apart a cache line of each in turn, an order the prefetcher
does not follow, and a lone query shares nothing across the four. `tr_attention_group` now takes a
group of one position by position (`tr_attention_head`): same bits, the attention 1.10-1.13× in the
agent's bench under load, not distinguishable a token on a still machine (the zone −9% at 2048;
`tests/bench_attn_bw.c`, MEASUREMENTS §The decode's attention, one position at a time). With it the
short measurement protocol, the default from now on: `sh tools/decode_context.sh change-short
<before>` (exactness, 8 threads at 512 / 2048 / 4000 with A/A, the profile after; ~40 min). Tests: `test_attention_group` covers both branches; a mutation of the new branch red (140
mismatches); on the real model logits of 600 positions and the tokens after 4000 identical to
commit 718a84c. Check: `make check`; `build/tests/bench_attn_bw.exe 2048 --runs 19`.

### 2026-09-24 — The decode's attention on the GPU, with the CPU's bits
`src/backend/gpu_attn.{h,c}`: the NVIDIA driver opened at run time (`tr_lib_open` in `platform`,
nothing linked), three PTX kernels a layer written in C (scores appending the new key and value,
`tr_expf` in double, values summed by shuffles in the lane tree's order), every float op `.rn`, the
KV mirrored in VRAM (prompt passes through `tr_gpu_attn_write`, keys position-minor), zero copy for
the 24 KB in and 8 KB out of a decode call, and one warp spinning between calls so the laptop GPU
keeps its clocks; the last pass of a prompt or of a speculative check wakes the GPU too
(`tr_gpu_attn_warm`, one warp at a time), so the first decode tokens do not pay the clocks' climb
(LESSONS #158, #159). The model opens the device (`tr_model_set_gpu`), each session its cache; the
command line turns it on when there is a GPU (`TR_GPU=0` off, `TR_GPU=1` says why not) and reports
`gpu: <name>, attention of N decode tokens`. A driver error sends the session back to the CPU,
whose cache is always whole. Tests: `tests/test_gpu_attn.c` (8010 head outputs against scalar and
AVX-512, 0 floats differ; counters on exceptions, underflows, subnormal exponentials, poisoned
rows; skipped without a GPU), `tools/mutate_gpu.sh` (7 of 7 red), `tools/gpu_exact.sh full` (logits
of 2000 decode positions, tokens after 4000, a speculative run: identical) and `quick` in the gate.
Premise benches: `tests/bench_gpu_attn.c`, `tests/bench_gpu_q8.c` (MEASUREMENTS §The decode's
attention on the GPU: the premise). Check: `make check`; `sh tools/gpu_exact.sh full`;
`build/trochilus run -m <gguf> -f prompt.txt -n 200` (the `gpu:` line).

### 2026-09-24 — The premises of questions 53-57 and 62 measured: the project read like a genome
Tools to measure a premise again on the next model or machine, no engine change.
`tools/draft_agreement.sh` + `tools/draft_agreement_report.py` (question 53): three real texts of
the repo, the exact model's greedy 512 tokens, the logits of every position of the exact model and
of each draft (Q4_K, Q4_K_M), top-1 agreement, KL, tokens a pass of k drafts; the report fails if
`generate` and `logits -b 1` disagree on a greedy token (seen red with one token changed).
`tools/route_union_report.py` (54, 62): the experts of k consecutive tokens against chance, and the
pairs of experts that fire together, from route traces. `tools/weights_genome.py` (56, 62): every
block of a GGUF once: code entropy three ways, scale entropy, zero blocks, repeated blocks and rows.
`tools/kv_repeats_report.py` (62): which layers' keys and values repeat exactly, on the probe's dumps.
`tests/bench_peak.c` (55, 59): the no-FMA and FMA peaks in inline assembly, the prefill kernel's
instruction stream in assembly with 4 and 8 tokens a weight, `dot_row2_x4` and `tr_matmul` on
OLMoE's shapes, 1 and 16 threads; fails if a kernel runs faster than the peak (mutant
`-DBENCH_PEAK_MUTATE`, seen red). `tests/bench_expf32.c` (58): the float-only exp's fma variant as
AVX-512 and AVX2 tiers, checked on all 2^32 floats beside the scalar ones and timed per value
(mutation 7 skips the slow path, seen red). `tools/bench_native.sh <bench> [args]`: any premise bench under
the native rules (marker, still machine, load declared, a copy one byte longer). Numbers in
MEASUREMENTS (§A Q4 draft, §Experts read by a pass of k rows, §The model read like a genome, §The
CPU's peak, §The KV packed in 28 bits). Check: `make check`; `sh tools/bench_native.sh bench_peak`;
`sh tools/bench_native.sh bench_kvpack time --run all`; `tools/.venv/Scripts/python.exe
tools/weights_genome.py models/<file>.gguf`; `tools/.venv/Scripts/python.exe
tools/route_union_report.py build/route/*.bin`; the container line in `tools/draft_agreement.sh`.

### 2026-09-24 — Two weight rows against eight tokens, and the lane tree in SIMD
`dot_row2_x8` (kernels.h, AVX-512, Q8_0, Q4_K, Q6_K): two weight rows against eight input rows,
sixteen accumulators, each sum its own `dot_row` bit for bit; `tr_matmul` takes eight tokens a call,
then four, then one. The lane contract's tree for many sums at once (`avx512_pair_sums`): the x8
kernels end in one store, the x4 two-row kernels too. Tests: the x8 kernels against scalar's
`dot_row` (special values, a counter of the calls compared), a wrong x8 table seen red, the tiled
matmul's roads counted exactly (`test_matmul_grouped`), the engine's 29-token pass through every road
(`test_tier_used`); `tools/mutate_row2.sh` 23 mutations, all red. Benches: `bench_peak` (row2_x8
lines, every matmul with and without x8), `bench_kernels --matrix` (each type with and without x8).
Numbers in MEASUREMENTS §Two rows against eight tokens. Check: `make check`; `sh tools/mutate_row2.sh`
in the container; `sh tools/bench_native.sh bench_peak --runs 15`; `sh tools/prefill_context.sh
change <binary before>`.

### 2026-09-24 — Pieces 1 and 4 against the references, faster mutations, the gate's native side beside it
- Piece 1 (the prompt's CPU matmul) read in llama.cpp, ik_llama.cpp, ds4 and colibri, and their
  kernels measured alone on bench_peak's shapes (`tools/bench_ggml.sh`: ggml's public API against
  the static libraries of `ref/llama.cpp/build-trochilus`); piece 4 (softmax and exp) read and
  measured on every float (`tools/bench_expf_refs.sh`). MEASUREMENTS §The prompt's matmul against
  the four references, §Softmax and exp against the references; ORIGINS rows 1 and 4.
- Question 63 (decode a panel of rows once, then F32 over every token): `bench_peak`'s `mix x8 f32`
  stream, `row2_x8_f32`, the panel matmul beside `tr_matmul` byte for byte (red under an FMA,
  `-DBENCH_PANEL_MUTATE`), `--one-core`.
- `test_dequant_is_the_dots_weight` (tests/test_kernels.c): `dot_row == dot_f32(dequant_row)` on
  every tier, every type, special values; red under an FMA in the AVX-512 Q8_0 dot (LESSONS #169).
- `tools/mutate_auto.py`: a gcov pass lists the mutants on lines no check runs (UNCOVERED) instead
  of building them; with `--asan` the sanitizers judge only the plain build's survivors; the trees
  built once and copied. `prof.c` 30 → 9 s with the same verdicts. `tools/mutate_files.sh` gains
  `gguf`, `experts`, `threads`, `platform`, all four run (build/mutate/).
- The gate: the native C tests run beside the container (`tools/beside.sh`, its test
  `tools/test_beside.sh` in the gate, red without the trap).
Check: `make check`; in the container `sh tools/bench_ggml.sh`, `sh tools/bench_expf_refs.sh`,
`sh tools/mutate_files.sh prof`; `sh tools/test_beside.sh`; `sh tools/bench_native.sh bench_peak
--runs 15 --one-core`.

### 2026-09-25 — Piece 1 closed; gguf.c's survivors 40 → 12
- Question 64 in `tests/bench_peak.c`: the 4 × 6 and 3 × 8 tiles as asm streams (one statement,
  registers named, LESSONS #174) beside 2 × 8 with the same decode, and as kernels (`row4_x6`,
  `row3_x8`, Q8_0 and Q4_K) in `tr_matmul`'s order, byte for byte against it before timing, red
  under `-DBENCH_TILE_MUTATE=4` and `=3`, counter `g_tile_kernel_calls`; `--q64` runs only those.
  Closed as no (MEASUREMENTS §More weight rows per input load).
- Measured natively: x8's prefill 1.13–1.19×, logits identical; the race against llama.cpp after
  x8 (MEASUREMENTS §Speed after x8). ORIGINS row 1 closed.
- `tests/test_gguf.c`: the agent's seventeen cases corrected (LESSONS #172) and five more against
  the survivors (the magic alone, a string array cut in its last item, the last key cut at the
  alignment, general.alignment 0 and 3); the truncations checked by the part of the file they end
  in; on Windows the shrink under an open reader is refused. `gguf.c` unchanged.
- `tests/test_spec.c`: the generation helper prints a refused session and where generation
  stopped (LESSONS #175).
Check: `make check`; `sh tools/bench_native.sh bench_peak --q64 --one-core --runs 11`; in the
container `python3 tools/mutate_auto.py src/format/gguf.c test_gguf test_model_load test_tokenizer
--asan --jobs 6`.

### 2026-09-25 — The decode's matmul against ggml's (ORIGINS row 2 closed)
- Read how colibri, ds4, llama.cpp and ik_llama.cpp run the one-token matmul (ORIGINS row 2).
- `tools/bench_ggml_decode.{c,sh}`: ggml's MUL_MAT / MUL_MAT_ID against `tr_matmul` /
  `tr_matmul_grouped` in one binary, the same weight bytes (~1 GiB of copies, every pass from RAM)
  and cores, alternating, beside a plain read of the same bytes; the first node of both checked
  against each other. Measured on a still machine (MEASUREMENTS §The decode's matmul against
  ggml's): Q8_0 level from 4 threads, Q4_K theirs 1.34-1.40x at 4. Three exact Q4_K kernels tried
  and closed as no; the engine's kernels unchanged.
- `tests/test_kernels.c`: the decode's one-token matmul (Q8_0, Q4_K, Q6_K, 2048 columns, dense and
  eight one-token groups) is one `dot_row` per product at 1, 3 and 7 threads, counted; seen red with
  `matmul_rows` stopping a row short.
- The wait for a still machine runs through `cleanup_run` at all nine call sites, so a stopped
  measurement stops at once and frees the marker; `tools/test_cleanup.sh` fails on a bare one
  (LESSONS #177).
Check: `make check`; `sh tools/bench_ggml_decode.sh` (container, ~3 min, guarded).

### 2026-09-25 — The Q4_K decode level with llama.cpp at 4 threads, exact (question 66)
- `tests/bench_q4k_genome.c` + `tools/bench_q4k_genome.sh`: the decode's Q4_K dot sequenced in L1
  (the cost of each instruction on this core, the kernel with one piece removed at a time, the
  candidates bit for bit against scalar), `--ram` / `--ram-smt` / `--dyn` from RAM, `--smt` in L1;
  built by the gate (RESEARCH_BIN). `tools/q4k_tables.py`: distinct (scale, min) pairs per tensor.
- `avx512_dot_row_q4_k`: the block's scales and mins in a vector (`avx512_q4_k_scales_store`, the
  same floats), one block ahead, and a prefetch 4608 bytes ahead: 1.25x in L1.
- `dot_row2` (kernels.h): two weight rows against one token; `avx512_dot_row2_q4_k` (the scales of
  8 blocks of both rows first); `tr_matmul`'s decode takes rows in pairs (`matmul_rows` and the
  one-token remainder of `matmul_tiled`). The real model's decode at 4 threads 40.3 -> 50.4 tok/s,
  llama.cpp 50.2; the matmul alone ggml 1.37x -> 1.04x at 4 threads.
- Tests: test_kernels compares every tier's `dot_row2` with scalar's `dot_row` (a wrong one seen),
  the decode matmul counts pairs and requires them; test_tier_used requires AVX-512's Q4_K pair and
  counts every product; eleven new mutations in `tools/mutate_row2.sh`.
- `make check` on Windows waits until Windows has 12 GB free before the container's lanes
  (`tools/host_memory.sh`, LESSONS #183). `RACE_THREADS` picks `tools/race_q4k.sh`'s threads.
Check: `make check`; `sh tools/bench_q4k_genome.sh` (L1), `sh tools/bench_q4k_genome.sh 9 --ram 4
--only pf4row` (RAM); `sh tools/bench_ggml_decode.sh 5 q4_k`; `RACE_THREADS=4 sh tools/race_q4k.sh`;
in the container `sh tools/mutate_row2.sh`.

### 2026-09-25 — The pool balanced at its tail; SMT in the decode closed as no (question 67)
- `tr_parallel_for_balanced` (threads.{h,c}): each thread runs its chunk in `TR_POOL_BLOCKS` = 64
  blocks and then takes the blocks the other chunks have not started (`run_chunk`, one atomic add a
  block, each region on its own line); every index runs once, the bits cannot move. The matmul uses
  it when every group has one input row (the decode, its experts), the decode's attention too; the
  prompt keeps `tr_parallel_for`. `TR_POOL_BLOCKS=1` is the static split, for A/B.
- `TR_POOL_TRACE` (research build: `EXTRA_CFLAGS=-DTR_POOL_TRACE`): every call's and chunk's start
  and end to `$TR_POOL_TRACE_FILE`; `tools/pool_trace.py` sums them by call shape.
- `tools/race_smt.sh`: Trochilus and llama.cpp confined by taskset to the same logical processors
  in the container, two thread counts (not SMT there: LESSONS #184).
- Test: test_base's `test_parallel_balanced` (every index once over widths, n and min_chunk; a held
  worker's chunk finished by the others, red with `TR_POOL_BLOCKS=1` and with the help loop cut).
- Measured (MEASUREMENTS §SMT in the decode, §The pool's tail): natively two threads a core give
  +1.6-5.7% at 4 cores, nothing at 8 (the engine's measured width, 60 tok/s), a collapse at 16
  (question 68); the balanced tail ~1.08x on the decode calls in the container at 8, level natively.
Check: `make check`; `sh tools/race_smt.sh`; a trace: `make BUILD=build/win-trace
EXTRA_CFLAGS=-DTR_POOL_TRACE build/win-trace/trochilus.exe`, run with `TR_POOL_TRACE_FILE=x.trace`,
then `tools/.venv/Scripts/python.exe tools/pool_trace.py x.trace`.

### 2026-09-26 — Piece 3, the attention: read, raced as a line, the prompt's tiles
- ORIGINS row 3 read in colibri, ds4, llama.cpp and ik_llama.cpp; the 09-25 race read as intercept
  and slope: level with no context, 5.31 against 3.24 µs a cached position (their F16 KV's bytes;
  exact F32 would need 81 GB/s). Question 69 closed on the CPU.
- `bench_mem streams` (T shares against one front taken in turns) and `bench_attn_bw front` (the
  decode's attention with one front): the premise refuted, both kept for the next of the kind.
- `dot_f32_4x4`, `axpy_f32_4x4` in the kernel table (scalar definition; AVX-512 tile with the lane
  trees in pair sums; AVX2 as four x4 calls) and `tr_attention_group` four queries at a time: the
  prompt's attention zone 1.31-1.33x, prefill 1.03x at 2048 and 1.06x at 4000, logits identical.
- Tests: test_kernels' `tile_diffs` (16 400 tiles a tier; two mutants of the definition and two of
  the AVX-512 kernel seen red), test_tier_used's 4x4 counters; `bench_attn` tile variants.
- `tools/ab_zone.sh`: one zone before and after, alternated run by run. `tools/race_llama.sh` takes
  RACE_MODEL, RACE_PROMPTS, RACE_THREADS, RACE_OUT. The backslash hook sees `write_bytes(` too.
Check: `make check`; `build/tests/bench_attn.exe 2048 --threads 1 --heads 1`; `sh tools/ab_zone.sh
<before> build/trochilus.exe attention 8`.
