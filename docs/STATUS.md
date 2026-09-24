# Status

Replace, do not append. Cap 40 KB. History is in `archive/DONE.md`.

## Decisions

- 2026-09-24 — **The long-context decode goes faster only exactly** (Marcello): no KV at 16 bits
  with rounding. The criterion: logits identical to the byte to today's engine, batch = token by
  token, every tier = scalar. Closes point 6's «KV at 16 bits» and order (b) as no. A GPU is a tier
  like the others (every op `.rn`, the CPU's orders: question 51): it runs the decode's attention
  by default when the machine has one (`TR_GPU=0` keeps the CPU).
- 2026-09-24 — **A change to the decode is measured with the short protocol** (Marcello):
  `decode_context.sh change-short` (exactness, the decode on 8 threads at 512 / 2048 / 4000 with A/A,
  the profile after; ~40 min); the full `change` only for what every token pays at any context.
- 2026-09-24 — **Sources to take the best from** (Marcello): besides colibri, ds4 and llama.cpp,
  ik_llama.cpp for M2 (K-quant and IQ_K CPU kernels, read before writing Q4_K), ktransformers
  for M3, Adaptive-K as our own non-exact experiment (question 50); ArcLight and bitnet.cpp
  later, PCoMoE not now. First read of each and why: `docs/ORIGINS.md` §Sources not yet studied.
- 2026-09-17 — New project, not a fork: own architecture, best pieces from colibri and ds4
  brought in by hand (Marcello's choice). Name: Trochilus. Licence Apache-2.0.
- First model OLMoE (exists real at 4–7 GB, runs everywhere); DeepSeek V4 Flash is the M4.
- Standard GGUF v3 format: Hugging Face files open without conversion, and ds4 CUDA kernels
  (ggml) already work on those types.
- Kernels chosen at runtime, not with `-march`; SIMD variants bit-identical to scalar.
- No OpenMP: own thread pool (ds4 idea), sized to physical cores (colibri).
- Weights with `pread`, not `mmap` (colibri: mmap keeps the model resident).
- GPU backend as runtime-loaded modules: the core stays without dependencies.
- From ds4 **not** brought: hyper-connections, DSA indexer, compressor, engram, DSpark (soldered to
  DeepSeek/GLM, they reuse the M4 on generic primitives); the GGUF reader to be cleaned of
  architecture hooks.
- From colibri **not** brought: int8 kernels tripled across engines, LRU scan O(cap),
  environment variables read inside kernels, global state per model.
- Correctness reference: transformers 5.14.1 + torch 2.13.0 CPU, numpy < 2.4 (2.5 breaks
  `import torch._dynamo` on Windows).

- 2026-09-17 — **Focus on coding** (Marcello's idea, confirmed): the engine stays general, but
  coding decides the order. Step 2 = Qwen3-Coder-30B-A3B; first the coding levers: file cache
  already read (prefix reuse), speculative decoding on text already in prompt, prefill speed on
  long prompts, long contexts; images and audio not implemented; quality also measured on a
  bench of programming exercises.

- 2026-09-17 — **Hot path** (Marcello: performance to the last drop): code that runs on every token
  has rules checked by lint and tests (no allocations, strings, I/O, element-wise math; logits
  identical with every thread count). Comments and line count do not change machine code (verified):
  they are not a speed lever. Rules in `docs/ARCHITECTURE.md`.
- 2026-09-17 — **Never a number from a single run** (Marcello): median of N with min, max, and
  spread; identical tokens in every run; tests with threads run multiple times and under ThreadSanitizer.
- 2026-09-17 — Thread pool with active wait (2 ms of `pause`, then sleep), one slot per thread:
  dispatch from 53 µs to 1.5 µs at 16 threads on Windows (`docs/MEASUREMENTS.md`).
- 2026-09-17 — RoPE from table per session, experts in stages (activations of all in one
  parallel work), attention per head: decode 26.3 → 32.8 tok/s at 16 threads, logits bit-identical.
  Every exact optimization is tested thus: `trochilus logits` on the same tokens with the binary
  before and after, `cmp` of files.
- 2026-09-17 — Private repository on GitHub (Marcello's request), first commit with all work so far.
- 2026-09-22 — **The repository becomes public** (Marcello's request):
  `github.com/namespaceMarcello/trochilus`. Checked before flipping: no model, no key and no
  third-party report has ever been in the history, the largest object is a 174 KB document, and
  `LICENSE` and `NOTICE` are at the root. From now on every commit is visible as it is pushed.
- 2026-09-17 — **Correctness first, then speed** (Marcello): tokenizer, same tokens as
  llama.cpp on the real model, and speed comparison come before other optimizations; then
  block prefill, then memory and VNNI. Yes to llama.cpp download (source in `ref/`, build in
  container). During measurements the Docker containers of other projects can be stopped, and
  restarted at the end.
- 2026-09-17 — **Tokenizer**: the reference is transformers, not llama.cpp (which does not do NFC). A
  pretokenizer family enters the `tokenizer.c` table only after an oracle on its `tokenizer.json`
  (today only `olmo`); an unknown family is rejected. Unicode classes and NFC probed by HF
  `tokenizers` 0.22.2 (pin), not by Python. Invalid UTF-8 kept byte by byte, except bytes
  the vocabulary does not have (discarded as HF does). From ds4 the load from GGUF, from colibri
  the regex replayed in C; new code (BPE n log n, no `exit`).
- 2026-09-17 — **`trochilus chat`** (Marcello): no Jinja engine; every supported template recognized
  by exact bytes and written in C, compared with `apply_chat_template`. Each turn renders and
  tokenizes the whole conversation (as transformers) and reuses the cache up to the first
  different token (`tr_session_rewind`): the reply is identical to one computed from scratch.
- 2026-09-17 — **colibri/ds4 comparison by component** (Marcello, LESSONS #29), with low usage: one
  folder at a time (kernels, OLMoE graph, GGUF, pool; platform only if needed), function map
  with Haiku, one Sonnet per folder on targeted lines (≤ 30 lines of report), my verification
  only of flagged points, stop after each folder. Result: one line per folder in `docs/ORIGINS.md`;
  fixes after, with tests; bugs of colibri/ds4 in `docs/UPSTREAM.md` with proof.
- 2026-09-17 — **Correctness on the real model**: llama.cpp not enough as an exact reference (8-bit
  activations, KV f16: average KL 9e-3, `docs/MEASUREMENTS.md`). Oracle made with transformers on
  the same real weights cut to 2 layers; Marcello: must not stop developments, a Sonnet agent does
  it in parallel (~5 GB free RAM needed: no chats open) and then enters `make check`.
- 2026-09-17 — DeepSpeed ideas (offload to SSD) evaluated: streaming for **experts**, not per layer; KV
  on disk as checkpoint of already-read files, not for old tokens with full attention. Decided
  with measurements 13–17 of `docs/MEASUREMENTS.md` §To measure; proposed order: 13–16 after steps
  1–2 below, 17 with KV work (awaiting Marcello's approval on order).
- 2026-09-17 — **llama.cpp third source** (Marcello): from llama.cpp/ggml (MIT) only pieces
  where measurements say it wins (block prefill, int8/VNNI activations as option), with
  colibri/ds4 method (header, `docs/ORIGINS.md`, bit-identity, benchmark). Not brought: ggml
  graph, allocator, backend, C++. llamafile's `sgemm.cpp` is C++ with intrinsics, not assembly: for
  Q8_0 weights it wants Q8_0 activations (not exact), and the float path uses FMA (not
  bit-identical to scalar).

- 2026-09-17 — **Block prefill in exact C** (step 1): map of three sources in `docs/ORIGINS.md`.
  Taken: passes of 512 tokens (llama.cpp), (token, expert) pairs ordered by expert (ds4),
  batch attention for (head, token) after writing all K/V (colibri), logits of last token only
  (all). Discarded: int8 activations in batch (llama.cpp, ds4: not exact) and ds4's per-row sum
  over all experts (rereads entire batch each row). Decode is one pass from one token: single
  path. Invariant: logits and cache bit-identical for every `n_batch` and every call split
  (`tests/test_prefill.c`, oracle to the byte). None of the sources prove it (LESSONS #42).
  `tr_session_create` takes `n_batch`, the `-b` command line.
- 2026-09-17 — **`dot_row_x4` kernel** (one weight row against 4 tokens in registers, ds4's idea which
  does 2): results bit-identical to `dot_row`, +45% on prefill, nothing on decode (which has one
  token only). Accumulators go in named registers, never in an array (LESSONS #45).
- 2026-09-17 — **Speed comparisons only with alternating runs** (`tools/ab_speed.sh`): measuring
  first all A then all B, the warming machine shifts numbers 10–25% and hides real gains
  (LESSONS #46: a good optimization had been discarded that way).

- 2026-09-17 — **Prefill did not scale due to thread pinning**, not power or CCD (question 20,
  `docs/MEASUREMENTS.md` §Where threads go). Clock drops 6–8% from 1 to 16 threads, and 8 threads
  split 4+4 across two chiplets do better than 8 on one only: Windows is pinning two of the 16
  threads to the same physical core. One thread per physical core: **+30%** on prefill, and 8
  to 16 cores **2.02×**. Thread pinning tests done native: inside Docker topology is not the
  machine's (LESSONS #47). Consequence: pin pool threads to physical cores and remeasure speed
  rows with that baseline.
- 2026-09-17 — **Speculative decoding from prompt** (coding lever, step 2): draft comes from
  text already in context (trailing n-gram searched backward, colibri idea; llama.cpp also has
  cache with counts, not taken), verification is **one single pass** over 1 + k positions, and
  kept only what the model would choose anyway. Unlike all three sources here tokens are
  **bit-identical** to generation without speculation, because each batch row is the same
  calculation as the one-token pass: it is a proven invariant (`tests/test_spec.c`, `make spec-check`),
  not a hope. Rejection costs `s->pos = n` (cache indexed by position: ds4 and colibri
  must restore a snapshot). Measured (§Speculation from prompt): **1.42×** rewriting a file
  already in prompt (64% drafts accepted), **0.62×** writing new code (13%); break-even
  around 15%. So `--spec` remains **off by default** until draft shortens itself after
  rejection: with that, worst case becomes «as without».

- 2026-09-18 — **Pool threads pin to physical core, not logical processor** (questions 22,
  27, and 3). One per core, cores taken round-robin across two chiplets, SMT siblings only if
  threads exceed cores; each thread free between the two siblings of **its** core. Remeasured with
  A/A control (8 rounds, `docs/MEASUREMENTS.md` §Revision): at 16 threads prefill **1.27×**
  against no pin and **+9%** on processor pin, which is also unstable (25% spread vs. 5%); on
  decode core pin is not distinguishable from the other two (2.2% threshold). The «−17% on decode»
  of processor pin, which started the choice, **does not reproduce** (−3.4% vs. no pin): choice
  stands, for prefill. 32 threads (SMT) sink: prefill −33%, decode −90%. Pin holds under loaded
  machine (+19% on prefill, 3 rounds). On macOS no pin (cannot), and a mask already set on the
  process is respected. `TR_POOL_PIN=0/1/2` for comparisons.
- 2026-09-18 — **On a MoE one extra draft row is not almost free**: measured by zone, native
  (`docs/MEASUREMENTS.md` §Revision), costs **17.6 ms** if alone and **13.7 ms** each if eight,
  against 31.3 of a pass. Cost is in experts (91% and 61%): draft row reads 2.3–4.7 new experts
  per layer out of 8 and time follows those MiB; dense multiplications free for first extra row.
  Speculation break-even at **44–56%** drafts accepted, not 15% measured in container (LESSONS #59).
  So adaptive draft not just shortens: after an all-wrong draft **stops** for 1, 3, 7, 15, then
  16 steps (LESSONS #60). Remeasured with A/A control, 8 rounds: worst case **0.953×** (previous
  3 rounds said 1.01×: wrong, LESSONS #66), best case **1.175×** (1.34× with `--spec-fixed`,
  3 rounds, for those rewriting a file). «Worst case as without» **not reached**, and pause
  constants do not reach it (replay with measured costs: at most 0.97–0.98×): `--spec`
  remains **off by default**.
- 2026-09-18 — **When a difference is a conclusion**: comparisons with `tools/ab_modes.sh` (8
  rounds, rotating order, one mode given twice as A/A control). A single A/A pair is noisy (same
  decode: 0.5% in one block, 2.2% in another): threshold is the worst A/A of the session for that
  phase, and difference must exceed it against both copies. Below threshold write «not
  distinguishable» (LESSONS #66).
- 2026-09-18 — **Lever 2 (int8/VNNI activations) is the prefill lever, and an open decision**
  (question 21, corrected by revision, LESSONS #65). First measurement compared one row against one
  token (+1–8%) and came out «not to be written». With our kernel's 4-token structure int8
  VNNI at 512 bits is **1.8–2.3×** our x4 float, and prefill is 90% multiplications; exact path
  «8 tokens in registers» instead does not pay (0.79× at n=2048). But int8 is not bit-identical:
  would break «block prefill = token by token» and can exist only as a declared mode.
- 2026-09-18, **remeasured on clean machine 2026-09-19** — **Threads per phase, and the engine
  measures the number** (question 26, `docs/MEASUREMENTS.md` §Threads per phase and §Remeasure on
  clean machine). Prefill wants full pool (16 vs. 8 threads: 1.49–1.57×). Decode wants **few**
  threads, 4 at short context and 8 at long context, because RAM hits cap (57 GB/s) with 4–6
  readers and beyond drops; 16 never wins but from 8 **not distinguishable** (1.00–1.09×, 4.6%
  threshold). Numbers from 18/09 (8 over 16 at 1.10× for 512) had four forgotten `yes` processes
  underneath (LESSONS #84): direction right, measurement inflated. Default remains a **measurement**,
  as principle 0 wants: each session tests full pool, half, and quarter (never under 4 threads) on
  its first one-token passes; `--decode-threads n` forces. Short pass = up to 4 rows; rest uses
  full pool. Exact by construction (result does not depend on threads) and by test. Old estimator
  **threw dice** (widest within 1% fixed, below noise of three passes: LESSONS #71, #88) and is
  **rewritten** (2026-09-19): center = fastest pass, noise = gap of second; wins narrowest within
  noise **of the pair** (it and fastest), with more passes on the two, up to 6, if margin decides;
  remeasures at every context doubling (from 32) and changes only with two agreed measurements
  (second arrives at most after 128 passes). The `threads:` line prints choice history. Remains
  validation on real model (first of next steps).
- 2026-09-19 — **KV holds positions of one head in a row** (`src/kv/`, first piece of KV layer;
  `docs/MEASUREMENTS.md` §Decode at long context). This machine's RAM gives **~57 GB/s** read
  with 4–6 threads and beyond drops (question 4, remeasured on clean machine: the «~54» of this
  section had four cores taken underneath, LESSONS #84; not 41, which was engine speed 17/09). Weight
  multiplications already at 50–55 GB/s per zone; below was only attention, at 32–35, because with KV
  `[layer][position][head]` each head read 512 bytes every 8 KiB. With `[layer][head][position]`
  reads at 47–48: decode **1.06–1.10×** at context 512, **1.12–1.14×** at 2048, **1.15–1.18×** at
  4000, prefill **1.10×** at 2048 and **1.39–1.43×** at 4000, not distinguishable at context 32
  (A/A 3.8%); logits byte-identical on real model. Context-drop hypothesis holds in shape not
  numbers: bandwidths were two (weights 47, KV 36 GB/s); now one, and `tok/s = 48.2 / (1.2236 + 0.000262 ×
  context)` wrong by at most 1.2% from 32 to 4000 tokens. Decode moves 48–51 GB/s of 54: exact side
  leaves 6–11%, scattered (questions 34–35); **from here decode at long context goes faster
  only by reading fewer bytes**, that is with inexact levers (KV at 16 or 8 bits, question 36),
  which Marcello decides.
- 2026-09-19 — **Prefill on long prompts: three exact levers, and the big piece left is `expf`**
  (`docs/MEASUREMENTS.md` §Prefill on long prompts). Hypothesis «prompt attention is repeated read
  of keys and values» half true: dismantled in bench (`make bench-attn`), zone is 51–72%
  **softmax**, i.e. C library `expf` (30 ns per call with MinGW, 2.3 with glibc), 27–37% products
  and weighted sum, repeated read matters only at 4000 tokens, 0 to 39% per run (64 MB keys/values
  against 64 MB L3: on edge). Exact levers written: attention in **groups of 16 tokens** per head
  over blocks of 64 positions (`tr_attention_group`, `dot_f32_x4`, `axpy_f32_x4`; decode one group
  of one), **one token's work split on pool** (norms, RoPE, KV, router, rows for experts: was 8% of
  prefill on one thread), and **F32 and F16 rows with tier kernel** (router ran scalar in each tier
  and no test saw it: now `tests/test_tier_used.c` requires every weight type pass through active
  tier, counts it in engine; LESSONS #78). Prefill **1.05–1.08× at 512, 1.07–1.08× at 2048,
  1.11–1.14× at 4000** (A/A 2.1%), decode not distinguishable, logits byte-identical on real model
  even on 4000-token prompt.
- 2026-09-19 — **`tr_expf`: exponential is ours, scalar, correctly rounded** (Marcello's decision;
  `docs/MEASUREMENTS.md` §`tr_expf`; not SIMD, in this step). `src/kernels/expf.c`: table of 64
  values from mpmath, polynomial, rounding tests, 8 exceptions computed at 200 bits, no C library
  calls inside; unproven rounding would come out NaN. **Proof is exhaustive and in the gate**: all
  4,278,190,082 floats against reference, with gcc and clang (`make bench-expf`); on Windows 0
  differences even from MinGW's `expf`. On Windows logits **byte-identical** to before; on Linux KL
  3.9e–13, 1000 of 1000 tokens equal, same bytes as emulation build; **Windows and Linux now give
  same logits to the byte** (fixture and real model at 2 layers, 1000 positions; RoPE tables
  identical to float up to 4096 positions). Measured on clean machine (A/A 3.0% and 1.9%): prefill
  **1.03–1.08× at 512, 1.20–1.23× at 2048, 1.29–1.31× at 4000**, decode at 8 threads **1.02–1.03× /
  1.05–1.07× / 1.08–1.10×**; `attention` 2.3–2.6× shorter, `expert_act` 5.5×. `tools/lint.py`
  refuses `expf(` in hot path and compares generated constants with their generator.
- 2026-09-19 — **A cap is measured with its own tool** (`make bench-mem`), and the profiler says
  per zone how many bytes it reads and at how many GB/s: a zone at cap is memory-bound, one below
  is bound by something else (LESSONS #75–#76). Measurements at multiple contexts fit in **one**
  `ab_modes.sh` session, in an order putting each context after every other (`tools/decode_context.sh`).
- 2026-09-23 — **The engine may outlive a command** (question 49, Marcello's go): `trochilus serve`
  keeps pool, model and expert store; `generate`, `logits`, `run`, `chat` run in it when it answers,
  transparently and byte-identical, else in their own process (`TR_SERVER=0`: never; the gate and
  the measurements set it). No auto-start: a server is started by the user, and leaves after
  `--idle` minutes without a command (default 30).

## Known issues

- Windows Smart App Control blocks a just-compiled .exe for up to 20 minutes, and on 2026-09-19
  for **four hours**: hash verdict stays cached and cloud is called every two hours
  (event log `CodeIntegrity`, event 3118). Correctness in Docker, native measurements when
  binary passes; after 30 minutes waiting a **second copy in another folder** is built, without
  touching the first, and measured with the one that starts (`TROCHILUS=<binary>`; LESSONS #12, #81).
  Since 2026-09-22 links carry no PE timestamp (same code, same bytes, same verdict) and `make check`
  runs the C tests natively too, a blocked one SKIPPED: the Windows branches of `src/base` had never
  run in the gate, and two bugs lived there (LESSONS #103, #106).
- **Measurements protect themselves and the machine is queried** (LESSONS #82, #84–#89): measurement
  scripts take a lock (one measurement at a time), keep the machine awake, refuse to start if
  `tools/orphans.sh` finds something of ours left running, wait for CPU quiet before session and
  before **every run** (`tools/machine_still.sh`), declare background load in log, and finish what
  they launched (`tools/cleanup.lib`): a measurement stops by killing the pid in the lock, and now
  truly stops (a shell waiting for child in foreground was not serving signal: long steps run with
  `cleanup_run`).
- **This machine's background load**: free machine leaves 1.0–2.5 logical processors occupied, and
  **0.7–0.9 cores are from kernel `System` process, always** (LESSONS #85): sits under every
  native measurement, declared and not corrected; what is in there asks for admin session. By day,
  with Marcello at machine and another window working, sessions cannot distinguish differences
  under 5%: deciding measurements done at night.
- **The machine is shared with other Claude windows** (rule of 2026-09-23, LESSONS #73, #87,
  #127): a timing measurement writes `~/.claude/macchina-ferma` ("trochilus: <script>") for its
  duration and the other windows pause their correctness work; the containers of other projects
  are no longer stopped nor waited for (OpenEMR's idle stack kept the old guard false for hours).
  `tools/measure_guard.lib` takes the marker with noclobber, waits up to 6 h for another window's,
  refreshes it before every run, gives back only its own (`tools/test_marker.sh`,
  `tools/mutate_marker.sh`); the CPU check before every run still stops a comparison on a busy
  machine. The other half, our Docker work pausing under another window's marker, is not done:
  `make check` and `mutate_files.sh` do not look at it yet.
- Project hooks (`.claude/settings.json`) work only if Claude Code starts from `trochilus` folder:
  a session opened from Desktop does not load them, even after `/hooks` (LESSONS #18).
- Free RAM on PC often ~10–14 GB of 31: Docker containers of other projects (OpenEMR,
  colibri-dev, Redis: ~3.6 GB) and VM cache after heavy tests (LESSONS #38). One open chat holds
  7 GB: must close before real model tests.
- Tokenizer and chat template: only OLMoE (family `olmo`, template OLMoE-0125-Instruct).
  Qwen3-Coder (`qwen2`, step 2) asks for oracle on its `tokenizer.json` and its C template before
  entry. `trochilus run` stays without template (raw text); `trochilus chat` applies it. Chat is
  greedy (no sampling) and stops at context full (`/reset`).
- Generated `src/tokenizer/unicode_data.h` weighs 157 KB source: compact if it becomes a problem;
  regenerated only with `tools/gen_unicode_tables.py` when `tokenizers` pin changes.
- OLMoE in transformers 5.x: experts saved fused (`gate_up_proj` [E, 2I, H], gate first),
  `q_norm` on whole projection (hidden), `k_norm` on kv_heads × head_dim, neox-style RoPE
  (`rotate_half`), softmax on all experts then top-k, normalization only with `norm_topk_prob`.
- Speed measurements in container: models in Docker volume `trochilus-models` (GGUF + colibri
  conversion, 14 GB), not from Windows disk (LESSONS #41).
- **`trochilus serve`** (question 49): one request at a time; a client killed mid-command leaves
  the command running to its end (on POSIX still printing on that terminal); on Windows the pipes
  keep the default DACL (other local users could read them), and a running server locks
  `build/trochilus.exe` (relink fails until `serve --stop` or the idle exit); `-t 0` and `-t 16`
  are two pools to it. A rebuilt client is declined (build stamp) and runs its command itself.
  The load's rainbow bar (`src/app/bar.c`) is drawn by the process that loads: a POSIX server
  draws on the client's terminal but reads `TR_BAR`/`NO_COLOR`/`TERM` from its own environment.
- PC has no clang or CUDA toolkit: Windows build with MinGW-w64 gcc 15.2 (scoop), Linux
  in `trochilus-dev` container (Ubuntu 24.04, gcc + clang; ASan, UBSan, TSan with gcc).
- **One pool at a time** (LESSONS #63): two live pools take same slots and share same cores, and
  a `tr_parallel_for` nested on another pool gets a worker id outside its range. Today process has
  one pool and one model: must solve (slots from process register, id per pool) before second model
  in same process.
- Hybrid P/E cores (Intel) and multiple processor groups on Windows: `cpu.c` reads them but no
  machine has tested them. Work divided equally, so on P+E slowest core sets pace; Windows
  `EfficiencyClass` not read.

- Native speed measurements: no agent compiles in Docker meanwhile (LESSONS #57); a run that
  produces no number stops the comparison; the scripts wait for the engine to see 12 GiB free
  (after `make check` Windows reclaims memory minutes late, LESSONS #72). With other projects'
  containers now left up, that wait may fail more often than with `docker stop`.

## Next steps

**Day of 2026-09-24: the long-context decode, exact** (MEASUREMENTS §Skipping cached positions
exactly and the five sections after it; LESSONS #152–#157):
- **skipping positions exactly: closed** (OLMoE's attention is flat: an oracle skips ≤ 0.2% of the
  bytes); `make attn-probe` checks the premise again on the next model;
- **the decode's attention on the GPU, built** (`src/backend/gpu_attn.{h,c}`, the same bytes,
  `tools/gpu_exact.sh`): **decode 1.31–1.32× at 2048, 1.54–1.58× at 4000**; one warp keeps the
  laptop GPU awake between layers (LESSONS #153, #158);
- CPU: **one position at a time** for a decode token: exact, kept, but not distinguishable on a
  still machine (0.96–1.05× a token, the zone −9% at 2048; the bench's 1.10–1.13× was under load,
  LESSONS #160); the KV packed in 28 bits exact, 1.13× fewer bytes, not timed yet (question 57);
- speculation at long context (question 52): nil on prose, 1.57 tokens a pass rewriting code,
  1.66× net on repetitive code; a pass of ~4 rows reads ~2.8× the experts;
- **evening: the premises measured** (the project read like a genome, CLAUDE.md; MEASUREMENTS §A Q4
  draft, §Experts read by a pass of k rows, §The model read like a genome, §The CPU's peak, §The
  KV packed in 28 bits; LESSONS #161–#164). **53**: the Q4_K picks the Q8_0's token 88.7 / 94.7 /
  90.6% of the time (prose, code, Italian; Q4_K_M 90.4 / 93.6 / 92.8), under the 95% threshold, but
  **54**'s union is small (a pass of k rows reads 1.55 / 2.33 / 3.34 / 4.42× one token's experts at
  k = 2 / 4 / 8 / 16, chance 1.88 / 3.31 / 5.25 / 7.06): a pass of 8 drafts is **~2.0× fewer bytes
  a token** on prose, ~1.3× serial and ~1.9× overlapped once the GPU draft is paid. **55**: the
  matmul at 49–53% of the no-FMA peak, the stream with 8 tokens 1.34× (built: below).
  **56**: codes at 7.59–7.69 bits of 8 (Q4_K 3.83–3.87 of 4): ~7% / ~4% lossless, closed. **62**:
  0.008% of blocks repeat (the output head's 220 rows of never-learned tokens); only layer 0's
  values are token-determined; co-activation grows with depth. **57**: the packed KV 0.97–1.05×,
  closed as no. **58**: the float-only exact exp in AVX-512 and AVX2, 0 of 2^32 differ, 4.8× / 4.4×
  `tr_expf` a value; the prefill ~1.02–1.03× (question 39's range): the GPU's reference, not wired;
- **two rows against eight tokens, built** (`dot_row2_x8`, AVX-512, Q8_0/Q4_K/Q6_K, the same bits;
  MEASUREMENTS §Two rows against eight tokens, LESSONS #165–#167): intrinsics were enough;
  the lane tree in SIMD was a tenth of every two-row call. The matmul on a core 87 → 102
  GFLOP/s (61% of the no-FMA peak); **prefill 1.19× on Q8_0 and Q4_K_M at 16 threads, 1.30× on
  Q4_K_M at 8** (container, load 3–4; the native run waits for 12 GiB free);
- **late evening: pieces 1 and 4 against the references** (MEASUREMENTS §The prompt's matmul
  against the four references, §Softmax and exp against the references; ORIGINS rows 1, 4; LESSONS
  #169–#170). All four references quantize the activations to int8 (colibri as an option). Their
  kernels alone on our shapes (`tools/bench_ggml.sh`, container, loaded): **Q8_0 experts ours
  1.42×** a core (their `mul_mat_id` runs one dot at a time), the dense projections theirs 1.6×
  (tinyBLAS 4 × 4 int8), repacked Q4_K theirs 1.26–2.3×. Question 63 (ik's decode once per row,
  exact) closed: Q8_0 1.00–1.04×, Q4_K 1.1×: the matmul is bound by its input rows' loads from L2.
  The exp: ggml's rounds 3.36% of floats otherwise (≤ 2 ulp), glibc's 0.004%, ours none;
- **next, one piece at a time**: (1) close piece 1: x8's native confirmation (`prefill_context.sh
  change build/before-x8/...`) and the native engine race, both waiting for 12 GiB free; then
  question 64 (4 weight rows × 6 tokens: half the input loads per op), prediction first; (2) row 2
  (the decode's matmul), measurable with `bench_ggml.c`'s harness on one token. Queued: M3's dense
  weights on the GPU, two rows on AVX2. **For Marcello, 59 and 60**: FMA would give the 8-token
  stream 1.19× on this Zen 4; 60 (an exact dot) is not measured. 61 after M4.

**Night of 2026-09-24**: the gate 428 → 238 s; against llama.cpp (Q8_0, container, before x8)
prefill 1.41× theirs at 16 threads, decode 1.12× at context 2048 (the KV's bytes: F32 against
F16), 1.00× at 512; **M2: Q4_K and Q6_K** exact (Q4_K decode 1.5× the Q8_0; Q4_K_M runs). DONE,
MEASUREMENTS.

**Tests and gate** (2026-09-24): `mutate_auto.py` lists the mutants no check runs (gcov) and
runs ASan on the survivors only (`prof.c` 30 → 9 s, the same verdicts); `gguf.c`, `experts.c`,
`threads.c`, `platform.c` mutated for the first time (MEASUREMENTS, after §Generated mutations):
open, `gguf.c`'s 40 survivors (17 cases in `build/gguf_tests_wip.diff`, 6 wrong: LESSONS #172), `threads.c`'s 22 (the spin and the pinning: speed,
not bits), `platform.c`'s Windows half never mutated (the container compiles its POSIX half). The
native C tests run beside the container (`tools/beside.sh`): the gate's new time not measured yet.
Review of 2026-09-22/23: LESSONS #102–#125. The native measurements follow the machine's marker.

Steps of 17–20/09 are in `docs/archive/DONE.md`, their numbers in `docs/MEASUREMENTS.md`
(questions closed and open in §Open questions).

**M1 in progress** (project in `docs/ARCHITECTURE.md` §Execution «Experts (M1)», from numbers above):
expert store with slots allocated at load, index and LRU O(1), not pin from use (LESSONS #93);
**on-demand reads on calling thread, no I/O thread and no prefetch** (with 1.5 GB/s disk disk is
bottleneck: k=8 neutral, k=12 −40% in time model; 11–15% only on 3× faster disk: question 43);
single path (full budget = previous engine); minimum budget one layer plus one token; read error
fails evaluation, not process. Read as colibri does (three flaws in `docs/UPSTREAM.md` #7–#9;
taken: expert index → slot, expert in one slot only). Volume encrypted (BitLocker XTS-AES 128 in
software, question 41) and untouched: factory state of target PCs, M1 designed on ~1.5 GB/s.
- **Batches 1–3, done** (2026-09-20, detail in DONE): `src/memory/experts.{h,c}`
  with index and LRU, `olmoe.c` fetching experts from expert store (`--expert-budget`,
  `TR_EXPERT_BUDGET_MIB`, `experts:` line, `weight_read` zone), and reads **without system cache**
  (`tr_file_open_direct`, aligned to 4096, `TR_EXPERT_DIRECT=0` to force). Tests: `tests/test_experts.c`,
  `tests/test_stream.c`, oracles under `min`, real model `cmp` to byte; 23 red mutations.
- **Measured 2026-09-20/21** (MEASUREMENTS §M1 measured, §Prefill reads model once per pass):
  **M1 works, the cost is the prompt**: decode 35.5 / 29.6 / 24.3 / 20.6 tok/s at 100 / 75 / 50 /
  25% (64 tokens), 0.90× and 0.87× of resident at 1000 tokens; direct reads give 2.42× the system
  cache's prefill (the guard was right). Decided: no prefetch in decode (0.03–0.14 units a token
  far from the prompt); the budget is not the lever (4% between 25 and 75%). Question 46 half
  closed: the fixed cost at generation start is the store settling (~0.35 s at 50%), absent when
  resident. Layer-major prefill (question 47): 2048 tokens read 6,273 MiB instead of 22,880, 12.41 s
  instead of 23.51 (**1.89×**), `tests/test_stream.c` §`once_per_prompt` (LESSONS #99).
- **M1 work order, decided 2026-09-21** (from numbers, not plan):
  1. ~~layer-major order in prefill~~ **done** (line above, 1.89×);
  2. **first prompt cost** (questions 45, 48, 49; 4.7 s at 2048, the whole gap to the resident
     model): **49 built 2026-09-23** (Marcello's go): `trochilus serve` keeps pool, model and
     store between commands, same bytes out (`tests/test_serve.c`). **Measured** (MEASUREMENTS
     §The engine kept between commands): first 2048-token prompt at full budget **12.79 → 7.81 s**
     (the load gone), at half **13.00 → 12.04 s** (0.926×): as predicted, a store smaller than one
     sweep is rewritten by every prompt. Below budget the lever is the eviction during a sweep
     (keep the first layers, evict the most recent: model, up to ~2.5 s with the server). The
     load shows the rainbow water bar (variant A, Marcello's choice, 2026-09-23). 48 (read while
     the user types) after 3;
  3. **overlap disk and computation**, done 2026-09-23 (MEASUREMENTS §Reading the next layer):
     an I/O thread reads layer L+1 while L computes in the layer-major prompt; 2048 tokens at half
     budget **11.45 → 9.39 s (1.22×)** of a 1.61× ceiling, same bytes out. Next: where the other
     2.07 s of disk wait go (profile per layer), and the one-block prompt (≤ 512 tokens, pass-major,
     nothing read ahead: ceiling 1.40×);
  4. **file reordering for co-activation** (mbolt, MIT): ≤ 1.15× on this disk with these
     experts, costs our format. Later.
- **Then**: questions 42 (layer 0), 43 (from which disk up prefetch pays), 46 (fixed cost of decode).
  Three matrices in one read: **no**, three separate GGUF tensors (docs/ORIGINS.md
  §The expert store).

- **Question 44, closed 2026-09-20** (`docs/MEASUREMENTS.md` §Is code behavior a small,
  deterministic graph?): no, in all four senses; a code region common to C, Python and shell
  (Jaccard 0.68–0.77, 0.07–0.09 with prose). What remains: a second model.

**Order decided by Marcello 2026-09-19 evening**: (a) **M1**, experts from disk, starting from
measurements 13–16 and with GGUF and pool folders from component comparison inside (point 5);
(b) ~~KV at 16 bits as declared mode~~ no (2026-09-24); (c) int8 in prefill as declared mode, after 16-bit
bench (point 3, question 33); (d) `--spec` from match length (point 1, question 32); (e) closed
as «no», with rationale in MEASUREMENTS: `tr_expf` in SIMD (question 39), 2 MB pages on Windows
(question 5), width per zone (question 34). **Estimator validation (point 0) comes after all
this**, one night only with measurements of (b) and (c). Until then **every speed measurement
forces width** (`--decode-threads`), never `auto`: no conclusion rests on unvalidated estimator.

0. **Validate rewritten estimator on real model** (question 31, LESSONS #88). Code there since
   2026-09-19 evening (decision above; `tests/test_phase.c` with fake clock, ten mutations seen
   red with `tools/mutate_tune.sh`, `make check` green) and **no number yet**: no run on real
   model. Validation **at quiet machine, at night**, Marcello launches at end of order above:
   `sh tools/decode_context.sh widths` (16 vs. 8 and 4 vs. 8 at four contexts: from 2048 up 4
   and 8 in noise, and 4.6% A/A does not see −4%), `sh tools/decode_context.sh measure 16`,
   `sh tools/decode_context.sh long` (1500 tokens after 1000-token prompt: crosses 1024 and 2048,
   ~1 minute per run; prints each run's changes from `threads:` line history), `sh tools/threads_phase.sh widths`.
   Criterion: automatic chooses what forced truth indicates at each context, changes zero between
   runs, at most one 4 → 8 change in long run. Logits do not change (result does not depend on
   threads). If criterion falls, before touching a constant look at pass times of each measurement
   (added to history).
1. **Turn on `--spec` by default?** Condition «worst case as without» **not met**: 0.953× where
   model invents, 1.175× where copies (A/A, 8 rounds). Changing pause constants not enough (replay
   with measured costs: at most 0.97–0.98×); lever is draft row cost, which for short drafts at
   91% in experts (question 12). Marcello decides: on accepting −5% / +17.5%, or off as today.
   With threads per phase both ratios remeasured: `--spec 0` takes 9% from tight decode, worst of
   `--spec 8` is 6.4%, best 2%. **Before deciding** (Marcello, 2026-09-19: «branch that avoids
   −5 and keeps +17»): pause already that branch, and −5% is cost of finding it wrong; path is
   decide before trying, from match length (question 32: count without timer, then replay).
2. **Measured width, on other machines** (question 31, after point 0): `sh
   tools/threads_phase.sh widths` and `sh tools/decode_context.sh widths` on different machine, as
   soon as one available (4–8 cores, more memory channels, Apple Silicon no pin, P/E cores).
3. **Int8/VNNI in prefill, yes or no** (questions 21 and 28, LESSONS #65): where gap with
   llama.cpp is (1.8–2.3× on kernel), but not exact. If yes: declared mode (`--fast-prefill`),
   off by default, with oracle measuring logit shift. Exact path «8 tokens in registers» measured
   and discarded. Marcello decides, asking 2026-09-19 if anything exists between float and int8:
   bench measures midway at 16 bits (question 33) before deciding; serial prefill part done
   (point 4), exact side leaves `expf` (question 37: multiplications 77% of prefill at 512 and 56%
   at 4000, rest mostly `expf`) and, for decode, 4-bit models. Every inexact mode (int8, and KV
   at 16 bits point 6 now raises) decided with quality numbers front: equal tokens and KL on real
   model vs. exact mode (`tools/compare_llamacpp.py` computes it: llama.cpp at 9e–3).
4. Prefill on long prompts: **exact part and scalar `tr_expf` done** (decisions 2026-09-19 above:
   clean machine 306–315 tok/s at 512, 303–308 at 2048, 276 at 4000). Remain:
   - `tr_expf` in SIMD: **no** (question 39, closed 2026-09-19: 1.01–1.04× estimated, at noise
     threshold, for ~200 delicate rows; scalar took almost all);
   - attention groups with **GQA** and beyond 4096 tokens (question 38), when Qwen3-Coder arrives:
     8 query heads per key head can fit same group;
   - KV at 16 bits **not** needed for prefill anymore (keys and values read once per group):
     remains decode lever (point 6).
5. Component-by-component comparison (decision above), then fixes from it. **Marcello's yes
   2026-09-19, on three sources**: colibri, ds4, llama.cpp. Review of what is already built
   (kernels, OLMoE graph, GGUF, pool), one folder at a time. In graph also watch how three sources
   hold KV: if per-position as ours was, 2026-09-19 measurement (attention from 32–35 to 47–48 GB/s
   with one head's positions in a row) is flag for `UPSTREAM.md`. GGUF and pool folders done within
   M1; measurements 13–16 (routing, expert cache, SSD) are its first piece (order above).
6. Decode at long context: KV at 16 bits **closed as no** (decision 2026-09-24); the exact levers
   are the day's block above (GPU attention, one position at a time, the packed KV). Remains the
   race with llama.cpp at long context after them (question 19).
7. Memory: bandwidth measured (~57 GB/s, question 4) and decode uses 89–94%. 2 MB pages
   (question 5) closed as «no» on Windows: ask privilege normal user does not have.
8. Pinning on Linux: code there and `make check` proves it, numbers no (need real Linux machine,
   WSL2 topology synthetic). Remains second half of question 22.
