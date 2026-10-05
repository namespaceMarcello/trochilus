# Status

Replace, do not append. Cap 40 KB. History is in `archive/DONE.md`.

## Decisions

- 2026-10-04 — **Qwen3.8-Flash-Next FP8's expert queue belongs to R3** (Marcello): its experts at the drive's
  limit and the VRAM as their cache, question 89 (colibri measured: 0.6 of 2-3 GB/s used, #296-#297). Not before R2.
- 2026-10-02 — **The roadmap is a ladder of models, for the machines most people own** (Marcello): R1 OLMoE,
  R2 Qwen3-Coder-30B, R3 a MoE larger than RAM, R4 DeepSeek V4; each through five phases (exact, studied, at
  the theoretical limit on three machines, usable, raced) before the next; one rung in progress (lint). The
  old milestones by technology map into the rungs (ARCHITECTURE §The roadmap, LESSONS #262).
- 2026-10-02 — **Every piece against colibri and ds4, and the verdict acts** (Marcello): worse, their way
  adopted; better, ours offered (PR ready, opened after his yes). `docs/UPSTREAM.md` §Our pieces against
  colibri and ds4, checked by the lint; today 1 of 14 raced (level).
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
- 2026-09-18, repriced 2026-09-27 — **On a MoE a draft row is not free**: its new experts (2.3–4.7
  of 8 a layer) are its price. 09-18: 17.6 ms alone, 13.7 each at eight, of a 31.3 ms pass; today
  (`tools/row_price.sh`, MEASUREMENTS §The post-it taken apart) 0.31–0.43 of a pass at 2–3 rows on
  Q8_0 and 0.33–0.39 on Q4_K since question 78 (0.50–0.72 before, above their bytes: LESSONS #224),
  0.23–0.30 at 9. The adaptive draft stops after an all-wrong
  draft for 1, 3, 7, 15, then 16 steps (LESSONS #60). `--spec` stays **off by default** (point 1).
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
  validation on real model (first of next steps). **2026-09-27**: each size of verify pass measures its
  own width the same way, the narrowest skipped (question 82: a 3-row pass wants 8 where the decode
  wants 4).
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

- **2026-10-02/03: the disk at its limit** (MEASUREMENTS §The disk at its limit; LESSONS #263-#269): the
  request's size is the lever (1.9 GB/s at 2 MiB, 3.5 at 64); the store's runs, the slots touched first and a
  run's three parts in flight: a resident load's read 4.07 -> 1.95 s, a first prompt at half budget 4.39 ->
  2.76 s (median of 8, the same tokens).
- **2026-09-24/26** (MEASUREMENTS to §Exact sums at the float's speed; LESSONS #152-#222): the GPU decode
  attention (1.31-1.58×), Q4_K decode level with llama.cpp at 4 threads, prompt attention in tiles,
  phase-major (164.7 GFLOP/s a core) and exact Q4_K; Q8_0 16 threads prompt 1.33-1.42x. No: SMT, 2 MB pages.
- **2026-09-27** (MEASUREMENTS from §The post-it taken apart to §The routers as bf16; LESSONS #223-#253):
  a calibrated gate 1.077x on new code (with a real context 1.146x); the short verify passes at their bytes
  (question 78: Q8_0's 3 rows 62.7 -> 43.0 ms; Q4_K's panel rows brought ahead, then **road (c)**, `q4x_dot_xt`
  decoding two weight rows once for 2-3 tokens: 3 rows 28.00 -> 26.45 ms); every verify size its own width
  (question 82). #242's decode +1% did not reproduce: two files of the same bytes differ by 1-2% (#243), so small
  changes race inside one process (`generate --ab`, `tools/ab_inproc.sh`, 0.28%; the arms aperiodic, #247).
  **Question 83** (Marcello: half a prep a token): q/k/v from one prep, gate/up through a map, no gather, the swiglu
  with the down's prep: the prompt **1.058 x 1.0087**, the decode **1.025**, a 3-row pass 1.026; then the argmax
  split over the pool (116-118 -> 6 us a token): a token ~0.7%, a 3-row pass 1.0175. Every bit the same.
  The KV's pages touched before a pass's layers: the decode +0.17% (#250). Environment switches raced process
  against process (`tools/ab_env.sh`). The routers as bf16: ~0.4% a token, the same bits. The parallel argmax fixed (#252: +0.74%; the lint refuses `malloc(sizeof *x)`).
  Question 84 answered (a region's end ~0.8% of a dense call; #254-#256). The idle workers (piece 3, B): level in the
  engine, off by default (`TR_POOL_HINT=2`; #255, question 85). **10-02, pretouch**: out (#259).
  **10-02, piece 4**: q, k, v and gate, up one call each, the weights' items one flat range
  (`tr_matmul_q4x_prepared_n`): the decode **1.0054 +- 0.0007**, the same bits (MEASUREMENTS §Piece 4, #260-#261).
- **next: the ladder** (ARCHITECTURE §The roadmap, status.json R1-R4): **R1, OLMoE to the end**. **10-03, the three
  machines measured** (`tools/machines.sh`): the average one at its RAM's limit, this PC at 98-99% of its RAM's measured
  ceiling. **10-03 night, the 8 GB machine's store** (MEASUREMENTS §The 8 GB machine's store; LESSONS #274-#279): its
  disk emulated (`TR_EXPERT_DISK_MBPS=500`, a SATA-class estimate), the plan on the run's own positions (the
  session's exact bytes: Q8_0 83 -> 290 slots, Q4_K 224 -> 615), ds4's eviction the default (replayed first:
  `tools/evict_replay.py`): the decode **0.57 -> 1.70 tok/s (Q8_0), 2.52 -> 8.02 (Q4_K)**, 3.67x fewer bytes a Q8_0
  token from disk at 64 tokens. There the decode is the disk's (72-94%), a first Q8_0 prompt 82% disk, the Q4_K
  prompt half compute. **10-03, AVX2's own Q4_K tile** (MEASUREMENTS §AVX2's own Q4_K tile;
  #280-#282): the W16 panel in two halves, tiles of 3 (T = 4 spills): the weak Q4_K prompt **20.6 -> 35.1 tok/s**, its
  cores' 36.4 -> 137.9, the same tokens. **10-03 afternoon, a pass reads the next layer ahead** (MEASUREMENTS §A pass
  reads the next layer ahead; #283-#285): a second run in flight gives the weak disk nothing (one queue); its one-pass
  prompt read on demand with the compute never overlapped. Now the next layer is read ahead in 8 MiB requests and
  what it does not ask dropped unread: the weak prompt **Q4_K 35.3 -> 45.5 tok/s, Q8_0 20.3 -> 24.6**, at 97-98% of
  its disk's time for its bytes (+2-3% bytes), the same tokens. **10-03 evening, the prompt's routings
  told to the store** (MEASUREMENTS, same name; #287-#289): a pass adds each unit its tokens' routings scaled to 64:
  the weak decode **Q4_K 8.00 -> 15.50, Q8_0 1.91 -> 2.38 tok/s**, the same tokens.
  **10-03 night, the KV grown from the store's room** (MEASUREMENTS, same name; #290-#291): the store gives the KV
  its slots as it is written; a chat's decode on weak **Q8_0 1.22 -> 2.29, Q4_K 4.55 -> 14.77 tok/s**.
  **10-04, offered to colibri** (UPSTREAM row 7; #293-#294): `kv_room_fit` (fork branch `perf/olmoe-kv-room`), on
  colibri's engine a request **1.19x**, the logits byte-identical: **colibri #1873**, waiting for its review.
  **10-04, the eviction told the future** (#304-#307): closed, heat stays. **10-04 night, the routes as time**
  (#309-#313, replayed): the window, not the prediction, is the limit. **10-05, the arrival order** (MEASUREMENTS
  §The arrival order built; #314-#317): a layer's misses read by the I/O thread while its present experts compute,
  the late ones in waves as they land, the same bytes and bits: the weak decode **Q4_K 14.98 -> 16.08, Q8_0 2.33 ->
  2.40 tok/s**; its handoffs 0.16% of a run (#318, q. 91).
  **10-05, the router's read ahead** (MEASUREMENTS, same name; #323-#332): the next layer's guess (27 MiB),
  read in pieces, stopped if not named: the weak decode **Q4_K 1.031, Q8_0 1.000-1.005**. Owed: the
  old binary's race (#332), the average machine's Q8_0.
  **Next**: question 94 (build/prompt-draft.md); a miss as one request, this PC
  at half budget; then the integrated GPU (r1-igpu:
  Vulkan; q. 87-88); then R1's first prompt's cost, generation's fixed cost, the review, the server; the races
  with llama.cpp, colibri and ds4 together at R1's close. The decode's
  speed backlog (R1 phase 3 where the weak machines need it): question 85's serial steps as messages; the pool's
  messages; the prep's bit arithmetic; the dense at 3 rows, AVX2's own xt, #229, questions 79-80, the gate in C,
  `--spec` by default; the GPU A/B at a free machine; W16's deletion series; AVX2 tile's tails (f64 16%, broadcasts
  15%); `dot_row2` dead code;
  the race of whole binaries (#243-#244); questions 68, 74, 76; **for Marcello, 59 and 60**.

Mutants open: `threads.c`'s 22 (speed, not bits), `platform.c`'s Windows half (#268). Steps of 17–20/09: DONE.

**M1** (`docs/ARCHITECTURE.md` §Execution «Experts (M1)»): slots allocated at load, ds4's eviction; demand reads
on the I/O thread under the compute, a prompt's next layer read ahead, a decode's router guesses the next; one path (full
budget = the engine before M1); a read error fails the evaluation, not the process. The volume encrypted
(BitLocker, question 41) and untouched: the target PCs' factory state.
- **Batches 1–3, done** (2026-09-20, in DONE): the store, `--expert-budget`, direct reads; 23 red mutations.
- **Measured 2026-09-20/21** (MEASUREMENTS §M1 measured, §Prefill reads model once per pass):
  **M1 works, the cost is the prompt**: decode 35.5 / 29.6 / 24.3 / 20.6 tok/s at 100 / 75 / 50 /
  25% (64 tokens), 0.90× and 0.87× of resident at 1000 tokens; direct reads give 2.42× the system
  cache's prefill (the guard was right). Decided then: no prefetch in decode (0.03–0.14 units a token
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
1. **Turn on `--spec` by default?** Condition «worst case as without»: 0.953× on new code in 09-18's
   A/A; **2026-09-27** at today's Q4_K prices the replay gives the adaptive lookup 1.014× on new code
   and a calibrated gate 1.077× (1.067× and 1.146× with a real context, question 81; on Q8_0 after
   question 78 1.040× and 1.103× on new code), the copy at a fixed draft of 8 1.78–1.95× (native,
   code-edit): a native A/B on new code first (after Q4_K's short groups), then Marcello decides.
2. **Measured width, on other machines** (question 31, after point 0): `sh
   tools/threads_phase.sh widths` and `sh tools/decode_context.sh widths` on different machine, as
   soon as one available (4–8 cores, more memory channels, Apple Silicon no pin, P/E cores).
3. **Superseded 2026-09-26** (MEASUREMENTS §Phase-major): the exact definition in phase-major order
   runs at the float's peak and levels llama.cpp's int8 prompt at 4 threads, so an int8 mode is no
   longer the lever (its history: MEASUREMENTS questions 21, 28, 33; LESSONS #65). Open the other
   way: E32, a more exact definition (4 digits, correctly rounded 96-99%) at ~0.85× phase-major.
   Every inexact mode is decided with equal tokens and KL in front (`tools/compare_llamacpp.py`).
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
   are the day's block above (GPU attention, one position at a time, the packed KV). The race at
   long context read as a line (2026-09-26, question 69): on the CPU the gap is F16's bytes.
7. Memory: bandwidth measured (~57 GB/s, question 4) and decode uses 89–94%. 2 MB pages
   (question 5) closed as «no» on Windows: ask privilege normal user does not have.
8. Pinning on Linux: code there and `make check` proves it, numbers no (need real Linux machine,
   WSL2 topology synthetic). Remains second half of question 22.
