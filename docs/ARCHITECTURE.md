# Architecture

Trochilus runs large MoE models on **any** machine and uses every byte: the target is
an ordinary person's computer (16–32 GB RAM, a 4–8 GB GPU or none, an SSD), not a
128 GB workstation. VRAM, RAM, and disk work together; experts that do not fit in memory
are read from disk. Results are exact to the token against the reference. The default build
has no dependencies.

Reference machine for measurements: Ryzen 9 7940HX laptop, 31 GB RAM, RTX 4070 Laptop 8 GB,
NVMe 1 TB. Low floor to keep alive: CPU only, 16 GB, no GPU.

Takes from colibri (true CPU, disk, exact tests) and ds4 (GPU, format, KV cache to disk).
Where each ported file comes from: `docs/ORIGINS.md`.

## Principles

0. **It adapts on its own.** On startup it measures the machine (physical cores, CPU
   instructions, free RAM, VRAM, disk speed) and decides where to put what: dense weights,
   hot experts, KV, threads. No mandatory option; options serve only to force a choice.
1. **CPU is an engine, not a reference.** Every model runs first on CPU; GPU accelerates.
2. **One binary for all CPUs.** Kernels chosen at runtime (cpuid / hwcap), not with `-march`:
   the same executable uses AVX-512 VNNI where it exists and AVX2 or scalar elsewhere. Neither
   colibri nor ds4 does this.
3. **Scalar defines the numbers.** Every SIMD or assembly variant sums in the same order
   and is **bit-identical** to scalar, verified by tests. A non-identical variant does not enter.
4. **Zero dependencies in the core**: libc, system threads. No OpenMP (on macOS and MinGW it
   is an extra library), no third-party libraries. GPU backends are modules loaded at runtime:
   the core starts even where CUDA or Metal are absent.
5. **The format is GGUF v3** with standard ggml type numbers and tensor names from llama.cpp:
   a file downloaded from Hugging Face opens without conversion.
6. **Generic operations, graphs per model.** The backend exposes primitives without model name
   (quantized matmul, rmsnorm, rope, attention, top-k, swiglu). A fused operation for one model
   is allowed only as acceleration, with generic composition as fallback.

## C and assembly

The engine is written in C, and the points where time is really spent are ported to assembly.
In an inference engine almost all time lives in a few kernels (dot products of quantized
matrices, attention, dequantization); loading, tokenizer, and orchestration gain nothing from
assembly and would lose portability.

The cycle for each kernel:
1. scalar C version: defines the result bit by bit;
2. version with intrinsics (AVX2, AVX-512, NEON): same arithmetic, identity test;
3. **assembly version** for the reference CPU (Zen 4, AVX-512 VNNI), written in GAS `.S` with
   macros for the two calling conventions (System V and Windows x64);
4. microbenchmark of the three: the fastest enters the dispatch table for that CPU.

The profiler decides the order: we port to assembly first the kernel that weighs most in
the measurement.

**Every optimization is taken, however small**, at any level (algorithm, memory, instructions,
assembly), under two conditions verified by tools, not by eye:
- **it does no harm**: tests find it bit-identical to the previous version;
- **it really gains**: the benchmark (median of at least 5 runs, machine at steady state) finds
  it faster by more than the noise measured on that benchmark. Below the noise is not a gain.

Every attempt, successful or rejected, goes in `docs/MEASUREMENTS.md` with the number: one
rejected today is not retried tomorrow without new reason.

**Below assembly.** Hand-written machine code adds nothing to assembly, which is already 1:1.
What remains at that level is the **JIT**: generate machine code at model load time, with
true dimensions (row lengths, blocks, heads) written as constants and instructions chosen
for the exact CPU. Experiment after assembly, with pages never writable and executable
together. Microcode and firmware are not touched; overclocking and undervolting are out. Of
the system only what is reversible is used: process priority, huge pages, power plan.

## Profiling

Data decides where we work. Three levels, from finest to widest:

| Level | Tool | Answers |
|---|---|---|
| kernel | `tests/bench_kernels.c` (`make bench`) | how fast is a single kernel per tier, with noise |
| memory | `tests/bench_mem.c` (`make bench-mem`), `tests/bench_attn.c` (`make bench-attn`) | what does the machine's RAM give, the decode cap: sequential and sparse reads at 1–16 threads, the engine's matmul on random expert matrices, one token of attention on KV; and the attention of a whole prompt on one layer, broken down: products, softmax and weighted sum alone, one query at a time against a group of queries per key block, with bit control. Threads are from the pool, pinned as in the engine |
| engine | internal profiler, `trochilus generate --profile` / `--profile-json <file>` | where time goes for one token: per phase (embedding, projections, norms, rope, attention, router, experts, lm_head, sampling), prefill and decode separate, thread pool waits, disk reads; **per zone bytes read (weights and KV) and GB/s**: a zone at RAM ceiling is memory-bound, one below is bound by something else |
| scenarios | `tools/profile_suite.py` (`make profile`), scenarios in `bench/scenarios.json` | how a real use fares (short prompt → long response; long file → short response; context that grows), median of N runs with spread, automatic comparison with the previous measurement on the same machine: improved, regressed, or noise |

The profiler is not global (lives in the session), costs a single conditional jump when off,
and uses the CPU counter (RDTSC with invariant TSC, else the system clock). Results of the
suite go in `bench/results/<date>-<commit>-<machine>.json`; the summary and decisions that
come from it in `docs/MEASUREMENTS.md`.

A native measurement is worth as much as the machine it runs on, and the machine is questioned,
not assumed (`docs/LESSONS.md` #84–#88: four forgotten processes under two days of measurements):

| Rule | Control |
|---|---|
| every script finishes what it launched, and a signal stops it right away | `tools/cleanup.lib` in every `tools/*.sh` (required by `tools/lint.py`); `tools/test_cleanup.sh` in `make check`, red without the trap at every run |
| nothing starts next to something the project left on | `tools/orphans.sh` at head of `make check` and every measurement (`measure_begin`) |
| the machine is loaded on purpose in one way only | `tools/busy_machine.sh <n> <command>`: load generators die with it |
| the machine is still before the session and before every run | `measure_still` and `tools/machine_still.sh` in `AB_GUARD`: occupied processors and container; waits, then stops and says who is holding the CPU |
| every measurement declares the background load; if it is not low conclusions are not drawn | `measure_declare` writes to log occupied processors and quota of `System` before the first run and after the last |
| a comparison has its own A/A, and a choice is judged from distribution, not from median | `tools/ab_modes.sh`; `tools/decode_context_report.py speed` prints choices and width changes |
| a change to the decode is measured with the short protocol by default (Marcello, 2026-09-24): exactness first, the decode forced on 8 at 512 / 2048 / 4000 with A/A, the profile after; ~40 min. **Short** for what grows with the context: attention (CPU or GPU), the KV cache and its layout, anything whose cost is per cached position. **Long** (~90 min: adds context 32, the measured width, bench_mem) for what every token pays at any context: the weights' kernels and quantized types, the expert store, norms, RoPE, router and experts' path, the thread pool, pinning, threads per phase and the width's estimator; and whenever the short one shows a surprise at 512 | `sh tools/decode_context.sh change-short <before>` / `change <before>` |

## Hot path

The code that runs every token sits between `/* hot: begin */` and `/* hot: end */` (model,
kernel, thread pool, profiler). The rest (loading, command line, reading files) is the
cold zone: runs once, and clarity and defense against malformed files count there.

Rules of the hot path, each with its control:

| Rule | Control |
|---|---|
| no memory allocation or release | `tools/lint.py` on source; `tests/test_hot.c` counts allocator calls while the model generates (must be zero) |
| no strings, prints, files, environment variables | `tools/lint.py` |
| no `pow`, `sin`, `cos`, `log` per element: computed once in table | `tools/lint.py` |
| no `expf`/`exp` from C library: exponent is `tr_expf` (`src/kernels/expf.c`), rounded correctly on every float, same bits on every platform, no library call inside | `tools/lint.py`; `make bench-expf` proves it on all 2^32 floats, in `make check` with gcc and clang; `tests/test_expf.c`; `tools/mutate_expf.sh` |
| same logits with any number of threads and with profiler on | `tests/test_hot.c` (pool of 1, 2, 3, 8 threads) |
| every optimization measured before and after, kept only above noise | line in `docs/MEASUREMENTS.md` §Attempts |

An exception is written on the same line, with reason: `/* hot-ok: pow -- reason */`. Lint
rejects it when the line no longer uses that name. Comments cost nothing at runtime (same
machine code with and without, verified): in the hot path only those explaining a constraint
remain.

## Machine safety

- Before loading we estimate the memory needed; if less than 2 GB or less than 10% free RAM
  would remain, the engine refuses and says how much is missing. Never swap.
- Threads at most as many physical cores; benchmarks last at most 60 s per run.
- No model larger than RAM while streaming from disk (M1) is not tested.

## Layers

| Folder | What it does | Where the idea came from |
|---|---|---|
| `src/base/` | platform (files, `pread`, O_DIRECT, time, aligned memory), thread pool, CPU detection | colibri `compat.h`, `omp_tune.h` (physical cores); ds4 pool `ds4_parallel_for` |
| `src/format/` | GGUF v3 reader, type table, metadata | ds4 `parse_metadata` / `parse_tensors`, without architecture hooks |
| `src/kernels/` | CPU kernels per quantized type: scalar + AVX2 + AVX-512 (+VNNI) + NEON, dispatch table | colibri `quant.h`, `expert_ffn.h`; ds4 K-quant references |
| `src/backend/` | backend interface (tensors on device, graph per token) and CPU backend; today `gpu_attn.{h,c}`: the decode's attention on an NVIDIA GPU with the CPU's bits (driver `nvcuda.dll`/`libcuda.so.1` opened at run time, PTX written in C and compiled by the driver, every float op `.rn`, `tr_expf` ported whole, the KV mirrored in VRAM, zero copy, one warp keeping the GPU awake between layers; `TR_GPU=0` keeps it on the CPU) | execution model: ds4 `ds4_gpu.h`, reduced to generic primitives; the attention kernels: new code, from measurements (`docs/MEASUREMENTS.md` §The decode's attention on the GPU) |
| `src/memory/` | expert store (M1: RAM / disk; VRAM at M3): units (layer, expert), slots allocated once, direct index and LRU O(1), reads on demand from GGUF | ideas: colibri `olmoe.c` (expert index → slot, expert in one slot), ds4 streaming; choices from our measurements (`docs/MEASUREMENTS.md` §M1): LRU not pin from use, no I/O pool, no prediction-based preloading on slow disks; in the layer-major prompt one I/O thread reads the next layer while this one computes (1.22× at 2048); new code |
| `src/kv/` | KV cache `[layer][head][position]`: a head's positions in row, so attention reads them at RAM bandwidth (`docs/MEASUREMENTS.md` §Decode at long context); then prefix reuse, checkpoint to disk with decay-weighted score | layout: new code, from measurements; ideas for the rest: colibri `kv_prefix.h`, `kv_fp8.h`; ds4 `ds4_kvstore.c` |
| `src/tokenizer/` | byte-level BPE from GGUF metadata (pretokenizer families allowed only with oracle), NFC and Unicode classes probed from HF `tokenizers`, chat template per architecture | ideas: colibri `tok.h` (regex replayed in C), ds4 `vocab_load` (from GGUF); new code |
| `src/models/` | one graph per family, built from primitives | colibri `olmoe.c`, ds4 / colibri DeepSeek V4 |
| `src/gen/` | how the token is chosen after the model: greedy, draft from prompt and verify in one pass (then sampling and stop criteria) | ideas: colibri `v4_ngram_draft`, llama.cpp `examples/lookup`; new code |
| `src/app/` | CLI; `bar.c` the rainbow water bar on stderr while the model loads (the core only reports bytes, `tr_model_load_progress`); `serve.c` keeps the engine (pool, model, expert store) between commands, which run in it byte-identical (local endpoint: named pipe / Unix socket); then HTTP server (OpenAI and Anthropic APIs) | both; the local server: new code |
| `backends/cuda/`, `backends/metal/` | loadable GPU modules | ds4 `cuda/mmq` (ggml, MIT), `metal/*.metal` |
| `tools/` | HF → GGUF converter, tiny model generators (Python, outside the engine) | colibri `tools/make_*_tiny.py` |
| `tests/` | kernel tests (SIMD = scalar), tiny oracles, microbenchmarks | colibri |

## Execution

- **Weights**: read with `pread` into own buffers, not `mmap`: resident memory stays under
  control (colibri `st.h`, mmap RSS bug). Dense: loaded at startup. Experts: on demand.
- **Experts (M1)**: the dense (attention, norms, router, embedding) always stay in RAM; experts
  pass through the store from `src/memory/`. A unit is a (layer, expert) with its three matrices
  in one slot only; slots are allocated all at load, as many fit in the **budget**. After the
  router the graph asks for units of the layer (`acquire`): present ones are touched (LRU),
  missing ones are read **right away and in sequence, on the calling thread**, from GGUF at tensor
  positions, evicting the least recently used; units asked by the current pass are not evicted.
  Read **without the system cache** (`tr_file_open_direct`), else the model would end in RAM
  a second time, the very memory the budget was to save, and every measurement would report
  RAM bandwidth instead of disk bandwidth. The price is alignment to 4096: the start of a
  part in the file is not aligned and its remainder depends on which expert, so the slot holds
  a margin of one sector per part and the aligned read lands inside without copies; where the
  part starts inside the slot is recorded at every fill. If the filesystem refuses (open trial
  and an aligned test read), we fall back to normal read and the `experts:` line says so.
  One path only: with the budget covering all, the store fills at load and never misses,
  which is the old engine. Same bytes in same kernels: logits are byte-identical with any
  budget, and the test demands it. Why this way (`docs/MEASUREMENTS.md` §M1): LRU beats
  pin from use at any capacity; disk gives the same bandwidth to one and eight readers, so
  no I/O thread until there is something to overlap; without prediction there is nothing to
  overlap, and prediction (the router of the next layer, 92–95%) on a 1.5 GB/s disk costs
  more than it pays. I/O threads and preloading come together, as option that the plan
  switches on for fast disks (question 43). Minimum budget: units of one whole layer plus
  those of one token, or the engine refuses. A read error makes evaluation return -1 and
  leaves the session as it was, never closes the process.
- **Auto plan (M1)**: at load we measure available RAM and subtract the reserve
  (§Machine safety), the dense, and one session at default context; what is left, up to
  covering all experts, is the budget. `--expert-budget <MiB>` forces it, for tests and
  measurements.
- **Passes**: one eval call runs in passes of at most `n_batch` tokens (512, `-b`); decode
  is one pass of one token, same code. In one pass every number is the same kernel call as
  with one token only (one matmul element = one `dot_row`; norms, RoPE, router, and expert
  sum per token; one token's attention on positions up to it), so logits and cache are
  bit-identical for every `n_batch`. Experts work on (token, expert) pairs ordered by expert;
  matrices are visited in token blocks, and one weight row goes against 4 tokens at a time
  in registers (`dot_row_x4`: one load and one conversion for four products, each identical
  to its `dot_row`). One head's attention runs at **groups of 16 tokens** (`tr_attention_group`):
  a block of 64 positions meets all queries of the group while in cache, 4 positions per
  query load (`dot_f32_x4`, `axpy_f32_x4`), so keys and values are read once per group and
  not once per token; decode is a group of one query. Work that is one token only (norms,
  RoPE, KV write, router choice, rows for experts) divides on the pool per token, at least
  8 per piece: below, stays on the calling thread. Logits of the last token, or of the last
  `n` positions when needed to verify a draft (`tr_session_eval_rows`, at most
  `TR_LOGIT_ROWS_MAX`).
- **Draft from prompt** (`src/gen/`): the draft is the continuation of the last occurrence
  of the tail n-gram in the context; one pass verifies 1 + k positions and keeps only tokens
  the model would have chosen anyway, the rest vanish with `tr_session_rewind`. Since each
  row of a pass is bit-identical to the one-token pass, generated tokens are the same with
  and without drafting: it is speed, never a different result.
- **Threads**: pool persistent, sized on **physical cores** (colibri: +2.3x on Zen 3 versus
  logical cores), `parallel_for` on row ranges. Counting is not enough: if not pinned to
  cores, Windows places two on the same physical core and prefill loses 30% (`docs/MEASUREMENTS.md`
  §Where do threads go). **Threads per phase**: a long pass (the prompt) is bound by compute
  and uses the whole pool; a short pass (decode, short draft: up to 4 rows) is bound by weight
  reads and uses the first n slots of the pool. n is not a constant: every session **measures**
  it on its first one-token passes (whole pool, half, quarter, never below 4 threads), keeps
  the tightest within measured noise on those same passes, remeasures every doubling of context,
  and changes only after two concordant measurements; `--decode-threads` forces it. Width
  changes speed, never a logit (`docs/MEASUREMENTS.md` §Threads per phase).
- **GPU**: all of one token in a single command batch, tensors stay on device (ds4). Today only a
  decode token's attention runs there, a call per layer, same bits (`src/backend/gpu_attn.h`); a
  laptop GPU sleeps in the CPU's gaps between layers unless kept awake (`docs/LESSONS.md` #153).
- **KV**: in memory per session; prefix reuse by token id; checkpoint to disk with
  score `(decayed hits + 1) × tokens / bytes` (ds4).

## Correctness

| Level | What it compares | Where |
|---|---|---|
| kernel | every SIMD/asm variant against scalar, bit by bit, on random input | `tests/test_kernels.c` |
| tier | the whole engine under every tier (`TR_CPU_MAX`): model test, and logits byte-identical across tiers, threads, `-b` | `make tier-check` (`tools/tier_check.sh`) |
| tier is used | same numbers do not say which code ran: every hot entry of every tier is its own function, per weight type; and in the engine products counted on the active table, type by type, are exactly rows × tokens | `tests/test_tier_used.c`, also under every tier in `make tier-check` |
| tiny model | greedy tokens **identical** to transformers (f32, f16); logits within tolerance per position; q8_0 report only, because reference is not quantized | `tools/make_tiny_olmoe.py` → `tools/oracle.py` (`make oracle`) |
| real model | true OLMoE cut to 2 layers against transformers on same Q8_0 weights dequantized: identical tokens, logits within 1e-3 | `make oracle-real` (skipped without the model) |
| exact optimization | logits of true model before and after, byte-identical, with multiple thread counts | `trochilus logits` + `cmp`, by hand |
| prefill in blocks | logits and cache with many tokens per pass identical bit-for-bit to one token per pass: `n_batch`, split calls, threads, f32/Q8_0, rewind | `tests/test_prefill.c`; `tools/oracle.py` (`logits -b 3/64/whole` to byte) on tiny and OLMoE 2-layer |
| threads per phase | every logit byte-identical to one thread only with measured width and every forced width; the narrowed pool uses only the first n workers; choice among widths on fake times | `tests/test_phase.c`, `tests/test_base.c` (`test_pool_active`), `make tier-check` (`--decode-threads`) |

## Model scale

We start small and climb only when the rung below is **exact and measured**. At each rung
we measure tokens/s (prefill and decode), RAM, first token, and the same model on **llama.cpp**
and **colibri** on the same machine: it is the only way to know what Trochilus is really worth.

| Rung | Model | Size | What it tests |
|---|---|---|---|
| 0 | OLMoE tiny, random weights | 1 MB | correctness against transformers |
| 1 | OLMoE-1B-7B | ~7 GB Q8_0, ~4 GB Q4 | CPU kernels, threads, tokenizer: all in RAM |
| 2 | Qwen3-Coder-30B-A3B | ~17 GB Q4 | a MoE for code that still fits in 31 GB; long prompts, files reread |
| 3 | a MoE larger than RAM | > 31 GB | experts from disk, VRAM + RAM + SSD plan |
| 4 | DeepSeek V4 Flash | hundreds of GB | the target of colibri and ds4, on an ordinary machine |

## Horizon (after M2)

The speed limit in generation is data movement: tokens/s ≈ memory bandwidth ÷ bytes read
per token. The directions that attack that division, one at a time, under the rule
«does no harm + really gains»:

| Direction | What it does | Attacks |
|---|---|---|
| speculative decode | a small model (or text already in prompt) proposes more tokens, the large one verifies them in one pass | more tokens per weight read |
| diffusion models | models trained to compose the whole response together and refine it in few passes (for code: DiffuCoder, Dream-Coder) | one pass for many tokens; a different family, supported separately |
| expert prediction | loads experts that will be needed before the router picks them | disk reads in wait |
| arrangement by frequency | hot experts in VRAM, warm in RAM, cold on SSD, in read order | data distance |
| activation sparsity | skips neurons that will stay near zero, predicted early | bytes read per token |
| weights at few bits (2-bit, ternary) | fewer bytes per weight; with -1/0/+1 multiplies become sums | bytes read per token |
| JIT | machine code generated at model load with model dimensions as constants | CPU work |
| superoptimization | automatic search for the fastest instruction sequence for tiny kernels | CPU work |

## Milestones

| Milestone | Contents | Done when |
|---|---|---|
| **M0** | base, GGUF, converter (F32/F16/Q8_0), CPU backend scalar + AVX2 + AVX-512 with dispatch, OLMoE graph, greedy, CLI | tiny oracle exact on Windows and Linux; true OLMoE-1B-7B answers |
| M1 | experts from disk with RAM budget: slot-based store, LRU O(1), reads on demand; **auto plan** (measures RAM, picks budget); then, on fast disks, I/O threads and preloading | small forced budget → byte-identical logits; no option necessary |
| M2 | K-quant on CPU — **Q4_K in** (2026-09-24: scalar, AVX2, AVX-512 bit-identical, gguf-py bit for bit, real model decode 1.5× Q8_0), then Q6_K, Q2_K, IQ2_XXS; assembly workshop | bit-identical kernels, microbenchmarks |
| M3 | CUDA, exact (every op `.rn`, the CPU's orders): **the decode's attention in** (2026-09-24, logits identical to the byte through the engine), then the dense weights (Q8_0 GEMV measured exact at 97% of VRAM bandwidth), hot experts in VRAM, VRAM + RAM + disk plan (model: ~4× at 2048) | same bytes as CPU; a model larger than RAM runs on reference PC |
| M4 | DeepSeek V4 Flash | tiny oracle exact; runs on reference PC |
| M5 | KV checkpoint to disk, server, speculative decode | — |
| M6 | Vulkan modules (AMD/Intel GPU, integrated; colibri has `backend_vulkan.c`) and Metal | same tokens as CPU |
