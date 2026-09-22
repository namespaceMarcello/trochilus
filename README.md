# Trochilus

An inference engine for Mixture-of-Experts models, written in C, with **no dependencies** —
libc and the operating system's threads, nothing else. It runs on any CPU (kernels are picked at
runtime, not at compile time), keeps the experts on disk when they don't fit in RAM, and produces
results that are **exact to the token** against `transformers`.

The target machine is an ordinary personal computer — 16-32 GB of RAM, a 4-8 GB GPU or none at
all, an SSD — not a 128 GB workstation. The long-term goal is to run a model of hundreds of
gigabytes on that machine, exactly, with no options to set.

> **Status: pre-alpha, under active development.** One model family works end to end (OLMoE);
> milestone M1 (experts streamed from disk) is in progress. See [Where we are](#where-we-are).
> The engineering documents under `docs/` are written in Italian; this README is the summary.

---

## What works today

- **OLMoE-1B-7B** (and any GGUF v3 file of the same family) loaded straight from Hugging Face —
  no conversion step. F32, F16 and Q8_0 weights.
- CPU kernels dispatched at runtime: scalar, AVX2, AVX-512, AVX-512+VNNI. One binary for every
  x86-64 machine. Every variant is **bit-identical** to the scalar one, and a test enforces it.
- Byte-level BPE tokenizer read from the GGUF metadata, with NFC and Unicode classes: the same
  tokens as Hugging Face `tokenizers`, at 22 MB/s.
- `run` (raw text in, greedy generation out) and `chat` (the model's own chat template, written
  in C, compared byte for byte against `apply_chat_template`).
- Block prefill, prompt-driven speculative decoding, per-phase thread widths, a KV cache laid out
  for the memory bus, and experts read from disk under a RAM budget.
- A built-in profiler (`--profile`), microbenchmarks for memory, disk, attention and `expf`, and
  a measurement harness that refuses to run on a busy machine.

## Measured

Ryzen 9 7940HX (16 physical cores), 31 GB RAM, NVMe ~1.5 GB/s, Windows 11, native build, quiet
machine, OLMoE-1B-7B Q8_0 resident in RAM. Every number is the median of N runs with the spread
recorded; full tables and method in `docs/MISURE.md`.

| Context | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|
| 32 | — | 38.9 |
| 512 | 306-315 | 35.5 |
| 2048 | 303-308 | 27.5 |
| 4000 | 276 | 20.9 |

With the experts streamed from disk instead of resident (M1, 512-token prompt): decode 35.5 /
29.6 / 24.3 / 20.6 tok/s at 100 / 75 / 50 / 25% of the expert budget over 64 generated tokens,
and 0.90x / 0.87x of the resident engine over 1000 tokens — **the budget barely matters once
generation is long; the cost is the first prompt**, which has to read the whole model once
(~4.7 s of disk).

Where the speed came from, each step measured with alternating runs and an A/A control:

| Change | Effect |
|---|---|
| Thread pool with active waiting, one slot per thread | dispatch 53 µs → 1.5 µs at 16 threads |
| One thread per **physical** core, spread across chiplets | prefill +30%, 8→16 cores 2.02x |
| One weight row against 4 tokens in registers (`dot_row_x4`) | prefill +45% |
| KV laid out `[layer][head][position]` | decode 1.15-1.18x at 4000 tokens |
| Attention in groups of 16 tokens, per-token work spread on the pool | prefill 1.11-1.14x at 4000 |
| Our own `expf` instead of the C library's | prefill 1.29-1.31x at 4000, `attention` 2.3-2.6x shorter |
| Layer-major prefill under a partial expert budget | 1.89x at 2048 tokens, 6 273 MiB read instead of 22 880 |

Tokenizer, on 2.2 MB of code and prose: **13x** Hugging Face `tokenizers`, **36x**
`llama-tokenize`, same tokens as Hugging Face.

## What we are proud of

**Exactness is an invariant, not a hope.** The logits do not change when you change the number of
threads, the batch size (`-b`), the SIMD tier, the expert budget, whether speculative decoding is
on, or the operating system. Each one of those is a test in the gate, not a claim in a README.
Windows and Linux produce the same logits **byte for byte** on the real model.

**Our `expf` is correctly rounded, and the proof is exhaustive.** `src/kernels/expf.c` is a
64-entry table plus a polynomial, with the eight hard cases computed at 200 bits. `make bench-expf`
checks **all 4 278 190 082 floats** against the reference, under gcc and under clang, and it runs
in the gate. That is what made the two operating systems agree.

**Speculative decoding that cannot change the answer.** The draft comes from the text already in
the context; verification is a single pass over 1 + k positions, and a row of a batch is bit for
bit the same computation as a single-token pass — so the generated tokens are identical with and
without speculation. It is speed, never a different result. None of the three engines we learned
from proves that invariant; two of them restore a cache snapshot when a draft is rejected, which
is both slower and weaker than a guarantee.

**One binary, every CPU.** Kernels are chosen from `cpuid`/`hwcap` at runtime, never from
`-march`. The scalar path is the definition; a SIMD variant that isn't bit-identical does not get
merged. A test also proves the tier you think is running *is* the one running — equal numbers do
not tell you which code executed.

**No number comes from a single run.** Comparisons alternate A and B run by run (measuring all of
A then all of B moves the result 10-25% as the machine heats up, and it once made us throw away a
good optimization). Every session includes an A/A control and a background-load declaration;
differences below the worst A/A of the session are written down as "not distinguishable".
Optimizations that were tried and **rejected** are documented with their numbers, not deleted.

**Every mistake becomes an automatic check.** `docs/LEZIONI.md` is the log — each entry names how
it was found and the test, lint rule or gate check that now prevents it. The hot path (code that
runs on every token) is enforced by `tools/lint.py` and `tests/test_hot.c`: no allocation, no
strings, no I/O.

## Where we are

| Milestone | Content | State |
|---|---|---|
| **M0** | base, GGUF v3, converter (F32/F16/Q8_0), CPU backend with runtime dispatch, OLMoE graph, greedy, CLI, tokenizer, chat template | **done** — exact against `transformers` on Windows and Linux; the real OLMoE-1B-7B answers |
| **M1** | experts from disk under a RAM budget: slot store, O(1) LRU, on-demand unbuffered reads, automatic plan | **in progress** — store, budget and layer-major prefill are done and measured; what remains is the cost of the first prompt (hide it behind the user's typing, or keep a store that outlives the session) and overlapping disk with compute |
| M2 | K-quants (Q4_K, Q6_K, Q2_K, IQ2_XXS), assembly lab | not started |
| M3 | CUDA module, hot experts in VRAM, VRAM + RAM + disk plan | not started |
| M4 | DeepSeek V4 Flash | not started |
| M5 | KV checkpoints on disk, server (OpenAI / Anthropic APIs), speculative decoding with a draft model | not started |
| M6 | Vulkan and Metal modules | not started |

Model ladder — we only climb when the step below is exact and measured: tiny OLMoE (done) →
**OLMoE-1B-7B (here)** → Qwen3-Coder-30B-A3B → a MoE larger than RAM → DeepSeek V4 Flash.

## What is missing

Honest list, so nobody has to find out by running it:

- **One model family.** The OLMoE graph only. The tokenizer accepts the `olmo` pretokenizer
  family and refuses anything else (a family enters only after an oracle on its `tokenizer.json`);
  one chat template. Qwen3-Coder is the next step.
- **No GPU.** The backend interface is designed for it; no CUDA, Vulkan or Metal module exists yet.
- **F32, F16 and Q8_0 only** — no K-quants, so no 4-bit models yet.
- **arm64 detection exists, NEON kernels do not.** On ARM it would take the scalar path.
- **Greedy only** — no sampling, no server, no API, no streaming interface beyond the CLI.
- **A model larger than RAM does not run yet**; the memory guard refuses to load one until M1 is
  finished.
- **The head-to-head against llama.cpp is stale.** The last full comparison (2026-09-17) predates
  a ~10x improvement in our prefill. The known remaining gap is the int8/VNNI dot kernel
  (1.8-2.3x on the kernel itself), which is not bit-exact and can therefore only ever exist here
  as a declared, off-by-default mode. A fresh comparison is pending.

## Goal

To make a large MoE model usable on the computer someone already owns. Concretely:

1. **It configures itself.** At startup the engine measures the machine — physical cores, CPU
   instructions, free RAM, VRAM, disk speed — and decides where everything goes. No option is
   mandatory; options exist only to force a choice, for tests and measurements.
2. **VRAM, RAM and disk work together**, and the experts that do not fit are read from disk while
   the dense part stays resident.
3. **The result stays exact.** Anything that is not bit-exact (int8 activations, a 16-bit KV
   cache) can only exist as a declared mode, off by default, decided with quality numbers in
   front of us.
4. **Down to the last drop of performance**, but never by trading away point 3 silently.

## Build and run

```sh
make                       # build/trochilus, no dependencies
make check                 # the gate: lint, 0-warning build, tests, ASan, TSan, oracles
build/trochilus cpu        # what the engine sees of this machine
build/trochilus run  -m model.gguf -f prompt.txt -n 200
build/trochilus chat -m model.gguf
```

C11 with intrinsics, a Makefile, gcc / clang / MinGW-w64. The Python tools (`tools/`) are only
used for conversion and for the oracles — the engine itself never needs them.

## Documents

The engineering log lives in `docs/` and is written in Italian:

| Document | Content |
|---|---|
| `docs/ARCHITETTURA.md` | principles, layers, execution, correctness ladder, milestones |
| `docs/STATO.md` | where the project stands, decisions taken, next steps |
| `docs/MISURE.md` | every measurement, including the optimizations that were rejected |
| `docs/LEZIONI.md` | every mistake and discovery, with the check that now prevents it |
| `docs/ORIGINI.md` | where each ported file comes from, commit by commit |
| `docs/COMANDI.md` | the commands: benchmarks, measurements, mutations, reports |

## Provenance and license

Trochilus is a new project, not a fork. Where a piece is taken from somewhere else it is taken by
hand, and it says so in its own file header and in `docs/ORIGINI.md` (project, commit, path, what
was changed): from [colibri](https://github.com/JustVugg/colibri) (Apache-2.0) the CPU work, the
disk handling and the exact tests; from [ds4](https://github.com/antirez/ds4) (MIT) the GGUF
reader, the thread pool and the GPU execution model. From llama.cpp / ggml (MIT) we have taken
ideas so far, not code, and only where a measurement said it was worth it. `NOTICE` is the
authority on all of this.

Apache-2.0 — see `LICENSE` and `NOTICE`.
