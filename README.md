<p align="center">
  <img src="assets/logo.png" alt="Trochilus" width="170">
</p>

<h1 align="center">Trochilus</h1>

<p align="center"><sub><em>First iteration of the logo. It will be revisited and will very likely change.</em></sub></p>

An inference engine for Mixture-of-Experts models, written in C, with **no dependencies** — libc
and the operating system's threads, nothing else. Kernels are chosen at runtime, so one binary
serves every CPU; the experts live on disk when they don't fit in RAM; and the result is **exact
to the token**, verified against `transformers` in the gate.

> **Status: pre-alpha, under active development.** One model family works end to end (OLMoE), in
> Q8_0, Q4_K and Q4_K_M; experts can be streamed from disk under a RAM budget (M1), and the
> decode's attention runs on an NVIDIA GPU with the CPU's exact bytes (M3, first piece).
>
> **This README is a first iteration too.** It will be refined, adjusted and recalibrated as the
> project grows — including cutting whatever turns out to be redundant here, or beside the point
> of what an open source repository is for. We will keep working to make it better.

---

## Why

Running a model locally should depend on the computer someone already owns, not on the computer
they would have to buy. That is the whole point of this project, and every decision in it answers
to that:

- **No minimum requirements page.** The scalar path is the definition of the engine, and AVX2,
  AVX-512 or VNNI are accelerations chosen at runtime. A machine without any of them is slower;
  it is never excluded.
- **Nothing to install.** No CUDA toolkit, no BLAS, no Python, no OpenMP runtime. A compiler and
  `make` produce a single binary that runs.
- **The machine decides, not the user.** At startup the engine measures physical cores, CPU
  instructions, free RAM and disk speed, and places the weights accordingly. Options exist to
  force a choice for tests and measurements, never because someone has to set them.
- **RAM should not decide which models you may run.** The dense part of the model stays resident
  and the experts are read from disk under a budget, so model size stops being a wall.
- **A smaller machine must not mean a worse answer.** The number of threads, the batch size, the
  SIMD level and the expert budget change the speed and nothing else: the tokens are the same.
  This is enforced by tests, not hoped for.
- **It refuses to hurt the machine it runs on.** Before loading, the engine estimates memory and
  declines if it would leave less than 2 GB or 10% of RAM free. It never swaps.

## What we have done

Working today, end to end:

- **OLMoE-1B-7B** and any GGUF v3 file of the same family, loaded straight from Hugging Face with
  no conversion step. F32, F16, Q8_0, **Q4_K and Q6_K** weights (so the Q4_K_M files people
  download), each exact against its own dequantized weights.
- CPU kernels dispatched at runtime — scalar, AVX2, AVX-512, AVX-512+VNNI — each variant
  **bit-identical** to the scalar one, with a test that enforces it and another that proves the
  tier you think is running is the one that ran. On AVX-512 the prompt's matrix products take two
  weight rows against eight tokens at a time, and the decode's Q4_K products two weight rows against
  the token, their scales decoded in vectors.
- **The decode's attention on an NVIDIA GPU**, loaded at runtime (no CUDA toolkit needed to build
  or run): the same bytes as the CPU, checked through the engine; without a GPU nothing changes.
- `trochilus serve`, which keeps the model loaded between commands; `run`, `chat` and `generate`
  go through it by themselves when it is there, with the same output byte for byte.
- A byte-level BPE tokenizer built from the GGUF metadata, with NFC and Unicode classes: the same
  tokens as Hugging Face `tokenizers`, at 22 MB/s.
- `run` (text in, greedy generation out) and `chat` (the model's own chat template, written in C
  and compared byte for byte against `apply_chat_template`).
- Block prefill, prompt-driven speculative decoding, thread widths measured per phase, a KV cache
  laid out for the memory bus, and experts read from disk under a RAM budget.
- A built-in profiler, microbenchmarks for memory, disk, attention and `expf`, and a measurement
  harness that refuses to run on a busy machine.

### Measured

Ryzen 9 7940HX (16 physical cores), 31 GB RAM, NVMe ~1.5 GB/s, Windows 11, native build, quiet
machine, OLMoE-1B-7B Q8_0 resident in RAM. Every number is the median of N runs with its spread
recorded; the full tables and the method are in `docs/MEASUREMENTS.md`.

| Context | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|
| 32 | — | 38.9 |
| 512 | 306-315 | 35.5 |
| 2048 | 303-308 | 27.5 |
| 4000 | 276 | 20.9 |

With the experts streamed from disk instead of resident (M1, 512-token prompt): decode 35.5 /
29.6 / 24.3 / 20.6 tok/s at 100 / 75 / 50 / 25% of the expert budget over 64 generated tokens,
and 0.90x / 0.87x of the resident engine over 1000 tokens — **once generation is long the budget
barely matters; what costs is the first prompt**, which reads the whole model once.

Where the speed came from. Each step was measured with alternating runs and an A/A control, and
the ones that did not pay are written down too, in `docs/MEASUREMENTS.md`, with their numbers:

| Change | Effect |
|---|---|
| Thread pool with active waiting, one slot per thread | dispatch 53 µs → 1.5 µs at 16 threads |
| One thread per **physical** core, spread across chiplets | prefill +30%, 8→16 cores 2.02x |
| One weight row against 4 tokens in registers | prefill +45% |
| KV laid out `[layer][head][position]` | decode 1.15-1.18x at 4000 tokens |
| Attention in groups of 16 tokens, single-token work spread on the pool | prefill 1.11-1.14x at 4000 |
| Our own `expf` in place of the C library's | prefill 1.29-1.31x at 4000, attention 2.3-2.6x shorter |
| Layer-major prefill under a partial expert budget | 1.89x at 2048 tokens, 6 273 MiB read instead of 22 880 |
| Disk reads of the next layer's experts overlapped with compute | prompt 1.22x at 2048 tokens, half budget |
| The decode's attention on the GPU, same bytes | decode 1.31x at 2048 tokens, 1.54-1.58x at 4000 |
| Two weight rows per load of the activations | prefill 1.20-1.26x (Q8_0) |
| Two weight rows against eight tokens, the lane sums in SIMD | matmul 87 → 102 GFLOP/s on a core; prefill 1.13-1.19x (Q8_0, native, 512 to 4000 tokens) |
| Q4_K weights instead of Q8_0 | decode 1.5x |
| Q4_K decode: the block scales in vectors one block ahead, two weight rows per call, a prefetch | decode 1.25x at 4 threads (40.3 → 50.4 tok/s), the same bits |

The context table above predates the GPU attention and the two-row kernels; it will be measured
again as a whole.

Against the references, piece by piece (`docs/ORIGINS.md` §Every piece). llama.cpp's own matrix
kernels called alone through ggml, on our shapes, one core (indicative: a loaded machine):

| Kernel | llama.cpp | Trochilus |
|---|---|---|
| MoE experts, Q8_0 (two thirds of OLMoE's prompt) | 72 GFLOP/s | **102 GFLOP/s** |
| Dense projections, Q8_0 | **163 GFLOP/s** (8-bit activations) | 102 GFLOP/s, exact |
| `exp` on all 2^32 floats | rounds 3.4% of them otherwise | correctly rounded, every one |

The whole engine against llama.cpp on the same OLMoE-1B-7B Q8_0 (2026-09-25, both in the same
container, a still machine): the prompt **level at 8 threads**, theirs 1.30-1.35x at 16 (their
8-bit activations); the decode theirs 1.04x at 512 tokens of context, 1.13x at 2048.

On the same OLMoE-1B-7B in **Q4_K**, 4 threads (2026-09-25, the same container, 512-token prompt,
128 generated): the decode **level, 50.4 tok/s against their 50.2**, with every weight still the
exact dequantized float (they round the activations to 8 bits); the prompt theirs 1.6x (222 against
139 tok/s: 8-bit activations again). Their decode kernels alone, one core: 2x ours; at 4 threads
the memory is the limit for both, 1.04x.

### What we are proud of

**Exactness is an invariant, not a hope.** The logits do not move when you change the number of
threads, the batch size, the SIMD tier, the expert budget, whether speculative decoding is on, or
the operating system. Each of those is a test in the gate. Windows and Linux produce the same
logits **byte for byte** on the real model.

**Our `expf` is correctly rounded, and the proof is exhaustive.** `src/kernels/expf.c` is a
64-entry table and a polynomial, with the eight hard cases computed at 200 bits. `make bench-expf`
checks **all 4 278 190 082 float values** against the reference, under gcc and under clang, and it
runs in the gate. That is what made the two operating systems agree to the byte.

**Speculative decoding that cannot change the answer.** The draft comes from the text already in
the context, verification is a single pass over 1 + k positions, and every row of a pass is bit
for bit the same computation as a single-token pass — so the generated tokens are identical with
speculation and without it. It buys speed and can never buy a different result.

**No number comes from a single run.** Comparisons alternate A and B run by run, because measuring
all of A and then all of B moves the result by 10-25% as the machine warms up — that mistake once
cost us a good optimization, which is how we know. Every session carries an A/A control and a
declared background load, and a difference smaller than the worst A/A of that session is written
down as "not distinguishable".

**Every mistake becomes an automatic check.** `docs/LESSONS.md` is the log: each entry names how
it was found and the test, lint rule or gate check that now prevents it from coming back. The hot
path — the code that runs on every token — is enforced by `tools/lint.py` and `tests/test_hot.c`:
no allocation, no strings, no I/O. The engine's files go through generated mutations, fourteen of
them so far (`tools/mutate_files.sh`): a changed operator that no test notices is read one by one,
and either a test is written that kills it or the reason it cannot matter is written down.

## What we will do

| Milestone | Content | State |
|---|---|---|
| **M0** | base, GGUF v3, converter (F32/F16/Q8_0), CPU backend with runtime dispatch, OLMoE graph, greedy, CLI, tokenizer, chat template | **done** — exact against `transformers` on Windows and Linux; the real OLMoE-1B-7B answers |
| **M1** | experts from disk under a RAM budget: slot store, O(1) LRU, on-demand unbuffered reads, automatic plan | **in progress** — the store, the budget, layer-major prefill and reads overlapped with compute are done and measured; what remains is the cost of the first prompt |
| **M2** | K-quants (Q4_K, Q6_K, Q2_K, IQ2_XXS) on CPU, kernels to the hardware's limit | **in progress** — Q4_K and Q6_K done (Q4_K_M runs); the prompt's kernel at 61% of the CPU's peak; the Q4_K decode level with llama.cpp at 4 threads, exact |
| **M3** | CUDA module, hot experts in VRAM, a plan over VRAM + RAM + disk | **in progress** — the decode's attention done, exact; the dense weights next |
| M4 | a model of hundreds of gigabytes on the reference laptop | — |
| M5 | KV checkpoints on disk, a server (OpenAI and Anthropic APIs), speculative decoding with a draft model | speculation from the prompt done; `serve` keeps the model loaded, no API yet |
| M6 | Vulkan and Metal modules, so the GPU path is not only NVIDIA | — |

How we work, one piece of the engine at a time and to the end: first read how
[colibri](https://github.com/JustVugg/colibri), [ds4](https://github.com/antirez/ds4) and
[llama.cpp](https://github.com/ggml-org/llama.cpp) (with ik_llama.cpp) solve that piece, measure
theirs against ours, then build something better and measure again. The table of every piece
against the three is in `docs/ORIGINS.md`; the next piece is the first one still unread.

We only climb a step when the one below is exact and measured: tiny OLMoE (done) →
**OLMoE-1B-7B (here)** → Qwen3-Coder-30B-A3B → a MoE larger than RAM → a frontier-size MoE.

## Where we want to arrive

A model of hundreds of gigabytes, answering on a laptop that cost what a laptop costs, with
nothing to configure and nothing installed alongside it. Concretely, that means four things we do
not have yet:

1. **Size decoupled from RAM.** VRAM, RAM and disk holding one model together, with the engine
   deciding the split from what it measured on that machine.
2. **Every GPU, or none.** CUDA first because we have one to test on, then Vulkan and Metal, so
   an AMD card, an Intel integrated chip or an Apple laptop are first-class — and so is a machine
   with no GPU at all.
3. **Small weights without silent losses.** K-quants and lower, where each format is validated
   against the exact path before it is offered, and anything that is not bit-exact — int8
   activations, a 16-bit KV cache — exists only as a mode you turn on knowingly, never as a
   default that quietly changes your answers.
4. **Still no options.** Everything above has to happen by itself on a machine the author of the
   code has never seen.

## What is missing

The honest list, so nobody has to find out by running it:

- **One model family.** The OLMoE graph only. The tokenizer accepts the `olmo` pretokenizer
  family and refuses anything else — a family is admitted only after an oracle on its
  `tokenizer.json` — and there is one chat template.
- **The GPU does one thing, on NVIDIA only.** The decode's attention runs there; the weights, the
  prompt and the experts are still on the CPU, and there is no Vulkan or Metal module.
- **Q4_K and Q6_K are the lowest formats.** No Q2_K or IQ2 yet, so no 2-bit models; AVX2 lacks the
  two-row kernels and the vector Q4_K scales AVX-512 has.
- **arm64 detection exists, NEON kernels do not.** On ARM the engine would take the scalar path.
- **Greedy only** — no sampling, no server, no API, nothing beyond the CLI.
- **A model larger than RAM has not been run yet.** The experts stream from disk under a budget,
  measured on a model that fits; the first one that does not fit is still ahead.
- **Measured on one machine.** All the numbers above come from a single laptop. Hybrid P/E cores
  and multi-group Windows machines are handled in code and have never been tested on real
  hardware.

## Build and run

```sh
make                       # build/trochilus, no dependencies
make check                 # the gate: lint, 0-warning build, tests, ASan, TSan, oracles
build/trochilus cpu        # what the engine sees of this machine
build/trochilus run  -m model.gguf -f prompt.txt -n 200
build/trochilus chat -m model.gguf
```

C11 with intrinsics, a Makefile, gcc / clang / MinGW-w64. The Python tools under `tools/` are used
only for conversion and for the oracles; the engine itself never needs them.

## Documents

The engineering log lives in `docs/`:

| Document | Content |
|---|---|
| `docs/ARCHITECTURE.md` | principles, layers, execution, the correctness ladder, milestones |
| `docs/STATUS.md` | where the project stands, the decisions taken, the next step |
| `docs/MEASUREMENTS.md` | every measurement, including the optimisations that were rejected |
| `docs/LESSONS.md` | every mistake and discovery, with the check that now prevents it |
| `docs/ORIGINS.md` | where each borrowed idea or file comes from, commit by commit; every piece of the engine against colibri, ds4 and llama.cpp |
| `docs/COMMANDS.md` | the commands: benchmarks, measurements, mutations, reports |

## What we read

Trochilus is written from scratch, but almost nothing in it was invented here. Four engines are
pinned at a commit in `ref/` and read as primary sources; every idea taken from one of them is
recorded in `docs/ORIGINS.md` with the file and function it came from, so that whoever improves
that piece next knows where to look first, and a table there says, piece by piece, which of them
has been read and measured against ours and which is still owed.

| Project | Some of what we learned from it |
|---|---|
| [colibri](https://github.com/JustVugg/colibri) — Apache-2.0 | threads on **physical** cores rather than logical ones; weights read with `pread` instead of `mmap`; the pretokenizer regex replayed in C over codepoints; oracles built on tiny generated models; a draft taken from the text already in the prompt; expert index → slot, one expert in one slot |
| [ds4](https://github.com/antirez/ds4) — MIT | a persistent thread pool instead of OpenMP; the vocabulary read from GGUF metadata; (token, expert) pairs sorted by expert with a counting sort; one weight row against several tokens held in registers; the GGUF type table; the GPU execution model for later |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — MIT | the prompt processed in passes of at most 512 tokens; a weight row against a block of tokens; the two system calls that pin a thread to a processor; truncating the cache index to undo a rejected draft; the K-quant block layouts; the engine we race |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — MIT | its K-quant CPU kernels (`iqk_gemm_kquants.cpp`), read before we wrote ours; its repacked row layouts, weighed and not taken yet |

And the ones that are not engines: **`transformers`** and Hugging Face **`tokenizers`** are the
definition of a correct result here — every oracle in the gate compares against them — and the
first model we run is **OLMoE-1B-7B**, from AI2.

Only two files carry code that came from elsewhere: the GGUF type table in `src/format/gguf.c` and
`tools/make_tiny_olmoe.py`. Each names its origin, commit and path in its header, as `NOTICE`
requires. Everything else is ours — which is exactly why the list above matters, because the ideas
were not.

## License

Apache-2.0 — see `LICENSE`, and `NOTICE` for third-party material.
