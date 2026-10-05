<p align="center">
  <img src="assets/logo.png" alt="Trochilus" width="170">
</p>

<h1 align="center">Trochilus</h1>

<p align="center"><b>Exact Mixture-of-Experts inference in C, on the computer you already own.</b></p>

Trochilus runs Mixture-of-Experts language models with **no dependencies**: libc and the
operating system's threads, nothing else. One binary picks its kernels at runtime for whatever CPU
it lands on, reads the experts from disk when they do not fit in RAM, and puts part of the work on
an NVIDIA GPU when one is there — without a CUDA toolkit. And it gives the **same tokens as
`transformers`**, checked in every build.

> **Speed may change; the result may not.** Threads, SIMD tier, batch size, expert budget,
> speculative decoding, Windows or Linux, CPU or GPU: each one changes how fast the answer comes,
> never a bit of it. Every one of them is a test in the gate.

## What it does today

- Runs **OLMoE-1B-7B** and any GGUF v3 file of its family, straight from Hugging Face: F32, F16,
  Q8_0, Q4_K and Q6_K weights (so the Q4_K_M files people download).
- CPU kernels for scalar, AVX2, AVX-512 and AVX-512 with VNNI and VBMI, chosen at runtime, each
  **bit-identical** to the scalar definition, with a test that the tier you think ran is the one that ran.
- **The decode's attention on an NVIDIA GPU**, loaded at runtime (hand-written PTX through the
  driver) with the CPU's bits: the logits of 600 positions and the tokens after a 4000-token prompt
  are the CPU's byte for byte. Without a GPU nothing changes.
- **Experts from disk** under a RAM budget planned on the run's own tokens: a slot store that gives
  up the coolest expert (ds4's rule), unbuffered reads, a layer's consecutive experts in one request a
  part, the prompt read layer by layer, the next layer's experts read while this one computes.
- **Speculative decoding** (`--spec`): drafts taken from the text already in the context, several
  tokens verified in one pass that loads each weight row from memory once for all of them (a pass
  of three tokens costs 1.7 times a pass of one).
- `run`, `chat` (the model's own chat template, byte for byte `apply_chat_template`), `generate`,
  and `serve`, which keeps the model loaded between commands.
- A byte-level BPE tokenizer built from the GGUF metadata, token for token Hugging Face
  `tokenizers`, at 22.0 MB/s.

## Results

OLMoE-1B-7B, the same GGUF for both engines, both in the same Linux container on one laptop (Ryzen 9
7940HX, 16 cores, 31 GB), a free machine, the night of 2026-09-26 to 27. Tokens per second, the median
of 10 runs: two series of 5 for each engine, alternated with the other's.

| | llama.cpp | Trochilus | Trochilus / llama.cpp |
|---|---|---|---|
| Q4_K, prompt of 512 tokens, 4 threads | 220.6 | 220.0 | 1.00× |
| Q4_K, generation after that prompt, 4 threads | 50.9 | 52.4 | 1.03× |
| Q8_0, prompt of 512 tokens, 16 threads | 370.7 | 492.9 | **1.33×** |
| Q8_0, prompt of 2048 tokens, 16 threads | 341.3 | 485.2 | **1.42×** |
| Q8_0, generation at 512 tokens of context, 8 threads | 33.1 | 33.1 | 1.00× |
| Q8_0, generation at 2048 tokens of context, 8 threads | 27.2 | 26.3 | 0.96× |

The difference is in what is computed: llama.cpp rounds the activations to 8 bits before each matrix
product and keeps its attention cache in 16 bits; Trochilus computes what `transformers` computes, the
activations in float (for Q4_K, exact integers) and the cache in 32 bits. Every number, the conditions,
and the optimizations that did not pay are in [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md).

Under a RAM budget the model is not read twice: with a partial expert budget a 2048-token prompt runs
1.89x faster reading 6 273 MiB instead of 22 880, and 1.22x more with the disk reading the next layer
while the cores compute this one. And the disk runs at its limit, 3.4 GB/s from this laptop's encrypted
drive: a layer's consecutive experts go in one request a part (96 requests for the whole model instead of
3072) into pages faulted beforehand by the engine's threads. With Q8_0 whole in RAM, the run from start to end (the
load, a 321-token prompt, 8 tokens) takes 3.48 s instead of 5.49; with half the experts in RAM, that first
prompt takes 2.81 s instead of 4.39 (2026-10-02, median of 8, the same tokens). A run's three parts then
went in flight together: the resident read 2.03 -> 1.95 s (3.5 GB/s), that first prompt 2.82 -> 2.76 s
(2026-10-03, median of 8, the same tokens).

On an 8 GB laptop the Q8_0's experts do not fit, and its disk is slower. Emulated here (4 cores, AVX2, the
memory Windows leaves free on 8 GB, the experts read at a SATA-class 0.5 GB/s), the memory planned on the
run's own tokens (290 experts in RAM instead of 83) and ds4's rule for which expert leaves take its generation
from 0.57 to 1.70 tok/s, and the Q4_K's from 2.52 to 8.02, the same tokens (2026-10-03, median of 5): the rule
was replayed on the model's real routes before it was written, and picked over ours, colibri's and the page
cache's. Its prompt was half computation: AVX2 now has a Q4_K tile of its own (each weight decoded once for a group
of tokens, in the same exact integers), and the Q4_K prompt goes from 20.6 to 35.1 tok/s there, from 36.4 to 137.9
on its four cores with every expert in RAM, the same tokens (2026-10-03, median of 5). Then it waited for the disk:
a prompt that fits one pass now reads each next layer while this one computes, in short requests, and drops what
that layer does not ask before it is read: the Q4_K prompt goes from 35.3 to 45.5 tok/s, the Q8_0's from 20.3 to
24.6, at 97-98% of the time its disk needs for those bytes, the same tokens (2026-10-03, median of 5 and of 3).
After a prompt the experts it chose most now stay in RAM (the prompt's routings tell the rule, as ds4's does, scaled
to 64 tokens): its generation goes from 8.00 to 15.50 tok/s (Q4_K) and from 1.91 to 2.38 (Q8_0), the same
tokens (2026-10-03, median of 3, stopped steady). A chat does not know how long it will be, so it used to set
aside the memory of its whole context (4096 tokens, 1 GiB) at load; now the experts hold that memory and hand it
to the conversation a page at a time as it is written, which none of colibri, ds4 or llama.cpp does: on the 8 GB
machine a chat keeps 278 Q8_0 experts in RAM instead of 131 and reads 193.7 MiB a token from disk instead of
375.9 (Q4_K: 591 instead of 314, 16.3 MiB instead of 87.4): its generation goes from 1.22 to 2.29 tok/s
(Q8_0) and from 4.55 to 14.77 (Q4_K), its Q8_0 prompt from 20.55 to 24.59, the same tokens (2026-10-04, median of
3, stopped steady). A layer whose experts are not all in RAM now computes the ones that are while its disk reads
the others, each late one as its bytes land, the same bytes and the same bits: on the 8 GB machine generation goes
from 14.98 to 16.08 tok/s (Q4_K) and from 2.33 to 2.40 (Q8_0), the same tokens (2026-10-05, median of 3, stopped
steady). ds4 splits a layer this way on its GPU from three misses; colibri and llama.cpp wait for each. Meanwhile
the next layer's router guesses which experts that layer will ask, and the disk reads them in short pieces, a
guess dropped mid-read as soon as the layer asks for others and read whole once it is asked: 15.92 -> 16.26 tok/s
(Q4_K) and 2.40 -> 2.41 (Q8_0), the same tokens (2026-10-05, median of 3, stopped steady). colibri's own guess,
off by default, reads whole experts, which on such a disk loses more than it gains; ds4 and llama.cpp guess nothing.

## How it keeps up, bit for bit

Every tier must give the scalar definition's bits, so the speed has to come from how the work is
arranged, not from rounding. Four ideas do it, each checked in the gate:

**Sixteen rows in sixteen lanes.** The scalar definition adds a row's products in sixteen interleaved
chains and joins them with a fixed tree. Rather than spread one row across a register's sixteen lanes,
which must then be joined at the end of every row, Trochilus gives each lane its own weight row: every
chain sums in the scalar order, the input is broadcast to sixteen rows at once, and nothing is decoded,
scaled or joined inside the loop. The prompt's matrix-product tile runs at 164.7 GFLOP/s on one core,
98.6% of what the CPU can do in float without fused multiply-adds (which would round once where the
scalar definition rounds twice), and every output is the scalar definition's, bit for bit.

**Q4_K in exact integers, each input converted once.** A Q4_K matrix product takes each block of 256
inputs as 32-bit fixed point and sums exact integers, then rounds once: the input's 32 bits are the only
approximation, where a float dot product rounds at every step. The order of the sums no longer matters,
so any thread count or SIMD width gives the same bits, as fast as the float kernels this replaced. Each
input row is converted once and shared: q, k and v read the same converted rows, gate and up read a
token's row through a map instead of eight gathered copies, and the SwiGLU converts the down
projection's input while it is still in the core's cache.

**A byte that can be deduced is not read.** In OLMoE's GGUF files the router matrices are 32-bit floats
whose low 16 bits are all zero: the model was converted from bf16. The load checks every value, keeps
only the top halves, and each row widens them back with a shift: the 32-bit row's bits, from half the
bytes, every token. A matrix with one value that does not fit stays in 32 bits.

**A correctly rounded `exp`, proved on every float.** Our `expf` is a 64-entry table and a polynomial,
with the eight hard cases computed at 200 bits; the gate checks it on **all 4 278 190 082** float
arguments that are not NaN, against the exact value, in every SIMD tier. (glibc's rounds 0.004% of
them otherwise; ggml's 3.36%, by up to 2 units in the last place.)

## What makes it different

**Exactness is an invariant, not a hope.** Windows and Linux give the same logits **byte for byte**
on the real model; tiny models built for the purpose are compared with `transformers` in every gate,
and so is the real model cut to two layers. Speculative decoding cannot change the answer: every
verified row is bit for bit the computation of a single-token pass.

**Nothing to configure.** At startup the engine measures the cores, the instructions, the free RAM
and the disk, and places the work itself; while it runs it times how many threads each kind of pass
wants (the decode, and each size of verify pass on its own). It refuses to load if it would leave the
machine with less than 2 GB or 10% of its RAM.

## Get started

```sh
git clone https://github.com/namespaceMarcello/trochilus.git
cd trochilus
make                                   # build/trochilus: no dependencies
build/trochilus cpu                    # what the engine sees of this machine
build/trochilus run  -m OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf -f prompt.txt -n 200
build/trochilus chat -m OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf
```

C11 with intrinsics and a Makefile: gcc, clang or MinGW-w64, on Linux and Windows. `make check` is
the gate (lint, a 0-warning build, the tests, ASan, TSan, the oracles). The Python tools under
`tools/` are for conversion and for the oracles only; the engine never needs them.

## Status

Pre-alpha, under active development, measured on one machine.

The aim is AI on the machines most people own. The roadmap is a ladder of models: each one is brought
to the end — exact, studied piece by piece, at the theoretical limit of three machines (below average:
4 cores, 8 GB, no GPU; average: 8 cores, 16 GB; this laptop), usable, raced — before the next one starts.

| Rung | Model | State |
|---|---|---|
| R0 | the exact engine: GGUF, CPU kernels for every tier, tokenizer, chat | **done** |
| R1 | OLMoE-1B-7B, to the end | exact; Q4_K, Q6_K, Q4_K_M; experts from disk under a RAM budget; the GPU's decode attention; speculation from the context, `serve`; the three machines measured (2026-10-03): an average one (8 cores, 16 GB) generates Q4_K at 69.9 tok/s, at its RAM's limit, this PC at 97-99% of what its RAM gives in practice, an 8 GB one with a SATA-class disk 15.50 tok/s from disk (2.52 before its memory was planned on the run, 8.00 before the prompt's routings told its eviction), its prompt 45.5 tok/s (20.6 before AVX2's own Q4_K tile, 35.3 before it read the next layer ahead) |
| R2 | Qwen3-Coder-30B-A3B, a coding model: on 16 GB its experts come from disk; 2-bit formats | — |
| R3 | a MoE larger than this laptop's RAM | — |
| R4 | DeepSeek V4 Flash, hundreds of gigabytes, on the same laptop | — |

Off the ladder, when a rung needs them: Vulkan and Metal (a GPU other than NVIDIA) and hand-written
assembly.

Not there yet: one model family and one chat template; the GPU does only the decode's attention, on
NVIDIA only; no 2-bit formats; no NEON kernels (ARM takes the scalar path); greedy decoding only, and
no HTTP API; a model larger than RAM has not been run yet.

## How it is built

Trochilus is written by **Claude Code (Anthropic's Opus) as the coding agent, with Marcello Costagliola
leading** — the direction, the questions, the reviews and every decision; every commit says so. One
piece of the engine at a time: first read how [colibri](https://github.com/JustVugg/colibri),
[ds4](https://github.com/antirez/ds4) and [llama.cpp](https://github.com/ggml-org/llama.cpp) solve it,
measure theirs against ours, then build and measure again. A comparison alternates its two sides run by
run and carries an A/A control, the prediction is written before the run, and every mistake becomes an
automatic check (a test, a lint rule, a mutation that must turn red). The engineering log is in `docs/`:

| Document | Content |
|---|---|
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | principles, layers, the roadmap: the rungs, the three machines, the five phases |
| [`STATUS.md`](docs/STATUS.md) | where the project stands, the decisions, the next step |
| [`MEASUREMENTS.md`](docs/MEASUREMENTS.md) | every measurement, including the optimizations that were rejected |
| [`LESSONS.md`](docs/LESSONS.md) | every mistake and discovery, with the check that now prevents it |
| [`ORIGINS.md`](docs/ORIGINS.md) | where each idea and file comes from; every piece against the three references |
| [`COMMANDS.md`](docs/COMMANDS.md) | benchmarks, measurements, mutations, reports |

## Acknowledgements

Trochilus is written from scratch, on what four other engines taught. They are pinned at a commit in
`ref/` and read as primary sources, and every idea taken from one of them is recorded in
[`docs/ORIGINS.md`](docs/ORIGINS.md) with the file and function it came from.

| Project | Some of what we learned from it |
|---|---|
| [colibri](https://github.com/JustVugg/colibri) — Apache-2.0 | threads on physical cores; weights read with `pread`; the pretokenizer regex replayed over codepoints; oracles on tiny generated models; drafts from the prompt's own text; one expert in one slot |
| [ds4](https://github.com/antirez/ds4) — MIT | a persistent thread pool instead of OpenMP; the vocabulary from GGUF metadata; (token, expert) pairs sorted by a counting sort; one weight row against several tokens in registers; the GGUF type table |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — MIT | prompts in passes of 512 tokens; the K-quant block layouts; pinning a thread to a processor; undoing a rejected draft; the engine we race |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — MIT | its K-quant CPU kernels, read before we wrote ours |

**`transformers`** and Hugging Face **`tokenizers`** define what a correct result is here, and the
first model is **OLMoE-1B-7B**, from AI2. Only two files carry code from elsewhere — the GGUF type
table in `src/format/gguf.c` and `tools/make_tiny_olmoe.py` — and each names its origin in its header.

## License

Apache-2.0 — see `LICENSE`, and `NOTICE` for third-party material.
