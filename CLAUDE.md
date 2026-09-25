<!-- preferenze: 39f397b3 -->
# Trochilus

Inference engine for MoE models in C, with no dependencies: it runs on every kind of CPU (kernels
dispatched at runtime), GPUs as loadable modules, experts read from disk, results exact to the
token against transformers. Born by taking the best of colibri (Apache-2.0) and ds4 (MIT).

C11 + intrinsics (and assembly where measured), a Makefile, gcc/clang/MinGW-w64; the Python tools
(`tools/.venv`: torch CPU + transformers) serve conversion and oracles only.

**The spirit: read the project like a genome** (Marcello, 2026-09-24). Every byte of the weights,
the KV, the routing and the engine's time is data to sequence patiently, where others filtered it
away. Measure before believing; write the prediction before measuring; invent the next idea from
the numbers; push every lever to the limit the hardware allows (the peak FLOP/s, the bandwidth, the
exact bytes), not to "good enough". A premise that gives zero is a discovery too: written, closed,
next. Pioneer: what nobody has measured is where to look first.

**The method for any speed work, in this order** (Marcello, 2026-09-25, "decisive, we use it": it
took the Q4_K decode level with llama.cpp, MEASUREMENTS §The Q4_K decode dot sequenced):
1. **measure before changing, one piece removed at a time**: the deletion series and each
   instruction's cost on this CPU (`tests/bench_q4k_genome.c` is the pattern); a limit is claimed
   only from that series, never inferred from failed attempts (LESSONS #176, #178, #179);
2. **remove the largest piece found**: work done one at a time done in bulk, and ready one step
   ahead so nothing waits for it (a value read right after its store waits for the store, #180);
3. **give the core independent work to alternate**: two chains where one waits on itself;
4. **bring the data before it is needed** (prefetch, early reads), every idea timed alone in cache,
   then from RAM, then in the engine, then in the race (#181);
5. **prove the result did not move**: bit for bit against scalar, and mutations the tests must see.
The prediction is written before the run; the report to Marcello follows the same five steps.

---

## How to work

- At every session start, without being asked: read `build/prompt-next.md`
  (the task Marcello left for this session: do it), then `docs/STATUS.md`.
- When a piece of work ends, without being asked: write the next session's prompt in
  `build/prompt-next.md` (what to read, the state, the task, how to measure it, how to report),
  so Marcello can `/clear` and start again.
- Agents: **never Fable**. Sonnet on closed briefs (one piece with its interface and its test),
  Haiku for mechanical work; design, interfaces and invariants belong to the orchestrator. One
  agent at a time, or in parallel only on disjoint files. Few agents: usage is the limit.
- **Agents speak English to each other**, in briefs and in reports. With Marcello, Italian.
- A brief is **short and complete**: goal, constraints, files to touch, shape of the answer, when
  it is done. An agent's report: what it did, files touched, what is still open.
- Report to Marcello: **schematic, short, foolproof** (Marcello, 2026-09-24: "the way you explain
  things, I don't follow them"). First line: the "so what?" in plain words (did we gain? how much,
  where). Then one line per item, no jargon (or a gloss in two words), at most 2–3 numbers, ≤ ~10
  lines. Two points: what was done, how to try it. The why and the detail go in `docs/STATUS.md`.
- Logic and kernels: choose, build, deliver. Questions are asked up front, not halfway through.
- **One piece at a time, to the end, against the three references** (Marcello, 2026-09-24, asked
  more than once): before building on a piece of the engine, read how colibri, ds4 and llama.cpp
  (with ik_llama.cpp) do it in `ref/`, measure theirs against ours where it runs, write it in the
  piece's row of `docs/ORIGINS.md` §Every piece against the references; then build something
  better, measure, compare again. A piece closes only with its row read and measured; the next
  piece starts after, never beside it.
- **Marcello asks for the commit.** Prepare everything (tests green, documents written) and stop.
- **Public repository** `namespaceMarcello/trochilus` (since 2026-09-22): everything committed is
  visible to anyone. Pushes and visibility are Marcello's call.

### Keep it all under control: the cycle of every step

Goal: the success rate rises with every request. It is measured in `docs/LESSONS.md` (the "found
by" column): no mistake should reach Marcello if a test could have caught it first.

1. **Before**: `docs/STATUS.md`, then `grep` in `docs/LESSONS.md` for the area being touched, then
   the piece's row in `docs/ORIGINS.md` §Every piece: if it is not "read and measured", reading and
   racing the references (`ref/`, `tools/bench_ggml.sh` for ggml's kernels alone) is the first step.
2. **A mistake or a discovery**, as soon as it happens, however small: one line in
   `docs/LESSONS.md`.
3. **Every mistake becomes a check**: a test that reproduces it (red before the fix, green after),
   or a check in `make check`, a pin, a hook. A sentence in a document is weak prevention: write a
   *rule*, or the lesson stays open.
   **A test is seen red at least once, and it says which branch it exercises**: red before the fix,
   or under a mutation (`tools/mutate_*.sh`) when the code is born together with the test; the head
   of the file states which branch it covers, and a counter (`TR_CHECK(n > 0)`) fails it if that
   branch was never taken. Equal results do not tell you which code ran (LESSONS #43, #50, #54,
   #78).
4. **Every new scenario enters the tests**: a model, a family, a tensor shape, a platform, a use
   (long prompt, full context, malformed file) becomes a case in the right suite: C tests in
   `tests/`, an oracle in `make oracle`, a performance scenario in `bench/scenarios.json`.
5. **Before saying "done"**: `make check` green (0-warning build, tests, oracles, sanitizers where
   available). An agent does not close a step: the orchestrator's verification does.
6. **After an agent**: read the report against the code, `make check`, `docker ps` and background
   processes (LESSONS #5).
7. **Upstream**: when porting or reading code from colibri and ds4, every bug found goes in
   `docs/UPSTREAM.md` with its proof; a PR or an issue only after Marcello says yes.
8. **Documents**, per the table below.

### Before every commit: document
| If this changed… | Write in |
|---|---|
| code (every commit) | `docs/archive/DONE.md`: `### <date> — <title>`, what it is and how to try it |
| a mistake, a discovery, a debt | `docs/LESSONS.md`, with the prevention and who found it |
| code ported from colibri or ds4 | `docs/ORIGINS.md` + the file header |
| a decision, a debt, a measurement, the next step | `docs/STATUS.md` (replace, do not append; 40 KB cap) |
| a step closed, a decision, a new question (together with STATUS) | `docs/status.json`, then `tools/status_html.py`, and republish the artifact (same URL) |
| layers, principles, milestones | `docs/ARCHITECTURE.md` (rewrite the line, do not add one) |
| a measurement or an attempt at optimisation (even a rejected one) | `docs/MEASUREMENTS.md` |

---

## Read before answering

| Question about… | Open |
|---|---|
| principles, layers, execution, milestones | `docs/ARCHITECTURE.md` |
| where we stand, decisions, the next step | `docs/STATUS.md` |
| where a file comes from, the pinned commit | `docs/ORIGINS.md` |
| measured numbers, optimisations that worked and that were rejected | `docs/MEASUREMENTS.md` |
| open questions to measure, where to go into detail | `docs/MEASUREMENTS.md` §Da misurare |
| the rules for code that runs on every token | `docs/ARCHITECTURE.md` §Hot path |
| mistakes already made, discoveries, how they are prevented | `docs/LESSONS.md` |
| bugs found in colibri or ds4, and what was reported to their authors | `docs/UPSTREAM.md` |
| profiling: levels, tools, scenarios | `docs/ARCHITECTURE.md` §Profiling |
| what has already been done | `docs/archive/DONE.md` |
| the command to do something (benches, measurements, mutations, reports) | `docs/COMMANDS.md` |
| where we stand, as a picture (milestones, dependencies, state) | `docs/status.json` → `tools/status_html.py` → artifact `6mx3NS4KtQrBFPRLkAYupr`; updated together with `docs/STATUS.md` |
| how colibri, ds4, llama.cpp or ik_llama.cpp do something | `ref/colibri`, `ref/ds4`, `ref/llama.cpp`, `ref/ik_llama.cpp` (read-only, commits in ORIGINS) |
| which piece has been read and measured against the three references, which is owed | `docs/ORIGINS.md` §Every piece against the references |

---

## Commands

```bash
make check               # the gate: lint, 0-warning build, tests, ASan, TSan, oracles
make                     # build/trochilus (the core, no dependencies)
make test                # C tests: SIMD kernels = scalar, malformed GGUF, pool, profiler
make oracle              # tiny models: generate, convert, compare against transformers
build/trochilus run -m <f.gguf> -f prompt.txt -n 200    # text in, greedy generation out
build/trochilus chat -m <file.gguf>                     # a conversation with the model's template
make profile             # profiler scenarios, median of N, identical tokens
tools/.venv/Scripts/python.exe tools/<script>.py        # on Linux/macOS: tools/.venv/bin/python
```

**Every other command is in `docs/COMMANDS.md`**: oracles per model and per platform, benches
(`bench-disk`, `bench-mem`, `bench-attn`, `bench-expf`), the native measurements with their guards
(`experts_budget.sh`, `prefill_overlap.sh`, `decode_context.sh`, `threads_phase.sh`,
`ab_modes.sh`), the mutations, the Python reports, the map of the project. A change to the decode is
measured with `sh tools/decode_context.sh change-short <binary-before>` (~40 min; the full `change`,
~90 min, only for what every token pays whatever its context: docs/ARCHITECTURE.md §Profiling).

Before delivering: `make check` green. On Windows correctness runs in Docker
(`trochilus-dev:local`): Smart App Control blocks freshly compiled binaries (LESSONS #12).

---

## Invariants

- **Safety of the machine**: before loading, the engine estimates memory and refuses if it would
  leave less than 2 GB or 10% of RAM free. No model larger than RAM before streaming (M1). Benches
  of at most 60 s per run, threads <= physical cores. A large download is announced (size, free
  space verified) before it starts.
- **Languages**: C is the definition (portable, readable, the bit-for-bit reference). Hot kernels
  are rewritten in assembly one at a time; the assembly version stays only if the test finds it
  identical to the C one and the benchmark finds it faster, and the C stays as the fallback for
  every other CPU.
- The core depends on nothing beyond libc and the operating system's threads.
- Every kernel variant (SIMD, assembly) is bit-identical to the scalar one; a test verifies it.
- Hot path (code that runs on every token): no allocation, no strings, no I/O; `tools/lint.py` and
  `tests/test_hot.c` verify it. Every measurement is a median of N runs, never a single run.
- **Native measurements**: every script finishes what it started (`tools/cleanup.lib`), nothing
  starts with orphans still running (`tools/orphans.sh`), the machine is loaded only with
  `tools/busy_machine.sh`, every session declares its background load in the log, and with a load
  that is not low no conclusion is drawn (`docs/ARCHITECTURE.md` §Profiling, LESSONS #84-#88).
- A file derived from colibri or ds4 states project, commit, path and modification in its header.
- **Everything in the repository is written in English**: file names, code, comments, documents
  under `docs/`. With Marcello the conversation stays in Italian.
- Prefix `tr_` for public symbols; no global state per model (several models in one process).
- Models, fixtures and binaries stay out of git (`.gitignore`).

---

## Maintenance

When a document is born in `docs/`, add its row to the table. What happened goes in
`docs/STATUS.md`, not here.
