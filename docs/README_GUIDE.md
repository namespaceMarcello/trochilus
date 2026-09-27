# How the README is written and kept

The README is the product's front page: what Trochilus is, what it gives, what it has that nobody
else has, and what it has conquered. It is not a log (that is `docs/`) and not a diary of attempts.
This guide comes from reading the READMEs of ds4 (whole), colibri and llama.cpp on 2026-09-26, and
from Marcello's rules of the same evening.

## The shape

What the three references share, in the order that serves a reader who has never heard of us:

1. **Logo, name, one line** that says what it is (ds4: "the best way to run a few excellent large
   language models on consumer hardware"; llama.cpp: "LLM inference in C/C++").
2. **One paragraph**: what it runs, on what, and why that matters.
3. **The guarantee** in a quote box (colibri: "no SLA on speed, and a hard guarantee on semantics";
   ours: "speed may change; the result may not").
4. **What it does today**: the capabilities a user can use now, as bullets (ds4: "So, what can I do
   with this software?").
5. **Results**: one table against the rival, both engines in the same conditions, the conditions and
   the date in one line above it; one paragraph on what differs in what is computed; a link to
   `docs/MEASUREMENTS.md` for everything else. Then a few milestones, each with its number (colibri's
   "What it achieves": a line per result, the full tables elsewhere; ds4's "Speed": one recorded
   baseline with its conditions, the rest in a performance document).
6. **What makes it different**: short paragraphs led by a bold sentence, each a property we can prove,
   with the proof (the test, the count, the check in the gate).
7. **Get started**: clone, build, run, in one code block (llama.cpp's "Quick start", ds4's "Start
   Here").
8. **Status**: one line, the milestones table, the honest list of what is missing.
9. **How it is built**: the AI disclosure (ds4 has one: say it plainly), the method in two lines, the
   documents table.
10. **Acknowledgements**: what we read and what we took, with the licenses.
11. **License**.

## The rules

- **Every number is today's.** The results table comes from the race scripts, the same GGUF for both
  engines, a free machine, the median of the runs, with the date and the conditions written once above
  it. A number that is no longer current is measured again or removed: never left under a note such as
  "this measurement is old" or "to be run again".
- **Show three things**: the results, what is unique, what has been conquered.
- **No history of gains.** A table of every optimization belongs in `docs/MEASUREMENTS.md` and
  `docs/archive/DONE.md`.
- **Only what is offered.** A prototype that is not in the engine stays out until it is in; roads
  left are never mentioned.
- **No apologies** ("first iteration", "will be refined").
- **Every claim true word for word**, checked against the documents before the commit (the oracles
  check tokens and a tolerance on the logits: "the same tokens as `transformers`", not "the same
  logits").
- **Marcello sees it before it is committed**: the README is visual work.

## When it changes

- With every significant implementation (a capability a user sees, a number against llama.cpp, a
  milestone step), in the same commit; the commit hook asks for it on every commit that changes
  `src/`, or for `no-readme: <why>` in the message.
- The results table is raced again when a change moves one of its numbers: `RACE_MODEL=models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf
  RACE_PROMPTS=512 RACE_THREADS=4 RACE_OUT=build/race_q4k_llama sh tools/race_llama.sh 5` (Q4_K, 4 threads,
  one model in memory: `tools/race_q4k.sh` loads two and presses Windows' memory, LESSONS #217) and
  `sh tools/race_llama.sh 5` (Q8_0, 8 and 16 threads, prompts of 512 and 2048), on a free machine and with
  no file written in the repo while they run (LESSONS #222); a run marked NOT FREE is not used.
- A section is rewritten to today's state, never appended to.
