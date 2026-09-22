#!/usr/bin/env python3
"""check_measurements.py -- a number from a model is not a measurement (docs/LESSONS.md #98).

Question 14 predicted 22.4 misses and 143 MiB per generated token with half the model in RAM;
the measurement found 0.3 and 1.8 on the same budget, because the simulation ran over a long
generation while the engine's cache is warm from the prompt. The question had been struck
through -- closed -- on the strength of the simulation alone, and the engine's design rested
on it for a day.

Two rules over docs/MEASUREMENTS.md, so the same thing cannot close a question again:

  A. a line that says a number comes from a simulation or a time model carries one of the two
     tags, so a reader cannot mistake it for something measured:
       "model, not measured"               nobody has measured this yet
       "model superseded by measurement"   history: a measurement has taken its place
     (the Italian ones, "modello, non misura" and "modello superato dalla misura", count too
     while the document is half translated). A tag may wrap onto the next or previous line.
  B. a question row tagged "model, not measured" stays OPEN -- no ~~strikethrough~~ -- until a
     measured number takes its place, wherever in the row's wrapped lines the tag sits. The
     other tag is history, so it may sit in a closed row.

Both rules read logical blocks, not lines: a table row with its wrapped continuation lines, or a
paragraph. Read line by line, a tag wrapped in two was not found (the gate went red on the
English translation) and a tag on a row's second line escaped rule B (docs/LESSONS.md #102).
The cases in SELF_TEST run first, every time: each is the shape of one of those misses.

Run: tools/.venv/Scripts/python.exe tools/check_measurements.py   (in `make check`)
"""
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "docs" / "MEASUREMENTS.md"
TAGS = ("model, not measured", "modello, non misura")
OLD_TAGS = ("model superseded by measurement", "modello superato dalla misura")
# the words that announce a number nobody measured
MODEL_WORDS = re.compile(r"simulat|simulazione|time model|modello a tempo|time estimate|stima a tempo", re.I)
# a row of the questions table: "| 14 | ..."
QUESTION_ROW = re.compile(r"^\|\s*(\d+)\s*\|")
# a closed question: the question itself is struck through
CLOSED = re.compile(r"^\|\s*\d+\s*\|\s*~~")


def blocks(lines):
    """(first line number, lines) for each logical block: a line starting with '|' opens a table
    row and the lines after it that do not start with '|' are its wrapping; a blank line ends a
    paragraph."""
    cur, start = [], 0
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("|") or line.startswith("#"):
            if cur:
                yield start, cur
            cur, start = ([line], n) if line.strip() else ([], 0)
            continue
        if not cur:
            start = n
        cur.append(line)
    if cur:
        yield start, cur


def flat(parts):
    return " ".join(" ".join(parts).split())


def problems(text):
    bad = []
    for start, block in blocks(text.splitlines()):
        for i, line in enumerate(block):
            if not MODEL_WORDS.search(line):
                continue
            near = flat(block[max(0, i - 1):i + 2])
            if not any(t in near for t in TAGS + OLD_TAGS):
                bad.append(
                    f"{DOC.name}:{start + i}: a number from a simulation or a time model, without "
                    f"the tag '{TAGS[0]}' or '{OLD_TAGS[0]}': {line.strip()[:120]}"
                )
        if CLOSED.match(block[0]) and any(t in flat(block) for t in TAGS):
            q = QUESTION_ROW.match(block[0]).group(1)
            bad.append(
                f"{DOC.name}:{start}: question {q} is closed (~~) on a '{TAGS[0]}': a question "
                f"stays open until a measured number takes the model's place"
            )
    return bad


# (text, how many problems it must give): the shapes the line-by-line reading got wrong
SELF_TEST = [
    ("Trace simulation said 22.4 units (model superseded by\nmeasurement): the engine found 0.3.", 0),
    ("Trace simulation said 22.4 units.", 1),
    ("| 14 | ~~How many bytes per token?~~ the answer\nfrom a trace simulation: 22.4 (model, not measured) |", 1),
    ("| 14 | How many bytes per token? A trace simulation\nsaid 22.4 (model, not measured) |", 0),
    ("| 14 | ~~How many bytes?~~ **Measured**: 1.8 MiB. The simulation said 143\n"
     "(modello superato dalla misura) |", 0),
    ("A time model says 1.61x (modello, non misura).\n\nThe simulation said 3.", 1),
]


def main() -> int:
    for text, want in SELF_TEST:
        got = len(problems(text))
        if got != want:
            print(f"check_measurements: self-test FAIL, {got} problems instead of {want} on:\n{text}",
                  file=sys.stderr)
            return 1
    bad = problems(DOC.read_text(encoding="utf-8"))
    if bad:
        print("check_measurements: FAIL", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        print(
            "\n  A model's number is a hypothesis: it is tagged, and its question stays in\n"
            "  the open-questions section until it is measured (docs/LESSONS.md #98).",
            file=sys.stderr,
        )
        return 1
    print(f"check_measurements: ok ({len(SELF_TEST)} self-test cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
