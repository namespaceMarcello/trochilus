#!/usr/bin/env python3
"""check_measurements.py -- a number from a model is not a measurement (docs/LESSONS.md #98).

Domanda 14 predicted 22.4 misses and 143 MiB per generated token with half the model in RAM;
the measurement found 0.3 and 1.8 on the same budget, because the simulation ran over a long
generation while the engine's cache is warm from the prompt. The question had been struck
through -- closed -- on the strength of the simulation alone, and the engine's design rested
on it for a day.

Two rules over docs/MEASUREMENTS.md, so the same thing cannot close a question again:

  A. a line that says a number comes from a simulation or a time model carries one of the two
     tags, so a reader cannot mistake it for something measured:
       «modello, non misura»            nobody has measured this yet
       «modello superato dalla misura»  history: a measurement has taken its place;
  B. a question row tagged «modello, non misura» stays OPEN -- no ~~strikethrough~~ -- until a
     measured number takes its place. The other tag is history, so it may sit in a closed row.

Run: tools/.venv/Scripts/python.exe tools/check_measurements.py   (in `make check`)
"""
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "docs" / "MEASUREMENTS.md"
TAG = "modello, non misura"
OLD_TAG = "modello superato dalla misura"
# the words that announce a number nobody measured
MODEL_WORDS = re.compile(r"simulat|simulazione|modello a tempo|stima a tempo", re.I)
# a row of the questions table: "| 14 | ..."
QUESTION_ROW = re.compile(r"^\|\s*(\d+)\s*\|")
# a closed question: the question itself is struck through
CLOSED = re.compile(r"^\|\s*\d+\s*\|\s*~~")


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    bad = []
    for n, line in enumerate(text.splitlines(), 1):
        tagged = TAG in line
        if MODEL_WORDS.search(line) and not tagged and OLD_TAG not in line:
            bad.append(
                f"{DOC.name}:{n}: a number from a simulation or a time model, without the tag "
                f"'{TAG}' or '{OLD_TAG}': {line.strip()[:120]}"
            )
        if tagged and CLOSED.match(line):
            q = QUESTION_ROW.match(line).group(1)
            bad.append(
                f"{DOC.name}:{n}: domanda {q} is closed (~~) on a '{TAG}': a question stays "
                f"open until a measured number takes the model's place"
            )
    if bad:
        print("check_misure: FAIL", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        print(
            "\n  A model's number is a hypothesis: it is tagged, and its question stays in\n"
            "  the open-questions section until it is measured (docs/LESSONS.md #98).",
            file=sys.stderr,
        )
        return 1
    print("check_misure: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
