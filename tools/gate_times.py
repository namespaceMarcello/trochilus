#!/usr/bin/env python3
"""gate_times.py -- seconds per step of a `make check` log whose lines start with HH:MM:SS.

Two steps hid 3 of the gate's 14 minutes (docs/LESSONS.md #107, #109) because nothing printed
times. Make the log with a time on each line, then read it here (steps under 3 s are left out):
  make check 2>&1 | awk '{ print strftime("%H:%M:%S"), $0; fflush() }' > build/check.log
  tools/.venv/Scripts/python.exe tools/gate_times.py build/check.log
With the parallel groups (-Orecurse) a group's output lands when the group ends, so inside a
group the table gives the order of completion, not each step's own time.
"""
import re
import sys
from datetime import datetime

STEP = re.compile(r"^(make |TR_EXPERT|TSAN_OPTIONS|build/linux-gcc/trochilus|! build|tools/\.venv|sh tools|MSYS_NO_PATHCONV|/usr/bin/make|\$\(MAKE\))")
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
steps = []
depth_top = True
for line in lines:
    m = re.match(r"^(\d\d:\d\d:\d\d) (.*)$", line)
    if not m:
        continue
    t, text = datetime.strptime(m.group(1), "%H:%M:%S"), m.group(2)
    if STEP.match(text) and "-c src/" not in text and " -o build" not in text:
        steps.append((t, text[:110]))
    if text.startswith("exit="):
        steps.append((t, text))
for (t0, s), (t1, _) in zip(steps, steps[1:]):
    d = (t1 - t0).total_seconds()
    if d < 0:
        d += 86400
    if d >= 3:
        print(f"{d:6.0f} s  {s}")
total = (steps[-1][0] - steps[0][0]).total_seconds()
print(f"{total if total >= 0 else total + 86400:6.0f} s  TOTAL")
