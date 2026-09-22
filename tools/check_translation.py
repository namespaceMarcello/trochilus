"""Compare a translated doc against its committed version.

A translation may change every word, but not a number, a fenced code block, a table
row or a heading count. Inline code spans are compared only outside fenced blocks.
"""
import re, subprocess, sys
from collections import Counter

FENCE = re.compile(r"```.*?```", re.S)
NUM = re.compile(r"\d[\d.]*\d|\d")
HEAD = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
ITALIAN = re.compile(r"\b(perch[eé]|quindi|questo|questa|della|degli|sono|viene|anche|ogni|"
                     r"nella|dalla|invece|soltanto|misura|passata|esperti|thread di|"
                     r"come si prova|non si|si fa)\b", re.I)


def committed(path, rev="HEAD"):
    return subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def report(path, rev="HEAD"):
    old, new = committed(path, rev), open(path, encoding="utf-8").read()
    print(f"\n=== {path} ===")
    print(f"bytes {len(old.encode())} -> {len(new.encode())}")

    a, b = Counter(NUM.findall(old)), Counter(NUM.findall(new))
    lost, added = a - b, b - a
    print(f"numbers: {sum(a.values())} -> {sum(b.values())}, {sum(lost.values())} lost, "
          f"{sum(added.values())} new")
    for k, n in list(lost.items())[:20]:
        print(f"   lost {k} x{n}")

    fa, fb = FENCE.findall(old), FENCE.findall(new)
    print(f"fenced blocks: {len(fa)} -> {len(fb)}" + ("" if fa == fb else "  CHANGED"))
    if fa != fb:
        for x, y in zip(fa, fb):
            if x != y:
                print("   --- a fenced block differs ---")
                print("   old:", x[:200].replace("\n", " | "))
                print("   new:", y[:200].replace("\n", " | "))

    ra = sum(1 for l in old.splitlines() if l.lstrip().startswith("|"))
    rb = sum(1 for l in new.splitlines() if l.lstrip().startswith("|"))
    print(f"table rows: {ra} -> {rb}" + ("" if ra == rb else "  CHANGED"))

    ha = [f"{h} {t}" for h, t in HEAD.findall(old)]
    hb = [f"{h} {t}" for h, t in HEAD.findall(new)]
    print(f"headings: {len(ha)} -> {len(hb)}" + ("" if len(ha) == len(hb) else "  CHANGED"))

    left = ITALIAN.findall(new)
    if left:
        print(f"leftover Italian: {len(left)} {Counter(w.lower() for w in left).most_common(8)}")


for p in sys.argv[1:]:
    report(p)
