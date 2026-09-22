"""Which line ranges of a document are still Italian."""
import re, sys
from collections import Counter

IT = re.compile(r"\b(perch[eé]|quindi|questo|questa|questi|della|dello|degli|delle|sono|viene|"
                r"anche|ogni|nella|nello|dalla|dallo|invece|soltanto|misura|misure|passata|"
                r"passate|esperti|come|non|che|con|per|una|uno|gli|alla|allo|dei|nel|sul|"
                r"pi[uù]|gi[aà]|fra|senza|dopo|prima|sopra|sotto|tutto|tutti|fatto|resta|"
                r"serve|vuole|legge|scrive|cambia|vale|costa|conta|tiene|entra|esce)\b", re.I)

path = sys.argv[1]
lines = open(path, encoding="utf-8").read().splitlines()

sections, cur, start = [], "(head)", 0
for i, l in enumerate(lines):
    if l.startswith("## "):
        sections.append((cur, start, i - 1))
        cur, start = l[3:].strip(), i
sections.append((cur, start, len(lines) - 1))

total = 0
for name, a, b in sections:
    hits = Counter()
    for l in lines[a:b + 1]:
        for w in IT.findall(l):
            hits[w.lower()] += 1
    n = sum(hits.values())
    total += n
    flag = "ITALIAN" if n > 12 else ("some" if n > 3 else "")
    if flag:
        print(f"{flag:8} lines {a+1:>5}-{b+1:<5} n={n:<5} {name[:70]}")
print(f"\ntotal markers: {total}")
