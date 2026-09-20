#!/usr/bin/env python3
"""mutate_reports.py — do the `--check` of the routing reports see a wrong formula? A check never
seen red proves nothing (docs/LEZIONI.md #43): each mutation is applied to a copy of the script in
a temporary directory, the copy's `--check` is run, and it must fail. A few seconds, no container:

    tools/.venv/Scripts/python.exe tools/mutate_reports.py

One line per mutation; exit 1 if any stays green or no longer applies."""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

MUTATIONS = {
    "route_trace_report.py": [
        ("wasted counts used units too", "wasted += e not in coming", "wasted += 1"),
        ("stalls at layer 0 count every layer", "stalls0 += layer == 0", "stalls0 += 1"),
        ("time model: reads do not queue on one disk",
         "disk_free = max(disk_free, asked_at) + read_ms", "disk_free = asked_at + read_ms"),
        ("time model: a unit in flight counts as there",
         "ready = max(ready, cache[unit])", "ready = max(ready, known)"),
    ],
    "route_graph_report.py": [
        ("table compares with the first occurrence, not the last",
         "        last[tok] = t", "        last.setdefault(tok, t)"),
        ("jaccard divides by the smaller set",
         "return len(a & b) / len(a | b) if a or b else 1.0",
         "return len(a & b) / min(len(a), len(b)) if a or b else 1.0"),
        ("mask has no floor per layer",
         "    floor = min(trace.n_expert, 2 * trace.n_used)", "    floor = 0"),
        ("margins: the gap is the first probability alone",
         "gaps = sorted(trace.margins[i] - trace.margins[i + 1]", "gaps = sorted(trace.margins[i]"),
        ("inside counts units, not activations",
         "return sum(c for unit, c in counts.items() if unit in hot) / sum(counts.values())",
         "return sum(1 for unit in counts if unit in hot) / len(counts)"),
        ("most frequent: learned on the test tokens too",
         "c = Counter(e for t in range(split) for e in rows[t][layer])",
         "c = Counter(e for t in range(trace.n_tokens) for e in rows[t][layer])"),
        ("static graph: tried on the tokens it learned from",
         "for t in range(split, trace.n_tokens):", "for t in range(0, split):"),
        ("coverage over the units used, not over all units",
         "k = int(fraction * trace.n_layers * trace.n_expert)", "k = int(fraction * len(counts))"),
    ],
}


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in MUTATIONS:
            shutil.copy(os.path.join(HERE, name), os.path.join(tmp, name))
        for name, cases in MUTATIONS.items():
            src = open(os.path.join(HERE, name), encoding="utf-8").read()
            target = os.path.join(tmp, name)
            rc = subprocess.run([sys.executable, target, "--check"], capture_output=True).returncode
            print(f"{name}: no mutation: {'green' if rc == 0 else 'RED'}")
            bad += rc != 0
            for label, old, new in cases:
                if src.count(old) < 1:
                    print(f"{name}: {label}: does not apply")
                    bad += 1
                    continue
                with open(target, "w", encoding="utf-8") as f:
                    f.write(src.replace(old, new, 1))
                rc = subprocess.run([sys.executable, target, "--check"], capture_output=True).returncode
                print(f"{name}: {label}: {'RED' if rc != 0 else 'green'}")
                bad += rc == 0
            with open(target, "w", encoding="utf-8") as f:
                f.write(src)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
