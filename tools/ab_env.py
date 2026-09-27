"""ab_env.py -- the report of tools/ab_env.sh: runs of one binary under environment modes, process by process.

Reads <dir>/<label>-<ns>.json (--profile-json of each run; the label is the mode's, ns the clock when it started).
The i-th run of every mode (in time order) belongs to round i; round 0 is dropped (tools/ab_modes.sh's warm-up).
Per mode: the prompt's ms (the prefill phase's token zone), the decode's ms a token, and every zone per token of
each phase (us), as medians over the rounds with min and max; then, per mode after the first, the median over the
rounds of this mode's time over the first's (above 1: the first mode faster; B/A when the first is the new road)
with a standard error over rounds.

    tools/.venv/Scripts/python.exe tools/ab_env.py build/ab_env
"""
import glob
import json
import os
import re
import statistics
import sys


def runs(d):
    by = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        m = re.match(r"(.+)-(\d+)\.json$", os.path.basename(f))
        if m:
            by.setdefault(m.group(1), []).append((int(m.group(2)), f))
    order = sorted(by, key=lambda k: min(t for t, _ in by[k]))
    return order, {k: [f for _, f in sorted(v)] for k, v in by.items()}


def metrics(f):
    ph = json.load(open(f))["engine"]["phases"]
    out = {}
    for name in ("prefill", "decode"):
        p = ph.get(name)
        if p is None or p["tokens"] == 0:
            continue
        n = p["tokens"]
        tok = p["zones"]["token"]["seconds"]
        if name == "prefill":
            out["prompt ms"] = tok * 1e3
        else:
            out["decode ms/token"] = tok / n * 1e3
        for z, v in p["zones"].items():
            if z != "token" and v["calls"] > 0:
                out["%s %s us/token" % (name, z)] = v["seconds"] / n * 1e6
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "build/ab_env"
    order, files = runs(d)
    if len(order) < 2:
        sys.exit("ab_env.py: fewer than two modes in " + d)
    rounds = min(len(files[k]) for k in order)
    table = {k: [metrics(f) for f in files[k][1:rounds]] for k in order}
    print("%s: modes %s, %d rounds kept (round 0 dropped)" % (d, " ".join(order), rounds - 1))
    keys = sorted({m for k in order for r in table[k] for m in r},
                  key=lambda m: (not m.startswith("prompt"), not m.startswith("decode ms"), m))
    first = order[0]
    for m in keys:
        cells = []
        for k in order:
            vals = [r[m] for r in table[k] if m in r]
            cells.append("%s %9.2f (%.2f-%.2f)" % (k, statistics.median(vals), min(vals), max(vals)) if vals else
                         "%s -" % k)
        ratios = []
        for k in order[1:]:
            pr = [b[m] / a[m] for a, b in zip(table[first], table[k]) if m in a and m in b and a[m] > 0]
            if len(pr) >= 2:
                se = statistics.stdev(pr) / len(pr) ** 0.5
                ratios.append("%s/%s %.4f +- %.4f" % (k, first, statistics.median(pr), se))
        print("%-34s %s   %s" % (m, "  ".join(cells), "  ".join(ratios)))


main()
