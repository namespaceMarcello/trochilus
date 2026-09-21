#!/usr/bin/env python3
"""prefill_overlap_report.py -- the two halves of the prefill, from the profiles of
`tools/prefill_overlap.sh`: the time the engine waits for the disk (zone `weight_read`) and the
time it computes, plus the ceiling of overlapping the two.

    ceiling = total / max(disk, compute)

The ceiling is an upper bound of a reader nobody has written yet -- modello, non misura -- while
the two halves it is built from are measured. Round 0 is dropped as warm-up, like every other
session; every number is the median over the rounds left, with min, max and spread.

    python tools/prefill_overlap_report.py build/prefill_overlap
"""
import json
import pathlib
import re
import sys

ORDER = ["p512b100", "p512b50", "p2048b100", "p2048b50", "p2048b50one"]
NAMES = {
    "p512b100": "prompt 512, residente",
    "p512b50": "prompt 512, budget 50%",
    "p2048b100": "prompt 2048, residente",
    "p2048b50": "prompt 2048, budget 50%",
    "p2048b50one": "prompt 2048, 50%, 1 passata",
}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def spread(xs):
    m = median(xs)
    return 0.0 if m == 0 else (max(xs) - min(xs)) / m * 100


def main(argv):
    out = pathlib.Path(argv[1] if len(argv) > 1 else "build/prefill_overlap")
    runs = {}
    for f in sorted(out.glob("*.json")):
        m = re.fullmatch(r"(.+)-(\d+)", f.stem)
        if not m or int(m.group(2)) == 0:  # round 0 is the warm-up
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        pre = d["engine"]["phases"]["prefill"]
        zones = pre.get("zones", {})
        read = zones.get("weight_read", {})
        runs.setdefault(m.group(1), []).append(
            (pre["seconds"], read.get("seconds", 0.0), read.get("bytes", 0), pre["tokens"]))
    if not runs:
        print("prefill_overlap_report: no profile in", out)
        return 1

    print(f"{'modo':<26}{'prefill s':>10}{'disco s':>9}{'calcolo s':>11}{'MiB letti':>11}"
          f"{'GB/s':>7}{'tetto':>8}{'spread':>8}{'n':>3}")
    for key in ORDER:
        if key not in runs:
            continue
        rows = runs[key]
        tot = median([r[0] for r in rows])
        disk = median([r[1] for r in rows])
        mib = median([r[2] for r in rows]) / (1024 * 1024)
        comp = tot - disk
        ceiling = tot / max(disk, comp) if max(disk, comp) > 0 else 1.0
        gbs = (mib / 1024) / disk if disk > 0 else 0.0
        print(f"{NAMES[key]:<26}{tot:>10.2f}{disk:>9.2f}{comp:>11.2f}{mib:>11.0f}"
              f"{gbs:>7.2f}{ceiling:>7.2f}x{spread([r[0] for r in rows]):>7.1f}%{len(rows):>3}")

    print("\nControllo: sui modi residenti il disco deve essere ~0, e la differenza fra i due")
    print("budget alla stessa lunghezza deve valere quanto il disco che la zona dichiara.")
    print("Il «tetto» e' un limite superiore (modello, non misura): con un lettore perfetto e")
    print("nessuna contesa il prefill non puo' scendere sotto max(disco, calcolo).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
