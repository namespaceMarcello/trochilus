#!/usr/bin/env python3
"""decode_context_report.py — what tools/decode_context.sh measured, as the tables of
docs/MEASUREMENTS.md §Decode a contesto lungo.

    decode_context_report.py speed <ab_modes output> [...]
        per context: median (min-max) of every mode, the ratio to the first mode of that context
        and the A/A differences; the threshold of the session is its worst A/A per phase.
    decode_context_report.py model <ab_modes output> <label> [...]
        ms per token of the modes "<label>-<context>" against the context: the line through them
        (ms at context zero, ms per 1000 tokens of context, the bandwidth the KV cache is read at)
        beside the hypothesis of docs/STATUS.md, tok/s = band / (weights + KV per token x context).
    decode_context_report.py zones <profile_suite results json> [...]
        decode zones of every scenario: ms per token, MiB read per token, GB/s in the zone.

Reads files only: every number it prints comes from a run kept on disk.
"""
import json
import re
import statistics
import sys

WEIGHTS_BYTES = 1223.6e6        # OLMoE-1B-7B Q8_0: bytes of weights one decode token reads (profiler)
KV_BYTES_PER_TOKEN = 262144.0   # 16 layers x 16 heads x 128 x 4 bytes x (K and V) per token of context
GEN_TOKENS = 48                 # `generate -n 48`: 47 evaluations, the context grows by that much


def read_runs(path):
    """{phase: {label: [values of rounds > 0]}} from an ab_modes.sh output, in order of appearance."""
    runs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^(\S+) (prefill|decode|width) (\d+) ([0-9.]+)\s*$", line)
            if not m:
                continue
            # round 0 is the warm-up: it gives the order of the modes (later rounds rotate it), no value
            values = runs.setdefault(m.group(2), {}).setdefault(m.group(1), [])
            if int(m.group(3)) > 0:
                values.append(float(m.group(4)))
    return runs


def context_of(label):
    # "after8-2048-again", and the "p2048-again" of tools/prefill_context.sh measure
    m = re.search(r"(?:-|^p)(\d+)(-again)?$", label)
    return int(m.group(1)) if m else -1


def speed(paths):
    for path in paths:
        runs = read_runs(path)
        print("## %s" % path)
        for phase in ("decode", "prefill"):
            worst = 0.0
            labels = list(runs.get(phase, {}))
            for ctx in sorted({context_of(x) for x in labels}):
                here = [x for x in labels if context_of(x) == ctx]
                med = {x: statistics.median(runs[phase][x]) for x in here}
                first = here[0]
                for x in here:
                    v = runs[phase][x]
                    line = "%-8s %-20s %8.2f (%.2f-%.2f)  %.3fx" % (phase, x, med[x], min(v), max(v),
                                                                     med[x] / med[first])
                    if x.endswith("-again"):
                        twin = x[:-len("-again")]
                        aa = abs(med[x] / med[twin] - 1.0) * 100.0
                        worst = max(worst, aa)
                        line += "   A/A %.1f%%" % aa
                    if phase == "decode" and x in runs.get("width", {}):
                        w = runs["width"][x]
                        line += "   width " + " ".join("%dx%d" % (w.count(k), int(k)) for k in sorted(set(w)))
                        # a choice that flips from one round to the next is not a choice: the count
                        # says how stable the measured default is, which a median hides
                        changes = sum(1 for p, q in zip(w, w[1:]) if p != q)
                        if len(set(w)) > 1:
                            line += " (%d changes in %d rounds)" % (changes, len(w))
                    print(line)
                # every real mode against every copy of the other
                real = [x for x in here if not x.endswith("-again")]
                if len(real) == 2:
                    a, b = real
                    pairs = [(b, a), (b + "-again", a), (b, a + "-again"), (b + "-again", a + "-again")]
                    print("%-8s %-20s %s" % (phase, "%s / %s" % (b, a), "  ".join(
                        "%.3fx" % (med[p] / med[q]) for p, q in pairs if p in med and q in med)))
            print("%-8s worst A/A of the session: %.1f%%\n" % (phase, worst))


def model(path, prefixes):
    runs = read_runs(path)["decode"]
    for prefix in prefixes:
        pts = []
        for label, v in runs.items():
            if label.startswith(prefix + "-") and not label.endswith("-again"):
                twin = runs.get(label + "-again", [])
                tps = statistics.median(v + twin)   # the mode and its copy are the same thing: 16 runs
                pts.append((context_of(label) + GEN_TOKENS / 2.0, 1000.0 / tps, tps))
        pts.sort()
        n = len(pts)
        mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
        slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
        base = my - slope * mx
        print("## %s, modes %s-<context> (each with its copy: 16 runs a point)" % (path, prefix))
        print("line through the points: %.2f ms at context zero (%.1f GB/s if it were all weights), "
              "%.3f ms per 1000 tokens of context (KV read at %.1f GB/s)" %
              (base, WEIGHTS_BYTES / base / 1e6, slope * 1000.0, KV_BYTES_PER_TOKEN / slope / 1e6))
        # the hypothesis with the band that makes it exact at the shortest context
        band = (WEIGHTS_BYTES + KV_BYTES_PER_TOKEN * pts[0][0]) / (pts[0][1] / 1000.0)
        print("%8s %9s %9s %12s %12s %14s" % ("context", "tok/s", "ms", "line ms", "41 GB/s", "%.1f GB/s" % (band / 1e9)))
        for ctx, ms, tps in pts:
            total = WEIGHTS_BYTES + KV_BYTES_PER_TOKEN * ctx
            print("%8d %9.2f %9.2f %12.2f %9.2f t/s %11.2f t/s   traffic %.1f GB/s" %
                  (ctx, tps, ms, base + slope * ctx, 41e9 / total, band / total, total * tps / 1e9))
        print("")


def zones(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print("## %s" % path)
        for sc in data["scenarios"]:
            d = sc["aggregate"]["decode"]
            print("== %s: decode %.2f tok/s (%.2f-%.2f), weights %.1f MiB/token, KV %.1f MiB/token, "
                  "weights + KV %.1f GB/s" % (sc["name"], d["tokens_per_sec_median"], d["tokens_per_sec_min"],
                                              d["tokens_per_sec_max"], d["mib_per_token_median"],
                                              d.get("kv_mib_per_token_median", 0.0),
                                              d.get("weights_kv_gb_per_sec_median", 0.0)))
            token_ms = d["zones"]["token"]["ms_per_token_median"]
            rest = token_ms
            for name, z in sorted(d["zones"].items(), key=lambda kv: -kv[1]["seconds_median"]):
                if name == "token":
                    continue
                rest -= z["ms_per_token_median"]
                line = "   %-16s %8.3f ms %5.1f%%" % (name, z["ms_per_token_median"], z["share_median"] * 100.0)
                if z.get("mib_per_token_median", 0.0) > 0:
                    line += "  %8.1f MiB  %6.1f GB/s" % (z["mib_per_token_median"], z["gb_per_sec_median"])
                print(line)
            print("   %-16s %8.3f ms        (token %.3f ms)" % ("outside zones", rest, token_ms))
        print("")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("speed", "model", "zones"):
        print(__doc__)
        return 2
    if sys.argv[1] == "speed":
        speed(sys.argv[2:])
    elif sys.argv[1] == "model":
        model(sys.argv[2], sys.argv[3:])
    else:
        zones(sys.argv[2:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
