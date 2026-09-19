#!/usr/bin/env python3
"""prefill_context_report.py — what tools/prefill_context.sh measured, as the tables of
docs/MISURE.md §Prefill su prompt lunghi.

    prefill_context_report.py attn <bench_attn output> [...]
        per prompt length and thread count: ms of one layer's attention for every variant, and
        where the time of the engine's kernel ("one full") goes: softmax, products and weighted
        sum with the keys and the values in cache ("blk nosm"), and the rest, which is the keys
        and the values read again for every query.
    prefill_context_report.py zones <profile_suite report> [<report after>]
        prefill zones of every scenario: ms, share, MiB read per token, GB/s; the zones that were
        one thread's work before this step are added up. With two reports, before and after.

The speed tables come from tools/decode_context_report.py speed <ab_modes output>.
Reads files only: every number it prints comes from a run kept on disk.
"""
import re
import sys

# one token's own work: a single thread ran it before the prefill step of 2026-09-19
PER_TOKEN_ZONES = ("embed", "attn_norm", "qk_norm", "rope", "kv_write", "ffn_norm", "expert_gather")
LAYERS = 16   # OLMoE-1B-7B: the benchmark times one layer, the engine's zone is 16 of them


def read_attn(path):
    """{(n_pos, threads, heads, group, block): {variant: (ms, ns)}} in order of appearance."""
    runs, key = {}, None
    group = block = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^bench_attn \d+, .* group (\d+), block (\d+)", line)
            if m:
                group, block = int(m.group(1)), int(m.group(2))
                continue
            m = re.match(r"^attn\s+(\d+) (\S+ \S+)\s+t=(\d+)\s+h=(\d+)\s+([0-9.]+) ms .*?\)\s+([0-9.]+) ns", line)
            if m:
                key = (int(m.group(1)), int(m.group(3)), int(m.group(4)), group, block)
                runs.setdefault(key, {})[m.group(2)] = (float(m.group(5)), float(m.group(6)))
    return runs


def attn(paths):
    for path in paths:
        print("## %s" % path)
        for key, v in read_attn(path).items():
            n_pos, threads, heads, group, block = key
            print("prompt %d, %d threads, %d heads, group %d, block %d (ms of one layer; ns per query and position)"
                  % key)
            for name, (ms, ns) in v.items():
                line = "  %-12s %9.3f ms  %6.2f ns" % (name, ms, ns)
                if "one full" in v and name != "one full":
                    line += "   %5.1f%% of one full" % (ms / v["one full"][0] * 100.0)
                print(line)
            if all(x in v for x in ("one full", "one softmax", "blk nosm")):
                full, soft, cached = v["one full"][0], v["one softmax"][0], v["blk nosm"][0]
                again = full - soft - cached
                print("  -> softmax %.0f%%, products and weighted sum with K and V in cache %.0f%%, "
                      "K and V read again %.0f%% (%.1f ms; %.1f ms by 'one nosm' - 'blk nosm')"
                      % (soft / full * 100, cached / full * 100, again / full * 100, again,
                         v["one nosm"][0] - cached if "one nosm" in v else float("nan")))
                print("  -> the engine's zone would be %.0f ms (x%d layers)" % (full * LAYERS, LAYERS))
            for name in ("blk full", "blkx full", "grp full"):
                if "one full" in v and name in v:
                    print("  -> %s: %.3fx the engine's kernel" % (name, v["one full"][0] / v[name][0]))


def read_zones(path):
    """{scenario: {"tok/s": float, "zones": {zone: (ms, share, mib_per_token, gb_per_sec)}}}, prefill only."""
    out, name, in_prefill = {}, None, False
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^== (\S+)", line)
            if m:
                name, in_prefill = m.group(1), False
                out[name] = {"tok/s": 0.0, "zones": {}}
                continue
            if name is None:
                continue
            m = re.match(r"^\s+prefill\s+([0-9.]+) tok/s", line)
            if m:
                out[name]["tok/s"] = float(m.group(1))
                continue
            if re.match(r"^\s+prefill zones", line):
                in_prefill = True
                continue
            if re.match(r"^\s+(decode zones|prefill weights)", line):
                in_prefill = False
                continue
            m = re.match(r"^\s+(\w+)\s+([0-9.]+) ms\s+([0-9.]+)%\s+[0-9.]+ ms/tok(?:\s+([0-9.]+) MiB/tok\s+([0-9.]+) GB/s)?", line)
            if m and in_prefill:
                out[name]["zones"][m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                                  float(m.group(4) or 0.0), float(m.group(5) or 0.0))
    return out


def zones(paths):
    before = read_zones(paths[0])
    after = read_zones(paths[1]) if len(paths) > 1 else None
    for name, sc in before.items():
        print("## %s: prefill %.2f tok/s%s" % (name, sc["tok/s"],
              (" -> %.2f (%.3fx)" % (after[name]["tok/s"], after[name]["tok/s"] / sc["tok/s"]))
              if after and name in after else ""))
        zs = sc["zones"]
        total = zs.get("token", (0.0,))[0]
        for z, (ms, share, mib, gbs) in sorted(zs.items(), key=lambda kv: -kv[1][0]):
            line = "  %-15s %10.1f ms %5.1f%%" % (z, ms, share)
            if mib > 0:
                line += "  %8.2f MiB/tok %7.2f GB/s" % (mib, gbs)
            if after and name in after and z in after[name]["zones"]:
                a = after[name]["zones"][z]
                line += "   -> %10.1f ms" % a[0]
                if a[2] > 0:
                    line += "  %8.2f MiB/tok %7.2f GB/s" % (a[2], a[3])
                if ms > 0:
                    line += "  (%+.1f ms)" % (a[0] - ms)
            print(line)
        serial = sum(zs[z][0] for z in PER_TOKEN_ZONES if z in zs)
        line = "  per-token zones (%s): %.1f ms, %.1f%% of the prefill" % (
            ", ".join(PER_TOKEN_ZONES), serial, serial / total * 100 if total else 0.0)
        if after and name in after:
            a_serial = sum(after[name]["zones"][z][0] for z in PER_TOKEN_ZONES if z in after[name]["zones"])
            line += "  -> %.1f ms" % a_serial
        print(line)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("attn", "zones"):
        print(__doc__)
        return 2
    if sys.argv[1] == "attn":
        attn(sys.argv[2:])
    else:
        zones(sys.argv[2:4])
    return 0


if __name__ == "__main__":
    sys.exit(main())
