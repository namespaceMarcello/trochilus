"""ab_inproc.py -- the arms of in-process A/Bs (`trochilus generate --ab <switch> --profile-json <file>`): the
passes alternated A B B A in one process, each arm's zones in its own phase ("decode", "decode_b"). Both arms run
in the same file, memory and threads, where two files of the same bytes differ by 1-2% (docs/LESSONS.md #243).

    tools/.venv/Scripts/python.exe tools/ab_inproc.py <profile.json> [...]

Per run: each arm's passes and mean ms a pass, B/A of the pass and of every zone (a zone's seconds over the arm's
passes). Over the runs: the median of B/A, and the mean with its standard error. "--ab none" (both arms the
same code) measures the tool's own noise: its B/A is what a real switch must be told from.

The blocks: passes 4j..4j+3 are A B B A or B A A B (the run's `ab_pass_arm`, since 2026-09-27: no period, so the
KV's fresh pages every 8 positions fall on both arms, docs/LESSONS.md #247; older runs A B B A A B B A); a block's
B/A is its two B passes over its two A passes, and the median over the blocks (the runs' blocks pooled) is the
estimate a pass the machine interrupted, or the first one, cannot move (the arms' means above can).
"""
import json
import statistics
import sys

ZONES = ["qkv_proj", "attention", "attn_out_proj", "router", "expert_gate_up", "expert_act", "expert_down",
         "lm_head", "sample", "kv_write"]


def arms(path):
    """{zone: (a ms a pass, b ms a pass)} for "pass" and ZONES, and the passes of each arm"""
    phases = json.load(open(path))["engine"]["phases"]
    a, b = phases["decode"], phases.get("decode_b")
    if b is None or "token" not in a["zones"] or "token" not in b.get("zones", {}):
        sys.exit("ab_inproc: %s has no two arms (was it run with --ab?)" % path)
    na, nb = a["zones"]["token"]["calls"], b["zones"]["token"]["calls"]
    out = {"pass": (1e3 * a["zones"]["token"]["seconds"] / na, 1e3 * b["zones"]["token"]["seconds"] / nb)}
    for z in ZONES:
        if z in a["zones"] and z in b["zones"]:
            out[z] = (1e3 * a["zones"][z]["seconds"] / na, 1e3 * b["zones"][z]["seconds"] / nb)
    return out, na, nb


def walls(d, key):
    """the walls under key ("ab_pass" or "ab_prompt") and their arms (0: A, 1: B)"""
    w = d.get(key + "_ms", [])
    a = d.get(key + "_arm")
    if a is None:  # runs before the arms were written: A B B A A B B A
        a = [((i + 1) // 2) % 2 for i in range(len(w))]
    return w, a


def block_ratios(w, a):
    """each whole block of four's B/A: its two B walls over its two A walls"""
    out = []
    for j in range(0, len(w) - 3, 4):
        sa = sum(w[i] for i in range(j, j + 4) if a[i] == 0)
        sb = sum(w[i] for i in range(j, j + 4) if a[i] == 1)
        if sa > 0 and sb > 0:
            out.append(sb / sa)
    return out


def blocks(path):
    """each whole block's B/A from the run's pass walls (empty for a run without them), and the walls and arms"""
    w, a = walls(json.load(open(path)), "ab_pass")
    return block_ratios(w, a), w, a


def trimmed(w, arm):
    """B/A of the arms' mean walls without the first block: the first pass after the prompt ran 15-24 ms against a
    median of 16 in the A/A of 2026-09-27 (docs/MEASUREMENTS.md §An A/B inside one process)"""
    a = [x for i, x in enumerate(w) if i >= 4 and arm[i] == 0]
    b = [x for i, x in enumerate(w) if i >= 4 and arm[i] == 1]
    return statistics.mean(b) / statistics.mean(a) if a and b else None


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    ratios, pooled, trims = {}, [], []
    for p in paths:
        z, na, nb = arms(p)
        ra, rb = z["pass"]
        bl, w, arm = blocks(p)
        line = "%s: A %d passes %.3f ms, B %d passes %.3f ms, B/A %.4f" % (p, na, ra, nb, rb, rb / ra)
        if bl:
            t = trimmed(w, arm)
            line += "; without the first block %.4f; blocks %d, median %.4f; first pass %.2f ms, median %.2f" % (
                t, len(bl), statistics.median(bl), w[0], statistics.median(w))
            trims.append(t)
        print(line)
        pooled += bl
        for k, (x, y) in z.items():
            if x > 0:
                ratios.setdefault(k, []).append(y / x)
    if len(paths) > 1:
        print("over %d runs, B/A: median, mean +- its standard error" % len(paths))
        for k, v in ratios.items():
            se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
            print("  %-16s %.4f  %.4f +- %.4f" % (k, statistics.median(v), statistics.mean(v), se))
    if len(pooled) > 1:
        # the median's standard error, from the blocks' spread (1.2533 sd / sqrt(n) for a normal spread)
        se = 1.2533 * statistics.stdev(pooled) / len(pooled) ** 0.5
        print("the blocks of every run pooled: %d, median B/A %.4f +- %.4f" % (len(pooled), statistics.median(pooled), se))
    if len(trims) > 1:
        se = statistics.stdev(trims) / len(trims) ** 0.5
        print("THE ESTIMATE, the arms' means without the first block: B/A %.4f +- %.4f over %d runs (median %.4f)" %
              (statistics.mean(trims), se, len(trims), statistics.median(trims)))
    # --ab-prompt: the prompt's evaluations in blocks of four (the warm-up eval is outside them already)
    prompt = []
    for p in paths:
        w, arm = walls(json.load(open(p)), "ab_prompt")
        prompt += block_ratios(w, arm)
        if w:
            a = [x for i, x in enumerate(w) if arm[i] == 0]
            b = [x for i, x in enumerate(w) if arm[i] == 1]
            print("%s: the prompt, A %d evals %.1f ms, B %d evals %.1f ms, B/A %.4f" %
                  (p, len(a), statistics.mean(a), len(b), statistics.mean(b), statistics.mean(b) / statistics.mean(a)))
    if len(prompt) > 1:
        se = statistics.stdev(prompt) / len(prompt) ** 0.5
        print("THE PROMPT'S ESTIMATE, its A B B A blocks: B/A %.4f +- %.4f over %d blocks (median %.4f)" %
              (statistics.mean(prompt), se, len(prompt), statistics.median(prompt)))


if __name__ == "__main__":
    main()
