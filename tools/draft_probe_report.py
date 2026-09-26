"""draft_probe_report.py -- what a draft read from the exact model's own bits would give, from
tools/draft_probe.c's lines (docs/MEASUREMENTS.md §The engine read as entangled pairs, pair 1).

    tools/.venv/Scripts/python.exe tools/draft_probe_report.py build/entangled/probe_*.tsv

Files probe_<text>_<ctx>.tsv follow the exact greedy continuation, probe_<text>-text_<ctx>.tsv the
real text (a greedy continuation can loop: rep4 is the share of its 4-grams seen before in it).

Per text, context and draft variant: the top-1 agreement with the exact model's token, split where
the exact model is unsure (its top-2 margin under 1 nat) or not; then speculation simulated along
the sequence, pass after pass: the draft proposes up to k tokens (or, adaptive, stops at the first
position where its own margin falls under tau), the exact pass keeps the leading agreements and adds
its own token. Bytes a token, in MiB:

    B = sum over passes (drafts computed * D + N + E1 * U(rows) + KV) / tokens

D the variant's own bytes a draft token, N the dense weights a pass reads once, E1 one token's
experts, U(r) the union of the experts of r consecutive tokens (question 54), KV the exact cache
read once a pass. Against ours (N + E1 + KV, F32) and llama.cpp's (their KV in F16, half).
OLMoE-1B-7B Q8_0: N = 385 MiB (attention 272, head 104, router and norms 9), E1 = 815, a position
0.25 MiB of F32 K and V; a Q8_0 block's high plane is 18 of its 34 bytes.

Question 75: the kv4 and kv8 variants read the history from a copy of 4 or 8 bits a value (and its
mins and steps, fp16: K per channel over 32 positions, V per position), the last 64 positions at
16 bits; every planes variant is also scored with a draft head of the first 16384 rows (+h16k: the
lowest ids, the tokenizer's most frequent merges; a draft whose exact token is not among them
misses). "oos": k and tau chosen on one half of the positions and scored on the other, both ways
(the in-sample best is an upper bound).
"""
import os
import re
import sys

N, E1, POS = 385.0, 815.0, 0.25
HI = 18.0 / 34.0
U = {1: 1.00, 2: 1.55, 3: 1.97, 4: 2.33, 6: 2.89, 8: 3.34, 9: 3.52, 12: 3.97, 16: 4.42}
TAUS = (0.5, 1.0, 2.0, 3.0)
RECENT = 64                    # positions the kv variants keep at 16 bits
SCALES = 5120.0 / 2 ** 20      # MiB of mins and steps a position: K 16 x 16 x 128 x 4 B / 32, V 16 x 16 x 4 B
HEAD, VOCAB, H16 = 104.0, 50304, 16384


def union(r):
    if r in U:
        return U[r]
    if r > 16:  # past the measured rows (a leaf on a chain of 15): the last slope
        return U[16] + (U[16] - U[12]) * (r - 16) / 4
    lo = max(k for k in U if k < r)
    hi = min(k for k in U if k > r)
    return U[lo] + (U[hi] - U[lo]) * (r - lo) / (hi - lo)


def draft_bytes(name, ctx):
    """MiB a draft token reads: the weights' high plane (or all of them), the used experts
    (8, or the top 4), the cache's high halves (or a window of 4 + 256 positions, or the history's
    copy of kvbits bits and the last RECENT), the head's first H16 rows only with +h16k."""
    cut = HI if "planes" in name else 1.0
    experts = E1 / 2 if "top4" in name else E1
    head_saved = HEAD * cut * (1 - H16 / VOCAB) if name.endswith("+h16k") else 0.0
    kv = re.search(r"kv(\d+)", name)
    if kv:
        old = max(0, ctx - RECENT)
        cache = old * (POS * int(kv.group(1)) / 32 + SCALES) + min(ctx, RECENT) * POS / 2
    else:
        positions = 4 + 256 if ("win256" in name or "sel192" in name or "nx96" in name) else ctx
        cache = positions * POS / 2
    return (N + experts) * cut - head_saved + cache


def simulate(flags, own, k, tau, d, ctx, second=None):
    """Passes one after the other along the sequence: (tokens a pass, MiB a token). With second
    (the draft's second choice matches the exact token), the chain stops at the first position
    whose own margin is under tau with two leaves there, its first and its second choice: one row
    more in the pass, the union of experts grown by that row."""
    i, passes, tokens, mib = 0, 0, 0, 0.0
    while i + k < len(flags):
        proposed = computed = leaf = 0
        while proposed < k:
            computed += 1
            if tau is not None and own[i + proposed] < tau:
                if second is not None:
                    leaf = 1
                    proposed += 1
                break
            proposed += 1
        acc = 0
        while acc < proposed and (flags[i + acc] or (leaf and acc == proposed - 1 and second[i + acc])):
            acc += 1
        passes += 1
        tokens += acc + 1
        mib += computed * d + N + E1 * union(proposed + 1 + leaf) + ctx * POS
        i += acc + 1
    if passes == 0:
        return 0.0, float("inf")
    return tokens / passes, mib / tokens


def cross(flags, own, d, ctx, sec):
    """The leaf's (k, tau) chosen on one half of the positions and scored on the other, both ways:
    the mean tokens a pass and MiB a token."""
    h = len(flags) // 2
    got = []
    for fit, score in ((slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))):
        _, _, k, tau = min((simulate(flags[fit], own[fit], k, tau, d, ctx, sec[fit]) + (k, tau)
                            for tau in TAUS for k in range(1, 16)), key=lambda s: s[1])
        got.append(simulate(flags[score], own[score], k, tau, d, ctx, sec[score]))
    return sum(g[0] for g in got) / 2, sum(g[1] for g in got) / 2


def main():
    rows = []
    for p in sys.argv[1:]:
        m = re.search(r"probe_([\w-]+?)_(\d+)\.tsv$", os.path.basename(p))
        if not m:
            continue
        text, ctx = m.group(1), int(m.group(2))
        lines = [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
        if len(lines) < 2:
            continue
        head, data = lines[0], lines[1:]
        names = [h for h in head[3:] if not h.endswith("_m") and not h.endswith("_2")]
        exact = [int(r[1]) for r in data]
        margin = [float(r[2]) for r in data]
        seen, rep = set(), 0
        for i in range(3, len(exact)):
            g4 = tuple(exact[i - 3:i + 1])
            rep += g4 in seen
            seen.add(g4)
        repeat = rep / max(1, len(exact) - 3)
        cover = sum(e < H16 for e in exact) / len(exact)
        runs = [(j, name, False) for j, name in enumerate(names)]
        runs += [(j, name, True) for j, name in enumerate(names) if "planes" in name]
        for j, column, pruned in runs:
            name = column + ("+h16k" if pruned else "")
            flags = [int(r[3 + j]) == e for r, e in zip(data, exact)]
            mcol = head.index(column + "_m") if column + "_m" in head else None
            own = [float(r[mcol]) for r in data] if mcol is not None else None
            scol = head.index(column + "_2") if column + "_2" in head else None
            sec = [int(r[scol]) == e for r, e in zip(data, exact)] if scol is not None else None
            if pruned:  # the head's first H16 rows: a token past them is never drafted
                flags = [f and e < H16 for f, e in zip(flags, exact)]
                sec = [s and e < H16 for s, e in zip(sec, exact)] if sec is not None else None
            agree = sum(flags) / len(flags)
            unsure = [f for f, mg in zip(flags, margin) if mg < 1.0]
            sure = [f for f, mg in zip(flags, margin) if mg >= 1.0]
            d = draft_bytes(name, ctx)
            best = min((simulate(flags, own, k, None, d, ctx) + (k, None) for k in range(1, 16)),
                       key=lambda s: s[1])
            if own is not None:
                for tau in TAUS:
                    s = min((simulate(flags, own, k, tau, d, ctx) + (k, tau) for k in range(1, 16)),
                            key=lambda s: s[1])
                    if s[1] < best[1]:
                        best = s
            leafbest = oos = None
            if own is not None and sec is not None:
                leafbest = min((simulate(flags, own, k, tau, d, ctx, sec) + (k, tau)
                                for tau in TAUS for k in range(1, 16)), key=lambda s: s[1])
                oos = cross(flags, own, d, ctx, sec)
            top2 = sum(f or g for f, g in zip(flags, sec)) / len(flags) if sec is not None else None
            rows.append((text, ctx, name, len(flags), repeat, agree,
                         (sum(unsure) / len(unsure) if unsure else 0.0, len(unsure),
                          sum(sure) / len(sure) if sure else 0.0),
                         best, N + E1 + ctx * POS, N + E1 + ctx * POS / 2, leafbest, top2, oos, cover))
    print("%-10s %5s %-24s %4s %5s %7s %13s %7s %3s %4s %6s %7s %6s %6s %6s" %
          ("text", "ctx", "draft", "n", "rep4", "agree", "unsure (n)", "sure", "k", "tau", "tok/ps", "MiB/tok",
           "ours", "llama", "vs ll"))
    for (text, ctx, name, n, repeat, agree, (au, nu, asu), (t, b, k, tau), ours, theirs, leaf, top2, oos,
         cover) in sorted(rows, key=lambda r: (r[1], r[0], r[2])):
        extra = ""
        if leaf is not None:
            extra = "  | top2 %5.1f%% leaf k %2d tau %.1f %5.2f tok/ps %5.0f MiB %5.2fx | oos %5.2f tok/ps %5.0f MiB %5.2fx" % (
                100 * top2, leaf[2], leaf[3], leaf[0], leaf[1], theirs / leaf[1], oos[0], oos[1], theirs / oos[1])
        if name.endswith("+h16k"):
            extra += " | h16k covers %4.1f%%" % (100 * cover)
        print("%-10s %5d %-24s %4d %4.0f%% %6.1f%% %6.1f%% (%3d) %6.1f%% %3d %4s %6.2f %7.0f %6.0f %6.0f %5.2fx%s" %
              (text, ctx, name, n, 100 * repeat, 100 * agree, 100 * au, nu, 100 * asu, k,
               "-" if tau is None else "%.1f" % tau, t, b, ours, theirs, theirs / b, extra))


if __name__ == "__main__":
    main()
