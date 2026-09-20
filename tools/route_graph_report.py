#!/usr/bin/env python3
"""route_graph_report.py — docs/MISURE.md domanda 44: is the model's behaviour on code a small,
deterministic graph? From traces written by `trochilus run ... --route-trace <file>`
(tools/route_trace_report.py reads the format; version 2 adds token ids and router margins).

Usage:
    route_graph_report.py <trace.bin>                  the tables below for one trace
    route_graph_report.py --compare <a.bin> <b.bin>... hot sets of several traces against each other
    route_graph_report.py --mask-from <trace.bin> --keep <fraction> --out <mask.txt> [--random <seed>]
                                                       the units to switch off for the ablation
    route_graph_report.py --check                      handmade traces, answers computed by hand

small      how many (layer, expert) units the trace uses, the share of activations the most used
           10/25/50/75% of all units cover, how many units 99% of the activations need, how fast
           the set of units seen grows, the entropy of each layer's usage.
static     how much of layer L+1's choice a graph without state gives: co-occurrence counts
           between the experts of consecutive layers learned on the first 75% of the tokens and
           tried on the rest, against the layer's most frequent experts (no path at all). The
           router on the live state is the Q13 table of route_trace_report.py.
repeats    tokens whose set of experts at a layer was already seen, whole 16-layer paths seen
           twice, experts shared with the previous token (what an LRU lives on).
table      (version 2) the same token id again: is its set of experts the same as the last time
           that id came? Per layer: share of identical sets and mean Jaccard.
margins    (version 2) router probability of the last expert chosen minus the first one left
           out: median, and share of choices decided by less than 0.01 and 0.001.
compare    for every pair of traces: Jaccard of their hot sets (the most used 25% and 50% of all
           units) and the share of B's activations that fall inside A's hot set.
mask       units outside the most used `keep` fraction, never fewer than 2 * n_used kept in a
           layer; one "layer expert" per line, for `--expert-mask` (a measurement-only mode).

No dependencies beyond the standard library."""
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_trace_report import MAGIC2, Trace, to_bytes  # noqa: E402

COVER_FRACTIONS = (0.10, 0.25, 0.50, 0.75)
HOT_FRACTIONS = (0.25, 0.50)
TRAIN_SHARE = 0.75
K_VALUES = (8, 12, 16)


def rows_of(trace):
    return [[tuple(trace.chosen_row(t, layer)) for layer in range(trace.n_layers)]
            for t in range(trace.n_tokens)]


def unit_counts(trace, rows):
    return Counter((layer, e) for t in range(trace.n_tokens) for layer in range(trace.n_layers)
                   for e in rows[t][layer])


# ---- small ----------------------------------------------------------------------------------

def coverage(trace, rows, fraction):
    """Share of all activations that the most used `fraction` of ALL units cover."""
    counts = sorted(unit_counts(trace, rows).values(), reverse=True)
    k = int(fraction * trace.n_layers * trace.n_expert)
    return sum(counts[:k]) / sum(counts)


def units_for(trace, rows, share):
    """Fewest units whose activations reach `share` of the total."""
    counts = sorted(unit_counts(trace, rows).values(), reverse=True)
    want, got = share * sum(counts), 0
    for i, c in enumerate(counts):
        got += c
        if got >= want - 1e-9:
            return i + 1
    return len(counts)


def growth(trace, rows, marks):
    seen, out = set(), {}
    for t in range(trace.n_tokens):
        for layer in range(trace.n_layers):
            for e in rows[t][layer]:
                seen.add((layer, e))
        if t + 1 in marks:
            out[t + 1] = len(seen)
    return out


def layer_entropy(trace, rows, layer):
    c = Counter(e for t in range(trace.n_tokens) for e in rows[t][layer])
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values())


# ---- static ---------------------------------------------------------------------------------

def static_graph_share(trace, rows, k):
    """Co-occurrence counts expert(L) -> expert(L+1) from the first TRAIN_SHARE of the tokens;
    on the rest, the k experts of L+1 with the highest summed conditional frequency given the
    experts chosen at L. Returns the share of L+1's chosen experts among them."""
    split = int(TRAIN_SHARE * trace.n_tokens)
    E = trace.n_expert
    co = [[[0] * E for _ in range(E)] for _ in range(trace.n_layers - 1)]
    for t in range(split):
        for layer in range(trace.n_layers - 1):
            for a in rows[t][layer]:
                ca = co[layer][a]
                for b in rows[t][layer + 1]:
                    ca[b] += 1
    hit = total = 0
    for t in range(split, trace.n_tokens):
        for layer in range(trace.n_layers - 1):
            score = [0.0] * E
            for a in rows[t][layer]:
                ca = co[layer][a]
                s = sum(ca)
                if s:
                    for b in range(E):
                        score[b] += ca[b] / s
            top = set(sorted(range(E), key=lambda b: (-score[b], b))[:k])
            hit += len(top & set(rows[t][layer + 1]))
            total += len(rows[t][layer + 1])
    return hit / total if total else 0.0


def frequency_share(trace, rows, k):
    """The k most frequent experts of each layer in the first TRAIN_SHARE of the tokens, tried on
    the rest (layers 1.., like static_graph_share): what usage alone gives, with no path."""
    split = int(TRAIN_SHARE * trace.n_tokens)
    hit = total = 0
    for layer in range(1, trace.n_layers):
        c = Counter(e for t in range(split) for e in rows[t][layer])
        top = set(e for e, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k])
        for t in range(split, trace.n_tokens):
            hit += len(top & set(rows[t][layer]))
            total += len(rows[t][layer])
    return hit / total if total else 0.0


# ---- repeats --------------------------------------------------------------------------------

def set_repeats(trace, rows, layer):
    seen, again = set(), 0
    for t in range(trace.n_tokens):
        if rows[t][layer] in seen:
            again += 1
        seen.add(rows[t][layer])
    return again / trace.n_tokens


def distinct_paths(trace, rows):
    return len(set(tuple(rows[t]) for t in range(trace.n_tokens)))


def shared_with_previous(trace, rows, layer):
    if trace.n_tokens < 2:
        return 0.0
    return sum(len(set(rows[t][layer]) & set(rows[t - 1][layer]))
               for t in range(1, trace.n_tokens)) / (trace.n_tokens - 1)


# ---- table (version 2) ----------------------------------------------------------------------

def table_test(trace, rows, layer):
    """For every token whose id came before: is its set at `layer` the one that id had the last
    time? Returns (repeats, identical, mean Jaccard)."""
    last = {}
    repeats = identical = 0
    jac = 0.0
    for t in range(trace.n_tokens):
        tok = trace.tokens[t]
        if tok in last:
            a, b = set(rows[t][layer]), set(rows[last[tok]][layer])
            repeats += 1
            identical += a == b
            jac += len(a & b) / len(a | b)
        last[tok] = t
    return repeats, identical, (jac / repeats if repeats else 0.0)


# ---- margins (version 2) --------------------------------------------------------------------

def margin_stats(trace):
    gaps = sorted(trace.margins[i] - trace.margins[i + 1] for i in range(0, len(trace.margins), 2))
    n = len(gaps)
    median = gaps[n // 2] if n % 2 else 0.5 * (gaps[n // 2 - 1] + gaps[n // 2])
    return median, sum(g < 0.01 for g in gaps) / n, sum(g < 0.001 for g in gaps) / n


def margin_relative(trace):
    """The gap as a share of the last chosen expert's probability (the probabilities of 64 experts are
    small numbers, so the absolute gap says little): median, and share of choices under 5%."""
    rel = sorted((trace.margins[i] - trace.margins[i + 1]) / trace.margins[i]
                 for i in range(0, len(trace.margins), 2))
    n = len(rel)
    median = rel[n // 2] if n % 2 else 0.5 * (rel[n // 2 - 1] + rel[n // 2])
    return median, sum(r < 0.05 for r in rel) / n


# ---- compare, mask --------------------------------------------------------------------------

def hot_set(trace, rows, fraction):
    """The most used `fraction` of ALL units (ties: lower layer, then lower expert first); units
    never used are never hot, so the set can be smaller than the fraction."""
    k = int(fraction * trace.n_layers * trace.n_expert)
    ranked = sorted(unit_counts(trace, rows).items(), key=lambda kv: (-kv[1], kv[0]))
    return set(unit for unit, _ in ranked[:k])


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def inside(trace, rows, hot):
    """Share of this trace's activations that fall inside `hot`."""
    counts = unit_counts(trace, rows)
    return sum(c for unit, c in counts.items() if unit in hot) / sum(counts.values())


def mask_units(trace, rows, keep, seed=None):
    """Units to switch off: everything outside the most used `keep` fraction of all units, but
    every layer keeps at least 2 * n_used (its own most used), so the router always has a choice.
    With a seed, the control: the same number of units kept, drawn at random instead of by usage
    (if that does as well, usage says nothing about which experts the output needs)."""
    counts = unit_counts(trace, rows)
    kept = hot_set(trace, rows, keep)
    if seed is not None:
        units = [(layer, e) for layer in range(trace.n_layers) for e in range(trace.n_expert)]
        random.Random(seed).shuffle(units)
        rank = {unit: i for i, unit in enumerate(units)}
        counts = {unit: len(units) - i for unit, i in rank.items()}  # the draw, as a fake usage
        kept = set(units[:int(keep * len(units))])
    floor = min(trace.n_expert, 2 * trace.n_used)
    for layer in range(trace.n_layers):
        mine = sorted(range(trace.n_expert), key=lambda e: (-counts.get((layer, e), 0), e))
        for e in mine:
            if sum(1 for unit in kept if unit[0] == layer) >= floor:
                break
            kept.add((layer, e))
    return sorted((layer, e) for layer in range(trace.n_layers) for e in range(trace.n_expert)
                  if (layer, e) not in kept)


# ---- printing -------------------------------------------------------------------------------

def print_one(trace):
    rows = rows_of(trace)
    total = trace.n_layers * trace.n_expert
    used = len(unit_counts(trace, rows))
    print(f"trace: {trace.n_tokens} tokens ({trace.n_prompt} prompt), {trace.n_layers} layers, "
          f"{trace.n_expert} experts, {trace.n_used} used")
    print(f"small: {used} of {total} units used ({100.0 * used / total:.1f}%)")
    for f in COVER_FRACTIONS:
        print(f"  the most used {int(f * 100):3d}% of the units cover {100.0 * coverage(trace, rows, f):5.1f}% "
              f"of the activations")
    n99 = units_for(trace, rows, 0.99)
    print(f"  99% of the activations need {n99} units ({100.0 * n99 / total:.1f}% of the model)")
    marks = sorted(set(m for m in (100, 300, 600, trace.n_prompt, trace.n_tokens) if 0 < m <= trace.n_tokens))
    for m, n in growth(trace, rows, set(marks)).items():
        print(f"  after {m:5d} tokens: {n:5d} units seen ({100.0 * n / total:.0f}%)")
    ent = [layer_entropy(trace, rows, layer) for layer in range(trace.n_layers)]
    print(f"  usage entropy per layer, of {math.log2(trace.n_expert):.2f} bits: min {min(ent):.2f}  "
          f"mean {sum(ent) / len(ent):.2f}  max {max(ent):.2f}")
    print("static: share of the next layer's experts found, last quarter of the tokens")
    for k in K_VALUES:
        k = min(k, trace.n_expert)
        print(f"  top {k:2d}: co-occurrence graph {100.0 * static_graph_share(trace, rows, k):5.1f}%   "
              f"most frequent {100.0 * frequency_share(trace, rows, k):5.1f}%   "
              f"chance {100.0 * k / trace.n_expert:5.1f}%")
    rep = [set_repeats(trace, rows, layer) for layer in range(trace.n_layers)]
    sh = [shared_with_previous(trace, rows, layer) for layer in range(trace.n_layers)]
    print(f"repeats: a layer's set already seen: layer 0 {100.0 * rep[0]:.1f}%  mean "
          f"{100.0 * sum(rep) / len(rep):.1f}%  last {100.0 * rep[-1]:.1f}%;  whole paths: "
          f"{distinct_paths(trace, rows)} distinct of {trace.n_tokens}")
    print(f"  experts shared with the previous token, of {trace.n_used} (chance "
          f"{trace.n_used * trace.n_used / trace.n_expert:.2f}): layer 0 {sh[0]:.2f}  mean "
          f"{sum(sh) / len(sh):.2f}  max {max(sh):.2f}")
    if trace.tokens is None:
        print("table, margins: not in this trace (version 1: no token ids, no router margins)")
        return
    tt = [table_test(trace, rows, layer) for layer in range(trace.n_layers)]
    print(f"table: {tt[0][0]} tokens whose id came before; same set as the last time: layer 0 "
          f"{100.0 * tt[0][1] / max(1, tt[0][0]):.1f}%  mean "
          f"{100.0 * sum(x[1] for x in tt) / max(1, sum(x[0] for x in tt)):.1f}%  last "
          f"{100.0 * tt[-1][1] / max(1, tt[-1][0]):.1f}%;  mean Jaccard: layer 0 {tt[0][2]:.2f}  "
          f"mean {sum(x[2] for x in tt) / len(tt):.2f}  last {tt[-1][2]:.2f}")
    median, under_01, under_001 = margin_stats(trace)
    print(f"margins: last chosen minus first left out: median {median:.4f}; under 0.01: "
          f"{100.0 * under_01:.1f}%  under 0.001: {100.0 * under_001:.1f}%")
    rel_median, rel_close = margin_relative(trace)
    print(f"  as a share of the last chosen one's probability: median {100.0 * rel_median:.1f}%; "
          f"decided by less than 5%: {100.0 * rel_close:.1f}% of the choices")


def print_compare(paths):
    traces = [(os.path.basename(p), Trace(open(p, "rb").read())) for p in paths]
    rows = [rows_of(tr) for _, tr in traces]
    for f in HOT_FRACTIONS:
        print(f"hot set = the most used {int(f * 100)}% of all units; Jaccard, and the share of the "
              f"second trace's activations inside the first one's hot set")
        hots = [hot_set(tr, r, f) for (_, tr), r in zip(traces, rows)]
        for i, (name_a, _) in enumerate(traces):
            for j, (name_b, tr_b) in enumerate(traces):
                if i == j:
                    continue
                print(f"  {name_a:24s} {name_b:24s} Jaccard {jaccard(hots[i], hots[j]):.3f}   "
                      f"inside {100.0 * inside(tr_b, rows[j], hots[i]):5.1f}%")


# ---- --check --------------------------------------------------------------------------------

def run_check():
    """Handmade traces (version 2). A and B: 3 layers x 4 experts, 1 used, 4 tokens.

    A: chosen t0 [1,2,0]  t1 [0,3,1]  t2 [1,2,0]  t3 [0,3,1]; token ids 5 5 7 5.
       6 units, each used twice, 12 activations, 12 units in the model.
       coverage: 25% of 12 = 3 units -> 6/12 = 0.5; 50% = 6 units -> 1.0. 99% needs all 6.
       growth: 3 units after token 1, 6 after token 2 and after.
       entropy: every layer uses 2 experts equally -> 1 bit.
       repeats: t2's and t3's sets were seen -> 2/4 at every layer; 2 distinct paths;
         shared with the previous token: t1/t0 0, t2/t1 0, t3/t2 0 -> 0.
       static (train = first 3 tokens, test = t3, k = 1): L0->L1 counts 1->2 twice, 0->3 once;
         t3 has 0 at L0 -> predicts 3, right; L1->L2 counts 2->0 twice, 3->1 once; t3 has 3 at
         L1 -> predicts 1, right -> 2/2 = 1.0. Most frequent (layers 1, 2; train t0..t2): L1
         says 2, t3 has 3; L2 says 0, t3 has 1 -> 0/2 = 0.0.
       table: t1 (id 5, last at t0): no layer's set equal, Jaccard 0; t3 (id 5, last at t1, NOT
         the first at t0): t3's sets are t1's, Jaccard 1 -> 2 repeats, 1 identical, mean 0.5.
       margins: gaps 0.30 0.20 0.0005 0.10 (x3 layers the same) -> sorted per 12 values; median
         of [0.0005 x3, 0.10 x3, 0.20 x3, 0.30 x3] = (0.10 + 0.20) / 2 = 0.15; under 0.01: 3/12;
         under 0.001: 3/12.
    B: chosen t0 [2,2,2]  t1 [1,2,2]  t2 [1,2,2]  t3 [1,2,2]: units (0,2) x1, (0,1) x3, (1,2) x4,
       (2,2) x4. Hot 50%: B's 4 units and A's 6 share (0,1) and (1,2), union 8 -> Jaccard 0.25;
       B's ACTIVATIONS inside A's hot set: 3 + 4 of 12 -> 0.5833 (its units: 2 of 4, not that).
    C: 2 layers x 4 experts, 1 used, 12 tokens, for the two predictors: t0..t8 choose [e, e]
       with e = 1,2,3,1,2,3,1,2,3; t9..t11 (the test quarter) choose [1,0] [2,0] [3,0].
       static k=1: the graph learned e -> e says 1, 2, 3; the truth is 0 -> 0/3 (on the tokens it
       learned from it would be 3/3). Most frequent k=1, layer 1: 1, 2, 3 three times each in the
       first nine, the tie goes to 1 -> 0/3 (counting the test tokens too, 0 joins the tie and wins).
    mask from A, keep 0.5 (6 units = exactly the 6 used), floor 2 * n_used = 2 a layer, already
       met -> masked = the other 6: (0,2) (0,3) (1,0) (1,1) (2,2) (2,3).
       keep 0.25 (3 units: ties by (layer, expert): (0,0) (0,1) (1,2)) -> layer 1 needs one more,
       its most used left is 3; layer 2 needs two, 0 and 1 -> kept = the same 6 -> same mask."""
    F = 0xFFFF
    fields = {"n_tokens": 4, "n_prompt": 2, "n_layers": 3, "n_expert": 4, "n_used": 1, "n_pred": 2,
              "expert_bytes": 1048576, "layer_bytes": 5242880}
    pred = [F] * (4 * 3 * 2)

    def build(chosen, tokens, gaps):
        import struct
        data = bytearray(to_bytes(fields, chosen, pred, pred))
        data[:8] = MAGIC2
        data += struct.pack("<4i", *tokens)
        margins = []
        for g in gaps:
            for _ in range(3):
                margins += [0.5, 0.5 - g]
        data += struct.pack("<%df" % len(margins), *margins)
        return Trace(bytes(data))

    a = build([1, 2, 0, 0, 3, 1, 1, 2, 0, 0, 3, 1], [5, 5, 7, 5], [0.30, 0.20, 0.0005, 0.10])
    b = build([2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2], [5, 5, 7, 5], [0.30, 0.20, 0.0005, 0.10])
    ra, rb = rows_of(a), rows_of(b)
    c_fields = dict(fields, n_tokens=12, n_prompt=9, n_layers=2)
    c_chosen = [e for e in (1, 2, 3, 1, 2, 3, 1, 2, 3) for _ in range(2)] + [1, 0, 2, 0, 3, 0]
    c = Trace(to_bytes(c_fields, c_chosen, [F] * (12 * 2 * 2), [F] * (12 * 2 * 2)))
    rc = rows_of(c)
    failures = []

    def check(label, got, want, tol=1e-6):
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        if not ok:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    check("coverage 25%", coverage(a, ra, 0.25), 0.5)
    check("coverage 50%", coverage(a, ra, 0.50), 1.0)
    check("units for 99%", units_for(a, ra, 0.99), 6)
    check("growth", growth(a, ra, {1, 2, 4}), {1: 3, 2: 6, 4: 6})
    check("entropy layer 1", layer_entropy(a, ra, 1), 1.0)
    check("set repeats", set_repeats(a, ra, 2), 0.5)
    check("distinct paths", distinct_paths(a, ra), 2)
    check("shared with previous", shared_with_previous(a, ra, 0), 0.0)
    check("static graph k=1", static_graph_share(a, ra, 1), 1.0)
    check("most frequent k=1", frequency_share(a, ra, 1), 0.0)
    check("table layer 0", table_test(a, ra, 0), (2, 1, 0.5))
    check("table layer 2", table_test(a, ra, 2), (2, 1, 0.5))
    median, under_01, under_001 = margin_stats(a)
    check("margin median", median, 0.15)
    check("margin under 0.01", under_01, 0.25)
    check("margin under 0.001", under_001, 0.25)
    # relative gaps 0.6 0.4 0.001 0.2 (x3): median (0.2 + 0.4) / 2 = 0.3; under 5%: 3 of 12
    rel_median, rel_close = margin_relative(a)
    check("relative margin median", rel_median, 0.3)
    check("relative margin under 5%", rel_close, 0.25)
    check("jaccard hot 50%", jaccard(hot_set(a, ra, 0.5), hot_set(b, rb, 0.5)), 0.25)
    check("B inside A's hot 50%", inside(b, rb, hot_set(a, ra, 0.5)), 7.0 / 12.0)
    check("C static graph k=1", static_graph_share(c, rc, 1), 0.0)
    check("C most frequent k=1", frequency_share(c, rc, 1), 0.0)
    want_mask = [(0, 2), (0, 3), (1, 0), (1, 1), (2, 2), (2, 3)]
    check("mask keep 0.5", mask_units(a, ra, 0.5), want_mask)
    check("mask keep 0.25 (the floor fills in)", mask_units(a, ra, 0.25), want_mask)
    # the random control: 6 of 12 units drawn to stay, plus what a layer's floor of 2 adds, so at
    # most 6 off; the same seed gives the same mask; some seed gives a mask that is not by usage
    r1 = mask_units(a, ra, 0.5, seed=1)
    check("random mask size", 4 <= len(r1) <= 6, True)
    check("random mask floor", min(sum(1 for e in range(4) if (layer, e) not in r1) for layer in range(3)) >= 2, True)
    check("random mask repeats", mask_units(a, ra, 0.5, seed=1), r1)
    check("random mask is not the usage mask", any(mask_units(a, ra, 0.5, seed=k) != want_mask for k in range(5)), True)

    if failures:
        for f in failures:
            print("FAIL:", f, file=sys.stderr)
        print(f"route_graph_report.py --check: {len(failures)} failure(s)", file=sys.stderr)
        return False
    print("route_graph_report.py --check: ok")
    return True


def main(argv):
    args = argv[1:]
    if args == ["--check"]:
        return 0 if run_check() else 1
    if len(args) >= 3 and args[0] == "--compare":
        print_compare(args[1:])
        return 0
    seed = None
    if len(args) == 8 and args[0] == "--mask-from" and args[6] == "--random":
        seed = int(args[7])
        args = args[:6]
    if len(args) == 6 and args[0] == "--mask-from" and args[2] == "--keep" and args[4] == "--out":
        trace = Trace(open(args[1], "rb").read())
        keep = float(args[3])
        if not 0.0 < keep <= 1.0:
            print("route_graph_report.py: --keep is a fraction in (0, 1]", file=sys.stderr)
            return 2
        masked = mask_units(trace, rows_of(trace), keep, seed)
        with open(args[5], "w", encoding="ascii", newline="") as f:
            for layer, e in masked:
                f.write("%d %d%s" % (layer, e, chr(10)))
        total = trace.n_layers * trace.n_expert
        print(f"{len(masked)} of {total} units masked ({100.0 * len(masked) / total:.1f}%), "
              f"{total - len(masked)} kept -> {args[5]}")
        return 0
    if len(args) == 1 and not args[0].startswith("--"):
        print_one(Trace(open(args[0], "rb").read()))
        return 0
    print(__doc__.split("small ")[0].rstrip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
