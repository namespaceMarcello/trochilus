#!/usr/bin/env python3
"""route_trace_report.py — docs/MEASUREMENTS.md domande 13-15, from a trace written by
`trochilus run ... --route-trace <file>` (src/models/model.h: tr_route_trace).

Usage:
    route_trace_report.py <trace.bin> [--disk-mib-s 1430] [--compute-ms 33]
                                        prints the tables below; the two numbers feed the time
                                        model only (the disk of tests/bench_disk.c, a decode token)
    route_trace_report.py --check       a handmade trace, hand-computed answers (see run_check);
                                         exit 1 on any mismatch (wired into `make lint`)

File format (little-endian): 8 bytes "TRROUTE1"; int64 x 8 (n_tokens, n_prompt, n_layers,
n_expert, n_used, n_pred, expert_bytes, layer_bytes); then chosen [n_tokens][n_layers][n_used],
pred_in and pred_out [n_tokens][n_layers][n_pred], all uint16 (0xFFFF: no prediction, the last
layer). No dependencies beyond the standard library.

Q13: does layer L+1's router already know, from layer L's own state, which experts it will
     pick? Share of chosen[t][L+1] found in the predicted top 8/12/16 of pred_in[t][L] and
     pred_out[t][L] (best-first, so "top K" is just the first K entries).
Q14: with an LRU cache of (layer, expert) units sized 25/50/75% of the model, how many misses
     (and MiB, at expert_bytes each) does a generated token cost? Simulated over the whole
     trace in order, counted from the generated tokens only (n_prompt..n_tokens). A static pin
     (the units used most during the prompt, fixed for the rest of the run) is measured the
     same way, for comparison.
     Then the same LRU with a prefetch of pred_in's first k units of the next layer: stalls,
     reads and wasted reads per generated token; and a MODEL on a clock (one disk, one read at a
     time) of what that means in tokens per second. A model, never a measurement.
Q15: for comparison, streaming whole layers instead of experts: every token touches every
     layer, so caching 25/50/75% of the layers still streams the rest, at layer_bytes each.
"""
import struct
import sys
from array import array
from collections import Counter, OrderedDict

MAGIC = b"TRROUTE1"
MAGIC2 = b"TRROUTE2"  # the same, then token ids and router margins (tools/route_graph_report.py)
HEADER_FIELDS = ("n_tokens", "n_prompt", "n_layers", "n_expert", "n_used", "n_pred",
                  "expert_bytes", "layer_bytes")
K_VALUES = (8, 12, 16)          # docs/MEASUREMENTS.md domanda 13
CACHE_FRACTIONS = (0.25, 0.50, 0.75)  # domande 14-15
MIB = 1024.0 * 1024.0


class Trace:
    """Parses the bytes of a route trace file (or a handmade one, for --check)."""

    def __init__(self, data):
        if data[:8] not in (MAGIC, MAGIC2):
            raise ValueError("not a route trace (bad magic)")
        header = struct.unpack_from("<8q", data, 8)
        for name, value in zip(HEADER_FIELDS, header):
            setattr(self, name, value)
        off = 8 + 8 * 8
        n_chosen = self.n_tokens * self.n_layers * self.n_used
        n_pred = self.n_tokens * self.n_layers * self.n_pred

        def take(n):
            nonlocal off
            a = array("H")
            a.frombytes(data[off:off + n * 2])
            off += n * 2
            return a

        self.chosen = take(n_chosen)
        self.pred_in = take(n_pred)
        self.pred_out = take(n_pred)
        # version 2: the id of every token (int32) and, per token and layer, the router
        # probability of the last expert chosen and of the first one left out (float32 pairs)
        self.tokens = self.margins = None
        if data[:8] == MAGIC2:
            self.tokens = array("i")
            self.tokens.frombytes(data[off:off + self.n_tokens * 4])
            off += self.n_tokens * 4
            self.margins = array("f")
            self.margins.frombytes(data[off:off + self.n_tokens * self.n_layers * 2 * 4])

    def chosen_row(self, t, L):
        i = (t * self.n_layers + L) * self.n_used
        return self.chosen[i:i + self.n_used]

    def pred_row(self, which, t, L):
        arr = self.pred_in if which == "in" else self.pred_out
        i = (t * self.n_layers + L) * self.n_pred
        return arr[i:i + self.n_pred]


def to_bytes(trace_fields, chosen, pred_in, pred_out):
    """The inverse of Trace(): builds the file bytes from plain lists, for run_check()."""
    header = tuple(trace_fields[name] for name in HEADER_FIELDS)
    out = bytearray(MAGIC)
    out += struct.pack("<8q", *header)
    for values in (chosen, pred_in, pred_out):
        out += array("H", values).tobytes()
    return bytes(out)


# ---- Q13: does the next layer's router already know? ---------------------------------------

def q13_layer_shares(trace, which, layer, tokens):
    """{k: (hits, total)} of chosen[t][layer+1] found in the top-k of pred_{which}[t][layer],
    over `tokens`, for every k in K_VALUES (clipped to n_pred, the most this trace stored)."""
    out = {}
    for k in K_VALUES:
        kk = min(k, trace.n_pred)
        hits = total = 0
        for t in tokens:
            topk = set(trace.pred_row(which, t, layer)[:kk])
            for e in trace.chosen_row(t, layer + 1):
                total += 1
                if e in topk:
                    hits += 1
        out[k] = (hits, total)
    return out


def q13_combine(trace, which, tokens):
    """Sums q13_layer_shares over every layer with a next one, for the given tokens."""
    totals = {k: [0, 0] for k in K_VALUES}
    for layer in range(trace.n_layers - 1):
        for k, (hits, total) in q13_layer_shares(trace, which, layer, tokens).items():
            totals[k][0] += hits
            totals[k][1] += total
    return {k: tuple(v) for k, v in totals.items()}


def q13_worst_layer(trace, which):
    """The layer (of n_layers - 1 candidates) with the lowest all-tokens share at the
    strictest column (top 8, the fewest predictions): the one a top-8 preloader would miss
    the most. Returns (layer, {k: (hits, total)}) for that layer, over all tokens."""
    all_tokens = range(trace.n_tokens)
    best_layer, best_share, best_shares = None, None, None
    for layer in range(trace.n_layers - 1):
        shares = q13_layer_shares(trace, which, layer, all_tokens)
        hits, total = shares[K_VALUES[0]]
        share = hits / total if total else 1.0
        if best_share is None or share < best_share:
            best_layer, best_share, best_shares = layer, share, shares
    return best_layer, best_shares


def share_str(hits, total):
    return f"{100.0 * hits / total:5.1f}%" if total else "  n/a"


def print_q13(trace):
    print("Q13: share of layer L+1's chosen experts found in the predicted top-K")
    print(f"{'':16s}{'pred_in':>24s}{'pred_out':>24s}")
    print(f"{'row':16s}" + "".join(f"{'top' + str(k):>8s}" for k in K_VALUES) * 2)
    all_tokens, prompt, generated = (range(trace.n_tokens), range(min(trace.n_prompt, trace.n_tokens)),
                                      range(trace.n_prompt, trace.n_tokens))
    rows = [("all tokens", all_tokens), ("prompt only", prompt), ("generated only", generated)]
    for label, tokens in rows:
        cells = []
        for which in ("in", "out"):
            shares = q13_combine(trace, which, tokens)
            cells.append("".join(f"{share_str(*shares[k]):>8s}" for k in K_VALUES))
        print(f"{label:16s}" + "".join(cells))
    worst_in_layer, worst_in = q13_worst_layer(trace, "in")
    worst_out_layer, worst_out = q13_worst_layer(trace, "out")
    label = "worst layer"
    cells = "".join(f"{share_str(*worst_in[k]):>8s}" for k in K_VALUES) + \
            "".join(f"{share_str(*worst_out[k]):>8s}" for k in K_VALUES)
    print(f"{label:16s}" + cells)
    print(f"  (worst layer: pred_in L={worst_in_layer}, pred_out L={worst_out_layer}, by top-{K_VALUES[0]})")


# ---- Q14: LRU cache of (layer, expert) units, and a static pin from the prompt --------------

def lru_misses(trace, capacity):
    """LRU over the whole trace in order; misses and tokens counted from the generated tokens
    (t >= n_prompt) only. capacity is a count of (layer, expert) units."""
    cache = OrderedDict()
    misses = tokens = 0
    for t in range(trace.n_tokens):
        generated = t >= trace.n_prompt
        if generated:
            tokens += 1
        for layer in range(trace.n_layers):
            for e in trace.chosen_row(t, layer):
                unit = (layer, e)
                if unit in cache:
                    cache.move_to_end(unit)
                    continue
                if generated:
                    misses += 1
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                cache[unit] = True
    return misses, tokens


def static_pin_misses(trace, capacity):
    """The `capacity` units used most during the prompt, pinned and never evicted; every other
    unit is always a miss. Misses and tokens counted from the generated tokens only."""
    freq = Counter()
    for t in range(min(trace.n_prompt, trace.n_tokens)):
        for layer in range(trace.n_layers):
            for e in trace.chosen_row(t, layer):
                freq[(layer, e)] += 1
    pinned = {unit for unit, _ in freq.most_common(capacity)}
    misses = tokens = 0
    for t in range(trace.n_prompt, trace.n_tokens):
        tokens += 1
        for layer in range(trace.n_layers):
            for e in trace.chosen_row(t, layer):
                if (layer, e) not in pinned:
                    misses += 1
    return misses, tokens


def lru_prefetch(trace, capacity, k):
    """The same LRU with a prefetch: after layer L's experts are known, the first k of pred_in
    (layer L+1's router on layer L's FFN input, known before L's experts run) are read if absent
    and touched if present. Layer 0 has no prediction. Counted on the generated tokens only:
    stalls (a chosen unit absent when its layer runs: compute waits for the disk), stalls at
    layer 0, reads (stalls + prefetch reads: what the disk does) and wasted reads (a prefetched
    unit its token did not choose at that layer). Returns (stalls, stalls0, reads, wasted, tokens)."""
    cache = OrderedDict()

    def load(unit):
        if len(cache) >= capacity:
            cache.popitem(last=False)
        cache[unit] = True

    stalls = stalls0 = reads = wasted = tokens = 0
    for t in range(trace.n_tokens):
        generated = t >= trace.n_prompt
        if generated:
            tokens += 1
        for layer in range(trace.n_layers):
            for e in trace.chosen_row(t, layer):
                unit = (layer, e)
                if unit in cache:
                    cache.move_to_end(unit)
                    continue
                if generated:
                    stalls += 1
                    reads += 1
                    stalls0 += layer == 0
                load(unit)
            if layer + 1 == trace.n_layers:
                continue
            coming = set(trace.chosen_row(t, layer + 1))
            for e in trace.pred_row("in", t, layer)[:k]:
                unit = (layer + 1, e)
                if unit in cache:
                    cache.move_to_end(unit)
                    continue
                if generated:
                    reads += 1
                    wasted += e not in coming
                load(unit)
    return stalls, stalls0, reads, wasted, tokens


def time_model(trace, capacity, k, read_ms, layer_ms):
    """A MODEL of the generated tokens' speed, not a measurement: the cache and prefetch of
    lru_prefetch on a clock. One disk, one read at a time, read_ms per unit, first come first
    served with a layer's own missing units queued before its guesses for the next layer. A layer
    takes layer_ms of compute: half before its experts are known (attention, router), half after
    every chosen unit has arrived. Returns tokens per second over the generated tokens."""
    cache = OrderedDict()  # unit -> the time its bytes are all in memory
    now = disk_free = 0.0
    total_ms = 0.0
    tokens = 0

    def read(unit, asked_at):
        nonlocal disk_free
        if len(cache) >= capacity:
            cache.popitem(last=False)
        disk_free = max(disk_free, asked_at) + read_ms
        cache[unit] = disk_free

    for t in range(trace.n_tokens):
        if t == trace.n_prompt:  # the clock starts with the first generated token
            now = disk_free = 0.0
            for unit in cache:
                cache[unit] = 0.0
        start = now
        for layer in range(trace.n_layers):
            known = now + layer_ms / 2
            ready = known
            for e in trace.chosen_row(t, layer):
                unit = (layer, e)
                if unit in cache:
                    cache.move_to_end(unit)
                else:
                    read(unit, known)
                ready = max(ready, cache[unit])
            if layer + 1 < trace.n_layers:
                for e in trace.pred_row("in", t, layer)[:k]:
                    unit = (layer + 1, e)
                    if unit in cache:
                        cache.move_to_end(unit)
                    else:
                        read(unit, known)
            now = ready + layer_ms / 2
        if t >= trace.n_prompt:
            total_ms += now - start
            tokens += 1
    return tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0


def per_token(misses, tokens):
    return misses / tokens if tokens else 0.0


def print_time_model(trace, disk_mib_s, compute_ms):
    read_ms = trace.expert_bytes / MIB / disk_mib_s * 1000.0
    layer_ms = compute_ms / trace.n_layers
    print(f"MODEL, not a measurement: tokens per second with one disk at {disk_mib_s:.0f} MiB/s "
          f"({read_ms:.2f} ms a unit) and {compute_ms:.0f} ms of compute a token")
    total_units = trace.n_layers * trace.n_expert
    ks = (0,) + tuple(min(k, trace.n_pred) for k in K_VALUES)
    print(f"{'capacity':16s}" + "".join(f"{f'k={k}':>10s}" for k in ks))
    for frac in CACHE_FRACTIONS:
        cap = max(1, round(frac * total_units))
        print(f"{f'{int(frac * 100)}% ({cap}/{total_units})':16s}"
              + "".join(f"{time_model(trace, cap, k, read_ms, layer_ms):10.2f}" for k in ks))


def print_q14_prefetch(trace):
    print("Q14 with a prefetch of pred_in's first k -- per generated token")
    total_units = trace.n_layers * trace.n_expert
    print(f"{'capacity':16s}{'k':>4s}{'stalls':>10s}{'at layer 0':>12s}{'reads':>10s}{'wasted':>10s}{'MiB read':>10s}")
    for frac in CACHE_FRACTIONS:
        cap = max(1, round(frac * total_units))
        for k in (0,) + tuple(K_VALUES):
            k = min(k, trace.n_pred)
            stalls, stalls0, reads, wasted, tokens = lru_prefetch(trace, cap, k)
            print(f"{f'{int(frac * 100)}% ({cap}/{total_units})':16s}{k:4d}{per_token(stalls, tokens):10.2f}"
                  f"{per_token(stalls0, tokens):12.2f}{per_token(reads, tokens):10.2f}"
                  f"{per_token(wasted, tokens):10.2f}{per_token(reads, tokens) * trace.expert_bytes / MIB:10.2f}")


def print_q14(trace):
    print("Q14: LRU cache of (layer, expert) units -- misses and MiB per generated token")
    total_units = trace.n_layers * trace.n_expert
    print(f"{'capacity':16s}{'LRU misses':>12s}{'LRU MiB':>12s}{'pin misses':>12s}{'pin MiB':>12s}")
    for frac in CACHE_FRACTIONS:
        cap = max(1, round(frac * total_units))
        lru_m, lru_t = lru_misses(trace, cap)
        pin_m, pin_t = static_pin_misses(trace, cap)
        lru_pt, pin_pt = per_token(lru_m, lru_t), per_token(pin_m, pin_t)
        print(f"{f'{int(frac * 100)}% ({cap}/{total_units})':16s}"
              f"{lru_pt:12.2f}{lru_pt * trace.expert_bytes / MIB:12.2f}"
              f"{pin_pt:12.2f}{pin_pt * trace.expert_bytes / MIB:12.2f}")


# ---- Q15: streaming whole layers instead of experts -----------------------------------------

def print_q15(trace):
    print("Q15: streaming whole layers instead of experts -- MiB per generated token")
    print(f"{'cached layers':20s}{'MiB/token':>12s}{'  (Q14 LRU MiB, same %)':>26s}")
    total_units = trace.n_layers * trace.n_expert
    for frac in CACHE_FRACTIONS:
        cached = min(trace.n_layers, max(0, round(frac * trace.n_layers)))
        streamed = trace.n_layers - cached
        mib = streamed * trace.layer_bytes / MIB
        cap = max(1, round(frac * total_units))
        lru_m, lru_t = lru_misses(trace, cap)
        lru_mib = per_token(lru_m, lru_t) * trace.expert_bytes / MIB
        print(f"{f'{int(frac * 100)}% ({cached}/{trace.n_layers})':20s}{mib:12.2f}{lru_mib:26.2f}")


# ---- --check: a handmade trace with answers computed by hand in this file's comments --------

def run_check():
    """A tiny handmade trace (3 layers, 4 experts, 1 used, 2 predicted, 4 tokens: 2 prompt + 2
    generated, chosen repeating token 0/1's units at tokens 2/3) with every Q13-Q15 number
    computed by hand (see the comments below); asserts the report functions reproduce them.

    chosen (t -> [L0, L1, L2]): 0:[1,2,0] 1:[0,3,1] 2:[1,2,0] (= token 0) 3:[0,3,1] (= token 1)

    pred_in (t -> {L: [best, second]}), targets are chosen[t][L+1]:
      L0 targets 2,3,2,3: t0=[2,1] hit, t1=[0,1] miss, t2=[3,2] hit, t3=[1,2] miss -> 2/4
      L1 targets 0,1,0,1: t0=[0,3] hit, t1=[2,3] miss, t2=[1,0] hit, t3=[1,2] hit  -> 3/4
      all = 5/8 = 0.625; prompt (t0,t1) = 2/4 = 0.5; generated (t2,t3) = 3/4 = 0.75
      worst layer (lowest all-tokens share) = L0 (0.5 < 0.75)

    pred_out (t -> {L: [best, second]}):
      L0 targets 2,3,2,3: t0=[2,3] hit, t1=[2,3] hit, t2=[2,3] hit, t3=[2,3] hit -> 4/4
      L1 targets 0,1,0,1: t0=[2,3] miss, t1=[2,3] miss, t2=[0,3] hit, t3=[0,3] miss -> 1/4
      all = 5/8 = 0.625; prompt = 2/4 = 0.5; generated = 3/4 = 0.75
      worst layer = L1 (0.25 < 1.0) -- a different layer than pred_in's, on purpose

    Q14, 12 units total (3 layers x 4 experts), capacities 3/6/9 (25/50/75%):
      Access order: (0,1)(1,2)(2,0) (0,0)(1,3)(2,1) (0,1)(1,2)(2,0) (0,0)(1,3)(2,1) -- token 2
      repeats token 0's units, token 3 repeats token 1's, so with room for both (capacity 6 or
      9, only 6 distinct units ever appear) every generated access is a cache hit: 0 misses.
      With room for only one token's worth (capacity 3), the other token's units always just
      got evicted: all 6 generated accesses miss -> 3.0 misses/token (2 generated tokens).
      Static pin at capacity 3: ties in prompt frequency (each of the 6 units seen once) break
      by first-seen order (Counter.most_common, documented behavior), pinning token 0's 3
      units. Token 2 (= token 0) then hits all 3; token 3 (= token 1, not pinned) misses all
      3: 3 misses / 2 generated tokens = 1.5. Capacity 6 or 9 pins every unit: 0 misses.

    Q15, 3 layers, 25/50/75% -> cached 1/2/2 layers (round(0.75)=1, round(1.5)=2, round(2.25)=2),
      streamed 2/1/1 layers; with layer_bytes = 5 MiB: 10.0 / 5.0 / 5.0 MiB per token.
    """
    chosen = [1, 2, 0, 0, 3, 1, 1, 2, 0, 0, 3, 1]
    F = 0xFFFF
    pred_in = [2, 1, 0, 3, F, F,
               0, 1, 2, 3, F, F,
               3, 2, 1, 0, F, F,
               1, 2, 1, 2, F, F]
    pred_out = [2, 3, 2, 3, F, F,
                2, 3, 2, 3, F, F,
                2, 3, 0, 3, F, F,
                2, 3, 0, 3, F, F]
    fields = {"n_tokens": 4, "n_prompt": 2, "n_layers": 3, "n_expert": 4, "n_used": 1, "n_pred": 2,
              "expert_bytes": 1048576, "layer_bytes": 5242880}
    trace = Trace(to_bytes(fields, chosen, pred_in, pred_out))

    failures = []

    def check(label, got, want, tol=1e-9):
        if isinstance(want, float):
            ok = abs(got - want) <= tol
        else:
            ok = got == want
        if not ok:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    for which, want_all, want_prompt, want_gen, want_layer, want_layer_share in (
        ("in", 0.625, 0.5, 0.75, 0, 0.5),
        ("out", 0.625, 0.5, 0.75, 1, 0.25),
    ):
        all_tokens = range(trace.n_tokens)
        prompt = range(trace.n_prompt)
        gen = range(trace.n_prompt, trace.n_tokens)
        for label, tokens, want in (("all", all_tokens, want_all), ("prompt", prompt, want_prompt),
                                     ("generated", gen, want_gen)):
            hits, total = q13_combine(trace, which, tokens)[K_VALUES[0]]
            check(f"Q13 pred_{which} {label} share", hits / total, want)
        layer, shares = q13_worst_layer(trace, which)
        check(f"Q13 pred_{which} worst layer", layer, want_layer)
        wh, wt = shares[K_VALUES[0]]
        check(f"Q13 pred_{which} worst layer share", wh / wt, want_layer_share)

    total_units = trace.n_layers * trace.n_expert
    check("Q14 total units", total_units, 12)
    for frac, want_lru, want_pin in ((0.25, 3.0, 1.5), (0.50, 0.0, 0.0), (0.75, 0.0, 0.0)):
        cap = max(1, round(frac * total_units))
        lru_m, lru_t = lru_misses(trace, cap)
        pin_m, pin_t = static_pin_misses(trace, cap)
        check(f"Q14 LRU {int(frac * 100)}% misses/token", per_token(lru_m, lru_t), want_lru)
        check(f"Q14 pin {int(frac * 100)}% misses/token", per_token(pin_m, pin_t), want_pin)

    # Q14 with a prefetch, followed by hand through the 12 demand steps and 8 prefetch steps:
    #   capacity 3, k = 1: a cache this small is polluted by every guess. Generated tokens:
    #     t2: (0,1) stall; prefetch (1,3) read, wasted; (1,2) stall; prefetch (2,1) read, wasted;
    #         (2,0) stall.  t3: (0,0) stall; prefetch (1,1) read, wasted; (1,3) stall; prefetch
    #         (2,1) read, used; (2,1) hit.  -> 5 stalls (2 at layer 0), 9 reads, 3 wasted.
    #   capacity 6, k = 2: without a prefetch this cache never misses (above); with it the guesses
    #     evict what the next token needs. t2: (0,1) stall; prefetch (1,2) read, used; (2,0) read,
    #     used.  t3: (0,0) stall; prefetch (1,1) read, wasted; (2,2) read, wasted.
    #     -> 2 stalls (both at layer 0), 6 reads, 2 wasted.
    #   k = 0 is lru_misses: capacity 3 -> 6 stalls, 6 reads, 0 wasted.
    check("Q14 prefetch cap 3 k 1", lru_prefetch(trace, 3, 1), (5, 2, 9, 3, 2))
    check("Q14 prefetch cap 6 k 2", lru_prefetch(trace, 6, 2), (2, 2, 6, 2, 2))
    check("Q14 prefetch cap 3 k 0", lru_prefetch(trace, 3, 0), (6, 2, 6, 0, 2))

    # The time model with a read of 10 ms and a layer of 2 ms (1 before the experts are known, 1
    # after they are all there), generated tokens only, clock at 0 when the first one starts:
    #   capacity 6, k 0: every unit is there: 3 layers x 2 ms = 6 ms a token -> 1000/6 tok/s
    #   capacity 3, k 0: every unit is read when asked: a layer is 1 + 10 + 1 = 12 ms, a token 36
    #   capacity 3, k 1: the guesses take the disk before the units that are needed. t2: layer 0
    #     asks at 1, read ends 11, the guess (1,3) holds the disk to 21, layer ends 12; layer 1 asks
    #     at 13, read 21-31, guess to 41, ends 32; layer 2 asks at 33, read 41-51, ends 52. t3:
    #     asks 53, read 53-63, guess to 73, ends 64; asks 65, read 73-83, guess (2,1) to 93, ends
    #     84; layer 2 needs (2,1), in flight until 93, ends 94. Two tokens in 94 ms.
    check("time model cap 6 k 0", time_model(trace, 6, 0, 10.0, 2.0), 1000.0 / 6.0, tol=1e-6)
    check("time model cap 3 k 0", time_model(trace, 3, 0, 10.0, 2.0), 1000.0 / 36.0, tol=1e-6)
    check("time model cap 3 k 1", time_model(trace, 3, 1, 10.0, 2.0), 2000.0 / 94.0, tol=1e-6)

    for frac, want_mib in ((0.25, 10.0), (0.50, 5.0), (0.75, 5.0)):
        cached = min(trace.n_layers, max(0, round(frac * trace.n_layers)))
        streamed = trace.n_layers - cached
        check(f"Q15 {int(frac * 100)}% MiB/token", streamed * trace.layer_bytes / MIB, want_mib)

    if failures:
        for f in failures:
            print("FAIL:", f, file=sys.stderr)
        print(f"route_trace_report.py --check: {len(failures)} failure(s)", file=sys.stderr)
        return False
    print("route_trace_report.py --check: ok")
    return True


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        return 0 if run_check() else 1
    disk_mib_s, compute_ms = 1430.0, 33.0
    args = argv[1:]
    try:
        while len(args) >= 3 and args[1] in ("--disk-mib-s", "--compute-ms"):
            if args[1] == "--disk-mib-s":
                disk_mib_s = float(args[2])
            else:
                compute_ms = float(args[2])
            del args[1:3]
    except ValueError:
        args = []
    if len(args) != 1 or disk_mib_s <= 0 or compute_ms <= 0:
        print("usage: route_trace_report.py <trace.bin> [--disk-mib-s 1430] [--compute-ms 33] | --check",
              file=sys.stderr)
        return 2
    with open(args[0], "rb") as f:
        data = f.read()
    trace = Trace(data)
    print_q13(trace)
    print()
    print_q14(trace)
    print()
    print_q14_prefetch(trace)
    print()
    print_time_model(trace, disk_mib_s, compute_ms)
    print()
    print_q15(trace)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
