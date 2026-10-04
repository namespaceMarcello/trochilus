#!/usr/bin/env python3
"""evict_replay.py -- the expert store's eviction, replayed on a route trace (docs/MEASUREMENTS.md §The three
machines; LESSONS #271: under one token's units the store's LRU hits nothing).

Usage:
    evict_replay.py <trace.bin> [--slots 72,83,128,...] [--disk-mbs 500] [--compute-ms 40] [--seeds 5]
    evict_replay.py --check      hand-made sequences with hand-computed answers (wired into `make lint`)

The trace: `trochilus run ... --route-trace <file>` (src/app/main.c write_route_trace; TRROUTE1 or 2). Only the
chosen experts are read: chosen[t][layer][n_used], increasing ids.

What is replayed is the store's own access sequence (src/models/olmoe.c olmoe_refresh_experts,
src/memory/experts.c tr_experts_acquire): a prompt in passes of 512 tokens (a trace turns the layer-major path
off), each pass asking every layer once for the union of its tokens' experts, in increasing id; then every
generated token asking every layer for its own n_used. A unit is (layer, expert), one slot each. Inside one call
the units it names are never its own victims (the store's rule). Hits and misses are counted on the generated
tokens only (the decode), and the prompt's misses apart.

The policies (a victim among the slots this call did not name):
    lru       the least recently used: the store today
    mru       the most recently used: under a cycle longer than the store, keeps what came first
    random    any, uniformly (seeded, the mean of --seeds runs): a page cache's rough recency, as an mmap gets
    layer     the least recently used of the layer holding the most slots: every layer keeps its share
    lfu       the fewest uses so far (the prompt's included), the least recent among equals: keeps the
              experts each layer's router likes
    opt       the one used again furthest in the future (Belady): no policy can hit more; the ceiling
    colibri   colibri's (ref/colibri c/olmoe.c:696-725): every layer its own LRU of slots // layers units;
              the units of a call beyond that are read into its working set and not kept
    ds4       ds4's streaming cache (ref/ds4 ds4_metal.m:15404-15439, 14272-14346): the lowest hotness, the
              oldest among equals; hotness counts every routing, hit or miss (a prefill batch every token row
              that chose the unit, 17437-17447 and 14359-14362), halved every 16 decode tokens (none before
              the first, 14292-14306)
    once      the engine of 10-03: ds4's, but a prompt's pass adding 1 to each unit it names (LESSONS #289)
    heat      the store's default (src/memory/experts.c tr_experts_acquire_counts): ds4's, a call adding to
              each unit the routings of its tokens that chose it, as if at most HOT_PROMPT tokens: c x
              min(1, 64 / n_tok), at least 1 (a decode token 1; a prompt pass its tokens' share, scaled to 64)
The time model (a model, never a measurement): a decode token's time = --compute-ms + its misses' bytes at
--disk-mbs; tok/s = 1000 / that.
"""
import heapq
import random
import struct
import sys
from collections import OrderedDict

PASS = 512  # OLMOE_DEFAULT_BATCH: the prompt's pass with a trace on
HOT_PROMPT = 64  # TR_EXPERTS_HOT_PROMPT: a call's routings count as if from at most this many tokens


def read_trace(path):
    with open(path, "rb") as f:
        data = f.read()
    magic = data[:8]
    if magic not in (b"TRROUTE1", b"TRROUTE2"):
        sys.exit(f"{path}: not a route trace ({magic!r})")
    n_tokens, n_prompt, n_layers, n_expert, n_used, n_pred, expert_bytes, layer_bytes = struct.unpack_from("<8q", data, 8)
    off = 8 + 64
    n = n_tokens * n_layers * n_used
    flat = struct.unpack_from(f"<{n}H", data, off)
    chosen = [[list(flat[(t * n_layers + L) * n_used:(t * n_layers + L + 1) * n_used]) for L in range(n_layers)]
              for t in range(n_tokens)]
    return dict(n_tokens=n_tokens, n_prompt=n_prompt, n_layers=n_layers, n_expert=n_expert, n_used=n_used,
                expert_bytes=expert_bytes, chosen=chosen)


def calls_of(tr):
    """[(token or -1 for the prompt, [units])]: the store's calls in order."""
    calls = []
    nl, ne, np_ = tr["n_layers"], tr["n_expert"], tr["n_prompt"]
    ch = tr["chosen"]
    for p0 in range(0, np_, PASS):
        toks = range(p0, min(np_, p0 + PASS))
        for L in range(nl):
            ids = sorted({e for t in toks for e in ch[t][L]})
            calls.append((-1, [L * ne + e for e in ids]))
    for t in range(np_, tr["n_tokens"]):
        for L in range(nl):
            calls.append((t, [L * ne + e for e in sorted(ch[t][L])]))
    return calls


def heat_add(count, n_tok):
    """What a call adds to a unit's hotness (tr_experts_acquire_counts): the routings of its n_tok tokens that
    chose it, as if from at most HOT_PROMPT tokens, rounded half up, at least 1."""
    if n_tok <= HOT_PROMPT:
        return count
    return max(1, (count * HOT_PROMPT + n_tok // 2) // n_tok)


def heats_of(tr, scaled=True):
    """Each call of calls_of(tr), what it adds to each of its units' hotness: heat_add, or (scaled False,
    ds4's) every routing whole."""
    heats = []
    nl, ne, np_ = tr["n_layers"], tr["n_expert"], tr["n_prompt"]
    ch = tr["chosen"]
    for p0 in range(0, np_, PASS):
        toks = range(p0, min(np_, p0 + PASS))
        for L in range(nl):
            c = {}
            for t in toks:
                for e in ch[t][L]:
                    c[e] = c.get(e, 0) + 1
            heats.append([heat_add(c[e], len(toks)) if scaled else c[e] for e in sorted(c)])
    for t in range(np_, tr["n_tokens"]):
        for L in range(nl):
            heats.append([1] * len(ch[t][L]))
    return heats


def replay_colibri(calls, slots, n_expert, n_layers):
    """colibri: per-layer LRU caches of slots // n_layers units, the rest of a call not kept."""
    cap = slots // n_layers
    cache = {}
    hits = misses = prompt_misses = 0
    for tok, units in calls:
        od = cache.setdefault(units[0] // n_expert, OrderedDict())
        for u in units:
            if u in od:
                if tok >= 0:
                    hits += 1
            elif tok >= 0:
                misses += 1
            else:
                prompt_misses += 1
        for u in units:
            od.pop(u, None)
            od[u] = None
        while len(od) > cap:
            od.popitem(last=False)
    return hits, misses, prompt_misses


def replay_ds4(calls, slots, n_expert, heats=None):
    """ds4: the lowest hotness out (the oldest among equals), this call's units never; hotness +1 a routing
    (heats: + heats[call][k] for the call's k-th unit), halved every 16 decode tokens."""
    present, hot, last = set(), {}, {}
    hits = misses = prompt_misses = 0
    tok_seen = 0
    prev_tok = None
    for i, (tok, units) in enumerate(calls):
        if tok >= 0 and tok != prev_tok:
            tok_seen += 1
            prev_tok = tok
            if tok_seen % 16 == 0:
                for u in hot:
                    hot[u] //= 2
        protect = set(units)
        for k, u in enumerate(units):
            hot[u] = hot.get(u, 0) + (heats[i][k] if heats is not None else 1)
            if u in present:
                if tok >= 0:
                    hits += 1
            else:
                if tok >= 0:
                    misses += 1
                else:
                    prompt_misses += 1
                if len(present) >= slots:
                    v = min((p for p in present if p not in protect), key=lambda p: (hot[p], last[p]))
                    present.discard(v)
                present.add(u)
            last[u] = i
    return hits, misses, prompt_misses


def replay(calls, slots, policy, n_expert, seed=0, n_layers=16, heats=None):
    """(decode hits, decode misses, prompt misses) of one policy at `slots`; ds4 and heat add `heats` (heats_of
    unscaled and scaled; None: 1 a routing, a call of one token's)."""
    if policy == "colibri":
        return replay_colibri(calls, slots, n_expert, n_layers)
    if policy == "once":
        return replay_ds4(calls, slots, n_expert)
    if policy in ("ds4", "heat"):
        return replay_ds4(calls, slots, n_expert, heats)
    rng = random.Random(seed)
    lru = OrderedDict()               # unit -> None, oldest first (every policy keeps recency)
    present = set()
    uses = {}                         # unit -> uses so far (lfu)
    last = {}                         # unit -> time of its last use
    by_layer = {}                     # layer -> OrderedDict of its units, oldest first (layer)
    rand_list, rand_pos = [], {}      # random: the present units, and where each sits
    # opt: every unit's call indices, and a heap of (-next use, unit) with lazy deletion
    nxt_list, nxt_ptr, heap = {}, {}, []
    if policy == "opt":
        for i, (_, units) in enumerate(calls):
            for u in units:
                nxt_list.setdefault(u, []).append(i)
    lfu_heap = []
    inf = float("inf")

    def next_use(u, i):
        lst = nxt_list[u]
        p = nxt_ptr.get(u, 0)
        while p < len(lst) and lst[p] <= i:
            p += 1
        nxt_ptr[u] = p
        return lst[p] if p < len(lst) else inf

    def victim(protect, i):
        if policy == "lru":
            for u in lru:
                if u not in protect:
                    return u
        elif policy == "mru":
            for u in reversed(lru):
                if u not in protect:
                    return u
        elif policy == "random":
            while True:
                u = rand_list[rng.randrange(len(rand_list))]
                if u not in protect:
                    return u
        elif policy == "layer":
            order = sorted(by_layer.items(), key=lambda kv: (-len(kv[1]), last[next(iter(kv[1]))] if kv[1] else 0))
            for _, od in order:
                for u in od:
                    if u not in protect:
                        return u
        elif policy == "lfu":
            back = []
            while lfu_heap:
                c, tl, u = heapq.heappop(lfu_heap)
                if u not in present or uses[u] != c or last[u] != tl:
                    continue
                if u in protect:
                    back.append((c, tl, u))
                    continue
                for b in back:
                    heapq.heappush(lfu_heap, b)
                return u
        elif policy == "opt":
            back = []
            while heap:
                k, u = heapq.heappop(heap)
                if u not in present or -k != next_use(u, i):
                    continue
                if u in protect:
                    back.append((k, u))
                    continue
                for b in back:
                    heapq.heappush(heap, b)
                return u
        raise RuntimeError("no victim: the store is smaller than one call")

    def drop(u):
        present.discard(u)
        del lru[u]
        by_layer[u // n_expert].pop(u)
        if policy == "random":
            j = rand_pos.pop(u)
            tail = rand_list.pop()
            if j < len(rand_list):
                rand_list[j] = tail
                rand_pos[tail] = j

    def use(u, i):
        lru.pop(u, None)
        lru[u] = None
        od = by_layer.setdefault(u // n_expert, OrderedDict())
        od.pop(u, None)
        od[u] = None
        uses[u] = uses.get(u, 0) + 1
        last[u] = i
        if policy == "lfu":
            heapq.heappush(lfu_heap, (uses[u], i, u))
        if policy == "opt":
            heapq.heappush(heap, (-next_use(u, i), u))

    hits = misses = prompt_misses = 0
    for i, (tok, units) in enumerate(calls):
        if len(units) > slots:
            raise RuntimeError(f"a call of {len(units)} units in {slots} slots")
        protect = set(units)
        for u in units:
            if u in present:
                if tok >= 0:
                    hits += 1
            else:
                if tok >= 0:
                    misses += 1
                else:
                    prompt_misses += 1
                if len(present) >= slots:
                    drop(victim(protect, i))
                present.add(u)
                if policy == "random":
                    rand_pos[u] = len(rand_list)
                    rand_list.append(u)
            use(u, i)
    return hits, misses, prompt_misses


POLICIES = ["lru", "mru", "random", "layer", "lfu", "colibri", "once", "ds4", "heat", "opt"]


def main(argv):
    if argv[:1] == ["--check"]:
        return run_check()
    if not argv:
        print(__doc__)
        return 2
    path = argv[0]
    opts = dict(zip(argv[1::2], argv[2::2]))
    tr = read_trace(path)
    slots_list = [int(s) for s in opts.get("--slots", "72,83,100,128,160,224,300,400,512,1024").split(",")]
    disk = float(opts.get("--disk-mbs", "500"))
    comp = float(opts.get("--compute-ms", "40"))
    seeds = int(opts.get("--seeds", "5"))
    calls = calls_of(tr)
    heats = {"ds4": heats_of(tr, False), "heat": heats_of(tr)}
    n_dec = tr["n_tokens"] - tr["n_prompt"]
    ub = tr["expert_bytes"]
    print(f"{path}: {tr['n_prompt']} prompt tokens, {n_dec} generated, {tr['n_layers']} layers x {tr['n_expert']} "
          f"experts, {tr['n_used']} used; a unit {ub / 2**20:.3f} MiB; a token {tr['n_layers'] * tr['n_used']} units")
    print(f"model: a decode token = {comp:g} ms + its misses at {disk:g} MB/s")
    print(f"{'slots':>6} {'policy':>7} {'hits/tok':>9} {'miss/tok':>9} {'MiB/tok':>8} {'prompt MiB':>11} {'tok/s':>7}")
    for S in slots_list:
        if S < max(len(u) for _, u in calls):
            continue
        for pol in POLICIES:
            runs = [replay(calls, S, pol, tr["n_expert"], seed, tr["n_layers"], heats.get(pol))
                    for seed in range(seeds if pol == "random" else 1)]
            h = sum(r[0] for r in runs) / len(runs)
            m = sum(r[1] for r in runs) / len(runs)
            pm = sum(r[2] for r in runs) / len(runs)
            mib = m / n_dec * ub / 2**20
            ms = comp + m / n_dec * ub / (disk * 1e6) * 1e3
            print(f"{S:>6} {pol:>7} {h / n_dec:>9.2f} {m / n_dec:>9.2f} {mib:>8.1f} {pm * ub / 2**20:>11.0f} "
                  f"{1000 / ms:>7.2f}")
    return 0


def run_check():
    fails = []
    ne = 4
    # one layer of 4 experts, 3 slots, decode calls of one unit each: 0 1 2 3 0 1 2 3 (a cycle of 4 in 3)
    calls = [(t, [u]) for t, u in enumerate([0, 1, 2, 3, 0, 1, 2, 3])]
    want = {"lru": (0, 8), "mru": (2, 6), "opt": (3, 5)}
    # lru: every unit evicted just before it comes back. mru: 0 1 2, 3 evicts 2; 0 hit, 1 hit, 2 evicts 1, 3 hit
    # -> hits 0,1,3 = 3? walk: [0,1,2] full; 3: victim mru=2 -> [0,1,3]; 0 hit; 1 hit; 2: mru=1 -> [0,2,3]; 3 hit
    want["mru"] = (3, 5)
    # opt: [0,1,2]; 3: furthest next use among 0(4),1(5),2(6) -> 2 out; 0 hit 1 hit; 2: 0(never),1(never),3(7):
    # evict 0 (inf, first found) -> 3 hit: hits 0,1,3 = 3
    for pol, (h, m) in want.items():
        got = replay(calls, 3, pol, ne)
        if got[:2] != (h, m):
            fails.append(f"{pol}: hits, misses {got[:2]} want {(h, m)}")
    # a call's own units are never its victims: 2 slots; [0 1], [2] evicts 0 -> [1 2]; [0 1]: 0 misses and the
    # LRU, 1, is this call's own, so 2 goes; 1 hits. A store that let a call evict its own units: 1 out, 1 misses
    calls = [(0, [0, 1]), (1, [2]), (2, [0, 1])]
    got = replay(calls, 2, "lru", ne)
    if got[:2] != (1, 4):
        fails.append(f"lru protect: {got[:2]} want (1, 4)")
    # layer: two layers, 4 slots; layer 0 holds 3, layer 1 one; a new layer-1 unit evicts layer 0's oldest
    calls = [(0, [0]), (1, [1]), (2, [2]), (3, [4]), (4, [5]), (5, [0])]
    got = replay(calls, 4, "layer", ne)
    # [0,1,2,4] full; 5 (layer 1): layer 0 holds 3 -> its oldest, 0, out; then 0 misses again
    if got[:2] != (0, 6):
        fails.append(f"layer: {got[:2]} want (0, 6)")
    calls = [(0, [0]), (1, [1]), (2, [2]), (3, [4]), (4, [5]), (5, [1])]
    got = replay(calls, 4, "layer", ne)
    if got[:2] != (1, 5):
        fails.append(f"layer keeps 1: {got[:2]} want (1, 5)")
    # lfu: 0 used twice, 1 once, then 2 and 3 in 2 slots... 3 slots: 0 0 1 2 3 1 -> 3 evicts 1 (one use, older
    # than 2), so 1 misses again
    calls = [(t, [u]) for t, u in enumerate([0, 0, 1, 2, 3, 1])]
    got = replay(calls, 3, "lfu", ne)
    if got[:2] != (1, 5):
        fails.append(f"lfu: {got[:2]} want (1, 5)")
    # the prompt's misses apart: tok -1 counts nothing in hits/misses
    got = replay([(-1, [0, 1]), (0, [0])], 2, "lru", ne)
    if got != (1, 0, 2):
        fails.append(f"prompt apart: {got} want (1, 0, 2)")
    # colibri: 2 layers, 4 slots: 2 a layer. 0 1 2 leaves [1 2]; 0 misses, leaves [2 0]; 2 hits
    calls = [(t, [u]) for t, u in enumerate([0, 1, 2, 0, 2])]
    got = replay(calls, 4, "colibri", ne, 0, 2)
    if got[:2] != (1, 4):
        fails.append(f"colibri: {got[:2]} want (1, 4)")
    # ds4: 2 slots, 0 0 1 2 0: 2 evicts the cooler 1 (hotness 1 against 2), so 0 hits; an LRU would evict 0
    calls = [(t, [u]) for t, u in enumerate([0, 0, 1, 2, 0])]
    got = replay(calls, 2, "ds4", ne)
    if got[:2] != (2, 3):
        fails.append(f"ds4: {got[:2]} want (2, 3)")
    # heat: 3 slots, the prompt's calls add 5, 1, 2 to units 0, 1, 2; the decode's 3 evicts the coolest, 1, so 0
    # hits. ds4's +1 leaves all three at 1 and evicts the oldest, 0, which then misses
    calls = [(-1, [0]), (-1, [1]), (-1, [2]), (0, [3]), (1, [0])]
    got = replay(calls, 3, "heat", ne, heats=[[5], [1], [2], [1], [1]])
    if got != (1, 1, 3):
        fails.append(f"heat: {got} want (1, 1, 3)")
    got = replay(calls, 3, "once", ne)
    if got != (0, 2, 3):
        fails.append(f"heat's once: {got} want (0, 2, 3)")
    # ds4 against heat, 3 slots: a pass of 320 tokens, 300 chose unit 0 and 20 unit 1: ds4 adds them whole (300,
    # 20), heat as 64 tokens (60, 4). Unit 2 is read at the first decode token and hits 15 times; the 16th halves
    # every unit first: ds4 150, 10, 8; heat 30, 2, 8. Unit 3 evicts the coolest: unit 2 under ds4, unit 1 under
    # heat; unit 2 asked again hits only under heat (15 hits against 16)
    tr = {"n_layers": 1, "n_expert": 8, "n_prompt": 320, "n_tokens": 337,
          "chosen": [[[0]] for _ in range(300)] + [[[1]] for _ in range(20)] + [[[2]] for _ in range(16)] + [[[3]]]}
    calls = calls_of(tr) + [(337, [2])]
    for pol, scaled, want in [("ds4", False, 15), ("heat", True, 16)]:
        got = replay(calls, 3, pol, 8, heats=heats_of(tr, scaled) + [[1]])
        if got[0] != want:
            fails.append(f"{pol} against heat: {got} want {want} hits")
    # heat_add: c as is up to 64 tokens, else c x 64 / n rounded half up, at least 1
    for c, n, want in [(5, 64, 5), (3, 128, 2), (10, 321, 2), (1, 321, 1), (1, 1, 1)]:
        if heat_add(c, n) != want:
            fails.append(f"heat_add({c}, {n}) = {heat_add(c, n)} want {want}")
    for f in fails:
        print("evict_replay --check:", f)
    if not fails:
        print("evict_replay --check: ok (13 cases)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
