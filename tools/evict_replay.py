#!/usr/bin/env python3
"""evict_replay.py -- the expert store's eviction, replayed on a route trace (docs/MEASUREMENTS.md §The three
machines; LESSONS #271: under one token's units the store's LRU hits nothing).

Usage:
    evict_replay.py <trace.bin> [--slots 72,83,128,...] [--disk-mbs 500] [--compute-ms 40] [--seeds 5] [--see 16,256]
    evict_replay.py <trace.bin> --prefetch 4,8,12 [--src in,out,oracle] [--chunk-kib 512,2048,0] [--arrive 0,1]
                    [--attn-frac 0.25] [--head-frac 0.05] [--slots ...] [--disk-mbs ...] [--compute-ms ...]
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
    see<H>    (--see H,...) heat knowing the next H calls: a unit not asked within them goes first (the lowest
              hotness), else the one asked furthest. How far ahead a policy must see to reach Belady (a decode token
              is n_layers calls; MEASUREMENTS §The eviction told the future)
The time model (a model, never a measurement): a decode token's time = --compute-ms + its misses' bytes at
--disk-mbs; tok/s = 1000 / that. With --prefetch, the time is replayed on one disk queue under `heat`
(replay_time): the next layer's units read ahead (k of the router's pred_in or pred_out, or the true ones), and
--arrive 1 a layer's experts computed as their bytes arrive (MEASUREMENTS §The routes as time).
"""
import bisect
import heapq
import inspect
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
    npd = n_tokens * n_layers * n_pred
    preds = []
    for k in range(2):                # pred_in, pred_out (src/models/model.h), best first, 0xFFFF on the last layer
        p = struct.unpack_from(f"<{npd}H", data, off + 2 * n + 2 * npd * k) if n_pred else ()
        preds.append([[list(p[(t * n_layers + L) * n_pred:(t * n_layers + L + 1) * n_pred]) for L in range(n_layers)]
                      for t in range(n_tokens)] if n_pred else None)
    return dict(n_tokens=n_tokens, n_prompt=n_prompt, n_layers=n_layers, n_expert=n_expert, n_used=n_used,
                expert_bytes=expert_bytes, chosen=chosen, pred_in=preds[0], pred_out=preds[1])


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


def replay_ds4(calls, slots, n_expert, heats=None, see=None):
    """ds4: the lowest hotness out (the oldest among equals), this call's units never; hotness +1 a routing
    (heats: + heats[call][k] for the call's k-th unit), halved every 16 decode tokens. see=H: the next H calls
    known, a unit they do not ask out first (by hotness), else the one they ask furthest."""
    present, hot, last = set(), {}, {}
    uses = {}
    if see is not None:
        for j, (_, us) in enumerate(calls):
            for u in us:
                uses.setdefault(u, []).append(j)

    def key(p, i):
        if see is None:
            return (hot[p], last[p])
        lst = uses[p]
        j = bisect.bisect_right(lst, i)
        if j == len(lst) or lst[j] > i + see:
            return (0, 0, hot[p], last[p])
        return (1, -lst[j], hot[p], last[p])
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
                    v = min((p for p in present if p not in protect), key=lambda p: key(p, i))
                    present.discard(v)
                present.add(u)
            last[u] = i
    return hits, misses, prompt_misses


def replay(calls, slots, policy, n_expert, seed=0, n_layers=16, heats=None):
    """(decode hits, decode misses, prompt misses) of one policy at `slots`; ds4, heat and see<H> add `heats`
    (heats_of unscaled and scaled; None: 1 a routing, a call of one token's)."""
    if policy.startswith("see"):
        return replay_ds4(calls, slots, n_expert, heats, int(policy[3:]))
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


def replay_time(tr, slots, heats, k, src="in", comp=40.0, attn=0.25, mbs=500.0, chunk=512 << 10, arrive=0,
                head=0.0, wave=0.0):
    """(decode misses, read-ahead units named, decode bytes, decode ms): heat's store, one disk queue, and the next
    call's units read ahead (MEASUREMENTS §The routes as time). A layer is attn x comp / n_layers of compute, its
    router, its units (the misses and a read ahead's rest first on the disk, in call order), the rest of its
    compute. src "in": at the router, layer L+1's router's top k on this FFN input (pred_in) not resident are
    queued, each taking a slot (heat's victim); "out": the same after the layer (pred_out); "oracle": the next
    call's true units. The queue reads while the disk is free, `chunk` bytes at a time (0: a unit whole), a demand
    waiting for the chunk in flight; at the next router a queued unit it names is finished as a demand, one it does
    not name dropped (its slot freed if not read whole). k = 0: the serial model, comp + misses at mbs. arrive 1: the
    rest of a layer's compute split evenly among its units, each computed as soon as its bytes are in (the resident
    ones first, a miss read under them); 2: two waves, the units in at the router, then every late one once the last
    is in; 3: 2 from three late units a layer, else 0 (ds4's split); 0: every unit after the last. wave: the ms
    each wave of late units adds (the engine's extra pool regions). head: the share of comp after the last layer
    (the output head, nothing to hide under it)."""
    calls = calls_of(tr)
    nl, ne, np_ = tr["n_layers"], tr["n_expert"], tr["n_prompt"]
    pred = tr["pred_out" if src == "out" else "pred_in"]
    ub = tr["expert_bytes"]
    rate = mbs * 1e3                  # bytes a ms
    a = comp * (1 - head) / nl * attn
    f = comp * (1 - head) / nl - a
    present, hot, last, ready = set(), {}, {}, {}
    queue = []                        # [unit, bytes left], rank order: the next call's read ahead
    st = dict(T=0.0, disk=0.0, q_t=0.0, rd=0, issued=set())
    misses = used = 0
    tok_seen, prev_tok = 0, None

    def take_slot(protect):
        if len(present) >= slots:
            present.discard(min((p for p in present if p not in protect), key=lambda p: (hot[p], last[p])))

    def progress(until):
        t = max(st["disk"], st["q_t"])
        ran = False
        while queue and t < until:
            u = queue[0]
            n = min(chunk, u[1]) if chunk else u[1]
            t += n / rate
            u[1] -= n
            st["rd"] += n
            ran = True
            if u[1] == 0:
                ready[u[0]] = t
                queue.pop(0)
        if ran:
            st["disk"] = t

    def issue(units, i, protect):
        for u in units:
            if u not in present:
                take_slot(protect | {q[0] for q in queue})
                present.add(u)
                hot.setdefault(u, 0)
                last[u] = i
                queue.append([u, ub])
                st["issued"].add(u)
        st["q_t"] = st["T"]

    for i, (tok, units) in enumerate(calls):
        L = units[0] // ne
        named = set(units)
        if tok >= 0 and tok != prev_tok:
            tok_seen += 1
            prev_tok = tok
            if tok_seen % 16 == 0:
                for u in hot:
                    hot[u] //= 2
        if tok >= 0:
            st["T"] += a
            progress(st["T"])
        left = {u: n for u, n in queue if u in named}
        for u, n in queue:
            if u not in named and n > 0:
                present.discard(u)
        queue.clear()
        used += len(named & st["issued"])
        st["issued"] = set()
        t = max(st["disk"], st["T"])
        arr = []                      # when each unit's bytes are in
        for j, u in enumerate(units):
            hot[u] = hot.get(u, 0) + heats[i][j]
            if u in left:
                t += left[u] / rate
                st["rd"] += left[u]
                arr.append(t)
            elif u in present:
                arr.append(max(st["T"], ready.get(u, 0.0)))
            else:
                take_slot(named)
                present.add(u)
                if tok >= 0:
                    misses += 1
                    t += ub / rate
                    st["rd"] += ub
                    arr.append(t)
            last[u] = i
        if tok < 0:
            continue
        if t > max(st["disk"], st["T"]):
            st["disk"] = t
        late = [r for r in arr if r > st["T"]]
        if arrive == 3 and len(late) < 3:  # ds4's split: worth it from 3 misses a layer (ds4_metal.m:13095)
            done = max(arr) + f
        elif arrive >= 2 and late:      # two waves: the resident units, then every late one after the last
            done = max(st["T"] + (len(arr) - len(late)) * f / len(units), max(late)) + len(late) * f / len(units) + wave
        elif arrive:
            done = st["T"]
            for r in sorted(arr):
                done = max(done, r) + f / len(units) + (wave if r > st["T"] else 0.0)
        else:
            done = max(arr) + f
        nxt = [] if k == 0 or L == nl - 1 else (
            [(L + 1) * ne + e for e in tr["chosen"][tok][L + 1]] if src == "oracle" else
            [(L + 1) * ne + e for e in pred[tok][L][:k] if e != 0xFFFF])
        if src != "out":
            issue(nxt, i, named)
        st["T"] = done + (comp * head if L == nl - 1 else 0.0)
        if src == "out":
            issue(nxt, i, named)
    return misses, used, st["rd"], st["T"]


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
    sees = [f"see{h}" for h in opts["--see"].split(",")] if "--see" in opts else []
    for pol in sees:
        heats[pol] = heats["heat"]
    n_dec = tr["n_tokens"] - tr["n_prompt"]
    ub = tr["expert_bytes"]
    print(f"{path}: {tr['n_prompt']} prompt tokens, {n_dec} generated, {tr['n_layers']} layers x {tr['n_expert']} "
          f"experts, {tr['n_used']} used; a unit {ub / 2**20:.3f} MiB; a token {tr['n_layers'] * tr['n_used']} units")
    print(f"model: a decode token = {comp:g} ms + its misses at {disk:g} MB/s")
    if "--prefetch" in opts:
        attn = float(opts.get("--attn-frac", "0.25"))
        chunks = [int(c) << 10 for c in opts.get("--chunk-kib", "512").split(",")]
        arrives = [int(x) for x in opts.get("--arrive", "0").split(",")]
        head = float(opts.get("--head-frac", "0.05"))
        wave = float(opts.get("--wave-ms", "0"))
        print(f"the next call read ahead (replay_time): {attn:g} of a layer's compute before its router, "
              f"{head:g} of a token's "
              f"after the last layer; gain against "
              f"serial (k 0); oracle: the next call's true units")
        print(f"{'slots':>6} {'src':>8} {'k':>3} {'chunk':>6} {'miss/tok':>9} {'ahead/tok':>10} {'MiB/tok':>8} "
              f"{'ms/tok':>7} {'tok/s':>7} {'gain':>6}")
        for S in slots_list:
            base = None
            ks = [int(x) for x in opts["--prefetch"].split(",")]
            runs = [(a, "serial", 0, chunks[0]) for a in arrives] + [
                (a, src, k, ch) for a in arrives for src in opts.get("--src", "in,out,oracle").split(",")
                for k in (ks[:1] if src == "oracle" else ks) for ch in chunks]
            for arrive, src, k, ch in runs:
                m, u, rd, ms = replay_time(tr, S, heats["heat"], k, src, comp, attn, disk, ch, arrive, head, wave)
                base = base or ms
                print(f"{S:>6} {src + (f'+a{arrive}' if arrive else ''):>8} {k:>3} {ch >> 10:>6} {m / n_dec:>9.2f} {u / n_dec:>10.2f} "
                      f"{rd / n_dec / 2**20:>8.1f} {ms / n_dec:>7.2f} {1000 * n_dec / ms:>7.2f} {base / ms:>6.3f}")
        return 0
    print(f"{'slots':>6} {'policy':>7} {'hits/tok':>9} {'miss/tok':>9} {'MiB/tok':>8} {'prompt MiB':>11} {'tok/s':>7}")
    for S in slots_list:
        if S < max(len(u) for _, u in calls):
            continue
        for pol in POLICIES + sees:
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
    ran = [0]

    def check(ok, msg):
        ran[0] += 1   # counted, never written by hand (LESSONS #309)
        if not ok:
            fails.append(msg)
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
        check(got[:2] == (h, m), f"{pol}: hits, misses {got[:2]} want {(h, m)}")
    # a call's own units are never its victims: 2 slots; [0 1], [2] evicts 0 -> [1 2]; [0 1]: 0 misses and the
    # LRU, 1, is this call's own, so 2 goes; 1 hits. A store that let a call evict its own units: 1 out, 1 misses
    calls = [(0, [0, 1]), (1, [2]), (2, [0, 1])]
    got = replay(calls, 2, "lru", ne)
    check(got[:2] == (1, 4), f"lru protect: {got[:2]} want (1, 4)")
    # layer: two layers, 4 slots; layer 0 holds 3, layer 1 one; a new layer-1 unit evicts layer 0's oldest
    calls = [(0, [0]), (1, [1]), (2, [2]), (3, [4]), (4, [5]), (5, [0])]
    got = replay(calls, 4, "layer", ne)
    # [0,1,2,4] full; 5 (layer 1): layer 0 holds 3 -> its oldest, 0, out; then 0 misses again
    check(got[:2] == (0, 6), f"layer: {got[:2]} want (0, 6)")
    calls = [(0, [0]), (1, [1]), (2, [2]), (3, [4]), (4, [5]), (5, [1])]
    got = replay(calls, 4, "layer", ne)
    check(got[:2] == (1, 5), f"layer keeps 1: {got[:2]} want (1, 5)")
    # lfu: 0 used twice, 1 once, then 2 and 3 in 2 slots... 3 slots: 0 0 1 2 3 1 -> 3 evicts 1 (one use, older
    # than 2), so 1 misses again
    calls = [(t, [u]) for t, u in enumerate([0, 0, 1, 2, 3, 1])]
    got = replay(calls, 3, "lfu", ne)
    check(got[:2] == (1, 5), f"lfu: {got[:2]} want (1, 5)")
    # the prompt's misses apart: tok -1 counts nothing in hits/misses
    got = replay([(-1, [0, 1]), (0, [0])], 2, "lru", ne)
    check(got == (1, 0, 2), f"prompt apart: {got} want (1, 0, 2)")
    # colibri: 2 layers, 4 slots: 2 a layer. 0 1 2 leaves [1 2]; 0 misses, leaves [2 0]; 2 hits
    calls = [(t, [u]) for t, u in enumerate([0, 1, 2, 0, 2])]
    got = replay(calls, 4, "colibri", ne, 0, 2)
    check(got[:2] == (1, 4), f"colibri: {got[:2]} want (1, 4)")
    # ds4: 2 slots, 0 0 1 2 0: 2 evicts the cooler 1 (hotness 1 against 2), so 0 hits; an LRU would evict 0
    calls = [(t, [u]) for t, u in enumerate([0, 0, 1, 2, 0])]
    got = replay(calls, 2, "ds4", ne)
    check(got[:2] == (2, 3), f"ds4: {got[:2]} want (2, 3)")
    # heat: 3 slots, the prompt's calls add 5, 1, 2 to units 0, 1, 2; the decode's 3 evicts the coolest, 1, so 0
    # hits. ds4's +1 leaves all three at 1 and evicts the oldest, 0, which then misses
    calls = [(-1, [0]), (-1, [1]), (-1, [2]), (0, [3]), (1, [0])]
    got = replay(calls, 3, "heat", ne, heats=[[5], [1], [2], [1], [1]])
    check(got == (1, 1, 3), f"heat: {got} want (1, 1, 3)")
    got = replay(calls, 3, "once", ne)
    check(got == (0, 2, 3), f"heat's once: {got} want (0, 2, 3)")
    # ds4 against heat, 3 slots: a pass of 320 tokens, 300 chose unit 0 and 20 unit 1: ds4 adds them whole (300,
    # 20), heat as 64 tokens (60, 4). Unit 2 is read at the first decode token and hits 15 times; the 16th halves
    # every unit first: ds4 150, 10, 8; heat 30, 2, 8. Unit 3 evicts the coolest: unit 2 under ds4, unit 1 under
    # heat; unit 2 asked again hits only under heat (15 hits against 16)
    tr = {"n_layers": 1, "n_expert": 8, "n_prompt": 320, "n_tokens": 337,
          "chosen": [[[0]] for _ in range(300)] + [[[1]] for _ in range(20)] + [[[2]] for _ in range(16)] + [[[3]]]}
    calls = calls_of(tr) + [(337, [2])]
    for pol, scaled, want in [("ds4", False, 15), ("heat", True, 16)]:
        got = replay(calls, 3, pol, 8, heats=heats_of(tr, scaled) + [[1]])
        check(got[0] == want, f"{pol} against heat: {got} want {want} hits")
    # see<H>: 2 slots, decode calls 0 0 1 2 1. heat (see0 too: nothing within 0 calls) lets 2 evict the cooler 1,
    # which then misses: 1 hit. see1 knows call 4 asks 1 and nothing asks 0 again: 0 goes, 1 hits: 2 hits
    calls = [(t, [u]) for t, u in enumerate([0, 0, 1, 2, 1])]
    for pol, want in [("heat", (1, 4, 0)), ("see0", (1, 4, 0)), ("see1", (2, 3, 0))]:
        got = replay(calls, 2, pol, ne, heats=[[1]] * 5)
        check(got == want, f"{pol}: {got} want {want}")
    # replay_time: 2 layers, one token, a unit 1000 bytes at 250 a ms (4 ms), compute 4 ms (a layer 0.5 + 1.5).
    # Serial: 0.5, unit 0 to 4.5, 6.0; 6.5, unit 5 to 10.5, 12.0. Right ahead (unit 5 queued at 0.5, read from 4.5):
    # whole, done at 8.5, waited: 10.0; in 250-byte chunks, 500 left at the router (6.5), read first: 10.0. Wrong
    # (unit 6): whole, 6.5 waits to 8.5, 12.5, 14.0; chunks, dropped at 6.5: 12.0, 500 bytes wasted. pred_out
    # right: queued at 6.0, one chunk to 7.0, 750 left: 10.0, 11.5. The oracle reads the true unit 5
    for p, src, k, ch, want in [(1, "in", 0, 250, (2, 0, 2000, 12.0)), (1, "in", 1, 0, (1, 1, 2000, 10.0)),
                                (1, "in", 1, 250, (1, 1, 2000, 10.0)), (2, "in", 1, 0, (2, 0, 3000, 14.0)),
                                (2, "in", 1, 250, (2, 0, 2500, 12.0)), (1, "out", 1, 250, (1, 1, 2000, 11.5)),
                                (2, "oracle", 1, 250, (1, 1, 2000, 10.0))]:
        tr = {"n_layers": 2, "n_expert": 4, "n_prompt": 0, "n_tokens": 1, "expert_bytes": 1000,
              "chosen": [[[0], [1]]], "pred_in": [[[p], [0xFFFF]]], "pred_out": [[[p], [0xFFFF]]]}
        got = replay_time(tr, 4, [[1], [1]], k, src, 4.0, 0.25, 0.25, ch)
        check(got[:3] == want[:3] and abs(got[3] - want[3]) <= 1e-9,
              f"replay_time pred {p} {src} k {k} chunk {ch}: {got} want {want}")
    # head 0.25: a layer 1.5 ms (0.375 + 1.125), 1 ms after the last; right ahead, whole: 0.375, unit 0 to 4.375,
    # 5.5; 5.875 waits unit 5 to 8.375, 9.5, + 1 = 10.5 (the window shrank by the head's share)
    tr = {"n_layers": 2, "n_expert": 4, "n_prompt": 0, "n_tokens": 1, "expert_bytes": 1000,
          "chosen": [[[0], [1]]], "pred_in": [[[1], [0xFFFF]]], "pred_out": None}
    got = replay_time(tr, 4, [[1], [1]], 1, "in", 4.0, 0.25, 0.25, 0, False, 0.25)
    check(got[:3] == (1, 1, 2000) and abs(got[3] - 10.5) <= 1e-9, f"replay_time head: {got} want (1, 1, 2000, 10.5)")
    # a wrong unit dropped half read is not resident: the next token asks unit 6 and misses (12.5, hit, 14.0; 14.5,
    # 6 to 18.5, 20.0). Kept, it would hit with 500 of its bytes never read
    tr = {"n_layers": 2, "n_expert": 4, "n_prompt": 0, "n_tokens": 2, "expert_bytes": 1000,
          "chosen": [[[0], [1]], [[0], [2]]], "pred_in": [[[2], [0xFFFF]], [[0xFFFF], [0xFFFF]]], "pred_out": None}
    got = replay_time(tr, 4, [[1]] * 4, 1, "in", 4.0, 0.25, 0.25, 250)
    check(got[:3] == (3, 0, 3500) and abs(got[3] - 20.0) <= 1e-9,
          f"replay_time drops a half-read unit: {got} want (3, 0, 3500, 20.0)")
    # arrive: one layer, compute 4 ms (1 + 3); the prompt leaves unit 1; the decode asks 0 (a miss, in at 5) and 1
    # (resident, in at 1). In arrival order: 1 computed 1 -> 2.5, 0 at 5 -> 6.5. In call order 0 waits to 5,
    # 6.5, then 1, 8.0; serial (after the last), 5 + 3 = 8.0
    tr = {"n_layers": 1, "n_expert": 4, "n_prompt": 1, "n_tokens": 2, "expert_bytes": 1000,
          "chosen": [[[1]], [[0, 1]]], "pred_in": None, "pred_out": None}
    for arrive, want in [(True, 6.5), (False, 8.0)]:
        got = replay_time(tr, 4, [[1], [1, 1]], 0, "in", 4.0, 0.25, 0.25, 250, arrive)
        check(got[:3] == (1, 0, 1000) and abs(got[3] - want) <= 1e-9,
              f"replay_time arrive {arrive}: {got} want (1, 0, 1000, {want})")
    # the waves: the decode asks 0 and 2 (misses, in at 5 and 9) and 1 (resident, in at 1), 1 ms a unit after the
    # attention's 1. Per arrival: 1 -> 2, 0 5 -> 6, 2 9 -> 10, a wave's 0.5 after each late one 10.5; two waves:
    # 1 -> 2, then both at 9 -> 11, 11.5; ds4's (two late, under 3): serial, 9 + 3 = 12; serial 12
    tr = {"n_layers": 1, "n_expert": 4, "n_prompt": 1, "n_tokens": 2, "expert_bytes": 1000,
          "chosen": [[[1]], [[0, 1, 2]]], "pred_in": None, "pred_out": None}
    for arrive, wave, want in [(1, 0.0, 10.0), (1, 0.5, 10.5), (2, 0.0, 11.0), (2, 0.5, 11.5), (3, 0.0, 12.0),
                               (0, 0.5, 12.0)]:
        got = replay_time(tr, 4, [[1], [1, 1, 1]], 0, "in", 4.0, 0.25, 0.25, 250, arrive, 0.0, wave)
        check(got[:3] == (2, 0, 2000) and abs(got[3] - want) <= 1e-9,
              f"replay_time arrive {arrive} wave {wave}: {got} want (2, 0, 2000, {want})")
    # ds4's split from three late units: 0, 2, 3 missing (in at 5, 9, 13), 1 resident; 0.75 ms a unit: 1 -> 1.75,
    # then the three at 13 -> 15.25; serial 13 + 3 = 16
    tr = {"n_layers": 1, "n_expert": 4, "n_prompt": 1, "n_tokens": 2, "expert_bytes": 1000,
          "chosen": [[[1]], [[0, 1, 2, 3]]], "pred_in": None, "pred_out": None}
    for arrive, want in [(3, 15.25), (0, 16.0)]:
        got = replay_time(tr, 4, [[1], [1, 1, 1, 1]], 0, "in", 4.0, 0.25, 0.25, 250, arrive)
        check(got[:3] == (3, 0, 3000) and abs(got[3] - want) <= 1e-9,
              f"replay_time ds4 split {arrive}: {got} want (3, 0, 3000, {want})")
    # heat_add: c as is up to 64 tokens, else c x 64 / n rounded half up, at least 1
    for c, n, want in [(5, 64, 5), (3, 128, 2), (10, 321, 2), (1, 321, 1), (1, 1, 1)]:
        check(heat_add(c, n) == want, f"heat_add({c}, {n}) = {heat_add(c, n)} want {want}")
    sites = sum(ln.lstrip().startswith("check(") for ln in inspect.getsource(run_check).splitlines())
    if ran[0] < sites:                # every check written ran at least once
        fails.append(f"{ran[0]} cases ran, {sites} written")
    for f in fails:
        print("evict_replay --check:", f)
    if not fails:
        print(f"evict_replay --check: ok ({ran[0]} cases)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
