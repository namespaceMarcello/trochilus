#!/usr/bin/env python3
"""spec_replay.py -- a speculative decode's verify passes replayed on the expert store (docs/MEASUREMENTS.md
§A draft from the bytes the machine holds, question 94). A model, never a measurement.

Usage:
    spec_replay.py <trace.bin>... [--slots 290] [--rows 1,2,3,4,6,8] [--accept <file>] [--draft 1,2,3,4,7]
    spec_replay.py --check       hand-made sequences with hand-computed answers

The store is `heat` (src/memory/experts.c, tools/evict_replay.py): the lowest hotness out, the oldest among
equals, a call's own units never; a call adds to each unit the routings of its tokens that chose it (as if from
at most 64 tokens), and every 16 x n_layers calls every hotness is halved (hot_period counts calls, so a pass of
r rows halves every 16 r tokens, as the engine would). The prompt is read in passes of 512 as in evict_replay.

The decode is replayed in passes. Without --accept, every pass holds `rows` consecutive generated tokens, all
accepted: the bytes of a perfect draft (the bound). With --accept (one line a generated position: 1 if the draft's
token there equals the exact one, from tools/draft_probe's output), a pass of k drafts holds k + 1 rows: the token
after the last accepted one and k drafted ones; it keeps the leading accepted drafts and the exact token after them
(j + 1 tokens); the rejected rows' experts are taken as the exact routes at the same positions (their own routes
are not in the trace: the same context, another last token). Per generated token it prints the store's disk units
(misses) and the RAM units (every unit a pass names, read once), each against the one-row decode.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evict_replay import HOT_PROMPT, PASS, heat_add, read_trace  # noqa: E402

HOT_CALLS = 16  # decode tokens between halvings (ds4), times n_layers calls
POLICY = {"halve": "calls", "heat": "all"}  # the store under passes: --halve, --heat


def store_replay(calls, slots, n_layers, snaps=None):
    """calls: [(decode, [units], [heat a unit], tokens)]; decode False for the prompt's, tokens the tokens its pass
    keeps (on the pass's first call, 0 on the others). Returns (disk units of the decode calls, units named by the
    decode calls). The heat store of evict_replay.replay_ds4, halved every 16 n_layers decode calls (POLICY halve
    "calls", the engine's hot_period) or every 16 kept tokens ("tokens", ds4's own words).
    snaps: {call index: []}, each list filled with the units present before that call."""
    present, hot, last = set(), {}, {}
    misses = named = 0
    n_dec = 0
    tok_done, next_halve = 0, HOT_CALLS
    for i, (dec, units, heats, adv) in enumerate(calls):
        if snaps is not None and i in snaps:
            snaps[i].extend(sorted(present))
        if dec and POLICY["halve"] == "calls":
            if n_dec > 0 and n_dec % (HOT_CALLS * n_layers) == 0:
                for u in hot:
                    hot[u] //= 2
            n_dec += 1
        elif dec and adv > 0:
            while tok_done >= next_halve:
                for u in hot:
                    hot[u] //= 2
                next_halve += HOT_CALLS
            tok_done += adv
        protect = set(units)
        for k, u in enumerate(units):
            hot[u] = hot.get(u, 0) + heats[k]
            if dec:
                named += 1
            if u not in present:
                if dec:
                    misses += 1
                if len(present) >= slots:
                    v = min((p for p in present if p not in protect), key=lambda p: (hot[p], last[p]))
                    present.discard(v)
                present.add(u)
            last[u] = i
    return misses, named


def pass_calls(ch, toks, n_layers, n_expert, n_kept=None):
    """One pass over the tokens toks: every layer once, the union of their experts, heat by routings: every row's
    (POLICY heat "all"), or only its first n_kept rows' ("kept": a rejected row's units are read, not counted)."""
    toks = list(toks)
    n_kept = len(toks) if n_kept is None else n_kept
    counted = toks if POLICY["heat"] == "all" else toks[:n_kept]
    out = []
    for L in range(n_layers):
        c = {}
        for t in toks:
            for e in ch[t][L]:
                c[e] = 0
        for t in counted:
            for e in ch[t][L]:
                c[e] += 1
        ids = sorted(c)
        out.append((True, [L * n_expert + e for e in ids], [heat_add(c[e], len(counted)) for e in ids],
                    n_kept if L == 0 else 0))
    return out


def prompt_calls(tr):
    nl, ne, np_ = tr["n_layers"], tr["n_expert"], tr["n_prompt"]
    out = []
    for p0 in range(0, np_, PASS):
        for dec, units, heats, adv in pass_calls(tr["chosen"], range(p0, min(np_, p0 + PASS)), nl, ne):
            out.append((False, units, heats, 0))
    return out


def schedule(n_gen, rows=None, accept=None, k=None):
    """[(first, n_rows, n_kept)] over generated positions 0..n_gen-1: fixed rows all kept, or k drafts judged by
    accept[pos] (1: the draft equals the exact token at pos)."""
    out = []
    p = 0
    while p < n_gen:
        if accept is None:
            r = min(rows, n_gen - p)
            out.append((p, r, r))
            p += r
            continue
        # the row at p is exact (the token after the last accepted one); drafts at p+1 .. p+k
        r = min(k + 1, n_gen - p)
        j = 0
        while j < r - 1 and accept[p + 1 + j]:
            j += 1
        out.append((p, r, j + 1))
        p += j + 1
    return out


def replay(tr, slots, sched):
    nl, ne, np_ = tr["n_layers"], tr["n_expert"], tr["n_prompt"]
    ch = tr["chosen"]
    calls = prompt_calls(tr)
    n_pass = 0
    for first, r, kept in sched:
        calls += pass_calls(ch, range(np_ + first, np_ + first + r), nl, ne, kept)
        n_pass += 1
    misses, named = store_replay(calls, slots, nl)
    return misses, named, n_pass


def dump_resident(tr, slots, path):
    """The one-row decode's store before each generated row: n_gen x n_layers x n_expert bytes, 1 = present
    (tools/resident_probe.c --resident)."""
    nl, ne, np_ = tr["n_layers"], tr["n_expert"], tr["n_prompt"]
    n_gen = tr["n_tokens"] - np_
    calls = prompt_calls(tr)
    base = len(calls)
    for g in range(n_gen):
        calls += pass_calls(tr["chosen"], [np_ + g], nl, ne)
    snaps = {base + g * nl: [] for g in range(n_gen)}
    store_replay(calls, slots, nl, snaps)
    out = bytearray(n_gen * nl * ne)
    for g in range(n_gen):
        for u in snaps[base + g * nl]:
            out[g * nl * ne + u] = 1
    with open(path, "wb") as f:
        f.write(bytes(out))
    return n_gen


def check_routes(tr, path):
    """The probe's exact rows routed as the trace's decode rows, every layer's set the same (LESSONS #329)."""
    import struct
    nl, nu, np_ = tr["n_layers"], tr["n_used"], tr["n_prompt"]
    data = open(path, "rb").read()
    n = len(data) // 2
    flat = struct.unpack(f"<{n}H", data)
    rows = n // (nl * nu)
    bad = 0
    for g in range(rows):
        for L in range(nl):
            got = sorted(flat[(g * nl + L) * nu:(g * nl + L + 1) * nu])
            if got != sorted(tr["chosen"][np_ + g][L]):
                bad += 1
    print(f"{path}: {rows} rows, {bad} (row, layer) sets unlike the trace's")
    return 1 if bad or rows == 0 else 0


def read_chain(path, column):
    """resident_probe's output: each row's matched length in `column` (sub, skip or norm); -1 kept."""
    lines = open(path).read().splitlines()
    head = lines[0].split("\t")
    c = head.index(column)
    return [int(x.split("\t")[c]) for x in lines[1:] if x.strip()]


def schedule_chain(n_gen, matched, k):
    """[(first, n_rows, n_kept)]: a pass at row p with k drafts keeps min(matched[p], k) of them and the exact
    token after (a row whose draft did not run, -1: a one-row pass)."""
    out = []
    p = 0
    while p < n_gen:
        m = matched[p] if p < len(matched) else -1
        r = min(k + 1, n_gen - p) if m >= 0 else 1
        j = max(0, min(m, r - 1))
        out.append((p, r, j + 1))
        p += j + 1
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--slots", default="290")
    ap.add_argument("--rows", default="1,2,3,4,6,8")
    ap.add_argument("--accept", default=None, help="files, comma-separated, one per trace")
    ap.add_argument("--draft", default="1,2,3,4,7")
    ap.add_argument("--p", default=None, help="instead of --accept: drafts agreeing at random with these rates")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--dump-resident", default=None, help="write the one-row store's resident sets (one trace)")
    ap.add_argument("--chain", default=None, help="resident_probe outputs, comma-separated, one per trace")
    ap.add_argument("--variants", default="sub,skip,norm")
    ap.add_argument("--routes", default=None, help="resident_probe --routes-out: must equal the trace's rows")
    ap.add_argument("--halve", choices=["calls", "tokens"], default="calls")
    ap.add_argument("--heat", choices=["all", "kept"], default="all")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    POLICY["halve"], POLICY["heat"] = a.halve, a.heat
    if a.check:
        return run_check()
    if a.routes is not None:
        return check_routes(read_trace(a.traces[0]), a.routes)
    if a.dump_resident is not None:
        tr = read_trace(a.traces[0])
        n = dump_resident(tr, int(a.slots), a.dump_resident)
        print(f"{a.dump_resident}: {n} rows of {tr['n_layers'] * tr['n_expert']} bytes, {a.slots} slots")
        return 0
    if a.p is not None:
        return main_p(a)
    if a.chain is not None:
        return main_chain(a)
    accepts = a.accept.split(",") if a.accept else [None] * len(a.traces)
    for path, acc_path in zip(a.traces, accepts):
        tr = read_trace(path)
        n_gen = tr["n_tokens"] - tr["n_prompt"]
        unit_mib = tr["expert_bytes"] / 2**20
        for slots in [int(s) for s in a.slots.split(",")]:
            base_m, base_n, _ = replay(tr, slots, schedule(n_gen, rows=1))
            print(f"{os.path.basename(path)} slots {slots} unit {unit_mib:.2f} MiB, {n_gen} generated: one row "
                  f"{base_m / n_gen:.2f} disk units a token ({base_m / n_gen * unit_mib:.1f} MiB), "
                  f"{base_n / n_gen:.1f} named")
            if acc_path is None:
                for r in [int(x) for x in a.rows.split(",")]:
                    m, n, _ = replay(tr, slots, schedule(n_gen, rows=r))
                    print(f"  rows {r}: disk {m / n_gen:6.2f} a token ({m / base_m:.3f} of one row), "
                          f"RAM {n / n_gen:6.1f} a token ({n / base_n:.3f})")
                continue
            acc = [int(x) for x in open(acc_path).read().split()][:n_gen]
            if len(acc) < n_gen:
                sys.exit(f"{acc_path}: {len(acc)} positions, the trace has {n_gen}")
            for k in [int(x) for x in a.draft.split(",")]:
                sch = schedule(n_gen, accept=acc, k=k)
                m, n, npass = replay(tr, slots, sch)
                rows = sum(r for _, r, _ in sch)
                print(f"  k {k}: {n_gen / npass:.2f} tokens a pass of {rows / npass:.2f} rows; disk "
                      f"{m / n_gen:6.2f} a token ({m / base_m:.3f}), RAM {n / n_gen:6.1f} ({n / base_n:.3f})")
    return 0


def main_p(a):
    """Every trace, slot count, rate p and k: the mean over seeds of tokens a pass, rows a pass, and the disk and
    RAM units a token against the one-row decode (each draft agreeing with probability p, independently)."""
    import random
    print("trace slots p k tokens/pass rows/pass disk_ratio ram_ratio")
    for path in a.traces:
        tr = read_trace(path)
        n_gen = tr["n_tokens"] - tr["n_prompt"]
        for slots in [int(s) for s in a.slots.split(",")]:
            base_m, base_n, _ = replay(tr, slots, schedule(n_gen, rows=1))
            for p in [float(x) for x in a.p.split(",")]:
                for k in [int(x) for x in a.draft.split(",")]:
                    tp = rp = dm = rn = 0.0
                    for seed in range(a.seeds):
                        rng = random.Random(seed)
                        acc = [1 if rng.random() < p else 0 for _ in range(n_gen)]
                        sch = schedule(n_gen, accept=acc, k=k)
                        m, n, npass = replay(tr, slots, sch)
                        tp += n_gen / npass
                        rp += sum(r for _, r, _ in sch) / npass
                        dm += m / base_m
                        rn += n / base_n
                    s = a.seeds
                    print(f"{os.path.basename(path)} {slots} {p} {k} {tp / s:.3f} {rp / s:.3f} {dm / s:.4f} "
                          f"{rn / s:.4f}", flush=True)
    return 0


def main_chain(a):
    """Every trace with its resident_probe output: each variant's agreement (its first draft), mean matched
    length, and for each k the tokens a pass, rows a pass and the disk and RAM units a token against one row."""
    print("trace slots variant agree1 mean_len k tokens/pass rows/pass disk_ratio ram_ratio")
    for path, cpath in zip(a.traces, a.chain.split(",")):
        tr = read_trace(path)
        n_gen = tr["n_tokens"] - tr["n_prompt"]
        slots = int(a.slots)
        base_m, base_n, _ = replay(tr, slots, schedule(n_gen, rows=1))
        for var in a.variants.split(","):
            mt = read_chain(cpath, var)
            ran = [x for x in mt if x >= 0]
            agree1 = sum(1 for x in ran if x >= 1) / max(1, len(ran))
            mean_len = sum(ran) / max(1, len(ran))
            for k in [int(x) for x in a.draft.split(",")]:
                sch = schedule_chain(n_gen, mt, k)
                m, n, npass = replay(tr, slots, sch)
                rows = sum(r for _, r, _ in sch)
                print(f"{os.path.basename(path)} {slots} {var} {agree1:.4f} {mean_len:.3f} {k} {n_gen / npass:.3f} "
                      f"{rows / npass:.3f} {m / base_m:.4f} {n / base_n:.4f}", flush=True)
    return 0


def run_check():
    bad = 0

    def check(ok, msg):
        nonlocal bad
        if not ok:
            bad += 1
            print("FAIL", msg)

    # two units in one slot, alternating: every call misses; two slots hold both
    calls = [(True, [0], [1], 1), (True, [1], [1], 1), (True, [0], [1], 1), (True, [1], [1], 1)]
    check(store_replay(calls, 1, 1) == (4, 4), "alternating units in one slot: 4 misses")
    check(store_replay(calls, 2, 1) == (2, 4), "two slots: 2 misses")
    # the lowest hotness goes: unit 0 named twice, 1 once, then 2 evicts 1 (the colder), 0 still hits
    calls = [(True, [0], [1], 1), (True, [0], [1], 1), (True, [1], [1], 1), (True, [2], [1], 1), (True, [0], [1], 1)]
    check(store_replay(calls, 2, 1) == (3, 5), "the colder unit out")
    # halving by kept tokens: two passes of 16 tokens halve unit 0's heat 2 -> 1 -> 0, so unit 2 evicts 0 (the
    # older of two at 0) and 0 misses again; halving by calls (16 of them) never comes in 4 calls: 1 goes
    calls = [(True, [0], [2], 16), (True, [1], [1], 16), (True, [2], [1], 1), (True, [0], [1], 1)]
    check(store_replay(calls, 2, 1) == (3, 4), "halved by calls: 3 misses")
    POLICY["halve"] = "tokens"
    check(store_replay(calls, 2, 1) == (4, 4), "halved by kept tokens: 4 misses")
    POLICY["halve"] = "calls"
    # a pass's heat: every row's, or only its kept rows' (row 1, rejected, adds nothing to its unit 1)
    ch = [[[0]], [[1]]]
    check([c[2] for c in pass_calls(ch, [0, 1], 1, 2, 1)] == [[1, 1]], "heat from every row")
    POLICY["heat"] = "kept"
    check([c[2] for c in pass_calls(ch, [0, 1], 1, 2, 1)] == [[1, 0]], "heat from the kept rows")
    ch2 = [[[0], [0]], [[1], [1]]]
    check([c[3] for c in pass_calls(ch2, [0, 1], 2, 2, 1)] == [1, 0], "the pass's kept tokens on its first call only")
    POLICY["heat"] = "all"
    # schedules: fixed rows; and drafts judged by accept (position 0 is always exact)
    check(schedule(5, rows=2) == [(0, 2, 2), (2, 2, 2), (4, 1, 1)], "fixed rows")
    check(schedule(6, accept=[0, 1, 0, 1, 1, 0], k=2) == [(0, 3, 2), (2, 3, 3), (5, 1, 1)], "drafts judged")
    check(schedule(4, accept=[0, 0, 0, 0], k=3) == [(0, 4, 1), (1, 3, 1), (2, 2, 1), (3, 1, 1)], "none accepted")
    # chains: a pass at p keeps min(matched[p], k) drafts and the exact token; -1 keeps none
    check(schedule_chain(6, [2, 0, 0, 5, 0, 0], 2) == [(0, 3, 3), (3, 3, 3)], "chains judged")
    check(schedule_chain(3, [-1, 1, 0], 4) == [(0, 1, 1), (1, 2, 2)], "a chain not run: a one-row pass")
    # the resident sets: one slot, a one-layer two-expert trace: the prompt asks 0, the rows 1 then 0
    tr = dict(n_layers=1, n_expert=2, n_prompt=1, n_tokens=3, chosen=[[[0]], [[1]], [[0]]])
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        n = dump_resident(tr, 1, os.path.join(d, "r"))
        got = open(os.path.join(d, "r"), "rb").read()
    check(n == 2 and got == bytes([1, 0, 0, 1]), f"resident sets before each row: {list(got)}")
    print("spec_replay --check:", "ok" if bad == 0 else f"{bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
