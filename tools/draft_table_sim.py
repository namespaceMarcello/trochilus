"""tools/draft_table_sim.py - how much would a cheap, non-neural draft guess of what the model writes?

Replays Trochilus's greedy speculative loop (src/gen/greedy.c) on token sequences the engine
already generated, with different draft sources:
  lookup  - src/gen/lookup.c replicated (what Trochilus has today), validated against the
            engine's own counters on the same runs;
  table   - an n-gram table (contexts of 4..1 tokens -> most frequent next token) built on
            public code already on this machine, pruned to a size budget;
  combo   - lookup when it proposes something, else the table.
A draft token is accepted only while it equals what the model wrote: the output never changes.
Speed is estimated with the costs Trochilus measured natively (docs/MEASUREMENTS.md
§Adaptive draft, §Revisione): a one-row pass 31.3 ms, an extra row 17.6 ms alone, 13.7 ms
each when eight.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "build" / "draft_table"  # tools/draft_table_gen.sh, tools/draft_corpus.py
GEN = HERE / "gen"
EOS = 0
P = np.uint64(1099511628211)
NMAX = 4

# ---------------------------------------------------------------- data


def read_ids(line):
    line = line.strip()
    return [int(x) for x in line.split(",")] if line else []


def load_runs():
    runs = []
    for pf in sorted(GEN.glob("*.prompt")):
        b = pf.stem
        prompt = read_ids(pf.read_text())
        out = (GEN / f"{b}.out").read_text()
        line = next(l for l in out.splitlines() if l.startswith("tokens:"))
        gen = read_ids(line[len("tokens:"):])
        err = (GEN / f"{b}.err").read_text()
        stats = next((l for l in err.splitlines() if l.startswith("speculation:")), "")
        n_gen = 256
        runs.append(dict(name=b, prompt=prompt, gen=gen, n_gen=n_gen, stats=stats))
    return runs


# ---------------------------------------------------------------- n-gram table


MASK = (1 << 64) - 1


def ctx_hash(tokens):
    h = len(tokens)
    for t in tokens:
        h = (h * int(P) + t + 1) & MASK
    return np.uint64(h)


def build_tables(corpus_path):
    seqs = [np.array(l.strip().split(","), dtype=np.int64) for l in open(corpus_path) if l.strip()]
    arr = np.concatenate([np.concatenate([s, [-1]]) for s in seqs if len(s)])
    print(f"corpus: {len(seqs)} files, {int((arr >= 0).sum())/1e6:.1f} M tokens", flush=True)
    tables = {}
    u = (arr + 1).astype(np.uint64)
    for n in range(1, NMAX + 1):
        N = len(arr) - n
        h = np.full(N, np.uint64(n), dtype=np.uint64)
        ok = arr[n:] >= 0
        with np.errstate(over="ignore"):
            for j in range(n):
                h = h * P + u[j:j + N]
                ok &= arr[j:j + N] >= 0
        nxt = arr[n:][ok]
        h = h[ok]
        order = np.lexsort((nxt, h))
        h, nxt = h[order], nxt[order]
        del order
        # runs of equal (context, next)
        brk = np.ones(len(h), dtype=bool)
        brk[1:] = (h[1:] != h[:-1]) | (nxt[1:] != nxt[:-1])
        starts = np.flatnonzero(brk)
        cnt = np.diff(np.append(starts, len(h)))
        ph, pn = h[starts], nxt[starts]
        # per context: total and the most frequent next (first on ties = smallest id)
        cb = np.ones(len(ph), dtype=bool)
        cb[1:] = ph[1:] != ph[:-1]
        cs = np.flatnonzero(cb)
        total = np.add.reduceat(cnt, cs)
        best = np.maximum.reduceat(cnt, cs)
        grp = np.repeat(np.arange(len(cs)), np.diff(np.append(cs, len(ph))))
        is_best = cnt == best[grp]
        first_best = np.full(len(cs), -1, dtype=np.int64)
        idx = np.flatnonzero(is_best)
        g = grp[idx]
        keep = np.ones(len(idx), dtype=bool)
        keep[1:] = g[1:] != g[:-1]
        first_best[g[keep]] = pn[idx[keep]]
        tables[n] = dict(key=ph[cs], nxt=first_best, best=best, total=total)
        print(f"  n={n}: {len(cs)/1e6:.2f} M contexts", flush=True)
        del h, nxt, brk, starts, cnt, ph, pn, cb, grp, is_best
    return tables


ENTRY_BYTES = 16  # 8-byte key, 4-byte next token, 4-byte counts (best share)


def prune(tables, budget_bytes):
    """Keeps the contexts seen most often, across every n, within the budget."""
    if budget_bytes is None:
        return tables
    allc = np.concatenate([t["total"] for t in tables.values()])
    k = min(len(allc), budget_bytes // ENTRY_BYTES)
    thr = np.partition(allc, len(allc) - k)[len(allc) - k] if k < len(allc) else 0
    out = {}
    for n, t in tables.items():
        m = t["total"] >= thr
        out[n] = {f: v[m] for f, v in t.items()}
    return out


def table_size(tables):
    return sum(len(t["key"]) for t in tables.values()) * ENTRY_BYTES


class Table:
    def __init__(self, tables, min_total=1):
        self.t = tables
        self.min_total = min_total

    def predict(self, ctx):
        """(next, confidence) from the longest context found, or None."""
        for n in range(NMAX, 0, -1):
            if len(ctx) < n:
                continue
            t = self.t[n]
            h = ctx_hash(ctx[-n:])
            i = np.searchsorted(t["key"], h)
            if i < len(t["key"]) and t["key"][i] == h and t["total"][i] >= self.min_total:
                return int(t["nxt"][i]), float(t["best"][i]) / float(t["total"][i])
        return None

    def draft(self, ctx, k, tau):
        cur = list(ctx[-NMAX:])
        out, p = [], 1.0
        for _ in range(k):
            r = self.predict(cur)
            if r is None:
                break
            tok, conf = r
            p *= conf
            if p < tau:
                break
            out.append(tok)
            cur = cur[1:] + [tok] if len(cur) >= NMAX else cur + [tok]
        return out


# ---------------------------------------------------------------- drafts


def lookup_draft(ctx, k, nmax=4, nmin=2):
    """src/gen/lookup.c, line for line."""
    if k <= 0:
        return []
    L = len(ctx)
    for n in range(nmax, nmin - 1, -1):
        if L < n + 1:
            continue
        tail = ctx[L - n:]
        for i in range(L - n - 1, -1, -1):
            if ctx[i + n - 1] != tail[n - 1]:
                continue
            if ctx[i:i + n] != tail:
                continue
            avail = L - (i + n)
            kk = min(avail, k)
            return ctx[i + n:i + n + kk]
    return []


# ---------------------------------------------------------------- the greedy loop


def cost_ms(extra):
    if extra == 0:
        return 31.3
    r = 17.6 + (extra - 1) * (13.7 - 17.6) / 7
    return 31.3 + extra * r


def simulate(run, draft_fn, n_draft=8, adaptive=False, n_gen=None, exact_tail=False):
    """Replays tr_greedy_step until n_gen tokens are out. A draft reaching past the tokens the
    engine wrote is cut there (exact_tail: the replay stops instead, for validation)."""
    prompt, gen = run["prompt"], run["gen"]
    n_gen = n_gen or run["n_gen"]
    if EOS in gen:
        gen = gen[:gen.index(EOS) + 1]
        n_gen = min(n_gen, len(gen))
    m = 0  # tokens emitted; gen[m] is `next`
    k_cur, cool, back = n_draft, 0, 0
    steps = drafted = accepted = 0
    t_ms = 0.0
    runs_len = []
    while m < n_gen:
        want = min(k_cur, n_draft) if adaptive else n_draft
        ctx = prompt + gen[:m + 1]
        d = draft_fn(ctx, want) if want > 0 else []
        if m + 1 + len(d) > len(gen):
            if exact_tail:
                break
            d = d[:len(gen) - m - 1]  # past the known tokens: what the model says there is unknown
        a = 0
        while a < len(d) and d[a] == gen[m + 1 + a]:
            a += 1
        steps += 1
        drafted += len(d)
        accepted += a
        t_ms += cost_ms(len(d))
        if d:
            runs_len.append(a)
        m += 1 + a
        if adaptive:
            if want == 0:
                if cool > 0:
                    cool -= 1
                    if cool == 0:
                        k_cur = 1
            elif d:
                if a == len(d):
                    k_cur = min(k_cur + 1, n_draft)
                    back = 0
                elif a > 0:
                    k_cur = a
                    back = 0
                else:
                    k_cur = 0
                    back = min(back * 2 + 1, 16)
                    cool = back
    base_ms = m * cost_ms(0)
    return dict(steps=steps, drafted=drafted, accepted=accepted, tokens=m, t_ms=t_ms, base_ms=base_ms)


def agg(results):
    s = {k: sum(r[k] for r in results) for k in ("steps", "drafted", "accepted", "tokens", "t_ms", "base_ms")}
    s["acc"] = s["accepted"] / s["drafted"] if s["drafted"] else 0.0
    s["tpp"] = s["tokens"] / s["steps"]
    s["speed"] = s["base_ms"] / s["t_ms"]
    return s


def fmt(s):
    return (f"accepted {s['accepted']:5d}/{s['drafted']:5d} = {100*s['acc']:5.1f}%  "
            f"tokens/pass {s['tpp']:.2f}  est. speed {s['speed']:.2f}x")


# ---------------------------------------------------------------- grammar ceiling


def structural_flags():
    """True for tokens with no letter, digit or underscore: whitespace, brackets, operators,
    punctuation. The most a grammar that knows the language but not the program could guess."""
    import gguf
    r = gguf.GGUFReader(str(ROOT / "models" / "OLMoE-1B-7B-0125-Instruct-Q4_K.gguf"))
    f = r.fields["tokenizer.ggml.tokens"]
    pieces = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
    # GPT-2 byte-level alphabet back to bytes
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    dec = {chr(c): b for b, c in zip(bs, cs)}
    flags = []
    for pc in pieces:
        try:
            text = bytes(dec[ch] for ch in pc).decode("utf-8", "replace")
        except KeyError:
            flags.append(False)
            continue
        flags.append(len(text) > 0 and not any(ch.isalnum() or ch == "_" for ch in text))
    return flags


def grammar_oracle(run, flags):
    """A perfect grammar: proposes the true next tokens as long as they are structural."""
    n_p = len(run["prompt"])
    gen = run["gen"]

    def fn(ctx, k):
        m1 = len(ctx) - n_p  # index in gen of the first drafted token
        out = []
        while len(out) < k and m1 + len(out) < len(gen) and flags[gen[m1 + len(out)]]:
            out.append(gen[m1 + len(out)])
        return out
    return fn


# ---------------------------------------------------------------- main


def main():
    runs = load_runs()
    new = runs
    print(f"runs: {len(runs)} ({len(new)} new code), generated tokens: {sum(len(r['gen']) for r in new)}")

    print("\n== validation: lookup, fixed 8, engine stopped at 200 tokens, replay on the 256 it wrote")
    for vf in sorted((HERE / "val").glob("*.err")):
        r = next(x for x in runs if x["name"] == vf.stem)
        s = simulate(r, lambda c, k: lookup_draft(c, k), 8, False, n_gen=200, exact_tail=True)
        eng = next(l for l in vf.read_text().splitlines() if l.startswith("speculation:"))
        same = f"{s['steps']} passes, {s['accepted']}/{s['drafted']} drafted" in eng
        print(f"  {r['name']:16s} replay {s['steps']} passes {s['accepted']}/{s['drafted']} | engine "
              f"{eng.split('speculation: ')[-1].split(' drafted')[0]} {'SAME' if same else 'DIFFERENT'}")

    flags = structural_flags()
    st = sum(flags[t] for r in new for t in r["gen"][:r["n_gen"]])
    tot = sum(len(r["gen"][:r["n_gen"]]) for r in new)
    print(f"\n== grammar ceiling: {st}/{tot} = {100*st/tot:.1f}% of the new code's tokens are structural")
    s = agg([simulate(r, grammar_oracle(r, flags), 8, False) for r in new])
    print(f"  perfect grammar (always right on structure)  {fmt(s)}")

    cache = HERE / "tables.npz"
    if cache.exists():
        z = np.load(cache)
        tables = {n: {f: z[f"{n}_{f}"] for f in ("key", "nxt", "best", "total")} for n in range(1, NMAX + 1)}
    else:
        tables = build_tables(HERE / "corpus.ids")
        np.savez(cache, **{f"{n}_{f}": v for n, t in tables.items() for f, v in t.items()})
    print(f"full table: {table_size(tables)/2**20:.0f} MiB")

    def report(label, fn, adaptive=False, n_draft=8, subset=new):
        s = agg([simulate(r, fn, n_draft, adaptive) for r in subset])
        print(f"  {label:44s} {fmt(s)}")
        return s

    print(f"\n== new code ({len(new)} prompts), fixed draft of 8 unless said")
    report("lookup (today)", lambda c, k: lookup_draft(c, k))
    report("lookup (today), adaptive", lambda c, k: lookup_draft(c, k), adaptive=True)
    for budget, name in ((4 << 20, "4 MiB"), (16 << 20, "16 MiB"), (64 << 20, "64 MiB"), (None, "full")):
        tb = Table(prune(tables, budget))
        size = table_size(tb.t) / 2**20
        print(f" table {name} ({size:.0f} MiB)")
        for tau in (0.0, 0.3, 0.5, 0.7):
            report(f"   table, stop below confidence {tau}", lambda c, k, tb=tb, tau=tau: tb.draft(c, k, tau))
        report("   table, adaptive", lambda c, k, tb=tb: tb.draft(c, k, 0.0), adaptive=True)
        for tau in (0.3, 0.5):
            def combo(c, k, tb=tb, tau=tau):
                d = lookup_draft(c, k)
                return d if d else tb.draft(c, k, tau)
            report(f"   lookup, else table (conf {tau})", combo)
            report(f"   lookup, else table (conf {tau}), adaptive", combo, adaptive=True)
        for kk in (1, 2, 4):
            report(f"   table conf 0.5, draft of {kk}", lambda c, k, tb=tb: tb.draft(c, k, 0.5), n_draft=kk)



if __name__ == "__main__":
    main()
