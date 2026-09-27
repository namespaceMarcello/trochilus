"""draft_gate_sim.py - the cheap drafts on new code, priced with what a verified row costs.

tools/draft_table_sim.py priced every pass with the costs of 2026-09-18 (an extra row 17.6 ms alone, 13.7
each at eight, of a 31.3 ms pass). This replays the same greedy loop on the same 18 texts
(bench/prompts/new-code, build/draft_table/gen) with three prices:
  measured  the passes of tools/row_price.sh (build/rowprice/prices.json, Q4_K, 8 threads), the byte
            model's increments past its largest pass;
  bytes     the row at its bytes: one token's experts at the one-row pass's MiB/ms times the union's growth
            (U(k) from the code route traces, the pair's new experts by token class, structural or not),
            plus 1.2 ms a row; the KV (0.25 MiB a position, read once a pass) and the attention's compute
            (0.4 ms a row per 1000 positions) at contexts 300, 2048 and 4000 (a model: to be measured);
  09-18     the old constants, for the comparison.
Draft sources: today's prompt lookup (src/gen/lookup.c, adaptive policy of src/gen/greedy.c); the n-gram
table (draft_table_sim.py); llama.cpp's lookup decoding (common/ngram-cache.cpp: a context cache of the text
itself, n = 1..4, lax gate, validated by a static cache of 2-token contexts from the same corpus); a
calibrated gate: each source's P(right) by bin (the context cache by n, occurrences and share; the table by
order and confidence), learned on one half of the prompts and scored on the other (both ways), proposing
token j while P(tokens 1..j right) > the price of row j; the perfect grammar's ceiling.

With a real context (docs/MEASUREMENTS.md question 81; the texts of tools/draft_ctx_gen.sh):
  --set DIR      the texts: DIR/gen/<name>.prompt, .out, .err (default build/draft_table, the new code);
                 DIR/val/<name>.err and .out, the engine stopped at 200 tokens with the lookup fixed at 8:
                 the replay must give its counters exactly (SAME), and its tokens are the 256's first 200
  --repo IDS     this repo's code, one file a line (`tokenize --batch`; the paths in IDS with .files for
                 .ids): the n-gram table of draft_table_sim.py from every file but the prompt's own (leave
                 one out, --sources names it), as a draft (4 MiB, >= 0.7) and as the gate's third source
                 (bins by order, occurrences and share: most of its contexts were seen once)
  --sources TSV  name, file under test (bench/prompts/real-context/sources.tsv)
  --ctx real     every pass priced at its own context (prompt + generated): the measured passes plus the
                 byte model's context terms (the KV read once a pass, the attention's compute a row), and
                 the bytes; beside them both at the 300 positions where the passes were measured
Other prices (question 78: the short pass before and after, and Q8_0's):
  --prices JSON  a prices.json of tools/row_price_report.py (default build/rowprice/prices.json)
  --config CFG   its model and threads: Q4_K-t8 (default), Q8_0-t8, Q8_0-t16
A replay waits while the machine's marker (~/.claude/macchina-ferma) says a native measurement is on.

  python tools/draft_gate_sim.py      (after tools/draft_table_gen.sh, draft_corpus.py and row_price.sh)
  python tools/draft_gate_sim.py --set build/draft_ctx --repo build/draft_ctx/repo.ids \
      --sources bench/prompts/real-context/sources.tsv --ctx real          (after tools/draft_ctx_gen.sh)
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_table_sim as dts  # noqa: E402
from route_union_report import chosen  # noqa: E402

ROOT = dts.ROOT
ROUTE = [ROOT / "build" / "route" / f for f in ("code-1000.bin", "trace-c2.bin", "trace-py.bin", "trace-sh.bin")]
PRICES = ROOT / "build" / "rowprice" / "prices.json"
KV_MIB_POS, BW = 0.25, 49.0  # F32 K and V of a position (OLMoE); MiB/ms of the decode's reads
MARKER = Path(os.environ.get("MACHINE_MARKER", str(Path.home() / ".claude" / "macchina-ferma")))
FLAGS = None


def wait_still():
    """nothing long runs beside a native measurement: the machine's marker (tools/measure_guard.lib)"""
    if MARKER.exists():
        print(f"(waiting: {MARKER} is there)", file=sys.stderr, flush=True)
        while MARKER.exists():
            time.sleep(60)


def whitespace_runs():
    """The vocabulary's added whitespace tokens (runs of 2-24 raw spaces, stored outside the byte alphabet):
    draft_table_sim.structural_flags cannot map them and counts them as names, though they are indentation
    (6.9% of the new code's tokens, 2026-09-27)."""
    import gguf
    r = gguf.GGUFReader(str(ROOT / "models" / "OLMoE-1B-7B-0125-Instruct-Q4_K.gguf"))
    f = r.fields["tokenizer.ggml.tokens"]
    pieces = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
    return {i for i, s in enumerate(pieces) if s and not s.strip()}


# ---------------------------------------------------------------- the cost side


def route_stats():
    """U[k]: the distinct experts of k consecutive tokens over one token's; NEW: the second of a pair's new
    experts (of n_used) by class, S structural (no letter, digit or underscore), N otherwise."""
    per, pair = defaultdict(list), defaultdict(list)
    for p in ROUTE:
        t, ch = chosen(str(p))
        for k in range(1, 18):
            n_win = t.n_tokens // k
            w = ch[:n_win * k].reshape(n_win, k, t.n_layers, t.n_used)
            oh = np.zeros((n_win, t.n_layers, t.n_expert), dtype=bool)
            for j in range(k):
                np.put_along_axis(oh, w[:, j], True, axis=2)
            per[k].append(oh.sum(axis=2).mean() / t.n_used)
        toks = list(t.tokens)
        for i in range(1, t.n_tokens):
            new = np.mean([len(set(ch[i][L]) - set(ch[i - 1][L])) for L in range(t.n_layers)])
            pair[("S" if FLAGS[toks[i - 1]] else "N") + ("S" if FLAGS[toks[i]] else "N")].append(new)
    U = [1.0] + [float(np.mean(per[k])) for k in range(1, 18)]
    return U, {k: float(np.mean(v)) for k, v in pair.items()}, t.n_used


def union_of(rows, U, NEW, n_used, classaware):
    k = len(rows)
    if not classaware:
        return U[k]
    u = 1.0
    for j in range(1, k):
        key = ("S" if FLAGS[rows[j - 1]] else "N") + ("S" if FLAGS[rows[j]] else "N")
        u += NEW[key] / n_used * (U[j + 1] - U[j]) / (U[2] - U[1])
    return u


# Every cost is cost(rows, at): at = the positions already in the KV when the pass runs (prompt + generated),
# which only the prices "at the real context" read.


def bytes_cost(T1, e_ms, ctx, U, NEW, n_used, classaware=True, c=1.2):
    """ctx None: each pass at its own context"""
    def cost(rows, at=None):
        cx = ctx if ctx is not None else at
        t1 = T1 + KV_MIB_POS * max(cx - 300, 0) / BW
        cr = c + 0.4 * max(cx - 300, 0) / 1000.0
        k = len(rows)
        return t1 if k == 1 else t1 + e_ms * (union_of(rows, U, NEW, n_used, classaware) - 1) + (k - 1) * cr
    return cost


def ctx_terms(at, k):
    """what a pass of k rows pays past the 300 positions its price was measured at: the byte model's KV read
    (once a pass) and attention compute (a row)"""
    over = max(at - 300, 0)
    return KV_MIB_POS * over / BW + (k - 1) * 0.4 * over / 1000.0


def with_ctx(cost):
    return lambda rows, at=None: cost(rows) + ctx_terms(at, len(rows))


def measured_cost(ms, plain):
    ks = sorted(ms)

    def at_k(k):
        if k in ms:
            return ms[k]
        if k > ks[-1]:
            return ms[ks[-1]] + plain([0] * k) - plain([0] * ks[-1])
        lo = max(x for x in ks if x < k)
        hi = min(x for x in ks if x > k)
        return ms[lo] + (ms[hi] - ms[lo]) * (k - lo) / (hi - lo)
    return lambda rows, at=None: at_k(len(rows))


def price_of(plain):
    """row j's price, a share of a one-row pass at the same context"""
    return lambda j, at=None: (plain([0] * (j + 1), at) - plain([0] * j, at)) / plain([0], at)


# ---------------------------------------------------------------- the replay (src/gen/greedy.c)


def replay(run, draft_fn, n_draft=8, adaptive=False):
    """The passes of tr_greedy_step until n_gen tokens are out: (rows, position, accepted) each. A draft
    reaching past the tokens the engine wrote is cut there."""
    prompt, gen, n_gen = run["prompt"], run["gen"], run["n_gen"]
    if dts.EOS in gen:
        gen = gen[:gen.index(dts.EOS) + 1]
        n_gen = min(n_gen, len(gen))
    n_p = len(prompt)
    m, k_cur, cool, back = 0, n_draft, 0, 0
    passes = []
    drafted = accepted = 0
    while m < n_gen:
        want = min(k_cur, n_draft) if adaptive else n_draft
        d = draft_fn(prompt + gen[:m + 1], want) if want > 0 else []
        d = d[:max(len(gen) - m - 1, 0)]
        a = 0
        while a < len(d) and d[a] == gen[m + 1 + a]:
            a += 1
        passes.append(([gen[m]] + list(d), n_p + m, a))
        drafted += len(d)
        accepted += a
        m += 1 + a
        if adaptive:
            if want == 0:
                if cool > 0:
                    cool -= 1
                    if cool == 0:
                        k_cur = 1
            elif d:
                if a == len(d):
                    k_cur, back = min(k_cur + 1, n_draft), 0
                elif a > 0:
                    k_cur, back = a, 0
                else:
                    k_cur, back = 0, min(back * 2 + 1, 16)
                    cool = back
    return dict(passes=passes, gen=gen, n_p=n_p, drafted=drafted, accepted=accepted, tokens=m)


def priced(rp, cost):
    """(passes, drafted, accepted, tokens, ms with the draft, ms without)"""
    gen, n_p = rp["gen"], rp["n_p"]
    t = base = 0.0
    for rows, at, a in rp["passes"]:
        m = at - n_p
        t += cost(rows, at)
        for j in range(1 + a):
            base += cost([gen[min(m + j, len(gen) - 1)]], at + j)
    return len(rp["passes"]), rp["drafted"], rp["accepted"], rp["tokens"], t, base


def simulate(run, draft_fn, cost, n_draft=8, adaptive=False):
    return priced(replay(run, draft_fn, n_draft, adaptive), cost)


def agg(res):
    steps, drafted, accepted, m, t, base = np.sum(np.array(res), axis=0)
    return accepted / max(drafted, 1), m / steps, base / t


# ---------------------------------------------------------------- draft sources


def lookup_src(ctx):
    L = len(ctx)
    for n in range(4, 1, -1):
        if L < n + 1:
            continue
        tail = ctx[L - n:]
        for i in range(L - n - 1, -1, -1):
            if ctx[i:i + n] == tail:
                return ctx[i + n], ("L", n)
    return None


def ctx_src(ctx):
    """llama.cpp's context cache: the most frequent next token of the longest n-gram (4..1) seen in the text."""
    L = len(ctx)
    for n in range(4, 0, -1):
        if L < n + 1:
            continue
        tail, cnt = ctx[L - n:], {}
        for i in range(L - n):
            if ctx[i:i + n] == tail:
                cnt[ctx[i + n]] = cnt.get(ctx[i + n], 0) + 1
        if cnt:
            tok = max(cnt, key=cnt.get)
            tot = sum(cnt.values())
            tb = 0 if tot == 1 else 1 if tot == 2 else 2 if tot <= 4 else 3
            return tok, ("C", n, tb, min(int(4 * cnt[tok] / tot), 3))
    return None


def count_bin(tot):
    return 0 if tot == 1 else 1 if tot == 2 else 2 if tot <= 4 else 3 if tot <= 16 else 4


def table_src_of(tb, tag="T", counts=False):
    def src(ctx):
        for n in range(dts.NMAX, 0, -1):
            if len(ctx) < n:
                continue
            t = tb.t[n]
            h = dts.ctx_hash(ctx[-n:])
            i = np.searchsorted(t["key"], h)
            if i < len(t["key"]) and t["key"][i] == h:
                cb = min(int(5 * float(t["best"][i]) / float(t["total"][i])), 4)
                if counts:
                    return int(t["nxt"][i]), (tag, n, count_bin(int(t["total"][i])), cb)
                return int(t["nxt"][i]), (tag, n, cb)
        return None
    return src


def calibrate(train, sources_of):
    hit, cnt = defaultdict(int), defaultdict(int)
    for r in train:
        wait_still()
        sources = sources_of(r)
        P, G = r["prompt"], r["gen"][:r["n_gen"]]
        for m in range(1, len(G)):
            for src in sources:
                p = src(P + G[:m])
                if p is not None:
                    cnt[p[1]] += 1
                    hit[p[1]] += p[0] == G[m]
    pooled = defaultdict(lambda: [0, 0])
    for b in cnt:
        pooled[b[0]][0] += cnt[b]
        pooled[b[0]][1] += hit[b]
    # a bin's rate shrunk toward its source's: a bin seen a few times does not bring its luck
    return {b: (hit[b] + 20 * pooled[b[0]][1] / pooled[b[0]][0]) / (cnt[b] + 20) for b in cnt}


def gate_of(cal, sources, price):
    def draft(ctx, k):
        out, prod, cur = [], 1.0, list(ctx)
        for j in range(1, k + 1):
            best = None
            for src in sources:
                p = src(cur)
                if p is not None and (best is None or cal.get(p[1], 0.0) > best[1]):
                    best = (p[0], cal.get(p[1], 0.0))
            if best is None:
                break
            prod *= best[1]
            if prod <= price(j, len(ctx) - 1):
                break
            out.append(best[0])
            cur.append(best[0])
        return out
    return draft


class Repo:
    """This repo's code, one file a line: the n-gram table of every file but the one a prompt was cut from."""

    def __init__(self, ids_path, sources_path):
        self.ids = Path(ids_path)
        lines = self.ids.read_text().split("\n")
        self.lines = lines[:-1] if lines and lines[-1] == "" else lines
        self.files = [f for f in self.ids.with_suffix(".files").read_text().split("\n") if f]
        if len(self.files) != len(self.lines):
            raise SystemExit(f"{self.ids}: {len(self.lines)} lines for {len(self.files)} files")
        self.under_test = {}
        for line in Path(sources_path).read_text().splitlines() if sources_path else []:
            if line and not line.startswith("#"):
                f = line.split("\t")
                if f[1] not in self.files:
                    raise SystemExit(f"{f[1]} ({f[0]}) is not in {self.ids.with_suffix('.files')}: nothing to leave out")
                self.under_test[f[0]] = f[1]
        self.key, self.cache = (), None

    def tables(self, run):
        """(the full table, the 4 MiB one) without the run's own file"""
        path = self.under_test.get(run["name"])
        if self.cache is None or path != self.key:
            # a file of this process's own: two replays side by side shared one name and read each other's
            fd, tmp = tempfile.mkstemp(suffix=".loo", dir=self.ids.parent)
            with os.fdopen(fd, "w") as out:
                out.write("\n".join(l for f, l in zip(self.files, self.lines) if f != path) + "\n")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    full = dts.build_tables(tmp)
            finally:
                os.unlink(tmp)
            self.cache, self.key = (dts.Table(full), dts.Table(dts.prune(full, 4 << 20))), path
        return self.cache

    def describe(self):
        full, small = self.tables({"name": None})
        tokens = sum(l.count(",") + 1 for l in self.lines)
        thr = min(int(t["total"].min()) for t in small.t.values() if len(t["total"]))
        return (f"repo table: {len(self.files)} files, {tokens / 1e6:.2f} M tokens; full {dts.table_size(full.t) / 2**20:.1f} "
                f"MiB, the '4 MiB' one keeps the contexts seen >= {thr} times ({dts.table_size(small.t) / 2**20:.1f} MiB); "
                f"each prompt's own file left out")


# ---------------------------------------------------------------- llama.cpp's lookup (common/ngram-cache.cpp)


LAX_N, LAX_P = [2, 2, 1, 1], [66, 50, 50, 50]
V = 1 << 17


class Static:
    """2-token contexts of the corpus with their next tokens' counts (LLAMA_NGRAM_STATIC = 2)."""

    def __init__(self, path):
        seqs = [np.array(l.strip().split(","), dtype=np.int64) for l in open(path) if l.strip()]
        ids = np.concatenate([np.concatenate([s, [-1]]) for s in seqs if len(s)])
        ok = (ids[:-2] >= 0) & (ids[1:-1] >= 0) & (ids[2:] >= 0)
        k, self.cnt = np.unique(((ids[:-2] * V + ids[1:-1]) * V + ids[2:])[ok], return_counts=True)
        self.ctx, self.nxt = k // V, k % V

    def part(self, a, b):
        lo, hi = np.searchsorted(self.ctx, a * V + b, "left"), np.searchsorted(self.ctx, a * V + b, "right")
        return {int(t): int(c) for t, c in zip(self.nxt[lo:hi], self.cnt[lo:hi])}


def llama_lookup(st):
    cache, seen = defaultdict(lambda: defaultdict(int)), [0]

    def primary(ngrams, ps):
        for i in range(len(ngrams) - 1, -1, -1):
            part = cache.get(ngrams[i])
            if not part:
                continue
            best, bcp, bcs, tot = None, 0, 0, 0
            for tok, cp in part.items():
                cs = 100 * ps[tok] if tok in ps else 1
                if cp * cs > bcp * bcs:
                    best, bcp, bcs = tok, cp, cs
                tot += cp
            if tot >= LAX_N[i] and 100 * bcp >= LAX_P[i] * tot:
                return best
        return None

    def fn(ctx, k):
        for i in range(max(seen[0], 1), len(ctx)):  # the context cache follows the accepted text
            for n in range(1, 5):
                if i - n >= 0:
                    cache[tuple(ctx[i - n:i])][ctx[i]] += 1
        seen[0] = len(ctx)
        out = []
        while len(out) < k:
            cur = ctx + out
            ps = st.part(cur[-2], cur[-1])
            tok = primary([tuple(cur[-n:]) for n in range(1, 5)], ps)
            if tok is None and ps:
                best = max(ps, key=ps.get)
                tot = sum(ps.values())
                tok = best if tot >= LAX_N[1] and 100 * ps[best] >= LAX_P[1] * tot else None
            if tok is None:
                break
            out.append(tok)
        return out
    return fn


# ---------------------------------------------------------------- main


def validate(set_dir, runs):
    """the replay against the engine's own counters: stopped at 200 tokens, replayed on the 256 it wrote"""
    vals = sorted((set_dir / "val").glob("*.err"))
    if vals:
        print("\n== validation: lookup fixed at 8, the engine stopped at 200 tokens, the replay on the 256 it wrote")
    for vf in vals:
        r = next(x for x in runs if x["name"] == vf.stem)
        s = dts.simulate(r, lambda c, k: dts.lookup_draft(c, k), 8, False, n_gen=200, exact_tail=True)
        eng = next(l for l in vf.read_text().splitlines() if l.startswith("speculation:"))
        same = f"{s['steps']} passes, {s['accepted']}/{s['drafted']} drafted" in eng
        out = next(l for l in vf.with_suffix(".out").read_text().splitlines() if l.startswith("tokens:"))
        toks = dts.read_ids(out[len("tokens:"):])
        tsame = len(toks) == 200 and toks == r["gen"][:200]
        print(f"  {r['name']:24s} replay {s['steps']} passes {s['accepted']}/{s['drafted']} | engine "
              f"{eng.split('speculation: ')[-1].split(' drafted')[0]} {'SAME' if same else 'DIFFERENT'}; "
              f"its 200 tokens {'= the first 200 of the 256' if tsame else 'DIFFER from the 256'}")


def main():
    global FLAGS
    ap = argparse.ArgumentParser(description="the cheap drafts replayed and priced (see the module's head)")
    ap.add_argument("--set", default=str(dts.HERE), help="texts: DIR/gen, DIR/val (default build/draft_table)")
    ap.add_argument("--repo", help="this repo's code, one file a line (IDS; the paths in IDS with .files)")
    ap.add_argument("--sources", help="name -> file under test (bench/prompts/real-context/sources.tsv)")
    ap.add_argument("--ctx", choices=("fixed", "real"), default="fixed", help="real: each pass at its context")
    ap.add_argument("--prices", default=str(PRICES), help="tools/row_price_report.py's prices.json (default build/rowprice)")
    ap.add_argument("--config", default="Q4_K-t8", help="the prices' model and threads: Q4_K-t8 (default), Q8_0-t8, Q8_0-t16")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")  # a name with a character the console lacks (build/draft_table/gen)
    wait_still()
    FLAGS = dts.structural_flags()
    prices = Path(args.prices)
    if not prices.exists():
        print(f"{prices} is missing: run tools/row_price.sh and tools/row_price_report.py first")
        return 1
    pr = json.load(open(prices))
    q4 = pr["configs"][args.config]
    ms = {int(k): v for k, v in q4["ms_by_rows"].items()}
    U, NEW, n_used = route_stats()
    print(f"prices: {prices} of {pr['measured']} ({pr['binary']}), {args.config}: " +
          ", ".join(f"{k} rows {v:.2f} ms" for k, v in sorted(ms.items())))
    print("U(k), code route traces: " + " ".join(f"{U[k]:.2f}" for k in range(1, 10)))
    print("new experts of the second of a pair: " + ", ".join(f"{k} {v:.2f}/{n_used}" for k, v in sorted(NEW.items())))

    T1, e = ms[1], q4["experts_ms_one_row"]
    plain300 = bytes_cost(T1, e, 300, U, NEW, n_used, classaware=False)
    meas = measured_cost(ms, plain300)
    if args.ctx == "real":
        models = [("measured", meas, meas),
                  ("measured, real ctx", with_ctx(meas), with_ctx(meas)),
                  ("bytes 300", bytes_cost(T1, e, 300, U, NEW, n_used), plain300),
                  ("bytes, real ctx", bytes_cost(T1, e, None, U, NEW, n_used),
                   bytes_cost(T1, e, None, U, NEW, n_used, classaware=False))]
    else:
        old = lambda rows, at=None: dts.cost_ms(len(rows) - 1)  # noqa: E731
        models = [("09-18", old, old), ("measured", meas, meas)]
        for ctx in (300, 2048, 4000):
            models.append((f"bytes {ctx}", bytes_cost(T1, e, ctx, U, NEW, n_used),
                           bytes_cost(T1, e, ctx, U, NEW, n_used, classaware=False)))

    set_dir = Path(args.set)
    dts.GEN = set_dir / "gen"
    runs = dts.load_runs()
    n_p = [len(r["prompt"]) for r in runs]
    print(f"texts: {len(runs)} in {dts.GEN}, prompts of {min(n_p)}-{max(n_p)} tokens (mean {np.mean(n_p):.0f}), "
          f"{sum(len(r['gen']) for r in runs)} generated")
    validate(set_dir, runs)
    ws = whitespace_runs()
    flags_ws = [fl or i in ws for i, fl in enumerate(FLAGS)]
    st_tok = sum(FLAGS[t] for r in runs for t in r["gen"][:r["n_gen"]])
    ws_tok = sum(t in ws for r in runs for t in r["gen"][:r["n_gen"]])
    tot_tok = sum(len(r["gen"][:r["n_gen"]]) for r in runs)
    print(f"structural tokens (no letter, digit or underscore): {st_tok}/{tot_tok} = {100 * st_tok / tot_tok:.1f}%; "
          f"with the whitespace runs the flags miss ({len(ws)} tokens of raw spaces, {100 * ws_tok / tot_tok:.1f}%): "
          f"{100 * (st_tok + ws_tok) / tot_tok:.1f}%")

    z = np.load(dts.HERE / "tables.npz")
    tables = {n: {f: z[f"{n}_{f}"] for f in ("key", "nxt", "best", "total")} for n in range(1, dts.NMAX + 1)}
    t4, t16 = dts.Table(dts.prune(tables, 4 << 20)), dts.Table(dts.prune(tables, 16 << 20))
    del tables, z
    st = Static(dts.HERE / "corpus.ids")
    repo = Repo(args.repo, args.sources) if args.repo else None
    if repo:
        print(repo.describe())

    halves = ([r for i, r in enumerate(runs) if i % 2 == 0], [r for i, r in enumerate(runs) if i % 2 == 1])
    half = {r["name"]: i % 2 for i, r in enumerate(runs)}
    gates = {"gate": lambda r: (table_src_of(t16), ctx_src)}
    if repo:
        gates["gate+repo"] = lambda r: (table_src_of(t16), ctx_src, table_src_of(repo.tables(r)[0], "R", True))
    cal = {g: (calibrate(halves[0], f), calibrate(halves[1], f)) for g, f in gates.items()}

    cfgs = [("prompt lookup, fixed 8", "lookup", 8, False)] if args.ctx == "real" else []
    cfgs += [("lookup, adaptive (the engine today)", "lookup", 8, True),
             ("table 4 MiB >= 0.7 (best of draft_table_sim)", "t4", 8, False)]
    if repo:
        cfgs.append(("repo table 4 MiB >= 0.7 (leave one out)", "r4", 8, False))
    cfgs += [("llama.cpp lookup, draft 15", "llama", 15, False),
             ("llama.cpp lookup, draft 3", "llama", 3, False),
             ("gate: context cache + table (out of sample)", "gate", 8, False)]
    if repo:
        cfgs.append(("gate: + the repo table (out of sample)", "gate+repo", 8, False))
    cfgs.append(("perfect grammar (ceiling)", "grammar", 8, False))
    if args.ctx == "real":
        cfgs.append(("perfect grammar, whitespace runs structural", "grammar_ws", 8, False))

    res = defaultdict(list)
    for r in runs:
        wait_still()
        for ci, (name, kind, nd, adaptive) in enumerate(cfgs):
            if kind in gates:  # the gate's draft depends on the price
                for mi, (_, cost, plain) in enumerate(models):
                    fn = gate_of(cal[kind][1 - half[r["name"]]], gates[kind](r), price_of(plain))
                    res[ci, mi].append(simulate(r, fn, cost, nd))
                continue
            if kind == "grammar":
                fn = dts.grammar_oracle(r, FLAGS)
            elif kind == "grammar_ws":
                fn = dts.grammar_oracle(r, flags_ws)
            elif kind == "llama":
                fn = llama_lookup(st)
            elif kind == "lookup":
                fn = lambda c, k: dts.lookup_draft(c, k)  # noqa: E731
            elif kind == "t4":
                fn = lambda c, k: t4.draft(c, k, 0.7)  # noqa: E731
            else:
                r4 = repo.tables(r)[1]
                fn = lambda c, k, r4=r4: r4.draft(c, k, 0.7)  # noqa: E731
            rp = replay(r, fn, nd, adaptive)
            for mi, (_, cost, plain) in enumerate(models):
                res[ci, mi].append(priced(rp, cost))

    label = f"{set_dir.name}, {len(runs)} texts"
    print(f"\n{label:46s} | " + " | ".join(f"{m[0]:>21s}" for m in models))
    for ci, (name, *_) in enumerate(cfgs):
        cells = []
        for mi in range(len(models)):
            acc, tpp, sp = agg(res[ci, mi])
            cells.append(f"{sp:.3f}x {tpp:.2f}t {100 * acc:3.0f}%")
        print(f"{name:46s} | " + " | ".join(f"{c:>21s}" for c in cells))
    print("(each cell: estimated speed against no draft, tokens a pass, drafted tokens accepted)")

    if args.ctx == "real":
        mi = [m[0] for m in models].index("measured, real ctx")
        names = {"t4": "table", "r4": "repo", "gate": "gate", "gate+repo": "gate+R", "grammar": "grammar",
                 "grammar_ws": "gram+ws"}
        short = [("look ad" if ad else f"look f{nd}") if kind == "lookup" else f"llama{nd}" if kind == "llama"
                 else names[kind] for _, kind, nd, ad in cfgs]
        print(f"\nper text: prompt tokens, structural share (the flags'); tokens a pass (the gates at '{models[mi][0]}'); "
              f"the speed there of the last gate")
        print(f"{'':24s} {'prompt':>6s} {'struct':>6s} " + " ".join(f"{s:>7s}" for s in short) + "    speed")
        for i, r in enumerate(runs):
            g = r["gen"][:r["n_gen"]]
            tpp = [res[ci, mi][i][3] / res[ci, mi][i][0] for ci in range(len(cfgs))]
            last = max(ci for ci, c in enumerate(cfgs) if c[1] in gates)
            p = res[last, mi][i]
            print(f"{r['name']:24s} {len(r['prompt']):6d} {100 * sum(FLAGS[t] for t in g) / len(g):5.1f}% " +
                  " ".join(f"{x:7.2f}" for x in tpp) + f"  {p[5] / p[4]:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
