"""token_timeline.py -- where a decode token's time goes against the bytes it reads, from a
TR_POOL_TRACE build's file (threads.c; docs/MEASUREMENTS.md §The engine read as entangled pairs).

    tools/.venv/Scripts/python.exe tools/token_timeline.py <trace> [--bw GBs] [--skip K]

Each trace line is one parallel_for: "n chunks t0 t1 (start end) x chunks bytes rows cols", times in
microseconds; bytes/rows/cols come from TR_TRACE_NOTE (a weight's shape, rows -1 for the decode's
attention with cols its positions; 0 0 0 for calls that read no weight). A decode token is the calls
from one q projection with one input row to the next; its wall runs from its first call's dispatch
to the next token's, so the code between calls (and between tokens) is the gap. Against --bw, the
GB/s a plain read of the same bytes gets on the same threads, every call's loss is its wall minus
bytes / bw: the time the RAM could have been reading and was not. --skip drops the first K decode
tokens (the pool and the caches settling).
"""
import statistics
import sys


def load(path):
    calls = []
    for line in open(path):
        v = line.split()
        n, chunks = int(v[0]), int(v[1])
        t0, t1 = float(v[2]), float(v[3])
        tail = v[4 + 2 * chunks:]
        b, r, c = (int(tail[0]), int(tail[1]), int(tail[2])) if len(tail) >= 3 else (0, 0, 0)
        calls.append({"n": n, "chunks": chunks, "t0": t0, "t1": t1, "bytes": b, "rows": r, "cols": c})
    return calls


def tokens(calls):
    """Decode tokens: a q projection (the first weight of a layer-0 attention) with n == rows opens
    one. The first weight call after a head call is layer 0's q."""
    head_rows = max((c["rows"] for c in calls), default=0)
    out, cur, after_head = [], None, True
    for c in calls:
        if c["rows"] > 0 and after_head:
            after_head = False
            if cur is not None:
                out.append(cur)
            cur = [c] if c["n"] == c["rows"] else None
            continue
        if cur is not None:
            cur.append(c)
        if c["rows"] == head_rows:
            after_head = True
    if cur is not None:
        out.append(cur)
    return out


def name_of(c, seen):
    if c["rows"] == -1:
        return "attention"
    if c["rows"] == 0:
        return "no weight (n=%d, %s)" % (c["n"], "inline" if c["chunks"] == 1 else "pool")
    key = (c["rows"], c["cols"])
    k = seen.get(key, 0)
    seen[key] = k + 1
    return "weight %dx%d" % key


def main():
    args = [a for a in sys.argv[1:]]
    bw, skip = 53.0, 4
    if "--bw" in args:
        i = args.index("--bw"); bw = float(args[i + 1]); del args[i:i + 2]
    if "--skip" in args:
        i = args.index("--skip"); skip = int(args[i + 1]); del args[i:i + 2]
    calls = load(args[0])
    toks = tokens(calls)
    if len(toks) <= skip + 1:
        sys.exit("token_timeline: %d decode tokens, need more than --skip + 1" % len(toks))
    # a token's wall ends where the next one begins: the last token has no end, dropped
    toks = toks[skip:]
    rows = []
    per = {}
    for i in range(len(toks) - 1):
        t = toks[i]
        wall = toks[i + 1][0]["t0"] - t[0]["t0"]
        in_calls = sum(c["t1"] - c["t0"] for c in t)
        nbytes = sum(c["bytes"] for c in t)
        seen = {}
        for c in t:
            nm = name_of(c, seen)
            d = per.setdefault(nm, {"wall": [], "bytes": [], "chunks": c["chunks"]})
            d["wall"].append(c["t1"] - c["t0"])
            d["bytes"].append(c["bytes"])
        ctx = max((c["cols"] for c in t if c["rows"] == -1), default=0)
        rows.append((wall, in_calls, nbytes, len(t), ctx))
    n_tok = len(rows)
    wall = statistics.median(r[0] for r in rows)
    in_calls = statistics.median(r[1] for r in rows)
    nbytes = statistics.median(r[2] for r in rows)
    ncalls = statistics.median(r[3] for r in rows)
    ideal = nbytes / (bw * 1e3)  # bytes / (GB/s) in microseconds
    print("%d decode tokens (context %d-%d), medians a token; plain read %.1f GB/s" %
          (n_tok, rows[0][4], rows[-1][4], bw))
    print("  wall %.0f us, in calls %.0f us (%.1f%%), between calls %.0f us (%.1f%%)" %
          (wall, in_calls, 100 * in_calls / wall, wall - in_calls, 100 * (wall - in_calls) / wall))
    print("  %d calls, %.1f MiB read, %.1f GB/s over the token; at the plain read %.0f us: the RAM idle %.0f us (%.1f%%)" %
          (ncalls, nbytes / 2**20, nbytes / wall / 1e3, ideal, wall - ideal, 100 * (wall - ideal) / wall))
    print()
    print("%-34s %6s %8s %9s %9s %7s %9s" % ("call", "/token", "wall us", "MiB", "GB/s", "chunks", "loss us"))
    lines = []
    for nm, d in per.items():
        cnt = len(d["wall"]) / n_tok
        w = statistics.median(d["wall"])
        b = statistics.median(d["bytes"])
        gbs = b / w / 1e3 if w > 0 and b > 0 else 0.0
        loss = cnt * (w - b / (bw * 1e3))
        lines.append((loss, nm, cnt, w, b, gbs, d["chunks"]))
    lines.sort(reverse=True)
    for loss, nm, cnt, w, b, gbs, ch in lines:
        print("%-34s %6.1f %8.1f %9.2f %9.1f %7d %9.0f" % (nm, cnt, w, b / 2**20, gbs, ch, loss))
    print("%-34s %6s %8s %9s %9s %7s %9.0f" % ("between calls", "", "", "", "", "", wall - in_calls))


if __name__ == "__main__":
    main()
