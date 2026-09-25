"""pool_trace.py -- where a parallel_for's time goes, from a TR_POOL_TRACE build's file (threads.c):
one line per call, "n chunks t_dispatch t_done start0 end0 start1 end1 ..." in microseconds.

    tools/.venv/Scripts/python.exe tools/pool_trace.py <trace> [<trace> ...] [--min-calls K]

Per shape of call (n, chunks), over every call of that shape: calls, the median wall time
(dispatch to the dispatcher's return), and how it splits: the mean chunk's work, the slowest
chunk's work, the start lag of the last chunk to start, and the tail (the last chunk's end minus
the mean chunk's end). "balanced" is the mean chunk's work over the wall: what a perfect split of
the same work would take, as a fraction of what it took. Shapes sorted by their total time.
"""
import statistics
import sys


def load(path):
    calls = []
    for line in open(path):
        v = line.split()
        n, chunks = int(v[0]), int(v[1])
        t0, t1 = float(v[2]), float(v[3])
        ch = [(float(v[4 + 2 * k]), float(v[5 + 2 * k])) for k in range(chunks)]
        calls.append((n, chunks, t0, t1, ch))
    return calls


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    min_calls = 50
    if "--min-calls" in sys.argv:
        min_calls = int(sys.argv[sys.argv.index("--min-calls") + 1])
        args.remove(str(min_calls))
    shapes = {}
    for path in args:
        for n, chunks, t0, t1, ch in load(path):
            wall = t1 - t0
            work = [e - s for s, e in ch]
            mean_work = sum(work) / chunks
            lag = max(s for s, _ in ch) - t0
            tail = max(e for _, e in ch) - statistics.mean(e for _, e in ch)
            shapes.setdefault((n, chunks), []).append((wall, mean_work, max(work), lag, tail))
    total = sum(sum(x[0] for x in v) for v in shapes.values())
    print("%9s %6s %7s %9s %9s %9s %9s %9s %8s %7s" % ("n", "chunks", "calls", "wall us", "mean wk", "max wk",
                                                       "last lag", "tail", "balanced", "share"))
    for (n, chunks), v in sorted(shapes.items(), key=lambda kv: -sum(x[0] for x in kv[1])):
        if len(v) < min_calls:
            continue
        med = lambda i: statistics.median(x[i] for x in v)
        s_wall = sum(x[0] for x in v)
        bal = sum(x[1] for x in v) / s_wall if s_wall > 0 else 0
        print("%9d %6d %7d %9.2f %9.2f %9.2f %9.2f %9.2f %8.3f %6.1f%%" % (n, chunks, len(v), med(0), med(1), med(2),
                                                                           med(3), med(4), bal,
                                                                           100 * s_wall / total))


main()
