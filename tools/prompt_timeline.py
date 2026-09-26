"""prompt_timeline.py -- where a prompt pass's time goes, per call type and per thread, from two
TR_POOL_TRACE build's files of the same prompt at two thread counts (threads.c; the brief of
docs/MEASUREMENTS.md §Why the prompt scales worse from 8 to 16 threads).

    tools/.venv/Scripts/python.exe tools/prompt_timeline.py <trace A> [<trace B>] [--ccd K]

Each trace line is one parallel_for: "n chunks t0 t1 (start end) x chunks bytes rows cols", times in
microseconds (token_timeline.py reads the same file). A call type is its (n, bytes, rows, cols); for
each type and each file: calls, the summed wall, the summed work (chunk time over all chunks), the
work per chunk, the tail (wall minus the mean chunk's work: what a perfect split of the same work
would have saved), and the lag of the last chunk to start. With two files the last columns compare
them: the wall ratio A/B, the work ratio B/A (1.00 = the same work spread on more threads costs
nothing more; above 1 = each unit of work got slower with more threads), and the share of B's extra
time over a perfect A/B scaling that each type carries. --ccd K splits each call's chunks into the
first K and the rest (chunk c runs on the pool's thread c; the pool ranks threads over the L3s, so
this reads which half of a call is slower when the halves sit on different CCDs).
"""
import sys


def load(path):
    calls = []
    for line in open(path):
        v = line.split()
        n, chunks = int(v[0]), int(v[1])
        t0, t1 = float(v[2]), float(v[3])
        ch = [(float(v[4 + 2 * k]), float(v[5 + 2 * k])) for k in range(chunks)]
        tail = v[4 + 2 * chunks:]
        note = tuple(int(x) for x in tail[:3]) if len(tail) >= 3 else (0, 0, 0)
        calls.append({"n": n, "chunks": chunks, "t0": t0, "t1": t1, "ch": ch, "note": note})
    return calls


def per_type(calls, ccd):
    types = {}
    order = []
    for c in calls:
        key = (c["n"],) + c["note"]
        if key not in types:
            types[key] = {"calls": 0, "wall": 0.0, "work": 0.0, "lag": 0.0, "tail": 0.0, "chunks": c["chunks"],
                          "work_lo": 0.0, "work_hi": 0.0, "end_lo": 0.0, "end_hi": 0.0, "first": c["t0"]}
            order.append(key)
        t = types[key]
        wall = c["t1"] - c["t0"]
        work = [e - s for s, e in c["ch"]]
        t["calls"] += 1
        t["wall"] += wall
        t["work"] += sum(work)
        t["lag"] += max(s for s, _ in c["ch"]) - c["t0"]
        t["tail"] += wall - sum(work) / len(work)
        if ccd and len(work) > ccd:
            t["work_lo"] += sum(work[:ccd]) / ccd
            t["work_hi"] += sum(work[ccd:]) / (len(work) - ccd)
            t["end_lo"] += max(e for _, e in c["ch"][:ccd]) - c["t0"]
            t["end_hi"] += max(e for _, e in c["ch"][ccd:]) - c["t0"]
    return types, order


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ccd = 0
    if "--ccd" in sys.argv:
        ccd = int(sys.argv[sys.argv.index("--ccd") + 1])
        args.remove(str(ccd))
    loaded = [load(p) for p in args]
    runs = [per_type(c, ccd) for c in loaded]
    a_types, order = runs[0]
    total = [sum(t["wall"] for t in r[0].values()) for r in runs]
    for i, p in enumerate(args):
        calls = loaded[i]
        span = calls[-1]["t1"] - calls[0]["t0"]
        tails = sum(t["tail"] for t in runs[i][0].values())
        print("%s: %d types; span %.1f ms = pool calls %.1f (mean work %.1f + tails %.1f, %.1f%%) + gaps %.1f" % (
            p, len(runs[i][0]), span / 1000, total[i] / 1000, (total[i] - tails) / 1000, tails / 1000,
            100 * tails / total[i], (span - total[i]) / 1000))
    hdr = "%-34s %5s %3s %9s %9s %9s %8s %7s" % ("type n/bytes/rows/cols", "calls", "ch", "wall ms", "work ms",
                                                "wk/ch ms", "tail ms", "lag ms")
    if len(runs) > 1:
        hdr += " | %3s %9s %9s %8s %7s %6s %6s %6s" % ("ch", "wall ms", "wk/ch ms", "tail ms", "lag ms", "A/B",
                                                    "workB/A", "extra%")
    print(hdr)
    extra_total = 0.0
    rows = []
    for key in order:
        a = a_types[key]
        line = "%-34s %5d %3d %9.2f %9.2f %9.2f %8.2f %7.2f" % ("/".join(str(x) for x in key), a["calls"],
                                                            a["chunks"], a["wall"] / 1000, a["work"] / 1000,
                                                            a["work"] / a["chunks"] / 1000, a["tail"] / 1000,
                                                            a["lag"] / 1000)
        extra = 0.0
        if len(runs) > 1 and key in runs[1][0]:
            b = runs[1][0][key]
            ideal = a["wall"] * a["chunks"] / b["chunks"]
            extra = b["wall"] - ideal
            line += " | %3d %9.2f %9.2f %8.2f %7.2f %6.2f %6.2f" % (b["chunks"], b["wall"] / 1000,
                                                                  b["work"] / b["chunks"] / 1000, b["tail"] / 1000,
                                                                  b["lag"] / 1000, a["wall"] / b["wall"],
                                                                  b["work"] / a["work"] if a["work"] > 0 else 0)
            extra_total += extra
        rows.append((line, extra, key))
    for line, extra, key in rows:
        share = 100 * extra / extra_total if extra_total > 0 else 0
        print(line + (" %6.1f" % share if len(runs) > 1 else ""))
    if len(runs) > 1:
        print("B's time over a perfect scaling of A: %.1f ms of %.1f ms" % (extra_total / 1000, total[1] / 1000))
    if ccd:
        print("\nper type, the two halves (chunks < %d, >= %d): mean work and the last end, ms, summed over calls" % (ccd, ccd))
        for i, p in enumerate(args):
            print(p)
            for key in order:
                t = runs[i][0].get(key)
                if t is None or t["work_hi"] == 0:
                    continue
                print("  %-34s work lo %8.2f hi %8.2f (hi/lo %.3f)  end lo %8.2f hi %8.2f" % (
                    "/".join(str(x) for x in key), t["work_lo"] / 1000, t["work_hi"] / 1000,
                    t["work_hi"] / t["work_lo"] if t["work_lo"] > 0 else 0, t["end_lo"] / 1000, t["end_hi"] / 1000))


main()
