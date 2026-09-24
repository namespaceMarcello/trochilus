"""attn_skip_report.py — which cached positions a decode token's attention could skip without
changing a bit, on the real model (docs/MEASUREMENTS.md, "Skipping positions exactly").

Reads the dumps of the diagnostic engine (`make attn-probe`, tools/attn_probe.c): per layer and
head the decode queries and the keys and values they attend to. Replays the attention in float32
exactly as src/kernels/kernels.c defines it (16-lane dot, max, exp, 16-lane sum, division, the
output accumulated in increasing position; numpy's float32 operations round like C's, and the exp
is rounded from float64, which differs from tr_expf only on the rare hard cases), and asks of
every position:

  zero        exp(s - max) is exactly 0: the position cannot change anything;
  sum_or      adding its e to its lane's running partial leaves the partial unchanged (oracle);
  v_or        adding a*v to the running output leaves every output element unchanged (oracle);
  sum_bf      as sum_or, but with an upper bound of e computed from the high 16 bits of the key
              (bf16 by truncation) and a rigorous error bound: decidable without the low half;
  v_vmax      as v_or, but decided from a and the largest |v| of the position (4 bytes stored at
              write time): with the exact a (key read) or with the bound of a from sum_bf;
  v_iv        the value's high 16 bits alone decide the new output: the two ends of the
              interval of a*v round the output to the same float.

and prints the bytes each scheme would read per token against today's F32 K and V.

Usage: tools/.venv/Scripts/python.exe tools/attn_skip_report.py <dump dir> [<dump dir> ...]
         [--layers 0,1,...] [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np

F32 = np.float32
U32 = np.uint32
HEAD_DIM = 128
LANES = 16
SCALE = F32(1.0) / np.sqrt(F32(128.0))  # olmoe.c: 1.0f / sqrtf((float)head_dim)
# The computed dot is within GAMMA * sum|q_i k_i| of the exact one: every product rounds once and
# every element goes through at most 7 lane additions and 4 tree additions (12 roundings).
GAMMA = 12.0 * 2.0**-24 / (1.0 - 12.0 * 2.0**-24) * 1.01
HIST_EDGES = [0.0, -2.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -40.0, -50.0, -70.0, -103.97]


def bits(x):
    return np.ascontiguousarray(x).view(U32)


def lane_combine(lane):
    """lane[..., 16] -> the fixed tree of src/kernels/kernels_internal.h."""
    l = [lane[..., i] for i in range(LANES)]
    lo = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))
    hi = ((l[8] + l[9]) + (l[10] + l[11])) + ((l[12] + l[13]) + (l[14] + l[15]))
    return lo + hi


def dot_f32(q, k):
    """q [nq, 128], k [n, 128] float32 -> [nq, n], the 16-lane contract."""
    p = q[:, None, :] * k[None, :, :]  # float32 products, rounded once
    p = p.reshape(q.shape[0], k.shape[0], HEAD_DIM // LANES, LANES)
    acc = p[:, :, 0, :].copy()
    for r in range(1, HEAD_DIM // LANES):
        acc = acc + p[:, :, r, :]
    return lane_combine(acc)


def up32(x):
    """float64 -> the smallest float32 >= x."""
    f = x.astype(F32)
    return np.where(f.astype(np.float64) < x, np.nextafter(f, F32(np.inf)), f)


def down32(x):
    f = x.astype(F32)
    return np.where(f.astype(np.float64) > x, np.nextafter(f, F32(-np.inf)), f)


def exp32(d):
    return np.exp(d.astype(np.float64)).astype(F32)


def split_hi(x):
    """High 16 bits (truncation toward zero) and the width of what was cut: |x - hi| < w."""
    b = bits(x)
    hi = (b & U32(0xFFFF0000)).view(F32)
    ef = ((b >> U32(23)) & U32(0xFF)).astype(np.int64)
    w = np.ldexp(1.0, np.where(ef >= 1, ef - 127 - 7, -126 - 7))
    return hi, w


def load_run(d):
    runs = {}
    names = sorted(f for f in os.listdir(d) if f.startswith("k_L"))
    for name in names:
        L, H = int(name[3:5]), int(name[7:9])
        runs.setdefault(L, []).append(H)
    return runs


def read_q(path):
    raw = np.fromfile(path, dtype=np.uint8)
    rec = 8 + 4 * HEAD_DIM
    n = raw.size // rec
    raw = raw[: n * rec].reshape(n, rec)
    npos = raw[:, :8].copy().view(np.int64)[:, 0]
    q = raw[:, 8:].copy().view(F32)
    return npos, q


def analyse_layer(d, L, heads):
    """Per head, per query, per position flags. Returns a dict of counters."""
    ks, vs, qs, nps = [], [], [], []
    for H in heads:
        k = np.fromfile(os.path.join(d, "k_L%02d_H%02d.bin" % (L, H)), dtype=F32).reshape(-1, HEAD_DIM)
        v = np.fromfile(os.path.join(d, "v_L%02d_H%02d.bin" % (L, H)), dtype=F32).reshape(-1, HEAD_DIM)
        npos, q = read_q(os.path.join(d, "q_L%02d_H%02d.bin" % (L, H)))
        keep = npos <= k.shape[0]
        ks.append(k)
        vs.append(v)
        qs.append(q[keep])
        nps.append(npos[keep])
    nq = min(len(x) for x in nps)
    N = ks[0].shape[0]
    nh = len(heads)
    Q = np.stack([x[:nq] for x in qs])  # [h, nq, 128]
    NP = np.stack([x[:nq] for x in nps])  # [h, nq]
    K = np.stack(ks)
    V = np.stack(vs)
    pos = np.arange(N)
    mask = pos[None, None, :] < NP[:, :, None]  # [h, nq, N]

    S = np.empty((nh, nq, N), F32)
    UB = np.empty((nh, nq, N), F32)
    LB = np.empty((nh, nq, N), F32)
    for h in range(nh):
        S[h] = dot_f32(Q[h], K[h]) * SCALE
        kh, w = split_hi(K[h])
        q64 = Q[h].astype(np.float64)
        dh = q64 @ kh.astype(np.float64).T
        uw = np.abs(q64) @ w.T
        aa = np.abs(q64) @ (np.abs(kh.astype(np.float64)) + w).T
        err = uw + GAMMA * aa + 1e-30
        sc = float(SCALE)
        ub = (dh + err) * sc
        lb = (dh - err) * sc
        UB[h] = up32(ub + np.abs(ub) * 2.0**-22)
        LB[h] = down32(lb - np.abs(lb) * 2.0**-22)
    S = np.where(mask, S, F32(-np.inf))
    UB = np.where(mask, UB, F32(-np.inf))
    LB = np.where(mask, LB, F32(-np.inf))
    assert np.all((S <= UB) | ~mask) and np.all((S >= LB) | ~mask), "bound violated"
    m = S.max(axis=2)  # [h, nq]
    D = S - m[:, :, None]
    E = np.where(mask, exp32(D), F32(0.0))
    Dhi = UB - m[:, :, None]
    Ehi = np.where(mask, up32(np.exp(Dhi.astype(np.float64))), F32(0.0))
    cand = UB >= LB.max(axis=2)[:, :, None]

    # the sum, lane by lane in increasing position, with the partial each position meets
    R = (N + LANES - 1) // LANES
    pad = R * LANES - N
    Ep = np.concatenate([E, np.zeros((nh, nq, pad), F32)], axis=2).reshape(nh, nq, R, LANES)
    Ehp = np.concatenate([Ehi, np.zeros((nh, nq, pad), F32)], axis=2).reshape(nh, nq, R, LANES)
    P = np.zeros((nh, nq, LANES), F32)
    sum_or = np.empty((nh, nq, R, LANES), bool)
    sum_bf = np.empty((nh, nq, R, LANES), bool)
    for r in range(R):
        nxt = P + Ep[:, :, r, :]
        sum_or[:, :, r, :] = bits(nxt) == bits(P)
        sum_bf[:, :, r, :] = bits(P + Ehp[:, :, r, :]) == bits(P)
        P = nxt
    total = lane_combine(P)  # [h, nq]
    sum_or = sum_or.reshape(nh, nq, R * LANES)[:, :, :N]
    sum_bf = sum_bf.reshape(nh, nq, R * LANES)[:, :, :N]
    A = E / total[:, :, None]
    Ahi = up32(Ehi.astype(np.float64) / total[:, :, None].astype(np.float64))

    vmax = np.abs(V).max(axis=2)  # [h, N]
    vh, vw = split_hi(V)
    vlo = np.where(vh >= 0, vh, vh - vw).astype(np.float64)
    vhi = np.where(vh >= 0, vh + vw, vh).astype(np.float64)
    vlo32 = down32(vlo)
    vhi32 = up32(vhi)

    out = np.zeros((nh, nq, HEAD_DIM), F32)
    v_or = np.empty((nh, nq, N), bool)
    v_vx = np.empty((nh, nq, N), bool)
    v_vb = np.empty((nh, nq, N), bool)
    v_iv = np.empty((nh, nq, N), bool)
    v_iv_lines = np.zeros((nh, nq, N), np.int8)  # cache lines of 16 floats that need the low half
    ob = bits(out)
    for t in range(N):
        a = A[:, :, t]
        vt = V[:, t, :]
        p = a[:, :, None] * vt[:, None, :]
        new = out + p
        nb = bits(new)
        v_or[:, :, t] = np.all(nb == ob, axis=2)
        pm = a * vmax[:, t][:, None]
        up = bits(out + pm[:, :, None])
        dn = bits(out - pm[:, :, None])
        v_vx[:, :, t] = np.all((up == ob) & (dn == ob), axis=2)
        pmb = Ahi[:, :, t] * vmax[:, t][:, None]
        upb = bits(out + pmb[:, :, None])
        dnb = bits(out - pmb[:, :, None])
        v_vb[:, :, t] = np.all((upb == ob) & (dnb == ob), axis=2)
        o1 = bits(out + a[:, :, None] * vlo32[:, t, :][:, None, :])
        o2 = bits(out + a[:, :, None] * vhi32[:, t, :][:, None, :])
        same = o1 == o2
        v_iv[:, :, t] = np.all(same, axis=2)
        v_iv_lines[:, :, t] = (~same).reshape(nh, nq, HEAD_DIM // 16, 16).any(axis=3).sum(axis=2)
        out = new
        ob = nb

    # the replay must give the engine's own output, bit for bit, before its counts mean anything
    checked = mismatched = 0
    for i, H in enumerate(heads):
        path = os.path.join(d, "o_L%02d_H%02d.bin" % (L, H))
        if not os.path.exists(path):
            continue
        onp, o = read_q(path)
        o = o[onp <= N][:nq]
        checked += o.shape[0]
        mismatched += int(np.any(bits(o) != bits(out[i, : o.shape[0]]), axis=1).sum())

    zero = E == 0
    return dict(mask=mask, D=D, zero=zero, sum_or=sum_or, sum_bf=sum_bf, v_or=v_or, v_vx=v_vx, v_vb=v_vb,
                v_iv=v_iv, v_iv_lines=v_iv_lines, cand=cand, nq=nq, N=N, NP=NP, checked=checked,
                mismatched=mismatched)


def schemes(r):
    """Bytes per position per head for each scheme, summed over the masked positions."""
    mk = r["mask"]
    n = mk.sum()
    f = lambda x: float((x & mk).sum())
    out = {}
    out["positions"] = float(n)
    out["today"] = 1024.0 * n
    # K read whole (exact scores), V skipped where the vmax test with the exact a proves it
    out["v_skip_vmax"] = 512.0 * n + 512.0 * f(~r["v_vx"]) + 4.0 * n
    out["v_skip_oracle"] = 512.0 * n + 512.0 * f(~r["v_or"])
    # hi/lo keys: the high half always; the low half and V where not proven
    skip_all = ~r["cand"] & r["sum_bf"] & r["v_vb"]
    need_klo = ~skip_all
    need_v = need_klo & ~r["v_vx"]
    out["hilo_k"] = 256.0 * n + 256.0 * f(need_klo) + 512.0 * f(need_v) + 4.0 * n
    need_vlo = need_v & ~r["v_iv"]
    out["hilo_kv"] = 256.0 * n + 256.0 * f(need_klo) + 256.0 * f(need_v) + 256.0 * f(need_vlo) + 4.0 * n
    # the same with the low half of V fetched by cache line (16 floats, 64 bytes) where needed
    lines = float((r["v_iv_lines"] * (need_v & mk)).sum())
    out["hilo_kv_lines"] = 256.0 * n + 256.0 * f(need_klo) + 256.0 * f(need_v) + 32.0 * lines + 4.0 * n
    both_or = r["sum_or"] & r["v_or"]
    out["oracle_all"] = 1024.0 * f(~both_or)
    out["checked"] = float(r["checked"])
    out["mismatched"] = float(r["mismatched"])
    out["zero"] = f(r["zero"])
    out["cand"] = f(r["cand"])
    out["skip_all"] = f(skip_all)
    out["sum_bf"] = f(r["sum_bf"])
    out["sum_or"] = f(r["sum_or"])
    out["v_or"] = f(r["v_or"])
    out["v_vx"] = f(r["v_vx"])
    out["v_vb"] = f(r["v_vb"])
    out["v_iv"] = f(r["v_iv"] & need_v)
    out["need_v"] = f(need_v)
    D = r["D"][mk]
    edges = HIST_EDGES + [-np.inf]
    out["hist"] = [float(((D <= edges[i]) & (D > edges[i + 1])).sum()) for i in range(len(edges) - 1)]
    return out


def add(acc, s):
    for k, v in s.items():
        if isinstance(v, list):
            acc[k] = [a + b for a, b in zip(acc.get(k, [0.0] * len(v)), v)]
        else:
            acc[k] = acc.get(k, 0.0) + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    report = {}
    failed = False
    for d in args.dirs:
        run = os.path.basename(os.path.normpath(d))
        layers = load_run(d)
        sel = sorted(layers) if args.layers is None else [int(x) for x in args.layers.split(",")]
        tot = {}
        per_layer = {}
        for L in sel:
            r = analyse_layer(d, L, sorted(layers[L]))
            s = schemes(r)
            per_layer[L] = s
            add(tot, s)
            sys.stderr.write("%s L%02d done\n" % (run, L))
        report[run] = dict(total=tot, layers=per_layer)
        t = tot["today"]
        n = tot["positions"]
        print("== %s: %d positions (heads x queries x context)" % (run, n))
        if tot["checked"] == 0:
            print("   replay NOT checked: no o_*.bin in the dump (an older probe); the counts are unproven")
        else:
            print("   replay = engine: %d head outputs compared bit for bit, %d differ"
                  % (tot["checked"], tot["mismatched"]))
            if tot["mismatched"]:
                failed = True
        print("   bytes vs today:  v_skip_vmax %.3f  v_skip_oracle %.3f  hilo_k %.3f  hilo_kv %.3f  hilo_kv_lines %.3f  oracle_all %.3f"
              % (tot["v_skip_vmax"] / t, tot["v_skip_oracle"] / t, tot["hilo_k"] / t, tot["hilo_kv"] / t,
                 tot["hilo_kv_lines"] / t, tot["oracle_all"] / t))
        print("   fractions: zero %.4f  cand %.4f  sum_or %.3f  sum_bf %.3f  v_or %.3f  v_vx %.3f  v_vb %.3f  skip_all %.3f  need_v %.3f  v_iv|need_v %.3f"
              % (tot["zero"] / n, tot["cand"] / n, tot["sum_or"] / n, tot["sum_bf"] / n, tot["v_or"] / n,
                 tot["v_vx"] / n, tot["v_vb"] / n, tot["skip_all"] / n, tot["need_v"] / n,
                 tot["v_iv"] / max(tot["need_v"], 1.0)))
        h = tot["hist"]
        labels = ["(%g,%g]" % (HIST_EDGES[i + 1] if i + 1 < len(HIST_EDGES) else -np.inf, HIST_EDGES[i]) for i in range(len(HIST_EDGES))]
        print("   s-max: " + "  ".join("%s %.3f" % (lab, x / n) for lab, x in zip(labels, h)))
        print("   per layer hilo_kv / v_skip_vmax / oracle_all:")
        print("   " + "  ".join("L%d %.2f/%.2f/%.2f" % (L, s["hilo_kv"] / s["today"], s["v_skip_vmax"] / s["today"],
                                                          s["oracle_all"] / s["today"]) for L, s in per_layer.items()))
        sys.stdout.flush()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    if failed:
        sys.exit("attn_skip_report: the replay does not give the engine's bits; fix it before reading the counts")


if __name__ == "__main__":
    main()
