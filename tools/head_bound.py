#!/usr/bin/env python3
"""head_bound.py -- the output head's argmax by a bound (docs/MEASUREMENTS.md §A draft from the bytes the machine
holds, the fast inverse square root's lens, question 73). Counts, never a clock.

    head_bound.py <model.gguf> <logits.f32>... [--bits 2,3,4,5,6]
    head_bound.py --check

A row's logit is the sum over its blocks of scale x codes . h. Keep each code's top b bits (Q8_0: of 8, Q4_K: of 4)
and take the low bits at their worst against the position's h: an upper bound of every row from its scales and
top bits alone. A greedy token then needs only the rows whose bound reaches the best exact logit computed exactly
(taken in the bound's order: the first is the likeliest). This counts, at every position of the engine's logits
(`trochilus logits -b 1`), how many rows that leaves, and the head's bytes it reads: every row's scales and top
planes, the survivors' low planes.

h, the head's input, is solved from the engine's own logits (least squares on the dequantized head: the logits hold
h, 50304 equations for 2048 unknowns); the solve's residual is printed, and every row's bound is checked against
its value (a bound below its own row's logit would be a bug, and fails the run).
"""
import argparse
import sys

import numpy as np

EPS = 1e-3  # the engine's float logits against the float64 bound: far above their rounding, far below a margin


def q8_0_rows(raw, n_cols):
    """raw: rows x bytes of Q8_0 blocks (2-byte f16 scale, 32 int8). Returns (scales rows x blocks, codes int8)."""
    nb = n_cols // 32
    blk = raw.reshape(raw.shape[0], nb, 34)
    d = blk[:, :, :2].copy().view(np.float16)[:, :, 0].astype(np.float64)
    q = blk[:, :, 2:].copy().view(np.int8).astype(np.int32)
    return d, q  # q: rows x blocks x 32


def q4_k_rows(raw, n_cols):
    """raw: rows x bytes of Q4_K superblocks (ggml block_q4_K: d, dmin f16, 12 scale bytes, 128 code bytes).
    Returns (d*sc rows x subblocks, dmin*m rows x subblocks, codes 0..15 rows x subblocks x 32)."""
    ns = n_cols // 256
    blk = raw.reshape(raw.shape[0], ns, 144)
    d = blk[:, :, 0:2].copy().view(np.float16)[:, :, 0].astype(np.float64)
    dmin = blk[:, :, 2:4].copy().view(np.float16)[:, :, 0].astype(np.float64)
    s = blk[:, :, 4:16].astype(np.int32)
    qs = blk[:, :, 16:144].astype(np.int32)
    sc = np.zeros(d.shape + (8,), np.int32)
    mn = np.zeros(d.shape + (8,), np.int32)
    for j in range(8):  # ggml get_scale_min_k4
        if j < 4:
            sc[..., j] = s[..., j] & 63
            mn[..., j] = s[..., j + 4] & 63
        else:
            sc[..., j] = (s[..., j + 4] & 0xF) | ((s[..., j - 4] >> 6) << 4)
            mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    q = np.zeros(d.shape + (8, 32), np.int32)
    for c in range(4):  # 64 weights a chunk: the low nibbles first, the high nibbles next
        b = qs[:, :, c * 32:(c + 1) * 32]
        q[:, :, 2 * c, :] = b & 0xF
        q[:, :, 2 * c + 1, :] = b >> 4
    r = raw.shape[0]
    return ((d[..., None] * sc).reshape(r, ns * 8), (dmin[..., None] * mn).reshape(r, ns * 8),
            q.reshape(r, ns * 8, 32))


def survivors(upper, best):
    """Rows whose bound reaches the best exact logit, a position a column: their count."""
    return (upper >= best[None, :] - EPS).sum(axis=0)


def analyse(kind, rows, H, logits, bits):
    """rows: the head decoded (kind q8_0: (d, q); q4_k: (dsc, dm, q)); H: n_embd x P; logits: P x vocab.
    Returns [(b, mean survivors, p50, p99, max, the head's bytes against the whole)]."""
    best = logits.max(axis=1).astype(np.float64)
    pos = np.maximum(H, 0.0)  # n_embd x P
    out = []
    if kind == "q8_0":
        d, q = rows
        nb = d.shape[1]
        exact = (np.repeat(d, 32, axis=1) * q.reshape(q.shape[0], -1)) @ H
        Ppos = pos.reshape(nb, 32, -1).sum(axis=1)  # blocks x P
        for b in bits:
            s = 8 - b
            hi = q >> s  # floor, the arithmetic shift
            coarse = (np.repeat(d, 32, axis=1) * (hi << s).reshape(q.shape[0], -1)) @ H
            upper = coarse + ((1 << s) - 1) * (d @ Ppos)
            if (upper - exact).min() < -1e-6:
                sys.exit(f"q8_0 b {b}: a bound below its own row's logit ({(upper - exact).min():.3g})")
            n = survivors(upper, best)
            plane = (2 + 4 * b) / 34.0
            ratio = plane + n.mean() / d.shape[0] * (32 - 4 * b) / 34.0
            out.append((b, n.mean(), np.percentile(n, 50), np.percentile(n, 99), n.max(), ratio))
    else:
        dsc, dm, q = rows
        nsub = dsc.shape[1]
        sumh = H.reshape(nsub, 32, -1).sum(axis=1)  # subblocks x P
        Ppos = pos.reshape(nsub, 32, -1).sum(axis=1)
        mins = dm @ sumh  # rows x P: the dmin * m * sum(h) term, known from the scales
        exact = (np.repeat(dsc, 32, axis=1) * q.reshape(q.shape[0], -1)) @ H - mins
        for b in bits:
            s = 4 - b
            hi = q >> s
            coarse = (np.repeat(dsc, 32, axis=1) * (hi << s).reshape(q.shape[0], -1)) @ H - mins
            upper = coarse + ((1 << s) - 1) * (dsc @ Ppos)
            if (upper - exact).min() < -1e-6:
                sys.exit(f"q4_k b {b}: a bound below its own row's logit ({(upper - exact).min():.3g})")
            n = survivors(upper, best)
            plane = (16 + 32 * b) / 144.0
            ratio = plane + n.mean() / dsc.shape[0] * (128 - 32 * b) / 144.0
            out.append((b, n.mean(), np.percentile(n, 50), np.percentile(n, 99), n.max(), ratio))
    return out, exact


def load_head(path):
    import gguf
    r = gguf.GGUFReader(path)
    t = next(t for t in r.tensors if t.name == "output.weight")
    n_cols, n_rows = int(t.shape[0]), int(t.shape[1])
    raw = np.asarray(t.data, dtype=np.uint8).reshape(n_rows, -1)
    kind = t.tensor_type.name.lower()
    if kind == "q8_0":
        return kind, q8_0_rows(raw, n_cols), n_rows, n_cols
    if kind == "q4_k":
        return kind, q4_k_rows(raw, n_cols), n_rows, n_cols
    sys.exit(f"{path}: output.weight is {kind}, not Q8_0 or Q4_K")


def dense(kind, rows):
    if kind == "q8_0":
        d, q = rows
        return np.repeat(d, 32, axis=1) * q.reshape(q.shape[0], -1)
    dsc, dm, q = rows
    return np.repeat(dsc, 32, axis=1) * q.reshape(q.shape[0], -1) - np.repeat(dm, 32, axis=1)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("logits", nargs="*")
    ap.add_argument("--bits", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        return run_check()
    kind, rows, n_rows, n_cols = load_head(a.model)
    W = dense(kind, rows)  # rows x n_embd, float64
    G = np.linalg.cholesky(W.T @ W)
    bits = [int(x) for x in a.bits.split(",")] if a.bits else ([2, 3, 4, 5, 6] if kind == "q8_0" else [0, 1, 2, 3])
    for path in a.logits:
        L = np.fromfile(path, dtype=np.float32).reshape(-1, n_rows)
        rhs = W.T @ L.T.astype(np.float64)  # n_embd x P
        H = np.linalg.solve(G.T, np.linalg.solve(G, rhs))
        res = np.abs(W @ H - L.T).max()
        out, exact = analyse(kind, rows, H, L, bits)
        agree = (exact.argmax(axis=0) == L.argmax(axis=1)).mean()
        print(f"{path}: {kind}, {L.shape[0]} positions, h solved (residual {res:.2e}, argmax kept {agree:.4f})")
        for b, mean, p50, p99, mx, ratio in out:
            print(f"  top {b} bits: rows left mean {mean:8.1f} ({mean / n_rows:.4f}), median {p50:.0f}, p99 {p99:.0f}, "
                  f"max {mx}; the head's bytes {ratio:.4f}")
    return 0


def run_check():
    """Hand-made heads: a Q8_0 and a Q4_K row whose bound is computed by hand, and a row that cannot win dropped."""
    bad = 0

    def check(ok, msg):
        nonlocal bad
        if not ok:
            bad += 1
            print("FAIL", msg)

    # Q8_0, one block of 32: codes 0x17 = 23 everywhere, d = 1; h = +1 on 16 lanes, -1 on 16. b 4: hi = 1, s = 4:
    # coarse = 16 * (16 - 16) = 0; bound = 0 + 15 * 16 = 240; exact = 23 * 0 = 0
    d = np.ones((1, 1))
    q = np.full((1, 1, 32), 23, np.int32)
    h = np.array([1.0] * 16 + [-1.0] * 16)[:, None]
    out, ex = analyse("q8_0", (d, q), h, np.array([[240.0]], np.float32), [4])
    check(out[0][1] == 1.0 and abs(ex[0, 0]) < 1e-9, "Q8_0: a bound of 240 reaches a best logit of 240")
    out, ex = analyse("q8_0", (d, q), h, np.array([[240.5]], np.float32), [4])
    check(out[0][1] == 0.0, "Q8_0: a bound of 240 does not reach 240.5")
    # a second row with codes -128 on the positive lanes: its bound -128*16 + ... stays below row 0's exact 0
    q2 = np.concatenate([q, np.where(np.arange(32) < 16, -128, 0).reshape(1, 1, 32).astype(np.int32)])
    d2 = np.ones((2, 1))
    logits = np.array([[0.0, -2048.0]], np.float32)
    out, ex = analyse("q8_0", (d2, q2), h, logits, [4])
    check(out[0][1] == 1.0, f"the row that cannot win is dropped: {out[0][1]} rows left")
    # Q4_K: one subblock, d*sc = 2, dmin*m = 1, codes 9 = 0b1001; h = 1 on every lane; b 2: hi = 2, s = 2:
    # coarse = 2 * (8 * 32) - 32 = 480; bound = 480 + 3 * 2 * 32 = 672; exact = 2 * 9 * 32 - 32 = 544
    dsc = np.full((1, 1), 2.0)
    dm = np.ones((1, 1))
    q = np.full((1, 1, 32), 9, np.int32)
    h = np.ones((32, 1))
    out, ex = analyse("q4_k", (dsc, dm, q), h, np.array([[672.0]], np.float32), [2])
    check(abs(ex[0, 0] - 544) < 1e-9, f"Q4_K exact by hand: {ex[0, 0]}")
    check(out[0][1] == 1.0, "Q4_K: a bound of 672 reaches 672")
    out, ex = analyse("q4_k", (dsc, dm, q), h, np.array([[672.5]], np.float32), [2])
    check(out[0][1] == 0.0, "Q4_K: a bound of 672 does not reach 672.5")
    # the Q4_K scales of ggml's packing: j >= 4 takes its high bits from the bytes j - 4 and j
    raw = np.zeros((1, 144), np.uint8)
    raw[0, 0:2] = np.frombuffer(np.float16(1.0).tobytes(), np.uint8)
    raw[0, 2:4] = np.frombuffer(np.float16(1.0).tobytes(), np.uint8)
    raw[0, 4 + 0] = 0b11000101  # sc0 = 5, its top 2 bits are sc4's bits 4-5
    raw[0, 4 + 4] = 0b01000111  # m0 = 7, its top 2 bits are m4's bits 4-5
    raw[0, 4 + 8] = 0b00100011  # sc4's low 4 = 3, m4's low 4 = 2
    raw[0, 16] = 0x21           # the first code byte: subblock 0's first code 1 (low nibble), subblock 1's 2
    dsc_, dm_, q_ = q4_k_rows(raw, 256)
    check(dsc_[0, 0] == 5 and dm_[0, 0] == 7 and dsc_[0, 4] == 3 + (3 << 4) and dm_[0, 4] == 2 + (1 << 4),
          f"Q4_K scales unpacked: {dsc_[0, [0, 4]]}, {dm_[0, [0, 4]]}")
    check(q_[0, 0, 0] == 1 and q_[0, 1, 0] == 2, f"Q4_K codes unpacked: {q_[0, 0, 0]}, {q_[0, 1, 0]}")
    print("head_bound --check:", "ok" if bad == 0 else f"{bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
