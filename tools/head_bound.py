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


TOL = 2e-3  # the study's float32 products: a bound this far below its row's value would be a bug


class Head:
    """The head for the study (question 73, build/q73): float32 planes, the bounds' variants on chunks of positions.
    Q8_0: scale d per block of 32, codes -128..127 (8 bits). Q4_K: d*sc per subblock of 32, codes 0..15 (4 bits),
    the dmin*m term exact from the block sums of h."""

    def __init__(self, kind, rows):
        self.kind = kind
        if kind == "q8_0":
            d, q = rows
            self.scale, self.q, self.dm, self.qmax, self.top = d.astype(np.float32), q.astype(np.int8), None, 128.0, 8
        else:
            dsc, dm, q = rows
            self.scale, self.q, self.dm, self.qmax, self.top = (dsc.astype(np.float32), q.astype(np.uint8),
                                                                dm.astype(np.float32), 15.0, 4)
        self.V, self.nb = self.scale.shape
        self.n_embd = self.nb * 32
        self.spos = np.maximum(self.scale, 0)
        self.sneg = np.minimum(self.scale, 0)
        self.sabs = np.abs(self.scale)
        self.planes = {0: self.plane_matrix(0)}
        self.svd = None

    def plane_matrix(self, s):
        q = self.q.astype(np.int16)
        if s:
            q = (q >> s) << s  # floor, the arithmetic shift
        return np.repeat(self.scale, 32, axis=1) * q.reshape(self.V, -1).astype(np.float32)

    def blk(self, H):
        return H.reshape(self.nb, 32, -1).sum(axis=1)

    def mins(self, H):
        return 0.0 if self.dm is None else self.dm @ self.blk(H)

    def exact(self, H):
        return self.planes[0] @ H - self.mins(H)

    def upper(self, b, H, Hu=None, E=None):
        """Every row's bound from its scales and top b bits; Hu: h as the bound pass sees it (a quantized copy),
        E: its error's block sums of |h - Hu| (the codes at their largest against it)."""
        s = self.top - b
        if s not in self.planes:
            self.planes[s] = self.plane_matrix(s)
        Hu = H if Hu is None else Hu
        u = self.planes[s] @ Hu - self.mins(H)
        if s:
            u += ((1 << s) - 1) * (self.spos @ self.blk(np.maximum(Hu, 0)) + self.sneg @ self.blk(np.minimum(Hu, 0)))
        if E is not None:
            u += self.qmax * (self.sabs @ E)
        return u

    def quant(self, H, bits):
        """h rounded per block of 32 to signed integers of `bits` bits (scale max|h| / (2^(bits-1) - 1))."""
        G = H.reshape(self.nb, 32, -1)
        delta = np.abs(G).max(axis=1, keepdims=True) / ((1 << (bits - 1)) - 1)
        delta[delta == 0] = 1.0
        Hq = (np.rint(G / delta) * delta).reshape(H.shape)
        return Hq.astype(np.float32), self.blk(np.abs(H - Hq)).astype(np.float32)

    def svd_prepare(self, ks):
        """FEXIPRO's bound (Li et al. 2017): the rows in the head's singular basis, the first k coordinates kept (as
        f16, their rounding counted), the rest bounded by Cauchy-Schwarz."""
        W = self.planes[0] if self.dm is None else self.planes[0] - np.repeat(self.dm, 32, axis=1)
        G = np.zeros((self.n_embd, self.n_embd))
        for r in range(0, self.V, 8192):
            c = W[r:r + 8192].astype(np.float64)
            G += c.T @ c
        _, vec = np.linalg.eigh(G)
        Vb = vec[:, ::-1].astype(np.float32)
        A = W @ Vb
        self.svd = {"V": Vb}
        for k in ks:
            Ah = A[:, :k].astype(np.float16).astype(np.float32)
            err = np.sqrt(((A[:, :k] - Ah) ** 2).sum(axis=1))
            rest = np.sqrt((A[:, k:] ** 2).sum(axis=1))
            self.svd[k] = (Ah, err * 1.0001 + 1e-6, rest * 1.0001 + 1e-6)

    def upper_svd(self, k, H):
        Z = self.svd["V"].T @ H
        Ah, err, rest = self.svd[k]
        zk = np.sqrt((Z[:k] ** 2).sum(axis=0))
        zr = np.sqrt((Z[k:] ** 2).sum(axis=0))
        return Ah @ Z[:k] + err[:, None] * zk[None, :] + rest[:, None] * zr[None, :]


def study_counts(U, ex, best, prev):
    """The rows the exact pass computes, a position a column, for four thresholds: the best exact score (the
    least), the exact score of the best bound's row, the best of the four best bounds' rows, the previous
    position's argmax row (the best bound's where there is none)."""
    p = np.arange(U.shape[1])
    n_min = (U >= best[None, :] - EPS).sum(axis=0)
    t1 = ex[U.argmax(axis=0), p]
    n_maxb = (U >= t1[None, :] - EPS).sum(axis=0)
    top4 = np.argpartition(-U, 4, axis=0)[:4]
    t4 = ex[top4, p[None, :]].max(axis=0)
    n_top4 = (U >= t4[None, :] - EPS).sum(axis=0)
    tp = np.where(prev >= 0, ex[np.maximum(prev, 0), p], t1)
    n_prev = (U >= tp[None, :] - EPS).sum(axis=0)
    return np.stack([n_min, n_maxb, n_top4, n_prev])


def study_variants(kind):
    if kind == "q8_0":
        return ["b3", "b4", "b5", "b6", "b4i8", "b4i16", "b5i8", "svd16", "svd64", "svd256", "svd64+b4"]
    return ["b2", "b3", "b3i8", "b3i16", "svd16", "svd64", "svd256", "svd64+b3"]


def study_bytes(kind, var, n, V, n_other=None):
    """The head's bytes a token reads, against the whole, for a variant whose exact pass computes n rows (means).
    svdK+bB: the f16 coordinates of every row, then the plane for the svd's survivors (n_other), then the rest of
    the rows both bounds keep (n)."""
    blk, row = (34.0, 2176.0) if kind == "q8_0" else (144.0, 1152.0)

    def plane(b):
        return (2 + 4 * b) / 34.0 if kind == "q8_0" else (16 + 32 * b) / 144.0

    if var.startswith("svd"):
        k = int(var[3:].split("+")[0])
        svd = (2 * k + 8) / row
        if "+" not in var:
            return svd + n / V
        b = int(var.split("+b")[1])
        return svd + n_other / V * plane(b) + n / V * (1 - plane(b))
    b = int(var[1])
    return plane(b) + n / V * (1 - plane(b))


def study(model, paths, out_dir, chunk):
    """Question 73, phase 1 (build/q73): every variant's rows left at every position of each logits file, saved as
    <out_dir>/<file>.npz (counts: variants x 4 thresholds x positions; the prompt's length when <file>.np says it),
    a line a variant printed."""
    import os
    kind, rows, n_rows, n_cols = load_head(model)
    hd = Head(kind, rows)
    del rows
    W = hd.planes[0] if hd.dm is None else hd.planes[0] - np.repeat(hd.dm, 32, axis=1)
    Gm = np.zeros((n_cols, n_cols))
    for r in range(0, n_rows, 8192):
        c = W[r:r + 8192].astype(np.float64)
        Gm += c.T @ c
    Gc = np.linalg.cholesky(Gm)
    variants = study_variants(kind)
    hd.svd_prepare([16, 64, 256])
    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        L = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, n_rows)
        P = L.shape[0]
        npf = path[:-4] + ".np"
        n_prompt = int(open(npf).read().split()[0]) if os.path.exists(npf) else P
        genf = path[:-4] + ".gen"  # generate's "tokens: a,b,..." line: the prompt and the generation fill the file
        if os.path.exists(genf):
            gen = [ln for ln in open(genf).read().splitlines() if ln.startswith("tokens:")]
            n_gen = len(gen[0].split(":", 1)[1].split(",")) if gen else 0
            if n_prompt + n_gen != P:
                sys.exit(f"{path}: {P} positions, but {n_prompt} prompt tokens ({npf}) and {n_gen} generated ({genf})")
        counts = np.zeros((len(variants), 4, P), np.int32)
        worst = np.inf
        prev_last = -1
        for c0 in range(0, P, chunk):
            Lc = np.asarray(L[c0:c0 + chunk], dtype=np.float64)
            p = Lc.shape[0]
            rhs = np.zeros((n_cols, p))
            for r in range(0, n_rows, 8192):
                rhs += W[r:r + 8192].astype(np.float64).T @ Lc[:, r:r + 8192].T
            H = np.linalg.solve(Gc.T, np.linalg.solve(Gc, rhs)).astype(np.float32)
            ex = hd.exact(H)
            best = Lc.max(axis=1).astype(np.float32)
            am = Lc.argmax(axis=1)
            prev = np.concatenate([[prev_last], am[:-1]])
            prev_last = am[-1]
            ups = {}
            for i, var in enumerate(variants):
                if var.startswith("svd") and "+" in var:
                    k, b = var[3:].split("+b")
                    U = np.minimum(ups["svd" + k], ups["b" + b])
                elif var.startswith("svd"):
                    U = hd.upper_svd(int(var[3:]), H)
                elif "i" in var:
                    b, bits = var[1:].split("i")
                    Hq, E = hd.quant(H, int(bits))
                    U = hd.upper(int(b), H, Hq, E)
                else:
                    U = hd.upper(int(var[1:]), H)
                ups[var] = U
                worst = min(worst, float((U - ex).min()))
                counts[i, :, c0:c0 + p] = study_counts(U, ex, best, prev)
            del ups
        if worst < -TOL:
            sys.exit(f"{path}: a bound below its own row's value ({worst:.3g})")
        stem = os.path.basename(path)[:-4]
        np.savez(os.path.join(out_dir, stem + ".npz"), counts=counts, variants=np.array(variants), n_prompt=n_prompt,
                 kind=kind)
        print(f"{path}: {kind}, {P} positions ({n_prompt} the prompt's), the bounds' worst margin {worst:.2e}")
        study_print(kind, variants, counts, n_rows)
    return 0


def study_print(kind, variants, counts, V, label=""):
    for i, var in enumerate(variants):
        n = counts[i]
        other = None
        if "+" in var:
            other = counts[variants.index(var.split("+")[0]), 0].mean()
        print(f"  {label}{var:9s} rows left mean {n[0].mean():8.1f} ({n[0].mean() / V:.4f}), median "
              f"{np.median(n[0]):.0f}, p99 {np.percentile(n[0], 99):.0f}, max {n[0].max()}; computed: best bound's "
              f"{n[1].mean():.1f}, top 4's {n[2].mean():.1f}, previous argmax's {n[3].mean():.1f}; the head's bytes "
              f"{study_bytes(kind, var, n[0].mean(), V, other):.4f} (top 4's "
              f"{study_bytes(kind, var, n[2].mean(), V, counts[variants.index(var.split('+')[0]), 2].mean() if other is not None else None):.4f})")


def study_report(npzs, V=50304):
    """The study's files pooled by kind and category: the prompts' positions, the greedy ones (the engine's own
    tokens, from the prompt's last position on), the chats' greedy ones, the long contexts' by position."""
    pools = {}
    for f in npzs:
        z = np.load(f)
        kind, variants, counts, n_prompt = str(z["kind"]), [str(v) for v in z["variants"]], z["counts"], int(z["n_prompt"])
        name = f.replace("\\", "/").split("/")[-1]
        P = counts.shape[2]
        cats = []
        pos = np.arange(P)
        if "long" in name:
            for lo, hi in ((0, 1024), (1024, 2048), (2048, 3072), (3072, 3800), (3800, P + 1)):
                cats.append((f"long {lo}-{hi - 1 if hi <= P else 'end'}", (pos >= lo) & (pos < hi)))
        elif "chat" in name:
            cats.append(("chat greedy", pos >= n_prompt - 1))
        else:
            cats.append(("prompt", pos < n_prompt - 1))
            cats.append(("greedy", pos >= n_prompt - 1))
        for c, m in cats:
            key = (kind, c)
            pools.setdefault(key, (variants, []))[1].append(counts[:, :, m])
    for (kind, c), (variants, parts) in sorted(pools.items()):
        counts = np.concatenate(parts, axis=2)
        print(f"{kind} {c}: {counts.shape[2]} positions")
        study_print(kind, variants, counts, V)
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("logits", nargs="*")
    ap.add_argument("--bits", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--study", default=None, help="question 73's variants: the folder of the counts (.npz)")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--report", action="store_true", help="the study's .npz files (given as the arguments) pooled")
    a = ap.parse_args(argv)
    if a.check:
        return run_check()
    if a.report:
        return study_report(([a.model] if a.model else []) + a.logits)
    if a.study:
        return study(a.model, a.logits, a.study, a.chunk)
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
    # the study's Head: the hand-made rows again, then random heads (negative scales too) against every variant
    hd = Head("q8_0", (np.ones((1, 1)), np.full((1, 1, 32), 23, np.int32)))
    h = np.array([1.0] * 16 + [-1.0] * 16, np.float32)[:, None]
    check(abs(hd.upper(4, h)[0, 0] - 240) < 1e-4 and abs(hd.exact(h)[0, 0]) < 1e-4, "Head q8_0: bound 240, exact 0")
    hd = Head("q4_k", (np.full((1, 1), 2.0), np.ones((1, 1)), np.full((1, 1, 32), 9, np.int32)))
    h = np.ones((32, 1), np.float32)
    check(abs(hd.upper(2, h)[0, 0] - 672) < 1e-4 and abs(hd.exact(h)[0, 0] - 544) < 1e-4, "Head q4_k: 672, 544")
    rng = np.random.default_rng(73)
    for kind in ("q8_0", "q4_k"):
        V, nb = 300, 4
        if kind == "q8_0":
            rows = (rng.normal(0, 0.01, (V, nb)), rng.integers(-128, 128, (V, nb, 32)))
        else:
            rows = (rng.normal(0, 0.01, (V, nb)), rng.uniform(0, 0.01, (V, nb)), rng.integers(0, 16, (V, nb, 32)))
        hd = Head(kind, rows)
        hd.svd_prepare([16])
        H = rng.normal(0, 1, (nb * 32, 5)).astype(np.float32)
        ex = hd.exact(H)
        worst, no_e = np.inf, np.inf
        for b in ((3, 4, 5) if kind == "q8_0" else (2, 3)):
            worst = min(worst, (hd.upper(b, H) - ex).min())
            Hq, E = hd.quant(H, 8)
            worst = min(worst, (hd.upper(b, H, Hq, E) - ex).min())
        Hq, E = hd.quant(H, 8)
        no_e = (hd.upper(hd.top, H, Hq, None) - ex).min()  # every bit kept: only the error term covers h's rounding
        worst = min(worst, (hd.upper(hd.top, H, Hq, E) - ex).min())
        worst = min(worst, (hd.upper_svd(16, H) - ex).min())
        check(worst > -1e-4, f"{kind}: every variant's bound above its row ({worst:.3g})")
        check(no_e < -1e-4, f"{kind}: without its error term the int8 bound fails somewhere ({no_e:.3g}): the term is seen")
        n = study_counts(hd.upper(4 if kind == "q8_0" else 3, H), ex, ex.max(axis=0), np.full(5, -1))
        check(n.min() >= 1 and (n[0] <= n[1]).all(), f"{kind}: the argmax's row survives, the least threshold the least")
    check(abs(study_bytes("q8_0", "b4", 0, 1) - 18 / 34) < 1e-12 and abs(study_bytes("q8_0", "svd64", 0, 1) - 136 / 2176)
          < 1e-12 and abs(study_bytes("q4_k", "svd64+b3", 1, 1, 1) - (136 / 1152 + 1)) < 1e-12, "study_bytes")
    print("head_bound --check:", "ok" if bad == 0 else f"{bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
