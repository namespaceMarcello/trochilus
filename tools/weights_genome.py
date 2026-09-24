"""weights_genome.py — a model's quantized weights read like a genome (docs/MEASUREMENTS.md
questions 56 and 62).

Approximate engines compress weights with loss and never look for exact structure; an exact
engine can use anything that repeats or is predictable, without changing a bit. This reads every
block of a GGUF file once and reports, per tensor and for the whole file:

  entropy   bits a code carries (Q8_0: 8-bit codes; Q4_K: 4-bit codes), three ways: order 0 with
            one table per tensor (what a static entropy coder reaches); given a per-block context
            (Q8_0: 16 magnitude classes by mean |code|, the class's own cost added; Q4_K: the
            sub-block's 6-bit scale, stored anyway); given the input column (one table per column
            of the tensor, table cost not counted: the most a column-aware coder could reach).
            The block scales: bits of their f16 (order 0) of 16;
  zero      blocks whose codes are all zero;
  repeats   blocks equal to an earlier block of the same tensor, whole (scale and codes), codes
            only (the same shape at another scale), codes up to sign; blocks equal anywhere in
            the file; rows (all the blocks of one row, e.g. of one expert) equal to another row.

Hashes are 64-bit (false repeats expected: blocks^2 / 2^65, ~0.001 on a 7 GB file).

  tools/.venv/Scripts/python.exe tools/weights_genome.py <model.gguf> [--tensors <regex>]

Supported: Q8_0 (32 codes + f16 scale) and Q4_K (256 codes in 8 sub-blocks, 6-bit scales and
mins); other types are counted and skipped.
"""
import argparse
import re
import sys

import numpy as np

import gguf

K = [np.uint64(c) for c in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                             0xD6E8FEB86659FD93, 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53)]
CHUNK = 1 << 20  # blocks at a time: the counting temporaries stay under ~1 GB


def mix(h):
    h = h ^ (h >> np.uint64(33))
    h = h * K[4]
    h = h ^ (h >> np.uint64(33))
    h = h * K[5]
    return h ^ (h >> np.uint64(33))


def hash_rows(words, seed):
    """64-bit hash of every row of a uint64 matrix (one row = one block)"""
    h = np.full(words.shape[0], seed, dtype=np.uint64)
    for j in range(words.shape[1]):
        h = mix(h ^ (words[:, j] * K[j % 4]) + np.uint64(j + 1))
    return h


def entropy(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def cond_entropy(joint):
    """H(code | context) in bits, joint[context, code]"""
    tot = joint.sum()
    return sum(row.sum() / tot * entropy(row) for row in joint if row.sum())


def repeats(h):
    """blocks whose hash equals an earlier block's"""
    return len(h) - len(np.unique(h))


def counts(nb, per, cols, nbins, code_of, ctx_of, nctx):
    """histograms over all blocks, chunked: order 0, joint (context, code), joint (column, code)"""
    h0 = np.zeros(nbins, dtype=np.int64)
    jctx = np.zeros(nctx * nbins, dtype=np.int64)
    jcol = np.zeros(cols * nbins, dtype=np.int64)
    bpr = cols // per
    for a in range(0, nb, CHUNK):
        b = min(nb, a + CHUNK)
        c = code_of(a, b).astype(np.int64)  # (b - a, per)
        h0 += np.bincount(c.ravel(), minlength=nbins)
        x = ctx_of(a, b).astype(np.int64)  # (b - a,) or (b - a, per)
        x = np.broadcast_to(x[:, None] if x.ndim == 1 else x, c.shape)
        jctx += np.bincount((x * nbins + c).ravel(), minlength=nctx * nbins)
        col = (np.arange(a, b) % bpr)[:, None] * per + np.arange(per)[None, :]
        jcol += np.bincount((col * nbins + c).ravel(), minlength=cols * nbins)
    return h0, jctx.reshape(nctx, nbins), jcol.reshape(cols, nbins)


def blocks_q8_0(raw):
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.uint16).ravel()
    q = b[:, 2:].copy().view(np.int8)
    return d, q


def blocks_q4_k(raw):
    """d and dmin (n, 2) f16 bits, codes (n, 256) in weight order, sub-block scales (n, 8)"""
    b = raw.reshape(-1, 144)
    d = b[:, 0:4].copy().view(np.uint16)
    sc = b[:, 4:16]
    qs = b[:, 16:144].reshape(-1, 4, 32)
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=2).reshape(-1, 256)  # 64j+l low, 64j+32+l high
    s = np.empty((b.shape[0], 8), dtype=np.uint8)  # llama.cpp get_scale_min_k4, scales only
    s[:, :4] = sc[:, 0:4] & 63
    s[:, 4:] = (sc[:, 8:12] & 0x0F) | ((sc[:, 0:4] >> 6) << 4)
    return d, q, s


def scan_tensor(t, cols, glob_hashes):
    raw = np.asarray(t.data).reshape(-1)
    typ = t.tensor_type.name
    out = {"name": t.name, "type": typ, "bytes": raw.size, "cols": cols}
    if typ == "Q8_0":
        d, q = blocks_q8_0(raw)
        nb, per = q.shape[0], 32
        codes = q.view(np.uint8)
        mag = np.abs(q.astype(np.int16)).sum(axis=1)
        cls = np.searchsorted(np.quantile(mag, np.linspace(0, 1, 17)[1:-1]), mag)
        h0, jctx, jcol = counts(nb, per, cols, 256, lambda a, b: codes[a:b], lambda a, b: cls[a:b], 16)
        out["hctx"] = cond_entropy(jctx) + entropy(np.bincount(cls, minlength=16)) / 32
        out["hscale"] = entropy(np.bincount(d, minlength=65536))
        out["code_bits"] = 8
        wc = q.view(np.uint64)
        w = np.concatenate([wc, d.astype(np.uint64)[:, None]], axis=1)
        first = q[np.arange(nb), np.argmax(q != 0, axis=1)]
        sign = np.where(first < 0, -1, 1).astype(np.int8)
        ws = np.ascontiguousarray(q * sign[:, None]).view(np.uint64)
    else:
        d, q, s = blocks_q4_k(raw)
        nb, per = q.shape[0], 256
        sub = np.repeat(np.arange(8), 32)
        h0, jctx, jcol = counts(nb, per, cols, 16, lambda a, b: q[a:b], lambda a, b: s[a:b][:, sub], 64)
        out["hctx"] = cond_entropy(jctx)
        out["hscale"] = entropy(np.bincount(d.ravel(), minlength=65536))
        out["code_bits"] = 4
        packed = raw.reshape(-1, 144)
        w = packed.copy().view(np.uint64)
        wc = np.ascontiguousarray(packed[:, 16:144]).view(np.uint64)
        ws = None
    rows = nb // (cols // per)
    out["rows"], out["blocks"] = rows, nb
    out["zero"] = int((~q.any(axis=1)).sum())
    out["h0"] = entropy(h0)
    out["hcol"] = cond_entropy(jcol) if rows >= 256 else float("nan")
    h = hash_rows(w, K[0])
    out["rep_full"] = repeats(h)
    out["rep_codes"] = repeats(hash_rows(wc, K[0]))
    out["rep_sign"] = repeats(hash_rows(ws, K[0])) if ws is not None else 0
    out["rep_rows"] = repeats(hash_rows(h.reshape(rows, nb // rows), K[2]))
    glob_hashes.append(h)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--tensors", default=".*", help="regex on tensor names")
    args = ap.parse_args()
    r = gguf.GGUFReader(args.model)
    rx = re.compile(args.tensors)
    glob_hashes, res, skipped = [], [], {}
    for t in r.tensors:
        typ = t.tensor_type.name
        if not rx.search(t.name):
            continue
        if typ not in ("Q8_0", "Q4_K"):
            skipped[typ] = skipped.get(typ, 0) + int(t.n_bytes)
            continue
        o = scan_tensor(t, int(t.shape[0]), glob_hashes)
        res.append(o)
        print(f"{o['name']:28s} {o['type']:4s} {o['blocks']:9d} blocks; code bits of {o['code_bits']}: "
              f"order0 {o['h0']:.3f} ctx {o['hctx']:.3f} column {o['hcol']:.3f}; scale {o['hscale']:.2f}/16; "
              f"zero {o['zero']}; repeats: whole {o['rep_full']} codes {o['rep_codes']} "
              f"sign {o['rep_sign']} rows {o['rep_rows']}", flush=True)
    allh = np.concatenate(glob_hashes)
    rep_glob = repeats(allh)
    print(f"\nwhole file: {len(allh)} blocks, {sum(o['bytes'] for o in res) / 2**20:.0f} MiB scanned; "
          f"skipped (bytes by type) {skipped}")
    print(f"  blocks equal to another anywhere in the file: {rep_glob} ({rep_glob / len(allh) * 100:.4f}%)")
    for key, label in (("rep_full", "same tensor, whole"), ("rep_codes", "same tensor, codes only"),
                       ("rep_sign", "same tensor, codes up to sign"), ("zero", "all-zero codes")):
        n = sum(o[key] for o in res)
        print(f"  {label}: {n} ({n / len(allh) * 100:.4f}%)")
    print(f"  rows equal to another row of the same tensor: {sum(o['rep_rows'] for o in res)}")
    # the bytes an entropy coder would need: codes at their entropy, scales at theirs, the rest as is
    for typ in sorted({o["type"] for o in res}):
        sel = [o for o in res if o["type"] == typ]
        per, bb, sb = (32, 34, 2) if typ == "Q8_0" else (256, 144, 4)
        orig = sum(o["blocks"] * bb for o in sel)
        codes_orig = sum(o["blocks"] * per * o["code_bits"] / 8 for o in sel)
        scale = sum(o["blocks"] * sb * o["hscale"] / 16 for o in sel)
        rest = sum(o["blocks"] * (bb - sb - per * o["code_bits"] // 8) for o in sel)
        for key, label in (("h0", "order 0"), ("hctx", "per-block context"), ("hcol", "per column")):
            code = sum(o["blocks"] * per * (o[key] if o[key] == o[key] else o["h0"]) / 8 for o in sel)
            tot = code + scale + rest
            print(f"  {typ} {label:17s}: {orig / 2**20:6.0f} -> {tot / 2**20:6.0f} MiB ({tot / orig * 100:5.1f}%); "
                  f"codes {codes_orig / 2**20:.0f} -> {code / 2**20:.0f} MiB ({code / codes_orig * 100:.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
