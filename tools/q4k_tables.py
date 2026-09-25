"""q4k_tables.py — how many distinct 16-value weight tables a Q4_K model has (docs/MEASUREMENTS.md
question 66, premise e: "the dictionary of tables").

The exact Q4_K kernel builds, for each sub-block of 32 weights, the table of its 16 possible
weights, scale * q - min for q = 0..15, with scale = d * sc and min = dmin * m (floats). If a
tensor had few distinct (scale, min) pairs, the tables could be built once at load and looked up
(a load instead of two broadcasts, a multiply and a subtract). This counts, per Q4_K tensor, the
distinct (scale, min) float pairs over its sub-blocks, and how many sub-blocks the most common
16384 pairs cover (16384 tables of 64 bytes fill a Zen 4 core's 1 MiB L2).

  tools/.venv/Scripts/python.exe tools/q4k_tables.py <model.gguf> [--tensors <regex>]
"""
import argparse
import re

import numpy as np

import gguf

L2_TABLES = 16384


def pairs_q4_k(raw):
    """(n*8,) uint64: the float bits of scale (high word) and min (low word) of every sub-block"""
    b = raw.reshape(-1, 144)
    h = b[:, 0:4].copy().view(np.float16).astype(np.float32)  # d, dmin: exact
    sc = b[:, 4:16]
    s = np.empty((b.shape[0], 8), dtype=np.float32)
    m = np.empty((b.shape[0], 8), dtype=np.float32)
    s[:, :4] = sc[:, 0:4] & 63
    m[:, :4] = sc[:, 4:8] & 63
    s[:, 4:] = (sc[:, 8:12] & 0x0F) | ((sc[:, 0:4] >> 6) << 4)
    m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    scale = (h[:, 0:1] * s).astype(np.float32)  # float products, as tr_q4_k_scales
    mn = (h[:, 1:2] * m).astype(np.float32)
    return (scale.view(np.uint32).astype(np.uint64) << np.uint64(32)) | mn.view(np.uint32).astype(np.uint64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--tensors", default=".*", help="regex on tensor names")
    a = ap.parse_args()
    r = gguf.GGUFReader(a.model)
    pat = re.compile(a.tensors)
    tot_sub = tot_dist = tot_cov = 0
    print(f"{'tensor':40s} {'sub-blocks':>11s} {'distinct':>10s} {'distinct%':>9s} {'top16K cover%':>13s}")
    for t in r.tensors:
        if t.tensor_type.name != "Q4_K" or not pat.search(t.name):
            continue
        p = pairs_q4_k(np.asarray(t.data).reshape(-1))
        _, cnt = np.unique(p, return_counts=True)
        cover = np.sort(cnt)[::-1][:L2_TABLES].sum()
        tot_sub += p.size
        tot_dist += cnt.size
        tot_cov += cover
        print(f"{t.name:40s} {p.size:11d} {cnt.size:10d} {100 * cnt.size / p.size:8.1f}% {100 * cover / p.size:12.1f}%")
    if tot_sub:
        print(f"{'all Q4_K tensors':40s} {tot_sub:11d} {tot_dist:10d} {100 * tot_dist / tot_sub:8.1f}% "
              f"{100 * tot_cov / tot_sub:12.1f}%  (distinct and cover summed per tensor)")


if __name__ == "__main__":
    main()
