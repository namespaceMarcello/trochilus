#!/usr/bin/env python3
"""rope_table_compare.py — the RoPE tables of two platforms (tests/dump_rope.c), entry by entry.

    rope_table_compare.py <dump A> <dump B> <n_pos> <head_dim> <theta>

tr_rope_table takes pow, cos and sin from the C library in double and rounds to float: after
tr_expf it is the one place where two platforms can still give the engine different numbers.
Prints how many of the doubles differ (pow per pair of dimensions, cos and sin per entry), how
many FLOAT entries differ (what the engine sees), where they are, and for each of those which
platform has the float the definition gives when every step is correctly rounded (mpmath, 200
bits): pow to double, 1/pow to double, angle = inv_freq * p to double, cos and sin to float.
"""
import struct
import sys

import mpmath
import numpy as np

from expf_hard_cases import round_to_float32_bits
from gen_expf_table import round_to_double


def load(path, n_pos, half):
    n = n_pos * half
    raw = open(path, "rb").read()
    want = n * 4 * 2 + half * 8 + n * 8 * 2
    if len(raw) != want:
        sys.exit("%s: %d bytes, %d expected" % (path, len(raw), want))
    o = 0
    out = {}
    for name, dt, count in (("cos_f", np.float32, n), ("sin_f", np.float32, n), ("pow_d", np.float64, half),
                            ("cos_d", np.float64, n), ("sin_d", np.float64, n)):
        size = np.dtype(dt).itemsize * count
        out[name] = np.frombuffer(raw[o:o + size], dtype=dt)
        o += size
    return out


def f32_of(v):
    """bits of the float nearest to the mpf v (any sign)."""
    if v == 0:
        return 0
    bits = round_to_float32_bits(abs(v))
    return bits | (0x80000000 if v < 0 else 0)


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        return 2
    n_pos, head_dim = int(sys.argv[3]), int(sys.argv[4])
    theta = struct.unpack("<f", struct.pack("<f", float(sys.argv[5])))[0]
    half = head_dim // 2
    a, b = load(sys.argv[1], n_pos, half), load(sys.argv[2], n_pos, half)
    mpmath.mp.prec = 200

    ideal_pow = [round_to_double(mpmath.power(mpmath.mpf(theta), mpmath.mpf(2 * i) / head_dim)) for i in range(half)]
    pow_diff = int((a["pow_d"].view(np.uint64) != b["pow_d"].view(np.uint64)).sum())
    a_pow_ok = sum(float(a["pow_d"][i]) == ideal_pow[i] for i in range(half))
    b_pow_ok = sum(float(b["pow_d"][i]) == ideal_pow[i] for i in range(half))
    print("pow(theta, 2i/d), %d doubles: %d differ between A and B; correctly rounded: A %d, B %d" %
          (half, pow_diff, a_pow_ok, b_pow_ok))
    for name in ("cos_d", "sin_d"):
        d = int((a[name].view(np.uint64) != b[name].view(np.uint64)).sum())
        print("%s, %d doubles: %d differ between A and B (%.4f%%)" % (name, a[name].size, d, 100.0 * d / a[name].size))

    total = 0
    for name, fn in (("cos_f", mpmath.cos), ("sin_f", mpmath.sin)):
        idx = np.nonzero(a[name].view(np.uint32) != b[name].view(np.uint32))[0]
        total += idx.size
        print("%s, %d floats (the engine's table): %d differ between A and B" % (name, a[name].size, idx.size))
        a_right = b_right = 0
        for n, flat in enumerate(idx):
            p, i = divmod(int(flat), half)
            inv = round_to_double(1 / mpmath.mpf(ideal_pow[i]))
            angle = round_to_double(mpmath.mpf(inv) * p) if p else 0.0
            want = f32_of(fn(mpmath.mpf(angle)))
            abits, bbits = int(a[name].view(np.uint32)[flat]), int(b[name].view(np.uint32)[flat])
            a_right += abits == want
            b_right += bbits == want
            if n < 12:
                print("  position %d, pair %d: A %08x, B %08x, definition %08x%s" %
                      (p, i, abits, bbits, want, "  (pow differs here)" if a["pow_d"][i] != b["pow_d"][i] else ""))
        if idx.size:
            print("  of those %d: A has the definition's float in %d, B in %d; first position %d, last %d" %
                  (idx.size, a_right, b_right, int(idx.min()) // half, int(idx.max()) // half))
    print("table entries that differ: %d of %d" % (total, 2 * a["cos_f"].size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
