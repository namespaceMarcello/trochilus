#!/usr/bin/env python3
"""gen_expf_table.py — writes src/kernels/expf_table.h, every constant of tr_expf (src/kernels/expf.c).

tr_expf is exp(x) correctly rounded to float on all 2^32 floats, with no C library call inside
(docs/MEASUREMENTS.md question 37). Its numbers come from here, computed with mpmath at 200 bits and
rounded with integer arithmetic, never from a C library (whose exp2 is not the same on two
platforms):

  the table      64 doubles, 2^(j/64) rounded to nearest
  the reduction  64/ln2, and ln2/64 in two pieces: the first a multiple of 2^-38 (32 bits, fdlibm's
                 split), so that k * hi is exact for every k the function meets, the second the
                 rest rounded to double
  the polynomial 1/6, 1/24, 1/120 rounded to double (Taylor, degree 5: |r| <= ln2/128)
  the borders    the largest float whose exp is finite
  the exceptions the arguments whose fast path cannot be trusted: the double it computes is so
                 close to the middle between two floats that an error of 2^-50 could move the
                 rounding. For each, exp(x) at 200 bits rounded to binary32 (and again at 400
                 bits: the same). They are SLOW_ARGS below; this script finds them again with
                 --scan, a bit-exact numpy mirror of the C fast path over every float in range.

    gen_expf_table.py            write the header
    gen_expf_table.py --check    exit 1 if the header on disk is not what this would write
    gen_expf_table.py --scan     find the exceptions again (about 5 minutes) and compare
    gen_expf_table.py --slow F   compare with the arguments `bench_expf --slow F` wrote

The proof that the function is right is not here: it is tests/bench_expf.c --check, every float
against the reference, in `make check`.
"""
import math
import struct
import sys
from pathlib import Path

import mpmath

from expf_hard_cases import round_to_float32_bits

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "src" / "kernels" / "expf_table.h"

MARGIN_LOG2 = -50          # the rounding test of expf.c: an error of 2^-50 must not move the float
UNDERFLOW_X = -104.0       # exp(-104) < 2^-150: rounds to zero

# bits of the arguments that fail the rounding test, found by --scan (and by bench_expf --slow)
SLOW_ARGS = [
    0x377eff81,
    0x39c6be5b,
    0x4001b249,
    0x40315b33,
    0xb3000000,
    0xbae0e25c,
    0xbbf0edf1,
    0xc16912cd,
]


def round_to_double(v):
    """the double nearest to the positive mpf v, ties to even."""
    man, exp = int(v.man), int(v.exp)             # v = man * 2^exp exactly
    e = man.bit_length() + exp - 1                # 2^e <= v < 2^(e+1)
    shift = (e - 52) - exp
    if shift <= 0:
        q = man << -shift
    else:
        q, rem = divmod(man, 1 << shift)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
    return math.ldexp(q, e - 52)


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def constants():
    mpmath.mp.prec = 200
    ln2 = mpmath.log(2)
    c = {}
    c["table"] = [round_to_double(mpmath.power(2, mpmath.mpf(j) / 64)) for j in range(64)]
    c["inv"] = round_to_double(64 / ln2)
    hi = mpmath.floor(ln2 / 64 * 2**38) / 2**38
    c["hi"] = round_to_double(hi)
    assert mpmath.mpf(c["hi"]) == hi, "hi must be exact in a double"
    c["lo"] = round_to_double(ln2 / 64 - hi)
    c["c3"] = round_to_double(mpmath.mpf(1) / 6)
    c["c4"] = round_to_double(mpmath.mpf(1) / 24)
    c["c5"] = round_to_double(mpmath.mpf(1) / 120)
    # the largest float whose exp does not round to infinity, found from above ln(FLT_MAX)
    bits = f32_bits(89.0)
    while round_to_float32_bits(mpmath.exp(mpmath.mpf(f32(bits)))) == 0x7F800000:
        bits -= 1
    c["overflow_bits"] = bits
    # below UNDERFLOW_X everything rounds to zero: the largest value there is under half a subnormal
    assert mpmath.exp(mpmath.mpf(UNDERFLOW_X)) < mpmath.mpf(2) ** -150
    return c


def fast_path(c, x):
    """(y, a bits, b bits): the C fast path on one float, the same doubles operation by operation."""
    z = x * c["inv"]
    kd = (z + 1.5 * 2.0**52) - 1.5 * 2.0**52
    k = int(kd)
    r = (x - kd * c["hi"]) - kd * c["lo"]
    p = r + r * r * (0.5 + r * (c["c3"] + r * (c["c4"] + r * c["c5"])))
    t = c["table"][k & 63]
    y = (t + t * p) * math.ldexp(1.0, k >> 6)
    m = 2.0**MARGIN_LOG2
    return y, f32_bits(y * (1.0 - m)), f32_bits(y * (1.0 + m))


def exception_value(xbits):
    """bits of exp(x) correctly rounded to binary32, at 200 bits and again at 400."""
    got = []
    for prec in (200, 400):
        mpmath.mp.prec = prec
        got.append(round_to_float32_bits(mpmath.exp(mpmath.mpf(f32(xbits)))))
    mpmath.mp.prec = 200
    assert got[0] == got[1], "exp(%r): 200 and 400 bits round differently" % f32(xbits)
    return got[0]


def scan(c):
    """bits of every float in range whose fast path fails the rounding test (numpy mirror)."""
    import numpy as np
    table = np.array(c["table"])
    m = 2.0**MARGIN_LOG2
    found = []
    ranges = [(0, c["overflow_bits"] + 1), (0x80000000, f32_bits(UNDERFLOW_X) + 1)]
    step = 1 << 22
    for lo, hi in ranges:
        for start in range(lo, hi, step):
            xb = np.arange(start, min(start + step, hi), dtype=np.uint32)
            x = xb.view(np.float32).astype(np.float64)
            z = x * c["inv"]
            kd = (z + 1.5 * 2.0**52) - 1.5 * 2.0**52
            k = kd.astype(np.int64)
            r = (x - kd * c["hi"]) - kd * c["lo"]
            p = r + r * r * (0.5 + r * (c["c3"] + r * (c["c4"] + r * c["c5"])))
            t = table[k & 63]
            y = (t + t * p) * np.ldexp(1.0, (k >> 6).astype(np.int32))
            a = (y * (1.0 - m)).astype(np.float32).view(np.uint32)
            b = (y * (1.0 + m)).astype(np.float32).view(np.uint32)
            found.extend(int(v) for v in xb[a != b])
    return sorted(found)


def render(c):
    mpmath.mp.prec = 200
    rows = []
    for xbits in SLOW_ARGS:
        _, a, b = fast_path(c, f32(xbits))
        assert a != b, "%08x passes the rounding test: not an exception" % xbits
        rows.append((xbits, exception_value(xbits)))
    out = []
    w = out.append
    w("/* expf_table.h — GENERATED by tools/gen_expf_table.py (mpmath %s, 200 bits). Do not edit:" % mpmath.__version__)
    w(" * change the script and run it again; tools/lint.py compares this file with what it writes.")
    w(" * The constants of tr_expf (expf.c); what each one is, and why, is said there and in the script. */")
    w("#ifndef TR_EXPF_TABLE_H")
    w("#define TR_EXPF_TABLE_H")
    w("")
    w("#include <stdint.h>")
    w("")
    w("#define TR_EXPF_INV_LN2_64 %s /* 64 / ln2 */" % c["inv"].hex())
    w("#define TR_EXPF_LN2_64_HI %s /* ln2 / 64, a multiple of 2^-38: k * hi is exact */" % c["hi"].hex())
    w("#define TR_EXPF_LN2_64_LO %s /* the rest of ln2 / 64 */" % c["lo"].hex())
    w("#define TR_EXPF_C3 %s /* 1/6 */" % c["c3"].hex())
    w("#define TR_EXPF_C4 %s /* 1/24 */" % c["c4"].hex())
    w("#define TR_EXPF_C5 %s /* 1/120 */" % c["c5"].hex())
    w("#define TR_EXPF_MARGIN 0x1p%d /* the error the fast path is allowed before it stops being trusted */" % MARGIN_LOG2)
    w("#define TR_EXPF_OVERFLOW_X %sf /* the largest float whose exp is finite */" % float(f32(c["overflow_bits"])).hex())
    w("#define TR_EXPF_UNDERFLOW_X %sf /* below it exp(x) < 2^-150 rounds to zero */" % float(UNDERFLOW_X).hex())
    w("")
    w("/* 2^(j/64), j = 0..63, rounded to nearest */")
    w("static const double tr_expf_pow2[64] = {")
    for j in range(0, 64, 4):
        w("    " + " ".join("%s," % v.hex() for v in c["table"][j:j + 4]))
    w("};")
    w("")
    w("/* The arguments the fast path cannot settle, and exp of each correctly rounded (200 bits):")
    w(" * bits of x, bits of the result. tests/bench_expf.c --check proves there is no other. */")
    w("#define TR_EXPF_N_EXCEPTIONS %d" % len(rows))
    w("static const uint32_t tr_expf_exceptions[%d][2] = {" % max(len(rows), 1))
    for xbits, ybits in rows:
        w("    {0x%08xu, 0x%08xu}, /* exp(%s) */" % (xbits, ybits, float(f32(xbits)).hex()))
    if not rows:
        w("    {0xffffffffu, 0xffffffffu}, /* none: a NaN never gets here */")
    w("};")
    w("")
    w("#endif")
    return "\n".join(out) + "\n"


def main():
    c = constants()
    args = sys.argv[1:]
    if args[:1] == ["--scan"]:
        found = scan(c)
        print("SLOW_ARGS = [\n%s]" % "".join("    0x%08x,\n" % b for b in found))
        print("%d arguments fail the rounding test; SLOW_ARGS in the script: %s" %
              (len(found), "the same" if found == sorted(SLOW_ARGS) else "DIFFERENT"))
        return 0 if found == sorted(SLOW_ARGS) else 1
    if args[:1] == ["--slow"] and len(args) == 2:
        with open(args[1], encoding="utf-8") as f:
            found = sorted(int(line.split()[0], 16) for line in f if line.strip())
        same = found == sorted(SLOW_ARGS)
        print("%d arguments in %s; SLOW_ARGS in the script: %s" % (len(found), args[1], "the same" if same else "DIFFERENT"))
        return 0 if same else 1
    text = render(c)
    if args[:1] == ["--check"]:
        on_disk = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if on_disk != text:
            print("%s is not what tools/gen_expf_table.py writes: run the script" % OUT_PATH.relative_to(ROOT))
            return 1
        return 0
    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print("wrote %s: 64 table entries, %d exceptions" % (OUT_PATH.relative_to(ROOT), len(SLOW_ARGS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
