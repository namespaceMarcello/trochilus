#!/usr/bin/env python3
"""expf_hard_cases.py — is (float)exp((double)x) really the correctly rounded exp(x)?

tests/bench_expf.c compares the C library's expf, and a candidate of ours, with the double
precision exp rounded to float, over all 2^32 floats. That reference is wrong only where the
double is: within one unit of ITS last place of the middle between two floats. `bench_expf --hard
<file>` writes every argument whose double exp is within 2^-45 of such a border (a wide margin:
about 1500 of 4 billion), as two hex words per line: the bits of x and the bits of the reference.
This computes exp(x) with 200 bits (mpmath), rounds it to binary32 to nearest even with integer
arithmetic, subnormals included, and compares.

    expf_hard_cases.py <file> [<file> ...]

No difference: on those 2^32 arguments the reference IS the correctly rounded value, so a library
whose expf agrees with it everywhere (MinGW-w64: docs/MISURE.md question 37) rounds correctly
everywhere, and "zero differences from the library" for an expf of ours is a proof.
"""
import struct
import sys

import mpmath


def round_to_float32_bits(v):
    """bits of the binary32 nearest to the positive mpf v, ties to even."""
    man, exp = int(v.man), int(v.exp)             # v = man * 2^exp exactly
    e = man.bit_length() + exp - 1                # 2^e <= v < 2^(e+1)
    qexp = -149 if e < -126 else e - 23           # the quantum of the result: 2^qexp
    shift = qexp - exp
    if shift <= 0:
        q = man << -shift
    else:
        q, rem = divmod(man, 1 << shift)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
    if e < -126:
        return q                                  # subnormal; 2^23 is already the smallest normal
    if q == 1 << 24:                              # the rounding carried into the next binade
        q, e = 1 << 23, e + 1
    if e + 127 >= 255:
        return 0x7F800000
    return ((e + 127) << 23) | (q - (1 << 23))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mpmath.mp.prec = 200
    total = wrong = 0
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                xbits, refbits = int(parts[0], 16), int(parts[1], 16)
                x = struct.unpack("<f", struct.pack("<I", xbits))[0]
                want = round_to_float32_bits(mpmath.exp(mpmath.mpf(x)))
                total += 1
                if want != refbits:
                    wrong += 1
                    print("x = %r (%08x): reference %08x, correctly rounded %08x" % (x, xbits, refbits, want))
    print("%d arguments near a rounding border checked with %d bits: %d where the reference is not the correctly "
          "rounded value" % (total, mpmath.mp.prec, wrong))
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
