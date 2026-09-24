#!/usr/bin/env python3
"""check_dequant.py -- the engine's scalar dequantization against gguf-py's, bit for bit.

gguf-py (llama.cpp's Python package, pinned in tools/requirements-oracle.txt) is what the
converters and the oracles read weights with: a weight the engine reads must be the very float
gguf-py reads. For each type, random blocks go through tests/dump_dequant (the engine's scalar
dequant_row, the definition every SIMD tier is held to by tests/test_kernels.c) and through
gguf.quants.dequantize; every float must have the same 32 bits (a NaN: any NaN).

The blocks: every byte random, so every nibble, every 6-bit scale and min and every packing of
Q4_K's 12 scale bytes occurs; the f16 scales ordinary (exponents 1..30, either sign) in 3 blocks
of 4 and any 16 bits in the fourth (zeros, subnormals, infinities, NaNs). The branches it covers,
and fails if never taken: per type, a block with a NaN, an infinite and a subnormal scale.

  tools/.venv/Scripts/python.exe tools/check_dequant.py --binary build/tests/dump_dequant.exe
  (in the gate: tools/.venv/bin/python ... --binary build/linux-gcc/tests/dump_dequant)
Seen red: tests/test_kernels.c and this file under a mutation of tr_q4_k_scales (the high bits of
min[4..7] taken from byte j instead of j + 4), docs/MEASUREMENTS.md §Q4_K.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

# name: (gguf type, elements per block, bytes per block, bytes of f16 scales at the block's head)
TYPES = {
    "q8_0": (GGMLQuantizationType.Q8_0, 32, 34, 2),
    "q4_k": (GGMLQuantizationType.Q4_K, 256, 144, 4),
}


def make_blocks(name, n, rng):
    _, _, nbytes, head = TYPES[name]
    raw = rng.integers(0, 256, size=(n, nbytes), dtype=np.uint8)
    halves = raw[:, :head].copy().view(np.uint16)
    ordinary = (halves & 0x83FF) | (rng.integers(1, 31, size=halves.shape, dtype=np.uint16) << 10)
    special = (np.arange(n) % 4 == 3)[:, None]
    halves = np.where(special, halves, ordinary).astype(np.uint16)
    # one block of each kind for sure: NaN, infinite, subnormal scale (the counters below)
    halves[3, 0], halves[7, 0], halves[11, 0] = 0x7E01, 0xFC00, 0x0003
    raw[:, :head] = halves.view(np.uint8).reshape(n, head)
    return raw


def check(binary, name, n_blocks, seed):
    qtype, elems, _, head = TYPES[name]
    raw = make_blocks(name, n_blocks, np.random.default_rng(seed))
    f16 = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    reached = {"nan": int(np.isnan(f16).sum()), "inf": int(np.isinf(f16).sum()),
               "subnormal": int(((f16 != 0) & (np.abs(f16) < 6.1035e-05)).sum())}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "blocks.bin"), os.path.join(tmp, "out.f32")
        raw.tofile(src)
        r = subprocess.run([binary, name, src, dst], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("check_dequant: %s %s failed (%d): %s" % (binary, name, r.returncode, r.stderr.strip()))
        got = np.fromfile(dst, dtype=np.float32)
    with np.errstate(all="ignore"):
        want = dequantize(raw, qtype).astype(np.float32).reshape(-1)
    if got.shape != want.shape:
        sys.exit("check_dequant: %s: %d floats from the engine, %d from gguf-py" % (name, got.size, want.size))
    same = (got.view(np.uint32) == want.view(np.uint32)) | (np.isnan(got) & np.isnan(want))
    bad = np.flatnonzero(~same)
    if bad.size:
        i = int(bad[0])
        sys.exit("check_dequant: %s: %d of %d floats differ from gguf-py; first at element %d (block %d): "
                 "engine %r (0x%08x), gguf-py %r (0x%08x)" %
                 (name, bad.size, got.size, i, i // elems, float(got[i]), int(got.view(np.uint32)[i]),
                  float(want[i]), int(want.view(np.uint32)[i])))
    missing = [k for k, v in reached.items() if v == 0]
    if missing:
        sys.exit("check_dequant: %s: no block with a %s scale: the special branches were not exercised" %
                 (name, ", ".join(missing)))
    print("check_dequant: %s: %d blocks, %d floats identical to gguf-py (scales: %d NaN, %d infinite, %d subnormal)"
          % (name, n_blocks, got.size, reached["nan"], reached["inf"], reached["subnormal"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="tests/dump_dequant built for this platform")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    for name in TYPES:
        check(args.binary, name, args.blocks, args.seed)


if __name__ == "__main__":
    main()
