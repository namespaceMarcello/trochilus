"""kv_repeats_report.py — which layers' keys and values repeat exactly across positions (docs/MEASUREMENTS.md
question 62): a layer whose value row is the same bytes wherever the same token sits is determined by the
token alone, and could be stored once per distinct token instead of once per position.

Reads the dumps of `make attn-probe` (tools/attn_probe.c: <dir>/<run>/{k,v}_L<l>_H<h>.bin, positions x
head_dim float32). Per run and layer: the share of positions whose value row (per head, and in every
head at once) equals an earlier position's, and the same for keys (after RoPE: a repeat would need the
same token at the same position, so 0 is expected).

  python tools/kv_repeats_report.py <probe dir> [head_dim]
"""
import os
import sys

import numpy as np


def rep_share(a):
    """share of rows equal to an earlier row, and the group id of every row"""
    _, inv = np.unique(a.view(np.dtype((np.void, a.shape[1] * 4))).ravel(), return_inverse=True)
    return 1 - len(np.unique(inv)) / a.shape[0], inv.ravel().astype(np.int64)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    hd = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    for run in sorted(os.listdir(root)):
        d = os.path.join(root, run)
        if not os.path.isdir(d):
            continue
        layers = sorted({int(f[3:5]) for f in os.listdir(d) if f.startswith("v_L")})
        heads = sorted({int(f[7:9]) for f in os.listdir(d) if f.startswith("v_L")})
        print(run)
        for L in layers:
            vr, kr, groups, n = [], [], None, 0
            for H in heads:
                v = np.fromfile(os.path.join(d, "v_L%02d_H%02d.bin" % (L, H)), dtype=np.float32).reshape(-1, hd)
                k = np.fromfile(os.path.join(d, "k_L%02d_H%02d.bin" % (L, H)), dtype=np.float32).reshape(-1, hd)
                n = v.shape[0]
                s, inv = rep_share(v)
                vr.append(s)
                kr.append(rep_share(k)[0])
                groups = inv if groups is None else groups * (n + 1) + inv
            every = 1 - len(np.unique(groups)) / n
            print(f"  layer {L:2d}, {n} positions: values repeated {np.mean(vr) * 100:5.1f}% (in every head at once "
                  f"{every * 100:5.1f}%), keys {np.mean(kr) * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
