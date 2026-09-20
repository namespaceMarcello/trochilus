#!/usr/bin/env python3
"""expf_quality.py — two builds of the engine on the same tokens: what a different expf changes.

    expf_quality.py <logits A> <logits B> [<tokens A> <tokens B>] [--vocab 50304]

The logits files are `trochilus logits` outputs over the same token sequence (float32, one row of
the vocabulary per position); the token files hold the comma separated ids each build generated
greedily from the same prompt. Prints, over the positions: how many rows are the same bytes (and
the first that is not), the largest difference between two logits, the mean and the largest
KL(A || B), how many positions choose another token; and, with the token files, how many
generated tokens are the same before the first difference.
Used by tools/expf_quality.sh (docs/MISURE.md question 37) and tools/platform_bits.sh.
"""
import sys

import numpy as np


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def read_tokens(path):
    with open(path, encoding="utf-8") as f:
        return [int(t) for t in f.read().replace("\n", "").split(",") if t.strip()]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    vocab = 50304
    if "--vocab" in sys.argv:
        vocab = int(sys.argv[sys.argv.index("--vocab") + 1])
        args.remove(str(vocab))
    if len(args) not in (2, 4):
        print(__doc__)
        return 2
    a = np.fromfile(args[0], dtype=np.float32).reshape(-1, vocab)
    b = np.fromfile(args[1], dtype=np.float32).reshape(-1, vocab)
    if a.shape != b.shape:
        print("the two logits files do not have the same shape: %s, %s" % (a.shape, b.shape))
        return 1
    row_same = (a.view(np.uint32) == b.view(np.uint32)).all(axis=1)
    same_rows = int(row_same.sum())
    if same_rows < a.shape[0]:
        print("first position whose row differs: %d" % int(np.argmin(row_same)))
    la, lb = log_softmax(a), log_softmax(b)
    kl = (np.exp(la) * (la - lb)).sum(axis=1)
    other = int((a.argmax(axis=1) != b.argmax(axis=1)).sum())
    print("positions: %d, of %d logits each" % a.shape)
    print("rows that are the same bytes: %d (%.1f%%)" % (same_rows, 100.0 * same_rows / a.shape[0]))
    print("largest difference between two logits: %.3g (logits range over %.1f)" %
          (float(np.abs(a.astype(np.float64) - b).max()), float(a.max() - a.min())))
    print("KL(A || B): mean %.3g, largest %.3g" % (float(kl.mean()), float(kl.max())))
    print("positions that choose another token: %d" % other)
    if len(args) == 2:
        return 0
    ta, tb = read_tokens(args[2]), read_tokens(args[3])
    same = 0
    while same < min(len(ta), len(tb)) and ta[same] == tb[same]:
        same += 1
    print("generated tokens: %d and %d, the same for the first %d%s" %
          (len(ta), len(tb), same, " (all of them)" if same == len(ta) == len(tb) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
