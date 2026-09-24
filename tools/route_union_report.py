"""route_union_report.py — what a pass of k consecutive rows reads, and which experts fire together
(docs/MEASUREMENTS.md questions 54 and 62), from route traces (`run ... --route-trace <file>`).

  union      per k = 1..16: the distinct experts of k consecutive tokens, per layer, in units of one
             token's (n_used), averaged over windows and layers; against k independent random
             n_used-subsets of n_expert (what no locality would give). It sets the expert bytes of a
             speculative pass of k rows, or of any batch of consecutive tokens.
  pairs      per layer, over all traces: the lift of every pair of experts, P(i and j in the same
             token) / what independence gives under "exactly n_used a token"; the share of pairs above
             2 and below 0.5; how much of an expert's co-firings its top-3 partners hold.

  python tools/route_union_report.py <trace.bin>...    e.g. build/route/*.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_trace_report import Trace  # noqa: E402

KS = list(range(1, 17))


def chosen(path):
    t = Trace(open(path, "rb").read())
    ch = np.frombuffer(t.chosen.tobytes(), dtype=np.uint16).reshape(t.n_tokens, t.n_layers, t.n_used)
    return t, ch.astype(np.int64)


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    per_trace = {k: [] for k in KS}
    cnt, n_tok, E, U = None, 0, None, None
    for path in paths:
        t, ch = chosen(path)
        E, U = t.n_expert, t.n_used
        line = []
        for k in KS:
            n_win = t.n_tokens // k
            w = ch[: n_win * k].reshape(n_win, k, t.n_layers, U)
            onehot = np.zeros((n_win, t.n_layers, E), dtype=bool)
            for j in range(k):
                np.put_along_axis(onehot, w[:, j], True, axis=2)
            u = onehot.sum(axis=2).mean() / U
            per_trace[k].append(u)
            line.append(f"{u:.2f}")
        print(f"{os.path.basename(path):22s} {t.n_tokens:5d} tokens, union / {U}, k = 1..16: " + " ".join(line))
        oh = np.zeros((t.n_tokens, t.n_layers, E))
        np.put_along_axis(oh, ch, 1.0, axis=2)
        c = np.einsum("tli,tlj->lij", oh, oh)
        cnt = c if cnt is None else cnt + c
        n_tok += t.n_tokens
    print(f"{'mean':22s} {'':12s} " + " ".join(f"{np.mean(per_trace[k]):.2f}" for k in KS))
    print(f"{'random subsets':22s} {'':12s} " + " ".join(f"{E * (1 - (1 - U / E) ** k) / U:.2f}" for k in KS))
    print(f"\npairs of experts in one token, {n_tok} tokens pooled (uniform top-3 share: {3 / (E - 1) * 100:.1f}%)")
    iu = np.triu_indices(E, 1)
    for layer in range(cnt.shape[0]):
        p = np.diag(cnt[layer]) / n_tok
        pp = np.outer(p, p)[iu]
        lift = (cnt[layer][iu] / n_tok) / np.maximum(pp * (U * (U - 1) / 2) / pp.sum(), 1e-12)
        m = cnt[layer].copy()
        np.fill_diagonal(m, 0)
        top3 = (np.sort(m, axis=1)[:, -3:].sum(axis=1) / np.maximum(m.sum(axis=1), 1)).mean()
        top = np.sort(lift)[::-1]
        print(f"  layer {layer:2d}: lift max {top[0]:5.1f}, 10th {top[9]:5.1f}; pairs above 2 {np.mean(lift > 2) * 100:4.1f}%, "
              f"below 0.5 {np.mean(lift < 0.5) * 100:4.1f}%; top-3 partners {top3 * 100:4.1f}% of co-firings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
