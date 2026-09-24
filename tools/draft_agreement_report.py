"""draft_agreement_report.py — a quantized draft against the exact model, from tools/draft_agreement.sh's
logits (docs/MEASUREMENTS.md question 53).

Per text: the exact model's greedy tokens through `logits -b 1` must be the ones `generate` gave
(the engine's exactness, checked first: the script fails otherwise); top-1 agreement between exact
and draft on the real text (teacher-forced) and on the exact greedy trajectory; KL(exact || draft);
next-token accuracy of both on the real text; agreement by the exact model's top-2 margin; tokens a
pass of a k-token draft verified exactly would give (accepted leading matches + 1), and with the
draft cut where its own top-2 margin falls under tau.

  python tools/draft_agreement_report.py <dir> <draft tag> [vocab]   e.g. models/draft q4_k
"""
import sys

import numpy as np


def ids(path):
    return [int(x) for x in open(path).read().strip().split(",") if x]


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def passes(match, start, end, k, cont=None):
    """tokens per pass of a draft of up to k tokens verified exactly; cont[i]: the draft goes on
    after drafting position i. Returns tokens/pass, accepted share of drafted, rows/pass."""
    s, n_pass, n_tok, n_acc, n_rows, n_draft = start, 0, 0, 0, 0, 0
    while s < end:
        n = 1
        while n < k and s + n < end and (cont is None or cont[s + n - 1]):
            n += 1
        a = 0
        while a < n and s + a < end and match[s + a]:
            a += 1
        n_pass += 1
        n_acc += a
        n_draft += n
        n_rows += n + 1
        n_tok += a + 1
        s += a + 1
    return n_tok / n_pass, n_acc / n_draft, n_rows / n_pass


def report(o, name, tag, vocab):
    P, G = ids(f"{o}/{name}.prompt"), ids(f"{o}/{name}.gen")
    A = P + G
    N, np_ = len(A), len(P)
    ex = np.memmap(f"{o}/{name}.exact.bin", dtype=np.float32, mode="r").reshape(-1, vocab)
    dr = np.memmap(f"{o}/{name}.{tag}.bin", dtype=np.float32, mode="r").reshape(-1, vocab)
    assert ex.shape[0] == N and dr.shape[0] == N, (ex.shape, dr.shape, N)
    ae, ad = np.asarray(ex.argmax(axis=1)), np.asarray(dr.argmax(axis=1))
    gen_rows = np.arange(np_ - 1, N - 1)
    same = np.mean(ae[gen_rows] == np.array(A[np_:]))
    if same != 1.0:
        print(f"{name}: generate and logits -b 1 disagree on {(1 - same) * 100:.2f}% of the greedy tokens")
        return 1
    kl, marg, margd = np.empty(N), np.empty(N), np.empty(N)
    for i in range(0, N, 128):
        le, ld = logsoftmax(ex[i:i + 128]), logsoftmax(dr[i:i + 128])
        kl[i:i + 128] = (np.exp(le) * (le - ld)).sum(axis=1)
        s, sd = np.sort(le, axis=1), np.sort(ld, axis=1)
        marg[i:i + 128] = s[:, -1] - s[:, -2]
        margd[i:i + 128] = sd[:, -1] - sd[:, -2]
    match = ae == ad
    real = np.arange(0, np_ - 1)
    acc_e = np.mean(ae[real] == np.array(A[1:np_]))
    acc_d = np.mean(ad[real] == np.array(A[1:np_]))
    seen, rep = set(), 0
    for j in range(3, N):
        g = tuple(A[j - 3:j + 1])
        rep += j >= np_ and g in seen
        seen.add(g)
    print(f"== {name}, draft {tag}: prompt {np_}, generated {len(G)} ({rep / len(G) * 100:.1f}% of its 4-grams seen "
          f"before); greedy through logits -b 1 = generate's")
    print(f"  real text ({len(real)} rows): agreement {match[real].mean() * 100:.2f}%, KL mean {kl[real].mean():.2e} "
          f"median {np.median(kl[real]):.2e}; next-token accuracy exact {acc_e * 100:.1f}% draft {acc_d * 100:.1f}%")
    print(f"  exact greedy trajectory ({len(gen_rows)} rows): agreement {match[gen_rows].mean() * 100:.2f}%, "
          f"KL mean {kl[gen_rows].mean():.2e}")
    for k in (4, 8, 16):
        t_r, f_r, _ = passes(match, 0, np_ - 1, k)
        t_g, f_g, r_g = passes(match, np_ - 1, N - 1, k)
        print(f"  draft {k:2d}: tokens/pass real text {t_r:.2f} ({f_r * 100:.0f}% of drafted accepted), "
              f"trajectory {t_g:.2f} ({f_g * 100:.0f}%), rows/pass {r_g:.2f}")
    for tau in (0.5, 1.0, 2.0):
        cont = margd >= tau
        t_g, f_g, r_g = passes(match, np_ - 1, N - 1, 8, cont)
        t_r, _, r_r = passes(match, 0, np_ - 1, 8, cont)
        print(f"  draft <= 8 cut at its margin {tau}: trajectory {t_g:.2f} tokens in {r_g:.2f} rows "
              f"({f_g * 100:.0f}% accepted); real text {t_r:.2f} in {r_r:.2f}")
    rows = np.arange(0, N - 1)
    for lo, hi in [(0, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, np.inf)]:
        sel = rows[(marg[rows] >= lo) & (marg[rows] < hi)]
        if len(sel):
            print(f"  exact margin [{lo}, {hi}) nats: {len(sel):5d} rows, agreement {match[sel].mean() * 100:6.2f}%")
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    o, tag = sys.argv[1], sys.argv[2]
    vocab = int(sys.argv[3]) if len(sys.argv) > 3 else 50304
    rc = 0
    for name in ("prose", "code", "ital"):
        rc |= report(o, name, tag, vocab)
    return rc


if __name__ == "__main__":
    sys.exit(main())
