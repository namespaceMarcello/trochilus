#!/usr/bin/env python3
"""Trochilus against llama.cpp on the same GGUF and prompt (docs/STATO.md, correctness on a real model).

Both engines get the same token ids. The script compares:
  1. tokenization of the prompt file (Trochilus `tokenize` against llama.cpp `llama-tokenize`);
  2. greedy generation of -n tokens (Trochilus `generate` against tools/llamacpp_logits.c);
  3. the logits at every position of prompt + Trochilus' generated tokens (teacher forcing): the
     chosen token, KL(llama.cpp || Trochilus) and, where the choice differs, the margin between the
     two best tokens (a near tie is arithmetic, a wide gap is a bug).

llama.cpp multiplies Q8_0 weights with activations quantized to 8 bits too, Trochilus with exact
activations: small logit differences are expected, systematic disagreement is not.

Runs in the trochilus-dev container (Linux binaries):
  tools/compare_llamacpp.py --trochilus build/linux-gcc/trochilus \
      --llamacpp-dir ref/llama.cpp/build-trochilus/bin --model <file.gguf> --prompt <file> [-n 128]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(args):
    r = subprocess.run([str(a) for a in args], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, args))}\n{r.stderr.decode('utf-8', 'replace')[-2000:]}")
    return r.stdout.decode("utf-8", "replace")


def parse_tokens_line(out):
    line = [l for l in out.splitlines() if l.startswith("tokens:")][0]
    return [int(x) for x in line.split(":", 1)[1].strip().split(",") if x]


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trochilus", required=True)
    ap.add_argument("--llamacpp-dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("-t", type=int, default=8)
    ap.add_argument("--hf", default="fixtures/olmoe-1b-7b-0125-instruct-tokenizer", help="only to show text")
    args = ap.parse_args()
    ok = True

    ids = [int(x) for x in run([args.trochilus, "tokenize", "-m", args.model, "-f", args.prompt]).strip().split(",")]
    ll_out = run([args.llamacpp_dir / "llama-tokenize", "-m", args.model, "-f", args.prompt, "--ids", "--log-disable"])
    ll_ids = [int(x) for x in ll_out.strip().splitlines()[-1].strip("[] ").split(",") if x.strip()]
    same_tok = ids == ll_ids
    ok &= same_tok
    print(f"prompt tokens: {len(ids)}, identical to llama.cpp: {same_tok}")
    if not same_tok:
        print(f"  trochilus: {ids}\n  llama.cpp: {ll_ids}")

    tok_str = ",".join(map(str, ids))
    with tempfile.TemporaryDirectory(prefix="cmp_llamacpp_") as tmp:
        tmp = Path(tmp)
        gen_tr = parse_tokens_line(run([args.trochilus, "generate", "-m", args.model, "--tokens", tok_str,
                                        "-n", args.n, "-t", args.t]))
        gen_ll = parse_tokens_line(run([args.llamacpp_dir / "llamacpp_logits", "-m", args.model, "--tokens", tok_str,
                                        "--out", tmp / "unused.bin", "-n", args.n, "-t", args.t]))
        first = next((i for i, (a, b) in enumerate(zip(gen_tr, gen_ll)) if a != b), None)
        print(f"greedy {args.n} tokens: identical {'up to the end' if first is None else f'for the first {first}'}")

        seq = ids + gen_tr
        seq_str = ",".join(map(str, seq))
        run([args.trochilus, "logits", "-m", args.model, "--tokens", seq_str, "--out", tmp / "tr.bin", "-t", args.t])
        run([args.llamacpp_dir / "llamacpp_logits", "-m", args.model, "--tokens", seq_str, "--out", tmp / "ll.bin",
             "-t", args.t])
        tr = np.fromfile(tmp / "tr.bin", dtype=np.float32).reshape(len(seq), -1)
        ll = np.fromfile(tmp / "ll.bin", dtype=np.float32).reshape(len(seq), -1)

    lp_tr, lp_ll = log_softmax(tr), log_softmax(ll)
    kl = (np.exp(lp_ll) * (lp_ll - lp_tr)).sum(axis=-1)
    top_tr, top_ll = tr.argmax(axis=-1), ll.argmax(axis=-1)
    agree = top_tr == top_ll
    print(f"positions: {len(seq)} (prompt {len(ids)} + generated {len(gen_tr)})")
    print(f"same top token: {int(agree.sum())}/{len(seq)}")
    print(f"KL(llama.cpp || trochilus): mean {kl.mean():.2e}, max {kl.max():.2e}")
    print(f"max |logit difference|: {np.abs(tr - ll).max():.4f}")
    for i in np.nonzero(~agree)[0]:
        s_tr, s_ll = np.sort(tr[i])[::-1], np.sort(ll[i])[::-1]
        gap_tr, gap_ll = s_tr[0] - s_tr[1], s_ll[0] - s_ll[1]
        print(f"  position {i}: trochilus {top_tr[i]} (margin {gap_tr:.4f}), llama.cpp {top_ll[i]} (margin {gap_ll:.4f})")
        if max(gap_tr, gap_ll) > 0.1:
            ok = False
    if kl.max() > 1e-2:
        ok = False

    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(args.hf)
        print("trochilus:", repr(t.decode(gen_tr)))
        if first is not None:
            print("llama.cpp:", repr(t.decode(gen_ll)))
    except Exception as exc:
        print(f"(text not shown: {exc})")
    print("RESULT:", "same model behaviour" if ok else "DIFFERENT: look at the positions above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
