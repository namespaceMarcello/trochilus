#!/usr/bin/env python3
"""oracle.py — compare Trochilus's forward pass against the transformers
reference captured in <fixture_dir>/ref.json.

Two fixture shapes are understood:
  * single-prompt (tools/make_tiny_olmoe.py): ref.json has prompt_ids/full_ids/logits
    for one prompt, logits as a nested JSON array.
  * multi-prompt (tools/make_olmoe_2layer_ref.py, `make oracle-real`): ref.json has a
    "prompts" list, each with name/prompt_ids/full_ids/logits_file; logits are a
    separate float32 binary per prompt (the same layout `trochilus logits --out`
    writes), read with numpy instead of nested JSON.

Usage:
    oracle.py <fixture_dir> <model.gguf> [--binary <path>] [--expect exact|report] [--logit-tol X]

Runs `<binary> generate` on the prompt ids for len(full_ids) - len(prompt_ids)
greedy tokens, and `<binary> logits` on the full teacher-forced sequence. Prints
token match, max abs logit diff (overall and per position), and argmax agreement.

--expect exact (default): exit 1 unless every prompt's generated tokens match
    exactly and the overall max abs logit diff is <= --logit-tol (default 1e-3).
--expect report: always exit 0 (results are still printed).
"""
import argparse
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np


def default_binary():
    return "build/trochilus.exe" if platform.system() == "Windows" else "build/trochilus"


def peak_rss_mb():
    """(this process, largest child so far) peak RSS in MB, or (None, None) on
    Windows (no resource module) -- the container run reports the real numbers."""
    try:
        import resource
    except ImportError:
        return None, None
    div = 1024 * 1024 if sys.platform == "darwin" else 1024
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / div
    return self_rss, child_rss


def parse_tokens_line(stdout):
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("tokens:"):
            rest = line[len("tokens:"):].strip()
            if not rest:
                return []
            return [int(x) for x in rest.split(",")]
    return None


def read_logits(path, n_tokens, vocab):
    with open(path, "rb") as f:
        data = f.read()
    expected = n_tokens * vocab * 4
    if len(data) != expected:
        raise RuntimeError("logits file size %d != expected %d" % (len(data), expected))
    values = struct.unpack("<%df" % (n_tokens * vocab), data)
    return [values[i * vocab:(i + 1) * vocab] for i in range(n_tokens)]


def read_logits_np(path, n_tokens, vocab):
    """Same binary layout as read_logits, but as one numpy array: the vocab here
    (tens of thousands) makes the pure-Python per-element loop too slow."""
    arr = np.fromfile(path, dtype="<f4")
    expected = n_tokens * vocab
    if arr.size != expected:
        raise RuntimeError("logits file size %d floats != expected %d" % (arr.size, expected))
    return arr.reshape(n_tokens, vocab)


def run_generate(binary, model, prompt_ids, n_gen):
    tokens_arg = ",".join(str(t) for t in prompt_ids)
    r = subprocess.run([binary, "generate", "-m", model, "--tokens", tokens_arg, "-n", str(n_gen)],
                        capture_output=True, text=True)
    if r.returncode != 0:
        print("generate failed (exit %d):\n%s" % (r.returncode, r.stderr), file=sys.stderr)
        return None
    return parse_tokens_line(r.stdout) or []


def run_logits(binary, model, full_ids, batch=1):
    """Returns the path to a temp file with Trochilus's logits for full_ids, or
    None on failure. Caller removes the file. batch > 1: the tokens run `batch` per
    forward pass, one row per pass (its last token)."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bin")
    os.close(tmp_fd)
    tokens_arg = ",".join(str(t) for t in full_ids)
    r = subprocess.run([binary, "logits", "-m", model, "--tokens", tokens_arg, "--out", tmp_path, "-b", str(batch)],
                        capture_output=True, text=True)
    if r.returncode != 0:
        print("logits failed (exit %d):\n%s" % (r.returncode, r.stderr), file=sys.stderr)
        os.remove(tmp_path)
        return None
    return tmp_path


# Tokens per forward pass checked against one token per pass, plus the whole sequence in one pass.
BATCH_SIZES = (3, 64)


def check_batches(binary, model, full_ids, per_token_path, vocab, label=""):
    """Prefill in blocks (docs/STATO.md): `logits -b k` must write, for the last token of
    each pass, the very bytes that one token per pass gives at that position. Exact by
    design, so any difference fails whatever --expect says. Returns True if all match."""
    with open(per_token_path, "rb") as f:
        per_token = f.read()
    row, n, ok = 4 * vocab, len(full_ids), True
    for k in sorted({k for k in BATCH_SIZES if 1 < k < n} | {n}):
        path = run_logits(binary, model, full_ids, batch=k)
        if path is None:
            ok = False
            continue
        try:
            with open(path, "rb") as f:
                got = f.read()
        finally:
            os.remove(path)
        ends = [min(i + k, n) - 1 for i in range(0, n, k)]
        same = got == b"".join(per_token[e * row:(e + 1) * row] for e in ends)
        print("%s%d tokens per pass: %d rows %s" % (label, k, len(ends),
              "bit-identical to one token per pass" if same else "DIFFERENT from one token per pass"))
        ok = ok and same
    return ok


def run_single(args, binary, ref):
    """tools/make_tiny_olmoe.py fixture: one prompt, logits inline as nested JSON."""
    prompt_ids = ref["prompt_ids"]
    full_ids = ref["full_ids"]
    ref_logits = ref["logits"]
    vocab = len(ref_logits[0])
    n_gen = len(full_ids) - len(prompt_ids)
    expected_gen = full_ids[len(prompt_ids):]

    ok, batches_ok = True, True
    generated = run_generate(binary, args.model, prompt_ids, n_gen)
    if generated is None:
        ok, generated = False, []
    n_match = sum(1 for a, b in zip(generated, expected_gen) if a == b)
    tokens_exact = generated == expected_gen
    print("token match: %d/%d (exact: %s)" % (n_match, n_gen, tokens_exact))
    if not tokens_exact:
        print("  expected: %s" % expected_gen)
        print("  got:      %s" % generated)
        ok = False

    max_diff = float("inf")
    tmp_path = run_logits(binary, args.model, full_ids)
    if tmp_path is None:
        ok = False
    else:
        try:
            got_logits = read_logits(tmp_path, len(full_ids), vocab)
            max_diff = 0.0
            argmax_agree = 0
            per_pos = []
            for pos in range(len(full_ids)):
                row_diff = max(abs(a - b) for a, b in zip(got_logits[pos], ref_logits[pos]))
                per_pos.append(row_diff)
                max_diff = max(max_diff, row_diff)
                got_arg = max(range(vocab), key=lambda i: got_logits[pos][i])
                ref_arg = max(range(vocab), key=lambda i: ref_logits[pos][i])
                if got_arg == ref_arg:
                    argmax_agree += 1
            print("max abs logit diff: %.6g (overall)" % max_diff)
            for pos, d in enumerate(per_pos):
                print("  pos %3d: max abs diff %.6g" % (pos, d))
            print("argmax agreement: %d/%d" % (argmax_agree, len(full_ids)))
            if max_diff > args.logit_tol:
                ok = False
            batches_ok = check_batches(binary, args.model, full_ids, tmp_path, vocab)
        finally:
            os.remove(tmp_path)

    return ok and tokens_exact and max_diff <= args.logit_tol, batches_ok


def run_multi(args, binary, ref):
    """`make oracle-real` fixture (tools/make_olmoe_2layer_ref.py): several
    prompts, logits as a separate float32 binary per prompt (numpy comparison --
    the real vocab is tens of thousands wide, prompts up to ~1000+ tokens)."""
    vocab = ref["vocab_size"]
    ok, batches_ok = True, True
    for p in ref["prompts"]:
        name = p["name"]
        prompt_ids, full_ids = p["prompt_ids"], p["full_ids"]
        n_gen = len(full_ids) - len(prompt_ids)
        expected_gen = full_ids[len(prompt_ids):]

        generated = run_generate(binary, args.model, prompt_ids, n_gen)
        tokens_exact = generated == expected_gen
        print("[%s] tokens: %d prompt + %d generated, greedy match: %s" %
              (name, len(prompt_ids), n_gen, tokens_exact))
        if not tokens_exact:
            print("  expected: %s" % expected_gen)
            print("  got:      %s" % generated)
            ok = False

        tmp_path = run_logits(binary, args.model, full_ids)
        if tmp_path is None:
            ok = False
            continue
        try:
            got = read_logits_np(tmp_path, len(full_ids), vocab)
            batches_ok = check_batches(binary, args.model, full_ids, tmp_path, vocab, "[%s] " % name) and batches_ok
        finally:
            os.remove(tmp_path)
        ref_logits = read_logits_np(os.path.join(args.fixture_dir, p["logits_file"]), len(full_ids), vocab)

        diff = np.abs(got - ref_logits)
        row_max = diff.max(axis=1)
        worst_pos = int(row_max.argmax())
        max_diff = float(row_max[worst_pos])
        argmax_agree = int((got.argmax(axis=1) == ref_logits.argmax(axis=1)).sum())
        print("[%s] max abs logit diff: %.6g at pos %d (of %d)" % (name, max_diff, worst_pos, len(full_ids)))
        print("[%s] argmax agreement: %d/%d" % (name, argmax_agree, len(full_ids)))
        if max_diff > args.logit_tol:
            ok = False

    return ok, batches_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture_dir")
    ap.add_argument("model")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--expect", choices=["exact", "report"], default="exact")
    ap.add_argument("--logit-tol", type=float, default=1e-3,
                     help="max abs logit diff allowed under --expect exact (default 1e-3)")
    args = ap.parse_args()

    # os.path.abspath: on Windows, CreateProcess does not reliably resolve a
    # relative forward-slash path (e.g. "build/trochilus.exe") even from the
    # right cwd; an absolute path always works.
    binary = os.path.abspath(args.binary or default_binary())

    with open(os.path.join(args.fixture_dir, "ref.json"), encoding="utf-8") as f:
        ref = json.load(f)

    t0 = time.perf_counter()
    ok, batches_ok = run_multi(args, binary, ref) if "prompts" in ref else run_single(args, binary, ref)
    self_rss, child_rss = peak_rss_mb()
    msg = ("peak RSS: oracle.py %.0f MB, %s %.0f MB" % (self_rss, os.path.basename(binary), child_rss)
           if self_rss is not None else "peak RSS: n/a on this platform")
    print("comparison: %.1fs, %s" % (time.perf_counter() - t0, msg))

    if not batches_ok:
        return 1
    if args.expect == "report":
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
