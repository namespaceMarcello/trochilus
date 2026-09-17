#!/usr/bin/env python3
"""oracle.py — compare Trochilus's forward pass against the transformers
reference captured in <fixture_dir>/ref.json (see tools/make_tiny_olmoe.py).

Usage:
    oracle.py <fixture_dir> <model.gguf> [--binary <path>] [--expect exact|report]

Runs `<binary> generate` on ref["prompt_ids"] for
len(full_ids) - len(prompt_ids) greedy tokens, and `<binary> logits` on the
full teacher-forced sequence ref["full_ids"]. Prints token match, max abs
logit diff (overall and per position), and argmax agreement.

--expect exact (default): exit 1 unless the generated tokens match exactly
    and the overall max abs logit diff is <= 1e-3.
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


def default_binary():
    return "build/trochilus.exe" if platform.system() == "Windows" else "build/trochilus"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture_dir")
    ap.add_argument("model")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--expect", choices=["exact", "report"], default="exact")
    args = ap.parse_args()

    # os.path.abspath: on Windows, CreateProcess does not reliably resolve a
    # relative forward-slash path (e.g. "build/trochilus.exe") even from the
    # right cwd; an absolute path always works.
    binary = os.path.abspath(args.binary or default_binary())

    with open(os.path.join(args.fixture_dir, "ref.json"), encoding="utf-8") as f:
        ref = json.load(f)
    prompt_ids = ref["prompt_ids"]
    full_ids = ref["full_ids"]
    ref_logits = ref["logits"]
    vocab = len(ref_logits[0])
    n_gen = len(full_ids) - len(prompt_ids)
    expected_gen = full_ids[len(prompt_ids):]

    ok = True

    # ---- generate: greedy tokens must match the reference exactly ----
    tokens_arg = ",".join(str(t) for t in prompt_ids)
    r = subprocess.run([binary, "generate", "-m", args.model, "--tokens", tokens_arg, "-n", str(n_gen)],
                        capture_output=True, text=True)
    if r.returncode != 0:
        print("generate failed (exit %d):\n%s" % (r.returncode, r.stderr), file=sys.stderr)
        ok = False
        generated = []
    else:
        generated = parse_tokens_line(r.stdout) or []

    n_match = sum(1 for a, b in zip(generated, expected_gen) if a == b)
    tokens_exact = generated == expected_gen
    print("token match: %d/%d (exact: %s)" % (n_match, n_gen, tokens_exact))
    if not tokens_exact:
        print("  expected: %s" % expected_gen)
        print("  got:      %s" % generated)
        ok = False

    # ---- logits: teacher-forced full sequence ----
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bin")
    os.close(tmp_fd)
    max_diff = float("inf")
    try:
        tokens_arg = ",".join(str(t) for t in full_ids)
        r = subprocess.run([binary, "logits", "-m", args.model, "--tokens", tokens_arg, "--out", tmp_path],
                            capture_output=True, text=True)
        if r.returncode != 0:
            print("logits failed (exit %d):\n%s" % (r.returncode, r.stderr), file=sys.stderr)
            ok = False
        else:
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
            if max_diff > 1e-3:
                ok = False
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if args.expect == "report":
        return 0
    return 0 if (ok and tokens_exact and max_diff <= 1e-3) else 1


if __name__ == "__main__":
    sys.exit(main())
