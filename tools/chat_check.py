#!/usr/bin/env python3
"""`trochilus chat` on a real model: two turns, then the same conversation from scratch.

The chat reuses the session cache between turns (it re-renders the whole conversation,
keeps the tokens that did not change and rewinds the rest). The second reply must be
identical to `trochilus run` on the conversation rendered by transformers'
apply_chat_template: cache reuse and rewinding may not change a single token.

  tools/chat_check.py --binary build/trochilus --model <file.gguf> --hf <tokenizer dir>

Prints SKIPPED and exits 0 when the model file is not there.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TURNS = ["Ciao! Come ti chiami? Rispondi in una frase.",
         "Scrivi una funzione Python che somma due numeri."]
MAX_TOKENS = "60"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--hf", required=True, type=Path)
    args = ap.parse_args()
    if not args.model.exists():
        print(f"chat-check: SKIPPED, {args.model} not found")
        return 0
    binary = os.path.abspath(args.binary)
    common = ["-m", str(args.model), "-n", MAX_TOKENS, "-c", "1024"]

    r = subprocess.run([binary, "chat", *common], input=("\n".join(TURNS) + "\n").encode("utf-8"),
                       capture_output=True)
    if r.returncode != 0:
        print(f"chat-check: chat failed (exit {r.returncode}): {r.stderr.decode('utf-8', 'replace')[-2000:]}")
        return 1
    # stdout: "\n> " before each input, then the reply and a newline
    replies = [p[:-1] if p.endswith("\n") else p for p in r.stdout.decode("utf-8").split("\n> ")[1:]]
    if len(replies) < 2 or not replies[0] or not replies[1]:
        print(f"chat-check: expected two replies, got {replies!r}")
        return 1

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.hf))
    msgs = [{"role": "user", "content": TURNS[0]}, {"role": "assistant", "content": replies[0]},
            {"role": "user", "content": TURNS[1]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    with tempfile.TemporaryDirectory(prefix="chat_check_") as tmp:
        path = Path(tmp) / "prompt.txt"
        path.write_bytes(prompt.encode("utf-8"))
        r2 = subprocess.run([binary, "run", "-f", str(path), *common], capture_output=True)
    if r2.returncode != 0:
        print(f"chat-check: run failed (exit {r2.returncode}): {r2.stderr.decode('utf-8', 'replace')[-2000:]}")
        return 1
    fresh = r2.stdout.decode("utf-8")
    fresh = fresh[:-1] if fresh.endswith("\n") else fresh
    if fresh != replies[1]:
        print("chat-check: the second reply differs from a fresh run on the same conversation")
        print(f"  chat: {replies[1]!r}")
        print(f"  run:  {fresh!r}")
        return 1
    print(f"chat-check: ok, second reply identical to a fresh run ({len(replies[1])} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
