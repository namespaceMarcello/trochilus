#!/usr/bin/env python3
"""fetch_hf_tokenizer.py — download the Hugging Face files needed to convert and
check a model's tokenizer, pinned to one revision.

Usage:
  tools/.venv/Scripts/python.exe tools/fetch_hf_tokenizer.py \
      --repo allenai/OLMoE-1B-7B-0125-Instruct \
      --revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e \
      --output fixtures/olmoe-1b-7b-0125-instruct-tokenizer

Downloads tokenizer.json, tokenizer_config.json and config.json (required: the
conversion and the oracle need all three - config.json in particular carries
vocab_size, which the tokenizer files alone do not always determine, see
hf_to_gguf.py write_tokenizer), plus special_tokens_map.json and
generation_config.json when the repo has them (optional).

Idempotent: a file already present under --output is left alone and no request
is made for it, so re-running after a partial download only fetches what is
missing. Every file is pinned to --revision, never "main", so the fixture
cannot silently drift under a later commit to the HF repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

REQUIRED_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json"]
OPTIONAL_FILES = ["special_tokens_map.json", "generation_config.json"]


def fetch_one(repo: str, revision: str, filename: str, output: Path) -> str:
    """Returns "skipped" (already there), "fetched", "absent" (optional file not
    in the repo) or raises on any other failure."""
    target = output / filename
    if target.exists():
        return "skipped"
    try:
        hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(output))
    except EntryNotFoundError:
        return "absent"
    return "fetched"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. allenai/OLMoE-1B-7B-0125-Instruct")
    parser.add_argument("--revision", required=True, help="commit SHA to pin to (never a branch name)")
    parser.add_argument("--output", required=True, type=Path, help="directory to fill in")
    args = parser.parse_args(argv)

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    ok = True
    for filename in REQUIRED_FILES:
        try:
            status = fetch_one(args.repo, args.revision, filename, output)
        except Exception as exc:  # noqa: BLE001 - report and keep going, then fail at the end
            print(f"error: {filename}: {exc}", file=sys.stderr)
            ok = False
            continue
        if status == "absent":
            print(f"error: {filename} not found in {args.repo}@{args.revision} (required)", file=sys.stderr)
            ok = False
        else:
            print(f"{filename}: {status}")

    for filename in OPTIONAL_FILES:
        try:
            status = fetch_one(args.repo, args.revision, filename, output)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {filename}: {exc}", file=sys.stderr)
            ok = False
            continue
        print(f"{filename}: {status}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
