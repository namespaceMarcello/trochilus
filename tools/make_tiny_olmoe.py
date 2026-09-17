#!/usr/bin/env python3
"""Build a deterministic tiny OLMoE checkpoint and its transformers reference.

Derived from colibri a90bed9 c/tools/make_olmoe_tiny.py (Apache-2.0), modified:
reference logits for every position are stored next to the greedy tokens, so a
mismatch can be located (which step, how far off) instead of only counted.

Output directory:
  config.json, model.safetensors   the HF source checkpoint (random weights)
  ref.json                         prompt_ids, full_ids (greedy, transformers)
                                   and logits[pos][vocab] of a teacher-forced
                                   forward over full_ids, as float32 values
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--new-tokens", type=int, default=16)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists (use --force): {output}")
        shutil.rmtree(output)
    torch.manual_seed(20260825)
    torch.set_num_threads(1)
    config = OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        eos_token_id=None,
        pad_token_id=None,
        bos_token_id=None,
        tie_word_embeddings=False,
    )
    model = OlmoeForCausalLM(config).eval().float()
    prompt = [3, 11, 29, 7, 41, 19]
    with torch.no_grad():
        full = model.generate(
            torch.tensor([prompt]), max_new_tokens=args.new_tokens,
            do_sample=False, use_cache=True,
        )[0].tolist()
        logits = model(torch.tensor([full])).logits[0].float()

    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    ref = {
        "prompt_ids": prompt,
        "full_ids": full,
        "logits": [[float(v) for v in row] for row in logits.tolist()],
    }
    (output / "ref.json").write_text(json.dumps(ref) + "\n", encoding="utf-8")
    print(f"wrote tiny OLMoE: {output} ({len(full) - len(prompt)} new tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
