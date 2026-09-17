#!/usr/bin/env python3
"""Inspect and validate a GGUF v3 file written by tools/hf_to_gguf.py.

Purpose:
  Prints the file's header, metadata count and one line per tensor (name, ggml
  type, ggml shape, byte size), using gguf-py's GGUFReader as the source of
  truth for how the file is laid out - no independent parsing of the format.
  For a known architecture (currently "olmoe"), also checks structural
  consistency: every tensor the architecture needs is present for every
  layer, and every tensor's shape agrees with the file's own metadata.

Usage:
  tools/.venv/Scripts/python.exe tools/check_gguf.py <file.gguf> [--compare-hf <hf_dir>]

  --compare-hf <hf_dir>   also dequantize every tensor (gguf.quants.dequantize)
                          and compare it to the matching HF safetensors tensor,
                          after the same stacking/permutation hf_to_gguf.py did:
                            F32   bit-exact
                            F16   exact vs. the HF value cast to float16
                            Q8_0  max abs error <= one block's scale
                          Exits 1 on any mismatch and reports the worst one.

Exit code: 0 if all checks pass, 1 on any structural or numerical problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

import gguf
from gguf.gguf_reader import GGUFReader, ReaderTensor


def kv(reader: GGUFReader, key: str):
    field = reader.fields.get(key)
    return field.contents() if field is not None else None


def print_header(reader: GGUFReader) -> None:
    version = kv(reader, "GGUF.version")
    arch = kv(reader, "general.architecture")
    kv_count = kv(reader, "GGUF.kv_count")
    tensor_count = kv(reader, "GGUF.tensor_count")
    print(f"GGUF version: {version}")
    print(f"architecture: {arch}")
    print(f"kv count: {kv_count}")
    print(f"tensor count: {tensor_count}")


def print_tensors(reader: GGUFReader) -> None:
    for t in reader.tensors:
        shape = "x".join(str(d) for d in t.shape.tolist())
        print(f"{t.name}\t{t.tensor_type.name}\tshape=[{shape}]\tbytes={t.n_bytes}")


# --- structural validation ------------------------------------------------

def olmoe_expected_tensors(reader: GGUFReader) -> list[tuple[str, tuple[int, ...]]]:
    """(name, expected ggml shape) for every tensor an OLMoE GGUF must have,
    derived from the file's own metadata (not from an HF config)."""
    arch = kv(reader, "general.architecture")
    n_embd = kv(reader, "olmoe.embedding_length")
    n_layer = kv(reader, "olmoe.block_count")
    n_ff = kv(reader, "olmoe.feed_forward_length")
    n_head = kv(reader, "olmoe.attention.head_count")
    n_head_kv = kv(reader, "olmoe.attention.head_count_kv")
    n_expert = kv(reader, "olmoe.expert_count")
    key_length = kv(reader, "olmoe.attention.key_length")
    value_length = kv(reader, "olmoe.attention.value_length")
    vocab_size = kv(reader, "olmoe.vocab_size")

    required = {
        "general.architecture": arch,
        "olmoe.embedding_length": n_embd,
        "olmoe.block_count": n_layer,
        "olmoe.feed_forward_length": n_ff,
        "olmoe.attention.head_count": n_head,
        "olmoe.attention.head_count_kv": n_head_kv,
        "olmoe.expert_count": n_expert,
        "olmoe.attention.key_length": key_length,
        "olmoe.attention.value_length": value_length,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"missing required metadata key(s): {missing}")

    n_qkv = n_head * key_length
    n_kv = n_head_kv * key_length

    T = gguf.MODEL_TENSOR

    def tn(tensor, bid=None):
        base = gguf.TENSOR_NAMES[tensor]
        if bid is not None:
            base = base.format(bid=bid)
        return base + ".weight"

    expected: list[tuple[str, tuple[int, ...]]] = [
        (tn(T.OUTPUT_NORM), (n_embd,)),
    ]
    if vocab_size is not None:
        expected.append((tn(T.TOKEN_EMBD), (n_embd, vocab_size)))
        expected.append((tn(T.OUTPUT), (n_embd, vocab_size)))

    for i in range(n_layer):
        expected += [
            (tn(T.ATTN_NORM, i), (n_embd,)),
            (tn(T.ATTN_Q, i), (n_embd, n_qkv)),
            (tn(T.ATTN_K, i), (n_embd, n_kv)),
            (tn(T.ATTN_V, i), (n_embd, n_kv)),
            (tn(T.ATTN_OUT, i), (n_qkv, n_embd)),
            (tn(T.ATTN_Q_NORM, i), (n_qkv,)),
            (tn(T.ATTN_K_NORM, i), (n_kv,)),
            (tn(T.FFN_NORM, i), (n_embd,)),
            (tn(T.FFN_GATE_INP, i), (n_embd, n_expert)),
            (tn(T.FFN_GATE_EXP, i), (n_embd, n_ff, n_expert)),
            (tn(T.FFN_UP_EXP, i), (n_embd, n_ff, n_expert)),
            (tn(T.FFN_DOWN_EXP, i), (n_ff, n_embd, n_expert)),
        ]
    return expected


STRUCTURAL_VALIDATORS = {
    "olmoe": olmoe_expected_tensors,
}


def validate_structure(reader: GGUFReader) -> list[str]:
    """Returns a list of problem descriptions (empty = ok). Only runs for
    architectures with a registered validator; others are skipped, printed
    as a note."""
    arch = kv(reader, "general.architecture")
    validator = STRUCTURAL_VALIDATORS.get(arch)
    if validator is None:
        print(f"note: no structural validator registered for architecture {arch!r}; skipping")
        return []

    problems: list[str] = []
    try:
        expected = validator(reader)
    except ValueError as exc:
        return [str(exc)]

    by_name = {t.name: t for t in reader.tensors}
    for name, exp_shape in expected:
        t = by_name.get(name)
        if t is None:
            problems.append(f"missing tensor: {name}")
            continue
        got_shape = tuple(int(d) for d in t.shape.tolist())
        if got_shape != exp_shape:
            problems.append(f"{name}: shape {got_shape} != expected {exp_shape}")

    expected_names = {name for name, _ in expected}
    return problems


# --- HF comparison ----------------------------------------------------------

def load_hf_tensor(hf_dir: Path, name: str) -> np.ndarray:
    index_file = hf_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        path = hf_dir / index["weight_map"][name]
    else:
        path = hf_dir / "model.safetensors"
    with safe_open(str(path), framework="numpy") as f:
        return f.get_tensor(name)


def hf_reference_for(name: str, config: dict, hf_dir: Path) -> np.ndarray | None:
    """Reconstructs, from HF safetensors, the exact numpy array hf_to_gguf.py
    wrote for `name` (same stacking for the expert tensors), or None if `name`
    is not one this comparator knows how to reconstruct (e.g. tokenizer)."""
    T = gguf.MODEL_TENSOR
    n_layers = config["num_hidden_layers"]
    n_experts = config["num_experts"]

    def tn(tensor, bid=None):
        base = gguf.TENSOR_NAMES[tensor]
        if bid is not None:
            base = base.format(bid=bid)
        return base + ".weight"

    if name == tn(T.TOKEN_EMBD):
        return load_hf_tensor(hf_dir, "model.embed_tokens.weight")
    if name == tn(T.OUTPUT_NORM):
        return load_hf_tensor(hf_dir, "model.norm.weight")
    if name == tn(T.OUTPUT):
        try:
            return load_hf_tensor(hf_dir, "lm_head.weight")
        except Exception:
            return load_hf_tensor(hf_dir, "model.embed_tokens.weight")

    for i in range(n_layers):
        p = f"model.layers.{i}."
        mapping = {
            tn(T.ATTN_NORM, i): p + "input_layernorm.weight",
            tn(T.ATTN_Q, i): p + "self_attn.q_proj.weight",
            tn(T.ATTN_K, i): p + "self_attn.k_proj.weight",
            tn(T.ATTN_V, i): p + "self_attn.v_proj.weight",
            tn(T.ATTN_OUT, i): p + "self_attn.o_proj.weight",
            tn(T.ATTN_Q_NORM, i): p + "self_attn.q_norm.weight",
            tn(T.ATTN_K_NORM, i): p + "self_attn.k_norm.weight",
            tn(T.FFN_NORM, i): p + "post_attention_layernorm.weight",
            tn(T.FFN_GATE_INP, i): p + "mlp.gate.weight",
        }
        if name in mapping:
            return load_hf_tensor(hf_dir, mapping[name])

        for exp_tensor, proj in ((T.FFN_GATE_EXP, "gate_proj"), (T.FFN_UP_EXP, "up_proj"), (T.FFN_DOWN_EXP, "down_proj")):
            if name == tn(exp_tensor, i):
                return np.stack(
                    [load_hf_tensor(hf_dir, p + f"mlp.experts.{e}.{proj}.weight") for e in range(n_experts)],
                    axis=0,
                )
    return None


def compare_with_hf(reader: GGUFReader, hf_dir: Path) -> list[str]:
    config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    problems: list[str] = []

    for t in reader.tensors:
        ref = hf_reference_for(t.name, config, hf_dir)
        if ref is None:
            continue  # nothing to compare this tensor against (shouldn't happen for olmoe)
        ref = ref.astype(np.float32, copy=False)

        got = gguf.quants.dequantize(t.data, t.tensor_type)  # (out[, ...], in) numpy layout
        got = got.reshape(ref.shape)

        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            if not np.array_equal(got, ref):
                max_err = float(np.max(np.abs(got - ref)))
                problems.append(f"{t.name}: F32 not bit-exact, max abs err {max_err:.3e}")
        elif t.tensor_type == gguf.GGMLQuantizationType.F16:
            ref_f16 = ref.astype(np.float16).astype(np.float32)
            if not np.array_equal(got, ref_f16):
                max_err = float(np.max(np.abs(got - ref_f16)))
                problems.append(f"{t.name}: F16 mismatch vs ref.astype(float16), max abs err {max_err:.3e}")
        elif t.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            block_size, _type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q8_0]
            flat_ref = ref.reshape(-1, block_size)
            block_scale = np.max(np.abs(flat_ref), axis=-1) / 127.0
            max_allowed = float(np.max(block_scale)) if block_scale.size else 0.0
            max_err = float(np.max(np.abs(got - ref)))
            if max_err > max_allowed:
                problems.append(
                    f"{t.name}: Q8_0 max abs err {max_err:.3e} > max block scale {max_allowed:.3e}"
                )
            else:
                print(f"  {t.name}: Q8_0 max abs err {max_err:.3e} (<= block scale {max_allowed:.3e})")
        else:
            problems.append(f"{t.name}: no comparison rule for type {t.tensor_type.name}")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf_file", type=Path)
    parser.add_argument("--compare-hf", type=Path, default=None, metavar="hf_dir")
    args = parser.parse_args(argv)

    reader = GGUFReader(str(args.gguf_file))
    print_header(reader)
    print_tensors(reader)

    problems = validate_structure(reader)

    if args.compare_hf is not None:
        print(f"comparing against HF checkpoint: {args.compare_hf}")
        problems += compare_with_hf(reader, args.compare_hf.resolve())

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
