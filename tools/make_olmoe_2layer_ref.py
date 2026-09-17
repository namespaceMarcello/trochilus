#!/usr/bin/env python3
"""Reference for oracle-real: OLMoE's real weights, cut to 2 layers and dequantized
from the GGUF written by tools/make_olmoe_2layer_gguf.py, run through transformers.

For each prompt: greedy-generate 32 tokens, then one teacher-forced forward pass over
prompt+generated tokens (every position's logits, not just the generated ones).
Config comes straight from the 2-layer GGUF's metadata -- every field used is
asserted, not assumed -- and weights are copied tensor by tensor into a fresh
OlmoeForCausalLM(num_hidden_layers=2). The three big per-layer expert tensors are
loaded a few experts at a time (see put_experts/EXPERT_CHUNK below) so the peak
stays close to the model's own resident float32 parameters (~4.2 GB for 2 layers)
plus reading the 2-layer GGUF once, not several hundred more MB per big tensor.

Tensor-name mapping mirrors tools/hf_to_gguf.py's OlmoeConverter in reverse. The
fused `experts.gate_up_proj` / `experts.down_proj` parameters this pinned
transformers version (5.14.1) actually uses are built here from the GGUF's separate
ffn_gate_exps / ffn_up_exps / ffn_down_exps tensors (gate rows first, matching
OlmoeExperts.forward's `.chunk(2, dim=-1)` on the linear output).

Output directory:
  ref.json              vocab_size, per-prompt name/prompt_ids/full_ids/logits_file
  <name>.logits.bin      float32 little-endian, row-major [len(full_ids), vocab] --
                         the same binary layout `trochilus logits --out` writes.

Usage:
  tools/.venv/Scripts/python.exe tools/make_olmoe_2layer_ref.py \
      --gguf fixtures/olmoe-2layer/model.gguf --hf fixtures/olmoe-1b-7b-0125-instruct-tokenizer \
      --output fixtures/olmoe-2layer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

import gguf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from hf_to_gguf import tensor_name  # noqa: E402 -- same llama.cpp tensor-name convention

NEW_TOKENS = 32
LONG_PROMPT_TOKENS = 1024  # far enough to exercise RoPE positions well past a short prompt


def kv(reader, arch, group, name):
    key = getattr(getattr(gguf.Keys, group), name).format(arch=arch)
    field = reader.get_field(key)
    return field.contents() if field is not None else None


def dequant(reader_tensor):
    return gguf.quants.dequantize(reader_tensor.data, reader_tensor.tensor_type).astype(np.float32, copy=False)


EXPERT_CHUNK = 16  # of 64: gguf-py's own dequantize buffers ~2x the chunk internally


def dequant_expert_chunks(reader_tensor, chunk=EXPERT_CHUNK):
    """Dequantizes reader_tensor (leading dim = expert) a few experts at a time.
    Q8_0 blocks only span the last axis, so slicing the leading (expert) axis
    before dequantizing is exact, not an approximation -- it just keeps gguf-py's
    own dequantize (which internally buffers about 2x what it is handed) from
    ever holding a whole 512 MB expert projection at once."""
    data = reader_tensor.data
    for start in range(0, data.shape[0], chunk):
        end = min(start + chunk, data.shape[0])
        yield start, end, gguf.quants.dequantize(data[start:end], reader_tensor.tensor_type).astype(
            np.float32, copy=False)


def logical_shape(reader_tensor):
    """Element shape in numpy/HF convention ([..., out, in]), valid for any storage
    type. reader_tensor.data.shape is only that for F32/F16; for a quantized type it
    is the packed *byte* shape (see gguf_reader.py), so this reverses .shape (ne,
    ggml declaration order) instead of dequantizing just to check a shape."""
    return tuple(int(x) for x in reversed(reader_tensor.shape.tolist()))


def build_config(reader):
    """Read every OlmoeConfig field this script needs from GGUF metadata, asserting
    each one (no silent defaults for values the reference must get right)."""
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    assert arch == "olmoe", f"unsupported architecture {arch!r}"

    byname = {t.name: t for t in reader.tensors}
    n_layers = len({int(n.split(".")[1]) for n in byname if n.startswith("blk.")})
    assert n_layers >= 1, "no blk.N tensors in this GGUF"

    embed = byname[tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD)]
    vocab_size, hidden_size = logical_shape(embed)  # numpy [vocab, hidden], see hf_to_gguf.py

    intermediate_size = kv(reader, arch, "LLM", "FEED_FORWARD_LENGTH")
    context_length = kv(reader, arch, "LLM", "CONTEXT_LENGTH")
    n_experts = kv(reader, arch, "LLM", "EXPERT_COUNT")
    n_experts_used = kv(reader, arch, "LLM", "EXPERT_USED_COUNT")
    norm_topk_prob = bool(kv(reader, arch, "LLM", "EXPERT_WEIGHTS_NORM") or False)
    n_head = kv(reader, arch, "Attention", "HEAD_COUNT")
    n_head_kv = kv(reader, arch, "Attention", "HEAD_COUNT_KV")
    rms_eps = kv(reader, arch, "Attention", "LAYERNORM_RMS_EPS")
    clip_qkv = kv(reader, arch, "Attention", "CLAMP_KQV")
    key_length = kv(reader, arch, "Attention", "KEY_LENGTH")
    value_length = kv(reader, arch, "Attention", "VALUE_LENGTH")
    rope_theta = kv(reader, arch, "Rope", "FREQ_BASE")
    rope_scaling_type = kv(reader, arch, "Rope", "SCALING_TYPE")

    required = {"feed_forward_length": intermediate_size, "context_length": context_length,
                "expert_count": n_experts, "expert_used_count": n_experts_used,
                "head_count": n_head, "head_count_kv": n_head_kv,
                "layer_norm_rms_epsilon": rms_eps, "rope.freq_base": rope_theta}
    for name, val in required.items():
        assert val is not None, f"required GGUF key missing: {arch}.{name}"
    assert rope_scaling_type is None, f"unsupported rope scaling {rope_scaling_type!r}: extend this script"
    assert hidden_size % n_head == 0, f"hidden_size {hidden_size} not divisible by head_count {n_head}"
    head_dim = hidden_size // n_head
    if key_length is not None:
        assert key_length == head_dim, f"key_length {key_length} != hidden_size/head_count {head_dim}"
    if value_length is not None:
        assert value_length == head_dim, f"value_length {value_length} != hidden_size/head_count {head_dim}"

    has_bias = any(n.endswith(".bias") for n in byname)
    assert not has_bias, "bias tensors present: this script does not support attention_bias=true"

    q = byname[tensor_name(gguf.MODEL_TENSOR.ATTN_Q, 0)]
    q_shape = logical_shape(q)
    assert q_shape == (n_head * head_dim, hidden_size), (
        f"attn_q shape {q_shape} disagrees with head_count*head_dim={n_head * head_dim}, hidden={hidden_size}")

    config = OlmoeConfig(
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_hidden_layers=n_layers,
        num_attention_heads=int(n_head),
        num_key_value_heads=int(n_head_kv),
        hidden_act="silu",
        max_position_embeddings=int(context_length),
        rms_norm_eps=float(rms_eps),
        rope_parameters={"rope_theta": float(rope_theta)},
        attention_bias=False,
        clip_qkv=float(clip_qkv) if clip_qkv is not None else None,
        num_experts_per_tok=int(n_experts_used),
        num_experts=int(n_experts),
        output_router_logits=False,
        norm_topk_prob=norm_topk_prob,
        tie_word_embeddings=(tensor_name(gguf.MODEL_TENSOR.OUTPUT) not in byname),
        eos_token_id=None, bos_token_id=None, pad_token_id=None,
    )
    return config, n_layers, byname


def to_writable(arr):
    # F32 tensors dequantize to a .view() of the read-only GGUF memmap; torch.from_numpy
    # needs writable storage even though we only ever read it here.
    return arr if arr.flags.writeable else arr.copy()


def load_weights(model, byname, n_layers):
    """Copies one dequantized tensor at a time into the model's own (already
    allocated) parameter storage, instead of building a full state_dict first.
    The three big per-layer expert tensors (gate/up/down, ~512 MB each in f32) are
    loaded EXPERT_CHUNK experts at a time (put_experts) so at most a small slice of
    one of them, doubled by gguf-py's own dequantize buffering, is ever alive on
    top of the model's resident float32 parameters (peak RSS)."""
    T = gguf.MODEL_TENSOR
    params = dict(model.named_parameters())

    def put(name, arr):
        p = params[name]
        arr = to_writable(np.ascontiguousarray(arr))
        assert tuple(p.shape) == arr.shape, f"{name}: model {tuple(p.shape)} vs gguf {arr.shape}"
        with torch.no_grad():
            p.copy_(torch.from_numpy(arr))

    def put_experts(name, reader_tensor, dim1_offset=0):
        """Copies reader_tensor into p[e_start:e_end, dim1_offset:dim1_offset+width, :]
        a few experts at a time (dequant_expert_chunks): down_proj gets dim1_offset=0
        over its full width; gate_up_proj gets one call for the gate half
        (dim1_offset=0) and one for the up half (dim1_offset=intermediate_size)."""
        p = params[name]
        for e_start, e_end, arr in dequant_expert_chunks(reader_tensor):
            arr = to_writable(np.ascontiguousarray(arr))
            width = arr.shape[1]
            dst = p[e_start:e_end, dim1_offset:dim1_offset + width, :]
            assert tuple(dst.shape) == arr.shape, f"{name}: model {tuple(dst.shape)} vs gguf {arr.shape}"
            with torch.no_grad():
                dst.copy_(torch.from_numpy(arr))

    put("model.embed_tokens.weight", dequant(byname[tensor_name(T.TOKEN_EMBD)]))
    put("model.norm.weight", dequant(byname[tensor_name(T.OUTPUT_NORM)]))
    out_name = tensor_name(T.OUTPUT)
    if out_name in byname:
        put("lm_head.weight", dequant(byname[out_name]))

    for i in range(n_layers):
        p = f"model.layers.{i}."
        put(p + "input_layernorm.weight", dequant(byname[tensor_name(T.ATTN_NORM, i)]))
        put(p + "self_attn.q_proj.weight", dequant(byname[tensor_name(T.ATTN_Q, i)]))
        put(p + "self_attn.k_proj.weight", dequant(byname[tensor_name(T.ATTN_K, i)]))
        put(p + "self_attn.v_proj.weight", dequant(byname[tensor_name(T.ATTN_V, i)]))
        put(p + "self_attn.o_proj.weight", dequant(byname[tensor_name(T.ATTN_OUT, i)]))
        put(p + "self_attn.q_norm.weight", dequant(byname[tensor_name(T.ATTN_Q_NORM, i)]))
        put(p + "self_attn.k_norm.weight", dequant(byname[tensor_name(T.ATTN_K_NORM, i)]))
        put(p + "post_attention_layernorm.weight", dequant(byname[tensor_name(T.FFN_NORM, i)]))
        put(p + "mlp.gate.weight", dequant(byname[tensor_name(T.FFN_GATE_INP, i)]))
        intermediate_size = int(byname[tensor_name(T.FFN_GATE_EXP, i)].shape[1])  # ne1 = intermediate (ggml order)
        gu_name = p + "mlp.experts.gate_up_proj"
        put_experts(gu_name, byname[tensor_name(T.FFN_GATE_EXP, i)], dim1_offset=0)
        put_experts(gu_name, byname[tensor_name(T.FFN_UP_EXP, i)], dim1_offset=intermediate_size)
        put_experts(p + "mlp.experts.down_proj", byname[tensor_name(T.FFN_DOWN_EXP, i)])


def load_prompt_ids(tokenizer, path, max_tokens=None):
    text = path.read_text(encoding="utf-8", errors="replace")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens] if max_tokens is not None else ids


def peak_rss_mb():
    try:
        import resource
    except ImportError:
        return None  # Windows: no resource module; the container run reports the real number
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def run_prompt(model, name, prompt_ids, output_dir):
    t0 = time.perf_counter()
    with torch.no_grad():
        full = model.generate(torch.tensor([prompt_ids]), max_new_tokens=NEW_TOKENS,
                               do_sample=False, use_cache=True)[0].tolist()
        logits = model(torch.tensor([full])).logits[0].float().numpy()
    dt = time.perf_counter() - t0
    logits_file = f"{name}.logits.bin"
    logits.astype("<f4").tofile(str(output_dir / logits_file))
    rss = peak_rss_mb()
    print(f"[{name}] {len(prompt_ids)} prompt + {len(full) - len(prompt_ids)} generated tokens, "
          f"{dt:.1f}s" + (f", peak RSS so far {rss:.0f} MB" if rss is not None else ""))
    return {"name": name, "prompt_ids": prompt_ids, "full_ids": full, "logits_file": logits_file}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--hf", required=True, type=Path, help="local HF tokenizer directory")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--long-source", type=Path, default=ROOT / "src" / "models" / "olmoe.c",
                     help="source file tokenized for the >=1024-token far-RoPE prompt")
    args = ap.parse_args()

    torch.manual_seed(20260917)
    torch.set_num_threads(1)  # deterministic reduction order, matches tools/make_tiny_olmoe.py

    t_start = time.perf_counter()
    reader = gguf.GGUFReader(str(args.gguf), "r")
    config, n_layers, byname = build_config(reader)
    model = OlmoeForCausalLM(config).eval().float()
    load_weights(model, byname, n_layers)
    rss = peak_rss_mb()
    print(f"model built and loaded: {n_layers} layers, vocab {config.vocab_size}, "
          f"{time.perf_counter() - t_start:.1f}s" + (f", peak RSS {rss:.0f} MB" if rss is not None else ""))

    tokenizer = AutoTokenizer.from_pretrained(str(args.hf))
    args.output.mkdir(parents=True, exist_ok=True)

    long_ids = load_prompt_ids(tokenizer, args.long_source, max_tokens=LONG_PROMPT_TOKENS)
    assert len(long_ids) >= LONG_PROMPT_TOKENS, (
        f"{args.long_source} only tokenizes to {len(long_ids)} tokens, need >= {LONG_PROMPT_TOKENS}")
    prompts = [
        ("dante", load_prompt_ids(tokenizer, ROOT / "bench" / "prompts" / "dante.txt")),
        ("long", long_ids),
    ]

    entries = [run_prompt(model, name, ids, args.output) for name, ids in prompts]

    ref = {"vocab_size": config.vocab_size, "n_layers": n_layers, "new_tokens": NEW_TOKENS, "prompts": entries}
    (args.output / "ref.json").write_text(json.dumps(ref) + "\n", encoding="utf-8")

    rss = peak_rss_mb()
    print(f"wrote {args.output / 'ref.json'}; total wall time {time.perf_counter() - t_start:.1f}s"
          + (f"; peak RSS {rss:.0f} MB" if rss is not None else "; peak RSS n/a on this platform"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
