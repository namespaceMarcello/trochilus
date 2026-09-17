#!/usr/bin/env python3
"""Convert a Hugging Face transformers checkpoint to a standard GGUF v3 file.

Purpose:
  Trochilus reads plain GGUF v3 with llama.cpp tensor naming (see docs/ARCHITETTURA.md,
  principle 5): this tool produces that file from an HF `transformers` checkpoint
  directory, so the C engine never has to understand HF's on-disk layout.

  Tensor naming, key-value metadata keys and the standard quantization formats all
  follow llama.cpp via gguf-py (the pip package built from llama.cpp's gguf-py, MIT):
  gguf.Keys, gguf.MODEL_TENSOR / gguf.TENSOR_NAMES, gguf.GGUFWriter and gguf.quants
  are the source of truth, not hand-typed strings.

Usage:
  tools/.venv/Scripts/python.exe tools/hf_to_gguf.py <hf_dir> <out.gguf> --type {f32,f16,q8_0}
  tools/.venv/Scripts/python.exe tools/hf_to_gguf.py <hf_dir> <out.gguf> --vocab-only [--tokenizer-pre NAME]

  <hf_dir>       directory with config.json and, for a full conversion,
                 model.safetensors (or a sharded model.safetensors.index.json);
                 tokenizer.json / tokenizer_config.json are read whenever present.
  <out.gguf>     output path.
  --type         weight storage type for everything except 1-D norms and the MoE
                 router, which are always F32 (see select_tensor_dtype below).
                 Required unless --vocab-only.
  --vocab-only   write only general.architecture, general.name and tokenizer.*
                 metadata: no tensors, no other hparams, --type not needed.
  --tokenizer-pre  value for tokenizer.ggml.pre (llama.cpp fingerprints
                 tokenizer.json against a signature table to pick this; that
                 table isn't available here, so without this flag the writer
                 falls back to "default" - see detect_bpe_pretokenizer).

Adding an architecture: implement ArchConverter and register it with
@register_architecture("HFArchClassName") - see OlmoeConverter below.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from safetensors import safe_open

import gguf

FILE_TYPE_FOR_OUT_TYPE = {
    "f32": gguf.LlamaFileType.ALL_F32,
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
}

Q8_0_BLOCK_SIZE = 32


@dataclass
class TensorSpec:
    """One GGUF tensor to write: its final llama.cpp name, the source numpy array
    (in numpy/HF [..., out, in] convention), and whether it must stay F32
    (1-D norms, the MoE router) regardless of the requested --type."""

    name: str
    data: np.ndarray
    force_f32: bool


class SafetensorsSource:
    """Lazy reader over a (possibly sharded) safetensors checkpoint."""

    def __init__(self, hf_dir: Path):
        self.hf_dir = hf_dir
        self._handles: dict[Path, object] = {}
        index_file = hf_dir / "model.safetensors.index.json"
        if index_file.exists():
            index = json.loads(index_file.read_text(encoding="utf-8"))
            self._name_to_file = {
                name: hf_dir / rel for name, rel in index["weight_map"].items()
            }
        else:
            single = hf_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"neither model.safetensors nor model.safetensors.index.json in {hf_dir}"
                )
            self._name_to_file = None
            self._single_file = single

    def _handle(self, path: Path):
        h = self._handles.get(path)
        if h is None:
            h = safe_open(str(path), framework="numpy")
            self._handles[path] = h
        return h

    def _file_for(self, name: str) -> Path:
        return self._name_to_file[name] if self._name_to_file is not None else self._single_file

    def has(self, name: str) -> bool:
        if self._name_to_file is not None:
            return name in self._name_to_file
        return name in self._handle(self._single_file).keys()

    def get(self, name: str) -> np.ndarray:
        if not self.has(name):
            raise KeyError(f"tensor {name!r} not found in {self.hf_dir}")
        return self._handle(self._file_for(name)).get_tensor(name)


class ArchConverter:
    """Interface every supported HF architecture implements."""

    gguf_arch: gguf.MODEL_ARCH

    def write_metadata(self, writer: gguf.GGUFWriter, config: dict) -> None:
        raise NotImplementedError

    def iter_tensors(self, source: SafetensorsSource, config: dict) -> Iterator[TensorSpec]:
        raise NotImplementedError


ARCH_CONVERTERS: dict[str, type[ArchConverter]] = {}


def register_architecture(hf_arch: str):
    def deco(cls: type[ArchConverter]) -> type[ArchConverter]:
        ARCH_CONVERTERS[hf_arch] = cls
        return cls
    return deco


def tensor_name(tensor: "gguf.MODEL_TENSOR", bid: int | None = None) -> str:
    """GGUF tensor name for a MODEL_TENSOR, straight from gguf-py's TENSOR_NAMES map
    (the same table llama.cpp's own converter uses), with the ".weight" suffix that
    every llama.cpp GGUF tensor carries."""
    base = gguf.TENSOR_NAMES[tensor]
    if bid is not None:
        base = base.format(bid=bid)
    return base + ".weight"


@register_architecture("OlmoeForCausalLM")
class OlmoeConverter(ArchConverter):
    """OLMoE (transformers `OlmoeForCausalLM`): GQA attention with whole-projection
    Q/K RMSNorm (not per-head - see modeling_olmoe.py, OlmoeAttention.forward: the
    norm is applied to the full [*, hidden] / [*, n_kv_head*head_dim] projection
    output, before it is reshaped into heads), and a per-layer MoE FFN with one
    router and N per-expert gate/up/down experts, all with the same shapes."""

    gguf_arch = gguf.MODEL_ARCH.OLMOE

    def write_metadata(self, writer: gguf.GGUFWriter, config: dict) -> None:
        hidden_size = config["hidden_size"]
        n_head = config["num_attention_heads"]
        n_head_kv = config["num_key_value_heads"]

        writer.add_vocab_size(config["vocab_size"])
        writer.add_context_length(config["max_position_embeddings"])
        writer.add_embedding_length(hidden_size)
        writer.add_block_count(config["num_hidden_layers"])
        writer.add_feed_forward_length(config["intermediate_size"])
        writer.add_head_count(n_head)
        writer.add_head_count_kv(n_head_kv)
        writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
        writer.add_expert_count(config["num_experts"])
        writer.add_expert_used_count(config["num_experts_per_tok"])
        writer.add_expert_weights_norm(bool(config["norm_topk_prob"]))

        # head_dim has no dedicated OlmoeConfig field (see configuration_olmoe.py):
        # modeling_olmoe.py derives it as getattr(config, "head_dim", None) or
        # hidden_size // num_attention_heads. Always write it explicitly via the
        # standard key/value length keys so the C engine never has to reproduce
        # that fallback.
        head_dim = config.get("head_dim") or hidden_size // n_head
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)

        rope_params = config.get("rope_parameters") or {}
        rope_theta = rope_params.get("rope_theta", config.get("rope_theta", 10000.0))
        writer.add_rope_freq_base(float(rope_theta))
        rope_type = rope_params.get("rope_type", "default")
        if rope_type not in ("default", "none", None):
            if rope_type == "linear":
                factor = rope_params.get("factor")
                if factor is None:
                    raise ValueError("rope_type='linear' but rope_parameters has no 'factor'")
                writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                writer.add_rope_scaling_factor(float(factor))
            else:
                # yarn/dynamic/etc: representable in principle (gguf.RopeScalingType
                # has YARN), but the exact HF <-> ggml parameter mapping needs
                # checking against a real config before trusting it. Fail loudly
                # instead of silently writing a wrong or incomplete scaling config.
                raise NotImplementedError(
                    f"rope_type={rope_type!r} is not supported by this converter yet"
                )

        clip_qkv = config.get("clip_qkv")
        if clip_qkv is not None:
            writer.add_clamp_kqv(float(clip_qkv))

        if config.get("attention_bias", False):
            # bias would need blk.N.attn_{q,k,v,output}.bias tensors, which this
            # converter does not emit.
            raise NotImplementedError(
                "attention_bias=true requires attn bias tensors, not implemented"
            )

        # attention_dropout, output_router_logits, router_aux_loss_coef: training-only,
        # do not affect the forward pass and have no GGUF representation. Not written.

    def iter_tensors(self, source: SafetensorsSource, config: dict) -> Iterator[TensorSpec]:
        n_layers = config["num_hidden_layers"]
        n_experts = config["num_experts"]
        T = gguf.MODEL_TENSOR

        yield TensorSpec(tensor_name(T.TOKEN_EMBD), source.get("model.embed_tokens.weight"), False)
        yield TensorSpec(tensor_name(T.OUTPUT_NORM), source.get("model.norm.weight"), True)

        if source.has("lm_head.weight"):
            output = source.get("lm_head.weight")
        elif config.get("tie_word_embeddings", False):
            output = source.get("model.embed_tokens.weight").copy()
        else:
            raise KeyError("lm_head.weight missing and tie_word_embeddings is false")
        yield TensorSpec(tensor_name(T.OUTPUT), output, False)

        for i in range(n_layers):
            p = f"model.layers.{i}."
            yield TensorSpec(tensor_name(T.ATTN_NORM, i), source.get(p + "input_layernorm.weight"), True)
            yield TensorSpec(tensor_name(T.ATTN_Q, i), source.get(p + "self_attn.q_proj.weight"), False)
            yield TensorSpec(tensor_name(T.ATTN_K, i), source.get(p + "self_attn.k_proj.weight"), False)
            yield TensorSpec(tensor_name(T.ATTN_V, i), source.get(p + "self_attn.v_proj.weight"), False)
            yield TensorSpec(tensor_name(T.ATTN_OUT, i), source.get(p + "self_attn.o_proj.weight"), False)
            yield TensorSpec(tensor_name(T.ATTN_Q_NORM, i), source.get(p + "self_attn.q_norm.weight"), True)
            yield TensorSpec(tensor_name(T.ATTN_K_NORM, i), source.get(p + "self_attn.k_norm.weight"), True)
            yield TensorSpec(tensor_name(T.FFN_NORM, i), source.get(p + "post_attention_layernorm.weight"), True)
            # Router: gate.weight is [n_expert, hidden] (only 2-D, not 1-D), but is
            # still forced F32 per spec.
            yield TensorSpec(tensor_name(T.FFN_GATE_INP, i), source.get(p + "mlp.gate.weight"), True)

            # Experts are separate nn.Linear per expert in HF (mlp.experts.{e}.*),
            # stacked here into one tensor per projection, expert order preserved:
            # numpy [n_expert, out, in] -> ggml ne [in, out, n_expert].
            gate = np.stack(
                [source.get(p + f"mlp.experts.{e}.gate_proj.weight") for e in range(n_experts)], axis=0
            )
            up = np.stack(
                [source.get(p + f"mlp.experts.{e}.up_proj.weight") for e in range(n_experts)], axis=0
            )
            down = np.stack(
                [source.get(p + f"mlp.experts.{e}.down_proj.weight") for e in range(n_experts)], axis=0
            )
            yield TensorSpec(tensor_name(T.FFN_GATE_EXP, i), gate, False)
            yield TensorSpec(tensor_name(T.FFN_UP_EXP, i), up, False)
            yield TensorSpec(tensor_name(T.FFN_DOWN_EXP, i), down, False)


def select_tensor_dtype(
    arr: np.ndarray, force_f32: bool, out_type: str
) -> tuple[np.ndarray, gguf.GGMLQuantizationType | None]:
    """Returns (data, raw_dtype) ready for GGUFWriter.add_tensor(name, data, raw_dtype=raw_dtype).
    raw_dtype is None for plain F32/F16 (inferred from the numpy dtype); Q8_0 needs
    it because the written array is packed quantized bytes, not float data."""
    if force_f32 or out_type == "f32":
        return arr.astype(np.float32, copy=False), None
    if out_type == "f16":
        return arr.astype(np.float16, copy=False), None
    if out_type == "q8_0":
        if arr.shape[-1] % Q8_0_BLOCK_SIZE == 0:
            q = gguf.quants.quantize(arr.astype(np.float32, copy=False), gguf.GGMLQuantizationType.Q8_0)
            return q, gguf.GGMLQuantizationType.Q8_0
        return arr.astype(np.float16, copy=False), None
    raise ValueError(f"unknown --type {out_type!r}")


def detect_bpe_pretokenizer(hf_dir: Path) -> str:
    """llama.cpp's real converter fingerprints tokenizer.json's pre-tokenizer
    against a large table of known models (convert_hf_to_gguf.py, not part of the
    gguf-py pip package) to pick tokenizer.ggml.pre. That table isn't available
    here, so this always falls back to "default" - llama.cpp's own fallback for an
    unrecognized pre-tokenizer. Verify against a real llama.cpp conversion before
    trusting this for anything beyond the tokenizer-less tiny fixture."""
    print(
        "note: BPE pre-tokenizer family not detected (no signature table available); "
        "writing tokenizer.ggml.pre='default' - verify before using on a real model",
        file=sys.stderr,
    )
    return "default"


def does_token_look_special(text: str) -> bool:
    """Mirrors llama.cpp convert_hf_to_gguf.py Model.does_token_look_special:
    added tokens that are not marked `special` in tokenizer.json but that look
    like they should be a control token by their spelling (some converters get
    this wrong upstream, e.g. deepseek-coder, gemma)."""
    if text in ("<pad>", "<mask>", "<2mass>", "[@BOS@]"):
        return True
    if text.startswith("<|") and text.endswith("|>"):
        return True
    if text.startswith("<｜") and text.endswith("｜>"):  # "<｜...｜>", e.g. deepseek_r1
        return True
    if text.startswith("<unused"):
        return True
    return False


def get_vocab_base(hf_dir: Path, config: dict) -> tuple[list[bytes], list[int]]:
    """Mirrors llama.cpp convert_hf_to_gguf.py Model.get_vocab_base for a gpt2
    (BPE) tokenizer: AutoTokenizer.from_pretrained, one token+type per id in
    range(vocab_size), ids missing from the vocab filled with UNUSED "[PAD{i}]"
    placeholders. Unlike gguf.BpeVocab (which marks every added token CONTROL),
    an added token is CONTROL only when tokenizer.json marks it `special` or its
    spelling looks special (does_token_look_special); otherwise it is
    USER_DEFINED, with the non-normalized round-trip and the U+2581 -> space
    substitution llama.cpp applies to that case."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    vocab_size = config.get("vocab_size") or len(tokenizer.vocab)

    reverse_vocab = {id_: tok for tok, id_ in tokenizer.vocab.items()}
    added_vocab = tokenizer.get_added_vocab()
    added_tokens_decoder = tokenizer.added_tokens_decoder

    tokens: list[bytes] = []
    toktypes: list[int] = []
    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]".encode("utf-8"))
            toktypes.append(int(gguf.TokenType.UNUSED))
            continue

        text = reverse_vocab[i]
        if text in added_vocab:
            decoder_entry = added_tokens_decoder.get(i)
            # llama.cpp assumes CONTROL/USER_DEFINED tokens are pre-normalized;
            # a non-normalized added token is re-encoded/decoded once to match.
            if decoder_entry is not None and not decoder_entry.normalized:
                text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
            is_special = (decoder_entry is not None and decoder_entry.special) or does_token_look_special(
                reverse_vocab[i]
            )
            if is_special:
                toktypes.append(int(gguf.TokenType.CONTROL))
            else:
                text = text.replace("▁", " ")
                toktypes.append(int(gguf.TokenType.USER_DEFINED))
        else:
            toktypes.append(int(gguf.TokenType.NORMAL))
        tokens.append(text.encode("utf-8"))

    return tokens, toktypes


def write_tokenizer(
    writer: gguf.GGUFWriter, hf_dir: Path, config: dict, tokenizer_pre: str | None
) -> None:
    tokens, toktypes = get_vocab_base(hf_dir, config)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(tokenizer_pre if tokenizer_pre is not None else detect_bpe_pretokenizer(hf_dir))
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(hf_dir, load_merges=True)
    special_vocab.add_to_gguf(writer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("hf_dir", type=Path, help="HF transformers checkpoint directory")
    parser.add_argument("out_gguf", type=Path, help="output .gguf path")
    parser.add_argument("--type", default=None, choices=sorted(FILE_TYPE_FOR_OUT_TYPE),
                         help="required unless --vocab-only")
    parser.add_argument("--tokenizer-pre", default=None, metavar="NAME",
                         help="tokenizer.ggml.pre value; default falls back to detect_bpe_pretokenizer")
    parser.add_argument("--vocab-only", action="store_true",
                         help="write only general.architecture/name and tokenizer.* metadata, no tensors")
    args = parser.parse_args(argv)

    if not args.vocab_only and args.type is None:
        parser.error("--type is required unless --vocab-only")

    hf_dir: Path = args.hf_dir.resolve()
    config_path = hf_dir / "config.json"
    if not config_path.exists():
        print(f"error: {config_path} not found", file=sys.stderr)
        return 1
    config = json.loads(config_path.read_text(encoding="utf-8"))

    architectures = config.get("architectures") or []
    if not architectures:
        print(f"error: {config_path} has no 'architectures' field", file=sys.stderr)
        return 1
    hf_arch = architectures[0]
    converter_cls = ARCH_CONVERTERS.get(hf_arch)
    if converter_cls is None:
        supported = ", ".join(sorted(ARCH_CONVERTERS)) or "(none registered)"
        print(f"error: unsupported architecture {hf_arch!r}; supported: {supported}", file=sys.stderr)
        return 1
    converter = converter_cls()

    arch_name = gguf.MODEL_ARCH_NAMES[converter.gguf_arch]
    model_name = config.get("_name_or_path") or hf_dir.name

    args.out_gguf.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out_gguf), arch_name, endianess=gguf.GGUFEndian.LITTLE)
    # GGUFWriter.__init__ already calls add_architecture() from the `arch` argument.
    writer.add_name(model_name)

    n_tensors = 0
    if args.vocab_only:
        print("note: --vocab-only: no tensors, no hparams beyond general.architecture/name", file=sys.stderr)
    else:
        converter.write_metadata(writer, config)
        writer.add_file_type(FILE_TYPE_FOR_OUT_TYPE[args.type])
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        source = SafetensorsSource(hf_dir)
        for spec in converter.iter_tensors(source, config):
            data, raw_dtype = select_tensor_dtype(spec.data, spec.force_f32, args.type)
            writer.add_tensor(spec.name, data, raw_dtype=raw_dtype)
            n_tensors += 1

    if (hf_dir / "tokenizer.json").exists():
        write_tokenizer(writer, hf_dir, config, args.tokenizer_pre)
    else:
        print(f"note: no tokenizer.json in {hf_dir}; GGUF written without tokenizer metadata", file=sys.stderr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    print(f"wrote {args.out_gguf} ({n_tensors} tensors, arch={arch_name}, type={args.type or 'vocab-only'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
