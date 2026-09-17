#!/usr/bin/env python3
"""Cut a real OLMoE GGUF down to its first N transformer blocks (default 2), for an
exact-reference oracle against transformers on real weights (see
tools/make_olmoe_2layer_ref.py and `make oracle-real`).

Tensor values are copied unchanged (same quantized bytes, never re-quantized): only
token_embd, output_norm, output and blk.0..blk.(N-1) are kept. Every other piece of
metadata -- including the tokenizer -- is copied as-is, except {arch}.block_count,
which is rewritten to N.

Pattern: ref/llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.py (MIT): stream a GGUF
file tensor-by-tensor through GGUFReader/GGUFWriter (add_tensor_info + write_tensor_data)
instead of buffering every kept tensor in memory at once via GGUFWriter.add_tensor.

Usage:
  tools/.venv/Scripts/python.exe tools/make_olmoe_2layer_gguf.py <src.gguf> <out.gguf> [--layers N]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import gguf

# Fields GGUFReader synthesizes for the file header itself, not real metadata.
PSEUDO_FIELDS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}

BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="source real GGUF (e.g. models/OLMoE-...-Q8_0.gguf)")
    ap.add_argument("out", type=Path, help="output GGUF path")
    ap.add_argument("--layers", type=int, default=2, help="number of leading blk.N kept (default 2)")
    args = ap.parse_args()

    reader = gguf.GGUFReader(str(args.src), "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    block_count_key = gguf.Keys.LLM.BLOCK_COUNT.format(arch=arch)

    n_present = len({int(m.group(1)) for t in reader.tensors if (m := BLOCK_RE.match(t.name))})
    if n_present < args.layers:
        sys.exit(f"error: {args.src} has only {n_present} blocks, need {args.layers}")

    keep_names = {"token_embd.weight", "output_norm.weight", "output.weight"}
    kept = [
        t for t in reader.tensors
        if t.name in keep_names
        or ((m := BLOCK_RE.match(t.name)) is not None and int(m.group(1)) < args.layers)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), arch=arch, endianess=reader.endianess)

    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    for field in reader.fields.values():
        if field.name in PSEUDO_FIELDS or field.name == gguf.Keys.General.ARCHITECTURE:
            continue  # GGUFWriter.__init__ already wrote general.architecture
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = args.layers if field.name == block_count_key else field.contents()
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    total_bytes = 0
    for t in kept:
        total_bytes += t.n_bytes
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in kept:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()

    print(f"wrote {args.out}: {len(kept)} tensors, {args.layers} layers, "
          f"{total_bytes / 1e6:.1f} MB of tensor data ({block_count_key}={args.layers})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
