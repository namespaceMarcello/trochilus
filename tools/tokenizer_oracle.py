#!/usr/bin/env python3
"""tokenizer_oracle.py — check the C tokenizer against transformers.

Usage:
  tools/.venv/Scripts/python.exe tools/tokenizer_oracle.py \
      --gguf <vocab.gguf> --hf <hf_dir> --binary <path/to/trochilus> \
      [--compare-gguf <model.gguf>] [--no-sweep] [--seed N]

--gguf          the GGUF file passed to the binary as `-m` (tokenizer.* metadata
                only is enough - see hf_to_gguf.py --vocab-only).
--hf            HF transformers directory the same tokenizer came from
                (AutoTokenizer.from_pretrained(hf) is the reference).
--binary        the `trochilus` executable; contract:
                    BINARY tokenize -m GGUF --batch FILE [--no-parse-special] [--pieces | --decode]
                Batch file: repeated records, each an ASCII decimal byte
                length, "\n", then exactly that many UTF-8 bytes. Output: one
                "\n"-terminated line per record, in order.
                  default:   token ids in decimal joined by ",", empty line
                             for zero tokens. Adds special tokens per GGUF and
                             parses special tokens found in the text.
                  --pieces:  items joined by " ": a BPE piece (bytes of the
                             normalized text handed to BPE) as lowercase hex,
                             or a matched added token as "#"+id.
                  --decode:  lowercase hex of the decoded bytes for that
                             record's own ids.
                  --no-parse-special: added/special tokens in the text are
                             not recognized (split_special_tokens=True side).
--compare-gguf  if the path exists, every tokenizer.* GGUF key (values and
                arrays) is compared between --gguf and this file; each
                difference is printed and counted as a failure. If the path
                does not exist: "compare-gguf: skipped, <path> not found"
                (not a failure - the real model GGUF may not be downloaded).
--no-sweep      skip the full Unicode scalar-value sweep (check (c) below);
                keeps the run to the text bank and the random bank.
--seed          RNG seed for the random text bank (default: fixed, so a run
                is reproducible unless the caller asks for a different seed).

Texts, three sources, all encoded as UTF-8 for the batch file:
  (a) tests/tokenizer_bank.json - a hand-picked bank: source code, whitespace
      edge cases, scripts, emoji, digits, contractions, every added token
      (inline and adjacent to other text), near-miss fragments of added
      tokens, the empty string, one space, long runs.
  (b) >= 3000 seeded random strings, length 0..60 atoms, from a mixed-script
      alphabet (see random_bank / _random_atoms below).
  (c) unless --no-sweep: every Unicode scalar value (surrogates excluded) in
      a small snippet, ~64 code points per batch record, plus the NFD form of
      every code point whose NFD differs, grouped the same way.

Checks, run in two modes:
  default mode:          (a)+(b)+(c) - add_special_tokens=True, split_special_tokens=False
  no-parse-special mode: (a)+(b)     - add_special_tokens=True, split_special_tokens=True
For each mode: ids, pieces (only compared per-text when that text's HF ids
contain no added-token id; always used for the normalized-text
reconstruction, which is checked for every text), and decode (against
tok.decode of the ids the C binary itself produced in the same run).

Exit 1 on any mismatch, on a --compare-gguf difference, or if the binary
fails; exit 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

from gguf.gguf_reader import GGUFReader
from gguf.vocab import bytes_to_unicode

REPO_ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = REPO_ROOT / "tests" / "tokenizer_bank.json"

GROUP_SIZE = 64          # code points per record in the full sweep
RANDOM_BANK_SIZE = 3000  # >= 3000 required by the brief
DEFAULT_SEED = 20260917
MAX_PRINTED_MISMATCHES = 10
MAX_REPR_LEN = 200        # truncate long values in mismatch printouts


# ---------------------------------------------------------------------------
# Text banks
# ---------------------------------------------------------------------------

def load_text_bank() -> list[str]:
    with open(BANK_PATH, encoding="utf-8") as f:
        return json.load(f)


def _random_atoms() -> list[str]:
    atoms: list[str] = []
    atoms += list(string.ascii_letters)
    atoms += list(string.digits)
    atoms += list(string.punctuation)
    atoms += ["'", " ", "\t", "\n", "\r"]
    atoms += list("àéìòùáíóúñüçøåöÀÉÌÒÙ")  # accented, precomposed
    atoms += [unicodedata.normalize("NFD", c) for c in "àéìòùáíóúñü"]  # decomposed
    atoms += ["́", "̣", "ͅ", "̴"]  # bare combining marks
    atoms += ["ᄀ", "ᅡ", "ᆨ", "가", "나"]  # jamo L/V/T + syllables
    atoms += list("中文字符测试漢字")
    atoms += ["\U0001F600", "\U0001F44D", "\U0001F34E", "\U0001F1EE\U0001F1F9"]  # emoji incl. flag
    atoms += [" ", " "]  # NBSP, LINE SEPARATOR
    atoms += list("０１２３４５６７８９")  # fullwidth digits
    atoms += [  # added-token fragments (full and partial)
        "<|endoftext|>", "<pad>", "<|endoftext", "|||IP_ADDRESS||",
        "<|padding|>", "|||EMAIL_ADDRESS|||", "<<|endoftext|>>",
    ]
    return atoms


def random_bank(seed: int, n: int = RANDOM_BANK_SIZE) -> list[str]:
    rng = random.Random(seed)
    atoms = _random_atoms()
    out = []
    for _ in range(n):
        length = rng.randint(0, 60)
        out.append("".join(rng.choice(atoms) for _ in range(length)))
    return out


def _sweep_snippet(c: str) -> str:
    return c + "a" + c + " " + c + "1" + c + c + "\n"


def unicode_sweep() -> list[str]:
    scalars = [cp for cp in range(0x110000) if not (0xD800 <= cp <= 0xDFFF)]
    records = []
    for i in range(0, len(scalars), GROUP_SIZE):
        chunk = scalars[i:i + GROUP_SIZE]
        records.append("".join(_sweep_snippet(chr(cp)) for cp in chunk))

    nfd_forms = []
    for cp in scalars:
        c = chr(cp)
        d = unicodedata.normalize("NFD", c)
        if d != c:
            nfd_forms.append(d)
    for i in range(0, len(nfd_forms), GROUP_SIZE):
        chunk = nfd_forms[i:i + GROUP_SIZE]
        records.append("".join(_sweep_snippet(d) for d in chunk))
    return records


# ---------------------------------------------------------------------------
# Batch file protocol / binary invocation
# ---------------------------------------------------------------------------

def write_batch_file(path: Path, texts: list[str]) -> None:
    with open(path, "wb") as f:
        for t in texts:
            b = t.encode("utf-8")
            f.write(str(len(b)).encode("ascii"))
            f.write(b"\n")
            f.write(b)


def run_binary(binary: str, gguf_path: Path, batch_path: Path, extra_args: list[str]) -> list[str]:
    args = [binary, "tokenize", "-m", str(gguf_path), "--batch", str(batch_path), *extra_args]
    r = subprocess.run(args, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"binary failed (exit {r.returncode}) for {args[1:]}: "
            f"{r.stderr.decode('utf-8', errors='replace')[:2000]}"
        )
    out = r.stdout.decode("utf-8")
    lines = out.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def parse_ids_line(line: str) -> list[int]:
    if line == "":
        return []
    return [int(x) for x in line.split(",")]


def parse_pieces_line(line: str) -> list[str]:
    if line == "":
        return []
    return line.split(" ")


# ---------------------------------------------------------------------------
# HF reference side
# ---------------------------------------------------------------------------

class HFSide:
    def __init__(self, hf_dir: Path):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(hf_dir)
        self.bt = self.tok.backend_tokenizer
        added_vocab = self.tok.get_added_vocab()  # content -> id
        self.added_ids = set(added_vocab.values())
        self.id_to_added_content = {v: k for k, v in added_vocab.items()}
        encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in encoder.items()}

    def ids(self, texts: list[str], split_special: bool) -> list[list[int]]:
        return self.tok(texts, add_special_tokens=True, split_special_tokens=split_special)["input_ids"]

    def normalize(self, text: str) -> str:
        return self.bt.normalizer.normalize_str(text)

    def pretokenize_bytes(self, text: str) -> list[bytes]:
        segs = self.bt.pre_tokenizer.pre_tokenize_str(self.normalize(text))
        return [bytes(self.byte_decoder[ch] for ch in s) for s, _ in segs]

    def decode(self, ids: list[int]) -> bytes:
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).encode("utf-8")


# ---------------------------------------------------------------------------
# GGUF tokenizer.* key comparison
# ---------------------------------------------------------------------------

def compare_gguf_tokenizer_keys(vocab_path: Path, model_path: Path) -> list[str]:
    a = GGUFReader(str(vocab_path))
    b = GGUFReader(str(model_path))
    keys_a = {k for k in a.fields if k.startswith("tokenizer.")}
    keys_b = {k for k in b.fields if k.startswith("tokenizer.")}

    diffs: list[str] = []
    for key in sorted(keys_a | keys_b):
        if key not in keys_a:
            diffs.append(f"{key}: missing in {vocab_path.name}")
            continue
        if key not in keys_b:
            diffs.append(f"{key}: missing in {model_path.name}")
            continue
        va = a.fields[key].contents()
        vb = b.fields[key].contents()
        if va == vb:
            continue
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                diffs.append(f"{key}: length {len(va)} != {len(vb)}")
                continue
            first = next(i for i in range(len(va)) if va[i] != vb[i])
            diffs.append(f"{key}: differs at index {first} ({va[first]!r} != {vb[first]!r})")
        else:
            diffs.append(f"{key}: {va!r} != {vb!r}")
    return diffs


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _first_diff(a, b) -> int:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def run_mode(hf: HFSide, binary: str, gguf_path: Path, texts: list[str], split_special: bool,
             label: str, tmp_dir: Path) -> dict:
    batch_path = tmp_dir / f"{label}.batch"
    write_batch_file(batch_path, texts)
    extra = ["--no-parse-special"] if split_special else []

    results = {}

    # ---- ids ----
    t0 = time.time()
    hf_ids = hf.ids(texts, split_special)
    hf_time = time.time() - t0
    t0 = time.time()
    c_lines = run_binary(binary, gguf_path, batch_path, extra)
    c_time = time.time() - t0
    if len(c_lines) != len(texts):
        raise RuntimeError(f"{label} ids: binary returned {len(c_lines)} lines for {len(texts)} records")
    c_ids = [parse_ids_line(l) for l in c_lines]

    mism = []
    for i, (h, c) in enumerate(zip(hf_ids, c_ids)):
        if h != c:
            mism.append((i, h, c, _first_diff(h, c)))
    results["ids"] = (len(texts), mism, hf_time + c_time)

    # ---- pieces + normalized-text reconstruction ----
    t0 = time.time()
    c_lines = run_binary(binary, gguf_path, batch_path, extra + ["--pieces"])
    pieces_time = time.time() - t0
    if len(c_lines) != len(texts):
        raise RuntimeError(f"{label} pieces: binary returned {len(c_lines)} lines for {len(texts)} records")
    c_pieces = [parse_pieces_line(l) for l in c_lines]

    piece_mism = []
    norm_mism = []
    for i, text in enumerate(texts):
        items = c_pieces[i]

        recon = bytearray()
        parse_ok = True
        for item in items:
            if item.startswith("#"):
                content = hf.id_to_added_content.get(int(item[1:])) if item[1:].isdigit() else None
                if content is None:
                    parse_ok = False
                    break
                recon += content.encode("utf-8")
            else:
                try:
                    recon += bytes.fromhex(item)
                except ValueError:
                    parse_ok = False
                    break
        expected_norm = hf.normalize(text).encode("utf-8")
        if not parse_ok or bytes(recon) != expected_norm:
            norm_mism.append((i, expected_norm, bytes(recon) if parse_ok else b"<unparseable>",
                               _first_diff(expected_norm, bytes(recon))))

        if not any(t in hf.added_ids for t in hf_ids[i]):
            expected_pieces = [seg.hex() for seg in hf.pretokenize_bytes(text)]
            if items != expected_pieces:
                piece_mism.append((i, expected_pieces, items, _first_diff(expected_pieces, items)))

    results["pieces"] = (len(texts), piece_mism, pieces_time)
    results["normalized_text"] = (len(texts), norm_mism, 0.0)

    # ---- decode (against the C binary's own ids from this run) ----
    t0 = time.time()
    c_lines = run_binary(binary, gguf_path, batch_path, extra + ["--decode"])
    decode_time = time.time() - t0
    if len(c_lines) != len(texts):
        raise RuntimeError(f"{label} decode: binary returned {len(c_lines)} lines for {len(texts)} records")

    decode_mism = []
    for i, line in enumerate(c_lines):
        try:
            got = bytes.fromhex(line)
        except ValueError:
            got = None
        expected = hf.decode(c_ids[i])
        if got is None or got != expected:
            decode_mism.append((i, expected, got if got is not None else b"<bad hex>",
                                 _first_diff(expected, got or b"")))
    results["decode"] = (len(texts), decode_mism, decode_time)

    return results


def _trunc(v):
    if isinstance(v, (bytes, bytearray)) and len(v) > MAX_REPR_LEN:
        return bytes(v[:MAX_REPR_LEN]) + b"...(%d more bytes)" % (len(v) - MAX_REPR_LEN)
    if isinstance(v, list) and len(v) > MAX_REPR_LEN:
        return v[:MAX_REPR_LEN] + [f"...({len(v) - MAX_REPR_LEN} more)"]
    return v


def print_check(label: str, n_texts: int, mismatches: list[tuple], seconds: float, texts: list[str]) -> None:
    print(f"{label}: {n_texts} texts, {len(mismatches)} mismatches, {seconds:.2f}s")
    for i, hf_val, c_val, pos in mismatches[:MAX_PRINTED_MISMATCHES]:
        text_repr = texts[i]
        if len(text_repr) > MAX_REPR_LEN:
            text_repr = text_repr[:MAX_REPR_LEN] + f"...({len(text_repr) - MAX_REPR_LEN} more chars)"
        print(f"  [{i}] text={text_repr!r}")
        print(f"      hf={_trunc(hf_val)!r}")
        print(f"      c ={_trunc(c_val)!r}")
        print(f"      first differing position: {pos}")


# ---------------------------------------------------------------------------
# Chat template: `trochilus chat-template` against apply_chat_template
# ---------------------------------------------------------------------------

CHAT_ROLES = ["system", "user", "assistant", "user", "assistant", "tool"]
CHAT_CONVERSATIONS = 600


def chat_conversations(hf: "HFSide", pool: list[str], seed: int) -> list[tuple[list[dict], bool]]:
    """Random conversations (roles in any order, unknown roles too) that HF can render."""
    rng = random.Random(seed + 1)
    usable = [t for t in pool if "\x1e" not in t and "\x1f" not in t and len(t) < 400]
    candidates = [([], False), ([], True)]
    for _ in range(CHAT_CONVERSATIONS):
        msgs = [{"role": rng.choice(CHAT_ROLES), "content": rng.choice(usable)} for _ in range(rng.randint(1, 6))]
        candidates.append((msgs, rng.random() < 0.7))
    convs = []
    for msgs, gen in candidates:
        try:
            hf.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
        except Exception:
            continue
        convs.append((msgs, gen))
    return convs


def run_chat_template(hf: "HFSide", binary: str, gguf_path: Path, convs, tmp_dir: Path) -> dict:
    batch = tmp_dir / "chat.batch"
    with open(batch, "wb") as f:
        for msgs, gen in convs:
            rec = (b"1" if gen else b"0") + b"\x1e".join(
                m["role"].encode("utf-8") + b"\x1f" + m["content"].encode("utf-8") for m in msgs)
            f.write(str(len(rec)).encode("ascii") + b"\n" + rec)

    def c_lines(extra):
        args = [binary, "chat-template", "-m", str(gguf_path), "--batch", str(batch), *extra]
        r = subprocess.run(args, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"binary failed (exit {r.returncode}) for {args[1:]}: "
                               f"{r.stderr.decode('utf-8', errors='replace')[:2000]}")
        lines = r.stdout.decode("utf-8").split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if len(lines) != len(convs):
            raise RuntimeError(f"chat-template: binary returned {len(lines)} lines for {len(convs)} records")
        return lines

    t0 = time.time()
    rendered = [hf.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=g) for m, g in convs]
    text_mism, ids_mism = [], []
    for i, line in enumerate(c_lines([])):
        want = rendered[i].encode("utf-8")
        got = bytes.fromhex(line)
        if got != want:
            text_mism.append((i, want, got, _first_diff(want, got)))
    for i, line in enumerate(c_lines(["--ids"])):
        want = hf.tok(rendered[i], add_special_tokens=False)["input_ids"]
        got = parse_ids_line(line)
        if got != want:
            ids_mism.append((i, want, got, _first_diff(want, got)))
    secs = time.time() - t0
    return {"text": (len(convs), text_mism, secs), "ids": (len(convs), ids_mism, 0.0)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--hf", required=True, type=Path)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--compare-gguf", type=Path, default=None)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    ok = True

    if args.compare_gguf is not None:
        if not args.compare_gguf.exists():
            print(f"compare-gguf: skipped, {args.compare_gguf} not found")
        else:
            diffs = compare_gguf_tokenizer_keys(args.gguf, args.compare_gguf)
            print(f"compare-gguf: {len(diffs)} differences")
            for d in diffs:
                print(f"  {d}")
            if diffs:
                ok = False

    hf = HFSide(args.hf)

    bank = load_text_bank()
    rnd = random_bank(args.seed)
    sweep = [] if args.no_sweep else unicode_sweep()

    default_texts = bank + rnd + sweep
    restricted_texts = bank + rnd

    try:
        with tempfile.TemporaryDirectory(prefix="tokenizer_oracle_") as tmp:
            tmp_dir = Path(tmp)
            modes = [
                ("default", default_texts, False),
                ("no-parse-special", restricted_texts, True),
            ]
            all_results = []
            for label, texts, split_special in modes:
                results = run_mode(hf, args.binary, args.gguf, texts, split_special, label, tmp_dir)
                all_results.append((label, texts, results))
            convs = chat_conversations(hf, restricted_texts, args.seed)
            chat_results = run_chat_template(hf, args.binary, args.gguf, convs, tmp_dir)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total_mismatches = 0
    for label, texts, results in all_results:
        for check_name in ("ids", "pieces", "normalized_text", "decode"):
            n, mism, secs = results[check_name]
            print_check(f"{label}/{check_name}", n, mism, secs, texts)
            total_mismatches += len(mism)
    conv_reprs = [repr((m, g)) for m, g in convs]
    for check_name in ("text", "ids"):
        n, mism, secs = chat_results[check_name]
        print_check(f"chat-template/{check_name}", n, mism, secs, conv_reprs)
        total_mismatches += len(mism)

    if total_mismatches:
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
