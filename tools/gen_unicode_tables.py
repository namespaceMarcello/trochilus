#!/usr/bin/env python3
"""Generates src/tokenizer/unicode_data.h: the \\p{L} \\p{N} \\s regex classes and the NFC
data (canonical combining class, full decompositions, composition pairs) that Hugging Face
`tokenizers` uses for OLMoE-family models -- probed directly from the installed library,
not copied from a Unicode data release, since the reference for correctness is `tokenizers`
itself (its Oniguruma regex engine, its normalizers.NFD()/NFC()), not a Unicode version.

Before writing anything, every table is verified against `tokenizers`: a Python mirror of
the exact C algorithm in src/tokenizer/unicode.c (decompose, canonically reorder, compose)
is compared with tokenizers' NFC on every single scalar value, the NFD form of every
decomposable code point, every composition pair, all Hangul L/V/T/LV/LVT sequences, and a
batch of random sequences mixing starters, combining marks and jamo. Any disagreement
prints examples and exits 1 without touching unicode_data.h.

Usage:
    tools/.venv/Scripts/python.exe tools/gen_unicode_tables.py   (Windows)
    tools/.venv/bin/python tools/gen_unicode_tables.py           (Linux/macOS)
"""
import random
import sys
import time
import unicodedata
from pathlib import Path

from tokenizers import Regex, __version__ as TOKENIZERS_VERSION
from tokenizers import normalizers, pre_tokenizers

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "src" / "tokenizer" / "unicode_data.h"

MAX_CP = 0x110000
SURROGATE_LO, SURROGATE_HI = 0xD800, 0xDFFF

# Hangul algorithmic constants (UAX #15); kept out of the generated tables, mirrored here
# only so the self-check can validate the C code's algorithmic path against `tokenizers`.
HANGUL_SBASE = 0xAC00
HANGUL_LBASE = 0x1100
HANGUL_VBASE = 0x1161
HANGUL_TBASE = 0x11A7
HANGUL_LCOUNT = 19
HANGUL_VCOUNT = 21
HANGUL_TCOUNT = 28
HANGUL_NCOUNT = HANGUL_VCOUNT * HANGUL_TCOUNT
HANGUL_SCOUNT = HANGUL_LCOUNT * HANGUL_NCOUNT


def scalar_values():
    for cp in range(MAX_CP):
        if SURROGATE_LO <= cp <= SURROGATE_HI:
            continue
        yield cp


def is_hangul_syllable(cp):
    return HANGUL_SBASE <= cp < HANGUL_SBASE + HANGUL_SCOUNT


# ============================================================================================
# \p{L} \p{N} \s classes, probed from tokenizers' own Split(Regex(...), behavior="removed").
# ============================================================================================

CLASS_ID = {"L": 1, "N": 2, "space": 3}  # matches tr_uclass in unicode.h
CLASS_NAME = {1: "LETTER", 2: "NUMBER", 3: "SPACE"}


def probe_classes():
    patterns = {"L": r"\p{L}", "N": r"\p{N}", "space": r"\s"}
    splitters = {
        name: pre_tokenizers.Split(Regex(pat), behavior="removed")
        for name, pat in patterns.items()
    }
    class_of = bytearray(MAX_CP)  # 0 = OTHER, default for surrogates and unassigned cps
    hits_per_class = {name: 0 for name in patterns}
    for name, splitter in splitters.items():
        cid = CLASS_ID[name]
        for cp in scalar_values():
            # behavior="removed": a matching character disappears from the output entirely,
            # so a single-character input matches the class iff pre_tokenize_str returns [].
            if len(splitter.pre_tokenize_str(chr(cp))) == 0:
                if class_of[cp] != 0:
                    print(
                        f"ERROR: U+{cp:04X} matches both {CLASS_NAME[class_of[cp]]} and "
                        f"{name} -- classes must be disjoint",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                class_of[cp] = cid
                hits_per_class[name] += 1
    return class_of, hits_per_class


def ranges_from_bytes(values, max_index):
    """Sorted (first, last, value) ranges over values[0:max_index], merged where the scalar
    range is contiguous and the value is the same; runs of 0 are not emitted (default)."""
    ranges = []
    first = None
    prev_value = None
    prev_cp = None
    for cp in range(max_index):
        v = values[cp]
        if v != 0 and v == prev_value and cp == prev_cp + 1:
            prev_cp = cp
            continue
        if first is not None:
            ranges.append((first, prev_cp, prev_value))
        if v != 0:
            first = cp
            prev_value = v
            prev_cp = cp
        else:
            first = None
    if first is not None:
        ranges.append((first, prev_cp, prev_value))
    return ranges


# ============================================================================================
# NFC data: canonical combining class, full canonical decomposition, composition pairs.
# ============================================================================================


def build_ccc():
    """Canonical combining class per code point, from Python's unicodedata (as instructed):
    Hangul jamo and syllables are ccc 0 there too, so no special-casing is needed."""
    ccc = bytearray(MAX_CP)
    for cp in scalar_values():
        c = unicodedata.combining(chr(cp))
        if c:
            ccc[cp] = c
    return ccc


# Three old, unambiguously-assigned-forever marks used to bracket every candidate's ccc
# against `tokenizers`' own reordering behaviour: U+0334 COMBINING TILDE OVERLAY (ccc 1,
# IPA Extensions), U+093C DEVANAGARI SIGN NUKTA (ccc 7, Unicode 1.0), U+0345 COMBINING
# GREEK YPOGEGRAMMENI (ccc 240, Unicode 1.0). None of the three is itself ever flagged.
_ANCHOR_LOW = 0x334
_ANCHOR_LOW2 = 0x93C  # only used to bracket candidates whose own ccc is 1 (ties with _ANCHOR_LOW)
_ANCHOR_HIGH = 0x345


def detect_ccc_anomalies(ccc_of, hf_nfd):
    """`tokenizers` 0.22.2 turns out to bundle a combining-class table older than Unicode
    15.1: for a few hundred fairly recently assigned marks (Arabic Extended-C, Combining
    Diacritical Marks Extended/Supplement, several 15.0/16.0 scripts...) it never reorders
    them at all, matching exactly the signature of ccc 0 (a run boundary) in both
    directions against both a known-low and a known-high anchor. Detected here by probing
    `tokenizers` itself -- not by guessing a Unicode version -- so the generated ccc table
    matches what `tokenizers` actually does, not what Python's unicodedata says."""
    anomalies = set()
    for cp in scalar_values():
        c = ccc_of[cp]
        if c == 0 or cp in (_ANCHOR_LOW, _ANCHOR_LOW2, _ANCHOR_HIGH):
            continue
        bad = False
        if c == 1:
            s = chr(0x61) + chr(_ANCHOR_LOW2) + chr(cp)  # wrong order: 7 then 1
            if hf_nfd.normalize_str(s) != chr(0x61) + chr(cp) + chr(_ANCHOR_LOW2):
                bad = True
        else:
            s = chr(0x61) + chr(cp) + chr(_ANCHOR_LOW)  # wrong order: c(>1) then 1
            if hf_nfd.normalize_str(s) != chr(0x61) + chr(_ANCHOR_LOW) + chr(cp):
                bad = True
        if c < 240:
            s = chr(0x61) + chr(_ANCHOR_HIGH) + chr(cp)  # wrong order: 240 then c(<240)
            if hf_nfd.normalize_str(s) != chr(0x61) + chr(cp) + chr(_ANCHOR_HIGH):
                bad = True
        if bad:
            anomalies.add(cp)
    return anomalies


def build_decomposition(hf_nfd):
    """Full (fully recursive) canonical decomposition per code point, taken directly from
    `tokenizers`' own NFD of the isolated code point -- not from Python's decomposition
    tables, which is how U+11938 (HF does not decompose it, Python does) is handled for
    free. Hangul syllables are excluded: algorithmic in C."""
    decomp = {}
    for cp in scalar_values():
        if is_hangul_syllable(cp):
            continue
        s = hf_nfd.normalize_str(chr(cp))
        if s != chr(cp):
            decomp[cp] = [ord(c) for c in s]
    return decomp


def build_composition_pairs(hf_nfc):
    """Primary composites: code points with a length-2 *canonical* decomposition per Python
    (a compatibility decomposition, tagged like "<compat> ...", is not a composite; a
    singleton, one code point, has nothing to pair). Kept only if `tokenizers`' NFC actually
    recomposes a+b back to c -- this is what filters out composition-exclusion characters
    and any HF/Python data mismatch (e.g. U+11938's would-be pair), without a separate
    exclusion table."""
    pairs = []
    for cp in scalar_values():
        if is_hangul_syllable(cp):
            continue
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        parts = d.split()
        if len(parts) != 2:
            continue
        a, b = int(parts[0], 16), int(parts[1], 16)
        composed = hf_nfc.normalize_str(chr(a) + chr(b))
        if composed == chr(cp):
            pairs.append((a, b, cp))
    pairs.sort(key=lambda p: (p[0], p[1]))
    return pairs


# ============================================================================================
# Python mirror of the exact C algorithm (unicode.c): decompose -> canonical order -> compose.
# Used only to self-check the tables below against `tokenizers`; not part of the output.
# ============================================================================================


class Mirror:
    def __init__(self, decomp_map, ccc_map, compose_map):
        self.decomp_map = decomp_map
        self.ccc_map = ccc_map
        self.compose_map = compose_map

    def ccc(self, cp):
        return self.ccc_map.get(cp, 0)

    def hangul_decompose(self, cp):
        if not is_hangul_syllable(cp):
            return None
        s_index = cp - HANGUL_SBASE
        l = HANGUL_LBASE + s_index // HANGUL_NCOUNT
        v = HANGUL_VBASE + (s_index % HANGUL_NCOUNT) // HANGUL_TCOUNT
        t_index = s_index % HANGUL_TCOUNT
        return [l, v] if t_index == 0 else [l, v, HANGUL_TBASE + t_index]

    def decompose_one(self, cp):
        h = self.hangul_decompose(cp)
        if h is not None:
            return h
        return self.decomp_map.get(cp, [cp])

    def decompose_full(self, cps):
        out = []
        for cp in cps:
            out.extend(self.decompose_one(cp))
        return out

    def canonical_order(self, cps):
        out = list(cps)
        n = len(out)
        i = 0
        while i < n:
            if self.ccc(out[i]) == 0:
                i += 1
                continue
            j = i
            while j < n and self.ccc(out[j]) != 0:
                j += 1
            out[i:j] = sorted(out[i:j], key=self.ccc)  # Python sort is stable
            i = j
        return out

    def try_compose(self, starter, ch):
        if (
            HANGUL_LBASE <= starter < HANGUL_LBASE + HANGUL_LCOUNT
            and HANGUL_VBASE <= ch < HANGUL_VBASE + HANGUL_VCOUNT
        ):
            l_index = starter - HANGUL_LBASE
            v_index = ch - HANGUL_VBASE
            return HANGUL_SBASE + (l_index * HANGUL_VCOUNT + v_index) * HANGUL_TCOUNT
        if (
            HANGUL_SBASE <= starter < HANGUL_SBASE + HANGUL_SCOUNT
            and (starter - HANGUL_SBASE) % HANGUL_TCOUNT == 0
            and HANGUL_TBASE < ch < HANGUL_TBASE + HANGUL_TCOUNT
        ):
            return starter + (ch - HANGUL_TBASE)
        return self.compose_map.get((starter, ch))

    def compose(self, cps):
        result = []
        starter_index = -1
        last_class = -1
        for cp in cps:
            ccc = self.ccc(cp)
            composite = None
            if starter_index != -1 and last_class < ccc:
                composite = self.try_compose(result[starter_index], cp)
            if composite is not None:
                result[starter_index] = composite
            else:
                result.append(cp)
                if ccc == 0:
                    starter_index = len(result) - 1
                    last_class = -1
                else:
                    last_class = ccc
        return result

    def nfc_cps(self, cps):
        return self.compose(self.canonical_order(self.decompose_full(cps)))

    def nfc_str(self, s):
        return "".join(chr(c) for c in self.nfc_cps([ord(c) for c in s]))


# ============================================================================================
# Self-check: every single check below must pass, or nothing is written.
# ============================================================================================


def fail(title, examples):
    print(f"SELF-CHECK FAILED: {title}", file=sys.stderr)
    for e in examples[:20]:
        print(f"    {e}", file=sys.stderr)
    sys.exit(1)


def self_check(mirror, decomp_map, pairs, hf_nfd, hf_nfc):
    # 1. Mirror NFC vs HF NFC, every single scalar value.
    bad = []
    for cp in scalar_values():
        got = mirror.nfc_str(chr(cp))
        want = hf_nfc.normalize_str(chr(cp))
        if got != want:
            bad.append(f"U+{cp:04X}: mirror={[hex(ord(c)) for c in got]} hf={[hex(ord(c)) for c in want]}")
    if bad:
        fail("mirror NFC != HF NFC on a single code point", bad)

    # 2. Mirror decompose+order vs HF NFD, every decomposable code point.
    bad = []
    for cp, seq in decomp_map.items():
        got = mirror.canonical_order(mirror.decompose_full([cp]))
        want = [ord(c) for c in hf_nfd.normalize_str(chr(cp))]
        if got != want:
            bad.append(f"U+{cp:04X}: mirror={[hex(c) for c in got]} hf_nfd={[hex(c) for c in want]}")
    if bad:
        fail("mirror decompose != HF NFD on a decomposable code point", bad)

    # 3. Every composition pair recomposes under the mirror too.
    bad = []
    for a, b, c in pairs:
        got = mirror.nfc_str(chr(a) + chr(b))
        want = hf_nfc.normalize_str(chr(a) + chr(b))
        if got != chr(c) or want != chr(c):
            bad.append(f"({a:04X},{b:04X})->{c:04X}: mirror={ascii(got)} hf={ascii(want)}")
    if bad:
        fail("composition pair mismatch", bad)

    # 4. Hangul: every L+V, every LV+T, every L+V+T triple, both directions.
    bad = []
    for li in range(HANGUL_LCOUNT):
        l = HANGUL_LBASE + li
        for vi in range(HANGUL_VCOUNT):
            v = HANGUL_VBASE + vi
            lv = HANGUL_SBASE + (li * HANGUL_VCOUNT + vi) * HANGUL_TCOUNT
            got = mirror.nfc_str(chr(l) + chr(v))
            want = hf_nfc.normalize_str(chr(l) + chr(v))
            if got != chr(lv) or want != chr(lv):
                bad.append(f"L+V U+{l:04X}+U+{v:04X}: mirror={ascii(got)} hf={ascii(want)}")
            got_d = mirror.decompose_full([lv])
            want_d = [ord(c) for c in hf_nfd.normalize_str(chr(lv))]
            if got_d != [l, v] or want_d != [l, v]:
                bad.append(f"decompose LV U+{lv:04X}: mirror={got_d} hf={want_d}")
            for ti in range(1, HANGUL_TCOUNT):
                t = HANGUL_TBASE + ti
                lvt = lv + ti
                got2 = mirror.nfc_str(chr(lv) + chr(t))
                want2 = hf_nfc.normalize_str(chr(lv) + chr(t))
                if got2 != chr(lvt) or want2 != chr(lvt):
                    bad.append(f"LV+T U+{lv:04X}+U+{t:04X}: mirror={ascii(got2)} hf={ascii(want2)}")
                got3 = mirror.nfc_str(chr(l) + chr(v) + chr(t))
                want3 = hf_nfc.normalize_str(chr(l) + chr(v) + chr(t))
                if got3 != chr(lvt) or want3 != chr(lvt):
                    bad.append(f"L+V+T U+{l:04X}+U+{v:04X}+U+{t:04X}: mirror={ascii(got3)} hf={ascii(want3)}")
                got_d3 = mirror.decompose_full([lvt])
                want_d3 = [ord(c) for c in hf_nfd.normalize_str(chr(lvt))]
                if got_d3 != [l, v, t] or want_d3 != [l, v, t]:
                    bad.append(f"decompose LVT U+{lvt:04X}: mirror={got_d3} hf={want_d3}")
        if bad:
            break
    if bad:
        fail("Hangul mismatch", bad)

    # 5. >=20000 random sequences mixing starters, combining marks (various ccc, incl. equal
    #    ccc together), jamo and precomposed letters.
    rng = random.Random(0x1D160)
    ccc_pool = {}
    for cp in scalar_values():
        c = unicodedata.combining(chr(cp))
        if c:
            ccc_pool.setdefault(c, []).append(cp)
    ccc_buckets = list(ccc_pool.values())
    starters = [ord(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"]
    starters += [a for a, b, c in pairs][:200]  # bases that participate in composition
    precomposed = [c for a, b, c in pairs][:500]
    jamo_l = list(range(HANGUL_LBASE, HANGUL_LBASE + HANGUL_LCOUNT))
    jamo_v = list(range(HANGUL_VBASE, HANGUL_VBASE + HANGUL_VCOUNT))
    jamo_t = list(range(HANGUL_TBASE + 1, HANGUL_TBASE + HANGUL_TCOUNT))
    pools = [starters, precomposed, jamo_l, jamo_v, jamo_t] + ccc_buckets

    bad = []
    n_seq = 25000
    for _ in range(n_seq):
        length = rng.randint(1, 12)
        cps = []
        for _ in range(length):
            pool = rng.choice(pools)
            cps.append(rng.choice(pool))
        # occasionally force a run of equal-ccc marks, to exercise stability specifically
        if rng.random() < 0.3 and len(ccc_buckets) > 0:
            bucket = rng.choice(ccc_buckets)
            if len(bucket) >= 2:
                run = [rng.choice(bucket) for _ in range(rng.randint(2, 4))]
                pos = rng.randint(0, len(cps))
                cps[pos:pos] = run
        s = "".join(chr(c) for c in cps)
        got = mirror.nfc_str(s)
        want = hf_nfc.normalize_str(s)
        if got != want:
            bad.append(f"{[hex(c) for c in cps]}: mirror={[hex(ord(c)) for c in got]} hf={[hex(ord(c)) for c in want]}")
            if len(bad) >= 20:
                break
    if bad:
        fail(f"random sequence mismatch (of {n_seq})", bad)


# ============================================================================================
# Emit unicode_data.h
# ============================================================================================


def fmt_hex_rows(items, fmt_one, per_line):
    lines = []
    for i in range(0, len(items), per_line):
        row = " ".join(fmt_one(x) for x in items[i : i + per_line])
        lines.append("    " + row)
    return "\n".join(lines)


def emit_header(class_ranges, ccc_ranges, decomp_map, pairs, threshold, max_decomp, py_uver, n_ccc_anomalies):
    ascii_class = [0] * 128
    for first, last, klass in class_ranges:
        if first >= 128:
            continue
        for cp in range(first, min(last, 127) + 1):
            ascii_class[cp] = klass

    # Flatten decompositions, sorted by code point, into one data array + entry table.
    decomp_items = sorted(decomp_map.items())
    flat = []
    entries = []
    for cp, seq in decomp_items:
        entries.append((cp, len(flat), len(seq)))
        flat.extend(seq)

    lines = []
    lines.append("/* unicode_data.h -- generated by tools/gen_unicode_tables.py. Do not edit by hand;")
    lines.append(" * re-run the generator instead.")
    lines.append(" *")
    lines.append(f" * Source of truth: Hugging Face `tokenizers` {TOKENIZERS_VERSION} (the \\p{{L}} \\p{{N}} \\s")
    lines.append(" * classes of its Oniguruma regex engine, and its normalizers.NFD()/NFC()) -- not a")
    lines.append(f" * Unicode data release. Canonical combining class starts from Python's unicodedata")
    lines.append(f" * ({py_uver}): `tokenizers`' NFD/NFC agree with Python's on every single code point")
    lines.append(" * except U+11938, which `tokenizers` does not decompose. On *sequences*, though,")
    lines.append(f" * `tokenizers` turns out to bundle an older combining-class table: for {n_ccc_anomalies} code")
    lines.append(" * points (mostly scripts and marks added in Unicode 10.0-16.0) it never reorders them")
    lines.append(" * at all, exactly as if their combining class were 0 -- found by probing `tokenizers`")
    lines.append(" * itself (see detect_ccc_anomalies), and forced to 0 below to match. Every table here")
    lines.append(" * was checked against `tokenizers` directly before this file was written.")
    lines.append(" *")
    lines.append(" * Hangul syllables (U+AC00..U+D7A3) are algorithmic in unicode.c and excluded here. */")
    lines.append("#ifndef TR_UNICODE_DATA_H")
    lines.append("#define TR_UNICODE_DATA_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("/* ---- \\p{L} \\p{N} \\s classes (tr_uclass values from unicode.h) ------------------- */")
    lines.append("")
    lines.append("static const uint8_t TR_UASCII_CLASS[128] = {")
    lines.append(fmt_hex_rows(ascii_class, lambda x: f"{x},", 16))
    lines.append("};")
    lines.append("")
    lines.append("typedef struct { uint32_t first, last; uint8_t klass; } tr_uclass_range;")
    lines.append("")
    lines.append("static const tr_uclass_range TR_UCLASS_RANGES[] = {")
    for first, last, klass in class_ranges:
        lines.append(f"    {{0x{first:06x}, 0x{last:06x}, {klass}}},")
    lines.append("};")
    lines.append("#define TR_UCLASS_RANGES_COUNT (sizeof(TR_UCLASS_RANGES) / sizeof(TR_UCLASS_RANGES[0]))")
    lines.append("")
    lines.append("/* ---- NFC: canonical combining class ------------------------------------------------ */")
    lines.append("")
    lines.append("typedef struct { uint32_t first, last; uint8_t ccc; } tr_uccc_range;")
    lines.append("")
    lines.append("static const tr_uccc_range TR_UCCC_RANGES[] = {")
    for first, last, ccc in ccc_ranges:
        lines.append(f"    {{0x{first:06x}, 0x{last:06x}, {ccc}}},")
    lines.append("};")
    lines.append("#define TR_UCCC_RANGES_COUNT (sizeof(TR_UCCC_RANGES) / sizeof(TR_UCCC_RANGES[0]))")
    lines.append("")
    lines.append("/* ---- NFC: full canonical decomposition ----------------------------------------------- */")
    lines.append("")
    lines.append(f"#define TR_NFC_MAX_DECOMP {max_decomp}  /* longest single-code-point decomposition */")
    lines.append("")
    lines.append("static const uint32_t TR_UDECOMP_DATA[] = {")
    lines.append(fmt_hex_rows(flat, lambda x: f"0x{x:06x},", 10))
    lines.append("};")
    lines.append("")
    lines.append("typedef struct { uint32_t cp; uint32_t offset; uint8_t len; } tr_udecomp_entry;")
    lines.append("")
    lines.append("static const tr_udecomp_entry TR_UDECOMP_ENTRIES[] = {")
    for cp, offset, length in entries:
        lines.append(f"    {{0x{cp:06x}, {offset}, {length}}},")
    lines.append("};")
    lines.append("#define TR_UDECOMP_ENTRIES_COUNT (sizeof(TR_UDECOMP_ENTRIES) / sizeof(TR_UDECOMP_ENTRIES[0]))")
    lines.append("")
    lines.append("/* ---- NFC: canonical composition pairs (a, b) -> c, sorted by (a, b) ------------------ */")
    lines.append("")
    lines.append("typedef struct { uint32_t a, b, c; } tr_ucompose_pair;")
    lines.append("")
    lines.append("static const tr_ucompose_pair TR_UCOMPOSE_PAIRS[] = {")
    for a, b, c in pairs:
        lines.append(f"    {{0x{a:06x}, 0x{b:06x}, 0x{c:06x}}},")
    lines.append("};")
    lines.append("#define TR_UCOMPOSE_PAIRS_COUNT (sizeof(TR_UCOMPOSE_PAIRS) / sizeof(TR_UCOMPOSE_PAIRS[0]))")
    lines.append("")
    lines.append("/* Lowest code point that NFC could affect (decomposes, has ccc != 0, or is the second")
    lines.append(" * element of a composition pair). Every scalar value below this is unchanged by NFC. */")
    lines.append(f"#define TR_NFC_SAFE_THRESHOLD 0x{threshold:06x}u")
    lines.append("")
    lines.append("#endif")
    lines.append("")
    return "\n".join(lines)


def main():
    t_start = time.time()
    py_uver = unicodedata.unidata_version
    print(f"tokenizers {TOKENIZERS_VERSION}, Python unicodedata {py_uver}")

    print("probing \\p{L} \\p{N} \\s classes...")
    t0 = time.time()
    class_of, hits = probe_classes()
    class_ranges = ranges_from_bytes(class_of, MAX_CP)
    print(f"  L={hits['L']} N={hits['N']} space={hits['space']} code points, "
          f"{len(class_ranges)} ranges, {time.time() - t0:.2f}s")

    print("building ccc, decomposition and composition tables...")
    t0 = time.time()
    ccc_of = build_ccc()

    hf_nfd = normalizers.NFD()
    hf_nfc = normalizers.NFC()

    anomalies = detect_ccc_anomalies(ccc_of, hf_nfd)
    if anomalies:
        print(f"  {len(anomalies)} code points where `tokenizers` never reorders "
              f"(older combining-class data than Unicode {py_uver}); ccc forced to 0 there")
        for cp in anomalies:
            ccc_of[cp] = 0
    ccc_ranges = ranges_from_bytes(ccc_of, MAX_CP)
    ccc_map = {cp: ccc_of[cp] for cp in range(MAX_CP) if ccc_of[cp]}

    decomp_map = build_decomposition(hf_nfd)
    pairs = build_composition_pairs(hf_nfc)
    max_decomp = max((len(v) for v in decomp_map.values()), default=0)
    max_decomp = max(max_decomp, 3)  # Hangul LVT, algorithmic, not in decomp_map
    threshold = min(list(decomp_map.keys()) + list(ccc_map.keys()) + [b for a, b, c in pairs])
    print(f"  ccc: {len(ccc_ranges)} ranges; decomposition: {len(decomp_map)} entries "
          f"(max length {max_decomp}); composition: {len(pairs)} pairs; "
          f"threshold U+{threshold:04X}; {time.time() - t0:.2f}s")

    print("self-checking against tokenizers (mirror algorithm)...")
    t0 = time.time()
    compose_map = {(a, b): c for a, b, c in pairs}
    mirror = Mirror(decomp_map, ccc_map, compose_map)
    self_check(mirror, decomp_map, pairs, hf_nfd, hf_nfc)
    print(f"  all checks passed, {time.time() - t0:.2f}s")

    header = emit_header(class_ranges, ccc_ranges, decomp_map, pairs, threshold, max_decomp, py_uver, len(anomalies))
    OUT_PATH.write_text(header, encoding="utf-8", newline="\n")

    n_bytes = len(header.encode("utf-8"))
    print(f"wrote {OUT_PATH.relative_to(ROOT)}: {n_bytes} bytes")
    print(f"  class ranges: {len(class_ranges)}")
    print(f"  ccc ranges: {len(ccc_ranges)}")
    print(f"  decomposition entries: {len(decomp_map)} (flattened {sum(len(v) for v in decomp_map.values())} code points)")
    print(f"  composition pairs: {len(pairs)}")
    print(f"  NFC safe threshold: U+{threshold:04X}")
    print(f"total time: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
