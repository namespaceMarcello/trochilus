#!/bin/sh
# mutate_q6k.sh — do the tests see a wrong Q6_K? A test never seen red proves nothing
# (docs/LESSONS.md #43): each mutation is applied to a copy of the tree, the copy is built and the
# checks that should notice are run: test_kernels (dequant on a ggml-packed block, dot == dot of
# the dequantized row, every tier == scalar), test_tier_used (the tier's own entries, the engine
# on a Q6_K model), test_gguf (the type table) and tools/check_dequant.py (gguf-py bit for bit).
# Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_q6k.sh
#
# One line per mutation: which checks went red. "no mutation" must be all green, every other line
# must have at least one RED. About 5 minutes.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PYBIN=${PY:-tools/.venv/bin/python}
cat > /tmp/mut.py <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit("mutation does not apply once: " + old)
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
# $1: name, $2: file, $3: text, $4: replacement
run() {
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ &&
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ ';'
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  RES=""
  if make BUILD=b CC=gcc WERROR=1 b/tests/test_kernels b/tests/test_tier_used b/tests/test_gguf b/tests/dump_dequant \
      > /dev/null 2>&1; then
    for T in test_kernels test_tier_used test_gguf; do
      if ./b/tests/$T > /dev/null 2>&1; then RES="$RES $T=green"; else RES="$RES $T=RED"; fi
    done
    if $PYBIN tools/check_dequant.py --binary b/tests/dump_dequant > /dev/null 2>&1; then RES="$RES dequant=green"
    else RES="$RES dequant=RED"; fi
  else
    RES=" BUILD FAILED (not a mutation seen: fix the mutation)"
  fi
  echo "$1:$RES"
  cd /src
}
K=src/kernels
run "no mutation" $K/kernels.c "static void k_dequant_q6_k(" "static void k_dequant_q6_k("
run "scalar: qh bits of quarter c & 1" $K/kernels_internal.h ">> (2 * c)) & 3u" ">> (2 * (c & 1))) & 3u"
run "scalar: the neighbour sub-block's scale" $K/kernels_internal.h "d * (float)sc[j]" "d * (float)sc[j ^ 1]"
run "scalar: d read two bytes early" $K/kernels_internal.h "#define TR_Q6_K_D_OFFSET 208" "#define TR_Q6_K_D_OFFSET 206"
run "scalar: q - 31" $K/kernels_internal.h "(int)(lo | (hi << 4)) - 32" "(int)(lo | (hi << 4)) - 31"
run "unpack: quarter 1 takes qh bits 1-2" $K/kernels_x86.c "_mm256_slli_epi16(hb, 2), m2)" "_mm256_slli_epi16(hb, 3), m2)"
run "unpack: quarter 3 from A's high nibbles" $K/kernels_x86.c \
  "_mm256_and_si256(_mm256_srli_epi16(b, 4), m4)," "_mm256_and_si256(_mm256_srli_epi16(a, 4), m4),"
run "avx2 dot_row: lanes 8-15 take lanes 0-7's quants" $K/kernels_x86.c \
  "avx2_i8_to_ps(q + 16 * j + 8));
            lo = " "avx2_i8_to_ps(q + 16 * j));
            lo = "
run "avx2 x4: lanes 8-15 one quant off" $K/kernels_x86.c \
  "avx2_i8_to_ps(q + 16 * j + 8));
            const float *x0" "avx2_i8_to_ps(q + 16 * j + 7));
            const float *x0"
run "avx512 dot_row: even sub-block's scale" $K/kernels_x86.c \
  "scale[j]), avx512_i8_to_ps(q + 16 * j));
            acc = " "scale[j & 14]), avx512_i8_to_ps(q + 16 * j));
            acc = "
run "avx512 x4: the neighbour sub-block's scale" $K/kernels_x86.c \
  "scale[j]), avx512_i8_to_ps(q + 16 * j));
            const float *x0" "scale[j ^ 1]), avx512_i8_to_ps(q + 16 * j));
            const float *x0"
run "table: avx512 dot_row left to scalar's" $K/kernels_x86.c \
  "g_avx512.dot_row[TR_TYPE_Q6_K] = avx512_dot_row_q6_k;" "(void)avx512_dot_row_q6_k;"
run "table: avx2 x4 left to scalar's" $K/kernels_x86.c \
  "g_avx2.dot_row_x4[TR_TYPE_Q6_K] = avx2_dot_row_x4_q6_k;" "(void)avx2_dot_row_x4_q6_k;"
run "engine: Q6_K not supported" $K/kernels.c "type == TR_TYPE_Q4_K ||
           type == TR_TYPE_Q6_K;" "type == TR_TYPE_Q4_K;"
run "gguf: Q6_K blocks of 208 bytes" src/format/gguf.c '{"q6_k",   256, 210}' '{"q6_k",   256, 208}'
}
main "$@"; exit
