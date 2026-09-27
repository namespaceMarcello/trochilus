#!/bin/sh
# mutate_bf16.sh — do the tests see a wrong BF16 road (an F32 matrix narrowed at load to bfloat16 when its
# values fit: kernels.h tr_f32_to_bf16_exact, the BF16 rows of every tier, olmoe.c read_mat)? A test never seen
# red proves nothing (docs/LESSONS.md #43): each mutation is applied to a copy of the tree, the copy is built and
# the checks that should notice are run: test_kernels (the narrowing refused and done, BF16 rows against the F32
# rows they widen to on every tier and on scalar, dot_row and dot_row_x4) and test_tier_used (the tiers' own BF16
# entries; a model whose router fits bf16: its products counted through the BF16 entries, every pass's logits the
# bits of the same model loaded with the router F32). Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_bf16.sh
#
# One line per mutation: which checks went red. "no mutation" must be all green, every other line
# must have at least one RED. About 6 minutes.
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
  if make BUILD=b CC=gcc WERROR=1 b/tests/test_kernels b/tests/test_tier_used > /dev/null 2>&1; then
    for T in test_kernels test_tier_used; do
      if ./b/tests/$T > /dev/null 2>&1; then RES="$RES $T=green"; else RES="$RES $T=RED"; fi
    done
  else
    RES=" BUILD FAILED (not a mutation seen: fix the mutation)"
  fi
  echo "$1:$RES"
  cd /src
}
K=src/kernels
run "no mutation" $K/kernels.c "static void k_dequant_bf16(" "static void k_dequant_bf16("
run "scalar: the widening shift one short" $K/kernels_internal.h 'uint32_t bits = (uint32_t)b << 16;' 'uint32_t bits = (uint32_t)b << 15;'
run "scalar dot_row: a product in the next lane" $K/kernels.c 'float w = tr_bf16_to_float(r[k]);
        lane[k % TR_LANES] += w * x[k];' 'float w = tr_bf16_to_float(r[k]);
        lane[(k + 1) % TR_LANES] += w * x[k];'
run "scalar x4: tokens 1 and 3 read tokens 0 and 2" $K/kernels.c 'float w = tr_bf16_to_float(r[k]);
        for (int t = 0; t < TR_DOT_TOKENS; t++) lane[t][k % TR_LANES] += w * x[t * stride + k];' 'float w = tr_bf16_to_float(r[k]);
        for (int t = 0; t < TR_DOT_TOKENS; t++) lane[t][k % TR_LANES] += w * x[(t & 2) * stride + k];'
run "narrowing: the lowest bit ignored" $K/kernels.c 'if ((v & 0xFFFFu) != 0) return 0;' 'if ((v & 0xFFFEu) != 0) return 0;'
run "narrowing: the halves left at the floats' places" $K/kernels.c 'memcpy(p + 2 * i, &b, sizeof b);' 'memcpy(p + 4 * i, &b, sizeof b);'
run "narrowing: the low half kept" $K/kernels.c 'uint16_t b = (uint16_t)(v >> 16);' 'uint16_t b = (uint16_t)v;'
run "avx2: the widening shift one long" $K/kernels_x86.c '_mm256_slli_epi32(w, 16)' '_mm256_slli_epi32(w, 17)'
run "avx2 dot_row: lanes 8-15 take lanes 0-7's weights" $K/kernels_x86.c \
  'hi = _mm256_add_ps(hi, _mm256_mul_ps(avx2_bf16_to_ps(p + 2 * k + 16), _mm256_loadu_ps(x + k + 8)));' \
  'hi = _mm256_add_ps(hi, _mm256_mul_ps(avx2_bf16_to_ps(p + 2 * k), _mm256_loadu_ps(x + k + 8)));'
run "avx2 x4: lanes 8-15 one weight off" $K/kernels_x86.c \
  '__m256 w0 = avx2_bf16_to_ps(p + 2 * k), w1 = avx2_bf16_to_ps(p + 2 * k + 16);' \
  '__m256 w0 = avx2_bf16_to_ps(p + 2 * k), w1 = avx2_bf16_to_ps(p + 2 * k + 14);'
run "avx512: the widening shift one long" $K/kernels_x86.c '_mm512_slli_epi32(w, 16)' '_mm512_slli_epi32(w, 17)'
run "avx512 dot_row: the weights one element on" $K/kernels_x86.c \
  'acc = _mm512_add_ps(acc, _mm512_mul_ps(avx512_bf16_to_ps(p + 2 * k), _mm512_loadu_ps(x + k)));' \
  'acc = _mm512_add_ps(acc, _mm512_mul_ps(avx512_bf16_to_ps(p + 2 * k + 2), _mm512_loadu_ps(x + k)));'
run "avx512 x4: odd blocks read the even blocks' weights" $K/kernels_x86.c \
  '__m512 w = avx512_bf16_to_ps(p + 2 * k);' '__m512 w = avx512_bf16_to_ps(p + 2 * (k & ~(int64_t)16));'
run "table: avx512 dot_row left to scalar's" $K/kernels_x86.c \
  'g_avx512.dot_row[TR_TYPE_BF16] = avx512_dot_row_bf16;' '(void)avx512_dot_row_bf16;'
run "table: avx2 x4 left to scalar's" $K/kernels_x86.c \
  'g_avx2.dot_row_x4[TR_TYPE_BF16] = avx2_dot_row_x4_bf16;' '(void)avx2_dot_row_x4_bf16;'
run "engine: no F32 matrix narrowed" src/models/olmoe.c \
  'if (t->type == TR_TYPE_F32 && m->bf16_exact && tr_f32_to_bf16_exact(raw, rows * cols)) out->type = TR_TYPE_BF16;' '(void)0;'
run "engine: narrowed, and read as F32" src/models/olmoe.c \
  'tr_f32_to_bf16_exact(raw, rows * cols)) out->type = TR_TYPE_BF16;' 'tr_f32_to_bf16_exact(raw, rows * cols)) out->type = TR_TYPE_F32;'
run "engine: off unless asked" src/models/olmoe.c \
  'm->bf16_exact = bf16 == NULL || strcmp(bf16, "0") != 0;' 'm->bf16_exact = bf16 != NULL && strcmp(bf16, "0") != 0;'
}
main "$@"; exit
