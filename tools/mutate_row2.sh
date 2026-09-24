#!/bin/sh
# mutate_row2.sh — do the tests see a wrong two-row kernel (dot_row2_x4 and dot_row2_x8: two weight
# rows against the same four or eight input rows, AVX-512), a wrong lane tree in SIMD, a wrong use
# of them in tr_matmul, or a wrong SIMD Q6_K scale? A test
# never seen red proves nothing (docs/LESSONS.md #43): each mutation is applied to a copy of the
# tree, the copy is built and the checks that should notice are run: test_kernels (every tier's
# kernels against scalar's, tr_matmul against dot_row), test_tier_used (the tier's entries, every
# product through the active table) and test_prefill (a pass of many tokens = one token a pass).
# Linux container on an AVX-512 CPU, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_row2.sh
#
# One line per mutation: which checks went red. "no mutation" must be all green, every other line
# must have at least one RED. About 10 minutes.
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
nth = int(sys.argv[4]) if len(sys.argv) > 4 else 1
s = open(path, encoding="utf-8").read()
if s.count(old) < nth:
    sys.exit("mutation does not apply: " + old)
at = -1
for _ in range(nth):
    at = s.index(old, at + 1)
open(path, "w", encoding="utf-8").write(s[:at] + new + s[at + len(old):])
EOF
# $1: name, $2: file, $3: text, $4: replacement, $5: which occurrence (default the first)
run() {
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ &&
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ ';'
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4" ${5:-1}
  RES=""
  if make BUILD=b CC=gcc WERROR=1 b/tests/test_kernels b/tests/test_tier_used b/tests/test_prefill > /dev/null 2>&1; then
    for T in test_kernels test_tier_used test_prefill; do
      if ./b/tests/$T > /dev/null 2>&1; then RES="$RES $T=green"; else RES="$RES $T=RED"; fi
    done
  else
    RES=" BUILD FAILED (not a mutation seen: fix the mutation)"
  fi
  echo "$1:$RES"
  cd /src
}
K=src/kernels/kernels_x86.c
run "no mutation" $K "static void avx512_dot_row2_x4_q8_0(" "static void avx512_dot_row2_x4_q8_0("
run "q8_0: row 1 takes row 0's block scale" $K \
  "d1 = _mm512_set1_ps(tr_q8_0_block_scale(k1))" "d1 = _mm512_set1_ps(tr_q8_0_block_scale(k0))"
run "every type: row 1's fourth input row times row 0's weights" $K \
  "b3 = _mm512_add_ps(b3, _mm512_mul_ps(u, v_));" "b3 = _mm512_add_ps(b3, _mm512_mul_ps(w, v_));"
run "q8_0: row 1's first two sums swapped" $K \
  "TR_ROW2_OUT(out, a0, a1, a2, a3, b0, b1, b2, b3);" "TR_ROW2_OUT(out, a0, a1, a2, a3, b1, b0, b2, b3);"
run "q4_k: row 1's high nibbles look up row 0's values" $K \
  "_mm512_permutexvar_ps(_mm512_srli_epi32(q10, 4), vhi1)" "_mm512_permutexvar_ps(_mm512_srli_epi32(q10, 4), vhi0)"
run "q6_k: row 1's quants unpacked from row 0" $K "avx2_q6_k_unpack(k1, q1);" "avx2_q6_k_unpack(k0, q1);"
run "q6_k scales, avx512: one byte late" $K \
  "avx512_i8_to_ps(blk + TR_Q6_K_SCALES_OFFSET)" "avx512_i8_to_ps(blk + TR_Q6_K_SCALES_OFFSET + 1)"
run "q6_k scales, avx2: the second eight one byte early" $K \
  "avx2_i8_to_ps(blk + TR_Q6_K_SCALES_OFFSET + 8)" "avx2_i8_to_ps(blk + TR_Q6_K_SCALES_OFFSET + 7)"
run "table: avx512 q6_k two-row kernel left out" $K \
  "g_avx512.dot_row2_x4[TR_TYPE_Q6_K] = avx512_dot_row2_x4_q6_k;" "(void)avx512_dot_row2_x4_q6_k;"
M=src/kernels/kernels.c
run "matmul: row 1's sums stored over row 0's" $M \
  "c->y[(q + j) * c->rows + r + 1] = out[TR_DOT_TOKENS + j];" "c->y[(q + j) * c->rows + r] = out[TR_DOT_TOKENS + j];"
run "matmul: the tail tokens of row 1 dotted with row 0" $M \
  "c->y[q * c->rows + r + 1] = c->dot_row(row1," "c->y[q * c->rows + r + 1] = c->dot_row(row0,"
run "matmul: the last pair of rows skipped" $M "for (; r + 2 <= c->rows; r += 2) {" "for (; r + 4 <= c->rows; r += 2) {"
# eight tokens a call (dot_row2_x8) and the lane tree in SIMD (avx512_pair_sums)
run "x8 q8_0: row 1 takes row 0's block scale" $K \
  "d1 = _mm512_set1_ps(tr_q8_0_block_scale(k1))" "d1 = _mm512_set1_ps(tr_q8_0_block_scale(k0))" 2
run "x8 every type: row 1's eighth input row times row 0's weights" $K \
  "TR_ROW2_PAIR(x7, e, w_, u_, a7, b7);" "TR_ROW2_PAIR(x7, e, w_, w_, a7, b7);"
run "x8 every type: input row 5 is input row 4" $K "*x5 = x4 + stride" "*x5 = x4"
run "x8 out: row 1's first two sums swapped" $K \
  "avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(b0, b1)" "avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(b1, b0)"
run "x4 out: row 0's first two sums swapped" $K \
  "h_ = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(a0, a1)" "h_ = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(a1, a0)"
run "lane tree: lanes 12 and 14 paired the other way" $K \
  "_mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14," "_mm512_setr_epi32(0, 2, 4, 6, 8, 10, 14, 12,"
run "x8 q4_k: row 1's high nibbles look up row 0's values" $K \
  "_mm512_permutexvar_ps(_mm512_srli_epi32(q10, 4), vhi1)" "_mm512_permutexvar_ps(_mm512_srli_epi32(q10, 4), vhi0)" 2
run "x8 q6_k: row 1's quants unpacked from row 0" $K "avx2_q6_k_unpack(k1, q1);" "avx2_q6_k_unpack(k0, q1);" 2
run "table: avx512 q4_k x8 kernel left out" $K \
  "g_avx512.dot_row2_x8[TR_TYPE_Q4_K] = avx512_dot_row2_x8_q4_k;" "(void)avx512_dot_row2_x8_q4_k;"
run "matmul x8: row 1's sums stored over row 0's" $M \
  "c->y[(q + j) * c->rows + r + 1] = out[TR_DOT_TOKENS_WIDE + j];" "c->y[(q + j) * c->rows + r] = out[TR_DOT_TOKENS_WIDE + j];"
run "matmul: the x8 road never taken" $M "if (c->dot_row2_x8 != NULL) {" "if (c->dot_row2_x8 != NULL && c->rows < 0) {"
run "matmul x8: a whole group of eight left to x4" $M \
  "for (; q + TR_DOT_TOKENS_WIDE <= b_end; q += TR_DOT_TOKENS_WIDE) {" "for (; q + TR_DOT_TOKENS_WIDE < b_end; q += TR_DOT_TOKENS_WIDE) {"
}
main "$@"; exit
