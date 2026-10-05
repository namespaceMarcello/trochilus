#!/bin/sh
# mutate_head.sh — do the tests see a wrong head by a bound? A test never seen red proves nothing (docs/LESSONS.md #43):
# each mutation is applied to a copy of the tree, the copy is built and the checks that should notice are run:
# test_head_bound (the planes, every tier against the scalar bound, every bound against the engine's row, the argmax
# against the scan, the logits from the planes) and test_head_model (the engine: greedy tokens, lazy logits, verify
# passes, one head call a greedy pass); the proof's rows (prove) run tools/proof.sh's harnesses that should go red
# (docs/MEASUREMENTS.md §The head's bound proved). Linux container with cbmc, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-proof:local sh tools/mutate_head.sh
#
# One line per mutation: which checks went red. "no mutation" must be all green, every other line must have at least
# one RED. ONLY=<text>: only the mutations whose name holds it (docker run -e ONLY=...). About 20 minutes.
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
  case "$1" in *"${ONLY:-}"*) ;; *) return 0 ;; esac
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ &&
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ ';'
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  RES=""
  if make -j8 BUILD=b CC=gcc WERROR=1 b/tests/test_head_bound b/tests/test_head_model > /dev/null 2>&1; then
    for T in test_head_bound test_head_model; do
      if TR_TEST_FAILFAST=1 ./b/tests/$T > /dev/null 2>&1; then RES="$RES $T=green"; else RES="$RES $T=RED"; fi
    done
  else
    RES=" BUILD FAILED (not a mutation seen: fix the mutation)"
  fi
  echo "$1:$RES"
  cd /src
}
# the same for the proof (tools/proof.sh): $5 the harnesses that should go red (proof.sh's ONLY); no cbmc, n/a
prove() {
  case "$1" in *"${ONLY:-}"*) ;; *) return 0 ;; esac
  command -v cbmc > /dev/null || { echo "$1: proof=n/a (no cbmc: the trochilus-proof image)"; return 0; }
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests /tmp/mut/ &&
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ ';'
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  if ONLY="$5" OUT=/tmp/mut/proof sh tools/proof.sh > /tmp/mut/proof.out 2>&1; then RES=" proof $5=green"
  elif grep -q FAILED /tmp/mut/proof.out; then RES=" proof $5=RED"
  else RES=" proof $5: NOT RUN (fix the row): $(tail -1 /tmp/mut/proof.out)"; fi
  echo "$1:$RES"
  cd /src
}
H=src/kernels/head_bound.c
run "no mutation" $H "#define HB_LIM 32639" "#define HB_LIM 32639"
prove "no mutation (proof)" $H "#define HB_LIM 32639" "#define HB_LIM 32639" proof_
prove "proof: the margin halved" $H "? d : HB_MARGIN) * 0x1p-24;" "? d : HB_MARGIN) * 0x1p-25;" /margin/
prove "proof: the engine's path one rounding short" $H "n / 64 + 5 : n / 16 + 4;" "n / 64 + 5 : n / 16 + 3;" /margin/
prove "proof: hb_up not rounding up" $H "return (double)f >= v ? f : nextafterf(f, INFINITY);" "return f;" /up/
prove "proof: Q8_0's second half's u dropped" $H "((c0 >> 4) | ((c1 >> 4) << 4))" "((c0 >> 4))" planes_q8_0
prove "proof: Q4_K's bit-1 plane dropped" $H "ph[80 + k / 8] |= (unsigned char)(((q >> 1) & 1) << (k % 8));" \
  "(void)0;" planes_q4_k
prove "proof: a Q8_0 lane's u against the next element's low digit" $H "hb_u_q8_0(nib, i) * (256 * X[i] + Xl[i])" \
  "hb_u_q8_0(nib, i) * (256 * X[i] + Xl[i ^ 1])" bound_q8_0
prove "proof: a Q4_K lane's digits swapped" $H "hb_hv_q4_k(ph, k) * (256 * X[k] + Xl[k])" \
  "hb_hv_q4_k(ph, k) * (256 * Xl[k] + X[k])" bound_q4_k
prove "proof: the high digit one off" $H "(q + 32896) / 256 - 128" "(q + 32896) / 256 - 127" /digits/
prove "proof: the threshold strict" $H "if (U[r] >= th) ids" "if (U[r] > th) ids" argmax_q8_0/
prove "proof: the workers' tie to the higher row" $H "(wk->best == best && wk->best_i < bi)" \
  "(wk->best == best && wk->best_i > bi)" argmax_q8_0_pool
run "prep: no float margin" $H "return (double)(d > HB_MARGIN ? d : HB_MARGIN) * 0x1p-24;" "return 0.0;"
run "prep: a block of h below 2^-64 kept (#349)" $H "if (!zero && m < HB_HMIN) return -1;" "(void)0;"
run "prep: Q4_K's rounding of h left out of g" $H "+ 480.0 * err_el" "+ 0.0 * err_el"
run "prep: Q4_K's rounding of h left out of mu" $H "+ 32.0 * err_el" "+ 0.0 * err_el"
run "prep: the high digit one off" $H "(q + 32896) / 256 - 128" "(q + 32896) / 256 - 127"
run "planes: Q8_0's low nibbles of the second half" $H "((c0 & 15) | ((c1 & 15) << 4))" "((c0 & 15) | ((c0 & 15) << 4))"
run "rebuild avx2: Q8_0 without the sign flip" $H \
  "_mm_or_si128(_mm_xor_si128(_mm_slli_epi16(_mm_and_si128(h, m4), 4), x80), _mm_and_si128(l, m4))" \
  "_mm_or_si128(_mm_slli_epi16(_mm_and_si128(h, m4), 4), _mm_and_si128(l, m4))"
run "scalar: the chains folded in another order" $H \
  "((acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l]))" "((acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]))"
run "avx2: the chains folded in another order" $H \
  "_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3)), side)" \
  "_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), side)"
run "avx512: the side added before the halves" $H \
  "hb_hsum8(_mm256_add_ps(_mm256_add_ps(lo, hi), side))" "hb_hsum8(_mm256_add_ps(_mm256_add_ps(lo, side), hi))"
run "avx2 Q4_K: the bit plane left out" $H "_mm256_sub_epi8(_mm256_add_epi8(t, t), bb)" "_mm256_add_epi8(t, t)"
run "avx2 Q4_K: dmin's term added" $H \
  "side = _mm256_sub_ps(side, _mm256_mul_ps(_mm256_mul_ps(vdm, vm)" "side = _mm256_add_ps(side, _mm256_mul_ps(_mm256_mul_ps(vdm, vm)"
run "argmax: a NaN bound not seen" $H "        } else if (u != u) {
            wk->bad = 1;" "        } else if (u != u) {
            (void)0;"
run "argmax: the threshold strict" $H "if (U[r] >= th) ids" "if (U[r] > th) ids"
run "argmax: a worker's tie to the higher row" $H "(v == wk->best && r < wk->best_i)" "(v == wk->best && r > wk->best_i)"
run "argmax: the workers' tie to the higher row" $H "(wk->best == best && wk->best_i < bi)" "(wk->best == best && wk->best_i > bi)"
run "argmax: the threshold from two rows" src/kernels/kernels.h "#define TR_HB_TOP 4" "#define TR_HB_TOP 2"
O=src/models/olmoe.c
run "engine: the normed row not kept" $O "        memcpy(s->head_h, h, (size_t)m->n_embd * sizeof(float));
" ""
run "engine: the lazy logits never computed" $O "    if (s->head_pending) {
        tr_prof *prof" "    if (0) {
        tr_prof *prof"
run "engine: the pass's token ignored" src/models/model.c "        if (r >= 0) return r;" "        (void)r;"
run "engine: a verify pass by the bound" $O "if (planes && s->greedy && n_logits == 1) {" "if (planes && s->greedy) {"
}
main "$@"; exit
