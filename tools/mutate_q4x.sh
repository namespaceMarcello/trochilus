#!/bin/sh
# mutate_q4x.sh — do the checks see a wrong Q4_K integer road (kernels.h q4x_*)? A test never seen red proves
# nothing (docs/LESSONS.md #43): each mutation is applied to a copy of the tree, the copy is rebuilt and the
# checks run on it. Linux container on an AVX-512 CPU with VBMI and VNNI, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_q4x.sh
#
# One line per mutation, three checks: kernels (tests/test_kernels on the best tier: the W16 panel and
# tiles), novbmi (the same under TR_CPU_MAX=avx512-novbmi: AVX2's kernels by rows) and tier (test_tier_used:
# the table's entries, every product through them). "no mutation" must be all green, every other line must
# have at least one RED. The copy is built once and each mutation rebuilds only its file. About 15 minutes.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PYBIN=${PY:-tools/.venv/bin/python}
# "@@" in a mutation's text stands for a newline
cat > /tmp/mutq.py <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2].replace("@@", "\n"), sys.argv[3].replace("@@", "\n")
s = open(path, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit("mutation does not apply once: " + old)
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
rm -rf /tmp/mutq && mkdir -p /tmp/mutq
tar -c --exclude=tools/.venv --exclude=tools/docker Makefile CLAUDE.md src tests tools bench docs | tar -x -C /tmp/mutq
cd /tmp/mutq
make -j8 BUILD=b CC=gcc b/tests/test_kernels b/tests/test_tier_used > /tmp/mutq-build.log 2>&1 ||
  { tail -20 /tmp/mutq-build.log; exit 1; }
for f in kernels.c kernels_x86.c kernels_internal.h; do cp src/kernels/$f /tmp/pristine-q4x-$f; done
# $1: name, $2: file, $3: text, $4: replacement
run() {
  for f in kernels.c kernels_x86.c kernels_internal.h; do cp /tmp/pristine-q4x-$f src/kernels/$f; done
  $PYBIN /tmp/mutq.py "$2" "$3" "$4"
  RES=""
  if make -j8 BUILD=b CC=gcc b/tests/test_kernels b/tests/test_tier_used > /tmp/mutq-build.log 2>&1; then
    if timeout 900 ./b/tests/test_kernels > /tmp/mutq-k.log 2>&1; then RES="$RES kernels=green"; else RES="$RES kernels=RED"; fi
    if TR_CPU_MAX=avx512-novbmi timeout 900 ./b/tests/test_kernels > /tmp/mutq-n.log 2>&1; then RES="$RES novbmi=green";
    else RES="$RES novbmi=RED"; fi
    if timeout 900 ./b/tests/test_tier_used > /tmp/mutq-t.log 2>&1; then RES="$RES tier=green"; else RES="$RES tier=RED"; fi
  else
    RES="$RES build=RED"
  fi
  echo "$1:$RES"
}
C=src/kernels/kernels.c
X=src/kernels/kernels_x86.c
I=src/kernels/kernels_internal.h
run "no mutation" $C "static void k_q4x_prep(" "static void k_q4x_prep("
# the scalar definition
run "scalar prep: X truncated, not rounded" $C "const int32_t X = bad ? 0 : (int32_t)(((double)x[k] * f + magic) - magic);" \
  "const int32_t X = bad ? 0 : (int32_t)((double)x[k] * f);"
run "shift: never one step down" $I "if (ldexp((double)m, sh) > TR_Q4X_XMAX) sh--;" "(void)0;"
run "digits: the quotient truncated" $I "const int64_t q = x >= 0 ? (x + b / 2) / b : -((-x + b / 2) / b);" "const int64_t q = x / b;"
run "block: the mins of the mirrored sub-block" $C "M += (int64_t)m[j] * (int64_t)v->bf[8 * s + j];" \
  "M += (int64_t)m[7 - j] * (int64_t)v->bf[8 * s + j];"
# (a fused multiply-add in the block's f64 steps is not here: it moves the f64 value's last bit, which
# the result's f32 rounding hides in all but about one output in 2^29; it survived 2026-09-26)
run "scalar panel and tile: the odd rows' constants at the even rows' places" $C \
  "static int q4x_eo(int r) { return (r & 1) ? 8 + r / 2 : r / 2; }" "static int q4x_eo(int r) { return r / 2; }"
run "scalar prep: every token term one more" $C "e += TR_Q4X_C * (a + c) + a * c;" "e += TR_Q4X_C * (a + c) + a * c + 1;"
# the driver
run "driver: the last input row never prepared" $C "tr_parallel_for(pool, n_in, 1, q4x_prep_body, &c);" \
  "tr_parallel_for(pool, n_in - 1, 1, q4x_prep_body, &c);"
run "driver: a one-row group sent to the tiles" $C "if (n_in < Q4X_MIN_TILE_ROWS || !own) {" "if (!own) {"
run "driver: a row that is not whole blocks goes through" $C \
  "if (rows % TR_PM_ROWS != 0 || tr_row_bytes(TR_TYPE_Q4_K, cols) == 0 || cols > s->max_cols ||" \
  "if (rows % TR_PM_ROWS != 0 || cols > s->max_cols ||"
# AVX-512 (W16)
run "avx512 prep: V1 truncated" $X "_mm512_cvtpd_epi32(_mm512_mul_pd(xl, inv))" "_mm512_cvttpd_epi32(_mm512_mul_pd(xl, inv))"
run "avx512 prep: the token term without its constant" $X \
  "(int32_t)(q4x_hsum_u32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), a, b)) - (uint32_t)c2);" \
  "(int32_t)(q4x_hsum_u32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), a, b)));"
run "avx512 panel: the bias on the odd rows" $X "const __m512i bias = _mm512_set_epi32(0, INT32_MIN," \
  "const __m512i bias = _mm512_set_epi32(INT32_MIN, 0,"
run "avx512 panel: the quant transpose's byte pairs swapped" $X "((2 * o + w) * 8 + (r & 7)) * 2 + k);" \
  "((2 * o + w) * 8 + (r & 7)) * 2 + (k ^ 1));"
run "avx512 panel: the row term of the low words squared" $X "acc[u & 3] = _mm512_dpwssd_epi32(acc[u & 3], wl, wh);" \
  "acc[u & 3] = _mm512_dpwssd_epi32(acc[u & 3], wl, wl);"
run "avx512 panel: the scales' byte transpose one index off" $X \
  "g_pm_byte_idx[1][ph * 8 + r] = (unsigned char)(src + 8);" "g_pm_byte_idx[1][ph * 8 + r] = (unsigned char)(src + 9);"
run "avx512 tile: the flush's bias for three windows" $X "const __m512i bias4 = _mm512_set1_epi64(4ll << 31);" \
  "const __m512i bias4 = _mm512_set1_epi64(3ll << 31);"
run "avx512 tile: digit 0's f2 from the high column" $X "(const void *)(v[t].v0 + k0 + 2 * u)" \
  "(const void *)(v[t].v0 + k0 + 32 + 2 * u)"
run "avx512 tile: the odd rows' mins from the even rows'" $X \
  "__m512d Mo = _mm512_mul_pd(_mm512_loadu_pd(cst + 40), _mm512_set1_pd(bf[0]));" \
  "__m512d Mo = _mm512_mul_pd(_mm512_loadu_pd(cst + 32), _mm512_set1_pd(bf[0]));"
run "avx512 dot2: row 1's header read from row 0" $X "_mm_loadu_si128((const __m128i *)(const void *)h1), 1);" \
  "_mm_loadu_si128((const __m128i *)(const void *)h0), 1);"
run "avx512 dot2: the reduction's last pairs not joined" $X "_mm512_permute_pd(C, 0x55)" "_mm512_permute_pd(C, 0x00)"
run "avx512 dot2: d and dmin swapped" $X "hdm[h][2 * r] * sums[4 * r] - hdm[h][2 * r + 1] * sums[4 * r + 2]" \
  "hdm[h][2 * r + 1] * sums[4 * r] - hdm[h][2 * r] * sums[4 * r + 2]"
run "avx512 dot2: the high nibbles take the low sub-block's scale" $X "_mm512_set1_epi16(hsc[h][r][2 * c + 1])" \
  "_mm512_set1_epi16(hsc[h][r][2 * c])"
# AVX2 (by rows; the avx512 tier's under the novbmi cap)
run "avx2 prep: the second factor dropped (blocks past 2^-96)" $X \
  "_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + k), fa), fb)" "_mm256_mul_ps(_mm256_loadu_ps(x + k), fa)"
run "avx2 dot2: digit 0 of the second 16 columns from the first" $X "(const __m256i *)(const void *)(v0 + 64 * c + 16 * i)" \
  "(const __m256i *)(const void *)(v0 + 64 * c + 16 * (i & 2))"
run "table: the W16 panel left out" $X "g_avx512.q4x_panel = avx512_q4x_panel;" "(void)avx512_q4x_panel;"
cd /src
}
main "$@"; exit
