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
# "@@" in a mutation's text stands for a newline; a mutation is one or more (text, replacement) pairs
cat > /tmp/mutq.py <<'EOF'
import sys
path = sys.argv[1]
s = open(path, encoding="utf-8").read()
for i in range(2, len(sys.argv) - 1, 2):
    old, new = sys.argv[i].replace("@@", "\n"), sys.argv[i + 1].replace("@@", "\n")
    if s.count(old) != 1:
        sys.exit("mutation does not apply once: " + old)
    s = s.replace(old, new, 1)
open(path, "w", encoding="utf-8").write(s)
EOF
rm -rf /tmp/mutq && mkdir -p /tmp/mutq
tar -c --exclude=tools/.venv --exclude=tools/docker Makefile CLAUDE.md src tests tools bench docs | tar -x -C /tmp/mutq
cd /tmp/mutq
# every mutation is first applied without a build (seconds): a text the code no longer has stops the script here,
# not half an hour later at its own line (MUTATE_DRY=1: only this pass)
DRY=1
# the files a mutation may touch, each put back before the next one
FILES="src/kernels/kernels.c src/kernels/kernels_x86.c src/kernels/kernels_internal.h src/models/olmoe.c"
for f in $FILES; do cp $f /tmp/pristine-q4x-$(basename $f); done
# $1: name, $2: file, then pairs of text and replacement
run() {
  NAME=$1
  shift
  for f in $FILES; do cp /tmp/pristine-q4x-$(basename $f) $f; done
  $PYBIN /tmp/mutq.py "$@"
  if [ "$DRY" = 1 ]; then
    echo "$NAME: applies"
    return 0
  fi
  RES=""
  if make -j8 BUILD=b CC=gcc b/tests/test_kernels b/tests/test_tier_used > /tmp/mutq-build.log 2>&1; then
    if timeout 900 ./b/tests/test_kernels > /tmp/mutq-k.log 2>&1; then RES="$RES kernels=green"; else RES="$RES kernels=RED"; fi
    if TR_CPU_MAX=avx512-novbmi timeout 900 ./b/tests/test_kernels > /tmp/mutq-n.log 2>&1; then RES="$RES novbmi=green";
    else RES="$RES novbmi=RED"; fi
    if timeout 900 ./b/tests/test_tier_used > /tmp/mutq-t.log 2>&1; then RES="$RES tier=green"; else RES="$RES tier=RED"; fi
  else
    RES="$RES build=RED"
  fi
  echo "$NAME:$RES"
}
C=src/kernels/kernels.c
X=src/kernels/kernels_x86.c
I=src/kernels/kernels_internal.h
O=src/models/olmoe.c
mutations() {
run "no mutation" $C "static void k_q4x_prep(" "static void k_q4x_prep("
# the scalar definition
run "scalar prep: X truncated, not rounded" $C "const int32_t X = bad ? 0 : (int32_t)(((double)x[k] * f + magic) - magic);" \
  "const int32_t X = bad ? 0 : (int32_t)((double)x[k] * f);"
run "shift: never one step down" $I "if (ldexp((double)m, sh) > TR_Q4X_XMAX) sh--;" "(void)0;"
run "digits: the quotient truncated" $I "const int64_t q = x >= 0 ? (x + b / 2) / b : -((-x + b / 2) / b);" "const int64_t q = x / b;"
run "block: the mins of the mirrored sub-block" $C "M += (int64_t)m[j] * (int64_t)v->bf[8 * s + j];" \
  "M += (int64_t)m[7 - j] * (int64_t)v->bf[8 * s + j];"
# a fused multiply-add in the f64 steps moves the value's last bit, which random inputs hide in all but about one
# output in 2^29 (it survived 2026-09-26): test_kernels' cancelling-mins witness makes it visible
run "block: the value fused (the cancelling mins)" $C "const double val = d * (double)T - dmin * (double)M;" \
  "const double val = fma(d, (double)T, -(dmin * (double)M));"
run "scalar xt: one input row short" $C "for (int t = 0; t < T; t++) {@@        float o[2];@@        k_q4x_dot2(row0, row1," \
  "for (int t = 0; t < T - 1; t++) {@@        float o[2];@@        k_q4x_dot2(row0, row1,"
run "scalar panel and tile: the odd rows' constants at the even rows' places" $C \
  "static int q4x_eo(int r) { return (r & 1) ? 8 + r / 2 : r / 2; }" "static int q4x_eo(int r) { return r / 2; }"
run "scalar prep: every token term one more" $C "e += TR_Q4X_C * (a + c) + a * c;" "e += TR_Q4X_C * (a + c) + a * c + 1;"
# the driver
run "driver: the last input row never prepared" $C "tr_parallel_for(pool, n, 1, q4x_prep_body, &pc);" \
  "tr_parallel_for(pool, n - 1, 1, q4x_prep_body, &pc);"
run "driver: a one-row group sent to the tiles" $C "if (n_in < c->min_rows || !own) {" "if (!own) {"
run "driver: groups of 3 back on the panel (4 rows made 3)" $C "#define Q4X_MIN_TILE_ROWS 4" "#define Q4X_MIN_TILE_ROWS 3"
run "driver: the rows' road ignores q4x_dot_xt" $C "if (c->k->q4x_dot_xt != NULL)@@        while (p_end - p >= 2) {" \
  "if (0)@@        while (p_end - p >= 2) {"
run "driver: a run of 2 steps 3 rows (4 by rows lose one)" $C "p += T;" "p += 3;"
run "driver: a row that is not whole blocks goes through" $C \
  "rows % TR_PM_ROWS == 0 && tr_row_bytes(TR_TYPE_Q4_K, cols) != 0 && cols <= s->max_cols &&" \
  "rows % TR_PM_ROWS == 0 && cols <= s->max_cols &&"
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
run "avx512 tile: the even rows' value fused (the cancelling mins)" $X \
  "const __m512d ve = _mm512_sub_pd(_mm512_mul_pd(_mm512_loadu_pd(cst), te), _mm512_mul_pd(_mm512_loadu_pd(cst + 16), Me));" \
  "const __m512d ve = _mm512_fmsub_pd(_mm512_loadu_pd(cst), te, _mm512_mul_pd(_mm512_loadu_pd(cst + 16), Me));"
run "avx512 tile: the sums from -0.0 (the signed zero)" $X "ye[t] = yo[t] = _mm512_setzero_pd();" \
  "ye[t] = yo[t] = _mm512_set1_pd(-0.0);"
run "avx512 dot2: row 1's header read from row 0" $X "_mm_loadu_si128((const __m128i *)(const void *)h1), 1);" \
  "_mm_loadu_si128((const __m128i *)(const void *)h0), 1);"
run "avx512 dot2: the reduction's last pairs not joined" $X "_mm512_permute_pd(C, 0x55)" "_mm512_permute_pd(C, 0x00)"
run "avx512 dot2: d and dmin swapped" $X "hdm[h][2 * r] * sums[4 * r] - hdm[h][2 * r + 1] * sums[4 * r + 2]" \
  "hdm[h][2 * r + 1] * sums[4 * r] - hdm[h][2 * r] * sums[4 * r + 2]"
run "avx512 dot2: the high nibbles take the low sub-block's scale" $X \
  "const __m512i wh = _mm512_mullo_epi16(_mm512_srli_epi16(q, 4), _mm512_set1_epi16(hsc[h][r][2 * c + 1]));" \
  "const __m512i wh = _mm512_mullo_epi16(_mm512_srli_epi16(q, 4), _mm512_set1_epi16(hsc[h][r][2 * c]));"
run "avx512 dot2: the value fused (the cancelling mins)" $X \
  "const double val = hdm[h][2 * r] * sums[4 * r] - hdm[h][2 * r + 1] * sums[4 * r + 2];" \
  "const double val = fma(hdm[h][2 * r], sums[4 * r], -(hdm[h][2 * r + 1] * sums[4 * r + 2]));"
# AVX-512 xt (a short verify pass's runs of 2-3 input rows) and its header pre-pass and block end
run "avx512 xt: row 1's header read from row 0" $X "_mm_loadu_si128((const __m128i *)(const void *)k1), 1);" \
  "_mm_loadu_si128((const __m128i *)(const void *)k0), 1);"
run "avx512 xt: the reduction's last pairs not joined" $X "_mm512_permute_pd(Q, 0x55)" "_mm512_permute_pd(Q, 0x00)"
run "avx512 xt: d and dmin swapped" $X "hdm[2 * r] * sums[4 * r] - hdm[2 * r + 1] * sums[4 * r + 2]" \
  "hdm[2 * r + 1] * sums[4 * r] - hdm[2 * r] * sums[4 * r + 2]"
run "avx512 xt: row 1's mins from row 0's" $X "mm[r] = _mm512_mul_pd(_mm512_loadu_pd(hmf[r]), bf);" \
  "mm[r] = _mm512_mul_pd(_mm512_loadu_pd(hmf[0]), bf);"
run "avx512 xt: the value fused (the cancelling mins)" $X \
  "const double val = hdm[2 * r] * sums[4 * r] - hdm[2 * r + 1] * sums[4 * r + 2];" \
  "const double val = fma(hdm[2 * r], sums[4 * r], -(hdm[2 * r + 1] * sums[4 * r + 2]));"
run "avx512 xt: token 2's digits from token 1" $X \
  "const int16_t *d0 = v[t].v0 + s * TR_Q4_K_BLOCK_ELEMS + 64 * c, *d1 = v[t].v1 + s * TR_Q4_K_BLOCK_ELEMS + 64 * c;" \
  "const int16_t *d0 = v[t == 2 ? 1 : t].v0 + s * TR_Q4_K_BLOCK_ELEMS + 64 * c, *d1 = v[t == 2 ? 1 : t].v1 + s * TR_Q4_K_BLOCK_ELEMS + 64 * c;"
run "avx512 xt: row 1 fed row 0's weights" $X "(bq[r] + 32 * c)" "(bq[0] + 32 * c)"
run "avx512 xt: the sums from -0.0 (the signed zero)" $X "y[t][0] = y[t][1] = 0.0;" "y[t][0] = y[t][1] = -0.0;"
run "avx512 xt: the sums split in even and odd blocks (the block order)" $X \
  "double y[TR_Q4X_XT_MAX][2];" "double y[TR_Q4X_XT_MAX][2], yo[TR_Q4X_XT_MAX][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};" \
  "for (int t = 0; t < T; t++) q4x_end2(acc[t], v[t].bf + 8 * s, hmf[h], hdm[h], v[t].scale[s], y[t]);" \
  "for (int t = 0; t < T; t++) q4x_end2(acc[t], v[t].bf + 8 * s, hmf[h], hdm[h], v[t].scale[s], (s & 1) ? yo[t] : y[t]);" \
  "out[(int64_t)t * ys] = (float)y[t][0];@@        out[(int64_t)t * ys + 1] = (float)y[t][1];" \
  "out[(int64_t)t * ys] = (float)(y[t][0] + yo[t][0]);@@        out[(int64_t)t * ys + 1] = (float)(y[t][1] + yo[t][1]);"
run "avx512 xt: the kernel of 3 in the slot of 2" $X "{NULL, NULL, avx512_q4x_dot_xt_2, avx512_q4x_dot_xt_3}" \
  "{NULL, NULL, avx512_q4x_dot_xt_3, avx512_q4x_dot_xt_3}"
run "avx512 xt: the kernel of 2 in the slot of 3" $X "{NULL, NULL, avx512_q4x_dot_xt_2, avx512_q4x_dot_xt_3}" \
  "{NULL, NULL, avx512_q4x_dot_xt_2, avx512_q4x_dot_xt_2}"
# AVX2 (by rows; the avx512 tier's under the novbmi cap)
run "avx2 prep: the second factor dropped (blocks past 2^-96)" $X \
  "_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + k), fa), fb)" "_mm256_mul_ps(_mm256_loadu_ps(x + k), fa)"
run "avx2 dot2: digit 0 of the second 16 columns from the first" $X "(const __m256i *)(const void *)(v0 + 64 * c + 16 * i)" \
  "(const __m256i *)(const void *)(v0 + 64 * c + 16 * (i & 2))"
run "avx2 dot2: the value fused (the cancelling mins)" $X \
  "(double)tr_half_to_float(hd) * (double)(t0 + (int64_t)TR_Q4X_BASE * t1) -@@                               (double)tr_half_to_float(hm) * (double)M;" \
  "fma((double)tr_half_to_float(hd), (double)(t0 + (int64_t)TR_Q4X_BASE * t1),@@                               -((double)tr_half_to_float(hm) * (double)M));"
run "avx2 dot2: the sums from -0.0 (the signed zero)" $X "double y[2] = {0.0, 0.0};@@    for (int64_t s = 0;" \
  "double y[2] = {-0.0, -0.0};@@    for (int64_t s = 0;"
run "table: the W16 panel left out" $X "g_avx512.q4x_panel = avx512_q4x_panel;" "(void)avx512_q4x_panel;"
run "table: the avx512 xt left scalar's" $X "g_avx512.q4x_dot_xt = avx512_q4x_dot_xt;" "(void)avx512_q4x_dot_xt;"
# the prep once a row (docs/MEASUREMENTS.md §The prep once a row): the rows read in place, a grouped row through the map
run "driver: the map ignored (a grouped row reads its own index)" $C \
  "return c->xq + (size_t)(c->map != NULL ? c->map[p] : p) * c->xq_row;" "return c->xq + (size_t)p * c->xq_row;"
run "driver: a run's prepared rows in reverse" $C "for (int t = 0; t < T; t++) xq[t] = q4x_row(c, p + t);" \
  "for (int t = 0; t < T; t++) xq[t] = q4x_row(c, p + T - 1 - t);"
run "driver: a tile's prepared rows in reverse" $C "for (int t = 0; t < T; t++) xq[t] = q4x_row(c, p0 + t);" \
  "for (int t = 0; t < T; t++) xq[t] = q4x_row(c, p0 + T - 1 - t);"
run "avx512 xt: every token reads the first prepared row" $X \
  "        v[t] = tr_q4x_view_of(xq[t], n);@@        y[t][0] = y[t][1] = 0.0;" \
  "        v[t] = tr_q4x_view_of(xq[0], n);@@        y[t][0] = y[t][1] = 0.0;"
run "avx512 tile: every token reads the first prepared row" $X \
  "        v[t] = tr_q4x_view_of(xq[t], n);@@        ye[t] = yo[t]" "        v[t] = tr_q4x_view_of(xq[0], n);@@        ye[t] = yo[t]"
run "model: a grouped row's token taken by its slot" $O "s->xmap[s->place[j]] = j / n_used;" "s->xmap[s->place[j]] = j % n_used;"
run "model: q, k and v on rows nobody prepared" $O \
  "if (any) tr_q4x_prepare(pool, s->normed, n_tok, n_embd, s->xq_tok);" "(void)any;"
# the swiglu and the down's prep in one call
run "act: each row prepared before its swiglu" $C \
  "        swiglu_body(&c->sw, r * c->n, (r + 1) * c->n, worker);@@        k->q4x_prep(c->sw.x + r * c->n, c->n, c->xq + (size_t)r * c->xq_row);" \
  "        k->q4x_prep(c->sw.x + r * c->n, c->n, c->xq + (size_t)r * c->xq_row);@@        swiglu_body(&c->sw, r * c->n, (r + 1) * c->n, worker);"
run "act: row r prepared into the place of row r + 1" $C "c->xq + (size_t)r * c->xq_row);" "c->xq + (size_t)(r + 1) * c->xq_row);"
run "model: the down on rows the act never prepared" $O \
  "if (act_road) tr_swiglu_prepare(pool, s->h1, s->h2, n_rows, n_ff, s->pm.xq);" \
  "if (act_road) tr_swiglu(pool, s->h1, s->h2, n_rows * n_ff);"
# the argmax split over the pool (test_kernels' test_argmax)
run "argmax: a lane's ties to the later index" $C "if (c->x[i + l] > bv[l]) {" "if (c->x[i + l] >= bv[l]) {"
run "argmax: the lanes' ties to the higher index" $C "(bv[l] == v && bi[l] < idx)" "(bv[l] == v && bi[l] > idx)"
run "argmax: the merge from -inf, not from x[0]" $C "    float bv = x[0];" "    float bv = -INFINITY;"
run "argmax: the chunks merged backwards" $C "for (int64_t ch = 0; ch < c.n_chunks; ch++)" \
  "for (int64_t ch = c.n_chunks - 1; ch >= 0; ch--)"
}
mutations > /tmp/mutq-dry.log
echo "every mutation applies once ($(grep -c ': applies' /tmp/mutq-dry.log))"
if [ "${MUTATE_DRY:-0}" = 1 ]; then
  cd /src
  return 0
fi
make -j8 BUILD=b CC=gcc b/tests/test_kernels b/tests/test_tier_used > /tmp/mutq-build.log 2>&1 ||
  { tail -20 /tmp/mutq-build.log; exit 1; }
DRY=0
mutations
cd /src
}
main "$@"; exit
