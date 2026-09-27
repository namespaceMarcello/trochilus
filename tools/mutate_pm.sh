#!/bin/sh
# mutate_pm.sh — do the checks see a wrong phase-major matmul? A test never seen red proves nothing
# (docs/LESSONS.md #43): each mutation is applied to a copy of the tree, the copy is rebuilt and the
# kernel tests are run on it. Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_pm.sh
#
# One line per mutation, two checks: kernels (tests/test_kernels on the best tier) and novbmi (the same
# under TR_CPU_MAX=avx512-novbmi: Q4_K's integer road by rows; its W16 road is tools/mutate_q4x.sh's).
# "no mutation" must be all green, every other line must have at least one RED. The copy is
# built once and each mutation rebuilds only its file. About 10 minutes.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PYBIN=${PY:-tools/.venv/bin/python}
# "@@" in a mutation's text stands for a newline
cat > /tmp/mut.py <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2].replace("@@", "\n"), sys.argv[3].replace("@@", "\n")
s = open(path, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit("mutation does not apply once: " + old)
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
rm -rf /tmp/mut && mkdir -p /tmp/mut
tar -c --exclude=tools/.venv --exclude=tools/docker Makefile CLAUDE.md src tests tools bench docs | tar -x -C /tmp/mut
cd /tmp/mut
make -j8 BUILD=b CC=gcc b/tests/test_kernels > /tmp/mut-build.log 2>&1 || { tail -20 /tmp/mut-build.log; exit 1; }
cp src/kernels/kernels.c /tmp/pristine-kernels.c
cp src/kernels/kernels_x86.c /tmp/pristine-kernels_x86.c
# $1: name, $2: file, $3: text, $4: replacement
run() {
  cp /tmp/pristine-kernels.c src/kernels/kernels.c
  cp /tmp/pristine-kernels_x86.c src/kernels/kernels_x86.c
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  RES=""
  if make -j8 BUILD=b CC=gcc b/tests/test_kernels > /tmp/mut-build.log 2>&1; then
    if timeout 600 ./b/tests/test_kernels > /tmp/mut-k.log 2>&1; then RES="$RES kernels=green"; else RES="$RES kernels=RED"; fi
    if TR_CPU_MAX=avx512-novbmi timeout 600 ./b/tests/test_kernels > /tmp/mut-n.log 2>&1; then RES="$RES novbmi=green";
    else RES="$RES novbmi=RED"; fi
  else
    RES="$RES build=RED"
  fi
  echo "$1:$RES"
}
X=src/kernels/kernels_x86.c
C=src/kernels/kernels.c
run "no mutation" $X "static void pm_build_byte_idx(void) {" "static void pm_build_byte_idx(void) {"
run "tile: one FMA instead of a multiply then an add" $X "a[t] = _mm512_add_ps(a[t], _mm512_mul_ps(w, _mm512_set1_ps(xm[t])));" "a[t] = _mm512_fmadd_ps(w, _mm512_set1_ps(xm[t]), a[t]);"
run "tile: the accumulators start from -0.0" $X "for (int t = 0; t < T; t++) a[t] = _mm512_setzero_ps();" "for (int t = 0; t < T; t++) a[t] = _mm512_set1_ps(-0.0f);"
run "tree: phases 1 and 2 swapped" $X "__m512 s01 = _mm512_add_ps(PM_L(0), PM_L(1)), s23 = _mm512_add_ps(PM_L(2), PM_L(3));" "__m512 s01 = _mm512_add_ps(PM_L(0), PM_L(2)), s23 = _mm512_add_ps(PM_L(1), PM_L(3));"
run "tree: the low half added as a chain" $X "__m512 lo = _mm512_add_ps(_mm512_add_ps(s01, s23), _mm512_add_ps(s45, s67));" "__m512 lo = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(s01, s23), s45), s67);"
run "tile table: width 11 runs the tile of 10" $X "avx512_pm_tile_10, avx512_pm_tile_11," "avx512_pm_tile_10, avx512_pm_tile_10,"
run "interleave: the steps in decreasing m (a short run's tail lands on a step already written)" $X "const size_t ps = (size_t)(M * T + TR_PM_PAD);@@    for (int64_t m = 0; m < M; m++) {" "const size_t ps = (size_t)(M * T + TR_PM_PAD);@@    for (int64_t m = M - 1; m >= 0; m--) {"
run "q4x panel: one byte index of the scales' transpose wrong" $X "g_pm_byte_idx[1][ph * 8 + r] = (unsigned char)(src + 8);" "g_pm_byte_idx[1][ph * 8 + r] = (unsigned char)(src + 9);"
run "Q8_0 panel: a block's second half read from its first" $X "TR_Q8_0_SCALE_BYTES + 16 * half;" "TR_Q8_0_SCALE_BYTES + 0 * half;"
run "plan: every tile one input row short" $C "pl->tile_t[nt_all] = c0 + (t + 1) * len / nt - pl->tile_p[nt_all];" "pl->tile_t[nt_all] = c0 + (t + 1) * len / nt - pl->tile_p[nt_all] - 1;"
run "plan: a group's last chunk one input row short" $C "c1 = offsets[g] + (ch + 1) * cg / nch, len = c1 - c0;" "c1 = offsets[g] + (ch + 1) * cg / nch - (ch == nch - 1), len = c1 - c0;"
run "panel kept by its rows alone (another group's served)" $C "if (last[0] == g && last[1] == rg) return 0;" "if (last[1] == rg) return 0;"
run "panel kept from the call before (last not reset)" $C "for (int i = 0; i < 2 * n_workers; i++) pl->last[i] = -1;" "(void)0;"
run "guard: a pool larger than the scratch goes through" $C "if (pool != NULL && s != NULL && tr_pool_size(pool) > s->n_workers) s = NULL;" "(void)0;"
run "guard: a worker past the scratch takes a panel" $C "const int own = worker < c->s->n_workers;" "const int own = 1;"
run "guard: a row that is not whole blocks goes through" $C "cols % 16 != 0 || tr_row_bytes(type, cols) == 0 ||" "cols % 16 != 0 ||"
# the short verify pass (question 78): the road it takes, and the kernel of 2 and 3 input rows
run "short pass: refused again (a group of 4 wanted)" $C "    if (most < 2) return 0;" "    if (most < PM_MIN_ROWS) return 0;"
run "short pass: groups of 2 and 3 left to dot_row" $C "if ((t == 2 || t == 3) && c->k->dot_row_xt[c->w[g].type] != NULL) {" "if (0) {"
run "short pass: a group of 3 written as 2" $C "for (int64_t j = 0; j < t; j++) yp[j * c->rows + r] = out[j];" "for (int64_t j = 0; j < 2; j++) yp[j * c->rows + r] = out[j];"
run "xt avx512: 3 input rows run the kernel of 2" $X "if (t == 3) avx512_dot_row_x3_q8_0(row, x, stride, n, out);" "if (t == 3) avx512_dot_row_x2_q8_0(row, x, stride, n, out);"
run "xt avx2: 3 input rows run the kernel of 2" $X "if (t == 3) avx2_dot_row_x3_q8_0(row, x, stride, n, out);" "if (t == 3) avx2_dot_row_x2_q8_0(row, x, stride, n, out);"
run "xt avx512: the second input row read from the first" $X "const float *x0 = xb + j, *x1 = x0 + stride;@@            acc0" "const float *x0 = xb + j, *x1 = x0;@@            acc0"
run "xt avx2: the third row's high half from the second's" $X "hi2 = _mm256_add_ps(hi2, _mm256_mul_ps(w1, _mm256_loadu_ps(x2 + 8)));@@        }@@    }" "hi2 = _mm256_add_ps(hi2, _mm256_mul_ps(w1, _mm256_loadu_ps(x1 + 8)));@@        }@@    }"
run "xt scalar: the third input row read at the second's" $C "for (int u = 0; u < t; u++) lane[u][k % TR_LANES] += w * x[u * stride + k];" "for (int u = 0; u < t; u++) lane[u][k % TR_LANES] += w * x[(u > 0) * stride + k];"
# the review's survivors (2026-09-27): each must be red now
run "xt scalar: a zero sum given as -0.0 (lanes started at -0.0)" $C "    for (int u = 0; u < t; u++) out[u] = lane_combine(lane[u]);" "    for (int u = 0; u < t; u++) out[u] = lane_combine(lane[u]) == 0.0f ? -0.0f : lane_combine(lane[u]);"
run "xt avx512: the second accumulator starts at -0.0" $X "acc1 = _mm512_setzero_ps();" "acc1 = _mm512_set1_ps(-0.0f);"
run "short pass: xt for groups of 2 only" $C "if ((t == 2 || t == 3) && c->k->dot_row_xt[c->w[g].type] != NULL) {" "if ((t == 2) && c->k->dot_row_xt[c->w[g].type] != NULL) {"
run "short pass: xt for groups of 3 only" $C "if ((t == 2 || t == 3) && c->k->dot_row_xt[c->w[g].type] != NULL) {" "if ((t == 3) && c->k->dot_row_xt[c->w[g].type] != NULL) {"
run "short pass: the decode's call taken too" $C "    if (most < 2) return 0;" "    if (most < 1) return 0;"
run "short pass: the pm kernels asked for again (the old guard)" $C "    if (s == NULL || s->plan == NULL) return 0;" "    if (s == NULL || s->plan == NULL || k->pm_tile == NULL || k->pm_interleave == NULL || k->pm_panel[type] == NULL) return 0;"
run "xt avx512: two input rows run the kernel of three" $X "    else avx512_dot_row_x2_q8_0(row, x, stride, n, out);" "    else avx512_dot_row_x3_q8_0(row, x, stride, n, out);"
cd /src
}
main "$@"; exit
