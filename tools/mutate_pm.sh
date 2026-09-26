#!/bin/sh
# mutate_pm.sh — do the checks see a wrong phase-major matmul? A test never seen red proves nothing
# (docs/LESSONS.md #43): each mutation is applied to a copy of the tree, the copy is rebuilt and the
# kernel tests are run on it. Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_pm.sh
#
# One line per mutation, two checks: kernels (tests/test_kernels on the best tier: the VBMI panel
# where the CPU has it) and novbmi (the same under TR_CPU_MAX=avx512-novbmi: the float-transpose Q4_K
# panel). "no mutation" must be all green, every other line must have at least one RED. The copy is
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
run "no mutation" $X "TR_TARGET_AVX512VBMI __attribute__" "TR_TARGET_AVX512VBMI __attribute__"
run "tile: one FMA instead of a multiply then an add" $X "a[t] = _mm512_add_ps(a[t], _mm512_mul_ps(w, _mm512_set1_ps(xm[t])));" "a[t] = _mm512_fmadd_ps(w, _mm512_set1_ps(xm[t]), a[t]);"
run "tile: the accumulators start from -0.0" $X "for (int t = 0; t < T; t++) a[t] = _mm512_setzero_ps();" "for (int t = 0; t < T; t++) a[t] = _mm512_set1_ps(-0.0f);"
run "tree: phases 1 and 2 swapped" $X "__m512 s01 = _mm512_add_ps(PM_L(0), PM_L(1)), s23 = _mm512_add_ps(PM_L(2), PM_L(3));" "__m512 s01 = _mm512_add_ps(PM_L(0), PM_L(2)), s23 = _mm512_add_ps(PM_L(1), PM_L(3));"
run "tree: the low half added as a chain" $X "__m512 lo = _mm512_add_ps(_mm512_add_ps(s01, s23), _mm512_add_ps(s45, s67));" "__m512 lo = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(s01, s23), s45), s67);"
run "tile table: width 11 runs the tile of 10" $X "avx512_pm_tile_10, avx512_pm_tile_11," "avx512_pm_tile_10, avx512_pm_tile_10,"
run "interleave: the steps in decreasing m (a short run's tail lands on a step already written)" $X "const size_t ps = (size_t)(M * T + TR_PM_PAD);@@    for (int64_t m = 0; m < M; m++) {" "const size_t ps = (size_t)(M * T + TR_PM_PAD);@@    for (int64_t m = M - 1; m >= 0; m--) {"
run "VBMI panel: one byte index of the transpose wrong" $X "g_pm_byte_idx[1][ph * 8 + r] = (unsigned char)(src + 8);" "g_pm_byte_idx[1][ph * 8 + r] = (unsigned char)(src + 9);"
run "VBMI panel: the high nibbles with the low ones' scale" $X "const __m512 wh = _mm512_sub_ps(_mm512_mul_ps(shi," "const __m512 wh = _mm512_sub_ps(_mm512_mul_ps(slo,"
run "float-transpose Q4_K panel: high nibbles from the 16th element" $X "high = (i0 % 64) >= 32;" "high = (i0 % 64) >= 16;"
run "Q8_0 panel: a block's second half read from its first" $X "TR_Q8_0_SCALE_BYTES + 16 * half;" "TR_Q8_0_SCALE_BYTES + 0 * half;"
run "plan: every tile one input row short" $C "pl->tile_t[n_tiles] = c0 + (t + 1) * len / nt - pl->tile_p[n_tiles];" "pl->tile_t[n_tiles] = c0 + (t + 1) * len / nt - pl->tile_p[n_tiles] - 1;"
run "plan: a group's last chunk one input row short" $C "c1 = offsets[g] + (ch + 1) * cg / nch, len = c1 - c0;" "c1 = offsets[g] + (ch + 1) * cg / nch - (ch == nch - 1), len = c1 - c0;"
run "panel kept by its rows alone (another group's served)" $C "if (last[0] != g || last[1] != rg) {" "if (last[1] != rg) {"
run "panel kept from the call before (last not reset)" $C "for (int i = 0; i < 2 * s->n_workers; i++) pl->last[i] = -1;" "(void)0;"
run "guard: a pool larger than the scratch goes through" $C "if (pool != NULL && s != NULL && tr_pool_size(pool) > s->n_workers) s = NULL;" "(void)0;"
run "guard: a worker past the scratch takes a panel" $C "const int own = worker < c->s->n_workers;" "const int own = 1;"
run "guard: a row that is not whole blocks goes through" $C "cols % 16 != 0 || tr_row_bytes(type, cols) == 0 ||" "cols % 16 != 0 ||"
cd /src
}
main "$@"; exit
