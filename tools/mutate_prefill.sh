#!/bin/sh
# mutate_prefill.sh — do the tests see an error in the attention by groups, in the x4 kernels
# and in the work of one token split over the pool? A test never seen red proves nothing
# (docs/LEZIONI.md #43): each mutation is applied to a copy of the tree, the copy is built and
# the tests that should notice are run. Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm --security-opt seccomp=unconfined -v "$(pwd -W):/src" -w /src \
#       trochilus-dev:local sh tools/mutate_prefill.sh
#
# One line per mutation: which tests went red. "no mutation" must be all green, every other
# line must have at least one RED. About 45 minutes (every mutation builds its own tree); with an
# argument, only the mutations whose name holds that text (and "no mutation").
set -e
# The body is one function, called on the last line (docs/LEZIONI.md #69).
main() {
PYBIN=${PY:-tools/.venv/bin/python}
cat > /tmp/mut.py <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
if s.count(old) < 1:
    sys.exit("mutation does not apply: " + old)
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
TESTS="test_kernels test_prefill test_hot test_spec test_model_prof test_tier_used"
FAILED=0
ONLY=$1
wanted() { [ -z "$ONLY" ] || [ "$1" = "no mutation" ] || case "$1" in *"$ONLY"*) true ;; *) false ;; esac; }
# $1: name, $2: file, $3: text, $4: replacement
run() {
  wanted "$1" || return 0
  rm -rf /tmp/mut && mkdir -p /tmp/mut && cp -r Makefile src tests tools bench /tmp/mut/
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  BINS=""
  for T in $TESTS; do BINS="$BINS b/tests/$T"; done
  make BUILD=b CC=gcc $BINS > /dev/null 2>&1 || { echo "$1: does not build"; cd /src; FAILED=1; return; }
  RES=""
  REDS=0
  for T in $TESTS; do
    if ./b/tests/$T > /dev/null 2>&1; then RES="$RES $T=green"; else RES="$RES $T=RED"; REDS=$((REDS + 1)); fi
  done
  echo "$1:$RES"
  if [ "$1" = "no mutation" ]; then [ "$REDS" = 0 ] || FAILED=1; else [ "$REDS" -gt 0 ] || FAILED=1; fi
  cd /src
}
# A mutation that is a data race gives the right numbers most of the time: only ThreadSanitizer
# sees it every time (test_hot under TSan, as in make check). Same arguments as run.
run_tsan() {
  wanted "$1" || return 0
  rm -rf /tmp/mut && mkdir -p /tmp/mut && cp -r Makefile src tests tools bench /tmp/mut/
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  make BUILD=bt CC=gcc EXTRA_CFLAGS="-O1 -g -fsanitize=thread" EXTRA_LDFLAGS="-fsanitize=thread" bt/tests/test_hot \
    > /dev/null 2>&1 || { echo "$1: does not build"; cd /src; FAILED=1; return; }
  if TSAN_OPTIONS=halt_on_error=1 setarch "$(uname -m)" -R bt/tests/test_hot > /dev/null 2>&1; then
    echo "$1: test_hot(tsan)=green"; FAILED=1
  else
    echo "$1: test_hot(tsan)=RED"
  fi
  cd /src
}
K=src/kernels/kernels.c
X=src/kernels/kernels_x86.c
O=src/models/olmoe.c
run "no mutation" $K "last_n_pos" "last_n_pos"
run "group: a query sees one position too many" $K "int64_t n_pos = first_n_pos + j, end = t1 < n_pos ? t1 : n_pos, t = t0;
            for (; t + TR_ATTN_X <= end; t += TR_ATTN_X) {" "int64_t n_pos = first_n_pos + j + 1, end = t1 < n_pos ? t1 : n_pos, t = t0;
            for (; t + TR_ATTN_X <= end; t += TR_ATTN_X) {"
run "group: the first query of a block starts one late" $K "int64_t j0 = t0 + 1 > first_n_pos ? t0 + 1 - first_n_pos : 0; /* the first query that sees t0 */" "int64_t j0 = t0 + 1 > first_n_pos ? t0 + 2 - first_n_pos : 0;"
run "group: the softmax of a row one position short" $K "tr_softmax(scores + j * score_stride, first_n_pos + j);" "tr_softmax(scores + j * score_stride, first_n_pos + j - (j > 0));"
run "group: the blocks of values overlap by one position" $K "            for (; t < end; t++) k->axpy_f32(oj, values + t * head_dim, row[t], head_dim);" "            for (; t < end; t++) k->axpy_f32(oj, values + t * head_dim, row[t], head_dim);
            if (t1 < n_pos && t1 < last_n_pos) k->axpy_f32(oj, values + t1 * head_dim, row[t1], head_dim);"
run "group: scores not scaled in the x4 path" $K "for (int x = 0; x < TR_ATTN_X; x++) row[t + x] = row[t + x] * scale;" "for (int x = 1; x < TR_ATTN_X; x++) row[t + x] = row[t + x] * scale;"
run "x4 avx512: the third key is the second" $X "acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(av, _mm512_loadu_ps(b2 + k)));" "acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(av, _mm512_loadu_ps(b1 + k)));"
run "x4 avx512: the tail of the last value is dropped" $X "v = _mm512_add_ps(v, _mm512_mul_ps(a3, _mm512_maskz_loadu_ps(m, x3 + k)));" "v = _mm512_add_ps(v, _mm512_mul_ps(a3, _mm512_maskz_loadu_ps(m, x2 + k)));"
run "x4 avx2: values added in another order" $X "        v = _mm256_add_ps(v, _mm256_mul_ps(a0, _mm256_loadu_ps(x0 + k)));
        v = _mm256_add_ps(v, _mm256_mul_ps(a1, _mm256_loadu_ps(x1 + k)));" "        v = _mm256_add_ps(v, _mm256_mul_ps(a1, _mm256_loadu_ps(x1 + k)));
        v = _mm256_add_ps(v, _mm256_mul_ps(a0, _mm256_loadu_ps(x0 + k)));"
run "engine: every group of a head writes over the first one's outputs" $O "c->out + i * c->n_qkv + h * c->head_dim, c->n_qkv);" "c->out + i0 * c->n_qkv + h * c->head_dim, c->n_qkv);"
run "engine: the second group of a head sees the first one's context" $O "n_q, c->pos0 + i + 1," "n_q, c->pos0 + i0 + 1,"
run_tsan "engine: the scores of every worker are worker 0's (a race: ThreadSanitizer)" $O "c->scores + (int64_t)worker * OLMOE_ATTN_QUERIES * c->score_stride" "c->scores"
run_tsan "engine: every worker routes with the same scratch (a race: ThreadSanitizer)" $O "c->taken + (int64_t)worker * n_expert" "c->taken"
run "engine: a chunk of tokens is written at the start of the pass" $O "tr_kv_write(c->kv, c->layer, c->pos0 + begin, end - begin," "tr_kv_write(c->kv, c->layer, c->pos0, end - begin,"
run "engine: RoPE of a chunk uses the positions of the first chunk" $O "c->rope_cos + (c->pos0 + i) * half, *sin_p = c->rope_sin + (c->pos0 + i) * half;" "c->rope_cos + (c->pos0 + i - begin) * half, *sin_p = c->rope_sin + (c->pos0 + i - begin) * half;"
run "engine: the norm of a chunk copies only its first token" $O "(size_t)((end - begin) * c->n) * sizeof(float));" "(size_t)c->n * sizeof(float));"
run "engine: the rows of the experts skip the last of a chunk" $O "for (int64_t j = begin; j < end; j++)
        memcpy(c->xg" "for (int64_t j = begin; j + 1 < end; j++)
        memcpy(c->xg"
# the numbers stay right in the next two: only a check on WHICH function runs can see them (#78)
run "tier: the avx512 table loses its F32 rows and falls back to scalar's" $X "g_avx512.dot_row[TR_TYPE_F32] = avx512_dot_row_f32;" "/* gone */"
run "tier: the matmul takes scalar's table instead of the active one" $K "const tr_kernels *k = tr_kernels_get();
    ctx.dot_row = k->dot_row[w[0].type];" "const tr_kernels *k = tr_kernels_scalar();
    ctx.dot_row = k->dot_row[w[0].type];"
run "tier: the avx512 F16 row starts one half late" $X "acc = _mm512_add_ps(acc, _mm512_mul_ps(avx512_f16_to_ps(p + 2 * k), _mm512_loadu_ps(x + k)));" "acc = _mm512_add_ps(acc, _mm512_mul_ps(avx512_f16_to_ps(p + 2 * k + 2), _mm512_loadu_ps(x + k)));"
run "profiler: the bytes of the attention counted query by query" $O "positions += (uint64_t)(pos0 + (i + OLMOE_ATTN_QUERIES < n_tok ? i + OLMOE_ATTN_QUERIES : n_tok));" "positions += (uint64_t)(pos0 + i + 1);"
[ "$FAILED" = 0 ] && echo "mutate_prefill: every mutation seen, no mutation green" || { echo "mutate_prefill: a mutation was NOT seen (or the clean tree is red)"; exit 1; }
}
main "$@"; exit
