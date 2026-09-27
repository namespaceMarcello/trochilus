#!/bin/sh
# mutate_idle.sh — do the tests see a wrong idle hint (threads.h tr_pool_hint, tr_pool_region; kernels.h
# tr_matmul_hint; olmoe.c hint_next and the idle switch)? A hint moves no data a result reads, so what the tests
# check is where it points, when it is taken and when it is not: test_base (each chunk's region as the call runs
# it; a hint taken whole, left the moment a call comes, ignored for chunk 0, past the width and with hints off, the
# workers awake), test_kernels (each worker's hint where its first block reads, on the Q4_K road and on the float
# road, capped, and nothing for a call of two rows a group) and test_tier_used (a decode's hints taken; none with
# the idle switch's arm B, whose logits are the engine's bits). A test never seen red proves nothing
# (docs/LESSONS.md #43): each mutation is applied to a copy of the tree, the copy is built and the checks run.
# Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_idle.sh
#
# One line per mutation: which checks went red. "no mutation" must be all green, every other line
# must have at least one RED. About 8 minutes.
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
  if make BUILD=b CC=gcc WERROR=1 b/tests/test_base b/tests/test_kernels b/tests/test_tier_used > /dev/null 2>&1; then
    for T in test_base test_kernels test_tier_used; do
      if timeout 300 ./b/tests/$T > /dev/null 2>&1; then RES="$RES $T=green"; else RES="$RES $T=RED"; fi
    done
  else
    RES=" BUILD FAILED (not a mutation seen: fix the mutation)"
  fi
  echo "$1:$RES"
  cd /src
}
B=src/base/threads.c
K=src/kernels/kernels.c
run "no mutation" $B "static void hint_run(slot *w) {" "static void hint_run(slot *w) {"
run "pool: a hint never left for a call" $B \
  'if (atomic_load_explicit(&w->state, memory_order_relaxed) == SLOT_READY) break;' '(void)0;'
run "pool: a spinning worker never looks at its hint" $B \
  'if (atomic_load_explicit(&w->hint_seq, memory_order_relaxed) != w->hint_seen) {' 'if (0) {'
run "pool: a hint past the width posted" $B \
  'chunk < 1 || chunk >= p->active || bytes == 0) return;' 'chunk < 1 || bytes == 0) return;'
run "pool: a hint posted with hints off" $B \
  'if (p == NULL || p->hint_level == 0 || chunk < 1' 'if (p == NULL || chunk < 1'
run "pool: hints on as a pool starts" $B '    tr_pool_set_hints(p, 0);
    const char *hint_env' '    tr_pool_set_hints(p, 2);
    const char *hint_env'
run "pool: a region one index past its end" $B \
  '*end = *begin + base + (chunk < rem ? 1 : 0);' '*end = *begin + base + (chunk <= rem ? 1 : 0);'
run "hint: an item's rows taken as its index" $K \
  'r0 = b % (rows / per) * per;' 'r0 = b % (rows / per);'
run "hint: the empty groups counted" $K \
  '            if (offsets[g + 1] > offsets[g]) k++;' '            k++;'
run "hint: the whole region, not half" $K \
  'size_t bytes = (size_t)((e - b) * per / 2) * rb' 'size_t bytes = (size_t)((e - b) * per) * rb'
run "hint: past max_bytes" $K '        if (bytes > max_bytes) bytes = max_bytes;
' ''
run "hint: a call of two rows a group hinted" $K \
  '        !tr_matmul_one_row_per_group(offsets, n_groups))' '        0)'
run "hint: the float road's chunks of one row" $K \
  'const int64_t min_chunk = per > 1 ? 1 : w[0].cols > 0 ? 4096 / w[0].cols + 1 : 1;' 'const int64_t min_chunk = 1;'
run "hint: the float road for a Q4_K call" $K \
  'const int64_t per = q4x_road(tr_kernels_get(), w, offsets, n_groups, s) ? TR_PM_ROWS : 1;' 'const int64_t per = 1;'
run "engine: the idle switch's arm B still hints" src/models/olmoe.c \
  'if (!s->ab_no_idle) tr_matmul_hint(' 'if (1) tr_matmul_hint('
run "engine: no hint at all" src/models/olmoe.c \
  'if (!s->ab_no_idle) tr_matmul_hint(' 'if (0) tr_matmul_hint('
}
main "$@"; exit
