#!/bin/sh
# mutate_kv.sh — do the tests see a wrong index in the KV cache? A test never seen red proves
# nothing (docs/LESSONS.md #43): each mutation is applied to a copy of the tree, the copy is
# built and the tests that should notice are run. Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_kv.sh
#
# One line per mutation: which tests went red. "no mutation" must be all green, every other
# line must have at least one RED. About 15 minutes.
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
if s.count(old) < 1:
    sys.exit("mutation does not apply: " + old)
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
# $1: name, $2: file, $3: text, $4: replacement
run() {
  # tools without .venv: the repo's own Python environment is 706 MiB over a Windows bind mount,
  # and the container has its own ($PY below), so copying it made one mutation take minutes
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ &&
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ ';' &&
    ln -s /src/fixtures /tmp/mut/fixtures
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  make BUILD=b CC=gcc b/trochilus b/tests/test_kv b/tests/test_prefill b/tests/test_spec b/tests/test_session > /dev/null 2>&1
  RES=""
  for T in test_kv test_prefill test_spec test_session; do
    if ./b/tests/$T > /dev/null 2>&1; then RES="$RES $T=green"; else RES="$RES $T=RED"; fi
  done
  if make BUILD=b CC=gcc PY=$PYBIN oracle > /dev/null 2>&1; then RES="$RES oracle=green"; else RES="$RES oracle=RED"; fi
  echo "$1:$RES"
  cd /src
}
run "no mutation" src/kv/kv.c "head_bytes" "head_bytes"
run "write: every head stores head 0 of K" src/kv/kv.c "k + i * n_kv + h * head_dim, head_bytes" "k + i * n_kv, head_bytes"
run "write: every pass starts at position 0" src/kv/kv.c "kv->k + head0 + (size_t)pos0 * (size_t)head_dim" "kv->k + head0"
run "read: keys of every head are head 0's" src/kv/kv.h "(size_t)layer * (size_t)kv->n_head_kv + (size_t)head)" "(size_t)layer * (size_t)kv->n_head_kv)"
run "attention: every query head reads kv head 0" src/models/olmoe.c "tr_kv_keys(c->kv, c->layer, h / c->group)" "tr_kv_keys(c->kv, c->layer, 0)"
run "attention: values of the wrong layer" src/models/olmoe.c "tr_kv_values(c->kv, c->layer, h / c->group)" "tr_kv_values(c->kv, 0, h / c->group)"
}
main "$@"; exit
