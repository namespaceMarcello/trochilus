#!/bin/sh
# mutate_route.sh — does tests/test_route.c see a wrong routing trace (src/models/olmoe.c,
# docs/MISURE.md domande 13-15)? A test never seen red proves nothing (docs/LEZIONI.md #43): each
# mutation is applied to a copy of the tree, the copy is built and test_route is run. Linux
# container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_route.sh
#
# One line per mutation: which tests went red. "no mutation" must be green, every other line must
# be RED. ONLY=<word> runs "no mutation" and the mutations with that word in the name.
set -e
# The body is one function, called on the last line (docs/LEZIONI.md #69).
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
  [ -z "$ONLY" ] || case "$1" in "no mutation"|*"$ONLY"*) ;; *) return 0 ;; esac
  # tools without .venv: the repo's own Python environment is 706 MiB over a Windows bind mount,
  # and the container has its own, so copying it made one mutation take minutes
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ &&
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ ';'
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  make BUILD=b CC=gcc b/tests/test_route > /dev/null 2>&1 || { echo "$1: does not build"; cd /src; return; }
  if ./b/tests/test_route > /dev/null 2>&1; then echo "$1: test_route=green"; else echo "$1: test_route=RED"; fi
  cd /src
}
M=src/models/olmoe.c
run "no mutation" $M \
  "tr_matmul(m->pool, &next->gate_inp, s->normed, rec, tr->router);" \
  "tr_matmul(m->pool, &next->gate_inp, s->normed, rec, tr->router);"
run "pred uses layer L's own gate_inp, not layer L+1's" $M \
  "tr_matmul(m->pool, &next->gate_inp, s->normed, rec, tr->router);" \
  "tr_matmul(m->pool, &m->layers[L].gate_inp, s->normed, rec, tr->router);"
run "pred_in taken from x instead of normed" $M \
  "tr_matmul(m->pool, &next->gate_inp, s->normed, rec, tr->router);" \
  "tr_matmul(m->pool, &next->gate_inp, s->x, rec, tr->router);"
run "pred_out uses layer L's own ffn_norm, not layer L+1's" $M \
  "for (int64_t i = 0; i < rec; i++) tr_rmsnorm(tr->normed + i * n_embd, next->ffn_norm, n_embd, m->rms_eps);" \
  "for (int64_t i = 0; i < rec; i++) tr_rmsnorm(tr->normed + i * n_embd, m->layers[L].ffn_norm, n_embd, m->rms_eps);"
run "pred_out taken from the FFN input instead of the output" $M \
  "memcpy(tr->normed, s->x, (size_t)rec * n_embd * sizeof(float));" \
  "memcpy(tr->normed, s->normed, (size_t)rec * n_embd * sizeof(float));"
run "chosen recorded with the wrong token index inside a batch" $M \
  "int64_t id = s->sel_id[i * n_used + slot];" \
  "int64_t id = s->sel_id[0 * n_used + slot];"
run "tokens: every token of a batch gets the id of the first" $M \
  "if (L == 0) tr->tokens[base + i] = tokens[i];" \
  "if (L == 0) tr->tokens[base + i] = tokens[0];"
run "margin: the best expert chosen, not the last" $M \
  "if (prob[id] < p_last) p_last = prob[id];" \
  "if (prob[id] > p_last || slot == 0) p_last = prob[id];"
run "margin: the best one left out is looked for among the chosen too" $M \
  "if (!tr->taken[e] && prob[e] > p_next) p_next = prob[e];" \
  "if (prob[e] > p_next) p_next = prob[e];"
run "mask: the router ignores it" $M \
  "taken[e] = off != NULL ? off[e] : 0;" \
  "taken[e] = 0;"
run "mask: layer 0's row for every layer" $M \
  "oc.off = m->expert_off != NULL ? m->expert_off + L * n_expert : NULL;" \
  "oc.off = m->expert_off != NULL ? m->expert_off : NULL;"
run "mask: a mask that leaves a layer too few experts is accepted" $M \
  "if (left < m->n_expert_used) return -1;" \
  "if (left < 0) return -1;"
run "mask: clearing it does nothing" $M \
  "        free(m->expert_off);
        m->expert_off = NULL;
        return 0;" \
  "        return 0;"
}
main "$@"; exit
