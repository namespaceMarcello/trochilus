#!/bin/sh
# mutate_stream.sh — does tests/test_stream.c see a wrong streaming engine (src/models/olmoe.c:
# expert pointer refresh, load-time offsets, acquire's own selection, failure handling)? A test
# never seen red proves nothing (docs/LESSONS.md #43): each mutation is applied to a copy of the
# tree, the copy is built and test_stream is run. Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_stream.sh
#
# One line per mutation: which tests went red (a crash counts as RED). "no mutation" must be
# green, every other line must be RED. ONLY=<word> runs "no mutation" and the mutations with that
# word in the name.
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
  [ -z "$ONLY" ] || case "$1" in "no mutation"|*"$ONLY"*) ;; *) return 0 ;; esac
  # tools without .venv: the repo's own Python environment is 706 MiB over a Windows bind mount,
  # and the container has its own, so copying it made one mutation take minutes
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ &&
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ ';'
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  make BUILD=b CC=gcc b/tests/test_stream > /dev/null 2>&1 || { echo "$1: does not build"; cd /src; return; }
  if ./b/tests/test_stream > /dev/null 2>&1; then echo "$1: test_stream=green"; else echo "$1: test_stream=RED"; fi
  cd /src
}
O=src/models/olmoe.c
run "no mutation" $O \
  "    for (int64_t e = 0; e < n_expert; e++) {
        layer->exps[e].data = tr_experts_part(m->experts, L, e, 0);
        layer->exps[n_expert + e].data = tr_experts_part(m->experts, L, e, 1);
        layer->exps[2 * n_expert + e].data = tr_experts_part(m->experts, L, e, 2);
    }" \
  "    for (int64_t e = 0; e < n_expert; e++) {
        layer->exps[e].data = tr_experts_part(m->experts, L, e, 0);
        layer->exps[n_expert + e].data = tr_experts_part(m->experts, L, e, 1);
        layer->exps[2 * n_expert + e].data = tr_experts_part(m->experts, L, e, 2);
    }"
run "the pointer refresh after acquire is skipped for up_exps" $O \
  "    for (int64_t e = 0; e < n_expert; e++) {
        layer->exps[e].data = tr_experts_part(m->experts, L, e, 0);
        layer->exps[n_expert + e].data = tr_experts_part(m->experts, L, e, 1);
        layer->exps[2 * n_expert + e].data = tr_experts_part(m->experts, L, e, 2);
    }" \
  "    for (int64_t e = 0; e < n_expert; e++) {
        layer->exps[e].data = tr_experts_part(m->experts, L, e, 0);
        layer->exps[2 * n_expert + e].data = tr_experts_part(m->experts, L, e, 2);
    }"
run "gate and up offsets swapped at load" $O \
  "        part_offset_tbl[L * TR_EXPERT_PARTS + 0] = t->offset;
        all_experts_bytes += t->n_bytes;

        snprintf(name, sizeof name, \"blk.%\" PRId64 \".ffn_up_exps.weight\", L);
        t = find_experts(g, name, m->n_embd, m->n_ff, m->n_expert, err, err_len);
        if (t == NULL) goto fail;
        part_type_tbl[L * TR_EXPERT_PARTS + 1] = t->type;
        part_bytes_tbl[L * TR_EXPERT_PARTS + 1] = (size_t)m->n_ff * tr_row_bytes(t->type, m->n_embd);
        part_offset_tbl[L * TR_EXPERT_PARTS + 1] = t->offset;" \
  "        part_offset_tbl[L * TR_EXPERT_PARTS + 1] = t->offset;
        all_experts_bytes += t->n_bytes;

        snprintf(name, sizeof name, \"blk.%\" PRId64 \".ffn_up_exps.weight\", L);
        t = find_experts(g, name, m->n_embd, m->n_ff, m->n_expert, err, err_len);
        if (t == NULL) goto fail;
        part_type_tbl[L * TR_EXPERT_PARTS + 1] = t->type;
        part_bytes_tbl[L * TR_EXPERT_PARTS + 1] = (size_t)m->n_ff * tr_row_bytes(t->type, m->n_embd);
        part_offset_tbl[L * TR_EXPERT_PARTS + 0] = t->offset;"
run "acquire only asks for the first non-empty expert" $O \
  "    int64_t n_ids = 0;
    for (int64_t e = 0; e < n_expert; e++)
        if (s->offsets[e + 1] > s->offsets[e]) s->acquire_ids[n_ids++] = e;" \
  "    int64_t n_ids = 0;
    for (int64_t e = 0; e < n_expert; e++)
        if (s->offsets[e + 1] > s->offsets[e]) { s->acquire_ids[n_ids++] = e; break; }"
run "pos is not restored after a failed pass" $O \
  "        if (forward_pass(m, s, tokens + off, len, off + len == n ? n_logits : 0) != 0) {
            s->pos = pos0;
            return -1;
        }" \
  "        if (forward_pass(m, s, tokens + off, len, off + len == n ? n_logits : 0) != 0) {
            return -1;
        }"
run "forward_pass ignores acquire's return value" $O \
  "        if (olmoe_refresh_experts(m, s, L) != 0) return -1;" \
  "        olmoe_refresh_experts(m, s, L);"
}
main "$@"; exit
