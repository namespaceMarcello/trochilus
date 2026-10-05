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
  "            s->acquire_ids[n_ids++] = e;
        }" \
  "            s->acquire_ids[n_ids++] = e;
            break;
        }"
run "pos is not restored after a failed pass" $O \
  "        if (forward_pass(m, s, tokens + off, len, off + len == n ? n_logits : 0) != 0) {
            s->pos = pos0;
            return -1;
        }" \
  "        if (forward_pass(m, s, tokens + off, len, off + len == n ? n_logits : 0) != 0) {
            return -1;
        }"
run "forward_pass ignores acquire's return value" $O \
  "    if (olmoe_refresh_experts(m, s, L, n_tok) != 0) return -1;" \
  "    olmoe_refresh_experts(m, s, L, n_tok);"
run "plan: the session's positions ignored (the default context's always)" $O \
  "        int64_t ctx_plan = session_ctx(m, plan_ctx);" \
  "        int64_t ctx_plan = session_ctx(m, 0);"
run "eviction: the LRU by default" $O \
  "        ecfg.evict = evict != NULL && strcmp(evict, \"lru\") == 0 ? TR_EXPERTS_EVICT_LRU : TR_EXPERTS_EVICT_HOT;" \
  "        ecfg.evict = evict != NULL && strcmp(evict, \"hot\") == 0 ? TR_EXPERTS_EVICT_HOT : TR_EXPERTS_EVICT_LRU;"
run "session guard: the KV left out of the session's bytes" $O \
  "    return kv_bytes + scratch_bytes + x_all_bytes;" \
  "    return scratch_bytes + x_all_bytes + 0 * kv_bytes;"
# the KV grown from the store's room (test_mem_available kv_room)
run "plan: the KV set aside again" $O \
  "                                     tr_kv_bytes(m->n_layers, m->n_head_kv, m->head_dim, ctx_plan);" \
  "                                     0 * tr_kv_bytes(m->n_layers, m->n_head_kv, m->head_dim, ctx_plan);"
run "kv room: a partial plan shares none" $O \
  "            if (budget < resident_bytes) m->kv_room = budget; /* partial: every byte the plan left */" \
  "            if (0) m->kv_room = budget;"
run "kv_hold: never called before a pass" $O \
  "    kv_hold(m, s, s->pos + n);" \
  "    if (0) kv_hold(m, s, s->pos + n);"
run "store_fit: the KV held not taken off the room" $O \
  "    uint64_t room = m->kv_held < m->kv_room ? m->kv_room - m->kv_held : 0;" \
  "    uint64_t room = m->kv_room;"
run "session free: its KV not given back to the room" $O \
  "    s->m->kv_held -= s->kv_held; /* its pages back to the system: the store takes their slots again */" \
  "    (void)0;"
run "session free: the slots not taken again" $O \
  "    store_fit(s->m);" \
  "    (void)s->m;"
run "session guard: the room's check off" $O \
  "        if (m->kv_held + kv_full + least > m->kv_room) {" \
  "        if (0) {"
# the arrival order (test_stream arrival): a layer's misses read by the I/O thread while its present experts compute
run "arrival: the misses read on the calling thread" $O \
  "    int64_t rc = s->ab_sync ? tr_experts_acquire_counts" \
  "    int64_t rc = 1 ? tr_experts_acquire_counts"
run "arrival: a row left in its id order's place" $O \
  "        s->place[j] = offs[grp_of[e]] + s->place[j] - s->offsets[e];" \
  "        (void)grp_of;"
run "arrival: the late groups first" $O \
  "            if (s->offsets[e + 1] > s->offsets[e] && (layer->gate_exps[e].data == NULL) == late) {" \
  "            if (s->offsets[e + 1] > s->offsets[e] && (layer->gate_exps[e].data == NULL) != late) {"
run "arrival: a wave's offsets from the layer's start" $O \
  "        for (int64_t i = 0; i <= k; i++) wo[i] = offs[g + i] - offs[g];" \
  "        for (int64_t i = 0; i <= k; i++) wo[i] = offs[g + i];"
run "arrival: a wave's tokens through the first rows' map" $O \
  "            !tr_matmul_q4x_prepared_n(pool, wgu, 2, offsets, n_groups, s->xq_tok, s->xmap + row0, ygu, &s->pm)) {" \
  "            !tr_matmul_q4x_prepared_n(pool, wgu, 2, offsets, n_groups, s->xq_tok, s->xmap, ygu, &s->pm)) {"
run "arrival: a wave's down reads another wave's activations" $O \
  "    if (act_road) tr_matmul_q4x_prepared(pool, w + 2 * stride, offsets, n_groups, xq, NULL, h3, &s->pm);" \
  "    if (act_road) tr_matmul_q4x_prepared(pool, w + 2 * stride, offsets, n_groups, xq + (row0 > 0 ? 64 : 0), NULL, h3, &s->pm);"
run "arrival: a late unit's weights not refreshed in its group" $O \
  "            s->wave_w[p * n_expert + i].data = d;" \
  "            (void)d;"
run "arrival: a wave's floats from the first rows" $O \
  "        tr_matmul_grouped_s(pool, w, offsets, n_groups, s->xg + row0 * n_embd, h1, &s->pm);" \
  "        tr_matmul_grouped_s(pool, w, offsets, n_groups, s->xg, h1, &s->pm);"
}
main "$@"; exit
