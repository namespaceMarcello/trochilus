#!/bin/sh
# mutate_experts.sh — does tests/test_experts.c see a wrong expert store (src/memory/experts.c:
# LRU list, per-part offsets, eviction, duplicate rejection, aligned reads for a direct handle)?
# A test never seen red proves nothing (docs/LESSONS.md #43): each mutation is applied to a copy of
# the tree, the copy is built and test_experts is run. Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_experts.sh
#
# One line per mutation: which tests went red. "no mutation" must be green, every other line must
# be RED. ONLY=<word> runs "no mutation" and the mutations with that word in the name.
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
  # tools/.venv (Python, torch/transformers) is 700+ MiB and irrelevant to a pure-C build: copying
  # it on every mutation, from a slow bind mount, is minutes of dead time per line, not seconds.
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ && \
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ \;
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  make BUILD=b CC=gcc b/tests/test_experts > /dev/null 2>&1 || { echo "$1: does not build"; cd /src; return; }
  if ./b/tests/test_experts > /dev/null 2>&1; then echo "$1: test_experts=green"; else echo "$1: test_experts=RED"; fi
  cd /src
}
E=src/memory/experts.c
run "no mutation" $E \
  "        if (slot != -1) {
            lru_unlink(x, slot);
            lru_push_front(x, slot);
            x->stats.hits++;
        }" \
  "        if (slot != -1) {
            lru_unlink(x, slot);
            lru_push_front(x, slot);
            x->stats.hits++;
        }"
run "hit path: FIFO instead of LRU, a hit never moves to the hot end" $E \
  "        if (slot != -1) {
            lru_unlink(x, slot);
            lru_push_front(x, slot);
            x->stats.hits++;
        }" \
  "        if (slot != -1) {
            x->stats.hits++;
        }"
run "eviction: the victim is the hot end" $E \
  "    int32_t victim = x->lru_tail;" \
  "    int32_t victim = x->lru_head;"
run "eviction: a fresh unit is inserted at the cold end" $E \
  "    x->slot_of[unit] = victim;
    sl->unit = (int32_t)unit;
    lru_unlink(x, victim);
    lru_push_front(x, victim);
    x->stats.misses++;" \
  "    x->slot_of[unit] = victim;
    sl->unit = (int32_t)unit;
    x->stats.misses++;"
run "read: part offset ignores the expert index" $E \
  "        uint64_t file_off = x->part_offset[idx] + (uint64_t)x->part_bytes[idx] * (uint64_t)id;" \
  "        uint64_t file_off = x->part_offset[idx];"
run "read: parts land in the same slot position, no per-part offset" $E \
  "        int rc = x->read(x->read_ctx, base + x->part_slot_offset[idx], n, lo);" \
  "        int rc = x->read(x->read_ctx, base, n, lo);"
run "every layer uses layer 0's part sizes" $E \
  "        uint64_t file_off = x->part_offset[idx] + (uint64_t)x->part_bytes[idx] * (uint64_t)id;
        /* align == 1: lo == file_off, hi == file_off + part_bytes -- today's exact behaviour, same
         * formula, no branch. align > 1: the aligned range the read must actually cover. */
        uint64_t lo = file_off / align * align;
        uint64_t hi = (file_off + (uint64_t)x->part_bytes[idx] + align - 1) / align * align;" \
  "        uint64_t file_off = x->part_offset[idx] + (uint64_t)x->part_bytes[(size_t)p] * (uint64_t)id;
        /* align == 1: lo == file_off, hi == file_off + part_bytes -- today's exact behaviour, same
         * formula, no branch. align > 1: the aligned range the read must actually cover. */
        uint64_t lo = file_off / align * align;
        uint64_t hi = (file_off + (uint64_t)x->part_bytes[(size_t)p] + align - 1) / align * align;"
run "failure: a failed read leaves the unit mapped" $E \
  "    unsigned char *base = x->slab + (uint64_t)victim * x->stats.slot_bytes;
    uint64_t align = x->read_align;
    for (int p = 0; p < TR_EXPERT_PARTS; p++) {" \
  "    unsigned char *base = x->slab + (uint64_t)victim * x->stats.slot_bytes;
    uint64_t align = x->read_align;
    x->slot_of[unit] = victim;
    sl->unit = (int32_t)unit;
    for (int p = 0; p < TR_EXPERT_PARTS; p++) {"
run "eviction: the evicted unit stays mapped in slot_of" $E \
  "    if (sl->unit != -1) {
        x->slot_of[sl->unit] = -1;
        sl->unit = -1;
        x->stats.evictions++;
    }" \
  "    if (sl->unit != -1) {
        sl->unit = -1;
        x->stats.evictions++;
    }"
run "arguments: duplicate ids within a call are accepted" $E \
  "        if (x->seen_call[unit] == stamp) return -1; /* duplicate within this call */
        x->seen_call[unit] = stamp;" \
  "        x->seen_call[unit] = stamp;"
run "budget gate: a model with few layers can never run, even fully resident" $E \
  "    if (n_slots < min_slots && n_slots < n_units) {" \
  "    if (n_slots < min_slots) {"
# ---- Step B: aligned reads for a direct handle (lot 3) ----
run "aligned range: the start is not floored to the alignment" $E \
  "        uint64_t lo = file_off / align * align;" \
  "        uint64_t lo = file_off;"
run "aligned range: the end is not rounded up to the alignment" $E \
  "        uint64_t hi = (file_off + (uint64_t)x->part_bytes[idx] + align - 1) / align * align;" \
  "        uint64_t hi = file_off + (uint64_t)x->part_bytes[idx];"
run "part pointer ignores where its own bytes start inside the aligned read" $E \
  "    return x->slab + (uint64_t)slot * x->stats.slot_bytes + x->part_slot_offset[idx] +
           x->slot_part_shift[(size_t)slot * TR_EXPERT_PARTS + (size_t)part];" \
  "    return x->slab + (uint64_t)slot * x->stats.slot_bytes + x->part_slot_offset[idx];"
run "per-slot part offset recorded once instead of at every fill" $E \
  "        x->slot_part_shift[(size_t)victim * TR_EXPERT_PARTS + (size_t)p] = file_off - lo;" \
  "        (void)file_off; (void)lo; /* not recorded here: stays whatever it was at create (0) */"
run "read_align ignored: every read stays byte-granular" $E \
  "    uint64_t align = x->read_align;" \
  "    uint64_t align = 1;"
}
main "$@"; exit
