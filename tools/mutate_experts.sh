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
# $1: name, $2: file, $3: text, $4: replacement, $5: the test (default test_experts; test_base for the pages)
run() {
  [ -z "$ONLY" ] || case "$1" in "no mutation"|*"$ONLY"*) ;; *) return 0 ;; esac
  T=${5:-test_experts}
  # tools/.venv (Python, torch/transformers) is 700+ MiB and irrelevant to a pure-C build: copying
  # it on every mutation, from a slow bind mount, is minutes of dead time per line, not seconds.
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ && \
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ \;
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  make BUILD=b CC=gcc b/tests/$T > /dev/null 2>&1 || { echo "$1: does not build"; cd /src; return; }
  if ./b/tests/$T > /dev/null 2>&1; then echo "$1: $T=green"; else echo "$1: $T=RED"; fi
  cd /src
}
E=src/memory/experts.c
run "no mutation" $E \
  "            lru_unlink(x, slot);
            lru_push_front(x, slot);
            x->stats.hits++;" \
  "            lru_unlink(x, slot);
            lru_push_front(x, slot);
            x->stats.hits++;"
run "hit path: FIFO instead of LRU, a hit never moves to the hot end" $E \
  "            lru_unlink(x, slot);
            lru_push_front(x, slot);
            x->stats.hits++;" \
  "            x->stats.hits++;"
run "eviction: the victim is the hot end" $E \
  "    for (int32_t s = x->lru_tail; s != -1; s = x->slots[s].prev) {" \
  "    for (int32_t s = x->lru_head; s != -1; s = x->slots[s].next) {"
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
  "        int rc = read(ctx, base + x->part_slot_offset[idx], n, lo);" \
  "        int rc = read(ctx, base, n, lo);"
run "every layer uses layer 0's part sizes" $E \
  "        uint64_t file_off = x->part_offset[idx] + (uint64_t)x->part_bytes[idx] * (uint64_t)id;
        /* align == 1: lo == file_off, hi == file_off + part_bytes -- the exact bytes. align > 1:
         * the aligned range the read must actually cover. */
        uint64_t lo = file_off / align * align;
        uint64_t hi = (file_off + (uint64_t)x->part_bytes[idx] + align - 1) / align * align;" \
  "        uint64_t file_off = x->part_offset[idx] + (uint64_t)x->part_bytes[(size_t)p] * (uint64_t)id;
        uint64_t lo = file_off / align * align;
        uint64_t hi = (file_off + (uint64_t)x->part_bytes[(size_t)p] + align - 1) / align * align;"
run "failure: a failed read leaves the unit mapped" $E \
  "    if (rc != 0) {
        lru_unlink(x, victim);
        lru_push_back(x, victim);
        return -1;
    }" \
  "    if (rc != 0) {
        x->slot_of[unit] = victim;
        sl->unit = (int32_t)unit;
        lru_unlink(x, victim);
        lru_push_back(x, victim);
        return -1;
    }"
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
  "        x->slot_part_shift[(size_t)victim * TR_EXPERT_PARTS + (size_t)p] = shift[p];" \
  "        (void)shift; /* not recorded here: stays whatever it was at create (0) */"
run "read_align ignored: every read stays byte-granular" $E \
  "    uint64_t align = x->read_align;" \
  "    uint64_t align = 1;"
# ---- runs: a layer's consecutive experts in one request a part (docs/MEASUREMENTS.md §The disk at its limit) ----
run "run: the page two neighbours share is not copied" $E \
  "                memcpy((unsigned char *)iov[e].base + iov[e].n, iov[e + 1].base, (size_t)(hi - lo - iov[e].n));" \
  "                (void)hi;"
run "run: each piece its unit's whole aligned range, overlapping the next" $E \
  "            uint64_t end = e + 1 < k ? (first + pb * (uint64_t)(e + 1)) / align * align /* the next one's lo */" \
  "            uint64_t end = e + 1 < k ? (first + pb * (uint64_t)(e + 1) + align - 1) / align * align"
run "run: never split, one request carries the whole run" $E \
  "    return (left + n_req - 1) / n_req;" \
  "    return left;"
run "run: the I/O thread leaves the last queued expert out" $E \
  "        while (left <= x->q_len && left < x->n_expert && (job.cap == 0 || left < job.cap)) {" \
  "        while (left < x->q_len && left < x->n_expert && (job.cap == 0 || left < job.cap)) {"
run "run: acquire joins a missing id that is not the next one" $E \
  "        while (i + left < n && ids[i + left] == id + left && x->slot_of[unit + left] == -1) left++;" \
  "        while (i + left < n && x->slot_of[layer * x->n_expert + ids[i + left]] == -1) left++;"
run "run: the units' shifts not recorded" $E \
  "            x->slot_part_shift[(size_t)slots[e] * TR_EXPERT_PARTS + (size_t)p] = x->run_shift[0][e][p];" \
  "            (void)p;"
run "touch (runs' test): a page's first byte overwritten" $E \
  "    for (int64_t pg = begin; pg < end; pg++) base[pg * 4096] = base[pg * 4096];" \
  "    for (int64_t pg = begin; pg < end; pg++) base[pg * 4096] = 0;"
run "run in flight: only the first part's request handed over" $E \
  "    int rc = readv(ctx, req, TR_EXPERT_PARTS, x->scratch[w]);" \
  "    int rc = readv(ctx, req, 1, x->scratch[w]);"
run "run in flight: the three parts' pieces in one list, each overwriting the last" $E \
  "        tr_iov *iov = x->iov[w] + (size_t)p * (size_t)x->n_expert;" \
  "        tr_iov *iov = x->iov[w];"
run "run in flight: a run counted as one request" $E \
  "    *requests += TR_EXPERT_PARTS;" \
  "    (*requests)++;"
run "hot: a run's picks not stamped (a slot taken twice)" $E \
  "        sl->pick = x->call_seq; /* emptied below: the next pick must not take it again (hot_victim) */" \
  "        (void)sl;"
run "hot: every slot taken at call 0 (the resident load fails)" $E \
  "        x->slots[s].pick = UINT64_MAX; /* never taken: tr_experts_load_all runs at call_seq 0 */" \
  "        x->slots[s].pick = 0;"
run "hot: never halved" $E \
  "            for (int64_t u = 0; u < x->stats.n_units; u++) x->hot[u] >>= 1;" \
  "            (void)0;"
run "hot: the most recent among equals" $E \
  "        if (best == -1 || x->hot[u] < best_hot) {" \
  "        if (best == -1 || x->hot[u] <= best_hot) {"
run "hot: a hit not counted" $E \
  "            x->hot[layer * x->n_expert + ids[i]] += a;" \
  "            if (x->slot_of[layer * x->n_expert + ids[i]] == -1) x->hot[layer * x->n_expert + ids[i]] += a;"
run "hot: a prompt's routings ignored (ds4's +1)" $E \
  "            uint32_t a = counts != NULL ? hot_add(counts[i], n_tok) : 1u;" \
  "            uint32_t a = 1u;"
run "hot: a pass's routings unscaled" $E \
  "    if (n_tok <= TR_EXPERTS_HOT_PROMPT) return (uint32_t)(count < n_tok ? count : n_tok);" \
  "    if (n_tok > 0) return (uint32_t)count;"
run "hot: the scaled routings rounded down" $E \
  "    int64_t a = (count * TR_EXPERTS_HOT_PROMPT + n_tok / 2) / n_tok;" \
  "    int64_t a = (count * TR_EXPERTS_HOT_PROMPT) / n_tok;"
run "hot: a scaled 0 left 0" $E \
  "    return a < 1 ? 1u : (uint32_t)(a < TR_EXPERTS_HOT_PROMPT ? a : TR_EXPERTS_HOT_PROMPT);" \
  "    return a < 1 ? 0u : (uint32_t)(a < TR_EXPERTS_HOT_PROMPT ? a : TR_EXPERTS_HOT_PROMPT);"
run "hot: a unit of this call evicted" $E \
  "        if (x->seen_call[u] == x->call_seq) continue;" \
  "        (void)0;"
run "hot: the LRU's victim" $E \
  "    if (x->hot != NULL) return hot_victim(x, skip_a, skip_b);" \
  "    if (x->hot != NULL && x->hot == NULL) return hot_victim(x, skip_a, skip_b);"
run "emulated disk: no wait" $E \
  "    tr_wait_until((double)end / 1e9);" \
  "    (void)end;"
run "emulated disk: a clock a request, not one disk" $E \
  "        end = (free_ns > t0_ns ? free_ns : t0_ns) + cost;" \
  "        end = t0_ns + cost;"
run "emulated disk: the runs' door not charged" $E \
  "    if (rc == 0) disk_emulate(x, t0, total);" \
  "    if (rc == 0) (void)total;"
run "emulated disk: a coarse sleep (whole 16 ms ticks)" $E \
  "    tr_wait_until((double)end / 1e9);" \
  "    { double w = (double)end / 1e9; while (tr_time_sec() < w) tr_wait_until(tr_time_sec() + 0.016); }"
# tr_experts_set_slots (test_experts slots): the slots a session's KV grows into
run "set_slots: the warmest evicted first (HOT)" $E \
  "        if (best == -1 || x->hot[u] < x->hot[x->slots[best].unit]) best = s;" \
  "        if (best == -1 || x->hot[u] > x->hot[x->slots[best].unit]) best = s;"
run "set_slots: a moved unit's bytes not copied" $E \
  "        memcpy(x->slab + (uint64_t)v * sb, x->slab + (uint64_t)s * sb, (size_t)sb);" \
  "        (void)sb;"
run "set_slots: a moved unit's shifts not copied" $E \
  "        memcpy(&x->slot_part_shift[(size_t)v * TR_EXPERT_PARTS], &x->slot_part_shift[(size_t)s * TR_EXPERT_PARTS]," \
  "        memcpy(&x->slot_part_shift[(size_t)v * TR_EXPERT_PARTS], &x->slot_part_shift[(size_t)v * TR_EXPERT_PARTS],"
run "set_slots: the index left on the slot given back" $E \
  "        x->slot_of[u] = v;" \
  "        x->slot_of[u] = (int32_t)s;"
run "set_slots: a moved unit's recency lost (to the cold end)" $E \
  "        lru_take_place(x, v, (int32_t)s);" \
  "        lru_unlink(x, (int32_t)s), lru_unlink(x, v), lru_push_back(x, v);"
run "set_slots: reads ahead not taken in first" $E \
  "    /* a read ahead still in flight lands first (a failed one stays for the next tr_experts_prefetch_wait) */" \
  "    if (0)"
run "set_slots: under the minimum" $E \
  "    if (n < least) n = least;" \
  "    if (n < 1) n = 1;"
run "set_slots: a slot taken back left out of the list" $E \
  "            lru_push_back(x, (int32_t)s); /* free, at the cold end */" \
  "            (void)s;"
# tr_experts_acquire_async / _take (test_experts arrival): a call's misses read by the I/O thread
run "arrival: a queued unit's victim by the read ahead's rule" $E \
  "            int32_t v = coldest_victim(x, -1, -1);" \
  "            int32_t v = coldest_victim(x, layer, -1);"
run "arrival: a reserved slot left cold" $E \
  "    lru_push_front(x, v);" \
  "    lru_push_back(x, v);"
run "arrival: a take waits for every unit" $E \
  "        if (i >= at_least) {" \
  "        if (0) {"
run "arrival: counted as read ahead" $E \
  "    if (job.demand) x->stats.arrived++;" \
  "    if (0) x->stats.arrived++;"
run "arrival: a failed call's unit reported again by the wait" $E \
  "        if (!job.demand) x->prefetch_failed = 1;" \
  "        x->prefetch_failed = 1;"
run "arrival: the misses read in the call" $E \
  "    if (async && x->io != NULL) {" \
  "    if (0) {"
# the handoffs' counters (test_experts arrival's order, disk): LESSONS #318
run "handoff: a blocked take not counted" $E \
  "        x->stats.take_waits++;" \
  "        (void)0;"
run "handoff: the I/O thread's gap never counted" $E \
  "        if (!waited && prev_done > 0) atomic_fetch_add" \
  "        if (0) atomic_fetch_add"
run "handoff: the emulated disk's lateness never counted" $E \
  "    if (late > 0) atomic_fetch_add" \
  "    if (0) atomic_fetch_add"
run "handoff: a take's wake from the wait's start" $E \
  "        x->stats.take_wake_sec += t1 - x->jobs[s].done_at;" \
  "        x->stats.take_wake_sec += t1 - t0;"
run "pages: release keeps the pages (test_base)" src/base/platform.c \
  "    return madvise((void *)lo, hi - lo, MADV_DONTNEED) == 0 ? 0 : -1;" \
  "    return 0;" test_base
}
main "$@"; exit
