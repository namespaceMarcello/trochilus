/* test_experts.c — the expert store (src/memory/experts.h): which (layer, expert) units are in
 * RAM under a budget, and the door to disk for the rest (docs/ARCHITECTURE.md "Esecuzione",
 * Esperti (M1)).
 *
 * A fake reader replaces the GGUF file: every byte of every part of every unit is a pure
 * function of its absolute file offset (pattern_byte), so a part read at the wrong offset, or
 * landing in the wrong slot, shows up as a mismatch. It counts its own calls and bytes, and can
 * be told to fail the k-th call, for the failure-injection tests.
 *
 * Each test says which branch it exercises and counts that it was actually taken (CLAUDE.md,
 * LESSONS #43 #50 #54 #78):
 *   arithmetic   tr_experts_slot_bytes / tr_experts_min_slots, and tr_experts_create's budget
 *                gate: one byte short of the minimum refused (message names the MiB), the exact
 *                minimum accepted, non-positive shapes refused
 *   contents     every part pointer exactly the file's bytes for that (layer, expert, part) --
 *                64-byte aligned when buffered, unconstrained when direct; never another unit's
 *                bytes; one layer has different part sizes than the rest (Step A: GGUF files mix
 *                quantization types across layers), and a slot it once held is later reused by a
 *                same-sized layer
 *   lru          a hand-scripted sequence on the minimum store: hits/misses/evictions checked
 *                after every step, including a touch that saves a unit from the next eviction
 *   full_layer   a whole layer at once on the minimum store, twice: the call's own units are
 *                never among its own victims
 *   resident     budget >= every unit: resident, load_all reads each unit once and nothing else
 *                is ever read again; load_all on a non-resident store fails
 *   failure      the reader fails on the 1st, 2nd and 3rd part of a missing unit in turn: the
 *                unit stays absent, the victim's old unit is gone, units handled earlier are
 *                untouched, a retry succeeds; load_all with a failing reader fails too
 *   arg_errors   bad layer, bad id, n == 0, n too large, a duplicate id: -1 and the store
 *                unchanged
 *   o1_evidence  a differential test: a long pseudo-random sequence of acquires against a plain
 *                array-based reference LRU (O(n), fine there); same presence and counters after
 *                every call, fixed seed
 *   few_layers   a shape with n_layers * n_expert (n_units) below n_expert + n_used (min_slots):
 *                impossible to run partially, but a resident budget (n_slots == n_units) is
 *                still accepted -- the margin only protects a store that can actually evict; one
 *                slot short of that (n_slots < n_units too) still refuses
 *   align_reuse  read_align > 1 only: a slot reused by two experts whose file offsets fall on
 *                different residues modulo the alignment both come back byte-exact -- the
 *                per-slot, per-part shift (Step B) is really taken at every fill
 *   prefetch     the I/O thread (tr_experts_prefetch), with a reader both threads call at once
 *                (atomic counters, a delay, a file offset that fails): start refused on a resident
 *                store and one slot short of 2 * n_expert + n_used, accepted at exactly that;
 *                (basic) a layer read ahead is invisible until taken in, then exact, counted as
 *                prefetched and misses, and the held layer untouched; (victims) never the kept
 *                layer, never a slot in flight, and nothing queued when no such victim is left;
 *                (wait) an acquire of units still in flight waits for them; (concurrent) demand
 *                reads beside reads ahead, no slot overwritten mid-read; (fail) a failed read
 *                ahead leaves only that unit absent, reported once by the wait, read on demand
 *                afterwards, and an acquire asking for it is -1; (free) freed with reads in flight
 *   hot          TR_EXPERTS_EVICT_HOT (ds4's policy): the store against a plain reference after every
 *                call of a random sequence (halvings included; a third of the calls a prompt's pass
 *                through tr_experts_acquire_counts, some of more tokens than TR_EXPERTS_HOT_PROMPT),
 *                apart from the LRU's; a run's several victims in one call, every unit exact in a
 *                slot of its own; and (prompt) the unit a pass's tokens chose most kept into the decode
 *   slots        tr_experts_set_slots: 14 -> 10 slots under HOT drops the four coolest and moves the four
 *                units above 10 below it, bytes exact; under the LRU the four least recent go and a moved
 *                unit keeps its recency (the next miss evicts the least recent left, not it); clamped to
 *                the minimum and to the slots made, the same count a no-op; taken back free and filled;
 *                reads ahead in flight taken in before any slot moves
 *   disk         the emulated disk (cfg.disk_bytes_per_sec): a layer on demand and a load in runs take
 *                at least their bytes at the rate, and not twice that; the I/O thread and the calling
 *                thread reading at once share one disk (their bytes one after the other)
 *   arrival      tr_experts_acquire_async / _take (a call's misses read by the I/O thread while the caller
 *                computes the present units): (twins) against tr_experts_acquire_counts on the same random
 *                calls, LRU and HOT, at the I/O thread's smallest store and two slots more, runs on: after
 *                every call the same units, hits, misses, evictions, bytes and requests, a queued unit
 *                unreadable until taken, takes of random lengths stopping where a read has not landed;
 *                (order) at 0.1 s a read nothing landed at once, the first waited for and the second not,
 *                counted arrived; (fail) a failed read fails its take, only that unit absent, nothing
 *                reported again by the wait; (sync) no I/O thread: the misses read in the call
 *
 * Every test above align_reuse runs twice, at read_align 1 (buffered, today's layout) and
 * TR_FILE_DIRECT_ALIGN (direct, Step B): same contents, same LRU behaviour, same counters either
 * way. The fake reader also checks that every offset and length it is asked for is itself a
 * multiple of the alignment in effect (fake_file.bad_align), which a store that ignored
 * cfg.read_align could not fake merely by returning the right bytes.
 *
 * Seen red: tools/mutate_experts.sh.
 */
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "test.h"
#include "../src/base/platform.h" /* TR_FILE_DIRECT_ALIGN */
#include "../src/memory/experts.h"

/* ---- the fake file: a byte at absolute offset o is pattern_byte(o), nothing is ever stored --- */

static uint64_t pattern_byte_hash(uint64_t o) {
    uint64_t h = o * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 32;
    return h;
}
static unsigned char pattern_byte(uint64_t o) {
    return (unsigned char)pattern_byte_hash(o);
}

typedef struct {
    int64_t n_calls;
    uint64_t bytes;
    int64_t fail_at;    /* 1-based call number to fail on, 0 = never */
    uint64_t align_check; /* > 1: every call's offset and n must be its multiple (Step B) */
    int64_t bad_align;    /* calls that were not (proves read_align actually reached the reader) */
} fake_file;

static int fake_read(void *ctx, void *buf, size_t n, uint64_t offset) {
    fake_file *f = (fake_file *)ctx;
    f->n_calls++;
    if (f->fail_at == f->n_calls) return -1;
    if (f->align_check > 1 && (offset % f->align_check != 0 || n % f->align_check != 0)) f->bad_align++;
    unsigned char *dst = (unsigned char *)buf;
    for (size_t i = 0; i < n; i++) dst[i] = pattern_byte(offset + i);
    f->bytes += n;
    return 0;
}

/* A fresh fake_file that checks every read it sees lands on `align` (1: no check, buffered). */
static fake_file mk_fake(uint64_t align) {
    fake_file f;
    memset(&f, 0, sizeof f);
    f.align_check = align;
    return f;
}

/* ---- small shape: 3 layers, 6 experts, 2 used, unaligned offsets. Layer 1 has different part
 * sizes than layers 0 and 2 (Step A: a GGUF file may quantize layers differently), and is the
 * layer every test below exercises first, so a store that used layer 0's sizes everywhere would
 * show up immediately as wrong content or a crash (tools/mutate_experts.sh). ---- */

enum { N_LAYERS = 3, N_EXPERT = 6, N_USED = 2 };
static const size_t PART_BYTES_TABLE[N_LAYERS][TR_EXPERT_PARTS] = {
    {100, 60, 37},
    {150, 60, 37}, /* layer 1: part 0 rounds to 192 instead of 128 -- the biggest layer sets slot_bytes */
    {100, 60, 37},
};
enum { SLOT_BYTES = 320 }; /* max(128+64+64, 192+64+64, 128+64+64) */
static uint64_t part_offset_table[N_LAYERS * TR_EXPERT_PARTS];
static const int64_t ALL_EXPERTS[N_EXPERT] = {0, 1, 2, 3, 4, 5};

static void build_part_offsets(void) {
    uint64_t off = 137; /* arbitrary, not 0, not 64-aligned */
    for (int64_t layer = 0; layer < N_LAYERS; layer++)
        for (int p = 0; p < TR_EXPERT_PARTS; p++) {
            part_offset_table[layer * TR_EXPERT_PARTS + p] = off;
            off += PART_BYTES_TABLE[layer][p] * (uint64_t)N_EXPERT + 13; /* gap: blocks never touch */
        }
}

static tr_experts_config mk_cfg(uint64_t budget, tr_experts_read_fn read, void *ctx, uint64_t align) {
    tr_experts_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_layers = N_LAYERS;
    cfg.n_expert = N_EXPERT;
    cfg.n_used = N_USED;
    cfg.part_bytes = &PART_BYTES_TABLE[0][0];
    cfg.part_offset = part_offset_table;
    cfg.budget_bytes = budget;
    cfg.read = read;
    cfg.read_ctx = ctx;
    cfg.read_align = align;
    return cfg;
}

/* align == 1 (buffered): a part pointer is always 64-byte aligned (today's layout). align > 1
 * (direct): a part starts wherever its own bytes happen to fall inside its aligned read (experts.h
 * tr_experts_part), which carries no alignment guarantee of its own, so that check does not apply. */
static int part_ok(const tr_experts *x, int64_t layer, int64_t expert, int p, uint64_t align) {
    const unsigned char *got = (const unsigned char *)tr_experts_part(x, layer, expert, p);
    if (got == NULL) return 0;
    if (align == 1 && ((uintptr_t)got) % 64 != 0) return 0;
    size_t part_bytes = PART_BYTES_TABLE[layer][p];
    uint64_t base = part_offset_table[layer * TR_EXPERT_PARTS + p] + part_bytes * (uint64_t)expert;
    for (size_t i = 0; i < part_bytes; i++)
        if (got[i] != pattern_byte(base + i)) return 0;
    return 1;
}
static int unit_ok(const tr_experts *x, int64_t layer, int64_t expert, uint64_t align) {
    for (int p = 0; p < TR_EXPERT_PARTS; p++)
        if (!part_ok(x, layer, expert, p, align)) return 0;
    return 1;
}
static int unit_absent(const tr_experts *x, int64_t layer, int64_t expert) {
    for (int p = 0; p < TR_EXPERT_PARTS; p++)
        if (tr_experts_part(x, layer, expert, p) != NULL) return 0;
    return 1;
}

enum { MIN_SLOTS = N_EXPERT + N_USED }; /* the minimum store: no room to spare */

/* Independent of experts.c: what tr_experts_slot_bytes ought to return, part by part (experts.h). */
static uint64_t expected_area(size_t part_bytes, uint64_t align) {
    uint64_t a = align > 64 ? align : 64;
    uint64_t area = ((uint64_t)part_bytes + a - 1) / a * a;
    if (align > 1) area += align;
    return area;
}
static uint64_t expected_slot_bytes(uint64_t align) {
    uint64_t max_total = 0;
    for (int64_t layer = 0; layer < N_LAYERS; layer++) {
        uint64_t total = 0;
        for (int p = 0; p < TR_EXPERT_PARTS; p++) total += expected_area(PART_BYTES_TABLE[layer][p], align);
        if (total > max_total) max_total = total;
    }
    return max_total;
}

/* ---- arithmetic: sizes, and tr_experts_create's budget gate ---- */

static void test_arithmetic(uint64_t align) {
    int checked = 0;

    /* layer 1 is the biggest (150 -> 192, plus 64 + 64 buffered): every slot is sized for it, not
     * layer 0. expected_slot_bytes is an independent formula (Step B), not a copy of the code
     * under test; SLOT_BYTES pins the known buffered number so that formula cannot drift unnoticed. */
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    TR_CHECK_EQ_INT(slot_bytes, expected_slot_bytes(align));
    if (align == 1) TR_CHECK_EQ_INT(slot_bytes, SLOT_BYTES);
    TR_CHECK_EQ_INT(tr_experts_min_slots(N_EXPERT, N_USED), MIN_SLOTS);
    checked++;

    char err[256];
    fake_file f = mk_fake(align);
    tr_experts_config cfg = mk_cfg((uint64_t)MIN_SLOTS * slot_bytes - 1, fake_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x == NULL);
    TR_CHECK(strstr(err, "MiB") != NULL);
    checked++;

    cfg.budget_bytes = (uint64_t)MIN_SLOTS * slot_bytes; /* the exact minimum */
    x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    tr_experts_free(x);
    checked++;

    tr_experts_config bad;
    bad = cfg;
    bad.n_layers = 0;
    TR_CHECK(tr_experts_create(&bad, err, sizeof err) == NULL);
    bad = cfg;
    bad.n_expert = 0;
    TR_CHECK(tr_experts_create(&bad, err, sizeof err) == NULL);
    bad = cfg;
    bad.n_used = 0;
    TR_CHECK(tr_experts_create(&bad, err, sizeof err) == NULL);
    bad = cfg;
    size_t bad_part_bytes[N_LAYERS * TR_EXPERT_PARTS];
    memcpy(bad_part_bytes, &PART_BYTES_TABLE[0][0], sizeof bad_part_bytes);
    bad_part_bytes[1] = 0; /* layer 0, part 1: zero bytes is refused wherever it happens */
    bad.part_bytes = bad_part_bytes;
    TR_CHECK(tr_experts_create(&bad, err, sizeof err) == NULL);
    checked++;

    /* every refusal says why, and one without an error buffer refuses all the same (mutations of
     * the err != NULL guards and of the shape checks lived: tools/mutate_auto.py, 2026-09-22) */
    bad = cfg;
    bad.n_used = 0;
    err[0] = '\0';
    TR_CHECK(tr_experts_create(&bad, err, sizeof err) == NULL && strstr(err, "positive") != NULL);
    bad.part_bytes = bad_part_bytes;
    bad.n_used = cfg.n_used;
    err[0] = '\0';
    TR_CHECK(tr_experts_create(&bad, err, sizeof err) == NULL && strstr(err, "zero bytes") != NULL);
    TR_CHECK(tr_experts_create(&bad, NULL, 0) == NULL);
    bad = cfg;
    bad.budget_bytes = 0;
    TR_CHECK(tr_experts_create(&bad, NULL, 0) == NULL);
    tr_experts_free(NULL);
    checked++;

    /* a part of exactly 64 bytes takes 64 in a buffered slot, not 128 (round up, not past) */
    size_t exact[TR_EXPERT_PARTS] = {64, 128, 192};
    TR_CHECK_EQ_INT(tr_experts_slot_bytes(exact, 1, 1), 64 + 128 + 192);
    TR_CHECK_EQ_INT(tr_experts_slot_bytes(exact, 1, 4096), 3 * (4096 + 4096));
    checked++;

    /* the store says whether it reads direct */
    x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x != NULL) {
        tr_experts_stats st;
        tr_experts_get_stats(x, &st);
        TR_CHECK_EQ_INT(st.direct, align > 1);
        tr_experts_free(x);
    }
    checked++;

    TR_CHECK_EQ_INT(f.bad_align, 0);
    TR_CHECK(checked > 0);
}

/* ---- contents: pointers 64-aligned, exact bytes, never another unit's ---- */

static void test_contents(uint64_t align) {
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    char err[256];
    fake_file f = mk_fake(align);
    tr_experts_config cfg = mk_cfg((uint64_t)MIN_SLOTS * slot_bytes, fake_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return;

    int64_t ids[3] = {0, 2, 5};
    TR_CHECK(tr_experts_acquire(x, 1, ids, 3) == 0); /* layer 1 has different part sizes (Step A) */

    int checked = 0;
    for (int i = 0; i < 3; i++) {
        TR_CHECK(unit_ok(x, 1, ids[i], align));
        checked++;
    }
    /* neighbours never loaded, and never confused with a loaded part */
    TR_CHECK(unit_absent(x, 1, 1));
    TR_CHECK(unit_absent(x, 1, 3));
    TR_CHECK(unit_absent(x, 0, 0));
    checked++;

    /* layer 0, plain sizes: fills every remaining slot and evicts layer 1's coldest unit
     * (its id 0), reusing that bigger slot with layer 0's own, smaller layout -- proves a slot
     * is reinterpreted by whichever layer currently occupies it, not by its own past occupant */
    TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK(unit_absent(x, 1, 0));
    for (int64_t e = 0; e < N_EXPERT; e++) {
        TR_CHECK(unit_ok(x, 0, e, align));
        checked++;
    }
    TR_CHECK(unit_ok(x, 1, 2, align));
    TR_CHECK(unit_ok(x, 1, 5, align));
    checked++;

    TR_CHECK_EQ_INT(f.bad_align, 0);
    TR_CHECK(checked > 0);
    tr_experts_free(x);
}

/* ---- lru: a hand-scripted sequence on the minimum store (8 slots) ----
 *
 * 1) acquire(layer0, [0,1,2,3]): store is empty -> 4 misses, 0 hits, 0 evictions. Hot-to-cold
 *    after: 3,2,1,0, then 4 still-free slots.
 * 2) acquire(layer0, [4,5]): 2 misses, consuming the last 2 free slots. Hot-to-cold: 5,4,3,2,1,0.
 *    Layer 0 is now fully resident (6 of 6), no free slots left.
 * 3) acquire(layer0, [0]): a hit -- 0 moves to the front. Hot-to-cold: 0,5,4,3,2,1.
 * 4) acquire(layer1, [0..5]), a whole layer at once: 6 misses, and only 8 slots exist, so 4 of
 *    layer0's 6 units must go. The coldest are 1,2,3,4 (in that order) -- 0 and 5 survive (0 was
 *    just touched, 5 was loaded last in step 2). 4 evictions.
 */
static void test_lru(uint64_t align) {
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    char err[256];
    fake_file f = mk_fake(align);
    tr_experts_config cfg = mk_cfg((uint64_t)MIN_SLOTS * slot_bytes, fake_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return;

    tr_experts_stats st;
    int checked = 0;

    int64_t s1[4] = {0, 1, 2, 3};
    TR_CHECK(tr_experts_acquire(x, 0, s1, 4) == 0);
    tr_experts_get_stats(x, &st);
    TR_CHECK_EQ_INT(st.hits, 0);
    TR_CHECK_EQ_INT(st.misses, 4);
    TR_CHECK_EQ_INT(st.evictions, 0);
    checked++;

    int64_t s2[2] = {4, 5};
    TR_CHECK(tr_experts_acquire(x, 0, s2, 2) == 0);
    tr_experts_get_stats(x, &st);
    TR_CHECK_EQ_INT(st.hits, 0);
    TR_CHECK_EQ_INT(st.misses, 6);
    TR_CHECK_EQ_INT(st.evictions, 0);
    for (int64_t e = 0; e < N_EXPERT; e++) TR_CHECK(unit_ok(x, 0, e, align));
    checked++;

    int64_t s3[1] = {0};
    TR_CHECK(tr_experts_acquire(x, 0, s3, 1) == 0);
    tr_experts_get_stats(x, &st);
    TR_CHECK_EQ_INT(st.hits, 1);
    TR_CHECK_EQ_INT(st.misses, 6);
    TR_CHECK_EQ_INT(st.evictions, 0);
    checked++;

    int64_t s4[6] = {0, 1, 2, 3, 4, 5};
    TR_CHECK(tr_experts_acquire(x, 1, s4, 6) == 0);
    tr_experts_get_stats(x, &st);
    TR_CHECK_EQ_INT(st.hits, 1);
    TR_CHECK_EQ_INT(st.misses, 12);
    TR_CHECK_EQ_INT(st.evictions, 4);
    checked++;

    for (int64_t e = 0; e < N_EXPERT; e++) {
        int should_survive = (e == 0 || e == 5);
        TR_CHECK(should_survive ? unit_ok(x, 0, e, align) : unit_absent(x, 0, e));
    }
    for (int64_t e = 0; e < N_EXPERT; e++) TR_CHECK(unit_ok(x, 1, e, align));
    checked++;

    TR_CHECK_EQ_INT(f.bad_align, 0);
    TR_CHECK(checked > 0);
    tr_experts_free(x);
}

/* ---- full_layer: a whole layer at once, twice, on the minimum store: a call's own units are
 * never among its own victims ---- */

static void test_full_layer(uint64_t align) {
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    char err[256];
    fake_file f = mk_fake(align);
    tr_experts_config cfg = mk_cfg((uint64_t)MIN_SLOTS * slot_bytes, fake_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return;

    int checked = 0;
    TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
    for (int64_t e = 0; e < N_EXPERT; e++) {
        TR_CHECK(unit_ok(x, 0, e, align));
        checked++;
    }

    TR_CHECK(tr_experts_acquire(x, 1, ALL_EXPERTS, N_EXPERT) == 0);
    for (int64_t e = 0; e < N_EXPERT; e++) {
        TR_CHECK(unit_ok(x, 1, e, align)); /* none of this call's own units got evicted mid-call */
        checked++;
    }

    TR_CHECK_EQ_INT(f.bad_align, 0);
    TR_CHECK(checked > 0);
    tr_experts_free(x);
}

/* ---- resident: budget covers every unit ---- */

/* The exact number of physical bytes one part's fill reads at read_align (Step B's lo/hi
 * formula, reproduced independently of experts.c so this stays a real check): read_align == 1
 * reduces to the part's own bytes, unchanged from before this store read anything aligned. */
static uint64_t part_read_bytes(uint64_t file_off, size_t part_bytes, uint64_t align) {
    uint64_t lo = file_off / align * align;
    uint64_t hi = (file_off + (uint64_t)part_bytes + align - 1) / align * align;
    return hi - lo;
}

static void test_resident(uint64_t align) {
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    char err[256];
    fake_file f = mk_fake(align);
    uint64_t full_budget = (uint64_t)(N_LAYERS * N_EXPERT) * slot_bytes;
    tr_experts_config cfg = mk_cfg(full_budget, fake_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return;

    int checked = 0;
    TR_CHECK(tr_experts_resident(x));
    checked++;

    TR_CHECK(tr_experts_load_all(x, NULL, NULL) == 0);
    TR_CHECK_EQ_INT(f.n_calls, N_LAYERS * N_EXPERT * TR_EXPERT_PARTS);
    uint64_t total = 0;
    for (int64_t layer = 0; layer < N_LAYERS; layer++)
        for (int p = 0; p < TR_EXPERT_PARTS; p++)
            for (int64_t e = 0; e < N_EXPERT; e++) {
                size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)p;
                uint64_t file_off = part_offset_table[idx] + (uint64_t)PART_BYTES_TABLE[layer][p] * (uint64_t)e;
                total += part_read_bytes(file_off, PART_BYTES_TABLE[layer][p], align);
            }
    TR_CHECK_EQ_INT(f.bytes, total);
    checked++;

    for (int64_t L = 0; L < N_LAYERS; L++)
        for (int64_t e = 0; e < N_EXPERT; e++) {
            TR_CHECK(unit_ok(x, L, e, align));
            checked++;
        }

    tr_experts_stats before;
    tr_experts_get_stats(x, &before);
    TR_CHECK_EQ_INT(before.evictions, 0);

    TR_CHECK(tr_experts_acquire(x, 2, ALL_EXPERTS, N_EXPERT) == 0);
    int64_t few[3] = {1, 3, 5};
    TR_CHECK(tr_experts_acquire(x, 0, few, 3) == 0);

    tr_experts_stats after;
    tr_experts_get_stats(x, &after);
    TR_CHECK_EQ_INT(after.misses, before.misses);
    TR_CHECK_EQ_INT(after.evictions, before.evictions);
    TR_CHECK_EQ_INT(f.n_calls, N_LAYERS * N_EXPERT * TR_EXPERT_PARTS); /* the reader was not called again */
    checked++;

    /* a non-resident store refuses load_all */
    fake_file f2 = mk_fake(align);
    tr_experts_config cfg2 = mk_cfg((uint64_t)MIN_SLOTS * slot_bytes, fake_read, &f2, align);
    tr_experts *y = tr_experts_create(&cfg2, err, sizeof err);
    TR_CHECK(y != NULL);
    if (y != NULL) {
        TR_CHECK(!tr_experts_resident(y));
        TR_CHECK(tr_experts_load_all(y, NULL, NULL) == -1);
        tr_experts_free(y);
        checked++;
    }

    TR_CHECK_EQ_INT(f.bad_align, 0);
    TR_CHECK_EQ_INT(f2.bad_align, 0);
    TR_CHECK(checked > 0);
    tr_experts_free(x);
}

/* ---- failure: the reader fails on the 1st, 2nd, 3rd part of a missing unit in turn ---- */

static void test_failure(uint64_t align) {
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    const int64_t layer0_full[6] = {0, 1, 2, 3, 4, 5};
    const int64_t layer1_two[2] = {0, 1};
    int checked = 0;

    for (int fail_part = 0; fail_part < TR_EXPERT_PARTS; fail_part++) {
        char err[256];
        fake_file f = mk_fake(align);
        tr_experts_config cfg = mk_cfg((uint64_t)MIN_SLOTS * slot_bytes, fake_read, &f, align);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x == NULL) continue;

        /* fill the store completely: layer 0 (6 units) + 2 of layer 1 = 8 slots, no eviction yet.
         * hot-to-cold at this point: (1,1),(1,0),(0,5),(0,4),(0,3),(0,2),(0,1),(0,0) -- (0,0) is
         * coldest, having never been touched since it loaded first. */
        TR_CHECK(tr_experts_acquire(x, 0, layer0_full, 6) == 0);
        TR_CHECK(tr_experts_acquire(x, 1, layer1_two, 2) == 0);
        tr_experts_stats before;
        tr_experts_get_stats(x, &before);
        TR_CHECK_EQ_INT(before.misses, 8);
        TR_CHECK_EQ_INT(before.evictions, 0);

        int64_t target[1] = {3};
        f.fail_at = f.n_calls + fail_part + 1; /* the (fail_part+1)-th read of the next unit */
        TR_CHECK(tr_experts_acquire(x, 2, target, 1) == -1);

        tr_experts_stats after;
        tr_experts_get_stats(x, &after);
        TR_CHECK_EQ_INT(after.misses, before.misses);           /* only completed units count */
        TR_CHECK_EQ_INT(after.evictions, before.evictions + 1); /* the victim's old unit is gone */
        TR_CHECK(unit_absent(x, 2, 3));                          /* the failed unit never appears */
        TR_CHECK(unit_absent(x, 0, 0));                          /* the coldest victim is gone */
        for (int64_t e = 1; e < N_EXPERT; e++) TR_CHECK(unit_ok(x, 0, e, align)); /* earlier units intact */
        for (int64_t e = 0; e < 2; e++) TR_CHECK(unit_ok(x, 1, e, align));

        f.fail_at = 0; /* never fail again */
        TR_CHECK(tr_experts_acquire(x, 2, target, 1) == 0);
        TR_CHECK(unit_ok(x, 2, 3, align));
        tr_experts_stats retried;
        tr_experts_get_stats(x, &retried);
        TR_CHECK_EQ_INT(retried.evictions, before.evictions + 1); /* no second eviction on retry */

        TR_CHECK_EQ_INT(f.bad_align, 0);
        tr_experts_free(x);
        checked++;
    }

    {
        char err[256];
        fake_file f = mk_fake(align);
        f.fail_at = 5; /* fails mid-way through load_all's file-order sweep */
        tr_experts_config cfg = mk_cfg((uint64_t)(N_LAYERS * N_EXPERT) * slot_bytes, fake_read, &f, align);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            TR_CHECK(tr_experts_load_all(x, NULL, NULL) == -1);
            TR_CHECK(unit_ok(x, 0, 0, align));     /* before the failure: loaded fine */
            TR_CHECK(unit_absent(x, 0, 1)); /* its 2nd part (call 5) is the one that failed */
            TR_CHECK_EQ_INT(f.bad_align, 0);
            tr_experts_free(x);
            checked++;
        }
    }

    TR_CHECK(checked > 0);
}

/* ---- arg_errors: bad layer, bad id, n == 0, n too large, a duplicate id ---- */

static void test_arg_errors(uint64_t align) {
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    char err[256];
    fake_file f = mk_fake(align);
    tr_experts_config cfg = mk_cfg((uint64_t)MIN_SLOTS * slot_bytes, fake_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return;

    int64_t some[3] = {1, 2, 4};
    TR_CHECK(tr_experts_acquire(x, 0, some, 3) == 0);
    tr_experts_stats before;
    tr_experts_get_stats(x, &before);

    int checked = 0;
    int64_t one[1];
    one[0] = 0;
    TR_CHECK(tr_experts_acquire(x, -1, one, 1) == -1);
    checked++;
    TR_CHECK(tr_experts_acquire(x, N_LAYERS, one, 1) == -1);
    checked++;
    one[0] = -1;
    TR_CHECK(tr_experts_acquire(x, 0, one, 1) == -1);
    checked++;
    one[0] = N_EXPERT;
    TR_CHECK(tr_experts_acquire(x, 0, one, 1) == -1);
    checked++;
    TR_CHECK(tr_experts_acquire(x, 0, some, 0) == -1);
    checked++;
    int64_t too_many[N_EXPERT + 1] = {0, 1, 2, 3, 4, 5, 0};
    TR_CHECK(tr_experts_acquire(x, 0, too_many, N_EXPERT + 1) == -1);
    checked++;
    int64_t dup[3] = {1, 4, 1};
    TR_CHECK(tr_experts_acquire(x, 0, dup, 3) == -1);
    checked++;

    tr_experts_stats after;
    tr_experts_get_stats(x, &after);
    TR_CHECK_EQ_INT(after.hits, before.hits);
    TR_CHECK_EQ_INT(after.misses, before.misses);
    TR_CHECK_EQ_INT(after.evictions, before.evictions);
    for (int64_t e = 0; e < N_EXPERT; e++) {
        int should_be_present = (e == 1 || e == 2 || e == 4);
        TR_CHECK(should_be_present ? unit_ok(x, 0, e, align) : unit_absent(x, 0, e));
    }
    checked++;

    /* tr_experts_part outside the store: NULL, never a read past its tables. (layer 1, expert 0)
     * is present first, so a bound that let expert == N_EXPERT through would find that unit
     * right behind the last expert of layer 0 and return it (tools/mutate_auto.py, 2026-09-22) */
    int64_t first[1] = {0};
    TR_CHECK(tr_experts_acquire(x, 1, first, 1) == 0 && tr_experts_part(x, 1, 0, 0) != NULL);
    TR_CHECK(tr_experts_part(x, 0, 1, 0) != NULL);
    TR_CHECK(tr_experts_part(x, -1, 1, 0) == NULL && tr_experts_part(x, N_LAYERS, 1, 0) == NULL);
    TR_CHECK(tr_experts_part(x, 0, -1, 0) == NULL && tr_experts_part(x, 0, N_EXPERT, 0) == NULL);
    TR_CHECK(tr_experts_part(x, 0, 1, -1) == NULL && tr_experts_part(x, 0, 1, TR_EXPERT_PARTS) == NULL);
    checked++;

    TR_CHECK_EQ_INT(f.bad_align, 0);
    TR_CHECK(checked > 0);
    tr_experts_free(x);
}

/* ---- o1_evidence: a larger store, a long pseudo-random acquire sequence, checked call by call
 * against a plain array-based reference LRU (O(n) per call there, on purpose) ---- */

enum { N_LAYERS2 = 64, N_EXPERT2 = 256, N_USED2 = 4 };
enum { N_UNITS2 = N_LAYERS2 * N_EXPERT2 };
/* every layer its own part sizes (Step A), small enough that round_up64 still folds them all to
 * one 64-byte slot per part: the variation exercises per-layer indexing without changing slot_bytes */
static size_t part_bytes_table2[N_LAYERS2 * TR_EXPERT_PARTS];
static uint64_t part_offset_table2[N_LAYERS2 * TR_EXPERT_PARTS];

static void build_part_bytes2(void) {
    for (int64_t layer = 0; layer < N_LAYERS2; layer++) {
        part_bytes_table2[layer * TR_EXPERT_PARTS + 0] = 5 + (size_t)(layer % 4);
        part_bytes_table2[layer * TR_EXPERT_PARTS + 1] = 3 + (size_t)((layer + 1) % 3);
        part_bytes_table2[layer * TR_EXPERT_PARTS + 2] = 2 + (size_t)((layer + 2) % 2);
    }
}

static void build_part_offsets2(void) {
    uint64_t off = 91;
    for (int64_t layer = 0; layer < N_LAYERS2; layer++)
        for (int p = 0; p < TR_EXPERT_PARTS; p++) {
            part_offset_table2[layer * TR_EXPERT_PARTS + p] = off;
            off += part_bytes_table2[layer * TR_EXPERT_PARTS + p] * (uint64_t)N_EXPERT2 + 7;
        }
}

/* xorshift32, fixed seed: reproducible without depending on libc's rand() state */
static uint32_t rng_state;
static uint32_t rng_next(void) {
    uint32_t v = rng_state;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    rng_state = v;
    return v;
}

static void shuffle_prefix(int64_t *pool, int64_t total, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int64_t j = i + (int64_t)(rng_next() % (uint32_t)(total - i));
        int64_t tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
    }
}

enum { REF_MAX_SLOTS = 700 };
typedef struct {
    int64_t list[REF_MAX_SLOTS]; /* resident units, hot (index 0) to cold */
    int32_t pos[N_UNITS2];       /* index into list, or -1 */
    int64_t count;
    int64_t n_slots;
    uint64_t hits, misses, evictions;
} ref_model;

static void ref_init(ref_model *r, int64_t n_slots) {
    memset(r, 0, sizeof *r);
    for (int64_t i = 0; i < N_UNITS2; i++) r->pos[i] = -1;
    r->n_slots = n_slots;
}

static void ref_move_front(ref_model *r, int64_t idx) {
    int64_t unit = r->list[idx];
    memmove(&r->list[1], &r->list[0], (size_t)idx * sizeof(int64_t));
    r->list[0] = unit;
    for (int64_t i = 0; i <= idx; i++) r->pos[r->list[i]] = (int32_t)i;
}

static int ref_acquire(ref_model *r, int64_t layer, const int64_t *ids, int64_t n) {
    if (layer < 0 || layer >= N_LAYERS2) return -1;
    if (n < 1 || n > N_EXPERT2) return -1;
    for (int64_t i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= N_EXPERT2) return -1;
        for (int64_t j = 0; j < i; j++)
            if (ids[j] == ids[i]) return -1;
    }

    for (int64_t i = 0; i < n; i++) {
        int64_t unit = layer * N_EXPERT2 + ids[i];
        if (r->pos[unit] != -1) {
            ref_move_front(r, r->pos[unit]);
            r->hits++;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        int64_t unit = layer * N_EXPERT2 + ids[i];
        if (r->pos[unit] != -1) continue;
        if (r->count < r->n_slots) {
            memmove(&r->list[1], &r->list[0], (size_t)r->count * sizeof(int64_t));
            r->list[0] = unit;
            r->count++;
        } else {
            int64_t victim_unit = r->list[r->count - 1];
            r->pos[victim_unit] = -1;
            r->evictions++;
            memmove(&r->list[1], &r->list[0], (size_t)(r->count - 1) * sizeof(int64_t));
            r->list[0] = unit;
        }
        for (int64_t k = 0; k < r->count; k++) r->pos[r->list[k]] = (int32_t)k;
        r->misses++;
    }
    return 0;
}

enum { O1_ITERS = 800 };

static void test_o1_evidence(uint64_t align) {
    build_part_bytes2();
    build_part_offsets2();
    char err[256];
    fake_file f = mk_fake(align);
    int64_t n_slots2 = 600; /* > min_slots (260), well short of n_units2 (16384) */
    tr_experts_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_layers = N_LAYERS2;
    cfg.n_expert = N_EXPERT2;
    cfg.n_used = N_USED2;
    cfg.part_bytes = part_bytes_table2;
    cfg.part_offset = part_offset_table2;
    cfg.budget_bytes = (uint64_t)n_slots2 * tr_experts_slot_bytes(part_bytes_table2, N_LAYERS2, align);
    cfg.read = fake_read;
    cfg.read_ctx = &f;
    cfg.read_align = align;
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return;

    ref_model ref;
    ref_init(&ref, n_slots2);
    rng_state = 0xC0FFEEu;

    int64_t pool[N_EXPERT2];
    int64_t mismatches = 0, iters_checked = 0;

    for (int it = 0; it < O1_ITERS; it++) {
        for (int64_t i = 0; i < N_EXPERT2; i++) pool[i] = i;
        int64_t n = 1 + (int64_t)(rng_next() % 8);
        if (rng_next() % 20 == 0) n = N_EXPERT2;
        int64_t layer = (int64_t)(rng_next() % N_LAYERS2);
        shuffle_prefix(pool, N_EXPERT2, n);

        int rc_store = tr_experts_acquire(x, layer, pool, n);
        int rc_ref = ref_acquire(&ref, layer, pool, n);
        if (rc_store != rc_ref) mismatches++;

        tr_experts_stats st;
        tr_experts_get_stats(x, &st);
        if (st.hits != ref.hits || st.misses != ref.misses || st.evictions != ref.evictions) mismatches++;

        for (int64_t u = 0; u < N_UNITS2; u++) {
            int store_present = tr_experts_part(x, u / N_EXPERT2, u % N_EXPERT2, 0) != NULL;
            int ref_present = ref.pos[u] != -1;
            if (store_present != ref_present) {
                mismatches++;
                break;
            }
        }
        iters_checked++;
    }

    TR_CHECK_EQ_INT(mismatches, 0);
    TR_CHECK_EQ_INT(iters_checked, O1_ITERS);
    TR_CHECK_EQ_INT(f.bad_align, 0);

    tr_experts_free(x);
}

/* ---- few_layers: n_units below min_slots -- only a fully resident budget can run ---- */

static void test_few_layers(uint64_t align) {
    enum { FEW_LAYERS = 1, FEW_EXPERT = 2, FEW_USED = 1 };          /* n_units = 2, min_slots = 3 */
    enum { FEW_UNITS = FEW_LAYERS * FEW_EXPERT, FEW_MIN = FEW_EXPERT + FEW_USED };
    static const size_t few_part_bytes[FEW_LAYERS * TR_EXPERT_PARTS] = {40, 20, 10};
    uint64_t few_part_offset[FEW_LAYERS * TR_EXPERT_PARTS];
    uint64_t off = 5;
    for (int p = 0; p < TR_EXPERT_PARTS; p++) {
        few_part_offset[p] = off;
        off += few_part_bytes[p] * (uint64_t)FEW_EXPERT + 3;
    }
    uint64_t slot_bytes = tr_experts_slot_bytes(few_part_bytes, FEW_LAYERS, align);
    TR_CHECK_EQ_INT(tr_experts_min_slots(FEW_EXPERT, FEW_USED), FEW_MIN);
    TR_CHECK(FEW_UNITS < FEW_MIN); /* the whole point of this test: impossible to run partially */

    int checked = 0;
    char err[256];
    fake_file f = mk_fake(align);
    tr_experts_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_layers = FEW_LAYERS;
    cfg.n_expert = FEW_EXPERT;
    cfg.n_used = FEW_USED;
    cfg.part_bytes = few_part_bytes;
    cfg.part_offset = few_part_offset;
    cfg.read = fake_read;
    cfg.read_ctx = &f;
    cfg.read_align = align;

    /* resident (n_slots == n_units): accepted despite being below min_slots */
    cfg.budget_bytes = (uint64_t)FEW_UNITS * slot_bytes;
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x != NULL) {
        TR_CHECK(tr_experts_resident(x));
        TR_CHECK(tr_experts_load_all(x, NULL, NULL) == 0);
        for (int64_t e = 0; e < FEW_EXPERT; e++)
            for (int p = 0; p < TR_EXPERT_PARTS; p++) TR_CHECK(tr_experts_part(x, 0, e, p) != NULL);
        checked++;
        tr_experts_free(x);
    }

    /* one slot short of resident: still below min_slots, and no longer every unit -- refused */
    cfg.budget_bytes = (uint64_t)(FEW_UNITS - 1) * slot_bytes;
    TR_CHECK(tr_experts_create(&cfg, err, sizeof err) == NULL);
    TR_CHECK(strstr(err, "MiB") != NULL);
    checked++;

    TR_CHECK_EQ_INT(f.bad_align, 0);
    TR_CHECK(checked > 0);
}

/* ---- align_reuse: a slot reused by two experts whose file offsets fall on different residues
 * modulo the read alignment -- proves the recorded shift (experts.h "The recorded offsets change
 * every time a slot is refilled") is really taken at every fill, not fixed once when the slot
 * was created or first used (tools/mutate_experts.sh). Only meaningful when read_align > 1: at
 * read_align == 1 every part always starts at shift 0. ---- */

static void test_align_reuse(void) {
    uint64_t align = TR_FILE_DIRECT_ALIGN;
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    char err[256];
    fake_file f = mk_fake(align);
    tr_experts_config cfg = mk_cfg((uint64_t)MIN_SLOTS * slot_bytes, fake_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return;

    int checked = 0;
    /* MIN_SLOTS (8) leaves no margin once both layers are in play: 3 of layer 1 (5 slots still
     * free) then all 6 of layer 0 evicts layer 1's coldest (its id 0, never touched again) to make
     * room -- neither part_bytes (100 and 150) divides 4096, so layer1/0 and whichever layer0
     * expert reuses its slot land at different offsets modulo 4096, and need different shifts. */
    int64_t ids[3] = {0, 2, 5};
    TR_CHECK(tr_experts_acquire(x, 1, ids, 3) == 0);
    TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK(unit_absent(x, 1, 0));
    for (int64_t e = 0; e < N_EXPERT; e++) {
        TR_CHECK(unit_ok(x, 0, e, align));
        checked++;
    }
    TR_CHECK(unit_ok(x, 1, 2, align));
    TR_CHECK(unit_ok(x, 1, 5, align));
    checked++;

    /* bring layer 1's expert 0 back: every slot is full now, so this evicts one of layer 0's
     * (its coldest), and layer1/0 reoccupies a slot it does not currently hold -- its shift
     * recorded fresh again, not whatever a past occupant of that slot left behind. */
    int64_t back[1] = {0};
    TR_CHECK(tr_experts_acquire(x, 1, back, 1) == 0);
    TR_CHECK(unit_ok(x, 1, 0, align));
    checked++;

    TR_CHECK_EQ_INT(f.bad_align, 0);
    TR_CHECK(checked > 0);
    tr_experts_free(x);
}

/* ---- prefetch: the I/O thread (tr_experts_prefetch) ----
 * A reader both threads may call at once: its counters are atomic, and it can be slowed down (so
 * a unit is still in flight when the test asks for it) or told to fail the next reads made by any
 * thread but the test's own -- the I/O thread's. (Not a file offset: a direct store reads 4096
 * bytes at a time, and one such read covers every unit of these tiny layers.) */

static _Thread_local int t_test_thread; /* global-ok: 1 on the thread that runs the tests */

typedef struct {
    atomic_llong calls, bytes, fails, bad_align;
    atomic_llong fail_io;     /* reads left to fail on a thread other than the test's */
    double delay;             /* seconds each read takes */
    uint64_t align_check;
} shared_file;

static int shared_read(void *ctx, void *buf, size_t n, uint64_t offset) {
    shared_file *f = (shared_file *)ctx;
    atomic_fetch_add(&f->calls, 1);
    if (f->delay > 0) {
        double t0 = tr_time_sec();
        while (tr_time_sec() - t0 < f->delay) {
        }
    }
    if (!t_test_thread && atomic_load(&f->fail_io) > 0 && atomic_fetch_sub(&f->fail_io, 1) > 0) {
        atomic_fetch_add(&f->fails, 1);
        return -1;
    }
    if (f->align_check > 1 && (offset % f->align_check != 0 || n % f->align_check != 0))
        atomic_fetch_add(&f->bad_align, 1);
    unsigned char *dst = (unsigned char *)buf;
    for (size_t i = 0; i < n; i++) dst[i] = pattern_byte(offset + i);
    atomic_fetch_add(&f->bytes, (long long)n);
    return 0;
}

static void shared_init(shared_file *f, uint64_t align) {
    atomic_init(&f->calls, 0);
    atomic_init(&f->bytes, 0);
    atomic_init(&f->fails, 0);
    atomic_init(&f->bad_align, 0);
    atomic_init(&f->fail_io, 0);
    f->delay = 0.0;
    f->align_check = align;
}

enum { PREFETCH_SLOTS = 2 * N_EXPERT + N_USED }; /* 14 of 18: the smallest store that reads ahead */

static int layer_ok(const tr_experts *x, int64_t layer, uint64_t align) {
    for (int64_t e = 0; e < N_EXPERT; e++)
        if (!unit_ok(x, layer, e, align)) return 0;
    return 1;
}

static int g_pf_start, g_pf_basic, g_pf_victims, g_pf_wait, g_pf_fail, g_pf_fail_acquire, g_pf_concurrent,
    g_pf_free_in_flight, g_pf_hot, g_pf_cancel;

static void test_prefetch(uint64_t align) {
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    shared_file f;
    shared_init(&f, align);
    char err[256];

    /* start: only a partial store with room for two layers and a margin */
    {
        uint64_t all = (uint64_t)N_LAYERS * N_EXPERT * slot_bytes; /* resident */
        tr_experts_config cfg = mk_cfg(all, shared_read, &f, align);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 1 && !tr_experts_prefetching(x));
        tr_experts_free(x);
        cfg.budget_bytes = (uint64_t)(PREFETCH_SLOTS - 1) * slot_bytes;
        x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 1 && !tr_experts_prefetching(x));
        TR_CHECK(x != NULL && tr_experts_prefetch(x, 1, 0) == 0); /* not started: nothing queued */
        tr_experts_free(x);
        cfg.budget_bytes = (uint64_t)PREFETCH_SLOTS * slot_bytes;
        x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 0 && tr_experts_prefetching(x));
        TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 0); /* again: already running */
        tr_experts_free(x);
        g_pf_start++;
    }

    tr_experts_config cfg = mk_cfg((uint64_t)PREFETCH_SLOTS * slot_bytes, shared_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return;
    TR_CHECK(tr_experts_prefetch_start(x) == 0);

    /* basic: layer 1 read ahead while layer 0 is held; nothing of it usable until taken in; then
     * acquire finds it all present and exact, counted as read ahead */
    tr_experts_stats st;
    TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK_EQ_INT(tr_experts_prefetch(x, 1, 0), N_EXPERT);
    int none_visible = 1;
    for (int64_t e = 0; e < N_EXPERT; e++) none_visible &= unit_absent(x, 1, e);
    TR_CHECK(none_visible); /* in flight, or done but not taken in: never readable */
    TR_CHECK(tr_experts_acquire(x, 1, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK(layer_ok(x, 1, align));
    TR_CHECK(layer_ok(x, 0, align));
    tr_experts_get_stats(x, &st);
    TR_CHECK_EQ_INT(st.prefetched, N_EXPERT);
    TR_CHECK_EQ_INT(st.misses, 2 * N_EXPERT);
    TR_CHECK_EQ_INT(st.hits, N_EXPERT);
    TR_CHECK_EQ_INT(atomic_load(&f.bad_align), 0);
    if (none_visible && st.prefetched == N_EXPERT) g_pf_basic++;

    /* victims: layer 2 ahead while layer 1 computes takes the 2 free slots and layer 0's coldest,
     * never layer 1's; layer 0 ahead while 1 is kept finds nothing left (layer 1 kept, layer 2 in
     * flight, layer 0 is the target itself) and queues nothing */
    TR_CHECK_EQ_INT(tr_experts_prefetch(x, 2, 1), N_EXPERT);
    TR_CHECK_EQ_INT(tr_experts_prefetch(x, 0, 1), 0);
    TR_CHECK(tr_experts_prefetch_wait(x) == 0);
    TR_CHECK(layer_ok(x, 1, align));
    TR_CHECK(layer_ok(x, 2, align));
    int layer0_left = 0;
    for (int64_t e = 0; e < N_EXPERT; e++) layer0_left += !unit_absent(x, 0, e);
    TR_CHECK_EQ_INT(layer0_left, PREFETCH_SLOTS - 2 * N_EXPERT);
    tr_experts_get_stats(x, &st);
    TR_CHECK_EQ_INT(st.prefetched, 2 * N_EXPERT);
    g_pf_victims++;

    /* wait: a slow reader, and the acquire comes at once: it waits for the units in flight */
    f.delay = 0.002;
    TR_CHECK_EQ_INT(tr_experts_prefetch(x, 0, 2), N_EXPERT - layer0_left);
    TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK(layer_ok(x, 0, align));
    tr_experts_get_stats(x, &st);
    TR_CHECK(st.prefetch_wait_sec > 0.0);
    if (st.prefetch_wait_sec > 0.0) g_pf_wait++;
    f.delay = 0.0;

    /* concurrent: layer 1 in flight (slowly) while the calling thread reads layer 2's missing
     * units on demand -- two readers at once, and no demand victim is a slot in flight */
    f.delay = 0.001;
    int64_t queued = tr_experts_prefetch(x, 1, 0);
    TR_CHECK(queued > 0);
    TR_CHECK(tr_experts_acquire(x, 2, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK(layer_ok(x, 2, align));
    TR_CHECK(tr_experts_prefetch_wait(x) == 0);
    int layer1_ok = 1;
    for (int64_t e = 0; e < N_EXPERT; e++) layer1_ok &= unit_absent(x, 1, e) || unit_ok(x, 1, e, align);
    TR_CHECK(layer1_ok); /* every unit of layer 1 still in RAM is exact: none was overwritten mid-read */
    TR_CHECK(layer_ok(x, 2, align));
    f.delay = 0.0;
    if (queued > 0 && layer1_ok) g_pf_concurrent++;

    /* fail: a read ahead that fails leaves that unit absent and the rest exact; the wait reports
     * it once; the unit is read on demand afterwards */
    TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
    int64_t gone = -1; /* the lowest unit of layer 1 not in RAM now: the first one read ahead */
    for (int64_t e = N_EXPERT - 1; e >= 0; e--)
        if (unit_absent(x, 1, e)) gone = e;
    TR_CHECK(gone >= 0);
    if (gone < 0) gone = 0;
    atomic_store(&f.fail_io, 1);
    long long fails0 = atomic_load(&f.fails);
    queued = tr_experts_prefetch(x, 1, 0);
    TR_CHECK(queued > 0);
    TR_CHECK(tr_experts_prefetch_wait(x) == -1);
    TR_CHECK(atomic_load(&f.fails) > fails0);
    TR_CHECK(unit_absent(x, 1, gone));
    int others_ok = 1;
    for (int64_t e = 0; e < N_EXPERT; e++)
        if (e != gone) others_ok &= unit_ok(x, 1, e, align);
    TR_CHECK(others_ok); /* layer 1's other units all read ahead and exact */
    TR_CHECK(tr_experts_prefetch_wait(x) == 0); /* reported once */
    TR_CHECK(tr_experts_acquire(x, 1, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK(layer_ok(x, 1, align));
    if (others_ok && unit_ok(x, 1, gone, align)) g_pf_fail++;

    /* fail, asked for: acquire of a unit whose read ahead failed is -1, like a failed read */
    TR_CHECK(tr_experts_acquire(x, 2, ALL_EXPERTS, N_EXPERT) == 0);
    gone = -1;
    for (int64_t e = N_EXPERT - 1; e >= 0; e--)
        if (unit_absent(x, 0, e)) gone = e;
    TR_CHECK(gone >= 0);
    if (gone < 0) gone = 0;
    atomic_store(&f.fail_io, 1);
    queued = tr_experts_prefetch(x, 0, 2);
    TR_CHECK(queued > 0);
    int rc = tr_experts_acquire(x, 0, &gone, 1);
    TR_CHECK_EQ_INT(rc, -1);
    TR_CHECK(unit_absent(x, 0, gone));
    TR_CHECK(tr_experts_prefetch_wait(x) == -1); /* the failure, still reported to the wait */
    TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK(layer_ok(x, 0, align));
    if (rc == -1) g_pf_fail_acquire++;
    tr_experts_free(x);

    /* hot: concurrent again under ds4's eviction. Layer 1 read ahead (never named: hotness 0) while
     * layer 2 is read on demand: the coolest units are layer 1's in flight, and none may be a victim;
     * layer 0's (hotness 1) go instead, and every unit left in RAM is exact */
    cfg.evict = TR_EXPERTS_EVICT_HOT;
    x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 0);
    if (x != NULL) {
        TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
        f.delay = 0.002;
        TR_CHECK_EQ_INT(tr_experts_prefetch(x, 1, 0), N_EXPERT);
        TR_CHECK(tr_experts_acquire(x, 2, ALL_EXPERTS, N_EXPERT) == 0);
        TR_CHECK(tr_experts_prefetch_wait(x) == 0);
        f.delay = 0.0;
        int exact = layer_ok(x, 2, align);
        for (int64_t e = 0; e < N_EXPERT; e++) exact &= unit_absent(x, 1, e) || unit_ok(x, 1, e, align);
        TR_CHECK(exact);
        int layer1_whole = 1; /* nothing of layer 1 was evicted: it was all in flight */
        for (int64_t e = 0; e < N_EXPERT; e++) layer1_whole &= unit_ok(x, 1, e, align);
        TR_CHECK(layer1_whole);
        if (exact && layer1_whole) g_pf_hot++;
        tr_experts_free(x);
    }
    cfg.evict = TR_EXPERTS_EVICT_LRU;

    /* cancel: layer 1 queued ahead (slow reads, one unit a read: no runs here), then layer 2 while the
     * I/O thread is still reading layer 1: the second call returns only once the thread has taken
     * layer 2's first unit, so a cancel naming only layer 2's last unit drops the N_EXPERT - 2 between
     * them unread (absent, slots free, never counted as read) while the first, in flight, and the last
     * land; the dropped ones are then read on demand, exact. A read ahead that returned at once would
     * find layer 2's first still queued behind layer 1 and drop it too: N_EXPERT - 1 */
    x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 0);
    if (x != NULL) {
        TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
        f.delay = 0.01;
        int64_t last_one[1] = {N_EXPERT - 1};
        int64_t q1 = tr_experts_prefetch(x, 1, 0);
        int64_t q = tr_experts_prefetch(x, 2, 1);
        int64_t dropped = tr_experts_prefetch_cancel(x, 2, last_one, 1);
        TR_CHECK_EQ_INT(q1, N_EXPERT);
        TR_CHECK_EQ_INT(q, N_EXPERT);
        TR_CHECK_EQ_INT(dropped, N_EXPERT - 2);
        TR_CHECK(tr_experts_prefetch_wait(x) == 0);
        f.delay = 0.0;
        tr_experts_stats cs;
        tr_experts_get_stats(x, &cs);
        TR_CHECK_EQ_INT(cs.cancelled, N_EXPERT - 2);
        TR_CHECK_EQ_INT(cs.prefetched, N_EXPERT + 2);
        int landed = layer_ok(x, 1, align) && unit_ok(x, 2, 0, align) && unit_ok(x, 2, N_EXPERT - 1, align);
        int absent = 1;
        for (int64_t e = 1; e < N_EXPERT - 1; e++) absent &= unit_absent(x, 2, e);
        TR_CHECK(landed && absent);
        TR_CHECK(tr_experts_acquire(x, 2, ALL_EXPERTS, N_EXPERT) == 0);
        TR_CHECK(layer_ok(x, 2, align));
        if (dropped == N_EXPERT - 2 && landed && absent && cs.cancelled == N_EXPERT - 2) g_pf_cancel++;
        tr_experts_free(x);
    }

    /* freed with reads in flight: the I/O thread finishes and is joined, nothing leaks (ASan) */
    x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x != NULL) {
        f.delay = 0.001;
        TR_CHECK(tr_experts_prefetch_start(x) == 0);
        TR_CHECK_EQ_INT(tr_experts_prefetch(x, 1, 0), N_EXPERT);
        tr_experts_free(x);
        f.delay = 0.0;
        g_pf_free_in_flight++;
    }
}

/* ---- runs (cfg.readv): a layer's consecutive experts in one request a part, each unit's pages
 * scattered into its own slot and the page two neighbours share copied after ---- */

/* The fake file again, with a scattered door: fake_file first, so fake_read takes the same ctx. */
typedef struct {
    fake_file f;
    int64_t calls, max_pieces, fail_at, bad_align;
    int64_t batches, batch_not_parts; /* readv calls, and those with other than a run's three requests */
    uint64_t bytes;
} fake_run_file;

/* calls counts the requests, as the store's stats do: fail_at names the request that fails, and the
 * others of its call still land (tr_file_preadv_n waits for every request it issued) */
static int fake_readv(void *ctx, const tr_readv_req *req, int n, void *scratch) {
    fake_run_file *r = (fake_run_file *)ctx;
    (void)scratch;
    r->batches++;
    if (n != TR_EXPERT_PARTS) r->batch_not_parts++;
    int failed = 0;
    for (int k = 0; k < n; k++) {
        r->calls++;
        if (r->fail_at == r->calls) {
            failed = 1;
            continue;
        }
        const tr_iov *iov = req[k].iov;
        int cnt = req[k].cnt;
        uint64_t offset = req[k].offset;
        if (cnt > r->max_pieces) r->max_pieces = cnt;
        uint64_t a = r->f.align_check, o = offset;
        if (a > 1 && offset % a != 0) r->bad_align++;
        for (int i = 0; i < cnt; i++) {
            if (a > 1 && ((uintptr_t)iov[i].base % a != 0 || iov[i].n % a != 0)) r->bad_align++;
            unsigned char *dst = (unsigned char *)iov[i].base;
            for (size_t j = 0; j < iov[i].n; j++) dst[j] = pattern_byte(o + j);
            o += iov[i].n;
        }
        r->bytes += o - offset;
    }
    return failed ? -1 : 0;
}

/* Parts of at least a page: layer 0 and 2 unaligned (sizes and offsets: every neighbour shares a
 * page on a direct store), layer 1 all aligned (none does: the copy is skipped). */
enum { RN_LAYERS = 3, RN_EXPERT = 6, RN_USED = 2 };
static const size_t RN_PART[RN_LAYERS][TR_EXPERT_PARTS] = {
    {3 * 4096 + 100, 2 * 4096 + 3000, 4096 + 5},
    {2 * 4096, 4096, 3 * 4096},
    {3 * 4096 + 100, 2 * 4096 + 3000, 4096 + 5},
};
static uint64_t rn_offset[RN_LAYERS * TR_EXPERT_PARTS];
static const int64_t RN_ALL[RN_EXPERT] = {0, 1, 2, 3, 4, 5};

static void rn_build_offsets(void) {
    uint64_t off = 3 * 4096 + 123;
    for (int64_t layer = 0; layer < RN_LAYERS; layer++)
        for (int p = 0; p < TR_EXPERT_PARTS; p++) {
            if (layer == 1) off = (off + 4095) / 4096 * 4096;
            if (layer == 2 && p == 0) off += 777; /* another residue than layer 0's */
            rn_offset[layer * TR_EXPERT_PARTS + p] = off;
            off += RN_PART[layer][p] * (uint64_t)RN_EXPERT + 4096 + 13;
        }
}

static tr_experts_config rn_cfg(uint64_t slots, fake_run_file *r, uint64_t align, uint64_t run_bytes) {
    tr_experts_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_layers = RN_LAYERS;
    cfg.n_expert = RN_EXPERT;
    cfg.n_used = RN_USED;
    cfg.part_bytes = &RN_PART[0][0];
    cfg.part_offset = rn_offset;
    cfg.budget_bytes = slots * tr_experts_slot_bytes(&RN_PART[0][0], RN_LAYERS, align);
    cfg.read = fake_read;
    cfg.readv = fake_readv;
    cfg.read_ctx = r;
    cfg.read_align = align;
    cfg.run_bytes = run_bytes;
    return cfg;
}

static int rn_unit_ok(const tr_experts *x, int64_t layer, int64_t e) {
    for (int p = 0; p < TR_EXPERT_PARTS; p++) {
        const unsigned char *got = (const unsigned char *)tr_experts_part(x, layer, e, p);
        if (got == NULL) return 0;
        uint64_t base = rn_offset[layer * TR_EXPERT_PARTS + p] + RN_PART[layer][p] * (uint64_t)e;
        for (size_t i = 0; i < RN_PART[layer][p]; i++)
            if (got[i] != pattern_byte(base + i)) return 0;
    }
    return 1;
}

static int rn_layer_ok(const tr_experts *x, int64_t layer) {
    for (int64_t e = 0; e < RN_EXPERT; e++)
        if (!rn_unit_ok(x, layer, e)) return 0;
    return 1;
}

/* Bytes of one request of the run (layer, part, e0 .. e0 + k - 1): its units' aligned ranges end to end. */
static uint64_t rn_request_bytes(int64_t layer, int p, int64_t e0, int64_t k, uint64_t align) {
    uint64_t pb = RN_PART[layer][p], first = rn_offset[layer * TR_EXPERT_PARTS + p] + pb * (uint64_t)e0;
    return (first + pb * (uint64_t)k + align - 1) / align * align - first / align * align;
}

static int64_t g_rn_shared, g_rn_unshared, g_rn_split, g_rn_demand, g_rn_ahead, g_rn_fail, g_rn_fallback;

static void test_runs(uint64_t align) {
    char err[256];

    /* the branch the copy serves, by construction: neighbours sharing a page, and not */
    for (int64_t layer = 0; layer < RN_LAYERS && align > 1; layer++)
        for (int p = 0; p < TR_EXPERT_PARTS; p++)
            for (int64_t e = 0; e + 1 < RN_EXPERT; e++) {
                uint64_t off = rn_offset[layer * TR_EXPERT_PARTS + p] + RN_PART[layer][p] * (uint64_t)e;
                uint64_t hi = (off + RN_PART[layer][p] + align - 1) / align * align;
                uint64_t next_lo = (off + RN_PART[layer][p]) / align * align;
                if (hi > next_lo) g_rn_shared++;
                else g_rn_unshared++;
            }

    /* (load) resident, one request a part a layer: every unit exact, the reader one at a time never called */
    {
        fake_run_file r;
        memset(&r, 0, sizeof r);
        r.f.align_check = align;
        tr_experts_config cfg = rn_cfg(RN_LAYERS * RN_EXPERT, &r, align, 0);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            tr_experts_touch(x, NULL); /* before any read: contents untouched by it */
            TR_CHECK(tr_experts_load_all(x, NULL, NULL) == 0);
            TR_CHECK_EQ_INT(r.calls, RN_LAYERS * TR_EXPERT_PARTS);
            /* a run's three parts handed over in one call, so the disk has them in flight together */
            TR_CHECK_EQ_INT(r.batches, RN_LAYERS);
            TR_CHECK_EQ_INT(r.batch_not_parts, 0);
            TR_CHECK_EQ_INT(r.f.n_calls, 0);
            TR_CHECK_EQ_INT(r.max_pieces, RN_EXPERT);
            uint64_t want = 0;
            for (int64_t layer = 0; layer < RN_LAYERS; layer++)
                for (int p = 0; p < TR_EXPERT_PARTS; p++) want += rn_request_bytes(layer, p, 0, RN_EXPERT, align);
            TR_CHECK_EQ_INT(r.bytes, want);
            tr_experts_stats st;
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.requests, RN_LAYERS * TR_EXPERT_PARTS);
            TR_CHECK_EQ_INT(st.bytes_read, want);
            TR_CHECK_EQ_INT(st.misses, RN_LAYERS * RN_EXPERT);
            for (int64_t layer = 0; layer < RN_LAYERS; layer++) TR_CHECK(rn_layer_ok(x, layer));
            tr_experts_touch(x, NULL); /* after: still the same bytes */
            for (int64_t layer = 0; layer < RN_LAYERS; layer++) TR_CHECK(rn_layer_ok(x, layer));
            tr_experts_free(x);
        }
    }

    /* (split) a run longer than a request carries: 6 experts at 2 a request, in equal requests */
    {
        fake_run_file r;
        memset(&r, 0, sizeof r);
        r.f.align_check = align;
        tr_experts_config cfg = rn_cfg(RN_LAYERS * RN_EXPERT, &r, align, (uint64_t)(2.5 * (3 * 4096 + 100)));
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            TR_CHECK(tr_experts_load_all(x, NULL, NULL) == 0);
            TR_CHECK_EQ_INT(r.calls, RN_LAYERS * TR_EXPERT_PARTS * 3);
            TR_CHECK_EQ_INT(r.max_pieces, 2);
            for (int64_t layer = 0; layer < RN_LAYERS; layer++) TR_CHECK(rn_layer_ok(x, layer));
            g_rn_split++;
            tr_experts_free(x);
        }
    }

    /* (demand) the smallest store: a layer given in id order is one run; ids out of order are read
     * one at a time; two consecutive missing ones are a run again */
    {
        fake_run_file r;
        memset(&r, 0, sizeof r);
        r.f.align_check = align;
        tr_experts_config cfg = rn_cfg(RN_EXPERT + RN_USED, &r, align, 0);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            TR_CHECK(tr_experts_acquire(x, 0, RN_ALL, RN_EXPERT) == 0);
            TR_CHECK_EQ_INT(r.calls, TR_EXPERT_PARTS);
            TR_CHECK(rn_layer_ok(x, 0));
            int64_t scattered[3] = {5, 0, 2};
            TR_CHECK(tr_experts_acquire(x, 1, scattered, 3) == 0);
            TR_CHECK_EQ_INT(r.calls, TR_EXPERT_PARTS);
            TR_CHECK_EQ_INT(r.f.n_calls, 3 * TR_EXPERT_PARTS);
            int64_t pair[2] = {3, 4};
            TR_CHECK(tr_experts_acquire(x, 1, pair, 2) == 0);
            TR_CHECK_EQ_INT(r.calls, 2 * TR_EXPERT_PARTS);
            for (int i = 0; i < 3; i++) TR_CHECK(rn_unit_ok(x, 1, scattered[i]));
            TR_CHECK(rn_unit_ok(x, 1, 3) && rn_unit_ok(x, 1, 4));
            tr_experts_stats st;
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.requests, 2 * TR_EXPERT_PARTS + 3 * TR_EXPERT_PARTS);
            TR_CHECK_EQ_INT(r.bad_align + r.f.bad_align, 0);
            g_rn_demand++;

            /* (fail) the second request of a run fails: the run's units absent, the store consistent */
            r.fail_at = r.calls + 2;
            TR_CHECK(tr_experts_acquire(x, 2, RN_ALL, RN_EXPERT) == -1);
            for (int64_t e = 0; e < RN_EXPERT; e++) TR_CHECK(unit_absent(x, 2, e));
            TR_CHECK(tr_experts_acquire(x, 2, RN_ALL, RN_EXPERT) == 0);
            TR_CHECK(rn_layer_ok(x, 2));
            g_rn_fail++;
            tr_experts_free(x);
        }
    }

    /* (ahead) the I/O thread takes a layer queued in id order as one run; a failed run's units are
     * absent after the wait and read on demand afterwards */
    {
        fake_run_file r;
        memset(&r, 0, sizeof r);
        r.f.align_check = align;
        tr_experts_config cfg = rn_cfg(2 * RN_EXPERT + RN_USED, &r, align, 0);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            TR_CHECK(tr_experts_prefetch_start(x) == 0);
            TR_CHECK(tr_experts_acquire(x, 0, RN_ALL, RN_EXPERT) == 0);
            int64_t before = r.calls;
            TR_CHECK_EQ_INT(tr_experts_prefetch(x, 1, 0), RN_EXPERT);
            TR_CHECK(tr_experts_prefetch_wait(x) == 0);
            TR_CHECK_EQ_INT(r.calls - before, TR_EXPERT_PARTS);
            TR_CHECK(tr_experts_acquire(x, 1, RN_ALL, RN_EXPERT) == 0);
            TR_CHECK(rn_layer_ok(x, 1));
            tr_experts_stats st;
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.prefetched, RN_EXPERT);
            TR_CHECK_EQ_INT(st.requests, 2 * TR_EXPERT_PARTS);
            g_rn_ahead++;

            r.fail_at = r.calls + 1;
            TR_CHECK_EQ_INT(tr_experts_prefetch(x, 2, 1), RN_EXPERT);
            TR_CHECK(tr_experts_prefetch_wait(x) == -1);
            for (int64_t e = 0; e < RN_EXPERT; e++) TR_CHECK(unit_absent(x, 2, e));
            TR_CHECK(tr_experts_acquire(x, 2, RN_ALL, RN_EXPERT) == 0);
            TR_CHECK(rn_layer_ok(x, 2));
            TR_CHECK_EQ_INT(r.bad_align + r.f.bad_align, 0);
            g_rn_fail++;
            tr_experts_free(x);
        }
    }

    /* (fallback) a direct store whose parts are under a page: no runs, every part one read */
    if (align > 1) {
        fake_run_file r;
        memset(&r, 0, sizeof r);
        r.f.align_check = align;
        tr_experts_config cfg = mk_cfg((uint64_t)MIN_SLOTS * tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align),
                                       fake_read, &r, align);
        cfg.readv = fake_readv;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            TR_CHECK(tr_experts_acquire(x, 1, ALL_EXPERTS, N_EXPERT) == 0);
            TR_CHECK_EQ_INT(r.calls, 0);
            TR_CHECK_EQ_INT(r.f.n_calls, N_EXPERT * TR_EXPERT_PARTS);
            for (int64_t e = 0; e < N_EXPERT; e++) TR_CHECK(unit_ok(x, 1, e, align));
            g_rn_fallback++;
            tr_experts_free(x);
        }
    }
}

/* ---- hot: TR_EXPERTS_EVICT_HOT, ds4's streaming cache, against a plain reference ---- */

enum { HOT_PERIOD = 23 };

typedef struct {
    ref_model lru;           /* the recency order and the counters, as the LRU's reference keeps them */
    uint32_t hot[N_UNITS2];
    int64_t calls;
} ref_hot_model;

/* The policy in its definition: every unit a call names +1, or with counts its tokens' routings scaled to
 * at most TR_EXPERTS_HOT_PROMPT tokens, rounded to the nearest (half up), at least 1 (after this call's
 * halving, every HOT_PERIOD calls); a miss takes a free slot, else the resident unit of the lowest hotness
 * not named by this call, the least recent among equals. */
static int ref_hot_acquire(ref_hot_model *h, int64_t layer, const int64_t *ids, const int64_t *counts,
                           int64_t n_tok, int64_t n) {
    ref_model *r = &h->lru;
    if (layer < 0 || layer >= N_LAYERS2 || n < 1 || n > N_EXPERT2) return -1;
    for (int64_t i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= N_EXPERT2) return -1;
        for (int64_t j = 0; j < i; j++)
            if (ids[j] == ids[i]) return -1;
    }
    if (++h->calls % HOT_PERIOD == 0)
        for (int64_t u = 0; u < N_UNITS2; u++) h->hot[u] >>= 1;
    for (int64_t i = 0; i < n; i++) {
        int64_t add = 1;
        if (counts != NULL && n_tok <= TR_EXPERTS_HOT_PROMPT) add = counts[i];
        if (counts != NULL && n_tok > TR_EXPERTS_HOT_PROMPT) {
            int64_t num = counts[i] * TR_EXPERTS_HOT_PROMPT; /* num / n_tok, up when the rest is half or more */
            add = num / n_tok + (2 * (num % n_tok) >= n_tok);
            if (add < 1) add = 1;
        }
        h->hot[layer * N_EXPERT2 + ids[i]] += (uint32_t)add;
    }
    for (int64_t i = 0; i < n; i++) {
        int64_t unit = layer * N_EXPERT2 + ids[i];
        if (r->pos[unit] != -1) {
            ref_move_front(r, r->pos[unit]);
            r->hits++;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        int64_t unit = layer * N_EXPERT2 + ids[i];
        if (r->pos[unit] != -1) continue;
        if (r->count < r->n_slots) {
            r->count++;
        } else {
            int64_t best = -1;
            for (int64_t k = r->count - 1; k >= 0; k--) {
                int64_t u = r->list[k], named = 0;
                for (int64_t j = 0; j < n && !named; j++) named = layer * N_EXPERT2 + ids[j] == u;
                if (named) continue;
                if (best == -1 || h->hot[u] < h->hot[r->list[best]]) best = k;
            }
            r->pos[r->list[best]] = -1;
            r->evictions++;
            memmove(&r->list[best], &r->list[best + 1], (size_t)(r->count - 1 - best) * sizeof(int64_t));
        }
        memmove(&r->list[1], &r->list[0], (size_t)(r->count - 1) * sizeof(int64_t));
        r->list[0] = unit;
        for (int64_t k = 0; k < r->count; k++) r->pos[r->list[k]] = (int32_t)k;
        r->misses++;
    }
    return 0;
}

static int64_t g_hot_diff, g_hot_apart, g_hot_runs, g_hot_own, g_hot_resident, g_hot_scaled, g_hot_prompt;

/* diff:  the store under EVICT_HOT against ref_hot_acquire, the o1 shape and sequence: the same presence
 *        and counters after every call, with halvings every HOT_PERIOD calls; and the LRU's reference on
 *        the same calls hits differently (the policy is not the LRU under another name);
 * runs:  a layer's missing experts read as a run under EVICT_HOT (several victims in one call): every
 *        unit exact, each in a slot of its own (a run's earlier picks are not taken again);
 * own:   the call's own units are the coolest in the store (layer 0 made hot first), and still never its
 *        victims: the call ends with all of them present;
 * prompt: a pass of 320 tokens names layer 0's experts, 300 of them chose expert 0, the others 2-5 each;
 *        the next layer's misses evict layer 0's coolest, and expert 0 is still present after them (ds4's
 *        +1 leaves all at 1 and evicts the least recent, layer 0's first ids, 0 among them). */
static void test_hot(uint64_t align) {
    char err[256];
    {
        fake_file f = mk_fake(align);
        uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
        tr_experts_config cfg = mk_cfg((uint64_t)(N_EXPERT + N_USED) * slot_bytes, fake_read, &f, align);
        cfg.evict = TR_EXPERTS_EVICT_HOT;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            for (int k = 0; k < 5; k++) TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
            static const int64_t two[2] = {0, 1}, three[3] = {0, 1, 2};
            TR_CHECK(tr_experts_acquire(x, 1, two, 2) == 0);   /* the 2 free slots: hotness 1 each */
            TR_CHECK(tr_experts_acquire(x, 1, three, 3) == 0); /* 0 and 1 at 2, layer 0 at 5: 2 needs a victim */
            int all = unit_ok(x, 1, 0, align) && unit_ok(x, 1, 1, align) && unit_ok(x, 1, 2, align);
            TR_CHECK(all);
            tr_experts_stats st;
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.evictions, 1);
            if (all && st.evictions == 1) g_hot_own++;
            tr_experts_free(x);
        }
    }
    {
        build_part_bytes2();
        build_part_offsets2();
        fake_file f = mk_fake(align);
        int64_t n_slots2 = 600;
        tr_experts_config cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.n_layers = N_LAYERS2;
        cfg.n_expert = N_EXPERT2;
        cfg.n_used = N_USED2;
        cfg.part_bytes = part_bytes_table2;
        cfg.part_offset = part_offset_table2;
        cfg.budget_bytes = (uint64_t)n_slots2 * tr_experts_slot_bytes(part_bytes_table2, N_LAYERS2, align);
        cfg.read = fake_read;
        cfg.read_ctx = &f;
        cfg.read_align = align;
        cfg.evict = TR_EXPERTS_EVICT_HOT;
        cfg.hot_period = HOT_PERIOD;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        static ref_hot_model ref;
        static ref_model lru;
        memset(&ref, 0, sizeof ref);
        ref_init(&ref.lru, n_slots2);
        ref_init(&lru, n_slots2);
        rng_state = 0xC0FFEEu;
        int64_t pool[N_EXPERT2], counts[N_EXPERT2], mismatches = 0, checked = 0, scaled = 0;
        for (int it = 0; it < O1_ITERS && x != NULL; it++) {
            for (int64_t i = 0; i < N_EXPERT2; i++) pool[i] = i;
            int64_t n = 1 + (int64_t)(rng_next() % 8);
            if (rng_next() % 20 == 0) n = N_EXPERT2;
            /* fewer layers than the o1 test: units come back, so hotness has something to tell */
            int64_t layer = (int64_t)(rng_next() % 6);
            shuffle_prefix(pool, N_EXPERT2, n);
            /* a third of the calls a prompt's pass: 1-300 tokens, each id chosen by 1..n_tok of them */
            int64_t n_tok = 1 + (int64_t)(rng_next() % 300);
            int with_counts = rng_next() % 3 == 0;
            for (int64_t i = 0; i < n; i++) counts[i] = 1 + (int64_t)(rng_next() % (uint64_t)n_tok);
            if (with_counts && n_tok > TR_EXPERTS_HOT_PROMPT) scaled++;
            int rc = with_counts ? tr_experts_acquire_counts(x, layer, pool, counts, n_tok, n)
                                 : tr_experts_acquire(x, layer, pool, n);
            if (rc != ref_hot_acquire(&ref, layer, pool, with_counts ? counts : NULL, n_tok, n)) mismatches++;
            ref_acquire(&lru, layer, pool, n);
            tr_experts_stats st;
            tr_experts_get_stats(x, &st);
            if (st.hits != ref.lru.hits || st.misses != ref.lru.misses || st.evictions != ref.lru.evictions)
                mismatches++;
            for (int64_t u = 0; u < N_UNITS2; u++)
                if ((tr_experts_part(x, u / N_EXPERT2, u % N_EXPERT2, 0) != NULL) != (ref.lru.pos[u] != -1)) {
                    mismatches++;
                    break;
                }
            checked++;
        }
        TR_CHECK_EQ_INT(mismatches, 0);
        TR_CHECK_EQ_INT(checked, O1_ITERS);
        TR_CHECK(ref.lru.hits != lru.hits);
        TR_CHECK(scaled > 0);
        if (mismatches == 0 && checked == O1_ITERS) g_hot_diff++;
        if (mismatches == 0 && scaled > 0) g_hot_scaled++;
        if (ref.lru.hits != lru.hits) g_hot_apart++;
        tr_experts_free(x);
    }
    {
        /* resident: tr_experts_load_all runs before any acquire (call_seq 0), its runs' picks included */
        fake_run_file r;
        memset(&r, 0, sizeof r);
        r.f.align_check = align;
        tr_experts_config cfg = rn_cfg(RN_LAYERS * RN_EXPERT, &r, align, 0);
        cfg.evict = TR_EXPERTS_EVICT_HOT;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            int ok = tr_experts_load_all(x, NULL, NULL) == 0;
            for (int64_t layer = 0; layer < RN_LAYERS && ok; layer++) ok &= rn_layer_ok(x, layer);
            TR_CHECK(ok);
            if (ok) g_hot_resident++;
            tr_experts_free(x);
        }
    }
    {
        fake_run_file r;
        memset(&r, 0, sizeof r);
        r.f.align_check = align;
        tr_experts_config cfg = rn_cfg(RN_EXPERT + RN_USED + 2, &r, align, 0);
        cfg.evict = TR_EXPERTS_EVICT_HOT;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            int ok = 1;
            static const int64_t all[RN_EXPERT] = {0, 1, 2, 3, 4, 5};
            for (int round = 0; round < 2; round++)
                for (int64_t layer = 0; layer < RN_LAYERS; layer++) {
                    TR_CHECK(tr_experts_acquire(x, layer, all, RN_EXPERT) == 0);
                    ok &= rn_layer_ok(x, layer);
                }
            TR_CHECK(ok);
            TR_CHECK(r.batches > 0); /* the runs' door ran: several victims in one call */
            if (ok && r.batches > 0) g_hot_runs++;
            tr_experts_free(x);
        }
    }
    {
        fake_file f = mk_fake(align);
        uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
        tr_experts_config cfg = mk_cfg((uint64_t)(N_EXPERT + N_USED) * slot_bytes, fake_read, &f, align);
        cfg.evict = TR_EXPERTS_EVICT_HOT;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            int64_t counts[N_EXPERT];
            for (int64_t e = 0; e < N_EXPERT; e++) counts[e] = e == 0 ? 300 : 1 + e;
            TR_CHECK(tr_experts_acquire_counts(x, 0, ALL_EXPERTS, counts, 320, N_EXPERT) == 0);
            /* the next layer's N_EXPERT units fill the N_USED free slots, then evict N_EXPERT - N_USED of layer 0 */
            TR_CHECK(tr_experts_acquire(x, 1, ALL_EXPERTS, N_EXPERT) == 0);
            int kept = tr_experts_part(x, 0, 0, 0) != NULL && unit_ok(x, 0, 0, align);
            TR_CHECK(kept);
            if (kept) g_hot_prompt++;
            tr_experts_free(x);
        }
    }
}

/* ---- the emulated disk (cfg.disk_bytes_per_sec, TR_EXPERT_DISK_MBPS): a slower disk for measurements ---- */

static int64_t g_disk_rate, g_disk_runs, g_disk_shared;

/* Bytes the reader moves for one layer read on demand into a fresh store, no disk emulated. */
static double disk_layer_bytes(uint64_t align, int64_t layer) {
    shared_file f;
    shared_init(&f, align);
    char err[256];
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    tr_experts_config cfg = mk_cfg((uint64_t)PREFETCH_SLOTS * slot_bytes, shared_read, &f, align);
    tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return 0.0;
    TR_CHECK(tr_experts_acquire(x, layer, ALL_EXPERTS, N_EXPERT) == 0);
    tr_experts_free(x);
    return (double)atomic_load(&f.bytes);
}

/* rate:   a layer read on demand, one request a part, takes at least its bytes at the rate (the wall and
 *         the store's read time) and not much more (a timer, not a 15.6 ms tick a request);
 * runs:   a resident load in runs (three requests in flight a call): the same, on the runs' door;
 * shared: a layer read ahead by the I/O thread while the calling thread reads another on demand: the
 *         two layers' bytes at the rate one after the other (one disk), where two clocks would overlap
 *         them and finish in about the larger one's time. */
/* ---- slots: tr_experts_set_slots, the slots in use fewer and more (a session's KV grows into them) ---- */

static int64_t g_sl_hot, g_sl_lru, g_sl_clamp, g_sl_back, g_sl_pending;

/* 14 slots filled in one known order: layer 0 three times (slots 0-5), layer 1's experts 0 and 1 (slots 6-7),
 * layer 2 (slots 8-13): every slot holds a unit, the last two calls' units the most recent */
static tr_experts *slots_filled(tr_experts_config *cfg) {
    char err[256];
    tr_experts *x = tr_experts_create(cfg, err, sizeof err);
    TR_CHECK(x != NULL);
    if (x == NULL) return NULL;
    static const int64_t two[2] = {0, 1};
    for (int k = 0; k < 3; k++) TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
    TR_CHECK(tr_experts_acquire(x, 1, two, 2) == 0);
    TR_CHECK(tr_experts_acquire(x, 2, ALL_EXPERTS, N_EXPERT) == 0);
    return x;
}

static void test_slots(uint64_t align) {
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    tr_experts_stats st;
    /* HOT: 14 -> 10 drops the four coolest, layer 1's two (hotness 1, older) then layer 2's experts 0 and 1
     * (hotness 1, the least recent of their call); layer 0 (hotness 3) stays where it is, and layer 2's
     * experts 2-5, in slots 10-13, move below 10 with their bytes */
    {
        fake_file f = mk_fake(align);
        tr_experts_config cfg = mk_cfg(14 * slot_bytes, fake_read, &f, align);
        cfg.evict = TR_EXPERTS_EVICT_HOT;
        tr_experts *x = slots_filled(&cfg);
        if (x != NULL) {
            TR_CHECK_EQ_INT(tr_experts_set_slots(x, 10), 10);
            tr_experts_get_stats(x, &st);
            int kept = 1, gone = unit_absent(x, 1, 0) && unit_absent(x, 1, 1) && unit_absent(x, 2, 0) &&
                                 unit_absent(x, 2, 1);
            for (int64_t e = 0; e < N_EXPERT; e++) kept &= unit_ok(x, 0, e, align);
            for (int64_t e = 2; e < N_EXPERT; e++) kept &= unit_ok(x, 2, e, align);
            TR_CHECK(kept && gone);
            TR_CHECK_EQ_INT(st.n_slots, 10);
            TR_CHECK_EQ_INT(st.n_slots_made, 14);
            TR_CHECK_EQ_INT(st.slots_given, 4);
            TR_CHECK_EQ_INT(st.moved, 4);
            TR_CHECK_EQ_INT(st.evictions, 4);
            /* the store goes on in 10 slots: a whole layer read in, every unit exact */
            uint64_t calls = (uint64_t)f.n_calls;
            TR_CHECK(tr_experts_acquire(x, 1, ALL_EXPERTS, N_EXPERT) == 0);
            TR_CHECK((uint64_t)f.n_calls > calls);
            int layer1 = 1;
            for (int64_t e = 0; e < N_EXPERT; e++) layer1 &= unit_ok(x, 1, e, align);
            TR_CHECK(layer1);
            if (kept && gone && st.moved == 4 && st.slots_given == 4 && layer1) g_sl_hot++;

            /* clamped: never under the minimum, never over the slots made; the same count changes nothing */
            TR_CHECK_EQ_INT(tr_experts_set_slots(x, 3), MIN_SLOTS);
            TR_CHECK_EQ_INT(tr_experts_set_slots(x, MIN_SLOTS), MIN_SLOTS);
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.slots_given, 14 - MIN_SLOTS);
            int clamp_ok = st.n_slots == MIN_SLOTS && st.slots_given == 14 - MIN_SLOTS;
            /* back: the slots taken again free, and filled as before */
            TR_CHECK_EQ_INT(tr_experts_set_slots(x, 100), 14);
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.slots_taken, 14 - MIN_SLOTS);
            uint64_t ev = st.evictions;
            TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
            TR_CHECK(tr_experts_acquire(x, 2, ALL_EXPERTS, N_EXPERT) == 0);
            tr_experts_get_stats(x, &st);
            int all = 1;
            for (int64_t e = 0; e < N_EXPERT; e++) all &= unit_ok(x, 0, e, align) && unit_ok(x, 2, e, align);
            TR_CHECK(all);
            /* 14 slots, MIN_SLOTS of them held: the 6 taken back free first, so two layers of 12 units need
             * at most 12 - 6 victims */
            TR_CHECK(st.evictions - ev <= 6);
            if (clamp_ok && all) g_sl_clamp++;
            if (all && st.slots_taken == 14 - MIN_SLOTS) g_sl_back++;
            tr_experts_free(x);
        }
    }
    /* LRU: 14 -> 10 drops the four least recent, layer 0's experts 0-3; layer 2's 2-5 move into their slots
     * and keep their recency: the next miss evicts layer 0's expert 4, the least recent left, not a moved unit */
    {
        fake_file f = mk_fake(align);
        tr_experts_config cfg = mk_cfg(14 * slot_bytes, fake_read, &f, align);
        tr_experts *x = slots_filled(&cfg);
        if (x != NULL) {
            TR_CHECK_EQ_INT(tr_experts_set_slots(x, 10), 10);
            int gone = 1, kept = unit_ok(x, 0, 4, align) && unit_ok(x, 0, 5, align) && unit_ok(x, 1, 0, align) &&
                                 unit_ok(x, 1, 1, align);
            for (int64_t e = 0; e < 4; e++) gone &= unit_absent(x, 0, e);
            for (int64_t e = 0; e < N_EXPERT; e++) kept &= unit_ok(x, 2, e, align);
            TR_CHECK(kept && gone);
            static const int64_t one[1] = {2};
            TR_CHECK(tr_experts_acquire(x, 1, one, 1) == 0);
            int next = unit_absent(x, 0, 4) && unit_ok(x, 0, 5, align) && unit_ok(x, 1, 2, align);
            for (int64_t e = 2; e < N_EXPERT; e++) next &= unit_ok(x, 2, e, align);
            TR_CHECK(next);
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.moved, 4);
            if (kept && gone && next && st.moved == 4) g_sl_lru++;
            tr_experts_free(x);
        }
    }
    /* reads ahead still in flight are taken in before any slot moves: layer 1 queued on a slow reader, then
     * 14 -> 10 at once */
    {
        shared_file f;
        shared_init(&f, align);
        f.delay = 0.002;
        char err[256];
        tr_experts_config cfg = mk_cfg((uint64_t)PREFETCH_SLOTS * slot_bytes, shared_read, &f, align);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            TR_CHECK(tr_experts_prefetch_start(x) == 0);
            TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
            TR_CHECK_EQ_INT(tr_experts_prefetch(x, 1, 0), N_EXPERT);
            TR_CHECK_EQ_INT(tr_experts_set_slots(x, 10), 10);
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.prefetched, N_EXPERT); /* all taken in by set_slots, none by an acquire */
            int present = 0, exact = 1;
            for (int64_t l = 0; l < 2; l++)
                for (int64_t e = 0; e < N_EXPERT; e++) {
                    if (unit_absent(x, l, e)) continue;
                    present++;
                    exact &= unit_ok(x, l, e, align);
                }
            TR_CHECK_EQ_INT(present, 10);
            TR_CHECK(exact);
            TR_CHECK(tr_experts_prefetch_wait(x) == 0);
            if (st.prefetched == N_EXPERT && present == 10 && exact) g_sl_pending++;
            tr_experts_free(x);
        }
    }
}

static void test_disk_emulated(uint64_t align) {
    char err[256];
    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    double rate = disk_layer_bytes(align, 1) / 0.04; /* layer 1 in 40 ms */
    TR_CHECK(rate > 0);
    if (!(rate > 0)) return;

    /* rate */
    {
        shared_file f;
        shared_init(&f, align);
        tr_experts_config cfg = mk_cfg((uint64_t)PREFETCH_SLOTS * slot_bytes, shared_read, &f, align);
        cfg.disk_bytes_per_sec = rate;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            double t0 = tr_time_sec();
            TR_CHECK(tr_experts_acquire(x, 1, ALL_EXPERTS, N_EXPERT) == 0);
            double wall = tr_time_sec() - t0;
            double want = (double)atomic_load(&f.bytes) / rate;
            tr_experts_stats st;
            tr_experts_get_stats(x, &st);
            TR_CHECK(layer_ok(x, 1, align));
            TR_CHECK(st.disk_bytes_per_sec == rate);
            TR_CHECK(want > 0.039 && want < 0.041);
            TR_CHECK(wall >= want * 0.999);
            TR_CHECK(st.read_sec >= want * 0.999);
            TR_CHECK(wall < 2 * want + 0.05);
            /* each emulated wait woke at or past its end: the lateness counted, under the read's own time */
            TR_CHECK(st.disk_late_sec > 0 && st.disk_late_sec < want);
            if (layer_ok(x, 1, align) && wall >= want * 0.999 && wall < 2 * want + 0.05 && st.disk_late_sec > 0)
                g_disk_rate++;
            tr_experts_free(x);
        }
    }

    /* runs */
    {
        fake_run_file r;
        memset(&r, 0, sizeof r);
        r.f.align_check = align;
        tr_experts_config cfg = rn_cfg(RN_LAYERS * RN_EXPERT, &r, align, 0);
        cfg.disk_bytes_per_sec = rate;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x != NULL) {
            double t0 = tr_time_sec();
            TR_CHECK(tr_experts_load_all(x, NULL, NULL) == 0);
            double wall = tr_time_sec() - t0;
            tr_experts_stats st;
            tr_experts_get_stats(x, &st);
            double want = (double)st.bytes_read / rate;
            TR_CHECK_EQ_INT(r.batches, RN_LAYERS); /* the runs' door, not the one a part */
            TR_CHECK(rn_layer_ok(x, 0) && rn_layer_ok(x, 1) && rn_layer_ok(x, 2));
            TR_CHECK(want > 0.0);
            TR_CHECK(wall >= want * 0.999);
            TR_CHECK(st.read_sec >= want * 0.999);
            if (r.batches == RN_LAYERS && want > 0.0 && wall >= want * 0.999) g_disk_runs++;
            tr_experts_free(x);
        }
    }

    /* shared */
    {
        shared_file f;
        shared_init(&f, align);
        tr_experts_config cfg = mk_cfg((uint64_t)PREFETCH_SLOTS * slot_bytes, shared_read, &f, align);
        cfg.disk_bytes_per_sec = rate;
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL);
        if (x == NULL) return;
        TR_CHECK(tr_experts_prefetch_start(x) == 0);
        TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
        long long b0 = atomic_load(&f.bytes);
        double t0 = tr_time_sec();
        int64_t queued = tr_experts_prefetch(x, 1, 0); /* the I/O thread reads layer 1 ... */
        TR_CHECK(tr_experts_acquire(x, 2, ALL_EXPERTS, N_EXPERT) == 0); /* ... while this one reads layer 2 */
        TR_CHECK(tr_experts_prefetch_wait(x) == 0);
        double wall = tr_time_sec() - t0;
        double want = (double)(atomic_load(&f.bytes) - b0) / rate;
        TR_CHECK_EQ_INT(queued, N_EXPERT);
        TR_CHECK(want > 0.06); /* both layers: 40 ms and at least 30 more */
        TR_CHECK(wall >= want * 0.999);
        TR_CHECK(tr_experts_acquire(x, 1, ALL_EXPERTS, N_EXPERT) == 0);
        TR_CHECK(layer_ok(x, 1, align));
        if (queued == N_EXPERT && want > 0.06 && wall >= want * 0.999) g_disk_shared++;
        tr_experts_free(x);
    }
}

/* ---- arrival: tr_experts_acquire_async and tr_experts_acquire_take, a call's misses read by the I/O thread ---- */

static int64_t g_ar_twin, g_ar_twin_hot, g_ar_twin_runs, g_ar_twin_partial, g_ar_order, g_ar_fail, g_ar_sync;

/* the twins: one store through tr_experts_acquire_counts, the other through tr_experts_acquire_async and takes
 * of random lengths, the same random calls (ids in increasing order, as the engine gives them): after every call
 * the same units present, the same hits, misses, evictions, bytes and requests, every unit of the call exact */
static void arrival_twins(uint64_t align, int evict, uint64_t slots) {
    char err[256];
    fake_run_file ra, rb;
    memset(&ra, 0, sizeof ra);
    memset(&rb, 0, sizeof rb);
    ra.f.align_check = rb.f.align_check = align;
    tr_experts_config ca = rn_cfg(slots, &ra, align, 0), cb = rn_cfg(slots, &rb, align, 0);
    ca.evict = cb.evict = evict;
    tr_experts *a = tr_experts_create(&ca, err, sizeof err), *b = tr_experts_create(&cb, err, sizeof err);
    TR_CHECK(a != NULL && b != NULL);
    if (a == NULL || b == NULL) {
        tr_experts_free(a);
        tr_experts_free(b);
        return;
    }
    TR_CHECK(tr_experts_prefetch_start(b) == 0);
    rng_state = 0x9e3779b9u ^ (uint32_t)slots ^ (uint32_t)(evict * 77);
    int same = 1, exact = 1;
    for (int call = 0; call < 400 && same && exact; call++) {
        int64_t layer = (int64_t)(rng_next() % RN_LAYERS), n = 1 + (int64_t)(rng_next() % RN_EXPERT), ids[RN_EXPERT];
        int64_t pool[RN_EXPERT] = {0, 1, 2, 3, 4, 5}, counts[RN_EXPERT];
        shuffle_prefix(pool, RN_EXPERT, n);
        int64_t k = 0; /* the chosen ones in increasing order */
        for (int64_t e = 0; e < RN_EXPERT; e++)
            for (int64_t i = 0; i < n; i++)
                if (pool[i] == e) ids[k++] = e;
        int64_t n_tok = rng_next() % 4 == 0 ? 100 : 1;
        for (int64_t i = 0; i < n; i++) counts[i] = n_tok == 1 ? 1 : 1 + (int64_t)(rng_next() % 40);
        TR_CHECK(tr_experts_acquire_counts(a, layer, ids, counts, n_tok, n) == 0);
        int64_t queued = tr_experts_acquire_async(b, layer, ids, counts, n_tok, n);
        TR_CHECK(queued >= 0);
        int none_visible = 1; /* a queued unit is never readable before it is taken in */
        int64_t absent = 0;
        for (int64_t i = 0; i < n; i++)
            if (tr_experts_part(b, layer, ids[i], 0) == NULL) absent++;
        none_visible = absent == queued;
        TR_CHECK(none_visible);
        for (int64_t done = 0; done < n;) {
            int64_t at_least = 1 + (int64_t)(rng_next() % (uint32_t)(n - done));
            int64_t got = tr_experts_acquire_take(b, layer, ids + done, n - done, at_least);
            TR_CHECK(got >= at_least && got <= n - done);
            if (got < at_least) break;
            if (got < n - done) g_ar_twin_partial++;
            done += got;
        }
        for (int64_t i = 0; i < n; i++) exact &= rn_unit_ok(b, layer, ids[i]);
        for (int64_t u = 0; u < RN_LAYERS * RN_EXPERT; u++)
            same &= (tr_experts_part(a, u / RN_EXPERT, u % RN_EXPERT, 0) != NULL) ==
                    (tr_experts_part(b, u / RN_EXPERT, u % RN_EXPERT, 0) != NULL);
        tr_experts_stats sa, sb;
        tr_experts_get_stats(a, &sa);
        tr_experts_get_stats(b, &sb);
        same &= sa.hits == sb.hits && sa.misses == sb.misses && sa.evictions == sb.evictions &&
                sa.bytes_read == sb.bytes_read && sa.requests == sb.requests && sb.arrived == sb.misses &&
                sb.prefetched == 0;
        if (queued > 0) g_ar_twin++;
        if (queued > 0 && evict == TR_EXPERTS_EVICT_HOT) g_ar_twin_hot++;
    }
    TR_CHECK(same);
    TR_CHECK(exact);
    TR_CHECK_EQ_INT(rb.bad_align, 0);
    if (rb.max_pieces > 1) g_ar_twin_runs++; /* the I/O thread joined consecutive misses into runs */
    tr_experts_free(a);
    tr_experts_free(b);
}

static void test_arrival(uint64_t align) {
    for (int evict = TR_EXPERTS_EVICT_LRU; evict <= TR_EXPERTS_EVICT_HOT; evict++) {
        arrival_twins(align, evict, 2 * RN_EXPERT + RN_USED);
        arrival_twins(align, evict, 2 * RN_EXPERT + RN_USED + 2);
    }

    uint64_t slot_bytes = tr_experts_slot_bytes(&PART_BYTES_TABLE[0][0], N_LAYERS, align);
    shared_file f;
    shared_init(&f, align);
    char err[256];
    static const int64_t three[3] = {0, 1, 2};

    /* order: three misses at 0.1 s a read (a unit is three): none landed at once, the first waited for and
     * the next still in flight, then the other two; each exact; counted arrived, not read ahead */
    {
        tr_experts_config cfg = mk_cfg((uint64_t)PREFETCH_SLOTS * slot_bytes, shared_read, &f, align);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 0);
        if (x != NULL) {
            TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
            f.delay = 0.1;
            TR_CHECK_EQ_INT(tr_experts_acquire_async(x, 1, three, NULL, 1, 3), 3);
            int64_t now = tr_experts_acquire_take(x, 1, three, 3, 0);
            int64_t first = tr_experts_acquire_take(x, 1, three, 3, 1);
            int ok = now == 0 && first == 1 && unit_ok(x, 1, 0, align) && unit_absent(x, 1, 1) && unit_absent(x, 1, 2);
            TR_CHECK_EQ_INT(now, 0);
            TR_CHECK_EQ_INT(first, 1);
            TR_CHECK(unit_absent(x, 1, 1) && unit_absent(x, 1, 2));
            TR_CHECK_EQ_INT(tr_experts_acquire_take(x, 1, three + 1, 2, 2), 2);
            f.delay = 0.0;
            ok &= layer_ok(x, 0, align) && unit_ok(x, 1, 0, align) && unit_ok(x, 1, 1, align) && unit_ok(x, 1, 2, align);
            TR_CHECK(ok);
            tr_experts_stats st;
            tr_experts_get_stats(x, &st);
            TR_CHECK_EQ_INT(st.arrived, 3);
            TR_CHECK_EQ_INT(st.prefetched, 0);
            TR_CHECK_EQ_INT(st.misses, N_EXPERT + 3);
            /* the handoffs counted (LESSONS #318): two takes blocked (the first unit, then the last two), each woken
             * after its publish and well within a read; the call waited for the I/O thread's first job; the I/O
             * thread went from one queued unit to the next without sleeping */
            TR_CHECK(st.take_waits >= 2);
            TR_CHECK(st.take_wake_sec > 0 && st.take_wake_sec < 0.1);
            TR_CHECK(st.queue_wake_sec > 0);
            TR_CHECK(st.io_gap_sec > 0 && st.io_gap_sec < 0.1);
            if (ok && st.arrived == 3 && st.take_waits >= 2 && st.io_gap_sec > 0) g_ar_order++;
            tr_experts_free(x);
        }
    }

    /* fail: the I/O thread's first read fails: the take is -1, that unit absent, the others in; the wait reports
     * nothing more (the call already failed), and the unit is read on demand after, exact */
    {
        tr_experts_config cfg = mk_cfg((uint64_t)PREFETCH_SLOTS * slot_bytes, shared_read, &f, align);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 0);
        if (x != NULL) {
            TR_CHECK(tr_experts_acquire(x, 0, ALL_EXPERTS, N_EXPERT) == 0);
            atomic_store(&f.fail_io, 1);
            TR_CHECK_EQ_INT(tr_experts_acquire_async(x, 1, three, NULL, 1, 3), 3);
            TR_CHECK_EQ_INT(tr_experts_acquire_take(x, 1, three, 3, 3), -1);
            int ok = unit_absent(x, 1, 0);
            TR_CHECK(ok);
            TR_CHECK(tr_experts_prefetch_wait(x) == 0);
            ok &= unit_ok(x, 1, 1, align) && unit_ok(x, 1, 2, align);
            TR_CHECK(tr_experts_acquire(x, 1, three, 3) == 0);
            ok &= unit_ok(x, 1, 0, align) && unit_ok(x, 1, 1, align) && unit_ok(x, 1, 2, align) && layer_ok(x, 0, align);
            TR_CHECK(ok);
            if (ok) g_ar_fail++;
            tr_experts_free(x);
        }
        atomic_store(&f.fail_io, 0);
    }

    /* sync: no I/O thread (one slot short of it): the misses read in the call, 0 queued, every unit present */
    {
        tr_experts_config cfg = mk_cfg((uint64_t)(PREFETCH_SLOTS - 1) * slot_bytes, shared_read, &f, align);
        tr_experts *x = tr_experts_create(&cfg, err, sizeof err);
        TR_CHECK(x != NULL && tr_experts_prefetch_start(x) == 1);
        if (x != NULL) {
            TR_CHECK_EQ_INT(tr_experts_acquire_async(x, 1, three, NULL, 1, 3), 0);
            TR_CHECK_EQ_INT(tr_experts_acquire_take(x, 1, three, 3, 3), 3);
            int ok = unit_ok(x, 1, 0, align) && unit_ok(x, 1, 1, align) && unit_ok(x, 1, 2, align);
            TR_CHECK(ok);
            if (ok) g_ar_sync++;
            tr_experts_free(x);
        }
    }
}

int main(void) {
    t_test_thread = 1;
    build_part_offsets();
    rn_build_offsets();

    static const uint64_t aligns[] = {1, TR_FILE_DIRECT_ALIGN};
    for (size_t i = 0; i < sizeof aligns / sizeof aligns[0]; i++) {
        uint64_t align = aligns[i];
        test_arithmetic(align);
        test_contents(align);
        test_lru(align);
        test_full_layer(align);
        test_resident(align);
        test_failure(align);
        test_arg_errors(align);
        test_o1_evidence(align);
        test_few_layers(align);
        test_prefetch(align);
        test_runs(align);
        test_disk_emulated(align);
        test_hot(align);
        test_slots(align);
        test_arrival(align);
    }
    test_align_reuse();
    TR_CHECK(g_ar_twin > 0);
    TR_CHECK(g_ar_twin_hot > 0);
    TR_CHECK(g_ar_twin_runs > 0);
    TR_CHECK(g_ar_twin_partial > 0);
    TR_CHECK(g_ar_order > 0);
    TR_CHECK(g_ar_fail > 0);
    TR_CHECK(g_ar_sync > 0);
    TR_CHECK(g_sl_hot > 0);
    TR_CHECK(g_sl_lru > 0);
    TR_CHECK(g_sl_clamp > 0);
    TR_CHECK(g_sl_back > 0);
    TR_CHECK(g_sl_pending > 0);
    TR_CHECK(g_hot_diff > 0);
    TR_CHECK(g_hot_scaled > 0);
    TR_CHECK(g_hot_prompt > 0);
    TR_CHECK(g_hot_apart > 0);
    TR_CHECK(g_hot_runs > 0);
    TR_CHECK(g_hot_own > 0);
    TR_CHECK(g_hot_resident > 0);
    TR_CHECK(g_disk_rate > 0);
    TR_CHECK(g_disk_runs > 0);
    TR_CHECK(g_disk_shared > 0);
    TR_CHECK(g_rn_shared > 0);
    TR_CHECK(g_rn_unshared > 0);
    TR_CHECK(g_rn_split > 0);
    TR_CHECK(g_rn_demand > 0);
    TR_CHECK(g_rn_ahead > 0);
    TR_CHECK(g_rn_fail > 0);
    TR_CHECK(g_rn_fallback > 0);
    TR_CHECK(g_pf_start > 0);
    TR_CHECK(g_pf_basic > 0);
    TR_CHECK(g_pf_victims > 0);
    TR_CHECK(g_pf_wait > 0);
    TR_CHECK(g_pf_concurrent > 0);
    TR_CHECK(g_pf_fail > 0);
    TR_CHECK(g_pf_fail_acquire > 0);
    TR_CHECK(g_pf_free_in_flight > 0); TR_CHECK(g_pf_hot > 0); TR_CHECK(g_pf_cancel > 0);

    TR_TEST_EXIT();
}
