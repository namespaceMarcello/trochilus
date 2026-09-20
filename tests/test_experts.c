/* test_experts.c — the expert store (src/memory/experts.h): which (layer, expert) units are in
 * RAM under a budget, and the door to disk for the rest (docs/ARCHITETTURA.md "Esecuzione",
 * Esperti (M1)).
 *
 * A fake reader replaces the GGUF file: every byte of every part of every unit is a pure
 * function of its absolute file offset (pattern_byte), so a part read at the wrong offset, or
 * landing in the wrong slot, shows up as a mismatch. It counts its own calls and bytes, and can
 * be told to fail the k-th call, for the failure-injection tests.
 *
 * Each test says which branch it exercises and counts that it was actually taken (CLAUDE.md,
 * LEZIONI #43 #50 #54 #78):
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
 *
 * Every test above align_reuse runs twice, at read_align 1 (buffered, today's layout) and
 * TR_FILE_DIRECT_ALIGN (direct, Step B): same contents, same LRU behaviour, same counters either
 * way. The fake reader also checks that every offset and length it is asked for is itself a
 * multiple of the alignment in effect (fake_file.bad_align), which a store that ignored
 * cfg.read_align could not fake merely by returning the right bytes.
 *
 * Seen red: tools/mutate_experts.sh.
 */
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

    TR_CHECK(tr_experts_load_all(x) == 0);
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
        TR_CHECK(tr_experts_load_all(y) == -1);
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
            TR_CHECK(tr_experts_load_all(x) == -1);
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
        TR_CHECK(tr_experts_load_all(x) == 0);
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

int main(void) {
    build_part_offsets();

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
    }
    test_align_reuse();

    TR_TEST_EXIT();
}
