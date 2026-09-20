/* experts.c — the expert store: RAM-resident units backed by on-demand reads from disk, one
 * doubly linked LRU list holding every slot (free slots at the cold end) and an O(1) unit -> slot
 * index (see experts.h for the contract; docs/ARCHITETTURA.md "Esecuzione" Esperti (M1)). */
#include "experts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../base/platform.h"

/* One slot's occupant and its place in the single LRU list that threads every slot together:
 * used slots in recency order toward lru_head (hot), free slots toward lru_tail (cold). */
typedef struct {
    int32_t unit;        /* -1 when the slot is free */
    int32_t prev, next;  /* -1 at the ends of the list */
} experts_slot;

struct tr_experts {
    int64_t n_layers, n_expert, n_used;
    size_t *part_bytes;           /* own copy: [n_layers][TR_EXPERT_PARTS] */
    uint64_t *part_slot_offset;   /* own: [n_layers][TR_EXPERT_PARTS], each layer's own aligned
                                   * layout inside a slot (layers can need different part sizes) --
                                   * fixed at create time, same for every slot */
    uint64_t *part_offset;        /* own copy: [n_layers][TR_EXPERT_PARTS] */
    uint64_t read_align;          /* 1 (buffered) or TR_FILE_DIRECT_ALIGN (direct); every read this
                                   * store issues has offset, length and buffer address multiples of
                                   * it (src/base/platform.h tr_file_alignment) */
    tr_experts_read_fn read;
    void *read_ctx;

    unsigned char *slab;  /* n_slots * slot_bytes, aligned to read_align (at least 64) */
    int32_t *slot_of;     /* [n_units]: the slot holding this unit, or -1 */
    experts_slot *slots;  /* [n_slots] */
    uint64_t *slot_part_shift; /* [n_slots][TR_EXPERT_PARTS]: recorded at the fill that last loaded
                               * this slot -- how far past its area's own aligned start (in
                               * part_slot_offset) that fill's part bytes actually begin
                               * (file_off - lo). Changes every time the slot is refilled, since a
                               * different (layer, expert) generally has a different file offset
                               * modulo read_align. */
    int32_t lru_head;     /* hot end: most recently touched or filled */
    int32_t lru_tail;     /* cold end: next victim */

    uint64_t *seen_call;  /* [n_units]: stamp of the acquire call that last named this unit */
    uint64_t call_seq;    /* bumped at the start of every tr_experts_acquire. 64 bits: 0, the stamp
                           * of a unit never named, never comes back (32 would wrap in months of
                           * decoding and call a first id a duplicate) */

    tr_experts_stats stats; /* n_units, n_slots, slot_bytes and the running counters */
};

static uint64_t round_up_to(uint64_t n, uint64_t align) {
    return (n + align - 1) / align * align;
}

/* Bytes one part occupies inside a slot (experts.h tr_experts_slot_bytes). read_align == 1: the
 * part's own bytes, rounded up to 64 (today's layout, unchanged). read_align > 1: the part's file
 * offset is not itself aligned, so the read landing it can start up to read_align - 1 bytes early
 * and end up to read_align - 1 bytes late; round_up_to(part_bytes, read_align) + read_align is
 * always room enough for that (the two slacks together are at most 2 * (read_align - 1), and this
 * gives at least read_align, plus whatever round_up_to already added). */
static uint64_t part_area_bytes(uint64_t part_bytes, uint64_t read_align) {
    uint64_t align = read_align > 64 ? read_align : 64;
    uint64_t area = round_up_to(part_bytes, align);
    if (read_align > 1) area += read_align;
    return area;
}

uint64_t tr_experts_slot_bytes(const size_t *part_bytes, int64_t n_layers, uint64_t read_align) {
    uint64_t max_total = 0;
    for (int64_t layer = 0; layer < n_layers; layer++) {
        uint64_t total = 0;
        for (int p = 0; p < TR_EXPERT_PARTS; p++)
            total += part_area_bytes((uint64_t)part_bytes[(size_t)layer * TR_EXPERT_PARTS + (size_t)p], read_align);
        if (total > max_total) max_total = total;
    }
    return max_total;
}

int64_t tr_experts_min_slots(int64_t n_expert, int64_t n_used) {
    return n_expert + n_used;
}

/* ---- the LRU list: unlink a slot from wherever it is, or push it to the hot end ---- */

static void lru_unlink(tr_experts *x, int32_t s) {
    experts_slot *sl = &x->slots[s];
    if (sl->prev != -1) x->slots[sl->prev].next = sl->next; else x->lru_head = sl->next;
    if (sl->next != -1) x->slots[sl->next].prev = sl->prev; else x->lru_tail = sl->prev;
    sl->prev = sl->next = -1;
}

static void lru_push_front(tr_experts *x, int32_t s) {
    experts_slot *sl = &x->slots[s];
    sl->prev = -1;
    sl->next = x->lru_head;
    if (x->lru_head != -1) x->slots[x->lru_head].prev = s;
    x->lru_head = s;
    if (x->lru_tail == -1) x->lru_tail = s;
}

void tr_experts_free(tr_experts *x) {
    if (x == NULL) return;
    free(x->part_bytes);
    free(x->part_slot_offset);
    free(x->part_offset);
    tr_free_aligned(x->slab);
    free(x->slot_of);
    free(x->slots);
    free(x->slot_part_shift);
    free(x->seen_call);
    free(x);
}

tr_experts *tr_experts_create(const tr_experts_config *cfg, char *err, size_t err_len) {
    if (cfg->n_layers <= 0 || cfg->n_expert <= 0 || cfg->n_used <= 0) {
        if (err != NULL)
            snprintf(err, err_len, "expert store: layers, experts and used-per-token must all be positive");
        return NULL;
    }
    for (int64_t layer = 0; layer < cfg->n_layers; layer++) {
        for (int p = 0; p < TR_EXPERT_PARTS; p++) {
            if (cfg->part_bytes[(size_t)layer * TR_EXPERT_PARTS + (size_t)p] == 0) {
                if (err != NULL)
                    snprintf(err, err_len, "expert store: layer %lld part %d has zero bytes", (long long)layer, p);
                return NULL;
            }
        }
    }

    uint64_t read_align = cfg->read_align != 0 ? cfg->read_align : 1;
    uint64_t slot_bytes = tr_experts_slot_bytes(cfg->part_bytes, cfg->n_layers, read_align);
    int64_t n_units = cfg->n_layers * cfg->n_expert;
    int64_t min_slots = tr_experts_min_slots(cfg->n_expert, cfg->n_used);

    uint64_t budget_slots = cfg->budget_bytes / slot_bytes;
    uint64_t n_units_u = (uint64_t)n_units;
    int64_t n_slots = (int64_t)(budget_slots < n_units_u ? budget_slots : n_units_u);

    /* Below the safety margin refuses, unless every unit gets its own slot anyway (n_slots ==
     * n_units): a fully resident store never evicts, so the margin (room for one call's own
     * units never to be their own victim) is moot -- this is the only way a model with few
     * layers (n_units, i.e. n_layers * n_expert, under n_expert + n_used) can run at all. */
    if (n_slots < min_slots && n_slots < n_units) {
        if (err != NULL) {
            double mib = 1024.0 * 1024.0;
            snprintf(err, err_len,
                     "expert budget %.2f MiB holds %lld slots, need at least %lld slots (%.2f MiB minimum)",
                     (double)cfg->budget_bytes / mib, (long long)n_slots, (long long)min_slots,
                     (double)min_slots * (double)slot_bytes / mib);
        }
        return NULL;
    }

    tr_experts *x = (tr_experts *)calloc(1, sizeof *x);
    if (x == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    x->n_layers = cfg->n_layers;
    x->n_expert = cfg->n_expert;
    x->n_used = cfg->n_used;
    x->read_align = read_align;
    x->read = cfg->read;
    x->read_ctx = cfg->read_ctx;
    x->stats.n_units = n_units;
    x->stats.n_slots = n_slots;
    x->stats.slot_bytes = slot_bytes;
    x->stats.direct = read_align > 1;
    x->lru_head = x->lru_tail = -1;

    size_t table_n = (size_t)(cfg->n_layers * TR_EXPERT_PARTS);
    uint64_t slab_align = read_align > 64 ? read_align : 64;
    x->part_bytes = (size_t *)malloc(sizeof(size_t) * table_n);
    x->part_slot_offset = (uint64_t *)malloc(sizeof(uint64_t) * table_n);
    x->part_offset = (uint64_t *)malloc(sizeof(uint64_t) * table_n);
    x->slab = (unsigned char *)tr_alloc_aligned((size_t)((uint64_t)n_slots * slot_bytes), (size_t)slab_align);
    x->slot_of = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_units);
    x->slots = (experts_slot *)malloc(sizeof(experts_slot) * (size_t)n_slots);
    x->slot_part_shift = (uint64_t *)calloc((size_t)n_slots * TR_EXPERT_PARTS, sizeof(uint64_t));
    x->seen_call = (uint64_t *)calloc((size_t)n_units, sizeof(uint64_t));
    if (x->part_bytes == NULL || x->part_slot_offset == NULL || x->part_offset == NULL || x->slab == NULL ||
        x->slot_of == NULL || x->slots == NULL || x->slot_part_shift == NULL || x->seen_call == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory");
        tr_experts_free(x);
        return NULL;
    }
    memcpy(x->part_bytes, cfg->part_bytes, sizeof(size_t) * table_n);
    memcpy(x->part_offset, cfg->part_offset, sizeof(uint64_t) * table_n);

    for (int64_t layer = 0; layer < cfg->n_layers; layer++) {
        uint64_t off = 0;
        for (int p = 0; p < TR_EXPERT_PARTS; p++) {
            size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)p;
            x->part_slot_offset[idx] = off;
            off += part_area_bytes((uint64_t)cfg->part_bytes[idx], read_align);
        }
    }

    for (int64_t u = 0; u < n_units; u++) x->slot_of[u] = -1;
    for (int64_t s = 0; s < n_slots; s++) {
        x->slots[s].unit = -1;
        x->slots[s].prev = x->slots[s].next = -1;
    }
    /* every slot starts free: pushed front in index order, so slot 0 ends at the cold end and
     * slot n_slots - 1 at the hot end -- the first eviction picks slot 0, then 1, 2, ... */
    for (int64_t s = 0; s < n_slots; s++) lru_push_front(x, (int32_t)s);

    return x;
}

int tr_experts_resident(const tr_experts *x) {
    return x->stats.n_slots == x->stats.n_units;
}

/* Evicts the coldest slot (freeing its unit, if it holds one, evictions++ regardless of what
 * follows) and reads (layer, id)'s parts into it. -1 on a failed part read: the slot is left
 * free at the cold end (it is never linked to the hot end until every part succeeds), `unit`
 * stays absent, and every part read before the failure already counted in bytes_read. */
static int experts_fill_slot(tr_experts *x, int64_t layer, int64_t id, int64_t unit) {
    int32_t victim = x->lru_tail;
    experts_slot *sl = &x->slots[victim];
    if (sl->unit != -1) {
        x->slot_of[sl->unit] = -1;
        sl->unit = -1;
        x->stats.evictions++;
    }

    unsigned char *base = x->slab + (uint64_t)victim * x->stats.slot_bytes;
    uint64_t align = x->read_align;
    for (int p = 0; p < TR_EXPERT_PARTS; p++) {
        size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)p;
        uint64_t file_off = x->part_offset[idx] + (uint64_t)x->part_bytes[idx] * (uint64_t)id;
        /* align == 1: lo == file_off, hi == file_off + part_bytes -- today's exact behaviour, same
         * formula, no branch. align > 1: the aligned range the read must actually cover. */
        uint64_t lo = file_off / align * align;
        uint64_t hi = (file_off + (uint64_t)x->part_bytes[idx] + align - 1) / align * align;
        size_t n = (size_t)(hi - lo);
        double t0 = tr_time_sec();
        int rc = x->read(x->read_ctx, base + x->part_slot_offset[idx], n, lo);
        x->stats.read_sec += tr_time_sec() - t0;
        if (rc != 0) return -1;
        x->stats.bytes_read += n;
        x->slot_part_shift[(size_t)victim * TR_EXPERT_PARTS + (size_t)p] = file_off - lo;
    }

    x->slot_of[unit] = victim;
    sl->unit = (int32_t)unit;
    lru_unlink(x, victim);
    lru_push_front(x, victim);
    x->stats.misses++;
    return 0;
}

int tr_experts_load_all(tr_experts *x) {
    if (!tr_experts_resident(x)) return -1;
    for (int64_t layer = 0; layer < x->n_layers; layer++) {
        for (int64_t id = 0; id < x->n_expert; id++) {
            int64_t unit = layer * x->n_expert + id;
            if (experts_fill_slot(x, layer, id, unit) != 0) return -1;
        }
    }
    return 0;
}

int tr_experts_acquire(tr_experts *x, int64_t layer, const int64_t *ids, int64_t n) {
    if (layer < 0 || layer >= x->n_layers) return -1;
    if (n < 1 || n > x->n_expert) return -1;

    /* range and duplicate check first, touching only the per-unit call stamp: on any failure
     * the store (slots, LRU, stats) is untouched */
    uint64_t stamp = ++x->call_seq;
    for (int64_t i = 0; i < n; i++) {
        int64_t id = ids[i];
        if (id < 0 || id >= x->n_expert) return -1;
        int64_t unit = layer * x->n_expert + id;
        if (x->seen_call[unit] == stamp) return -1; /* duplicate within this call */
        x->seen_call[unit] = stamp;
    }

    /* first pass: touch every unit already present, in the order given */
    for (int64_t i = 0; i < n; i++) {
        int64_t unit = layer * x->n_expert + ids[i];
        int32_t slot = x->slot_of[unit];
        if (slot != -1) {
            lru_unlink(x, slot);
            lru_push_front(x, slot);
            x->stats.hits++;
        }
    }

    /* second pass: read the missing ones, in the order given. n_slots > n (tr_experts_create's
     * minimum), and the first pass already made every present unit of this call hot, so the
     * victim taken here is never one of this call's own units. */
    for (int64_t i = 0; i < n; i++) {
        int64_t id = ids[i];
        int64_t unit = layer * x->n_expert + id;
        if (x->slot_of[unit] != -1) continue;
        if (experts_fill_slot(x, layer, id, unit) != 0) return -1;
    }
    return 0;
}

const void *tr_experts_part(const tr_experts *x, int64_t layer, int64_t expert, int part) {
    if (layer < 0 || layer >= x->n_layers || expert < 0 || expert >= x->n_expert || part < 0 ||
        part >= TR_EXPERT_PARTS)
        return NULL;
    int32_t slot = x->slot_of[layer * x->n_expert + expert];
    if (slot == -1) return NULL;
    size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)part;
    return x->slab + (uint64_t)slot * x->stats.slot_bytes + x->part_slot_offset[idx] +
           x->slot_part_shift[(size_t)slot * TR_EXPERT_PARTS + (size_t)part];
}

void tr_experts_get_stats(const tr_experts *x, tr_experts_stats *out) {
    *out = x->stats;
}

void tr_experts_get_reader(const tr_experts *x, tr_experts_read_fn *read, void **ctx) {
    *read = x->read;
    *ctx = x->read_ctx;
}

void tr_experts_set_reader(tr_experts *x, tr_experts_read_fn read, void *ctx) {
    x->read = read;
    x->read_ctx = ctx;
}
