/* experts.h — the expert store: which experts are in RAM, and the door to the disk for the
 * others (milestone M1, docs/ARCHITECTURE.md "Esecuzione", docs/MEASUREMENTS.md "M1, prima di scrivere
 * codice").
 *
 * A unit is one expert of one layer: its TR_EXPERT_PARTS matrices (gate, up, down), bytes exactly
 * as they are in the file, in one slot. Every slot is allocated when the store is created, as
 * many as the budget holds; nothing is allocated afterwards. After the router has chosen, the
 * graph asks for a layer's units (tr_experts_acquire): the ones present are touched, the missing
 * ones are read at once, one after the other, on the calling thread, into the slot of the unit
 * used least recently. Index unit -> slot and LRU list are O(1).
 *
 * One path: with a budget that holds every unit the store is filled at load (tr_experts_load_all)
 * and nothing is ever missing or evicted, which is the engine before M1. The same bytes reach the
 * same kernels whatever the budget, so no logit depends on it (tests/test_experts.c).
 *
 * Why this shape and not the one the sources have (measured, docs/MEASUREMENTS.md): an LRU reads less
 * than a pin learned from usage at every capacity; one reader gets the whole bandwidth of the
 * disk, eight get no more, so there is no I/O thread until there is something to overlap; and the
 * only thing to overlap would be a prefetch, which on a 1.5 GB/s disk costs more reads than it
 * saves waiting.
 *
 * Not thread-safe, like the model's pool: one evaluation at a time per store. No global state. */
#ifndef TR_EXPERTS_H
#define TR_EXPERTS_H

#include <stddef.h>
#include <stdint.h>

#define TR_EXPERT_PARTS 3

typedef struct tr_experts tr_experts;

/* The store's only door to the disk: n bytes at `offset` into buf, 0 on success. The engine
 * passes tr_file_pread on the model's file; a test passes a reader that counts, or fails. */
typedef int (*tr_experts_read_fn)(void *ctx, void *buf, size_t n, uint64_t offset);

typedef struct {
    int64_t n_layers, n_expert;
    int64_t n_used;                       /* experts a token uses in a layer: sets the smallest store */
    const size_t *part_bytes;             /* [n_layers][TR_EXPERT_PARTS]: a GGUF file may quantize
                                           * layers differently, so a part's bytes can differ layer to
                                           * layer; expert e of a (layer, part) is part_bytes[..] * e
                                           * further into that (layer, part)'s block */
    const uint64_t *part_offset;          /* [n_layers][TR_EXPERT_PARTS]: where expert 0's part starts in
                                           * the file */
    uint64_t budget_bytes;                /* RAM for the slots; more than every unit needs is not used */
    tr_experts_read_fn read;
    void *read_ctx;
    uint64_t read_align;                  /* what every read through `read` must land on: 1 for an
                                           * ordinary (buffered) handle, TR_FILE_DIRECT_ALIGN
                                           * (src/base/platform.h) for one opened with
                                           * tr_file_open_direct -- offset, length and the slot's own
                                           * buffer address are then all multiples of it */
} tr_experts_config;

typedef struct {
    int64_t n_units, n_slots;             /* n_layers * n_expert, and how many of them fit */
    uint64_t slot_bytes;                  /* tr_experts_slot_bytes */
    uint64_t hits, misses, evictions;     /* units asked for and present / read / thrown out */
    uint64_t bytes_read;                  /* actually transferred: rounded up to read_align, so it
                                           * can exceed the units' own bytes on a direct store */
    double read_sec;                      /* time inside the reader */
    int direct;                           /* 1: cfg.read_align asked for aligned reads (a direct file
                                           * handle is behind read_ctx); 0: an ordinary buffered one */
} tr_experts_stats;

/* Bytes of one slot: for each of the n_layers layers, its own parts one after the other, the
 * largest layer's total, since one slot must hold any layer's unit. part_bytes: [n_layers]
 * [TR_EXPERT_PARTS]. read_align == 1: each part starts on a 64-byte boundary, exactly its own
 * bytes (today's layout). read_align > 1 (TR_FILE_DIRECT_ALIGN): a part's offset in the file is
 * not itself aligned, so the read landing it can start up to read_align - 1 bytes early and end up
 * to read_align - 1 bytes late; each part therefore gets align_up(part_bytes, read_align) +
 * read_align bytes, starting at a read_align-aligned position, which is always room enough. */
uint64_t tr_experts_slot_bytes(const size_t *part_bytes, int64_t n_layers, uint64_t read_align);
/* The smallest store that can run: one pass may ask for every expert of a layer at once (a
 * prompt), and the units it asks for are never evicted by that same call. n_expert + n_used. */
int64_t tr_experts_min_slots(int64_t n_expert, int64_t n_used);

/* NULL with a message in err when the shape is not positive, the budget holds fewer than
 * tr_experts_min_slots slots and also fewer than every unit (a model with few enough layers that
 * n_layers * n_expert < tr_experts_min_slots(...) can still run fully resident: nothing is ever
 * evicted there, so the margin does not apply), or memory runs out. Copies cfg (part_offset
 * included). Reads nothing. */
tr_experts *tr_experts_create(const tr_experts_config *cfg, char *err, size_t err_len);
void tr_experts_free(tr_experts *x);

/* 1 when every unit has a slot: after tr_experts_load_all nothing is read again. */
int tr_experts_resident(const tr_experts *x);
/* Reads every unit, in file order. Only for a resident store, at load; -1 on a failed read
 * or a store that is not resident. */
int tr_experts_load_all(tr_experts *x);

/* The experts ids[0..n) of `layer` (distinct, 0 <= id < n_expert, 1 <= n <= n_expert) are
 * wanted now. Present: touched. Missing: read, in the order given, each into the slot of the
 * unit used least recently; a unit this call asked for is never the victim. On success (0)
 * tr_experts_part is valid for these units until the next tr_experts_acquire. -1 on a failed
 * read: the unit that failed is absent, every other unit is as it was or freshly read, the store
 * stays consistent and the next call may try again. Allocates nothing. */
int tr_experts_acquire(tr_experts *x, int64_t layer, const int64_t *ids, int64_t n);
/* Part p of (layer, expert), 64-byte aligned on a buffered store (read_align == 1); on a direct
 * store its address follows wherever that part's bytes actually start inside its own aligned read,
 * which moves every time the slot is refilled, and carries no alignment guarantee of its own. NULL
 * when the unit is not in RAM. */
const void *tr_experts_part(const tr_experts *x, int64_t layer, int64_t expert, int part);

void tr_experts_get_stats(const tr_experts *x, tr_experts_stats *out);

/* Tests only: swaps the reader out (to wrap it, e.g. to fail the k-th call) and back. */
void tr_experts_get_reader(const tr_experts *x, tr_experts_read_fn *read, void **ctx);
void tr_experts_set_reader(tr_experts *x, tr_experts_read_fn read, void *ctx);

#endif
