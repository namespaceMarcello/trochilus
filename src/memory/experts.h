/* experts.h — the expert store: which experts are in RAM, and the door to the disk for the
 * others (milestone M1, docs/ARCHITECTURE.md "Esecuzione", docs/MEASUREMENTS.md "M1, prima di scrivere
 * codice").
 *
 * A unit is one expert of one layer: its TR_EXPERT_PARTS matrices (gate, up, down), bytes exactly
 * as they are in the file, in one slot. Every slot is allocated when the store is created, as
 * many as the budget holds; nothing is allocated afterwards. After the router has chosen, the
 * graph asks for a layer's units (tr_experts_acquire): the ones present are touched, the missing
 * ones are read at once, one after the other, on the calling thread, into the slot of the unit
 * the policy gives up (cfg.evict: the least recently used, or ds4's coolest). Index unit -> slot and
 * LRU list are O(1); ds4's victim is a walk of the slots.
 *
 * One path: with a budget that holds every unit the store is filled at load (tr_experts_load_all)
 * and nothing is ever missing or evicted, which is the engine before M1. The same bytes reach the
 * same kernels whatever the budget, so no logit depends on it (tests/test_experts.c).
 *
 * Why this shape and not the one the sources have (measured, docs/MEASUREMENTS.md): an LRU reads less
 * than a pin learned from usage at every capacity; one reader gets the whole bandwidth of the
 * disk, eight get no more, so there is one I/O thread and only for what can be overlapped: a
 * prompt reads the next layer's units while the current one computes (tr_experts_prefetch, below;
 * in one pass in short requests, what the next layer does not ask dropped unread:
 * tr_experts_prefetch_cancel). In decode a prefetch would cost more reads than it saves waiting.
 *
 * Not thread-safe, like the model's pool: one evaluation at a time per store, every call from the
 * same thread. No global state. */
#ifndef TR_EXPERTS_H
#define TR_EXPERTS_H

#include <stddef.h>
#include <stdint.h>

#define TR_EXPERT_PARTS 3
/* The most one request of a run asks for (tr_experts_config.readv): the disk is flat from 64 MiB
 * on (3.5 GB/s at 64 and 96 MiB against 1.9 at one expert matrix, docs/MEASUREMENTS.md §The disk at
 * its limit), so a layer's 136 MiB part goes in two requests of 68. */
#define TR_EXPERTS_RUN_BYTES ((uint64_t)96 << 20)

/* Which unit a miss evicts (tr_experts_config.evict). LRU: the least recently used, the store's own
 * since M1. HOT: ds4's streaming cache (ref/ds4 ds4_metal.m:15404-15439): the lowest hotness, the
 * least recent among equals; a unit's hotness grows by one at every call naming it, hit or miss, and
 * every unit's is halved each TR_EXPERTS_HOT_TOKENS tokens' calls (hot_period). Below one token's
 * units an LRU over the layers' cycle evicts each unit just before it comes back (LESSONS #271): on
 * OLMoE's route trace at 83 slots the LRU hits 0 of 128 a token, HOT 47 (tools/evict_replay.py).
 * A call of more tokens (tr_experts_acquire_counts, a prompt's pass) adds to each unit the routings of
 * its tokens that chose it, as if from at most TR_EXPERTS_HOT_PROMPT tokens. ds4 adds a prefill's
 * routings whole (ds4_metal.m:17437-17447); the store of 10-03 added 1 a call, so a prompt left every
 * unit at 1 and the decode began on the prompt's last layers (the 8 GB machine's Q4_K replay: 12.5
 * misses a token in the first 32, 3.8 scaled, 4.3 whole; over 256 tokens 7.09, 5.96, 5.81; over 7
 * traces scaled has 7% fewer than whole in the first 32, 1% over 256: a long prompt's whole counts
 * outlive many halvings, LESSONS #287, #289). */
#define TR_EXPERTS_EVICT_LRU 0
#define TR_EXPERTS_EVICT_HOT 1
#define TR_EXPERTS_HOT_TOKENS 16
#define TR_EXPERTS_HOT_PROMPT 64

typedef struct tr_experts tr_experts;

/* The store's door to the disk: n bytes at `offset` into buf, 0 on success. The engine passes
 * tr_file_pread on the model's file; a test passes a reader that counts, or fails. */
typedef int (*tr_experts_read_fn)(void *ctx, void *buf, size_t n, uint64_t offset);
/* Its door for a run: n requests at once (a run's three parts, n = TR_EXPERT_PARTS), each the file's
 * bytes from its offset on into its pieces in order, all in flight together where the system allows
 * (tr_file_preadv_n, src/base/platform.h, with its scratch: tr_file_preadv_n_scratch(3 (run bytes +
 * 2 pages), 3) bytes, the store's own). 0, or -1 when any request failed, with none still writing.
 * Not included here: a header reached through four ".." comes out of MinGW's gcc as an absolute path
 * in the .d files, which the container's make refuses. */
struct tr_readv_req;
typedef int (*tr_experts_readv_fn)(void *ctx, const struct tr_readv_req *req, int n, void *scratch);

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
    tr_experts_readv_fn readv;            /* NULL: every part of every unit is one read through `read`.
                                           * Set (on read_ctx): consecutive experts of a layer, read
                                           * together at load (tr_experts_load_all) and ahead
                                           * (tr_experts_prefetch), are one request a part of at most
                                           * TR_EXPERTS_RUN_BYTES, the three in flight together, each
                                           * expert's pages scattered into its own slot; a layer whose
                                           * part is under read_align bytes
                                           * keeps the reads one at a time. The same bytes land in
                                           * every slot either way. */
    uint64_t run_bytes;                   /* the most one request of a run asks for: 0 is
                                           * TR_EXPERTS_RUN_BYTES (tests set less, to split runs) */
    int evict;                            /* which unit a miss evicts: TR_EXPERTS_EVICT_LRU (0) or
                                           * TR_EXPERTS_EVICT_HOT */
    int64_t hot_period;                   /* EVICT_HOT: successful acquire calls between two halvings
                                           * of every unit's hotness; 0: TR_EXPERTS_HOT_TOKENS * n_layers */
    double disk_bytes_per_sec;            /* > 0, measurement only (TR_EXPERT_DISK_MBPS, tools/machines.sh):
                                           * a slower disk emulated on this one. Every request holds
                                           * one emulated disk, shared by the calling thread and the I/O
                                           * thread, for its bytes at this rate, from when it is issued
                                           * or the disk is free, and returns at the end: a request
                                           * takes max(the real read, its turn on the slower disk).
                                           * 0: off, the real disk alone */
} tr_experts_config;

typedef struct {
    int64_t n_units, n_slots;             /* n_layers * n_expert, and how many of them fit */
    uint64_t slot_bytes;                  /* tr_experts_slot_bytes */
    uint64_t hits, misses, evictions;     /* units asked for and present / read / thrown out */
    uint64_t bytes_read;                  /* actually transferred: rounded up to read_align, so it
                                           * can exceed the units' own bytes on a direct store */
    uint64_t requests;                    /* calls of the reader, read and readv together */
    double read_sec;                      /* time inside the reader (the I/O thread's too) */
    int direct;                           /* 1: cfg.read_align asked for aligned reads (a direct file
                                           * handle is behind read_ctx); 0: an ordinary buffered one */
    uint64_t prefetched;                  /* units the I/O thread read ahead (tr_experts_prefetch),
                                           * counted in misses and bytes_read too once taken in */
    uint64_t arrived;                     /* units a call's I/O thread read while it computed the present
                                           * ones (tr_experts_acquire_async), in misses and bytes_read too */
    /* the handoffs between the calling thread and the I/O thread (LESSONS #318), in seconds: */
    uint64_t take_waits;                  /* takes of a unit still in flight: the calling thread blocked */
    double take_wake_sec;                 /* their wakes: from the I/O thread's publish to the take's return */
    double queue_wake_sec;                /* a call's wait for the I/O thread to take its first job */
    double io_gap_sec;                    /* the I/O thread between a job's publish and its next read, the queue
                                           * not empty (its own time between two requests) */
    double disk_late_sec;                 /* the emulated disk's sleeps woken past their end, both threads */
    double prefetch_wait_sec;             /* time tr_experts_acquire and tr_experts_prefetch_wait
                                           * spent waiting for units still in flight */
    uint64_t cancelled;                   /* units queued ahead and dropped before a byte of them was
                                           * read (tr_experts_prefetch_cancel): never counted as read */
    double touch_sec;                     /* time of tr_experts_touch, the slots' pages faulted */
    double disk_bytes_per_sec;            /* cfg.disk_bytes_per_sec: > 0 while a slower disk is emulated */
    int evict;                            /* cfg.evict: the policy a miss evicts by */
    uint64_t hot_extra;                   /* EVICT_HOT: hotness the calls added beyond one a unit named
                                           * (tr_experts_acquire_counts: a prompt's routings) */
    int64_t n_slots_made;                 /* the slots the store was created with; n_slots of them in use */
    uint64_t slots_given, slots_taken;    /* slots given back to the system, and taken again
                                           * (tr_experts_set_slots) */
    uint64_t moved;                       /* units moved to another slot when theirs was given back */
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
 * or a store that is not resident. on_unit (may be NULL) is called after every unit read, with
 * that unit's own bytes (the sum of its parts' part_bytes, not the aligned transfer
 * tr_experts_stats.bytes_read counts): how the load reports its progress (model.h tr_progress). */
typedef void (*tr_experts_unit_fn)(void *ctx, uint64_t unit_bytes);
int tr_experts_load_all(tr_experts *x, tr_experts_unit_fn on_unit, void *ctx);
/* Every page of the slots written once with the byte it holds, the pages split over the pool: a
 * fresh page's first fault (the system zeroing it) taken by the pool's threads, not inside the
 * disk's request. Contents unchanged; not while reads ahead are in flight. pool NULL: serial. */
struct tr_pool;
void tr_experts_touch(tr_experts *x, struct tr_pool *pool);

/* The slots in use, n clamped to [tr_experts_min_slots (or every slot of a smaller store), the slots
 * created]: a session's KV grows into the pages of the slots given back (src/models/olmoe.c kv_hold).
 * Fewer: the policy's coldest units go until n slots hold the others (evictions), the ones above n move
 * into slots below it (moved, their bytes copied whole), and the pages of slots [n, n_slots) go back to the
 * system (tr_pages_release). More: slots [n_slots, n) taken back free (tr_pages_commit; when the system
 * refuses, nothing changes). Reads ahead still in flight are taken in first. Between calls only: a
 * unit's tr_experts_part may move. The same bytes reach the same kernels. Returns the slots in use. */
int64_t tr_experts_set_slots(tr_experts *x, int64_t n);

/* The experts ids[0..n) of `layer` (distinct, 0 <= id < n_expert, 1 <= n <= n_expert) are
 * wanted now. Present: touched. Missing: read, in the order given, each into the slot of the
 * unit used least recently; a unit this call asked for is never the victim. Missing ones given
 * one after the other with consecutive ids are a run (cfg.readv): one request a part. On success
 * (0) tr_experts_part is valid for these units until the next tr_experts_acquire. -1 on a failed
 * read: the units of the failed request are absent, every other unit is as it was or freshly read,
 * the store stays consistent and the next call may try again. Allocates nothing. */
int tr_experts_acquire(tr_experts *x, int64_t layer, const int64_t *ids, int64_t n);
/* tr_experts_acquire for a call of n_tok tokens, counts[i] of them routed to ids[i] (NULL: one each):
 * under TR_EXPERTS_EVICT_HOT each unit's hotness grows by counts[i] x min(1, TR_EXPERTS_HOT_PROMPT /
 * n_tok), rounded half up, at least 1. tr_experts_acquire is this with NULL and 1. */
int tr_experts_acquire_counts(tr_experts *x, int64_t layer, const int64_t *ids, const int64_t *counts,
                              int64_t n_tok, int64_t n);
/* tr_experts_acquire_counts with the missing units read by the I/O thread while the caller computes the present
 * ones (docs/MEASUREMENTS.md §The arrival order built): the same hotness, the same victims, the same runs and
 * bytes, the reads queued in the order given under one wake, and the call returns once the I/O thread has taken
 * the first. Returns the units queued (0: every one present), -1 as tr_experts_acquire_counts. Until
 * tr_experts_acquire_take takes a queued unit in, tr_experts_part gives NULL for it and it counts as present (no
 * victim, no second read). Reading ahead not started (tr_experts_prefetch_start): tr_experts_acquire_counts,
 * 0 on success. Allocates nothing. */
int64_t tr_experts_acquire_async(tr_experts *x, int64_t layer, const int64_t *ids, const int64_t *counts,
                                 int64_t n_tok, int64_t n);
/* Takes in the units (layer, ids[i]) of a tr_experts_acquire_async call in the order given (the order its reads
 * land): the first at_least waited for, then each next one only while its read has already landed. A unit not
 * in flight is passed over. Returns how many of ids were passed (at_least <= it <= n; n: all in), or -1 if a
 * read failed or a unit is absent: the failed unit absent, the store consistent. Allocates nothing. */
int64_t tr_experts_acquire_take(tr_experts *x, int64_t layer, const int64_t *ids, int64_t n, int64_t at_least);
/* Part p of (layer, expert), 64-byte aligned on a buffered store (read_align == 1); on a direct
 * store its address follows wherever that part's bytes actually start inside its own aligned read,
 * which moves every time the slot is refilled, and carries no alignment guarantee of its own. NULL
 * when the unit is not in RAM. */
const void *tr_experts_part(const tr_experts *x, int64_t layer, int64_t expert, int part);

void tr_experts_get_stats(const tr_experts *x, tr_experts_stats *out);

/* ---- reading ahead (docs/ARCHITECTURE.md Esperti M1: disk and computation overlapped) ----
 * One I/O thread per store, started once and joined by tr_experts_free, fills slots reserved for
 * it while the calling thread computes. Everything else stays on the calling thread: the index,
 * the LRU, the stats, the choice of victims; the I/O thread only reads into a reserved slot's
 * bytes and publishes that it is done. A reserved unit counts as present: tr_experts_acquire
 * waits for it if it is still in flight, tr_experts_part gives NULL until it has been taken in,
 * and no victim is ever a slot in flight. Once started, the reader may be called from the I/O
 * thread and the calling thread at the same time (tr_file_pread may: positional reads).
 *
 * 0: started (or already running). 1: nothing to do -- the store is resident, or it holds fewer
 * than 2 * n_expert + n_used slots, too few for one layer's units, the next layer's, and a
 * token's margin; nothing changes, every read stays on demand. -1: out of memory, or no thread. */
int tr_experts_prefetch_start(tr_experts *x);
int tr_experts_prefetching(const tr_experts *x);
/* Reserves a slot for every unit of `layer` not in RAM and hands it to the I/O thread, in id
 * order (file order). A victim is the coldest slot not in flight and holding no unit of `layer`
 * nor of `keep` (the layer computing now: its units are never evicted for the next one's); it
 * stops at the first unit no such victim is left for. Returns the units queued; 0 when reading
 * ahead is not started. Allocates nothing. */
int64_t tr_experts_prefetch(tr_experts *x, int64_t layer, int64_t keep);
/* tr_experts_prefetch with the I/O thread's requests at most run_bytes a part (0: the store's own,
 * cfg.run_bytes): short requests leave the rest of the layer in the queue, where
 * tr_experts_prefetch_cancel can still drop what the layer's call will not ask for. Both return once
 * the I/O thread has taken the first unit queued (one wake): the read starts before the caller
 * computes on, whatever else holds the cores. */
int64_t tr_experts_prefetch_n(tr_experts *x, int64_t layer, int64_t keep, uint64_t run_bytes);
/* Drops every unit of `layer` still queued ahead (no byte of it read yet) that ids[0..n) does not
 * name: its slot free at the cold end, its unit absent, never counted as read. A unit already in
 * flight is left to land. For the layer's only call (a pass's), before it, so it never waits
 * behind a read nobody asked for. Returns the units dropped; 0 when reading ahead is not started.
 * Allocates nothing. */
int64_t tr_experts_prefetch_cancel(tr_experts *x, int64_t layer, const int64_t *ids, int64_t n);
/* Waits until nothing is in flight and takes every unit in. 0, or -1 if a read ahead failed
 * since the last call: that unit is absent again (the next acquire reads it on demand), the store
 * consistent. tr_experts_acquire returns -1 too for a unit it asked for whose read ahead failed. */
int tr_experts_prefetch_wait(tr_experts *x);

/* Tests only: swaps both doors out (to wrap them, e.g. to fail the k-th call) and back; they share
 * one ctx. A readv set on a store created without one stays unused (no run memory). */
void tr_experts_get_reader(const tr_experts *x, tr_experts_read_fn *read, tr_experts_readv_fn *readv, void **ctx);
void tr_experts_set_reader(tr_experts *x, tr_experts_read_fn read, tr_experts_readv_fn readv, void *ctx);

#endif
