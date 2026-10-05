/* experts.c — the expert store: RAM-resident units backed by on-demand reads from disk, one
 * doubly linked LRU list holding every slot (free slots at the cold end) and an O(1) unit -> slot
 * index (see experts.h for the contract; docs/ARCHITECTURE.md "Esecuzione" Esperti (M1)). */
#include "experts.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../base/platform.h"
#include "../base/threads.h"

/* One slot's occupant and its place in the single LRU list that threads every slot together:
 * used slots in recency order toward lru_head (hot), free slots toward lru_tail (cold). */
typedef struct {
    int32_t unit;        /* -1 when the slot is free */
    int32_t prev, next;  /* -1 at the ends of the list */
    uint64_t pick;       /* the acquire call that last took it as a victim: a run's next pick skips it */
} experts_slot;

/* A slot's read ahead (tr_experts_prefetch). Every field is read and written under the store's
 * monitor: the calling thread fills the first group and queues the slot, the I/O thread copies
 * them, reads without the lock, and writes the second group with done = 1. */
typedef struct {
    int64_t layer, id;
    int64_t cap; /* the most units one run of it carries (tr_experts_prefetch_n), 0: run_take's */
    int demand;  /* 1: a call's own unit (tr_experts_acquire_async), counted arrived; 0: read ahead */
    tr_experts_read_fn read;
    tr_experts_readv_fn readv;
    void *read_ctx;
    int done, rc;
    uint64_t bytes, requests; /* a run's whole count on its first job, 0 on the others */
    double sec;
    double done_at; /* tr_time_sec when the I/O thread published it done */
    uint64_t shift[TR_EXPERT_PARTS];
} experts_job;

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
    tr_experts_readv_fn readv;    /* NULL: no runs */
    void *read_ctx;
    int64_t *run_units;           /* [n_layers]: experts one request of a run may carry; < 2: no runs */
    /* a run's working memory, one set for the calling thread (load) and one for the I/O thread */
    void *scratch[2];             /* tr_file_preadv_n_scratch(3 (run bytes + 2 pages), 3) */
    tr_iov *iov[2];               /* [TR_EXPERT_PARTS][n_expert]: a run's three requests at once */
    int32_t *run_slot[2];         /* [n_expert] */
    uint64_t (*run_shift[2])[TR_EXPERT_PARTS]; /* [n_expert] */

    unsigned char *slab;  /* n_alloc * slot_bytes on whole pages (tr_pages_alloc), aligned to read_align */
    int64_t n_alloc;      /* the slots the slab was made with: stats.n_slots of them in use, the others' pages
                           * given back (tr_experts_set_slots). The per-slot arrays are n_alloc long */
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

    /* reading ahead (tr_experts_prefetch_start); io == NULL: never started, every field NULL/0 */
    tr_thread *io;
    tr_monitor *mon;
    experts_job *jobs;       /* [n_slots], under mon */
    int32_t *queue;          /* [n_slots] ring of slots for the I/O thread, under mon */
    int64_t q_head, q_len;   /* under mon */
    int io_stop;             /* under mon: the I/O thread ends once the queue is empty */
    unsigned char *pending;  /* [n_slots]: reserved and not taken in yet -- calling thread only */
    int64_t n_pending;       /* calling thread only */
    int prefetch_failed;     /* calling thread only: a read ahead failed since the last wait */

    /* the emulated disk (cfg.disk_bytes_per_sec): NULL when off. The pointer is fixed at create time;
     * the time the disk is free again, in ns of tr_time_sec, is shared by both threads */
    _Atomic uint64_t *disk_free_ns;
    double disk_bps;
    /* the handoffs' clocks written by either thread (tr_experts_stats io_gap_sec, disk_late_sec), in ns */
    _Atomic uint64_t io_gap_ns, disk_late_ns;

    /* cfg.evict == TR_EXPERTS_EVICT_HOT: every unit's hotness, +1 a call that names it, every one
     * halved each hot_period successful calls (hot_calls counts them); NULL under the LRU */
    uint32_t *hot;
    int64_t hot_period, hot_calls;
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

static void lru_push_back(tr_experts *x, int32_t s) {
    experts_slot *sl = &x->slots[s];
    sl->next = -1;
    sl->prev = x->lru_tail;
    if (x->lru_tail != -1) x->slots[x->lru_tail].next = s;
    x->lru_tail = s;
    if (x->lru_head == -1) x->lru_head = s;
}

/* The victim under TR_EXPERTS_EVICT_HOT (ds4's streaming cache, ref/ds4 ds4_metal.m:15404-15439): a
 * free slot if there is one, else the unit of the lowest hotness, the least recent among equals (the
 * walk goes from the cold end and takes only a strictly cooler one). Never a slot in flight, one this
 * call already took (a run's earlier picks), a unit of layer skip_a or skip_b, or a unit this call
 * names. -1 if there is none. */
static int32_t hot_victim(const tr_experts *x, int64_t skip_a, int64_t skip_b) {
    int32_t best = -1;
    uint32_t best_hot = UINT32_MAX;
    for (int32_t s = x->lru_tail; s != -1; s = x->slots[s].prev) {
        if (x->pending != NULL && x->pending[s]) continue;
        if (x->slots[s].pick == x->call_seq) continue;
        int32_t u = x->slots[s].unit;
        if (u == -1) return s;
        int64_t layer = u / x->n_expert;
        if (layer == skip_a || layer == skip_b) continue;
        if (x->seen_call[u] == x->call_seq) continue;
        if (best == -1 || x->hot[u] < best_hot) {
            best = s;
            best_hot = x->hot[u];
        }
    }
    return best;
}

/* The coldest slot the store may take: never one in flight, and never one holding a unit of
 * layer skip_a or skip_b (-1: no such layer). -1 if there is none. Without reading ahead this is
 * lru_tail, the plain LRU victim. Under TR_EXPERTS_EVICT_HOT, hot_victim. */
static int32_t coldest_victim(const tr_experts *x, int64_t skip_a, int64_t skip_b) {
    if (x->hot != NULL) return hot_victim(x, skip_a, skip_b);
    for (int32_t s = x->lru_tail; s != -1; s = x->slots[s].prev) {
        if (x->pending != NULL && x->pending[s]) continue;
        int32_t u = x->slots[s].unit;
        if (u != -1) {
            int64_t layer = u / x->n_expert;
            if (layer == skip_a || layer == skip_b) continue;
        }
        return s;
    }
    return -1;
}

/* A request of `bytes` issued at t0 on the emulated disk (cfg.disk_bytes_per_sec): it holds the disk
 * from t0, or from when the request before it ends, for its bytes at the disk's rate, and returns at
 * its end. One clock for both threads: their requests queue as on one drive. The real read issued
 * at t0 has landed by then whenever the real disk is the faster one. Off: returns at once. */
static void disk_emulate(const tr_experts *x, double t0, uint64_t bytes) {
    if (x->disk_free_ns == NULL) return;
    uint64_t t0_ns = (uint64_t)(t0 * 1e9);
    uint64_t cost = (uint64_t)((double)bytes / x->disk_bps * 1e9);
    uint64_t free_ns = atomic_load(x->disk_free_ns), end;
    do {
        end = (free_ns > t0_ns ? free_ns : t0_ns) + cost;
    } while (!atomic_compare_exchange_weak(x->disk_free_ns, &free_ns, end));
    tr_wait_until((double)end / 1e9);
    double late = tr_time_sec() - (double)end / 1e9;
    if (late > 0) atomic_fetch_add((_Atomic uint64_t *)&x->disk_late_ns, (uint64_t)(late * 1e9));
}

/* (layer, id)'s parts into the slot area at base, one read each, through read/ctx: 0, or -1 at
 * the first part that fails. *bytes and *sec grow by what the reads before a failure moved too.
 * Touches only fields fixed at create time (the I/O thread calls it). */
static int read_parts(const tr_experts *x, tr_experts_read_fn read, void *ctx, int64_t layer, int64_t id,
                      unsigned char *base, uint64_t shift[TR_EXPERT_PARTS], uint64_t *bytes, double *sec,
                      uint64_t *requests) {
    uint64_t align = x->read_align;
    for (int p = 0; p < TR_EXPERT_PARTS; p++) {
        size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)p;
        uint64_t file_off = x->part_offset[idx] + (uint64_t)x->part_bytes[idx] * (uint64_t)id;
        /* align == 1: lo == file_off, hi == file_off + part_bytes -- the exact bytes. align > 1:
         * the aligned range the read must actually cover. */
        uint64_t lo = file_off / align * align;
        uint64_t hi = (file_off + (uint64_t)x->part_bytes[idx] + align - 1) / align * align;
        size_t n = (size_t)(hi - lo);
        double t0 = tr_time_sec();
        int rc = read(ctx, base + x->part_slot_offset[idx], n, lo);
        if (rc == 0) disk_emulate(x, t0, n);
        *sec += tr_time_sec() - t0;
        (*requests)++;
        if (rc != 0) return -1;
        *bytes += n;
        shift[p] = file_off - lo;
    }
    return 0;
}

/* The units (layer, id0 .. id0 + k - 1) into slots[0..k), one request a part through readv, the
 * three in flight together (two give this disk +7.6% over one at 68 MiB: docs/MEASUREMENTS.md §The
 * disk at its limit): each unit's aligned range [lo, hi) as read_parts reads it, laid end to end.
 * Two neighbours' ranges share at most one page (hi of one past lo of the next, when a part's end is
 * not aligned): the request lands it once, in the next unit's slot as its first page, and it is
 * copied after into this unit's slot at its place. Needs every part at least read_align bytes (a
 * unit's own pages then start past the previous unit's) and k * part bytes within the run bytes
 * (x->run_units). w: which thread's working memory. Same arguments and failure as read_parts. */
_Static_assert(TR_EXPERT_PARTS <= TR_FILE_PREADV_MAX, "a run's requests go in one tr_file_preadv_n");
static int read_run(const tr_experts *x, int w, tr_experts_readv_fn readv, void *ctx, int64_t layer, int64_t id0,
                    int64_t k, const int32_t *slots, uint64_t (*shift)[TR_EXPERT_PARTS], uint64_t *bytes,
                    double *sec, uint64_t *requests) {
    uint64_t align = x->read_align;
    tr_readv_req req[TR_EXPERT_PARTS];
    uint64_t total = 0;
    for (int p = 0; p < TR_EXPERT_PARTS; p++) {
        size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)p;
        uint64_t pb = (uint64_t)x->part_bytes[idx];
        uint64_t first = x->part_offset[idx] + pb * (uint64_t)id0;
        uint64_t lo0 = first / align * align;
        tr_iov *iov = x->iov[w] + (size_t)p * (size_t)x->n_expert;
        for (int64_t e = 0; e < k; e++) {
            uint64_t lo = (first + pb * (uint64_t)e) / align * align;
            uint64_t end = e + 1 < k ? (first + pb * (uint64_t)(e + 1)) / align * align /* the next one's lo */
                                     : (first + pb * (uint64_t)k + align - 1) / align * align;
            iov[e].base = x->slab + (uint64_t)slots[e] * x->stats.slot_bytes + x->part_slot_offset[idx];
            iov[e].n = (size_t)(end - lo);
        }
        req[p].iov = iov;
        req[p].cnt = (int)k;
        req[p].offset = lo0;
        total += (first + pb * (uint64_t)k + align - 1) / align * align - lo0;
    }
    double t0 = tr_time_sec();
    int rc = readv(ctx, req, TR_EXPERT_PARTS, x->scratch[w]);
    if (rc == 0) disk_emulate(x, t0, total); /* the three in flight share the one disk */
    *sec += tr_time_sec() - t0;
    *requests += TR_EXPERT_PARTS;
    if (rc != 0) return -1;
    *bytes += total;
    for (int p = 0; p < TR_EXPERT_PARTS; p++) {
        size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)p;
        uint64_t pb = (uint64_t)x->part_bytes[idx];
        uint64_t first = x->part_offset[idx] + pb * (uint64_t)id0;
        const tr_iov *iov = req[p].iov;
        for (int64_t e = 0; e < k; e++) {
            uint64_t off = first + pb * (uint64_t)e;
            uint64_t lo = off / align * align;
            uint64_t hi = (off + pb + align - 1) / align * align;
            shift[e][p] = off - lo;
            if (e + 1 < k && hi > lo + iov[e].n) /* the shared page, landed in the next slot */
                memcpy((unsigned char *)iov[e].base + iov[e].n, iov[e + 1].base, (size_t)(hi - lo - iov[e].n));
        }
    }
    return 0;
}

/* Experts of `layer` one request of a run carries, out of `left` consecutive ones: the run split in
 * equal requests (64 at 45 a request: 32 and 32, not 45 and 19). 1: no run. */
static int64_t run_take(const tr_experts *x, int64_t layer, int64_t left) {
    int64_t cap = x->run_units != NULL ? x->run_units[layer] : 1;
    if (cap < 2 || left < 2) return 1;
    int64_t n_req = (left + cap - 1) / cap;
    return (left + n_req - 1) / n_req;
}

/* The I/O thread: takes queued slots in order, reads each one's unit, publishes it done. Ends
 * when told to and the queue is empty. */
static void io_main(void *arg) {
    tr_experts *x = (tr_experts *)arg;
    double prev_done = 0.0; /* the last publish, for the gap to the next read */
    tr_monitor_lock(x->mon);
    for (;;) {
        int waited = 0;
        while (x->q_len == 0 && !x->io_stop) {
            waited = 1;
            tr_monitor_wait(x->mon);
        }
        if (x->q_len == 0) break;
        int32_t s = x->queue[x->q_head];
        x->q_head = (x->q_head + 1) % x->n_alloc;
        x->q_len--;
        experts_job job = x->jobs[s];
        /* the queued slots right behind it holding this layer's next experts: one run */
        int64_t left = 1;
        while (left <= x->q_len && left < x->n_expert && (job.cap == 0 || left < job.cap)) {
            const experts_job *nx = &x->jobs[x->queue[(x->q_head + left - 1) % x->n_alloc]];
            if (nx->layer != job.layer || nx->id != job.id + left || nx->readv != job.readv || job.readv == NULL)
                break;
            left++;
        }
        int64_t k = job.readv != NULL ? run_take(x, job.layer, left) : 1;
        int32_t alone = s;
        int32_t *slots = k > 1 ? x->run_slot[1] : &alone;
        slots[0] = s;
        for (int64_t e = 1; e < k; e++) {
            slots[e] = x->queue[x->q_head];
            x->q_head = (x->q_head + 1) % x->n_alloc;
            x->q_len--;
        }
        tr_monitor_broadcast(x->mon); /* taken: tr_experts_prefetch_n waits for its first */
        tr_monitor_unlock(x->mon);
        if (!waited && prev_done > 0) atomic_fetch_add(&x->io_gap_ns, (uint64_t)((tr_time_sec() - prev_done) * 1e9));

        job.bytes = job.requests = 0;
        job.sec = 0.0;
        if (k > 1) {
            job.rc = read_run(x, 1, job.readv, job.read_ctx, job.layer, job.id, k, slots, x->run_shift[1],
                              &job.bytes, &job.sec, &job.requests);
        } else {
            unsigned char *base = x->slab + (uint64_t)s * x->stats.slot_bytes;
            job.rc = read_parts(x, job.read, job.read_ctx, job.layer, job.id, base, job.shift, &job.bytes,
                                &job.sec, &job.requests);
        }

        tr_monitor_lock(x->mon);
        prev_done = tr_time_sec();
        job.done_at = prev_done;
        for (int64_t e = 0; e < k; e++) {
            experts_job *j = &x->jobs[slots[e]];
            if (e == 0) {
                *j = job;
            } else {
                j->bytes = j->requests = 0;
                j->sec = 0.0;
                j->rc = job.rc;
                j->done_at = job.done_at;
            }
            if (k > 1) memcpy(j->shift, x->run_shift[1][e], sizeof j->shift);
            j->done = 1;
        }
        tr_monitor_broadcast(x->mon);
    }
    tr_monitor_unlock(x->mon);
}

static void prefetch_stop(tr_experts *x) {
    if (x->io != NULL) {
        tr_monitor_lock(x->mon);
        x->io_stop = 1;
        tr_monitor_broadcast(x->mon);
        tr_monitor_unlock(x->mon);
        tr_thread_join(x->io);
        x->io = NULL;
    }
    tr_monitor_free(x->mon);
    free(x->jobs);
    free(x->queue);
    free(x->pending);
    x->mon = NULL;
    x->jobs = NULL;
    x->queue = NULL;
    x->pending = NULL;
}

void tr_experts_free(tr_experts *x) {
    if (x == NULL) return;
    prefetch_stop(x); /* first: the I/O thread may still be writing into the slab */
    free(x->run_units);
    for (int w = 0; w < 2; w++) {
        free(x->scratch[w]);
        free(x->iov[w]);
        free(x->run_slot[w]);
        free(x->run_shift[w]);
    }
    free(x->part_bytes);
    free(x->part_slot_offset);
    free(x->part_offset);
    tr_pages_free(x->slab, (size_t)((uint64_t)x->n_alloc * x->stats.slot_bytes));
    free(x->slot_of);
    free(x->slots);
    free(x->slot_part_shift);
    free(x->seen_call);
    free((void *)x->disk_free_ns);
    free(x->hot);
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
    x->stats.n_slots_made = n_slots;
    x->n_alloc = n_slots;
    x->stats.slot_bytes = slot_bytes;
    x->stats.direct = read_align > 1;
    x->lru_head = x->lru_tail = -1;

    size_t table_n = (size_t)(cfg->n_layers * TR_EXPERT_PARTS);
    uint64_t slab_align = read_align > 64 ? read_align : 64;
    x->part_bytes = (size_t *)malloc(sizeof(size_t) * table_n);
    x->part_slot_offset = (uint64_t *)malloc(sizeof(uint64_t) * table_n);
    x->part_offset = (uint64_t *)malloc(sizeof(uint64_t) * table_n);
    x->slab = (unsigned char *)tr_pages_alloc((size_t)((uint64_t)n_slots * slot_bytes));
    if (x->slab != NULL && (uintptr_t)x->slab % slab_align != 0) { /* a page smaller than read_align */
        tr_pages_free(x->slab, (size_t)((uint64_t)n_slots * slot_bytes));
        x->slab = NULL;
    }
    x->slot_of = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_units);
    x->slots = (experts_slot *)malloc(sizeof(experts_slot) * (size_t)n_slots);
    x->slot_part_shift = (uint64_t *)calloc((size_t)n_slots * TR_EXPERT_PARTS, sizeof(uint64_t));
    x->seen_call = (uint64_t *)calloc((size_t)n_units, sizeof(uint64_t));
    atomic_init(&x->io_gap_ns, 0);
    atomic_init(&x->disk_late_ns, 0);
    if (x->part_bytes == NULL || x->part_slot_offset == NULL || x->part_offset == NULL || x->slab == NULL ||
        x->slot_of == NULL || x->slots == NULL || x->slot_part_shift == NULL || x->seen_call == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory");
        tr_experts_free(x);
        return NULL;
    }
    memcpy(x->part_bytes, cfg->part_bytes, sizeof(size_t) * table_n);
    memcpy(x->part_offset, cfg->part_offset, sizeof(uint64_t) * table_n);
    if (cfg->disk_bytes_per_sec > 0) {
        x->disk_free_ns = (_Atomic uint64_t *)calloc(1, sizeof *x->disk_free_ns);
        if (x->disk_free_ns == NULL) {
            if (err != NULL) snprintf(err, err_len, "out of memory");
            tr_experts_free(x);
            return NULL;
        }
        atomic_init(x->disk_free_ns, 0);
        x->disk_bps = cfg->disk_bytes_per_sec;
        x->stats.disk_bytes_per_sec = cfg->disk_bytes_per_sec;
    }

    for (int64_t layer = 0; layer < cfg->n_layers; layer++) {
        uint64_t off = 0;
        for (int p = 0; p < TR_EXPERT_PARTS; p++) {
            size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)p;
            x->part_slot_offset[idx] = off;
            off += part_area_bytes((uint64_t)cfg->part_bytes[idx], read_align);
        }
    }

    /* runs: a layer's experts a request carries within the run bytes, none for a layer whose part
     * is under read_align bytes (its neighbours' pages would overlap by more than one) */
    if (cfg->readv != NULL) {
        uint64_t run_bytes = cfg->run_bytes != 0 ? cfg->run_bytes : TR_EXPERTS_RUN_BYTES;
        x->readv = cfg->readv;
        x->run_units = (int64_t *)calloc((size_t)cfg->n_layers, sizeof(int64_t));
        size_t scratch = tr_file_preadv_n_scratch((size_t)(TR_EXPERT_PARTS * (run_bytes + 2 * read_align)), TR_EXPERT_PARTS);
        int ok = x->run_units != NULL;
        for (int w = 0; w < 2 && ok; w++) {
            x->scratch[w] = scratch > 0 ? malloc(scratch) : NULL;
            x->iov[w] = (tr_iov *)malloc(sizeof(tr_iov) * TR_EXPERT_PARTS * (size_t)cfg->n_expert);
            x->run_slot[w] = (int32_t *)malloc(sizeof(int32_t) * (size_t)cfg->n_expert);
            x->run_shift[w] = calloc((size_t)cfg->n_expert, sizeof *x->run_shift[w]);
            ok = (scratch == 0 || x->scratch[w] != NULL) && x->iov[w] != NULL && x->run_slot[w] != NULL &&
                 x->run_shift[w] != NULL;
        }
        if (!ok) {
            if (err != NULL) snprintf(err, err_len, "out of memory");
            tr_experts_free(x);
            return NULL;
        }
        for (int64_t layer = 0; layer < cfg->n_layers; layer++) {
            uint64_t most = 0, least = UINT64_MAX;
            for (int p = 0; p < TR_EXPERT_PARTS; p++) {
                uint64_t pb = (uint64_t)cfg->part_bytes[(size_t)layer * TR_EXPERT_PARTS + (size_t)p];
                if (pb > most) most = pb;
                if (pb < least) least = pb;
            }
            x->run_units[layer] = least < read_align ? 0 : (int64_t)(run_bytes / most);
        }
    }

    if (cfg->evict == TR_EXPERTS_EVICT_HOT) {
        x->hot = (uint32_t *)calloc((size_t)n_units, sizeof(uint32_t));
        if (x->hot == NULL) {
            if (err != NULL) snprintf(err, err_len, "out of memory");
            tr_experts_free(x);
            return NULL;
        }
        x->hot_period = cfg->hot_period > 0 ? cfg->hot_period : TR_EXPERTS_HOT_TOKENS * cfg->n_layers;
        x->stats.evict = TR_EXPERTS_EVICT_HOT;
    }

    for (int64_t u = 0; u < n_units; u++) x->slot_of[u] = -1;
    for (int64_t s = 0; s < n_slots; s++) {
        x->slots[s].unit = -1;
        x->slots[s].prev = x->slots[s].next = -1;
        x->slots[s].pick = UINT64_MAX; /* never taken: tr_experts_load_all runs at call_seq 0 */
    }
    /* every slot starts free: pushed front in index order, so slot 0 ends at the cold end and
     * slot n_slots - 1 at the hot end -- the first eviction picks slot 0, then 1, 2, ... */
    for (int64_t s = 0; s < n_slots; s++) lru_push_front(x, (int32_t)s);

    return x;
}

int tr_experts_resident(const tr_experts *x) {
    return x->stats.n_slots == x->stats.n_units;
}

/* Evicts the coldest slot not in flight (freeing its unit, if it holds one, evictions++
 * regardless of what follows) and reads (layer, id)'s parts into it on this thread. -1 on a
 * failed part read: the slot is left free at the cold end (it is never linked to the hot end until
 * every part succeeds), `unit` stays absent, and every part read before the failure already
 * counted in bytes_read. */
static int experts_fill_slot(tr_experts *x, int64_t layer, int64_t id, int64_t unit) {
    int32_t victim = coldest_victim(x, -1, -1);
    if (victim == -1) return -1; /* every slot in flight: tr_experts_prefetch_start's margin forbids it */
    experts_slot *sl = &x->slots[victim];
    if (sl->unit != -1) {
        x->slot_of[sl->unit] = -1;
        sl->unit = -1;
        x->stats.evictions++;
    }

    unsigned char *base = x->slab + (uint64_t)victim * x->stats.slot_bytes;
    uint64_t shift[TR_EXPERT_PARTS] = {0, 0, 0};
    int rc = read_parts(x, x->read, x->read_ctx, layer, id, base, shift, &x->stats.bytes_read, &x->stats.read_sec,
                        &x->stats.requests);
    if (rc != 0) {
        lru_unlink(x, victim);
        lru_push_back(x, victim);
        return -1;
    }
    for (int p = 0; p < TR_EXPERT_PARTS; p++)
        x->slot_part_shift[(size_t)victim * TR_EXPERT_PARTS + (size_t)p] = shift[p];

    x->slot_of[unit] = victim;
    sl->unit = (int32_t)unit;
    lru_unlink(x, victim);
    lru_push_front(x, victim);
    x->stats.misses++;
    return 0;
}

/* experts_fill_slot for the run (layer, id0 .. id0 + k - 1), on the calling thread: k victims taken
 * as experts_fill_slot takes one (each made hot at once, so the next is another slot), then one
 * request a part (read_run). -1 on a failed request: every slot of the run is left free at the cold
 * end and its unit absent. */
static int experts_fill_run(tr_experts *x, int64_t layer, int64_t id0, int64_t k) {
    int32_t *slots = x->run_slot[0];
    for (int64_t e = 0; e < k; e++) {
        slots[e] = coldest_victim(x, -1, -1);
        if (slots[e] == -1) { /* every slot in flight: tr_experts_prefetch_start's margin forbids it */
            for (int64_t f = 0; f < e; f++) {
                lru_unlink(x, slots[f]);
                lru_push_back(x, slots[f]);
            }
            return -1;
        }
        experts_slot *sl = &x->slots[slots[e]];
        sl->pick = x->call_seq; /* emptied below: the next pick must not take it again (hot_victim) */
        if (sl->unit != -1) {
            x->slot_of[sl->unit] = -1;
            sl->unit = -1;
            x->stats.evictions++;
        }
        lru_unlink(x, slots[e]);
        lru_push_front(x, slots[e]);
    }
    if (read_run(x, 0, x->readv, x->read_ctx, layer, id0, k, slots, x->run_shift[0], &x->stats.bytes_read,
                 &x->stats.read_sec, &x->stats.requests) != 0) {
        for (int64_t e = 0; e < k; e++) {
            lru_unlink(x, slots[e]);
            lru_push_back(x, slots[e]);
        }
        return -1;
    }
    for (int64_t e = 0; e < k; e++) {
        int64_t unit = layer * x->n_expert + id0 + e;
        for (int p = 0; p < TR_EXPERT_PARTS; p++)
            x->slot_part_shift[(size_t)slots[e] * TR_EXPERT_PARTS + (size_t)p] = x->run_shift[0][e][p];
        x->slot_of[unit] = slots[e];
        x->slots[slots[e]].unit = (int32_t)unit;
        x->stats.misses++;
    }
    return 0;
}

/* Waits for slot s's read ahead and takes it in: counted (misses, prefetched, bytes, time) and an
 * ordinary resident unit from here on, or, if the read failed, freed to the cold end with its
 * unit absent. 0, or -1 for a failed read. */
static int prefetch_take(tr_experts *x, int32_t s) {
    tr_monitor_lock(x->mon);
    if (!x->jobs[s].done) {
        double t0 = tr_time_sec();
        while (!x->jobs[s].done) tr_monitor_wait(x->mon);
        double t1 = tr_time_sec();
        x->stats.prefetch_wait_sec += t1 - t0;
        x->stats.take_waits++;
        x->stats.take_wake_sec += t1 - x->jobs[s].done_at;
    }
    experts_job job = x->jobs[s];
    tr_monitor_unlock(x->mon);

    x->pending[s] = 0;
    x->n_pending--;
    x->stats.bytes_read += job.bytes;
    x->stats.requests += job.requests;
    x->stats.read_sec += job.sec;
    if (job.rc != 0) {
        x->slot_of[x->slots[s].unit] = -1;
        x->slots[s].unit = -1;
        lru_unlink(x, s);
        lru_push_back(x, s);
        if (!job.demand) x->prefetch_failed = 1; /* a call's own unit fails that call (tr_experts_acquire_take) */
        return -1;
    }
    for (int p = 0; p < TR_EXPERT_PARTS; p++)
        x->slot_part_shift[(size_t)s * TR_EXPERT_PARTS + (size_t)p] = job.shift[p];
    x->stats.misses++;
    if (job.demand) x->stats.arrived++;
    else x->stats.prefetched++;
    return 0;
}

/* Slot v reserved for (layer, id) and queued for the I/O thread, under x->mon: its unit evicted, the new one
 * present to the index, hot in the LRU, in flight until taken in. cap: the job's (tr_experts_prefetch_n);
 * demand: the job's. */
static void reserve_slot(tr_experts *x, int32_t v, int64_t layer, int64_t id, int64_t cap, int demand) {
    experts_slot *sl = &x->slots[v];
    if (sl->unit != -1) {
        x->slot_of[sl->unit] = -1;
        x->stats.evictions++;
    }
    int64_t unit = layer * x->n_expert + id;
    sl->unit = (int32_t)unit;
    x->slot_of[unit] = v;
    lru_unlink(x, v);
    lru_push_front(x, v);
    x->pending[v] = 1;
    x->n_pending++;

    experts_job *job = &x->jobs[v];
    memset(job, 0, sizeof *job);
    job->layer = layer;
    job->id = id;
    job->cap = cap;
    job->demand = demand;
    job->read = x->read;
    job->readv = x->readv;
    job->read_ctx = x->read_ctx;
    x->queue[(x->q_head + x->q_len) % x->n_alloc] = v;
    x->q_len++;
}

/* After queuing `queued` slots under x->mon: the I/O thread woken, and the first of them taken before the
 * caller computes on: the disk starts now, not whenever the I/O thread is next scheduled (a loaded machine, a
 * pool busy on every core: LESSONS #286). The queue is in order: fewer than `queued` left, the first is out. */
static void queue_wake(tr_experts *x, int64_t queued) {
    if (queued <= 0) return;
    double t0 = tr_time_sec();
    tr_monitor_broadcast(x->mon);
    while (x->q_len >= queued) tr_monitor_wait(x->mon);
    x->stats.queue_wake_sec += tr_time_sec() - t0;
}

int tr_experts_prefetch_start(tr_experts *x) {
    if (x->io != NULL) return 0;
    if (tr_experts_resident(x) || x->stats.n_slots < 2 * x->n_expert + x->n_used) return 1;
    size_t n = (size_t)x->n_alloc;
    x->mon = tr_monitor_create();
    x->jobs = (experts_job *)calloc(n, sizeof(experts_job));
    x->queue = (int32_t *)calloc(n, sizeof(int32_t));
    x->pending = (unsigned char *)calloc(n, 1);
    x->q_head = x->q_len = 0;
    x->io_stop = 0;
    x->n_pending = 0;
    x->prefetch_failed = 0;
    if (x->mon == NULL || x->jobs == NULL || x->queue == NULL || x->pending == NULL ||
        (x->io = tr_thread_start(io_main, x)) == NULL) {
        prefetch_stop(x);
        return -1;
    }
    return 0;
}

int tr_experts_prefetching(const tr_experts *x) {
    return x->io != NULL;
}

int64_t tr_experts_prefetch(tr_experts *x, int64_t layer, int64_t keep) {
    return tr_experts_prefetch_n(x, layer, keep, 0);
}

int64_t tr_experts_prefetch_n(tr_experts *x, int64_t layer, int64_t keep, uint64_t run_bytes) {
    if (x->io == NULL || layer < 0 || layer >= x->n_layers) return 0;
    int64_t cap = 0; /* the experts of the layer's largest part that fit run_bytes, at least one */
    if (run_bytes > 0) {
        uint64_t most = 1;
        for (int p = 0; p < TR_EXPERT_PARTS; p++) {
            uint64_t pb = (uint64_t)x->part_bytes[(size_t)layer * TR_EXPERT_PARTS + (size_t)p];
            if (pb > most) most = pb;
        }
        cap = run_bytes / most > 1 ? (int64_t)(run_bytes / most) : 1;
    }
    int64_t queued = 0;
    /* the whole layer queued under one lock and one wake: the I/O thread sees it all at once and its
     * first run is as long as the next ones (woken at the first unit, it read that unit alone) */
    tr_monitor_lock(x->mon);
    for (int64_t id = 0; id < x->n_expert; id++) {
        int64_t unit = layer * x->n_expert + id;
        if (x->slot_of[unit] != -1) continue;
        int32_t v = coldest_victim(x, layer, keep);
        if (v == -1) break;
        reserve_slot(x, v, layer, id, cap, 0);
        queued++;
    }
    queue_wake(x, queued);
    tr_monitor_unlock(x->mon);
    return queued;
}

int64_t tr_experts_prefetch_cancel(tr_experts *x, int64_t layer, const int64_t *ids, int64_t n) {
    if (x->io == NULL) return 0;
    int64_t dropped = 0, kept = 0;
    tr_monitor_lock(x->mon);
    /* the queue compacted in place: what stays keeps its order (a kept entry moves only back) */
    for (int64_t i = 0; i < x->q_len; i++) {
        int32_t s = x->queue[(x->q_head + i) % x->n_alloc];
        const experts_job *j = &x->jobs[s];
        int asked = j->layer != layer;
        for (int64_t k = 0; k < n && !asked; k++) asked = ids[k] == j->id;
        if (asked) {
            x->queue[(x->q_head + kept) % x->n_alloc] = s;
            kept++;
            continue;
        }
        /* never read: the slot free at the cold end, as a failed read leaves it (prefetch_take) */
        x->slot_of[x->slots[s].unit] = -1;
        x->slots[s].unit = -1;
        lru_unlink(x, s);
        lru_push_back(x, s);
        x->pending[s] = 0;
        x->n_pending--;
        dropped++;
    }
    x->q_len = kept;
    tr_monitor_unlock(x->mon);
    x->stats.cancelled += (uint64_t)dropped;
    return dropped;
}

int tr_experts_prefetch_wait(tr_experts *x) {
    if (x->io == NULL) return 0;
    for (int64_t s = 0; s < x->n_alloc && x->n_pending > 0; s++)
        if (x->pending[s]) prefetch_take(x, (int32_t)s);
    int failed = x->prefetch_failed;
    x->prefetch_failed = 0;
    return failed ? -1 : 0;
}

/* tr_experts_touch: one write a page of the slab, the pages split over the pool */
typedef struct {
    unsigned char *base;
} touch_ctx;

static void touch_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    volatile unsigned char *base = ((touch_ctx *)ctx_)->base;
    (void)worker;
    for (int64_t pg = begin; pg < end; pg++) base[pg * 4096] = base[pg * 4096];
}

void tr_experts_touch(tr_experts *x, tr_pool *pool) {
    touch_ctx c = {x->slab};
    int64_t pages = (int64_t)(((uint64_t)x->stats.n_slots * x->stats.slot_bytes + 4095) / 4096);
    double t0 = tr_time_sec();
    tr_parallel_for(pool, pages, 4096, touch_body, &c);
    x->stats.touch_sec += tr_time_sec() - t0;
}

/* ---- tr_experts_set_slots: the slots in use, fewer or more ---- */

/* The coldest unit present among the slots in use, as the policy ranks them with no call under way: under
 * the LRU the least recent, under HOT the lowest hotness, the least recent among equals. -1 if none. */
static int32_t coldest_unit_slot(const tr_experts *x) {
    int32_t best = -1;
    for (int32_t s = x->lru_tail; s != -1; s = x->slots[s].prev) {
        int32_t u = x->slots[s].unit;
        if (u == -1) continue;
        if (x->hot == NULL) return s;
        if (best == -1 || x->hot[u] < x->hot[x->slots[best].unit]) best = s;
    }
    return best;
}

/* Slot v, free, takes slot s's place in the list (and so its unit's recency); s leaves the list */
static void lru_take_place(tr_experts *x, int32_t v, int32_t s) {
    lru_unlink(x, v);
    experts_slot *sl = &x->slots[s], *vl = &x->slots[v];
    vl->prev = sl->prev;
    vl->next = sl->next;
    if (sl->prev != -1) x->slots[sl->prev].next = v; else x->lru_head = v;
    if (sl->next != -1) x->slots[sl->next].prev = v; else x->lru_tail = v;
    sl->prev = sl->next = -1;
}

int64_t tr_experts_set_slots(tr_experts *x, int64_t n) {
    int64_t least = tr_experts_min_slots(x->n_expert, x->n_used);
    if (least > x->n_alloc) least = x->n_alloc;
    if (n < least) n = least;
    if (n > x->n_alloc) n = x->n_alloc;
    int64_t have = x->stats.n_slots;
    if (n == have) return have;
    /* a read ahead still in flight lands first (a failed one stays for the next tr_experts_prefetch_wait) */
    for (int32_t s = 0; s < x->n_alloc && x->n_pending > 0; s++)
        if (x->pending[s] != 0) prefetch_take(x, s);
    uint64_t sb = x->stats.slot_bytes;

    if (n > have) {
        if (tr_pages_commit(x->slab + (uint64_t)have * sb, (size_t)((uint64_t)(n - have) * sb)) != 0) return have;
        for (int64_t s = have; s < n; s++) {
            x->slots[s].unit = -1;
            x->slots[s].pick = UINT64_MAX;
            lru_push_back(x, (int32_t)s); /* free, at the cold end */
        }
        x->stats.slots_taken += (uint64_t)(n - have);
        x->stats.n_slots = n;
        return n;
    }

    /* fewer: the policy's coldest units go until n slots hold the rest, and the ones in slots [n, have) move
     * below n, into slots freed for them (n - (units kept below n) of them are free there) */
    int64_t used = 0;
    for (int64_t s = 0; s < have; s++) used += x->slots[s].unit != -1;
    for (; used > n; used--) {
        int32_t s = coldest_unit_slot(x);
        x->slot_of[x->slots[s].unit] = -1;
        x->slots[s].unit = -1;
        lru_unlink(x, s);
        lru_push_back(x, s);
        x->stats.evictions++;
    }
    for (int64_t s = have - 1; s >= n; s--) {
        int32_t u = x->slots[s].unit;
        if (u == -1) {
            lru_unlink(x, (int32_t)s);
            continue;
        }
        int32_t v = x->lru_tail; /* free slots wait at the cold end */
        while (v != -1 && (v >= n || x->slots[v].unit != -1)) v = x->slots[v].prev;
        memcpy(x->slab + (uint64_t)v * sb, x->slab + (uint64_t)s * sb, (size_t)sb);
        memcpy(&x->slot_part_shift[(size_t)v * TR_EXPERT_PARTS], &x->slot_part_shift[(size_t)s * TR_EXPERT_PARTS],
               TR_EXPERT_PARTS * sizeof(uint64_t));
        x->slots[v].unit = u;
        x->slots[v].pick = x->slots[s].pick;
        x->slot_of[u] = v;
        lru_take_place(x, v, (int32_t)s);
        x->slots[s].unit = -1;
        x->stats.moved++;
    }
    tr_pages_release(x->slab + (uint64_t)n * sb, (size_t)((uint64_t)(have - n) * sb));
    x->stats.slots_given += (uint64_t)(have - n);
    x->stats.n_slots = n;
    return n;
}

int tr_experts_load_all(tr_experts *x, tr_experts_unit_fn on_unit, void *ctx) {
    if (!tr_experts_resident(x)) return -1;
    for (int64_t layer = 0; layer < x->n_layers; layer++) {
        uint64_t unit_bytes = 0;
        for (int p = 0; p < TR_EXPERT_PARTS; p++)
            unit_bytes += x->part_bytes[(size_t)layer * TR_EXPERT_PARTS + (size_t)p];
        for (int64_t id = 0; id < x->n_expert;) {
            int64_t k = run_take(x, layer, x->n_expert - id);
            if ((k == 1 ? experts_fill_slot(x, layer, id, layer * x->n_expert + id)
                        : experts_fill_run(x, layer, id, k)) != 0)
                return -1;
            for (int64_t e = 0; e < k && on_unit != NULL; e++) on_unit(ctx, unit_bytes);
            id += k;
        }
    }
    return 0;
}

int tr_experts_acquire(tr_experts *x, int64_t layer, const int64_t *ids, int64_t n) {
    return tr_experts_acquire_counts(x, layer, ids, NULL, 1, n);
}

/* What a call of n_tok tokens adds to the hotness of a unit `count` of them chose: the routings as if from
 * at most TR_EXPERTS_HOT_PROMPT tokens, rounded half up, at least 1 (a decode token: 1) */
static uint32_t hot_add(int64_t count, int64_t n_tok) {
    if (count < 1) count = 1;
    if (n_tok <= TR_EXPERTS_HOT_PROMPT) return (uint32_t)(count < n_tok ? count : n_tok);
    int64_t a = (count * TR_EXPERTS_HOT_PROMPT + n_tok / 2) / n_tok;
    return a < 1 ? 1u : (uint32_t)(a < TR_EXPERTS_HOT_PROMPT ? a : TR_EXPERTS_HOT_PROMPT);
}

/* tr_experts_acquire_counts, and with async (the I/O thread started) tr_experts_acquire_async: the second pass
 * reserves and queues where the other reads, the victims taken in the same order by the same rule. Returns the
 * units queued, or -1. */
static int64_t acquire_body(tr_experts *x, int64_t layer, const int64_t *ids, const int64_t *counts, int64_t n_tok,
                            int64_t n, int async) {
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

    /* TR_EXPERTS_EVICT_HOT: every unit named counts, hit or miss, after the halving this call's turn
     * may bring (ds4 halves every 16 decode tokens: hot_period calls); a prompt's pass by its tokens'
     * routings (hot_add), so the units its router liked stay into the decode */
    if (x->hot != NULL) {
        if (++x->hot_calls % x->hot_period == 0)
            for (int64_t u = 0; u < x->stats.n_units; u++) x->hot[u] >>= 1;
        for (int64_t i = 0; i < n; i++) {
            uint32_t a = counts != NULL ? hot_add(counts[i], n_tok) : 1u;
            x->hot[layer * x->n_expert + ids[i]] += a;
            x->stats.hot_extra += a - 1u;
        }
    }

    /* first pass: touch every unit already present, in the order given; one still in flight is
     * waited for and taken in first, and one whose read ahead failed fails the call like a read */
    for (int64_t i = 0; i < n; i++) {
        int64_t unit = layer * x->n_expert + ids[i];
        int32_t slot = x->slot_of[unit];
        if (slot != -1) {
            if (x->pending != NULL && x->pending[slot] && prefetch_take(x, slot) != 0) return -1;
            lru_unlink(x, slot);
            lru_push_front(x, slot);
            x->stats.hits++;
        }
    }

    /* second pass: read the missing ones, in the order given. n_slots > n (tr_experts_create's
     * minimum), and the first pass already made every present unit of this call hot, so the
     * victim taken here is never one of this call's own units. Async: each one's victim taken as
     * experts_fill_slot takes it (a reserved slot is hot and in flight, as a filled one is hot and
     * named), reserved and queued; the I/O thread joins consecutive ids into the runs read here */
    if (async && x->io != NULL) {
        int64_t queued = 0;
        tr_monitor_lock(x->mon);
        for (int64_t i = 0; i < n; i++) {
            int64_t unit = layer * x->n_expert + ids[i];
            if (x->slot_of[unit] != -1) continue;
            int32_t v = coldest_victim(x, -1, -1);
            if (v == -1) break; /* every slot in flight: tr_experts_prefetch_start's margin forbids it */
            reserve_slot(x, v, layer, ids[i], 0, 1);
            queued++;
        }
        queue_wake(x, queued);
        tr_monitor_unlock(x->mon);
        for (int64_t i = 0; i < n; i++)
            if (x->slot_of[layer * x->n_expert + ids[i]] == -1) return -1;
        return queued;
    }
    for (int64_t i = 0; i < n;) {
        int64_t id = ids[i];
        int64_t unit = layer * x->n_expert + id;
        if (x->slot_of[unit] != -1) {
            i++;
            continue;
        }
        /* the missing ones right after it in the order given, the next experts in the file: a run */
        int64_t left = 1;
        while (i + left < n && ids[i + left] == id + left && x->slot_of[unit + left] == -1) left++;
        int64_t k = run_take(x, layer, left);
        if ((k == 1 ? experts_fill_slot(x, layer, id, unit) : experts_fill_run(x, layer, id, k)) != 0) return -1;
        i += k;
    }
    return 0;
}

int tr_experts_acquire_counts(tr_experts *x, int64_t layer, const int64_t *ids, const int64_t *counts,
                              int64_t n_tok, int64_t n) {
    return acquire_body(x, layer, ids, counts, n_tok, n, 0) < 0 ? -1 : 0;
}

int64_t tr_experts_acquire_async(tr_experts *x, int64_t layer, const int64_t *ids, const int64_t *counts,
                                 int64_t n_tok, int64_t n) {
    return acquire_body(x, layer, ids, counts, n_tok, n, 1);
}

int64_t tr_experts_acquire_take(tr_experts *x, int64_t layer, const int64_t *ids, int64_t n, int64_t at_least) {
    if (layer < 0 || layer >= x->n_layers) return -1;
    int64_t i = 0;
    for (; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= x->n_expert) return -1;
        int32_t s = x->slot_of[layer * x->n_expert + ids[i]];
        if (s == -1) return -1;
        if (x->pending == NULL || !x->pending[s]) continue;
        if (i >= at_least) {
            tr_monitor_lock(x->mon);
            int landed = x->jobs[s].done;
            tr_monitor_unlock(x->mon);
            if (!landed) break;
        }
        if (prefetch_take(x, s) != 0) return -1;
    }
    return i;
}

const void *tr_experts_part(const tr_experts *x, int64_t layer, int64_t expert, int part) {
    if (layer < 0 || layer >= x->n_layers || expert < 0 || expert >= x->n_expert || part < 0 ||
        part >= TR_EXPERT_PARTS)
        return NULL;
    int32_t slot = x->slot_of[layer * x->n_expert + expert];
    if (slot == -1) return NULL;
    if (x->pending != NULL && x->pending[slot]) return NULL; /* in flight: its bytes are not ours yet */
    size_t idx = (size_t)layer * TR_EXPERT_PARTS + (size_t)part;
    return x->slab + (uint64_t)slot * x->stats.slot_bytes + x->part_slot_offset[idx] +
           x->slot_part_shift[(size_t)slot * TR_EXPERT_PARTS + (size_t)part];
}

void tr_experts_get_stats(const tr_experts *x, tr_experts_stats *out) {
    *out = x->stats;
    out->io_gap_sec = (double)atomic_load((_Atomic uint64_t *)&x->io_gap_ns) / 1e9;
    out->disk_late_sec = (double)atomic_load((_Atomic uint64_t *)&x->disk_late_ns) / 1e9;
}

void tr_experts_get_reader(const tr_experts *x, tr_experts_read_fn *read, tr_experts_readv_fn *readv, void **ctx) {
    *read = x->read;
    *readv = x->readv;
    *ctx = x->read_ctx;
}

void tr_experts_set_reader(tr_experts *x, tr_experts_read_fn read, tr_experts_readv_fn readv, void *ctx) {
    x->read = read;
    x->readv = readv;
    x->read_ctx = ctx;
}
