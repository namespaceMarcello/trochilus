/* prof.h — engine profiler: where the time of a token goes.
 *
 * A tr_prof belongs to one session (no global state). Zones are a fixed enum so
 * a sample is an array index, not a string lookup. Time is wall-clock seen by the
 * thread that drives the forward pass: a zone around a parallel_for measures the
 * whole parallel work, which is what a user waits for.
 *
 * Cost: with the profiler disabled a scope is one predictable branch; enabled,
 * two tick reads (RDTSC on x86-64 with an invariant TSC, the OS monotonic clock
 * otherwise). Nested zones are allowed; each zone accounts its own inclusive time.
 *
 * Usage:
 *     uint64_t t = tr_prof_begin(p);
 *     ... work ...
 *     tr_prof_end(p, TR_PROF_ROPE, t);
 */
#ifndef TR_PROF_H
#define TR_PROF_H

#include <stdint.h>
#include <stdio.h>

typedef enum {
    TR_PROF_TOKEN = 0,        /* one whole forward pass (inclusive of everything below) */
    TR_PROF_EMBED,
    TR_PROF_ATTN_NORM,
    TR_PROF_QKV_PROJ,
    TR_PROF_QK_NORM,
    TR_PROF_ROPE,
    TR_PROF_KV_WRITE,
    TR_PROF_ATTENTION,        /* scores, softmax, weighted sum over the context */
    TR_PROF_ATTN_OUT_PROJ,
    TR_PROF_FFN_NORM,
    TR_PROF_ROUTER,           /* router projection, softmax, top-k */
    TR_PROF_EXPERT_GATHER,    /* the pass's token inputs copied, grouped by expert */
    TR_PROF_EXPERT_GATE_UP,
    TR_PROF_EXPERT_ACT,
    TR_PROF_EXPERT_DOWN,
    TR_PROF_EXPERT_MIX,       /* weighted sum of expert outputs */
    TR_PROF_OUTPUT_NORM,
    TR_PROF_LM_HEAD,
    TR_PROF_SAMPLE,
    TR_PROF_WEIGHT_READ,      /* reading weights from disk (load and streaming) */
    TR_PROF_POOL_WAIT,        /* driving thread waiting for workers to finish */
    TR_PROF_ZONE_COUNT
} tr_prof_zone;

typedef enum { TR_PHASE_PREFILL = 0, TR_PHASE_DECODE = 1, TR_PHASE_COUNT = 2 } tr_prof_phase;

typedef struct {
    uint64_t calls;
    uint64_t ticks;
} tr_prof_acc;

typedef struct tr_prof {
    int enabled;
    tr_prof_phase phase;
    tr_prof_acc acc[TR_PHASE_COUNT][TR_PROF_ZONE_COUNT];
    uint64_t tokens[TR_PHASE_COUNT];
    uint64_t weight_bytes_touched[TR_PHASE_COUNT];  /* bytes of weights read by kernels */
    uint64_t io_bytes[TR_PHASE_COUNT];              /* bytes read from disk */
} tr_prof;

/* Stable short name, e.g. "attention". */
const char *tr_prof_zone_name(tr_prof_zone z);

/* Clears all counters; enabled and phase are kept. */
void tr_prof_reset(tr_prof *p);

/* Tick source and its unit, detected once. */
uint64_t tr_prof_ticks(void);
double tr_prof_ticks_per_sec(void);

/* hot: begin */
static inline uint64_t tr_prof_begin(const tr_prof *p) {
    return (p != NULL && p->enabled) ? tr_prof_ticks() : 0;
}

static inline void tr_prof_end(tr_prof *p, tr_prof_zone z, uint64_t start) {
    if (p != NULL && p->enabled) {
        tr_prof_acc *a = &p->acc[p->phase][z];
        a->calls++;
        a->ticks += tr_prof_ticks() - start;
    }
}

static inline void tr_prof_count(tr_prof *p, uint64_t weight_bytes, uint64_t io_bytes) {
    if (p != NULL && p->enabled) {
        p->weight_bytes_touched[p->phase] += weight_bytes;
        p->io_bytes[p->phase] += io_bytes;
    }
}
/* hot: end */

/* Human-readable table per phase: zone, calls, total ms, % of token time, µs per
 * call; then tokens/s and weight bytes per token (GB/s of effective memory traffic). */
void tr_prof_print(const tr_prof *p, FILE *out);

/* The same numbers as one JSON object (seconds, not ticks), for tools/profile_suite.py. */
void tr_prof_write_json(const tr_prof *p, FILE *out);

#endif
