/* bench_attn_bw.c — why the decode's attention reads its KV below the RAM's bandwidth, and what
 * an exact change of order, of threads or of prefetching gives (docs/MEASUREMENTS.md §"Decode at
 * context 2048" and §"Decode at long context and RAM bandwidth" point 2).
 *
 * The cache is shaped like OLMoE-1B-7B's (16 layers, 16 heads of 128, f32, K and V apart, a
 * head's positions in a row, context 4096: src/kv/kv.h) and filled with random floats. One
 * decode token is 16 layers of one query per head over n_pos cached positions. After every
 * layer's attention every core of the pool reads 64 MiB of other memory, not timed, as the
 * engine reads its weights between two attentions (the set-up of `bench_mem kv`). The engine's
 * pool, pinned as the engine pins it; the width of the attention is a parameter of every variant
 * (tr_pool_set_active).
 *
 * A variant is `scheme[:option=value...][@threads]` (threads: 8 if not given):
 *   engine   tr_attention_group as src/models/olmoe.c calls it for a decode token: the reference
 *   head     tr_attention_head (the kernels' own one-query attention: dot_f32 and axpy_f32 one
 *            position at a time, the bits of dot_f32_x4 and axpy_f32_x4 by their contract)
 *   copy     the same calls in the same order written out here: K pass in blocks of 64 positions
 *            (dot_f32_x4), tr_softmax's four loops, V pass (axpy_f32_x4)
 *   fused    the softmax's max taken in the K pass and its division in the V pass, both in
 *            increasing position: between K and V only the exponentials and the lane sums remain
 *   pair     two heads of a worker interleaved: K(a); K(b) with the exponentials of a; V(a) with
 *            the exponentials of b; V(b). No thread stops reading memory for a softmax
 *   split    the K pass of every head cut in blocks of 64 positions spread over the threads (every
 *            thread the same number of blocks), then softmax and V per head, a second parallel_for
 *   read     plain reads of the same bytes, K then V per head (diagnostic, writes no output)
 *   read4    the same reads in the kernels' order: line k of 4 positions, k = 0..7 (diagnostic)
 *   nosm     copy without the softmax (diagnostic: what the softmax costs; other bits)
 *   mutant   copy with the V pass's blocks in decreasing order: must differ (the check seen red;
 *            a no-op at 64 positions or fewer, one block)
 * options: kd=B vd=B   software prefetch B bytes ahead of each call of the K / V pass
 *          kl=N vl=N   lines prefetched per call of 4 positions (32: all; -1: one per 4 KiB page)
 *          h=0..3      prefetch hint: 0 nta, 1 t2, 2 t1, 3 t0 (default)
 *          v0=B        the first B bytes of V prefetched along the exponentials (copy, fused, split)
 *          v0h=0..3    their hint (default 2, t1: L2 rather than the 32 KiB of L1)
 *          g=N         pair: the other head's exponentials every N positions (default 4)
 *          t=1         time the phases inside every worker (K, softmax, V, start and end waits)
 *          one=1       dot_f32 and axpy_f32 one position at a time instead of the x4 kernels (the
 *                      same bits by their contract): the rows read strictly one after the other
 *
 *   bench_attn_bw [n_pos] [--runs N] [variant ...]
 *
 * A repetition is 2 tokens of 16 steps; a step runs every variant once, each on its own layer, in
 * an order rotated at every step, so a burst of load elsewhere on the machine (it is shared)
 * falls on the variants of a step alike. A line: median of N repetitions after one warm-up, ms per token,
 * GB/s of KV, spread (max - min) / median, and the ratio to the first variant: the median
 * [p25-p75] (min-max) of the per-repetition ratios (> 1: faster than the first).
 * Bits: after every layer of every run the 16 outputs are compared with the engine's, computed
 * once at set-up; the output is poisoned before each layer. An exact variant that differs fails
 * the run, the mutant must differ, and no comparison at all fails it too (the branch counter).
 * A run lasts about (N + 1) x variants x 0.07 s at 2048 positions, 0.09 s at 4000; one that
 * would pass 55 s is cut to fewer repetitions after the warm-up. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"
#include "../src/models/model_internal.h"

#ifdef _WIN32
#include <windows.h>
#endif

#define MAX_RUNS 41
#define MAX_VARIANTS 24
#define MAX_WORKERS 64
#define MIB ((size_t)1 << 20)

/* OLMoE-1B-7B */
#define N_LAYER 16
#define N_HEAD 16
#define HEAD_DIM 128
#define N_KV (N_HEAD * HEAD_DIM)
#define KV_CTX 4096
#define POS_BYTES ((int64_t)(HEAD_DIM * sizeof(float)))
#define CALL_BYTES ((int64_t)TR_ATTN_X * POS_BYTES)
#define FLUSH_BYTES (64 * MIB)
#define N_FLUSH 8
#define TOKENS 2
#define SCALE 0.088388348f /* 1 / sqrt(128) */

enum { S_ENGINE, S_HEAD, S_COPY, S_FUSED, S_PAIR, S_SPLIT, S_READ, S_READ4, S_NOSM, S_MUTANT, S_COUNT };
static const char *const scheme_name[S_COUNT] = {"engine", "head", "copy",  "fused", "pair",  "split",
                                                 "read",   "read4", "nosm", "mutant"};

typedef struct {
    int64_t dist; /* bytes ahead of the first position of the call; 0: no prefetch */
    int lines;    /* per call of 4 positions (2 KiB): 32 every line, 1 the first; -1 one per 4 KiB page */
    int hint;     /* 0 nta, 1 t2, 2 t1, 3 t0 */
    int one;      /* 1: each x4 kernel call replaced by 4 single-position calls, rows read one after the other */
} pfopt;

typedef struct {
    char spec[64];
    int scheme, threads, gran, timed, exact;
    pfopt kpf, vpf;
    int64_t v0; /* bytes of V's start prefetched along the exponentials */
    int v0_hint;
    double secs[MAX_RUNS];
    long long compared, differ;
    /* the phases, summed over layers (seconds; per worker: summed over the active workers) */
    double b_wall, b_start, b_k, b_sm, b_v, b_idle, b_spread;
    long long b_layers, b_workers;
    double w_end[MAX_WORKERS]; /* when each worker finished, from the dispatch, summed over layers */
} variant;

typedef struct {
    double t_begin, t_end, k, sm, v;
    char pad[24];
} wstat;
_Static_assert(sizeof(wstat) == 64, "one line per worker");

typedef struct {
    const tr_kernels *kt;
    const variant *v;
    const float *keys, *values, *q; /* this layer's */
    float *scores;                  /* [MAX_WORKERS][2][KV_CTX] */
    float *scores_all;              /* split: [N_HEAD][KV_CTX] */
    float *out;                     /* [N_HEAD][HEAD_DIM] */
    int64_t n_pos, n_blk;
    wstat *stats;
} run_ctx;

/* one 64-byte line per worker: a sum nobody can optimize away */
typedef struct {
    volatile uint64_t sum;
    char pad[56];
} sink_t;
static sink_t sinks[MAX_WORKERS];

static uint64_t rng_next(uint64_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s * 0x2545F4914F6CDD1Dull;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* n bytes, a multiple of 64 */
static uint64_t read_bytes(const unsigned char *p, int64_t n) {
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (int64_t i = 0; i < n; i += 64) {
        uint64_t w[8];
        memcpy(w, p + i, 64);
        a += w[0] + w[4];
        b += w[1] + w[5];
        c += w[2] + w[6];
        d += w[3] + w[7];
    }
    return a ^ b ^ c ^ d;
}

/* ---- filling ---------------------------------------------------------------------------- */

typedef struct {
    unsigned char *base;
    size_t share;
} fill_ctx;

static void fill_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const fill_ctx *c = (const fill_ctx *)ctx_;
    for (int64_t i = begin; i < end; i++) {
        uint64_t s = 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1) + (uint64_t)(uintptr_t)c->base;
        unsigned char *p = c->base + (size_t)i * c->share;
        for (size_t k = 0; k < c->share; k += 4) {
            float v = (float)(int32_t)(rng_next(&s) >> 40) * (1.0f / 8388608.0f) - 1.0f;
            memcpy(p + k, &v, 4);
        }
    }
}

/* random floats in [-1, 1); bytes a multiple of 256 */
static void fill(tr_pool *pool, void *base, size_t bytes) {
    fill_ctx c = {(unsigned char *)base, bytes / 64};
    tr_parallel_for(pool, 64, 1, fill_body, &c);
}

/* ---- prefetching -------------------------------------------------------------------------- */

static inline void pf(const char *p, int hint) {
    switch (hint) {
    case 0: __builtin_prefetch(p, 0, 0); break;
    case 1: __builtin_prefetch(p, 0, 1); break;
    case 2: __builtin_prefetch(p, 0, 2); break;
    default: __builtin_prefetch(p, 0, 3); break;
    }
}

/* the lines o asks for, ahead of the call whose first position is `off` bytes into base;
 * nothing past lim, the bytes the pass reads */
static inline void prefetch_call(const char *base, int64_t lim, int64_t off, const pfopt *o) {
    int64_t at = off + o->dist;
    if (o->lines < 0) {
        if ((at & 4095) == 0 && at < lim) pf(base + at, o->hint);
        return;
    }
    int64_t step = CALL_BYTES / o->lines;
    for (int64_t b = 0; b < CALL_BYTES; b += step)
        if (at + b < lim) pf(base + at + b, o->hint);
}

/* ---- plain reads of a pass ----------------------------------------------------------------- */

/* the 32 lines of 4 positions in the order dot_f32_x4 and axpy_f32_x4 load them: line k of
 * each of the 4 positions, k = 0..7 */
static uint64_t read_call4(const unsigned char *p) {
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (int k = 0; k < 8; k++)
        for (int j = 0; j < TR_ATTN_X; j++) {
            uint64_t w[8];
            memcpy(w, p + j * POS_BYTES + k * 64, 64);
            a += w[0] + w[4];
            b += w[1] + w[5];
            c += w[2] + w[6];
            d += w[3] + w[7];
        }
    return a ^ b ^ c ^ d;
}

/* n_pos positions of K or V read 2 KiB (a call of 4 positions) at a time, in a row or in the
 * kernels' order, with the prefetching of o */
static uint64_t read_pass(const float *base, int64_t n_pos, const pfopt *o, int kernel_order) {
    const unsigned char *p = (const unsigned char *)base;
    const int64_t lim = n_pos * POS_BYTES;
    uint64_t s = 0;
    for (int64_t off = 0; off < lim; off += CALL_BYTES) {
        if (o->dist != 0) prefetch_call((const char *)p, lim, off, o);
        if (kernel_order && lim - off >= CALL_BYTES) s += read_call4(p + off);
        else s += read_bytes(p + off, lim - off < CALL_BYTES ? lim - off : CALL_BYTES);
    }
    return s;
}

/* ---- the attention of one head, written out --------------------------------------------- */

/* the exponentials of another head's row, done a piece at a time in increasing position, with
 * the lane sums of tr_softmax taken as they come */
typedef struct {
    float *row;
    float m;
    float lane[TR_LANES];
    int64_t done;
} exp_job;

static void exp_upto(exp_job *e, int64_t upto) {
    for (int64_t i = e->done; i < upto; i++) {
        float x = tr_expf(e->row[i] - e->m);
        e->row[i] = x;
        e->lane[i % TR_LANES] += x;
    }
    if (upto > e->done) e->done = upto;
}

/* x[i] = tr_expf(x[i] - m) in increasing i, with pf_bytes of pfb prefetched evenly along the
 * loop; lane, if not NULL, takes the lane sums as they come */
static void exp_loop(float *x, int64_t n, float m, const char *pfb, int64_t pf_bytes, int hint, float *lane) {
    int64_t lines = pf_bytes / 64, acc = 0, next = 0;
    for (int64_t i = 0; i < n; i++) {
        float e = tr_expf(x[i] - m);
        x[i] = e;
        if (lane != NULL) lane[i % TR_LANES] += e;
        for (acc += lines; acc >= n; acc -= n) pf(pfb + 64 * next++, hint);
    }
}

/* tr_softmax: the same loops in the same order */
static void softmax_copy(float *x, int64_t n, const char *pfb, int64_t pf_bytes, int hint) {
    float m = x[0];
    for (int64_t i = 1; i < n; i++)
        if (x[i] > m) m = x[i];
    exp_loop(x, n, m, pfb, pf_bytes, hint, NULL);
    float lane[TR_LANES] = {0};
    for (int64_t k = 0; k < n; k++) lane[k % TR_LANES] += x[k];
    float sum = tr_lane_combine(lane);
    for (int64_t i = 0; i < n; i++) x[i] /= sum;
}

/* positions [t0, t1) of the K pass (t0 a multiple of 4), as tr_attention_group does them */
static void kblock(const tr_kernels *kt, const float *q, const float *keys, int64_t t0, int64_t t1, int64_t lim,
                   float *row, const pfopt *o, exp_job *e, int gran) {
    int64_t t = t0;
    for (; t + TR_ATTN_X <= t1; t += TR_ATTN_X) {
        if (o->dist != 0) prefetch_call((const char *)keys, lim, t * POS_BYTES, o);
        if (o->one) /* dot_f32_x4's out[j] is dot_f32 of row j, bit for bit (kernels.h) */
            for (int x = 0; x < TR_ATTN_X; x++) row[t + x] = kt->dot_f32(q, keys + (t + x) * HEAD_DIM, HEAD_DIM);
        else
            kt->dot_f32_x4(q, keys + t * HEAD_DIM, HEAD_DIM, HEAD_DIM, row + t);
        for (int x = 0; x < TR_ATTN_X; x++) row[t + x] = row[t + x] * SCALE;
        if (e != NULL && (t + TR_ATTN_X) % gran == 0) exp_upto(e, t + TR_ATTN_X);
    }
    for (; t < t1; t++) row[t] = kt->dot_f32(q, keys + t * HEAD_DIM, HEAD_DIM) * SCALE;
    if (e != NULL) exp_upto(e, t1);
}

/* The K pass; with want_max, tr_softmax's max taken block by block in increasing position */
static float kpass(const tr_kernels *kt, const float *q, const float *keys, int64_t n_pos, float *row, const pfopt *o,
                   int want_max, exp_job *e, int gran) {
    float m = 0.0f;
    for (int64_t t0 = 0; t0 < n_pos; t0 += TR_ATTN_BLOCK) {
        int64_t t1 = t0 + TR_ATTN_BLOCK < n_pos ? t0 + TR_ATTN_BLOCK : n_pos;
        kblock(kt, q, keys, t0, t1, n_pos * POS_BYTES, row, o, e, gran);
        if (want_max) {
            int64_t i = t0;
            if (i == 0) m = row[i++];
            for (; i < t1; i++)
                if (row[i] > m) m = row[i];
        }
    }
    return m;
}

/* The V pass; with div, tr_softmax's division done just before a position's weight is used */
static void vpass(const tr_kernels *kt, const float *values, int64_t n_pos, float *row, float *out, const pfopt *o,
                  int div, float sum, exp_job *e, int gran, int reverse) {
    const int64_t lim = n_pos * POS_BYTES, n_blk = (n_pos + TR_ATTN_BLOCK - 1) / TR_ATTN_BLOCK;
    for (int d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;
    for (int64_t bi = 0; bi < n_blk; bi++) {
        int64_t t0 = (reverse ? n_blk - 1 - bi : bi) * TR_ATTN_BLOCK;
        int64_t t1 = t0 + TR_ATTN_BLOCK < n_pos ? t0 + TR_ATTN_BLOCK : n_pos, t = t0;
        for (; t + TR_ATTN_X <= t1; t += TR_ATTN_X) {
            if (o->dist != 0) prefetch_call((const char *)values, lim, t * POS_BYTES, o);
            if (div)
                for (int x = 0; x < TR_ATTN_X; x++) row[t + x] = row[t + x] / sum;
            if (o->one) /* axpy_f32_x4 is 4 axpy_f32 in order, bit for bit (kernels.h) */
                for (int x = 0; x < TR_ATTN_X; x++)
                    kt->axpy_f32(out, values + (t + x) * HEAD_DIM, row[t + x], HEAD_DIM);
            else
                kt->axpy_f32_x4(out, values + t * HEAD_DIM, HEAD_DIM, row + t, HEAD_DIM);
            if (e != NULL && (t + TR_ATTN_X) % gran == 0) exp_upto(e, t + TR_ATTN_X);
        }
        for (; t < t1; t++) {
            if (div) row[t] = row[t] / sum;
            kt->axpy_f32(out, values + t * HEAD_DIM, row[t], HEAD_DIM);
        }
        if (e != NULL) exp_upto(e, t1);
    }
}

static const float *head_keys(const run_ctx *c, int64_t h) { return c->keys + h * (int64_t)KV_CTX * HEAD_DIM; }
static const float *head_values(const run_ctx *c, int64_t h) { return c->values + h * (int64_t)KV_CTX * HEAD_DIM; }

static void head_single(const run_ctx *c, int64_t h, float *row, wstat *st, int worker) {
    const variant *v = c->v;
    const tr_kernels *kt = c->kt;
    const float *q = c->q + h * HEAD_DIM, *keys = head_keys(c, h), *values = head_values(c, h);
    float *out = c->out + h * HEAD_DIM;
    const int64_t n = c->n_pos, v0 = v->v0 < n * POS_BYTES ? v->v0 : n * POS_BYTES;
    double t0 = st != NULL ? tr_time_sec() : 0.0, t1 = t0, t2 = t0;
    switch (v->scheme) {
    case S_ENGINE:
        tr_attention_group(q, HEAD_DIM, keys, values, 1, n, HEAD_DIM, SCALE, row, KV_CTX, out, HEAD_DIM);
        break;
    case S_HEAD:
        tr_attention_head(q, keys, values, HEAD_DIM, 0, n, HEAD_DIM, SCALE, row, out);
        break;
    case S_READ:
    case S_READ4: {
        uint64_t s = read_pass(keys, n, &v->kpf, v->scheme == S_READ4);
        if (st != NULL) t1 = t2 = tr_time_sec();
        s += read_pass(values, n, &v->vpf, v->scheme == S_READ4);
        sinks[worker].sum += s;
        break;
    }
    case S_FUSED:
    case S_PAIR: {
        float m = kpass(kt, q, keys, n, row, &v->kpf, 1, NULL, TR_ATTN_X);
        if (st != NULL) t1 = tr_time_sec();
        float lane[TR_LANES] = {0};
        exp_loop(row, n, m, (const char *)values, v0, v->v0_hint, lane);
        float sum = tr_lane_combine(lane);
        if (st != NULL) t2 = tr_time_sec();
        vpass(kt, values, n, row, out, &v->vpf, 1, sum, NULL, TR_ATTN_X, 0);
        break;
    }
    default: /* copy, nosm, mutant */
        kpass(kt, q, keys, n, row, &v->kpf, 0, NULL, TR_ATTN_X);
        if (st != NULL) t1 = tr_time_sec();
        if (v->scheme != S_NOSM) softmax_copy(row, n, (const char *)values, v0, v->v0_hint);
        if (st != NULL) t2 = tr_time_sec();
        vpass(kt, values, n, row, out, &v->vpf, 0, 0.0f, NULL, TR_ATTN_X, v->scheme == S_MUTANT);
        break;
    }
    if (st != NULL) {
        double t3 = tr_time_sec();
        st->k += t1 - t0;
        st->sm += t2 - t1;
        st->v += t3 - t2;
    }
}

/* heads a and a + 1: K(a); K(b) with exp(a); V(a) with exp(b); V(b) */
static void head_pair(const run_ctx *c, int64_t a, float *rowa, float *rowb, wstat *st) {
    const variant *v = c->v;
    const tr_kernels *kt = c->kt;
    const int64_t n = c->n_pos;
    double t0 = st != NULL ? tr_time_sec() : 0.0;
    float ma = kpass(kt, c->q + a * HEAD_DIM, head_keys(c, a), n, rowa, &v->kpf, 1, NULL, TR_ATTN_X);
    exp_job ea = {rowa, ma, {0}, 0};
    float mb = kpass(kt, c->q + (a + 1) * HEAD_DIM, head_keys(c, a + 1), n, rowb, &v->kpf, 1, &ea, v->gran);
    float sa = tr_lane_combine(ea.lane);
    double t1 = st != NULL ? tr_time_sec() : 0.0;
    exp_job eb = {rowb, mb, {0}, 0};
    vpass(kt, head_values(c, a), n, rowa, c->out + a * HEAD_DIM, &v->vpf, 1, sa, &eb, v->gran, 0);
    float sb = tr_lane_combine(eb.lane);
    vpass(kt, head_values(c, a + 1), n, rowb, c->out + (a + 1) * HEAD_DIM, &v->vpf, 1, sb, NULL, TR_ATTN_X, 0);
    if (st != NULL) {
        double t2 = tr_time_sec();
        st->k += t1 - t0;
        st->v += t2 - t1;
    }
}

static void heads_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const run_ctx *c = (const run_ctx *)ctx_;
    wstat *st = c->v->timed ? &c->stats[worker] : NULL;
    if (st != NULL) st->t_begin = tr_time_sec();
    float *rowa = c->scores + (int64_t)worker * 2 * KV_CTX, *rowb = rowa + KV_CTX;
    for (int64_t h = begin; h < end;) {
        if (c->v->scheme == S_PAIR && h + 1 < end) {
            head_pair(c, h, rowa, rowb, st);
            h += 2;
        } else {
            head_single(c, h, rowa, st, worker);
            h++;
        }
    }
    if (st != NULL) st->t_end = tr_time_sec();
}

/* split, first call: items (head, block of 64 positions) */
static void split_k_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const run_ctx *c = (const run_ctx *)ctx_;
    for (int64_t i = begin; i < end; i++) {
        int64_t h = i / c->n_blk, t0 = (i % c->n_blk) * TR_ATTN_BLOCK;
        int64_t t1 = t0 + TR_ATTN_BLOCK < c->n_pos ? t0 + TR_ATTN_BLOCK : c->n_pos;
        kblock(c->kt, c->q + h * HEAD_DIM, head_keys(c, h), t0, t1, c->n_pos * POS_BYTES,
               c->scores_all + h * KV_CTX, &c->v->kpf, NULL, TR_ATTN_X);
    }
}

/* split, second call: softmax and V per head */
static void split_sv_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const run_ctx *c = (const run_ctx *)ctx_;
    const int64_t v0 = c->v->v0 < c->n_pos * POS_BYTES ? c->v->v0 : c->n_pos * POS_BYTES;
    for (int64_t h = begin; h < end; h++) {
        float *row = c->scores_all + h * KV_CTX;
        softmax_copy(row, c->n_pos, (const char *)head_values(c, h), v0, c->v->v0_hint);
        vpass(c->kt, head_values(c, h), c->n_pos, row, c->out + h * HEAD_DIM, &c->v->vpf, 0, 0.0f, NULL, TR_ATTN_X,
              0);
    }
}

/* ---- the flush between two layers ------------------------------------------------------- */

typedef struct {
    const unsigned char *base;
    int64_t block;
} flush_ctx;

static void flush_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const flush_ctx *c = (const flush_ctx *)ctx_;
    uint64_t s = 0;
    for (int64_t i = begin; i < end; i++) s += read_bytes(c->base + i * c->block, c->block);
    sinks[worker].sum += s;
}

/* ---- the variants ------------------------------------------------------------------------- */

static int parse_variant(const char *spec, variant *v, int max_threads) {
    memset(v, 0, sizeof *v);
    char buf[64];
    if (strlen(spec) >= sizeof buf) return -1;
    memcpy(buf, spec, strlen(spec) + 1);
    snprintf(v->spec, sizeof v->spec, "%s", spec);
    v->threads = 8;
    v->gran = TR_ATTN_X;
    v->kpf.lines = v->vpf.lines = 32;
    v->kpf.hint = v->vpf.hint = 3;
    v->v0_hint = 2;
    char *at = strchr(buf, '@');
    if (at != NULL) {
        *at = '\0';
        v->threads = atoi(at + 1);
    }
    if (v->threads < 1) v->threads = 1;
    if (v->threads > max_threads) v->threads = max_threads;
    char *opt = strchr(buf, ':');
    if (opt != NULL) *opt++ = '\0';
    v->scheme = -1;
    for (int s = 0; s < S_COUNT; s++)
        if (strcmp(buf, scheme_name[s]) == 0) v->scheme = s;
    if (v->scheme < 0) return -1;
    while (opt != NULL && *opt != '\0') {
        char *next = strchr(opt, ':');
        if (next != NULL) *next++ = '\0';
        char *eq = strchr(opt, '=');
        if (eq == NULL) return -1;
        *eq = '\0';
        long long val = atoll(eq + 1);
        if (strcmp(opt, "kd") == 0) v->kpf.dist = val;
        else if (strcmp(opt, "vd") == 0) v->vpf.dist = val;
        else if (strcmp(opt, "kl") == 0) v->kpf.lines = (int)val;
        else if (strcmp(opt, "vl") == 0) v->vpf.lines = (int)val;
        else if (strcmp(opt, "h") == 0) v->kpf.hint = v->vpf.hint = (int)val;
        else if (strcmp(opt, "v0") == 0) v->v0 = val;
        else if (strcmp(opt, "v0h") == 0) v->v0_hint = (int)val;
        else if (strcmp(opt, "g") == 0) v->gran = (int)val;
        else if (strcmp(opt, "t") == 0) v->timed = (int)val;
        else if (strcmp(opt, "one") == 0) v->kpf.one = v->vpf.one = (int)val;
        else return -1;
        opt = next;
    }
    if (v->kpf.lines == 0 || v->kpf.lines > 32 || v->vpf.lines == 0 || v->vpf.lines > 32) return -1;
    if (v->gran < TR_ATTN_X || v->gran % TR_ATTN_X != 0) return -1;
    if (v->scheme == S_SPLIT) v->timed = 0;
    v->exact = v->scheme != S_READ && v->scheme != S_READ4 && v->scheme != S_NOSM && v->scheme != S_MUTANT;
    return 0;
}

typedef struct {
    tr_pool *pool;
    const tr_kernels *kt;
    float *kv, *q_all, *ref, *out, *scores, *scores_all;
    unsigned char *flush;
    int64_t n_pos;
    wstat *stats;
} bench;

/* One layer's attention with variant v, timed, its bits checked; then the flush: 64 MiB of other
 * memory read by every core of the pool, its seconds in *flush_secs (how loaded the machine is).
 * Returns the attention's seconds. */
static double run_layer(bench *b, variant *v, int L, long long *n_flush, double *flush_secs) {
    run_ctx c;
    memset(&c, 0, sizeof c);
    c.kt = b->kt;
    c.v = v;
    c.scores = b->scores;
    c.scores_all = b->scores_all;
    c.out = b->out;
    c.n_pos = b->n_pos;
    c.n_blk = (b->n_pos + TR_ATTN_BLOCK - 1) / TR_ATTN_BLOCK;
    c.stats = b->stats;
    c.keys = b->kv + (size_t)L * 2 * KV_CTX * N_KV;
    c.values = c.keys + (size_t)KV_CTX * N_KV;
    c.q = b->q_all + (size_t)L * N_KV;
    const int active = v->threads < N_HEAD ? v->threads : N_HEAD;
    memset(b->out, 0xFF, N_KV * sizeof(float)); /* NaN: a head nobody wrote cannot pass */
    if (v->timed) memset(b->stats, 0, sizeof(wstat) * MAX_WORKERS);
    tr_pool_set_active(b->pool, v->threads);
    double t0 = tr_time_sec();
    if (v->scheme == S_SPLIT) {
        tr_parallel_for(b->pool, N_HEAD * c.n_blk, 1, split_k_body, &c);
        tr_parallel_for(b->pool, N_HEAD, 1, split_sv_body, &c);
    } else {
        tr_parallel_for(b->pool, N_HEAD, 1 + 256 / b->n_pos, heads_body, &c);
    }
    double t1 = tr_time_sec();
    if (v->timed) {
        double first_end = 1e300, last_end = -1e300;
        for (int w = 0; w < active; w++) {
            const wstat *st = &b->stats[w];
            v->b_start += st->t_begin - t0;
            v->b_k += st->k;
            v->b_sm += st->sm;
            v->b_v += st->v;
            v->b_idle += t1 - st->t_end;
            v->w_end[w] += st->t_end - t0;
            if (st->t_end < first_end) first_end = st->t_end;
            if (st->t_end > last_end) last_end = st->t_end;
        }
        v->b_wall += t1 - t0;
        v->b_spread += last_end - first_end;
        v->b_layers++;
        v->b_workers += active;
    }
    v->compared++;
    if (memcmp(b->out, b->ref + (size_t)L * N_KV, N_KV * sizeof(float)) != 0) v->differ++;
    /* the engine reads its weights between two attentions: other memory through the caches */
    tr_pool_set_active(b->pool, 0);
    flush_ctx fl;
    fl.block = (int64_t)(FLUSH_BYTES / (size_t)tr_pool_size(b->pool)) & ~(int64_t)4095;
    fl.base = b->flush + (size_t)((*n_flush)++ % N_FLUSH) * FLUSH_BYTES;
    double f0 = tr_time_sec();
    tr_parallel_for(b->pool, tr_pool_size(b->pool), 1, flush_body, &fl);
    *flush_secs = tr_time_sec() - f0;
    return t1 - t0;
}

static void large_page_probe(void) {
#ifdef _WIN32
    SIZE_T min = GetLargePageMinimum();
    void *p = min > 0 ? VirtualAlloc(NULL, min, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE) : NULL;
    unsigned long err = p == NULL ? (unsigned long)GetLastError() : 0ul;
    printf("large pages: minimum %llu KiB, VirtualAlloc(MEM_LARGE_PAGES) %s (error %lu; 1314: privilege not held)\n",
           (unsigned long long)min / 1024, p != NULL ? "succeeded" : "failed", err);
    if (p != NULL) VirtualFree(p, 0, MEM_RELEASE);
#else
    printf("large pages: not probed here (Linux: transparent huge pages, madvise)\n");
#endif
}

/* the quantile p of s[0..n) sorted, linear between two neighbours */
static double quantile(const double *s, int n, double p) {
    double at = p * (double)(n - 1);
    int i = (int)at;
    return i + 1 < n ? s[i] + (at - (double)i) * (s[i + 1] - s[i]) : s[n - 1];
}

/* median of x[0..n); q[0..4] = min, p25, median, p75, max */
static double median_of(const double *x, int n, double *q) {
    double s[MAX_RUNS];
    memcpy(s, x, (size_t)n * sizeof s[0]);
    qsort(s, (size_t)n, sizeof s[0], cmp_double);
    q[0] = s[0];
    q[1] = quantile(s, n, 0.25);
    q[2] = quantile(s, n, 0.5);
    q[3] = quantile(s, n, 0.75);
    q[4] = s[n - 1];
    return q[2];
}

#define MAX_STEPS (MAX_RUNS * TOKENS * N_LAYER)

/* ratios x_ref / x_i over the steps in `keep` (all if NULL): median, p25, p75 into q; returns the count */
static int step_ratios(const double *ref, const double *x, int n, const unsigned char *keep, double *q) {
    static double r[MAX_STEPS];
    int m = 0;
    for (int s = 0; s < n; s++)
        if (keep == NULL || keep[s]) r[m++] = ref[s] / x[s];
    if (m == 0) {
        q[0] = q[1] = q[2] = 0.0;
        return 0;
    }
    qsort(r, (size_t)m, sizeof r[0], cmp_double);
    q[0] = quantile(r, m, 0.5);
    q[1] = quantile(r, m, 0.25);
    q[2] = quantile(r, m, 0.75);
    return m;
}

int main(int argc, char **argv) {
    int64_t n_pos = 2048;
    int n_runs = 11, n_vars = 0;
    static variant vars[MAX_VARIANTS];
    static double step_secs[MAX_VARIANTS][MAX_STEPS], step_flush[MAX_STEPS];
    static unsigned char quiet[MAX_STEPS];
    const char *specs[MAX_VARIANTS];
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) n_runs = atoi(argv[++i]);
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') n_pos = atoll(argv[i]);
        else if (n_vars < MAX_VARIANTS) specs[n_vars++] = argv[i];
    }
    if (n_vars == 0) {
        /* the finding, its two exact fixes, the plain reads, the check seen red, and an A/A copy */
        static const char *const dflt[] = {"engine@8",  "head@8",   "copy:one=1@8", "copy:kd=4096:vd=4096@8",
                                           "read@8",    "read4@8",  "mutant@8",     "engine@8"};
        for (size_t i = 0; i < sizeof dflt / sizeof dflt[0]; i++) specs[n_vars++] = dflt[i];
    }
    if (n_runs < 3) n_runs = 3;
    if (n_runs > MAX_RUNS) n_runs = MAX_RUNS;
    if (n_pos < 1 || n_pos > KV_CTX) {
        fprintf(stderr, "bench_attn_bw: n_pos from 1 to %d\n", KV_CTX);
        return 2;
    }

    tr_kernels_init();
    int cores = tr_cpu()->physical_cores;
    for (int i = 0; i < n_vars; i++) {
        if (parse_variant(specs[i], &vars[i], cores) != 0) {
            fprintf(stderr, "bench_attn_bw: bad variant '%s'\n", specs[i]);
            return 2;
        }
    }
    char cpu_line[512], err[256];
    tr_cpu_describe(tr_cpu(), cpu_line, sizeof cpu_line);
    printf("bench_attn_bw %d positions, built %s %s, kernels %s, median of %d\n%s\n", (int)n_pos, __DATE__, __TIME__,
           tr_kernels_get()->tier, n_runs, cpu_line);
    large_page_probe();

    const size_t kv_bytes = (size_t)N_LAYER * 2 * KV_CTX * N_KV * sizeof(float); /* 1 GiB */
    const size_t flush_bytes = (size_t)N_FLUSH * FLUSH_BYTES;                   /* 512 MiB */
    if (tr_mem_guard((uint64_t)(kv_bytes + flush_bytes + 16 * MIB), err, sizeof err) != 0) {
        fprintf(stderr, "bench_attn_bw: %s\n", err);
        return 1;
    }
    bench b;
    memset(&b, 0, sizeof b);
    b.n_pos = n_pos;
    b.kt = tr_kernels_get();
    b.kv = (float *)tr_alloc_aligned(kv_bytes, 4096);
    b.flush = (unsigned char *)tr_alloc_aligned(flush_bytes, 4096);
    b.q_all = (float *)tr_alloc_aligned((size_t)N_LAYER * N_KV * sizeof(float), 64);
    b.ref = (float *)tr_alloc_aligned((size_t)N_LAYER * N_KV * sizeof(float), 64);
    b.out = (float *)tr_alloc_aligned((size_t)N_KV * sizeof(float), 64);
    b.scores = (float *)tr_alloc_aligned((size_t)MAX_WORKERS * 2 * KV_CTX * sizeof(float), 64);
    b.scores_all = (float *)tr_alloc_aligned((size_t)N_HEAD * KV_CTX * sizeof(float), 64);
    b.stats = (wstat *)tr_alloc_aligned(sizeof(wstat) * MAX_WORKERS, 64);
    if (b.kv == NULL || b.flush == NULL || b.q_all == NULL || b.ref == NULL || b.out == NULL || b.scores == NULL ||
        b.scores_all == NULL || b.stats == NULL) {
        fprintf(stderr, "bench_attn_bw: out of memory\n");
        return 1;
    }
    b.pool = tr_pool_create(0);
    if (b.pool == NULL) return 1;
    fill(b.pool, b.kv, kv_bytes);
    fill(b.pool, b.flush, flush_bytes);
    fill(b.pool, b.q_all, (size_t)N_LAYER * N_KV * sizeof(float));

    /* the reference bits: the engine's kernel, one head after the other */
    for (int L = 0; L < N_LAYER; L++) {
        const float *keys = b.kv + (size_t)L * 2 * KV_CTX * N_KV, *values = keys + (size_t)KV_CTX * N_KV;
        for (int h = 0; h < N_HEAD; h++)
            tr_attention_group(b.q_all + (size_t)L * N_KV + h * HEAD_DIM, HEAD_DIM,
                               keys + (size_t)h * KV_CTX * HEAD_DIM, values + (size_t)h * KV_CTX * HEAD_DIM, 1, n_pos,
                               HEAD_DIM, SCALE, b.scores, KV_CTX, b.ref + (size_t)L * N_KV + h * HEAD_DIM, HEAD_DIM);
    }

    /* A repetition is TOKENS x 16 steps; every step runs every variant once, in an order rotated
     * at every step, so a burst of load elsewhere on the machine falls on all the variants of a
     * step alike. Run number n reads layer n mod 16: two runs on the same layer have 15 other
     * layers and 16 flushes between them, as a token has in the engine (two runs of the same layer
     * in a row, with a 128 MiB flush between them, read it at 55 GB/s instead of 47: the L3 keeps
     * part of it). r = -1 is the warm-up. */
    double t_start = tr_time_sec();
    long long n_flush = 0, n_run = 0;
    int n_steps = 0; /* timed steps */
    for (int r = -1; r < n_runs; r++) {
        double acc[MAX_VARIANTS] = {0};
        for (int step = 0; step < TOKENS * N_LAYER; step++) {
            double slowest_flush = 0.0;
            for (int j = 0; j < n_vars; j++) {
                int i = (j + step + (r > 0 ? r : 0)) % n_vars;
                double fs, s = run_layer(&b, &vars[i], (int)(n_run++ % N_LAYER), &n_flush, &fs);
                acc[i] += s;
                if (r >= 0) step_secs[i][n_steps] = s;
                if (fs > slowest_flush) slowest_flush = fs;
            }
            if (r >= 0) step_flush[n_steps++] = slowest_flush;
        }
        if (r < 0) { /* the breakdown counts timed runs only */
            /* safety of the machine: a run stays under 60 s */
            double warm = tr_time_sec() - t_start;
            if (warm * (double)(n_runs + 1) > 55.0) {
                int fit = (int)(55.0 / warm) - 1;
                if (fit < 3) {
                    fprintf(stderr, "bench_attn_bw: %.1f s per repetition, too long for 3 in 60 s: fewer variants\n",
                            warm);
                    return 2;
                }
                printf("(%.1f s per repetition: %d repetitions instead of %d)\n", warm, fit, n_runs);
                n_runs = fit;
            }
            for (int i = 0; i < n_vars; i++) {
                variant *v = &vars[i];
                v->b_wall = v->b_start = v->b_k = v->b_sm = v->b_v = v->b_idle = v->b_spread = 0.0;
                v->b_layers = v->b_workers = 0;
                memset(v->w_end, 0, sizeof v->w_end);
            }
            continue;
        }
        for (int i = 0; i < n_vars; i++) vars[i].secs[r] = acc[i] / TOKENS;
    }
    double elapsed = tr_time_sec() - t_start;

    const double kv_token = (double)N_LAYER * (double)n_pos * N_KV * 2 * sizeof(float);
    printf("KV read per token: %.1f MiB; ratio = per-repetition time of the first variant / this one: median [p25-p75]"
           " (min-max)\n",
           kv_token / (double)MIB);
    printf("%-38s %2s %8s %6s %6s  %-34s %s\n", "variant", "T", "ms/tok", "GB/s", "spread", "ratio", "bits");
    int bad = 0;
    for (int i = 0; i < n_vars; i++) {
        variant *v = &vars[i];
        double q[5], rq[5], med = median_of(v->secs, n_runs, q);
        double ratio[MAX_RUNS];
        for (int r = 0; r < n_runs; r++) ratio[r] = vars[0].secs[r] / v->secs[r];
        double rmed = median_of(ratio, n_runs, rq);
        char bits[64];
        if (v->compared == 0) {
            snprintf(bits, sizeof bits, "NOT COMPARED");
            bad = 1;
        } else if (v->exact) {
            if (v->differ == 0) snprintf(bits, sizeof bits, "same (%lld layers)", v->compared);
            else snprintf(bits, sizeof bits, "DIFFER in %lld of %lld", v->differ, v->compared);
            if (v->differ != 0) bad = 1;
        } else if (v->scheme == S_MUTANT && n_pos > TR_ATTN_BLOCK) {
            snprintf(bits, sizeof bits, "differ in %lld of %lld (must)", v->differ, v->compared);
            if (v->differ == 0) bad = 1;
        } else if (v->scheme == S_MUTANT) { /* one block: reversing the blocks changes nothing */
            snprintf(bits, sizeof bits, "no-op at <= %d positions", TR_ATTN_BLOCK);
        } else {
            snprintf(bits, sizeof bits, "diagnostic (differ %lld)", v->differ);
        }
        char rs[64];
        snprintf(rs, sizeof rs, "%.3f [%.3f-%.3f] (%.3f-%.3f)", rmed, rq[1], rq[3], rq[0], rq[4]);
        printf("%-38s %2d %8.3f %6.2f %5.1f%%  %-34s %s\n", v->spec, v->threads, med * 1e3, kv_token / med / 1e9,
               (q[4] - q[0]) / med * 100.0, rs, bits);
    }

    /* The same ratios step by step: the variants of a step run within a few ms of each other.
     * Quiet steps: those whose slowest flush (64 MiB, every core, not a variant) was within 10%
     * of the session's 5th percentile, i.e. when nothing else on the machine was reading memory. */
    {
        static double fs[MAX_STEPS];
        memcpy(fs, step_flush, (size_t)n_steps * sizeof fs[0]);
        qsort(fs, (size_t)n_steps, sizeof fs[0], cmp_double);
        double p5 = quantile(fs, n_steps, 0.05), fmed = quantile(fs, n_steps, 0.5);
        int n_quiet = 0;
        for (int s = 0; s < n_steps; s++) n_quiet += quiet[s] = step_flush[s] <= 1.10 * p5;
        printf("per step (%d steps; flush %.1f GB/s at p5, %.1f at the median; %d quiet steps, flush within 10%% of"
               " p5): ratio to the first variant median [p25-p75]; ms/token and GB/s from the median layer\n",
               n_steps, (double)FLUSH_BYTES / p5 / 1e9, (double)FLUSH_BYTES / fmed / 1e9, n_quiet);
        printf("%-38s %-24s %-24s %8s %6s\n", "variant", "all steps", "quiet steps", "ms/tok", "GB/s");
        for (int i = 0; i < n_vars; i++) {
            static double lay[MAX_STEPS];
            double qa[3], qq[3];
            step_ratios(step_secs[0], step_secs[i], n_steps, NULL, qa);
            int m = step_ratios(step_secs[0], step_secs[i], n_steps, quiet, qq);
            int k = 0;
            for (int s = 0; s < n_steps; s++)
                if (quiet[s]) lay[k++] = step_secs[i][s];
            qsort(lay, (size_t)k, sizeof lay[0], cmp_double);
            double lmed = k > 0 ? quantile(lay, k, 0.5) : 0.0;
            char sa[32], sq[32];
            snprintf(sa, sizeof sa, "%.3f [%.3f-%.3f]", qa[0], qa[1], qa[2]);
            snprintf(sq, sizeof sq, "%.3f [%.3f-%.3f]", qq[0], qq[1], qq[2]);
            printf("%-38s %-24s %-24s %8.3f %6.2f%s\n", vars[i].spec, sa, sq, lmed * N_LAYER * 1e3,
                   lmed > 0 ? kv_token / N_LAYER / lmed / 1e9 : 0.0, m < 10 ? "  (few quiet steps)" : "");
        }
    }
    for (int i = 0; i < n_vars; i++) {
        const variant *v = &vars[i];
        if (!v->timed || v->b_layers == 0) continue;
        double per_layer = 1e6 / (double)v->b_layers, per_worker = 1e6 / (double)v->b_workers;
        int heads_per_worker = N_HEAD / (v->threads < N_HEAD ? v->threads : N_HEAD);
        double kbytes = (double)N_HEAD * (double)n_pos * POS_BYTES; /* one layer's K, or V */
        printf("phases of %s (us per layer, mean over the %d workers): wall %.1f = start %.1f + K %.1f + softmax %.1f"
               " + V %.1f + end wait %.1f; ends spread %.1f; K %.1f GB/s, V %.1f GB/s while all read (%d heads each)\n",
               v->spec, v->threads < N_HEAD ? v->threads : N_HEAD, v->b_wall * per_layer, v->b_start * per_worker,
               v->b_k * per_worker, v->b_sm * per_worker, v->b_v * per_worker, v->b_idle * per_worker,
               v->b_spread * per_layer, kbytes / (v->b_k * per_worker * 1e-6) / 1e9,
               kbytes / (v->b_v * per_worker * 1e-6) / 1e9, heads_per_worker);
        printf("  each worker's end, us after the dispatch (worker 0 is the caller):");
        for (int w = 0; w < (v->threads < N_HEAD ? v->threads : N_HEAD); w++) printf(" %.1f", v->w_end[w] * per_layer);
        printf("\n");
    }
    uint64_t s = 0;
    for (int w = 0; w < MAX_WORKERS; w++) s += sinks[w].sum;
    printf("elapsed %.1f s (sink %llu)\n", elapsed, (unsigned long long)(s & 1));
    tr_pool_destroy(b.pool);
    tr_free_aligned(b.stats);
    tr_free_aligned(b.scores_all);
    tr_free_aligned(b.scores);
    tr_free_aligned(b.out);
    tr_free_aligned(b.ref);
    tr_free_aligned(b.q_all);
    tr_free_aligned(b.flush);
    tr_free_aligned(b.kv);
    if (bad) fprintf(stderr, "bench_attn_bw: bits check failed\n");
    return bad;
}
