/* bench_attn.c — the attention of a whole prompt on one layer, the way the prefill runs it, taken
 * apart (docs/MEASUREMENTS.md question 7 and "Prefill su prompt lunghi").
 *
 * The prefill runs a prompt in passes of 512 tokens. In a pass every (head, token) is one item:
 * token i at position pos0 + i multiplies its query with the keys of positions 0..pos0+i, takes
 * the softmax of that row and adds up the values with those weights. One thread ends up with one
 * head and all its tokens, so it reads the same keys and the same values again for every token:
 * 1 KiB per (token, position), 2 TB on a prompt of 4000. This benchmark says where the time of
 * that zone goes, with the shapes of OLMoE-1B-7B (16 heads of 128, f32, context 4096, keys and
 * values of a head in a row as in src/kv/kv.h):
 *
 *   one full      the engine's kernel (tr_attention_head), one query at a time
 *   one nosm      the same without the softmax: what expf, max, sum and division cost
 *   one dots      only the products of a query with its keys (reads K)
 *   one softmax   only the softmax of a row as long as the token's context (reads nothing)
 *   one wsum      only the weighted sum of the values (reads V)
 *   blk full      several queries against a block of keys while the block is in cache: a group
 *                 of G queries fills G rows of scores block by block of B positions, then the
 *                 softmax of every row, then the values block by block. Same kernel calls on the
 *                 same numbers in the same order for every output: the bits do not change, and
 *                 K and V are read once per group instead of once per query
 *   blk nosm      the same without the softmax
 *   blkx full     blk with one query against 4 keys in the registers and 4 values added to one
 *   blkx nosm     output per load (AVX-512 only): what is left to take from the arithmetic
 *   grp full      the engine's kernel for a group (tr_attention_group), called as the engine
 *                 calls it: what the measurements above became
 *
 *   bench_attn <n_pos> [--threads T] [--heads H] [--runs N] [--group G] [--block B] [--only one|blk|grp]
 *
 * n_pos is the length of the prompt; H heads are computed (16, or 1 to see one core alone with
 * its caches to itself). Between two passes the threads read 64 MiB of other memory, not timed:
 * the engine reads every weight of the model between two attentions of the same layer. Every
 * line is the median of N timed repetitions (default 5) after one warm-up, with min, max and
 * spread; "ns" is thread time per (query, position) pair; "bits" is a hash of every output of
 * the prompt, and blk, blkx and grp must print the hash of the matching "one" line (--group and
 * --block size blk and blkx; grp takes --group, its block is TR_ATTN_BLOCK). One run stays
 * well under 60 s (docs/ARCHITECTURE.md, safety of the machine). */
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

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAVE_X86 1
#else
#define HAVE_X86 0
#endif

#define MAX_RUNS 31
#define MAX_WORKERS 64
#define MAX_GROUP 64
#define MIB ((size_t)1 << 20)

/* OLMoE-1B-7B */
#define N_HEAD 16
#define HEAD_DIM 128
#define N_QKV (N_HEAD * HEAD_DIM)
#define KV_CTX 4096
#define N_BATCH 512
#define FLUSH_BYTES (64 * MIB)
#define SCALE 0.088388348f   /* 1 / sqrt(128) */
/* a row of scores per query of a group; not a multiple of 4 KiB, or the same position of every
 * row would fall in the same set of the L1 cache */
#define ROW_STRIDE (KV_CTX + TR_LANES)

enum {
    ONE_FULL, ONE_NOSM, ONE_DOTS, ONE_SOFTMAX, ONE_WSUM, BLK_FULL, BLK_NOSM, BLKX_FULL, BLKX_NOSM, GRP_FULL, N_VARIANT
};
static const char *const variant_name[N_VARIANT] = {"one full",  "one nosm",  "one dots",  "one softmax", "one wsum",
                                                    "blk full",  "blk nosm",  "blkx full", "blkx nosm",   "grp full"};

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static uint64_t rng_next(uint64_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s * 0x2545F4914F6CDD1Dull;
}

/* one 64-byte line per worker: a sum nobody can optimize away, and no false sharing */
typedef struct {
    volatile uint64_t sum;
    char pad[56];
} sink_t;
static sink_t sinks[MAX_WORKERS];

static uint32_t float_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    return bits;
}

/* floats in [-1, 1), every page written */
static void fill_floats(float *p, size_t n, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; i++) p[i] = (float)(int32_t)(rng_next(&s) >> 40) * (1.0f / 8388608.0f) - 1.0f;
}

/* ---- what the engine does between two attentions: other memory through the caches ---- */

typedef struct {
    const unsigned char *base;
    size_t share;
} flush_ctx;

static void flush_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const flush_ctx *c = (const flush_ctx *)ctx_;
    for (int64_t i = begin; i < end; i++) {
        const unsigned char *p = c->base + (size_t)i * c->share;
        uint64_t a = 0;
        for (size_t k = 0; k + 8 <= c->share; k += 8) {
            uint64_t w;
            memcpy(&w, p + k, 8);
            a += w;
        }
        sinks[worker].sum += a;
    }
}

/* ---- the variants -------------------------------------------------------------------- */

typedef struct {
    const float *q, *keys, *values, *row;
    float *scores, *out;
    int64_t n_tok, pos0, group, block;
    int variant;
} attn_ctx;

static void dots_plain(const tr_kernels *k, const float *q, const float *keys, int64_t t0, int64_t t1, float *scores) {
    for (int64_t t = t0; t < t1; t++) scores[t] = k->dot_f32(q, keys + t * HEAD_DIM, HEAD_DIM) * SCALE;
}

static void wsum_plain(const tr_kernels *k, float *out, const float *values, int64_t t0, int64_t t1,
                       const float *scores) {
    for (int64_t t = t0; t < t1; t++) k->axpy_f32(out, values + t * HEAD_DIM, scores[t], HEAD_DIM);
}

#if HAVE_X86
/* One query against 4 keys: one load of the query per 4 products, an accumulator per key in a
 * named register (docs/LESSONS.md #45). Each score is bit for bit avx512_dot_f32's. */
__attribute__((target("avx512f")))
static void dots_x4(const tr_kernels *k, const float *q, const float *keys, int64_t t0, int64_t t1, float *scores) {
    int64_t t = t0;
    for (; t + 4 <= t1; t += 4) {
        const float *k0 = keys + t * HEAD_DIM, *k1 = k0 + HEAD_DIM, *k2 = k1 + HEAD_DIM, *k3 = k2 + HEAD_DIM;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(),
               a3 = _mm512_setzero_ps();
        for (int64_t d = 0; d < HEAD_DIM; d += TR_LANES) {
            __m512 qv = _mm512_loadu_ps(q + d);
            a0 = _mm512_add_ps(a0, _mm512_mul_ps(qv, _mm512_loadu_ps(k0 + d)));
            a1 = _mm512_add_ps(a1, _mm512_mul_ps(qv, _mm512_loadu_ps(k1 + d)));
            a2 = _mm512_add_ps(a2, _mm512_mul_ps(qv, _mm512_loadu_ps(k2 + d)));
            a3 = _mm512_add_ps(a3, _mm512_mul_ps(qv, _mm512_loadu_ps(k3 + d)));
        }
        float lane[TR_LANES];
        _mm512_storeu_ps(lane, a0);
        scores[t] = tr_lane_combine(lane) * SCALE;
        _mm512_storeu_ps(lane, a1);
        scores[t + 1] = tr_lane_combine(lane) * SCALE;
        _mm512_storeu_ps(lane, a2);
        scores[t + 2] = tr_lane_combine(lane) * SCALE;
        _mm512_storeu_ps(lane, a3);
        scores[t + 3] = tr_lane_combine(lane) * SCALE;
    }
    dots_plain(k, q, keys, t, t1, scores);
}

/* 4 values added to one output, in increasing position: one load and one store of the output
 * per 4 additions. Each addition is bit for bit avx512_axpy_f32's. */
__attribute__((target("avx512f")))
static void wsum_x4(const tr_kernels *k, float *out, const float *values, int64_t t0, int64_t t1,
                    const float *scores) {
    int64_t t = t0;
    for (; t + 4 <= t1; t += 4) {
        const float *v0 = values + t * HEAD_DIM, *v1 = v0 + HEAD_DIM, *v2 = v1 + HEAD_DIM, *v3 = v2 + HEAD_DIM;
        __m512 s0 = _mm512_set1_ps(scores[t]), s1 = _mm512_set1_ps(scores[t + 1]),
               s2 = _mm512_set1_ps(scores[t + 2]), s3 = _mm512_set1_ps(scores[t + 3]);
        for (int64_t d = 0; d < HEAD_DIM; d += TR_LANES) {
            __m512 y = _mm512_loadu_ps(out + d);
            y = _mm512_add_ps(y, _mm512_mul_ps(s0, _mm512_loadu_ps(v0 + d)));
            y = _mm512_add_ps(y, _mm512_mul_ps(s1, _mm512_loadu_ps(v1 + d)));
            y = _mm512_add_ps(y, _mm512_mul_ps(s2, _mm512_loadu_ps(v2 + d)));
            y = _mm512_add_ps(y, _mm512_mul_ps(s3, _mm512_loadu_ps(v3 + d)));
            _mm512_storeu_ps(out + d, y);
        }
    }
    wsum_plain(k, out, values, t, t1, scores);
}
#endif

static void dots(int x, const tr_kernels *k, const float *q, const float *keys, int64_t t0, int64_t t1,
                 float *scores) {
#if HAVE_X86
    if (x) {
        dots_x4(k, q, keys, t0, t1, scores);
        return;
    }
#endif
    (void)x;
    dots_plain(k, q, keys, t0, t1, scores);
}

static void wsum(int x, const tr_kernels *k, float *out, const float *values, int64_t t0, int64_t t1,
                 const float *scores) {
#if HAVE_X86
    if (x) {
        wsum_x4(k, out, values, t0, t1, scores);
        return;
    }
#endif
    (void)x;
    wsum_plain(k, out, values, t0, t1, scores);
}

/* One query at a time, one item per (head, token) with the head first: the engine's attn_body. */
static void one_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const attn_ctx *c = (const attn_ctx *)ctx_;
    const tr_kernels *k = tr_kernels_get();
    float *scores = c->scores + (int64_t)worker * MAX_GROUP * ROW_STRIDE;
    for (int64_t idx = begin; idx < end; idx++) {
        int64_t h = idx / c->n_tok, i = idx % c->n_tok, n = c->pos0 + i + 1;
        const float *q = c->q + i * N_QKV + h * HEAD_DIM;
        const float *keys = c->keys + h * KV_CTX * HEAD_DIM, *values = c->values + h * KV_CTX * HEAD_DIM;
        float *out = c->out + i * N_QKV + h * HEAD_DIM;
        switch (c->variant) {
        case ONE_FULL:
            tr_attention_head(q, keys, values, HEAD_DIM, 0, n, HEAD_DIM, SCALE, scores, out);
            break;
        case ONE_NOSM:
            dots_plain(k, q, keys, 0, n, scores);
            for (int64_t d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;
            wsum_plain(k, out, values, 0, n, scores);
            break;
        case ONE_DOTS:
            dots_plain(k, q, keys, 0, n, scores);
            sinks[worker].sum += float_bits(scores[n - 1]);
            break;
        case ONE_SOFTMAX:
            memcpy(scores, c->row, (size_t)n * sizeof(float));
            tr_softmax(scores, n);
            sinks[worker].sum += float_bits(scores[n - 1]);
            break;
        default: /* ONE_WSUM */
            for (int64_t d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;
            wsum_plain(k, out, values, 0, n, c->row);
            break;
        }
    }
}

/* n_q queries of one head, tokens g0 .. g0 + n_q - 1 of the pass: query j sees first_n + j
 * positions. Row j of scores is filled block by block, only by the queries that see the block. */
static void attn_group(const attn_ctx *c, const tr_kernels *k, int64_t h, int64_t g0, int64_t n_q, float *scores) {
    int x = c->variant == BLKX_FULL || c->variant == BLKX_NOSM;
    int softmax = c->variant == BLK_FULL || c->variant == BLKX_FULL;
    const float *keys = c->keys + h * KV_CTX * HEAD_DIM, *values = c->values + h * KV_CTX * HEAD_DIM;
    int64_t first_n = c->pos0 + g0 + 1, last_n = first_n + n_q - 1;
    for (int64_t t0 = 0; t0 < last_n; t0 += c->block) {
        int64_t t1 = t0 + c->block < last_n ? t0 + c->block : last_n;
        int64_t j_min = t0 + 1 > first_n ? t0 + 1 - first_n : 0;
        for (int64_t j = j_min; j < n_q; j++) {
            int64_t n_j = first_n + j;
            dots(x, k, c->q + (g0 + j) * N_QKV + h * HEAD_DIM, keys, t0, t1 < n_j ? t1 : n_j, scores + j * ROW_STRIDE);
        }
    }
    if (softmax)
        for (int64_t j = 0; j < n_q; j++) tr_softmax(scores + j * ROW_STRIDE, first_n + j);
    for (int64_t j = 0; j < n_q; j++) {
        float *out = c->out + (g0 + j) * N_QKV + h * HEAD_DIM;
        for (int64_t d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;
    }
    for (int64_t t0 = 0; t0 < last_n; t0 += c->block) {
        int64_t t1 = t0 + c->block < last_n ? t0 + c->block : last_n;
        int64_t j_min = t0 + 1 > first_n ? t0 + 1 - first_n : 0;
        for (int64_t j = j_min; j < n_q; j++) {
            int64_t n_j = first_n + j;
            wsum(x, k, c->out + (g0 + j) * N_QKV + h * HEAD_DIM, values, t0, t1 < n_j ? t1 : n_j,
                 scores + j * ROW_STRIDE);
        }
    }
}

/* The same items as one_body; a chunk is cut into heads, a head's tokens into groups. */
static void blk_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const attn_ctx *c = (const attn_ctx *)ctx_;
    const tr_kernels *k = tr_kernels_get();
    float *scores = c->scores + (int64_t)worker * MAX_GROUP * ROW_STRIDE;
    int64_t idx = begin;
    while (idx < end) {
        int64_t h = idx / c->n_tok, i0 = idx % c->n_tok;
        int64_t i_end = (h + 1) * c->n_tok <= end ? c->n_tok : end - h * c->n_tok;
        for (int64_t g0 = i0; g0 < i_end; g0 += c->group) {
            int64_t n_q = g0 + c->group <= i_end ? c->group : i_end - g0;
            if (c->variant == GRP_FULL)
                tr_attention_group(c->q + g0 * N_QKV + h * HEAD_DIM, N_QKV, c->keys + h * KV_CTX * HEAD_DIM,
                                   c->values + h * KV_CTX * HEAD_DIM, n_q, c->pos0 + g0 + 1, HEAD_DIM, SCALE, scores,
                                   ROW_STRIDE, c->out + g0 * N_QKV + h * HEAD_DIM, N_QKV);
            else
                attn_group(c, k, h, g0, n_q, scores);
        }
        idx = h * c->n_tok + i_end;
    }
}

/* every output of the pass, heads 0..n_heads-1 */
static uint64_t hash_out(uint64_t hash, const float *out, int64_t n_tok, int64_t n_heads) {
    for (int64_t i = 0; i < n_tok; i++)
        for (int64_t d = 0; d < n_heads * HEAD_DIM; d++)
            hash = (hash ^ float_bits(out[i * N_QKV + d])) * 0x100000001B3ull;
    return hash;
}

int main(int argc, char **argv) {
    int64_t n_pos = 0, group = 16, block = 64;
    int n_runs = 5, threads = 0, heads = N_HEAD;
    const char *only = "";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) n_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--group") == 0 && i + 1 < argc) group = atoll(argv[++i]);
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = atoll(argv[++i]);
        else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[++i];
        else n_pos = atoll(argv[i]);
    }
    if (n_pos < 1 || n_pos > KV_CTX || heads < 1 || heads > N_HEAD || group < 1 || group > MAX_GROUP || block < 1) {
        fprintf(stderr,
                "usage: bench_attn <n_pos 1..%d> [--threads T] [--heads 1..%d] [--runs N] [--group 1..%d] "
                "[--block B] [--only one|blk|grp]\n",
                KV_CTX, N_HEAD, MAX_GROUP);
        return 2;
    }
    if (n_runs < 3) n_runs = 3;
    if (n_runs > MAX_RUNS) n_runs = MAX_RUNS;
    int cores = tr_cpu()->physical_cores;
    if (threads < 1 || threads > cores) threads = cores;   /* never more threads than cores */
    if (threads > MAX_WORKERS) threads = MAX_WORKERS;

    tr_kernels_init();
    char cpu_line[512], err[256];
    tr_cpu_describe(tr_cpu(), cpu_line, sizeof cpu_line);
    /* which build this is: a stale benchmark measures the code of another day (docs/LESSONS.md #24) */
    printf("bench_attn %d, built %s %s, kernels %s, median of %d, group %d, block %d\n%s\n", (int)n_pos, __DATE__,
           __TIME__, tr_kernels_get()->tier, n_runs, (int)group, (int)block, cpu_line);
    if (tr_mem_guard((uint64_t)256 * MIB, err, sizeof err) != 0) {
        fprintf(stderr, "bench_attn: %s\n", err);
        return 1;
    }

    const size_t kv_floats = (size_t)N_HEAD * KV_CTX * HEAD_DIM;
    float *keys = (float *)tr_alloc_aligned(kv_floats * sizeof(float), 4096);
    float *values = (float *)tr_alloc_aligned(kv_floats * sizeof(float), 4096);
    float *q = (float *)tr_alloc_aligned((size_t)N_BATCH * N_QKV * sizeof(float), 64);
    float *out = (float *)tr_alloc_aligned((size_t)N_BATCH * N_QKV * sizeof(float), 64);
    float *row = (float *)tr_alloc_aligned((size_t)KV_CTX * sizeof(float), 64);
    float *scores = (float *)tr_alloc_aligned((size_t)threads * MAX_GROUP * ROW_STRIDE * sizeof(float), 64);
    unsigned char *other = (unsigned char *)tr_alloc_aligned(FLUSH_BYTES, 4096);
    if (keys == NULL || values == NULL || q == NULL || out == NULL || row == NULL || scores == NULL || other == NULL) {
        fprintf(stderr, "bench_attn: out of memory\n");
        return 1;
    }
    fill_floats(keys, kv_floats, 1);
    fill_floats(values, kv_floats, 2);
    fill_floats(q, (size_t)N_BATCH * N_QKV, 3);
    memset(out, 0, (size_t)N_BATCH * N_QKV * sizeof(float));
    memset(scores, 0, (size_t)threads * MAX_GROUP * ROW_STRIDE * sizeof(float));
    memset(other, 1, FLUSH_BYTES);
    /* a row of scores as the products leave them: where "one softmax" starts from, and the
     * weights of "one wsum" */
    dots_plain(tr_kernels_get(), q, keys, 0, KV_CTX, row);

    tr_pool *pool = tr_pool_create(threads);
    if (pool == NULL) return 1;
    double pairs = (double)heads * (double)n_pos * (double)(n_pos + 1) / 2.0;
    double t_start = tr_time_sec();
    uint64_t hashes[N_VARIANT] = {0};

    for (int v = 0; v < N_VARIANT; v++) {
        if (only[0] != '\0' && strncmp(variant_name[v], only, strlen(only)) != 0) continue;
        if ((v == BLKX_FULL || v == BLKX_NOSM) && !(HAVE_X86 && tr_cpu()->avx512f)) continue;
        attn_ctx c;
        c.q = q;
        c.keys = keys;
        c.values = values;
        c.row = row;
        c.scores = scores;
        c.out = out;
        c.group = group;
        c.block = block;
        c.variant = v;
        flush_ctx f;
        f.base = other;
        f.share = (FLUSH_BYTES / (size_t)threads) & ~(size_t)4095;
        double secs[MAX_RUNS];
        for (int r = -1; r < n_runs; r++) {
            double sum = 0.0;
            uint64_t hash = 0xCBF29CE484222325ull;
            for (int64_t pos0 = 0; pos0 < n_pos; pos0 += N_BATCH) {
                c.pos0 = pos0;
                c.n_tok = n_pos - pos0 < N_BATCH ? n_pos - pos0 : N_BATCH;
                tr_parallel_for(pool, threads, 1, flush_body, &f);
                double t0 = tr_time_sec();
                tr_parallel_for(pool, heads * c.n_tok, 1 + 256 / (pos0 + c.n_tok), v < BLK_FULL ? one_body : blk_body,
                                &c);
                sum += tr_time_sec() - t0;
                if (r < 0 && v != ONE_DOTS && v != ONE_SOFTMAX) hash = hash_out(hash, out, c.n_tok, heads);
            }
            if (r >= 0) secs[r] = sum;
            else hashes[v] = hash;
        }
        qsort(secs, (size_t)n_runs, sizeof secs[0], cmp_double);
        double med = (n_runs % 2) ? secs[n_runs / 2] : 0.5 * (secs[n_runs / 2 - 1] + secs[n_runs / 2]);
        printf("attn %4d %-11s t=%-2d h=%-2d  %9.3f ms  (min %9.3f  max %9.3f  spread %4.1f%%)  %6.2f ns  %7.2f GB/s",
               (int)n_pos, variant_name[v], threads, heads, med * 1e3, secs[0] * 1e3, secs[n_runs - 1] * 1e3,
               (secs[n_runs - 1] - secs[0]) / med * 100.0, med * 1e9 * (double)threads / pairs,
               pairs * 2.0 * HEAD_DIM * sizeof(float) / med / 1e9);
        if (v != ONE_DOTS && v != ONE_SOFTMAX) printf("  bits %016llx", (unsigned long long)hashes[v]);
        printf("\n");
        fflush(stdout);
    }

    /* the blocked variants are worth a number only if they give the engine's bits */
    int bad = 0, compared = 0;
    for (int v = BLK_FULL; v < N_VARIANT; v++) {
        int ref = (v == BLK_FULL || v == BLKX_FULL || v == GRP_FULL) ? ONE_FULL : ONE_NOSM;
        if (hashes[v] == 0 || hashes[ref] == 0) continue;
        compared++;
        if (hashes[v] != hashes[ref]) {
            printf("bench_attn: '%s' does not give the bits of '%s'\n", variant_name[v], variant_name[ref]);
            bad = 1;
        }
    }
    if (compared == 0) printf("bits: not compared (no blocked variant ran together with its 'one' line)\n");
    else if (!bad) printf("bits: the %d blocked variants that ran give the bits of their 'one' line\n", compared);
    printf("elapsed %.1f s\n", tr_time_sec() - t_start);

    tr_pool_destroy(pool);
    tr_free_aligned(other);
    tr_free_aligned(scores);
    tr_free_aligned(row);
    tr_free_aligned(out);
    tr_free_aligned(q);
    tr_free_aligned(values);
    tr_free_aligned(keys);
    return bad;
}
