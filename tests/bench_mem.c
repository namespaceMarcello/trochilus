/* bench_mem.c — what the RAM of this machine gives, and what the two big readers of a decode
 * token get out of it (docs/MISURE.md question 4 and "Decode a contesto lungo").
 *
 * A decode token reads ~1.2 GB of weights plus the KV cache of its whole context, far more
 * than any cache level holds: its ceiling is the bandwidth of the RAM, and the profiler's
 * bytes per zone (src/base/prof.h) are to be held against the numbers printed here.
 * Three groups, each one run well under 60 s (docs/ARCHITETTURA.md, safety of the machine):
 *
 *   bench_mem ram          plain reads (64-bit sums: no arithmetic worth the name) of 2 GiB:
 *                            seq      every thread walks its own contiguous share
 *                            blocks   the same memory as blocks of 2 MiB (one expert matrix),
 *                                     256 KiB (one thread's share of one) and 4 KiB (a page),
 *                                     visited in random order
 *   bench_mem weights      the engine's own matmul (tr_matmul_grouped, Q8_0, one token) over 8
 *                          random expert-sized matrices per call, out of 1 GiB of them: what
 *                          the kernels pull through the same bus
 *   bench_mem kv <n_pos>   one decode token's attention over n_pos cached positions of a KV
 *                          cache shaped like OLMoE-1B-7B's (16 layers, 16 heads of 128, f32, K
 *                          and V, context 4096), with 64 MiB of other memory read between two
 *                          layers as the engine reads its weights (not timed). The real kernel
 *                          (tr_attention_head) and plain reads of the same bytes, in two layouts:
 *                            rows     [position][head]: a head reads 512 B every 8 KiB
 *                            heads    [head][position]: a head reads its positions in a row
 *
 *   bench_mem <group> [--runs N]
 *
 * Threads are the engine's pool, so they sit on the cores the engine would use. Every line is
 * the median of N timed repetitions (default 7) after one warm-up, with min, max and spread
 * (max - min) / median: a difference smaller than the spread is not a difference. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/models/model_internal.h"

#define MAX_RUNS 31
#define MAX_WORKERS 64
#define GIB ((size_t)1 << 30)
#define MIB ((size_t)1 << 20)

/* OLMoE-1B-7B: the shapes the engine reads at every token */
#define N_LAYER 16
#define N_HEAD 16
#define HEAD_DIM 128
#define N_KV (N_HEAD * HEAD_DIM)
#define KV_CTX 4096
#define EXP_ROWS 1024
#define EXP_COLS 2048
#define EXP_USED 8
#define FLUSH_BYTES (64 * MIB)

static int n_runs = 7;

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

/* n bytes, a multiple of 64 */
static uint64_t read_bytes(const unsigned char *p, size_t n) {
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (size_t i = 0; i < n; i += 64) {
        uint64_t w[8];
        memcpy(w, p + i, 64);
        a += w[0] + w[4];
        b += w[1] + w[5];
        c += w[2] + w[6];
        d += w[3] + w[7];
    }
    return a ^ b ^ c ^ d;
}

/* a result kept alive without converting a float that may not fit an integer */
static uint32_t float_bits(const float *p) {
    uint32_t bits;
    memcpy(&bits, p, sizeof bits);
    return bits;
}

/* ---- filling: every page written, so that none is the shared zero page ------ */

typedef struct {
    unsigned char *base;
    size_t bytes;
    int shares;
    int as_float;
} fill_ctx;

static void fill_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const fill_ctx *c = (const fill_ctx *)ctx_;
    size_t share = c->bytes / (size_t)c->shares;
    for (int64_t i = begin; i < end; i++) {
        uint64_t s = 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1);
        unsigned char *p = c->base + (size_t)i * share;
        if (c->as_float) {
            for (size_t k = 0; k < share; k += 4) {
                float v = (float)(int32_t)(rng_next(&s) >> 40) * (1.0f / 8388608.0f) - 1.0f;
                memcpy(p + k, &v, 4);
            }
        } else {
            for (size_t k = 0; k < share; k += 8) {
                uint64_t v = rng_next(&s);
                memcpy(p + k, &v, 8);
            }
        }
    }
}

static void fill(tr_pool *pool, unsigned char *base, size_t bytes, int as_float) {
    fill_ctx c = {base, bytes, 64, as_float};
    tr_parallel_for(pool, c.shares, 1, fill_body, &c);
}

/* ---- one line of output ------------------------------------------------------ */

/* secs[0..n): the timed repetitions; bytes: what each one read */
static void report(const char *name, int threads, double *secs, int n, double bytes, double items) {
    qsort(secs, (size_t)n, sizeof secs[0], cmp_double);
    double med = (n % 2) ? secs[n / 2] : 0.5 * (secs[n / 2 - 1] + secs[n / 2]);
    printf("%-26s t=%-2d  %7.2f GB/s  (min %6.2f  max %6.2f  spread %4.1f%%)", name, threads, bytes / med / 1e9,
           bytes / secs[n - 1] / 1e9, bytes / secs[0] / 1e9, (secs[n - 1] - secs[0]) / med * 100.0);
    if (items > 0) printf("  %7.3f ms  %6.1f ns per (head, position) of thread time", med * 1e3,
                          med * 1e9 * (double)(threads < N_HEAD ? threads : N_HEAD) / items);
    printf("\n");
    fflush(stdout);
}

static int thread_counts(int *out) {
    static const int want[] = {1, 2, 4, 6, 8, 12, 16};
    int cores = tr_cpu()->physical_cores, n = 0;
    for (size_t i = 0; i < sizeof want / sizeof want[0]; i++)
        if (want[i] <= cores) out[n++] = want[i];
    if (n == 0 || out[n - 1] != cores) out[n++] = cores;   /* never more threads than cores */
    return n;
}

/* ---- ram: plain reads --------------------------------------------------------- */

typedef struct {
    const unsigned char *base;
    size_t block;            /* bytes of one item */
    const uint32_t *order;   /* item -> block index; NULL: in a row */
} ram_ctx;

static void ram_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const ram_ctx *c = (const ram_ctx *)ctx_;
    uint64_t s = 0;
    for (int64_t i = begin; i < end; i++) {
        size_t blk = c->order != NULL ? c->order[i] : (size_t)i;
        s += read_bytes(c->base + blk * c->block, c->block);
    }
    sinks[worker].sum += s;
}

static void shuffle(uint32_t *a, size_t n, uint64_t seed) {
    for (size_t i = 0; i < n; i++) a[i] = (uint32_t)i;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rng_next(&seed) % (uint64_t)(i + 1));
        uint32_t t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

static int group_ram(void) {
    const size_t arena_bytes = 2 * GIB;
    unsigned char *arena = (unsigned char *)tr_alloc_aligned(arena_bytes, 4096);
    uint32_t *order = (uint32_t *)malloc((arena_bytes / 4096) * sizeof(uint32_t));
    if (arena == NULL || order == NULL) {
        fprintf(stderr, "bench_mem: out of memory\n");
        return 1;
    }
    tr_pool *all = tr_pool_create(0);
    fill(all, arena, arena_bytes, 0);
    tr_pool_destroy(all);

    /* name, block bytes (0: one contiguous share per thread), bytes read per repetition */
    static const struct {
        const char *name;
        size_t block, volume;
    } pat[] = {
        {"ram seq", 0, 2 * GIB},
        {"ram blocks 2 MiB", 2 * MIB, 2 * GIB},
        {"ram blocks 256 KiB", 256 * 1024, 1 * GIB},
        {"ram blocks 4 KiB", 4096, 512 * MIB},
    };
    int counts[16], n_counts = thread_counts(counts);
    for (size_t p = 0; p < sizeof pat / sizeof pat[0]; p++) {
        for (int ti = 0; ti < n_counts; ti++) {
            int T = counts[ti];
            tr_pool *pool = tr_pool_create(T);
            if (pool == NULL) return 1;
            ram_ctx c;
            c.base = arena;
            int64_t n_items;
            if (pat[p].block == 0) {
                c.block = (arena_bytes / (size_t)T) & ~(size_t)4095;
                c.order = NULL;
                n_items = T;
            } else {
                c.block = pat[p].block;
                shuffle(order, arena_bytes / c.block, 0xC0FFEEull + p);
                c.order = order;
                n_items = (int64_t)(pat[p].volume / c.block);
            }
            double secs[MAX_RUNS];
            for (int r = -1; r < n_runs; r++) {
                double t0 = tr_time_sec();
                tr_parallel_for(pool, n_items, 1, ram_body, &c);
                double t1 = tr_time_sec();
                if (r >= 0) secs[r] = t1 - t0;
            }
            report(pat[p].name, T, secs, n_runs, (double)n_items * (double)c.block, 0);
            tr_pool_destroy(pool);
        }
    }
    free(order);
    tr_free_aligned(arena);
    return 0;
}

/* ---- weights: the engine's matmul over random expert-sized matrices -------------- */

static int group_weights(void) {
    size_t row_bytes = tr_row_bytes(TR_TYPE_Q8_0, EXP_COLS);
    size_t mat_bytes = (size_t)EXP_ROWS * row_bytes;
    size_t n_mats = GIB / mat_bytes;
    size_t arena_bytes = n_mats * mat_bytes;
    unsigned char *arena = (unsigned char *)tr_alloc_aligned(GIB, 4096);
    uint32_t *order = (uint32_t *)malloc(n_mats * sizeof(uint32_t));
    float *x = (float *)tr_alloc_aligned((size_t)EXP_USED * EXP_COLS * sizeof(float), 64);
    float *y = (float *)tr_alloc_aligned((size_t)EXP_USED * EXP_ROWS * sizeof(float), 64);
    if (arena == NULL || order == NULL || x == NULL || y == NULL) {
        fprintf(stderr, "bench_mem: out of memory\n");
        return 1;
    }
    tr_pool *all = tr_pool_create(0);
    fill(all, arena, GIB, 0);
    fill(all, (unsigned char *)x, (size_t)EXP_USED * EXP_COLS * sizeof(float), 1);
    tr_pool_destroy(all);
    /* a Q8_0 block is an fp16 scale and 32 int8: a small positive scale, the quants as they are */
    uint64_t s = 0xABCDEFull;
    for (size_t b = 0; b + 34 <= arena_bytes; b += 34) {
        uint16_t d = (uint16_t)(0x3000 | (rng_next(&s) & 0x3FF));
        memcpy(arena + b, &d, 2);
    }
    shuffle(order, n_mats, 0xFEEDull);

    int64_t offsets[EXP_USED + 1];
    for (int i = 0; i <= EXP_USED; i++) offsets[i] = i;
    size_t n_calls = n_mats / EXP_USED;
    int counts[16], n_counts = thread_counts(counts);
    for (int ti = 0; ti < n_counts; ti++) {
        int T = counts[ti];
        tr_pool *pool = tr_pool_create(T);
        if (pool == NULL) return 1;
        double secs[MAX_RUNS];
        for (int r = -1; r < n_runs; r++) {
            double t0 = tr_time_sec();
            for (size_t call = 0; call < n_calls; call++) {
                tr_mat w[EXP_USED];
                for (int e = 0; e < EXP_USED; e++) {
                    w[e].type = TR_TYPE_Q8_0;
                    w[e].rows = EXP_ROWS;
                    w[e].cols = EXP_COLS;
                    w[e].data = arena + (size_t)order[call * EXP_USED + (size_t)e] * mat_bytes;
                }
                tr_matmul_grouped(pool, w, offsets, EXP_USED, x, y);
            }
            double t1 = tr_time_sec();
            if (r >= 0) secs[r] = t1 - t0;
        }
        sinks[0].sum += float_bits(y);
        report("weights matmul q8_0 x8", T, secs, n_runs, (double)(n_calls * EXP_USED) * (double)mat_bytes, 0);
        tr_pool_destroy(pool);
    }
    tr_free_aligned(y);
    tr_free_aligned(x);
    free(order);
    tr_free_aligned(arena);
    return 0;
}

/* ---- kv: one token's attention, two layouts --------------------------------------- */

typedef struct {
    const float *keys, *values;   /* the layer's K and V */
    const float *q;
    float *scores, *out;
    int64_t n_pos;
    int by_head;                  /* layout: 0 rows [position][head], 1 heads [head][position] */
    int plain;                    /* 1: read the bytes, no kernel */
} kv_ctx;

static void kv_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const kv_ctx *c = (const kv_ctx *)ctx_;
    float scale = 0.088388348f;   /* 1 / sqrt(128) */
    for (int64_t h = begin; h < end; h++) {
        const float *k = c->by_head ? c->keys + h * KV_CTX * HEAD_DIM : c->keys;
        const float *v = c->by_head ? c->values + h * KV_CTX * HEAD_DIM : c->values;
        int64_t stride = c->by_head ? HEAD_DIM : N_KV, offset = c->by_head ? 0 : h * HEAD_DIM;
        if (c->plain) {
            uint64_t s = 0;
            for (int64_t t = 0; t < c->n_pos; t++)
                s += read_bytes((const unsigned char *)(k + t * stride + offset), HEAD_DIM * sizeof(float));
            for (int64_t t = 0; t < c->n_pos; t++)
                s += read_bytes((const unsigned char *)(v + t * stride + offset), HEAD_DIM * sizeof(float));
            sinks[worker].sum += s;
        } else {
            tr_attention_head(c->q + h * HEAD_DIM, k, v, stride, offset, c->n_pos, HEAD_DIM, scale,
                              c->scores + (int64_t)worker * KV_CTX, c->out + h * HEAD_DIM);
        }
    }
}

static int group_kv(int64_t n_pos) {
    if (n_pos < 1 || n_pos > KV_CTX) {
        fprintf(stderr, "bench_mem: kv wants 1 to %d positions\n", KV_CTX);
        return 2;
    }
    /* the same cache twice, once per layout; the flush reads the other copy */
    const size_t layer_bytes = (size_t)KV_CTX * N_KV * sizeof(float);
    const size_t cache_bytes = 2 * (size_t)N_LAYER * layer_bytes;   /* K and V: 1 GiB */
    unsigned char *arena[2];
    arena[0] = (unsigned char *)tr_alloc_aligned(cache_bytes, 4096);
    arena[1] = (unsigned char *)tr_alloc_aligned(cache_bytes, 4096);
    float *q = (float *)tr_alloc_aligned((size_t)N_KV * sizeof(float), 64);
    float *out = (float *)tr_alloc_aligned((size_t)N_KV * sizeof(float), 64);
    float *scores = (float *)tr_alloc_aligned((size_t)MAX_WORKERS * KV_CTX * sizeof(float), 64);
    if (arena[0] == NULL || arena[1] == NULL || q == NULL || out == NULL || scores == NULL) {
        fprintf(stderr, "bench_mem: out of memory\n");
        return 1;
    }
    tr_pool *all = tr_pool_create(0);
    fill(all, arena[0], cache_bytes, 1);
    fill(all, arena[1], cache_bytes, 1);
    fill(all, (unsigned char *)q, (size_t)N_KV * sizeof(float), 1);
    tr_pool_destroy(all);

    static const int counts[] = {4, 8, 16};
    static const char *const layout_name[2] = {"rows", "heads"};
    const int tokens = 2;
    int cores = tr_cpu()->physical_cores;
    for (int plain = 0; plain <= 1; plain++) {
        for (int by_head = 0; by_head <= 1; by_head++) {
            for (size_t ti = 0; ti < sizeof counts / sizeof counts[0]; ti++) {
                int T = counts[ti] <= cores ? counts[ti] : cores;
                tr_pool *pool = tr_pool_create(T);
                if (pool == NULL) return 1;
                kv_ctx c;
                c.q = q;
                c.scores = scores;
                c.out = out;
                c.n_pos = n_pos;
                c.by_head = by_head;
                c.plain = plain;
                ram_ctx flush;
                flush.base = arena[1 - by_head];
                flush.block = (FLUSH_BYTES / (size_t)T) & ~(size_t)4095;
                flush.order = NULL;
                double secs[MAX_RUNS];
                for (int r = -1; r < n_runs; r++) {
                    double sum = 0.0;
                    for (int tok = 0; tok < tokens; tok++) {
                        for (int L = 0; L < N_LAYER; L++) {
                            const float *base = (const float *)(const void *)arena[by_head];
                            c.keys = base + (size_t)L * 2 * (size_t)KV_CTX * N_KV;
                            c.values = c.keys + (size_t)KV_CTX * N_KV;
                            double t0 = tr_time_sec();
                            tr_parallel_for(pool, N_HEAD, 1 + 256 / n_pos, kv_body, &c);
                            sum += tr_time_sec() - t0;
                            /* what the engine does between two attentions: other memory through the caches */
                            flush.base = arena[1 - by_head] + (size_t)((tok * N_LAYER + L) % 8) * 2 * FLUSH_BYTES;
                            tr_parallel_for(pool, T, 1, ram_body, &flush);
                        }
                    }
                    if (r >= 0) secs[r] = sum / tokens;
                }
                sinks[0].sum += float_bits(out);
                char name[64];
                snprintf(name, sizeof name, "kv %4d %-5s %s", (int)n_pos, layout_name[by_head],
                         plain ? "read" : "attention");
                report(name, T, secs, n_runs, (double)N_LAYER * (double)n_pos * N_KV * 2 * sizeof(float),
                       (double)N_LAYER * N_HEAD * (double)n_pos);
                tr_pool_destroy(pool);
            }
        }
    }
    tr_free_aligned(scores);
    tr_free_aligned(out);
    tr_free_aligned(q);
    tr_free_aligned(arena[1]);
    tr_free_aligned(arena[0]);
    return 0;
}

int main(int argc, char **argv) {
    const char *group = argc > 1 ? argv[1] : "";
    int64_t n_pos = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) n_runs = atoi(argv[++i]);
        else n_pos = atoll(argv[i]);
    }
    if (n_runs < 3) n_runs = 3;
    if (n_runs > MAX_RUNS) n_runs = MAX_RUNS;
    int known = strcmp(group, "ram") == 0 || strcmp(group, "weights") == 0 || strcmp(group, "kv") == 0;
    if (!known) {
        fprintf(stderr, "usage: bench_mem ram | weights | kv <n_pos>   [--runs N]\n");
        return 2;
    }

    tr_kernels_init();
    char cpu_line[512], err[256];
    tr_cpu_describe(tr_cpu(), cpu_line, sizeof cpu_line);
    /* which build this is: a stale benchmark measures the code of another day (docs/LEZIONI.md #24) */
    printf("bench_mem %s, built %s %s, kernels %s, median of %d\n%s\n", group, __DATE__, __TIME__,
           tr_kernels_get()->tier, n_runs, cpu_line);
    if (tr_mem_guard((uint64_t)2 * GIB + 64 * MIB, err, sizeof err) != 0) {
        fprintf(stderr, "bench_mem: %s\n", err);
        return 1;
    }

    double t0 = tr_time_sec();
    int rc = strcmp(group, "ram") == 0 ? group_ram() : strcmp(group, "weights") == 0 ? group_weights() : group_kv(n_pos);
    printf("elapsed %.1f s\n", tr_time_sec() - t0);
    return rc;
}
