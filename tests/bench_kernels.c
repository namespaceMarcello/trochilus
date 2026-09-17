/* bench_kernels.c — microbenchmark of the CPU kernels, one line per (tier, kernel, size).
 *
 * Method, so that a difference can be told apart from noise:
 *   - every measurement is `runs` timed repetitions of ~`run_ms` each, after a warm-up;
 *   - the reported value is the median; `spread` is (max - min) / median of those runs,
 *     the noise floor of that line on this machine at this moment;
 *   - an optimization counts only if it moves the median by more than the spread.
 * The whole benchmark stays well under 60 seconds (docs/ARCHITETTURA.md, safety).
 *
 *   bench_kernels [--runs N] [--ms M] [--threads T]
 *
 * Output columns: tier, kernel, n (elements per row), median ns per call,
 * median million elements per second, spread %. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/platform.h"
#include "../src/base/cpu.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"

#define MAX_RUNS 31

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static float frand(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 40) & 0xFFFFFF) / (float)0x1000000 * 2.0f - 1.0f;
}

/* Fills one row of n elements of `type` with plausible random content. */
static void fill_row(tr_type type, uint8_t *row, int64_t n) {
    switch (type) {
    case TR_TYPE_F32:
        for (int64_t i = 0; i < n; i++) { float v = frand(); memcpy(row + 4 * i, &v, 4); }
        break;
    case TR_TYPE_F16:
        for (int64_t i = 0; i < n; i++) {
            uint16_t h = (uint16_t)((rng_state >> 17) & 0x7BFF);   /* finite half values */
            frand();
            memcpy(row + 2 * i, &h, 2);
        }
        break;
    case TR_TYPE_Q8_0:
        for (int64_t b = 0; b < n / 32; b++) {
            uint8_t *blk = row + 34 * b;
            uint16_t d = 0x3000 | (uint16_t)((rng_state >> 30) & 0x3FF);  /* small positive scale */
            frand();
            memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)(int8_t)(frand() * 127.0f);
        }
        break;
    default:
        break;
    }
}

typedef struct {
    const tr_kernels *k;
    tr_type type;
    const uint8_t *row;
    const float *a, *x;
    int64_t n;
    float sink;
} dot_job;

static void run_dot(dot_job *j, long calls) {
    float s = 0;
    if (j->type == TR_TYPE_COUNT) {
        for (long c = 0; c < calls; c++) s += j->k->dot_f32(j->a, j->x, j->n);
    } else {
        for (long c = 0; c < calls; c++) s += j->k->dot_row[j->type](j->row, j->x, j->n);
    }
    j->sink = s;
}

/* Times `calls` calls repeatedly; returns median ns per call and the spread. */
static void measure(dot_job *j, int runs, double run_ms, double *ns_median, double *spread) {
    long calls = 1;
    double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < 0.02) { run_dot(j, calls); calls *= 2; }   /* warm-up + calibration */
    double per_call = (tr_time_sec() - t0) / (double)(calls - 1);    /* 1 + 2 + ... + calls/2 */
    calls = (long)(run_ms / 1000.0 / (per_call > 0 ? per_call : 1e-9));
    if (calls < 1) calls = 1;

    double ns[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        run_dot(j, calls);
        ns[r] = (tr_time_sec() - a) * 1e9 / (double)calls;
    }
    qsort(ns, (size_t)runs, sizeof ns[0], cmp_double);
    *ns_median = ns[runs / 2];
    *spread = (ns[runs - 1] - ns[0]) / ns[runs / 2];
}

/* Same row against TR_DOT_TOKENS input rows (the block path of tr_matmul): ns per call
 * covers TR_DOT_TOKENS * n elements. */
static void measure_x4(const tr_kernels *k, const uint8_t *row, const float *x, int64_t n, int runs, double run_ms,
                        double *ns_median, double *spread) {
    float out[TR_DOT_TOKENS];
    long calls = 1;
    double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < 0.02) {
        for (long c = 0; c < calls; c++) k->dot_row_x4[TR_TYPE_Q8_0](row, x, n, n, out);
        calls *= 2;
    }
    double per_call = (tr_time_sec() - t0) / (double)(calls - 1);
    calls = (long)(run_ms / 1000.0 / (per_call > 0 ? per_call : 1e-9));
    if (calls < 1) calls = 1;

    double ns[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        for (long c = 0; c < calls; c++) k->dot_row_x4[TR_TYPE_Q8_0](row, x, n, n, out);
        ns[r] = (tr_time_sec() - a) * 1e9 / (double)calls;
    }
    qsort(ns, (size_t)runs, sizeof ns[0], cmp_double);
    *ns_median = ns[runs / 2];
    *spread = (ns[runs - 1] - ns[0]) / ns[runs / 2];
}

typedef struct {
    tr_mat w;
    const float *x;
    float *y;
} mat_job;

/* One run = `calls` matmuls of `n_tokens` input rows back to back: n_tokens 1 is the rhythm
 * of a decode step, a larger one the block path of a prefill pass. */
static void measure_matmul(tr_pool *pool, mat_job *m, int runs, int calls, int64_t n_tokens, double *ms_median,
                            double *spread) {
    tr_matmul(pool, &m->w, m->x, n_tokens, m->y);          /* warm-up */
    double ms[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        for (int c = 0; c < calls; c++) tr_matmul(pool, &m->w, m->x, n_tokens, m->y);
        ms[r] = (tr_time_sec() - a) * 1e3 / calls;
    }
    qsort(ms, (size_t)runs, sizeof ms[0], cmp_double);
    *ms_median = ms[runs / 2];
    *spread = (ms[runs - 1] - ms[0]) / ms[runs / 2];
}

static void empty_body(void *ctx, int64_t begin, int64_t end, int worker) {
    (void)ctx; (void)begin; (void)end; (void)worker;
}

/* Cost of one tr_parallel_for that hands one empty chunk to every thread: the price of
 * waking the pool, paid on every parallel matmul. */
static void measure_dispatch(tr_pool *pool, int runs, int calls, double *us_median, double *spread) {
    int64_t n = tr_pool_size(pool);
    tr_parallel_for(pool, n, 1, empty_body, NULL);
    double us[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        for (int c = 0; c < calls; c++) tr_parallel_for(pool, n, 1, empty_body, NULL);
        us[r] = (tr_time_sec() - a) * 1e6 / calls;
    }
    qsort(us, (size_t)runs, sizeof us[0], cmp_double);
    *us_median = us[runs / 2];
    *spread = (us[runs - 1] - us[0]) / us[runs / 2];
}

int main(int argc, char **argv) {
    int runs = 7, threads = 0;
    double run_ms = 150;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ms") == 0 && i + 1 < argc) run_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else { fprintf(stderr, "usage: bench_kernels [--runs N] [--ms M] [--threads T]\n"); return 2; }
    }
    if (runs < 3) runs = 3;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
    if (runs % 2 == 0) runs++;

    char line[512];
    tr_cpu_describe(tr_cpu(), line, sizeof line);
    printf("cpu: %s\n", line);
    tr_kernels_init();
    printf("active tier: %s; %d runs x %.0f ms, median and spread\n\n", tr_kernels_get()->tier, runs, run_ms);

    static const char *tiers[] = {"scalar", "avx2", "avx512", "avx512vnni", "neon", "asm"};
    static const int64_t sizes[] = {64, 1024, 2048, 4096};
    static const struct { const char *name; tr_type type; } kinds[] = {
        {"dot_f32", TR_TYPE_COUNT}, {"dot_row f16", TR_TYPE_F16}, {"dot_row q8_0", TR_TYPE_Q8_0},
    };

    int64_t max_n = 4096;
    float *a = tr_alloc_aligned((size_t)max_n * sizeof(float), 64);
    float *x = tr_alloc_aligned((size_t)max_n * sizeof(float), 64);
    uint8_t *row = tr_alloc_aligned((size_t)max_n * 4, 64);
    if (!a || !x || !row) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int64_t i = 0; i < max_n; i++) { a[i] = frand(); x[i] = frand(); }

    printf("%-12s %-14s %6s %12s %12s %8s\n", "tier", "kernel", "n", "ns/call", "M elem/s", "spread");
    for (size_t t = 0; t < sizeof tiers / sizeof tiers[0]; t++) {
        const tr_kernels *k = tr_kernels_tier(tiers[t]);
        if (k == NULL) continue;
        for (size_t kind = 0; kind < sizeof kinds / sizeof kinds[0]; kind++) {
            tr_type type = kinds[kind].type;
            if (type != TR_TYPE_COUNT && k->dot_row[type] == NULL) continue;
            for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
                int64_t n = sizes[s];
                if (type != TR_TYPE_COUNT) fill_row(type, row, n);
                dot_job j = {k, type, row, a, x, n, 0};
                double ns, spread;
                measure(&j, runs, run_ms, &ns, &spread);
                printf("%-12s %-14s %6lld %12.1f %12.1f %7.1f%%\n", k->tier, kinds[kind].name,
                       (long long)n, ns, (double)n / ns * 1e3, spread * 100.0);
            }
        }
    }

    /* The same row against TR_DOT_TOKENS input rows: elements per second counted per input
     * row, so it compares directly with the dot_row q8_0 line above. */
    float *x4 = tr_alloc_aligned((size_t)max_n * TR_DOT_TOKENS * sizeof(float), 64);
    if (!x4) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int64_t i = 0; i < max_n * TR_DOT_TOKENS; i++) x4[i] = frand();
    for (size_t t = 0; t < sizeof tiers / sizeof tiers[0]; t++) {
        const tr_kernels *k = tr_kernels_tier(tiers[t]);
        if (k == NULL || k->dot_row_x4[TR_TYPE_Q8_0] == NULL) continue;
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            int64_t n = sizes[s];
            fill_row(TR_TYPE_Q8_0, row, n);
            double ns, spread;
            measure_x4(k, row, x4, n, runs, run_ms, &ns, &spread);
            printf("%-12s %-14s %6lld %12.1f %12.1f %7.1f%%\n", k->tier, "dot_row q8_0 x4", (long long)n, ns,
                   (double)n * TR_DOT_TOKENS / ns * 1e3, spread * 100.0);
        }
    }
    tr_free_aligned(x4);

    /* One whole matrix through tr_matmul with the active tier: 1024 x 2048 q8_0,
     * the shape of an OLMoE expert projection. */
    int64_t rows = 1024, cols = 2048;
    size_t rb = tr_row_bytes(TR_TYPE_Q8_0, cols);
    uint8_t *w = tr_alloc_aligned(rb * (size_t)rows, 64);
    float *y = tr_alloc_aligned((size_t)rows * sizeof(float), 64);
    if (!w || !y) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int64_t r = 0; r < rows; r++) fill_row(TR_TYPE_Q8_0, w + rb * (size_t)r, cols);
    int64_t bench_tokens = 64;
    float *xb = tr_alloc_aligned((size_t)cols * (size_t)bench_tokens * sizeof(float), 64);
    float *yb = tr_alloc_aligned((size_t)rows * (size_t)bench_tokens * sizeof(float), 64);
    if (!xb || !yb) { fprintf(stderr, "out of memory (block matmul)"); return 1; }
    for (int64_t i = 0; i < cols * bench_tokens; i++) xb[i] = frand();
    mat_job m = {{TR_TYPE_Q8_0, rows, cols, w}, x, y};
    mat_job mb = {{TR_TYPE_Q8_0, rows, cols, w}, xb, yb};

    int n_phys = tr_cpu()->physical_cores;
    int counts[8] = {1, 2, 4, 8, 16, 32, 0, 0}, n_counts = 0;
    for (int c = 0; c < 6; c++)
        if (counts[c] <= n_phys) counts[n_counts++] = counts[c];
    if (counts[n_counts - 1] != n_phys) counts[n_counts++] = n_phys;
    if (threads > 0) counts[n_counts++] = threads;
    printf("\n%-8s %13s %7s %18s %7s %25s %7s\n", "threads", "dispatch us", "spread", "1024x2048 1 token",
           "spread", "1024x2048 64 tokens/tok", "spread");
    for (int c = 0; c < n_counts; c++) {
        tr_pool *pool = tr_pool_create(counts[c]);
        double us, ms, msb, s1, s2, s3;
        measure_dispatch(pool, runs, 2000, &us, &s1);
        measure_matmul(pool, &m, runs, 50, 1, &ms, &s2);
        measure_matmul(pool, &mb, runs, 5, bench_tokens, &msb, &s3);
        printf("%-8d %13.3f %6.1f%% %15.3f ms %6.1f%% %22.3f ms %6.1f%%\n", counts[c], us, s1 * 100.0, ms,
               s2 * 100.0, msb / (double)bench_tokens, s3 * 100.0);
        tr_pool_destroy(pool);
    }

    tr_free_aligned(a); tr_free_aligned(x); tr_free_aligned(row); tr_free_aligned(w); tr_free_aligned(y);
    tr_free_aligned(xb); tr_free_aligned(yb);
    return 0;
}
