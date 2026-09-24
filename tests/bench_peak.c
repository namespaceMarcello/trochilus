/* bench_peak.c — how many FLOP/s this CPU does at best without FMA, and how close the prefill's
 * matmul comes to it (docs/MEASUREMENTS.md question 55: the premise of an assembly microkernel).
 *
 * Per thread count (1 and every physical core), the median of N runs and its spread:
 *   peak       independent multiplies and adds on 12 registers, nothing loaded, as zmm (AVX-512)
 *              and as ymm (AVX2): the no-FMA peak at the clock the cores hold under this load;
 *   row2 L1    the active tier's dot_row2_x4 on Q8_0 (the prefill's inner kernel: two rows'
 *              codes widened, converted, scaled, then 8 multiplies and 8 adds into 8
 *              accumulators) over rows and inputs that stay in L1 (512 columns) or in L2 (2048),
 *              and its dot_row2_x8 (the same against 8 input rows, 16 accumulators);
 *   matmul     tr_matmul on OLMoE's shapes: an expert's gate/up (1024 x 2048) and down
 *              (2048 x 1024) with the 64 tokens an expert sees in a 512-token pass, and an
 *              attention projection (2048 x 2048) with 512 tokens; where the tier has
 *              dot_row2_x8, also without it ("-x8": a copy of the table made active), the two
 *              run by run in turn.
 * FLOPs are the matmul's useful ones, 2 per weight and token (the scale multiply not counted).
 *
 *   build/tests/bench_peak.exe [--runs N]      native, still machine (tools/measure_guard.lib)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAVE_X86 1
#endif

#define MAX_RUNS 31

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static volatile float g_one = 1.0f, g_zero = 0.0f;  /* not foldable: the loops really run */
static volatile float g_sink;

#if HAVE_X86
/* The peak loop is inline assembly: twelve named registers the compiler can neither merge nor
 * drop. Written in C first, gcc merged the twelve chains into two (they started from the same
 * values) and the "peak" came out 2x too high (docs/LESSONS.md #161). Six multiply chains and six
 * add chains, each op waiting only on its own chain: 3-cycle latency, 6 chains per pipe pair. The
 * multiplies are grouped, then the adds: on Zen 4 the same twelve ops strictly alternating
 * (mul, add, mul, add...) run at 83% of this (139 against 167 GFLOP/s on one core, LESSONS #162). */
#ifndef BENCH_PEAK_MUTATE
#define PEAK_ASM \
    "vmulps %[c], %[m0], %[m0]\n\tvmulps %[c], %[m1], %[m1]\n\tvmulps %[c], %[m2], %[m2]\n\t" \
    "vmulps %[c], %[m3], %[m3]\n\tvmulps %[c], %[m4], %[m4]\n\tvmulps %[c], %[m5], %[m5]\n\t" \
    "vaddps %[e], %[a0], %[a0]\n\tvaddps %[e], %[a1], %[a1]\n\tvaddps %[e], %[a2], %[a2]\n\t" \
    "vaddps %[e], %[a3], %[a3]\n\tvaddps %[e], %[a4], %[a4]\n\tvaddps %[e], %[a5], %[a5]\n\t"
#else
/* the mutation (-DBENCH_PEAK_MUTATE): the chains made dependent, one of each kind, so the loop
 * waits on latency; the check at the end of main must fail on it (a peak below a real kernel) */
#define PEAK_ASM \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t"
#endif
#define PEAK_BODY(V, SET1, iters) \
    do { \
        V c = SET1(g_one), e = SET1(g_zero); \
        V m0 = c, m1 = c, m2 = c, m3 = c, m4 = c, m5 = c, a0 = e, a1 = e, a2 = e, a3 = e, a4 = e, a5 = e; \
        for (int64_t i_ = 0; i_ < (iters); i_++) \
            __asm__ volatile(PEAK_ASM \
                             : [m0] "+v"(m0), [m1] "+v"(m1), [m2] "+v"(m2), [m3] "+v"(m3), [m4] "+v"(m4), \
                               [m5] "+v"(m5), [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), \
                               [a4] "+v"(a4), [a5] "+v"(a5) \
                             : [c] "v"(c), [e] "v"(e)); \
        float f_[16]; /* the asm is volatile: nothing to keep alive, one value to show */ \
        V s_ = m0; \
        memcpy(f_, &s_, sizeof s_); \
        g_sink = f_[0]; \
        (void)m1; (void)m2; (void)m3; (void)m4; (void)m5; \
        (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5; \
    } while (0)

__attribute__((target("avx512f"))) static void peak512(int64_t iters) {
    PEAK_BODY(__m512, _mm512_set1_ps, iters);
}
__attribute__((target("avx2"))) static void peak256(int64_t iters) {
    PEAK_BODY(__m256, _mm256_set1_ps, iters);
}

/* The prefill's inner step as an instruction stream, over one block that stays in L1: two rows'
 * 16 codes widened, converted and scaled, then per token a multiply and an add for each row into
 * its own accumulator; tokens two at a time, their four multiplies before their four adds. Not
 * the kernel's bits (the same block every iteration): the ceiling of its arithmetic, with 4
 * tokens as dot_row2_x4 is today and with 8 as an assembly microkernel could hold (16
 * accumulators, 26 of the 32 zmm). */
#define MIX_CODES \
    "vpmovsxbd (%[q]), %[w]\n\tvpmovsxbd 16(%[q]), %[u]\n\tvcvtdq2ps %[w], %[w]\n\t" \
    "vcvtdq2ps %[u], %[u]\n\tvmulps %[d0], %[w], %[w]\n\tvmulps %[d1], %[u], %[u]\n\t"
#define MIX_PAIR(off0, off1, a, b, c, d) \
    "vmovups " off0 "(%[x]), %[v0]\n\tvmovups " off1 "(%[x]), %[v1]\n\t" \
    "vmulps %[v0], %[w], %[t0]\n\tvmulps %[v0], %[u], %[t1]\n\t" \
    "vmulps %[v1], %[w], %[t2]\n\tvmulps %[v1], %[u], %[t3]\n\t" \
    "vaddps %[t0], %[" a "], %[" a "]\n\tvaddps %[t1], %[" b "], %[" b "]\n\t" \
    "vaddps %[t2], %[" c "], %[" c "]\n\tvaddps %[t3], %[" d "], %[" d "]\n\t"
#define MIX_TEMPS [w] "=&v"(w), [u] "=&v"(u), [v0] "=&v"(v0), [v1] "=&v"(v1), [t0] "=&v"(t0), \
                  [t1] "=&v"(t1), [t2] "=&v"(t2), [t3] "=&v"(t3)
#define MIX_IN [q] "r"(q), [x] "r"(x), [d0] "v"(d0), [d1] "v"(d1)

__attribute__((target("avx512f"))) static void mix_x4(int64_t iters, const uint8_t *q, const float *x) {
    __m512 d0 = _mm512_set1_ps(g_one), d1 = _mm512_set1_ps(g_one), z = _mm512_set1_ps(g_zero);
    __m512 a0 = z, a1 = z, a2 = z, a3 = z, b0 = z, b1 = z, b2 = z, b3 = z, w, u, v0, v1, t0, t1, t2, t3;
    for (int64_t i = 0; i < iters; i++)
        __asm__ volatile(MIX_CODES MIX_PAIR("0", "64", "a0", "b0", "a1", "b1") MIX_PAIR("128", "192", "a2", "b2", "a3", "b3")
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [b0] "+v"(b0), [b1] "+v"(b1),
                           [b2] "+v"(b2), [b3] "+v"(b3), MIX_TEMPS
                         : MIX_IN
                         : "memory");
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}

/* For question 59 (FMA in the definition?): the peak and the 8-token stream with fused
 * multiply-adds, one rounding instead of two; not the engine's bits. */
#define FMA_ASM \
    "vfmadd231ps %[c], %[m0], %[a0]\n\tvfmadd231ps %[c], %[m1], %[a1]\n\tvfmadd231ps %[c], %[m2], %[a2]\n\t" \
    "vfmadd231ps %[c], %[m3], %[a3]\n\tvfmadd231ps %[c], %[m4], %[a4]\n\tvfmadd231ps %[c], %[m5], %[a5]\n\t"
__attribute__((target("avx512f"))) static void peak_fma(int64_t iters) {
    __m512 c = _mm512_set1_ps(g_one), e = _mm512_set1_ps(g_zero);
    __m512 m0 = c, m1 = c, m2 = c, m3 = c, m4 = c, m5 = c, a0 = e, a1 = e, a2 = e, a3 = e, a4 = e, a5 = e;
    for (int64_t i = 0; i < iters; i++)
        __asm__ volatile(FMA_ASM FMA_ASM
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [a4] "+v"(a4), [a5] "+v"(a5)
                         : [c] "v"(c), [m0] "v"(m0), [m1] "v"(m1), [m2] "v"(m2), [m3] "v"(m3), [m4] "v"(m4),
                           [m5] "v"(m5));
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}
#define FMA_PAIR(off0, off1, a, b, c, d) \
    "vmovups " off0 "(%[x]), %[v0]\n\tvmovups " off1 "(%[x]), %[v1]\n\t" \
    "vfmadd231ps %[v0], %[w], %[" a "]\n\tvfmadd231ps %[v0], %[u], %[" b "]\n\t" \
    "vfmadd231ps %[v1], %[w], %[" c "]\n\tvfmadd231ps %[v1], %[u], %[" d "]\n\t"
__attribute__((target("avx512f"))) static void mix_x8_fma(int64_t iters, const uint8_t *q, const float *x) {
    __m512 d0 = _mm512_set1_ps(g_one), d1 = _mm512_set1_ps(g_one), z = _mm512_set1_ps(g_zero);
    __m512 a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z;
    __m512 b0 = z, b1 = z, b2 = z, b3 = z, b4 = z, b5 = z, b6 = z, b7 = z, w, u, v0, v1, t0, t1, t2, t3;
    for (int64_t i = 0; i < iters; i++) {
        __asm__ volatile(MIX_CODES FMA_PAIR("0", "64", "a0", "b0", "a1", "b1") FMA_PAIR("128", "192", "a2", "b2", "a3", "b3")
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [b0] "+v"(b0), [b1] "+v"(b1),
                           [b2] "+v"(b2), [b3] "+v"(b3), MIX_TEMPS
                         : MIX_IN
                         : "memory");
        __asm__ volatile(FMA_PAIR("256", "320", "a4", "b4", "a5", "b5") FMA_PAIR("384", "448", "a6", "b6", "a7", "b7")
                         : [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7), [b4] "+v"(b4), [b5] "+v"(b5),
                           [b6] "+v"(b6), [b7] "+v"(b7), [v0] "=&v"(v0), [v1] "=&v"(v1)
                         : [x] "r"(x), [w] "v"(w), [u] "v"(u)
                         : "memory");
    }
    (void)t0; (void)t1; (void)t2; (void)t3;
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}

__attribute__((target("avx512f"))) static void mix_x8(int64_t iters, const uint8_t *q, const float *x) {
    __m512 d0 = _mm512_set1_ps(g_one), d1 = _mm512_set1_ps(g_one), z = _mm512_set1_ps(g_zero);
    __m512 a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z;
    __m512 b0 = z, b1 = z, b2 = z, b3 = z, b4 = z, b5 = z, b6 = z, b7 = z, w, u, v0, v1, t0, t1, t2, t3;
    /* two statements an iteration: an asm takes at most 30 operands, and "+v" counts twice */
    for (int64_t i = 0; i < iters; i++) {
        __asm__ volatile(MIX_CODES MIX_PAIR("0", "64", "a0", "b0", "a1", "b1") MIX_PAIR("128", "192", "a2", "b2", "a3", "b3")
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [b0] "+v"(b0), [b1] "+v"(b1),
                           [b2] "+v"(b2), [b3] "+v"(b3), MIX_TEMPS
                         : MIX_IN
                         : "memory");
        __asm__ volatile(MIX_PAIR("256", "320", "a4", "b4", "a5", "b5") MIX_PAIR("384", "448", "a6", "b6", "a7", "b7")
                         : [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7), [b4] "+v"(b4), [b5] "+v"(b5),
                           [b6] "+v"(b6), [b7] "+v"(b7), [v0] "=&v"(v0), [v1] "=&v"(v1), [t0] "=&v"(t0),
                           [t1] "=&v"(t1), [t2] "=&v"(t2), [t3] "=&v"(t3)
                         : [x] "r"(x), [w] "v"(w), [u] "v"(u)
                         : "memory");
    }
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}
#endif

typedef struct {
    int kind;          /* 0: peak512, 1: peak256, 2: row2, 3: mix_x4, 4: mix_x8, 5: peak_fma, 6: mix_x8_fma,
                        * 7: row2_x8 */
    int64_t iters;     /* loop iterations (peak) or calls (row2) per thread */
    int64_t cols;
    uint8_t *rows[64]; /* per worker: two Q8_0 rows */
    float *x[64];      /* per worker: 8 input rows */
} peak_job;

static void peak_fn(void *ctx, int64_t begin, int64_t end, int worker) {
    peak_job *j = (peak_job *)ctx;
    for (int64_t t = begin; t < end; t++) {
#if HAVE_X86
        if (j->kind == 0) peak512(j->iters);
        if (j->kind == 1) peak256(j->iters);
        if (j->kind == 3) mix_x4(j->iters, j->rows[worker], j->x[worker]);
        if (j->kind == 4) mix_x8(j->iters, j->rows[worker], j->x[worker]);
        if (j->kind == 5) peak_fma(j->iters);
        if (j->kind == 6) mix_x8_fma(j->iters, j->rows[worker], j->x[worker]);
#endif
        if (j->kind == 2) {
            const tr_kernels *k = tr_kernels_get();
            size_t rb = tr_row_bytes(TR_TYPE_Q8_0, j->cols);
            float out[8], acc = 0.0f;
            for (int64_t c = 0; c < j->iters; c++) {
                k->dot_row2_x4[TR_TYPE_Q8_0](j->rows[worker], j->rows[worker] + rb, j->x[worker], j->cols, j->cols,
                                             out);
                acc += out[0];
            }
            g_sink = acc;
        }
        if (j->kind == 7) {
            const tr_kernels *k = tr_kernels_get();
            size_t rb = tr_row_bytes(TR_TYPE_Q8_0, j->cols);
            float out[16], acc = 0.0f;
            for (int64_t c = 0; c < j->iters; c++) {
                k->dot_row2_x8[TR_TYPE_Q8_0](j->rows[worker], j->rows[worker] + rb, j->x[worker], j->cols, j->cols,
                                             out);
                acc += out[0];
            }
            g_sink = acc;
        }
    }
}

/* GFLOP/s of n_threads running the job, median of runs; flop is per thread */
static double run_job(tr_pool *pool, int n_threads, peak_job *j, double flop, int runs, double *spread) {
    double g[MAX_RUNS];
    tr_parallel_for(pool, n_threads, 1, peak_fn, j); /* warm-up */
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        tr_parallel_for(pool, n_threads, 1, peak_fn, j);
        g[r] = flop * n_threads / (tr_time_sec() - a) / 1e9;
    }
    qsort(g, (size_t)runs, sizeof g[0], cmp_double);
    *spread = (g[runs - 1] - g[0]) / g[runs / 2];
    return g[runs / 2];
}

static float frand_state(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((*s >> 40) & 0xFFFF) / 32768.0f - 1.0f;
}

static void fill_q8_0(uint8_t *row, int64_t n, uint64_t *s) {
    for (int64_t b = 0; b < n / 32; b++) {
        uint16_t d = (uint16_t)(0x2000 | (uint16_t)((*s >> 30) & 0x3FF));
        frand_state(s);
        memcpy(row + 34 * b, &d, 2);
        for (int i = 0; i < 32; i++) row[34 * b + 2 + i] = (uint8_t)(int8_t)(frand_state(s) * 127.0f);
    }
}

int main(int argc, char **argv) {
    int runs = 7, below_kernel = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else { fprintf(stderr, "usage: bench_peak [--runs N]\n"); return 2; }
    }
    if (runs < 3) runs = 3;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
    tr_kernels_init();
    const tr_cpu_info *cpu = tr_cpu();
    char desc[512];
    tr_cpu_describe(cpu, desc, (int)sizeof desc);
    printf("%s\nkernel tier: %s; runs %d, median and spread (max-min)/median\n", desc, tr_kernels_get()->tier,
           runs);
    int counts[2] = {1, cpu->physical_cores > 64 ? 64 : cpu->physical_cores};
    uint64_t seed = 12345;
    peak_job j;
    memset(&j, 0, sizeof j);
    for (int w = 0; w < 64; w++) {
        j.rows[w] = tr_alloc_aligned(2 * tr_row_bytes(TR_TYPE_Q8_0, 2048), 64);
        j.x[w] = tr_alloc_aligned(8 * 2048 * sizeof(float), 64);
        if (j.rows[w] == NULL || j.x[w] == NULL) { fprintf(stderr, "out of memory\n"); return 1; }
    }
    printf("\n%-8s %-26s %12s %7s\n", "threads", "what", "GFLOP/s", "spread");
    for (int c = 0; c < 2; c++) {
        int n = counts[c];
        tr_pool *pool = tr_pool_create(n);
        double sp, g, peak = 0.0;
#if HAVE_X86
        if (cpu->avx512f) {
            j.kind = 0; j.iters = 20000000;
            g = run_job(pool, n, &j, 20000000.0 * 12 * 16, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "peak zmm mul+add", g, sp * 100);
            peak = g > peak ? g : peak;
        }
        if (cpu->avx2) {
            j.kind = 1; j.iters = 20000000;
            g = run_job(pool, n, &j, 20000000.0 * 12 * 8, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "peak ymm mul+add", g, sp * 100);
            peak = g > peak ? g : peak;
        }
        if (cpu->avx512f) {
            for (int w = 0; w < n; w++) {
                memset(j.rows[w], 1, 64);
                for (int i = 0; i < 128; i++) j.x[w][i] = 0.5f;
            }
            j.kind = 3; j.iters = 10000000;
            g = run_job(pool, n, &j, 10000000.0 * 256, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "mix x4 (asm, L1)", g, sp * 100);
            below_kernel |= peak > 0.0 && g > peak * 1.02;
            j.kind = 4; j.iters = 5000000;
            g = run_job(pool, n, &j, 5000000.0 * 512, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "mix x8 (asm, L1)", g, sp * 100);
            below_kernel |= peak > 0.0 && g > peak * 1.02;
        }
        if (cpu->avx512f && cpu->fma) {
            j.kind = 5; j.iters = 10000000;
            g = run_job(pool, n, &j, 10000000.0 * 12 * 32, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "peak zmm fma (q. 59)", g, sp * 100);
            j.kind = 6; j.iters = 5000000;
            g = run_job(pool, n, &j, 5000000.0 * 512, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "mix x8 fma (q. 59)", g, sp * 100);
        }
#endif
        if (tr_kernels_get()->dot_row2_x4[TR_TYPE_Q8_0] != NULL) {
            static const int64_t colss[2] = {512, 2048};
            for (int s = 0; s < 2; s++) {
                j.kind = 2; j.cols = colss[s];
                for (int w = 0; w < n; w++) {
                    fill_q8_0(j.rows[w], 2 * j.cols, &seed);
                    for (int64_t i = 0; i < 8 * j.cols; i++) j.x[w][i] = frand_state(&seed);
                }
                j.iters = 40000000 / j.cols;
                g = run_job(pool, n, &j, (double)j.iters * 16.0 * (double)j.cols, runs, &sp);
                char what[64];
                snprintf(what, sizeof what, "row2_x4 q8_0, %lld cols", (long long)j.cols);
                printf("%-8d %-26s %12.1f %6.1f%%\n", n, what, g, sp * 100);
                below_kernel |= peak > 0.0 && g > peak * 1.02;
                if (tr_kernels_get()->dot_row2_x8[TR_TYPE_Q8_0] == NULL) continue;
                j.kind = 7; j.iters = 20000000 / j.cols;
                g = run_job(pool, n, &j, (double)j.iters * 32.0 * (double)j.cols, runs, &sp);
                snprintf(what, sizeof what, "row2_x8 q8_0, %lld cols", (long long)j.cols);
                printf("%-8d %-26s %12.1f %6.1f%%\n", n, what, g, sp * 100);
                below_kernel |= peak > 0.0 && g > peak * 1.02;
            }
        }
        static const int64_t shapes[3][3] = {{1024, 2048, 64}, {2048, 1024, 64}, {2048, 2048, 512}};
        for (int s = 0; s < 3; s++) {
            int64_t rows = shapes[s][0], cols = shapes[s][1], nt = shapes[s][2];
            size_t rb = tr_row_bytes(TR_TYPE_Q8_0, cols);
            uint8_t *w = tr_alloc_aligned(rb * (size_t)rows, 64);
            float *x = tr_alloc_aligned((size_t)(cols * nt) * sizeof(float), 64);
            float *y = tr_alloc_aligned((size_t)(rows * nt) * sizeof(float), 64);
            if (!w || !x || !y) { fprintf(stderr, "out of memory\n"); return 1; }
            fill_q8_0(w, rows * cols, &seed);
            for (int64_t i = 0; i < cols * nt; i++) x[i] = frand_state(&seed);
            tr_mat m = {TR_TYPE_Q8_0, rows, cols, w};
            int calls = (int)(2000000000LL / (rows * cols * nt) / (n == 1 ? 8 : 1)) + 1;
            /* the tier as it is, and (where it has one) without its eight-token kernel */
            static tr_kernels no_x8;
            no_x8 = *tr_kernels_get();
            no_x8.dot_row2_x8[TR_TYPE_Q8_0] = NULL;
            int ab = tr_kernels_get()->dot_row2_x8[TR_TYPE_Q8_0] != NULL ? 2 : 1;
            double g2[2][MAX_RUNS];
            tr_matmul(pool, &m, x, nt, y);
            for (int r = 0; r < runs; r++) {
                for (int v = 0; v < ab; v++) {
                    int which = ab == 1 ? 0 : (r + v) % 2; /* the order turns every run */
                    tr_kernels_set_active(which == 1 ? &no_x8 : NULL);
                    double a = tr_time_sec();
                    for (int k = 0; k < calls; k++) tr_matmul(pool, &m, x, nt, y);
                    g2[which][r] = 2.0 * (double)(rows * cols * nt) * calls / (tr_time_sec() - a) / 1e9;
                }
            }
            tr_kernels_set_active(NULL);
            for (int v = 0; v < ab; v++) {
                qsort(g2[v], (size_t)runs, sizeof g2[v][0], cmp_double);
                char what[64];
                snprintf(what, sizeof what, "matmul %lldx%lld, %lld tok%s", (long long)rows, (long long)cols,
                         (long long)nt, v == 1 ? " -x8" : "");
                printf("%-8d %-26s %12.1f %6.1f%%\n", n, what, g2[v][runs / 2],
                       (g2[v][runs - 1] - g2[v][0]) / g2[v][runs / 2] * 100);
                below_kernel |= peak > 0.0 && g2[v][runs / 2] > peak * 1.02;
            }
            g_sink = y[0];
            tr_free_aligned(w); tr_free_aligned(x); tr_free_aligned(y);
        }
        tr_pool_destroy(pool);
    }
    for (int w = 0; w < 64; w++) { tr_free_aligned(j.rows[w]); tr_free_aligned(j.x[w]); }
    /* a peak below a real kernel is not a peak: the compiler merged or dropped its chains (it
     * did once, docs/LESSONS.md #161) */
    if (below_kernel) {
        printf("bench_peak: a kernel ran faster than the peak loop: the peak loop is not what it claims\n");
        return 1;
    }
    return 0;
}
