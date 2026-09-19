/* bench_expf.c — the expf of this platform's C library, and the trial of an expf of our own
 * (docs/MISURE.md "Prefill su prompt lunghi" and question 37).
 *
 * The softmax of the attention and of the router, and the SiLU of the experts, call expf once
 * per element. It is the one kernel of the hot zone the engine does not own: its speed and its
 * bits are the C library's, and the two C libraries the engine is built with are not the same
 * code (MinGW-w64: x87 instructions; glibc: a table and a polynomial in double). What this
 * prints, for the decision on an expf of our own and later as its proof:
 *
 *   cost     ns per call over the arguments a softmax produces (x <= 0), one thread: the
 *            library's expf, and the candidate's
 *   bits     over EVERY float (2^32 bit patterns), on the pool:
 *              library    how many expf(x) are not (float)exp((double)x), the double-precision
 *                         exp of the same library rounded to float. That reference is the
 *                         correctly rounded value except where exp(x) falls within about 2^-29
 *                         of the middle between two floats, a handful of arguments in 4 billion
 *                         (tools/expf_hard_cases.py checks exactly those with 200 bits);
 *              candidate  how many of its results are not the library's, and how many are not
 *                         the reference: ZERO against the library is what "the same bits as
 *                         today" means on that platform, and it is a proof, not a sample.
 *
 * The candidate is TR_EXPF_CANDIDATE, a float (*)(float) given at build time
 * (make bench-expf EXTRA_CFLAGS=-DTR_EXPF_CANDIDATE=tr_expf, once the engine has one). Without
 * it the candidate is the trial below: NOT engine code, a sketch that measures what such a
 * function costs and whether its fast path can be trusted. 64 table entries of 2^(j/64), a
 * polynomial of degree 5 in double, a rounding test, and for the arguments that fail the test
 * (the slow path, counted) the library's double exp: the real one needs a slow path of its own.
 *
 *   bench_expf [--threads T] [--hard <file>]     --hard: the arguments near a rounding border
 *
 * One run, well under 60 s on 16 threads (docs/ARCHITETTURA.md, safety of the machine). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"

#define MAX_WORKERS 64
#define BLOCK_BITS 16   /* 65536 blocks of 65536 bit patterns */
#define HARD_PER_WORKER 4096

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static uint32_t to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

/* ---- the trial: a sketch of a correctly rounded expf, for this benchmark only --------------- */

static double trial_table[64];                /* 2^(j/64) */
static _Thread_local uint64_t trial_slow;     /* arguments that took the slow path, per thread */

static void trial_init(void) {
    for (int j = 0; j < 64; j++) trial_table[j] = exp2((double)j / 64.0);
}

static float trial_expf(float x) {
    if (x != x) return x;
    if (x > 88.8f) return INFINITY;
    if (x < -104.0f) return 0.0f;             /* exp(-104) < 2^-150: rounds to zero */
    /* ln2/64 in two pieces, the first with 21 trailing zero bits: k * hi is exact (fdlibm's split) */
    const double inv = 0x1.71547652b82fep+6, hi = 0x1.62e42feep-7, lo = 0x1.a39ef35793c76p-39;
    double xd = (double)x, z = xd * inv;
    double kd = (z + 0x1.8p52) - 0x1.8p52;    /* round to nearest, no library call */
    int64_t k = (int64_t)kd;
    double r = (xd - kd * hi) - kd * lo;      /* |r| <= ln2/128 */
    double p = r + r * r * (0.5 + r * (0x1.5555555555555p-3 + r * (0x1.5555555555555p-5 + r * 0x1.1111111111111p-7)));
    double t = trial_table[k & 63];
    uint64_t scale_bits = (uint64_t)((k >> 6) + 1023) << 52;   /* 2^(k div 64), exact */
    double scale;
    memcpy(&scale, &scale_bits, sizeof scale);
    double y = (t + t * p) * scale;
    /* the fast path is trusted where an error of 2^-50 cannot move the rounding */
    float a = (float)(y * (1.0 - 0x1p-50)), b = (float)(y * (1.0 + 0x1p-50));
    if (to_bits(a) == to_bits(b)) return a;
    trial_slow++;
    return (float)exp(xd);
}

#ifdef TR_EXPF_CANDIDATE
float TR_EXPF_CANDIDATE(float x);
#define CANDIDATE TR_EXPF_CANDIDATE
#define CANDIDATE_NAME "the engine's"
#else
#define CANDIDATE trial_expf
#define CANDIDATE_NAME "the trial in this file"
#endif

/* ---- every float ---------------------------------------------------------------------------- */

typedef struct {
    uint64_t finite, finite_softmax;
    uint64_t lib_ref, lib_ref_softmax;        /* library's expf is not the reference */
    uint64_t cand_lib, cand_ref, slow;        /* the candidate is not the library's / the reference */
    double worst_ulp;
    float worst_x;
} tally;
static tally tallies[MAX_WORKERS];
static float hard[MAX_WORKERS][HARD_PER_WORKER];
static int n_hard[MAX_WORKERS];

static double ulps(float got, float want) {
    float next = nextafterf(want, got);
    double ulp = fabs((double)next - (double)want);
    return ulp > 0.0 ? fabs((double)got - (double)want) / ulp : 0.0;
}

static void bits_body(void *ctx, int64_t begin, int64_t end, int worker) {
    (void)ctx;
    tally local; /* counted on the stack: two workers' tallies may share a cache line */
    memset(&local, 0, sizeof local);
    uint64_t slow0 = trial_slow;
    for (int64_t b = begin; b < end; b++) {
        for (uint32_t i = 0; i < (1u << BLOCK_BITS); i++) {
            float x = from_bits(((uint32_t)b << BLOCK_BITS) | i);
            if (x != x) continue; /* NaN in, NaN out: nothing to round */
            volatile float lib = expf(x);
            volatile double ref_d = exp((double)x);
            volatile float ref = (float)ref_d;
            volatile float cand = CANDIDATE(x);
            int softmax_range = x <= 0.0f && x >= -104.0f; /* below -104 every exp is 0 */
            local.finite++;
            local.finite_softmax += (uint64_t)softmax_range;
            local.cand_lib += to_bits(cand) != to_bits(lib);
            local.cand_ref += to_bits(cand) != to_bits(ref);
            /* near a border: a change of 2^-45 of the double moves the float. For the 200-bit check. */
            if (to_bits((float)(ref_d * (1.0 - 0x1p-45))) != to_bits((float)(ref_d * (1.0 + 0x1p-45))) &&
                n_hard[worker] < HARD_PER_WORKER)
                hard[worker][n_hard[worker]++] = x;
            if (to_bits(lib) == to_bits(ref)) continue;
            local.lib_ref++;
            local.lib_ref_softmax += (uint64_t)softmax_range;
            double d = ulps(lib, ref);
            if (d > local.worst_ulp) {
                local.worst_ulp = d;
                local.worst_x = x;
            }
        }
    }
    tally *mine = &tallies[worker];
    mine->finite += local.finite;
    mine->finite_softmax += local.finite_softmax;
    mine->lib_ref += local.lib_ref;
    mine->lib_ref_softmax += local.lib_ref_softmax;
    mine->cand_lib += local.cand_lib;
    mine->cand_ref += local.cand_ref;
    mine->slow += trial_slow - slow0;
    if (local.worst_ulp > mine->worst_ulp) {
        mine->worst_ulp = local.worst_ulp;
        mine->worst_x = local.worst_x;
    }
}

/* ns per call over the arguments of a softmax row, scores minus their maximum: -12 .. 0 */
static double cost_ns(float (*fn)(float), const float *xs, int n, float *sum) {
    enum { REPS = 64 };
    double best = 1e30;
    for (int r = 0; r < 7; r++) {
        double t0 = tr_time_sec();
        float acc = 0.0f;
        for (int k = 0; k < REPS; k++)
            for (int i = 0; i < n; i++) acc += fn(xs[i]);
        double dt = tr_time_sec() - t0;
        *sum += acc;
        if (dt < best) best = dt;
    }
    return best * 1e9 / ((double)n * REPS);
}

int main(int argc, char **argv) {
    int threads = 0;
    const char *hard_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hard") == 0 && i + 1 < argc) hard_path = argv[++i];
    }
    int cores = tr_cpu()->physical_cores;
    if (threads < 1 || threads > cores) threads = cores; /* never more threads than cores */
    if (threads > MAX_WORKERS) threads = MAX_WORKERS;
    trial_init();

    char cpu_line[512];
    tr_cpu_describe(tr_cpu(), cpu_line, sizeof cpu_line);
    printf("bench_expf, built %s %s, %d threads, candidate: %s\n%s\n", __DATE__, __TIME__, threads, CANDIDATE_NAME,
           cpu_line);

    enum { N = 1 << 16 };
    static float xs[N];
    float sum = 0.0f;
    for (int i = 0; i < N; i++) xs[i] = -12.0f * (float)((i * 2654435761u) >> 8 & 0xFFFF) / 65536.0f;
    double lib_ns = cost_ns(expf, xs, N, &sum), cand_ns = cost_ns(CANDIDATE, xs, N, &sum);
    printf("cost: library %.2f ns per call, candidate %.2f ns (one thread, best of 7, arguments in -12..0; sum %.1f)\n",
           lib_ns, cand_ns, (double)sum);

    tr_pool *pool = tr_pool_create(threads);
    if (pool == NULL) return 1;
    double t0 = tr_time_sec();
    tr_parallel_for(pool, (int64_t)1 << (32 - BLOCK_BITS), 16, bits_body, NULL);
    double secs = tr_time_sec() - t0;
    tally sum_t;
    memset(&sum_t, 0, sizeof sum_t);
    for (int w = 0; w < threads; w++) {
        sum_t.finite += tallies[w].finite;
        sum_t.finite_softmax += tallies[w].finite_softmax;
        sum_t.lib_ref += tallies[w].lib_ref;
        sum_t.lib_ref_softmax += tallies[w].lib_ref_softmax;
        sum_t.cand_lib += tallies[w].cand_lib;
        sum_t.cand_ref += tallies[w].cand_ref;
        sum_t.slow += tallies[w].slow;
        if (tallies[w].worst_ulp > sum_t.worst_ulp) {
            sum_t.worst_ulp = tallies[w].worst_ulp;
            sum_t.worst_x = tallies[w].worst_x;
        }
    }
    printf("bits, library: %llu of %llu non-NaN floats give an expf that is not (float)exp((double)x): %.6f%%\n",
           (unsigned long long)sum_t.lib_ref, (unsigned long long)sum_t.finite,
           (double)sum_t.lib_ref / (double)sum_t.finite * 100.0);
    printf("bits, library: in the range of a softmax (-104 <= x <= 0): %llu of %llu: %.6f%%\n",
           (unsigned long long)sum_t.lib_ref_softmax, (unsigned long long)sum_t.finite_softmax,
           (double)sum_t.lib_ref_softmax / (double)sum_t.finite_softmax * 100.0);
    printf("bits, library: largest difference %.3f units of the last place, at x = %.9g\n", sum_t.worst_ulp,
           (double)sum_t.worst_x);
    printf("bits, candidate: %llu results are not the library's expf, %llu are not the reference\n",
           (unsigned long long)sum_t.cand_lib, (unsigned long long)sum_t.cand_ref);
#ifndef TR_EXPF_CANDIDATE
    printf("bits, candidate: the trial took its slow path for %llu arguments (%.6f%%)\n",
           (unsigned long long)sum_t.slow, (double)sum_t.slow / (double)sum_t.finite * 100.0);
#endif
    int total_hard = 0;
    for (int w = 0; w < threads; w++) total_hard += n_hard[w];
    printf("hard: %d arguments whose double exp is within 2^-45 of a rounding border%s\n", total_hard,
           hard_path != NULL ? "" : " (--hard <file> writes them for tools/expf_hard_cases.py)");
    if (hard_path != NULL) {
        FILE *f = fopen(hard_path, "w");
        if (f == NULL) return 1;
        for (int w = 0; w < threads; w++)
            for (int i = 0; i < n_hard[w]; i++)
                fprintf(f, "%08x %08x\n", (unsigned)to_bits(hard[w][i]), (unsigned)to_bits((float)exp((double)hard[w][i])));
        fclose(f);
    }
    printf("elapsed %.1f s\n", secs);
    tr_pool_destroy(pool);
    return 0;
}
