/* bench_expf.c — tr_expf against the C library's expf and against the correctly rounded value,
 * on EVERY float: the proof of src/kernels/expf.c (docs/MEASUREMENTS.md "Prefill su prompt lunghi" and
 * question 37).
 *
 * The softmax of the attention and of the router, and the SiLU of the experts, take one
 * exponential per element. It used to be the C library's expf, whose speed and bits are the
 * library's, and the two C libraries the engine is built with are not the same code (MinGW-w64:
 * x87 instructions; glibc: a table and a polynomial in double). What this prints:
 *
 *   cost     ns per call over the arguments a softmax produces (x <= 0), one thread: the
 *            library's expf, and tr_expf
 *   bits     over EVERY float (2^32 bit patterns), on the pool:
 *              library    how many expf(x) are not (float)exp((double)x), the double-precision
 *                         exp of the same library rounded to float. That reference is the
 *                         correctly rounded value except where exp(x) falls within about 2^-29
 *                         of the middle between two floats, a handful of arguments in 4 billion
 *                         (tools/expf_hard_cases.py checks exactly those with 200 bits);
 *              tr_expf    how many of its results are not the library's, and how many are not
 *                         the reference: ZERO against the reference is "correctly rounded", and
 *                         zero against the library is what "the same bits as before" means on
 *                         that platform. It is a proof, not a sample.
 *
 *   bench_expf [--threads T] [--check] [--like-library] [--hard <file>] [--slow <file>]
 *
 *   --check         exit 1 unless every float gives the reference, none leaves tr_expf unproven
 *                   (TR_EXPF_UNPROVEN, a NaN) and every entry of its table of exceptions is an
 *                   argument that really takes the table: the gate of `make check`
 *   --like-library  with --check: also zero differences from the library's expf (MinGW-w64,
 *                   whose expf is correctly rounded everywhere; glibc's is not)
 *   --hard <file>   the arguments whose double exp is near a rounding border, for
 *                   tools/expf_hard_cases.py
 *   --slow <file>   the arguments tr_expf cannot settle with its fast path, for
 *                   tools/gen_expf_table.py --slow
 *
 * One run, well under 60 s on 16 threads (docs/ARCHITECTURE.md, safety of the machine). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"

#define MAX_WORKERS 64
#define BLOCK_BITS 16   /* 65536 blocks of 65536 bit patterns */
#define HARD_PER_WORKER 4096
#define SLOW_PER_WORKER 4096

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

/* ---- every float ---------------------------------------------------------------------------- */

typedef struct {
    uint64_t finite, finite_softmax;
    uint64_t lib_ref, lib_ref_softmax;        /* library's expf is not the reference */
    uint64_t cand_lib, cand_ref, unproven;    /* tr_expf is not the library's / the reference / a number */
    uint64_t slow;                            /* arguments past the fast path (counted with --slow) */
    double worst_ulp;
    float worst_x;
} tally;
static tally tallies[MAX_WORKERS];
static float hard[MAX_WORKERS][HARD_PER_WORKER];
static int n_hard[MAX_WORKERS];
static float slow[MAX_WORKERS][SLOW_PER_WORKER];
static int n_slow[MAX_WORKERS];

static double ulps(float got, float want) {
    float next = nextafterf(want, got);
    double ulp = fabs((double)next - (double)want);
    return ulp > 0.0 ? fabs((double)got - (double)want) / ulp : 0.0;
}

static void bits_body(void *ctx, int64_t begin, int64_t end, int worker) {
    int want_slow = *(const int *)ctx;
    tally local; /* counted on the stack: two workers' tallies may share a cache line */
    memset(&local, 0, sizeof local);
    for (int64_t b = begin; b < end; b++) {
        for (uint32_t i = 0; i < (1u << BLOCK_BITS); i++) {
            float x = from_bits(((uint32_t)b << BLOCK_BITS) | i);
            if (x != x) continue; /* NaN in, NaN out: nothing to round */
            volatile float lib = expf(x);
            volatile double ref_d = exp((double)x);
            volatile float ref = (float)ref_d;
            volatile float cand = tr_expf(x);
            int softmax_range = x <= 0.0f && x >= -104.0f; /* below -104 every exp is 0 */
            local.finite++;
            local.finite_softmax += (uint64_t)softmax_range;
            local.cand_lib += to_bits(cand) != to_bits(lib);
            local.cand_ref += to_bits(cand) != to_bits(ref);
            local.unproven += cand != cand;
            if (want_slow) {
                int path = tr_expf_path(x);
                if (path == TR_EXPF_TABLE || path == TR_EXPF_UNPROVEN) {
                    local.slow++;
                    if (n_slow[worker] < SLOW_PER_WORKER) slow[worker][n_slow[worker]++] = x;
                }
            }
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
    mine->unproven += local.unproven;
    mine->slow += local.slow;
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

/* every entry of the table of exceptions is an argument that takes the table, once */
static int dead_table_entries(void) {
    int dead = 0;
    for (int i = 0; i < tr_expf_n_exceptions(); i++) {
        uint32_t xb, yb, other_x, other_y;
        tr_expf_exception(i, &xb, &yb);
        if (tr_expf_path(from_bits(xb)) != TR_EXPF_TABLE || to_bits(tr_expf(from_bits(xb))) != yb) dead++;
        for (int j = 0; j < i; j++) {
            tr_expf_exception(j, &other_x, &other_y);
            dead += other_x == xb;
        }
    }
    return dead;
}

int main(int argc, char **argv) {
    int threads = 0, check = 0, like_library = 0;
    const char *hard_path = NULL, *slow_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hard") == 0 && i + 1 < argc) hard_path = argv[++i];
        else if (strcmp(argv[i], "--slow") == 0 && i + 1 < argc) slow_path = argv[++i];
        else if (strcmp(argv[i], "--check") == 0) check = 1;
        else if (strcmp(argv[i], "--like-library") == 0) like_library = 1;
    }
    int cores = tr_cpu()->physical_cores;
    if (threads < 1 || threads > cores) threads = cores; /* never more threads than cores */
    if (threads > MAX_WORKERS) threads = MAX_WORKERS;

    char cpu_line[512];
    tr_cpu_describe(tr_cpu(), cpu_line, sizeof cpu_line);
    printf("bench_expf, built %s %s, %d threads\n%s\n", __DATE__, __TIME__, threads, cpu_line);

    enum { N = 1 << 16 };
    static float xs[N];
    float sum = 0.0f;
    for (int i = 0; i < N; i++) xs[i] = -12.0f * (float)((i * 2654435761u) >> 8 & 0xFFFF) / 65536.0f;
    double lib_ns = cost_ns(expf, xs, N, &sum), cand_ns = cost_ns(tr_expf, xs, N, &sum);
    printf("cost: library %.2f ns per call, tr_expf %.2f ns (one thread, best of 7, arguments in -12..0; sum %.1f)\n",
           lib_ns, cand_ns, (double)sum);

    tr_pool *pool = tr_pool_create(threads);
    if (pool == NULL) return 1;
    int want_slow = slow_path != NULL;
    double t0 = tr_time_sec();
    tr_parallel_for(pool, (int64_t)1 << (32 - BLOCK_BITS), 16, bits_body, &want_slow);
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
        sum_t.unproven += tallies[w].unproven;
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
    printf("bits, tr_expf: %llu results are not the library's expf, %llu are not the reference\n",
           (unsigned long long)sum_t.cand_lib, (unsigned long long)sum_t.cand_ref);
    int dead = dead_table_entries();
    printf("bits, tr_expf: %llu arguments left unproven (a NaN); table of exceptions: %d entries, %d that do not "
           "take the table\n",
           (unsigned long long)sum_t.unproven, tr_expf_n_exceptions(), dead);
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
    if (slow_path != NULL) {
        FILE *f = fopen(slow_path, "w");
        if (f == NULL) return 1;
        for (int w = 0; w < threads; w++)
            for (int i = 0; i < n_slow[w]; i++)
                fprintf(f, "%08x %08x\n", (unsigned)to_bits(slow[w][i]), (unsigned)to_bits((float)exp((double)slow[w][i])));
        fclose(f);
        printf("slow: %llu arguments fail the rounding test of tr_expf\n", (unsigned long long)sum_t.slow);
    }
    printf("elapsed %.1f s\n", secs);
    tr_pool_destroy(pool);
    if (check) {
        int bad = sum_t.cand_ref != 0 || sum_t.unproven != 0 || dead != 0 || (like_library && sum_t.cand_lib != 0);
        printf("check: %s\n", bad ? "FAILED" : "tr_expf is the correctly rounded exp on every float");
        return bad;
    }
    return 0;
}
