/* bench_expf_refs.c — the references' exp against ours (docs/ORIGINS.md §Every piece, row 4): on
 * every one of the 2^32 floats, how many results differ from the correctly rounded exp (tr_expf,
 * proven by bench_expf --check), by how many ulps, how many small results go to zero and large
 * ones to infinity early; then the cost per value of each, one core, 4096 softmax-like arguments.
 *
 * ggml_v_expf below is copied unmodified from llama.cpp b49650a, ggml/src/ggml-cpu/vec.h (MIT,
 * "adapted from arm limited optimized routine"), the AVX-512 variant a GGML_NATIVE build uses on
 * this CPU for softmax and SiLU; ik_llama.cpp f3d6e6e has the same routine (ggml/src/iqk/
 * iqk_utils.h, v_expf). llama.cpp's CPU flash attention, ds4 and colibri call the C library's
 * expf, one value at a time: its bits are the platform's (glibc here). A comparison tool: not part
 * of the engine, not in `make check`.
 *
 *   sh tools/bench_expf_refs.sh        in the trochilus-dev container; about a minute
 */
#define _POSIX_C_SOURCE 199309L /* clock_gettime under -std=c11 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <immintrin.h>

float tr_expf(float x);

#define TARGET __attribute__((target("avx512f,avx512dq,fma")))

/* llama.cpp b49650a, ggml/src/ggml-cpu/vec.h, unmodified: */
// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
TARGET inline static __m512 ggml_v_expf(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d))
    return res;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}
/* end of the copy */

static uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static float u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* one exp's disagreements with the correctly rounded one */
typedef struct {
    uint64_t differ, zero_early, inf_early, other; /* other: NaN or sign where the reference has none */
    uint64_t max_ulp;
    float worst_x;
} tally;

/* how far b is from the reference a, both finite and non-negative: the distance of their bits */
static void count(tally *t, float x, float ref, float got) {
    if (f2u(ref) == f2u(got) || (isnan(ref) && isnan(got))) return;
    t->differ++;
    if (got == 0.0f && ref > 0.0f) {
        t->zero_early++;
    } else if (isinf(got) && !isinf(ref)) {
        t->inf_early++;
    } else if (isfinite(got) && isfinite(ref) && got >= 0.0f && ref >= 0.0f) {
        uint32_t a = f2u(ref), b = f2u(got);
        uint64_t d = a > b ? a - b : b - a;
        if (d > t->max_ulp) {
            t->max_ulp = d;
            t->worst_x = x;
        }
    } else {
        t->other++;
    }
}

typedef struct {
    uint64_t lo, hi; /* bit patterns [lo, hi) */
    tally ggml, libm;
} job;

TARGET static void *sweep(void *arg) {
    job *j = (job *)arg;
    float xs[16], got[16];
    for (uint64_t u = j->lo; u < j->hi; u += 16) {
        for (int l = 0; l < 16; l++) xs[l] = u2f((uint32_t)(u + (uint64_t)l));
        _mm512_storeu_ps(got, ggml_v_expf(_mm512_loadu_ps(xs)));
        for (int l = 0; l < 16; l++) {
            float ref = tr_expf(xs[l]);
            count(&j->ggml, xs[l], ref, got[l]);
            count(&j->libm, xs[l], ref, expf(xs[l]));
        }
    }
    return NULL;
}

static void add(tally *a, const tally *b) {
    a->differ += b->differ;
    a->zero_early += b->zero_early;
    a->inf_early += b->inf_early;
    a->other += b->other;
    if (b->max_ulp > a->max_ulp) {
        a->max_ulp = b->max_ulp;
        a->worst_x = b->worst_x;
    }
}

static void report(const char *name, const tally *t) {
    printf("%-22s differ %11llu of 4294967296 (%.3f%%), max %llu ulp (x = %a), zero early %llu, inf early %llu, "
           "other %llu\n",
           name, (unsigned long long)t->differ, 100.0 * (double)t->differ / 4294967296.0,
           (unsigned long long)t->max_ulp, (double)t->worst_x, (unsigned long long)t->zero_early,
           (unsigned long long)t->inf_early, (unsigned long long)t->other);
}

enum { NV = 4096, RUNS = 15 };
static float g_x[NV], g_y[NV];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

TARGET static void run_ggml(void) {
    for (int i = 0; i < NV; i += 16) _mm512_storeu_ps(g_y + i, ggml_v_expf(_mm512_loadu_ps(g_x + i)));
}
static void run_tr(void) {
    for (int i = 0; i < NV; i++) g_y[i] = tr_expf(g_x[i]);
}
static void run_libm(void) {
    for (int i = 0; i < NV; i++) g_y[i] = expf(g_x[i]);
}

/* ns a value, median of RUNS runs of reps passes over the 4096 arguments */
static double ns_per_value(void (*fn)(void), int reps, double *spread) {
    double t[RUNS];
    fn();
    for (int r = 0; r < RUNS; r++) {
        double a = now();
        for (int k = 0; k < reps; k++) fn();
        t[r] = (now() - a) / ((double)reps * NV) * 1e9;
    }
    qsort(t, RUNS, sizeof t[0], cmp_d);
    *spread = (t[RUNS - 1] - t[0]) / t[RUNS / 2];
    return t[RUNS / 2];
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 16;
    if (threads < 1 || threads > 64) threads = 16;
    pthread_t th[64];
    static job jobs[64];
    uint64_t step = (1ull << 32) / (uint64_t)threads;
    for (int i = 0; i < threads; i++) {
        memset(&jobs[i], 0, sizeof jobs[i]);
        jobs[i].lo = step * (uint64_t)i;
        jobs[i].hi = i == threads - 1 ? (1ull << 32) : step * (uint64_t)(i + 1);
        pthread_create(&th[i], NULL, sweep, &jobs[i]);
    }
    tally g = {0}, l = {0};
    for (int i = 0; i < threads; i++) {
        pthread_join(th[i], NULL);
        add(&g, &jobs[i].ggml);
        add(&l, &jobs[i].libm);
    }
    printf("every float, against tr_expf (correctly rounded):\n");
    report("ggml_v_expf (AVX-512)", &g);
    report("libm expf", &l);

    /* softmax-like arguments: x - max in [-20, 0] */
    uint64_t s = 12345;
    for (int i = 0; i < NV; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        g_x[i] = -20.0f * (float)((s >> 40) & 0xFFFFFF) / 16777216.0f;
    }
    double sp;
    printf("cost, one core, 4096 arguments in [-20, 0], median of %d:\n", RUNS);
    double a = ns_per_value(run_ggml, 2000, &sp);
    printf("  ggml_v_expf (AVX-512)  %.3f ns a value (spread %.1f%%)\n", a, 100 * sp);
    double b = ns_per_value(run_tr, 200, &sp);
    printf("  tr_expf (scalar)       %.3f ns a value (spread %.1f%%)\n", b, 100 * sp);
    double c = ns_per_value(run_libm, 500, &sp);
    printf("  libm expf (scalar)     %.3f ns a value (spread %.1f%%)\n", c, 100 * sp);
    return 0;
}
