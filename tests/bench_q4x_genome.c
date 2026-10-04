/* bench_q4x_genome.c — AVX2's Q4_K tile sequenced like a genome (docs/MEASUREMENTS.md §AVX2's own Q4_K tile).
 *
 * The prompt's Q4_K road on AVX2 (kernels_x86.c's avx2_q4x_panel and q4x_tile2_t): a W16 panel built once per
 * (group, 16 rows), then tiles of 16 rows against T prepared rows, a half of 8 rows after the other. Where a
 * tile's nanoseconds go is measured here, one piece removed at a time (docs/LESSONS.md #176, #178):
 *
 *   clock      a chain of dependent integer adds (1 cycle each): the core's clock in this run;
 *   tier       the active tier's q4x_tile itself;
 *   full       a copy of the tier's tile, every piece in it;
 *   oddb       candidate: the odd rows biased by 2^31 in the tile too (one add a window), so a window's split to
 *              int64 is one vpsrlq and two vpaddq a digit instead of a blend of two shifts and two adds;
 *   fma        candidate: the block's exact sums by fma (te, to and the mins' chain: every value an integer under
 *              2^53, so the same bits as mul and add);
 *   oddb+fma   both;
 *   noblock    full without the block's f64 tail (the int64 sums kept alive, not converted);
 *   nowin      noblock without the windows' int64 split (a window's int32 lanes summed on, wrong but timed);
 *   nobcast    nowin with the prepared rows' digits taken from registers (no broadcast loads);
 *   halves     candidate at T = 2: both halves in one pass, each broadcast serving 16 rows;
 *   tails      only the windows' split and the block's tail (no Winograd products).
 * The lines that keep the arithmetic (tier, full, oddb, fma, oddb+fma) are checked bit for bit against the tier's
 * tile before timing; a mismatch fails the run. Lines run in turn, run after run: medians of N runs, spread
 * (max - min) / median; ns and cycles a call, and weight MACs a cycle (16 rows x n x T a call).
 *
 *   bench_q4x_genome [--runs N] [--ms M]     one core, n = 2048, T = 3 and 4, the panel in L2 (built before)
 * Run natively on a still machine (tools/measure_guard.lib), with TR_CPU_MAX=avx2 or on any AVX2 CPU. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAVE_X86 1
#endif

#define MAX_RUNS 31
#define N 2048
#define ROWS TR_PM_ROWS

static volatile float g_sink;

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_spread(double *v, int n, double *spread) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    double med = v[n / 2];
    *spread = med > 0 ? (v[n - 1] - v[0]) / med : 0;
    return med;
}

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(g_rng >> 33);
}

#if HAVE_X86

/* 16 dependent adds an iteration: 1 cycle each, the loop's dec/jnz run beside them */
static void clock_chain(int64_t iters) {
    uint64_t a = 0;
    __asm__ volatile("1:\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "dec %1\n\tjnz 1b"
                     : "+r"(a), "+r"(iters)
                     :
                     : "cc");
    g_sink = (float)a;
}

static double clock_ghz(void) {
    const int64_t iters = 600000;
    clock_chain(iters / 10);
    double t0 = tr_time_sec();
    clock_chain(iters);
    double t1 = tr_time_sec();
    return (double)iters * 16 / ((t1 - t0) * 1e9);
}

/* the pieces a variant keeps */
enum { P_INNER = 1, P_WIN = 2, P_BLOCK = 4, P_BCAST = 8, P_ODDB = 16, P_FMA = 32 };

#define TGT __attribute__((target("avx2,fma")))

TGT static inline __m256i hi64(__m256i a) {
    return _mm256_blend_epi32(_mm256_srli_epi64(a, 32), _mm256_srai_epi32(a, 31), 0xAA);
}

TGT static inline __m256d i64_pd(__m256i x) {
    return _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(x, _mm256_set1_epi64x(0x4338000000000000ll))),
                         _mm256_set1_pd(6755399441055744.0));
}

static inline int32_t ld32(const int16_t *p) {
    int32_t x;
    memcpy(&x, p, sizeof x);
    return x;
}

TGT static inline __m256d mad(__m256d a, __m256d b, __m256d c, const int fma) {
    return fma ? _mm256_fmadd_pd(a, b, c) : _mm256_add_pd(_mm256_mul_pd(a, b), c);
}

/* kernels_x86.c's q4x_tile2_t with the pieces of P */
TGT static inline __attribute__((always_inline)) void tile_v(const unsigned char *panel, const void *const *xq, int64_t n,
                                                             float *y, int64_t ys, const int T, const int P) {
    tr_q4x_view v[4];
#pragma GCC unroll 4
    for (int t = 0; t < T; t++) v[t] = tr_q4x_view_of(xq[t], n);
    const __m256i bias4 = _mm256_set1_epi64x(4ll << 31), oddb = _mm256_setr_epi32(0, INT32_MIN, 0, INT32_MIN, 0, INT32_MIN, 0, INT32_MIN);
    const __m256d b16 = _mm256_set1_pd((double)TR_Q4X_BASE);
    __m256i sinki = _mm256_setzero_si256();
    for (int hh = 0; hh < 2; hh++) {
        __m256d ye[4], yo[4];
#pragma GCC unroll 4
        for (int t = 0; t < T; t++) ye[t] = yo[t] = _mm256_setzero_pd();
        for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
            const unsigned char *sb = panel + (size_t)s * TR_Q4X_SB_BYTES;
            const double *cst = (const double *)(const void *)sb + 4 * hh;
            __m256i ae[4][2], ao[4][2], a[4][2];
#pragma GCC unroll 4
            for (int t = 0; t < T; t++) ae[t][0] = ae[t][1] = ao[t][0] = ao[t][1] = a[t][0] = a[t][1] = _mm256_setzero_si256();
            for (int c = 0; c < 4; c++) {
                const unsigned char *win = sb + TR_Q4X_CONST_BYTES + (size_t)c * TR_Q4X_WIN_BYTES + 32 * hh;
                const int64_t w = 4 * s + c, k0 = 64 * w;
                __m256i rn = _mm256_loadu_si256((const __m256i *)(const void *)win);
                if (P & P_ODDB) rn = _mm256_add_epi32(rn, oddb);
                if ((P & P_WIN) || c == 0)
#pragma GCC unroll 4
                    for (int t = 0; t < T; t++) {
                        a[t][0] = _mm256_sub_epi32(rn, _mm256_set1_epi32(v[t].tok[2 * w]));
                        a[t][1] = _mm256_sub_epi32(rn, _mm256_set1_epi32(v[t].tok[2 * w + 1]));
                    }
                if (P & P_INNER) {
                    __m256i r0[4], r1[4];
#pragma GCC unroll 4
                    for (int t = 0; t < T; t++) {
                        r0[t] = _mm256_set1_epi32(ld32(v[t].v0 + k0));
                        r1[t] = _mm256_set1_epi32(ld32(v[t].v1 + k0));
                    }
#pragma GCC unroll 4
                    for (int u = 0; u < 16; u++) {
                        const __m256i wl = _mm256_loadu_si256((const __m256i *)(const void *)(win + 64 + 128 * u));
                        const __m256i wh = _mm256_loadu_si256((const __m256i *)(const void *)(win + 128 + 128 * u));
#pragma GCC unroll 4
                        for (int t = 0; t < T; t++) {
                            const __m256i b0a = (P & P_BCAST) ? _mm256_set1_epi32(ld32(v[t].v0 + k0 + 32 + 2 * u)) : r0[t];
                            const __m256i b0b = (P & P_BCAST) ? _mm256_set1_epi32(ld32(v[t].v0 + k0 + 2 * u)) : r1[t];
                            const __m256i b1a = (P & P_BCAST) ? _mm256_set1_epi32(ld32(v[t].v1 + k0 + 32 + 2 * u)) : r1[t];
                            const __m256i b1b = (P & P_BCAST) ? _mm256_set1_epi32(ld32(v[t].v1 + k0 + 2 * u)) : r0[t];
                            a[t][0] = _mm256_add_epi32(a[t][0], _mm256_madd_epi16(_mm256_add_epi16(wl, b0a), _mm256_add_epi16(wh, b0b)));
                            a[t][1] = _mm256_add_epi32(a[t][1], _mm256_madd_epi16(_mm256_add_epi16(wl, b1a), _mm256_add_epi16(wh, b1b)));
                        }
                    }
                }
                if (P & P_WIN)
#pragma GCC unroll 4
                    for (int t = 0; t < T; t++)
#pragma GCC unroll 2
                        for (int d = 0; d < 2; d++) {
                            ao[t][d] = _mm256_add_epi64(ao[t][d], (P & P_ODDB) ? _mm256_srli_epi64(a[t][d], 32) : hi64(a[t][d]));
                            ae[t][d] = _mm256_add_epi64(ae[t][d], a[t][d]);
                        }
            }
            if (!(P & P_BLOCK)) {
#pragma GCC unroll 4
                for (int t = 0; t < T; t++)
                    sinki = _mm256_add_epi64(sinki, _mm256_add_epi64(_mm256_add_epi64(ae[t][0], ao[t][1]),
                                                                     _mm256_add_epi64(a[t][0], a[t][1])));
                continue;
            }
#pragma GCC unroll 4
            for (int t = 0; t < T; t++) {
                const __m256i ev0 = _mm256_sub_epi64(_mm256_sub_epi64(ae[t][0], _mm256_slli_epi64(ao[t][0], 32)), bias4);
                const __m256i ev1 = _mm256_sub_epi64(_mm256_sub_epi64(ae[t][1], _mm256_slli_epi64(ao[t][1], 32)), bias4);
                const __m256i od0 = (P & P_ODDB) ? _mm256_sub_epi64(ao[t][0], bias4) : ao[t][0];
                const __m256i od1 = (P & P_ODDB) ? _mm256_sub_epi64(ao[t][1], bias4) : ao[t][1];
                const int F = (P & P_FMA) != 0;
                const __m256d te = mad(i64_pd(ev1), b16, i64_pd(ev0), F);
                const __m256d to = mad(i64_pd(od1), b16, i64_pd(od0), F);
                const double *bf = v[t].bf + 8 * s;
                __m256d Me = _mm256_mul_pd(_mm256_loadu_pd(cst + 32), _mm256_set1_pd(bf[0]));
                __m256d Mo = _mm256_mul_pd(_mm256_loadu_pd(cst + 40), _mm256_set1_pd(bf[0]));
#pragma GCC unroll 8
                for (int j = 1; j < 8; j++) {
                    const __m256d bj = _mm256_set1_pd(bf[j]);
                    Me = mad(_mm256_loadu_pd(cst + 32 + 16 * j), bj, Me, F);
                    Mo = mad(_mm256_loadu_pd(cst + 40 + 16 * j), bj, Mo, F);
                }
                const __m256d sc = _mm256_set1_pd(v[t].scale[s]);
                const __m256d ve = _mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(cst), te), _mm256_mul_pd(_mm256_loadu_pd(cst + 16), Me));
                const __m256d vo = _mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(cst + 8), to), _mm256_mul_pd(_mm256_loadu_pd(cst + 24), Mo));
                ye[t] = _mm256_add_pd(ye[t], _mm256_mul_pd(ve, sc));
                yo[t] = _mm256_add_pd(yo[t], _mm256_mul_pd(vo, sc));
            }
        }
#pragma GCC unroll 4
        for (int t = 0; t < T; t++) {
            const __m128 e = _mm256_cvtpd_ps(ye[t]), o = _mm256_cvtpd_ps(yo[t]);
            float *yt = y + (size_t)t * (size_t)ys + 8 * hh;
            _mm_storeu_ps(yt, _mm_unpacklo_ps(e, o));
            _mm_storeu_ps(yt + 4, _mm_unpackhi_ps(e, o));
        }
    }
    if (!(P & P_BLOCK)) {
        int32_t l[8];
        _mm256_storeu_si256((__m256i *)(void *)l, sinki);
        y[0] += (float)l[0];
    }
}

/* candidate "halves": both halves in one pass (16 rows, T = 2): each broadcast serves 16 rows instead of 8, and the
 * block's bf broadcasts both halves; the windows' split and the f64 steps as full's */
TGT static void halves_2(const unsigned char *panel, const void *const *xq, int64_t n, float *y, int64_t ys) {
    enum { T = 2 };
    tr_q4x_view v[T];
    for (int t = 0; t < T; t++) v[t] = tr_q4x_view_of(xq[t], n);
    const __m256i bias4 = _mm256_set1_epi64x(4ll << 31);
    const __m256d b16 = _mm256_set1_pd((double)TR_Q4X_BASE);
    __m256d yv[T][4]; /* [t][even 0-7, even 8-15, odd 0-7, odd 8-15] */
    for (int t = 0; t < T; t++) yv[t][0] = yv[t][1] = yv[t][2] = yv[t][3] = _mm256_setzero_pd();
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        const unsigned char *sb = panel + (size_t)s * TR_Q4X_SB_BYTES;
        const double *cst = (const double *)(const void *)sb;
        __m256i ae[T][2][2], ao[T][2][2]; /* [t][digit][half] */
        for (int t = 0; t < T; t++)
            for (int d = 0; d < 2; d++) ae[t][d][0] = ae[t][d][1] = ao[t][d][0] = ao[t][d][1] = _mm256_setzero_si256();
        for (int c = 0; c < 4; c++) {
            const unsigned char *win = sb + TR_Q4X_CONST_BYTES + (size_t)c * TR_Q4X_WIN_BYTES;
            const int64_t w = 4 * s + c, k0 = 64 * w;
            const __m256i rn0 = _mm256_loadu_si256((const __m256i *)(const void *)win);
            const __m256i rn1 = _mm256_loadu_si256((const __m256i *)(const void *)(win + 32));
            __m256i a[T][2][2];
#pragma GCC unroll 2
            for (int t = 0; t < T; t++)
#pragma GCC unroll 2
                for (int d = 0; d < 2; d++) {
                    const __m256i tk = _mm256_set1_epi32(v[t].tok[2 * w + d]);
                    a[t][d][0] = _mm256_sub_epi32(rn0, tk);
                    a[t][d][1] = _mm256_sub_epi32(rn1, tk);
                }
#pragma GCC unroll 2
            for (int u = 0; u < 16; u++) {
                const __m256i wl0 = _mm256_loadu_si256((const __m256i *)(const void *)(win + 64 + 128 * u));
                const __m256i wl1 = _mm256_loadu_si256((const __m256i *)(const void *)(win + 96 + 128 * u));
                const __m256i wh0 = _mm256_loadu_si256((const __m256i *)(const void *)(win + 128 + 128 * u));
                const __m256i wh1 = _mm256_loadu_si256((const __m256i *)(const void *)(win + 160 + 128 * u));
#pragma GCC unroll 2
                for (int t = 0; t < T; t++)
#pragma GCC unroll 2
                    for (int d = 0; d < 2; d++) {
                        const int16_t *vd = d ? v[t].v1 : v[t].v0;
                        const __m256i ba = _mm256_set1_epi32(ld32(vd + k0 + 32 + 2 * u)), bb = _mm256_set1_epi32(ld32(vd + k0 + 2 * u));
                        a[t][d][0] = _mm256_add_epi32(a[t][d][0], _mm256_madd_epi16(_mm256_add_epi16(wl0, ba), _mm256_add_epi16(wh0, bb)));
                        a[t][d][1] = _mm256_add_epi32(a[t][d][1], _mm256_madd_epi16(_mm256_add_epi16(wl1, ba), _mm256_add_epi16(wh1, bb)));
                    }
            }
#pragma GCC unroll 2
            for (int t = 0; t < T; t++)
#pragma GCC unroll 2
                for (int d = 0; d < 2; d++)
#pragma GCC unroll 2
                    for (int hh = 0; hh < 2; hh++) {
                        ao[t][d][hh] = _mm256_add_epi64(ao[t][d][hh], hi64(a[t][d][hh]));
                        ae[t][d][hh] = _mm256_add_epi64(ae[t][d][hh], a[t][d][hh]);
                    }
        }
#pragma GCC unroll 2
        for (int t = 0; t < T; t++) {
            const double *bf = v[t].bf + 8 * s;
            __m256d M[4];
            const __m256d b0 = _mm256_set1_pd(bf[0]);
            for (int q = 0; q < 4; q++) M[q] = _mm256_mul_pd(_mm256_loadu_pd(cst + 32 + 4 * q), b0);
#pragma GCC unroll 8
            for (int j = 1; j < 8; j++) {
                const __m256d bj = _mm256_set1_pd(bf[j]);
#pragma GCC unroll 4
                for (int q = 0; q < 4; q++) M[q] = _mm256_add_pd(M[q], _mm256_mul_pd(_mm256_loadu_pd(cst + 32 + 16 * j + 4 * q), bj));
            }
            const __m256d sc = _mm256_set1_pd(v[t].scale[s]);
#pragma GCC unroll 2
            for (int hh = 0; hh < 2; hh++) {
                const __m256i ev0 = _mm256_sub_epi64(_mm256_sub_epi64(ae[t][0][hh], _mm256_slli_epi64(ao[t][0][hh], 32)), bias4);
                const __m256i ev1 = _mm256_sub_epi64(_mm256_sub_epi64(ae[t][1][hh], _mm256_slli_epi64(ao[t][1][hh], 32)), bias4);
                const __m256d te = _mm256_add_pd(_mm256_mul_pd(i64_pd(ev1), b16), i64_pd(ev0));
                const __m256d to = _mm256_add_pd(_mm256_mul_pd(i64_pd(ao[t][1][hh]), b16), i64_pd(ao[t][0][hh]));
                /* M[0..1]: the even rows' mins (rows 0-6, 8-14), M[2..3] the odd ones' */
                const __m256d ve = _mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(cst + 4 * hh), te),
                                                 _mm256_mul_pd(_mm256_loadu_pd(cst + 16 + 4 * hh), M[hh]));
                const __m256d vo = _mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(cst + 8 + 4 * hh), to),
                                                 _mm256_mul_pd(_mm256_loadu_pd(cst + 24 + 4 * hh), M[2 + hh]));
                yv[t][hh] = _mm256_add_pd(yv[t][hh], _mm256_mul_pd(ve, sc));
                yv[t][2 + hh] = _mm256_add_pd(yv[t][2 + hh], _mm256_mul_pd(vo, sc));
            }
        }
    }
    for (int t = 0; t < T; t++)
        for (int hh = 0; hh < 2; hh++) {
            const __m128 e = _mm256_cvtpd_ps(yv[t][hh]), o = _mm256_cvtpd_ps(yv[t][2 + hh]);
            float *yt = y + (size_t)t * (size_t)ys + 8 * hh;
            _mm_storeu_ps(yt, _mm_unpacklo_ps(e, o));
            _mm_storeu_ps(yt + 4, _mm_unpackhi_ps(e, o));
        }
}

#define ALL (P_INNER | P_WIN | P_BLOCK | P_BCAST)
typedef void (*tile_fn)(const unsigned char *, const void *const *, int64_t, float *, int64_t);
#define V(name, T, P) \
    TGT static void name(const unsigned char *p, const void *const *x, int64_t n, float *y, int64_t ys) { tile_v(p, x, n, y, ys, T, P); }
#define VS(T)                                                     \
    V(full_##T, T, ALL)                                           \
    V(oddb_##T, T, ALL | P_ODDB)                                  \
    V(fma_##T, T, ALL | P_FMA)                                    \
    V(both_##T, T, ALL | P_ODDB | P_FMA)                          \
    V(noblock_##T, T, P_INNER | P_WIN | P_BCAST)                  \
    V(nowin_##T, T, P_INNER | P_BCAST)                            \
    V(nobcast_##T, T, P_INNER)                                    \
    V(tails_##T, T, P_WIN | P_BLOCK)
V(full_2, 2, ALL)
VS(3)
VS(4)

typedef struct {
    const char *name;
    int T, checked;
    tile_fn f;
} variant;

static const variant VARS[] = {
    {"tier", 2, 1, NULL},        {"full", 2, 1, full_2},       {"halves", 2, 1, halves_2},
    {"tier", 3, 1, NULL},        {"full", 3, 1, full_3},       {"oddb", 3, 1, oddb_3},       {"fma", 3, 1, fma_3},
    {"oddb+fma", 3, 1, both_3},  {"noblock", 3, 0, noblock_3}, {"nowin", 3, 0, nowin_3},     {"nobcast", 3, 0, nobcast_3},
    {"tails", 3, 0, tails_3},    {"tier", 4, 1, NULL},         {"full", 4, 1, full_4},       {"oddb", 4, 1, oddb_4},
    {"fma", 4, 1, fma_4},        {"oddb+fma", 4, 1, both_4},   {"noblock", 4, 0, noblock_4}, {"nowin", 4, 0, nowin_4},
    {"nobcast", 4, 0, nobcast_4}, {"tails", 4, 0, tails_4},
};
#define NV ((int)(sizeof VARS / sizeof VARS[0]))

static int genome(const tr_kernels *K, int runs, double run_ms) {
    static unsigned char rows[ROWS * (N / 256) * TR_Q4_K_BLOCK_BYTES];
    static unsigned char xq[4 * (N / 256) * 1152] __attribute__((aligned(64)));
    static unsigned char panel[N / 256 * 9728] __attribute__((aligned(64)));
    static float x[4][N];
    const size_t rb = tr_row_bytes(TR_TYPE_Q4_K, N), xb = tr_q4x_bytes(N);
    for (size_t b = 0; b < sizeof rows / TR_Q4_K_BLOCK_BYTES; b++) {
        unsigned char *blk = rows + b * TR_Q4_K_BLOCK_BYTES;
        uint16_t d = (uint16_t)(0x2C00 | (rnd() & 0x3FF)), dmin = (uint16_t)(0x2800 | (rnd() & 0x3FF));
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (int i = 4; i < TR_Q4_K_BLOCK_BYTES; i++) blk[i] = (unsigned char)rnd();
    }
    const void *xp[4];
    for (int t = 0; t < 4; t++) {
        for (int i = 0; i < N; i++) x[t][i] = (float)rnd() * 0x1p-30f - 1.0f;
        K->q4x_prep(x[t], N, xq + (size_t)t * xb);
        xp[t] = xq + (size_t)t * xb;
    }
    K->q4x_panel(rows, rb, N, panel, NULL);
    float want[ROWS * 4], got[ROWS * 4];
    long checked = 0;
    for (int i = 0; i < NV; i++) {
        if (!VARS[i].checked) continue;
        K->q4x_tile(panel, xp, N, VARS[i].T, want, ROWS);
        if (VARS[i].f != NULL) VARS[i].f(panel, xp, N, got, ROWS);
        else K->q4x_tile(panel, xp, N, VARS[i].T, got, ROWS);
        if (memcmp(want, got, sizeof(float) * (size_t)(ROWS * VARS[i].T)) != 0) {
            printf("MISMATCH %s T=%d\n", VARS[i].name, VARS[i].T);
            return 1;
        }
        checked += ROWS * VARS[i].T;
    }
    if (checked == 0) return 1;
    printf("# the checked lines' outputs equal the tier's tile bit for bit: %ld compared\n", checked);
    long calls[NV];
    double ns[NV][MAX_RUNS], ghz[MAX_RUNS];
    for (int pass = -1; pass < runs; pass++) {
        const double g1 = clock_ghz();
        for (int i = 0; i < NV; i++) {
            const long c = pass < 0 ? 300 : calls[i];
            double t0 = tr_time_sec();
            for (long k = 0; k < c; k++) {
                if (VARS[i].f != NULL) VARS[i].f(panel, xp, N, got, ROWS);
                else K->q4x_tile(panel, xp, N, VARS[i].T, got, ROWS);
            }
            const double v = (tr_time_sec() - t0) * 1e9 / (double)c;
            if (pass < 0) calls[i] = (long)(run_ms * 1e6 / v) + 1;
            else ns[i][pass] = v;
        }
        const double g2 = clock_ghz();
        if (pass >= 0) ghz[pass] = g2 < g1 ? g2 : g1;
    }
    g_sink = got[0];
    double sp;
    const double g = median_spread(ghz, runs, &sp);
    printf("clock %.3f GHz  spread %.1f%%\n", g, sp * 100);
    printf("%-10s %2s %10s %8s %10s %10s\n", "line", "T", "ns a call", "spread", "cycles", "MAC/cycle");
    for (int i = 0; i < NV; i++) {
        const double m = median_spread(ns[i], runs, &sp);
        printf("%-10s %2d %10.0f %7.1f%% %10.0f %10.2f\n", VARS[i].name, VARS[i].T, m, sp * 100, m * g,
               (double)ROWS * N * VARS[i].T / (m * g));
    }
    return 0;
}
#endif

int main(int argc, char **argv) {
    int runs = 11;
    double run_ms = 5;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ms") == 0 && i + 1 < argc) run_ms = atof(argv[++i]);
        else {
            fprintf(stderr, "usage: bench_q4x_genome [--runs N] [--ms M]\n");
            return 2;
        }
    }
    if (runs < 3) runs = 3;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
#if HAVE_X86
    tr_kernels_init();
    const tr_kernels *K = tr_kernels_tier("avx2");
    if (K == NULL || K->q4x_tile == NULL) {
        printf("bench_q4x_genome: no AVX2 tier with a Q4_K tile on this CPU\n");
        return 0;
    }
    printf("# AVX2's Q4_K tile, one core, n = %d, %d runs, median and spread (max-min)/median\n", N, runs);
    tr_pool *pool = tr_pool_create(1); /* pins this thread to the first slot */
    const int r = genome(K, runs, run_ms);
    tr_pool_destroy(pool);
    return r;
#else
    printf("bench_q4x_genome: x86 only\n");
    return 0;
#endif
}
