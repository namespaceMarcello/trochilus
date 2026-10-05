/* bench_head_bound.c — question 73: the output head's argmax by a bound, its time piece by piece before any engine
 * code (docs/MEASUREMENTS.md §The head's argmax by a bound).
 *
 * The head is 50304 rows of 2048. Today a greedy token computes every row (tr_matmul_s, the engine's call). The
 * bound reads each row's scales and its codes' top bits only (a plane), against h rounded per block of 32 to int8
 * (its rounding error counted), and gives every row a provable upper bound of the engine's own float result; the
 * rows whose bound reaches the best exact score are then computed exactly (scattered rows). Measured here:
 *
 *   engine     the full head call as the decode makes it (Q8_0: dot_row2 pairs; Q4_K: the prepared integers);
 *   plane      the bound pass over the 50304 rows: Q8_0's high nibbles (1152 B a row, 0.53 of 2176), Q4_K's top 3
 *              bits as a two-bit and a one-bit plane (896 B a row, 0.78 of 1152), int8 h, AVX2 + F16C;
 *   rows N     N random rows computed exactly, in increasing order, by the engine's own kernel: from the original
 *              rows ("orig") or rebuilt from the two planes first ("planes"), N = 100, 1000, 12000;
 *   scan       the rows whose bound reaches a threshold, collected from the 50304 bounds (one thread);
 *   region     an empty tr_parallel_for: what each extra pass over the pool costs.
 * Every time from RAM (a 256 MiB buffer read between runs), at 1, 4, 8 and 16 threads; "plane" also in cache (the
 * first 256 rows, one thread). Each kernel's bound is checked against the engine's value of every row before
 * timing (g_checked counts them and fails the run if none was), and against the bound's formula in double.
 *
 *   bench_head_bound [--runs N] [--threads 1,4,8,16]     TR_CPU_MAX=avx2 for the engine's AVX2 tier
 * Run by tools/bench_native.sh bench_head_bound (the marker, a still machine). */
#include <math.h>
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

#define VOCAB 50304
#define COLS 2048
#define NB8 (COLS / 32)  /* Q8_0 blocks a row */
#define NS4 (COLS / 256) /* Q4_K super-blocks a row */
#define P8_HI (NB8 * 2 + NB8 * 16)
#define P8_LO (NB8 * 16)
#define P4_HI (NS4 * 112)
#define P4_LO (NS4 * 32)
#define MAX_RUNS 31
#define FLUSH_BYTES ((size_t)256 << 20)
/* every timed run reads its own copy of the heads, untouched for NCOPY - 1 runs: a flush from one thread leaves
 * the other CCD's L3 warm (docs/LESSONS.md #344) */
#define NCOPY 4
#define CP(base, row_bytes, c) ((base) + (size_t)(c) * (size_t)VOCAB * (size_t)(row_bytes))
#define MARGIN 1.52587890625e-05 /* 2^-16: every float sum's rounding here and in the engine is below it */

static int64_t g_checked, g_pruned; /* rows checked; rows the argmax by the bound never computed (0 fails the run) */
static volatile uint64_t g_sink;

static int cmp_float_desc(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x < y) - (x > y);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_spread(double *v, int n, double *spread) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    *spread = v[n / 2] > 0 ? (v[n - 1] - v[0]) / v[n / 2] : 0;
    return v[n / 2];
}

static uint32_t g_seed = 73;
static uint32_t rnd(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}
static float rnd_normal(void) {
    float s = 0;
    for (int i = 0; i < 12; i++) s += (float)(rnd() & 0xFFFF) / 65536.0f;
    return s - 6.0f;
}

static unsigned char *g_flush;
static void flush_caches(void) {
    uint64_t s = 0;
    for (size_t i = 0; i < FLUSH_BYTES; i += 64) s += g_flush[i];
    g_sink += s;
}

/* ---- the token prepared for the bound: h per block of 32 as int8, and each block's terms ------------------- */

typedef struct {
    int8_t X[COLS], Xl[COLS]; /* wide: h ~ delta (256 X + Xl), two int8 digits of a 16-bit code; else h ~ delta X */
    int wide;
    /* Q8_0 (blocks of 32): U = sum_b d_b (a_b I_b + be_b) + |d_b| g_b, I_b = sum u_k X_k, u = high nibble + 8 */
    float a8[NB8], be8[NB8], g8[NB8];
    /* Q4_K (sub-blocks of 32): U = sum_s sum_j d_s sc_j (a_j I_j + be_j) + |d_s| sc_j g_j - dmin_s m_j (H_j - mu_j)
     * with I_j = sum hi_k X_k, hi = q >> 1; mu_j carries dmin's sign through |dmin| (see bound_ref4) */
    float a4[NB8], be4[NB8], g4[NB8], H4[NB8], mu4[NB8];
} prep;

static float up(double v) { /* the float at or above v */
    float f = (float)v;
    return (double)f >= v ? f : nextafterf(f, INFINITY);
}

static void prep_build(const float *h, prep *p, int wide) {
    const long lim = wide ? 32639 : 127; /* 127 * 257: the high digit stays within int8 */
    p->wide = wide;
    for (int b = 0; b < NB8; b++) {
        const float *x = h + 32 * b;
        float m = 0;
        for (int k = 0; k < 32; k++) m = fmaxf(m, fabsf(x[k]));
        float delta = m > 0 ? m / (float)lim : 1.0f;
        double sx = 0, sax = 0, e = 0, ha = 0, hs = 0;
        for (int k = 0; k < 32; k++) {
            long q = lrintf(x[k] / delta);
            if (q > lim) q = lim;
            if (q < -lim) q = -lim;
            long qh = wide ? (long)floor((double)(q + 128) / 256.0) : q;
            p->X[32 * b + k] = (int8_t)qh;
            p->Xl[32 * b + k] = (int8_t)(q - 256 * qh);
            sx += (double)q;
            sax += fabs((double)q);
            e += fabs((double)x[k] - (double)delta * (double)q);
            ha += fabs((double)x[k]);
            hs += (double)x[k];
        }
        e *= 1.0 + 1e-12;
        /* Q8_0: d (16 delta I - 120.5 delta sumX) + |d| (7.5 delta sum|X| + 128 E), the margins on |d| */
        double g = 7.5 * delta * sax + 128.0 * e;
        double mag = 240.0 * delta * sax + 120.5 * delta * fabs(sx) + g + 128.0 * ha;
        p->a8[b] = 16.0f * delta;
        p->be8[b] = (float)(-120.5 * delta * sx);
        p->g8[b] = up(g + MARGIN * (mag + fabs(-120.5 * delta * sx)));
        /* Q4_K: d sc (2 delta I + delta sumX / 2) + |d| sc (delta sum|X| / 2 + 15 E), the min term -dmin m H */
        double g4 = 0.5 * delta * sax + 15.0 * e;
        double mag4 = 14.0 * delta * sax + 0.5 * delta * fabs(sx) + g4 + 15.0 * ha;
        p->a4[b] = 2.0f * delta;
        p->be4[b] = (float)(0.5 * delta * sx);
        p->g4[b] = up(g4 + MARGIN * mag4);
        p->H4[b] = (float)hs;
        p->mu4[b] = up(MARGIN * 4 * ha + fabs(hs - (double)(float)hs));
    }
}

/* ---- the planes, built from the original rows (as a load would) ------------------------------------------- */

static void planes8(const unsigned char *row, unsigned char *hi, unsigned char *lo) {
    for (int b = 0; b < NB8; b++) {
        const unsigned char *blk = row + 34 * b;
        memcpy(hi + 2 * b, blk, 2);
        const int8_t *q = (const int8_t *)(blk + 2);
        for (int i = 0; i < 16; i++) {
            int u0 = (q[i] >> 4) + 8, u1 = (q[i + 16] >> 4) + 8;
            hi[2 * NB8 + 16 * b + i] = (unsigned char)(u0 | (u1 << 4));
            lo[16 * b + i] = (unsigned char)((q[i] & 15) | ((q[i + 16] & 15) << 4));
        }
    }
}

static void rebuild8(const unsigned char *hi, const unsigned char *lo, unsigned char *row) {
    for (int b = 0; b < NB8; b++) {
        unsigned char *blk = row + 34 * b;
        memcpy(blk, hi + 2 * b, 2);
        for (int i = 0; i < 16; i++) {
            int h = hi[2 * NB8 + 16 * b + i], l = lo[16 * b + i];
            blk[2 + i] = (unsigned char)(int8_t)(((h & 15) - 8) * 16 + (l & 15));
            blk[2 + 16 + i] = (unsigned char)(int8_t)(((h >> 4) - 8) * 16 + (l >> 4));
        }
    }
}

/* Q4_K: the codes in the elements' order (ggml's 64-weight chunks: low nibbles the first 32, high the next) */
static void codes4(const unsigned char *sb, uint8_t q[256]) {
    for (int c = 0; c < 4; c++)
        for (int i = 0; i < 32; i++) {
            q[64 * c + i] = sb[16 + 32 * c + i] & 15;
            q[64 * c + 32 + i] = sb[16 + 32 * c + i] >> 4;
        }
}

static void planes4(const unsigned char *row, unsigned char *hi, unsigned char *lo) {
    for (int s = 0; s < NS4; s++) {
        const unsigned char *sb = row + 144 * s;
        unsigned char *ph = hi + 112 * s;
        uint8_t q[256];
        codes4(sb, q);
        memcpy(ph, sb, 16);
        memset(ph + 16, 0, 96);
        memset(lo + 32 * s, 0, 32);
        for (int k = 0; k < 256; k++) {
            int half = k / 128, j = (k % 128) / 32, i = k % 32;
            ph[16 + 32 * half + i] |= (unsigned char)((q[k] >> 2) << (2 * j));
            ph[80 + k / 8] |= (unsigned char)(((q[k] >> 1) & 1) << (k % 8));
            lo[32 * s + k / 8] |= (unsigned char)((q[k] & 1) << (k % 8));
        }
    }
}

static void rebuild4(const unsigned char *hi, const unsigned char *lo, unsigned char *row) {
    for (int s = 0; s < NS4; s++) {
        const unsigned char *ph = hi + 112 * s;
        unsigned char *sb = row + 144 * s;
        memcpy(sb, ph, 16);
        uint8_t q[256];
        for (int k = 0; k < 256; k++) {
            int half = k / 128, j = (k % 128) / 32, i = k % 32;
            q[k] = (uint8_t)((((ph[16 + 32 * half + i] >> (2 * j)) & 3) << 2) | (((ph[80 + k / 8] >> (k % 8)) & 1) << 1) |
                             ((lo[32 * s + k / 8] >> (k % 8)) & 1));
        }
        for (int c = 0; c < 4; c++)
            for (int i = 0; i < 32; i++) sb[16 + 32 * c + i] = (unsigned char)(q[64 * c + i] | (q[64 * c + 32 + i] << 4));
    }
}

static void scale_min4(const unsigned char *s12, int j, int *sc, int *m) { /* ggml's get_scale_min_k4 */
    if (j < 4) {
        *sc = s12[j] & 63;
        *m = s12[j + 4] & 63;
    } else {
        *sc = (s12[j + 4] & 0xF) | ((s12[j - 4] >> 6) << 4);
        *m = (s12[j + 4] >> 4) | ((s12[j] >> 6) << 4);
    }
}

/* ---- the bound's formula in double (the reference the kernels are checked against) ------------------------- */

static long xv(const prep *p, int k) { return p->wide ? 256L * p->X[k] + p->Xl[k] : (long)p->X[k]; }

static double bound_ref8(const unsigned char *hi, const prep *p) {
    double U = 0;
    for (int b = 0; b < NB8; b++) {
        uint16_t dh;
        memcpy(&dh, hi + 2 * b, 2);
        double d = tr_half_to_float(dh);
        long I = 0;
        for (int i = 0; i < 16; i++) {
            int v = hi[2 * NB8 + 16 * b + i];
            I += (long)(v & 15) * xv(p, 32 * b + i) + (long)(v >> 4) * xv(p, 32 * b + 16 + i);
        }
        U += d * ((double)p->a8[b] * (double)I + p->be8[b]) + fabs(d) * p->g8[b];
    }
    return U;
}

static double bound_ref4(const unsigned char *hi, const prep *p) {
    double U = 0;
    for (int s = 0; s < NS4; s++) {
        const unsigned char *ph = hi + 112 * s;
        uint16_t dh, mh;
        memcpy(&dh, ph, 2);
        memcpy(&mh, ph + 2, 2);
        double d = tr_half_to_float(dh), dmin = tr_half_to_float(mh);
        for (int j = 0; j < 8; j++) {
            int sc, m, sb = 8 * s + j;
            scale_min4(ph + 4, j, &sc, &m);
            long I = 0;
            for (int i = 0; i < 32; i++) {
                int k = 32 * j + i, half = k / 128;
                int hv = (((ph[16 + 32 * half + i] >> (2 * (j % 4))) & 3) << 1) | ((ph[80 + k / 8] >> (k % 8)) & 1);
                I += (long)hv * xv(p, 256 * s + k);
            }
            U += d * sc * ((double)p->a4[sb] * (double)I + p->be4[sb]) + fabs(d) * sc * p->g4[sb] -
                 dmin * m * p->H4[sb] + fabs(dmin) * m * p->mu4[sb];
        }
    }
    return U;
}

#if HAVE_X86

#define TGT __attribute__((target("avx2,fma,f16c")))

TGT static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* sum_k u_k X_k over 32 lanes as 8 int32 lanes: X one int8 digit, or (wide) two: 256 X + Xl */
TGT static inline __attribute__((always_inline)) __m256i dig(__m256i u, const prep *p, int off, int wide) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(u, _mm256_loadu_si256((const __m256i *)(p->X + off))),
                                  wide ? _mm256_set1_epi16(256) : ones);
    if (wide)
        s = _mm256_add_epi32(
            s, _mm256_madd_epi16(_mm256_maddubs_epi16(u, _mm256_loadu_si256((const __m256i *)(p->Xl + off))), ones));
    return s;
}

/* Q8_0's bound of one row from its high plane: two chains of blocks, the scales' terms 8 blocks a vector */
TGT static inline __attribute__((always_inline)) float bound8_body(const unsigned char *hi, const prep *p, int wide) {
    const __m256i m4 = _mm256_set1_epi8(15);
    __m256 side = _mm256_setzero_ps();
    float coef[NB8];
    for (int b = 0; b < NB8; b += 8) {
        __m256 d = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(hi + 2 * b)));
        __m256 ad = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), d);
        side = _mm256_fmadd_ps(d, _mm256_loadu_ps(p->be8 + b), side);
        side = _mm256_fmadd_ps(ad, _mm256_loadu_ps(p->g8 + b), side);
        _mm256_storeu_ps(coef + b, _mm256_mul_ps(d, _mm256_loadu_ps(p->a8 + b)));
    }
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    const unsigned char *nib = hi + 2 * NB8;
    for (int b = 0; b < NB8; b += 2) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(nib + 16 * b));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(nib + 16 * b + 16));
        __m256i u0 = _mm256_and_si256(_mm256_insertf128_si256(_mm256_castsi128_si256(v0), _mm_srli_epi16(v0, 4), 1), m4);
        __m256i u1 = _mm256_and_si256(_mm256_insertf128_si256(_mm256_castsi128_si256(v1), _mm_srli_epi16(v1, 4), 1), m4);
        __m256i s0 = dig(u0, p, 32 * b, wide);
        __m256i s1 = dig(u1, p, 32 * b + 32, wide);
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s0), _mm256_set1_ps(coef[b]), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_set1_ps(coef[b + 1]), acc1);
    }
    return hsum8(_mm256_add_ps(_mm256_add_ps(acc0, acc1), side));
}
TGT static float bound8_n(const unsigned char *hi, const prep *p) { return bound8_body(hi, p, 0); }
TGT static float bound8_w(const unsigned char *hi, const prep *p) { return bound8_body(hi, p, 1); }
static float bound8_avx2(const unsigned char *hi, const prep *p) { return p->wide ? bound8_w(hi, p) : bound8_n(hi, p); }

/* Q4_K's bound of one row from its two-bit and one-bit planes */
TGT static inline __attribute__((always_inline)) float bound4_body(const unsigned char *hi, const prep *p, int wide) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i bsel = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    const __m256i bmask = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    __m256 side = _mm256_setzero_ps(), acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    float coef[NB8];
    /* first every super-block's scales (a coefficient read right after its store would wait for it, #180) */
    for (int s = 0; s < NS4; s++) {
        const unsigned char *ph = hi + 112 * s;
        uint32_t u0, u1, u2, d2; /* ggml's unpacking of the twelve scale bytes, in registers */
        memcpy(&d2, ph, 4);
        memcpy(&u0, ph + 4, 4);
        memcpy(&u1, ph + 8, 4);
        memcpy(&u2, ph + 12, 4);
        const uint32_t m1 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
        const uint32_t m0 = u1 & 0x3f3f3f3fu;
        const uint32_t s1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
        const uint32_t s0 = u0 & 0x3f3f3f3fu;
        __m128i sm = _mm_setr_epi32((int)s0, (int)s1, (int)m0, (int)m1);
        __m256 vsc = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sm));
        __m256 vm = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(sm, 8)));
        __m128 dd = _mm_cvtph_ps(_mm_cvtsi32_si128((int)d2));
        __m256 vd = _mm256_broadcastss_ps(dd), vdm = _mm256_broadcastss_ps(_mm_movehdup_ps(dd));
        __m256 vad = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vd), vadm = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vdm);
        const float *a = p->a4 + 8 * s, *be = p->be4 + 8 * s, *g = p->g4 + 8 * s, *H = p->H4 + 8 * s, *mu = p->mu4 + 8 * s;
        __m256 dsc = _mm256_mul_ps(vd, vsc);
        side = _mm256_fmadd_ps(dsc, _mm256_loadu_ps(be), side);
        side = _mm256_fmadd_ps(_mm256_mul_ps(vad, vsc), _mm256_loadu_ps(g), side);
        side = _mm256_fnmadd_ps(_mm256_mul_ps(vdm, vm), _mm256_loadu_ps(H), side);
        side = _mm256_fmadd_ps(_mm256_mul_ps(vadm, vm), _mm256_loadu_ps(mu), side);
        _mm256_storeu_ps(coef + 8 * s, _mm256_mul_ps(dsc, _mm256_loadu_ps(a)));
    }
    for (int s = 0; s < NS4; s++) {
        const unsigned char *ph = hi + 112 * s;
        for (int half = 0; half < 2; half++) {
            const __m256i two = _mm256_loadu_si256((const __m256i *)(ph + 16 + 32 * half));
            __m256i t[4] = {_mm256_and_si256(two, m3), _mm256_and_si256(_mm256_srli_epi16(two, 2), m3),
                            _mm256_and_si256(_mm256_srli_epi16(two, 4), m3), _mm256_and_si256(_mm256_srli_epi16(two, 6), m3)};
            for (int jj = 0; jj < 4; jj += 2) {
                const int j = 4 * half + jj;
                __m256i b0 = _mm256_set1_epi32(*(const int32_t *)(const void *)(ph + 80 + 4 * j));
                __m256i b1 = _mm256_set1_epi32(*(const int32_t *)(const void *)(ph + 84 + 4 * j));
                b0 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(b0, bsel), bmask), bmask);
                b1 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(b1, bsel), bmask), bmask);
                __m256i h0 = _mm256_sub_epi8(_mm256_add_epi8(t[jj], t[jj]), b0); /* 2t + bit: a set bit's cmpeq is -1 */
                __m256i h1 = _mm256_sub_epi8(_mm256_add_epi8(t[jj + 1], t[jj + 1]), b1);
                __m256i s0 = dig(h0, p, 256 * s + 32 * j, wide);
                __m256i s1 = dig(h1, p, 256 * s + 32 * j + 32, wide);
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s0), _mm256_set1_ps(coef[8 * s + j]), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_set1_ps(coef[8 * s + j + 1]), acc1);
            }
        }
    }
    return hsum8(_mm256_add_ps(_mm256_add_ps(acc0, acc1), side));
}
TGT static float bound4_n(const unsigned char *hi, const prep *p) { return bound4_body(hi, p, 0); }
TGT static float bound4_w(const unsigned char *hi, const prep *p) { return bound4_body(hi, p, 1); }
static float bound4_avx2(const unsigned char *hi, const prep *p) { return p->wide ? bound4_w(hi, p) : bound4_n(hi, p); }

/* ---- the same bounds with AVX-512 (BW, VNNI): two blocks a vector, vpdpbusd for the codes against h, a bit plane
 * to bytes by vpmovm2b. The float sums run in another order than AVX2's: each kernel's bound is rigorous on its
 * own (MARGIN covers both orders), they need not be equal ------------------------------------------------------ */

#define TGT5 __attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")))

/* sum_k u_k X_k for two blocks of 32 at once (16 int32 lanes, 8 a block): one digit, or (wide) 256 X + Xl */
TGT5 static inline __attribute__((always_inline)) __m512i dig5(__m512i u, const prep *p, int off, int wide) {
    __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, _mm512_loadu_si512((const void *)(p->X + off)));
    if (wide)
        s = _mm512_add_epi32(_mm512_slli_epi32(s, 8),
                             _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, _mm512_loadu_si512((const void *)(p->Xl + off))));
    return s;
}

/* [coef[j] x 8 | coef[j + 1] x 8] */
TGT5 static inline __attribute__((always_inline)) __m512 pair5(const float *coef, int j) {
    return _mm512_mask_broadcastss_ps(_mm512_set1_ps(coef[j]), 0xFF00, _mm_load_ss(coef + j + 1));
}

TGT5 static inline __attribute__((always_inline)) float bound8_body5(const unsigned char *hi, const prep *p, int wide) {
    const __m512i m4 = _mm512_set1_epi8(15);
    __m256 side = _mm256_setzero_ps();
    float coef[NB8];
    for (int b = 0; b < NB8; b += 8) {
        __m256 d = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(hi + 2 * b)));
        __m256 ad = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), d);
        side = _mm256_fmadd_ps(d, _mm256_loadu_ps(p->be8 + b), side);
        side = _mm256_fmadd_ps(ad, _mm256_loadu_ps(p->g8 + b), side);
        _mm256_storeu_ps(coef + b, _mm256_mul_ps(d, _mm256_loadu_ps(p->a8 + b)));
    }
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    const unsigned char *nib = hi + 2 * NB8;
    for (int b = 0; b < NB8; b += 4) {
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(nib + 16 * b));      /* blocks b, b + 1 */
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(nib + 16 * b + 32)); /* blocks b + 2, b + 3 */
        /* [b lo, b + 1 lo, b hi, b + 1 hi] -> [b lo, b hi, b + 1 lo, b + 1 hi] */
        __m512i u0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), _mm256_srli_epi16(v0, 4), 1);
        __m512i u1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), _mm256_srli_epi16(v1, 4), 1);
        u0 = _mm512_and_si512(_mm512_shuffle_i64x2(u0, u0, _MM_SHUFFLE(3, 1, 2, 0)), m4);
        u1 = _mm512_and_si512(_mm512_shuffle_i64x2(u1, u1, _MM_SHUFFLE(3, 1, 2, 0)), m4);
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dig5(u0, p, 32 * b, wide)), pair5(coef, b), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dig5(u1, p, 32 * b + 64, wide)), pair5(coef, b + 2), acc1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) + hsum8(side);
}
TGT5 static float bound8_n5(const unsigned char *hi, const prep *p) { return bound8_body5(hi, p, 0); }
TGT5 static float bound8_w5(const unsigned char *hi, const prep *p) { return bound8_body5(hi, p, 1); }

TGT5 static inline __attribute__((always_inline)) float bound4_body5(const unsigned char *hi, const prep *p, int wide) {
    const __m512i m3 = _mm512_set1_epi8(3);
    __m256 side = _mm256_setzero_ps();
    float coef[NB8];
    for (int s = 0; s < NS4; s++) { /* the scales, as the AVX2 kernel */
        const unsigned char *ph = hi + 112 * s;
        uint32_t u0, u1, u2, d2;
        memcpy(&d2, ph, 4);
        memcpy(&u0, ph + 4, 4);
        memcpy(&u1, ph + 8, 4);
        memcpy(&u2, ph + 12, 4);
        const uint32_t m1 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
        const uint32_t m0 = u1 & 0x3f3f3f3fu;
        const uint32_t s1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
        const uint32_t s0 = u0 & 0x3f3f3f3fu;
        __m128i sm = _mm_setr_epi32((int)s0, (int)s1, (int)m0, (int)m1);
        __m256 vsc = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sm));
        __m256 vm = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(sm, 8)));
        __m128 dd = _mm_cvtph_ps(_mm_cvtsi32_si128((int)d2));
        __m256 vd = _mm256_broadcastss_ps(dd), vdm = _mm256_broadcastss_ps(_mm_movehdup_ps(dd));
        __m256 vad = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vd), vadm = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vdm);
        const float *a = p->a4 + 8 * s, *be = p->be4 + 8 * s, *g = p->g4 + 8 * s, *H = p->H4 + 8 * s, *mu = p->mu4 + 8 * s;
        __m256 dsc = _mm256_mul_ps(vd, vsc);
        side = _mm256_fmadd_ps(dsc, _mm256_loadu_ps(be), side);
        side = _mm256_fmadd_ps(_mm256_mul_ps(vad, vsc), _mm256_loadu_ps(g), side);
        side = _mm256_fnmadd_ps(_mm256_mul_ps(vdm, vm), _mm256_loadu_ps(H), side);
        side = _mm256_fmadd_ps(_mm256_mul_ps(vadm, vm), _mm256_loadu_ps(mu), side);
        _mm256_storeu_ps(coef + 8 * s, _mm256_mul_ps(dsc, _mm256_loadu_ps(a)));
    }
    const __m512i sh01 = _mm512_setr_epi64(0, 0, 0, 0, 2, 2, 2, 2), sh23 = _mm512_setr_epi64(4, 4, 4, 4, 6, 6, 6, 6);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int s = 0; s < NS4; s++) {
        const unsigned char *ph = hi + 112 * s;
        for (int half = 0; half < 2; half++) {
            const __m512i two = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(ph + 16 + 32 * half)));
            const int j = 4 * half;
            uint64_t b01, b23;
            memcpy(&b01, ph + 80 + 4 * j, 8);
            memcpy(&b23, ph + 88 + 4 * j, 8);
            __m512i t01 = _mm512_and_si512(_mm512_srlv_epi64(two, sh01), m3); /* sub-blocks j, j + 1 */
            __m512i t23 = _mm512_and_si512(_mm512_srlv_epi64(two, sh23), m3); /* sub-blocks j + 2, j + 3 */
            __m512i h01 = _mm512_sub_epi8(_mm512_add_epi8(t01, t01), _mm512_movm_epi8((__mmask64)b01));
            __m512i h23 = _mm512_sub_epi8(_mm512_add_epi8(t23, t23), _mm512_movm_epi8((__mmask64)b23));
            acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dig5(h01, p, 256 * s + 32 * j, wide)), pair5(coef, 8 * s + j), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dig5(h23, p, 256 * s + 32 * j + 64, wide)),
                                   pair5(coef, 8 * s + j + 2), acc1);
        }
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) + hsum8(side);
}
TGT5 static float bound4_n5(const unsigned char *hi, const prep *p) { return bound4_body5(hi, p, 0); }
TGT5 static float bound4_w5(const unsigned char *hi, const prep *p) { return bound4_body5(hi, p, 1); }

static int g_avx512; /* the AVX-512 kernels in the passes (set by --avx512 on a CPU that has them) */
static float bound8_k(const unsigned char *hi, const prep *p) {
    if (g_avx512) return p->wide ? bound8_w5(hi, p) : bound8_n5(hi, p);
    return bound8_avx2(hi, p);
}
static float bound4_k(const unsigned char *hi, const prep *p) {
    if (g_avx512) return p->wide ? bound4_w5(hi, p) : bound4_n5(hi, p);
    return bound4_avx2(hi, p);
}

/* ---- the passes over the pool ------------------------------------------------------------------------------ */

/* the original rows rebuilt from the two planes, 16 bytes of codes at a time (the scalar rebuild8/rebuild4 above are
 * their definition: the check compares them) */
TGT static void rebuild8_avx2(const unsigned char *hi, const unsigned char *lo, unsigned char *row) {
    const __m128i m4 = _mm_set1_epi8(15), x80 = _mm_set1_epi8((char)0x80);
    for (int b = 0; b < NB8; b++) {
        unsigned char *blk = row + 34 * b;
        memcpy(blk, hi + 2 * b, 2);
        __m128i h = _mm_loadu_si128((const __m128i *)(hi + 2 * NB8 + 16 * b));
        __m128i l = _mm_loadu_si128((const __m128i *)(lo + 16 * b));
        __m128i q0 = _mm_or_si128(_mm_xor_si128(_mm_slli_epi16(_mm_and_si128(h, m4), 4), x80), _mm_and_si128(l, m4));
        __m128i q1 = _mm_or_si128(_mm_xor_si128(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), m4), 4), x80),
                                  _mm_and_si128(_mm_srli_epi16(l, 4), m4));
        _mm_storeu_si128((__m128i *)(blk + 2), q0);
        _mm_storeu_si128((__m128i *)(blk + 18), q1);
    }
}

TGT static void rebuild4_avx2(const unsigned char *hi, const unsigned char *lo, unsigned char *row) {
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    const __m256i bsel = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    const __m256i bmask = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    for (int s = 0; s < NS4; s++) {
        const unsigned char *ph = hi + 112 * s, *pl = lo + 32 * s;
        unsigned char *sb = row + 144 * s;
        memcpy(sb, ph, 16);
        for (int half = 0; half < 2; half++) {
            const __m256i two = _mm256_loadu_si256((const __m256i *)(ph + 16 + 32 * half));
            __m256i q[4];
            for (int jj = 0; jj < 4; jj++) {
                const int j = 4 * half + jj;
                __m256i t = _mm256_and_si256(_mm256_srli_epi16(two, 2 * jj), m3);
                __m256i b1 = _mm256_set1_epi32(*(const int32_t *)(const void *)(ph + 80 + 4 * j));
                __m256i b0 = _mm256_set1_epi32(*(const int32_t *)(const void *)(pl + 4 * j));
                b1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(b1, bsel), bmask), bmask), one);
                b0 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(b0, bsel), bmask), bmask), one);
                q[jj] = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi16(t, 2), _mm256_add_epi8(b1, b1)), b0);
            }
            for (int cc = 0; cc < 2; cc++) /* chunk 2 half + cc: sub-block 2c in the low nibbles, 2c + 1 in the high */
                _mm256_storeu_si256((__m256i *)(sb + 16 + 32 * (2 * half + cc)),
                                    _mm256_or_si256(q[2 * cc], _mm256_slli_epi16(q[2 * cc + 1], 4)));
        }
    }
}

#define MAX_W 64
typedef struct {
    int kind; /* 8: Q8_0, 4: Q4_K */
    const unsigned char *hi, *lo, *orig; /* orig != NULL: the exact rows from the original rows, not the planes */
    const prep *p;
    float *U;
    const float *x;
    const void *xq;
    int32_t *ids; /* each worker's candidates, in its own range */
    const tr_kernels *k;
    float th;                                 /* the rows whose bound reaches th are computed exactly */
    float wmax[MAX_W], wbest[MAX_W];          /* each worker's largest bound, its best exact score */
    int32_t widx[MAX_W], wbidx[MAX_W], wn[MAX_W]; /* and their rows, and how many rows it computed */
} pass_ctx;

static void plane_body(void *c_, int64_t b, int64_t e, int w) {
    pass_ctx *c = (pass_ctx *)c_;
    float m = -INFINITY;
    int32_t mi = (int32_t)b;
    for (int64_t r = b; r < e; r++) {
        float u = c->kind == 8 ? bound8_k(c->hi + (size_t)r * P8_HI, c->p) : bound4_k(c->hi + (size_t)r * P4_HI, c->p);
        c->U[r] = u;
        if (u > m) {
            m = u;
            mi = (int32_t)r;
        }
    }
    if (m > c->wmax[w]) { /* a worker may run more than one chunk: its rows ascend, so a tie keeps the first */
        c->wmax[w] = m;
        c->widx[w] = mi;
    }
}

/* one row's exact score, as the engine's kernel computes it (Q4_K: the decode's two-row road, rows a and b) */
static void exact2(const pass_ctx *c, int32_t a, int32_t b, float *out) {
    unsigned char t0[NS4 * 144 > NB8 * 34 ? NS4 * 144 : NB8 * 34], t1[NS4 * 144];
    if (c->kind == 8) {
        const unsigned char *row = c->orig ? c->orig + (size_t)a * NB8 * 34 : t0;
        if (!c->orig) rebuild8_avx2(c->hi + (size_t)a * P8_HI, c->lo + (size_t)a * P8_LO, t0);
        out[0] = c->k->dot_row[TR_TYPE_Q8_0](row, c->x, COLS);
        return;
    }
    const unsigned char *ra = c->orig ? c->orig + (size_t)a * NS4 * 144 : t0;
    const unsigned char *rb = c->orig ? c->orig + (size_t)b * NS4 * 144 : t1;
    if (!c->orig) {
        rebuild4_avx2(c->hi + (size_t)a * P4_HI, c->lo + (size_t)a * P4_LO, t0);
        rebuild4_avx2(c->hi + (size_t)b * P4_HI, c->lo + (size_t)b * P4_LO, t1);
    }
    c->k->q4x_dot2(ra, rb, c->xq, COLS, out);
}

/* the second region: each worker collects its rows whose bound reaches th, computes them, keeps its best */
static void pick_body(void *c_, int64_t b, int64_t e, int w) {
    pass_ctx *c = (pass_ctx *)c_;
    int32_t n = 0, *ids = c->ids + b;
    for (int64_t r = b; r < e; r++)
        if (c->U[r] >= c->th) ids[n++] = (int32_t)r;
    float best = -INFINITY;
    int32_t bi = -1;
    const int step = c->kind == 8 ? 1 : 2; /* Q4_K's road computes two rows a call */
    for (int32_t i = 0; i < n; i += step) {
        float o[2];
        const int two = step == 2 && i + 1 < n;
        exact2(c, ids[i], ids[two ? i + 1 : i], o);
        for (int t = 0; t <= two; t++)
            if (o[t] > best) {
                best = o[t];
                bi = ids[i + t];
            }
    }
    if (bi >= 0 && (best > c->wbest[w] || (best == c->wbest[w] && bi < c->wbidx[w]))) {
        c->wbest[w] = best;
        c->wbidx[w] = bi;
    }
    c->wn[w] += n;
}

static void empty_body(void *c_, int64_t b, int64_t e, int w) {}

/* the whole argmax by the bound: the bounds and each worker's largest, the best bound's row exactly, then the rows
 * whose bound reaches th (forced = 0: that row's score, the real rule; else the given th), the best of them (the
 * lowest row on a tie). Returns the row; *n_exact the rows computed in the second region. */
static int32_t argmax_by_bound(tr_pool *pool, pass_ctx *c, int T, int forced, float th, int *n_exact) {
    int nw = T < MAX_W ? T : MAX_W;
    for (int w = 0; w < nw; w++) c->wmax[w] = -INFINITY, c->widx[w] = 0;
    tr_parallel_for(pool, VOCAB, 64, plane_body, c);
    int32_t top = c->widx[0];
    float tm = c->wmax[0];
    for (int w = 1; w < nw; w++)
        if (c->wmax[w] > tm) {
            tm = c->wmax[w];
            top = c->widx[w];
        }
    float o[2];
    exact2(c, top, top, o);
    c->th = forced ? th : o[0];
    for (int w = 0; w < nw; w++) c->wn[w] = 0, c->wbest[w] = -INFINITY, c->wbidx[w] = -1;
    tr_parallel_for(pool, VOCAB, 64, pick_body, c);
    float best = o[0];
    int32_t bi = top;
    *n_exact = 0;
    for (int w = 0; w < nw; w++) {
        *n_exact += c->wn[w];
        if (c->wbidx[w] >= 0 && (c->wbest[w] > best || (c->wbest[w] == best && c->wbidx[w] < bi))) {
            best = c->wbest[w];
            bi = c->wbidx[w];
        }
    }
    return bi;
}

#endif /* HAVE_X86 */

int main(int argc, char **argv) {
    int runs = 11;
    int threads[8] = {1, 4, 8, 16}, n_threads = 4;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--avx512") == 0) {
            g_avx512 = 1;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = 0;
            for (char *t = strtok(argv[++i], ","); t != NULL && n_threads < 8; t = strtok(NULL, ","))
                threads[n_threads++] = atoi(t);
        } else {
            fprintf(stderr, "usage: bench_head_bound [--runs N] [--threads 1,4,8,16]\n");
            return 2;
        }
    }
    if (runs < 3) runs = 3;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
#if !HAVE_X86
    printf("bench_head_bound: x86 only\n");
    return 0;
#else
    tr_kernels_init();
    const tr_kernels *k = tr_kernels_get();
    const tr_cpu_info *ci = tr_cpu();
    if (!ci->avx2 || !ci->fma || !ci->f16c) {
        printf("bench_head_bound: the plane kernels need AVX2, FMA and F16C\n");
        return 0;
    }
    if (g_avx512 && !(ci->avx512f && ci->avx512bw && ci->avx512vl && ci->avx512dq && ci->avx512vnni)) {
        printf("bench_head_bound: --avx512 needs AVX-512 F, BW, VL, DQ and VNNI (the engine's tier may be capped)\n");
        return 0;
    }
    printf("plane kernels: %s\n", g_avx512 ? "AVX-512 VNNI" : "AVX2");
    const size_t rb8 = (size_t)NB8 * 34, rb4 = (size_t)NS4 * 144;
    unsigned char *o8 = tr_alloc_aligned(NCOPY * (size_t)VOCAB * rb8, 64), *o4 = tr_alloc_aligned(NCOPY * (size_t)VOCAB * rb4, 64);
    unsigned char *h8 = tr_alloc_aligned(NCOPY * (size_t)VOCAB * P8_HI, 64), *l8 = tr_alloc_aligned(NCOPY * (size_t)VOCAB * P8_LO, 64);
    unsigned char *h4 = tr_alloc_aligned(NCOPY * (size_t)VOCAB * P4_HI, 64), *l4 = tr_alloc_aligned(NCOPY * (size_t)VOCAB * P4_LO, 64);
    g_flush = tr_alloc_aligned(FLUSH_BYTES, 64);
    float *x = tr_alloc_aligned(COLS * sizeof(float), 64), *y = tr_alloc_aligned(VOCAB * sizeof(float), 64);
    float *U = tr_alloc_aligned(VOCAB * sizeof(float), 64), *out = tr_alloc_aligned(VOCAB * sizeof(float), 64);
    int32_t *ids = tr_alloc_aligned(VOCAB * sizeof(int32_t), 64);
    void *xq = tr_alloc_aligned(tr_q4x_bytes(COLS), 64);
    prep *p = tr_alloc_aligned(sizeof(prep), 64), *pw = tr_alloc_aligned(sizeof(prep), 64);
    if (!o8 || !o4 || !h8 || !l8 || !h4 || !l4 || !g_flush || !x || !y || !U || !out || !ids || !xq || !p || !pw) {
        fprintf(stderr, "bench_head_bound: out of memory\n");
        return 1;
    }
    memset(g_flush, 1, FLUSH_BYTES);
    /* the heads: Q8_0 scales near 2^-9 (one row in 8 negative), random codes; Q4_K's d and dmin near 2^-10 */
    for (size_t r = 0; r < VOCAB; r++) {
        for (int b = 0; b < NB8; b++) {
            unsigned char *blk = o8 + r * rb8 + 34 * (size_t)b;
            uint16_t dh = (uint16_t)((6u << 10) | (rnd() & 1023u) | ((rnd() & 7u) == 0 ? 0x8000u : 0u));
            memcpy(blk, &dh, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (unsigned char)(int8_t)(rnd_normal() * 40.0f);
        }
        for (int s = 0; s < NS4; s++) {
            unsigned char *sb = o4 + r * rb4 + 144 * (size_t)s;
            uint16_t dh = (uint16_t)((5u << 10) | (rnd() & 1023u)), mh = (uint16_t)((4u << 10) | (rnd() & 1023u));
            memcpy(sb, &dh, 2);
            memcpy(sb + 2, &mh, 2);
            for (int i = 4; i < 144; i++) sb[i] = (unsigned char)rnd();
        }
        planes8(o8 + r * rb8, h8 + r * P8_HI, l8 + r * P8_LO);
        planes4(o4 + r * rb4, h4 + r * P4_HI, l4 + r * P4_LO);
    }
    for (int i = 0; i < COLS; i++) x[i] = rnd_normal() * ((rnd() & 63u) == 0 ? 8.0f : 1.0f);
    { /* one row aligned with h in each head: its score stands out, so the bound can drop rows (the check's argmax) */
        const size_t r = 12345;
        for (int b = 0; b < NB8; b++) {
            unsigned char *blk = o8 + r * rb8 + 34 * (size_t)b;
            blk[1] &= 0x7F;
            for (int i = 0; i < 32; i++) blk[2 + i] = (unsigned char)(int8_t)(x[32 * b + i] > 0 ? 100 : -100);
        }
        for (int s = 0; s < NS4; s++) {
            unsigned char *sb = o4 + r * rb4 + 144 * (size_t)s;
            const unsigned char sc12[12] = {63, 63, 63, 63, 0, 0, 0, 0, 15, 15, 15, 15};
            memcpy(sb + 4, sc12, 12);
            for (int cc = 0; cc < 4; cc++)
                for (int i = 0; i < 32; i++)
                    sb[16 + 32 * cc + i] = (unsigned char)((x[256 * s + 64 * cc + i] > 0 ? 15 : 0) |
                                                           ((x[256 * s + 64 * cc + 32 + i] > 0 ? 15 : 0) << 4));
        }
        planes8(o8 + r * rb8, h8 + r * P8_HI, l8 + r * P8_LO);
        planes4(o4 + r * rb4, h4 + r * P4_HI, l4 + r * P4_LO);
        /* and its copy at a lower row: a tie of bounds and of scores, which the lowest row wins */
        memcpy(o8 + 777 * rb8, o8 + r * rb8, rb8);
        memcpy(o4 + 777 * rb4, o4 + r * rb4, rb4);
        planes8(o8 + 777 * rb8, h8 + 777 * P8_HI, l8 + 777 * P8_LO);
        planes4(o4 + 777 * rb4, h4 + 777 * P4_HI, l4 + 777 * P4_LO);
    }
    for (size_t cp = 1; cp < NCOPY; cp++) { /* the copies a timed run rotates through */
        memcpy(CP(o8, rb8, cp), o8, (size_t)VOCAB * rb8);
        memcpy(CP(o4, rb4, cp), o4, (size_t)VOCAB * rb4);
        memcpy(CP(h8, P8_HI, cp), h8, (size_t)VOCAB * P8_HI);
        memcpy(CP(l8, P8_LO, cp), l8, (size_t)VOCAB * P8_LO);
        memcpy(CP(h4, P4_HI, cp), h4, (size_t)VOCAB * P4_HI);
        memcpy(CP(l4, P4_LO, cp), l4, (size_t)VOCAB * P4_LO);
    }
    prep_build(x, p, 0);
    prep_build(x, pw, 1);
    const prep *pv[2] = {p, pw};
    k->q4x_prep(x, COLS, xq);

    /* the planes rebuild the rows byte for byte, and every bound is above the engine's value of its row */
    {
        unsigned char tmp[NB8 * 34 > NS4 * 144 ? NB8 * 34 : NS4 * 144];
        int bad = 0;
        double worst8 = INFINITY, worst4 = INFINITY, dev = 0;
        for (size_t r = 0; r < VOCAB; r += 97) {
            rebuild8(h8 + r * P8_HI, l8 + r * P8_LO, tmp);
            bad += memcmp(tmp, o8 + r * rb8, rb8) != 0;
            rebuild4(h4 + r * P4_HI, l4 + r * P4_LO, tmp);
            bad += memcmp(tmp, o4 + r * rb4, rb4) != 0;
            rebuild8_avx2(h8 + r * P8_HI, l8 + r * P8_LO, tmp);
            bad += memcmp(tmp, o8 + r * rb8, rb8) != 0;
            rebuild4_avx2(h4 + r * P4_HI, l4 + r * P4_LO, tmp);
            bad += memcmp(tmp, o4 + r * rb4, rb4) != 0;
        }
        tr_pool *pool = tr_pool_create(16);
        tr_pm_scratch pm;
        if (pool == NULL || tr_pm_scratch_init(&pm, 16, 1, 4, COLS, 4 * COLS) != 0) return 1;
        for (int tw = 0; tw < 4; tw++) {
            const int t = tw / 2;
            const prep *q = pv[tw % 2];
            tr_mat w = {t == 0 ? TR_TYPE_Q8_0 : TR_TYPE_Q4_K, VOCAB, COLS, t == 0 ? o8 : o4};
            tr_matmul_s(pool, &w, x, 1, y, &pm);
            for (size_t r = 0; r < VOCAB; r++) {
                float b = t == 0 ? bound8_k(h8 + r * P8_HI, q) : bound4_k(h4 + r * P4_HI, q);
                double ref = t == 0 ? bound_ref8(h8 + r * P8_HI, q) : bound_ref4(h4 + r * P4_HI, q);
                double gap = (double)b - (double)y[r];
                if (t == 0) worst8 = fmin(worst8, gap);
                else worst4 = fmin(worst4, gap);
                double rel = fabs((double)b - ref) / (1.0 + fabs(ref));
                if (rel > dev) dev = rel;
                g_checked++;
            }
            /* the whole argmax by the bound against the engine's (the lowest row on a tie), from the planes */
            int32_t want = 0;
            for (int32_t r = 1; r < VOCAB; r++)
                if (y[r] > y[want]) want = r;
            pass_ctx c;
            memset(&c, 0, sizeof c);
            c.kind = t == 0 ? 8 : 4;
            c.hi = t == 0 ? h8 : h4;
            c.lo = t == 0 ? l8 : l4;
            c.p = q;
            c.U = U;
            c.x = x;
            c.xq = xq;
            c.ids = ids;
            c.k = k;
            int n_exact = 0;
            int32_t got = argmax_by_bound(pool, &c, 16, 0, 0, &n_exact);
            if (got != want) bad++;
            g_pruned += VOCAB - n_exact;
            printf("check: %s, h as %s: the argmax by the bound %d, the engine's %d; %d rows computed exactly\n",
                   t == 0 ? "Q8_0" : "Q4_K", tw % 2 ? "two digits" : "int8", (int)got, (int)want, n_exact);
        }
        tr_pm_scratch_free(&pm);
        tr_pool_destroy(pool);
        printf("check: planes rebuild the rows (%s); bound - engine value, the least: Q8_0 %.4g, Q4_K %.4g; kernel "
               "against the double formula %.2g; %lld rows\n",
               bad ? "NO" : "yes", worst8, worst4, dev, (long long)g_checked);
        if (bad || worst8 < 0 || worst4 < 0 || dev > 1e-4 || g_checked == 0 || g_pruned == 0) {
            printf("bench_head_bound: check FAILED\n");
            return 1;
        }
    }

    /* in cache, one thread: the first 256 rows' planes, ns a row and GB/s of plane bytes */
    {
        double v[4][MAX_RUNS], sp;
        for (int run = 0; run < runs; run++) {
            for (int tw = 0; tw < 4; tw++) {
                const int t = tw / 2;
                const prep *q = pv[tw % 2];
                double t0 = tr_time_sec();
                float acc = 0;
                for (int it = 0; it < 64; it++)
                    for (int r = 0; r < 256; r++)
                        acc += t == 0 ? bound8_k(h8 + (size_t)r * P8_HI, q) : bound4_k(h4 + (size_t)r * P4_HI, q);
                double dt = (tr_time_sec() - t0) / (64.0 * 256) * 1e9;
                g_sink += (uint64_t)(acc != 0);
                v[tw][run] = dt;
            }
        }
        for (int tw = 0; tw < 4; tw++) {
            double m = median_spread(v[tw], runs, &sp);
            printf("plane in cache, 1 thread: %s, h as %s: %.1f ns a row (%.1f GB/s, spread %.1f%%)\n",
                   tw < 2 ? "Q8_0" : "Q4_K", tw % 2 ? "two int8 digits" : "int8", m, (tw < 2 ? P8_HI : P4_HI) / m,
                   100 * sp);
        }
    }

    /* from RAM: every piece at every thread count */
    printf("%-22s %7s %10s %8s %8s\n", "piece", "threads", "us", "GB/s", "spread");
    for (int ti = 0; ti < n_threads; ti++) {
        int T = threads[ti];
        tr_pool *pool = tr_pool_create(T);
        tr_pm_scratch pm;
        if (pool == NULL || tr_pm_scratch_init(&pm, T, 1, 4, COLS, 4 * COLS) != 0) return 1;
        const char *names[] = {"engine Q8_0", "plane Q8_0 int8", "plane Q8_0 2 digits", "engine Q4_K", "plane Q4_K int8",
                               "plane Q4_K 2 digits", "region cold", "region hot"};
        const double bytes[] = {(double)VOCAB * rb8, (double)VOCAB * P8_HI, (double)VOCAB * P8_HI, (double)VOCAB * rb4,
                                (double)VOCAB * P4_HI, (double)VOCAB * P4_HI, 0, 0};
        for (int piece = 0; piece < 8; piece++) {
            double v[MAX_RUNS], s;
            const int q8 = piece < 3;
            for (int run = 0; run < runs; run++) {
                flush_caches();
                pass_ctx c;
                memset(&c, 0, sizeof c);
                c.kind = q8 ? 8 : 4;
                c.hi = q8 ? CP(h8, P8_HI, run % NCOPY) : CP(h4, P4_HI, run % NCOPY);
                c.p = pv[piece % 3 == 2];
                c.U = U;
                tr_mat w = {q8 ? TR_TYPE_Q8_0 : TR_TYPE_Q4_K, VOCAB, COLS,
                             q8 ? CP(o8, rb8, run % NCOPY) : CP(o4, rb4, run % NCOPY)};
                if (piece == 7) tr_parallel_for(pool, VOCAB, 64, empty_body, NULL); /* the workers awake */
                double t0 = tr_time_sec();
                if (piece >= 6) tr_parallel_for(pool, VOCAB, 64, empty_body, NULL);
                else if (piece % 3 == 0) tr_matmul_s(pool, &w, x, 1, y, &pm);
                else tr_parallel_for(pool, VOCAB, 64, plane_body, &c);
                v[run] = (tr_time_sec() - t0) * 1e6;
            }
            double m = median_spread(v, runs, &s);
            printf("%-22s %7d %10.1f %8.1f %7.1f%%\n", names[piece], T, m, bytes[piece] > 0 ? bytes[piece] / m / 1e3 : 0,
                   100 * s);
        }
        /* the whole argmax by the bound (h as two digits), the rows left forced to n by the threshold: the n largest
         * bounds of this head; "orig": the exact rows read from the original rows instead of the planes */
        const int counts[] = {0, 100, 1000, 3000, 12000, -1000};
        for (int kind = 8; kind >= 4; kind -= 4) {
            pass_ctx c;
            memset(&c, 0, sizeof c);
            c.kind = kind;
            c.hi = kind == 8 ? h8 : h4;
            c.lo = kind == 8 ? l8 : l4;
            c.p = pw;
            c.U = U;
            c.x = x;
            c.xq = xq;
            c.ids = ids;
            c.k = k;
            int n0 = 0;
            argmax_by_bound(pool, &c, T, 1, INFINITY, &n0);
            float *sorted = (float *)malloc(VOCAB * sizeof(float));
            if (sorted == NULL) return 1;
            memcpy(sorted, U, VOCAB * sizeof(float));
            qsort(sorted, VOCAB, sizeof(float), cmp_float_desc);
            for (int ni = 0; ni < 6; ni++) {
                const int n = counts[ni] < 0 ? -counts[ni] : counts[ni];
                const float th = n == 0 ? INFINITY : sorted[n - 1];
                double v[MAX_RUNS], s;
                int got = 0;
                for (int run = 0; run < runs; run++) {
                    const size_t cp = (size_t)run % NCOPY;
                    c.hi = kind == 8 ? CP(h8, P8_HI, cp) : CP(h4, P4_HI, cp);
                    c.lo = kind == 8 ? CP(l8, P8_LO, cp) : CP(l4, P4_LO, cp);
                    c.orig = counts[ni] < 0 ? (kind == 8 ? CP(o8, rb8, cp) : CP(o4, rb4, cp)) : NULL;
                    flush_caches();
                    double t0 = tr_time_sec();
                    argmax_by_bound(pool, &c, T, 1, th, &got);
                    v[run] = (tr_time_sec() - t0) * 1e6;
                }
                double m = median_spread(v, runs, &s);
                char name[64];
                snprintf(name, sizeof name, "argmax %s n %d%s", kind == 8 ? "Q8_0" : "Q4_K", got, c.orig ? " orig" : "");
                printf("%-22s %7d %10.1f %8.1f %7.1f%%\n", name, T, m,
                       (double)VOCAB * (kind == 8 ? P8_HI : P4_HI) / m / 1e3, 100 * s);
            }
            free(sorted);
        }
        tr_pm_scratch_free(&pm);
        tr_pool_destroy(pool);
    }
    /* the scan: the rows at or above a threshold that keeps ~1000 of them, one thread, the bounds in cache */
    {
        double v[MAX_RUNS], s;
        float th = 0;
        for (int i = 0; i < VOCAB; i++) U[i] = rnd_normal();
        th = 3.09f; /* ~0.1% of a normal's mass above */
        int n = 0;
        for (int run = 0; run < runs; run++) {
            double t0 = tr_time_sec();
            n = 0;
            for (int i = 0; i < VOCAB; i++)
                if (U[i] >= th) ids[n++] = i;
            v[run] = (tr_time_sec() - t0) * 1e6;
        }
        printf("scan of the bounds, 1 thread: %.1f us (%d rows kept, spread %.1f%%)\n", median_spread(v, runs, &s), n,
               100 * s);
    }
    printf("g_checked %lld\n", (long long)g_checked);
    return g_checked > 0 ? 0 : 1;
#endif
}
