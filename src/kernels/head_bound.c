/* head_bound.c — the output head's argmax by a bound (kernels.h §The output head by a bound; docs/MEASUREMENTS.md
 * §The head's argmax by a bound, question 73). Born in tests/bench_head_bound.c, phase 1: the planes' layout, the
 * token's prep and the AVX2 and AVX-512 plane kernels. Here the float order is fixed once, so every tier gives the
 * scalar definition's bits, and every multiply-add is a multiply then an add (kernels.h's numerics contract). */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"
#include "kernels_internal.h"
#include "../base/platform.h"

#define HB_LIM 32639                /* 127 * 257: h's 16-bit code, its high digit within int8 */
#define HB_MARGIN 1.52587890625e-05 /* 2^-16: every float sum's rounding here and in the engine is below it */
#define HB_HMAX 0x1p64f             /* a larger |h| leaves the bound to the full head: no float of a row overflows */
#define HB_WORKER_BYTES 64

/* ---- sizes ---------------------------------------------------------------------------------------------------- */

int tr_hb_supports(tr_type type, int64_t n) {
    return (type == TR_TYPE_Q8_0 || type == TR_TYPE_Q4_K) && n > 0 && n % 256 == 0 && n <= TR_HB_COLS_MAX;
}

size_t tr_hb_hi_bytes(tr_type type, int64_t n) {
    if (!tr_hb_supports(type, n)) return 0;
    return type == TR_TYPE_Q8_0 ? (size_t)(n / 32) * 18 : (size_t)(n / 256) * 112;
}

size_t tr_hb_lo_bytes(tr_type type, int64_t n) {
    if (!tr_hb_supports(type, n)) return 0;
    return type == TR_TYPE_Q8_0 ? (size_t)(n / 32) * 16 : (size_t)(n / 256) * 32;
}

size_t tr_hb_prep_bytes(int64_t n) {
    const size_t b = 2 * (size_t)n + 5 * sizeof(float) * (size_t)(n / 32);
    return (b + 63) / 64 * 64;
}

tr_hb_prep tr_hb_prep_view(void *buf, int64_t n) {
    tr_hb_prep p;
    unsigned char *c = (unsigned char *)buf;
    const size_t nb = (size_t)(n / 32);
    p.X = (int8_t *)c;
    p.Xl = (int8_t *)(c + n);
    float *f = (float *)(void *)(c + 2 * n);
    p.a = f;
    p.be = f + nb;
    p.g = f + 2 * nb;
    p.H = f + 3 * nb;
    p.mu = f + 4 * nb;
    return p;
}

/* ---- the token prepared for the bound ------------------------------------------------------------------------- */

/* hot: begin */
static float hb_up(double v) { /* the float at or above v */
    float f = (float)v;
    return (double)f >= v ? f : nextafterf(f, INFINITY);
}

int tr_hb_prep_build(tr_type type, const float *h, int64_t n, const tr_hb_prep *p) {
    if (!tr_hb_supports(type, n)) return -1;
    const int q4 = type == TR_TYPE_Q4_K;
    double err_el = 0.0; /* Q4_K: the engine's rounding of h, 2^-30 of its super-block's max|h| an element */
    for (int64_t b = 0; b < n / 32; b++) {
        const float *x = h + 32 * b;
        if (q4 && b % 8 == 0) {
            float ms = 0.0f;
            for (int k = 0; k < 256; k++) {
                const float a = fabsf(x[k]);
                if (!(a <= HB_HMAX)) return -1; /* a NaN, an infinity, or past the floats' room */
                if (a > ms) ms = a;
            }
            err_el = (double)ms * 0x1p-30;
        }
        float m = 0.0f;
        for (int k = 0; k < 32; k++) {
            const float a = fabsf(x[k]);
            if (!(a <= HB_HMAX)) return -1;
            if (a > m) m = a;
        }
        /* a block below 2^-64 keeps X = 0: all of it counted as h's rounding */
        const int tiny = m < 0x1p-64f;
        const float delta = tiny ? 1.0f : m / (float)HB_LIM;
        int64_t sx = 0, sax = 0;
        double e = 0.0, ha = 0.0, hs = 0.0;
        for (int k = 0; k < 32; k++) {
            /* the nearest integer, ties to even: adding and taking away 1.5 2^52 (|x / delta| <= 32640, as q4x_prep) */
            long q = tiny ? 0 : (long)(((double)(x[k] / delta) + 6755399441055744.0) - 6755399441055744.0);
            if (q > HB_LIM) q = HB_LIM;
            if (q < -HB_LIM) q = -HB_LIM;
            const long qh = (q + 32896) / 256 - 128; /* floor((q + 128) / 256), the dividend positive */
            p->X[32 * b + k] = (int8_t)qh;
            p->Xl[32 * b + k] = (int8_t)(q - 256 * qh);
            sx += q;
            sax += q < 0 ? -q : q;
            e += fabs((double)x[k] - (double)delta * (double)q);
            ha += fabs((double)x[k]);
            hs += (double)x[k];
        }
        e *= 1.0 + 1e-12;
        const double dsx = (double)delta * (double)sx, dsax = (double)delta * (double)sax;
        if (!q4) {
            /* Q8_0, q = 16 (u - 8) + lo: d (16 delta I - 120.5 delta sumX) + |d| (7.5 delta sum|X| + 128 E) */
            const double g = 7.5 * dsax + 128.0 * e;
            const double mag = 240.0 * dsax + 120.5 * fabs(dsx) + g + 128.0 * ha;
            p->a[b] = 16.0f * delta;
            p->be[b] = (float)(-120.5 * dsx);
            p->g[b] = hb_up(g + HB_MARGIN * (mag + 120.5 * fabs(dsx)));
            p->H[b] = 0.0f;
            p->mu[b] = 0.0f;
        } else {
            /* Q4_K, q = 2 (q >> 1) + bit: d sc (2 delta I + delta sumX / 2) + |d| sc (delta sum|X| / 2 + 15 E), and
             * -dmin m H; the engine's own rounding of h, err_el an element, on both */
            const double g = 0.5 * dsax + 15.0 * e + 480.0 * err_el;
            const double mag = 14.0 * dsax + 0.5 * fabs(dsx) + g + 15.0 * ha;
            p->a[b] = 2.0f * delta;
            p->be[b] = (float)(0.5 * dsx);
            p->g[b] = hb_up(g + HB_MARGIN * mag);
            p->H[b] = (float)hs;
            p->mu[b] = hb_up(HB_MARGIN * 4.0 * ha + fabs(hs - (double)(float)hs) + 32.0 * err_el);
        }
    }
    return 0;
}
/* hot: end */

/* ---- the planes ----------------------------------------------------------------------------------------------- */

void tr_hb_planes(tr_type type, const void *row_, int64_t n, unsigned char *hi, unsigned char *lo) {
    const unsigned char *row = (const unsigned char *)row_;
    if (type == TR_TYPE_Q8_0) {
        const int64_t nb = n / 32;
        for (int64_t b = 0; b < nb; b++) {
            const unsigned char *blk = row + TR_Q8_0_BLOCK_BYTES * b;
            memcpy(hi + 2 * b, blk, 2);
            const int8_t *q = (const int8_t *)(blk + 2);
            for (int i = 0; i < 16; i++) {
                const int c0 = q[i] + 128, c1 = q[i + 16] + 128; /* u = c >> 4 = (q >> 4) + 8, low = c & 15 = q & 15 */
                hi[2 * nb + 16 * b + i] = (unsigned char)((c0 >> 4) | ((c1 >> 4) << 4));
                lo[16 * b + i] = (unsigned char)((c0 & 15) | ((c1 & 15) << 4));
            }
        }
        return;
    }
    for (int64_t s = 0; s < n / 256; s++) {
        const unsigned char *sb = row + TR_Q4_K_BLOCK_BYTES * s;
        unsigned char *ph = hi + 112 * s, *pl = lo + 32 * s;
        memcpy(ph, sb, 16);
        memset(ph + 16, 0, 96);
        memset(pl, 0, 32);
        for (int k = 0; k < 256; k++) {
            const unsigned q = tr_q4_k_quant(sb + TR_Q4_K_QS_OFFSET, k);
            ph[16 + 32 * (k / 128) + k % 32] |= (unsigned char)((q >> 2) << (2 * ((k % 128) / 32)));
            ph[80 + k / 8] |= (unsigned char)(((q >> 1) & 1) << (k % 8));
            pl[k / 8] |= (unsigned char)((q & 1) << (k % 8));
        }
    }
}

/* hot: begin */
static void hb_rebuild_q8_0(const unsigned char *hi, const unsigned char *lo, int64_t n, unsigned char *row) {
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        unsigned char *blk = row + TR_Q8_0_BLOCK_BYTES * b;
        memcpy(blk, hi + 2 * b, 2);
        for (int i = 0; i < 16; i++) {
            const int u = hi[2 * nb + 16 * b + i], l = lo[16 * b + i];
            blk[2 + i] = (unsigned char)(((u & 15) << 4 | (l & 15)) ^ 0x80);
            blk[2 + 16 + i] = (unsigned char)(((u >> 4) << 4 | (l >> 4)) ^ 0x80);
        }
    }
}

static void hb_rebuild_q4_k(const unsigned char *hi, const unsigned char *lo, int64_t n, unsigned char *row) {
    for (int64_t s = 0; s < n / 256; s++) {
        const unsigned char *ph = hi + 112 * s, *pl = lo + 32 * s;
        unsigned char *sb = row + TR_Q4_K_BLOCK_BYTES * s;
        memcpy(sb, ph, 16);
        unsigned char q[256];
        for (int k = 0; k < 256; k++)
            q[k] = (unsigned char)((((ph[16 + 32 * (k / 128) + k % 32] >> (2 * ((k % 128) / 32))) & 3) << 2) |
                                   (((ph[80 + k / 8] >> (k % 8)) & 1) << 1) | ((pl[k / 8] >> (k % 8)) & 1));
        for (int c = 0; c < 4; c++)
            for (int i = 0; i < 32; i++)
                sb[TR_Q4_K_QS_OFFSET + 32 * c + i] = (unsigned char)(q[64 * c + i] | (q[64 * c + 32 + i] << 4));
    }
}

/* ---- the bound, the definition --------------------------------------------------------------------------------
 * A block's I as eight int32 lanes of four elements (exact: |P| < 2^21), lane l of block b into chain b % 4, the
 * side terms of block b into lane b % 8, then the fixed fold below. */

static float hb_fold(float acc[4][8], const float side[8]) {
    float v[8];
    for (int l = 0; l < 8; l++) v[l] = ((acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l])) + side[l];
    return ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
}

static void hb_bounds_q8_0(const unsigned char *hi, int64_t rows, const tr_hb_prep *p, int64_t n, float *U) {
    const int64_t nb = n / 32;
    const size_t stride = (size_t)nb * 18;
    for (int64_t r = 0; r < rows; r++) {
        const unsigned char *h = hi + (size_t)r * stride, *nib = h + 2 * nb;
        float acc[4][8] = {{0}}, side[8] = {0};
        for (int64_t b = 0; b < nb; b++) {
            uint16_t dh;
            memcpy(&dh, h + 2 * b, 2);
            const float d = tr_half_to_float(dh), ad = fabsf(d), coef = d * p->a[b];
            for (int l = 0; l < 8; l++) {
                int32_t P = 0;
                for (int i = 4 * l; i < 4 * l + 4; i++) {
                    const int v = nib[16 * b + (i & 15)], u = i < 16 ? (v & 15) : (v >> 4);
                    P += u * (256 * p->X[32 * b + i] + p->Xl[32 * b + i]);
                }
                acc[b & 3][l] = acc[b & 3][l] + (float)P * coef;
            }
            side[b & 7] = side[b & 7] + d * p->be[b];
            side[b & 7] = side[b & 7] + ad * p->g[b];
        }
        U[r] = hb_fold(acc, side);
    }
}

static void hb_bounds_q4_k(const unsigned char *hi, int64_t rows, const tr_hb_prep *p, int64_t n, float *U) {
    const int64_t ns = n / 256;
    const size_t stride = (size_t)ns * 112;
    for (int64_t r = 0; r < rows; r++) {
        const unsigned char *h = hi + (size_t)r * stride;
        float acc[4][8] = {{0}}, side[8] = {0};
        for (int64_t s = 0; s < ns; s++) {
            const unsigned char *ph = h + 112 * s;
            int sc[8], m[8];
            tr_q4_k_sc_m(ph, sc, m);
            uint16_t dh, mh;
            memcpy(&dh, ph, 2);
            memcpy(&mh, ph + 2, 2);
            const float d = tr_half_to_float(dh), dmin = tr_half_to_float(mh), ad = fabsf(d), adm = fabsf(dmin);
            for (int j = 0; j < 8; j++) {
                const int64_t sb = 8 * s + j;
                const float fsc = (float)sc[j], fm = (float)m[j], dsc = d * fsc, coef = dsc * p->a[sb];
                for (int l = 0; l < 8; l++) {
                    int32_t P = 0;
                    for (int i = 4 * l; i < 4 * l + 4; i++) {
                        const int k = 32 * j + i;
                        const int hv = (((ph[16 + 32 * (k / 128) + i] >> (2 * (j % 4))) & 3) << 1) |
                                       ((ph[80 + k / 8] >> (k % 8)) & 1);
                        P += hv * (256 * p->X[256 * s + k] + p->Xl[256 * s + k]);
                    }
                    acc[j & 3][l] = acc[j & 3][l] + (float)P * coef;
                }
                side[j] = side[j] + dsc * p->be[sb];
                side[j] = side[j] + (ad * fsc) * p->g[sb];
                side[j] = side[j] - (dmin * fm) * p->H[sb];
                side[j] = side[j] + (adm * fm) * p->mu[sb];
            }
        }
        U[r] = hb_fold(acc, side);
    }
}

void tr_hb_fill_scalar(tr_kernels *k) {
    k->hb_bounds[TR_TYPE_Q8_0] = hb_bounds_q8_0;
    k->hb_bounds[TR_TYPE_Q4_K] = hb_bounds_q4_k;
    k->hb_rebuild[TR_TYPE_Q8_0] = hb_rebuild_q8_0;
    k->hb_rebuild[TR_TYPE_Q4_K] = hb_rebuild_q4_k;
}
/* hot: end */

/* ---- x86: AVX2 + F16C, and AVX-512 (BW, VL, DQ, VNNI) for the bounds ------------------------------------------ */

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#define HB_AVX2 __attribute__((target("avx2,f16c")))
#define HB_AVX512 __attribute__((target("avx2,f16c,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")))

/* hot: begin */
HB_AVX2 static inline float hb_hsum8(__m256 v) { /* ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7)) */
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* sum_k u_k (256 X_k + Xl_k) over 32 bytes as 8 int32 lanes of four */
HB_AVX2 static inline __attribute__((always_inline)) __m256i hb_dig(__m256i u, const int8_t *X, const int8_t *Xl) {
    const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(u, _mm256_loadu_si256((const __m256i *)(const void *)X)),
                                        _mm256_set1_epi16(256));
    return _mm256_add_epi32(
        s, _mm256_madd_epi16(_mm256_maddubs_epi16(u, _mm256_loadu_si256((const __m256i *)(const void *)Xl)),
                             _mm256_set1_epi16(1)));
}

HB_AVX2 static inline __attribute__((always_inline)) __m256 hb_lane(__m256 acc, __m256i P, float coef) {
    return _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(P), _mm256_set1_ps(coef)));
}

/* a Q4_K super-block's side terms into side, its eight coefficients (d sc_j) a_j into coef */
HB_AVX2 static inline __attribute__((always_inline)) __m256 hb_side4(const unsigned char *ph, const tr_hb_prep *p,
                                                                      int64_t sb, __m256 side, float *coef) {
    uint32_t u0, u1, u2, d2; /* ggml's unpacking of the twelve scale bytes, in registers (LESSONS #341) */
    memcpy(&d2, ph, 4);
    memcpy(&u0, ph + 4, 4);
    memcpy(&u1, ph + 8, 4);
    memcpy(&u2, ph + 12, 4);
    const uint32_t m1 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
    const uint32_t m0 = u1 & 0x3f3f3f3fu;
    const uint32_t s1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
    const uint32_t s0 = u0 & 0x3f3f3f3fu;
    const __m128i sm = _mm_setr_epi32((int)s0, (int)s1, (int)m0, (int)m1);
    const __m256 vsc = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sm));
    const __m256 vm = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(sm, 8)));
    const __m128 dd = _mm_cvtph_ps(_mm_cvtsi32_si128((int)d2));
    const __m256 vd = _mm256_broadcastss_ps(dd), vdm = _mm256_broadcastss_ps(_mm_movehdup_ps(dd));
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 vad = _mm256_andnot_ps(sign, vd), vadm = _mm256_andnot_ps(sign, vdm);
    const __m256 dsc = _mm256_mul_ps(vd, vsc);
    side = _mm256_add_ps(side, _mm256_mul_ps(dsc, _mm256_loadu_ps(p->be + sb)));
    side = _mm256_add_ps(side, _mm256_mul_ps(_mm256_mul_ps(vad, vsc), _mm256_loadu_ps(p->g + sb)));
    side = _mm256_sub_ps(side, _mm256_mul_ps(_mm256_mul_ps(vdm, vm), _mm256_loadu_ps(p->H + sb)));
    side = _mm256_add_ps(side, _mm256_mul_ps(_mm256_mul_ps(vadm, vm), _mm256_loadu_ps(p->mu + sb)));
    _mm256_storeu_ps(coef, _mm256_mul_ps(dsc, _mm256_loadu_ps(p->a + sb)));
    return side;
}

/* a Q8_0 row's side terms (8 blocks a vector) and every coefficient d_b a_b first: a coefficient read right after
 * its store waits for it (#180) */
HB_AVX2 static inline __attribute__((always_inline)) __m256 hb_side8(const unsigned char *h, const tr_hb_prep *p,
                                                                      int64_t nb, float *coef) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 side = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b += 8) {
        const __m256 d = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(h + 2 * b)));
        side = _mm256_add_ps(side, _mm256_mul_ps(d, _mm256_loadu_ps(p->be + b)));
        side = _mm256_add_ps(side, _mm256_mul_ps(_mm256_andnot_ps(sign, d), _mm256_loadu_ps(p->g + b)));
        _mm256_storeu_ps(coef + b, _mm256_mul_ps(d, _mm256_loadu_ps(p->a + b)));
    }
    return side;
}

HB_AVX2 static void avx2_hb_bounds_q8_0(const unsigned char *hi, int64_t rows, const tr_hb_prep *p, int64_t n,
                                        float *U) {
    const int64_t nb = n / 32;
    const size_t stride = (size_t)nb * 18;
    const __m256i m4 = _mm256_set1_epi8(15);
    float coef[TR_HB_COLS_MAX / 32];
    for (int64_t r = 0; r < rows; r++) {
        const unsigned char *h = hi + (size_t)r * stride, *nib = h + 2 * nb;
        const __m256 side = hb_side8(h, p, nb, coef);
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (int64_t b = 0; b < nb; b += 4) {
            __m256i u[4];
            for (int i = 0; i < 4; i++) {
                const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(nib + 16 * (b + i)));
                u[i] = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(v), _mm_srli_epi16(v, 4), 1), m4);
            }
            a0 = hb_lane(a0, hb_dig(u[0], p->X + 32 * b, p->Xl + 32 * b), coef[b]);
            a1 = hb_lane(a1, hb_dig(u[1], p->X + 32 * b + 32, p->Xl + 32 * b + 32), coef[b + 1]);
            a2 = hb_lane(a2, hb_dig(u[2], p->X + 32 * b + 64, p->Xl + 32 * b + 64), coef[b + 2]);
            a3 = hb_lane(a3, hb_dig(u[3], p->X + 32 * b + 96, p->Xl + 32 * b + 96), coef[b + 3]);
        }
        U[r] = hb_hsum8(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3)), side));
    }
}

HB_AVX2 static void avx2_hb_bounds_q4_k(const unsigned char *hi, int64_t rows, const tr_hb_prep *p, int64_t n,
                                        float *U) {
    const int64_t ns = n / 256;
    const size_t stride = (size_t)ns * 112;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i bsel = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    const __m256i bmask = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    float coef[TR_HB_COLS_MAX / 32];
    for (int64_t r = 0; r < rows; r++) {
        const unsigned char *h = hi + (size_t)r * stride;
        __m256 side = _mm256_setzero_ps();
        for (int64_t s = 0; s < ns; s++) side = hb_side4(h + 112 * s, p, 8 * s, side, coef + 8 * s);
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int64_t s = 0; s < ns; s++) {
            const unsigned char *ph = h + 112 * s;
            for (int half = 0; half < 2; half++) {
                const __m256i two = _mm256_loadu_si256((const __m256i *)(const void *)(ph + 16 + 32 * half));
                for (int jj = 0; jj < 4; jj++) {
                    const int j = 4 * half + jj;
                    const __m256i t = _mm256_and_si256(_mm256_srli_epi16(two, 2 * jj), m3);
                    int32_t bits;
                    memcpy(&bits, ph + 80 + 4 * j, 4);
                    __m256i bb = _mm256_shuffle_epi8(_mm256_set1_epi32(bits), bsel);
                    bb = _mm256_cmpeq_epi8(_mm256_and_si256(bb, bmask), bmask);
                    const __m256i hv = _mm256_sub_epi8(_mm256_add_epi8(t, t), bb); /* 2t + bit: a set bit is -1 */
                    acc[jj] = hb_lane(acc[jj], hb_dig(hv, p->X + 256 * s + 32 * j, p->Xl + 256 * s + 32 * j),
                                      coef[8 * s + j]);
                }
            }
        }
        U[r] = hb_hsum8(
            _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(acc[0], acc[2]), _mm256_add_ps(acc[1], acc[3])), side));
    }
}

/* two blocks of 32 at once: 16 int32 lanes, 8 a block, sum u (256 X + Xl) */
HB_AVX512 static inline __attribute__((always_inline)) __m512i hb_dig5(__m512i u, const int8_t *X, const int8_t *Xl) {
    const __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, _mm512_loadu_si512((const void *)X));
    return _mm512_add_epi32(_mm512_slli_epi32(s, 8),
                            _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, _mm512_loadu_si512((const void *)Xl)));
}

/* [coef[j] x 8 | coef[j + 1] x 8] */
HB_AVX512 static inline __attribute__((always_inline)) __m512 hb_pair5(const float *coef, int64_t j) {
    return _mm512_mask_broadcastss_ps(_mm512_set1_ps(coef[j]), 0xFF00, _mm_load_ss(coef + j + 1));
}

/* chains [0 | 1] in a0, [2 | 3] in a1: lane l = ((c0 + c2) + (c1 + c3)) + side, then hb_hsum8 */
HB_AVX512 static inline __attribute__((always_inline)) float hb_fold5(__m512 a0, __m512 a1, __m256 side) {
    const __m512 t = _mm512_add_ps(a0, a1);
    const __m256 lo = _mm512_castps512_ps256(t);
    const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(t), 1));
    return hb_hsum8(_mm256_add_ps(_mm256_add_ps(lo, hi), side));
}

HB_AVX512 static void avx512_hb_bounds_q8_0(const unsigned char *hi, int64_t rows, const tr_hb_prep *p, int64_t n,
                                            float *U) {
    const int64_t nb = n / 32;
    const size_t stride = (size_t)nb * 18;
    const __m512i m4 = _mm512_set1_epi8(15);
    float coef[TR_HB_COLS_MAX / 32];
    for (int64_t r = 0; r < rows; r++) {
        const unsigned char *h = hi + (size_t)r * stride, *nib = h + 2 * nb;
        const __m256 side = hb_side8(h, p, nb, coef);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        for (int64_t b = 0; b < nb; b += 4) {
            const __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)(nib + 16 * b));      /* b, b + 1 */
            const __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)(nib + 16 * b + 32)); /* b + 2, b + 3 */
            /* [b lo, b + 1 lo, b hi, b + 1 hi] -> [b lo, b hi, b + 1 lo, b + 1 hi] */
            __m512i u0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), _mm256_srli_epi16(v0, 4), 1);
            __m512i u1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), _mm256_srli_epi16(v1, 4), 1);
            u0 = _mm512_and_si512(_mm512_shuffle_i64x2(u0, u0, _MM_SHUFFLE(3, 1, 2, 0)), m4);
            u1 = _mm512_and_si512(_mm512_shuffle_i64x2(u1, u1, _MM_SHUFFLE(3, 1, 2, 0)), m4);
            a0 = _mm512_add_ps(a0, _mm512_mul_ps(_mm512_cvtepi32_ps(hb_dig5(u0, p->X + 32 * b, p->Xl + 32 * b)),
                                                 hb_pair5(coef, b)));
            a1 = _mm512_add_ps(a1, _mm512_mul_ps(_mm512_cvtepi32_ps(hb_dig5(u1, p->X + 32 * b + 64, p->Xl + 32 * b + 64)),
                                                 hb_pair5(coef, b + 2)));
        }
        U[r] = hb_fold5(a0, a1, side);
    }
}

HB_AVX512 static void avx512_hb_bounds_q4_k(const unsigned char *hi, int64_t rows, const tr_hb_prep *p, int64_t n,
                                            float *U) {
    const int64_t ns = n / 256;
    const size_t stride = (size_t)ns * 112;
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i sh01 = _mm512_setr_epi64(0, 0, 0, 0, 2, 2, 2, 2), sh23 = _mm512_setr_epi64(4, 4, 4, 4, 6, 6, 6, 6);
    float coef[TR_HB_COLS_MAX / 32];
    for (int64_t r = 0; r < rows; r++) {
        const unsigned char *h = hi + (size_t)r * stride;
        __m256 side = _mm256_setzero_ps();
        for (int64_t s = 0; s < ns; s++) side = hb_side4(h + 112 * s, p, 8 * s, side, coef + 8 * s);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        for (int64_t s = 0; s < ns; s++) {
            const unsigned char *ph = h + 112 * s;
            for (int half = 0; half < 2; half++) {
                const __m512i two =
                    _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(const void *)(ph + 16 + 32 * half)));
                const int j = 4 * half;
                uint64_t b01, b23;
                memcpy(&b01, ph + 80 + 4 * j, 8);
                memcpy(&b23, ph + 88 + 4 * j, 8);
                const __m512i t01 = _mm512_and_si512(_mm512_srlv_epi64(two, sh01), m3); /* sub-blocks j, j + 1 */
                const __m512i t23 = _mm512_and_si512(_mm512_srlv_epi64(two, sh23), m3); /* j + 2, j + 3 */
                const __m512i h01 = _mm512_sub_epi8(_mm512_add_epi8(t01, t01), _mm512_movm_epi8((__mmask64)b01));
                const __m512i h23 = _mm512_sub_epi8(_mm512_add_epi8(t23, t23), _mm512_movm_epi8((__mmask64)b23));
                const int64_t o = 256 * s + 32 * j;
                a0 = _mm512_add_ps(a0, _mm512_mul_ps(_mm512_cvtepi32_ps(hb_dig5(h01, p->X + o, p->Xl + o)),
                                                     hb_pair5(coef, 8 * s + j)));
                a1 = _mm512_add_ps(a1, _mm512_mul_ps(_mm512_cvtepi32_ps(hb_dig5(h23, p->X + o + 64, p->Xl + o + 64)),
                                                     hb_pair5(coef, 8 * s + j + 2)));
            }
        }
        U[r] = hb_fold5(a0, a1, side);
    }
}

/* the original rows from the two planes, 16 bytes of codes at a time (the scalar rebuilds are their definition) */
HB_AVX2 static void avx2_hb_rebuild_q8_0(const unsigned char *hi, const unsigned char *lo, int64_t n,
                                         unsigned char *row) {
    const int64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(15), x80 = _mm_set1_epi8((char)0x80);
    for (int64_t b = 0; b < nb; b++) {
        unsigned char *blk = row + TR_Q8_0_BLOCK_BYTES * b;
        memcpy(blk, hi + 2 * b, 2);
        const __m128i h = _mm_loadu_si128((const __m128i *)(const void *)(hi + 2 * nb + 16 * b));
        const __m128i l = _mm_loadu_si128((const __m128i *)(const void *)(lo + 16 * b));
        const __m128i q0 = _mm_or_si128(_mm_xor_si128(_mm_slli_epi16(_mm_and_si128(h, m4), 4), x80), _mm_and_si128(l, m4));
        const __m128i q1 = _mm_or_si128(_mm_xor_si128(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), m4), 4), x80),
                                        _mm_and_si128(_mm_srli_epi16(l, 4), m4));
        _mm_storeu_si128((__m128i *)(void *)(blk + 2), q0);
        _mm_storeu_si128((__m128i *)(void *)(blk + 18), q1);
    }
}

HB_AVX2 static void avx2_hb_rebuild_q4_k(const unsigned char *hi, const unsigned char *lo, int64_t n,
                                         unsigned char *row) {
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    const __m256i bsel = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    const __m256i bmask = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    for (int64_t s = 0; s < n / 256; s++) {
        const unsigned char *ph = hi + 112 * s, *pl = lo + 32 * s;
        unsigned char *sb = row + TR_Q4_K_BLOCK_BYTES * s;
        memcpy(sb, ph, 16);
        for (int half = 0; half < 2; half++) {
            const __m256i two = _mm256_loadu_si256((const __m256i *)(const void *)(ph + 16 + 32 * half));
            __m256i q[4];
            for (int jj = 0; jj < 4; jj++) {
                const int j = 4 * half + jj;
                const __m256i t = _mm256_and_si256(_mm256_srli_epi16(two, 2 * jj), m3);
                int32_t w1, w0;
                memcpy(&w1, ph + 80 + 4 * j, 4);
                memcpy(&w0, pl + 4 * j, 4);
                __m256i b1 = _mm256_shuffle_epi8(_mm256_set1_epi32(w1), bsel);
                __m256i b0 = _mm256_shuffle_epi8(_mm256_set1_epi32(w0), bsel);
                b1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(b1, bmask), bmask), one);
                b0 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(b0, bmask), bmask), one);
                q[jj] = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi16(t, 2), _mm256_add_epi8(b1, b1)), b0);
            }
            for (int cc = 0; cc < 2; cc++) /* chunk 2 half + cc: sub-block 2c in the low nibbles, 2c + 1 in the high */
                _mm256_storeu_si256((__m256i *)(void *)(sb + TR_Q4_K_QS_OFFSET + 32 * (2 * half + cc)),
                                    _mm256_or_si256(q[2 * cc], _mm256_slli_epi16(q[2 * cc + 1], 4)));
        }
    }
}

void tr_hb_fill_x86(tr_kernels *k, int f16c, int vnni512) {
    if (!f16c) return; /* the scales are f16: the scalar definition stays */
    k->hb_bounds[TR_TYPE_Q8_0] = vnni512 ? avx512_hb_bounds_q8_0 : avx2_hb_bounds_q8_0;
    k->hb_bounds[TR_TYPE_Q4_K] = vnni512 ? avx512_hb_bounds_q4_k : avx2_hb_bounds_q4_k;
    k->hb_rebuild[TR_TYPE_Q8_0] = avx2_hb_rebuild_q8_0;
    k->hb_rebuild[TR_TYPE_Q4_K] = avx2_hb_rebuild_q4_k;
}
/* hot: end */

#else

void tr_hb_fill_x86(tr_kernels *k, int f16c, int vnni512) {
    (void)k;
    (void)f16c;
    (void)vnni512;
}

#endif

/* ---- the calls over the pool ---------------------------------------------------------------------------------- */

typedef struct {
    float top[TR_HB_TOP];
    int32_t top_i[TR_HB_TOP];
    float best;
    int32_t best_i;
    int64_t n;
    int n_top, bad;
} hb_worker;
_Static_assert(sizeof(hb_worker) <= HB_WORKER_BYTES, "a worker's results fit its cache line");

/* hot: begin */
typedef struct {
    const tr_kernels *k;
    const tr_hb_head *w;
    tr_hb_scratch *s;
    size_t hi_b, lo_b;
    const float *x; /* the input rows, cols floats apart */
    int64_t T;      /* how many (tr_hb_logits) */
    float *y;
    float th;
} hb_ctx;

static hb_worker *hb_w(const tr_hb_scratch *s, int worker) {
    return (hb_worker *)(void *)(s->workers + (size_t)worker * HB_WORKER_BYTES);
}

static unsigned char *hb_buf(const tr_hb_scratch *s, int worker, int i) {
    return s->rows + ((size_t)worker * 2 + (size_t)i) * s->row_stride;
}

static const unsigned char *hb_rebuilt(const hb_ctx *c, int64_t r, unsigned char *buf) {
    c->k->hb_rebuild[c->w->type](c->w->hi + (size_t)r * c->hi_b, c->w->lo + (size_t)r * c->lo_b, c->w->cols, buf);
    return buf;
}

/* rows r0 and r1 (r1 may be r0) exactly against the token, from their planes: dot_row's bits (Q8_0), q4x_dot2's
 * (Q4_K), on the worker's own buffers */
static void hb_exact2(const hb_ctx *c, int worker, int64_t r0, int64_t r1, float *out) {
    const unsigned char *b0 = hb_rebuilt(c, r0, hb_buf(c->s, worker, 0));
    const unsigned char *b1 = r1 == r0 ? b0 : hb_rebuilt(c, r1, hb_buf(c->s, worker, 1));
    if (c->w->type == TR_TYPE_Q4_K) {
        c->k->q4x_dot2(b0, b1, c->s->xq, c->w->cols, out);
        return;
    }
    out[0] = c->k->dot_row[TR_TYPE_Q8_0](b0, c->x, c->w->cols);
    out[1] = r1 == r0 ? out[0] : c->k->dot_row[TR_TYPE_Q8_0](b1, c->x, c->w->cols);
}

static void hb_top_insert(hb_worker *wk, float u, int32_t r) {
    int i = wk->n_top < TR_HB_TOP ? wk->n_top++ : TR_HB_TOP - 1;
    while (i > 0 && u > wk->top[i - 1]) {
        wk->top[i] = wk->top[i - 1];
        wk->top_i[i] = wk->top_i[i - 1];
        i--;
    }
    wk->top[i] = u;
    wk->top_i[i] = r;
}

/* region 1: the rows' bounds, each worker's TR_HB_TOP largest; a NaN bound marks the worker */
static void hb_bound_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const hb_ctx *c = (const hb_ctx *)ctx_;
    float *U = c->s->U;
    c->k->hb_bounds[c->w->type](c->w->hi + (size_t)begin * c->hi_b, end - begin, &c->s->prep, c->w->cols, U + begin);
    hb_worker *wk = hb_w(c->s, worker);
    float lim = wk->n_top == TR_HB_TOP ? wk->top[TR_HB_TOP - 1] : -INFINITY;
    for (int64_t r = begin; r < end; r++) {
        const float u = U[r];
        if (u > lim) {
            hb_top_insert(wk, u, (int32_t)r);
            if (wk->n_top == TR_HB_TOP) lim = wk->top[TR_HB_TOP - 1];
        } else if (u != u) {
            wk->bad = 1;
        }
    }
}

/* region 2: each worker's rows whose bound reaches th, computed exactly; its best (the lowest row on a tie) */
static void hb_pick_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const hb_ctx *c = (const hb_ctx *)ctx_;
    const float *U = c->s->U;
    const float th = c->th;
    int32_t *ids = c->s->ids + begin;
    int64_t m = 0;
    for (int64_t r = begin; r < end; r++)
        if (U[r] >= th) ids[m++] = (int32_t)r;
    hb_worker *wk = hb_w(c->s, worker);
    const int64_t step = c->w->type == TR_TYPE_Q4_K ? 2 : 1; /* Q4_K's road computes two rows a call */
    for (int64_t i = 0; i < m; i += step) {
        const int two = step == 2 && i + 1 < m;
        float o[2];
        hb_exact2(c, worker, ids[i], ids[two ? i + 1 : i], o);
        for (int t = 0; t <= two; t++) {
            const float v = o[t];
            const int32_t r = ids[i + t];
            if (v != v) wk->bad = 1;
            else if (wk->best_i < 0 || v > wk->best || (v == wk->best && r < wk->best_i)) {
                wk->best = v;
                wk->best_i = r;
            }
        }
    }
    wk->n += m;
}

static void hb_logits_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const hb_ctx *c = (const hb_ctx *)ctx_;
    const tr_kernels *k = c->k;
    const int64_t rows = c->w->rows, n = c->w->cols, T = c->T;
    for (int64_t pr = begin; pr < end; pr++) {
        const int64_t r0 = 2 * pr, r1 = r0 + 1 < rows ? r0 + 1 : r0;
        const unsigned char *b[2];
        b[0] = hb_rebuilt(c, r0, hb_buf(c->s, worker, 0));
        b[1] = r1 == r0 ? b[0] : hb_rebuilt(c, r1, hb_buf(c->s, worker, 1));
        if (c->w->type == TR_TYPE_Q4_K) {
            int64_t t = 0;
            float o[2 * TR_Q4X_XT_MAX];
            while (T - t >= 2 && k->q4x_dot_xt != NULL) {
                const int g = T - t < TR_Q4X_XT_MAX ? (int)(T - t) : TR_Q4X_XT_MAX;
                const void *xs[TR_Q4X_XT_MAX];
                for (int i = 0; i < g; i++) xs[i] = c->s->xq + (size_t)(t + i) * c->s->xq_row;
                k->q4x_dot_xt(b[0], b[1], xs, n, g, o, 2);
                for (int i = 0; i < g; i++) {
                    c->y[(t + i) * rows + r0] = o[2 * i];
                    c->y[(t + i) * rows + r1] = o[2 * i + 1];
                }
                t += g;
            }
            for (; t < T; t++) {
                k->q4x_dot2(b[0], b[1], c->s->xq + (size_t)t * c->s->xq_row, n, o);
                c->y[t * rows + r0] = o[0];
                c->y[t * rows + r1] = o[1];
            }
            continue;
        }
        for (int i = 0; i <= (r1 != r0); i++) {
            const int64_t r = i ? r1 : r0;
            int64_t t = 0;
            float o[TR_DOT_TOKENS];
            while (T - t >= TR_DOT_TOKENS && k->dot_row_x4[TR_TYPE_Q8_0] != NULL) {
                k->dot_row_x4[TR_TYPE_Q8_0](b[i], c->x + t * n, n, n, o);
                for (int j = 0; j < TR_DOT_TOKENS; j++) c->y[(t + j) * rows + r] = o[j];
                t += TR_DOT_TOKENS;
            }
            while (T - t >= 2 && k->dot_row_xt[TR_TYPE_Q8_0] != NULL) {
                const int g = T - t < 3 ? (int)(T - t) : 3;
                k->dot_row_xt[TR_TYPE_Q8_0](b[i], c->x + t * n, n, n, g, o);
                for (int j = 0; j < g; j++) c->y[(t + j) * rows + r] = o[j];
                t += g;
            }
            for (; t < T; t++) c->y[t * rows + r] = k->dot_row[TR_TYPE_Q8_0](b[i], c->x + t * n, n);
        }
    }
}

static void hb_ctx_init(hb_ctx *c, const tr_hb_head *w, tr_hb_scratch *s, const float *x, int64_t T, float *y) {
    c->k = tr_kernels_get();
    c->w = w;
    c->s = s;
    c->hi_b = tr_hb_hi_bytes(w->type, w->cols);
    c->lo_b = tr_hb_lo_bytes(w->type, w->cols);
    c->x = x;
    c->T = T;
    c->y = y;
    c->th = 0.0f;
}

void tr_hb_hint(tr_pool *pool, const tr_hb_head *w, int64_t n, size_t max_bytes) {
    if (tr_pool_hints(pool) == 0 || max_bytes == 0 || n <= 0) return;
    if (n > w->rows) n = w->rows;
    const size_t hb = tr_hb_hi_bytes(w->type, w->cols);
    int64_t b, e;
    const int chunks = tr_pool_region(pool, n, 64, 0, &b, &e); /* region 1's call */
    for (int t = 1; t < chunks; t++) {
        tr_pool_region(pool, n, 64, t, &b, &e);
        size_t bytes = (size_t)((e - b) / 2) * hb;
        if (bytes > max_bytes) bytes = max_bytes;
        tr_pool_hint(pool, t, w->hi + (size_t)b * hb, bytes);
    }
}

void tr_hb_logits(tr_pool *pool, const tr_hb_head *w, const float *x, int64_t n_tokens, float *y, tr_hb_scratch *s) {
    if (n_tokens < 1) return;
    hb_ctx c;
    hb_ctx_init(&c, w, s, x, n_tokens, y);
    if (w->type == TR_TYPE_Q4_K) tr_q4x_prepare(pool, x, n_tokens, w->cols, s->xq);
    tr_parallel_for_balanced(pool, (w->rows + 1) / 2, 16, hb_logits_body, &c);
}

int32_t tr_hb_argmax(tr_pool *pool, const tr_hb_head *w, const float *h, int64_t n, tr_hb_scratch *s, float *y,
                     tr_hb_result *res) {
    hb_ctx c;
    hb_ctx_init(&c, w, s, h, 1, y);
    const int nw = pool != NULL ? tr_pool_size(pool) : 1;
    if (n > w->rows) n = w->rows;
    res->rows = 0;
    res->full = 0;
    if (n < 1 || nw > s->n_workers || tr_hb_prep_build(w->type, h, w->cols, &s->prep) != 0) goto full;
    if (w->type == TR_TYPE_Q4_K) c.k->q4x_prep(h, w->cols, s->xq);
    for (int i = 0; i < nw; i++) {
        hb_worker *wk = hb_w(s, i);
        memset(wk, 0, sizeof *wk);
        wk->best_i = -1;
    }
    tr_parallel_for_balanced(pool, n, 64, hb_bound_body, &c);

    /* the TR_HB_TOP largest bounds of all, computed exactly: the best of them is the threshold */
    hb_worker top;
    memset(&top, 0, sizeof top);
    for (int i = 0; i < nw; i++) {
        const hb_worker *wk = hb_w(s, i);
        if (wk->bad) goto full;
        for (int j = 0; j < wk->n_top; j++)
            if (top.n_top < TR_HB_TOP || wk->top[j] > top.top[TR_HB_TOP - 1]) hb_top_insert(&top, wk->top[j], wk->top_i[j]);
    }
    float th = -INFINITY;
    for (int j = 0; j < top.n_top; j += 2) {
        const int32_t r0 = top.top_i[j], r1 = j + 1 < top.n_top ? top.top_i[j + 1] : r0;
        float o[2];
        hb_exact2(&c, 0, r0, r1, o);
        for (int t = 0; t < 2; t++) {
            if (o[t] != o[t]) goto full;
            if (o[t] > th) th = o[t];
        }
    }
    c.th = th;
    tr_parallel_for(pool, n, 64, hb_pick_body, &c);

    float best = 0.0f;
    int32_t bi = -1;
    res->rows = top.n_top;
    for (int i = 0; i < nw; i++) {
        const hb_worker *wk = hb_w(s, i);
        if (wk->bad) goto full;
        res->rows += wk->n;
        if (wk->best_i >= 0 && (bi < 0 || wk->best > best || (wk->best == best && wk->best_i < bi))) {
            best = wk->best;
            bi = wk->best_i;
        }
    }
    if (bi >= 0) return bi;

full: /* h, a bound or a score not finite: every row, and the scan's own rule */
    tr_hb_logits(pool, w, h, 1, y, s);
    res->rows = w->rows;
    res->full = 1;
    return tr_argmax_f32(pool, y, n);
}
/* hot: end */

/* ---- the scratch ---------------------------------------------------------------------------------------------- */

void tr_hb_scratch_free(tr_hb_scratch *s) {
    tr_free_aligned(s->prep_buf);
    tr_free_aligned(s->U);
    tr_free_aligned(s->ids);
    tr_free_aligned(s->xq);
    tr_free_aligned(s->rows);
    tr_free_aligned(s->workers);
    memset(s, 0, sizeof *s);
}

int tr_hb_scratch_init(tr_hb_scratch *s, const tr_hb_head *w, int n_workers, int64_t max_tokens) {
    memset(s, 0, sizeof *s);
    if (!tr_hb_supports(w->type, w->cols) || n_workers < 1 || max_tokens < 1) return -1;
    s->n_workers = n_workers;
    s->max_tokens = max_tokens;
    s->prep_buf = tr_alloc_aligned(tr_hb_prep_bytes(w->cols), 64);
    s->U = (float *)tr_alloc_aligned((size_t)w->rows * sizeof(float), 64);
    s->ids = (int32_t *)tr_alloc_aligned((size_t)w->rows * sizeof(int32_t), 64);
    s->row_stride = (tr_row_bytes(w->type, w->cols) + 63) / 64 * 64;
    s->rows = (unsigned char *)tr_alloc_aligned((size_t)n_workers * 2 * s->row_stride, 64);
    s->workers = (unsigned char *)tr_alloc_aligned((size_t)n_workers * HB_WORKER_BYTES, 64);
    if (w->type == TR_TYPE_Q4_K) {
        s->xq_row = tr_q4x_bytes(w->cols);
        s->xq = (unsigned char *)tr_alloc_aligned((size_t)max_tokens * s->xq_row, 64);
    }
    if (s->prep_buf == NULL || s->U == NULL || s->ids == NULL || s->rows == NULL || s->workers == NULL ||
        (w->type == TR_TYPE_Q4_K && s->xq == NULL)) {
        tr_hb_scratch_free(s);
        return -1;
    }
    s->prep = tr_hb_prep_view(s->prep_buf, w->cols);
    return 0;
}
