/* test_kernels.c — tests for src/kernels/kernels.c: half->float, Q8_0 dequant,
 * the 16-lane reduction contract (kernels.h) against an independent in-test
 * implementation, dot_row == dot_f32(dequantized), every SIMD tier == scalar,
 * rope/swiglu/attention == their first per-element definitions, tr_matmul
 * thread-count determinism, tr_matmul_grouped == one dot_row per element. */
#include "test.h"

#include "../src/kernels/kernels.h"
#include "../src/base/threads.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int bit_eq(float a, float b) {
    uint32_t ba, bb;
    memcpy(&ba, &a, sizeof ba);
    memcpy(&bb, &b, sizeof bb);
    return ba == bb;
}

/* Independent implementation of the lane contract in kernels.h, kept
 * deliberately separate from kernels.c's internal lane_combine(). */
static float ref_dot_f32(const float *a, const float *b, int64_t n) {
    float lane[TR_LANES] = {0};
    for (int64_t k = 0; k < n; k++) lane[k % TR_LANES] += a[k] * b[k];

    float s01 = lane[0] + lane[1], s23 = lane[2] + lane[3];
    float s45 = lane[4] + lane[5], s67 = lane[6] + lane[7];
    float lo = (s01 + s23) + (s45 + s67);

    float s89 = lane[8] + lane[9], s1011 = lane[10] + lane[11];
    float s1213 = lane[12] + lane[13], s1415 = lane[14] + lane[15];
    float hi = (s89 + s1011) + (s1213 + s1415);

    return lo + hi;
}

static unsigned next_rand(unsigned *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return *seed;
}

static float rand_float(unsigned *seed) {
    return ((float)(next_rand(seed) % 20001u) - 10000.0f) / 1000.0f; /* [-10, 10] */
}

/* ---- half -> float, via dequant_row[F16] (no other public entry point) --- */

static void test_half_to_float(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    uint16_t bits[7] = {
        0x0000, /* +0 */
        0x8000, /* -0 */
        0x3C00, /* 1.0 */
        0xC000, /* -2.0 */
        0x7BFF, /* 65504, max normal half */
        0x0001, /* smallest subnormal: 2^-24 */
        0x7C00, /* +inf */
    };
    float want[7] = {0.0f, -0.0f, 1.0f, -2.0f, 65504.0f, 5.9604644775390625e-8f, INFINITY};

    float got[7];
    K->dequant_row[TR_TYPE_F16](bits, got, 7);
    for (int i = 0; i < 7; i++) TR_CHECK(bit_eq(got[i], want[i]));

    uint16_t nan_bits = 0x7E00; /* exponent all-1, nonzero mantissa */
    float nan_out;
    K->dequant_row[TR_TYPE_F16](&nan_bits, &nan_out, 1);
    TR_CHECK(isnan(nan_out));
}

/* ---- Q8_0 dequant on a hand-built block ----------------------------------- */

static void test_q8_0_dequant(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    unsigned char blk[34];
    uint16_t scale_bits = 0x4000; /* 2.0 in binary16 */
    memcpy(blk, &scale_bits, 2);
    for (int i = 0; i < 32; i++) blk[2 + i] = (unsigned char)(int8_t)(i - 16); /* -16..15 */

    float out[32];
    K->dequant_row[TR_TYPE_Q8_0](blk, out, 32);
    for (int i = 0; i < 32; i++) {
        float expect = 2.0f * (float)(int8_t)(i - 16);
        TR_CHECK(bit_eq(out[i], expect));
    }
}

/* ---- dot_f32 against the explicit in-test lane contract ------------------- */

static void test_dot_f32_contract(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    static float a[1000], b[1000];
    unsigned seed = 12345u;
    for (int i = 0; i < 1000; i++) {
        a[i] = rand_float(&seed);
        b[i] = rand_float(&seed);
    }

    for (int64_t n = 0; n <= 70; n++) {
        float got = K->dot_f32(a, b, n);
        float want = ref_dot_f32(a, b, n);
        TR_CHECK(bit_eq(got, want));
    }
    float got1000 = K->dot_f32(a, b, 1000);
    float want1000 = ref_dot_f32(a, b, 1000);
    TR_CHECK(bit_eq(got1000, want1000));
}

/* ---- dot_row(F16/Q8_0) == dot_f32(dequantized) bit for bit ---------------- */

static void test_dot_row_f16(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    const int64_t n = 65; /* not a multiple of TR_LANES: exercises tail lanes */
    uint16_t row[65];
    float x[65];
    unsigned seed = 777u;
    for (int64_t i = 0; i < n; i++) {
        unsigned r = next_rand(&seed);
        unsigned sign = (r >> 20) & 1u;
        unsigned exp = (r >> 10) & 0x1Fu;
        if (exp == 0x1Fu) exp = 0x1Eu; /* avoid inf/nan for this comparison */
        unsigned mant = r & 0x3FFu;
        row[i] = (uint16_t)((sign << 15) | (exp << 10) | mant);
        x[i] = rand_float(&seed);
    }

    float dequant[65];
    K->dequant_row[TR_TYPE_F16](row, dequant, n);
    float want = ref_dot_f32(dequant, x, n);
    float got = K->dot_row[TR_TYPE_F16](row, x, n);
    TR_CHECK(bit_eq(got, want));
}

static void test_dot_row_q8_0(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    const int64_t n = 64; /* two blocks */
    unsigned char row[2 * 34];
    float x[64];
    unsigned seed = 999u;
    for (int b = 0; b < 2; b++) {
        uint16_t scale_bits = (uint16_t)((5u + (unsigned)b) << 10); /* modest positive scales */
        memcpy(row + b * 34, &scale_bits, 2);
        for (int j = 0; j < 32; j++) {
            int v = (int)(next_rand(&seed) >> 8) % 256 - 128;
            row[b * 34 + 2 + j] = (unsigned char)(int8_t)v;
        }
    }
    for (int64_t i = 0; i < n; i++) x[i] = rand_float(&seed);

    float dequant[64];
    K->dequant_row[TR_TYPE_Q8_0](row, dequant, n);
    float want = ref_dot_f32(dequant, x, n);
    float got = K->dot_row[TR_TYPE_Q8_0](row, x, n);
    TR_CHECK(bit_eq(got, want));
}

/* ---- every SIMD tier == scalar, bit for bit (NaN: any NaN) ---------------- */

static int same_float(float a, float b) {
    return bit_eq(a, b) || (isnan(a) && isnan(b));
}

/* Ordinary values; with `special`, sometimes the ones that break careless SIMD code.
 * Specials stay rare: an inf or NaN in a lane hides every rounding difference. */
static float rand_special(unsigned *seed, int special) {
    if (!special) return rand_float(seed);
    switch (next_rand(seed) % 256u) {
    case 0: return 0.0f;
    case 1: return -0.0f;
    case 2: return 1.0e-45f;              /* subnormal */
    case 3: return -3.0e38f;              /* product overflows to inf */
    case 4: return 1.0e-30f;              /* product underflows */
    default: return rand_float(seed);
    }
}

static void fill_q8_0(unsigned char *row, int64_t nb, unsigned *seed, int special) {
    for (int64_t b = 0; b < nb; b++) {
        unsigned r = next_rand(seed);
        /* with `special`, now and then any exponent: subnormal and inf/nan scales */
        uint16_t scale_bits = (uint16_t)((r >> 8) & 0xFFFFu);
        if (!special || (r & 63u) != 0) scale_bits = (uint16_t)((scale_bits & 0x83FFu) | ((8u + (r >> 3) % 14u) << 10));
        memcpy(row + b * 34, &scale_bits, 2);
        for (int j = 0; j < 32; j++) row[b * 34 + 2 + j] = (unsigned char)(next_rand(seed) >> 16);
    }
}

enum { CMP_MAX_N = 4096 + 17 };

/* n halves: ordinary ones (any sign, exponent bits 1..30), or with `special` any 16 bits at
 * all, so subnormals, zeros, infinities and NaNs too */
static void fill_f16(unsigned char *row, int64_t n, unsigned *seed, int special) {
    for (int64_t i = 0; i < n; i++) {
        unsigned r = next_rand(seed);
        uint16_t h = (uint16_t)((r >> 8) & 0xFFFFu);
        if (!special || (r & 63u) != 0) h = (uint16_t)((h & 0x83FFu) | ((1u + (r >> 3) % 30u) << 10));
        memcpy(row + 2 * i, &h, sizeof h);
    }
}

/* Runs K and S on the same inputs; counts the calls whose results differ. */
static void count_diffs(const tr_kernels *K, const tr_kernels *S, unsigned seed, int *bad_dot, int *bad_row,
                        int *bad_axpy, int *bad_x4) {
    static float a[CMP_MAX_N], b[CMP_MAX_N], yk[CMP_MAX_N], ys[CMP_MAX_N], y4[CMP_MAX_N];
    static unsigned char row[(4096 / 32) * 34], hrow[2 * CMP_MAX_N];
    static const int64_t big[] = {255, 256, 257, 1000, 2048, 4096};

    *bad_dot = *bad_row = *bad_axpy = *bad_x4 = 0;
    for (int round = 0; round < 40; round++) {
        for (int i = 0; i < CMP_MAX_N; i++) {
            a[i] = rand_special(&seed, round % 2);
            b[i] = rand_special(&seed, round % 2);
        }
        fill_f16(hrow, CMP_MAX_N, &seed, round % 2);
        /* every length up to 200 (all tails), then large rows, from two offsets */
        for (int64_t n = 0; n <= 200 + (int64_t)(sizeof big / sizeof big[0]); n++) {
            int64_t len = n <= 200 ? n : big[n - 201];
            for (int off = 0; off < 2; off++) {
                if (!same_float(K->dot_f32(a + off, b + off, len), S->dot_f32(a + off, b + off, len))) (*bad_dot)++;
                /* an F32 weight row (the router's): the tier's kernel against scalar's */
                if (!same_float(K->dot_row[TR_TYPE_F32](a + off, b + off, len),
                                S->dot_row[TR_TYPE_F32](a + off, b + off, len)))
                    (*bad_row)++;
                /* an F16 weight row: the tier's conversion and lanes against scalar's */
                if (!same_float(K->dot_row[TR_TYPE_F16](hrow + 2 * off, b + off, len),
                                S->dot_row[TR_TYPE_F16](hrow + 2 * off, b + off, len)))
                    (*bad_row)++;
                /* axpy: y from a, x from b; the elements past len must stay untouched */
                float alpha = rand_special(&seed, round % 2);
                memcpy(yk, a, sizeof yk);
                memcpy(ys, a, sizeof ys);
                K->axpy_f32(yk + off, b + off, alpha, len);
                S->axpy_f32(ys + off, b + off, alpha, len);
                for (int i = 0; i < CMP_MAX_N; i++) {
                    if (!same_float(yk[i], ys[i])) {
                        (*bad_axpy)++;
                        break;
                    }
                }
                /* 4 cache positions per call, rows of b a stride apart that is not the length:
                 * the same as scalar's, and as the tier's own dot_f32 and axpy_f32 called 4
                 * times in a row (the 4 rows must fit in b) */
                int64_t stride = len + 3;
                if (off + TR_ATTN_X * stride > CMP_MAX_N) continue;
                float dk[TR_ATTN_X], ds[TR_ATTN_X], alpha4[TR_ATTN_X];
                K->dot_f32_x4(a + off, b + off, stride, len, dk);
                S->dot_f32_x4(a + off, b + off, stride, len, ds);
                for (int j = 0; j < TR_ATTN_X; j++) {
                    if (!same_float(dk[j], ds[j])) (*bad_x4)++;
                    if (!same_float(dk[j], K->dot_f32(a + off, b + off + j * stride, len))) (*bad_x4)++;
                    alpha4[j] = rand_special(&seed, round % 2);
                }
                /* the same F32 row, and the same F16 row, against 4 input rows */
                K->dot_row_x4[TR_TYPE_F32](a + off, b + off, stride, len, dk);
                S->dot_row_x4[TR_TYPE_F32](a + off, b + off, stride, len, ds);
                for (int j = 0; j < TR_DOT_TOKENS; j++)
                    if (!same_float(dk[j], ds[j])) (*bad_row)++;
                K->dot_row_x4[TR_TYPE_F16](hrow + 2 * off, b + off, stride, len, dk);
                S->dot_row_x4[TR_TYPE_F16](hrow + 2 * off, b + off, stride, len, ds);
                for (int j = 0; j < TR_DOT_TOKENS; j++)
                    if (!same_float(dk[j], ds[j])) (*bad_row)++;
                memcpy(yk, a, sizeof yk);
                memcpy(ys, a, sizeof ys);
                memcpy(y4, a, sizeof y4);
                K->axpy_f32_x4(yk + off, b + off, stride, alpha4, len);
                S->axpy_f32_x4(ys + off, b + off, stride, alpha4, len);
                for (int j = 0; j < TR_ATTN_X; j++) K->axpy_f32(y4 + off, b + off + j * stride, alpha4[j], len);
                for (int i = 0; i < CMP_MAX_N; i++) {
                    if (!same_float(yk[i], ys[i]) || !same_float(yk[i], y4[i])) {
                        (*bad_x4)++;
                        break;
                    }
                }
            }
        }
        for (int64_t nb = 1; nb <= 128; nb += (nb < 8 ? 1 : 15)) {
            fill_q8_0(row, nb, &seed, round % 2);
            int64_t n = nb * 32;
            if (!same_float(K->dot_row[TR_TYPE_Q8_0](row, b, n), S->dot_row[TR_TYPE_Q8_0](row, b, n))) (*bad_row)++;
            /* the same row against 4 consecutive input rows: same as the tier's own
             * dot_row on each of them, and as scalar's (4 * n floats must fit in b) */
            if (K->dot_row_x4[TR_TYPE_Q8_0] != NULL && 4 * n <= CMP_MAX_N) {
                float xk[TR_DOT_TOKENS], xs[TR_DOT_TOKENS];
                K->dot_row_x4[TR_TYPE_Q8_0](row, b, n, n, xk);
                S->dot_row_x4[TR_TYPE_Q8_0](row, b, n, n, xs);
                for (int t = 0; t < TR_DOT_TOKENS; t++) {
                    if (!same_float(xk[t], xs[t])) (*bad_row)++;
                    if (!same_float(xk[t], K->dot_row[TR_TYPE_Q8_0](row, b + t * n, n))) (*bad_row)++;
                }
            }
        }
    }
}

/* Wrong only in rounding: the scalar arithmetic with lanes 0 and 8 swapped before the
 * tree. count_diffs must see it, or the comparison is too weak to prove anything
 * (docs/LESSONS.md #20: too many infinities once hid exactly this). */
static float wrong_dot_f32(const float *a, const float *b, int64_t n) {
    float lane[TR_LANES] = {0};
    for (int64_t k = 0; k < n; k++) lane[k % TR_LANES] += a[k] * b[k];
    float t = lane[0];
    lane[0] = lane[8];
    lane[8] = t;
    float s01 = lane[0] + lane[1], s23 = lane[2] + lane[3];
    float s45 = lane[4] + lane[5], s67 = lane[6] + lane[7];
    float s89 = lane[8] + lane[9], s1011 = lane[10] + lane[11];
    float s1213 = lane[12] + lane[13], s1415 = lane[14] + lane[15];
    return ((s01 + s23) + (s45 + s67)) + ((s89 + s1011) + (s1213 + s1415));
}

static float wrong_dot_row_q8_0(const void *row, const float *x, int64_t n) {
    static float w[CMP_MAX_N];
    tr_kernels_tier("scalar")->dequant_row[TR_TYPE_Q8_0](row, w, n);
    return wrong_dot_f32(w, x, n);
}

static float wrong_dot_row_f16(const void *row, const float *x, int64_t n) {
    static float w[CMP_MAX_N];
    tr_kernels_tier("scalar")->dequant_row[TR_TYPE_F16](row, w, n);
    return wrong_dot_f32(w, x, n);
}

/* Wrong only in rounding: a fused multiply-add, what FMA contraction would silently do. */
static void wrong_axpy_f32(float *y, const float *x, float a, int64_t n) {
    for (int64_t k = 0; k < n; k++) y[k] = fmaf(a, x[k], y[k]);
}

/* Wrong only in rounding: the 4 products added to each other first and to y last, what a
 * kernel that keeps y out of the chain of additions would do. */
static void wrong_axpy_f32_x4(float *y, const float *x, int64_t stride, const float *a, int64_t n) {
    for (int64_t k = 0; k < n; k++)
        y[k] = y[k] + ((a[0] * x[k] + a[1] * x[stride + k]) + (a[2] * x[2 * stride + k] + a[3] * x[3 * stride + k]));
}

static void test_tiers_match_scalar(void) {
    static const char *const tiers[] = {"avx2", "avx512"};
    const tr_kernels *S = tr_kernels_tier("scalar");
    TR_CHECK(S != NULL);
    if (S == NULL) return;
    int bad_dot, bad_row, bad_axpy, bad_x4;

    tr_kernels wrong = *S;
    wrong.dot_f32 = wrong_dot_f32;
    wrong.axpy_f32 = wrong_axpy_f32;
    wrong.dot_row[TR_TYPE_Q8_0] = wrong_dot_row_q8_0;
    count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
    TR_CHECK(bad_dot > 0);
    TR_CHECK(bad_row > 0);
    TR_CHECK(bad_axpy > 0);
    /* the x4 kernels of `wrong` are still scalar's: they differ from its own wrong dot and axpy */
    TR_CHECK(bad_x4 > 0);
    wrong = *S;
    wrong.axpy_f32_x4 = wrong_axpy_f32_x4;
    count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
    TR_CHECK_EQ_INT(bad_dot + bad_row + bad_axpy, 0);
    TR_CHECK(bad_x4 > 0);
    /* and a wrong F16 row alone: the Q8_0 rows above must not be what made bad_row move */
    wrong = *S;
    wrong.dot_row[TR_TYPE_F16] = wrong_dot_row_f16;
    count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
    TR_CHECK_EQ_INT(bad_dot + bad_axpy + bad_x4, 0);
    TR_CHECK(bad_row > 0);

    for (size_t t = 0; t < sizeof tiers / sizeof tiers[0]; t++) {
        const tr_kernels *K = tr_kernels_tier(tiers[t]);
        if (K == NULL) {
            printf("  tier %-8s not available on this CPU: skipped\n", tiers[t]);
            continue;
        }
        /* same numbers says nothing on WHICH function ran: that the tier's entries are its own,
         * and that the engine goes through them, is tests/test_tier_used.c */
        count_diffs(K, S, 2026u + (unsigned)t, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
        TR_CHECK_EQ_INT(bad_dot, 0);
        TR_CHECK_EQ_INT(bad_row, 0);
        TR_CHECK_EQ_INT(bad_axpy, 0);
        TR_CHECK_EQ_INT(bad_x4, 0);
        printf("  tier %-8s dot_f32, axpy_f32, their x4, dot_row and dot_row_x4 of f32, f16 and q8_0 %s\n", K->tier,
               bad_dot == 0 && bad_row == 0 && bad_axpy == 0 && bad_x4 == 0 ? "identical to scalar"
                                                                            : "DIFFER from scalar");
    }
}

/* ---- rope, swiglu, attention: identical to their first definitions ---------- */

/* The per-element RoPE the engine had before the table (trigonometry at every call). */
static void ref_rope_neox(float *x, int64_t n_heads, int64_t head_dim, int64_t pos, float theta) {
    int64_t half = head_dim / 2;
    for (int64_t h = 0; h < n_heads; h++) {
        float *base = x + h * head_dim;
        for (int64_t i = 0; i < half; i++) {
            double inv_freq = 1.0 / pow((double)theta, (2.0 * (double)i) / (double)head_dim);
            double angle = inv_freq * (double)pos;
            float c = (float)cos(angle), s = (float)sin(angle);
            float x1 = base[i], x2 = base[i + half];
            base[i] = x1 * c - x2 * s;
            base[i + half] = x2 * c + x1 * s;
        }
    }
}

static void test_rope_table(void) {
    enum { N_POS = 5000, HEADS = 3, MAX_DIM = 256 };
    static const int64_t dims[] = {2, 8, 64, 128, 256};
    static const float thetas[] = {10000.0f, 500000.0f};
    static float cos_t[N_POS * (MAX_DIM / 2)], sin_t[N_POS * (MAX_DIM / 2)];
    static float x[HEADS * MAX_DIM], got[HEADS * MAX_DIM], want[HEADS * MAX_DIM];
    unsigned seed = 11u;
    int bad = 0;
    for (size_t di = 0; di < sizeof dims / sizeof dims[0]; di++) {
        int64_t dim = dims[di], half = dim / 2;
        for (size_t ti = 0; ti < sizeof thetas / sizeof thetas[0]; ti++) {
            tr_rope_table(cos_t, sin_t, N_POS, dim, thetas[ti]);
            for (int64_t pos = 0; pos < N_POS; pos += (pos < 70 ? 1 : 263)) {
                for (int64_t i = 0; i < HEADS * dim; i++) x[i] = rand_float(&seed);
                memcpy(got, x, sizeof x);
                memcpy(want, x, sizeof x);
                tr_rope_neox(got, HEADS, dim, cos_t + pos * half, sin_t + pos * half);
                ref_rope_neox(want, HEADS, dim, pos, thetas[ti]);
                if (memcmp(got, want, (size_t)(HEADS * dim) * sizeof(float)) != 0) bad++;
            }
        }
    }
    TR_CHECK_EQ_INT(bad, 0);
}

static void test_swiglu_threads(void) {
    enum { MAX_N = 8192 + 7 };
    static const int64_t sizes[] = {1, 255, 256, 257, 1024, 8192 + 7};
    static const int threads[] = {1, 2, 3, 8};
    static float x0[MAX_N], y[MAX_N], want[MAX_N], got[MAX_N];
    unsigned seed = 5u;
    for (int i = 0; i < MAX_N; i++) {
        x0[i] = rand_special(&seed, 1);
        y[i] = rand_special(&seed, 1);
    }
    int bad = 0;
    for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        int64_t n = sizes[si];
        for (int64_t i = 0; i < n; i++) {
            float v = x0[i];
            want[i] = v / (1.0f + tr_expf(-v)) * y[i];
        }
        memcpy(got, x0, sizeof got);
        tr_swiglu(NULL, got, y, n);
        for (int64_t i = 0; i < n; i++) bad += !same_float(got[i], want[i]);
        for (size_t ti = 0; ti < sizeof threads / sizeof threads[0]; ti++) {
            tr_pool *p = tr_pool_create(threads[ti]);
            TR_CHECK(p != NULL);
            if (p == NULL) continue;
            memcpy(got, x0, sizeof got);
            tr_swiglu(p, got, y, n);
            for (int64_t i = 0; i < n; i++) bad += !same_float(got[i], want[i]);
            tr_pool_destroy(p);
        }
    }
    TR_CHECK_EQ_INT(bad, 0);
}

/* The attention loop the engine had before tr_attention_head. */
static void ref_attention_head(const float *q, const float *keys, const float *values, int64_t stride,
                               int64_t offset, int64_t n_pos, int64_t head_dim, float scale, float *scores,
                               float *out) {
    for (int64_t t = 0; t < n_pos; t++) scores[t] = ref_dot_f32(q, keys + t * stride + offset, head_dim) * scale;
    tr_softmax(scores, n_pos);
    for (int64_t d = 0; d < head_dim; d++) out[d] = 0.0f;
    for (int64_t t = 0; t < n_pos; t++) {
        const float *v = values + t * stride + offset;
        for (int64_t d = 0; d < head_dim; d++) out[d] += scores[t] * v[d];
    }
}

static void test_attention_head(void) {
    enum { MAX_POS = 700, MAX_DIM = 130 };
    static const int64_t dims[] = {1, 5, 16, 64, 100, 128};
    static const int64_t positions[] = {1, 2, 17, 255, 700};
    static float q[MAX_DIM], got[MAX_DIM], want[MAX_DIM];
    static float keys[MAX_POS * 3 * MAX_DIM], values[MAX_POS * 3 * MAX_DIM];
    static float s_got[MAX_POS], s_want[MAX_POS];
    unsigned seed = 3u;
    int bad = 0;
    for (size_t di = 0; di < sizeof dims / sizeof dims[0]; di++) {
        int64_t dim = dims[di], stride = 3 * dim;
        float scale = 1.0f / sqrtf((float)dim);
        for (size_t pi = 0; pi < sizeof positions / sizeof positions[0]; pi++) {
            int64_t n_pos = positions[pi];
            /* small q: scores spread over many positions instead of one-hot softmax */
            for (int64_t i = 0; i < dim; i++) q[i] = rand_float(&seed) * 0.01f;
            for (int64_t i = 0; i < n_pos * stride; i++) {
                keys[i] = rand_float(&seed);
                values[i] = rand_float(&seed);
            }
            for (int64_t off = 0; off < stride; off += dim) {
                tr_attention_head(q, keys, values, stride, off, n_pos, dim, scale, s_got, got);
                ref_attention_head(q, keys, values, stride, off, n_pos, dim, scale, s_want, want);
                if (memcmp(got, want, (size_t)dim * sizeof(float)) != 0) bad++;
            }
        }
    }
    TR_CHECK_EQ_INT(bad, 0);
}

/* tr_attention_group: every query of a group gets the bits tr_attention_head gives it alone,
 * scores included, and nothing is written outside its own slice of the output or past the end
 * of its own row of scores. Groups start and
 * end all around the borders of a block of positions (TR_ATTN_BLOCK) and of the 4 positions of
 * an x4 kernel; strides are wider than what is used. */
static void test_attention_group(void) {
    enum { MAX_POS = 300, MAX_DIM = 130, MAX_Q = 19, SCORE_STRIDE = MAX_POS + 7 };
    static const int64_t dims[] = {1, 5, 16, 100, 128};
    static const int64_t firsts[] = {1, 2, 3, 4, 5, 61, 63, 64, 65, 66, 127, 128, 129, 190, 250};
    static const int64_t groups[] = {1, 2, 3, 4, 5, 16, 19};
    static float q[MAX_Q * 2 * MAX_DIM], got[MAX_Q * 3 * MAX_DIM], want[MAX_DIM];
    static float keys[MAX_POS * MAX_DIM], values[MAX_POS * MAX_DIM];
    static float s_got[MAX_Q * SCORE_STRIDE], s_want[MAX_POS];
    unsigned seed = 17u;
    int bad = 0, cases = 0;
    TR_CHECK(TR_ATTN_BLOCK == 64); /* the borders above are chosen around 64 and 128 */
    for (size_t di = 0; di < sizeof dims / sizeof dims[0]; di++) {
        int64_t dim = dims[di], q_stride = 2 * dim, out_stride = 3 * dim;
        float scale = 1.0f / sqrtf((float)dim);
        for (size_t fi = 0; fi < sizeof firsts / sizeof firsts[0]; fi++) {
            for (size_t gi = 0; gi < sizeof groups / sizeof groups[0]; gi++) {
                int64_t first = firsts[fi], n_q = groups[gi], last = first + n_q - 1;
                if (last > MAX_POS) continue;
                /* small q: scores spread over many positions instead of one-hot softmax */
                for (int64_t i = 0; i < n_q * q_stride; i++) q[i] = rand_float(&seed) * 0.01f;
                for (int64_t i = 0; i < last * dim; i++) {
                    keys[i] = rand_float(&seed);
                    values[i] = rand_float(&seed);
                }
                for (int64_t i = 0; i < n_q * out_stride; i++) got[i] = 7.0f;
                for (int64_t i = 0; i < n_q * SCORE_STRIDE; i++) s_got[i] = 7.0f;
                tr_attention_group(q, q_stride, keys, values, n_q, first, dim, scale, s_got, SCORE_STRIDE, got,
                                   out_stride);
                for (int64_t j = 0; j < n_q; j++) {
                    tr_attention_head(q + j * q_stride, keys, values, dim, 0, first + j, dim, scale, s_want, want);
                    if (memcmp(got + j * out_stride, want, (size_t)dim * sizeof(float)) != 0) bad++;
                    if (memcmp(s_got + j * SCORE_STRIDE, s_want, (size_t)(first + j) * sizeof(float)) != 0) bad++;
                    for (int64_t d = dim; d < out_stride; d++) bad += got[j * out_stride + d] != 7.0f;
                    /* a query's row of scores is as long as its context: not a float further */
                    for (int64_t t = first + j; t < SCORE_STRIDE; t++) bad += s_got[j * SCORE_STRIDE + t] != 7.0f;
                }
                cases++;
            }
        }
    }
    TR_CHECK_EQ_INT(bad, 0);
    printf("  tr_attention_group: %d groups (1..19 queries, first context 1..250, head_dim 1..128) identical to "
           "tr_attention_head query by query\n",
           cases);
}

/* ---- tr_matmul: identical output regardless of thread count --------------- */

static void test_matmul_thread_determinism(void) {
    const int64_t rows = 37, cols = 53, n_tokens = 3;
    float *wdata = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
    float *x = (float *)malloc((size_t)n_tokens * (size_t)cols * sizeof(float));
    TR_CHECK(wdata != NULL && x != NULL);
    if (wdata == NULL || x == NULL) {
        free(wdata);
        free(x);
        return;
    }

    unsigned seed = 42u;
    for (int64_t i = 0; i < rows * cols; i++) wdata[i] = rand_float(&seed);
    for (int64_t i = 0; i < n_tokens * cols; i++) x[i] = rand_float(&seed);

    tr_mat w;
    w.type = TR_TYPE_F32;
    w.rows = rows;
    w.cols = cols;
    w.data = wdata;

    int thread_counts[3] = {1, 2, 8};
    float *outputs[3] = {NULL, NULL, NULL};
    int all_ok = 1;
    for (int t = 0; t < 3; t++) {
        tr_pool *p = tr_pool_create(thread_counts[t]);
        TR_CHECK(p != NULL);
        if (p == NULL) {
            all_ok = 0;
            continue;
        }
        float *y = (float *)malloc((size_t)rows * (size_t)n_tokens * sizeof(float));
        TR_CHECK(y != NULL);
        outputs[t] = y;
        if (y != NULL) tr_matmul(p, &w, x, n_tokens, y);
        tr_pool_destroy(p);
    }

    if (all_ok && outputs[0] != NULL && outputs[1] != NULL && outputs[2] != NULL) {
        size_t bytes = (size_t)rows * (size_t)n_tokens * sizeof(float);
        TR_CHECK(memcmp(outputs[0], outputs[1], bytes) == 0);
        TR_CHECK(memcmp(outputs[0], outputs[2], bytes) == 0);
    }

    for (int t = 0; t < 3; t++) free(outputs[t]);
    free(wdata);
    free(x);
}

/* ---- tr_matmul_grouped: each element is one dot_row of its own group -------- */

static void test_matmul_grouped(void) {
    enum { G = 6, ROWS = 21, COLS = 64, N = 45 };
    /* empty groups first, in the middle and last; a one-row group; runs longer than
     * TR_MATMUL_TILE; ROWS odd, so thread chunks start and end inside input rows */
    static const int64_t offsets[G + 1] = {0, 0, 1, 20, 20, 45, 45};
    static const tr_type types[2] = {TR_TYPE_F32, TR_TYPE_Q8_0};
    static const int threads[4] = {1, 2, 3, 7};
    const tr_kernels *K = tr_kernels_get();

    for (int ti = 0; ti < 2; ti++) {
        tr_type type = types[ti];
        size_t rb = tr_row_bytes(type, COLS);
        unsigned char *wdata = (unsigned char *)malloc((size_t)G * ROWS * rb);
        float *x = (float *)malloc((size_t)N * COLS * sizeof(float));
        float *ref = (float *)malloc((size_t)N * ROWS * sizeof(float));
        float *y = (float *)malloc((size_t)N * ROWS * sizeof(float));
        TR_CHECK(wdata != NULL && x != NULL && ref != NULL && y != NULL);
        if (wdata == NULL || x == NULL || ref == NULL || y == NULL) {
            free(wdata);
            free(x);
            free(ref);
            free(y);
            return;
        }

        unsigned seed = 7u + (unsigned)ti;
        for (int64_t i = 0; i < G * ROWS; i++) {
            unsigned char *row = wdata + (size_t)i * rb;
            if (type == TR_TYPE_Q8_0) {
                fill_q8_0(row, COLS / 32, &seed, 0);
            } else {
                for (int64_t j = 0; j < COLS; j++) {
                    float v = rand_float(&seed);
                    memcpy(row + (size_t)j * sizeof(float), &v, sizeof v);
                }
            }
        }
        for (int64_t i = 0; i < N * COLS; i++) x[i] = rand_float(&seed);

        tr_mat w[G];
        for (int64_t g = 0; g < G; g++) {
            w[g].type = type;
            w[g].rows = ROWS;
            w[g].cols = COLS;
            w[g].data = wdata + (size_t)g * ROWS * rb;
            for (int64_t p = offsets[g]; p < offsets[g + 1]; p++)
                for (int64_t r = 0; r < ROWS; r++)
                    ref[p * ROWS + r] = K->dot_row[type]((const unsigned char *)w[g].data + (size_t)r * rb,
                                                        x + p * COLS, COLS);
        }

        memset(y, 0, (size_t)N * ROWS * sizeof(float));
        tr_matmul_grouped(NULL, w, offsets, G, x, y);
        TR_CHECK(memcmp(y, ref, (size_t)N * ROWS * sizeof(float)) == 0);
        for (int t = 0; t < 4; t++) {
            tr_pool *pool = tr_pool_create(threads[t]);
            TR_CHECK(pool != NULL);
            if (pool == NULL) continue;
            memset(y, 0, (size_t)N * ROWS * sizeof(float));
            tr_matmul_grouped(pool, w, offsets, G, x, y);
            TR_CHECK(memcmp(y, ref, (size_t)N * ROWS * sizeof(float)) == 0);
            /* one group alone through tr_matmul */
            memset(y, 0, (size_t)N * ROWS * sizeof(float));
            tr_matmul(pool, &w[4], x + 20 * COLS, 25, y);
            TR_CHECK(memcmp(y, ref + 20 * ROWS, 25 * ROWS * sizeof(float)) == 0);
            tr_pool_destroy(pool);
        }
        free(wdata);
        free(x);
        free(ref);
        free(y);
    }
}

int main(void) {
    tr_kernels_init();

    test_half_to_float();
    test_q8_0_dequant();
    test_dot_f32_contract();
    test_dot_row_f16();
    test_dot_row_q8_0();
    test_tiers_match_scalar();
    test_rope_table();
    test_swiglu_threads();
    test_attention_head();
    test_attention_group();
    test_matmul_thread_determinism();
    test_matmul_grouped();

    TR_TEST_EXIT();
}
