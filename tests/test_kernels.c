/* test_kernels.c — tests for src/kernels/kernels.c: half->float, Q8_0, Q4_K and Q6_K dequant,
 * the 16-lane reduction contract (kernels.h) against an independent in-test
 * implementation, dot_row == dot_f32(dequantized), every SIMD tier == scalar,
 * rope/swiglu/attention == their first per-element definitions, tr_matmul
 * thread-count determinism, tr_matmul_grouped == one dot_row per element, the decode's one-token
 * matmul == one dot_row per row at every thread count. */
#include "test.h"

#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"
#include "../src/base/threads.h"
#include "../src/base/cpu.h"

#include <math.h>
#include <stdatomic.h>
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

/* ---- Q4_K dequant on a block packed as ggml's quantize_row_q4_K_ref packs it ---- */

/* d = 1 and dmin = 0.5 make every weight exact: sc * q - m / 2. The scales of sub-blocks 4..7
 * use their two high bits (37, 50, 63), which live in bytes 0..7 beside the low ones. */
static void test_q4_k_dequant(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    static const int ls[8] = {1, 17, 33, 63, 2, 37, 50, 63}, lm[8] = {0, 5, 63, 20, 41, 7, 60, 16};
    unsigned char blk[2 * 144] = {0};
    uint16_t d = 0x3C00, dmin = 0x3800; /* 1.0 and 0.5 in binary16 */
    memcpy(blk, &d, 2);
    memcpy(blk + 2, &dmin, 2);
    unsigned char *sc = blk + 4, *qs = blk + 16;
    for (int j = 0; j < 8; j++) {
        if (j < 4) {
            sc[j] = (unsigned char)ls[j];
            sc[j + 4] = (unsigned char)lm[j];
        } else {
            sc[j + 4] = (unsigned char)((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
            sc[j - 4] |= (unsigned char)((ls[j] >> 4) << 6);
            sc[j] |= (unsigned char)((lm[j] >> 4) << 6);
        }
    }
    for (int i = 0; i < 256; i++) {
        unsigned q = (unsigned)(i * 7 + 3) % 16u;
        qs[32 * (i / 64) + i % 32] |= (unsigned char)((i / 32) % 2 ? q << 4 : q);
    }
    memcpy(blk + 144, blk, 144);   /* a second block, the same */

    float out[512];
    K->dequant_row[TR_TYPE_Q4_K](blk, out, 512);
    int checked = 0;
    for (int i = 0; i < 512; i++, checked++) {
        int j = (i % 256) / 32;
        float expect = (float)(ls[j] * ((i % 256 * 7 + 3) % 16)) - 0.5f * (float)lm[j];
        TR_CHECK(bit_eq(out[i], expect));
    }
    TR_CHECK_EQ_INT(checked, 512);
}

/* ---- Q6_K dequant on blocks packed as ggml's quantize_row_q6_K_ref packs them ---- */

/* L = q + 32 runs through all 64 values, the scales through both signs and the int8 ends; d = 1
 * then 0.5 (read at the block's end) make every weight exact: d * sc * (L - 32). */
static void test_q6_k_dequant(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    static const int sc[16] = {1, -1, 127, -128, 2, 37, -50, 63, 0, 5, -7, 100, -3, 11, 64, -64};
    unsigned char blk[2 * 210] = {0};
    for (int b = 0; b < 2; b++) {
        unsigned char *ql = blk + 210 * b, *qh = ql + 128;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; l++) {
                int L[4];
                for (int c = 0; c < 4; c++) L[c] = ((j + l + 32 * c) * 7 + 3) % 64;
                ql[l] = (unsigned char)((L[0] & 0xF) | ((L[2] & 0xF) << 4));
                ql[l + 32] = (unsigned char)((L[1] & 0xF) | ((L[3] & 0xF) << 4));
                qh[l] = (unsigned char)((L[0] >> 4) | ((L[1] >> 4) << 2) | ((L[2] >> 4) << 4) | ((L[3] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
        for (int j = 0; j < 16; j++) blk[210 * b + 192 + j] = (unsigned char)(int8_t)sc[j];
        uint16_t d = b == 0 ? 0x3C00 : 0x3800; /* 1.0 and 0.5 in binary16 */
        memcpy(blk + 210 * b + 208, &d, 2);
    }

    float out[512];
    K->dequant_row[TR_TYPE_Q6_K](blk, out, 512);
    int checked = 0;
    for (int i = 0; i < 512; i++, checked++) {
        float d = i < 256 ? 1.0f : 0.5f;
        /* (d * sc) * (q - 32) in floats: scale 0 times a negative quant is -0, as in ggml */
        float expect = (d * (float)sc[(i % 256) / 16]) * (float)((i % 256 * 7 + 3) % 64 - 32);
        TR_CHECK(bit_eq(out[i], expect));
    }
    TR_CHECK_EQ_INT(checked, 512);
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

/* Q4_K: the weight in the dot is the dequantized weight, element by element, and dot_row_x4 is
 * four dot_rows */
static void test_dot_row_q4_k(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    enum { N = 512 };                  /* two blocks */
    static unsigned char row[2 * 144];
    static float x[4 * N], dequant[N];
    unsigned seed = 4242u;
    for (int i = 0; i < 2 * 144; i++) row[i] = (unsigned char)(next_rand(&seed) >> 16);
    for (int b = 0; b < 2; b++) {
        uint16_t d = (uint16_t)(0x2C00u + (next_rand(&seed) & 0x3FFu));    /* ~0.06..0.12 */
        uint16_t dmin = (uint16_t)(0x2800u + (next_rand(&seed) & 0x3FFu)); /* ~0.03..0.06 */
        memcpy(row + 144 * b, &d, 2);
        memcpy(row + 144 * b + 2, &dmin, 2);
    }
    for (int i = 0; i < 4 * N; i++) x[i] = rand_float(&seed);

    K->dequant_row[TR_TYPE_Q4_K](row, dequant, N);
    TR_CHECK(bit_eq(K->dot_row[TR_TYPE_Q4_K](row, x, N), ref_dot_f32(dequant, x, N)));
    float out[TR_DOT_TOKENS];
    K->dot_row_x4[TR_TYPE_Q4_K](row, x, N, N, out);
    for (int t = 0; t < TR_DOT_TOKENS; t++) TR_CHECK(bit_eq(out[t], ref_dot_f32(dequant, x + t * N, N)));
}

/* Q6_K: the same two checks */
static void test_dot_row_q6_k(void) {
    const tr_kernels *K = tr_kernels_tier("scalar");
    TR_CHECK(K != NULL);
    if (K == NULL) return;

    enum { N = 512 };                  /* two blocks */
    static unsigned char row[2 * 210];
    static float x[4 * N], dequant[N];
    unsigned seed = 6262u;
    for (int i = 0; i < 2 * 210; i++) row[i] = (unsigned char)(next_rand(&seed) >> 16);
    for (int b = 0; b < 2; b++) {
        uint16_t d = (uint16_t)(0x1C00u + (next_rand(&seed) & 0x3FFu)); /* ~0.004..0.008 */
        memcpy(row + 210 * b + 208, &d, 2);
    }
    for (int i = 0; i < 4 * N; i++) x[i] = rand_float(&seed);

    K->dequant_row[TR_TYPE_Q6_K](row, dequant, N);
    TR_CHECK(bit_eq(K->dot_row[TR_TYPE_Q6_K](row, x, N), ref_dot_f32(dequant, x, N)));
    float out[TR_DOT_TOKENS];
    K->dot_row_x4[TR_TYPE_Q6_K](row, x, N, N, out);
    for (int t = 0; t < TR_DOT_TOKENS; t++) TR_CHECK(bit_eq(out[t], ref_dot_f32(dequant, x + t * N, N)));
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

/* Q4_K blocks: every byte random (every nibble, scale and min); d and dmin ordinary, or with
 * `special` now and then any 16 bits */
static void fill_q4_k(unsigned char *row, int64_t nb, unsigned *seed, int special) {
    for (int64_t b = 0; b < nb; b++) {
        unsigned char *blk = row + b * 144;
        for (int i = 4; i < 144; i++) blk[i] = (unsigned char)(next_rand(seed) >> 16);
        for (int h = 0; h < 2; h++) {
            unsigned r = next_rand(seed);
            uint16_t bits = (uint16_t)((r >> 8) & 0xFFFFu);
            if (!special || (r & 63u) != 0) bits = (uint16_t)((bits & 0x83FFu) | ((8u + (r >> 3) % 14u) << 10));
            memcpy(blk + 2 * h, &bits, 2);
        }
    }
}

/* Q6_K blocks: every byte random (every quant and int8 scale); d as Q4_K's */
static void fill_q6_k(unsigned char *row, int64_t nb, unsigned *seed, int special) {
    for (int64_t b = 0; b < nb; b++) {
        unsigned char *blk = row + b * 210;
        for (int i = 0; i < 208; i++) blk[i] = (unsigned char)(next_rand(seed) >> 16);
        unsigned r = next_rand(seed);
        uint16_t bits = (uint16_t)((r >> 8) & 0xFFFFu);
        if (!special || (r & 63u) != 0) bits = (uint16_t)((bits & 0x83FFu) | ((4u + (r >> 3) % 14u) << 10));
        memcpy(blk + 208, &bits, 2);
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

/* Two rows against the same four input rows: each sum the scalar x4 of its own row, bit for bit
 * (the scalar table has no dot_row2_x4: two dot_row_x4 are its definition). */
static void row2_diffs(const tr_kernels *K, const tr_kernels *S, tr_type type, const unsigned char *r0,
                       const unsigned char *r1, const float *x, int64_t n, int *bad) {
    float o[2 * TR_DOT_TOKENS], s0[TR_DOT_TOKENS], s1[TR_DOT_TOKENS];
    K->dot_row2_x4[type](r0, r1, x, n + 3, n, o);
    S->dot_row_x4[type](r0, x, n + 3, n, s0);
    S->dot_row_x4[type](r1, x, n + 3, n, s1);
    for (int t = 0; t < TR_DOT_TOKENS; t++) {
        if (!same_float(o[t], s0[t])) (*bad)++;
        if (!same_float(o[TR_DOT_TOKENS + t], s1[t])) (*bad)++;
    }
}

/* Two rows against one input row (dot_row2, the decode's pair): each result scalar's dot_row of its
 * own row, bit for bit. Counted: a tier whose dot_row2 is never compared proves nothing. */
static long g_pair_compared;
static void pair_diffs(const tr_kernels *K, const tr_kernels *S, tr_type type, const unsigned char *r0,
                       const unsigned char *r1, const float *x, int64_t n, int *bad) {
    float o[2];
    K->dot_row2[type](r0, r1, x, n, o);
    if (!same_float(o[0], S->dot_row[type](r0, x, n))) (*bad)++;
    if (!same_float(o[1], S->dot_row[type](r1, x, n))) (*bad)++;
    g_pair_compared++;
}

/* Two rows against the same eight input rows (x8, a stride of n + 3 floats): each sum scalar's
 * dot_row of its own row and input row, bit for bit. Counts its calls: a tier whose x8 is never
 * compared proves nothing (docs/LESSONS.md #43). */
enum { X8_MAX_N = 4096 };
static float g_x8_in[TR_DOT_TOKENS_WIDE * (X8_MAX_N + 3)];
static long g_x8_compared;
static void row2x8_diffs(const tr_kernels *K, const tr_kernels *S, tr_type type, const unsigned char *r0,
                         const unsigned char *r1, int64_t n, int *bad) {
    float o[2 * TR_DOT_TOKENS_WIDE];
    K->dot_row2_x8[type](r0, r1, g_x8_in, n + 3, n, o);
    for (int t = 0; t < TR_DOT_TOKENS_WIDE; t++) {
        if (!same_float(o[t], S->dot_row[type](r0, g_x8_in + t * (n + 3), n))) (*bad)++;
        if (!same_float(o[TR_DOT_TOKENS_WIDE + t], S->dot_row[type](r1, g_x8_in + t * (n + 3), n))) (*bad)++;
    }
    g_x8_compared++;
}

/* A tile of a prompt's attention: 4 query rows of a (len + 5 apart: not b's stride) against the 4
 * rows of b `stride` apart. Every score must be scalar's dot_f32 of its pair times the scale, and
 * the tier's 4x4 must equal scalar's 4x4; every one of the 4 outputs (rows of y, len + 2 apart,
 * the floats between them never touched) scalar's axpy_f32_x4 with its own 4 weights. Counts its
 * calls: a tile never compared proves nothing (docs/LESSONS.md #43). `avail`: floats of a from a on. */
static long g_tile_compared;
static void tile_diffs(const tr_kernels *K, const tr_kernels *S, const float *a, const float *b, int64_t stride,
                       int64_t len, int64_t avail, unsigned *seed, int special, int *bad) {
    enum { OUT_STRIDE = TR_ATTN_X + 3, Y_MAX = 4 * (1000 + 2) };
    static float yk[Y_MAX], ys[Y_MAX];
    const int64_t a_stride = len + 5, y_stride = len + 2;
    if ((TR_ATTN_X - 1) * a_stride + len > avail || TR_ATTN_X * y_stride > Y_MAX) return; /* the rows must fit */
    float scale = special ? rand_special(seed, 1) : 0.0883883f;
    float ok[TR_ATTN_X * OUT_STRIDE], os[TR_ATTN_X * OUT_STRIDE], w[TR_ATTN_X * OUT_STRIDE];
    for (int i = 0; i < TR_ATTN_X * OUT_STRIDE; i++) ok[i] = os[i] = 7.0f;
    K->dot_f32_4x4(a, a_stride, b, stride, len, scale, ok, OUT_STRIDE);
    S->dot_f32_4x4(a, a_stride, b, stride, len, scale, os, OUT_STRIDE);
    for (int i = 0; i < TR_ATTN_X; i++)
        for (int j = 0; j < OUT_STRIDE; j++) {
            float want = j < TR_ATTN_X ? S->dot_f32(a + i * a_stride, b + j * stride, len) * scale : 7.0f;
            if (!same_float(ok[i * OUT_STRIDE + j], want) || !same_float(ok[i * OUT_STRIDE + j], os[i * OUT_STRIDE + j]))
                (*bad)++;
        }
    for (int i = 0; i < TR_ATTN_X * OUT_STRIDE; i++) w[i] = rand_special(seed, special);
    for (int64_t i = 0; i < TR_ATTN_X * y_stride; i++) yk[i] = ys[i] = rand_special(seed, special);
    K->axpy_f32_4x4(yk, y_stride, b, stride, w, OUT_STRIDE, len);
    for (int i = 0; i < TR_ATTN_X; i++) S->axpy_f32_x4(ys + i * y_stride, b, stride, w + i * OUT_STRIDE, len);
    for (int64_t i = 0; i < TR_ATTN_X * y_stride; i++)
        if (!same_float(yk[i], ys[i])) {
            (*bad)++;
            break;
        }
    g_tile_compared++;
}

/* Runs K and S on the same inputs; counts the calls whose results differ. */
static void count_diffs(const tr_kernels *K, const tr_kernels *S, unsigned seed, int *bad_dot, int *bad_row,
                        int *bad_axpy, int *bad_x4) {
    static float a[CMP_MAX_N], b[CMP_MAX_N], yk[CMP_MAX_N], ys[CMP_MAX_N], y4[CMP_MAX_N];
    static unsigned char row[(4096 / 32) * 34], row2[(4096 / 32) * 34], hrow[2 * CMP_MAX_N];
    static const int64_t big[] = {255, 256, 257, 1000, 2048, 4096};

    *bad_dot = *bad_row = *bad_axpy = *bad_x4 = 0;
    for (int round = 0; round < 40; round++) {
        for (int i = 0; i < CMP_MAX_N; i++) {
            a[i] = rand_special(&seed, round % 2);
            b[i] = rand_special(&seed, round % 2);
        }
        for (size_t i = 0; i < sizeof g_x8_in / sizeof g_x8_in[0]; i++) g_x8_in[i] = rand_special(&seed, round % 2);
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
                tile_diffs(K, S, a + off, b + off, stride, len, CMP_MAX_N - off, &seed, round % 2, bad_x4);
            }
        }
        for (int64_t nb = 1; nb <= 128; nb += (nb < 8 ? 1 : 15)) {
            fill_q8_0(row, nb, &seed, round % 2);
            int64_t n = nb * 32;
            if (!same_float(K->dot_row[TR_TYPE_Q8_0](row, b, n), S->dot_row[TR_TYPE_Q8_0](row, b, n))) (*bad_row)++;
            if (K->dot_row2[TR_TYPE_Q8_0] != NULL) {
                fill_q8_0(row2, nb, &seed, round % 2);
                pair_diffs(K, S, TR_TYPE_Q8_0, row, row2, b, n, bad_row);
            }
            /* the same row against 4 consecutive input rows: same as the tier's own
             * dot_row on each of them, and as scalar's (4 * n floats must fit in b) */
            if (K->dot_row2_x4[TR_TYPE_Q8_0] != NULL && 4 * (n + 3) <= CMP_MAX_N) {
                fill_q8_0(row2, nb, &seed, round % 2);
                row2_diffs(K, S, TR_TYPE_Q8_0, row, row2, b, n, bad_row);
            }
            if (K->dot_row2_x8[TR_TYPE_Q8_0] != NULL) {
                fill_q8_0(row2, nb, &seed, round % 2);
                row2x8_diffs(K, S, TR_TYPE_Q8_0, row, row2, n, bad_row);
            }
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
        /* Q4_K rows of 1 to 16 blocks (256 to 4096 elements), the same two checks */
        for (int64_t nb = 1; nb <= 16; nb++) {
            fill_q4_k(row, nb, &seed, round % 2);
            int64_t n = nb * 256;
            if (!same_float(K->dot_row[TR_TYPE_Q4_K](row, b, n), S->dot_row[TR_TYPE_Q4_K](row, b, n))) (*bad_row)++;
            if (K->dot_row2[TR_TYPE_Q4_K] != NULL) {
                fill_q4_k(row2, nb, &seed, round % 2);
                pair_diffs(K, S, TR_TYPE_Q4_K, row, row2, b, n, bad_row);
            }
            if (K->dot_row2_x4[TR_TYPE_Q4_K] != NULL && 4 * (n + 3) <= CMP_MAX_N) {
                fill_q4_k(row2, nb, &seed, round % 2);
                row2_diffs(K, S, TR_TYPE_Q4_K, row, row2, b, n, bad_row);
            }
            if (K->dot_row2_x8[TR_TYPE_Q4_K] != NULL) {
                fill_q4_k(row2, nb, &seed, round % 2);
                row2x8_diffs(K, S, TR_TYPE_Q4_K, row, row2, n, bad_row);
            }
            if (K->dot_row_x4[TR_TYPE_Q4_K] != NULL && 4 * n <= CMP_MAX_N) {
                float xk[TR_DOT_TOKENS], xs[TR_DOT_TOKENS];
                K->dot_row_x4[TR_TYPE_Q4_K](row, b, n, n, xk);
                S->dot_row_x4[TR_TYPE_Q4_K](row, b, n, n, xs);
                for (int t = 0; t < TR_DOT_TOKENS; t++) {
                    if (!same_float(xk[t], xs[t])) (*bad_row)++;
                    if (!same_float(xk[t], K->dot_row[TR_TYPE_Q4_K](row, b + t * n, n))) (*bad_row)++;
                }
            }
        }
        /* Q6_K rows of 1 to 16 blocks, the same two checks */
        for (int64_t nb = 1; nb <= 16; nb++) {
            fill_q6_k(row, nb, &seed, round % 2);
            int64_t n = nb * 256;
            if (!same_float(K->dot_row[TR_TYPE_Q6_K](row, b, n), S->dot_row[TR_TYPE_Q6_K](row, b, n))) (*bad_row)++;
            if (K->dot_row2[TR_TYPE_Q6_K] != NULL) {
                fill_q6_k(row2, nb, &seed, round % 2);
                pair_diffs(K, S, TR_TYPE_Q6_K, row, row2, b, n, bad_row);
            }
            if (K->dot_row2_x4[TR_TYPE_Q6_K] != NULL && 4 * (n + 3) <= CMP_MAX_N) {
                fill_q6_k(row2, nb, &seed, round % 2);
                row2_diffs(K, S, TR_TYPE_Q6_K, row, row2, b, n, bad_row);
            }
            if (K->dot_row2_x8[TR_TYPE_Q6_K] != NULL) {
                fill_q6_k(row2, nb, &seed, round % 2);
                row2x8_diffs(K, S, TR_TYPE_Q6_K, row, row2, n, bad_row);
            }
            if (K->dot_row_x4[TR_TYPE_Q6_K] != NULL && 4 * n <= CMP_MAX_N) {
                float xk[TR_DOT_TOKENS], xs[TR_DOT_TOKENS];
                K->dot_row_x4[TR_TYPE_Q6_K](row, b, n, n, xk);
                S->dot_row_x4[TR_TYPE_Q6_K](row, b, n, n, xs);
                for (int t = 0; t < TR_DOT_TOKENS; t++) {
                    if (!same_float(xk[t], xs[t])) (*bad_row)++;
                    if (!same_float(xk[t], K->dot_row[TR_TYPE_Q6_K](row, b + t * n, n))) (*bad_row)++;
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

static float wrong_dot_row_q4_k(const void *row, const float *x, int64_t n) {
    static float w[CMP_MAX_N];
    tr_kernels_tier("scalar")->dequant_row[TR_TYPE_Q4_K](row, w, n);
    return wrong_dot_f32(w, x, n);
}

static float wrong_dot_row_q6_k(const void *row, const float *x, int64_t n) {
    static float w[CMP_MAX_N];
    tr_kernels_tier("scalar")->dequant_row[TR_TYPE_Q6_K](row, w, n);
    return wrong_dot_f32(w, x, n);
}

/* Two rows, each through the wrong dot of its type: count_diffs must see a wrong dot_row2_x4 */
#define WRONG_ROW2(name) \
    static void wrong_row2_##name(const void *r0, const void *r1, const float *x, int64_t stride, int64_t n, \
                                  float *out) { \
        for (int t = 0; t < TR_DOT_TOKENS; t++) { \
            out[t] = wrong_dot_row_##name(r0, x + t * stride, n); \
            out[TR_DOT_TOKENS + t] = wrong_dot_row_##name(r1, x + t * stride, n); \
        } \
    }
WRONG_ROW2(q8_0)
WRONG_ROW2(q4_k)
WRONG_ROW2(q6_k)

/* the same against eight input rows: count_diffs must see a wrong dot_row2_x8 */
#define WRONG_ROW2X8(name) \
    static void wrong_row2x8_##name(const void *r0, const void *r1, const float *x, int64_t stride, int64_t n, \
                                    float *out) { \
        for (int t = 0; t < TR_DOT_TOKENS_WIDE; t++) { \
            out[t] = wrong_dot_row_##name(r0, x + t * stride, n); \
            out[TR_DOT_TOKENS_WIDE + t] = wrong_dot_row_##name(r1, x + t * stride, n); \
        } \
    }
WRONG_ROW2X8(q8_0)
WRONG_ROW2X8(q4_k)
WRONG_ROW2X8(q6_k)

/* two rows against one input row: count_diffs must see a wrong dot_row2 */
#define WRONG_PAIR(name) \
    static void wrong_pair_##name(const void *r0, const void *r1, const float *x, int64_t n, float *out) { \
        out[0] = wrong_dot_row_##name(r0, x, n); \
        out[1] = wrong_dot_row_##name(r1, x, n); \
    }
WRONG_PAIR(q8_0)
WRONG_PAIR(q4_k)
WRONG_PAIR(q6_k)

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

/* Wrong only in rounding: the scale folded into the query before the products, what a tile
 * that saves the last multiply would do. */
static void wrong_dot_f32_4x4(const float *a, int64_t a_stride, const float *b, int64_t stride, int64_t n, float scale,
                              float *out, int64_t out_stride) {
    for (int i = 0; i < TR_ATTN_X; i++)
        for (int j = 0; j < TR_ATTN_X; j++) {
            float lane[TR_LANES] = {0};
            for (int64_t k = 0; k < n; k++) lane[k % TR_LANES] += (a[i * a_stride + k] * scale) * b[j * stride + k];
            out[i * out_stride + j] = tr_lane_combine(lane);
        }
}

/* Wrong only in rounding: each output's four products added to each other first */
static void wrong_axpy_f32_4x4(float *y, int64_t y_stride, const float *x, int64_t stride, const float *a,
                               int64_t a_stride, int64_t n) {
    for (int i = 0; i < TR_ATTN_X; i++) wrong_axpy_f32_x4(y + i * y_stride, x, stride, a + i * a_stride, n);
}

/* ---- the weight in every dot is dequant_row's float, in every tier ------------------------ */
/* dot_row(row, x) == dot_f32(dequant_row(row), x) bit for bit (NaN: any NaN), for every type and
 * tier, ordinary and special values, rows up to 4096 (and 4113 for F16's tail lanes): a matmul
 * may then decode a panel of rows once and run F32 over every token without moving a bit
 * (docs/MEASUREMENTS.md question 63). test_dot_row_* hold it for the scalar tier on two blocks.
 * Branch covered: each tier's own dot_row and dot_f32 (never scalar's through a NULL); a panel
 * one ulp off in every weight must fail it for each type, and the comparisons are counted. */
enum { DQ_TYPES = 4 };
static const tr_type g_dq_types[DQ_TYPES] = {TR_TYPE_F16, TR_TYPE_Q8_0, TR_TYPE_Q4_K, TR_TYPE_Q6_K};

static void fill_row_of(tr_type type, unsigned char *row, int64_t n, unsigned *seed, int special) {
    if (type == TR_TYPE_F16) fill_f16(row, n, seed, special);
    if (type == TR_TYPE_Q8_0) fill_q8_0(row, n / 32, seed, special);
    if (type == TR_TYPE_Q4_K) fill_q4_k(row, n / 256, seed, special);
    if (type == TR_TYPE_Q6_K) fill_q6_k(row, n / 256, seed, special);
}

/* per type, the rows whose dot differs from dot_f32 over the decoded panel; with `wrong`, every
 * finite nonzero weight of the panel one ulp up (a panel decoded otherwise than the dot) */
static void dequant_dot_diffs(const tr_kernels *K, int wrong, int bad[DQ_TYPES], long *compared) {
    static unsigned char row[2 * CMP_MAX_N];
    static float x[CMP_MAX_N], w[CMP_MAX_N];
    static const int64_t sizes[] = {256, 512, 2048, 4096, 4113};
    unsigned seed = 2463u;
    for (int t = 0; t < DQ_TYPES; t++) {
        tr_type type = g_dq_types[t];
        bad[t] = 0;
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            int64_t n = sizes[s];
            if (type != TR_TYPE_F16 && n % 256 != 0) continue;
            for (int special = 0; special < 2; special++) {
                fill_row_of(type, row, n, &seed, special);
                for (int64_t i = 0; i < n; i++) x[i] = rand_special(&seed, special);
                K->dequant_row[type](row, w, n);
                if (wrong)
                    for (int64_t i = 0; i < n; i++)
                        if (isfinite(w[i]) && w[i] != 0.0f) w[i] = nextafterf(w[i], INFINITY);
                if (!same_float(K->dot_row[type](row, x, n), K->dot_f32(w, x, n))) bad[t]++;
                (*compared)++;
            }
        }
    }
}

static void test_dequant_is_the_dots_weight(void) {
    static const char *const tiers[] = {"scalar", "avx2", "avx512"};
    int bad[DQ_TYPES];
    for (size_t i = 0; i < sizeof tiers / sizeof tiers[0]; i++) {
        const tr_kernels *K = tr_kernels_tier(tiers[i]);
        if (K == NULL) continue; /* this CPU (or TR_CPU_MAX, tools/tier_check.sh) lacks the tier */
        long compared = 0;
        dequant_dot_diffs(K, 0, bad, &compared);
        for (int t = 0; t < DQ_TYPES; t++) TR_CHECK_EQ_INT(bad[t], 0);
        TR_CHECK(compared > 0);
        dequant_dot_diffs(K, 1, bad, &compared);
        for (int t = 0; t < DQ_TYPES; t++) TR_CHECK(bad[t] > 0);
    }
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
    /* and each wrong tile alone: the attention's comparisons see them */
    wrong = *S;
    wrong.dot_f32_4x4 = wrong_dot_f32_4x4;
    count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
    TR_CHECK_EQ_INT(bad_dot + bad_row + bad_axpy, 0);
    TR_CHECK(bad_x4 > 0);
    wrong = *S;
    wrong.axpy_f32_4x4 = wrong_axpy_f32_4x4;
    count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
    TR_CHECK_EQ_INT(bad_dot + bad_row + bad_axpy, 0);
    TR_CHECK(bad_x4 > 0);
    /* and a wrong F16 row alone: the Q8_0 rows above must not be what made bad_row move */
    wrong = *S;
    wrong.dot_row[TR_TYPE_F16] = wrong_dot_row_f16;
    count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
    TR_CHECK_EQ_INT(bad_dot + bad_axpy + bad_x4, 0);
    TR_CHECK(bad_row > 0);
    /* and a wrong Q4_K row alone: its comparisons are live too */
    wrong = *S;
    wrong.dot_row[TR_TYPE_Q4_K] = wrong_dot_row_q4_k;
    count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
    TR_CHECK_EQ_INT(bad_dot + bad_axpy + bad_x4, 0);
    TR_CHECK(bad_row > 0);
    /* and a wrong Q6_K row alone */
    wrong = *S;
    wrong.dot_row[TR_TYPE_Q6_K] = wrong_dot_row_q6_k;
    count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
    TR_CHECK_EQ_INT(bad_dot + bad_axpy + bad_x4, 0);
    TR_CHECK(bad_row > 0);
    /* and a wrong two-row kernel of each quantized type alone: the scalar table has none, so
     * these are the only calls of dot_row2_x4 the wrong table makes */
    void (*const wrong_row2[3])(const void *, const void *, const float *, int64_t, int64_t, float *) = {
        wrong_row2_q8_0, wrong_row2_q4_k, wrong_row2_q6_k};
    static const tr_type row2_types[3] = {TR_TYPE_Q8_0, TR_TYPE_Q4_K, TR_TYPE_Q6_K};
    for (int i = 0; i < 3; i++) {
        wrong = *S;
        wrong.dot_row2_x4[row2_types[i]] = wrong_row2[i];
        count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
        TR_CHECK_EQ_INT(bad_dot + bad_axpy + bad_x4, 0);
        TR_CHECK(bad_row > 0);
    }
    /* and a wrong eight-token kernel of each quantized type alone (the scalar table has none either) */
    void (*const wrong_row2x8[3])(const void *, const void *, const float *, int64_t, int64_t, float *) = {
        wrong_row2x8_q8_0, wrong_row2x8_q4_k, wrong_row2x8_q6_k};
    for (int i = 0; i < 3; i++) {
        wrong = *S;
        wrong.dot_row2_x8[row2_types[i]] = wrong_row2x8[i];
        count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
        TR_CHECK_EQ_INT(bad_dot + bad_axpy + bad_x4, 0);
        TR_CHECK(bad_row > 0);
    }
    /* and a wrong one-token pair of each quantized type alone (the scalar table has none either) */
    void (*const wrong_pair[3])(const void *, const void *, const float *, int64_t, float *) = {
        wrong_pair_q8_0, wrong_pair_q4_k, wrong_pair_q6_k};
    for (int i = 0; i < 3; i++) {
        wrong = *S;
        wrong.dot_row2[row2_types[i]] = wrong_pair[i];
        count_diffs(&wrong, S, 7u, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
        TR_CHECK_EQ_INT(bad_dot + bad_axpy + bad_x4, 0);
        TR_CHECK(bad_row > 0);
    }

    for (size_t t = 0; t < sizeof tiers / sizeof tiers[0]; t++) {
        const tr_kernels *K = tr_kernels_tier(tiers[t]);
        if (K == NULL) {
            printf("  tier %-8s not available on this CPU: skipped\n", tiers[t]);
            continue;
        }
        /* same numbers says nothing on WHICH function ran: that the tier's entries are its own,
         * and that the engine goes through them, is tests/test_tier_used.c */
        g_x8_compared = 0;
        g_pair_compared = 0;
        g_tile_compared = 0;
        count_diffs(K, S, 2026u + (unsigned)t, &bad_dot, &bad_row, &bad_axpy, &bad_x4);
        TR_CHECK_EQ_INT(bad_dot, 0);
        TR_CHECK_EQ_INT(bad_row, 0);
        TR_CHECK_EQ_INT(bad_axpy, 0);
        TR_CHECK_EQ_INT(bad_x4, 0);
        /* the tier's x8 kernels were compared, if it has any */
        int has_x8 = 0;
        for (int type = 0; type < TR_TYPE_COUNT; type++) has_x8 |= K->dot_row2_x8[type] != NULL;
        if (has_x8) TR_CHECK(g_x8_compared > 0);
        /* and its two-rows-one-token kernels, if it has any */
        int has_pair = 0;
        for (int type = 0; type < TR_TYPE_COUNT; type++) has_pair |= K->dot_row2[type] != NULL;
        if (has_pair) TR_CHECK(g_pair_compared > 0);
        TR_CHECK(g_tile_compared > 0); /* the prompt's tiles */
        printf("  tier %-8s dot_f32, axpy_f32, their x4 and 4x4 (%ld tiles), dot_row, dot_row2 (%ld calls), dot_row_x4, "
               "dot_row2_x4 and "
               "dot_row2_x8 (%ld calls) of f32, f16, q8_0, q4_k and q6_k %s\n",
               K->tier, g_tile_compared, g_pair_compared, g_x8_compared,
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
 * an x4 kernel; strides are wider than what is used. Both branches: groups of 1 (a decode token,
 * position by position) and groups of 2 to 19 (blocks and the x4 kernels). */
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

/* The roads of the tiled matmul, counted on the single-thread call: two rows against 8 tokens,
 * against 4, one row against 4, one dot. Equal results do not say which road ran (LESSONS #43). */
static const tr_kernels *g_mm_real;
static long g_mm_x8, g_mm_r2, g_mm_x4, g_mm_row;
static void mm_x8(const void *r0, const void *r1, const float *x, int64_t stride, int64_t n, float *out) {
    g_mm_x8++;
    g_mm_real->dot_row2_x8[TR_TYPE_Q8_0](r0, r1, x, stride, n, out);
}
static void mm_r2(const void *r0, const void *r1, const float *x, int64_t stride, int64_t n, float *out) {
    g_mm_r2++;
    g_mm_real->dot_row2_x4[TR_TYPE_Q8_0](r0, r1, x, stride, n, out);
}
static void mm_x4(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    g_mm_x4++;
    g_mm_real->dot_row_x4[TR_TYPE_Q8_0](row, x, stride, n, out);
}
static float mm_row(const void *row, const float *x, int64_t n) {
    g_mm_row++;
    return g_mm_real->dot_row[TR_TYPE_Q8_0](row, x, n);
}

static void test_matmul_grouped(void) {
    enum { G = 6, ROWS = 21, COLS = 64, N = 50 };
    /* empty groups first, in the middle and last; a one-row group; runs longer than
     * TR_MATMUL_TILE; ROWS odd, so thread chunks start and end inside input rows. The run of 29
     * is a tile of 16 then one of 13 (8 + 4 + 1 tokens), the run of 20 a tile of 16 then one of 4:
     * every road of the tiled matmul */
    static const int64_t offsets[G + 1] = {0, 0, 1, 30, 30, 50, 50};
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
        static tr_kernels counting;
        g_mm_real = K;
        counting = *K;
        if (K->dot_row2_x8[TR_TYPE_Q8_0] != NULL) counting.dot_row2_x8[TR_TYPE_Q8_0] = mm_x8;
        if (K->dot_row2_x4[TR_TYPE_Q8_0] != NULL) counting.dot_row2_x4[TR_TYPE_Q8_0] = mm_r2;
        if (K->dot_row_x4[TR_TYPE_Q8_0] != NULL) counting.dot_row_x4[TR_TYPE_Q8_0] = mm_x4;
        counting.dot_row[TR_TYPE_Q8_0] = mm_row;
        g_mm_x8 = g_mm_r2 = g_mm_x4 = g_mm_row = 0;
        tr_kernels_set_active(&counting);
        tr_matmul_grouped(NULL, w, offsets, G, x, y);
        tr_kernels_set_active(NULL);
        TR_CHECK(memcmp(y, ref, (size_t)N * ROWS * sizeof(float)) == 0);
        if (type == TR_TYPE_Q8_0) {
            /* every road the tier has was taken; one dot always (a tile's last token). With both
             * two-row kernels the count is exact, per pair of rows: 8 + 8 and 8 + 4 + 1 tokens in
             * the run of 29, 8 + 8 and 4 in the run of 20 (five calls of x8, two of x4) */
            if (K->dot_row2_x8[type] != NULL && K->dot_row2_x4[type] != NULL) {
                TR_CHECK_EQ_INT(g_mm_x8, 5 * (ROWS / 2));
                TR_CHECK_EQ_INT(g_mm_r2, 2 * (ROWS / 2));
            }
            if (K->dot_row2_x8[type] != NULL) TR_CHECK(g_mm_x8 > 0);
            if (K->dot_row2_x4[type] != NULL) TR_CHECK(g_mm_r2 > 0);
            if (K->dot_row_x4[type] != NULL) TR_CHECK(g_mm_x4 > 0);
            TR_CHECK(g_mm_row > 0);
            TR_CHECK_EQ_INT(16 * g_mm_x8 + 8 * g_mm_r2 + 4 * g_mm_x4 + g_mm_row, N * ROWS);
            printf("  matmul q8_0 on tier %s: %ld calls of 2 rows x 8 tokens, %ld of 2 x 4, %ld of 1 x 4, %ld dots\n",
                   K->tier, g_mm_x8, g_mm_r2, g_mm_x4, g_mm_row);
        }
        for (int t = 0; t < 4; t++) {
            tr_pool *pool = tr_pool_create(threads[t]);
            TR_CHECK(pool != NULL);
            if (pool == NULL) continue;
            memset(y, 0, (size_t)N * ROWS * sizeof(float));
            tr_matmul_grouped(pool, w, offsets, G, x, y);
            TR_CHECK(memcmp(y, ref, (size_t)N * ROWS * sizeof(float)) == 0);
            /* one group alone through tr_matmul */
            memset(y, 0, (size_t)N * ROWS * sizeof(float));
            tr_matmul(pool, &w[4], x + 30 * COLS, 20, y);
            TR_CHECK(memcmp(y, ref + 30 * ROWS, 20 * ROWS * sizeof(float)) == 0);
            tr_pool_destroy(pool);
        }
        free(wdata);
        free(x);
        free(ref);
        free(y);
    }
}

/* The edges: a row length that is not a whole number of blocks, a type the reader knows nothing
 * of, and an empty softmax, which touches nothing (NULL is never read). */
static void test_edges(void) {
    TR_CHECK_EQ_INT(tr_row_bytes(TR_TYPE_Q8_0, 64), 68);
    TR_CHECK_EQ_INT(tr_row_bytes(TR_TYPE_Q8_0, 33), 0);
    TR_CHECK_EQ_INT(tr_row_bytes(TR_TYPE_Q4_K, 512), 288);
    TR_CHECK_EQ_INT(tr_row_bytes(TR_TYPE_Q4_K, 288), 0);
    TR_CHECK(tr_kernels_support(TR_TYPE_Q4_K));
    TR_CHECK_EQ_INT(tr_row_bytes(TR_TYPE_Q6_K, 512), 420);
    TR_CHECK_EQ_INT(tr_row_bytes(TR_TYPE_Q6_K, 128), 0);
    TR_CHECK(tr_kernels_support(TR_TYPE_Q6_K));
    TR_CHECK_EQ_INT(tr_row_bytes((tr_type)4, 32), 0); /* 4: a type number no longer in use */
    TR_CHECK_EQ_INT(tr_row_bytes((tr_type)TR_TYPE_COUNT, 32), 0);
    tr_softmax(NULL, 0);
}

/* ---- the decode's matmul: one token, chunk borders inside it ------------------------------ */
/* One token against quantized matrices of 2048 columns, as the decode: tr_parallel_for's chunks (at
 * least 4096 / cols + 1 = 3 rows) split the only input row among the threads, so with three or seven
 * threads the rows go through matmul_rows, with one through matmul_tiled; the eight one-token groups
 * of an expert layer go through both. Each product must be dot_row's of its own row, bit for bit,
 * and exactly one dot_row call or half a dot_row2 call (counted), at every thread count and for
 * every quantized type; where the tier has dot_row2 for the type, the pairs must have been taken
 * (67 rows: pairs and a row left alone). Branch covered: matmul_rows on quantized rows (seen red
 * with its loop stopping one row short), and its pairs. */
/* Which grouped matmuls run balanced (tr_matmul_one_row_per_group): the branch the experts of a
 * decode token take. Their shape, 8 rows in 8 of 64 groups, was the one the old test ("rows ==
 * groups") missed (docs/LESSONS.md #192); every case below names the call it stands for. */
static void test_one_row_per_group(void) {
    int64_t dense_one[2] = {0, 1};  /* a decode token's projection */
    int64_t dense_four[2] = {0, 4}; /* a prompt's */
    int64_t experts[65];            /* a decode token's experts: 8 of 64 groups, one row each */
    int64_t shared[65];             /* two tokens that share an expert */
    int64_t k = 0, j = 0;
    for (int g = 0; g < 64; g++) {
        experts[g] = k;
        k += g % 8 == 3;
        shared[g] = j;
        j += g == 5 ? 2 : g % 8 == 3;
    }
    experts[64] = k;
    shared[64] = j;
    TR_CHECK(tr_matmul_one_row_per_group(dense_one, 1) == 1);
    TR_CHECK(tr_matmul_one_row_per_group(dense_four, 1) == 0);
    TR_CHECK(experts[64] == 8 && tr_matmul_one_row_per_group(experts, 64) == 1);
    TR_CHECK(tr_matmul_one_row_per_group(shared, 64) == 0);
    printf("  one row per group: a decode's projection and experts balanced, a prompt's and two tokens' not\n");
}

static atomic_long g_dec_dots, g_dec_pairs;
static const tr_kernels *g_dec_real;
static tr_type g_dec_type;
static float dec_dot(const void *row, const float *x, int64_t n) {
    atomic_fetch_add(&g_dec_dots, 1);
    return g_dec_real->dot_row[g_dec_type](row, x, n);
}
static void dec_pair(const void *row0, const void *row1, const float *x, int64_t n, float *out) {
    atomic_fetch_add(&g_dec_pairs, 1);
    g_dec_real->dot_row2[g_dec_type](row0, row1, x, n, out);
}

static void test_matmul_decode(void) {
    enum { ROWS = 67, COLS = 2048, G = 8 };
    static const tr_type types[3] = {TR_TYPE_Q8_0, TR_TYPE_Q4_K, TR_TYPE_Q6_K};
    static const int threads[3] = {1, 3, 7};
    const tr_kernels *K = tr_kernels_get();
    for (int ti = 0; ti < 3; ti++) {
        tr_type type = types[ti];
        size_t rb = tr_row_bytes(type, COLS);
        unsigned char *wdata = (unsigned char *)malloc((size_t)G * ROWS * rb);
        float *x = (float *)malloc((size_t)G * COLS * sizeof(float));
        float *ref = (float *)malloc((size_t)G * ROWS * sizeof(float));
        float *y = (float *)malloc((size_t)G * ROWS * sizeof(float));
        TR_CHECK(wdata != NULL && x != NULL && ref != NULL && y != NULL);
        if (wdata == NULL || x == NULL || ref == NULL || y == NULL) {
            free(wdata);
            free(x);
            free(ref);
            free(y);
            return;
        }
        unsigned seed = 91u + (unsigned)ti;
        for (int64_t i = 0; i < G * ROWS; i++) {
            unsigned char *row = wdata + (size_t)i * rb;
            if (type == TR_TYPE_Q8_0) fill_q8_0(row, COLS / 32, &seed, 0);
            if (type == TR_TYPE_Q4_K) fill_q4_k(row, COLS / 256, &seed, 0);
            if (type == TR_TYPE_Q6_K) fill_q6_k(row, COLS / 256, &seed, 0);
        }
        for (int64_t i = 0; i < G * COLS; i++) x[i] = rand_float(&seed);
        tr_mat w[G];
        int64_t offsets[G + 1];
        for (int64_t g = 0; g < G; g++) {
            w[g].type = type;
            w[g].rows = ROWS;
            w[g].cols = COLS;
            w[g].data = wdata + (size_t)g * ROWS * rb;
            offsets[g] = g;
            for (int64_t r = 0; r < ROWS; r++)
                ref[g * ROWS + r] = K->dot_row[type]((const unsigned char *)w[g].data + (size_t)r * rb, x + g * COLS, COLS);
        }
        offsets[G] = G;

        static tr_kernels counting;
        counting = *K;
        g_dec_real = K;
        g_dec_type = type;
        counting.dot_row[type] = dec_dot;
        if (K->dot_row2[type] != NULL) counting.dot_row2[type] = dec_pair;
        tr_kernels_set_active(&counting);
        for (int t = 0; t < 3; t++) {
            tr_pool *pool = tr_pool_create(threads[t]);
            TR_CHECK(pool != NULL);
            if (pool == NULL) continue;
            atomic_store(&g_dec_dots, 0);
            atomic_store(&g_dec_pairs, 0);
            memset(y, 0, (size_t)ROWS * sizeof(float));
            tr_matmul(pool, &w[0], x, 1, y);
            TR_CHECK(memcmp(y, ref, (size_t)ROWS * sizeof(float)) == 0);
            TR_CHECK_EQ_INT(atomic_load(&g_dec_dots) + 2 * atomic_load(&g_dec_pairs), ROWS);
            if (K->dot_row2[type] != NULL) TR_CHECK(atomic_load(&g_dec_pairs) > 0);
            atomic_store(&g_dec_dots, 0);
            atomic_store(&g_dec_pairs, 0);
            memset(y, 0, (size_t)G * ROWS * sizeof(float));
            tr_matmul_grouped(pool, w, offsets, G, x, y);
            TR_CHECK(memcmp(y, ref, (size_t)G * ROWS * sizeof(float)) == 0);
            TR_CHECK_EQ_INT(atomic_load(&g_dec_dots) + 2 * atomic_load(&g_dec_pairs), G * ROWS);
            if (K->dot_row2[type] != NULL) TR_CHECK(atomic_load(&g_dec_pairs) > 0);
            tr_pool_destroy(pool);
        }
        tr_kernels_set_active(NULL);
        printf("  decode matmul %s on tier %s: one token and eight one-token groups, every product one dot_row "
               "%sat 1, 3 and 7 threads\n",
               type == TR_TYPE_Q8_0 ? "q8_0" : type == TR_TYPE_Q4_K ? "q4_k" : "q6_k", K->tier,
               K->dot_row2[type] != NULL ? "or half a dot_row2 (pairs taken) " : "");
        free(wdata);
        free(x);
        free(ref);
        free(y);
    }
}

/* an activation for the phase-major test: ordinary, with a zero now and then, or (special) the values
 * careless code breaks on, kept rare: an inf or NaN in a lane hides every rounding difference */
static float pm_input(unsigned *seed, int special) {
    if (!special) return (next_rand(seed) % 13u == 0) ? 0.0f : rand_float(seed);
    switch (next_rand(seed) % 512u) {
    case 0: return INFINITY;
    case 1: return -INFINITY;
    case 2: case 3: return 3.0e38f;              /* the product overflows */
    case 4: return -3.0e38f;
    default: return rand_special(seed, 1);       /* +-0, subnormals, underflowing products */
    }
}

/* ---- the prompt's matmul in phase-major order (kernels.h pm_*) ---------------------------------------
 * Branch: every tier with pm_panel[type], pm_interleave and pm_tile: 16 rows' panel, T input rows
 * interleaved, every tile width T = 4..TR_PM_TILE_MAX, against the scalar tier's dot_row bit for bit;
 * rows of ordinary and of special blocks (the halves' whole bit range), inputs ordinary (with zeros) and
 * special (+-0, subnormals, +-inf, overflowing and underflowing products), n of one block, of two and
 * OLMoE's 2048; the interleaved rows in a buffer of exactly TR_PM_XIL_FLOATS(n, T) floats (ASan sees a
 * run past the pad). And the sign of an all-zero dot: Q8_0 rows of negative weights against +0.0 inputs
 * and positive ones against -0.0, whose every product is -0.0: the sum starts from +0.0 (docs/LESSONS.md
 * #214). A tier without the entries is skipped; the counter fails the test if nothing was compared. The
 * line printed says which Q4_K panel the avx512 tier has (tools/tier_check.sh checks the cap). */
static void test_phase_major(void) {
    static const char *tiers[2] = {"avx2", "avx512"};
    static const tr_type types[2] = {TR_TYPE_Q4_K, TR_TYPE_Q8_0};
    static const int64_t ns[3] = {256, 512, 2048};
    const tr_kernels *S = tr_kernels_tier("scalar");
    TR_CHECK(S != NULL);
    if (S == NULL) return;
    int64_t compared = 0;
    for (int ti = 0; ti < 2; ti++) {
        const tr_kernels *K = tr_kernels_tier(tiers[ti]);
        if (K == NULL || K->pm_tile == NULL || K->pm_interleave == NULL) continue;
        for (int yi = 0; yi < 2; yi++) {
            tr_type type = types[yi];
            if (K->pm_panel[type] == NULL) continue;
            for (int ni = 0; ni < 3; ni++) {
                const int64_t n = ns[ni];
                const size_t rb = tr_row_bytes(type, n);
                unsigned char *w = (unsigned char *)malloc(TR_PM_ROWS * rb);
                float *x = (float *)malloc((size_t)TR_PM_TILE_MAX * (size_t)n * sizeof(float));
                float *panel = (float *)malloc((size_t)TR_PM_PANEL_FLOATS(n) * sizeof(float));
                float *part = (float *)malloc(TR_PM_PART * sizeof(float));
                float *y = (float *)malloc((size_t)TR_PM_TILE_MAX * 40 * sizeof(float));
                TR_CHECK(w != NULL && x != NULL && panel != NULL && part != NULL && y != NULL);
                if (w != NULL && x != NULL && panel != NULL && part != NULL && y != NULL) {
                    unsigned seed = 1234u + (unsigned)(ni * 7 + ti);
                    /* weights ordinary or special, inputs ordinary or special; 4: signed zeros (Q8_0) */
                    for (int mode = 0; mode < 5; mode++) {
                        const int wspecial = mode & 1, xspecial = (mode >> 1) & 1, zeros = mode == 4;
                        if (zeros && type != TR_TYPE_Q8_0) continue;
                        for (int r = 0; r < TR_PM_ROWS; r++) fill_row_of(type, w + (size_t)r * rb, n, &seed, wspecial);
                        for (int64_t i = 0; i < TR_PM_TILE_MAX * n; i++) x[i] = pm_input(&seed, xspecial);
                        if (zeros) { /* rows 0-7 negative against +0.0 in inputs 0-11, rows 8-15 positive, -0.0 */
                            for (int r = 0; r < TR_PM_ROWS; r++)
                                for (int64_t b = 0; b < n / 32; b++) {
                                    unsigned char *blk = w + (size_t)r * rb + (size_t)b * TR_Q8_0_BLOCK_BYTES;
                                    const uint16_t d = 0x3C00u; /* 1.0 */
                                    memcpy(blk, &d, 2);
                                    for (int i = 0; i < 32; i++)
                                        blk[2 + i] = (unsigned char)(int8_t)(r < 8 ? -1 - (int)(next_rand(&seed) % 127u)
                                                                               : 1 + (int)(next_rand(&seed) % 127u));
                                }
                            for (int64_t i = 0; i < TR_PM_TILE_MAX * n; i++) x[i] = i < 12 * n ? 0.0f : -0.0f;
                        }
                        K->pm_panel[type](w, rb, n, panel);
                        for (int T = 4; T <= TR_PM_TILE_MAX; T++) {
                            float *xil = (float *)malloc((size_t)TR_PM_XIL_FLOATS(n, T) * sizeof(float));
                            TR_CHECK(xil != NULL);
                            if (xil == NULL) continue;
                            /* y rows 40 floats apart: the kernel writes 16, the rest stays a canary */
                            for (int i = 0; i < TR_PM_TILE_MAX * 40; i++) y[i] = -777.0f;
                            K->pm_interleave(x, n, n, T, xil);
                            K->pm_tile(panel, xil, n, T, part, y, 40);
                            for (int t = 0; t < T; t++) {
                                for (int r = 0; r < TR_PM_ROWS; r++) {
                                    float want = S->dot_row[type](w + (size_t)r * rb, x + (size_t)t * n, n);
                                    TR_CHECK(xspecial ? same_float(y[t * 40 + r], want) : bit_eq(y[t * 40 + r], want));
                                    if (zeros && (r < 8) == (t < 12)) TR_CHECK(bit_eq(want, 0.0f));
                                    compared++;
                                }
                                for (int r = TR_PM_ROWS; r < 40; r++) TR_CHECK(y[t * 40 + r] == -777.0f);
                            }
                            free(xil);
                        }
                    }
                }
                free(w);
                free(x);
                free(panel);
                free(part);
                free(y);
            }
        }
    }
    /* on a CPU without AVX-512 no tier has the entries yet: nothing to compare, and nothing to fail */
    const tr_kernels *A = tr_kernels_tier("avx512");
    if (A != NULL) TR_CHECK(compared > 0);
    if (A != NULL && A->pm_panel[TR_TYPE_Q4_K] != NULL)
        printf("  phase-major on tier avx512: %lld outputs equal to dot_row, the Q4_K panel from %s\n", (long long)compared,
               tr_cpu()->avx512vbmi ? "byte permutes (vbmi)" : "the float transpose");
}

/* ---- tr_matmul_grouped_s: the phase-major road's orchestration (kernels.c pm_*) --------------------
 * Branch: the plan (chunks, tiles, groups under PM_MIN_ROWS on dot_row2), the panel a worker keeps
 * between its items, and the guards that send a call back to tr_matmul_grouped. Against
 * tr_matmul_grouped with no pool (every output one dot_row) byte for byte: 13 groups of 0, 1..5 input
 * rows (both sides of PM_MIN_ROWS), 23..25 (one tile and two), 255..257 and 513 (PM_CHUNK's edges);
 * 16 and 48 weight rows; Q4_K and Q8_0; no pool and pools of 1, 3, 7 and 16; each case twice in a row,
 * on other weights of the same shape. With 16 rows and no pool, consecutive groups share their rows'
 * index: a panel kept by its rows alone would serve the wrong group; one group of 5 inputs called
 * twice: a panel kept from the call before would serve. Counted (pm_tile): the road was taken, and
 * never where it must not be: a pool larger than the scratch, a worker index past the scratch's (no
 * pool, inside a larger pool's body), a Q4_K row that is not whole blocks (docs/LESSONS.md #214). */
static atomic_long g_pms_tiles;
static const tr_kernels *g_pms_real;
static void pms_tile(const float *panel, const float *xil, int64_t n, int T, float *part, float *y, int64_t y_stride) {
    atomic_fetch_add(&g_pms_tiles, 1);
    g_pms_real->pm_tile(panel, xil, n, T, part, y, y_stride);
}

typedef struct {
    const tr_mat *w;
    const int64_t *offsets;
    int64_t n_groups;
    const float *x;
    float *y;
    const tr_pm_scratch *s;
    atomic_int called;
} pms_nested;

/* one chunk run by a worker the scratch has no room for calls the matmul with no pool */
static void pms_nested_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)begin;
    (void)end;
    pms_nested *c = (pms_nested *)ctx_;
    int expected = 0;
    if (worker < c->s->n_workers || !atomic_compare_exchange_strong(&c->called, &expected, 1)) return;
    tr_matmul_grouped_s(NULL, c->w, c->offsets, c->n_groups, c->x, c->y, c->s);
}

static void test_matmul_grouped_s(void) {
    enum { G = 13, COLS = 256 };
    static const int64_t sizes[G] = {0, 1, 2, 3, 4, 5, 23, 24, 25, 255, 256, 257, 513};
    static const tr_type types[2] = {TR_TYPE_Q4_K, TR_TYPE_Q8_0};
    static const int64_t rows_of[2] = {16, 48};
    static const int threads[4] = {1, 3, 7, 16};
    const tr_kernels *K = tr_kernels_get();
    if (K->pm_tile == NULL || K->pm_interleave == NULL) {
        printf("  matmul_grouped_s: tier %s has no phase-major entries, nothing to compare\n", K->tier);
        return;
    }
    int64_t offsets[G + 1];
    offsets[0] = 0;
    for (int g = 0; g < G; g++) offsets[g + 1] = offsets[g] + sizes[g];
    const int64_t N = offsets[G];
    static tr_kernels counting;
    g_pms_real = K;
    counting = *K;
    counting.pm_tile = pms_tile;
    tr_kernels_set_active(&counting);
    long taken = 0, refused = 0;
    for (int ti = 0; ti < 2; ti++) {
        const tr_type type = types[ti];
        if (K->pm_panel[type] == NULL) continue;
        const size_t rb = tr_row_bytes(type, COLS);
        for (int ri = 0; ri < 2; ri++) {
            const int64_t R = rows_of[ri];
            const size_t wbytes = (size_t)G * (size_t)R * rb, ybytes = (size_t)N * (size_t)R * sizeof(float);
            unsigned char *wd[2] = {(unsigned char *)malloc(wbytes), (unsigned char *)malloc(wbytes)};
            float *x = (float *)malloc((size_t)N * COLS * sizeof(float));
            float *ref[2] = {(float *)malloc(ybytes), (float *)malloc(ybytes)};
            float *y = (float *)malloc(ybytes);
            tr_pm_scratch s, s4;
            int ok = wd[0] != NULL && wd[1] != NULL && x != NULL && ref[0] != NULL && ref[1] != NULL && y != NULL;
            ok = ok && tr_pm_scratch_init(&s, 16, G, N, COLS, N * COLS) == 0;
            ok = ok && tr_pm_scratch_init(&s4, 4, G, N, COLS, N * COLS) == 0;
            TR_CHECK(ok);
            if (ok) {
                unsigned seed = 91u + (unsigned)(ti * 2 + ri);
                for (int v = 0; v < 2; v++)
                    for (int64_t r = 0; r < G * R; r++) fill_row_of(type, wd[v] + (size_t)r * rb, COLS, &seed, 0);
                for (int64_t i = 0; i < N * COLS; i++) x[i] = (next_rand(&seed) % 13u == 0) ? 0.0f : rand_float(&seed);
                tr_mat w[2][G];
                for (int v = 0; v < 2; v++) {
                    for (int g = 0; g < G; g++) {
                        w[v][g].type = type;
                        w[v][g].rows = R;
                        w[v][g].cols = COLS;
                        w[v][g].data = wd[v] + (size_t)g * (size_t)R * rb;
                    }
                    tr_matmul_grouped(NULL, w[v], offsets, G, x, ref[v]);
                }
                for (int pi = -1; pi < 4; pi++) {
                    tr_pool *pool = pi < 0 ? NULL : tr_pool_create(threads[pi]);
                    TR_CHECK(pi < 0 || pool != NULL);
                    if (pi >= 0 && pool == NULL) continue;
                    for (int v = 0; v < 2; v++) {
                        memset(y, 0x7F, ybytes);
                        atomic_store(&g_pms_tiles, 0);
                        tr_matmul_grouped_s(pool, w[v], offsets, G, x, y, &s);
                        TR_CHECK(memcmp(y, ref[v], ybytes) == 0);
                        TR_CHECK(atomic_load(&g_pms_tiles) > 0);
                        taken++;
                    }
                    if (pool != NULL) tr_pool_destroy(pool);
                }
                /* one group of 5 inputs, twice on other weights: with 16 rows the same (group, rows) */
                const int64_t one[2] = {0, 5};
                for (int v = 0; v < 2; v++) {
                    tr_matmul_grouped(NULL, &w[v][12], one, 1, x, ref[1 - v]);
                    memset(y, 0x7F, ybytes);
                    tr_matmul_grouped_s(NULL, &w[v][12], one, 1, x, y, &s);
                    TR_CHECK(memcmp(y, ref[1 - v], 5 * (size_t)R * sizeof(float)) == 0);
                }
                for (int v = 0; v < 2; v++) tr_matmul_grouped(NULL, w[v], offsets, G, x, ref[v]);
                /* the refusals: none may run a tile, every one gives the old road's bytes */
                tr_pool *p8 = tr_pool_create(8);
                TR_CHECK(p8 != NULL);
                if (p8 != NULL) {
                    memset(y, 0x7F, ybytes);
                    atomic_store(&g_pms_tiles, 0);
                    tr_matmul_grouped_s(p8, w[0], offsets, G, x, y, &s4); /* 8 workers, room for 4 */
                    TR_CHECK(memcmp(y, ref[0], ybytes) == 0);
                    TR_CHECK_EQ_INT(atomic_load(&g_pms_tiles), 0);
                    pms_nested nc;
                    nc.w = w[1];
                    nc.offsets = offsets;
                    nc.n_groups = G;
                    nc.x = x;
                    nc.y = y;
                    nc.s = &s4;
                    atomic_store(&nc.called, 0);
                    memset(y, 0x7F, ybytes);
                    atomic_store(&g_pms_tiles, 0);
                    tr_parallel_for(p8, 8, 1, pms_nested_body, &nc);
                    TR_CHECK_EQ_INT(atomic_load(&nc.called), 1);
                    TR_CHECK(memcmp(y, ref[1], ybytes) == 0);
                    TR_CHECK_EQ_INT(atomic_load(&g_pms_tiles), 0);
                    refused += 2;
                    tr_pool_destroy(p8);
                }
            }
            if (ok || s.plan != NULL) tr_pm_scratch_free(&s);
            if (ok || s4.plan != NULL) tr_pm_scratch_free(&s4);
            free(wd[0]);
            free(wd[1]);
            free(x);
            free(ref[0]);
            free(ref[1]);
            free(y);
        }
    }
    /* a Q4_K row of 272 columns is not whole blocks (row bytes 0): no tile may run on it */
    {
        tr_pm_scratch s;
        unsigned char *wq = (unsigned char *)calloc(16, 2 * 144);
        float *x = (float *)calloc(8 * 272, sizeof(float)), *y = (float *)calloc(8 * 16, sizeof(float));
        const int64_t off[2] = {0, 8};
        int ok = wq != NULL && x != NULL && y != NULL && tr_pm_scratch_init(&s, 1, 1, 8, 272, 8 * 272) == 0;
        TR_CHECK(ok);
        if (ok && K->pm_panel[TR_TYPE_Q4_K] != NULL) {
            tr_mat wm = {TR_TYPE_Q4_K, 16, 272, wq};
            atomic_store(&g_pms_tiles, 0);
            tr_matmul_grouped_s(NULL, &wm, off, 1, x, y, &s);
            TR_CHECK_EQ_INT(atomic_load(&g_pms_tiles), 0);
            refused++;
        }
        if (ok) tr_pm_scratch_free(&s);
        free(wq);
        free(x);
        free(y);
    }
    tr_kernels_set_active(NULL);
    TR_CHECK(taken > 0 && refused > 0);
    printf("  matmul_grouped_s on tier %s: %ld calls through phase-major equal to dot_row, %ld refusals on "
           "the old road\n", K->tier, taken, refused);
}

int main(void) {
    tr_kernels_init();

    test_edges();
    test_half_to_float();
    test_q8_0_dequant();
    test_q4_k_dequant();
    test_q6_k_dequant();
    test_dot_f32_contract();
    test_dot_row_f16();
    test_dot_row_q8_0();
    test_dot_row_q4_k();
    test_dot_row_q6_k();
    test_tiers_match_scalar();
    test_dequant_is_the_dots_weight();
    test_rope_table();
    test_swiglu_threads();
    test_attention_head();
    test_attention_group();
    test_matmul_thread_determinism();
    test_matmul_grouped();
    test_matmul_decode();
    test_one_row_per_group();
    test_phase_major();
    test_matmul_grouped_s();

    TR_TEST_EXIT();
}
