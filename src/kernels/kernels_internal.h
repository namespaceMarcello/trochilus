/* kernels_internal.h — pieces shared by the scalar kernels (kernels.c) and the
 * SIMD tiers (kernels_x86.c ...). Not part of the public API. */
#ifndef TR_KERNELS_INTERNAL_H
#define TR_KERNELS_INTERNAL_H

#include "kernels.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define TR_Q8_0_BLOCK_ELEMS 32
#define TR_Q8_0_SCALE_BYTES 2
#define TR_Q8_0_BLOCK_BYTES 34

/* hot: begin */
/* IEEE binary16 -> binary32, exact, NaN payload preserved. */
static inline float tr_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign; /* +/-0 */
        } else {
            /* subnormal half -> normalize into a normal float */
            int e = -1;
            do {
                e++;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            uint32_t exp32 = (uint32_t)(127 - 15 - e);
            bits = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / nan, payload preserved */
    } else {
        uint32_t exp32 = exp - 15u + 127u;
        bits = sign | (exp32 << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* bfloat16 -> binary32: the top half of a float, the low half zero; exact for every value, NaN included. */
static inline float tr_bf16_to_float(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static inline float tr_q8_0_block_scale(const unsigned char *blk) {
    uint16_t dbits;
    memcpy(&dbits, blk, TR_Q8_0_SCALE_BYTES);
    return tr_half_to_float(dbits);
}

/* Q4_K (ggml's block_q4_K): 256 elements in 144 bytes. f16 d, f16 dmin, 12 bytes holding eight
 * 6-bit scales and eight 6-bit mins, then 128 bytes of 4-bit quants. Element i belongs to
 * sub-block j = i / 32 and weighs scale[j] * q - min[j], with scale[j] = d * sc_j and
 * min[j] = dmin * m_j, every product and the difference rounded to float: the order ggml's
 * dequantize_row_q4_K and gguf-py's Q4_K.dequantize_blocks use, so the dequantized row is theirs
 * bit for bit (tools/check_dequant.py). Quant bytes 32c .. 32c+31 hold sub-block 2c in their low
 * nibbles and 2c+1 in their high ones. */
#define TR_Q4_K_BLOCK_ELEMS 256
#define TR_Q4_K_BLOCK_BYTES 144
#define TR_Q4_K_QS_OFFSET 16

/* scale[j] and min[j] of the block at blk, j = 0..7 (ggml's get_scale_min_k4, unrolled) */
static inline void tr_q4_k_scales(const unsigned char *blk, float scale[8], float min[8]) {
    uint16_t hd, hm;
    memcpy(&hd, blk, 2);
    memcpy(&hm, blk + 2, 2);
    float d = tr_half_to_float(hd), dmin = tr_half_to_float(hm);
    const unsigned char *s = blk + 4;
    for (int j = 0; j < 4; j++) {
        scale[j] = d * (float)(s[j] & 63);
        min[j] = dmin * (float)(s[j + 4] & 63);
        scale[j + 4] = d * (float)((s[j + 8] & 0x0F) | ((s[j] >> 6) << 4));
        min[j + 4] = dmin * (float)((s[j + 8] >> 4) | ((s[j + 4] >> 6) << 4));
    }
}

/* the 4-bit quant of element i (0..255) of a block whose quants start at qs */
static inline unsigned tr_q4_k_quant(const unsigned char *qs, int i) {
    unsigned byte = qs[32 * (i >> 6) + (i & 31)];
    return (i & 32) ? byte >> 4 : byte & 0x0F;
}

/* the six-bit sc_j and m_j of the block at blk, as integers (tr_q4_k_scales without d and dmin) */
static inline void tr_q4_k_sc_m(const unsigned char *blk, int sc[8], int m[8]) {
    const unsigned char *s = blk + 4;
    for (int j = 0; j < 4; j++) {
        sc[j] = s[j] & 63;
        m[j] = s[j + 4] & 63;
        sc[j + 4] = (s[j + 8] & 0x0F) | ((s[j] >> 6) << 4);
        m[j + 4] = (s[j + 8] >> 4) | ((s[j + 4] >> 6) << 4);
    }
}

/* ---- Q4_K's integer definition (kernels.h q4x_*) ----
 * An input row of n columns prepared once (tr_q4x_bytes(n) bytes): int16 v0[n], v1[n], the balanced digits
 * of X = v0 + TR_Q4X_BASE v1 in natural column order; int32 tok[n/64][2], each 64-column window's token term
 * per digit, mod 2^32; double bf[n/256][8], each 32-column sub-block's exact sum of X; double scale[n/256],
 * 2^-sh of each super-block, NaN for one that holds a NaN or an infinity (its digits and sums are then 0).
 * The panel of 16 rows (TR_Q4X_PANEL_BYTES(n)): per super-block 1280 bytes of doubles, d[16], dmin[16] and
 * m[8][16], each with the even rows first, then the odd ones (the order the tile's int64 split gives), then
 * 4 windows of 64 columns, window c the block's quant chunk c (bytes 32c..32c+31: low nibbles columns
 * 64c..64c+31, high nibbles 64c+32..64c+63): 16 int32, minus the window's row term mod 2^32, plus 2^31 on
 * the even rows; then for u = 0..15 two vectors of 32 int16, lane r (words 2r, 2r+1) of the byte pair
 * i = 2u, 2u+1 of row r: wl = sc_2c lo(b_i) + TR_Q4X_C, wh = sc_2c+1 hi(b_i) + TR_Q4X_C. The tile pairs the
 * low and the high nibble of a byte (columns a = 64c+i and a+32): (wl_a + V_a+32)(wh_a + V_a) = w_a V_a +
 * w_a+32 V_a+32 + a row term (wl_a wh_a) + a token term (C (V_a + V_a+32) + V_a V_a+32), the words and the
 * digits each fit 16 bits and so does their sum. */
#define TR_Q4X_C (-473)
#define TR_Q4X_CONST_BYTES 1280
#define TR_Q4X_WIN_BYTES (64 + 16 * 128)
#define TR_Q4X_SB_BYTES (TR_Q4X_CONST_BYTES + 4 * TR_Q4X_WIN_BYTES)

typedef struct {
    int16_t *v0, *v1;
    int32_t *tok;
    double *bf, *scale;
} tr_q4x_view;

static inline tr_q4x_view tr_q4x_view_of(const void *xq, int64_t n) {
    tr_q4x_view v;
    unsigned char *p = (unsigned char *)(uintptr_t)xq;
    v.v0 = (int16_t *)(void *)p;
    v.v1 = v.v0 + n;
    v.tok = (int32_t *)(void *)(p + 4 * (size_t)n);
    v.bf = (double *)(void *)(p + 4 * (size_t)n + (size_t)n / 8);
    v.scale = v.bf + n / 32;
    return v;
}

/* 2^n from its bits, what ldexp(1.0, n) gives without the C library: a double for |n| <= 1022, a float for
 * |n| <= 126 (the prep's shifts stay within -99..179, and a float's within -99..127) */
static inline double tr_pow2(int n) {
    uint64_t b = (uint64_t)(n + 1023) << 52;
    double d;
    memcpy(&d, &b, sizeof d);
    return d;
}

static inline float tr_pow2f(int n) {
    uint32_t b = (uint32_t)(n + 127) << 23;
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

/* the shift of a block whose largest |x| is m (finite): the largest sh with m 2^sh <= TR_Q4X_XMAX, 0 for 0.
 * e is frexpf's exponent (m = f 2^e, f in [0.5, 1)), read from the bits: the exponent field less 126, or for a
 * subnormal (mantissa M, m = M 2^-149) M's bit length less 149. m 2^(31 - e) lies in [2^30, 2^31) and
 * TR_Q4X_XMAX in between: one step down at most; the product is exact (a power of two, in double's range). */
static inline int tr_q4x_shift(float m) {
    if (m == 0.0f) return 0;
    uint32_t b;
    memcpy(&b, &m, sizeof b);
    int field = (int)((b >> 23) & 0xffu), e;
    if (field != 0) {
        e = field - 126;
    } else {
        uint32_t man = b & 0x7fffffu;
        int len = 0;
        while (man >> len) len++;
        e = len - 149;
    }
    int sh = 31 - e;
    if ((double)m * tr_pow2(sh) > TR_Q4X_XMAX) sh--;
    return sh;
}

/* X = v0 + TR_Q4X_BASE v1, v1 the nearest integer to X / TR_Q4X_BASE (never a tie: the base is odd) */
static inline void tr_q4x_digits(int32_t X, int16_t *v0, int16_t *v1) {
    const int64_t b = TR_Q4X_BASE, x = X;
    const int64_t q = x >= 0 ? (x + b / 2) / b : -((-x + b / 2) / b);
    *v1 = (int16_t)q;
    *v0 = (int16_t)(x - b * q);
}

/* Q6_K (ggml's block_q6_K): 256 elements in 210 bytes. 128 bytes of low 4 bits (ql), 64 of high 2
 * bits (qh), sixteen int8 scales, then f16 d. Element i belongs to sub-block j = i / 16 and weighs
 * scale[j] * (q - 32), q its 6-bit quant, with scale[j] = d * sc_j, every product rounded to float:
 * the order ggml's dequantize_row_q6_K and gguf-py's Q6_K.dequantize_blocks use (tools/
 * check_dequant.py). Element i = 128h + 32c + l (h = 0..1, c = 0..3, l = 0..31) takes the low
 * (c < 2) or high (c >= 2) nibble of ql[64h + 32(c & 1) + l] and bits 2c, 2c+1 of qh[32h + l]. */
#define TR_Q6_K_BLOCK_ELEMS 256
#define TR_Q6_K_BLOCK_BYTES 210
#define TR_Q6_K_QH_OFFSET 128
#define TR_Q6_K_SCALES_OFFSET 192
#define TR_Q6_K_D_OFFSET 208

/* scale[j] of the block at blk, j = 0..15 */
static inline void tr_q6_k_scales(const unsigned char *blk, float scale[16]) {
    uint16_t hd;
    memcpy(&hd, blk + TR_Q6_K_D_OFFSET, 2);
    float d = tr_half_to_float(hd);
    const int8_t *sc = (const int8_t *)(blk + TR_Q6_K_SCALES_OFFSET);
    for (int j = 0; j < 16; j++) scale[j] = d * (float)sc[j];
}

/* q - 32 of element i (0..255) of the block at blk: -32..31 */
static inline int tr_q6_k_quant(const unsigned char *blk, int i) {
    int h = i >> 7, c = (i >> 5) & 3, l = i & 31;
    unsigned lo = blk[64 * h + 32 * (c & 1) + l];
    lo = (c & 2) ? lo >> 4 : lo & 0x0F;
    unsigned hi = (blk[TR_Q6_K_QH_OFFSET + 32 * h + l] >> (2 * c)) & 3u;
    return (int)(lo | (hi << 4)) - 32;
}

/* ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), same for 8..15, halves added last. */
static inline float tr_lane_combine(const float lane[TR_LANES]) {
    float s01 = lane[0] + lane[1], s23 = lane[2] + lane[3];
    float s45 = lane[4] + lane[5], s67 = lane[6] + lane[7];
    float lo = (s01 + s23) + (s45 + s67);

    float s89 = lane[8] + lane[9], s1011 = lane[10] + lane[11];
    float s1213 = lane[12] + lane[13], s1415 = lane[14] + lane[15];
    float hi = (s89 + s1011) + (s1213 + s1415);

    return lo + hi;
}
/* hot: end */

/* The scalar table; SIMD tiers start from a copy and replace what they accelerate. */
const tr_kernels *tr_kernels_scalar(void);

/* x86-64 tiers ("avx2", "avx512"): the table, or NULL if the name is unknown, not
 * compiled for this architecture, or not supported by this CPU and OS. */
const tr_kernels *tr_kernels_x86_tier(const char *tier);

/* Tests only: makes k the active table; NULL goes back to the tier tr_kernels_init chooses.
 * The engine reads the active table at every matmul and every attention, so a test can wrap a
 * tier's kernels with counters and see which ones a model really runs
 * (tests/test_tier_used.c, docs/LESSONS.md #78). Never while a forward pass is running. */
void tr_kernels_set_active(const tr_kernels *k);

#endif
