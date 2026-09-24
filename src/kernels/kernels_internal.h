/* kernels_internal.h — pieces shared by the scalar kernels (kernels.c) and the
 * SIMD tiers (kernels_x86.c ...). Not part of the public API. */
#ifndef TR_KERNELS_INTERNAL_H
#define TR_KERNELS_INTERNAL_H

#include "kernels.h"

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
