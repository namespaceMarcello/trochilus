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

#endif
