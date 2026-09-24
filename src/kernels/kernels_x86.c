/* kernels_x86.c — AVX2 and AVX-512 tiers of the hot kernels (dot_f32, axpy_f32, dot_row Q8_0).
 *
 * Each function reproduces the scalar arithmetic of kernels.c bit for bit (kernels.h):
 * one accumulator register per group of lanes (one __m512 = lanes 0..15, two __m256 =
 * lanes 0..7 and 8..15), separate multiply and add instructions, never FMA, the same
 * operand order, lanes combined by the scalar tree. Tails go into the lanes they belong
 * to, after the full chunks, so every lane still adds in increasing k.
 *
 * The functions carry target attributes instead of the whole file being built with
 * -mavx512f: the binary runs on any x86-64 and tr_kernels_x86_tier only hands out a
 * tier that tr_cpu() (CPUID + XGETBV, capped by TR_CPU_MAX) says is usable.
 * gcc and clang only. */
#include "kernels_internal.h"

#include "../base/cpu.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#define TR_TARGET_AVX2 __attribute__((target("avx2")))
#define TR_TARGET_AVX512 __attribute__((target("avx512f")))

/* ---- AVX2: lanes 0..7 in lo, 8..15 in hi --------------------------------- */
/* hot: begin */

TR_TARGET_AVX2
static float avx2_dot_f32(const float *a, const float *b, int64_t n) {
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        lo = _mm256_add_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k)));
        hi = _mm256_add_ps(hi, _mm256_mul_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8)));
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo);
    _mm256_storeu_ps(lane + 8, hi);
    for (; k < n; k++) lane[k % TR_LANES] += a[k] * b[k];
    return tr_lane_combine(lane);
}

TR_TARGET_AVX2
static void avx2_axpy_f32(float *y, const float *x, float a, int64_t n) {
    __m256 va = _mm256_set1_ps(a);
    int64_t k = 0;
    for (; k + 8 <= n; k += 8)
        _mm256_storeu_ps(y + k, _mm256_add_ps(_mm256_loadu_ps(y + k), _mm256_mul_ps(va, _mm256_loadu_ps(x + k))));
    for (; k < n; k++) y[k] = y[k] + a * x[k];
}

/* One a against 4 b: a is loaded once per 16 elements, each b keeps its own lo and hi, named
 * registers as in dot_row_x4 below. out[j] is bit for bit avx2_dot_f32(a, b + j*stride, n). */
TR_TARGET_AVX2
static void avx2_dot_f32_x4(const float *a, const float *b, int64_t stride, int64_t n, float *out) {
    const float *b0 = b, *b1 = b0 + stride, *b2 = b1 + stride, *b3 = b2 + stride;
    __m256 lo0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps(), lo2 = _mm256_setzero_ps(),
           lo3 = _mm256_setzero_ps();
    __m256 hi0 = _mm256_setzero_ps(), hi1 = _mm256_setzero_ps(), hi2 = _mm256_setzero_ps(),
           hi3 = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        __m256 alo = _mm256_loadu_ps(a + k), ahi = _mm256_loadu_ps(a + k + 8);
        lo0 = _mm256_add_ps(lo0, _mm256_mul_ps(alo, _mm256_loadu_ps(b0 + k)));
        hi0 = _mm256_add_ps(hi0, _mm256_mul_ps(ahi, _mm256_loadu_ps(b0 + k + 8)));
        lo1 = _mm256_add_ps(lo1, _mm256_mul_ps(alo, _mm256_loadu_ps(b1 + k)));
        hi1 = _mm256_add_ps(hi1, _mm256_mul_ps(ahi, _mm256_loadu_ps(b1 + k + 8)));
        lo2 = _mm256_add_ps(lo2, _mm256_mul_ps(alo, _mm256_loadu_ps(b2 + k)));
        hi2 = _mm256_add_ps(hi2, _mm256_mul_ps(ahi, _mm256_loadu_ps(b2 + k + 8)));
        lo3 = _mm256_add_ps(lo3, _mm256_mul_ps(alo, _mm256_loadu_ps(b3 + k)));
        hi3 = _mm256_add_ps(hi3, _mm256_mul_ps(ahi, _mm256_loadu_ps(b3 + k + 8)));
    }
    float l0[TR_LANES], l1[TR_LANES], l2[TR_LANES], l3[TR_LANES];
    _mm256_storeu_ps(l0, lo0);
    _mm256_storeu_ps(l0 + 8, hi0);
    _mm256_storeu_ps(l1, lo1);
    _mm256_storeu_ps(l1 + 8, hi1);
    _mm256_storeu_ps(l2, lo2);
    _mm256_storeu_ps(l2 + 8, hi2);
    _mm256_storeu_ps(l3, lo3);
    _mm256_storeu_ps(l3 + 8, hi3);
    for (; k < n; k++) {
        l0[k % TR_LANES] += a[k] * b0[k];
        l1[k % TR_LANES] += a[k] * b1[k];
        l2[k % TR_LANES] += a[k] * b2[k];
        l3[k % TR_LANES] += a[k] * b3[k];
    }
    out[0] = tr_lane_combine(l0);
    out[1] = tr_lane_combine(l1);
    out[2] = tr_lane_combine(l2);
    out[3] = tr_lane_combine(l3);
}

/* 4 axpys on one y, in order: y is loaded and stored once per 8 elements instead of 4 times.
 * Bit for bit 4 calls of avx2_axpy_f32. */
TR_TARGET_AVX2
static void avx2_axpy_f32_x4(float *y, const float *x, int64_t stride, const float *a, int64_t n) {
    const float *x0 = x, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
    __m256 a0 = _mm256_set1_ps(a[0]), a1 = _mm256_set1_ps(a[1]), a2 = _mm256_set1_ps(a[2]),
           a3 = _mm256_set1_ps(a[3]);
    int64_t k = 0;
    for (; k + 8 <= n; k += 8) {
        __m256 v = _mm256_loadu_ps(y + k);
        v = _mm256_add_ps(v, _mm256_mul_ps(a0, _mm256_loadu_ps(x0 + k)));
        v = _mm256_add_ps(v, _mm256_mul_ps(a1, _mm256_loadu_ps(x1 + k)));
        v = _mm256_add_ps(v, _mm256_mul_ps(a2, _mm256_loadu_ps(x2 + k)));
        v = _mm256_add_ps(v, _mm256_mul_ps(a3, _mm256_loadu_ps(x3 + k)));
        _mm256_storeu_ps(y + k, v);
    }
    for (; k < n; k++) {
        float v = y[k];
        v = v + a[0] * x0[k];
        v = v + a[1] * x1[k];
        v = v + a[2] * x2[k];
        v = v + a[3] * x3[k];
        y[k] = v;
    }
}

/* An F32 weight row is a plain dot_f32 of the row with x, and against 4 input rows it is
 * dot_f32_x4 with the row as `a`: the tier's own kernels instead of the scalar loop (the
 * router's matrix is F32 in every GGUF). */
_Static_assert(TR_DOT_TOKENS == TR_ATTN_X, "x4"); /* hot-ok: stringa -- a compile-time message, no code */

TR_TARGET_AVX2
static float avx2_dot_row_f32(const void *row, const float *x, int64_t n) {
    return avx2_dot_f32((const float *)row, x, n);
}

TR_TARGET_AVX2
static void avx2_dot_row_x4_f32(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    avx2_dot_f32_x4((const float *)row, x, stride, n, out);
}

/* F16 rows: 8 halves become 8 floats with one F16C instruction, an exact conversion (what
 * tr_half_to_float computes, subnormals included), then the products go into the lanes as in
 * dot_f32. The tail converts half by half, as scalar does. */
#define TR_TARGET_AVX2_F16C __attribute__((target("avx2,f16c")))

TR_TARGET_AVX2_F16C
static inline __m256 avx2_f16_to_ps(const unsigned char *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)p));
}

static inline float tail_half(const unsigned char *row, int64_t k) {
    uint16_t h;
    memcpy(&h, row + 2 * (size_t)k, sizeof h);
    return tr_half_to_float(h);
}

TR_TARGET_AVX2_F16C
static float avx2_dot_row_f16(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        lo = _mm256_add_ps(lo, _mm256_mul_ps(avx2_f16_to_ps(p + 2 * k), _mm256_loadu_ps(x + k)));
        hi = _mm256_add_ps(hi, _mm256_mul_ps(avx2_f16_to_ps(p + 2 * k + 16), _mm256_loadu_ps(x + k + 8)));
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo);
    _mm256_storeu_ps(lane + 8, hi);
    for (; k < n; k++) lane[k % TR_LANES] += tail_half(p, k) * x[k];
    return tr_lane_combine(lane);
}

TR_TARGET_AVX2_F16C
static void avx2_dot_row_x4_f16(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    const float *x0 = x, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
    __m256 lo0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps(), lo2 = _mm256_setzero_ps(),
           lo3 = _mm256_setzero_ps();
    __m256 hi0 = _mm256_setzero_ps(), hi1 = _mm256_setzero_ps(), hi2 = _mm256_setzero_ps(),
           hi3 = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        __m256 w0 = avx2_f16_to_ps(p + 2 * k), w1 = avx2_f16_to_ps(p + 2 * k + 16);
        lo0 = _mm256_add_ps(lo0, _mm256_mul_ps(w0, _mm256_loadu_ps(x0 + k)));
        hi0 = _mm256_add_ps(hi0, _mm256_mul_ps(w1, _mm256_loadu_ps(x0 + k + 8)));
        lo1 = _mm256_add_ps(lo1, _mm256_mul_ps(w0, _mm256_loadu_ps(x1 + k)));
        hi1 = _mm256_add_ps(hi1, _mm256_mul_ps(w1, _mm256_loadu_ps(x1 + k + 8)));
        lo2 = _mm256_add_ps(lo2, _mm256_mul_ps(w0, _mm256_loadu_ps(x2 + k)));
        hi2 = _mm256_add_ps(hi2, _mm256_mul_ps(w1, _mm256_loadu_ps(x2 + k + 8)));
        lo3 = _mm256_add_ps(lo3, _mm256_mul_ps(w0, _mm256_loadu_ps(x3 + k)));
        hi3 = _mm256_add_ps(hi3, _mm256_mul_ps(w1, _mm256_loadu_ps(x3 + k + 8)));
    }
    float l0[TR_LANES], l1[TR_LANES], l2[TR_LANES], l3[TR_LANES];
    _mm256_storeu_ps(l0, lo0);
    _mm256_storeu_ps(l0 + 8, hi0);
    _mm256_storeu_ps(l1, lo1);
    _mm256_storeu_ps(l1 + 8, hi1);
    _mm256_storeu_ps(l2, lo2);
    _mm256_storeu_ps(l2 + 8, hi2);
    _mm256_storeu_ps(l3, lo3);
    _mm256_storeu_ps(l3 + 8, hi3);
    for (; k < n; k++) {
        float w = tail_half(p, k);
        l0[k % TR_LANES] += w * x0[k];
        l1[k % TR_LANES] += w * x1[k];
        l2[k % TR_LANES] += w * x2[k];
        l3[k % TR_LANES] += w * x3[k];
    }
    out[0] = tr_lane_combine(l0);
    out[1] = tr_lane_combine(l1);
    out[2] = tr_lane_combine(l2);
    out[3] = tr_lane_combine(l3);
}

/* 8 int8 at p -> 8 exact floats */
TR_TARGET_AVX2
static inline __m256 avx2_i8_to_ps(const unsigned char *p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)p)));
}

TR_TARGET_AVX2
static float avx2_dot_row_q8_0(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m256 d = _mm256_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m256 w0 = _mm256_mul_ps(d, avx2_i8_to_ps(qs + j));
            __m256 w1 = _mm256_mul_ps(d, avx2_i8_to_ps(qs + j + 8));
            lo = _mm256_add_ps(lo, _mm256_mul_ps(w0, _mm256_loadu_ps(xb + j)));
            hi = _mm256_add_ps(hi, _mm256_mul_ps(w1, _mm256_loadu_ps(xb + j + 8)));
        }
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo);
    _mm256_storeu_ps(lane + 8, hi);
    return tr_lane_combine(lane);
}

/* ---- Q4_K (kernels_internal.h): each weight scale * q - min, as scalar computes it ---- */

/* 8 quant bytes at p -> their low (hi = 0) or high (hi = 1) nibbles, as 8 exact floats */
TR_TARGET_AVX2
static inline __m256 avx2_nibbles_to_ps(const unsigned char *p, int hi) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)p));
    v = hi ? _mm256_srli_epi32(v, 4) : _mm256_and_si256(v, _mm256_set1_epi32(0x0F));
    return _mm256_cvtepi32_ps(v);
}

TR_TARGET_AVX2
static float avx2_dot_row_q4_k(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        const float *xb = x + b * TR_Q4_K_BLOCK_ELEMS;
        float scale[8], min[8];
        tr_q4_k_scales(blk, scale, min);
        /* sub-block j: 32 elements, the nibbles (j & 1) of bytes 32 * (j / 2) .. + 31 */
        for (int j = 0; j < 8; j++) {
            __m256 s = _mm256_set1_ps(scale[j]), m = _mm256_set1_ps(min[j]);
            const unsigned char *q = qs + 32 * (j >> 1);
            const float *xj = xb + 32 * j;
            for (int g = 0; g < 32; g += TR_LANES) {
                __m256 w0 = _mm256_sub_ps(_mm256_mul_ps(s, avx2_nibbles_to_ps(q + g, j & 1)), m);
                __m256 w1 = _mm256_sub_ps(_mm256_mul_ps(s, avx2_nibbles_to_ps(q + g + 8, j & 1)), m);
                lo = _mm256_add_ps(lo, _mm256_mul_ps(w0, _mm256_loadu_ps(xj + g)));
                hi = _mm256_add_ps(hi, _mm256_mul_ps(w1, _mm256_loadu_ps(xj + g + 8)));
            }
        }
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo);
    _mm256_storeu_ps(lane + 8, hi);
    return tr_lane_combine(lane);
}

TR_TARGET_AVX2
static void avx2_dot_row_x4_q4_k(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    /* named registers, never an array (see avx2_dot_row_x4_q8_0) */
    __m256 lo0 = _mm256_setzero_ps(), hi0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps(),
           hi1 = _mm256_setzero_ps(), lo2 = _mm256_setzero_ps(), hi2 = _mm256_setzero_ps(),
           lo3 = _mm256_setzero_ps(), hi3 = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        const float *xb = x + b * TR_Q4_K_BLOCK_ELEMS;
        float scale[8], min[8];
        tr_q4_k_scales(blk, scale, min);
        for (int j = 0; j < 8; j++) {
            __m256 s = _mm256_set1_ps(scale[j]), m = _mm256_set1_ps(min[j]);
            const unsigned char *q = qs + 32 * (j >> 1);
            for (int g = 0; g < 32; g += TR_LANES) {
                __m256 w0 = _mm256_sub_ps(_mm256_mul_ps(s, avx2_nibbles_to_ps(q + g, j & 1)), m);
                __m256 w1 = _mm256_sub_ps(_mm256_mul_ps(s, avx2_nibbles_to_ps(q + g + 8, j & 1)), m);
                const float *x0 = xb + 32 * j + g, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
                lo0 = _mm256_add_ps(lo0, _mm256_mul_ps(w0, _mm256_loadu_ps(x0)));
                hi0 = _mm256_add_ps(hi0, _mm256_mul_ps(w1, _mm256_loadu_ps(x0 + 8)));
                lo1 = _mm256_add_ps(lo1, _mm256_mul_ps(w0, _mm256_loadu_ps(x1)));
                hi1 = _mm256_add_ps(hi1, _mm256_mul_ps(w1, _mm256_loadu_ps(x1 + 8)));
                lo2 = _mm256_add_ps(lo2, _mm256_mul_ps(w0, _mm256_loadu_ps(x2)));
                hi2 = _mm256_add_ps(hi2, _mm256_mul_ps(w1, _mm256_loadu_ps(x2 + 8)));
                lo3 = _mm256_add_ps(lo3, _mm256_mul_ps(w0, _mm256_loadu_ps(x3)));
                hi3 = _mm256_add_ps(hi3, _mm256_mul_ps(w1, _mm256_loadu_ps(x3 + 8)));
            }
        }
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo0);
    _mm256_storeu_ps(lane + 8, hi0);
    out[0] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo1);
    _mm256_storeu_ps(lane + 8, hi1);
    out[1] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo2);
    _mm256_storeu_ps(lane + 8, hi2);
    out[2] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo3);
    _mm256_storeu_ps(lane + 8, hi3);
    out[3] = tr_lane_combine(lane);
}

/* ---- AVX-512: lanes 0..15 in one register -------------------------------- */

TR_TARGET_AVX512
static float avx512_dot_f32(const float *a, const float *b, int64_t n) {
    __m512 acc = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES)
        acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k)));
    if (k < n) {
        /* masked add: the lanes past the tail keep their value, -0.0 included */
        __mmask16 m = (__mmask16)((1u << (unsigned)(n - k)) - 1u);
        __m512 prod = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, a + k), _mm512_maskz_loadu_ps(m, b + k));
        acc = _mm512_mask_add_ps(acc, m, acc, prod);
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc);
    return tr_lane_combine(lane);
}

TR_TARGET_AVX512
static void avx512_axpy_f32(float *y, const float *x, float a, int64_t n) {
    __m512 va = _mm512_set1_ps(a);
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES)
        _mm512_storeu_ps(y + k, _mm512_add_ps(_mm512_loadu_ps(y + k), _mm512_mul_ps(va, _mm512_loadu_ps(x + k))));
    if (k < n) {
        __mmask16 m = (__mmask16)((1u << (unsigned)(n - k)) - 1u);
        __m512 prod = _mm512_mul_ps(va, _mm512_maskz_loadu_ps(m, x + k));
        _mm512_mask_storeu_ps(y + k, m, _mm512_add_ps(_mm512_maskz_loadu_ps(m, y + k), prod));
    }
}

/* One a against 4 b, an accumulator per b in a named register. out[j] is bit for bit
 * avx512_dot_f32(a, b + j*stride, n), tail included. */
TR_TARGET_AVX512
static void avx512_dot_f32_x4(const float *a, const float *b, int64_t stride, int64_t n, float *out) {
    const float *b0 = b, *b1 = b0 + stride, *b2 = b1 + stride, *b3 = b2 + stride;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps(),
           acc3 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        __m512 av = _mm512_loadu_ps(a + k);
        acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(av, _mm512_loadu_ps(b0 + k)));
        acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(av, _mm512_loadu_ps(b1 + k)));
        acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(av, _mm512_loadu_ps(b2 + k)));
        acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(av, _mm512_loadu_ps(b3 + k)));
    }
    if (k < n) {
        __mmask16 m = (__mmask16)((1u << (unsigned)(n - k)) - 1u);
        __m512 av = _mm512_maskz_loadu_ps(m, a + k);
        acc0 = _mm512_mask_add_ps(acc0, m, acc0, _mm512_mul_ps(av, _mm512_maskz_loadu_ps(m, b0 + k)));
        acc1 = _mm512_mask_add_ps(acc1, m, acc1, _mm512_mul_ps(av, _mm512_maskz_loadu_ps(m, b1 + k)));
        acc2 = _mm512_mask_add_ps(acc2, m, acc2, _mm512_mul_ps(av, _mm512_maskz_loadu_ps(m, b2 + k)));
        acc3 = _mm512_mask_add_ps(acc3, m, acc3, _mm512_mul_ps(av, _mm512_maskz_loadu_ps(m, b3 + k)));
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc0);
    out[0] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc1);
    out[1] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc2);
    out[2] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc3);
    out[3] = tr_lane_combine(lane);
}

/* 4 axpys on one y, in order, y loaded and stored once per 16 elements. Bit for bit 4 calls of
 * avx512_axpy_f32. */
TR_TARGET_AVX512
static void avx512_axpy_f32_x4(float *y, const float *x, int64_t stride, const float *a, int64_t n) {
    const float *x0 = x, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
    __m512 a0 = _mm512_set1_ps(a[0]), a1 = _mm512_set1_ps(a[1]), a2 = _mm512_set1_ps(a[2]),
           a3 = _mm512_set1_ps(a[3]);
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        __m512 v = _mm512_loadu_ps(y + k);
        v = _mm512_add_ps(v, _mm512_mul_ps(a0, _mm512_loadu_ps(x0 + k)));
        v = _mm512_add_ps(v, _mm512_mul_ps(a1, _mm512_loadu_ps(x1 + k)));
        v = _mm512_add_ps(v, _mm512_mul_ps(a2, _mm512_loadu_ps(x2 + k)));
        v = _mm512_add_ps(v, _mm512_mul_ps(a3, _mm512_loadu_ps(x3 + k)));
        _mm512_storeu_ps(y + k, v);
    }
    if (k < n) {
        __mmask16 m = (__mmask16)((1u << (unsigned)(n - k)) - 1u);
        __m512 v = _mm512_maskz_loadu_ps(m, y + k);
        v = _mm512_add_ps(v, _mm512_mul_ps(a0, _mm512_maskz_loadu_ps(m, x0 + k)));
        v = _mm512_add_ps(v, _mm512_mul_ps(a1, _mm512_maskz_loadu_ps(m, x1 + k)));
        v = _mm512_add_ps(v, _mm512_mul_ps(a2, _mm512_maskz_loadu_ps(m, x2 + k)));
        v = _mm512_add_ps(v, _mm512_mul_ps(a3, _mm512_maskz_loadu_ps(m, x3 + k)));
        _mm512_mask_storeu_ps(y + k, m, v);
    }
}

TR_TARGET_AVX512
static float avx512_dot_row_f32(const void *row, const float *x, int64_t n) {
    return avx512_dot_f32((const float *)row, x, n);
}

TR_TARGET_AVX512
static void avx512_dot_row_x4_f32(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    avx512_dot_f32_x4((const float *)row, x, stride, n, out);
}

/* F16 rows: 16 halves become 16 floats with one instruction (exact), then as dot_f32. */
TR_TARGET_AVX512
static inline __m512 avx512_f16_to_ps(const unsigned char *p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(const void *)p));
}

TR_TARGET_AVX512
static float avx512_dot_row_f16(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    __m512 acc = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES)
        acc = _mm512_add_ps(acc, _mm512_mul_ps(avx512_f16_to_ps(p + 2 * k), _mm512_loadu_ps(x + k)));
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc);
    for (; k < n; k++) lane[k % TR_LANES] += tail_half(p, k) * x[k];
    return tr_lane_combine(lane);
}

TR_TARGET_AVX512
static void avx512_dot_row_x4_f16(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    const float *x0 = x, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps(),
           acc3 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        __m512 w = avx512_f16_to_ps(p + 2 * k);
        acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w, _mm512_loadu_ps(x0 + k)));
        acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w, _mm512_loadu_ps(x1 + k)));
        acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(w, _mm512_loadu_ps(x2 + k)));
        acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(w, _mm512_loadu_ps(x3 + k)));
    }
    float l0[TR_LANES], l1[TR_LANES], l2[TR_LANES], l3[TR_LANES];
    _mm512_storeu_ps(l0, acc0);
    _mm512_storeu_ps(l1, acc1);
    _mm512_storeu_ps(l2, acc2);
    _mm512_storeu_ps(l3, acc3);
    for (; k < n; k++) {
        float w = tail_half(p, k);
        l0[k % TR_LANES] += w * x0[k];
        l1[k % TR_LANES] += w * x1[k];
        l2[k % TR_LANES] += w * x2[k];
        l3[k % TR_LANES] += w * x3[k];
    }
    out[0] = tr_lane_combine(l0);
    out[1] = tr_lane_combine(l1);
    out[2] = tr_lane_combine(l2);
    out[3] = tr_lane_combine(l3);
}

/* 16 int8 at p -> 16 exact floats */
TR_TARGET_AVX512
static inline __m512 avx512_i8_to_ps(const unsigned char *p) {
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)p)));
}

TR_TARGET_AVX512
static float avx512_dot_row_q8_0(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m512 acc = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m512 d = _mm512_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            /* 16 bytes read at qs + 16 end exactly at the end of the 34-byte block */
            __m512 q = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + j))));
            __m512 w = _mm512_mul_ps(d, q);
            acc = _mm512_add_ps(acc, _mm512_mul_ps(w, _mm512_loadu_ps(xb + j)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc);
    return tr_lane_combine(lane);
}

/* Q4_K. A sub-block's weights take only 16 values, scale * q - min for q = 0..15: they are
 * computed once per sub-block, exactly as scalar computes each weight, into one register, and
 * each quant picks its own with vpermps, which reads the low 4 bits of every index. No
 * conversion to float and no mask per element: the high nibbles need one shift, the low ones
 * nothing. Bytes 32c .. 32c+31 give elements 64c .. 64c+31 (low nibbles, sub-block 2c) and
 * 64c+32 .. 64c+63 (high nibbles, sub-block 2c+1), four groups of 16 lanes in order. */
TR_TARGET_AVX512
static inline __m512 avx512_q4_k_values(float scale, float min) {
    const __m512 q = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f);
    return _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(scale), q), _mm512_set1_ps(min));
}

TR_TARGET_AVX512
static float avx512_dot_row_q4_k(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    __m512 acc = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        const float *xb = x + b * TR_Q4_K_BLOCK_ELEMS;
        float scale[8], min[8];
        tr_q4_k_scales(blk, scale, min);
        for (int c = 0; c < 4; c++) {
            __m512 vlo = avx512_q4_k_values(scale[2 * c], min[2 * c]);
            __m512 vhi = avx512_q4_k_values(scale[2 * c + 1], min[2 * c + 1]);
            __m512i q0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c)));
            __m512i q1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c + 16)));
            const float *xc = xb + 64 * c;
            acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_permutexvar_ps(q0, vlo), _mm512_loadu_ps(xc)));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_permutexvar_ps(q1, vlo), _mm512_loadu_ps(xc + 16)));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q0, 4), vhi),
                                                   _mm512_loadu_ps(xc + 32)));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q1, 4), vhi),
                                                   _mm512_loadu_ps(xc + 48)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc);
    return tr_lane_combine(lane);
}

/* ---- one weight row against TR_DOT_TOKENS input rows ---------------------- */
/* The block scale and the decoded weights are computed once and multiplied into each
 * input row's own accumulators: out[t] is bit for bit the dot_row of that row, with a
 * quarter of the weight loads and conversions per element. */

TR_TARGET_AVX2
static void avx2_dot_row_x4_q8_0(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    /* one named register per input row, never an array: an indexed accumulator ends up on
     * the stack, and with it the whole point of one pass over the weight row */
    __m256 lo0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps(), lo2 = _mm256_setzero_ps(),
           lo3 = _mm256_setzero_ps();
    __m256 hi0 = _mm256_setzero_ps(), hi1 = _mm256_setzero_ps(), hi2 = _mm256_setzero_ps(),
           hi3 = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m256 d = _mm256_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m256 w0 = _mm256_mul_ps(d, avx2_i8_to_ps(qs + j));
            __m256 w1 = _mm256_mul_ps(d, avx2_i8_to_ps(qs + j + 8));
            const float *x0 = xb + j, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
            lo0 = _mm256_add_ps(lo0, _mm256_mul_ps(w0, _mm256_loadu_ps(x0)));
            hi0 = _mm256_add_ps(hi0, _mm256_mul_ps(w1, _mm256_loadu_ps(x0 + 8)));
            lo1 = _mm256_add_ps(lo1, _mm256_mul_ps(w0, _mm256_loadu_ps(x1)));
            hi1 = _mm256_add_ps(hi1, _mm256_mul_ps(w1, _mm256_loadu_ps(x1 + 8)));
            lo2 = _mm256_add_ps(lo2, _mm256_mul_ps(w0, _mm256_loadu_ps(x2)));
            hi2 = _mm256_add_ps(hi2, _mm256_mul_ps(w1, _mm256_loadu_ps(x2 + 8)));
            lo3 = _mm256_add_ps(lo3, _mm256_mul_ps(w0, _mm256_loadu_ps(x3)));
            hi3 = _mm256_add_ps(hi3, _mm256_mul_ps(w1, _mm256_loadu_ps(x3 + 8)));
        }
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo0);
    _mm256_storeu_ps(lane + 8, hi0);
    out[0] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo1);
    _mm256_storeu_ps(lane + 8, hi1);
    out[1] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo2);
    _mm256_storeu_ps(lane + 8, hi2);
    out[2] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo3);
    _mm256_storeu_ps(lane + 8, hi3);
    out[3] = tr_lane_combine(lane);
}

TR_TARGET_AVX512
static void avx512_dot_row_x4_q8_0(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    /* named registers, never an array (see the AVX2 variant above) */
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps(),
           acc3 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m512 d = _mm512_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m512 q = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + j))));
            __m512 w = _mm512_mul_ps(d, q);
            const float *x0 = xb + j, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w, _mm512_loadu_ps(x0)));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w, _mm512_loadu_ps(x1)));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(w, _mm512_loadu_ps(x2)));
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(w, _mm512_loadu_ps(x3)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc0);
    out[0] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc1);
    out[1] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc2);
    out[2] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc3);
    out[3] = tr_lane_combine(lane);
}

TR_TARGET_AVX512
static void avx512_dot_row_x4_q4_k(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps(),
           acc3 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        const float *xb = x + b * TR_Q4_K_BLOCK_ELEMS;
        float scale[8], min[8];
        tr_q4_k_scales(blk, scale, min);
        for (int c = 0; c < 4; c++) {
            __m512 vlo = avx512_q4_k_values(scale[2 * c], min[2 * c]);
            __m512 vhi = avx512_q4_k_values(scale[2 * c + 1], min[2 * c + 1]);
            __m512i q0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c)));
            __m512i q1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c + 16)));
            /* the four groups of 16 in element order (named registers, never an array: see
             * avx2_dot_row_x4_q8_0), each weight used for the four input rows */
            __m512 w0 = _mm512_permutexvar_ps(q0, vlo), w1 = _mm512_permutexvar_ps(q1, vlo);
            __m512 w2 = _mm512_permutexvar_ps(_mm512_srli_epi32(q0, 4), vhi);
            __m512 w3 = _mm512_permutexvar_ps(_mm512_srli_epi32(q1, 4), vhi);
            const float *x0 = xb + 64 * c, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w0, _mm512_loadu_ps(x0)));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w0, _mm512_loadu_ps(x1)));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(w0, _mm512_loadu_ps(x2)));
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(w0, _mm512_loadu_ps(x3)));
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w1, _mm512_loadu_ps(x0 + 16)));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w1, _mm512_loadu_ps(x1 + 16)));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(w1, _mm512_loadu_ps(x2 + 16)));
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(w1, _mm512_loadu_ps(x3 + 16)));
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w2, _mm512_loadu_ps(x0 + 32)));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w2, _mm512_loadu_ps(x1 + 32)));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(w2, _mm512_loadu_ps(x2 + 32)));
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(w2, _mm512_loadu_ps(x3 + 32)));
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w3, _mm512_loadu_ps(x0 + 48)));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w3, _mm512_loadu_ps(x1 + 48)));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(w3, _mm512_loadu_ps(x2 + 48)));
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(w3, _mm512_loadu_ps(x3 + 48)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc0);
    out[0] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc1);
    out[1] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc2);
    out[2] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc3);
    out[3] = tr_lane_combine(lane);
}

/* ---- Q6_K (kernels_internal.h): each weight scale * (q - 32), as scalar computes it ---- */
/* A sub-block is 16 weights of 64 possible values each: no lookup (building 64 values costs more
 * than the 16 conversions they would replace). The block's 256 quants are unpacked once, 32 bytes
 * at a time, into q - 32 as int8 in element order (ik_llama.cpp's DequantizerQ6K), and from there
 * the weights come as Q8_0's do: widen, convert, multiply by the sub-block's scale. */

/* Half h of a block (128 elements): A = ql[64h .. +31], B = ql[64h + 32 .. +31], H = qh[32h ..
 * +31]; its four runs of 32 are A's low nibbles with H's bits 0-1 above them, B's low with bits
 * 2-3, A's high with bits 4-5, B's high with bits 6-7. The 16-bit shifts carry bits across the
 * two bytes of a word only where the masks drop them. */
TR_TARGET_AVX2
static inline void avx2_q6_k_unpack(const unsigned char *blk, unsigned char q[TR_Q6_K_BLOCK_ELEMS]) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(0x30), k32 = _mm256_set1_epi8(32);
    for (int h = 0; h < 2; h++) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)(blk + 64 * h));
        __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)(blk + 64 * h + 32));
        __m256i hb = _mm256_loadu_si256((const __m256i *)(const void *)(blk + TR_Q6_K_QH_OFFSET + 32 * h));
        __m256i q0 = _mm256_or_si256(_mm256_and_si256(a, m4), _mm256_and_si256(_mm256_slli_epi16(hb, 4), m2));
        __m256i q1 = _mm256_or_si256(_mm256_and_si256(b, m4), _mm256_and_si256(_mm256_slli_epi16(hb, 2), m2));
        __m256i q2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a, 4), m4), _mm256_and_si256(hb, m2));
        __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b, 4), m4),
                                     _mm256_and_si256(_mm256_srli_epi16(hb, 2), m2));
        __m256i *o = (__m256i *)(void *)(q + 128 * h);
        _mm256_storeu_si256(o, _mm256_sub_epi8(q0, k32));
        _mm256_storeu_si256(o + 1, _mm256_sub_epi8(q1, k32));
        _mm256_storeu_si256(o + 2, _mm256_sub_epi8(q2, k32));
        _mm256_storeu_si256(o + 3, _mm256_sub_epi8(q3, k32));
    }
}

/* the sixteen d * sc of a block, as tr_q6_k_scales computes them (one rounding each) */
TR_TARGET_AVX2
static inline void avx2_q6_k_scales(const unsigned char *blk, float scale[16]) {
    uint16_t hd;
    memcpy(&hd, blk + TR_Q6_K_D_OFFSET, 2);
    __m256 d = _mm256_set1_ps(tr_half_to_float(hd));
    _mm256_storeu_ps(scale, _mm256_mul_ps(d, avx2_i8_to_ps(blk + TR_Q6_K_SCALES_OFFSET)));
    _mm256_storeu_ps(scale + 8, _mm256_mul_ps(d, avx2_i8_to_ps(blk + TR_Q6_K_SCALES_OFFSET + 8)));
}

TR_TARGET_AVX2
static float avx2_dot_row_q6_k(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        const float *xb = x + b * TR_Q6_K_BLOCK_ELEMS;
        unsigned char q[TR_Q6_K_BLOCK_ELEMS];
        float scale[16];
        avx2_q6_k_unpack(blk, q);
        avx2_q6_k_scales(blk, scale);
        for (int j = 0; j < 16; j++) {
            __m256 s = _mm256_set1_ps(scale[j]);
            __m256 w0 = _mm256_mul_ps(s, avx2_i8_to_ps(q + 16 * j));
            __m256 w1 = _mm256_mul_ps(s, avx2_i8_to_ps(q + 16 * j + 8));
            lo = _mm256_add_ps(lo, _mm256_mul_ps(w0, _mm256_loadu_ps(xb + 16 * j)));
            hi = _mm256_add_ps(hi, _mm256_mul_ps(w1, _mm256_loadu_ps(xb + 16 * j + 8)));
        }
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo);
    _mm256_storeu_ps(lane + 8, hi);
    return tr_lane_combine(lane);
}

TR_TARGET_AVX2
static void avx2_dot_row_x4_q6_k(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    /* named registers, never an array (see avx2_dot_row_x4_q8_0) */
    __m256 lo0 = _mm256_setzero_ps(), hi0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps(),
           hi1 = _mm256_setzero_ps(), lo2 = _mm256_setzero_ps(), hi2 = _mm256_setzero_ps(),
           lo3 = _mm256_setzero_ps(), hi3 = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        const float *xb = x + b * TR_Q6_K_BLOCK_ELEMS;
        unsigned char q[TR_Q6_K_BLOCK_ELEMS];
        float scale[16];
        avx2_q6_k_unpack(blk, q);
        avx2_q6_k_scales(blk, scale);
        for (int j = 0; j < 16; j++) {
            __m256 s = _mm256_set1_ps(scale[j]);
            __m256 w0 = _mm256_mul_ps(s, avx2_i8_to_ps(q + 16 * j));
            __m256 w1 = _mm256_mul_ps(s, avx2_i8_to_ps(q + 16 * j + 8));
            const float *x0 = xb + 16 * j, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
            lo0 = _mm256_add_ps(lo0, _mm256_mul_ps(w0, _mm256_loadu_ps(x0)));
            hi0 = _mm256_add_ps(hi0, _mm256_mul_ps(w1, _mm256_loadu_ps(x0 + 8)));
            lo1 = _mm256_add_ps(lo1, _mm256_mul_ps(w0, _mm256_loadu_ps(x1)));
            hi1 = _mm256_add_ps(hi1, _mm256_mul_ps(w1, _mm256_loadu_ps(x1 + 8)));
            lo2 = _mm256_add_ps(lo2, _mm256_mul_ps(w0, _mm256_loadu_ps(x2)));
            hi2 = _mm256_add_ps(hi2, _mm256_mul_ps(w1, _mm256_loadu_ps(x2 + 8)));
            lo3 = _mm256_add_ps(lo3, _mm256_mul_ps(w0, _mm256_loadu_ps(x3)));
            hi3 = _mm256_add_ps(hi3, _mm256_mul_ps(w1, _mm256_loadu_ps(x3 + 8)));
        }
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo0);
    _mm256_storeu_ps(lane + 8, hi0);
    out[0] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo1);
    _mm256_storeu_ps(lane + 8, hi1);
    out[1] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo2);
    _mm256_storeu_ps(lane + 8, hi2);
    out[2] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo3);
    _mm256_storeu_ps(lane + 8, hi3);
    out[3] = tr_lane_combine(lane);
}

TR_TARGET_AVX512
static inline void avx512_q6_k_scales(const unsigned char *blk, float scale[16]) {
    uint16_t hd;
    memcpy(&hd, blk + TR_Q6_K_D_OFFSET, 2);
    _mm512_storeu_ps(scale, _mm512_mul_ps(_mm512_set1_ps(tr_half_to_float(hd)), avx512_i8_to_ps(blk + TR_Q6_K_SCALES_OFFSET)));
}

TR_TARGET_AVX512
static float avx512_dot_row_q6_k(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    __m512 acc = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        const float *xb = x + b * TR_Q6_K_BLOCK_ELEMS;
        unsigned char q[TR_Q6_K_BLOCK_ELEMS];
        float scale[16];
        avx2_q6_k_unpack(blk, q);
        avx512_q6_k_scales(blk, scale);
        for (int j = 0; j < 16; j++) {
            __m512 w = _mm512_mul_ps(_mm512_set1_ps(scale[j]), avx512_i8_to_ps(q + 16 * j));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(w, _mm512_loadu_ps(xb + 16 * j)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc);
    return tr_lane_combine(lane);
}

TR_TARGET_AVX512
static void avx512_dot_row_x4_q6_k(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps(),
           acc3 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        const float *xb = x + b * TR_Q6_K_BLOCK_ELEMS;
        unsigned char q[TR_Q6_K_BLOCK_ELEMS];
        float scale[16];
        avx2_q6_k_unpack(blk, q);
        avx512_q6_k_scales(blk, scale);
        for (int j = 0; j < 16; j++) {
            __m512 w = _mm512_mul_ps(_mm512_set1_ps(scale[j]), avx512_i8_to_ps(q + 16 * j));
            const float *x0 = xb + 16 * j, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w, _mm512_loadu_ps(x0)));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w, _mm512_loadu_ps(x1)));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(w, _mm512_loadu_ps(x2)));
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(w, _mm512_loadu_ps(x3)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc0);
    out[0] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc1);
    out[1] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc2);
    out[2] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc3);
    out[3] = tr_lane_combine(lane);
}

/* ---- two weight rows against the same four input rows --------------------------------- */
/* Eight accumulators, row 0's and row 1's for each input row: every input vector is loaded once
 * and multiplied into both rows' (kernels.h, dot_row2_x4), so a matmul reads its activations half
 * as often. Each sum is still its own dot_row, bit for bit: the same weights, the same order. */

/* the eight sums, row 0's four then row 1's (named registers, never an array) */
#define TR_ROW2_OUT(out, a0, a1, a2, a3, b0, b1, b2, b3) \
    do { \
        float lane_[TR_LANES]; \
        _mm512_storeu_ps(lane_, a0); (out)[0] = tr_lane_combine(lane_); \
        _mm512_storeu_ps(lane_, a1); (out)[1] = tr_lane_combine(lane_); \
        _mm512_storeu_ps(lane_, a2); (out)[2] = tr_lane_combine(lane_); \
        _mm512_storeu_ps(lane_, a3); (out)[3] = tr_lane_combine(lane_); \
        _mm512_storeu_ps(lane_, b0); (out)[4] = tr_lane_combine(lane_); \
        _mm512_storeu_ps(lane_, b1); (out)[5] = tr_lane_combine(lane_); \
        _mm512_storeu_ps(lane_, b2); (out)[6] = tr_lane_combine(lane_); \
        _mm512_storeu_ps(lane_, b3); (out)[7] = tr_lane_combine(lane_); \
    } while (0)

/* 16 elements at offset e of the four input rows, times row 0's w and row 1's u */
#define TR_ROW2_STEP(x, stride, e, w, u) \
    do { \
        __m512 v_ = _mm512_loadu_ps((x) + (e)); \
        a0 = _mm512_add_ps(a0, _mm512_mul_ps(w, v_)); \
        b0 = _mm512_add_ps(b0, _mm512_mul_ps(u, v_)); \
        v_ = _mm512_loadu_ps((x) + (stride) + (e)); \
        a1 = _mm512_add_ps(a1, _mm512_mul_ps(w, v_)); \
        b1 = _mm512_add_ps(b1, _mm512_mul_ps(u, v_)); \
        v_ = _mm512_loadu_ps((x) + 2 * (stride) + (e)); \
        a2 = _mm512_add_ps(a2, _mm512_mul_ps(w, v_)); \
        b2 = _mm512_add_ps(b2, _mm512_mul_ps(u, v_)); \
        v_ = _mm512_loadu_ps((x) + 3 * (stride) + (e)); \
        a3 = _mm512_add_ps(a3, _mm512_mul_ps(w, v_)); \
        b3 = _mm512_add_ps(b3, _mm512_mul_ps(u, v_)); \
    } while (0)

TR_TARGET_AVX512
static void avx512_dot_row2_x4_q8_0(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n,
                                    float *out) {
    const unsigned char *p0 = (const unsigned char *)row0, *p1 = (const unsigned char *)row1;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps(), b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *k0 = p0 + (size_t)b * TR_Q8_0_BLOCK_BYTES, *k1 = p1 + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m512 d0 = _mm512_set1_ps(tr_q8_0_block_scale(k0)), d1 = _mm512_set1_ps(tr_q8_0_block_scale(k1));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m512 w = _mm512_mul_ps(d0, avx512_i8_to_ps(k0 + TR_Q8_0_SCALE_BYTES + j));
            __m512 u = _mm512_mul_ps(d1, avx512_i8_to_ps(k1 + TR_Q8_0_SCALE_BYTES + j));
            TR_ROW2_STEP(xb, stride, j, w, u);
        }
    }
    TR_ROW2_OUT(out, a0, a1, a2, a3, b0, b1, b2, b3);
}

TR_TARGET_AVX512
static void avx512_dot_row2_x4_q4_k(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n,
                                    float *out) {
    const unsigned char *p0 = (const unsigned char *)row0, *p1 = (const unsigned char *)row1;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps(), b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *k0 = p0 + (size_t)b * TR_Q4_K_BLOCK_BYTES, *k1 = p1 + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs0 = k0 + TR_Q4_K_QS_OFFSET, *qs1 = k1 + TR_Q4_K_QS_OFFSET;
        const float *xb = x + b * TR_Q4_K_BLOCK_ELEMS;
        float s0[8], m0[8], s1[8], m1[8];
        tr_q4_k_scales(k0, s0, m0);
        tr_q4_k_scales(k1, s1, m1);
        for (int c = 0; c < 4; c++) {
            __m512 vlo0 = avx512_q4_k_values(s0[2 * c], m0[2 * c]), vhi0 = avx512_q4_k_values(s0[2 * c + 1], m0[2 * c + 1]);
            __m512 vlo1 = avx512_q4_k_values(s1[2 * c], m1[2 * c]), vhi1 = avx512_q4_k_values(s1[2 * c + 1], m1[2 * c + 1]);
            __m512i q00 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs0 + 32 * c)));
            __m512i q01 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs0 + 32 * c + 16)));
            __m512i q10 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs1 + 32 * c)));
            __m512i q11 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs1 + 32 * c + 16)));
            const float *xc = xb + 64 * c;
            TR_ROW2_STEP(xc, stride, 0, _mm512_permutexvar_ps(q00, vlo0), _mm512_permutexvar_ps(q10, vlo1));
            TR_ROW2_STEP(xc, stride, 16, _mm512_permutexvar_ps(q01, vlo0), _mm512_permutexvar_ps(q11, vlo1));
            TR_ROW2_STEP(xc, stride, 32, _mm512_permutexvar_ps(_mm512_srli_epi32(q00, 4), vhi0),
                         _mm512_permutexvar_ps(_mm512_srli_epi32(q10, 4), vhi1));
            TR_ROW2_STEP(xc, stride, 48, _mm512_permutexvar_ps(_mm512_srli_epi32(q01, 4), vhi0),
                         _mm512_permutexvar_ps(_mm512_srli_epi32(q11, 4), vhi1));
        }
    }
    TR_ROW2_OUT(out, a0, a1, a2, a3, b0, b1, b2, b3);
}

TR_TARGET_AVX512
static void avx512_dot_row2_x4_q6_k(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n,
                                    float *out) {
    const unsigned char *p0 = (const unsigned char *)row0, *p1 = (const unsigned char *)row1;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps(), b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *k0 = p0 + (size_t)b * TR_Q6_K_BLOCK_BYTES, *k1 = p1 + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        const float *xb = x + b * TR_Q6_K_BLOCK_ELEMS;
        unsigned char q0[TR_Q6_K_BLOCK_ELEMS], q1[TR_Q6_K_BLOCK_ELEMS];
        float s0[16], s1[16];
        avx2_q6_k_unpack(k0, q0);
        avx2_q6_k_unpack(k1, q1);
        avx512_q6_k_scales(k0, s0);
        avx512_q6_k_scales(k1, s1);
        for (int j = 0; j < 16; j++) {
            __m512 w = _mm512_mul_ps(_mm512_set1_ps(s0[j]), avx512_i8_to_ps(q0 + 16 * j));
            __m512 u = _mm512_mul_ps(_mm512_set1_ps(s1[j]), avx512_i8_to_ps(q1 + 16 * j));
            TR_ROW2_STEP(xb, stride, 16 * j, w, u);
        }
    }
    TR_ROW2_OUT(out, a0, a1, a2, a3, b0, b1, b2, b3);
}

/* hot: end */

/* ---- tables --------------------------------------------------------------- */

static tr_kernels g_avx2, g_avx512;                 /* global-ok: kernel tables depend only on the CPU */
static int g_avx2_built = 0, g_avx512_built = 0;    /* global-ok: same */

const tr_kernels *tr_kernels_x86_tier(const char *tier) {
    const tr_cpu_info *c = tr_cpu();
    if (strcmp(tier, "avx2") == 0) {
        if (!c->avx2) return NULL;
        if (!g_avx2_built) {
            g_avx2 = *tr_kernels_scalar();
            g_avx2.tier = "avx2";
            g_avx2.dot_f32 = avx2_dot_f32;
            g_avx2.axpy_f32 = avx2_axpy_f32;
            g_avx2.dot_f32_x4 = avx2_dot_f32_x4;
            g_avx2.axpy_f32_x4 = avx2_axpy_f32_x4;
            g_avx2.dot_row[TR_TYPE_F32] = avx2_dot_row_f32;
            g_avx2.dot_row_x4[TR_TYPE_F32] = avx2_dot_row_x4_f32;
            if (c->f16c) {
                g_avx2.dot_row[TR_TYPE_F16] = avx2_dot_row_f16;
                g_avx2.dot_row_x4[TR_TYPE_F16] = avx2_dot_row_x4_f16;
            }
            g_avx2.dot_row[TR_TYPE_Q8_0] = avx2_dot_row_q8_0;
            g_avx2.dot_row_x4[TR_TYPE_Q8_0] = avx2_dot_row_x4_q8_0;
            g_avx2.dot_row[TR_TYPE_Q4_K] = avx2_dot_row_q4_k;
            g_avx2.dot_row_x4[TR_TYPE_Q4_K] = avx2_dot_row_x4_q4_k;
            g_avx2.dot_row[TR_TYPE_Q6_K] = avx2_dot_row_q6_k;
            g_avx2.dot_row_x4[TR_TYPE_Q6_K] = avx2_dot_row_x4_q6_k;
            g_avx2_built = 1;
        }
        return &g_avx2;
    }
    if (strcmp(tier, "avx512") == 0) {
        if (!c->avx512f) return NULL;
        if (!g_avx512_built) {
            g_avx512 = *tr_kernels_scalar();
            g_avx512.tier = "avx512";
            g_avx512.dot_f32 = avx512_dot_f32;
            g_avx512.axpy_f32 = avx512_axpy_f32;
            g_avx512.dot_f32_x4 = avx512_dot_f32_x4;
            g_avx512.axpy_f32_x4 = avx512_axpy_f32_x4;
            g_avx512.dot_row[TR_TYPE_F32] = avx512_dot_row_f32;
            g_avx512.dot_row_x4[TR_TYPE_F32] = avx512_dot_row_x4_f32;
            g_avx512.dot_row[TR_TYPE_F16] = avx512_dot_row_f16;
            g_avx512.dot_row_x4[TR_TYPE_F16] = avx512_dot_row_x4_f16;
            g_avx512.dot_row[TR_TYPE_Q8_0] = avx512_dot_row_q8_0;
            g_avx512.dot_row_x4[TR_TYPE_Q8_0] = avx512_dot_row_x4_q8_0;
            g_avx512.dot_row2_x4[TR_TYPE_Q8_0] = avx512_dot_row2_x4_q8_0;
            g_avx512.dot_row2_x4[TR_TYPE_Q4_K] = avx512_dot_row2_x4_q4_k;
            g_avx512.dot_row2_x4[TR_TYPE_Q6_K] = avx512_dot_row2_x4_q6_k;
            g_avx512.dot_row[TR_TYPE_Q4_K] = avx512_dot_row_q4_k;
            g_avx512.dot_row_x4[TR_TYPE_Q4_K] = avx512_dot_row_x4_q4_k;
            g_avx512.dot_row[TR_TYPE_Q6_K] = avx512_dot_row_q6_k;
            g_avx512.dot_row_x4[TR_TYPE_Q6_K] = avx512_dot_row_x4_q6_k;
            g_avx512_built = 1;
        }
        return &g_avx512;
    }
    return NULL;
}

#else

const tr_kernels *tr_kernels_x86_tier(const char *tier) {
    (void)tier;
    return NULL;
}

#endif
