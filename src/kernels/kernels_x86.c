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

/* The tiles of a prompt's attention as four x4 calls: sixteen accumulators of two ymm each would
 * not fit the sixteen registers. The same bits as the definition. */
TR_TARGET_AVX2
static void avx2_dot_f32_4x4(const float *a, int64_t a_stride, const float *b, int64_t stride, int64_t n, float scale,
                             float *out, int64_t out_stride) {
    for (int i = 0; i < TR_ATTN_X; i++) {
        float *o = out + i * out_stride;
        avx2_dot_f32_x4(a + i * a_stride, b, stride, n, o);
        for (int j = 0; j < TR_ATTN_X; j++) o[j] = o[j] * scale;
    }
}

TR_TARGET_AVX2
static void avx2_axpy_f32_4x4(float *y, int64_t y_stride, const float *x, int64_t stride, const float *a,
                              int64_t a_stride, int64_t n) {
    for (int i = 0; i < TR_ATTN_X; i++) avx2_axpy_f32_x4(y + i * y_stride, x, stride, a + i * a_stride, n);
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

/* The decode's Q4_K row (docs/MEASUREMENTS.md question 66, tests/bench_q4k_genome.c). Timed in
 * L1, tr_q4_k_scales's sixteen scalar conversions and multiplies were a quarter of the row; here
 * the sixteen scales and mins are computed in vectors (the same floats), stored, and each table
 * takes its scale and min as broadcasts from memory, which the multiply and the subtract fold in.
 * The block's scales are computed one block ahead: a broadcast of a float stored just before
 * waits for the store (0.96x the scalar decode), one stored a block earlier does not. From RAM,
 * a prefetch TR_Q4_K_PREFETCH bytes ahead keeps enough lines in flight for one thread. */
#define TR_Q4_K_PREFETCH 4608

/* sm[0..7] = scale[0..7], sm[8..15] = min[0..7] of the block at blk, as tr_q4_k_scales computes
 * them: the 12 packed bytes rearranged by two byte shuffles into sc0..7, m0..7 (6 bits each),
 * widened and converted (small integers, exact), times [d x 8, dmin x 8] (the halves converted
 * by vcvtph2ps: exact, and a NaN comes out quieted as the multiply would make it anyway) */
TR_TARGET_AVX512
static inline void avx512_q4_k_scales_store(const unsigned char *blk, float sm[16]) {
    const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(blk + 4));
    __m128i a = _mm_shuffle_epi8(v, _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11));
    __m128i hb = _mm_shuffle_epi8(v, _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7));
    /* bytes 12..15 (the mins 4..7) take the high nibble of theirs */
    __m128i sel = _mm_blend_epi16(a, _mm_and_si128(_mm_srli_epi16(a, 4), _mm_set1_epi8(0x0F)), 0xC0);
    __m128i low = _mm_and_si128(sel, _mm_setr_epi8(63, 63, 63, 63, 15, 15, 15, 15, 63, 63, 63, 63, 15, 15, 15, 15));
    __m128i bytes = _mm_or_si128(low, _mm_srli_epi16(_mm_and_si128(hb, _mm_set1_epi8((char)0xC0)), 2));
    uint32_t dd;
    memcpy(&dd, blk, 4);
    __m512 h = _mm512_cvtph_ps(_mm256_castsi128_si256(_mm_cvtsi32_si128((int)dd)));
    __m512 mul = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1), h);
    _mm512_store_ps(sm, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes)), mul));
}

TR_TARGET_AVX512
static float avx512_dot_row_q4_k(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    __m512 acc = _mm512_setzero_ps();
    const __m512 q = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f);
    float sm[2][16] __attribute__((aligned(64)));   /* block b's scales in sm[b & 1] */
    if (nb > 0) avx512_q4_k_scales_store(p, sm[0]);
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        const float *xb = x + b * TR_Q4_K_BLOCK_ELEMS;
        _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH, _MM_HINT_T0);
        _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH + 64, _MM_HINT_T0);
        _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH + 128, _MM_HINT_T0);
        if (b + 1 < nb) avx512_q4_k_scales_store(blk + TR_Q4_K_BLOCK_BYTES, sm[(b + 1) & 1]);
        const float *s = sm[b & 1];
        for (int c = 0; c < 4; c++) {
            /* scale * q - min for q = 0..15, as avx512_q4_k_values */
            __m512 vlo = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(s[2 * c]), q), _mm512_set1_ps(s[8 + 2 * c]));
            __m512 vhi = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(s[2 * c + 1]), q), _mm512_set1_ps(s[9 + 2 * c]));
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

/* Two Q4_K rows against one token (kernels.h, dot_row2), the decode's matmul two rows at a time.
 * Each row keeps its own accumulator, lanes and order: out[0], out[1] are avx512_dot_row_q4_k's
 * results bit for bit. The pair buys two chains of adds where one row has one, and the token's
 * loads shared: 450 against 497 cycles a row in L1 (tests/bench_q4k_genome.c). The scales of
 * TR_Q4_K_SPAN blocks of both rows are computed first, then those blocks: the early reads of each
 * block's first bytes are what keeps two interleaved rows streaming from RAM (computed one block
 * ahead instead, as the one-row kernel does, two rows ran 0.72x from RAM: short streams the
 * prefetchers do not follow, LESSONS #178). */
#define TR_Q4_K_SPAN 8

TR_TARGET_AVX512
static void avx512_dot_row2_q4_k(const void *row0, const void *row1, const float *x, int64_t n, float *out) {
    const unsigned char *p0 = (const unsigned char *)row0, *p1 = (const unsigned char *)row1;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    const __m512 q = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f);
    float s0[TR_Q4_K_SPAN][16] __attribute__((aligned(64))), s1[TR_Q4_K_SPAN][16] __attribute__((aligned(64)));
    for (int64_t b0 = 0; b0 < nb; b0 += TR_Q4_K_SPAN) {
        int64_t b1 = b0 + TR_Q4_K_SPAN < nb ? b0 + TR_Q4_K_SPAN : nb;
        for (int64_t b = b0; b < b1; b++) {
            avx512_q4_k_scales_store(p0 + (size_t)b * TR_Q4_K_BLOCK_BYTES, s0[b - b0]);
            avx512_q4_k_scales_store(p1 + (size_t)b * TR_Q4_K_BLOCK_BYTES, s1[b - b0]);
        }
        for (int64_t b = b0; b < b1; b++) {
            const unsigned char *k0 = p0 + (size_t)b * TR_Q4_K_BLOCK_BYTES, *k1 = p1 + (size_t)b * TR_Q4_K_BLOCK_BYTES;
            _mm_prefetch((const char *)k0 + TR_Q4_K_PREFETCH, _MM_HINT_T0);
            _mm_prefetch((const char *)k0 + TR_Q4_K_PREFETCH + 64, _MM_HINT_T0);
            _mm_prefetch((const char *)k0 + TR_Q4_K_PREFETCH + 128, _MM_HINT_T0);
            _mm_prefetch((const char *)k1 + TR_Q4_K_PREFETCH, _MM_HINT_T0);
            _mm_prefetch((const char *)k1 + TR_Q4_K_PREFETCH + 64, _MM_HINT_T0);
            _mm_prefetch((const char *)k1 + TR_Q4_K_PREFETCH + 128, _MM_HINT_T0);
            const float *xb = x + b * TR_Q4_K_BLOCK_ELEMS;
            const float *a = s0[b - b0], *d = s1[b - b0];
            for (int c = 0; c < 4; c++) {
                __m512 lo0 = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(a[2 * c]), q), _mm512_set1_ps(a[8 + 2 * c]));
                __m512 hi0 = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(a[2 * c + 1]), q), _mm512_set1_ps(a[9 + 2 * c]));
                __m512 lo1 = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(d[2 * c]), q), _mm512_set1_ps(d[8 + 2 * c]));
                __m512 hi1 = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(d[2 * c + 1]), q), _mm512_set1_ps(d[9 + 2 * c]));
                const unsigned char *q0p = k0 + TR_Q4_K_QS_OFFSET + 32 * c, *q1p = k1 + TR_Q4_K_QS_OFFSET + 32 * c;
                __m512i a0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)q0p));
                __m512i a1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(q0p + 16)));
                __m512i b0v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)q1p));
                __m512i b1v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(q1p + 16)));
                const float *xc = xb + 64 * c;
                __m512 x0 = _mm512_loadu_ps(xc), x1 = _mm512_loadu_ps(xc + 16);
                __m512 x2 = _mm512_loadu_ps(xc + 32), x3 = _mm512_loadu_ps(xc + 48);
                acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(_mm512_permutexvar_ps(a0, lo0), x0));
                acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(_mm512_permutexvar_ps(b0v, lo1), x0));
                acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(_mm512_permutexvar_ps(a1, lo0), x1));
                acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(_mm512_permutexvar_ps(b1v, lo1), x1));
                acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(a0, 4), hi0), x2));
                acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(b0v, 4), hi1), x2));
                acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(a1, 4), hi0), x3));
                acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(b1v, 4), hi1), x3));
            }
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc0);
    out[0] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc1);
    out[1] = tr_lane_combine(lane);
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

/* One level of the lane contract's tree (tr_lane_combine) for many sums at once: lanes 0..7 of
 * the result are a's adjacent pairs, 8..15 b's, each even lane plus the odd one after it, the even
 * on the left as in the scalar tree. Four levels take sixteen accumulators to their sixteen sums in
 * leaf order, each the tr_lane_combine of its own accumulator bit for bit: 45 vector instructions
 * instead of sixteen stores and 240 scalar adds, which cost a two-row call 10-20% at 1024-2048
 * columns (docs/MEASUREMENTS.md §Two rows against eight tokens). */
TR_TARGET_AVX512
static inline __m512 avx512_pair_sums(__m512 a, __m512 b) {
    const __m512i even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    return _mm512_add_ps(_mm512_permutex2var_ps(a, even, b), _mm512_permutex2var_ps(a, odd, b));
}

/* ---- a prompt's attention: four queries against four cache positions ------------------ */
/* Sixteen accumulators, one per (query, position): every key and query vector is loaded once for
 * four products, and the sixteen lane trees take four levels of pair sums (45 instructions)
 * instead of sixteen stores and scalar trees: 1.62x the x4 kernels on the products of a prompt's
 * attention in L1 (docs/MEASUREMENTS.md §The prompt's attention in tiles). Each score is
 * avx512_dot_f32's, times scale. A head_dim that is not a multiple of 16 goes through the x4
 * kernels, which handle the tail. */
#define TR_TILE_QUERY(qp, a0, a1, a2, a3) \
    do { \
        __m512 x_ = _mm512_loadu_ps((qp) + d); \
        a0 = _mm512_add_ps(a0, _mm512_mul_ps(x_, w0)); \
        a1 = _mm512_add_ps(a1, _mm512_mul_ps(x_, w1)); \
        a2 = _mm512_add_ps(a2, _mm512_mul_ps(x_, w2)); \
        a3 = _mm512_add_ps(a3, _mm512_mul_ps(x_, w3)); \
    } while (0)

TR_TARGET_AVX512
static void avx512_dot_f32_4x4(const float *a, int64_t a_stride, const float *b, int64_t stride, int64_t n,
                               float scale, float *out, int64_t out_stride) {
    if (n % TR_LANES != 0) {
        for (int i = 0; i < TR_ATTN_X; i++) {
            float *o = out + i * out_stride;
            avx512_dot_f32_x4(a + i * a_stride, b, stride, n, o);
            for (int j = 0; j < TR_ATTN_X; j++) o[j] = o[j] * scale;
        }
        return;
    }
    const float *b0 = b, *b1 = b0 + stride, *b2 = b1 + stride, *b3 = b2 + stride;
    const float *q0 = a, *q1 = q0 + a_stride, *q2 = q1 + a_stride, *q3 = q2 + a_stride;
    __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00,
           a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    for (int64_t d = 0; d < n; d += TR_LANES) {
        __m512 w0 = _mm512_loadu_ps(b0 + d), w1 = _mm512_loadu_ps(b1 + d), w2 = _mm512_loadu_ps(b2 + d),
               w3 = _mm512_loadu_ps(b3 + d);
        TR_TILE_QUERY(q0, a00, a01, a02, a03);
        TR_TILE_QUERY(q1, a10, a11, a12, a13);
        TR_TILE_QUERY(q2, a20, a21, a22, a23);
        TR_TILE_QUERY(q3, a30, a31, a32, a33);
    }
    /* leaf order: lanes 4i..4i+3 are query i's four scores */
    __m512 lo = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(a00, a01), avx512_pair_sums(a02, a03)),
                                 avx512_pair_sums(avx512_pair_sums(a10, a11), avx512_pair_sums(a12, a13)));
    __m512 hi = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(a20, a21), avx512_pair_sums(a22, a23)),
                                 avx512_pair_sums(avx512_pair_sums(a30, a31), avx512_pair_sums(a32, a33)));
    __m512 r = _mm512_mul_ps(avx512_pair_sums(lo, hi), _mm512_set1_ps(scale));
    _mm_storeu_ps(out, _mm512_extractf32x4_ps(r, 0));
    _mm_storeu_ps(out + out_stride, _mm512_extractf32x4_ps(r, 1));
    _mm_storeu_ps(out + 2 * out_stride, _mm512_extractf32x4_ps(r, 2));
    _mm_storeu_ps(out + 3 * out_stride, _mm512_extractf32x4_ps(r, 3));
}

/* 16 elements at offset d of the value row w_, into the four outputs with their weights */
#define TR_TILE_VALUE(vp, s0, s1, s2, s3) \
    do { \
        __m512 w_ = _mm512_loadu_ps((vp) + d); \
        y0 = _mm512_add_ps(y0, _mm512_mul_ps(s0, w_)); \
        y1 = _mm512_add_ps(y1, _mm512_mul_ps(s1, w_)); \
        y2 = _mm512_add_ps(y2, _mm512_mul_ps(s2, w_)); \
        y3 = _mm512_add_ps(y3, _mm512_mul_ps(s3, w_)); \
    } while (0)

/* Four outputs against four value rows: each value vector loaded once for the four queries, each
 * output loaded and stored once per four additions, which go in increasing position for every
 * output. Each addition is avx512_axpy_f32's. */
TR_TARGET_AVX512
static void avx512_axpy_f32_4x4(float *y, int64_t y_stride, const float *x, int64_t stride, const float *a,
                                int64_t a_stride, int64_t n) {
    if (n % TR_LANES != 0) {
        for (int i = 0; i < TR_ATTN_X; i++) avx512_axpy_f32_x4(y + i * y_stride, x, stride, a + i * a_stride, n);
        return;
    }
    const float *x0 = x, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
    const float *r0 = a, *r1 = r0 + a_stride, *r2 = r1 + a_stride, *r3 = r2 + a_stride;
    __m512 s00 = _mm512_set1_ps(r0[0]), s01 = _mm512_set1_ps(r0[1]), s02 = _mm512_set1_ps(r0[2]),
           s03 = _mm512_set1_ps(r0[3]), s10 = _mm512_set1_ps(r1[0]), s11 = _mm512_set1_ps(r1[1]),
           s12 = _mm512_set1_ps(r1[2]), s13 = _mm512_set1_ps(r1[3]), s20 = _mm512_set1_ps(r2[0]),
           s21 = _mm512_set1_ps(r2[1]), s22 = _mm512_set1_ps(r2[2]), s23 = _mm512_set1_ps(r2[3]),
           s30 = _mm512_set1_ps(r3[0]), s31 = _mm512_set1_ps(r3[1]), s32 = _mm512_set1_ps(r3[2]),
           s33 = _mm512_set1_ps(r3[3]);
    float *o0 = y, *o1 = o0 + y_stride, *o2 = o1 + y_stride, *o3 = o2 + y_stride;
    for (int64_t d = 0; d < n; d += TR_LANES) {
        __m512 y0 = _mm512_loadu_ps(o0 + d), y1 = _mm512_loadu_ps(o1 + d), y2 = _mm512_loadu_ps(o2 + d),
               y3 = _mm512_loadu_ps(o3 + d);
        TR_TILE_VALUE(x0, s00, s10, s20, s30);
        TR_TILE_VALUE(x1, s01, s11, s21, s31);
        TR_TILE_VALUE(x2, s02, s12, s22, s32);
        TR_TILE_VALUE(x3, s03, s13, s23, s33);
        _mm512_storeu_ps(o0 + d, y0);
        _mm512_storeu_ps(o1 + d, y1);
        _mm512_storeu_ps(o2 + d, y2);
        _mm512_storeu_ps(o3 + d, y3);
    }
}

/* the eight sums, row 0's four then row 1's (named registers, never an array); the last level
 * pairs the vector with itself and keeps its low half */
#define TR_ROW2_OUT(out, a0, a1, a2, a3, b0, b1, b2, b3) \
    do { \
        __m512 h_ = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(a0, a1), avx512_pair_sums(a2, a3)), \
                                     avx512_pair_sums(avx512_pair_sums(b0, b1), avx512_pair_sums(b2, b3))); \
        _mm256_storeu_ps(out, _mm512_castps512_ps256(avx512_pair_sums(h_, h_))); \
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

/* ---- two weight rows against the same eight input rows -------------------------------- */
/* Sixteen accumulators, row 0's and row 1's for each input row (kernels.h, dot_row2_x8): each
 * weight vector decoded once for eight tokens instead of four, 26 of the 32 zmm, no spill (gcc,
 * checked in the disassembly). Each sum is still its own dot_row, bit for bit. Two instruction
 * orders measured the same (per token, or two tokens' loads, then four multiplies, then four adds:
 * docs/MEASUREMENTS.md §Two rows against eight tokens), so the compiler keeps its own. */

/* the sixteen sums, row 0's eight then row 1's, in one store */
#define TR_ROW2X8_OUT(out) \
    do { \
        __m512 s0_ = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(a0, a1), avx512_pair_sums(a2, a3)), \
                                      avx512_pair_sums(avx512_pair_sums(a4, a5), avx512_pair_sums(a6, a7))); \
        __m512 s1_ = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(b0, b1), avx512_pair_sums(b2, b3)), \
                                      avx512_pair_sums(avx512_pair_sums(b4, b5), avx512_pair_sums(b6, b7))); \
        _mm512_storeu_ps(out, avx512_pair_sums(s0_, s1_)); \
    } while (0)

/* 16 elements at offset e of input row xr, times row 0's w into A and row 1's u into B */
#define TR_ROW2_PAIR(xr, e, w, u, A, B) \
    do { \
        __m512 v_ = _mm512_loadu_ps((xr) + (e)); \
        A = _mm512_add_ps(A, _mm512_mul_ps(w, v_)); \
        B = _mm512_add_ps(B, _mm512_mul_ps(u, v_)); \
    } while (0)

/* 16 elements at offset e of the eight input rows x, x1 .. x7 */
#define TR_ROW2X8_STEP(e, w, u) \
    do { \
        __m512 w_ = (w), u_ = (u); \
        TR_ROW2_PAIR(x, e, w_, u_, a0, b0); \
        TR_ROW2_PAIR(x1, e, w_, u_, a1, b1); \
        TR_ROW2_PAIR(x2, e, w_, u_, a2, b2); \
        TR_ROW2_PAIR(x3, e, w_, u_, a3, b3); \
        TR_ROW2_PAIR(x4, e, w_, u_, a4, b4); \
        TR_ROW2_PAIR(x5, e, w_, u_, a5, b5); \
        TR_ROW2_PAIR(x6, e, w_, u_, a6, b6); \
        TR_ROW2_PAIR(x7, e, w_, u_, a7, b7); \
    } while (0)

/* the sixteen accumulators and the eight input rows */
#define TR_ROW2X8_BEGIN \
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps(); \
    __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps(), a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps(); \
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps(), b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps(); \
    __m512 b4 = _mm512_setzero_ps(), b5 = _mm512_setzero_ps(), b6 = _mm512_setzero_ps(), b7 = _mm512_setzero_ps(); \
    const float *x1 = x + stride, *x2 = x1 + stride, *x3 = x2 + stride, *x4 = x3 + stride, *x5 = x4 + stride, \
                *x6 = x5 + stride, *x7 = x6 + stride

TR_TARGET_AVX512
static void avx512_dot_row2_x8_q8_0(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n,
                                    float *out) {
    const unsigned char *p0 = (const unsigned char *)row0, *p1 = (const unsigned char *)row1;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    TR_ROW2X8_BEGIN;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *k0 = p0 + (size_t)b * TR_Q8_0_BLOCK_BYTES, *k1 = p1 + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        int64_t o = b * TR_Q8_0_BLOCK_ELEMS;
        /* the scalar conversion runs on the integer units, beside the FP pipes that bound this loop:
         * converted ahead, by vcvtph2ps or a gather, the scales cost 1-12% (MEASUREMENTS) */
        __m512 d0 = _mm512_set1_ps(tr_q8_0_block_scale(k0)), d1 = _mm512_set1_ps(tr_q8_0_block_scale(k1));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES)
            TR_ROW2X8_STEP(o + j, _mm512_mul_ps(d0, avx512_i8_to_ps(k0 + TR_Q8_0_SCALE_BYTES + j)),
                           _mm512_mul_ps(d1, avx512_i8_to_ps(k1 + TR_Q8_0_SCALE_BYTES + j)));
    }
    TR_ROW2X8_OUT(out);
}

TR_TARGET_AVX512
static void avx512_dot_row2_x8_q4_k(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n,
                                    float *out) {
    const unsigned char *p0 = (const unsigned char *)row0, *p1 = (const unsigned char *)row1;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    TR_ROW2X8_BEGIN;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *k0 = p0 + (size_t)b * TR_Q4_K_BLOCK_BYTES, *k1 = p1 + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs0 = k0 + TR_Q4_K_QS_OFFSET, *qs1 = k1 + TR_Q4_K_QS_OFFSET;
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
            int64_t o = b * TR_Q4_K_BLOCK_ELEMS + 64 * c;
            TR_ROW2X8_STEP(o, _mm512_permutexvar_ps(q00, vlo0), _mm512_permutexvar_ps(q10, vlo1));
            TR_ROW2X8_STEP(o + 16, _mm512_permutexvar_ps(q01, vlo0), _mm512_permutexvar_ps(q11, vlo1));
            TR_ROW2X8_STEP(o + 32, _mm512_permutexvar_ps(_mm512_srli_epi32(q00, 4), vhi0),
                           _mm512_permutexvar_ps(_mm512_srli_epi32(q10, 4), vhi1));
            TR_ROW2X8_STEP(o + 48, _mm512_permutexvar_ps(_mm512_srli_epi32(q01, 4), vhi0),
                           _mm512_permutexvar_ps(_mm512_srli_epi32(q11, 4), vhi1));
        }
    }
    TR_ROW2X8_OUT(out);
}

TR_TARGET_AVX512
static void avx512_dot_row2_x8_q6_k(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n,
                                    float *out) {
    const unsigned char *p0 = (const unsigned char *)row0, *p1 = (const unsigned char *)row1;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    TR_ROW2X8_BEGIN;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *k0 = p0 + (size_t)b * TR_Q6_K_BLOCK_BYTES, *k1 = p1 + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        int64_t o = b * TR_Q6_K_BLOCK_ELEMS;
        unsigned char q0[TR_Q6_K_BLOCK_ELEMS], q1[TR_Q6_K_BLOCK_ELEMS];
        float s0[16], s1[16];
        avx2_q6_k_unpack(k0, q0);
        avx2_q6_k_unpack(k1, q1);
        avx512_q6_k_scales(k0, s0);
        avx512_q6_k_scales(k1, s1);
        for (int j = 0; j < 16; j++)
            TR_ROW2X8_STEP(o + 16 * j, _mm512_mul_ps(_mm512_set1_ps(s0[j]), avx512_i8_to_ps(q0 + 16 * j)),
                           _mm512_mul_ps(_mm512_set1_ps(s1[j]), avx512_i8_to_ps(q1 + 16 * j)));
    }
    TR_ROW2X8_OUT(out);
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
            g_avx2.dot_f32_4x4 = avx2_dot_f32_4x4;
            g_avx2.axpy_f32_4x4 = avx2_axpy_f32_4x4;
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
            g_avx512.dot_f32_4x4 = avx512_dot_f32_4x4;
            g_avx512.axpy_f32_4x4 = avx512_axpy_f32_4x4;
            g_avx512.dot_row[TR_TYPE_F32] = avx512_dot_row_f32;
            g_avx512.dot_row_x4[TR_TYPE_F32] = avx512_dot_row_x4_f32;
            g_avx512.dot_row[TR_TYPE_F16] = avx512_dot_row_f16;
            g_avx512.dot_row_x4[TR_TYPE_F16] = avx512_dot_row_x4_f16;
            g_avx512.dot_row[TR_TYPE_Q8_0] = avx512_dot_row_q8_0;
            g_avx512.dot_row_x4[TR_TYPE_Q8_0] = avx512_dot_row_x4_q8_0;
            g_avx512.dot_row2_x4[TR_TYPE_Q8_0] = avx512_dot_row2_x4_q8_0;
            g_avx512.dot_row2_x4[TR_TYPE_Q4_K] = avx512_dot_row2_x4_q4_k;
            g_avx512.dot_row2_x4[TR_TYPE_Q6_K] = avx512_dot_row2_x4_q6_k;
            g_avx512.dot_row2_x8[TR_TYPE_Q8_0] = avx512_dot_row2_x8_q8_0;
            g_avx512.dot_row2_x8[TR_TYPE_Q4_K] = avx512_dot_row2_x8_q4_k;
            g_avx512.dot_row2_x8[TR_TYPE_Q6_K] = avx512_dot_row2_x8_q6_k;
            g_avx512.dot_row[TR_TYPE_Q4_K] = avx512_dot_row_q4_k;
            g_avx512.dot_row2[TR_TYPE_Q4_K] = avx512_dot_row2_q4_k;
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
