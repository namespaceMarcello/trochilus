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


/* ---- exp: tr_expf on 16 (AVX-512) or 8 (AVX2) floats at a time (docs/MEASUREMENTS.md question 70) -- *
 * expf32's fast path, the variant without fma (tests/bench_expf32.c, question 58), float and int32
 * operations only: k = round(x 256/ln2), j = k mod 256, e = floor(k/256); r = x - k ln2/256 in two
 * floats, ln2/256 in three pieces of 8 bits so that every product is exact; exp(r) - 1 - r a degree-3
 * polynomial; 2^(j/256) = th + tl from a table of 256 pairs (gathers); th rh = ph + pl by Veltkamp's
 * split and Dekker's product; y = yh + yl within 2^-39.9. A lane is settled where an error of D =
 * 2^-39 cannot move its rounding (below 2^-126 on the subnormal grid), and its bits are then the
 * correctly rounded exp, that is tr_expf's. The lanes the test does not settle, about 1 in 33 000,
 * take tr_expf itself, the scalar definition: no second slow path to prove. The same operations lane
 * by lane in both tiers, no FMA; tests/test_expf.c compares every tier with tr_expf on all 2^32
 * floats. Constants and table generated by gen_consts.py (mpmath, 300 bits), as in bench_expf32.c. */
#define XEXP_INVL 0x1.7154760000000p+8f    /* 256/ln2 */
#define XEXP_SHIFT 0x1.8p23f               /* 1.5 2^23: adding it rounds to an integer */
#define XEXP_LA 0x1.6200000000000p-9f      /* ln2/256 rounded to float = LA + LB + LC, 8 bits each */
#define XEXP_LB 0x1.c800000000000p-18f
#define XEXP_LC 0x1.8000000000000p-28f
#define XEXP_NLL 0x1.05c6100000000p-37f    /* -(ln2/256 - (LA + LB + LC)) rounded to float */
#define XEXP_C2 0x1.0000020000000p-1f      /* 1/2 + 2^-24: minimax with C3 */
#define XEXP_C3 0x1.5555560000000p-3f      /* 1/6 */
#define XEXP_X_NORM -0x1.5d589e0000000p+6f /* smallest float >= -126 ln2 */
#define XEXP_X_OVF 0x1.62e42ep+6f          /* the largest float whose exp is finite */
#define XEXP_X_UF -0x1.a0p+6f              /* below -104 exp(x) < 2^-150: zero */
#define XEXP_D 0x1p-39f                    /* what the fast path may be off by, in units of y */
#define XEXP_SPLIT 4097.0f                 /* Veltkamp: 2^12 + 1 */

/* 2^(j/256) = hi + lo, laid out hi, lo: a gather at 2j reads hi, at 2j + 1 lo */
static const float xexp_tab[] = {
    0x1.0000000000000p+0f, 0.0f, 0x1.00b1b00000000p+0f, -0x1.6950d00000000p-26f,
    0x1.0163da0000000p+0f, 0x1.3f66660000000p-25f, 0x1.0216820000000p+0f, -0x1.789fb00000000p-25f,
    0x1.02c9a40000000p+0f, -0x1.887fa00000000p-28f, 0x1.037d420000000p+0f, 0x1.c2377a0000000p-25f,
    0x1.04315e0000000p+0f, 0x1.0dcff00000000p-25f, 0x1.04e5f80000000p+0f, -0x1.a1356a0000000p-25f,
    0x1.059b0e0000000p+0f, -0x1.9d4f520000000p-25f, 0x1.0650a00000000p+0f, 0x1.c783f20000000p-25f,
    0x1.0706b20000000p+0f, 0x1.3bbedc0000000p-25f, 0x1.07bd420000000p+0f, 0x1.6e55060000000p-25f,
    0x1.0874520000000p+0f, -0x1.e2990e0000000p-26f, 0x1.092be00000000p+0f, -0x1.333f040000000p-25f,
    0x1.09e3ec0000000p+0f, 0x1.58de700000000p-25f, 0x1.0a9c7a0000000p+0f, -0x1.3831ba0000000p-26f,
    0x1.0b55860000000p+0f, 0x1.9f31220000000p-25f, 0x1.0c0f140000000p+0f, 0x1.791b220000000p-26f,
    0x1.0cc9220000000p+0f, 0x1.6e48fe0000000p-25f, 0x1.0d83b20000000p+0f, 0x1.9caef60000000p-27f,
    0x1.0e3ec40000000p+0f, -0x1.a585cc0000000p-25f, 0x1.0efa560000000p+0f, -0x1.02b1da0000000p-31f,
    0x1.0fb66a0000000p+0f, 0x1.ffda640000000p-25f, 0x1.1073020000000p+0f, 0x1.1ae4680000000p-25f,
    0x1.11301e0000000p+0f, -0x1.fdb4960000000p-25f, 0x1.11edba0000000p+0f, 0x1.6bc5560000000p-25f,
    0x1.12abdc0000000p+0f, 0x1.b0c7300000000p-30f, 0x1.136a820000000p+0f, -0x1.61bf6a0000000p-25f,
    0x1.1429aa0000000p+0f, 0x1.d525bc0000000p-25f, 0x1.14e95a0000000p+0f, -0x1.9619da0000000p-25f,
    0x1.15a98c0000000p+0f, 0x1.14b1ca0000000p-25f, 0x1.166a460000000p+0f, -0x1.71c7880000000p-25f,
    0x1.172b840000000p+0f, -0x1.c157420000000p-27f, 0x1.17ed480000000p+0f, 0x1.a56ef00000000p-26f,
    0x1.18af940000000p+0f, -0x1.dcdc860000000p-26f, 0x1.1972660000000p+0f, -0x1.f228b40000000p-26f,
    0x1.1a35be0000000p+0f, 0x1.6df96e0000000p-25f, 0x1.1af9a00000000p+0f, -0x1.fb1d780000000p-26f,
    0x1.1bbe080000000p+0f, 0x1.0117340000000p-26f, 0x1.1c82fa0000000p+0f, -0x1.5afc720000000p-25f,
    0x1.1d48740000000p+0f, -0x1.d2e8ca0000000p-25f, 0x1.1e0e760000000p+0f, -0x1.4bbfda0000000p-28f,
    0x1.1ed5020000000p+0f, 0x1.7e6c8e0000000p-27f, 0x1.1f9c180000000p+0f, 0x1.0e33940000000p-26f,
    0x1.2063b80000000p+0f, 0x1.0c519a0000000p-25f, 0x1.212be40000000p+0f, -0x1.50eafc0000000p-25f,
    0x1.21f49a0000000p+0f, -0x1.d0446e0000000p-25f, 0x1.22bdda0000000p+0f, 0x1.3c89680000000p-27f,
    0x1.2387a60000000p+0f, 0x1.ceac480000000p-25f, 0x1.2452000000000p+0f, -0x1.1f7afe0000000p-26f,
    0x1.251ce40000000p+0f, 0x1.f654c80000000p-25f, 0x1.25e8580000000p+0f, -0x1.dc26320000000p-25f,
    0x1.26b4560000000p+0f, 0x1.789f380000000p-26f, 0x1.2780e40000000p+0f, -0x1.7c441a0000000p-25f,
    0x1.284dfe0000000p+0f, 0x1.f563800000000p-28f, 0x1.291ba80000000p+0f, -0x1.4dc8920000000p-25f,
    0x1.29e9e00000000p+0f, -0x1.5c04240000000p-25f, 0x1.2ab8a60000000p+0f, 0x1.b443c40000000p-26f,
    0x1.2b87fe0000000p+0f, -0x1.e4a4ce0000000p-25f, 0x1.2c57e40000000p+0f, -0x1.a239340000000p-26f,
    0x1.2d285a0000000p+0f, 0x1.b900c20000000p-26f, 0x1.2df9620000000p+0f, -0x1.37d4ee0000000p-29f,
    0x1.2ecafa0000000p+0f, 0x1.27c5ea0000000p-25f, 0x1.2f9d240000000p+0f, 0x1.57b10e0000000p-25f,
    0x1.306fe00000000p+0f, 0x1.4636e20000000p-25f, 0x1.31432e0000000p+0f, 0x1.bdd6600000000p-25f,
    0x1.3217100000000p+0f, -0x1.d993e80000000p-27f, 0x1.32eb840000000p+0f, -0x1.15c5740000000p-26f,
    0x1.33c08c0000000p+0f, -0x1.b37d200000000p-25f, 0x1.3496260000000p+0f, 0x1.b8fe8c0000000p-26f,
    0x1.356c560000000p+0f, -0x1.b5803c0000000p-30f, 0x1.36431a0000000p+0f, 0x1.6f441e0000000p-27f,
    0x1.371a740000000p+0f, -0x1.18aac60000000p-25f, 0x1.37f2620000000p+0f, 0x1.8f3aa40000000p-27f,
    0x1.38cae60000000p+0f, 0x1.a0bb0c0000000p-25f, 0x1.39a4020000000p+0f, -0x1.23afc40000000p-26f,
    0x1.3a7db40000000p+0f, -0x1.634c020000000p-25f, 0x1.3b57fc0000000p+0f, -0x1.3930ba0000000p-32f,
    0x1.3c32dc0000000p+0f, 0x1.89d4720000000p-27f, 0x1.3d0e540000000p+0f, 0x1.3b785c0000000p-26f,
    0x1.3dea640000000p+0f, 0x1.8246840000000p-25f, 0x1.3ec70e0000000p+0f, -0x1.c75d160000000p-29f,
    0x1.3fa4500000000p+0f, 0x1.2b20060000000p-26f, 0x1.40822c0000000p+0f, 0x1.b3d0120000000p-27f,
    0x1.4160a20000000p+0f, 0x1.f72e2a0000000p-28f, 0x1.423fb20000000p+0f, 0x1.c251a20000000p-26f,
    0x1.431f5e0000000p+0f, -0x1.abd5da0000000p-26f, 0x1.43ffa40000000p+0f, -0x1.ed18b00000000p-30f,
    0x1.44e0860000000p+0f, 0x1.8624b40000000p-30f, 0x1.45c2040000000p+0f, 0x1.53e9180000000p-27f,
    0x1.46a41e0000000p+0f, 0x1.a3a00a0000000p-25f, 0x1.4786d60000000p+0f, 0x1.a2cc8e0000000p-26f,
    0x1.486a2c0000000p+0f, -0x1.47d8660000000p-25f, 0x1.494e1e0000000p+0f, 0x1.92aed20000000p-28f,
    0x1.4a32b00000000p+0f, -0x1.e505840000000p-25f, 0x1.4b17de0000000p+0f, 0x1.4db6fa0000000p-25f,
    0x1.4bfdae0000000p+0f, -0x1.593abc0000000p-25f, 0x1.4ce41c0000000p+0f, -0x1.fa0fba0000000p-26f,
    0x1.4dcb2a0000000p+0f, -0x1.8088bc0000000p-26f, 0x1.4eb2d80000000p+0f, 0x1.d8abfe0000000p-28f,
    0x1.4f9b280000000p+0f, -0x1.2c5a6c0000000p-25f, 0x1.5084180000000p+0f, -0x1.759c240000000p-29f,
    0x1.516daa0000000p+0f, 0x1.67b3200000000p-27f, 0x1.5257de0000000p+0f, 0x1.07e9de0000000p-25f,
    0x1.5342b60000000p+0f, -0x1.2c56100000000p-25f, 0x1.542e300000000p+0f, -0x1.612a5c0000000p-25f,
    0x1.551a4c0000000p+0f, 0x1.4bb2420000000p-25f, 0x1.56070e0000000p+0f, -0x1.0b77980000000p-27f,
    0x1.56f4740000000p+0f, -0x1.295b040000000p-25f, 0x1.57e27e0000000p+0f, -0x1.074ecc0000000p-26f,
    0x1.58d12e0000000p+0f, -0x1.6d07000000000p-25f, 0x1.59c0820000000p+0f, 0x1.ffc1f20000000p-26f,
    0x1.5ab07e0000000p+0f, -0x1.5bd5ec0000000p-27f, 0x1.5ba1200000000p+0f, -0x1.15e1800000000p-26f,
    0x1.5c92680000000p+0f, 0x1.4b28d60000000p-25f, 0x1.5d845a0000000p+0f, -0x1.ecce8e0000000p-25f,
    0x1.5e76f20000000p+0f, -0x1.4a5bd60000000p-25f, 0x1.5f6a320000000p+0f, 0x1.b9d6e20000000p-29f,
    0x1.605e1c0000000p+0f, -0x1.a248fe0000000p-26f, 0x1.6152ae0000000p+0f, 0x1.b37dbe0000000p-26f,
    0x1.6247ec0000000p+0f, -0x1.f8b5500000000p-25f, 0x1.633dd20000000p+0f, -0x1.736b020000000p-27f,
    0x1.6434640000000p+0f, -0x1.66679c0000000p-25f, 0x1.652ba00000000p+0f, -0x1.43704a0000000p-28f,
    0x1.6623880000000p+0f, 0x1.2a91120000000p-27f, 0x1.671c1c0000000p+0f, 0x1.c20cfe0000000p-26f,
    0x1.68155e0000000p+0f, -0x1.766ad20000000p-25f, 0x1.690f4c0000000p+0f, -0x1.cc2d580000000p-25f,
    0x1.6a09e60000000p+0f, 0x1.9fcef40000000p-26f, 0x1.6b05300000000p+0f, -0x1.62ba300000000p-26f,
    0x1.6c01280000000p+0f, -0x1.5e84a80000000p-25f, 0x1.6cfdce0000000p+0f, -0x1.15c4de0000000p-27f,
    0x1.6dfb240000000p+0f, -0x1.cd72e80000000p-27f, 0x1.6ef92a0000000p+0f, -0x1.e9b1460000000p-26f,
    0x1.6ff7e00000000p+0f, -0x1.ab9ae00000000p-26f, 0x1.70f7460000000p+0f, 0x1.bd0ba20000000p-26f,
    0x1.71f75e0000000p+0f, 0x1.1d8bee0000000p-25f, 0x1.72f8280000000p+0f, 0x1.bab4220000000p-26f,
    0x1.73f9a40000000p+0f, 0x1.14b02e0000000p-25f, 0x1.74fbd40000000p+0f, -0x1.4506800000000p-25f,
    0x1.75feb60000000p+0f, -0x1.37b3060000000p-25f, 0x1.77024c0000000p+0f, -0x1.ca923e0000000p-25f,
    0x1.7806940000000p+0f, 0x1.fbcba80000000p-25f, 0x1.790b940000000p+0f, -0x1.d4f8c20000000p-26f,
    0x1.7a11480000000p+0f, -0x1.829fd00000000p-25f, 0x1.7b17b00000000p+0f, 0x1.2ed9fc0000000p-25f,
    0x1.7c1ed00000000p+0f, 0x1.30c1320000000p-28f, 0x1.7d26a60000000p+0f, 0x1.7fc3780000000p-27f,
    0x1.7e2f340000000p+0f, -0x1.2616340000000p-25f, 0x1.7f38780000000p+0f, 0x1.2471240000000p-26f,
    0x1.8042760000000p+0f, -0x1.783cbe0000000p-25f, 0x1.814d2a0000000p+0f, 0x1.ba20dc0000000p-25f,
    0x1.82589a0000000p+0f, -0x1.accc7c0000000p-26f, 0x1.8364c20000000p+0f, -0x1.46be080000000p-28f,
    0x1.8471a40000000p+0f, 0x1.88f1ec0000000p-26f, 0x1.857f420000000p+0f, -0x1.0c149c0000000p-25f,
    0x1.868d9a0000000p+0f, -0x1.2edb440000000p-26f, 0x1.879cae0000000p+0f, -0x1.b396f20000000p-26f,
    0x1.88ac7e0000000p+0f, -0x1.9d665a0000000p-26f, 0x1.89bd0a0000000p+0f, 0x1.1e16040000000p-26f,
    0x1.8ace540000000p+0f, 0x1.15506e0000000p-27f, 0x1.8be05c0000000p+0f, -0x1.4a7a220000000p-26f,
    0x1.8cf3220000000p+0f, -0x1.29576e0000000p-25f, 0x1.8e06a60000000p+0f, -0x1.f799280000000p-28f,
    0x1.8f1aea0000000p+0f, -0x1.baa2320000000p-26f, 0x1.902fee0000000p+0f, -0x1.fafa6e0000000p-25f,
    0x1.9145b00000000p+0f, 0x1.723ff80000000p-25f, 0x1.925c360000000p+0f, -0x1.8aba040000000p-25f,
    0x1.93737c0000000p+0f, -0x1.e647440000000p-25f, 0x1.948b820000000p+0f, 0x1.6bf31c0000000p-25f,
    0x1.95a44c0000000p+0f, 0x1.790a420000000p-25f, 0x1.96bdda0000000p+0f, -0x1.6263d40000000p-26f,
    0x1.97d82a0000000p+0f, -0x1.0d8d840000000p-31f, 0x1.98f33e0000000p+0f, 0x1.1e88a80000000p-26f,
    0x1.9a0f180000000p+0f, -0x1.e6bf080000000p-25f, 0x1.9b2bb40000000p+0f, 0x1.aa7fc20000000p-25f,
    0x1.9c49180000000p+0f, 0x1.51f8480000000p-27f, 0x1.9d67420000000p+0f, -0x1.ad11ca0000000p-26f,
    0x1.9e86320000000p+0f, -0x1.8737380000000p-26f, 0x1.9fa5e80000000p+0f, 0x1.a0fe540000000p-25f,
    0x1.a0c6680000000p+0f, -0x1.2886a60000000p-26f, 0x1.a1e7ae0000000p+0f, 0x1.b1d7180000000p-25f,
    0x1.a309be0000000p+0f, 0x1.8945a60000000p-25f, 0x1.a42c980000000p+0f, 0x1.182b5e0000000p-30f,
    0x1.a5503c0000000p+0f, -0x1.b83b540000000p-25f, 0x1.a674a80000000p+0f, 0x1.5e8c0a0000000p-25f,
    0x1.a799e20000000p+0f, -0x1.99e9940000000p-25f, 0x1.a8bfe60000000p+0f, -0x1.87da340000000p-25f,
    0x1.a9e6b60000000p+0f, -0x1.50c0480000000p-25f, 0x1.ab0e520000000p+0f, 0x1.356eba0000000p-28f,
    0x1.ac36bc0000000p+0f, -0x1.6064320000000p-31f, 0x1.ad5ff40000000p+0f, -0x1.70f6220000000p-26f,
    0x1.ae89fa0000000p+0f, -0x1.a94b140000000p-26f, 0x1.afb4ce0000000p+0f, 0x1.88bcc00000000p-26f,
    0x1.b0e0720000000p+0f, 0x1.31b6cc0000000p-25f, 0x1.b20ce60000000p+0f, 0x1.93512a0000000p-25f,
    0x1.b33a2c0000000p+0f, -0x1.ec3a820000000p-26f, 0x1.b468420000000p+0f, -0x1.4916ca0000000p-25f,
    0x1.b597280000000p+0f, 0x1.bcab280000000p-25f, 0x1.b6c6e20000000p+0f, 0x1.3e38a60000000p-25f,
    0x1.b7f7700000000p+0f, -0x1.a094380000000p-25f, 0x1.b928d00000000p+0f, -0x1.bb16c40000000p-25f,
    0x1.ba5b040000000p+0f, -0x1.ebdf360000000p-25f, 0x1.bb8e0c0000000p+0f, -0x1.0cb21c0000000p-25f,
    0x1.bcc1ea0000000p+0f, -0x1.f687c60000000p-25f, 0x1.bdf69c0000000p+0f, 0x1.f9d1040000000p-27f,
    0x1.bf2c260000000p+0f, -0x1.0a387e0000000p-26f, 0x1.c062860000000p+0f, 0x1.41b33c0000000p-28f,
    0x1.c199be0000000p+0f, -0x1.3d56b20000000p-27f, 0x1.c2d1ce0000000p+0f, -0x1.8166b60000000p-26f,
    0x1.c40ab60000000p+0f, -0x1.7c2c980000000p-39f, 0x1.c544780000000p+0f, -0x1.c141380000000p-26f,
    0x1.c67f120000000p+0f, 0x1.cafa2a0000000p-25f, 0x1.c7ba880000000p+0f, 0x1.3119260000000p-25f,
    0x1.c8f6da0000000p+0f, -0x1.7f230a0000000p-25f, 0x1.ca34060000000p+0f, -0x1.15c7640000000p-25f,
    0x1.cb720e0000000p+0f, -0x1.8837cc0000000p-27f, 0x1.ccb0f20000000p+0f, 0x1.cda2ce0000000p-25f,
    0x1.cdf0b60000000p+0f, -0x1.5447800000000p-25f, 0x1.cf31560000000p+0f, -0x1.2915240000000p-26f,
    0x1.d072d40000000p+0f, 0x1.40f1300000000p-25f, 0x1.d1b5320000000p+0f, 0x1.61192e0000000p-25f,
    0x1.d2f8700000000p+0f, 0x1.01b13e0000000p-25f, 0x1.d43c8e0000000p+0f, 0x1.59543a0000000p-25f,
    0x1.d5818e0000000p+0f, -0x1.822dbc0000000p-27f, 0x1.d6c76e0000000p+0f, 0x1.0c5cda0000000p-25f,
    0x1.d80e320000000p+0f, -0x1.26cf8e0000000p-25f, 0x1.d955d80000000p+0f, -0x1.c013f20000000p-25f,
    0x1.da9e600000000p+0f, 0x1.ed99420000000p-27f, 0x1.dbe7ce0000000p+0f, -0x1.38af9e0000000p-25f,
    0x1.dd32200000000p+0f, -0x1.9fc9740000000p-25f, 0x1.de7d560000000p+0f, 0x1.0701960000000p-26f,
    0x1.dfc9740000000p+0f, -0x1.908c940000000p-25f, 0x1.e116760000000p+0f, 0x1.632fa20000000p-25f,
    0x1.e264620000000p+0f, -0x1.614bda0000000p-25f, 0x1.e3b3340000000p+0f, -0x1.3a447c0000000p-26f,
    0x1.e502ee0000000p+0f, 0x1.e2cffe0000000p-26f, 0x1.e653920000000p+0f, 0x1.19db5e0000000p-26f,
    0x1.e7a5200000000p+0f, -0x1.0e2ce00000000p-26f, 0x1.e8f7980000000p+0f, -0x1.0649180000000p-25f,
    0x1.ea4afa0000000p+0f, 0x1.52486c0000000p-27f, 0x1.eb9f480000000p+0f, 0x1.9f329c0000000p-26f,
    0x1.ecf4820000000p+0f, 0x1.b1ccfe0000000p-25f, 0x1.ee4aaa0000000p+0f, 0x1.0c42880000000p-27f,
    0x1.efa1be0000000p+0f, 0x1.cc2b440000000p-25f, 0x1.f0f9c20000000p+0f, -0x1.a4df6c0000000p-27f,
    0x1.f252b40000000p+0f, -0x1.1288ae0000000p-25f, 0x1.f3ac940000000p+0f, 0x1.1bae4e0000000p-25f,
    0x1.f507660000000p+0f, -0x1.246eb00000000p-26f, 0x1.f663280000000p+0f, -0x1.9deec20000000p-26f,
    0x1.f7bfda0000000p+0f, 0x1.b397c20000000p-25f, 0x1.f91d800000000p+0f, 0x1.121e440000000p-27f,
    0x1.fa7c180000000p+0f, 0x1.9e90d80000000p-28f, 0x1.fbdba40000000p+0f, -0x1.2da55e0000000p-25f,
    0x1.fd3c220000000p+0f, 0x1.71ee3e0000000p-25f, 0x1.fe9d960000000p+0f, 0x1.65447c0000000p-25f,
};
typedef char xexp_tab_has_256_pairs[sizeof xexp_tab / sizeof xexp_tab[0] == 512 ? 1 : -1];

/* the lanes in bad, from their saved arguments, through the definition */
static void xexp_fallback(const float *xs, float *y, unsigned bad) {
    while (bad) {
        int l = __builtin_ctz(bad);
        y[l] = tr_expf(xs[l]);
        bad &= bad - 1;
    }
}

TR_TARGET_AVX512
static void avx512_expf_f32(const float *x, float *y, int64_t n) {
    const __m512 invl = _mm512_set1_ps(XEXP_INVL), shift = _mm512_set1_ps(XEXP_SHIFT);
    const __m512 la = _mm512_set1_ps(XEXP_LA), lb = _mm512_set1_ps(XEXP_LB), lc = _mm512_set1_ps(XEXP_LC);
    const __m512 nll = _mm512_set1_ps(XEXP_NLL), split = _mm512_set1_ps(XEXP_SPLIT);
    const __m512 c2 = _mm512_set1_ps(XEXP_C2), c3 = _mm512_set1_ps(XEXP_C3), d = _mm512_set1_ps(XEXP_D);
    const __m512 xnorm = _mm512_set1_ps(XEXP_X_NORM), xovf = _mm512_set1_ps(XEXP_X_OVF);
    const __m512 xuf = _mm512_set1_ps(XEXP_X_UF), g22 = _mm512_set1_ps(0x1p-22f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 kf = _mm512_sub_ps(_mm512_add_ps(_mm512_mul_ps(vx, invl), shift), shift);
        __m512i k = _mm512_cvttps_epi32(kf);
        __m512i j2 = _mm512_slli_epi32(_mm512_and_si512(k, _mm512_set1_epi32(255)), 1);
        __m512 rh = _mm512_sub_ps(_mm512_sub_ps(_mm512_sub_ps(vx, _mm512_mul_ps(kf, la)), _mm512_mul_ps(kf, lb)),
                                  _mm512_mul_ps(kf, lc));
        __m512 rl = _mm512_mul_ps(kf, nll);
        __m512 rs = _mm512_add_ps(rh, rl);
        __m512 s = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(rs, rs), _mm512_add_ps(_mm512_mul_ps(c3, rs), c2)), rl);
        __m512 th = _mm512_i32gather_ps(j2, xexp_tab, 4);
        __m512 tl = _mm512_i32gather_ps(_mm512_add_epi32(j2, _mm512_set1_epi32(1)), xexp_tab, 4);
        __m512 ph = _mm512_mul_ps(th, rh);
        __m512 ct = _mm512_mul_ps(th, split), cr = _mm512_mul_ps(rh, split);
        __m512 th1 = _mm512_sub_ps(ct, _mm512_sub_ps(ct, th)), th2 = _mm512_sub_ps(th, th1);
        __m512 rh1 = _mm512_sub_ps(cr, _mm512_sub_ps(cr, rh)), rh2 = _mm512_sub_ps(rh, rh1);
        __m512 pl = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(th1, rh1), ph), _mm512_mul_ps(th1, rh2));
        pl = _mm512_add_ps(_mm512_add_ps(pl, _mm512_mul_ps(th2, rh1)), _mm512_mul_ps(th2, rh2));
        __m512 yh = _mm512_add_ps(th, ph);
        __m512 e = _mm512_add_ps(_mm512_sub_ps(th, yh), ph);
        __m512 a = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(tl, rs), tl), pl), e);
        __m512 yl = _mm512_add_ps(_mm512_mul_ps(th, s), a);
        __m512i ee = _mm512_srai_epi32(k, 8);
        __m512 u1 = _mm512_add_ps(yh, _mm512_sub_ps(yl, d)), u2 = _mm512_add_ps(yh, _mm512_add_ps(yl, d));
        __m512i bits = _mm512_add_epi32(_mm512_castps_si512(u1), _mm512_slli_epi32(ee, 23));
        __mmask16 ok = _mm512_cmp_ps_mask(u1, u2, _CMP_EQ_OQ);
        __mmask16 sub = _mm512_cmp_ps_mask(vx, xnorm, _CMP_LT_OQ);
        if (sub) { /* exp(x) < 2^-126: m + y rounds y on the grid 2^-149 (bench_expf32.c finish32) */
            __m512i mb = _mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(1), ee), 23);
            __m512 m = _mm512_castsi512_ps(mb);
            __m512 g = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(-22), ee), 23));
            __m512 s2 = _mm512_add_ps(m, yh);
            __m512 v = _mm512_add_ps(_mm512_add_ps(_mm512_sub_ps(m, s2), yh), yl);
            __m512 d2 = _mm512_add_ps(_mm512_mul_ps(g, g22), d);
            __m512 w1 = _mm512_add_ps(s2, _mm512_sub_ps(v, d2)), w2 = _mm512_add_ps(s2, _mm512_add_ps(v, d2));
            bits = _mm512_mask_mov_epi32(bits, sub, _mm512_sub_epi32(_mm512_castps_si512(w1), mb));
            ok = (__mmask16)((ok & ~sub) | (_mm512_cmp_ps_mask(w1, w2, _CMP_EQ_OQ) & sub));
        }
        __mmask16 nan = _mm512_cmp_ps_mask(vx, vx, _CMP_UNORD_Q);
        __mmask16 ovf = _mm512_cmp_ps_mask(vx, xovf, _CMP_GT_OQ);
        __mmask16 uf = _mm512_cmp_ps_mask(vx, xuf, _CMP_LT_OQ);
        bits = _mm512_mask_mov_epi32(bits, ovf, _mm512_set1_epi32(0x7F800000));
        bits = _mm512_mask_mov_epi32(bits, uf, _mm512_setzero_si512());
        bits = _mm512_mask_mov_epi32(bits, nan, _mm512_castps_si512(_mm512_add_ps(vx, vx)));
        unsigned bad = (unsigned)(uint16_t)~(ok | nan | ovf | uf);
        if (bad == 0) {
            _mm512_storeu_si512((void *)(y + i), bits);
        } else { /* the arguments saved first: y may be x */
            float xs[16];
            _mm512_storeu_ps(xs, vx);
            _mm512_storeu_si512((void *)(y + i), bits);
            xexp_fallback(xs, y + i, bad);
        }
    }
    for (; i < n; i++) y[i] = tr_expf(x[i]);
}

TR_TARGET_AVX2
static void avx2_expf_f32(const float *x, float *y, int64_t n) {
    const __m256 invl = _mm256_set1_ps(XEXP_INVL), shift = _mm256_set1_ps(XEXP_SHIFT);
    const __m256 la = _mm256_set1_ps(XEXP_LA), lb = _mm256_set1_ps(XEXP_LB), lc = _mm256_set1_ps(XEXP_LC);
    const __m256 nll = _mm256_set1_ps(XEXP_NLL), split = _mm256_set1_ps(XEXP_SPLIT);
    const __m256 c2 = _mm256_set1_ps(XEXP_C2), c3 = _mm256_set1_ps(XEXP_C3), d = _mm256_set1_ps(XEXP_D);
    const __m256 xnorm = _mm256_set1_ps(XEXP_X_NORM), xovf = _mm256_set1_ps(XEXP_X_OVF);
    const __m256 xuf = _mm256_set1_ps(XEXP_X_UF), g22 = _mm256_set1_ps(0x1p-22f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 kf = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(vx, invl), shift), shift);
        __m256i k = _mm256_cvttps_epi32(kf);
        __m256i j2 = _mm256_slli_epi32(_mm256_and_si256(k, _mm256_set1_epi32(255)), 1);
        __m256 rh = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(vx, _mm256_mul_ps(kf, la)), _mm256_mul_ps(kf, lb)),
                                  _mm256_mul_ps(kf, lc));
        __m256 rl = _mm256_mul_ps(kf, nll);
        __m256 rs = _mm256_add_ps(rh, rl);
        __m256 s = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(rs, rs), _mm256_add_ps(_mm256_mul_ps(c3, rs), c2)), rl);
        __m256 th = _mm256_i32gather_ps(xexp_tab, j2, 4);
        __m256 tl = _mm256_i32gather_ps(xexp_tab + 1, j2, 4);
        __m256 ph = _mm256_mul_ps(th, rh);
        __m256 ct = _mm256_mul_ps(th, split), cr = _mm256_mul_ps(rh, split);
        __m256 th1 = _mm256_sub_ps(ct, _mm256_sub_ps(ct, th)), th2 = _mm256_sub_ps(th, th1);
        __m256 rh1 = _mm256_sub_ps(cr, _mm256_sub_ps(cr, rh)), rh2 = _mm256_sub_ps(rh, rh1);
        __m256 pl = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(th1, rh1), ph), _mm256_mul_ps(th1, rh2));
        pl = _mm256_add_ps(_mm256_add_ps(pl, _mm256_mul_ps(th2, rh1)), _mm256_mul_ps(th2, rh2));
        __m256 yh = _mm256_add_ps(th, ph);
        __m256 e = _mm256_add_ps(_mm256_sub_ps(th, yh), ph);
        __m256 a = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(tl, rs), tl), pl), e);
        __m256 yl = _mm256_add_ps(_mm256_mul_ps(th, s), a);
        __m256i ee = _mm256_srai_epi32(k, 8);
        __m256 u1 = _mm256_add_ps(yh, _mm256_sub_ps(yl, d)), u2 = _mm256_add_ps(yh, _mm256_add_ps(yl, d));
        __m256i bits = _mm256_add_epi32(_mm256_castps_si256(u1), _mm256_slli_epi32(ee, 23));
        __m256 ok = _mm256_cmp_ps(u1, u2, _CMP_EQ_OQ);
        __m256 sub = _mm256_cmp_ps(vx, xnorm, _CMP_LT_OQ);
        if (_mm256_movemask_ps(sub)) { /* exp(x) < 2^-126: m + y rounds y on the grid 2^-149 */
            __m256i mb = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(1), ee), 23);
            __m256 m = _mm256_castsi256_ps(mb);
            __m256 g = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(-22), ee), 23));
            __m256 s2 = _mm256_add_ps(m, yh);
            __m256 v = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(m, s2), yh), yl);
            __m256 d2 = _mm256_add_ps(_mm256_mul_ps(g, g22), d);
            __m256 w1 = _mm256_add_ps(s2, _mm256_sub_ps(v, d2)), w2 = _mm256_add_ps(s2, _mm256_add_ps(v, d2));
            __m256i sbits = _mm256_sub_epi32(_mm256_castps_si256(w1), mb);
            bits = _mm256_blendv_epi8(bits, sbits, _mm256_castps_si256(sub));
            ok = _mm256_blendv_ps(ok, _mm256_cmp_ps(w1, w2, _CMP_EQ_OQ), sub);
        }
        __m256 nan = _mm256_cmp_ps(vx, vx, _CMP_UNORD_Q);
        __m256 ovf = _mm256_cmp_ps(vx, xovf, _CMP_GT_OQ);
        __m256 uf = _mm256_cmp_ps(vx, xuf, _CMP_LT_OQ);
        bits = _mm256_blendv_epi8(bits, _mm256_set1_epi32(0x7F800000), _mm256_castps_si256(ovf));
        bits = _mm256_blendv_epi8(bits, _mm256_setzero_si256(), _mm256_castps_si256(uf));
        bits = _mm256_blendv_epi8(bits, _mm256_castps_si256(_mm256_add_ps(vx, vx)), _mm256_castps_si256(nan));
        unsigned spec = (unsigned)_mm256_movemask_ps(_mm256_or_ps(nan, _mm256_or_ps(ovf, uf)));
        unsigned bad = ~((unsigned)_mm256_movemask_ps(ok) | spec) & 0xFFu;
        if (bad == 0) {
            _mm256_storeu_si256((__m256i *)(void *)(y + i), bits);
        } else { /* the arguments saved first: y may be x */
            float xs[8];
            _mm256_storeu_ps(xs, vx);
            _mm256_storeu_si256((__m256i *)(void *)(y + i), bits);
            xexp_fallback(xs, y + i, bad);
        }
    }
    for (; i < n; i++) y[i] = tr_expf(x[i]);
}

/* ---- the prompt's matmul in phase-major order (kernels.h pm_*) ---------------- */
/* The lane contract's lane p is the chain of k = p, p + 16, ... in increasing k: a sequence in time.
 * One register holds that chain for 16 weight rows (one row a lane) against one input row, so a tile of
 * T input rows keeps T accumulators and every step is one panel load, T multiplies and T adds, with the
 * input broadcast from memory: no decode, no reduction and no scale in the loop (docs/MEASUREMENTS.md
 * §Phase-major; the idea is the thinker's of 2026-09-26). Each lane adds dot_row's products in dot_row's
 * order, from +0.0, multiply then add as the x8 kernel (w first, then the input; the accumulator first),
 * and the 16 phases are combined by tr_lane_combine's tree: the same bits by construction. */

/* 16 x 16 floats in v[0..15] (row r = v[r]) transposed in registers: v[c] becomes column c. Every loop
 * unrolled: a loop left rolled keeps v and t in memory (the panel ran 5x slower, 45 stack accesses). */
TR_TARGET_AVX512
static inline __attribute__((always_inline)) void pm_transpose16(__m512 v[16]) {
    __m512 t[16];
#pragma GCC unroll 16
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_ps(v[i], v[i + 1]);
        t[i + 1] = _mm512_unpackhi_ps(v[i], v[i + 1]);
    }
#pragma GCC unroll 16
    for (int i = 0; i < 16; i += 4) {
        v[i] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i]), _mm512_castps_pd(t[i + 2])));
        v[i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i]), _mm512_castps_pd(t[i + 2])));
        v[i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
        v[i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
    }
#pragma GCC unroll 16
    for (int i = 0; i < 16; i += 8) {
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
            t[i + j] = _mm512_shuffle_f32x4(v[i + j], v[i + j + 4], 0x88);
            t[i + j + 4] = _mm512_shuffle_f32x4(v[i + j], v[i + j + 4], 0xdd);
        }
    }
#pragma GCC unroll 8
    for (int j = 0; j < 8; j++) {
        v[j] = _mm512_shuffle_f32x4(t[j], t[j + 8], 0x88);
        v[j + 8] = _mm512_shuffle_f32x4(t[j], t[j + 8], 0xdd);
    }
}

/* 16 Q4_K rows into the panel: 16 columns at a time, each row's 16 weights w = scale * q - min (dot_row's,
 * the product exact) straight into a register, then transposed there. The block's scales and mins come
 * from avx512_q4_k_scales_store (tr_q4_k_scales's floats in a vector; scalar they were a fifth of the
 * panel, 0.69 of 3.29 us at n = 2048). */
TR_TARGET_AVX512
static void avx512_pm_panel_q4_k(const void *rows, size_t row_bytes, int64_t n, float *panel) {
    const unsigned char *base = (const unsigned char *)rows;
    const int64_t M = n / 16;
    float sm[TR_PM_ROWS][16] __attribute__((aligned(64)));
    for (int64_t m = 0; m < M; m++) {
        const int64_t b = m / 16;
        const int i0 = (int)((16 * m) % TR_Q4_K_BLOCK_ELEMS), c = i0 / 64, high = (i0 % 64) >= 32;
        const int o = i0 % 32, j = i0 / 32;
        if (i0 == 0)
            for (int r = 0; r < TR_PM_ROWS; r++)
                avx512_q4_k_scales_store(base + (size_t)r * row_bytes + (size_t)b * TR_Q4_K_BLOCK_BYTES, sm[r]);
        __m512 v[16];
#pragma GCC unroll 16
        for (int r = 0; r < TR_PM_ROWS; r++) {
            const unsigned char *qs =
                base + (size_t)r * row_bytes + (size_t)b * TR_Q4_K_BLOCK_BYTES + TR_Q4_K_QS_OFFSET + 32 * c + o;
            __m512i q = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)qs));
            q = high ? _mm512_srli_epi32(q, 4) : _mm512_and_si512(q, _mm512_set1_epi32(15));
            v[r] = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(sm[r][j]), _mm512_cvtepi32_ps(q)),
                                 _mm512_set1_ps(sm[r][8 + j]));
        }
        pm_transpose16(v);
#pragma GCC unroll 16
        for (int p = 0; p < 16; p++)
            _mm512_storeu_ps(panel + (size_t)p * (size_t)(n + TR_PM_PAD) + (size_t)m * TR_PM_ROWS, v[p]);
    }
}

/* 16 Q8_0 rows into the panel: each row's 16 weights w = d * q (dot_row's, the product exact) straight into a
 * register, the block's d read once for its two steps, then transposed in registers */
TR_TARGET_AVX512
static void avx512_pm_panel_q8_0(const void *rows, size_t row_bytes, int64_t n, float *panel) {
    const unsigned char *base = (const unsigned char *)rows;
    const int64_t M = n / 16;
    const size_t ps = (size_t)(n + TR_PM_PAD);
    float d[TR_PM_ROWS];
    for (int64_t m = 0; m < M; m++) {
        const int64_t b = m / 2;
        const int half = (int)(m % 2);
        if (half == 0)
            for (int r = 0; r < TR_PM_ROWS; r++)
                d[r] = tr_q8_0_block_scale(base + (size_t)r * row_bytes + (size_t)b * TR_Q8_0_BLOCK_BYTES);
        __m512 v[16];
#pragma GCC unroll 16
        for (int r = 0; r < TR_PM_ROWS; r++) {
            const unsigned char *qs =
                base + (size_t)r * row_bytes + (size_t)b * TR_Q8_0_BLOCK_BYTES + TR_Q8_0_SCALE_BYTES + 16 * half;
            __m512i q = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)qs));
            v[r] = _mm512_mul_ps(_mm512_set1_ps(d[r]), _mm512_cvtepi32_ps(q));
        }
        pm_transpose16(v);
#pragma GCC unroll 16
        for (int p = 0; p < 16; p++) _mm512_storeu_ps(panel + (size_t)p * ps + (size_t)m * TR_PM_ROWS, v[p]);
    }
}

/* The same panel from the quants' bytes: 16 rows x 16 bytes of a chunk (the low nibbles of 16 weights and the
 * high nibbles of the 16 that sit 32 later) transposed as bytes by 8 two-register byte permutes, then each
 * phase's 16 rows decoded at once, lane = row: w = scale16 * q - min16 with the rows' scales and mins of
 * the sub-block as vectors (the block's 16 x 16 scales transposed once). The same floats in the same
 * places; 8 permutes for 512 weights where the float transpose took 128 shuffles (0.38 of the panel). */
#define TR_TARGET_AVX512VBMI __attribute__((target("avx512f,avx512bw,avx512vbmi")))
static unsigned char g_pm_byte_idx[4][64] __attribute__((aligned(64))); /* global-ok: filled once with the table */

/* the 16 x 16 byte transpose's indices: A and B take phases 0-7 and 8-15 of rows 0-7 from (Z0, Z1) (and of
 * rows 8-15 from (Z2, Z3)) as [phase][8 rows]; O0 and O1 join two of those as [4 phases][16 rows] */
static void pm_build_byte_idx(void) {
    for (int ph = 0; ph < 8; ph++)
        for (int r = 0; r < 8; r++) {
            int src = r < 4 ? r * 16 + ph : 64 + (r - 4) * 16 + ph;
            g_pm_byte_idx[0][ph * 8 + r] = (unsigned char)src;
            g_pm_byte_idx[1][ph * 8 + r] = (unsigned char)(src + 8);
        }
    for (int ph = 0; ph < 4; ph++)
        for (int row = 0; row < 16; row++) {
            g_pm_byte_idx[2][ph * 16 + row] = (unsigned char)(row < 8 ? ph * 8 + row : 64 + ph * 8 + row - 8);
            g_pm_byte_idx[3][ph * 16 + row] = (unsigned char)(row < 8 ? (ph + 4) * 8 + row : 64 + (ph + 4) * 8 + row - 8);
        }
}

TR_TARGET_AVX512VBMI
static void avx512_pm_panel_q4_k_vbmi(const void *rows, size_t row_bytes, int64_t n, float *panel) {
    const unsigned char *base = (const unsigned char *)rows;
    const int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    const size_t ps = (size_t)(n + TR_PM_PAD);
    const __m512i iA = _mm512_load_si512(g_pm_byte_idx[0]), iB = _mm512_load_si512(g_pm_byte_idx[1]);
    const __m512i iO0 = _mm512_load_si512(g_pm_byte_idx[2]), iO1 = _mm512_load_si512(g_pm_byte_idx[3]);
    const __m512i m15 = _mm512_set1_epi32(15);
    float sm[TR_PM_ROWS][16] __attribute__((aligned(64)));
    unsigned char tbytes[256] __attribute__((aligned(64)));
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = base + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        for (int r = 0; r < TR_PM_ROWS; r++) avx512_q4_k_scales_store(blk + (size_t)r * row_bytes, sm[r]);
        __m512 sv[16];
#pragma GCC unroll 16
        for (int r = 0; r < 16; r++) sv[r] = _mm512_load_ps(sm[r]);
        pm_transpose16(sv); /* sv[j] = scale j of the 16 rows, sv[8 + j] = their min j */
#pragma GCC unroll 4
        for (int c = 0; c < 4; c++) {
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++) {
                const unsigned char *q0 = blk + TR_Q4_K_QS_OFFSET + 32 * c + 16 * h;
                __m512i z[4];
#pragma GCC unroll 4
                for (int i = 0; i < 4; i++) {
                    const unsigned char *qi = q0 + (size_t)(4 * i) * row_bytes;
                    __m512i t = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *)(const void *)qi));
                    t = _mm512_inserti32x4(t, _mm_loadu_si128((const __m128i *)(const void *)(qi + row_bytes)), 1);
                    t = _mm512_inserti32x4(t, _mm_loadu_si128((const __m128i *)(const void *)(qi + 2 * row_bytes)), 2);
                    t = _mm512_inserti32x4(t, _mm_loadu_si128((const __m128i *)(const void *)(qi + 3 * row_bytes)), 3);
                    z[i] = t;
                }
                const __m512i A = _mm512_permutex2var_epi8(z[0], iA, z[1]), B = _mm512_permutex2var_epi8(z[0], iB, z[1]);
                const __m512i C = _mm512_permutex2var_epi8(z[2], iA, z[3]), D = _mm512_permutex2var_epi8(z[2], iB, z[3]);
                /* bytes 16p..16p+15: phase p of rows 0..15, the low nibbles at step m_lo, the high at m_lo + 2;
                 * read back 16 bytes a phase (an extract takes a constant lane, which only -O2's unrolling gives) */
                _mm512_store_si512(tbytes, _mm512_permutex2var_epi8(A, iO0, C));
                _mm512_store_si512(tbytes + 64, _mm512_permutex2var_epi8(A, iO1, C));
                _mm512_store_si512(tbytes + 128, _mm512_permutex2var_epi8(B, iO0, D));
                _mm512_store_si512(tbytes + 192, _mm512_permutex2var_epi8(B, iO1, D));
                const int64_t m_lo = 16 * b + 4 * c + h;
                const __m512 slo = sv[2 * c], mlo = sv[8 + 2 * c], shi = sv[2 * c + 1], mhi = sv[8 + 2 * c + 1];
#pragma GCC unroll 4
                for (int j = 0; j < 4; j++) {
#pragma GCC unroll 4
                    for (int k = 0; k < 4; k++) {
                        const int p = 4 * j + k;
                        const __m512i q = _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(const void *)(tbytes + 16 * p)));
                        const __m512 wl = _mm512_sub_ps(_mm512_mul_ps(slo, _mm512_cvtepi32_ps(_mm512_and_si512(q, m15))), mlo);
                        const __m512 wh = _mm512_sub_ps(_mm512_mul_ps(shi, _mm512_cvtepi32_ps(_mm512_srli_epi32(q, 4))), mhi);
                        _mm512_storeu_ps(panel + (size_t)p * ps + (size_t)m_lo * TR_PM_ROWS, wl);
                        _mm512_storeu_ps(panel + (size_t)p * ps + (size_t)(m_lo + 2) * TR_PM_ROWS, wh);
                    }
                }
            }
        }
    }
}

/* 16 input rows at a time: the 16 x 16 block of each step transposed in registers and written as runs of
 * the tile's width (element by element, the stores 8-12 KB apart fought over the cache's sets: 0.9x to 7x
 * the tile's own time, tools' bench of 2026-09-26). A run shorter than 16 is written whole, in increasing
 * m and increasing t0: its tail lands on the next step's first floats, which that step writes after it,
 * and the last step's tail lands in the phase's pad (at most 15 floats of TR_PM_PAD). Masked stores took
 * 1.3x the time. */
TR_TARGET_AVX512
static void avx512_pm_interleave(const float *x, int64_t stride, int64_t n, int T, float *xil) {
    const int64_t M = n / 16;
    const size_t ps = (size_t)(M * T + TR_PM_PAD);
    for (int64_t m = 0; m < M; m++) {
        for (int t0 = 0; t0 < T; t0 += 16) {
            const int nt = T - t0 < 16 ? T - t0 : 16;
            const float *xt = x + (size_t)t0 * (size_t)stride + 16 * m;
            __m512 v[16];
#pragma GCC unroll 16
            for (int r = 0; r < 16; r++) v[r] = r < nt ? _mm512_loadu_ps(xt + (size_t)r * (size_t)stride) : _mm512_setzero_ps();
            pm_transpose16(v);
#pragma GCC unroll 16
            for (int p = 0; p < 16; p++) _mm512_storeu_ps(xil + (size_t)p * ps + (size_t)m * T + t0, v[p]);
        }
    }
}

/* the 16 phases of T accumulators, then the tree into y; T a compile-time constant in every caller */
TR_TARGET_AVX512
static inline __attribute__((always_inline)) void pm_tile_t(const float *panel, const float *xil, int64_t n,
                                                            float *part, float *y, int64_t y_stride, const int T) {
    const int64_t M = n / 16;
    for (int p = 0; p < 16; p++) {
        __m512 a[TR_PM_TILE_MAX];
#pragma GCC unroll 24
        for (int t = 0; t < T; t++) a[t] = _mm512_setzero_ps();
        const float *wp = panel + (size_t)p * (size_t)(n + TR_PM_PAD);
        const float *xp = xil + (size_t)p * (size_t)(M * T + TR_PM_PAD);
        for (int64_t m = 0; m < M; m++) {
            const __m512 w = _mm512_loadu_ps(wp + m * TR_PM_ROWS);
            const float *xm = xp + m * T;
#pragma GCC unroll 24
            for (int t = 0; t < T; t++) a[t] = _mm512_add_ps(a[t], _mm512_mul_ps(w, _mm512_set1_ps(xm[t])));
        }
#pragma GCC unroll 24
        for (int t = 0; t < T; t++) _mm512_storeu_ps(part + ((size_t)p * T + t) * TR_PM_ROWS, a[t]);
    }
    /* tr_lane_combine's tree, the 16 phases of each input row at once for the 16 rows */
    for (int t = 0; t < T; t++) {
        const float *q = part + (size_t)t * TR_PM_ROWS;
#define PM_L(p) _mm512_loadu_ps(q + (size_t)(p) * T * TR_PM_ROWS)
        __m512 s01 = _mm512_add_ps(PM_L(0), PM_L(1)), s23 = _mm512_add_ps(PM_L(2), PM_L(3));
        __m512 s45 = _mm512_add_ps(PM_L(4), PM_L(5)), s67 = _mm512_add_ps(PM_L(6), PM_L(7));
        __m512 lo = _mm512_add_ps(_mm512_add_ps(s01, s23), _mm512_add_ps(s45, s67));
        __m512 s89 = _mm512_add_ps(PM_L(8), PM_L(9)), s1011 = _mm512_add_ps(PM_L(10), PM_L(11));
        __m512 s1213 = _mm512_add_ps(PM_L(12), PM_L(13)), s1415 = _mm512_add_ps(PM_L(14), PM_L(15));
        __m512 hi = _mm512_add_ps(_mm512_add_ps(s89, s1011), _mm512_add_ps(s1213, s1415));
#undef PM_L
        _mm512_storeu_ps(y + (size_t)t * (size_t)y_stride, _mm512_add_ps(lo, hi));
    }
}

#define PM_TILE_FN(T)                                                                                          \
    TR_TARGET_AVX512 static void avx512_pm_tile_##T(const float *panel, const float *xil, int64_t n, float *part, \
                                                    float *y, int64_t y_stride) {                              \
        pm_tile_t(panel, xil, n, part, y, y_stride, T);                                                        \
    }
PM_TILE_FN(4) PM_TILE_FN(5) PM_TILE_FN(6) PM_TILE_FN(7) PM_TILE_FN(8) PM_TILE_FN(9) PM_TILE_FN(10)
PM_TILE_FN(11) PM_TILE_FN(12) PM_TILE_FN(13) PM_TILE_FN(14) PM_TILE_FN(15) PM_TILE_FN(16) PM_TILE_FN(17)
PM_TILE_FN(18) PM_TILE_FN(19) PM_TILE_FN(20) PM_TILE_FN(21) PM_TILE_FN(22) PM_TILE_FN(23) PM_TILE_FN(24)
#undef PM_TILE_FN

typedef void (*pm_tile_fn)(const float *, const float *, int64_t, float *, float *, int64_t);
static const pm_tile_fn g_pm_tiles[TR_PM_TILE_MAX + 1] = {
    NULL, NULL, NULL, NULL, avx512_pm_tile_4, avx512_pm_tile_5, avx512_pm_tile_6, avx512_pm_tile_7,
    avx512_pm_tile_8, avx512_pm_tile_9, avx512_pm_tile_10, avx512_pm_tile_11, avx512_pm_tile_12,
    avx512_pm_tile_13, avx512_pm_tile_14, avx512_pm_tile_15, avx512_pm_tile_16, avx512_pm_tile_17,
    avx512_pm_tile_18, avx512_pm_tile_19, avx512_pm_tile_20, avx512_pm_tile_21, avx512_pm_tile_22,
    avx512_pm_tile_23, avx512_pm_tile_24};

static void avx512_pm_tile(const float *panel, const float *xil, int64_t n, int T, float *part, float *y,
                           int64_t y_stride) {
    g_pm_tiles[T](panel, xil, n, part, y, y_stride);
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
            g_avx2.expf_f32 = avx2_expf_f32;
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
            g_avx512.expf_f32 = avx512_expf_f32;
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
            g_avx512.pm_panel[TR_TYPE_Q4_K] = avx512_pm_panel_q4_k;
            g_avx512.pm_panel[TR_TYPE_Q8_0] = avx512_pm_panel_q8_0;
            if (c->avx512bw && c->avx512vbmi) {
                pm_build_byte_idx();
                g_avx512.pm_panel[TR_TYPE_Q4_K] = avx512_pm_panel_q4_k_vbmi;
            }
            g_avx512.pm_interleave = avx512_pm_interleave;
            g_avx512.pm_tile = avx512_pm_tile;
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
