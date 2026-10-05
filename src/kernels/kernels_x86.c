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
_Static_assert(TR_DOT_TOKENS == TR_ATTN_X, "x4"); /* hot-ok: string -- a compile-time message, no code */

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

/* BF16 rows: 8 halves zero-extended and shifted into the top of 8 words are 8 exact floats (the F32 values they were
 * cut from, tr_f32_to_bf16_exact), then the products go into the lanes as in dot_f32: an F32 row's bits. */
TR_TARGET_AVX2
static inline __m256 avx2_bf16_to_ps(const unsigned char *p) {
    __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(const void *)p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
}

static inline float tail_bf16(const unsigned char *row, int64_t k) {
    uint16_t b;
    memcpy(&b, row + 2 * (size_t)k, sizeof b);
    return tr_bf16_to_float(b);
}

TR_TARGET_AVX2
static float avx2_dot_row_bf16(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        lo = _mm256_add_ps(lo, _mm256_mul_ps(avx2_bf16_to_ps(p + 2 * k), _mm256_loadu_ps(x + k)));
        hi = _mm256_add_ps(hi, _mm256_mul_ps(avx2_bf16_to_ps(p + 2 * k + 16), _mm256_loadu_ps(x + k + 8)));
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo);
    _mm256_storeu_ps(lane + 8, hi);
    for (; k < n; k++) lane[k % TR_LANES] += tail_bf16(p, k) * x[k];
    return tr_lane_combine(lane);
}

TR_TARGET_AVX2
static void avx2_dot_row_x4_bf16(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    const float *x0 = x, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
    __m256 lo0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps(), lo2 = _mm256_setzero_ps(),
           lo3 = _mm256_setzero_ps();
    __m256 hi0 = _mm256_setzero_ps(), hi1 = _mm256_setzero_ps(), hi2 = _mm256_setzero_ps(),
           hi3 = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        __m256 w0 = avx2_bf16_to_ps(p + 2 * k), w1 = avx2_bf16_to_ps(p + 2 * k + 16);
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
        float w = tail_bf16(p, k);
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

/* ---- Q4_K's integer definition in AVX2 (kernels.h q4x_*; kernels_internal.h for the layouts) -----------
 * By rows: the prepared input row and the decode's pairs, for the AVX2 tier and for an AVX-512 CPU without
 * the W16 panel's byte permutes, VNNI or DQ. The same bits as kernels.c: exact integers, then the f64 steps.
 * The decode's Q4_K rows stream from RAM: a prefetch TR_Q4_K_PREFETCH bytes ahead keeps enough lines in
 * flight for one thread (docs/MEASUREMENTS.md question 66). */
#define TR_Q4_K_PREFETCH 4608

/* an input row prepared: the shift per super-block, X = nearest-even(x 2^sh) by two exact multiplies (2^sh
 * or, past 2^127, 2^100 then 2^(sh - 100): the product stays normal or rounds to 0 anyway) and vcvtps2dq,
 * V1 = round(X / TR_Q4X_BASE) in f64, V0 by vpmulld, the sub-blocks' sums in f64; then each window's token
 * term by vpmaddwd, mod 2^32 */
TR_TARGET_AVX2
static void avx2_q4x_prep(const float *x, int64_t n, void *xq) {
    const tr_q4x_view v = tr_q4x_view_of(xq, n);
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)), fmax = _mm256_set1_ps(3.40282347e38f);
    const __m256d inv = _mm256_set1_pd(1.0 / (double)TR_Q4X_BASE);
    const __m256i base = _mm256_set1_epi32(TR_Q4X_BASE);
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        const float *xs = x + s * TR_Q4_K_BLOCK_ELEMS;
        __m256 m0 = _mm256_setzero_ps(), m1 = _mm256_setzero_ps();
        int bad = 0;
        for (int i = 0; i < TR_Q4_K_BLOCK_ELEMS; i += 16) {
            const __m256 a = _mm256_and_ps(_mm256_loadu_ps(xs + i), absm), b = _mm256_and_ps(_mm256_loadu_ps(xs + i + 8), absm);
            bad |= _mm256_movemask_ps(_mm256_cmp_ps(a, fmax, _CMP_NLE_UQ)) | _mm256_movemask_ps(_mm256_cmp_ps(b, fmax, _CMP_NLE_UQ));
            m0 = _mm256_max_ps(m0, a);
            m1 = _mm256_max_ps(m1, b);
        }
        if (bad) {
            v.scale[s] = (double)NAN;
            memset(v.v0 + s * TR_Q4_K_BLOCK_ELEMS, 0, TR_Q4_K_BLOCK_ELEMS * sizeof(int16_t));
            memset(v.v1 + s * TR_Q4_K_BLOCK_ELEMS, 0, TR_Q4_K_BLOCK_ELEMS * sizeof(int16_t));
            for (int j = 0; j < 8; j++) v.bf[8 * s + j] = 0.0;
            continue;
        }
        float mx[8];
        _mm256_storeu_ps(mx, _mm256_max_ps(m0, m1));
        float m = mx[0];
        for (int i = 1; i < 8; i++) m = mx[i] > m ? mx[i] : m;
        const int sh = tr_q4x_shift(m);
        v.scale[s] = tr_pow2(-sh);
        const __m256 fa = _mm256_set1_ps(tr_pow2f(sh <= 127 ? sh : 100)), fb = _mm256_set1_ps(tr_pow2f(sh <= 127 ? 0 : sh - 100));
        for (int j = 0; j < 8; j++) {
            __m256d bs = _mm256_setzero_pd();
            for (int h = 0; h < 4; h++) {
                const int64_t k = s * TR_Q4_K_BLOCK_ELEMS + 32 * j + 8 * h;
                const __m256i X = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + k), fa), fb));
                const __m256d xl = _mm256_cvtepi32_pd(_mm256_castsi256_si128(X)), xh = _mm256_cvtepi32_pd(_mm256_extracti128_si256(X, 1));
                bs = _mm256_add_pd(bs, _mm256_add_pd(xl, xh));
                const __m256i v1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_cvtpd_epi32(_mm256_mul_pd(xl, inv))),
                                                           _mm256_cvtpd_epi32(_mm256_mul_pd(xh, inv)), 1);
                const __m256i v0 = _mm256_sub_epi32(X, _mm256_mullo_epi32(v1, base));
                _mm_storeu_si128((__m128i *)(void *)(v.v0 + k), _mm_packs_epi32(_mm256_castsi256_si128(v0), _mm256_extracti128_si256(v0, 1)));
                _mm_storeu_si128((__m128i *)(void *)(v.v1 + k), _mm_packs_epi32(_mm256_castsi256_si128(v1), _mm256_extracti128_si256(v1, 1)));
            }
            double b4[4];
            _mm256_storeu_pd(b4, bs);
            v.bf[8 * s + j] = (b4[0] + b4[1]) + (b4[2] + b4[3]);
        }
    }
    const __m256i cw = _mm256_set1_epi16(TR_Q4X_C);
    const uint32_t c2 = (uint32_t)(32 * TR_Q4X_C * TR_Q4X_C);
    for (int64_t w = 0; w < n / 64; w++)
        for (int d = 0; d < 2; d++) {
            const int16_t *vd = (d ? v.v1 : v.v0) + 64 * w;
            __m256i e = _mm256_madd_epi16(_mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(const void *)vd), cw),
                                          _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(vd + 32)), cw));
            e = _mm256_add_epi32(e, _mm256_madd_epi16(_mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(vd + 16)), cw),
                                                      _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(vd + 48)), cw)));
            uint32_t lanes[8], t = 0;
            _mm256_storeu_si256((__m256i *)(void *)lanes, e);
            for (int i = 0; i < 8; i++) t += lanes[i];
            v.tok[2 * w + d] = (int32_t)(t - c2);
        }
}

/* The decode's pairs by rows: per block and row the quants widened to words and times their sub-block's
 * scale (vpmullw), vpmaddwd per 16 columns and digit (a lane's 32 products stay under 2^31 over a block),
 * the lanes summed in int64 at the block's end, the mins in int64, then kernels.c's f64 steps. */
TR_TARGET_AVX2
static void avx2_q4x_dot2(const void *row0, const void *row1, const void *xq, int64_t n, float *out) {
    const unsigned char *p[2] = {(const unsigned char *)row0, (const unsigned char *)row1};
    const tr_q4x_view v = tr_q4x_view_of(xq, n);
    const __m256i m15 = _mm256_set1_epi8(15);
    double y[2] = {0.0, 0.0};
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        const int16_t *v0 = v.v0 + s * TR_Q4_K_BLOCK_ELEMS, *v1 = v.v1 + s * TR_Q4_K_BLOCK_ELEMS;
        for (int r = 0; r < 2; r++) {
            const unsigned char *blk = p[r] + (size_t)s * TR_Q4_K_BLOCK_BYTES;
            _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH, _MM_HINT_T0);
            _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH + 64, _MM_HINT_T0);
            _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH + 128, _MM_HINT_T0);
            int sc[8], m[8];
            tr_q4_k_sc_m(blk, sc, m);
            __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
            for (int c = 0; c < 4; c++) {
                const __m256i q = _mm256_loadu_si256((const __m256i *)(const void *)(blk + TR_Q4_K_QS_OFFSET + 32 * c));
                const __m256i lo = _mm256_and_si256(q, m15), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m15);
                const __m256i sl = _mm256_set1_epi16((short)sc[2 * c]), sh = _mm256_set1_epi16((short)sc[2 * c + 1]);
                const __m256i w[4] = {_mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo)), sl),
                                      _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(lo, 1)), sl),
                                      _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi)), sh),
                                      _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(hi, 1)), sh)};
                for (int i = 0; i < 4; i++) {
                    a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w[i], _mm256_loadu_si256((const __m256i *)(const void *)(v0 + 64 * c + 16 * i))));
                    a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w[i], _mm256_loadu_si256((const __m256i *)(const void *)(v1 + 64 * c + 16 * i))));
                }
            }
            int32_t l0[8], l1[8];
            _mm256_storeu_si256((__m256i *)(void *)l0, a0);
            _mm256_storeu_si256((__m256i *)(void *)l1, a1);
            int64_t t0 = 0, t1 = 0, M = 0;
            for (int i = 0; i < 8; i++) {
                t0 += l0[i];
                t1 += l1[i];
                M += (int64_t)m[i] * (int64_t)v.bf[8 * s + i];
            }
            uint16_t hd, hm;
            memcpy(&hd, blk, 2);
            memcpy(&hm, blk + 2, 2);
            const double val = (double)tr_half_to_float(hd) * (double)(t0 + (int64_t)TR_Q4X_BASE * t1) -
                               (double)tr_half_to_float(hm) * (double)M;
            y[r] = y[r] + val * v.scale[s];
        }
    }
    out[0] = (float)y[0];
    out[1] = (float)y[1];
}

/* A short verify pass's pairs by AVX2: the pairs one input row after the other (the same bits; each weight
 * decoded once for all of them, as AVX-512's, is owed here) */
TR_TARGET_AVX2
static void avx2_q4x_dot_xt(const void *row0, const void *row1, const void *const *xq, int64_t n, int T, float *y,
                            int64_t y_stride) {
    for (int t = 0; t < T; t++) {
        float o[2];
        avx2_q4x_dot2(row0, row1, xq[t], n, o);
        y[(int64_t)t * y_stride] = o[0];
        y[(int64_t)t * y_stride + 1] = o[1];
    }
}

/* AVX2's W16 road: the panel of kernels_internal.h in scalar's bytes, and a tile that runs it as two halves of
 * 8 rows (a ymm holds a half of the panel's 16 lanes). The tile is AVX-512's arithmetic in 256 bits: vpmaddwd
 * then vpaddd for vpdpwssd (the same products, mod 2^32), the odd row's int64 by a blend for vpsraq, int64 to f64
 * by the magic constant for vcvtqq2pd (every value an integer under 2^51), the f64 steps by mul and add (each
 * one exact until d and dmin, so no fma is wanted for the same bits). */

/* 8 rows' dwords transposed in place: r[i] dword u -> r[u] dword i */
TR_TARGET_AVX2
static inline __attribute__((always_inline)) void q4x_tr8(__m256i r[8]) {
    const __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    const __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    const __m256i s0 = _mm256_unpacklo_epi64(t0, t2), s1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i s2 = _mm256_unpacklo_epi64(t1, t3), s3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i s4 = _mm256_unpacklo_epi64(t4, t6), s5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i s6 = _mm256_unpacklo_epi64(t5, t7), s7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256(s0, s4, 0x20);
    r[1] = _mm256_permute2x128_si256(s1, s5, 0x20);
    r[2] = _mm256_permute2x128_si256(s2, s6, 0x20);
    r[3] = _mm256_permute2x128_si256(s3, s7, 0x20);
    r[4] = _mm256_permute2x128_si256(s0, s4, 0x31);
    r[5] = _mm256_permute2x128_si256(s1, s5, 0x31);
    r[6] = _mm256_permute2x128_si256(s2, s6, 0x31);
    r[7] = _mm256_permute2x128_si256(s3, s7, 0x31);
}

/* the panel: per block the rows' headers as scalar's (d, dmin and the mins in the even/odd order), then per window
 * and half each row's 32 quant bytes as words, times their sub-block's scale plus TR_Q4X_C, the rows' word pairs
 * transposed 8 x 8 into the panel's vectors; the row terms by vpmaddwd over the stored vectors (under 2^31) */
TR_TARGET_AVX2
static void avx2_q4x_panel(const void *rows, size_t row_bytes, int64_t n, void *panel, const void *next) {
    const unsigned char *base = (const unsigned char *)rows;
    const int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    const __m256i m15 = _mm256_set1_epi8(15), cw = _mm256_set1_epi16(TR_Q4X_C);
    const __m256i bias = _mm256_setr_epi32(INT32_MIN, 0, INT32_MIN, 0, INT32_MIN, 0, INT32_MIN, 0);
    for (int64_t s = 0; s < nb; s++) {
        unsigned char *sb = (unsigned char *)panel + (size_t)s * TR_Q4X_SB_BYTES;
        const unsigned char *blk0 = base + (size_t)s * TR_Q4_K_BLOCK_BYTES;
        double *cst = (double *)(void *)sb;
        int sc[TR_PM_ROWS][8];
        for (int r = 0; r < TR_PM_ROWS; r++) {
            const unsigned char *blk = blk0 + (size_t)r * row_bytes;
            const int lane = (r & 1) ? 8 + r / 2 : r / 2;
            int m[8];
            tr_q4_k_sc_m(blk, sc[r], m);
            uint16_t hd, hm;
            memcpy(&hd, blk, 2);
            memcpy(&hm, blk + 2, 2);
            cst[lane] = (double)tr_half_to_float(hd);
            cst[16 + lane] = (double)tr_half_to_float(hm);
            for (int j = 0; j < 8; j++) cst[32 + 16 * j + lane] = (double)m[j];
        }
        for (int c = 0; c < 4; c++) {
            unsigned char *win = sb + TR_Q4X_CONST_BYTES + (size_t)c * TR_Q4X_WIN_BYTES;
            /* the next rows, one contiguous run of 16 row_bytes = 9 lines a window of 64 columns */
            if (next != NULL) {
                const char *np = (const char *)next + (size_t)(9 * (4 * s + c)) * 64;
                for (int j = 0; j < 9; j++) _mm_prefetch(np + 64 * j, _MM_HINT_T0);
            }
            /* these rows' block s + 2, four rows a window */
            if (s + 2 < nb)
                for (int r = 4 * c; r < 4 * c + 4; r++) {
                    const char *pa = (const char *)(base + (size_t)r * row_bytes + (size_t)(s + 2) * TR_Q4_K_BLOCK_BYTES);
                    _mm_prefetch(pa, _MM_HINT_T0);
                    _mm_prefetch(pa + 64, _MM_HINT_T0);
                    _mm_prefetch(pa + 128, _MM_HINT_T0);
                }
            for (int hh = 0; hh < 2; hh++) {
                __m256i l0[8], l1[8], h0[8], h1[8];
                for (int i = 0; i < 8; i++) {
                    const int r = 8 * hh + i;
                    const __m256i q = _mm256_loadu_si256((const __m256i *)(const void *)(blk0 + (size_t)r * row_bytes +
                                                                                           TR_Q4_K_QS_OFFSET + 32 * c));
                    const __m256i lo = _mm256_and_si256(q, m15), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m15);
                    const __m256i sl = _mm256_set1_epi16((short)sc[r][2 * c]), sh = _mm256_set1_epi16((short)sc[r][2 * c + 1]);
                    l0[i] = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo)), sl), cw);
                    l1[i] = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(lo, 1)), sl), cw);
                    h0[i] = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi)), sh), cw);
                    h1[i] = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(hi, 1)), sh), cw);
                }
                q4x_tr8(l0);
                q4x_tr8(l1);
                q4x_tr8(h0);
                q4x_tr8(h1);
                __m256i e = _mm256_setzero_si256();
                for (int u = 0; u < 8; u++) {
                    _mm256_storeu_si256((__m256i *)(void *)(win + 64 + 128 * u + 32 * hh), l0[u]);
                    _mm256_storeu_si256((__m256i *)(void *)(win + 64 + 128 * (u + 8) + 32 * hh), l1[u]);
                    _mm256_storeu_si256((__m256i *)(void *)(win + 128 + 128 * u + 32 * hh), h0[u]);
                    _mm256_storeu_si256((__m256i *)(void *)(win + 128 + 128 * (u + 8) + 32 * hh), h1[u]);
                    e = _mm256_add_epi32(e, _mm256_add_epi32(_mm256_madd_epi16(l0[u], h0[u]), _mm256_madd_epi16(l1[u], h1[u])));
                }
                _mm256_storeu_si256((__m256i *)(void *)(win + 32 * hh), _mm256_add_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), e), bias));
            }
        }
    }
}

/* the odd row's dword of each qword, sign-extended (vpsraq by 32) */
TR_TARGET_AVX2
static inline __m256i q4x_hi64(__m256i a) {
    return _mm256_blend_epi32(_mm256_srli_epi64(a, 32), _mm256_srai_epi32(a, 31), 0xAA);
}

/* int64 lanes to f64, exact for |x| < 2^51: x added to the bits of 1.5 2^52, the constant taken back */
TR_TARGET_AVX2
static inline __m256d q4x_i64_pd(__m256i x) {
    return _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(x, _mm256_set1_epi64x(0x4338000000000000ll))),
                         _mm256_set1_pd(6755399441055744.0));
}

static inline int32_t q4x_ld32(const int16_t *p) {
    int32_t x;
    memcpy(&x, p, sizeof x);
    return x;
}

/* the panel against T prepared rows (T a constant in every caller), a half of 8 rows after the other: the lanes of
 * a window start at minus the row and token terms (+ 2^31 on the even rows), take 16 Winograd products per digit
 * and token, and join the block's int64 sums by the even/odd split; at the block's end AVX-512's f64 steps on 4
 * even and 4 odd rows */
TR_TARGET_AVX2
static inline __attribute__((always_inline)) void q4x_tile2_t(const unsigned char *panel, const void *const *xq,
                                                              int64_t n, float *y, int64_t ys, const int T) {
    tr_q4x_view v[TR_Q4X_TILE_MAX];
#pragma GCC unroll 4
    for (int t = 0; t < T; t++) v[t] = tr_q4x_view_of(xq[t], n);
    const __m256i bias4 = _mm256_set1_epi64x(4ll << 31);
    const __m256d b16 = _mm256_set1_pd((double)TR_Q4X_BASE);
    for (int hh = 0; hh < 2; hh++) {
        __m256d ye[TR_Q4X_TILE_MAX], yo[TR_Q4X_TILE_MAX];
#pragma GCC unroll 4
        for (int t = 0; t < T; t++) ye[t] = yo[t] = _mm256_setzero_pd();
        for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
            const unsigned char *sb = panel + (size_t)s * TR_Q4X_SB_BYTES;
            const double *cst = (const double *)(const void *)sb + 4 * hh;
            __m256i ae[TR_Q4X_TILE_MAX][2], ao[TR_Q4X_TILE_MAX][2];
#pragma GCC unroll 4
            for (int t = 0; t < T; t++) ae[t][0] = ae[t][1] = ao[t][0] = ao[t][1] = _mm256_setzero_si256();
            for (int c = 0; c < 4; c++) {
                const unsigned char *win = sb + TR_Q4X_CONST_BYTES + (size_t)c * TR_Q4X_WIN_BYTES + 32 * hh;
                const int64_t w = 4 * s + c, k0 = 64 * w;
                const __m256i rn = _mm256_loadu_si256((const __m256i *)(const void *)win);
                __m256i a[TR_Q4X_TILE_MAX][2];
#pragma GCC unroll 4
                for (int t = 0; t < T; t++) {
                    a[t][0] = _mm256_sub_epi32(rn, _mm256_set1_epi32(v[t].tok[2 * w]));
                    a[t][1] = _mm256_sub_epi32(rn, _mm256_set1_epi32(v[t].tok[2 * w + 1]));
                }
#pragma GCC unroll 4
                for (int u = 0; u < 16; u++) {
                    const __m256i wl = _mm256_loadu_si256((const __m256i *)(const void *)(win + 64 + 128 * u));
                    const __m256i wh = _mm256_loadu_si256((const __m256i *)(const void *)(win + 128 + 128 * u));
#pragma GCC unroll 4
                    for (int t = 0; t < T; t++) {
                        const __m256i f1a = _mm256_add_epi16(wl, _mm256_set1_epi32(q4x_ld32(v[t].v0 + k0 + 32 + 2 * u)));
                        const __m256i f2a = _mm256_add_epi16(wh, _mm256_set1_epi32(q4x_ld32(v[t].v0 + k0 + 2 * u)));
                        a[t][0] = _mm256_add_epi32(a[t][0], _mm256_madd_epi16(f1a, f2a));
                        const __m256i f1b = _mm256_add_epi16(wl, _mm256_set1_epi32(q4x_ld32(v[t].v1 + k0 + 32 + 2 * u)));
                        const __m256i f2b = _mm256_add_epi16(wh, _mm256_set1_epi32(q4x_ld32(v[t].v1 + k0 + 2 * u)));
                        a[t][1] = _mm256_add_epi32(a[t][1], _mm256_madd_epi16(f1b, f2b));
                    }
                }
#pragma GCC unroll 4
                for (int t = 0; t < T; t++)
#pragma GCC unroll 2
                    for (int d = 0; d < 2; d++) {
                        ao[t][d] = _mm256_add_epi64(ao[t][d], q4x_hi64(a[t][d]));
                        ae[t][d] = _mm256_add_epi64(ae[t][d], a[t][d]);
                    }
            }
#pragma GCC unroll 4
            for (int t = 0; t < T; t++) {
                const __m256i ev0 = _mm256_sub_epi64(_mm256_sub_epi64(ae[t][0], _mm256_slli_epi64(ao[t][0], 32)), bias4);
                const __m256i ev1 = _mm256_sub_epi64(_mm256_sub_epi64(ae[t][1], _mm256_slli_epi64(ao[t][1], 32)), bias4);
                const __m256d te = _mm256_add_pd(_mm256_mul_pd(q4x_i64_pd(ev1), b16), q4x_i64_pd(ev0));
                const __m256d to = _mm256_add_pd(_mm256_mul_pd(q4x_i64_pd(ao[t][1]), b16), q4x_i64_pd(ao[t][0]));
                const double *bf = v[t].bf + 8 * s;
                __m256d Me = _mm256_mul_pd(_mm256_loadu_pd(cst + 32), _mm256_set1_pd(bf[0]));
                __m256d Mo = _mm256_mul_pd(_mm256_loadu_pd(cst + 40), _mm256_set1_pd(bf[0]));
#pragma GCC unroll 8
                for (int j = 1; j < 8; j++) {
                    const __m256d bj = _mm256_set1_pd(bf[j]);
                    Me = _mm256_add_pd(Me, _mm256_mul_pd(_mm256_loadu_pd(cst + 32 + 16 * j), bj));
                    Mo = _mm256_add_pd(Mo, _mm256_mul_pd(_mm256_loadu_pd(cst + 40 + 16 * j), bj));
                }
                const __m256d sc = _mm256_set1_pd(v[t].scale[s]);
                const __m256d ve = _mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(cst), te), _mm256_mul_pd(_mm256_loadu_pd(cst + 16), Me));
                const __m256d vo = _mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(cst + 8), to), _mm256_mul_pd(_mm256_loadu_pd(cst + 24), Mo));
                ye[t] = _mm256_add_pd(ye[t], _mm256_mul_pd(ve, sc));
                yo[t] = _mm256_add_pd(yo[t], _mm256_mul_pd(vo, sc));
            }
        }
#pragma GCC unroll 4
        for (int t = 0; t < T; t++) {
            const __m128 e = _mm256_cvtpd_ps(ye[t]), o = _mm256_cvtpd_ps(yo[t]);
            float *yt = y + (size_t)t * (size_t)ys + 8 * hh;
            _mm_storeu_ps(yt, _mm_unpacklo_ps(e, o));
            _mm_storeu_ps(yt + 4, _mm_unpackhi_ps(e, o));
        }
    }
}

#define Q4X_TILE2_FN(T)                                                                                        \
    TR_TARGET_AVX2 static void avx2_q4x_tile_##T(const unsigned char *panel, const void *const *xq, int64_t n,  \
                                                 float *y, int64_t ys) {                                      \
        q4x_tile2_t(panel, xq, n, y, ys, T);                                                                  \
    }
Q4X_TILE2_FN(1) Q4X_TILE2_FN(2) Q4X_TILE2_FN(3) Q4X_TILE2_FN(4)
#undef Q4X_TILE2_FN

typedef void (*q4x_tile2_fn)(const unsigned char *, const void *const *, int64_t, float *, int64_t);
static const q4x_tile2_fn g_q4x_tiles2[TR_Q4X_TILE_MAX + 1] = {NULL, avx2_q4x_tile_1, avx2_q4x_tile_2,
                                                               avx2_q4x_tile_3, avx2_q4x_tile_4};

static void avx2_q4x_tile(const void *panel, const void *const *xq, int64_t n, int T, float *y, int64_t y_stride) {
    g_q4x_tiles2[T]((const unsigned char *)panel, xq, n, y, y_stride);
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

/* BF16 rows: 16 halves zero-extended and shifted into the top of 16 words (exact), then as dot_f32. */
TR_TARGET_AVX512
static inline __m512 avx512_bf16_to_ps(const unsigned char *p) {
    __m512i w = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(const void *)p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(w, 16));
}

TR_TARGET_AVX512
static float avx512_dot_row_bf16(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    __m512 acc = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES)
        acc = _mm512_add_ps(acc, _mm512_mul_ps(avx512_bf16_to_ps(p + 2 * k), _mm512_loadu_ps(x + k)));
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc);
    for (; k < n; k++) lane[k % TR_LANES] += tail_bf16(p, k) * x[k];
    return tr_lane_combine(lane);
}

TR_TARGET_AVX512
static void avx512_dot_row_x4_bf16(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    const float *x0 = x, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps(),
           acc3 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + TR_LANES <= n; k += TR_LANES) {
        __m512 w = avx512_bf16_to_ps(p + 2 * k);
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
        float w = tail_bf16(p, k);
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

/* ---- one weight row against 2 or 3 input rows (kernels.h dot_row_xt): a short verify pass's group ----
 * x4's loop with two or three named accumulators: out[t] is the dot_row of its own row, bit for bit. */
TR_TARGET_AVX2
static void avx2_dot_row_x2_q8_0(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m256 lo0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps();
    __m256 hi0 = _mm256_setzero_ps(), hi1 = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m256 d = _mm256_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m256 w0 = _mm256_mul_ps(d, avx2_i8_to_ps(qs + j));
            __m256 w1 = _mm256_mul_ps(d, avx2_i8_to_ps(qs + j + 8));
            const float *x0 = xb + j, *x1 = x0 + stride;
            lo0 = _mm256_add_ps(lo0, _mm256_mul_ps(w0, _mm256_loadu_ps(x0)));
            hi0 = _mm256_add_ps(hi0, _mm256_mul_ps(w1, _mm256_loadu_ps(x0 + 8)));
            lo1 = _mm256_add_ps(lo1, _mm256_mul_ps(w0, _mm256_loadu_ps(x1)));
            hi1 = _mm256_add_ps(hi1, _mm256_mul_ps(w1, _mm256_loadu_ps(x1 + 8)));
        }
    }
    float lane[TR_LANES];
    _mm256_storeu_ps(lane, lo0);
    _mm256_storeu_ps(lane + 8, hi0);
    out[0] = tr_lane_combine(lane);
    _mm256_storeu_ps(lane, lo1);
    _mm256_storeu_ps(lane + 8, hi1);
    out[1] = tr_lane_combine(lane);
}

TR_TARGET_AVX2
static void avx2_dot_row_x3_q8_0(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m256 lo0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps(), lo2 = _mm256_setzero_ps();
    __m256 hi0 = _mm256_setzero_ps(), hi1 = _mm256_setzero_ps(), hi2 = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m256 d = _mm256_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m256 w0 = _mm256_mul_ps(d, avx2_i8_to_ps(qs + j));
            __m256 w1 = _mm256_mul_ps(d, avx2_i8_to_ps(qs + j + 8));
            const float *x0 = xb + j, *x1 = x0 + stride, *x2 = x1 + stride;
            lo0 = _mm256_add_ps(lo0, _mm256_mul_ps(w0, _mm256_loadu_ps(x0)));
            hi0 = _mm256_add_ps(hi0, _mm256_mul_ps(w1, _mm256_loadu_ps(x0 + 8)));
            lo1 = _mm256_add_ps(lo1, _mm256_mul_ps(w0, _mm256_loadu_ps(x1)));
            hi1 = _mm256_add_ps(hi1, _mm256_mul_ps(w1, _mm256_loadu_ps(x1 + 8)));
            lo2 = _mm256_add_ps(lo2, _mm256_mul_ps(w0, _mm256_loadu_ps(x2)));
            hi2 = _mm256_add_ps(hi2, _mm256_mul_ps(w1, _mm256_loadu_ps(x2 + 8)));
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
}

TR_TARGET_AVX2
static void avx2_dot_row_xt_q8_0(const void *row, const float *x, int64_t stride, int64_t n, int t, float *out) {
    if (t == 3) avx2_dot_row_x3_q8_0(row, x, stride, n, out);
    else avx2_dot_row_x2_q8_0(row, x, stride, n, out);
}

TR_TARGET_AVX512
static void avx512_dot_row_x2_q8_0(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m512 d = _mm512_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m512 q = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + j))));
            __m512 w = _mm512_mul_ps(d, q);
            const float *x0 = xb + j, *x1 = x0 + stride;
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w, _mm512_loadu_ps(x0)));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w, _mm512_loadu_ps(x1)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc0);
    out[0] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc1);
    out[1] = tr_lane_combine(lane);
}

TR_TARGET_AVX512
static void avx512_dot_row_x3_q8_0(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m512 d = _mm512_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m512 q = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + j))));
            __m512 w = _mm512_mul_ps(d, q);
            const float *x0 = xb + j, *x1 = x0 + stride, *x2 = x1 + stride;
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(w, _mm512_loadu_ps(x0)));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(w, _mm512_loadu_ps(x1)));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(w, _mm512_loadu_ps(x2)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc0);
    out[0] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc1);
    out[1] = tr_lane_combine(lane);
    _mm512_storeu_ps(lane, acc2);
    out[2] = tr_lane_combine(lane);
}

TR_TARGET_AVX512
static void avx512_dot_row_xt_q8_0(const void *row, const float *x, int64_t stride, int64_t n, int t, float *out) {
    if (t == 3) avx512_dot_row_x3_q8_0(row, x, stride, n, out);
    else avx512_dot_row_x2_q8_0(row, x, stride, n, out);
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

/* A 16 x 16 byte transpose by 8 two-register byte permutes (VBMI), 16 rows of 16 bytes in 4 vectors (4 rows
 * a vector, a row a 128-bit lane) into [byte][16 rows]: the q4x panel's scales and mins (below). */
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

/* ---- Q4_K's integer definition (kernels.h q4x_*; kernels_internal.h for the layouts) ----------------------
 * W16 (docs/MEASUREMENTS.md §Exact sums at the float's speed): the digits and the weights in 16 bits, one
 * vpdpwssd lane takes two Winograd pairs of a row (the low and the high nibble of a quant byte, with the
 * other's digit riding in each factor); a 64-column window's int32 lanes start at minus its row and token
 * terms, wrap, and end exact (the window's sum is under 2^31); the windows go to int64 by the biased even/odd
 * split, the super-block's end to f64 with every value an integer under 2^53 until d and dmin. They need
 * VBMI (the panel's byte transposes), VNNI and DQ: Ice Lake, Zen 4 and later. */
#define TR_TARGET_Q4X __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx512vbmi,f16c")))

/* the panel's quants transposed: 16 rows x 32 bytes (a window) into [byte pair][16 rows] by two-register byte
 * permutes in three stages: [row][32 B] x 4 rows -> [pairs 0-7 | 8-15][4 rows]; two of those -> [4 pairs][8
 * rows]; two of those -> [2 pairs][16 rows] */
static unsigned char g_q4x_idx[3][2][64] __attribute__((aligned(64))); /* global-ok: filled once with the table */

static void q4x_build_idx(void) {
    for (int o = 0; o < 2; o++)
        for (int k = 0; k < 2; k++) {
            for (int w = 0; w < 8; w++)
                for (int r = 0; r < 4; r++)
                    g_q4x_idx[0][o][(w * 4 + r) * 2 + k] = (unsigned char)(r * 32 + 2 * (8 * o + w) + k);
            for (int w = 0; w < 4; w++)
                for (int r = 0; r < 8; r++)
                    g_q4x_idx[1][o][(w * 8 + r) * 2 + k] = (unsigned char)((r < 4 ? 0 : 64) + ((4 * o + w) * 4 + (r & 3)) * 2 + k);
            for (int w = 0; w < 2; w++)
                for (int r = 0; r < 16; r++)
                    g_q4x_idx[2][o][(w * 16 + r) * 2 + k] = (unsigned char)((r < 8 ? 0 : 64) + ((2 * o + w) * 8 + (r & 7)) * 2 + k);
        }
}

/* the 16 lanes' sum mod 2^32 (vector adds wrap; _mm512_reduce_add_epi32 adds signed ints, and a wrap there is
 * undefined behaviour, which UBSan stops on) */
TR_TARGET_Q4X
static inline uint32_t q4x_hsum_u32(__m512i v) {
    const __m256i a = _mm256_add_epi32(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
    __m128i b = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0x4E));
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0xB1));
    return (uint32_t)_mm_cvtsi128_si32(b);
}

/* an input row prepared: per super-block the largest |x| (a NaN or an infinity found by an unordered
 * compare), the shift, X by vscalefps and vcvtps2dq (exact: a power of two, then nearest-even), V1 =
 * round(X / TR_Q4X_BASE) in f64 (never near a tie: the base is odd and X under 2^31), V0 by vpmulld, the
 * sub-blocks' sums in f64 (integers under 2^53); then each window's token term, sum (V_a + C)(V_a+32 + C)
 * - 32 C^2 = kernels.c's, mod 2^32 */
TR_TARGET_Q4X
static void avx512_q4x_prep(const float *x, int64_t n, void *xq) {
    const tr_q4x_view v = tr_q4x_view_of(xq, n);
    const __m512 absm = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)), fmax = _mm512_set1_ps(3.40282347e38f);
    const __m512d inv = _mm512_set1_pd(1.0 / (double)TR_Q4X_BASE);
    const __m512i base = _mm512_set1_epi32(TR_Q4X_BASE);
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        const float *xs = x + s * TR_Q4_K_BLOCK_ELEMS;
        __m512 m0 = _mm512_setzero_ps(), m1 = _mm512_setzero_ps();
        __mmask16 bad = 0;
        for (int i = 0; i < TR_Q4_K_BLOCK_ELEMS; i += 32) {
            const __m512 a = _mm512_and_ps(_mm512_loadu_ps(xs + i), absm), b = _mm512_and_ps(_mm512_loadu_ps(xs + i + 16), absm);
            bad |= _mm512_cmp_ps_mask(a, fmax, _CMP_NLE_UQ) | _mm512_cmp_ps_mask(b, fmax, _CMP_NLE_UQ);
            m0 = _mm512_max_ps(m0, a);
            m1 = _mm512_max_ps(m1, b);
        }
        if (bad) {
            v.scale[s] = (double)NAN;
            memset(v.v0 + s * TR_Q4_K_BLOCK_ELEMS, 0, TR_Q4_K_BLOCK_ELEMS * sizeof(int16_t));
            memset(v.v1 + s * TR_Q4_K_BLOCK_ELEMS, 0, TR_Q4_K_BLOCK_ELEMS * sizeof(int16_t));
            for (int j = 0; j < 8; j++) v.bf[8 * s + j] = 0.0;
            continue;
        }
        const int sh = tr_q4x_shift(_mm512_reduce_max_ps(_mm512_max_ps(m0, m1)));
        v.scale[s] = tr_pow2(-sh);
        const __m512 shv = _mm512_set1_ps((float)sh);
        for (int j = 0; j < 8; j++) {
            __m512d bs = _mm512_setzero_pd();
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++) {
                const int64_t k = s * TR_Q4_K_BLOCK_ELEMS + 32 * j + 16 * h;
                const __m512i X = _mm512_cvtps_epi32(_mm512_scalef_ps(_mm512_loadu_ps(x + k), shv));
                const __m512d xl = _mm512_cvtepi32_pd(_mm512_castsi512_si256(X));
                const __m512d xh = _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(X, 1));
                bs = _mm512_add_pd(bs, _mm512_add_pd(xl, xh));
                const __m512i v1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtpd_epi32(_mm512_mul_pd(xl, inv))),
                                                      _mm512_cvtpd_epi32(_mm512_mul_pd(xh, inv)), 1);
                const __m512i v0 = _mm512_sub_epi32(X, _mm512_mullo_epi32(v1, base));
                _mm256_storeu_si256((__m256i *)(void *)(v.v0 + k), _mm512_cvtepi32_epi16(v0));
                _mm256_storeu_si256((__m256i *)(void *)(v.v1 + k), _mm512_cvtepi32_epi16(v1));
            }
            v.bf[8 * s + j] = _mm512_reduce_add_pd(bs);
        }
    }
    const __m512i cw = _mm512_set1_epi16(TR_Q4X_C);
    const int32_t c2 = 32 * TR_Q4X_C * TR_Q4X_C;
    for (int64_t w = 0; w < n / 64; w++)
#pragma GCC unroll 2
        for (int d = 0; d < 2; d++) {
            const int16_t *vd = (d ? v.v1 : v.v0) + 64 * w;
            const __m512i a = _mm512_add_epi16(_mm512_loadu_si512(vd), cw), b = _mm512_add_epi16(_mm512_loadu_si512(vd + 32), cw);
            v.tok[2 * w + d] = (int32_t)(q4x_hsum_u32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), a, b)) - (uint32_t)c2);
        }
}

/* 128-bit lane i of v (the extract takes an immediate: a switch on a loop's index folds away once unrolled) */
TR_TARGET_Q4X
static inline __m128i q4x_lane(__m512i v, int i) {
    switch (i & 3) {
    case 0: return _mm512_castsi512_si128(v);
    case 1: return _mm512_extracti32x4_epi32(v, 1);
    case 2: return _mm512_extracti32x4_epi32(v, 2);
    default: return _mm512_extracti32x4_epi32(v, 3);
    }
}

/* one Q4_K block's header, 4 rows a zmm (a row's 16 bytes a 128-bit lane: d, dmin, the 12 scale bytes) into
 * sc0..7, m0..7 per lane (the byte shuffles of tr_q4_k_sc_m, lane by lane) */
TR_TARGET_Q4X
static inline __m512i q4x_sc_m4(__m512i hdr) {
    const __m512i sA = _mm512_broadcast_i32x4(_mm_setr_epi8(4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15));
    const __m512i sB = _mm512_broadcast_i32x4(_mm_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, 8, 9, 10, 11));
    const __m512i mlow = _mm512_broadcast_i32x4(_mm_setr_epi8(63, 63, 63, 63, 15, 15, 15, 15, 63, 63, 63, 63, 15, 15, 15, 15));
    const __m512i a = _mm512_shuffle_epi8(hdr, sA), hb = _mm512_shuffle_epi8(hdr, sB);
    const __m512i sel = _mm512_mask_blend_epi16(0xC0C0C0C0u, a, _mm512_and_si512(_mm512_srli_epi16(a, 4), _mm512_set1_epi8(0x0F)));
    return _mm512_or_si512(_mm512_and_si512(sel, mlow), _mm512_srli_epi16(_mm512_and_si512(hb, _mm512_set1_epi8((char)0xC0)), 2));
}

/* The decode's road: two rows against one prepared row. Per block and row the quants widened to words once
 * and split into nibbles, times their sub-block's scale (vpmullw, the scale a word broadcast from the span's
 * table), one vpdpwssd per 32 columns and digit (a lane's 16 products stay under 2^31 over a block); at the
 * block's end each row's lanes to f64 (exact), its two digits joined (lanes under 2^47) beside its mins'
 * products, and the four sums of the two rows reduced together (every value an integer under 2^53: any order
 * is exact), then kernels.c's f64 steps. The headers of TR_Q4_K_SPAN blocks of both rows are read first, then
 * those blocks: the early reads of each block's first line keep two interleaved rows streaming from RAM
 * (LESSONS #178: the float kernel ran 0.72x from RAM without them). */
#define TR_Q4_K_SPAN 8

TR_TARGET_Q4X
static void avx512_q4x_dot2(const void *row0, const void *row1, const void *xq, int64_t n, float *out) {
    const unsigned char *p[2] = {(const unsigned char *)row0, (const unsigned char *)row1};
    const tr_q4x_view v = tr_q4x_view_of(xq, n);
    const __m512i m15 = _mm512_set1_epi16(15);
    const __m512d b16 = _mm512_set1_pd((double)TR_Q4X_BASE);
    const int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    double y[2] = {0.0, 0.0};
    int16_t hsc[TR_Q4_K_SPAN][2][16]; /* each block's sc0..7 and m0..7 as words */
    double hmf[TR_Q4_K_SPAN][2][8];   /* its m0..7 */
    double hdm[TR_Q4_K_SPAN][4];      /* d, dmin of row 0, then of row 1 */
    const __m256i sA = _mm256_broadcastsi128_si256(_mm_setr_epi8(4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15));
    const __m256i sB = _mm256_broadcastsi128_si256(_mm_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, 8, 9, 10, 11));
    const __m256i mlow = _mm256_broadcastsi128_si256(_mm_setr_epi8(63, 63, 63, 63, 15, 15, 15, 15, 63, 63, 63, 63, 15, 15, 15, 15));
    for (int64_t sp = 0; sp < nb; sp += TR_Q4_K_SPAN) {
        const int64_t se = sp + TR_Q4_K_SPAN < nb ? sp + TR_Q4_K_SPAN : nb;
        /* both rows' headers a ymm, a row a 128-bit lane: tr_q4_k_sc_m's byte shuffles, then d and dmin of
         * both rows by one vcvtph2ps */
        for (int64_t b = sp; b < se; b++) {
            const unsigned char *h0 = p[0] + (size_t)b * TR_Q4_K_BLOCK_BYTES, *h1 = p[1] + (size_t)b * TR_Q4_K_BLOCK_BYTES;
            const __m256i hdr = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)h0)),
                                                        _mm_loadu_si128((const __m128i *)(const void *)h1), 1);
            const __m256i a = _mm256_shuffle_epi8(hdr, sA), hb = _mm256_shuffle_epi8(hdr, sB);
            const __m256i sel = _mm256_blend_epi16(a, _mm256_and_si256(_mm256_srli_epi16(a, 4), _mm256_set1_epi8(0x0F)), 0xC0);
            const __m256i sm = _mm256_or_si256(_mm256_and_si256(sel, mlow),
                                               _mm256_srli_epi16(_mm256_and_si256(hb, _mm256_set1_epi8((char)0xC0)), 2));
            _mm256_storeu_si256((__m256i *)(void *)hsc[b - sp][0], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(sm)));
            _mm256_storeu_si256((__m256i *)(void *)hsc[b - sp][1], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(sm, 1)));
            _mm512_storeu_pd(hmf[b - sp][0], _mm512_cvtepi64_pd(_mm512_cvtepu8_epi64(_mm_srli_si128(_mm256_castsi256_si128(sm), 8))));
            _mm512_storeu_pd(hmf[b - sp][1], _mm512_cvtepi64_pd(_mm512_cvtepu8_epi64(_mm_srli_si128(_mm256_extracti128_si256(sm, 1), 8))));
            const __m128i dd = _mm_unpacklo_epi32(_mm256_castsi256_si128(hdr), _mm256_extracti128_si256(hdr, 1));
            _mm256_storeu_pd(hdm[b - sp], _mm256_cvtps_pd(_mm_cvtph_ps(dd)));
        }
        for (int64_t s = sp; s < se; s++) {
            const int16_t *v0 = v.v0 + s * TR_Q4_K_BLOCK_ELEMS, *v1 = v.v1 + s * TR_Q4_K_BLOCK_ELEMS;
            const int64_t h = s - sp;
            __m512i acc[2][2];
#pragma GCC unroll 2
            for (int r = 0; r < 2; r++) {
                const unsigned char *blk = p[r] + (size_t)s * TR_Q4_K_BLOCK_BYTES;
                _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH, _MM_HINT_T0);
                _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH + 64, _MM_HINT_T0);
                _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH + 128, _MM_HINT_T0);
                acc[r][0] = acc[r][1] = _mm512_setzero_si512();
            }
#pragma GCC unroll 4
            for (int c = 0; c < 4; c++) {
                const __m512i a0 = _mm512_loadu_si512(v0 + 64 * c), a1 = _mm512_loadu_si512(v1 + 64 * c);
                const __m512i b0 = _mm512_loadu_si512(v0 + 64 * c + 32), b1 = _mm512_loadu_si512(v1 + 64 * c + 32);
#pragma GCC unroll 2
                for (int r = 0; r < 2; r++) {
                    const __m512i q = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(p[r] + (size_t)s * TR_Q4_K_BLOCK_BYTES +
                                                                                                          TR_Q4_K_QS_OFFSET + 32 * c)));
                    const __m512i wl = _mm512_mullo_epi16(_mm512_and_si512(q, m15), _mm512_set1_epi16(hsc[h][r][2 * c]));
                    const __m512i wh = _mm512_mullo_epi16(_mm512_srli_epi16(q, 4), _mm512_set1_epi16(hsc[h][r][2 * c + 1]));
                    acc[r][0] = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(acc[r][0], wl, a0), wh, b0);
                    acc[r][1] = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(acc[r][1], wl, a1), wh, b1);
                }
            }
            const __m512d bf = _mm512_loadu_pd(v.bf + 8 * s);
            __m512d t[2], mm[2];
#pragma GCC unroll 2
            for (int r = 0; r < 2; r++) {
                const __m512d s0 = _mm512_add_pd(_mm512_cvtepi32_pd(_mm512_castsi512_si256(acc[r][0])),
                                                 _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(acc[r][0], 1)));
                const __m512d s1 = _mm512_add_pd(_mm512_cvtepi32_pd(_mm512_castsi512_si256(acc[r][1])),
                                                 _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(acc[r][1], 1)));
                t[r] = _mm512_fmadd_pd(s1, b16, s0);
                mm[r] = _mm512_mul_pd(_mm512_loadu_pd(hmf[h][r]), bf);
            }
            /* [T0 | M0] and [T1 | M1] by 256-bit halves, then [T0 M0 T1 M1] by 128-bit quarters, then the pairs */
            const __m512d A = _mm512_add_pd(_mm512_shuffle_f64x2(t[0], mm[0], 0x44), _mm512_shuffle_f64x2(t[0], mm[0], 0xEE));
            const __m512d B = _mm512_add_pd(_mm512_shuffle_f64x2(t[1], mm[1], 0x44), _mm512_shuffle_f64x2(t[1], mm[1], 0xEE));
            const __m512d C = _mm512_add_pd(_mm512_shuffle_f64x2(A, B, 0x88), _mm512_shuffle_f64x2(A, B, 0xDD));
            double sums[8];
            _mm512_storeu_pd(sums, _mm512_add_pd(C, _mm512_permute_pd(C, 0x55)));
#pragma GCC unroll 2
            for (int r = 0; r < 2; r++) {
                const double val = hdm[h][2 * r] * sums[4 * r] - hdm[h][2 * r + 1] * sums[4 * r + 2];
                y[r] = y[r] + val * v.scale[s];
            }
        }
    }
    out[0] = (float)y[0];
    out[1] = (float)y[1];
}

/* The pairs' header pre-pass and block end, for the short passes' kernel below: the same statements as
 * avx512_q4x_dot2's, which keeps them inline (the decode's kernel stays the code measured, LESSONS #239).
 * The headers of blocks sp..se-1 of rows p[0] and p[1] into a span's tables, both rows a ymm, a row a 128-bit
 * lane: tr_q4_k_sc_m's byte shuffles, then d and dmin of both rows by one vcvtph2ps. hsc: each block's sc0..7
 * and m0..7 as words; hmf: its m0..7; hdm: d, dmin of row 0, then of row 1 */
TR_TARGET_Q4X
static inline __attribute__((always_inline)) void q4x_headers2(const unsigned char *const p[2], int64_t sp, int64_t se,
                                                               int16_t hsc[][2][16], double hmf[][2][8], double hdm[][4]) {
    const __m256i sA = _mm256_broadcastsi128_si256(_mm_setr_epi8(4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15));
    const __m256i sB = _mm256_broadcastsi128_si256(_mm_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, 8, 9, 10, 11));
    const __m256i mlow = _mm256_broadcastsi128_si256(_mm_setr_epi8(63, 63, 63, 63, 15, 15, 15, 15, 63, 63, 63, 63, 15, 15, 15, 15));
    for (int64_t b = sp; b < se; b++) {
        const unsigned char *k0 = p[0] + (size_t)b * TR_Q4_K_BLOCK_BYTES, *k1 = p[1] + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const __m256i hdr = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)k0)),
                                                    _mm_loadu_si128((const __m128i *)(const void *)k1), 1);
        const __m256i a = _mm256_shuffle_epi8(hdr, sA), hb = _mm256_shuffle_epi8(hdr, sB);
        const __m256i sel = _mm256_blend_epi16(a, _mm256_and_si256(_mm256_srli_epi16(a, 4), _mm256_set1_epi8(0x0F)), 0xC0);
        const __m256i sm = _mm256_or_si256(_mm256_and_si256(sel, mlow),
                                           _mm256_srli_epi16(_mm256_and_si256(hb, _mm256_set1_epi8((char)0xC0)), 2));
        _mm256_storeu_si256((__m256i *)(void *)hsc[b - sp][0], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(sm)));
        _mm256_storeu_si256((__m256i *)(void *)hsc[b - sp][1], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(sm, 1)));
        _mm512_storeu_pd(hmf[b - sp][0], _mm512_cvtepi64_pd(_mm512_cvtepu8_epi64(_mm_srli_si128(_mm256_castsi256_si128(sm), 8))));
        _mm512_storeu_pd(hmf[b - sp][1], _mm512_cvtepi64_pd(_mm512_cvtepu8_epi64(_mm_srli_si128(_mm256_extracti128_si256(sm, 1), 8))));
        const __m128i dd = _mm_unpacklo_epi32(_mm256_castsi256_si128(hdr), _mm256_extracti128_si256(hdr, 1));
        _mm256_storeu_pd(hdm[b - sp], _mm256_cvtps_pd(_mm_cvtph_ps(dd)));
    }
}

/* a block's end for two rows against one prepared row (acc: each row's lanes of digit 0 and 1; bf8: the row's
 * sub-block sums; hmf, hdm: the block's entries of the span's tables): each row's lanes to f64, its two digits
 * joined beside its mins' products, the four sums reduced together, then kernels.c's f64 steps into y[0], y[1] */
TR_TARGET_Q4X
static inline __attribute__((always_inline)) void q4x_end2(__m512i acc[2][2], const double *bf8, const double hmf[2][8],
                                                           const double hdm[4], double scale, double y[2]) {
    const __m512d b16 = _mm512_set1_pd((double)TR_Q4X_BASE);
    const __m512d bf = _mm512_loadu_pd(bf8);
    __m512d t[2], mm[2];
#pragma GCC unroll 2
    for (int r = 0; r < 2; r++) {
        const __m512d s0 = _mm512_add_pd(_mm512_cvtepi32_pd(_mm512_castsi512_si256(acc[r][0])),
                                         _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(acc[r][0], 1)));
        const __m512d s1 = _mm512_add_pd(_mm512_cvtepi32_pd(_mm512_castsi512_si256(acc[r][1])),
                                         _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(acc[r][1], 1)));
        t[r] = _mm512_fmadd_pd(s1, b16, s0);
        mm[r] = _mm512_mul_pd(_mm512_loadu_pd(hmf[r]), bf);
    }
    /* [T0 | M0] and [T1 | M1] by 256-bit halves, then [T0 M0 T1 M1] by 128-bit quarters, then the pairs */
    const __m512d A = _mm512_add_pd(_mm512_shuffle_f64x2(t[0], mm[0], 0x44), _mm512_shuffle_f64x2(t[0], mm[0], 0xEE));
    const __m512d B = _mm512_add_pd(_mm512_shuffle_f64x2(t[1], mm[1], 0x44), _mm512_shuffle_f64x2(t[1], mm[1], 0xEE));
    const __m512d Q = _mm512_add_pd(_mm512_shuffle_f64x2(A, B, 0x88), _mm512_shuffle_f64x2(A, B, 0xDD));
    double sums[8];
    _mm512_storeu_pd(sums, _mm512_add_pd(Q, _mm512_permute_pd(Q, 0x55)));
#pragma GCC unroll 2
    for (int r = 0; r < 2; r++) {
        const double val = hdm[2 * r] * sums[4 * r] - hdm[2 * r + 1] * sums[4 * r + 2];
        y[r] = y[r] + val * scale;
    }
}

/* A short verify pass's pairs: two rows against T prepared rows (T a constant in every caller, 2 or 3), each
 * weight decoded once for all of them: per block and row the quants widened to words and split into nibbles,
 * times their sub-block's scale, once a window; then per token what avx512_q4x_dot2 does with them (one vpdpwssd
 * per 32 columns, digit and row; q4x_end2), each (token, row) its own chain. The headers of TR_Q4_K_SPAN blocks
 * of both rows are read first, as the pairs read them (LESSONS #178, #181). From RAM at 8 threads it reads a
 * dense matrix at 0.97 (T = 2) and 0.88 (T = 3) of a plain read, the W16 panel and a tile at 0.85 and 0.76
 * (docs/MEASUREMENTS.md §A Q4_K weight row decoded once for 2-3 tokens). */
TR_TARGET_Q4X
static inline __attribute__((always_inline)) void q4x_dot_xt_t(const unsigned char *row0, const unsigned char *row1,
                                                               const void *const *xq, int64_t n, float *out, int64_t ys,
                                                               const int T) {
    const unsigned char *p[2] = {row0, row1};
    tr_q4x_view v[TR_Q4X_XT_MAX];
    double y[TR_Q4X_XT_MAX][2];
#pragma GCC unroll 3
    for (int t = 0; t < T; t++) {
        v[t] = tr_q4x_view_of(xq[t], n);
        y[t][0] = y[t][1] = 0.0;
    }
    const __m512i m15 = _mm512_set1_epi16(15);
    const int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    int16_t hsc[TR_Q4_K_SPAN][2][16];
    double hmf[TR_Q4_K_SPAN][2][8];
    double hdm[TR_Q4_K_SPAN][4];
    for (int64_t sp = 0; sp < nb; sp += TR_Q4_K_SPAN) {
        const int64_t se = sp + TR_Q4_K_SPAN < nb ? sp + TR_Q4_K_SPAN : nb;
        q4x_headers2(p, sp, se, hsc, hmf, hdm);
        for (int64_t s = sp; s < se; s++) {
            const int64_t h = s - sp;
            const unsigned char *bq[2] = {p[0] + (size_t)s * TR_Q4_K_BLOCK_BYTES + TR_Q4_K_QS_OFFSET,
                                          p[1] + (size_t)s * TR_Q4_K_BLOCK_BYTES + TR_Q4_K_QS_OFFSET};
            __m512i acc[TR_Q4X_XT_MAX][2][2];
#pragma GCC unroll 2
            for (int r = 0; r < 2; r++) {
                const unsigned char *blk = p[r] + (size_t)s * TR_Q4_K_BLOCK_BYTES;
                _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH, _MM_HINT_T0);
                _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH + 64, _MM_HINT_T0);
                _mm_prefetch((const char *)blk + TR_Q4_K_PREFETCH + 128, _MM_HINT_T0);
            }
#pragma GCC unroll 3
            for (int t = 0; t < T; t++) acc[t][0][0] = acc[t][0][1] = acc[t][1][0] = acc[t][1][1] = _mm512_setzero_si512();
#pragma GCC unroll 4
            for (int c = 0; c < 4; c++) {
                __m512i wl[2], wh[2];
#pragma GCC unroll 2
                for (int r = 0; r < 2; r++) {
                    const __m512i q = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(bq[r] + 32 * c)));
                    wl[r] = _mm512_mullo_epi16(_mm512_and_si512(q, m15), _mm512_set1_epi16(hsc[h][r][2 * c]));
                    wh[r] = _mm512_mullo_epi16(_mm512_srli_epi16(q, 4), _mm512_set1_epi16(hsc[h][r][2 * c + 1]));
                }
#pragma GCC unroll 3
                for (int t = 0; t < T; t++) {
                    const int16_t *d0 = v[t].v0 + s * TR_Q4_K_BLOCK_ELEMS + 64 * c, *d1 = v[t].v1 + s * TR_Q4_K_BLOCK_ELEMS + 64 * c;
                    const __m512i a0 = _mm512_loadu_si512(d0), a1 = _mm512_loadu_si512(d1);
                    const __m512i b0 = _mm512_loadu_si512(d0 + 32), b1 = _mm512_loadu_si512(d1 + 32);
#pragma GCC unroll 2
                    for (int r = 0; r < 2; r++) {
                        acc[t][r][0] = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(acc[t][r][0], wl[r], a0), wh[r], b0);
                        acc[t][r][1] = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(acc[t][r][1], wl[r], a1), wh[r], b1);
                    }
                }
            }
#pragma GCC unroll 3
            for (int t = 0; t < T; t++) q4x_end2(acc[t], v[t].bf + 8 * s, hmf[h], hdm[h], v[t].scale[s], y[t]);
        }
    }
#pragma GCC unroll 3
    for (int t = 0; t < T; t++) {
        out[(int64_t)t * ys] = (float)y[t][0];
        out[(int64_t)t * ys + 1] = (float)y[t][1];
    }
}

#define Q4X_XT_FN(T)                                                                                           \
    TR_TARGET_Q4X static void avx512_q4x_dot_xt_##T(const unsigned char *row0, const unsigned char *row1,        \
                                                    const void *const *xq, int64_t n, float *y, int64_t ys) {    \
        q4x_dot_xt_t(row0, row1, xq, n, y, ys, T);                                                             \
    }
Q4X_XT_FN(2) Q4X_XT_FN(3)
#undef Q4X_XT_FN

typedef void (*q4x_xt_fn)(const unsigned char *, const unsigned char *, const void *const *, int64_t, float *, int64_t);
static const q4x_xt_fn g_q4x_xts[TR_Q4X_XT_MAX + 1] = {NULL, NULL, avx512_q4x_dot_xt_2, avx512_q4x_dot_xt_3};

static void avx512_q4x_dot_xt(const void *row0, const void *row1, const void *const *xq, int64_t n, int T, float *y,
                              int64_t y_stride) {
    g_q4x_xts[T]((const unsigned char *)row0, (const unsigned char *)row1, xq, n, y, y_stride);
}

/* 16 rows into the panel: per block the rows' headers 4 a zmm, their scales and mins transposed to [j][16 rows]
 * by the 16 x 16 byte transpose, d and dmin gathered to the even/odd order by three dword permutes; per window
 * the quants transposed to [byte pair][16 rows], the nibbles widened to words, times the rows' scales plus
 * TR_Q4X_C, stored, and the row term of the window's 32 pairs by vpdpwssd (four chains). The 16 rows are 16
 * short streams no prefetcher follows: each window brings the rows' block s + 2 ahead, and the next rows (one
 * run of 16 row_bytes, 9 lines a window) when the caller gives them (docs/MEASUREMENTS.md §The Q4_K short
 * passes: a 3-row verify pass 0.94x with both). */
TR_TARGET_Q4X
static void avx512_q4x_panel(const void *rows, size_t row_bytes, int64_t n, void *panel, const void *next) {
    const unsigned char *base = (const unsigned char *)rows;
    const int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    const __m512i i00 = _mm512_load_si512(g_q4x_idx[0][0]), i01 = _mm512_load_si512(g_q4x_idx[0][1]);
    const __m512i i10 = _mm512_load_si512(g_q4x_idx[1][0]), i11 = _mm512_load_si512(g_q4x_idx[1][1]);
    const __m512i i20 = _mm512_load_si512(g_q4x_idx[2][0]), i21 = _mm512_load_si512(g_q4x_idx[2][1]);
    const __m512i tA = _mm512_load_si512(g_pm_byte_idx[0]), tB = _mm512_load_si512(g_pm_byte_idx[1]);
    const __m512i tO0 = _mm512_load_si512(g_pm_byte_idx[2]), tO1 = _mm512_load_si512(g_pm_byte_idx[3]);
    const __m512i m15 = _mm512_set1_epi8(15), cw = _mm512_set1_epi16(TR_Q4X_C);
    const __m512i bias = _mm512_set_epi32(0, INT32_MIN, 0, INT32_MIN, 0, INT32_MIN, 0, INT32_MIN, 0, INT32_MIN, 0,
                                          INT32_MIN, 0, INT32_MIN, 0, INT32_MIN);
    const __m512i eo = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    /* the (d, dmin) dword of row r sits at dword 4 (r % 4) of header vector r / 4: rows 0-7 even then odd,
     * rows 8-15 the same, then the eight evens and the eight odds */
    const __m512i dA = _mm512_setr_epi32(0, 8, 16, 24, 4, 12, 20, 28, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i dB = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        unsigned char *sb = (unsigned char *)panel + (size_t)s * TR_Q4X_SB_BYTES;
        const unsigned char *blk0 = base + (size_t)s * TR_Q4_K_BLOCK_BYTES;
        __m512i hdr[4], smv[4];
#pragma GCC unroll 4
        for (int g = 0; g < 4; g++) {
            const unsigned char *h = blk0 + (size_t)(4 * g) * row_bytes;
            __m512i t = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *)(const void *)h));
            t = _mm512_inserti32x4(t, _mm_loadu_si128((const __m128i *)(const void *)(h + row_bytes)), 1);
            t = _mm512_inserti32x4(t, _mm_loadu_si128((const __m128i *)(const void *)(h + 2 * row_bytes)), 2);
            t = _mm512_inserti32x4(t, _mm_loadu_si128((const __m128i *)(const void *)(h + 3 * row_bytes)), 3);
            hdr[g] = t;
            smv[g] = q4x_sc_m4(t);
        }
        /* [j][16 rows]: sm[0] j 0-3, sm[1] j 4-7 (the scales), sm[2] j 8-11, sm[3] j 12-15 (the mins) */
        __m512i sm[4];
        {
            const __m512i A = _mm512_permutex2var_epi8(smv[0], tA, smv[1]), B = _mm512_permutex2var_epi8(smv[0], tB, smv[1]);
            const __m512i C = _mm512_permutex2var_epi8(smv[2], tA, smv[3]), D = _mm512_permutex2var_epi8(smv[2], tB, smv[3]);
            sm[0] = _mm512_permutex2var_epi8(A, tO0, C);
            sm[1] = _mm512_permutex2var_epi8(A, tO1, C);
            sm[2] = _mm512_permutex2var_epi8(B, tO0, D);
            sm[3] = _mm512_permutex2var_epi8(B, tO1, D);
        }
        double *cst = (double *)(void *)sb;
        {
            const __m512i lo = _mm512_permutex2var_epi32(hdr[0], dA, hdr[1]), hi = _mm512_permutex2var_epi32(hdr[2], dA, hdr[3]);
            const __m512i dd = _mm512_permutex2var_epi32(lo, dB, hi); /* (d | dmin << 16) of the rows in even/odd order */
            const __m512 fd = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(dd));
            const __m512 fm = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_srli_epi32(dd, 16)));
            _mm512_storeu_pd(cst, _mm512_cvtps_pd(_mm512_castps512_ps256(fd)));
            _mm512_storeu_pd(cst + 8, _mm512_cvtps_pd(_mm512_extractf32x8_ps(fd, 1)));
            _mm512_storeu_pd(cst + 16, _mm512_cvtps_pd(_mm512_castps512_ps256(fm)));
            _mm512_storeu_pd(cst + 24, _mm512_cvtps_pd(_mm512_extractf32x8_ps(fm, 1)));
#pragma GCC unroll 8
            for (int j = 0; j < 8; j++) {
                const __m512i mj = _mm512_permutexvar_epi32(eo, _mm512_cvtepu8_epi32(q4x_lane(sm[2 + j / 4], j)));
                _mm512_storeu_pd(cst + 32 + 16 * j, _mm512_cvtepi32_pd(_mm512_castsi512_si256(mj)));
                _mm512_storeu_pd(cst + 40 + 16 * j, _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(mj, 1)));
            }
        }
#pragma GCC unroll 1
        for (int c = 0; c < 4; c++) {
            unsigned char *win = sb + TR_Q4X_CONST_BYTES + (size_t)c * TR_Q4X_WIN_BYTES;
            /* the next rows, one contiguous run of 16 row_bytes = 9 lines a window of 64 columns */
            if (next != NULL) {
                const char *np = (const char *)next + (size_t)(9 * (4 * s + c)) * 64;
#pragma GCC unroll 9
                for (int j = 0; j < 9; j++) _mm_prefetch(np + 64 * j, _MM_HINT_T0);
            }
            /* these rows' block s + 2, four rows a window */
            if (s + 2 < nb) {
#pragma GCC unroll 4
                for (int r = 4 * c; r < 4 * c + 4; r++) {
                    const char *pa = (const char *)(base + (size_t)r * row_bytes + (size_t)(s + 2) * TR_Q4_K_BLOCK_BYTES);
                    _mm_prefetch(pa, _MM_HINT_T0);
                    _mm_prefetch(pa + 64, _MM_HINT_T0);
                    _mm_prefetch(pa + 128, _MM_HINT_T0);
                }
            }
            /* the rows' scales of sub-blocks 2c and 2c + 1, in both words of lane r */
            const __m512i s0 = _mm512_cvtepu8_epi32(q4x_lane(sm[c / 2], 2 * c));
            const __m512i s1 = _mm512_cvtepu8_epi32(q4x_lane(sm[c / 2], 2 * c + 1));
            const __m512i sc0 = _mm512_or_si512(s0, _mm512_slli_epi32(s0, 16)), sc1 = _mm512_or_si512(s1, _mm512_slli_epi32(s1, 16));
            const unsigned char *q0 = blk0 + TR_Q4_K_QS_OFFSET + 32 * c;
            __m512i z[8];
#pragma GCC unroll 8
            for (int i = 0; i < 8; i++) {
                const __m256i ra = _mm256_loadu_si256((const __m256i *)(const void *)(q0 + (size_t)(2 * i) * row_bytes));
                const __m256i rb = _mm256_loadu_si256((const __m256i *)(const void *)(q0 + (size_t)(2 * i + 1) * row_bytes));
                z[i] = _mm512_inserti64x4(_mm512_castsi256_si512(ra), rb, 1);
            }
            __m512i A[4], B[4], Cv[8], O[8];
#pragma GCC unroll 4
            for (int k = 0; k < 4; k++) {
                A[k] = _mm512_permutex2var_epi8(z[2 * k], i00, z[2 * k + 1]);
                B[k] = _mm512_permutex2var_epi8(z[2 * k], i01, z[2 * k + 1]);
            }
            Cv[0] = _mm512_permutex2var_epi8(A[0], i10, A[1]);
            Cv[1] = _mm512_permutex2var_epi8(A[0], i11, A[1]);
            Cv[2] = _mm512_permutex2var_epi8(A[2], i10, A[3]);
            Cv[3] = _mm512_permutex2var_epi8(A[2], i11, A[3]);
            Cv[4] = _mm512_permutex2var_epi8(B[0], i10, B[1]);
            Cv[5] = _mm512_permutex2var_epi8(B[0], i11, B[1]);
            Cv[6] = _mm512_permutex2var_epi8(B[2], i10, B[3]);
            Cv[7] = _mm512_permutex2var_epi8(B[2], i11, B[3]);
#pragma GCC unroll 4
            for (int h = 0; h < 4; h++) {
                const __m512i lo = Cv[h < 2 ? h : 2 + h], hi = Cv[h < 2 ? h + 2 : 4 + h];
                O[2 * h] = _mm512_permutex2var_epi8(lo, i20, hi);
                O[2 * h + 1] = _mm512_permutex2var_epi8(lo, i21, hi);
            }
            __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
#pragma GCC unroll 8
            for (int pp = 0; pp < 8; pp++) {
                const __m512i lo4 = _mm512_and_si512(O[pp], m15), hi4 = _mm512_and_si512(_mm512_srli_epi16(O[pp], 4), m15);
#pragma GCC unroll 2
                for (int h = 0; h < 2; h++) {
                    const int u = 2 * pp + h;
                    const __m512i wl = _mm512_add_epi16(_mm512_mullo_epi16(_mm512_cvtepu8_epi16(h ? _mm512_extracti64x4_epi64(lo4, 1)
                                                                                                   : _mm512_castsi512_si256(lo4)), sc0), cw);
                    const __m512i wh = _mm512_add_epi16(_mm512_mullo_epi16(_mm512_cvtepu8_epi16(h ? _mm512_extracti64x4_epi64(hi4, 1)
                                                                                                   : _mm512_castsi512_si256(hi4)), sc1), cw);
                    _mm512_storeu_si512(win + 64 + 128 * u, wl);
                    _mm512_storeu_si512(win + 128 + 128 * u, wh);
                    acc[u & 3] = _mm512_dpwssd_epi32(acc[u & 3], wl, wh);
                }
            }
            const __m512i e = _mm512_add_epi32(_mm512_add_epi32(acc[0], acc[1]), _mm512_add_epi32(acc[2], acc[3]));
            _mm512_storeu_si512(win, _mm512_add_epi32(_mm512_sub_epi32(_mm512_setzero_si512(), e), bias));
        }
    }
}

/* the panel against T prepared rows (T a constant in every caller): per window each lane starts at minus the
 * row and token terms (+ 2^31 on the even rows), takes 16 Winograd products per digit and token, and joins
 * the super-block's int64 sums by the even/odd split; at the block's end T exact in f64 (one fma), the mins
 * by exact fmas, then the f64 steps; the even rows' outputs in one vector, the odd rows' in another */
typedef int __attribute__((may_alias)) q4x_i32;
TR_TARGET_Q4X
static inline __attribute__((always_inline)) void q4x_tile_t(const unsigned char *panel, const void *const *xq,
                                                             int64_t n, float *y, int64_t ys, const int T) {
    tr_q4x_view v[TR_Q4X_TILE_MAX];
    __m512d ye[TR_Q4X_TILE_MAX], yo[TR_Q4X_TILE_MAX];
#pragma GCC unroll 4
    for (int t = 0; t < T; t++) {
        v[t] = tr_q4x_view_of(xq[t], n);
        ye[t] = yo[t] = _mm512_setzero_pd();
    }
    const __m512i bias4 = _mm512_set1_epi64(4ll << 31);
    const __m512d b16 = _mm512_set1_pd((double)TR_Q4X_BASE);
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        const unsigned char *sb = panel + (size_t)s * TR_Q4X_SB_BYTES;
        const double *cst = (const double *)(const void *)sb;
        __m512i ae[TR_Q4X_TILE_MAX][2], ao[TR_Q4X_TILE_MAX][2];
#pragma GCC unroll 4
        for (int t = 0; t < T; t++) ae[t][0] = ae[t][1] = ao[t][0] = ao[t][1] = _mm512_setzero_si512();
        for (int c = 0; c < 4; c++) {
            const unsigned char *win = sb + TR_Q4X_CONST_BYTES + (size_t)c * TR_Q4X_WIN_BYTES;
            const int64_t w = 4 * s + c, k0 = 64 * w;
            const __m512i rn = _mm512_loadu_si512(win);
            __m512i a[TR_Q4X_TILE_MAX][2];
#pragma GCC unroll 4
            for (int t = 0; t < T; t++) {
                a[t][0] = _mm512_sub_epi32(rn, _mm512_set1_epi32(v[t].tok[2 * w]));
                a[t][1] = _mm512_sub_epi32(rn, _mm512_set1_epi32(v[t].tok[2 * w + 1]));
            }
#pragma GCC unroll 4
            for (int u = 0; u < 16; u++) {
                const __m512i wl = _mm512_loadu_si512(win + 64 + 128 * u), wh = _mm512_loadu_si512(win + 128 + 128 * u);
#pragma GCC unroll 4
                for (int t = 0; t < T; t++) {
                    const __m512i f1a = _mm512_add_epi16(wl, _mm512_set1_epi32(*(const q4x_i32 *)(const void *)(v[t].v0 + k0 + 32 + 2 * u)));
                    const __m512i f2a = _mm512_add_epi16(wh, _mm512_set1_epi32(*(const q4x_i32 *)(const void *)(v[t].v0 + k0 + 2 * u)));
                    a[t][0] = _mm512_dpwssd_epi32(a[t][0], f1a, f2a);
                    const __m512i f1b = _mm512_add_epi16(wl, _mm512_set1_epi32(*(const q4x_i32 *)(const void *)(v[t].v1 + k0 + 32 + 2 * u)));
                    const __m512i f2b = _mm512_add_epi16(wh, _mm512_set1_epi32(*(const q4x_i32 *)(const void *)(v[t].v1 + k0 + 2 * u)));
                    a[t][1] = _mm512_dpwssd_epi32(a[t][1], f1b, f2b);
                }
            }
#pragma GCC unroll 4
            for (int t = 0; t < T; t++)
#pragma GCC unroll 2
                for (int d = 0; d < 2; d++) {
                    ao[t][d] = _mm512_add_epi64(ao[t][d], _mm512_srai_epi64(a[t][d], 32));
                    ae[t][d] = _mm512_add_epi64(ae[t][d], a[t][d]);
                }
        }
#pragma GCC unroll 4
        for (int t = 0; t < T; t++) {
            const __m512i ev0 = _mm512_sub_epi64(_mm512_sub_epi64(ae[t][0], _mm512_slli_epi64(ao[t][0], 32)), bias4);
            const __m512i ev1 = _mm512_sub_epi64(_mm512_sub_epi64(ae[t][1], _mm512_slli_epi64(ao[t][1], 32)), bias4);
            const __m512d te = _mm512_fmadd_pd(_mm512_cvtepi64_pd(ev1), b16, _mm512_cvtepi64_pd(ev0));
            const __m512d to = _mm512_fmadd_pd(_mm512_cvtepi64_pd(ao[t][1]), b16, _mm512_cvtepi64_pd(ao[t][0]));
            const double *bf = v[t].bf + 8 * s;
            __m512d Me = _mm512_mul_pd(_mm512_loadu_pd(cst + 32), _mm512_set1_pd(bf[0]));
            __m512d Mo = _mm512_mul_pd(_mm512_loadu_pd(cst + 40), _mm512_set1_pd(bf[0]));
#pragma GCC unroll 8
            for (int j = 1; j < 8; j++) {
                const __m512d bj = _mm512_set1_pd(bf[j]);
                Me = _mm512_fmadd_pd(_mm512_loadu_pd(cst + 32 + 16 * j), bj, Me);
                Mo = _mm512_fmadd_pd(_mm512_loadu_pd(cst + 40 + 16 * j), bj, Mo);
            }
            const __m512d sc = _mm512_set1_pd(v[t].scale[s]);
            const __m512d ve = _mm512_sub_pd(_mm512_mul_pd(_mm512_loadu_pd(cst), te), _mm512_mul_pd(_mm512_loadu_pd(cst + 16), Me));
            const __m512d vo = _mm512_sub_pd(_mm512_mul_pd(_mm512_loadu_pd(cst + 8), to), _mm512_mul_pd(_mm512_loadu_pd(cst + 24), Mo));
            ye[t] = _mm512_add_pd(ye[t], _mm512_mul_pd(ve, sc));
            yo[t] = _mm512_add_pd(yo[t], _mm512_mul_pd(vo, sc));
        }
    }
    const __m512i il = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
#pragma GCC unroll 4
    for (int t = 0; t < T; t++) {
        const __m512 e = _mm512_castps256_ps512(_mm512_cvtpd_ps(ye[t])), o = _mm512_castps256_ps512(_mm512_cvtpd_ps(yo[t]));
        _mm512_storeu_ps(y + (size_t)t * (size_t)ys, _mm512_permutex2var_ps(e, il, o));
    }
}

#define Q4X_TILE_FN(T)                                                                                         \
    TR_TARGET_Q4X static void avx512_q4x_tile_##T(const unsigned char *panel, const void *const *xq, int64_t n,   \
                                                  float *y, int64_t ys) {                                     \
        q4x_tile_t(panel, xq, n, y, ys, T);                                                                   \
    }
Q4X_TILE_FN(1) Q4X_TILE_FN(2) Q4X_TILE_FN(3) Q4X_TILE_FN(4)
#undef Q4X_TILE_FN

typedef void (*q4x_tile_fn)(const unsigned char *, const void *const *, int64_t, float *, int64_t);
static const q4x_tile_fn g_q4x_tiles[TR_Q4X_TILE_MAX + 1] = {NULL, avx512_q4x_tile_1, avx512_q4x_tile_2,
                                                             avx512_q4x_tile_3, avx512_q4x_tile_4};

static void avx512_q4x_tile(const void *panel, const void *const *xq, int64_t n, int T, float *y, int64_t y_stride) {
    g_q4x_tiles[T]((const unsigned char *)panel, xq, n, y, y_stride);
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
            g_avx2.dot_row[TR_TYPE_BF16] = avx2_dot_row_bf16;
            g_avx2.dot_row_x4[TR_TYPE_BF16] = avx2_dot_row_x4_bf16;
            g_avx2.dot_row[TR_TYPE_Q8_0] = avx2_dot_row_q8_0;
            g_avx2.dot_row_x4[TR_TYPE_Q8_0] = avx2_dot_row_x4_q8_0;
            g_avx2.dot_row_xt[TR_TYPE_Q8_0] = avx2_dot_row_xt_q8_0;
            /* Q4_K's integer definition: by rows, and the W16 panel road in two halves */
            g_avx2.q4x_prep = avx2_q4x_prep;
            g_avx2.q4x_dot2 = avx2_q4x_dot2;
            g_avx2.q4x_dot_xt = avx2_q4x_dot_xt;
            g_avx2.q4x_panel = avx2_q4x_panel;
            g_avx2.q4x_tile = avx2_q4x_tile;
            g_avx2.q4x_tile_max = 3;
            g_avx2.dot_row[TR_TYPE_Q6_K] = avx2_dot_row_q6_k;
            g_avx2.dot_row_x4[TR_TYPE_Q6_K] = avx2_dot_row_x4_q6_k;
            tr_hb_fill_x86(&g_avx2, c->f16c, 0);
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
            g_avx512.dot_row[TR_TYPE_BF16] = avx512_dot_row_bf16;
            g_avx512.dot_row_x4[TR_TYPE_BF16] = avx512_dot_row_x4_bf16;
            g_avx512.dot_row[TR_TYPE_Q8_0] = avx512_dot_row_q8_0;
            g_avx512.dot_row_x4[TR_TYPE_Q8_0] = avx512_dot_row_x4_q8_0;
            g_avx512.dot_row_xt[TR_TYPE_Q8_0] = avx512_dot_row_xt_q8_0;
            g_avx512.dot_row2_x4[TR_TYPE_Q8_0] = avx512_dot_row2_x4_q8_0;
            g_avx512.dot_row2_x4[TR_TYPE_Q6_K] = avx512_dot_row2_x4_q6_k;
            g_avx512.dot_row2_x8[TR_TYPE_Q8_0] = avx512_dot_row2_x8_q8_0;
            g_avx512.dot_row2_x8[TR_TYPE_Q6_K] = avx512_dot_row2_x8_q6_k;
            g_avx512.dot_row[TR_TYPE_Q6_K] = avx512_dot_row_q6_k;
            g_avx512.dot_row_x4[TR_TYPE_Q6_K] = avx512_dot_row_x4_q6_k;
            g_avx512.pm_panel[TR_TYPE_Q8_0] = avx512_pm_panel_q8_0;
            /* Q4_K's integer definition: W16 where the CPU has the byte permutes, VNNI and DQ; AVX2's kernels
             * elsewhere (their panel road in two halves) */
            if (c->avx512bw && c->avx512vl && c->avx512dq && c->avx512vnni && c->avx512vbmi && c->f16c) {
                pm_build_byte_idx();
                q4x_build_idx();
                g_avx512.q4x_prep = avx512_q4x_prep;
                g_avx512.q4x_dot2 = avx512_q4x_dot2;
                g_avx512.q4x_dot_xt = avx512_q4x_dot_xt;
                g_avx512.q4x_panel = avx512_q4x_panel;
                g_avx512.q4x_tile = avx512_q4x_tile;
                g_avx512.q4x_tile_max = TR_Q4X_TILE_MAX;
            } else {
                g_avx512.q4x_prep = avx2_q4x_prep;
                g_avx512.q4x_dot2 = avx2_q4x_dot2;
                g_avx512.q4x_dot_xt = avx2_q4x_dot_xt;
                g_avx512.q4x_panel = avx2_q4x_panel;
                g_avx512.q4x_tile = avx2_q4x_tile;
                g_avx512.q4x_tile_max = 3;
            }
            g_avx512.pm_interleave = avx512_pm_interleave;
            g_avx512.pm_tile = avx512_pm_tile;
            tr_hb_fill_x86(&g_avx512, c->f16c,
                           c->f16c && c->avx512bw && c->avx512vl && c->avx512dq && c->avx512vnni);
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
