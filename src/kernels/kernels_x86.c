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

/* hot: end */

/* ---- tables --------------------------------------------------------------- */

static tr_kernels g_avx2, g_avx512;
static int g_avx2_built = 0, g_avx512_built = 0;

const tr_kernels *tr_kernels_x86_tier(const char *tier) {
    const tr_cpu_info *c = tr_cpu();
    if (strcmp(tier, "avx2") == 0) {
        if (!c->avx2) return NULL;
        if (!g_avx2_built) {
            g_avx2 = *tr_kernels_scalar();
            g_avx2.tier = "avx2";
            g_avx2.dot_f32 = avx2_dot_f32;
            g_avx2.axpy_f32 = avx2_axpy_f32;
            g_avx2.dot_row[TR_TYPE_Q8_0] = avx2_dot_row_q8_0;
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
            g_avx512.dot_row[TR_TYPE_Q8_0] = avx512_dot_row_q8_0;
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
