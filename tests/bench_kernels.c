/* bench_kernels.c — microbenchmark of the CPU kernels, one line per (tier, kernel, size).
 *
 * Method, so that a difference can be told apart from noise:
 *   - every measurement is `runs` timed repetitions of ~`run_ms` each, after a warm-up;
 *   - the reported value is the median; `spread` is (max - min) / median of those runs,
 *     the noise floor of that line on this machine at this moment;
 *   - an optimization counts only if it moves the median by more than the spread.
 * The whole benchmark stays well under 60 seconds (docs/ARCHITECTURE.md, safety).
 *
 *   bench_kernels [--runs N] [--ms M] [--threads T]
 *
 * Output columns: tier, kernel, n (elements per row), median ns per call,
 * median million elements per second, spread %. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/platform.h"
#include "../src/base/cpu.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define MAX_RUNS 31

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static float frand(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 40) & 0xFFFFFF) / (float)0x1000000 * 2.0f - 1.0f;
}

/* Fills one row of n elements of `type` with plausible random content. */
static void fill_row(tr_type type, uint8_t *row, int64_t n) {
    switch (type) {
    case TR_TYPE_F32:
        for (int64_t i = 0; i < n; i++) { float v = frand(); memcpy(row + 4 * i, &v, 4); }
        break;
    case TR_TYPE_F16:
        for (int64_t i = 0; i < n; i++) {
            uint16_t h = (uint16_t)((rng_state >> 17) & 0x7BFF);   /* finite half values */
            frand();
            memcpy(row + 2 * i, &h, 2);
        }
        break;
    case TR_TYPE_Q8_0:
        for (int64_t b = 0; b < n / 32; b++) {
            uint8_t *blk = row + 34 * b;
            uint16_t d = 0x3000 | (uint16_t)((rng_state >> 30) & 0x3FF);  /* small positive scale */
            frand();
            memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)(int8_t)(frand() * 127.0f);
        }
        break;
    default:
        break;
    }
}

/* ---- candidate: int8 x int8 with VNNI (docs/MEASUREMENTS.md question 21) --------
 *
 * Not a kernel and not a tier: a measurement. Quantizing the activations to int8 is a
 * different number from the float dot, so this can never be one of our bit-identical
 * variants; it lives here to say how much lever 2 (int8 activations, llama.cpp's prefill
 * path) could be worth before a line of it is written. Same row layout as dot_row q8_0
 * (34-byte blocks: fp16 scale + 32 int8), same elements per call, so the two lines compare
 * directly. The per-block weight sums the unsigned trick needs are precomputed, as a real
 * kernel would store them beside the row; the cost of quantizing the activations is measured
 * on its own line, because a matmul pays it once for every row of the matrix. */
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define TR_BENCH_VNNI 1

/* x -> int8 with one scale per 32 elements, the same blocking as Q8_0. */
static void quantize_q8_0(const float *x, int64_t n, int8_t *q, float *d) {
    for (int64_t b = 0; b < n / TR_Q8_0_BLOCK_ELEMS; b++) {
        const float *src = x + b * TR_Q8_0_BLOCK_ELEMS;
        float amax = 0.0f;
        for (int i = 0; i < TR_Q8_0_BLOCK_ELEMS; i++) {
            float v = src[i] < 0.0f ? -src[i] : src[i];
            if (v > amax) amax = v;
        }
        float scale = amax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        d[b] = scale;
        for (int i = 0; i < TR_Q8_0_BLOCK_ELEMS; i++) {
            float v = src[i] * inv;
            int r = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
            q[b * TR_Q8_0_BLOCK_ELEMS + i] = (int8_t)(r < -127 ? -127 : (r > 127 ? 127 : r));
        }
    }
}

/* sum of the 32 weights of each block, for the +128 correction below */
static void q8_0_row_sums(const uint8_t *row, int64_t n, int32_t *sums) {
    for (int64_t b = 0; b < n / TR_Q8_0_BLOCK_ELEMS; b++) {
        const int8_t *w = (const int8_t *)(row + TR_Q8_0_BLOCK_BYTES * b + TR_Q8_0_SCALE_BYTES);
        int32_t s = 0;
        for (int i = 0; i < TR_Q8_0_BLOCK_ELEMS; i++) s += w[i];
        sums[b] = s;
    }
}

/* One 32-element block per iteration: _mm256_dpbusd_epi32 wants the first operand unsigned,
 * so the activations are shifted by +128 and the block's weight sum pays the shift back
 * (dot(x + 128, w) = dot(x, w) + 128 * sum(w)). The int32 lanes are converted and scaled into
 * a float accumulator without a horizontal sum per block, as ggml does. */
__attribute__((target("avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")))
static float dot_q8_q8_vnni(const uint8_t *row, const int8_t *xq, const float *xd,
                            const int32_t *wsums, int64_t n) {
    const __m256i off = _mm256_set1_epi8((char)-128); /* xor flips the sign bit: +128 unsigned */
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.0f;
    for (int64_t b = 0; b < n / TR_Q8_0_BLOCK_ELEMS; b++) {
        const uint8_t *blk = row + TR_Q8_0_BLOCK_BYTES * b;
        uint16_t half;
        memcpy(&half, blk, sizeof half);
        float dw = tr_half_to_float(half);
        __m256i w = _mm256_loadu_si256((const __m256i *)(blk + TR_Q8_0_SCALE_BYTES));
        __m256i x = _mm256_loadu_si256((const __m256i *)(xq + b * TR_Q8_0_BLOCK_ELEMS));
        __m256i xu = _mm256_xor_si256(x, off);
        __m256i prod = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xu, w);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * xd[b]), _mm256_cvtepi32_ps(prod), acc);
        corr += dw * xd[b] * 128.0f * (float)wsums[b];
    }
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s) - corr;
}

/* ---- question 21, second take (docs/LESSONS.md #65) ------------------------------------
 * The candidate above is one row against ONE token, with a scalar half->float and a scalar
 * correction chain inside the loop, and it was compared with our one-token float dot: +1-8%,
 * "lever 2 is not worth writing". But the kernel the prefill really runs is one row against
 * FOUR tokens, so the honest question is what int8 gives with that same structure. Three
 * candidates, measured alternately with the engine's own x4 kernel:
 *   vnni256 x4   the weight block loaded and made unsigned once (ggml's sign trick), then one
 *                dpbusd per token; scale by F16C
 *   vnni512 x4   two weight blocks per 512-bit dpbusd; activations stored unsigned and the
 *                +128 shift paid back in the integer domain, from a per-block constant that a
 *                real kernel would keep beside the row together with the float scale
 *   float x8     the exact lever (one row against 8 tokens, same arithmetic per token as x4,
 *                bit-identical): what "wider blocks" buys without giving up exactness
 * Still measurements, not kernels: int8 activations are another number, never bit-identical. */
__attribute__((target("avx2,fma")))
static inline float hsum256_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* xq: 4 input rows of n int8 each, xd: their 4 x n/32 scales */
__attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")))
static void dot_q8_q8_vnni256_x4(const uint8_t *row, const int8_t *xq, const float *xd, int64_t n, float *out) {
    const int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    const __m256i zero = _mm256_setzero_si256();
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    const int8_t *q0 = xq, *q1 = xq + n, *q2 = xq + 2 * n, *q3 = xq + 3 * n;
    const float *d0 = xd, *d1 = xd + nb, *d2 = xd + 2 * nb, *d3 = xd + 3 * nb;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *blk = row + TR_Q8_0_BLOCK_BYTES * b;
        uint16_t half;
        memcpy(&half, blk, sizeof half);
        __m256 dw = _mm256_set1_ps(_cvtsh_ss(half));
        __m256i w = _mm256_loadu_si256((const __m256i *)(blk + TR_Q8_0_SCALE_BYTES));
        __m256i aw = _mm256_sign_epi8(w, w);
        int64_t at = b * TR_Q8_0_BLOCK_ELEMS;
        __m256i p0 = _mm256_dpbusd_epi32(zero, aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q0 + at)), w));
        __m256i p1 = _mm256_dpbusd_epi32(zero, aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q1 + at)), w));
        __m256i p2 = _mm256_dpbusd_epi32(zero, aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q2 + at)), w));
        __m256i p3 = _mm256_dpbusd_epi32(zero, aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q3 + at)), w));
        a0 = _mm256_fmadd_ps(_mm256_mul_ps(dw, _mm256_broadcast_ss(d0 + b)), _mm256_cvtepi32_ps(p0), a0);
        a1 = _mm256_fmadd_ps(_mm256_mul_ps(dw, _mm256_broadcast_ss(d1 + b)), _mm256_cvtepi32_ps(p1), a1);
        a2 = _mm256_fmadd_ps(_mm256_mul_ps(dw, _mm256_broadcast_ss(d2 + b)), _mm256_cvtepi32_ps(p2), a2);
        a3 = _mm256_fmadd_ps(_mm256_mul_ps(dw, _mm256_broadcast_ss(d3 + b)), _mm256_cvtepi32_ps(p3), a3);
    }
    out[0] = hsum256_ps(a0);
    out[1] = hsum256_ps(a1);
    out[2] = hsum256_ps(a2);
    out[3] = hsum256_ps(a3);
}

/* xu: 4 input rows of n bytes, each int8 value + 128; dwf and wcorr (= -128 * sum of the block's
 * weights) are the per-block side data of the row. n must hold an even number of blocks. */
__attribute__((target("avx2,fma,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")))
static void dot_q8_q8_vnni512_x4(const uint8_t *row, const float *dwf, const int32_t *wcorr, const uint8_t *xu,
                                 const float *xd, int64_t n, float *out) {
    const int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    const uint8_t *q0 = xu, *q1 = xu + n, *q2 = xu + 2 * n, *q3 = xu + 3 * n;
    const float *d0 = xd, *d1 = xd + nb, *d2 = xd + 2 * nb, *d3 = xd + 3 * nb;
    for (int64_t b = 0; b + 1 < nb; b += 2) {
        const uint8_t *blk = row + TR_Q8_0_BLOCK_BYTES * b;
        __m512i w = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(blk + TR_Q8_0_SCALE_BYTES))),
            _mm256_loadu_si256((const __m256i *)(blk + TR_Q8_0_BLOCK_BYTES + TR_Q8_0_SCALE_BYTES)), 1);
        /* lane 0 of each half starts from its block's correction; the casts leave the other
         * lanes undefined, the mask clears them */
        __m512i corr = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castsi128_si256(_mm_cvtsi32_si128(wcorr[b]))),
                                          _mm256_castsi128_si256(_mm_cvtsi32_si128(wcorr[b + 1])), 1);
        corr = _mm512_maskz_mov_epi32(0x0101, corr);
        __m512 dw = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_broadcast_ss(dwf + b)),
                                       _mm256_broadcast_ss(dwf + b + 1), 1);
        int64_t at = b * TR_Q8_0_BLOCK_ELEMS;
#define TR_BENCH_TOKEN(acc, q, d)                                                                         \
    do {                                                                                                  \
        __m512i p = _mm512_dpbusd_epi32(corr, _mm512_loadu_si512((const void *)((q) + at)), w);           \
        __m512 dx = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_broadcast_ss((d) + b)),              \
                                       _mm256_broadcast_ss((d) + b + 1), 1);                              \
        acc = _mm512_fmadd_ps(_mm512_mul_ps(dw, dx), _mm512_cvtepi32_ps(p), acc);                         \
    } while (0)
        TR_BENCH_TOKEN(a0, q0, d0);
        TR_BENCH_TOKEN(a1, q1, d1);
        TR_BENCH_TOKEN(a2, q2, d2);
        TR_BENCH_TOKEN(a3, q3, d3);
#undef TR_BENCH_TOKEN
    }
    out[0] = _mm512_reduce_add_ps(a0);
    out[1] = _mm512_reduce_add_ps(a1);
    out[2] = _mm512_reduce_add_ps(a2);
    out[3] = _mm512_reduce_add_ps(a3);
}

/* One weight row against 8 float input rows: the engine's avx512 x4 kernel with twice the
 * accumulators, the same multiply and add per element, the same lane tree. */
__attribute__((target("avx512f")))
static void dot_row_x8_q8_0_float(const uint8_t *row, const float *x, int64_t stride, int64_t n, float *out) {
    const int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *blk = row + TR_Q8_0_BLOCK_BYTES * b;
        const uint8_t *qs = blk + TR_Q8_0_SCALE_BYTES;
        const float *xb = x + b * TR_Q8_0_BLOCK_ELEMS;
        __m512 d = _mm512_set1_ps(tr_q8_0_block_scale(blk));
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m512 q = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + j))));
            __m512 w = _mm512_mul_ps(d, q);
            const float *p = xb + j;
            a0 = _mm512_add_ps(a0, _mm512_mul_ps(w, _mm512_loadu_ps(p)));
            a1 = _mm512_add_ps(a1, _mm512_mul_ps(w, _mm512_loadu_ps(p + stride)));
            a2 = _mm512_add_ps(a2, _mm512_mul_ps(w, _mm512_loadu_ps(p + 2 * stride)));
            a3 = _mm512_add_ps(a3, _mm512_mul_ps(w, _mm512_loadu_ps(p + 3 * stride)));
            a4 = _mm512_add_ps(a4, _mm512_mul_ps(w, _mm512_loadu_ps(p + 4 * stride)));
            a5 = _mm512_add_ps(a5, _mm512_mul_ps(w, _mm512_loadu_ps(p + 5 * stride)));
            a6 = _mm512_add_ps(a6, _mm512_mul_ps(w, _mm512_loadu_ps(p + 6 * stride)));
            a7 = _mm512_add_ps(a7, _mm512_mul_ps(w, _mm512_loadu_ps(p + 7 * stride)));
        }
    }
    float lane[TR_LANES];
#define TR_BENCH_OUT(i, a) do { _mm512_storeu_ps(lane, a); out[i] = tr_lane_combine(lane); } while (0)
    TR_BENCH_OUT(0, a0); TR_BENCH_OUT(1, a1); TR_BENCH_OUT(2, a2); TR_BENCH_OUT(3, a3);
    TR_BENCH_OUT(4, a4); TR_BENCH_OUT(5, a5); TR_BENCH_OUT(6, a6); TR_BENCH_OUT(7, a7);
#undef TR_BENCH_OUT
}
#endif

typedef struct {
    const tr_kernels *k;
    tr_type type;
    const uint8_t *row;
    const float *a, *x;
    int64_t n;
    float sink;
} dot_job;

static void run_dot(dot_job *j, long calls) {
    float s = 0;
    if (j->type == TR_TYPE_COUNT) {
        for (long c = 0; c < calls; c++) s += j->k->dot_f32(j->a, j->x, j->n);
    } else {
        for (long c = 0; c < calls; c++) s += j->k->dot_row[j->type](j->row, j->x, j->n);
    }
    j->sink = s;
}

/* Times `calls` calls repeatedly; returns median ns per call and the spread. */
static void measure(dot_job *j, int runs, double run_ms, double *ns_median, double *spread) {
    long calls = 1;
    double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < 0.02) { run_dot(j, calls); calls *= 2; }   /* warm-up + calibration */
    double per_call = (tr_time_sec() - t0) / (double)(calls - 1);    /* 1 + 2 + ... + calls/2 */
    calls = (long)(run_ms / 1000.0 / (per_call > 0 ? per_call : 1e-9));
    if (calls < 1) calls = 1;

    double ns[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        run_dot(j, calls);
        ns[r] = (tr_time_sec() - a) * 1e9 / (double)calls;
    }
    qsort(ns, (size_t)runs, sizeof ns[0], cmp_double);
    *ns_median = ns[runs / 2];
    *spread = (ns[runs - 1] - ns[0]) / ns[runs / 2];
}

/* Same row against TR_DOT_TOKENS input rows (the block path of tr_matmul): ns per call
 * covers TR_DOT_TOKENS * n elements. */
static void measure_x4(const tr_kernels *k, const uint8_t *row, const float *x, int64_t n, int runs, double run_ms,
                        double *ns_median, double *spread) {
    float out[TR_DOT_TOKENS];
    long calls = 1;
    double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < 0.02) {
        for (long c = 0; c < calls; c++) k->dot_row_x4[TR_TYPE_Q8_0](row, x, n, n, out);
        calls *= 2;
    }
    double per_call = (tr_time_sec() - t0) / (double)(calls - 1);
    calls = (long)(run_ms / 1000.0 / (per_call > 0 ? per_call : 1e-9));
    if (calls < 1) calls = 1;

    double ns[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        for (long c = 0; c < calls; c++) k->dot_row_x4[TR_TYPE_Q8_0](row, x, n, n, out);
        ns[r] = (tr_time_sec() - a) * 1e9 / (double)calls;
    }
    qsort(ns, (size_t)runs, sizeof ns[0], cmp_double);
    *ns_median = ns[runs / 2];
    *spread = (ns[runs - 1] - ns[0]) / ns[runs / 2];
}

#if defined(TR_BENCH_VNNI)
/* Same shape as measure_x4, for the two candidate lines of question 21. */
static void measure_vnni(const uint8_t *row, const int8_t *xq, const float *xd, const int32_t *wsums, int64_t n,
                         int runs, double run_ms, double *ns_median, double *spread) {
    volatile float sink = 0;
    long calls = 1;
    double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < 0.02) {
        for (long c = 0; c < calls; c++) sink += dot_q8_q8_vnni(row, xq, xd, wsums, n);
        calls *= 2;
    }
    double per_call = (tr_time_sec() - t0) / (double)(calls - 1);
    calls = (long)(run_ms / 1000.0 / (per_call > 0 ? per_call : 1e-9));
    if (calls < 1) calls = 1;

    double ns[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        for (long c = 0; c < calls; c++) sink += dot_q8_q8_vnni(row, xq, xd, wsums, n);
        ns[r] = (tr_time_sec() - a) * 1e9 / (double)calls;
    }
    qsort(ns, (size_t)runs, sizeof ns[0], cmp_double);
    *ns_median = ns[runs / 2];
    *spread = (ns[runs - 1] - ns[0]) / ns[runs / 2];
}

static void measure_quant(const float *x, int64_t n, int8_t *q, float *d, int runs, double run_ms,
                          double *ns_median, double *spread) {
    long calls = 1;
    double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < 0.02) {
        for (long c = 0; c < calls; c++) quantize_q8_0(x, n, q, d);
        calls *= 2;
    }
    double per_call = (tr_time_sec() - t0) / (double)(calls - 1);
    calls = (long)(run_ms / 1000.0 / (per_call > 0 ? per_call : 1e-9));
    if (calls < 1) calls = 1;

    double ns[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        for (long c = 0; c < calls; c++) quantize_q8_0(x, n, q, d);
        ns[r] = (tr_time_sec() - a) * 1e9 / (double)calls;
    }
    qsort(ns, (size_t)runs, sizeof ns[0], cmp_double);
    *ns_median = ns[runs / 2];
    *spread = (ns[runs - 1] - ns[0]) / ns[runs / 2];
}

/* The four kernels of the second take on one weight row, run ALTERNATELY (a b c d a b c d ...):
 * a comparison measured one kernel after the other moves with the temperature of the core
 * (docs/LESSONS.md #46). ns are per input row, so the lines compare directly. */
enum { CAND_X4_FLOAT = 0, CAND_VNNI256_X4, CAND_VNNI512_X4, CAND_X8_FLOAT, CAND_COUNT };

typedef struct {
    const tr_kernels *k;
    const uint8_t *row;
    const float *x;      /* 8 rows of n floats */
    const int8_t *xq;    /* 4 rows of n int8 */
    const uint8_t *xu;   /* the same + 128 */
    const float *xd;     /* 4 rows of n/32 scales */
    const float *dwf;    /* per block: the weight scale as float */
    const int32_t *wcorr; /* per block: -128 * sum of the weights */
    int64_t n;
} cand_job;

static volatile float cand_sink;

static void cand_run(int which, const cand_job *j, long calls) {
    float out[8], s = 0;
    for (long c = 0; c < calls; c++) {
        switch (which) {
        case CAND_X4_FLOAT: j->k->dot_row_x4[TR_TYPE_Q8_0](j->row, j->x, j->n, j->n, out); break;
        case CAND_VNNI256_X4: dot_q8_q8_vnni256_x4(j->row, j->xq, j->xd, j->n, out); break;
        case CAND_VNNI512_X4: dot_q8_q8_vnni512_x4(j->row, j->dwf, j->wcorr, j->xu, j->xd, j->n, out); break;
        default: dot_row_x8_q8_0_float(j->row, j->x, j->n, j->n, out); break;
        }
        s += out[0] + out[3];
    }
    cand_sink = s;
}

static void measure_candidates(const cand_job *j, int runs, double run_ms, double ns_median[CAND_COUNT],
                               double spread[CAND_COUNT]) {
    static const double rows_per_call[CAND_COUNT] = {4, 4, 4, 8};
    static double ns[CAND_COUNT][MAX_RUNS];
    long calls[CAND_COUNT];
    for (int w = 0; w < CAND_COUNT; w++) {
        long c = 1;
        double t0 = tr_time_sec();
        while (tr_time_sec() - t0 < 0.02) { cand_run(w, j, c); c *= 2; }
        double per_call = (tr_time_sec() - t0) / (double)(c - 1);
        calls[w] = (long)(run_ms / 1000.0 / (per_call > 0 ? per_call : 1e-9));
        if (calls[w] < 1) calls[w] = 1;
    }
    for (int r = 0; r < runs; r++)
        for (int w = 0; w < CAND_COUNT; w++) {
            double a = tr_time_sec();
            cand_run(w, j, calls[w]);
            ns[w][r] = (tr_time_sec() - a) * 1e9 / (double)calls[w] / rows_per_call[w];
        }
    for (int w = 0; w < CAND_COUNT; w++) {
        qsort(ns[w], (size_t)runs, sizeof ns[w][0], cmp_double);
        ns_median[w] = ns[w][runs / 2];
        spread[w] = (ns[w][runs - 1] - ns[w][0]) / ns[w][runs / 2];
    }
}
#endif

typedef struct {
    tr_mat w;
    const float *x;
    float *y;
} mat_job;

/* One run = `calls` matmuls of `n_tokens` input rows back to back: n_tokens 1 is the rhythm
 * of a decode step, a larger one the block path of a prefill pass. */
static void measure_matmul(tr_pool *pool, mat_job *m, int runs, int calls, int64_t n_tokens, double *ms_median,
                            double *spread) {
    tr_matmul(pool, &m->w, m->x, n_tokens, m->y);          /* warm-up */
    double ms[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        for (int c = 0; c < calls; c++) tr_matmul(pool, &m->w, m->x, n_tokens, m->y);
        ms[r] = (tr_time_sec() - a) * 1e3 / calls;
    }
    qsort(ms, (size_t)runs, sizeof ms[0], cmp_double);
    *ms_median = ms[runs / 2];
    *spread = (ms[runs - 1] - ms[0]) / ms[runs / 2];
}

static void empty_body(void *ctx, int64_t begin, int64_t end, int worker) {
    (void)ctx; (void)begin; (void)end; (void)worker;
}

/* Cost of one tr_parallel_for that hands one empty chunk to every thread: the price of
 * waking the pool, paid on every parallel matmul. */
static void measure_dispatch(tr_pool *pool, int runs, int calls, double *us_median, double *spread) {
    int64_t n = tr_pool_size(pool);
    tr_parallel_for(pool, n, 1, empty_body, NULL);
    double us[MAX_RUNS];
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        for (int c = 0; c < calls; c++) tr_parallel_for(pool, n, 1, empty_body, NULL);
        us[r] = (tr_time_sec() - a) * 1e6 / calls;
    }
    qsort(us, (size_t)runs, sizeof us[0], cmp_double);
    *us_median = us[runs / 2];
    *spread = (us[runs - 1] - us[0]) / us[runs / 2];
}

int main(int argc, char **argv) {
    int runs = 7, threads = 0;
    double run_ms = 150;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ms") == 0 && i + 1 < argc) run_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else { fprintf(stderr, "usage: bench_kernels [--runs N] [--ms M] [--threads T]\n"); return 2; }
    }
    if (runs < 3) runs = 3;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
    if (runs % 2 == 0) runs++;

    char line[512];
    tr_cpu_describe(tr_cpu(), line, sizeof line);
    printf("cpu: %s\n", line);
    tr_kernels_init();
    printf("active tier: %s; %d runs x %.0f ms, median and spread\n\n", tr_kernels_get()->tier, runs, run_ms);

    static const char *tiers[] = {"scalar", "avx2", "avx512", "avx512vnni", "neon", "asm"};
    static const int64_t sizes[] = {64, 1024, 2048, 4096};
    static const struct { const char *name; tr_type type; } kinds[] = {
        {"dot_f32", TR_TYPE_COUNT}, {"dot_row f16", TR_TYPE_F16}, {"dot_row q8_0", TR_TYPE_Q8_0},
    };

    int64_t max_n = 4096;
    float *a = tr_alloc_aligned((size_t)max_n * sizeof(float), 64);
    float *x = tr_alloc_aligned((size_t)max_n * sizeof(float), 64);
    uint8_t *row = tr_alloc_aligned((size_t)max_n * 4, 64);
    if (!a || !x || !row) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int64_t i = 0; i < max_n; i++) { a[i] = frand(); x[i] = frand(); }

    printf("%-12s %-14s %6s %12s %12s %8s\n", "tier", "kernel", "n", "ns/call", "M elem/s", "spread");
    for (size_t t = 0; t < sizeof tiers / sizeof tiers[0]; t++) {
        const tr_kernels *k = tr_kernels_tier(tiers[t]);
        if (k == NULL) continue;
        for (size_t kind = 0; kind < sizeof kinds / sizeof kinds[0]; kind++) {
            tr_type type = kinds[kind].type;
            if (type != TR_TYPE_COUNT && k->dot_row[type] == NULL) continue;
            for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
                int64_t n = sizes[s];
                if (type != TR_TYPE_COUNT) fill_row(type, row, n);
                dot_job j = {k, type, row, a, x, n, 0};
                double ns, spread;
                measure(&j, runs, run_ms, &ns, &spread);
                printf("%-12s %-14s %6lld %12.1f %12.1f %7.1f%%\n", k->tier, kinds[kind].name,
                       (long long)n, ns, (double)n / ns * 1e3, spread * 100.0);
            }
        }
    }

    /* The same row against TR_DOT_TOKENS input rows: elements per second counted per input
     * row, so it compares directly with the dot_row q8_0 line above. */
    float *x4 = tr_alloc_aligned((size_t)max_n * TR_DOT_TOKENS * sizeof(float), 64);
    if (!x4) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int64_t i = 0; i < max_n * TR_DOT_TOKENS; i++) x4[i] = frand();
    for (size_t t = 0; t < sizeof tiers / sizeof tiers[0]; t++) {
        const tr_kernels *k = tr_kernels_tier(tiers[t]);
        if (k == NULL || k->dot_row_x4[TR_TYPE_Q8_0] == NULL) continue;
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            int64_t n = sizes[s];
            fill_row(TR_TYPE_Q8_0, row, n);
            double ns, spread;
            measure_x4(k, row, x4, n, runs, run_ms, &ns, &spread);
            printf("%-12s %-14s %6lld %12.1f %12.1f %7.1f%%\n", k->tier, "dot_row q8_0 x4", (long long)n, ns,
                   (double)n * TR_DOT_TOKENS / ns * 1e3, spread * 100.0);
        }
    }
    tr_free_aligned(x4);

#if defined(TR_BENCH_VNNI)
    /* Question 21: how much faster is int8 x int8 with VNNI than our float activations on the
     * very same row? The dot line assumes the activations are already quantized (a matmul
     * quantizes them once and reuses them for every row); the quantize line says what that
     * one-off costs, per element of one input row. */
    if (tr_cpu()->avx512vnni) {
        int64_t nblk = max_n / TR_Q8_0_BLOCK_ELEMS;
        int8_t *xq = tr_alloc_aligned((size_t)max_n, 64);
        float *xd = tr_alloc_aligned((size_t)nblk * sizeof(float), 64);
        int32_t *wsums = tr_alloc_aligned((size_t)nblk * sizeof(int32_t), 64);
        if (xq == NULL || xd == NULL || wsums == NULL) { fprintf(stderr, "out of memory\n"); return 1; }
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            int64_t n = sizes[s];
            fill_row(TR_TYPE_Q8_0, row, n);
            q8_0_row_sums(row, n, wsums);
            quantize_q8_0(x, n, xq, xd);
            double ns, spread;
            measure_vnni(row, xq, xd, wsums, n, runs, run_ms, &ns, &spread);
            printf("%-12s %-14s %6lld %12.1f %12.1f %7.1f%%\n", "avx512vnni", "dot q8_0xq8_0", (long long)n, ns,
                   (double)n / ns * 1e3, spread * 100.0);
            measure_quant(x, n, xq, xd, runs, run_ms, &ns, &spread);
            printf("%-12s %-14s %6lld %12.1f %12.1f %7.1f%%\n", "scalar", "quantize x q8_0", (long long)n, ns,
                   (double)n / ns * 1e3, spread * 100.0);
        }
        tr_free_aligned(xq);
        tr_free_aligned(xd);
        tr_free_aligned(wsums);
    }

    /* Question 21, second take: int8 with the structure of our best kernel (one row against 4
     * tokens), and the exact alternative (one row against 8), alternated with the engine's x4. */
    const tr_kernels *best = tr_kernels_tier("avx512");
    if (tr_cpu()->avx512vnni && tr_cpu()->avx512bw && tr_cpu()->avx512dq && tr_cpu()->f16c && best != NULL) {
        static const char *const cand_names[CAND_COUNT] = {"float x4 (ours)", "vnni256 x4", "vnni512 x4", "float x8 exact"};
        int64_t nblk = max_n / TR_Q8_0_BLOCK_ELEMS;
        float *x8 = tr_alloc_aligned((size_t)max_n * 8 * sizeof(float), 64);
        int8_t *xq4 = tr_alloc_aligned((size_t)max_n * 4, 64);
        uint8_t *xu4 = tr_alloc_aligned((size_t)max_n * 4, 64);
        float *xd4 = tr_alloc_aligned((size_t)nblk * 4 * sizeof(float), 64);
        float *dwf = tr_alloc_aligned((size_t)nblk * sizeof(float), 64);
        int32_t *wcorr = tr_alloc_aligned((size_t)nblk * sizeof(int32_t), 64);
        if (!x8 || !xq4 || !xu4 || !xd4 || !dwf || !wcorr) { fprintf(stderr, "out of memory\n"); return 1; }
        for (int64_t i = 0; i < max_n * 8; i++) x8[i] = frand();
        printf("\n%-16s %6s %12s %12s %8s %8s\n", "one row, by", "n", "ns/row", "M elem/s", "spread", "vs ours");
        for (size_t s = 1; s < sizeof sizes / sizeof sizes[0]; s++) {
            int64_t n = sizes[s], nb = n / TR_Q8_0_BLOCK_ELEMS;
            fill_row(TR_TYPE_Q8_0, row, n);
            q8_0_row_sums(row, n, wcorr);
            for (int64_t b = 0; b < nb; b++) {
                dwf[b] = tr_q8_0_block_scale(row + TR_Q8_0_BLOCK_BYTES * b);
                wcorr[b] *= -128;
            }
            for (int t = 0; t < 4; t++) quantize_q8_0(x8 + t * n, n, xq4 + t * n, xd4 + t * nb);
            for (int64_t i = 0; i < 4 * n; i++) xu4[i] = (uint8_t)(xq4[i] + 128);
            cand_job j = {best, row, x8, xq4, xu4, xd4, dwf, wcorr, n};

            /* the candidates must compute the dot they are timed for: x8 bit for bit, int8
             * within the rounding of the activations */
            float ref[8], got[8];
            best->dot_row_x4[TR_TYPE_Q8_0](row, x8, n, n, ref);
            dot_row_x8_q8_0_float(row, x8, n, n, got);
            if (memcmp(ref, got, 4 * sizeof(float)) != 0) { fprintf(stderr, "float x8 is not bit-identical to x4\n"); return 1; }
            double want = 0, mag = 0; /* the int8 dot of input row 3 in plain C, and the size of its terms */
            for (int64_t e = 0; e < n; e++) {
                int64_t b = e / TR_Q8_0_BLOCK_ELEMS;
                double term = (double)dwf[b] * (double)(int8_t)row[TR_Q8_0_BLOCK_BYTES * b + TR_Q8_0_SCALE_BYTES + e % TR_Q8_0_BLOCK_ELEMS] *
                              (double)xd4[3 * nb + b] * (double)xq4[3 * n + e];
                want += term;
                mag += term < 0 ? -term : term;
            }
            dot_q8_q8_vnni256_x4(row, xq4, xd4, n, got);
            double off256 = (double)got[3] - want;
            dot_q8_q8_vnni512_x4(row, dwf, wcorr, xu4, xd4, n, got);
            double off512 = (double)got[3] - want;
            if (off256 < 0) off256 = -off256;
            if (off512 < 0) off512 = -off512;
            if (off256 > 1e-4 * mag || off512 > 1e-4 * mag) {
                fprintf(stderr, "an int8 candidate does not compute its dot (off by %g and %g, terms %g)\n", off256, off512, mag);
                return 1;
            }

            double ns[CAND_COUNT], spread[CAND_COUNT];
            measure_candidates(&j, runs, run_ms, ns, spread);
            for (int w = 0; w < CAND_COUNT; w++)
                printf("%-16s %6lld %12.1f %12.1f %7.1f%% %7.2fx\n", cand_names[w], (long long)n, ns[w],
                       (double)n / ns[w] * 1e3, spread[w] * 100.0, ns[CAND_X4_FLOAT] / ns[w]);
        }
        tr_free_aligned(x8); tr_free_aligned(xq4); tr_free_aligned(xu4);
        tr_free_aligned(xd4); tr_free_aligned(dwf); tr_free_aligned(wcorr);
    }
#endif

    /* One whole matrix through tr_matmul with the active tier: 1024 x 2048 q8_0,
     * the shape of an OLMoE expert projection. */
    int64_t rows = 1024, cols = 2048;
    size_t rb = tr_row_bytes(TR_TYPE_Q8_0, cols);
    uint8_t *w = tr_alloc_aligned(rb * (size_t)rows, 64);
    float *y = tr_alloc_aligned((size_t)rows * sizeof(float), 64);
    if (!w || !y) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int64_t r = 0; r < rows; r++) fill_row(TR_TYPE_Q8_0, w + rb * (size_t)r, cols);
    int64_t bench_tokens = 64;
    float *xb = tr_alloc_aligned((size_t)cols * (size_t)bench_tokens * sizeof(float), 64);
    float *yb = tr_alloc_aligned((size_t)rows * (size_t)bench_tokens * sizeof(float), 64);
    if (!xb || !yb) { fprintf(stderr, "out of memory (block matmul)"); return 1; }
    for (int64_t i = 0; i < cols * bench_tokens; i++) xb[i] = frand();
    mat_job m = {{TR_TYPE_Q8_0, rows, cols, w}, x, y};
    mat_job mb = {{TR_TYPE_Q8_0, rows, cols, w}, xb, yb};

    int n_phys = tr_cpu()->physical_cores;
    int counts[8] = {1, 2, 4, 8, 16, 32, 0, 0}, n_counts = 0;
    for (int c = 0; c < 6; c++)
        if (counts[c] <= n_phys) counts[n_counts++] = counts[c];
    if (counts[n_counts - 1] != n_phys) counts[n_counts++] = n_phys;
    if (threads > 0) counts[n_counts++] = threads;
    printf("\n%-8s %13s %7s %18s %7s %25s %7s\n", "threads", "dispatch us", "spread", "1024x2048 1 token",
           "spread", "1024x2048 64 tokens/tok", "spread");
    for (int c = 0; c < n_counts; c++) {
        tr_pool *pool = tr_pool_create(counts[c]);
        double us, ms, msb, s1, s2, s3;
        measure_dispatch(pool, runs, 2000, &us, &s1);
        measure_matmul(pool, &m, runs, 50, 1, &ms, &s2);
        measure_matmul(pool, &mb, runs, 5, bench_tokens, &msb, &s3);
        printf("%-8d %13.3f %6.1f%% %15.3f ms %6.1f%% %22.3f ms %6.1f%%\n", counts[c], us, s1 * 100.0, ms,
               s2 * 100.0, msb / (double)bench_tokens, s3 * 100.0);
        tr_pool_destroy(pool);
    }

    tr_free_aligned(a); tr_free_aligned(x); tr_free_aligned(row); tr_free_aligned(w); tr_free_aligned(y);
    tr_free_aligned(xb); tr_free_aligned(yb);
    return 0;
}
