/* bench_peak.c — how many FLOP/s this CPU does at best without FMA, and how close the prefill's
 * matmul comes to it (docs/MEASUREMENTS.md question 55: the premise of an assembly microkernel).
 *
 * Per thread count (1 and every physical core), the median of N runs and its spread:
 *   peak       independent multiplies and adds on 12 registers, nothing loaded, as zmm (AVX-512)
 *              and as ymm (AVX2): the no-FMA peak at the clock the cores hold under this load;
 *   row2 L1    the active tier's dot_row2_x4 on Q8_0 (the prefill's inner kernel: two rows'
 *              codes widened, converted, scaled, then 8 multiplies and 8 adds into 8
 *              accumulators) over rows and inputs that stay in L1 (512 columns) or in L2 (2048),
 *              and its dot_row2_x8 (the same against 8 input rows, 16 accumulators);
 *   matmul     tr_matmul on OLMoE's shapes: an expert's gate/up (1024 x 2048) and down
 *              (2048 x 1024) with the 64 tokens an expert sees in a 512-token pass, and an
 *              attention projection (2048 x 2048) with 512 tokens; where the tier has
 *              dot_row2_x8, also without it ("-x8": a copy of the table made active), the two
 *              run by run in turn.
 * FLOPs are the matmul's useful ones, 2 per weight and token (the scale multiply not counted).
 *
 * The panel premise (ik_llama.cpp's idea, docs/MEASUREMENTS.md "Two rows against eight tokens"):
 * decode a panel of P weight rows once with dequant_row into f32, then row2_x8_f32 (the same lane
 * contract and mul-then-add as avx512_dot_row2_x8_q8_0, never FMA) over every group of 8 tokens,
 * instead of decoding each weight vector once per 8 tokens as tr_matmul does today. "panel P=16"
 * and "P=32" lines run beside each matmul shape and type; "mix x8 f32" is the same asm stream with
 * the decode replaced by a plain load, the ceiling that arithmetic could reach without one. Every
 * panel shape and type is checked against tr_matmul by memcmp before it is timed; the counter
 * g_panel_kernel_calls (checked > 0 at the end of main) is the branch that check exercises, so it
 * cannot pass without row2_x8_f32 ever having run.
 *
 * Question 64 (more weight rows per input load): the stream with 4 rows x 6 tokens and 3 x 8 beside
 * 2 x 8, then row4_x6 and row3_x8 as kernels in tr_matmul's order against tr_matmul, exact first
 * (tile_lines; its own counter and mutations, -DBENCH_TILE_MUTATE=4 or 3).
 *
 *   build/tests/bench_peak.exe [--runs N] [--one-core] [--q64]   native, still machine (tools/measure_guard.lib)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAVE_X86 1
#endif

#define MAX_RUNS 31

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static volatile float g_one = 1.0f, g_zero = 0.0f;  /* not foldable: the loops really run */
static volatile float g_sink;

#if HAVE_X86
/* The peak loop is inline assembly: twelve named registers the compiler can neither merge nor
 * drop. Written in C first, gcc merged the twelve chains into two (they started from the same
 * values) and the "peak" came out 2x too high (docs/LESSONS.md #161). Six multiply chains and six
 * add chains, each op waiting only on its own chain: 3-cycle latency, 6 chains per pipe pair. The
 * multiplies are grouped, then the adds: on Zen 4 the same twelve ops strictly alternating
 * (mul, add, mul, add...) run at 83% of this (139 against 167 GFLOP/s on one core, LESSONS #162). */
#ifndef BENCH_PEAK_MUTATE
#define PEAK_ASM \
    "vmulps %[c], %[m0], %[m0]\n\tvmulps %[c], %[m1], %[m1]\n\tvmulps %[c], %[m2], %[m2]\n\t" \
    "vmulps %[c], %[m3], %[m3]\n\tvmulps %[c], %[m4], %[m4]\n\tvmulps %[c], %[m5], %[m5]\n\t" \
    "vaddps %[e], %[a0], %[a0]\n\tvaddps %[e], %[a1], %[a1]\n\tvaddps %[e], %[a2], %[a2]\n\t" \
    "vaddps %[e], %[a3], %[a3]\n\tvaddps %[e], %[a4], %[a4]\n\tvaddps %[e], %[a5], %[a5]\n\t"
#else
/* the mutation (-DBENCH_PEAK_MUTATE): the chains made dependent, one of each kind, so the loop
 * waits on latency; the check at the end of main must fail on it (a peak below a real kernel) */
#define PEAK_ASM \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t" \
    "vmulps %[c], %[m0], %[m0]\n\tvaddps %[e], %[a0], %[a0]\n\t"
#endif
#define PEAK_BODY(V, SET1, iters) \
    do { \
        V c = SET1(g_one), e = SET1(g_zero); \
        V m0 = c, m1 = c, m2 = c, m3 = c, m4 = c, m5 = c, a0 = e, a1 = e, a2 = e, a3 = e, a4 = e, a5 = e; \
        for (int64_t i_ = 0; i_ < (iters); i_++) \
            __asm__ volatile(PEAK_ASM \
                             : [m0] "+v"(m0), [m1] "+v"(m1), [m2] "+v"(m2), [m3] "+v"(m3), [m4] "+v"(m4), \
                               [m5] "+v"(m5), [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), \
                               [a4] "+v"(a4), [a5] "+v"(a5) \
                             : [c] "v"(c), [e] "v"(e)); \
        float f_[16]; /* the asm is volatile: nothing to keep alive, one value to show */ \
        V s_ = m0; \
        memcpy(f_, &s_, sizeof s_); \
        g_sink = f_[0]; \
        (void)m1; (void)m2; (void)m3; (void)m4; (void)m5; \
        (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5; \
    } while (0)

__attribute__((target("avx512f"))) static void peak512(int64_t iters) {
    PEAK_BODY(__m512, _mm512_set1_ps, iters);
}
__attribute__((target("avx2"))) static void peak256(int64_t iters) {
    PEAK_BODY(__m256, _mm256_set1_ps, iters);
}

/* The prefill's inner step as an instruction stream, over one block that stays in L1: two rows'
 * 16 codes widened, converted and scaled, then per token a multiply and an add for each row into
 * its own accumulator; tokens two at a time, their four multiplies before their four adds. Not
 * the kernel's bits (the same block every iteration): the ceiling of its arithmetic, with 4
 * tokens as dot_row2_x4 is today and with 8 as an assembly microkernel could hold (16
 * accumulators, 26 of the 32 zmm). */
#define MIX_CODES \
    "vpmovsxbd (%[q]), %[w]\n\tvpmovsxbd 16(%[q]), %[u]\n\tvcvtdq2ps %[w], %[w]\n\t" \
    "vcvtdq2ps %[u], %[u]\n\tvmulps %[d0], %[w], %[w]\n\tvmulps %[d1], %[u], %[u]\n\t"
#define MIX_PAIR(off0, off1, a, b, c, d) \
    "vmovups " off0 "(%[x]), %[v0]\n\tvmovups " off1 "(%[x]), %[v1]\n\t" \
    "vmulps %[v0], %[w], %[t0]\n\tvmulps %[v0], %[u], %[t1]\n\t" \
    "vmulps %[v1], %[w], %[t2]\n\tvmulps %[v1], %[u], %[t3]\n\t" \
    "vaddps %[t0], %[" a "], %[" a "]\n\tvaddps %[t1], %[" b "], %[" b "]\n\t" \
    "vaddps %[t2], %[" c "], %[" c "]\n\tvaddps %[t3], %[" d "], %[" d "]\n\t"
#define MIX_TEMPS [w] "=&v"(w), [u] "=&v"(u), [v0] "=&v"(v0), [v1] "=&v"(v1), [t0] "=&v"(t0), \
                  [t1] "=&v"(t1), [t2] "=&v"(t2), [t3] "=&v"(t3)
#define MIX_IN [q] "r"(q), [x] "r"(x), [d0] "v"(d0), [d1] "v"(d1)

__attribute__((target("avx512f"))) static void mix_x4(int64_t iters, const uint8_t *q, const float *x) {
    __m512 d0 = _mm512_set1_ps(g_one), d1 = _mm512_set1_ps(g_one), z = _mm512_set1_ps(g_zero);
    __m512 a0 = z, a1 = z, a2 = z, a3 = z, b0 = z, b1 = z, b2 = z, b3 = z, w, u, v0, v1, t0, t1, t2, t3;
    for (int64_t i = 0; i < iters; i++)
        __asm__ volatile(MIX_CODES MIX_PAIR("0", "64", "a0", "b0", "a1", "b1") MIX_PAIR("128", "192", "a2", "b2", "a3", "b3")
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [b0] "+v"(b0), [b1] "+v"(b1),
                           [b2] "+v"(b2), [b3] "+v"(b3), MIX_TEMPS
                         : MIX_IN
                         : "memory");
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}

/* For question 59 (FMA in the definition?): the peak and the 8-token stream with fused
 * multiply-adds, one rounding instead of two; not the engine's bits. */
#define FMA_ASM \
    "vfmadd231ps %[c], %[m0], %[a0]\n\tvfmadd231ps %[c], %[m1], %[a1]\n\tvfmadd231ps %[c], %[m2], %[a2]\n\t" \
    "vfmadd231ps %[c], %[m3], %[a3]\n\tvfmadd231ps %[c], %[m4], %[a4]\n\tvfmadd231ps %[c], %[m5], %[a5]\n\t"
__attribute__((target("avx512f"))) static void peak_fma(int64_t iters) {
    __m512 c = _mm512_set1_ps(g_one), e = _mm512_set1_ps(g_zero);
    __m512 m0 = c, m1 = c, m2 = c, m3 = c, m4 = c, m5 = c, a0 = e, a1 = e, a2 = e, a3 = e, a4 = e, a5 = e;
    for (int64_t i = 0; i < iters; i++)
        __asm__ volatile(FMA_ASM FMA_ASM
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [a4] "+v"(a4), [a5] "+v"(a5)
                         : [c] "v"(c), [m0] "v"(m0), [m1] "v"(m1), [m2] "v"(m2), [m3] "v"(m3), [m4] "v"(m4),
                           [m5] "v"(m5));
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}
#define FMA_PAIR(off0, off1, a, b, c, d) \
    "vmovups " off0 "(%[x]), %[v0]\n\tvmovups " off1 "(%[x]), %[v1]\n\t" \
    "vfmadd231ps %[v0], %[w], %[" a "]\n\tvfmadd231ps %[v0], %[u], %[" b "]\n\t" \
    "vfmadd231ps %[v1], %[w], %[" c "]\n\tvfmadd231ps %[v1], %[u], %[" d "]\n\t"
__attribute__((target("avx512f"))) static void mix_x8_fma(int64_t iters, const uint8_t *q, const float *x) {
    __m512 d0 = _mm512_set1_ps(g_one), d1 = _mm512_set1_ps(g_one), z = _mm512_set1_ps(g_zero);
    __m512 a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z;
    __m512 b0 = z, b1 = z, b2 = z, b3 = z, b4 = z, b5 = z, b6 = z, b7 = z, w, u, v0, v1, t0, t1, t2, t3;
    for (int64_t i = 0; i < iters; i++) {
        __asm__ volatile(MIX_CODES FMA_PAIR("0", "64", "a0", "b0", "a1", "b1") FMA_PAIR("128", "192", "a2", "b2", "a3", "b3")
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [b0] "+v"(b0), [b1] "+v"(b1),
                           [b2] "+v"(b2), [b3] "+v"(b3), MIX_TEMPS
                         : MIX_IN
                         : "memory");
        __asm__ volatile(FMA_PAIR("256", "320", "a4", "b4", "a5", "b5") FMA_PAIR("384", "448", "a6", "b6", "a7", "b7")
                         : [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7), [b4] "+v"(b4), [b5] "+v"(b5),
                           [b6] "+v"(b6), [b7] "+v"(b7), [v0] "=&v"(v0), [v1] "=&v"(v1)
                         : [x] "r"(x), [w] "v"(w), [u] "v"(u)
                         : "memory");
    }
    (void)t0; (void)t1; (void)t2; (void)t3;
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}

__attribute__((target("avx512f"))) static void mix_x8(int64_t iters, const uint8_t *q, const float *x) {
    __m512 d0 = _mm512_set1_ps(g_one), d1 = _mm512_set1_ps(g_one), z = _mm512_set1_ps(g_zero);
    __m512 a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z;
    __m512 b0 = z, b1 = z, b2 = z, b3 = z, b4 = z, b5 = z, b6 = z, b7 = z, w, u, v0, v1, t0, t1, t2, t3;
    /* two statements an iteration: an asm takes at most 30 operands, and "+v" counts twice */
    for (int64_t i = 0; i < iters; i++) {
        __asm__ volatile(MIX_CODES MIX_PAIR("0", "64", "a0", "b0", "a1", "b1") MIX_PAIR("128", "192", "a2", "b2", "a3", "b3")
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [b0] "+v"(b0), [b1] "+v"(b1),
                           [b2] "+v"(b2), [b3] "+v"(b3), MIX_TEMPS
                         : MIX_IN
                         : "memory");
        __asm__ volatile(MIX_PAIR("256", "320", "a4", "b4", "a5", "b5") MIX_PAIR("384", "448", "a6", "b6", "a7", "b7")
                         : [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7), [b4] "+v"(b4), [b5] "+v"(b5),
                           [b6] "+v"(b6), [b7] "+v"(b7), [v0] "=&v"(v0), [v1] "=&v"(v1), [t0] "=&v"(t0),
                           [t1] "=&v"(t1), [t2] "=&v"(t2), [t3] "=&v"(t3)
                         : [x] "r"(x), [w] "v"(w), [u] "v"(u)
                         : "memory");
    }
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}

/* The same stream with the decode removed: two plain loads of already-decoded f32 weights (w, u)
 * from L1 instead of MIX_CODES's widen-convert-scale. The arithmetic after the loads is unchanged
 * (the same MIX_PAIR), so this is the ceiling the panel premise below could reach if decoding a
 * row cost nothing at all. */
#define MIX_CODES_F32 "vmovups (%[q]), %[w]\n\tvmovups 64(%[q]), %[u]\n\t"
#define MIX_IN_F32 [q] "r"(q), [x] "r"(x)

__attribute__((target("avx512f"))) static void mix_x8_f32(int64_t iters, const uint8_t *q, const float *x) {
    __m512 z = _mm512_set1_ps(g_zero);
    __m512 a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z;
    __m512 b0 = z, b1 = z, b2 = z, b3 = z, b4 = z, b5 = z, b6 = z, b7 = z, w, u, v0, v1, t0, t1, t2, t3;
    for (int64_t i = 0; i < iters; i++) {
        __asm__ volatile(MIX_CODES_F32 MIX_PAIR("0", "64", "a0", "b0", "a1", "b1") MIX_PAIR("128", "192", "a2", "b2", "a3", "b3")
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3), [b0] "+v"(b0), [b1] "+v"(b1),
                           [b2] "+v"(b2), [b3] "+v"(b3), MIX_TEMPS
                         : MIX_IN_F32
                         : "memory");
        __asm__ volatile(MIX_PAIR("256", "320", "a4", "b4", "a5", "b5") MIX_PAIR("384", "448", "a6", "b6", "a7", "b7")
                         : [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7), [b4] "+v"(b4), [b5] "+v"(b5),
                           [b6] "+v"(b6), [b7] "+v"(b7), [v0] "=&v"(v0), [v1] "=&v"(v1), [t0] "=&v"(t0),
                           [t1] "=&v"(t1), [t2] "=&v"(t2), [t3] "=&v"(t3)
                         : [x] "r"(x), [w] "v"(w), [u] "v"(u)
                         : "memory");
    }
    float f[16];
    memcpy(f, &a0, sizeof f);
    g_sink = f[0];
}

/* Question 64: more weight rows per input load, as streams over one block in L1 like mix_x8.
 * Each is one asm statement with its loop inside and every register named: split into several
 * statements (an asm takes at most 30 operands), gcc moved accumulators between registers from
 * one statement to the next. The decode is mix_x8's with the scale a broadcast from memory (four
 * scales in registers would not fit), so 2 x 8 runs again with it: only the tile differs.
 *   4 x 6: zmm0-23 accumulators (row r, token t in 6r + t), 24-27 weights, 28 input, 29-30 products
 *   3 x 8: zmm0-23 accumulators (8r + t), 24-26 weights, 27 input, 28-30 products
 *   2 x 8: zmm0-15 accumulators (8r + t), 16-17 weights, 18-19 inputs, 20-23 products
 * Per token one input load, then the products two or three at a time and their adds. */
#define MZ(n) "%%zmm" #n
#define MIX_DEC(w, off, doff) \
    "vpmovsxbd " #off "(%[q]), " MZ(w) "\n\tvcvtdq2ps " MZ(w) ", " MZ(w) "\n\t" \
    "vmulps " #doff "(%[d])%{1to16%}, " MZ(w) ", " MZ(w) "\n\t"
#define MIX_M(v, w, t) "vmulps " MZ(v) ", " MZ(w) ", " MZ(t) "\n\t"
#define MIX_A(t, a) "vaddps " MZ(t) ", " MZ(a) ", " MZ(a) "\n\t"
#define MIX_LD(off, v) "vmovups " #off "(%[x]), " MZ(v) "\n\t"
/* the accumulators are named by number: the preprocessor stringifies, it does not add */
#define MIX4_TOK(off, a, b, c, e) \
    MIX_LD(off, 28) MIX_M(28, 24, 29) MIX_M(28, 25, 30) MIX_A(29, a) MIX_A(30, b) \
    MIX_M(28, 26, 29) MIX_M(28, 27, 30) MIX_A(29, c) MIX_A(30, e)
#define MIX3_TOK(off, a, b, c) \
    MIX_LD(off, 27) MIX_M(27, 24, 28) MIX_M(27, 25, 29) MIX_M(27, 26, 30) MIX_A(28, a) MIX_A(29, b) MIX_A(30, c)
#define MIX2_TOK2(off0, off1, a0, b0, a1, b1) \
    MIX_LD(off0, 18) MIX_LD(off1, 19) MIX_M(18, 16, 20) MIX_M(18, 17, 21) MIX_M(19, 16, 22) MIX_M(19, 17, 23) \
    MIX_A(20, a0) MIX_A(21, b0) MIX_A(22, a1) MIX_A(23, b1)
#define MIX_ZERO4(a, b, c, e) \
    "vpxord " MZ(a) ", " MZ(a) ", " MZ(a) "\n\tvpxord " MZ(b) ", " MZ(b) ", " MZ(b) "\n\t" \
    "vpxord " MZ(c) ", " MZ(c) ", " MZ(c) "\n\tvpxord " MZ(e) ", " MZ(e) ", " MZ(e) "\n\t"
#define MIX_ZERO24 \
    MIX_ZERO4(0, 1, 2, 3) MIX_ZERO4(4, 5, 6, 7) MIX_ZERO4(8, 9, 10, 11) MIX_ZERO4(12, 13, 14, 15) \
    MIX_ZERO4(16, 17, 18, 19) MIX_ZERO4(20, 21, 22, 23)
#define MIX_LOOP(body) \
    float d[4] = {g_one, g_one, g_one, g_one}, f[16]; \
    if (iters < 1) return; \
    __asm__ volatile(MIX_ZERO24 "1:\n\t" body "dec %[n]\n\tjnz 1b\n\tvmovups " MZ(0) ", (%[f])\n\t" \
                     : [n] "+r"(iters) \
                     : [q] "r"(q), [x] "r"(x), [d] "r"(d), [f] "r"(f) \
                     : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", \
                       "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", \
                       "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", \
                       "memory", "cc"); \
    g_sink = f[0]

__attribute__((target("avx512f"))) static void mix_4x6(int64_t iters, const uint8_t *q, const float *x) {
    MIX_LOOP(MIX_DEC(24, 0, 0) MIX_DEC(25, 16, 4) MIX_DEC(26, 32, 8) MIX_DEC(27, 48, 12)
             MIX4_TOK(0, 0, 6, 12, 18) MIX4_TOK(64, 1, 7, 13, 19) MIX4_TOK(128, 2, 8, 14, 20)
             MIX4_TOK(192, 3, 9, 15, 21) MIX4_TOK(256, 4, 10, 16, 22) MIX4_TOK(320, 5, 11, 17, 23));
}
__attribute__((target("avx512f"))) static void mix_3x8(int64_t iters, const uint8_t *q, const float *x) {
    MIX_LOOP(MIX_DEC(24, 0, 0) MIX_DEC(25, 16, 4) MIX_DEC(26, 32, 8)
             MIX3_TOK(0, 0, 8, 16) MIX3_TOK(64, 1, 9, 17) MIX3_TOK(128, 2, 10, 18) MIX3_TOK(192, 3, 11, 19)
             MIX3_TOK(256, 4, 12, 20) MIX3_TOK(320, 5, 13, 21) MIX3_TOK(384, 6, 14, 22) MIX3_TOK(448, 7, 15, 23));
}
__attribute__((target("avx512f"))) static void mix_2x8m(int64_t iters, const uint8_t *q, const float *x) {
    MIX_LOOP(MIX_DEC(16, 0, 0) MIX_DEC(17, 16, 4)
             MIX2_TOK2(0, 64, 0, 8, 1, 9) MIX2_TOK2(128, 192, 2, 10, 3, 11) MIX2_TOK2(256, 320, 4, 12, 5, 13)
             MIX2_TOK2(384, 448, 6, 14, 7, 15));
}
#endif

typedef struct {
    int kind;          /* 0: peak512, 1: peak256, 2: row2, 3: mix_x4, 4: mix_x8, 5: peak_fma, 6: mix_x8_fma,
                        * 7: row2_x8, 8: mix_x8_f32, 9: mix_4x6, 10: mix_3x8, 11: mix_2x8m */
    int64_t iters;     /* loop iterations (peak) or calls (row2) per thread */
    int64_t cols;
    uint8_t *rows[64]; /* per worker: two Q8_0 rows */
    float *x[64];      /* per worker: 8 input rows */
} peak_job;

static void peak_fn(void *ctx, int64_t begin, int64_t end, int worker) {
    peak_job *j = (peak_job *)ctx;
    for (int64_t t = begin; t < end; t++) {
#if HAVE_X86
        if (j->kind == 0) peak512(j->iters);
        if (j->kind == 1) peak256(j->iters);
        if (j->kind == 3) mix_x4(j->iters, j->rows[worker], j->x[worker]);
        if (j->kind == 4) mix_x8(j->iters, j->rows[worker], j->x[worker]);
        if (j->kind == 5) peak_fma(j->iters);
        if (j->kind == 6) mix_x8_fma(j->iters, j->rows[worker], j->x[worker]);
        if (j->kind == 8) mix_x8_f32(j->iters, j->rows[worker], j->x[worker]);
        if (j->kind == 9) mix_4x6(j->iters, j->rows[worker], j->x[worker]);
        if (j->kind == 10) mix_3x8(j->iters, j->rows[worker], j->x[worker]);
        if (j->kind == 11) mix_2x8m(j->iters, j->rows[worker], j->x[worker]);
#endif
        if (j->kind == 2) {
            const tr_kernels *k = tr_kernels_get();
            size_t rb = tr_row_bytes(TR_TYPE_Q8_0, j->cols);
            float out[8], acc = 0.0f;
            for (int64_t c = 0; c < j->iters; c++) {
                k->dot_row2_x4[TR_TYPE_Q8_0](j->rows[worker], j->rows[worker] + rb, j->x[worker], j->cols, j->cols,
                                             out);
                acc += out[0];
            }
            g_sink = acc;
        }
        if (j->kind == 7) {
            const tr_kernels *k = tr_kernels_get();
            size_t rb = tr_row_bytes(TR_TYPE_Q8_0, j->cols);
            float out[16], acc = 0.0f;
            for (int64_t c = 0; c < j->iters; c++) {
                k->dot_row2_x8[TR_TYPE_Q8_0](j->rows[worker], j->rows[worker] + rb, j->x[worker], j->cols, j->cols,
                                             out);
                acc += out[0];
            }
            g_sink = acc;
        }
    }
}

/* GFLOP/s of n_threads running the job, median of runs; flop is per thread */
static double run_job(tr_pool *pool, int n_threads, peak_job *j, double flop, int runs, double *spread) {
    double g[MAX_RUNS];
    tr_parallel_for(pool, n_threads, 1, peak_fn, j); /* warm-up */
    for (int r = 0; r < runs; r++) {
        double a = tr_time_sec();
        tr_parallel_for(pool, n_threads, 1, peak_fn, j);
        g[r] = flop * n_threads / (tr_time_sec() - a) / 1e9;
    }
    qsort(g, (size_t)runs, sizeof g[0], cmp_double);
    *spread = (g[runs - 1] - g[0]) / g[runs / 2];
    return g[runs / 2];
}

static float frand_state(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((*s >> 40) & 0xFFFF) / 32768.0f - 1.0f;
}

static void fill_q8_0(uint8_t *row, int64_t n, uint64_t *s) {
    for (int64_t b = 0; b < n / 32; b++) {
        uint16_t d = (uint16_t)(0x2000 | (uint16_t)((*s >> 30) & 0x3FF));
        frand_state(s);
        memcpy(row + 34 * b, &d, 2);
        for (int i = 0; i < 32; i++) row[34 * b + 2 + i] = (uint8_t)(int8_t)(frand_state(s) * 127.0f);
    }
}

#if HAVE_X86
/* Q4_K row filler, copied from tests/test_kernels.c (next_rand, fill_q4_k): every byte random
 * (every nibble, scale and min). special=0 always keeps d and dmin ordinary (never a subnormal,
 * inf or NaN scale): kernels.h's contract only promises the same NaN on every variant, not the
 * same payload inside it, so a NaN block could make an exact panel/tr_matmul memcmp fail for a
 * reason that has nothing to do with this bench's premise. */
static unsigned next_rand(unsigned *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return *seed;
}
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

/* ---- the panel premise (ik_llama.cpp's idea, docs/MEASUREMENTS.md "Two rows against eight
 * tokens"): decode a panel of P weight rows once with dequant_row, then run a pure-f32 two-row
 * eight-token kernel over every group of 8 tokens, instead of decoding each weight vector once per
 * 8 tokens as avx512_dot_row2_x8_q8_0 does today. avx512_pair_sums and the tree it builds
 * (BP_ROW2X8_OUT/BP_ROW2_PAIR/BP_ROW2X8_STEP) are copied verbatim from src/kernels/kernels_x86.c's
 * avx512_pair_sums/TR_ROW2X8_OUT/TR_ROW2_PAIR/TR_ROW2X8_STEP: same lane contract (element e of
 * token t into lane e%16, increasing e), same mul-then-add (never FMA outside BENCH_PANEL_MUTATE),
 * same pairwise tree, so row2_x8_f32 fed dequant_row's output must be bit-identical to tr_matmul
 * (kernels.h: dot_row(row,x) is dot_f32(dequant_row(row),x) bit for bit; dot_row2_x8's own doc:
 * "each sum is still its own dot_row, bit for bit"). */
__attribute__((target("avx512f")))
static inline __m512 avx512_pair_sums(__m512 a, __m512 b) {
    const __m512i even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    return _mm512_add_ps(_mm512_permutex2var_ps(a, even, b), _mm512_permutex2var_ps(a, odd, b));
}

#define BP_ROW2X8_OUT(out) \
    do { \
        __m512 s0_ = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(a0, a1), avx512_pair_sums(a2, a3)), \
                                      avx512_pair_sums(avx512_pair_sums(a4, a5), avx512_pair_sums(a6, a7))); \
        __m512 s1_ = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(b0, b1), avx512_pair_sums(b2, b3)), \
                                      avx512_pair_sums(avx512_pair_sums(b4, b5), avx512_pair_sums(b6, b7))); \
        _mm512_storeu_ps(out, avx512_pair_sums(s0_, s1_)); \
    } while (0)

#ifndef BENCH_PANEL_MUTATE
#define BP_ROW2_PAIR(xr, e, w, u, A, B) \
    do { \
        __m512 v_ = _mm512_loadu_ps((xr) + (e)); \
        A = _mm512_add_ps(A, _mm512_mul_ps(w, v_)); \
        B = _mm512_add_ps(B, _mm512_mul_ps(u, v_)); \
    } while (0)
#else
/* the mutation (-DBENCH_PANEL_MUTATE): FMA instead of separate mul and add, one rounding instead
 * of two; the exactness check below must fail on it against tr_matmul's mul-then-add Q8_0/Q4_K */
#define BP_ROW2_PAIR(xr, e, w, u, A, B) \
    do { \
        __m512 v_ = _mm512_loadu_ps((xr) + (e)); \
        A = _mm512_fmadd_ps(w, v_, A); \
        B = _mm512_fmadd_ps(u, v_, B); \
    } while (0)
#endif

#define BP_ROW2X8_STEP(e, w, u) \
    do { \
        __m512 w_ = (w), u_ = (u); \
        BP_ROW2_PAIR(x, e, w_, u_, a0, b0); \
        BP_ROW2_PAIR(x1, e, w_, u_, a1, b1); \
        BP_ROW2_PAIR(x2, e, w_, u_, a2, b2); \
        BP_ROW2_PAIR(x3, e, w_, u_, a3, b3); \
        BP_ROW2_PAIR(x4, e, w_, u_, a4, b4); \
        BP_ROW2_PAIR(x5, e, w_, u_, a5, b5); \
        BP_ROW2_PAIR(x6, e, w_, u_, a6, b6); \
        BP_ROW2_PAIR(x7, e, w_, u_, a7, b7); \
    } while (0)

/* row2_x8_f32 calls from the panel path below: covers the branch the exactness check exercises
 * (checked > 0 at the end of main), so that check cannot pass without ever running it. */
static int64_t g_panel_kernel_calls = 0;

/* Two already-decoded f32 weight rows against eight input rows, sixteen accumulators: the same
 * contract as avx512_dot_row2_x8_q8_0, minus the decode. out[0..7]: row0's dot with tokens 0..7;
 * out[8..15]: row1's, the same layout as dot_row2_x8. */
__attribute__((target("avx512f")))
static void row2_x8_f32(const float *w0, const float *w1, const float *x, int64_t stride, int64_t n, float *out) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps(), a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps(), b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
    __m512 b4 = _mm512_setzero_ps(), b5 = _mm512_setzero_ps(), b6 = _mm512_setzero_ps(), b7 = _mm512_setzero_ps();
    const float *x1 = x + stride, *x2 = x1 + stride, *x3 = x2 + stride, *x4 = x3 + stride, *x5 = x4 + stride,
                *x6 = x5 + stride, *x7 = x6 + stride;
    for (int64_t e = 0; e < n; e += TR_LANES)
        BP_ROW2X8_STEP(e, _mm512_loadu_ps(w0 + e), _mm512_loadu_ps(w1 + e));
    BP_ROW2X8_OUT(out);
    g_panel_kernel_calls++;
}

/* One panel job: dequant_row's P rows [r0, r0+P) into this worker's scratch (panel[worker],
 * allocated before timing), then row2_x8_f32 over every group of 8 tokens and row pair.
 * y[t * rows + r], the same layout as tr_matmul. n_tokens is always a multiple of 8 and rows a
 * multiple of P in this bench's shapes (checked once by the caller before timing starts, never on
 * this hot loop). */
typedef struct {
    tr_type type;
    int64_t rows, cols, n_tokens, P;
    const void *w;
    const float *x;
    float *y;
    float *panel[64]; /* per worker, P * cols floats */
} panel_job;

/* The panel's decode in AVX-512, the floats dequant_row gives (Q8_0 scale * q, one rounding; Q4_K
 * scale * q - min, the product exact and the difference rounded once: docs/LESSONS.md #169). The
 * first run decoded with the scalar dequant_row, one element at a time against x8's sixteen: at 64
 * tokens that decode cost about as much as the panel's arithmetic, which is not the premise. */
__attribute__((target("avx512f")))
static void panel_dequant_q8_0(const unsigned char *row, float *out, int64_t n) {
    for (int64_t b = 0; b < n / TR_Q8_0_BLOCK_ELEMS; b++) {
        const unsigned char *blk = row + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        __m512 d = _mm512_set1_ps(tr_q8_0_block_scale(blk));
        for (int e = 0; e < TR_Q8_0_BLOCK_ELEMS; e += TR_LANES) {
            __m128i q = _mm_loadu_si128((const __m128i *)(const void *)(blk + TR_Q8_0_SCALE_BYTES + e));
            _mm512_storeu_ps(out + b * TR_Q8_0_BLOCK_ELEMS + e,
                             _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q))));
        }
    }
}
__attribute__((target("avx512f")))
static void panel_dequant_q4_k(const unsigned char *row, float *out, int64_t n) {
    const __m512i low = _mm512_set1_epi32(0x0F);
    for (int64_t b = 0; b < n / TR_Q4_K_BLOCK_ELEMS; b++) {
        const unsigned char *blk = row + (size_t)b * TR_Q4_K_BLOCK_BYTES, *qs = blk + TR_Q4_K_QS_OFFSET;
        float scale[8], min[8];
        tr_q4_k_scales(blk, scale, min);
        float *o = out + b * TR_Q4_K_BLOCK_ELEMS;
        for (int c = 0; c < 4; c++)
            for (int h = 0; h < 2; h++) {
                __m512i q = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c + 16 * h)));
                __m512 s0 = _mm512_set1_ps(scale[2 * c]), m0 = _mm512_set1_ps(min[2 * c]);
                __m512 s1 = _mm512_set1_ps(scale[2 * c + 1]), m1 = _mm512_set1_ps(min[2 * c + 1]);
                __m512 lo = _mm512_cvtepi32_ps(_mm512_and_si512(q, low));
                __m512 hi = _mm512_cvtepi32_ps(_mm512_srli_epi32(q, 4));
                _mm512_storeu_ps(o + 64 * c + 16 * h, _mm512_sub_ps(_mm512_mul_ps(s0, lo), m0));
                _mm512_storeu_ps(o + 64 * c + 32 + 16 * h, _mm512_sub_ps(_mm512_mul_ps(s1, hi), m1));
            }
    }
}

static void panel_fn(void *ctx, int64_t begin, int64_t end, int worker) {
    panel_job *j = (panel_job *)ctx;
    const unsigned char *base = (const unsigned char *)j->w;
    size_t rb = tr_row_bytes(j->type, j->cols);
    float *buf = j->panel[worker];
    for (int64_t p = begin; p < end; p++) {
        int64_t r0 = p * j->P;
        for (int64_t i = 0; i < j->P; i++) {
            const unsigned char *r = base + (size_t)(r0 + i) * rb;
            if (j->type == TR_TYPE_Q4_K) panel_dequant_q4_k(r, buf + i * j->cols, j->cols);
            else panel_dequant_q8_0(r, buf + i * j->cols, j->cols);
        }
        for (int64_t t = 0; t < j->n_tokens; t += 8) {
            const float *xt = j->x + t * j->cols;
            for (int64_t rp = 0; rp < j->P; rp += 2) {
                float out[16];
                row2_x8_f32(buf + rp * j->cols, buf + (rp + 1) * j->cols, xt, j->cols, j->cols, out);
                for (int e = 0; e < 8; e++) {
                    j->y[(t + e) * j->rows + r0 + rp] = out[e];
                    j->y[(t + e) * j->rows + r0 + rp + 1] = out[8 + e];
                }
            }
        }
    }
}

/* ---- question 64 as kernels (docs/MEASUREMENTS.md §More weight rows per input load):
 * row4_x6 runs four weight rows against six input rows (24 accumulators), row3_x8 three against
 * eight. The decode, the lane contract (element e into lane e % 16, increasing e), mul-then-add
 * and the tree are avx512_dot_row2_x8_q8_0's and _q4_k's (src/kernels/kernels_x86.c), so each
 * output must be its own dot_row bit for bit. tile_fn runs them in tr_matmul's order (a block of
 * tokens, every row group against it; the tokens past the last whole group through the tier's
 * dot_row2_x4 and dot_row) and is checked against tr_matmul by memcmp before it is timed.
 * g_tile_kernel_calls (checked > 0 at the end of main) counts the branch that check exercises;
 * -DBENCH_TILE_MUTATE=4 (FMA for mul-then-add in row4_x6) and =3 (in row3_x8) must each make that
 * check fail. */
static int64_t g_tile_kernel_calls = 0;

#define BP_MA(A, w, v) A = _mm512_add_ps(A, _mm512_mul_ps(w, v))
#define BP_FMA(A, w, v) A = _mm512_fmadd_ps(w, v, A)
#if defined(BENCH_TILE_MUTATE) && BENCH_TILE_MUTATE == 4
#define BP_MA4 BP_FMA
#else
#define BP_MA4 BP_MA
#endif
#if defined(BENCH_TILE_MUTATE) && BENCH_TILE_MUTATE == 3
#define BP_MA3 BP_FMA
#else
#define BP_MA3 BP_MA
#endif

/* sixteen accumulators to their sixteen sums in leaf order, and eight to eight (kernels_x86.c's
 * TR_ROW2X8_OUT and TR_ROW2_OUT: each sum is tr_lane_combine's tree of its own accumulator) */
#define BP_TREE16(out, A0, A1, A2, A3, A4, A5, A6, A7, B0, B1, B2, B3, B4, B5, B6, B7) \
    _mm512_storeu_ps(out, avx512_pair_sums( \
        avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(A0, A1), avx512_pair_sums(A2, A3)), \
                         avx512_pair_sums(avx512_pair_sums(A4, A5), avx512_pair_sums(A6, A7))), \
        avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(B0, B1), avx512_pair_sums(B2, B3)), \
                         avx512_pair_sums(avx512_pair_sums(B4, B5), avx512_pair_sums(B6, B7)))))
#define BP_TREE8(out, A0, A1, A2, A3, B0, B1, B2, B3) \
    do { \
        __m512 h_ = avx512_pair_sums(avx512_pair_sums(avx512_pair_sums(A0, A1), avx512_pair_sums(A2, A3)), \
                                     avx512_pair_sums(avx512_pair_sums(B0, B1), avx512_pair_sums(B2, B3))); \
        _mm256_storeu_ps(out, _mm512_castps512_ps256(avx512_pair_sums(h_, h_))); \
    } while (0)

/* 4 x 6: accumulators a (row 0), b, c, h (row 3), one per token; out[6 * row + token] */
#define BP_TOK4(xt, e, A, B, C, H) \
    do { \
        __m512 v_ = _mm512_loadu_ps((xt) + (e)); \
        BP_MA4(A, w0, v_); BP_MA4(B, w1, v_); BP_MA4(C, w2, v_); BP_MA4(H, w3, v_); \
    } while (0)
#define BP_STEP4X6(e) \
    do { \
        BP_TOK4(x, e, a0, b0, c0, h0); BP_TOK4(x1, e, a1, b1, c1, h1); BP_TOK4(x2, e, a2, b2, c2, h2); \
        BP_TOK4(x3, e, a3, b3, c3, h3); BP_TOK4(x4, e, a4, b4, c4, h4); BP_TOK4(x5, e, a5, b5, c5, h5); \
    } while (0)
#define BP_BEGIN4X6 \
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0; \
    __m512 b0 = a0, b1 = a0, b2 = a0, b3 = a0, b4 = a0, b5 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0, c4 = a0, c5 = a0; \
    __m512 h0 = a0, h1 = a0, h2 = a0, h3 = a0, h4 = a0, h5 = a0; \
    const float *x1 = x + stride, *x2 = x1 + stride, *x3 = x2 + stride, *x4 = x3 + stride, *x5 = x4 + stride
#define BP_OUT4X6(out) \
    do { \
        BP_TREE16(out, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3); \
        BP_TREE8((out) + 16, c4, c5, h0, h1, h2, h3, h4, h5); \
    } while (0)

/* 3 x 8: accumulators a, b, c; out[8 * row + token] */
#define BP_TOK3(xt, e, A, B, C) \
    do { \
        __m512 v_ = _mm512_loadu_ps((xt) + (e)); \
        BP_MA3(A, w0, v_); BP_MA3(B, w1, v_); BP_MA3(C, w2, v_); \
    } while (0)
#define BP_STEP3X8(e) \
    do { \
        BP_TOK3(x, e, a0, b0, c0); BP_TOK3(x1, e, a1, b1, c1); BP_TOK3(x2, e, a2, b2, c2); BP_TOK3(x3, e, a3, b3, c3); \
        BP_TOK3(x4, e, a4, b4, c4); BP_TOK3(x5, e, a5, b5, c5); BP_TOK3(x6, e, a6, b6, c6); BP_TOK3(x7, e, a7, b7, c7); \
    } while (0)
#define BP_BEGIN3X8 \
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0; \
    __m512 b0 = a0, b1 = a0, b2 = a0, b3 = a0, b4 = a0, b5 = a0, b6 = a0, b7 = a0; \
    __m512 c0 = a0, c1 = a0, c2 = a0, c3 = a0, c4 = a0, c5 = a0, c6 = a0, c7 = a0; \
    const float *x1 = x + stride, *x2 = x1 + stride, *x3 = x2 + stride, *x4 = x3 + stride, *x5 = x4 + stride, \
                *x6 = x5 + stride, *x7 = x6 + stride
#define BP_OUT3X8(out) \
    do { \
        BP_TREE16(out, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7); \
        BP_TREE8((out) + 16, c0, c1, c2, c3, c4, c5, c6, c7); \
    } while (0)

__attribute__((target("avx512f")))
static inline __m512 bp_i8_to_ps(const unsigned char *p) {
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)p)));
}
/* Q8_0: the 16 weights at j of block o of the row at p, scale * code (dot_row's one rounding) */
#define BP_Q8_W(p, o, j) \
    _mm512_mul_ps(_mm512_set1_ps(tr_q8_0_block_scale((p) + (o))), bp_i8_to_ps((p) + (o) + TR_Q8_0_SCALE_BYTES + (j)))

__attribute__((target("avx512f")))
static void row4_x6_q8_0(const unsigned char *row, size_t rb, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p0 = row, *p1 = row + rb, *p2 = row + 2 * rb, *p3 = row + 3 * rb;
    BP_BEGIN4X6;
    for (int64_t b = 0; b < n / TR_Q8_0_BLOCK_ELEMS; b++) {
        size_t o = (size_t)b * TR_Q8_0_BLOCK_BYTES;
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m512 w0 = BP_Q8_W(p0, o, j), w1 = BP_Q8_W(p1, o, j), w2 = BP_Q8_W(p2, o, j), w3 = BP_Q8_W(p3, o, j);
            BP_STEP4X6(b * TR_Q8_0_BLOCK_ELEMS + j);
        }
    }
    BP_OUT4X6(out);
    g_tile_kernel_calls++;
}

__attribute__((target("avx512f")))
static void row3_x8_q8_0(const unsigned char *row, size_t rb, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p0 = row, *p1 = row + rb, *p2 = row + 2 * rb;
    BP_BEGIN3X8;
    for (int64_t b = 0; b < n / TR_Q8_0_BLOCK_ELEMS; b++) {
        size_t o = (size_t)b * TR_Q8_0_BLOCK_BYTES;
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j += TR_LANES) {
            __m512 w0 = BP_Q8_W(p0, o, j), w1 = BP_Q8_W(p1, o, j), w2 = BP_Q8_W(p2, o, j);
            BP_STEP3X8(b * TR_Q8_0_BLOCK_ELEMS + j);
        }
    }
    BP_OUT3X8(out);
    g_tile_kernel_calls++;
}

/* Q4_K: kernels_x86.c's avx512_q4_k_values, the sub-block's 16 values scale * q - min, each quant
 * picking its own with vpermps (low nibble as it is, high nibble shifted) */
__attribute__((target("avx512f")))
static inline __m512 bp_q4_k_values(float scale, float min) {
    const __m512 q = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f);
    return _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(scale), q), _mm512_set1_ps(min));
}
#define BP_Q4_IDX(qs, c, h) _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)((qs) + 32 * (c) + 16 * (h))))

/* one Q4_K block's four sub-block pairs, for R rows (1 <= R <= 4): the tables per pair, then
 * the four steps of 16, as avx512_dot_row2_x8_q4_k; STEP is the tile's step macro */
#define BP_Q4_BLOCK(R, STEP) \
    do { \
        float s_[4][8], m_[4][8]; \
        const unsigned char *qs_[4]; \
        for (int r_ = 0; r_ < (R); r_++) { \
            const unsigned char *k_ = row + (size_t)r_ * rb + (size_t)b * TR_Q4_K_BLOCK_BYTES; \
            tr_q4_k_scales(k_, s_[r_], m_[r_]); \
            qs_[r_] = k_ + TR_Q4_K_QS_OFFSET; \
        } \
        for (int c = 0; c < 4; c++) { \
            __m512 lo_[4], hi_[4]; \
            __m512i i0_[4], i1_[4]; \
            for (int r_ = 0; r_ < (R); r_++) { \
                lo_[r_] = bp_q4_k_values(s_[r_][2 * c], m_[r_][2 * c]); \
                hi_[r_] = bp_q4_k_values(s_[r_][2 * c + 1], m_[r_][2 * c + 1]); \
                i0_[r_] = BP_Q4_IDX(qs_[r_], c, 0); \
                i1_[r_] = BP_Q4_IDX(qs_[r_], c, 1); \
            } \
            int64_t o_ = b * TR_Q4_K_BLOCK_ELEMS + 64 * c; \
            for (int k_ = 0; k_ < 4; k_++) { \
                __m512 wv_[4]; \
                for (int r_ = 0; r_ < (R); r_++) { \
                    __m512i ix_ = (k_ & 1) ? i1_[r_] : i0_[r_]; \
                    wv_[r_] = (k_ & 2) ? _mm512_permutexvar_ps(_mm512_srli_epi32(ix_, 4), hi_[r_]) \
                                       : _mm512_permutexvar_ps(ix_, lo_[r_]); \
                } \
                __m512 w0 = wv_[0], w1 = wv_[1], w2 = wv_[2], w3 = (R) > 3 ? wv_[3] : wv_[0]; \
                (void)w3; \
                STEP(o_ + 16 * k_); \
            } \
        } \
    } while (0)

__attribute__((target("avx512f")))
static void row4_x6_q4_k(const unsigned char *row, size_t rb, const float *x, int64_t stride, int64_t n, float *out) {
    BP_BEGIN4X6;
    for (int64_t b = 0; b < n / TR_Q4_K_BLOCK_ELEMS; b++) BP_Q4_BLOCK(4, BP_STEP4X6);
    BP_OUT4X6(out);
    g_tile_kernel_calls++;
}

__attribute__((target("avx512f")))
static void row3_x8_q4_k(const unsigned char *row, size_t rb, const float *x, int64_t stride, int64_t n, float *out) {
    BP_BEGIN3X8;
    for (int64_t b = 0; b < n / TR_Q4_K_BLOCK_ELEMS; b++) BP_Q4_BLOCK(3, BP_STEP3X8);
    BP_OUT3X8(out);
    g_tile_kernel_calls++;
}

/* One tile job: row groups [begin, end) of R rows against every block of `tile` tokens, in
 * tr_matmul's order; the tokens past the last whole group of T go through the tier's dot_row2_x4
 * (four at a time, row pairs) and dot_row; the rows past the last whole group (rows % R) go one
 * dot at a time, by the worker that has the last group. y[t * rows + r] as tr_matmul. */
typedef struct {
    int R, T;          /* 4 x 6 or 3 x 8 */
    tr_type type;
    int64_t rows, cols, n_tokens, tile;
    const unsigned char *w;
    const float *x;
    float *y;
} tile_job;

static void tile_fn(void *ctx, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const tile_job *j = (const tile_job *)ctx;
    const tr_kernels *k = tr_kernels_get();
    size_t rb = tr_row_bytes(j->type, j->cols);
    int q4 = j->type == TR_TYPE_Q4_K;
    void (*kern)(const unsigned char *, size_t, const float *, int64_t, int64_t, float *) =
        j->R == 4 ? (q4 ? row4_x6_q4_k : row4_x6_q8_0) : (q4 ? row3_x8_q4_k : row3_x8_q8_0);
    int64_t n_groups = j->rows / j->R;
    for (int64_t b = 0; b < j->n_tokens; b += j->tile) {
        int64_t b_end = b + j->tile < j->n_tokens ? b + j->tile : j->n_tokens;
        for (int64_t g = begin; g < end; g++) {
            int64_t r = g * j->R;
            const unsigned char *row = j->w + (size_t)r * rb;
            int64_t q = b;
            float out[24];
            for (; q + j->T <= b_end; q += j->T) {
                kern(row, rb, j->x + q * j->cols, j->cols, j->cols, out);
                for (int i = 0; i < j->R; i++)
                    for (int t = 0; t < j->T; t++) j->y[(q + t) * j->rows + r + i] = out[i * j->T + t];
            }
            for (; q + 4 <= b_end; q += 4) {
                int i = 0;
                for (; i + 2 <= j->R; i += 2) {
                    k->dot_row2_x4[j->type](row + (size_t)i * rb, row + (size_t)(i + 1) * rb, j->x + q * j->cols, j->cols,
                                            j->cols, out);
                    for (int t = 0; t < 4; t++) {
                        j->y[(q + t) * j->rows + r + i] = out[t];
                        j->y[(q + t) * j->rows + r + i + 1] = out[4 + t];
                    }
                }
                for (; i < j->R; i++)
                    for (int t = 0; t < 4; t++)
                        j->y[(q + t) * j->rows + r + i] = k->dot_row[j->type](row + (size_t)i * rb, j->x + (q + t) * j->cols, j->cols);
            }
            for (; q < b_end; q++)
                for (int i = 0; i < j->R; i++)
                    j->y[q * j->rows + r + i] = k->dot_row[j->type](row + (size_t)i * rb, j->x + q * j->cols, j->cols);
        }
        if (end == n_groups)
            for (int64_t r = n_groups * j->R; r < j->rows; r++)
                for (int64_t q = b; q < b_end; q++)
                    j->y[q * j->rows + r] = k->dot_row[j->type](j->w + (size_t)r * rb, j->x + q * j->cols, j->cols);
    }
}

/* Question 64's lines: on each shape and type, tr_matmul (today's 2 x 8) against row4_x6 (tokens
 * in blocks of 24) and row3_x8 (blocks of 16, tr_matmul's), each checked byte for byte first,
 * then the three timed in turn, the order turning every run. Returns 1 on a result that differs. */
static int tile_lines(tr_pool *pool, int n, int runs, double peak, int *below_kernel, uint64_t *seed) {
    static const struct { int64_t rows, cols, nt; int q4k; } shapes[4] = {
        {1024, 2048, 64, 0}, {2048, 1024, 64, 0}, {2048, 2048, 512, 0}, {1024, 2048, 64, 1},
    };
    static const struct { int R, T; int64_t tile; const char *name; } vars[2] = {
        {4, 6, 24, "4x6"}, {3, 8, 16, "3x8"},
    };
    for (int s = 0; s < 4; s++) {
        int64_t rows = shapes[s].rows, cols = shapes[s].cols, nt = shapes[s].nt;
        tr_type type = shapes[s].q4k ? TR_TYPE_Q4_K : TR_TYPE_Q8_0;
        const char *tname = shapes[s].q4k ? " q4_k" : "";
        size_t rb = tr_row_bytes(type, cols);
        uint8_t *w = tr_alloc_aligned(rb * (size_t)rows, 64);
        float *x = tr_alloc_aligned((size_t)(cols * nt) * sizeof(float), 64);
        float *y_ref = tr_alloc_aligned((size_t)(rows * nt) * sizeof(float), 64);
        float *y_tile = tr_alloc_aligned((size_t)(rows * nt) * sizeof(float), 64);
        if (!w || !x || !y_ref || !y_tile) { fprintf(stderr, "out of memory\n"); return 1; }
        if (shapes[s].q4k) {
            unsigned q4k_seed = 778u;
            fill_q4_k(w, (rows * cols) / TR_Q4_K_BLOCK_ELEMS, &q4k_seed, 0);
        } else {
            fill_q8_0(w, rows * cols, seed);
        }
        for (int64_t i = 0; i < cols * nt; i++) x[i] = frand_state(seed);
        tr_mat m = {type, rows, cols, w};
        tr_matmul(pool, &m, x, nt, y_ref); /* warm-up, and the exactness reference */
        tile_job tj[2];
        for (int v = 0; v < 2; v++) {
            tile_job t = {vars[v].R, vars[v].T, type, rows, cols, nt, vars[v].tile, w, x, y_tile};
            tj[v] = t;
            memset(y_tile, 0xFF, (size_t)(rows * nt) * sizeof(float));
            tr_parallel_for(pool, rows / vars[v].R, 1, tile_fn, &tj[v]);
            if (memcmp(y_ref, y_tile, (size_t)(rows * nt) * sizeof(float)) != 0) {
                int64_t bad = 0;
                while (bad < rows * nt && memcmp(&y_ref[bad], &y_tile[bad], sizeof(float)) == 0) bad++;
                fprintf(stderr, "bench_peak: row%s %lldx%lld, %lld tok%s not exact: first differing index %lld "
                        "(matmul=%.9g tile=%.9g)\n", vars[v].name, (long long)rows, (long long)cols, (long long)nt,
                        tname, (long long)bad, (double)y_ref[bad], (double)y_tile[bad]);
                return 1;
            }
        }
        int calls = (int)(2000000000LL / (rows * cols * nt) / (n == 1 ? 8 : 1)) + 1;
        double gt[3][MAX_RUNS];
        for (int r = 0; r < runs; r++)
            for (int v = 0; v < 3; v++) {
                int which = (r + v) % 3; /* 0: tr_matmul, 1: 4 x 6, 2: 3 x 8 */
                double a = tr_time_sec();
                for (int cc = 0; cc < calls; cc++) {
                    if (which == 0) tr_matmul(pool, &m, x, nt, y_ref);
                    else tr_parallel_for(pool, rows / vars[which - 1].R, 1, tile_fn, &tj[which - 1]);
                }
                gt[which][r] = 2.0 * (double)(rows * cols * nt) * calls / (tr_time_sec() - a) / 1e9;
            }
        for (int v = 0; v < 3; v++) {
            qsort(gt[v], (size_t)runs, sizeof gt[v][0], cmp_double);
            char what[80];
            snprintf(what, sizeof what, "%s %lldx%lld, %lld tok%s", v == 0 ? "matmul (q64)" : vars[v - 1].name,
                     (long long)rows, (long long)cols, (long long)nt, tname);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, what, gt[v][runs / 2], (gt[v][runs - 1] - gt[v][0]) / gt[v][runs / 2] * 100);
            *below_kernel |= peak > 0.0 && gt[v][runs / 2] > peak * 1.02;
        }
        g_sink = y_ref[0] + y_tile[0];
        tr_free_aligned(w); tr_free_aligned(x); tr_free_aligned(y_ref); tr_free_aligned(y_tile);
    }
    return 0;
}
#endif

int main(int argc, char **argv) {
    int runs = 7, below_kernel = 0, n_counts = 2, only_q64 = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--one-core") == 0) n_counts = 1; /* the 16-core lines are noise anyway */
        else if (strcmp(argv[i], "--q64") == 0) only_q64 = 1; /* the peaks, mix x8 and question 64 only */
        else { fprintf(stderr, "usage: bench_peak [--runs N] [--one-core] [--q64]\n"); return 2; }
    }
    if (runs < 3) runs = 3;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
    tr_kernels_init();
    const tr_cpu_info *cpu = tr_cpu();
    char desc[512];
    tr_cpu_describe(cpu, desc, (int)sizeof desc);
    printf("%s\nkernel tier: %s; runs %d, median and spread (max-min)/median\n", desc, tr_kernels_get()->tier,
           runs);
    int counts[2] = {1, cpu->physical_cores > 64 ? 64 : cpu->physical_cores};
    uint64_t seed = 12345;
    peak_job j;
    memset(&j, 0, sizeof j);
    for (int w = 0; w < 64; w++) {
        j.rows[w] = tr_alloc_aligned(2 * tr_row_bytes(TR_TYPE_Q8_0, 2048), 64);
        j.x[w] = tr_alloc_aligned(8 * 2048 * sizeof(float), 64);
        if (j.rows[w] == NULL || j.x[w] == NULL) { fprintf(stderr, "out of memory\n"); return 1; }
    }
    printf("\n%-8s %-26s %12s %7s\n", "threads", "what", "GFLOP/s", "spread");
    for (int c = 0; c < n_counts; c++) {
        int n = counts[c];
        tr_pool *pool = tr_pool_create(n);
        double sp, g, peak = 0.0;
#if HAVE_X86
        if (cpu->avx512f) {
            j.kind = 0; j.iters = 20000000;
            g = run_job(pool, n, &j, 20000000.0 * 12 * 16, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "peak zmm mul+add", g, sp * 100);
            peak = g > peak ? g : peak;
        }
        if (cpu->avx2) {
            j.kind = 1; j.iters = 20000000;
            g = run_job(pool, n, &j, 20000000.0 * 12 * 8, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "peak ymm mul+add", g, sp * 100);
            peak = g > peak ? g : peak;
        }
        if (cpu->avx512f) {
            for (int w = 0; w < n; w++) {
                memset(j.rows[w], 1, 64);
                for (int i = 0; i < 128; i++) j.x[w][i] = 0.5f;
            }
            j.kind = 3; j.iters = 10000000;
            g = only_q64 ? 0.0 : run_job(pool, n, &j, 10000000.0 * 256, runs, &sp);
            if (!only_q64) printf("%-8d %-26s %12.1f %6.1f%%\n", n, "mix x4 (asm, L1)", g, sp * 100);
            below_kernel |= peak > 0.0 && g > peak * 1.02;
            j.kind = 4; j.iters = 5000000;
            g = run_job(pool, n, &j, 5000000.0 * 512, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "mix x8 (asm, L1)", g, sp * 100);
            below_kernel |= peak > 0.0 && g > peak * 1.02;
            /* question 64: the same stream with 4 rows x 6 tokens and 3 x 8, and 2 x 8 with their
             * decode; each the median of its own runs, the three in turn every round */
            {
                static const int kinds[3] = {11, 9, 10};
                static const char *names[3] = {"mix 2x8 bcast (asm, L1)", "mix 4x6 (asm, L1)", "mix 3x8 (asm, L1)"};
                static const double flops[3] = {512.0, 768.0, 768.0};
                double gm[3][MAX_RUNS];
                for (int r = 0; r < runs; r++)
                    for (int v = 0; v < 3; v++) {
                        int which = (r + v) % 3;
                        j.kind = kinds[which]; j.iters = 3000000;
                        gm[which][r] = run_job(pool, n, &j, 3000000.0 * flops[which], 3, &sp);
                    }
                for (int v = 0; v < 3; v++) {
                    qsort(gm[v], (size_t)runs, sizeof gm[v][0], cmp_double);
                    printf("%-8d %-26s %12.1f %6.1f%%\n", n, names[v], gm[v][runs / 2],
                           (gm[v][runs - 1] - gm[v][0]) / gm[v][runs / 2] * 100);
                    below_kernel |= peak > 0.0 && gm[v][runs / 2] > peak * 1.02;
                }
            }
            for (int w = 0; w < n; w++) {
                float *wf = (float *)(void *)j.rows[w];
                for (int i = 0; i < 32; i++) wf[i] = 1.0f;
            }
            j.kind = 8; j.iters = 5000000;
            g = only_q64 ? 0.0 : run_job(pool, n, &j, 5000000.0 * 512, runs, &sp);
            if (!only_q64) printf("%-8d %-26s %12.1f %6.1f%%\n", n, "mix x8 f32 (asm, L1)", g, sp * 100);
            /* not checked against peak: it has 16 independent accumulator chains (a0-a7, b0-b7)
             * against peak's 6, plus its loads run on ports the multiplies and adds do not use, so
             * it is not bound by the 6-chain peak the same way (like peak_fma/mix_x8_fma above,
             * measured but not gated: a different ceiling, not a broken one). */
        }
        if (!only_q64 && cpu->avx512f && cpu->fma) {
            j.kind = 5; j.iters = 10000000;
            g = run_job(pool, n, &j, 10000000.0 * 12 * 32, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "peak zmm fma (q. 59)", g, sp * 100);
            j.kind = 6; j.iters = 5000000;
            g = run_job(pool, n, &j, 5000000.0 * 512, runs, &sp);
            printf("%-8d %-26s %12.1f %6.1f%%\n", n, "mix x8 fma (q. 59)", g, sp * 100);
        }
#endif
        if (!only_q64 && tr_kernels_get()->dot_row2_x4[TR_TYPE_Q8_0] != NULL) {
            static const int64_t colss[2] = {512, 2048};
            for (int s = 0; s < 2; s++) {
                j.kind = 2; j.cols = colss[s];
                for (int w = 0; w < n; w++) {
                    fill_q8_0(j.rows[w], 2 * j.cols, &seed);
                    for (int64_t i = 0; i < 8 * j.cols; i++) j.x[w][i] = frand_state(&seed);
                }
                j.iters = 40000000 / j.cols;
                g = run_job(pool, n, &j, (double)j.iters * 16.0 * (double)j.cols, runs, &sp);
                char what[64];
                snprintf(what, sizeof what, "row2_x4 q8_0, %lld cols", (long long)j.cols);
                printf("%-8d %-26s %12.1f %6.1f%%\n", n, what, g, sp * 100);
                below_kernel |= peak > 0.0 && g > peak * 1.02;
                if (tr_kernels_get()->dot_row2_x8[TR_TYPE_Q8_0] == NULL) continue;
                j.kind = 7; j.iters = 20000000 / j.cols;
                g = run_job(pool, n, &j, (double)j.iters * 32.0 * (double)j.cols, runs, &sp);
                snprintf(what, sizeof what, "row2_x8 q8_0, %lld cols", (long long)j.cols);
                printf("%-8d %-26s %12.1f %6.1f%%\n", n, what, g, sp * 100);
                below_kernel |= peak > 0.0 && g > peak * 1.02;
            }
        }
        static const int64_t shapes[3][3] = {{1024, 2048, 64}, {2048, 1024, 64}, {2048, 2048, 512}};
        for (int s = 0; s < (only_q64 ? 0 : 3); s++) {
            int64_t rows = shapes[s][0], cols = shapes[s][1], nt = shapes[s][2];
            size_t rb = tr_row_bytes(TR_TYPE_Q8_0, cols);
            uint8_t *w = tr_alloc_aligned(rb * (size_t)rows, 64);
            float *x = tr_alloc_aligned((size_t)(cols * nt) * sizeof(float), 64);
            float *y = tr_alloc_aligned((size_t)(rows * nt) * sizeof(float), 64);
            if (!w || !x || !y) { fprintf(stderr, "out of memory\n"); return 1; }
            fill_q8_0(w, rows * cols, &seed);
            for (int64_t i = 0; i < cols * nt; i++) x[i] = frand_state(&seed);
            tr_mat m = {TR_TYPE_Q8_0, rows, cols, w};
            int calls = (int)(2000000000LL / (rows * cols * nt) / (n == 1 ? 8 : 1)) + 1;
            /* the tier as it is, and (where it has one) without its eight-token kernel */
            static tr_kernels no_x8;
            no_x8 = *tr_kernels_get();
            no_x8.dot_row2_x8[TR_TYPE_Q8_0] = NULL;
            int ab = tr_kernels_get()->dot_row2_x8[TR_TYPE_Q8_0] != NULL ? 2 : 1;
            double g2[2][MAX_RUNS];
            tr_matmul(pool, &m, x, nt, y);
            for (int r = 0; r < runs; r++) {
                for (int v = 0; v < ab; v++) {
                    int which = ab == 1 ? 0 : (r + v) % 2; /* the order turns every run */
                    tr_kernels_set_active(which == 1 ? &no_x8 : NULL);
                    double a = tr_time_sec();
                    for (int k = 0; k < calls; k++) tr_matmul(pool, &m, x, nt, y);
                    g2[which][r] = 2.0 * (double)(rows * cols * nt) * calls / (tr_time_sec() - a) / 1e9;
                }
            }
            tr_kernels_set_active(NULL);
            for (int v = 0; v < ab; v++) {
                qsort(g2[v], (size_t)runs, sizeof g2[v][0], cmp_double);
                char what[64];
                snprintf(what, sizeof what, "matmul %lldx%lld, %lld tok%s", (long long)rows, (long long)cols,
                         (long long)nt, v == 1 ? " -x8" : "");
                printf("%-8d %-26s %12.1f %6.1f%%\n", n, what, g2[v][runs / 2],
                       (g2[v][runs - 1] - g2[v][0]) / g2[v][runs / 2] * 100);
                below_kernel |= peak > 0.0 && g2[v][runs / 2] > peak * 1.02;
            }
            g_sink = y[0];
            tr_free_aligned(w); tr_free_aligned(x); tr_free_aligned(y);
        }
#if HAVE_X86
        if (cpu->avx512f) {
            static const struct { int64_t rows, cols, nt; int q4k; } pshapes[4] = {
                {1024, 2048, 64, 0}, {2048, 1024, 64, 0}, {2048, 2048, 512, 0}, {1024, 2048, 64, 1},
            };
            for (int s = 0; s < (only_q64 ? 0 : 4); s++) {
                int64_t rows = pshapes[s].rows, cols = pshapes[s].cols, nt = pshapes[s].nt;
                tr_type type = pshapes[s].q4k ? TR_TYPE_Q4_K : TR_TYPE_Q8_0;
                const char *tname = pshapes[s].q4k ? " q4_k" : "";
                /* assert: every shape here has a token count a multiple of 8 and a row count a
                 * multiple of 32, so the panel loop below never meets a remainder of either */
                if (nt % 8 != 0 || rows % 32 != 0) {
                    fprintf(stderr, "bench_peak: panel shape %lldx%lld, %lld tok is not a multiple of 8 tokens and 32 rows\n",
                            (long long)rows, (long long)cols, (long long)nt);
                    return 1;
                }
                size_t rb = tr_row_bytes(type, cols);
                uint8_t *w = tr_alloc_aligned(rb * (size_t)rows, 64);
                float *x = tr_alloc_aligned((size_t)(cols * nt) * sizeof(float), 64);
                float *y_ref = tr_alloc_aligned((size_t)(rows * nt) * sizeof(float), 64);
                float *y_panel = tr_alloc_aligned((size_t)(rows * nt) * sizeof(float), 64);
                if (!w || !x || !y_ref || !y_panel) { fprintf(stderr, "out of memory\n"); return 1; }
                if (pshapes[s].q4k) {
                    unsigned q4k_seed = 777u;
                    fill_q4_k(w, (rows * cols) / TR_Q4_K_BLOCK_ELEMS, &q4k_seed, 0);
                } else {
                    fill_q8_0(w, rows * cols, &seed);
                }
                for (int64_t i = 0; i < cols * nt; i++) x[i] = frand_state(&seed);
                tr_mat m = {type, rows, cols, w};
                tr_matmul(pool, &m, x, nt, y_ref); /* warm-up, and the exactness reference */

                int calls = (int)(2000000000LL / (rows * cols * nt) / (n == 1 ? 8 : 1)) + 1;
                for (int pidx = 0; pidx < 2; pidx++) {
                    int64_t P = pidx == 0 ? 16 : 32;
                    int64_t n_panels = rows / P;
                    panel_job pj;
                    memset(&pj, 0, sizeof pj);
                    pj.type = type; pj.rows = rows; pj.cols = cols; pj.n_tokens = nt; pj.P = P;
                    pj.w = w; pj.x = x; pj.y = y_panel;
                    for (int wk = 0; wk < n; wk++) {
                        pj.panel[wk] = tr_alloc_aligned((size_t)(P * cols) * sizeof(float), 64);
                        if (!pj.panel[wk]) { fprintf(stderr, "out of memory\n"); return 1; }
                    }

                    tr_parallel_for(pool, n_panels, 1, panel_fn, &pj); /* warm-up, and the check below */
                    if (memcmp(y_ref, y_panel, (size_t)(rows * nt) * sizeof(float)) != 0) {
                        int64_t bad = -1;
                        for (int64_t i = 0; i < rows * nt; i++)
                            if (y_ref[i] != y_panel[i]) { bad = i; break; }
                        fprintf(stderr,
                                "bench_peak: panel P=%lld %lldx%lld, %lld tok%s not exact: first differing "
                                "index %lld (matmul=%.9g panel=%.9g)\n",
                                (long long)P, (long long)rows, (long long)cols, (long long)nt, tname,
                                (long long)bad, (double)y_ref[bad], (double)y_panel[bad]);
                        return 1;
                    }

                    double g2p[2][MAX_RUNS];
                    for (int r = 0; r < runs; r++) {
                        for (int v = 0; v < 2; v++) {
                            int which = (r + v) % 2; /* 0: tr_matmul, 1: panel; the order turns every run */
                            double a = tr_time_sec();
                            if (which == 0) {
                                for (int cc = 0; cc < calls; cc++) tr_matmul(pool, &m, x, nt, y_ref);
                            } else {
                                for (int cc = 0; cc < calls; cc++) tr_parallel_for(pool, n_panels, 1, panel_fn, &pj);
                            }
                            g2p[which][r] = 2.0 * (double)(rows * cols * nt) * calls / (tr_time_sec() - a) / 1e9;
                        }
                    }
                    for (int v = 0; v < 2; v++) {
                        qsort(g2p[v], (size_t)runs, sizeof g2p[v][0], cmp_double);
                        char what[80];
                        if (v == 0)
                            snprintf(what, sizeof what, "matmul %lldx%lld, %lld tok%s (ab P%lld)", (long long)rows,
                                     (long long)cols, (long long)nt, tname, (long long)P);
                        else
                            snprintf(what, sizeof what, "panel P=%lld %lldx%lld, %lld tok%s", (long long)P,
                                     (long long)rows, (long long)cols, (long long)nt, tname);
                        printf("%-8d %-26s %12.1f %6.1f%%\n", n, what, g2p[v][runs / 2],
                               (g2p[v][runs - 1] - g2p[v][0]) / g2p[v][runs / 2] * 100);
                        below_kernel |= peak > 0.0 && g2p[v][runs / 2] > peak * 1.02;
                    }
                    for (int wk = 0; wk < n; wk++) tr_free_aligned(pj.panel[wk]);
                }
                g_sink = y_ref[0];
                tr_free_aligned(w); tr_free_aligned(x); tr_free_aligned(y_ref); tr_free_aligned(y_panel);
            }
            if (tr_kernels_get()->dot_row2_x4[TR_TYPE_Q8_0] != NULL &&
                tile_lines(pool, n, runs, peak, &below_kernel, &seed) != 0)
                return 1;
        }
#endif
        tr_pool_destroy(pool);
    }
    for (int w = 0; w < 64; w++) { tr_free_aligned(j.rows[w]); tr_free_aligned(j.x[w]); }
#if HAVE_X86
    /* the panel exactness check above must not pass vacuously: row2_x8_f32 has to have run */
    if (cpu->avx512f && g_tile_kernel_calls == 0) {
        fprintf(stderr, "bench_peak: no row4_x6 or row3_x8 ran: their exactness check passed vacuously\n");
        return 1;
    }
    if (cpu->avx512f && !only_q64 && g_panel_kernel_calls == 0) {
        fprintf(stderr, "bench_peak: row2_x8_f32 never ran: the panel exactness check passed vacuously\n");
        return 1;
    }
#endif
    /* a peak below a real kernel is not a peak: the compiler merged or dropped its chains (it
     * did once, docs/LESSONS.md #161) */
    if (below_kernel) {
        printf("bench_peak: a kernel ran faster than the peak loop: the peak loop is not what it claims\n");
        return 1;
    }
    return 0;
}
