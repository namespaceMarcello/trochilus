/* bench_q4k_genome.c — the decode's Q4_K dot sequenced like a genome before it is changed
 * (docs/MEASUREMENTS.md question 66: reach llama.cpp's decode at 4 threads, exactly).
 *
 * The one-row kernel (kernels_x86.c's avx512_dot_row_q4_k) takes ~120 ns a row of 2048 in L1.
 * Where those nanoseconds go is measured here, not inferred (docs/LESSONS.md #176, #178):
 *
 *   clock      a chain of dependent integer adds (1 cycle each on every x86): the core's clock
 *              in this run, so every time below is also given in cycles;
 *   op ...     the cost of each instruction the kernel uses, on this core: 12 independent ops a
 *              loop iteration ("tput", cycles per op) or 12 dependent ones ("lat", its latency);
 *              "mix" lines put several kinds together, to see which ones share a pipe;
 *   kernel ... the kernel over 4 rows of 2048 that stay in L1 with the token's 8 KiB, called one
 *              row after another as tr_matmul does, ns and cycles a row. "full" is a copy of the
 *              tier's kernel; every other line removes pieces from it:
 *                prescale   the sixteen scales and mins read as floats (decoded before timing)
 *                f16c       the two halves converted by vcvtph2ps instead of in software
 *                pretable   the 16-value tables read (built before timing): no scale decode,
 *                           no table build
 *                constidx   the permute's indices constant: no load/widen/shift of the quants
 *                noperm     the table multiplied as it is: no index, no permute
 *                notree     the sixteen lanes not combined at the end (lane 0 returned)
 *              and their combinations; "tier" calls the active tier's dot_row itself.
 *
 * The lines that keep the kernel's arithmetic (tier, full, prescale, f16c, pretable) are checked
 * bit for bit against the scalar kernel before timing; g_exact_checked counts the rows compared
 * and fails the run if none was. Timed variants run in turn, run after run, median of N runs and
 * spread (max - min) / median; the clock is measured in every round.
 *
 *   bench_q4k_genome [--runs N] [--ms M]          one core; AVX-512 only (prints why otherwise)
 * Run by tools/bench_q4k_genome.sh (the marker, a still machine, the container by default). */
#include <stdatomic.h>
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
#define L1_ROWS 4         /* 4 x 1152 B of weights + 8 KiB of token + 16 KiB of tables < 32 KiB */
#define COLS 2048
#define NB (COLS / TR_Q4_K_BLOCK_ELEMS)

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_spread(double *v, int n, double *spread) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    double med = v[n / 2];
    *spread = med > 0 ? (v[n - 1] - v[0]) / med : 0;
    return med;
}

static volatile float g_sink;
static int64_t g_exact_checked;

#if HAVE_X86

/* ---- the core's clock -------------------------------------------------------------------- */

/* 16 dependent adds an iteration: 1 cycle each, the loop's dec/jnz run beside them */
static void clock_chain(int64_t iters) {
    uint64_t a = 0;
    __asm__ volatile("1:\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "dec %1\n\tjnz 1b"
                     : "+r"(a), "+r"(iters)
                     :
                     : "cc");
    g_sink = (float)a;
}

/* GHz: cycles per ns, from ~2 ms of the chain */
static double clock_ghz(void) {
    const int64_t iters = 600000;
    clock_chain(iters / 10);
    double t0 = tr_time_sec();
    clock_chain(iters);
    double t1 = tr_time_sec();
    return (double)iters * 16 / ((t1 - t0) * 1e9);
}

/* ---- the cost of one instruction ------------------------------------------------------- */

/* Twelve copies of an instruction a loop iteration. In the "tput" loops every copy writes its
 * own register and reads constant ones (zmm20..23, set before the loop), so nothing waits; in
 * the "lat" loops each copy reads the previous one's result. zmm/xmm 0..11 and 20..23 are
 * clobbered; `m` points to 4 KiB in L1. */
#define R12(f) f(0) f(1) f(2) f(3) f(4) f(5) f(6) f(7) f(8) f(9) f(10) f(11)
#define CLOB_V "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", \
               "xmm10", "xmm11", "xmm20", "xmm21", "xmm22", "xmm23"
#define SETUP "vbroadcastss (%[m]), %%zmm20\n\tvbroadcastss 4(%[m]), %%zmm21\n\t" \
              "vpmovzxbd 64(%[m]), %%zmm22\n\tvbroadcastss 8(%[m]), %%zmm23\n\t" \
              "vxorps %%xmm0, %%xmm0, %%xmm0\n\t"
#define OP_FN(name, text)                                                                   \
    __attribute__((noinline, target("avx512f,f16c"))) static void name(int64_t iters,        \
                                                                           const void *m) { \
        __asm__ volatile(SETUP "1:\n\t" R12(text) "dec %[n]\n\tjnz 1b"                      \
                         : [n] "+r"(iters)                                                   \
                         : [m] "r"(m)                                                        \
                         : CLOB_V, "cc", "memory");                                          \
    }
#define S_(k) #k
#define Z(k) "%%zmm" S_(k)
#define X(k) "%%xmm" S_(k)
#define Y(k) "%%ymm" S_(k)

#define I_MUL(k) "vmulps %%zmm20, %%zmm21, " Z(k) "\n\t"
#define I_MUL_LAT(k) "vmulps %%zmm20, %%zmm0, %%zmm0\n\t"
#define I_ADD(k) "vaddps %%zmm20, %%zmm21, " Z(k) "\n\t"
#define I_ADD_LAT(k) "vaddps %%zmm20, %%zmm0, %%zmm0\n\t"
#define I_PERM(k) "vpermps %%zmm20, %%zmm22, " Z(k) "\n\t"
#define I_PERM_LAT(k) "vpermps %%zmm0, %%zmm22, %%zmm0\n\t"
#define I_PERMY(k) "vpermps %%ymm20, %%ymm22, " Y(k) "\n\t"
#define I_PERMT2(k) "vpermt2ps %%zmm20, %%zmm22, " Z(k) "\n\t"
#define I_ZXM(k) "vpmovzxbd " S_(k) "*16(%[m]), " Z(k) "\n\t"
#define I_ZXR(k) "vpmovzxbd %%xmm23, " Z(k) "\n\t"
#define I_SRL(k) "vpsrld $4, %%zmm22, " Z(k) "\n\t"
#define I_BCM(k) "vbroadcastss " S_(k) "*4+128(%[m]), " Z(k) "\n\t"
#define I_BCR(k) "vbroadcastss %%xmm20, " Z(k) "\n\t"
#define I_CVTSI(k) "vcvtsi2ss %k[n], " X(k) ", " X(k) "\n\t"
#define I_CVTPH(k) "vcvtph2ps %%xmm23, " X(k) "\n\t"
#define I_MULSS(k) "vmulss %%xmm20, %%xmm21, " X(k) "\n\t"
#define I_LOADZ(k) "vmovups " S_(k) "*64+256(%[m]), " Z(k) "\n\t"

OP_FN(op_mul, I_MUL)
OP_FN(op_mul_lat, I_MUL_LAT)
OP_FN(op_add, I_ADD)
OP_FN(op_add_lat, I_ADD_LAT)
OP_FN(op_perm, I_PERM)
OP_FN(op_perm_lat, I_PERM_LAT)
OP_FN(op_permy, I_PERMY)
OP_FN(op_permt2, I_PERMT2)
OP_FN(op_zxm, I_ZXM)
OP_FN(op_zxr, I_ZXR)
OP_FN(op_srl, I_SRL)
OP_FN(op_bcm, I_BCM)
OP_FN(op_bcr, I_BCR)
OP_FN(op_cvtsi, I_CVTSI)
OP_FN(op_cvtph, I_CVTPH)
OP_FN(op_mulss, I_MULSS)
OP_FN(op_loadz, I_LOADZ)

/* the mixes: 12 instructions an iteration, the kinds in turn */
#define MIX_FN(name, text)                                                                  \
    __attribute__((noinline, target("avx512f"))) static void name(int64_t iters, const void *m) { \
        __asm__ volatile(SETUP "1:\n\t" text "dec %[n]\n\tjnz 1b"                            \
                         : [n] "+r"(iters)                                                   \
                         : [m] "r"(m)                                                        \
                         : CLOB_V, "cc", "memory");                                          \
    }
/* permute + multiply + add, 4 of each */
#define PMA(a, b, c) I_PERM(a) I_MUL(b) I_ADD(c)
MIX_FN(mix_pma, PMA(0, 1, 2) PMA(3, 4, 5) PMA(6, 7, 8) PMA(9, 10, 11))
/* multiply + add, 6 of each: the definition's own two ops */
#define MA(a, b) I_MUL(a) I_ADD(b)
MIX_FN(mix_ma, MA(0, 1) MA(2, 3) MA(4, 5) MA(6, 7) MA(8, 9) MA(10, 11))
/* permute + widen from memory, 6 of each: the lookup and its index */
#define PZ(a, b) I_PERM(a) I_ZXM(b)
MIX_FN(mix_pz, PZ(0, 1) PZ(2, 3) PZ(4, 5) PZ(6, 7) PZ(8, 9) PZ(10, 11))
/* the kernel's proportions for 64 weights: 4 permutes, 2 widens, 2 shifts, 4 multiplies, 4 adds
 * (16 ops; the table's 2 muls, 2 subs and 4 broadcasts left out), independent */
MIX_FN(mix_kernel, I_ZXM(0) I_ZXM(1) I_SRL(2) I_SRL(3) I_PERM(4) I_PERM(5) I_PERM(6) I_PERM(7)
                       I_MUL(8) I_MUL(9) I_MUL(10) I_MUL(11) I_ADD(0) I_ADD(1) I_ADD(2) I_ADD(3))

typedef struct {
    const char *name;
    void (*fn)(int64_t iters, const void *m);
    int ops;    /* instructions an iteration */
} op_case;

static const op_case OPS[] = {
    {"vmulps zmm tput", op_mul, 12},         {"vmulps zmm lat", op_mul_lat, 12},
    {"vaddps zmm tput", op_add, 12},         {"vaddps zmm lat", op_add_lat, 12},
    {"vpermps zmm tput", op_perm, 12},       {"vpermps zmm lat", op_perm_lat, 12},
    {"vpermps ymm tput", op_permy, 12},      {"vpermt2ps zmm tput", op_permt2, 12},
    {"vpmovzxbd zmm,m128 tput", op_zxm, 12}, {"vpmovzxbd zmm,xmm tput", op_zxr, 12},
    {"vpsrld zmm tput", op_srl, 12},         {"vbroadcastss zmm,m32 tput", op_bcm, 12},
    {"vbroadcastss zmm,xmm tput", op_bcr, 12}, {"vcvtsi2ss tput", op_cvtsi, 12},
    {"vcvtph2ps xmm tput", op_cvtph, 12},    {"vmulss tput", op_mulss, 12},
    {"vmovups zmm load tput", op_loadz, 12},
    {"mix perm+mul+add (per op)", mix_pma, 12}, {"mix mul+add (per op)", mix_ma, 12},
    {"mix perm+widen (per op)", mix_pz, 12},  {"mix kernel 16 ops (per op)", mix_kernel, 16},
};
#define N_OPS ((int)(sizeof OPS / sizeof OPS[0]))

/* ---- the kernel and its deletions ------------------------------------------------------ */

typedef struct {
    const unsigned char *rows;   /* L1_ROWS rows of COLS Q4_K elements */
    const float *x;              /* COLS */
    const float *pre_scales;     /* per row and block: scale[0..7], min[0..7] */
    const float *pre_tables;     /* per row, block and sub-block: the 16 weights scale*q - min */
} l1_job;

enum { SC_DECODE, SC_F16C, SC_PRE };

/* tr_q4_k_scales, line for line, always inlined: in kernels_x86.c gcc inlines it into the tier's
 * kernel, here (a file with many callers) it left a call, which the copy must not have */
__attribute__((always_inline)) static inline void scales_inl(const unsigned char *blk, float scale[8], float min[8]) {
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

/* tr_q4_k_scales with the halves converted by the hardware (exact: IEEE half to float, a NaN's
 * payload kept but quieted, which the following multiply does anyway) */
__attribute__((target("avx512f,f16c"))) static inline void scales_f16c(const unsigned char *blk, float scale[8],
                                                                        float min[8]) {
    uint32_t dd;
    memcpy(&dd, blk, 4);
    __m128 h = _mm_cvtph_ps(_mm_cvtsi32_si128((int)dd));
    float d = _mm_cvtss_f32(h), dmin = _mm_cvtss_f32(_mm_shuffle_ps(h, h, 1));
    const unsigned char *s = blk + 4;
    for (int j = 0; j < 4; j++) {
        scale[j] = d * (float)(s[j] & 63);
        min[j] = dmin * (float)(s[j + 4] & 63);
        scale[j + 4] = d * (float)((s[j + 8] & 0x0F) | ((s[j] >> 6) << 4));
        min[j + 4] = dmin * (float)((s[j + 8] >> 4) | ((s[j + 4] >> 6) << 4));
    }
}

/* One template, specialised by constant flags in the noinline wrappers below: `sc` where the
 * scales come from, `tab_pre` the tables read instead of built, `idx` the quants widened into
 * indices (else constant ones), `perm` the lookup done (else the table used as the weights),
 * `tree` the lanes combined. With every flag on it is avx512_dot_row_q4_k, line for line. */
__attribute__((always_inline, target("avx512f,f16c"))) static inline float
q4k_tmpl(const l1_job *j, int r, int sc, int tab_pre, int idx, int perm, int tree) {
    const unsigned char *p = j->rows + (size_t)r * NB * TR_Q4_K_BLOCK_BYTES;
    const __m512 qv = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                     12.0f, 13.0f, 14.0f, 15.0f);
    /* constant indices that cross lanes, so the permute is a real one */
    const __m512i c0 = _mm512_setr_epi32(7, 12, 1, 14, 3, 9, 0, 5, 15, 2, 11, 6, 13, 4, 10, 8);
    const __m512i c1 = _mm512_setr_epi32(3, 8, 13, 0, 10, 5, 15, 1, 6, 11, 2, 12, 9, 14, 4, 7);
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < NB; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        const float *xb = j->x + b * TR_Q4_K_BLOCK_ELEMS;
        const float *tb = j->pre_tables + ((size_t)r * NB + b) * 8 * 16;
        float scale[8], min[8];
        if (!tab_pre) {
            if (sc == SC_DECODE) scales_inl(blk, scale, min);
            else if (sc == SC_F16C) scales_f16c(blk, scale, min);
            else {
                memcpy(scale, j->pre_scales + ((size_t)r * NB + b) * 16, sizeof scale);
                memcpy(min, j->pre_scales + ((size_t)r * NB + b) * 16 + 8, sizeof min);
            }
        }
        for (int c = 0; c < 4; c++) {
            __m512 vlo, vhi;
            if (tab_pre) {
                vlo = _mm512_loadu_ps(tb + 32 * c);
                vhi = _mm512_loadu_ps(tb + 32 * c + 16);
            } else {
                vlo = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(scale[2 * c]), qv), _mm512_set1_ps(min[2 * c]));
                vhi = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(scale[2 * c + 1]), qv),
                                    _mm512_set1_ps(min[2 * c + 1]));
            }
            __m512 w0 = vlo, w1 = vlo, w2 = vhi, w3 = vhi;
            if (perm) {
                __m512i q0 = c0, q1 = c1, h0 = c1, h1 = c0;
                if (idx) {
                    q0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c)));
                    q1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c + 16)));
                    h0 = _mm512_srli_epi32(q0, 4);
                    h1 = _mm512_srli_epi32(q1, 4);
                }
                w0 = _mm512_permutexvar_ps(q0, vlo);
                w1 = _mm512_permutexvar_ps(q1, vlo);
                w2 = _mm512_permutexvar_ps(h0, vhi);
                w3 = _mm512_permutexvar_ps(h1, vhi);
            }
            const float *xc = xb + 64 * c;
            acc = _mm512_add_ps(acc, _mm512_mul_ps(w0, _mm512_loadu_ps(xc)));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(w1, _mm512_loadu_ps(xc + 16)));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(w2, _mm512_loadu_ps(xc + 32)));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(w3, _mm512_loadu_ps(xc + 48)));
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc);
    return tree ? tr_lane_combine(lane) : lane[0];
}

#define VARIANT(name, sc, tab, idx, perm, tree)                                                 \
    __attribute__((noinline, target("avx512f,f16c"))) static float name(const l1_job *j, int r) { \
        return q4k_tmpl(j, r, sc, tab, idx, perm, tree);                                        \
    }
VARIANT(k_full, SC_DECODE, 0, 1, 1, 1)
VARIANT(k_f16c, SC_F16C, 0, 1, 1, 1)
VARIANT(k_prescale, SC_PRE, 0, 1, 1, 1)
VARIANT(k_pretable, SC_PRE, 1, 1, 1, 1)
VARIANT(k_constidx, SC_DECODE, 0, 0, 1, 1)
VARIANT(k_noperm, SC_DECODE, 0, 0, 0, 1)
VARIANT(k_notree, SC_DECODE, 0, 1, 1, 0)
VARIANT(k_prescale_constidx, SC_PRE, 0, 0, 1, 1)
VARIANT(k_prescale_noperm, SC_PRE, 0, 0, 0, 1)
VARIANT(k_pretable_constidx, SC_PRE, 1, 0, 1, 1)
VARIANT(k_pretable_noperm, SC_PRE, 1, 0, 0, 1)

/* ---- candidates -------------------------------------------------------------------------- */

/* The sixteen scales and mins of a block in one vector, the same floats as tr_q4_k_scales:
 * the 12 packed bytes rearranged by two byte shuffles into sc0..7, m0..7 (6 bits each), widened,
 * converted (exact: small integers) and multiplied by [d x 8, dmin x 8] (the halves by
 * vcvtph2ps, exact), then stored: each sub-block's scale and min are broadcast from memory,
 * which costs a load, where the earlier vector attempt (MEASUREMENTS §The decode's matmul
 * against ggml's) broadcast them by lane permutes and lost. */
__attribute__((always_inline, target("avx512f,f16c"))) static inline void vscales_store(const unsigned char *blk,
                                                                                         float sm[16]) {
    const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(blk + 4));
    const __m128i ia = _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11);
    const __m128i ib = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7);
    const __m128i lowmask = _mm_setr_epi8(63, 63, 63, 63, 15, 15, 15, 15, 63, 63, 63, 63, 15, 15, 15, 15);
    __m128i a = _mm_shuffle_epi8(v, ia);
    __m128i b = _mm_shuffle_epi8(v, ib);
    /* lanes 12..15 take the high nibble of their byte */
    __m128i ahi = _mm_and_si128(_mm_srli_epi16(a, 4), _mm_set1_epi8(0x0F));
    __m128i sel = _mm_blend_epi16(a, ahi, 0xC0);
    __m128i hi2 = _mm_srli_epi16(_mm_and_si128(b, _mm_set1_epi8((char)0xC0)), 2);
    __m128i bytes = _mm_or_si128(_mm_and_si128(sel, lowmask), hi2);
    uint32_t dd;
    memcpy(&dd, blk, 4);
    __m512 h = _mm512_castps128_ps512(_mm_cvtph_ps(_mm_cvtsi32_si128((int)dd)));
    __m512 mul = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1), h);
    _mm512_store_ps(sm, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes)), mul));
}

/* the tier's kernel with the scales decoded by vscales_store, four ways of getting each
 * sub-block's scale and min to the table:
 *   VS_INLINE   decoded into a 64-byte buffer at the top of each block, then broadcast from it:
 *               the broadcasts load what the store just wrote (store-to-load forwarding);
 *   VS_PREPASS  the whole row's scales decoded first into NB buffers, then the blocks: each load
 *               is ~a block or more behind its store;
 *   VS_AHEAD    block b+1 decoded while block b runs (two buffers);
 *   VS_PERM     no memory: the decoded vector kept in a register and each scale and min
 *               broadcast by a lane permute (two per sub-block). */
enum { VS_INLINE, VS_PREPASS, VS_AHEAD, VS_PERM };
__attribute__((always_inline, target("avx512f,f16c"))) static inline float cand_vscale_tmpl(const l1_job *j, int r,
                                                                                            int mode, int pfd) {
    const unsigned char *p = j->rows + (size_t)r * NB * TR_Q4_K_BLOCK_BYTES;
    const __m512 qv = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                     12.0f, 13.0f, 14.0f, 15.0f);
    __m512 acc = _mm512_setzero_ps();
    float sm[NB][16] __attribute__((aligned(64)));
    if (mode == VS_PREPASS)
        for (int b = 0; b < NB; b++) vscales_store(p + (size_t)b * TR_Q4_K_BLOCK_BYTES, sm[b]);
    if (mode == VS_AHEAD) vscales_store(p, sm[0]);
    for (int b = 0; b < NB; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        const float *xb = j->x + b * TR_Q4_K_BLOCK_ELEMS;
        const float *s = sm[b];
        __m512 sv = _mm512_setzero_ps();
        if (pfd) {
            _mm_prefetch((const char *)blk + pfd, _MM_HINT_T0);
            _mm_prefetch((const char *)blk + pfd + 64, _MM_HINT_T0);
            _mm_prefetch((const char *)blk + pfd + 128, _MM_HINT_T0);
        }
        if (mode == VS_INLINE) vscales_store(blk, sm[b]);
        if (mode == VS_AHEAD && b + 1 < NB) vscales_store(blk + TR_Q4_K_BLOCK_BYTES, sm[b + 1]);
        if (mode == VS_PERM) {
            vscales_store(blk, sm[b]);
            sv = _mm512_load_ps(sm[b]);   /* gcc forwards the register: no memory round trip */
        }
        __asm__ volatile("" : : "r"(sm) : "memory");
        for (int c = 0; c < 4; c++) {
            __m512 slo, mlo, shi, mhi;
            if (mode == VS_PERM) {
                slo = _mm512_permutexvar_ps(_mm512_set1_epi32(2 * c), sv);
                mlo = _mm512_permutexvar_ps(_mm512_set1_epi32(8 + 2 * c), sv);
                shi = _mm512_permutexvar_ps(_mm512_set1_epi32(2 * c + 1), sv);
                mhi = _mm512_permutexvar_ps(_mm512_set1_epi32(8 + 2 * c + 1), sv);
            } else {
                slo = _mm512_set1_ps(s[2 * c]);
                mlo = _mm512_set1_ps(s[8 + 2 * c]);
                shi = _mm512_set1_ps(s[2 * c + 1]);
                mhi = _mm512_set1_ps(s[8 + 2 * c + 1]);
            }
            __m512 vlo = _mm512_sub_ps(_mm512_mul_ps(slo, qv), mlo);
            __m512 vhi = _mm512_sub_ps(_mm512_mul_ps(shi, qv), mhi);
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
#define VSCALE(name, mode, pfd)                                                                        \
    __attribute__((noinline, target("avx512f,f16c"))) static float name(const l1_job *j, int r) { \
        return cand_vscale_tmpl(j, r, mode, pfd);                                                      \
    }
VSCALE(c_vscale, VS_INLINE, 0)
VSCALE(c_vscale_pre, VS_PREPASS, 0)
VSCALE(c_vscale_ahead, VS_AHEAD, 0)
VSCALE(c_vscale_perm, VS_PERM, 0)
VSCALE(c_vscale_ahead_pf1, VS_AHEAD, 1152)
VSCALE(c_vscale_ahead_pf2, VS_AHEAD, 2304)
VSCALE(c_vscale_ahead_pf4, VS_AHEAD, 4608)
VSCALE(c_vscale_ahead_pf8, VS_AHEAD, 9216)

/* NR rows (r .. r+NR-1) at once, block by block: NR chains of adds instead of one, the token's
 * loads shared; each row keeps its own lanes and order, so out[i] is row r+i's one-row result.
 * `mode` as cand_vscale_tmpl (VS_PREPASS or VS_AHEAD). `fmatab` builds each table entry
 * scale * q - min with one fused multiply-subtract: the same float, because scale = d * sc has at
 * most 11 + 6 significant bits and q 4, so scale * q is exact and the fused op rounds once where
 * the definition rounds once too (the product's rounding is the identity). Finite scales only:
 * for a NaN or infinite d or dmin the NaN chosen could differ (not in these rows). */
__attribute__((always_inline, target("avx512f,f16c,fma"))) static inline void rows_tmpl(const l1_job *j, int r,
                                                                                         float *out, int nr,
                                                                                         int mode, int fmatab, int pfd) {
    const unsigned char *p = j->rows + (size_t)r * NB * TR_Q4_K_BLOCK_BYTES;
    const size_t rb = (size_t)NB * TR_Q4_K_BLOCK_BYTES;
    const __m512 qv = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                     12.0f, 13.0f, 14.0f, 15.0f);
    __m512 acc[4];
    float sm[4][NB][16] __attribute__((aligned(64)));
#pragma GCC unroll 4
    for (int i = 0; i < nr; i++) acc[i] = _mm512_setzero_ps();
    if (mode == VS_PREPASS) {
        for (int b = 0; b < NB; b++)
#pragma GCC unroll 4
            for (int i = 0; i < nr; i++) vscales_store(p + i * rb + (size_t)b * TR_Q4_K_BLOCK_BYTES, sm[i][b]);
    } else {
#pragma GCC unroll 4
        for (int i = 0; i < nr; i++) vscales_store(p + i * rb, sm[i][0]);
    }
    for (int b = 0; b < NB; b++) {
        if (mode == VS_AHEAD && b + 1 < NB) {
#pragma GCC unroll 4
            for (int i = 0; i < nr; i++) vscales_store(p + i * rb + (size_t)(b + 1) * TR_Q4_K_BLOCK_BYTES, sm[i][b + 1]);
        }
        __asm__ volatile("" : : "r"(sm) : "memory");
        if (pfd) {
#pragma GCC unroll 4
            for (int i = 0; i < nr; i++) {
                const char *q = (const char *)p + i * rb + (size_t)b * TR_Q4_K_BLOCK_BYTES + pfd;
                _mm_prefetch(q, _MM_HINT_T0);
                _mm_prefetch(q + 64, _MM_HINT_T0);
                _mm_prefetch(q + 128, _MM_HINT_T0);
            }
        }
        const float *xb = j->x + b * TR_Q4_K_BLOCK_ELEMS;
        for (int c = 0; c < 4; c++) {
            const float *xc = xb + 64 * c;
            __m512 x0 = _mm512_loadu_ps(xc), x1 = _mm512_loadu_ps(xc + 16);
            __m512 x2 = _mm512_loadu_ps(xc + 32), x3 = _mm512_loadu_ps(xc + 48);
#pragma GCC unroll 4
            for (int i = 0; i < nr; i++) {
                const float *s = sm[i][b];
                __m512 lo, hi;
                if (fmatab) {
                    lo = _mm512_fmsub_ps(_mm512_set1_ps(s[2 * c]), qv, _mm512_set1_ps(s[8 + 2 * c]));
                    hi = _mm512_fmsub_ps(_mm512_set1_ps(s[2 * c + 1]), qv, _mm512_set1_ps(s[8 + 2 * c + 1]));
                } else {
                    lo = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(s[2 * c]), qv), _mm512_set1_ps(s[8 + 2 * c]));
                    hi = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(s[2 * c + 1]), qv), _mm512_set1_ps(s[8 + 2 * c + 1]));
                }
                const unsigned char *qp = p + i * rb + (size_t)b * TR_Q4_K_BLOCK_BYTES + TR_Q4_K_QS_OFFSET + 32 * c;
                __m512i q0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)qp));
                __m512i q1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qp + 16)));
                __m512 a = acc[i];
                a = _mm512_add_ps(a, _mm512_mul_ps(_mm512_permutexvar_ps(q0, lo), x0));
                a = _mm512_add_ps(a, _mm512_mul_ps(_mm512_permutexvar_ps(q1, lo), x1));
                a = _mm512_add_ps(a, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q0, 4), hi), x2));
                a = _mm512_add_ps(a, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q1, 4), hi), x3));
                acc[i] = a;
            }
        }
    }
    float lane[TR_LANES];
#pragma GCC unroll 4
    for (int i = 0; i < nr; i++) {
        _mm512_storeu_ps(lane, acc[i]);
        out[i] = tr_lane_combine(lane);
    }
}
#define ROWS(name, nr, mode, fmatab, pfd)                                                                          \
    __attribute__((noinline, target("avx512f,f16c,fma"))) static void name(const l1_job *j, int r, float *out) { \
        rows_tmpl(j, r, out, nr, mode, fmatab, pfd);                                                               \
    }
ROWS(c_x2_pre, 2, VS_PREPASS, 0, 0)
ROWS(c_x2_ahead, 2, VS_AHEAD, 0, 0)
ROWS(c_x2_ahead_fmatab, 2, VS_AHEAD, 1, 0)
ROWS(c_x4_ahead, 4, VS_AHEAD, 0, 0)
ROWS(c_x4_ahead_fmatab, 4, VS_AHEAD, 1, 0)
ROWS(c_x1_ahead_fmatab, 1, VS_AHEAD, 1, 0)
ROWS(c_x2_pre_pf2, 2, VS_PREPASS, 0, 2304)
ROWS(c_x2_pre_pf4, 2, VS_PREPASS, 0, 4608)
ROWS(c_x2_ahead_pf2, 2, VS_AHEAD, 0, 2304)
ROWS(c_x2_ahead_pf4, 2, VS_AHEAD, 0, 4608)

/* question 59's kernel: the tier's with a fused multiply-add into the accumulator (another
 * definition: not checked against scalar) */
__attribute__((noinline, target("avx512f,fma"))) static float k_fma(const l1_job *j, int r) {
    const unsigned char *p = j->rows + (size_t)r * NB * TR_Q4_K_BLOCK_BYTES;
    const __m512 qv = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                     12.0f, 13.0f, 14.0f, 15.0f);
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < NB; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        const float *xb = j->x + b * TR_Q4_K_BLOCK_ELEMS;
        float scale[8], min[8];
        scales_inl(blk, scale, min);
        for (int c = 0; c < 4; c++) {
            __m512 vlo = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(scale[2 * c]), qv), _mm512_set1_ps(min[2 * c]));
            __m512 vhi = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(scale[2 * c + 1]), qv), _mm512_set1_ps(min[2 * c + 1]));
            __m512i q0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c)));
            __m512i q1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(qs + 32 * c + 16)));
            const float *xc = xb + 64 * c;
            acc = _mm512_fmadd_ps(_mm512_permutexvar_ps(q0, vlo), _mm512_loadu_ps(xc), acc);
            acc = _mm512_fmadd_ps(_mm512_permutexvar_ps(q1, vlo), _mm512_loadu_ps(xc + 16), acc);
            acc = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q0, 4), vhi), _mm512_loadu_ps(xc + 32), acc);
            acc = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q1, 4), vhi), _mm512_loadu_ps(xc + 48), acc);
        }
    }
    float lane[TR_LANES];
    _mm512_storeu_ps(lane, acc);
    return tr_lane_combine(lane);
}

static float (*g_tier_dot)(const void *row, const float *x, int64_t n);
static const char *g_only;   /* --only: the kernels whose name contains it, and "tier" */
__attribute__((noinline)) static float k_tier(const l1_job *j, int r) {
    return g_tier_dot(j->rows + (size_t)r * NB * TR_Q4_K_BLOCK_BYTES, j->x, COLS);
}

typedef struct {
    const char *name;
    float (*fn)(const l1_job *j, int r);              /* one row */
    void (*fn2)(const l1_job *j, int r, float *out);   /* or rows r .. r+nrows-1 (r a multiple) */
    int exact;   /* keeps the kernel's arithmetic: checked against scalar */
    int nrows;   /* fn2's rows a call */
} kernel_case;

static const kernel_case KERNELS[] = {
    {"tier", k_tier, NULL, 1, 0},
    {"full", k_full, NULL, 1, 0},
    {"f16c", k_f16c, NULL, 1, 0},
    {"prescale", k_prescale, NULL, 1, 0},
    {"pretable", k_pretable, NULL, 1, 0},
    {"constidx", k_constidx, NULL, 0, 0},
    {"noperm", k_noperm, NULL, 0, 0},
    {"notree", k_notree, NULL, 0, 0},
    {"prescale+constidx", k_prescale_constidx, NULL, 0, 0},
    {"prescale+noperm", k_prescale_noperm, NULL, 0, 0},
    {"pretable+constidx", k_pretable_constidx, NULL, 0, 0},
    {"pretable+noperm", k_pretable_noperm, NULL, 0, 0},
    {"cand vscale", c_vscale, NULL, 1, 0},
    {"cand vscale prepass", c_vscale_pre, NULL, 1, 0},
    {"cand vscale ahead", c_vscale_ahead, NULL, 1, 0},
    {"cand vscale perm", c_vscale_perm, NULL, 1, 0},
    {"cand vscale ahead pf1row", c_vscale_ahead_pf1, NULL, 1, 0},
    {"cand vscale ahead pf2row", c_vscale_ahead_pf2, NULL, 1, 0},
    {"cand vscale ahead pf4row", c_vscale_ahead_pf4, NULL, 1, 0},
    {"cand vscale ahead pf8row", c_vscale_ahead_pf8, NULL, 1, 0},
    {"cand x1 ahead fmatab", NULL, c_x1_ahead_fmatab, 1, 1},
    {"cand x2 prepass", NULL, c_x2_pre, 1, 2},
    {"cand x2 ahead", NULL, c_x2_ahead, 1, 2},
    {"cand x2 ahead fmatab", NULL, c_x2_ahead_fmatab, 1, 2},
    {"cand x4 ahead", NULL, c_x4_ahead, 1, 4},
    {"cand x4 ahead fmatab", NULL, c_x4_ahead_fmatab, 1, 4},
    {"cand x2 prepass pf2row", NULL, c_x2_pre_pf2, 1, 2},
    {"cand x2 prepass pf4row", NULL, c_x2_pre_pf4, 1, 2},
    {"cand x2 ahead pf2row", NULL, c_x2_ahead_pf2, 1, 2},
    {"cand x2 ahead pf4row", NULL, c_x2_ahead_pf4, 1, 2},
    {"fma (q59, not exact)", k_fma, NULL, 0, 0},
};
#define N_KERNELS ((int)(sizeof KERNELS / sizeof KERNELS[0]))

static int selected(int k) {
    return !g_only || k == 0 || strstr(KERNELS[k].name, g_only) != NULL;
}

/* ---- data ------------------------------------------------------------------------------ */

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(g_rng >> 33);
}

/* plausible Q4_K blocks: small positive d and dmin, random scale bytes and quants */
static void fill_q4_k(unsigned char *row) {
    for (int b = 0; b < NB; b++) {
        unsigned char *blk = row + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        uint16_t d = (uint16_t)(0x2C00 | (rnd() & 0x3FF)), dmin = (uint16_t)(0x2800 | (rnd() & 0x3FF));
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (int i = 4; i < TR_Q4_K_BLOCK_BYTES; i++) blk[i] = (unsigned char)rnd();
    }
}

static void decode_all(const unsigned char *rows, float *pre_scales, float *pre_tables) {
    for (int r = 0; r < L1_ROWS; r++)
        for (int b = 0; b < NB; b++) {
            const unsigned char *blk = rows + ((size_t)r * NB + b) * TR_Q4_K_BLOCK_BYTES;
            float *ps = pre_scales + ((size_t)r * NB + b) * 16;
            float *pt = pre_tables + ((size_t)r * NB + b) * 8 * 16;
            tr_q4_k_scales(blk, ps, ps + 8);
            for (int s = 0; s < 8; s++)
                for (int q = 0; q < 16; q++) pt[16 * s + q] = ps[s] * (float)q - ps[8 + s];
        }
}

/* ns a row */
static double time_kernel(const kernel_case *k, const l1_job *j, long calls) {
    float out[L1_ROWS];
    double t0 = tr_time_sec();
    if (k->fn2)
        for (long c = 0; c < calls; c += k->nrows) {
            int r = (int)(c & (L1_ROWS - k->nrows));
            k->fn2(j, r, out + r);
        }
    else
        for (long c = 0; c < calls; c++) out[c & (L1_ROWS - 1)] = k->fn(j, (int)(c & (L1_ROWS - 1)));
    double t1 = tr_time_sec();
    g_sink = out[0] + out[L1_ROWS - 1];
    long rows = k->fn2 ? (calls + k->nrows - 1) / k->nrows * k->nrows : calls;
    return (t1 - t0) * 1e9 / (double)rows;
}

/* row r of kernel k (the pair's call for a two-row kernel) */
static float kernel_row(const kernel_case *k, const l1_job *j, int r) {
    if (!k->fn2) return k->fn(j, r);
    float out[4];
    k->fn2(j, r - r % k->nrows, out);
    return out[r % k->nrows];
}

static double time_op(const op_case *o, const void *m, int64_t iters) {
    double t0 = tr_time_sec();
    o->fn(iters, m);
    double t1 = tr_time_sec();
    return (t1 - t0) * 1e9 / ((double)iters * o->ops);
}

/* ---- SMT: two threads on the two logical processors of one core ------------------------ */

/* Each worker pins itself (worker 0 and 1: the two siblings of slot 0's core, or the first
 * processors of slots 0 and 1, two cores) and times `calls` rows of the kernel on the shared
 * L1 data; the core's cost of a row is the slower thread's time over the rows both did. */
typedef struct {
    const kernel_case *k;
    const l1_job *j;
    long calls;
    int siblings;
    double sec[2];
    int pinned[2];
} smt_job;

static void smt_body(void *ctx, int64_t begin, int64_t end, int worker) {
    smt_job *s = (smt_job *)ctx;
    const tr_cpu_info *cpu = tr_cpu();
    const tr_cpu_slot *sl = &cpu->slot[s->siblings ? 0 : worker];
    unsigned short lc = s->siblings ? sl->core[worker] : sl->core[0];
    s->pinned[worker] = tr_thread_pin(sl->group, &lc, 1, NULL) == 0;
    time_kernel(s->k, s->j, s->calls / 8);    /* warm */
    double t0 = tr_time_sec();
    time_kernel(s->k, s->j, s->calls);
    s->sec[worker] = tr_time_sec() - t0;
    (void)begin;
    (void)end;
}

static void smt_lines(const l1_job *job, int runs, double run_ms) {
    const tr_cpu_info *cpu = tr_cpu();
    if (cpu->n_slots < 2 || cpu->slot[0].n_core < 2) {
        printf("smt: no SMT sibling known on slot 0 (n_slots %d): skipped\n", cpu->n_slots);
        return;
    }
    tr_pool *pool = tr_pool_create(2);
    printf("smt: siblings %u,%u of one core; two cores %u,%u\n", cpu->slot[0].core[0], cpu->slot[0].core[1],
           cpu->slot[0].core[0], cpu->slot[1].core[0]);
    printf("%-28s %10s %10s %10s  (ns a row per core; spread)\n", "kernel", "1 thread", "2 siblings", "2 cores");
    for (int k = 0; k < N_KERNELS; k++) {
        if (strcmp(KERNELS[k].name, "tier") != 0 && strncmp(KERNELS[k].name, "cand", 4) != 0) continue;
        double one[MAX_RUNS], sib[MAX_RUNS], two[MAX_RUNS];
        long calls = (long)(run_ms * 1e6 / 100.0);
        for (int run = 0; run < runs; run++) {
            unsigned short lc = cpu->slot[0].core[0];
            tr_thread_pin(cpu->slot[0].group, &lc, 1, NULL);
            one[run] = time_kernel(&KERNELS[k], job, calls);
            smt_job s = {&KERNELS[k], job, calls, 1, {0, 0}, {0, 0}};
            tr_parallel_for(pool, 2, 1, smt_body, &s);
            if (!s.pinned[0] || !s.pinned[1]) { printf("smt: pinning failed: skipped\n"); tr_pool_destroy(pool); return; }
            sib[run] = (s.sec[0] > s.sec[1] ? s.sec[0] : s.sec[1]) * 1e9 / (2.0 * (double)calls);
            s.siblings = 0;
            tr_parallel_for(pool, 2, 1, smt_body, &s);
            two[run] = (s.sec[0] > s.sec[1] ? s.sec[0] : s.sec[1]) * 1e9 / (2.0 * (double)calls);
        }
        double s1, s2, s3;
        double m1 = median_spread(one, runs, &s1), m2 = median_spread(sib, runs, &s2), m3 = median_spread(two, runs, &s3);
        printf("%-28s %10.1f %10.1f %10.1f  (%.0f%% %.0f%% %.0f%%)\n", KERNELS[k].name, m1, m2, m3, s1 * 100, s2 * 100,
               s3 * 100);
    }
    tr_pool_destroy(pool);
}

/* ---- from RAM: the kernels as the decode runs them -------------------------------------- */

/* --ram T: ~1 GiB of Q4_K rows of 2048 (1024 random rows repeated: equal bytes change nothing for
 * the memory), far beyond the 2 x 32 MiB of L3, split in T contiguous chunks over a pool of T
 * threads on distinct cores (tr_matmul's split), each chunk's rows run through a kernel one call
 * after another; "read" is a plain read of the same bytes, the ceiling. GB/s of weight bytes,
 * kernels in turn, R rounds of P timed passes, median and spread. --ram-smt C: 2C threads on C
 * cores, worker w pinned to the logical processor w % 2 of slot w / 2 (native only: in the
 * container the logical processors are virtual). Every kernel's outputs are compared with the
 * tier's on the first pass. */
#define RAM_UNIT_ROWS 1024
#define RAM_PASSES 2
__attribute__((target("avx512f"))) static float read_rows(const unsigned char *p, size_t bytes) {
    __m512i a = _mm512_setzero_si512(), b = a, c = a, d = a;
    size_t i = 0;
    for (; i + 256 <= bytes; i += 256) {
        a = _mm512_xor_si512(a, _mm512_loadu_si512((const void *)(p + i)));
        b = _mm512_xor_si512(b, _mm512_loadu_si512((const void *)(p + i + 64)));
        c = _mm512_xor_si512(c, _mm512_loadu_si512((const void *)(p + i + 128)));
        d = _mm512_xor_si512(d, _mm512_loadu_si512((const void *)(p + i + 192)));
    }
    a = _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_xor_si512(c, d));
    return (float)_mm512_reduce_add_epi32(a);
}

typedef struct {
    const kernel_case *k;    /* NULL: the plain read */
    l1_job j;
    float *out;
    int smt;
    int pin_fail;
    int dyn;                 /* > 0: rows claimed dyn at a time from `next` (ggml's way), else chunks */
    int64_t n_rows;
    atomic_llong next;
    volatile float sink[64];
} ram_job;

static void ram_rows(ram_job *m, int64_t r0, int64_t r1, int worker) {
    const kernel_case *k = m->k;
    if (!k) {
        size_t rb = (size_t)NB * TR_Q4_K_BLOCK_BYTES;
        m->sink[worker & 63] = read_rows(m->j.rows + (size_t)r0 * rb, (size_t)(r1 - r0) * rb);
    } else if (k->fn2) {
        for (int64_t r = r0; r < r1; r += k->nrows) k->fn2(&m->j, (int)r, m->out + r);
    } else {
        for (int64_t r = r0; r < r1; r++) m->out[r] = k->fn(&m->j, (int)r);
    }
}


static void ram_body(void *ctx, int64_t begin, int64_t end, int worker) {
    ram_job *m = (ram_job *)ctx;
    if (m->smt) {
        const tr_cpu_slot *sl = &tr_cpu()->slot[worker / 2];
        unsigned short lc = sl->core[worker % 2 < sl->n_core ? worker % 2 : 0];
        if (tr_thread_pin(sl->group, &lc, 1, NULL) != 0) m->pin_fail = 1;
    }
    if (m->dyn > 0) {   /* one index per worker; the rows go to whoever asks next */
        for (;;) {
            int64_t r0 = atomic_fetch_add(&m->next, m->dyn);
            if (r0 >= m->n_rows) return;
            ram_rows(m, r0, r0 + m->dyn < m->n_rows ? r0 + m->dyn : m->n_rows, worker);
        }
    }
    ram_rows(m, begin * 4, end * 4, worker);   /* the index space counts groups of 4 rows */
}

static int ram_lines(int threads, int smt, int rounds, int dyn) {
    const int64_t unit_bytes = (int64_t)RAM_UNIT_ROWS * NB * TR_Q4_K_BLOCK_BYTES;
    const int64_t units = ((int64_t)1 << 30) / unit_bytes;
    const int64_t n_rows = units * RAM_UNIT_ROWS;
    const size_t bytes = (size_t)n_rows * NB * TR_Q4_K_BLOCK_BYTES;
    unsigned char *w = (unsigned char *)malloc(bytes);
    float *out = (float *)malloc((size_t)n_rows * sizeof(float));
    float *ref = (float *)malloc((size_t)n_rows * sizeof(float));
    static float x[COLS];
    if (!w || !out || !ref) {
        printf("ram: no memory for %zu bytes\n", bytes);
        free(w), free(out), free(ref);
        return 1;
    }
    for (int r = 0; r < RAM_UNIT_ROWS; r++) fill_q4_k(w + (size_t)r * NB * TR_Q4_K_BLOCK_BYTES);
    for (int64_t u = 1; u < units; u++) memcpy(w + (size_t)u * unit_bytes, w, (size_t)unit_bytes);
    for (int i = 0; i < COLS; i++) x[i] = (float)rnd() * 0x1p-30f - 1.0f;
    int n_threads = smt ? 2 * threads : threads;
    tr_pool *pool = tr_pool_create(n_threads);
    /* each kernel (and the read) with one contiguous chunk a thread, and with --dyn also with
     * rows claimed dyn at a time: ks[i] the kernel (-1 the read), dy[i] the claim size or 0 */
    int ks[2 * (N_KERNELS + 1)], dy[2 * (N_KERNELS + 1)], nk = 0;
    for (int k = -1; k < N_KERNELS; k++) {
        if (k >= 0 && !(KERNELS[k].exact && KERNELS[k].fn != k_full && KERNELS[k].fn != k_f16c &&
                        KERNELS[k].fn != k_prescale && KERNELS[k].fn != k_pretable && selected(k)))
            continue;
        ks[nk] = k, dy[nk++] = 0;
        if (dyn > 0) ks[nk] = k, dy[nk++] = dyn;
    }
    static double gbs[2 * (N_KERNELS + 1)][MAX_RUNS * RAM_PASSES];
    ram_job m;
    memset(&m, 0, sizeof m);
    m.j.rows = w;
    m.j.x = x;
    m.out = out;
    m.smt = smt;
    m.n_rows = n_rows;
    int bad = 0;
    for (int round = 0; round < rounds; round++) {
        for (int i = 0; i < nk; i++) {
            m.k = ks[i] < 0 ? NULL : &KERNELS[ks[i]];
            m.dyn = dy[i];
            atomic_store(&m.next, 0);
            tr_parallel_for(pool, dy[i] > 0 ? n_threads : n_rows / 4, 1, ram_body, &m);   /* untimed */
            if (round == 0 && ks[i] >= 0) {
                if (ks[i] == 0 && dy[i] == 0) memcpy(ref, out, (size_t)n_rows * sizeof(float));
                else if (memcmp(ref, out, (size_t)n_rows * sizeof(float)) != 0) {
                    printf("ram: %s differs from the tier\n", KERNELS[ks[i]].name);
                    bad = 1;
                }
            }
            for (int pass = 0; pass < RAM_PASSES; pass++) {
                atomic_store(&m.next, 0);
                double t0 = tr_time_sec();
                tr_parallel_for(pool, dy[i] > 0 ? n_threads : n_rows / 4, 1, ram_body, &m);
                gbs[i][round * RAM_PASSES + pass] = (double)bytes / ((tr_time_sec() - t0) * 1e9);
            }
        }
    }
    tr_pool_destroy(pool);
    if (m.pin_fail) printf("ram: pinning to SMT siblings failed: the smt lines are not what they say\n");
    printf("ram: %d threads%s, %.2f GiB of Q4_K rows of %d, %d rounds x %d passes, %s\n", n_threads,
           smt ? " (two on each of the cores)" : "", (double)bytes / (1 << 30), COLS, rounds, RAM_PASSES,
           dyn > 0 ? "chunks, and rows claimed dynamically (dyn)" : "one contiguous chunk a thread");
    printf("%-28s %6s %8s %7s %9s\n", "kernel", "rows", "GB/s", "spread", "vs tier");
    double med[2 * (N_KERNELS + 1)], sp[2 * (N_KERNELS + 1)], tier = 1;
    for (int i = 0; i < nk; i++) {
        med[i] = median_spread(gbs[i], rounds * RAM_PASSES, &sp[i]);
        if (ks[i] == 0 && dy[i] == 0) tier = med[i];
    }
    for (int i = 0; i < nk; i++)
        printf("%-28s %6s %8.1f %6.1f%% %8.2fx\n", ks[i] < 0 ? "read (ceiling)" : KERNELS[ks[i]].name,
               dy[i] > 0 ? "dyn" : "chunk", med[i], sp[i] * 100, med[i] / tier);
    free(w), free(out), free(ref);
    return bad || m.pin_fail;
}

int main(int argc, char **argv) {
    int runs = 11, smt = 0, ram = 0, ram_smt = 0, dyn = 0;
    double run_ms = 5;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ms") == 0 && i + 1 < argc) run_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--smt") == 0) smt = 1;
        else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) g_only = argv[++i];
        else if (strcmp(argv[i], "--ram") == 0 && i + 1 < argc) ram = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ram-smt") == 0 && i + 1 < argc) ram = ram_smt = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dyn") == 0 && i + 1 < argc) dyn = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: bench_q4k_genome [--runs N] [--ms M] [--smt] [--only S] [--ram T | --ram-smt C] [--dyn G]\n");
            return 2;
        }
    }
    if (runs < 3) runs = 3;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
    const tr_cpu_info *cpu = tr_cpu();
    tr_kernels_init();
    const tr_kernels *t512 = tr_kernels_tier("avx512");
    if (!cpu->avx512f || !cpu->f16c || !t512) {
        printf("bench_q4k_genome: no AVX-512 tier on this CPU: nothing to measure\n");
        return 0;
    }
    char line[256];
    tr_cpu_describe(cpu, line, (int)sizeof line);
    printf("# %s\n# one core, %d runs of ~%.0f ms, median and spread (max-min)/median\n", line, runs, run_ms);
    tr_pool *pool = tr_pool_create(1);    /* pins this thread to the first slot */
    g_tier_dot = t512->dot_row[TR_TYPE_Q4_K];

    static unsigned char rows[L1_ROWS * NB * TR_Q4_K_BLOCK_BYTES];
    static float x[COLS], pre_scales[L1_ROWS * NB * 16], pre_tables[L1_ROWS * NB * 8 * 16];
    static unsigned char mem[4096] __attribute__((aligned(64)));
    for (int r = 0; r < L1_ROWS; r++) fill_q4_k(rows + (size_t)r * NB * TR_Q4_K_BLOCK_BYTES);
    for (int i = 0; i < COLS; i++) x[i] = (float)rnd() * 0x1p-30f - 1.0f;
    for (int i = 0; i < 4096; i++) mem[i] = (unsigned char)(i * 37);
    memset(mem, 0, 16);   /* the broadcast sources: zeros, no denormal and no NaN */
    decode_all(rows, pre_scales, pre_tables);
    l1_job job = {rows, x, pre_scales, pre_tables};

    /* the exact lines against scalar, row by row */
    float (*scalar_dot)(const void *, const float *, int64_t) = tr_kernels_scalar()->dot_row[TR_TYPE_Q4_K];
    int bad = 0;
    for (int k = 0; k < N_KERNELS; k++) {
        if (!KERNELS[k].exact) continue;
        for (int r = 0; r < L1_ROWS; r++) {
            float want = scalar_dot(rows + (size_t)r * NB * TR_Q4_K_BLOCK_BYTES, x, COLS);
            float got = kernel_row(&KERNELS[k], &job, r);
            g_exact_checked++;
            if (memcmp(&want, &got, sizeof want) != 0) {
                printf("MISMATCH %s row %d: %.9g scalar %.9g\n", KERNELS[k].name, r, (double)got, (double)want);
                bad = 1;
            }
        }
    }
    if (bad || g_exact_checked == 0) {
        printf("bench_q4k_genome: exactness check failed (%lld rows compared)\n", (long long)g_exact_checked);
        tr_pool_destroy(pool);
        return 1;
    }
    printf("# exact lines equal scalar bit for bit: %lld rows compared\n", (long long)g_exact_checked);
    if (ram > 0) {
        tr_pool_destroy(pool);
        return ram_lines(ram, ram_smt > 0, runs, dyn);
    }

    /* calls and iterations for ~run_ms each */
    long kcalls[N_KERNELS];
    int n_ops = g_only ? 0 : N_OPS;
    for (int k = 0; k < N_KERNELS; k++) {
        kcalls[k] = 0;
        if (!selected(k)) continue;
        double ns = time_kernel(&KERNELS[k], &job, 2000);
        ns = time_kernel(&KERNELS[k], &job, 2000);
        kcalls[k] = (long)(run_ms * 1e6 / ns) + 1;
    }
    int64_t oiters[N_OPS];
    for (int o = 0; o < n_ops; o++) {
        double ns = time_op(&OPS[o], mem, 20000);
        oiters[o] = (int64_t)(run_ms * 1e6 / (ns * OPS[o].ops)) + 1;
    }

    static double kns[N_KERNELS][MAX_RUNS], kcyc[N_KERNELS][MAX_RUNS], ocyc[N_OPS][MAX_RUNS], ghz[MAX_RUNS];
    for (int run = 0; run < runs; run++) {
        double g1 = clock_ghz();
        for (int k = 0; k < N_KERNELS; k++)
            if (kcalls[k]) kns[k][run] = time_kernel(&KERNELS[k], &job, kcalls[k]);
        for (int o = 0; o < n_ops; o++) ocyc[o][run] = time_op(&OPS[o], mem, oiters[o]);
        double g2 = clock_ghz();
        ghz[run] = g2 < g1 ? g2 : g1;   /* the lower of the round's two clocks: cycles not overcounted */
        for (int k = 0; k < N_KERNELS; k++) kcyc[k][run] = kns[k][run] * ghz[run];
        for (int o = 0; o < n_ops; o++) ocyc[o][run] *= ghz[run];
    }
    double sp;
    double g = median_spread(ghz, runs, &sp);
    printf("clock %.3f GHz  spread %.1f%%\n", g, sp * 100);
    printf("%-28s %8s %7s\n", "op", "cycles", "spread");
    for (int o = 0; o < n_ops; o++) {
        double c = median_spread(ocyc[o], runs, &sp);
        printf("%-28s %8.3f %6.1f%%\n", OPS[o].name, c, sp * 100);
    }
    printf("%-28s %8s %8s %7s\n", "kernel (row of 2048)", "ns", "cycles", "spread");
    for (int k = 0; k < N_KERNELS; k++) {
        if (!kcalls[k]) continue;
        double spc;
        double ns = median_spread(kns[k], runs, &sp);
        double cy = median_spread(kcyc[k], runs, &spc);
        printf("%-28s %8.1f %8.0f %6.1f%%\n", KERNELS[k].name, ns, cy, sp * 100);
    }
    tr_pool_destroy(pool);
    if (smt) smt_lines(&job, runs, run_ms);
    return 0;
}

#else
int main(void) {
    printf("bench_q4k_genome: x86-64 only\n");
    (void)cmp_double;
    (void)median_spread;
    (void)g_exact_checked;
    return 0;
}
#endif
