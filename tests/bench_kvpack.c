/* bench_kvpack.c — premise benchmark: decode attention over a lossless 28-bit packed KV cache,
 * bit-identical to the F32 one (docs/MEASUREMENTS.md "Decode a contesto lungo": the CPU decode at
 * context 2048 spends 11.3 of 36.8 ms/token reading 518 MiB of F32 KV; sign+exponent carry the
 * entropy, the mantissa's low bits carry none of it worth keeping. A fixed 28-bit layout reads
 * ~12% fewer bytes; whether decoding it stays under the memory time it replaces is what this file
 * measures, on real K/V/Q dumps, not on synthetic noise.
 *
 * ---- packed format (per layer, head, and K or V stream; positions in blocks of TR_ATTN_BLOCK;
 * the last partial block stays plain F32) ----
 *   For each value's float bits b: sign s = b>>31, exponent e = (b>>23)&0xFF, mantissa m = b&0x7FFFFF.
 *   Per (block, channel d): base[d] = smallest e over the block's 64 values with 1<=e<=254, else 0.
 *   4-bit code per value: 0 if e==0 (zero/subnormal, exact from sign+mantissa alone); e-base+1 if
 *   1<=e<=254 and that fits in 1..14; 15 = escape (inf, NaN, or an exponent too far from base) —
 *   the value's full 32 bits go in the block's exception list, in the (position, channel) raster
 *   order the planes already use, so decoding recovers them by walking the same order and
 *   consuming exceptions as code 15 is seen (no separate index needed).
 *   Planes per block: lo16[64][128] = m&0xFFFF, hi8[64][128] = s<<7|m>>16, code[64][128] as 4-bit
 *   nibbles (2/byte), base[128], exception count + list. 16+8+4 = 28 bits/value, +base and
 *   exceptions as headers.
 *   Decode: code 1..14 -> s<<31 | (base+code-1)<<23 | m; code 0 -> s<<31 | m; code 15 -> the list.
 *
 * ---- kernel ----
 * The attention gives the SAME bits as tr_attention_group on plain F32 rows: exactness comes from
 * decoding TR_ATTN_X positions of K (or V) into a small F32 scratch, then handing that scratch to
 * the active tier's own dot_f32_x4 / axpy_f32_x4 (tr_kernels_get()), in exactly the block/x4/tail
 * order tr_attention_group uses for one query (src/kernels/kernels.c, tr_attention_group). This
 * file only ever calls it with n_q == 1 (a decode token): first_n_pos == n_pos, j is always 0, so
 * kvpack_attention_decode below is that specialization, written out rather than parameterized.
 * TR_ATTN_BLOCK (64) is reused as the pack block size on purpose: an attention block then maps
 * entirely to either the packed region or the plain tail (never straddles), so the packed/tail
 * branch is taken once per block, not once per position.
 * The decoder itself (reconstructing bits from code/lo16/hi8/base) has scalar, AVX2 and AVX-512
 * variants dispatched at runtime like the engine's own kernels (src/kernels/kernels_x86.c: target
 * attributes, tr_cpu() feature bits, TR_CPU_MAX to cap them for testing). Nibble unpacking (4-bit
 * codes, 2/byte, the smallest of the four planes) is a shared scalar step; the bit-reconstruction
 * itself is real per-tier SIMD. All three are bit-identical: they are pure integer bit arithmetic
 * on the reconstructed exponent, never a rounding path.
 *
 * MEASURE 1 (round trip), 2 (bits, with a mutation shown red then reverted) and 3 (time, F32 vs
 * packed) are three phases of this binary; see main() / usage(). Real dumps: KVPACK_PROBE_DIR
 * (env) or the default below, six runs (prose/code/synth x 2k/4k), per (layer, head): k_/v_ =
 * N x 128 f32 rows, q_ = records of (int64 n_pos, 128 f32). A record's n_pos can exceed the dump's
 * N by one (the capture's last step, whose own K/V never got written to the snapshot) — those
 * records are skipped, per (layer, head), wherever n_pos > N. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "test.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define KVPACK_X86 1
#define KVPACK_TARGET_AVX2 __attribute__((target("avx2")))
#define KVPACK_TARGET_AVX512 __attribute__((target("avx512f")))
#endif

#define HEAD_DIM 128
#define N_LAYER 16
#define N_HEAD 16
#define MAX_N_POS 4096
#define MAX_TIME_RUNS 64

_Static_assert(TR_ATTN_BLOCK == 64, "kvpack blocks must align with the engine's attention blocks");
_Static_assert(TR_ATTN_X == 4, "kvpack decodes TR_ATTN_X positions at a time for dot/axpy _x4");
_Static_assert(HEAD_DIM % TR_ATTN_BLOCK == 0 || HEAD_DIM > 0, "sanity");

/* Deliberately wrong under -DKVPACK_MUTATE_BASE_OFF_BY_ONE, in every tier alike: proves MEASURE 2
 * goes red when the decoder is broken (docs: "a test is seen red at least once"). Off by default. */
#ifdef KVPACK_MUTATE_BASE_OFF_BY_ONE
#define KVPACK_BASE_BIAS 2
#else
#define KVPACK_BASE_BIAS 1
#endif

/* ---- bits, no strict-aliasing games -------------------------------------------------------- */

static inline uint32_t f2b(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static inline float b2f(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

static uint64_t rng_next(uint64_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s * 0x2545F4914F6CDD1Dull;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ---- packed stream: one (layer, head, K-or-V) --------------------------------------------- */

typedef struct {
    int64_t n_pos;
    int64_t n_full_blocks;   /* n_pos / TR_ATTN_BLOCK */
    int64_t tail_rows;       /* n_pos % TR_ATTN_BLOCK, stored as plain f32 */
    uint8_t *base;           /* [n_full_blocks][HEAD_DIM] */
    uint8_t *code;           /* [n_full_blocks][TR_ATTN_BLOCK][HEAD_DIM/2], 2 nibbles/byte */
    uint16_t *lo16;          /* [n_full_blocks][TR_ATTN_BLOCK][HEAD_DIM] */
    uint8_t *hi8;            /* [n_full_blocks][TR_ATTN_BLOCK][HEAD_DIM] */
    uint32_t *exc_off;       /* [n_full_blocks+1], prefix sums; NULL if n_full_blocks == 0 */
    uint32_t *exc_bits;      /* [exc_off[n_full_blocks]] raw bits, escapes in raster order */
    float *tail;             /* [tail_rows][HEAD_DIM], NULL if tail_rows == 0 */
} kv_stream_packed;

static int kvpack_encode(const float *f32, int64_t n_pos, kv_stream_packed *out) {
    memset(out, 0, sizeof *out);
    out->n_pos = n_pos;
    out->n_full_blocks = n_pos / TR_ATTN_BLOCK;
    out->tail_rows = n_pos % TR_ATTN_BLOCK;
    int64_t nb = out->n_full_blocks;

    if (nb > 0) {
        out->base = (uint8_t *)tr_alloc_aligned((size_t)nb * HEAD_DIM, 64);
        out->code = (uint8_t *)tr_alloc_aligned((size_t)nb * TR_ATTN_BLOCK * (HEAD_DIM / 2), 64);
        out->lo16 = (uint16_t *)tr_alloc_aligned((size_t)nb * TR_ATTN_BLOCK * HEAD_DIM * sizeof(uint16_t), 64);
        out->hi8 = (uint8_t *)tr_alloc_aligned((size_t)nb * TR_ATTN_BLOCK * HEAD_DIM, 64);
        out->exc_off = (uint32_t *)tr_alloc_aligned((size_t)(nb + 1) * sizeof(uint32_t), 64);
        if (!out->base || !out->code || !out->lo16 || !out->hi8 || !out->exc_off) return -1;
        memset(out->exc_off, 0, (size_t)(nb + 1) * sizeof(uint32_t));
    }

    /* pass 1: base[], code[], lo16[], hi8[]; tally escapes per block into exc_off[b+1] */
    for (int64_t b = 0; b < nb; b++) {
        const float *blockf = f32 + (size_t)b * TR_ATTN_BLOCK * HEAD_DIM;
        uint8_t *baseb = out->base + (size_t)b * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            uint8_t mn = 0;
            int have = 0;
            for (int r = 0; r < TR_ATTN_BLOCK; r++) {
                uint32_t bits = f2b(blockf[(size_t)r * HEAD_DIM + d]);
                uint8_t e = (uint8_t)((bits >> 23) & 0xFF);
                if (e >= 1 && e <= 254 && (!have || e < mn)) { mn = e; have = 1; }
            }
            baseb[d] = have ? mn : 0;
        }
        uint32_t escapes = 0;
        uint8_t *codeb = out->code + (size_t)b * TR_ATTN_BLOCK * (HEAD_DIM / 2);
        uint16_t *lo16b = out->lo16 + (size_t)b * TR_ATTN_BLOCK * HEAD_DIM;
        uint8_t *hi8b = out->hi8 + (size_t)b * TR_ATTN_BLOCK * HEAD_DIM;
        for (int r = 0; r < TR_ATTN_BLOCK; r++) {
            const float *row = blockf + (size_t)r * HEAD_DIM;
            uint8_t *coderow = codeb + (size_t)r * (HEAD_DIM / 2);
            uint16_t *lo16row = lo16b + (size_t)r * HEAD_DIM;
            uint8_t *hi8row = hi8b + (size_t)r * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                uint32_t bits = f2b(row[d]);
                uint32_t s = bits >> 31;
                uint8_t e = (uint8_t)((bits >> 23) & 0xFF);
                uint32_t m = bits & 0x7FFFFFu;
                lo16row[d] = (uint16_t)(m & 0xFFFFu);
                hi8row[d] = (uint8_t)((s << 7) | (m >> 16));
                uint8_t code;
                if (e == 0) {
                    code = 0;
                } else if (e <= 254) {
                    uint32_t diff = (uint32_t)e - baseb[d] + 1;
                    code = (diff >= 1 && diff <= 14) ? (uint8_t)diff : 15;
                } else {
                    code = 15; /* inf or NaN */
                }
                if (code == 15) escapes++;
                if (d & 1) coderow[d >> 1] = (uint8_t)(coderow[d >> 1] | (uint8_t)(code << 4));
                else coderow[d >> 1] = code;
            }
        }
        out->exc_off[b + 1] = escapes;
    }
    for (int64_t b = 0; b < nb; b++) out->exc_off[b + 1] += out->exc_off[b];
    uint32_t total_exc = nb > 0 ? out->exc_off[nb] : 0;
    if (total_exc > 0) {
        out->exc_bits = (uint32_t *)tr_alloc_aligned((size_t)total_exc * sizeof(uint32_t), 64);
        if (!out->exc_bits) return -1;
    }

    /* pass 2: fill exceptions, reusing the code[] already written (no recompute) */
    for (int64_t b = 0; b < nb; b++) {
        const float *blockf = f32 + (size_t)b * TR_ATTN_BLOCK * HEAD_DIM;
        uint8_t *codeb = out->code + (size_t)b * TR_ATTN_BLOCK * (HEAD_DIM / 2);
        uint32_t cursor = out->exc_off[b];
        for (int r = 0; r < TR_ATTN_BLOCK; r++) {
            const float *row = blockf + (size_t)r * HEAD_DIM;
            uint8_t *coderow = codeb + (size_t)r * (HEAD_DIM / 2);
            for (int d = 0; d < HEAD_DIM; d++) {
                uint8_t byte = coderow[d >> 1];
                uint8_t code = (d & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
                if (code == 15) out->exc_bits[cursor++] = f2b(row[d]);
            }
        }
    }

    if (out->tail_rows > 0) {
        out->tail = (float *)tr_alloc_aligned((size_t)out->tail_rows * HEAD_DIM * sizeof(float), 64);
        if (!out->tail) return -1;
        memcpy(out->tail, f32 + (size_t)nb * TR_ATTN_BLOCK * HEAD_DIM, (size_t)out->tail_rows * HEAD_DIM * sizeof(float));
    }
    return 0;
}

static void kvpack_free(kv_stream_packed *s) {
    tr_free_aligned(s->base);
    tr_free_aligned(s->code);
    tr_free_aligned(s->lo16);
    tr_free_aligned(s->hi8);
    tr_free_aligned(s->exc_off);
    tr_free_aligned(s->exc_bits);
    tr_free_aligned(s->tail);
    memset(s, 0, sizeof *s);
}

/* Bytes actually stored (headers and exceptions included): what a "bytes/value" report should use,
 * not a flat 3.5 assumption. */
static uint64_t kvpack_bytes(const kv_stream_packed *s) {
    int64_t nb = s->n_full_blocks;
    uint64_t total = (uint64_t)nb * HEAD_DIM                                  /* base */
                    + (uint64_t)nb * TR_ATTN_BLOCK * (HEAD_DIM / 2)           /* code */
                    + (uint64_t)nb * TR_ATTN_BLOCK * HEAD_DIM * sizeof(uint16_t) /* lo16 */
                    + (uint64_t)nb * TR_ATTN_BLOCK * HEAD_DIM;                /* hi8 */
    uint64_t exc = (nb > 0 && s->exc_off) ? s->exc_off[nb] : 0;
    total += exc * sizeof(uint32_t);
    total += (uint64_t)s->tail_rows * HEAD_DIM * sizeof(float);
    return total;
}

/* ---- decode: scalar (the definition), AVX2, AVX-512, dispatched like the engine's kernels --- */

static inline void unpack_nibbles_128(const uint8_t *packed, uint8_t out[HEAD_DIM]) {
    for (int i = 0; i < HEAD_DIM / 2; i++) {
        uint8_t byte = packed[i];
        out[2 * i] = (uint8_t)(byte & 0x0F);
        out[2 * i + 1] = (uint8_t)(byte >> 4);
    }
}

/* Consumes escapes from exc_row (already offset to this block) in increasing channel order,
 * matching the raster order they were appended in during encode. Shared by the vectorized tiers;
 * scalar patches inline instead (see below), same result either way. */
static inline void kvpack_patch_escapes(const uint8_t codebuf[HEAD_DIM], const uint32_t *exc_row, int64_t *cursor,
                                        uint32_t bits[HEAD_DIM]) {
    int64_t c = *cursor;
    for (int d = 0; d < HEAD_DIM; d++)
        if (codebuf[d] == 15) bits[d] = exc_row[c++];
    *cursor = c;
}

/* Definition: one channel at a time, in order. code 0 -> s<<31|m; 1..14 -> s<<31|(base+code-1)<<23|m;
 * 15 -> next value off the block's exception list. */
static void kvpack_decode_row_scalar(const kv_stream_packed *s, int64_t block, int64_t blockpos, int64_t *cursor,
                                     float *out) {
    const uint8_t *baserow = s->base + (size_t)block * HEAD_DIM;
    const uint8_t *coderow = s->code + ((size_t)block * TR_ATTN_BLOCK + blockpos) * (HEAD_DIM / 2);
    const uint16_t *lo16row = s->lo16 + ((size_t)block * TR_ATTN_BLOCK + blockpos) * HEAD_DIM;
    const uint8_t *hi8row = s->hi8 + ((size_t)block * TR_ATTN_BLOCK + blockpos) * HEAD_DIM;
    const uint32_t *excrow = s->exc_bits + s->exc_off[block];
    int64_t c = *cursor;
    uint32_t bits[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        uint8_t byte = coderow[d >> 1];
        uint8_t code = (d & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
        uint32_t m = (uint32_t)lo16row[d] | ((uint32_t)(hi8row[d] & 0x7F) << 16);
        uint32_t sbit = (uint32_t)(hi8row[d] >> 7) << 31;
        uint32_t b;
        if (code == 15) b = excrow[c++];
        else if (code == 0) b = sbit | m;
        else b = sbit | (((uint32_t)baserow[d] + code - KVPACK_BASE_BIAS) << 23) | m;
        bits[d] = b;
    }
    *cursor = c;
    memcpy(out, bits, sizeof bits);
}

#ifdef KVPACK_X86

KVPACK_TARGET_AVX2
static void kvpack_decode_row_avx2(const kv_stream_packed *s, int64_t block, int64_t blockpos, int64_t *cursor,
                                   float *out) {
    const uint8_t *baserow = s->base + (size_t)block * HEAD_DIM;
    const uint8_t *coderow = s->code + ((size_t)block * TR_ATTN_BLOCK + blockpos) * (HEAD_DIM / 2);
    const uint16_t *lo16row = s->lo16 + ((size_t)block * TR_ATTN_BLOCK + blockpos) * HEAD_DIM;
    const uint8_t *hi8row = s->hi8 + ((size_t)block * TR_ATTN_BLOCK + blockpos) * HEAD_DIM;
    const uint32_t *excrow = s->exc_bits + s->exc_off[block];

    uint8_t codebuf[HEAD_DIM];
    unpack_nibbles_128(coderow, codebuf);

    uint32_t bits[HEAD_DIM];
    const __m256i zero = _mm256_setzero_si256();
    const __m256i mask7f = _mm256_set1_epi32(0x7F);
    const __m256i bias = _mm256_set1_epi32(KVPACK_BASE_BIAS);
    for (int d = 0; d < HEAD_DIM; d += 8) {
        __m256i codev = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(codebuf + d)));
        __m256i lo16v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(lo16row + d)));
        __m256i hi8v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(hi8row + d)));
        __m256i basev = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(baserow + d)));
        __m256i mvec = _mm256_or_si256(lo16v, _mm256_slli_epi32(_mm256_and_si256(hi8v, mask7f), 16));
        __m256i svec = _mm256_slli_epi32(_mm256_srli_epi32(hi8v, 7), 31);
        __m256i expraw = _mm256_add_epi32(_mm256_sub_epi32(basev, bias), codev);
        __m256i iszero = _mm256_cmpeq_epi32(codev, zero);
        __m256i expv = _mm256_andnot_si256(iszero, expraw);
        __m256i bitsv = _mm256_or_si256(svec, _mm256_or_si256(_mm256_slli_epi32(expv, 23), mvec));
        _mm256_storeu_si256((__m256i *)(bits + d), bitsv);
    }
    int64_t c = *cursor;
    kvpack_patch_escapes(codebuf, excrow, &c, bits);
    *cursor = c;
    memcpy(out, bits, sizeof bits);
}

KVPACK_TARGET_AVX512
static void kvpack_decode_row_avx512(const kv_stream_packed *s, int64_t block, int64_t blockpos, int64_t *cursor,
                                     float *out) {
    const uint8_t *baserow = s->base + (size_t)block * HEAD_DIM;
    const uint8_t *coderow = s->code + ((size_t)block * TR_ATTN_BLOCK + blockpos) * (HEAD_DIM / 2);
    const uint16_t *lo16row = s->lo16 + ((size_t)block * TR_ATTN_BLOCK + blockpos) * HEAD_DIM;
    const uint8_t *hi8row = s->hi8 + ((size_t)block * TR_ATTN_BLOCK + blockpos) * HEAD_DIM;
    const uint32_t *excrow = s->exc_bits + s->exc_off[block];

    uint8_t codebuf[HEAD_DIM];
    unpack_nibbles_128(coderow, codebuf);

    uint32_t bits[HEAD_DIM];
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask7f = _mm512_set1_epi32(0x7F);
    const __m512i bias = _mm512_set1_epi32(KVPACK_BASE_BIAS);
    for (int d = 0; d < HEAD_DIM; d += 16) {
        __m512i codev = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(codebuf + d)));
        __m512i lo16v = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(lo16row + d)));
        __m512i hi8v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(hi8row + d)));
        __m512i basev = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(baserow + d)));
        __m512i mvec = _mm512_or_si512(lo16v, _mm512_slli_epi32(_mm512_and_si512(hi8v, mask7f), 16));
        __m512i svec = _mm512_slli_epi32(_mm512_srli_epi32(hi8v, 7), 31);
        __m512i expraw = _mm512_add_epi32(_mm512_sub_epi32(basev, bias), codev);
        __mmask16 notzero = _mm512_cmpneq_epi32_mask(codev, zero);
        __m512i expv = _mm512_maskz_mov_epi32(notzero, expraw);
        __m512i bitsv = _mm512_or_si512(svec, _mm512_or_si512(_mm512_slli_epi32(expv, 23), mvec));
        _mm512_storeu_si512((void *)(bits + d), bitsv);
    }
    int64_t c = *cursor;
    kvpack_patch_escapes(codebuf, excrow, &c, bits);
    *cursor = c;
    memcpy(out, bits, sizeof bits);
}

#endif /* KVPACK_X86 */

typedef struct {
    const char *tier;
    void (*decode_row)(const kv_stream_packed *s, int64_t block, int64_t blockpos, int64_t *cursor, float *out);
} kvpack_kernels;

static const kvpack_kernels KVPACK_SCALAR = {"scalar", kvpack_decode_row_scalar};
#ifdef KVPACK_X86
static const kvpack_kernels KVPACK_AVX2 = {"avx2", kvpack_decode_row_avx2};
static const kvpack_kernels KVPACK_AVX512 = {"avx512", kvpack_decode_row_avx512};
#endif

/* Named tier, or NULL if not compiled in or not supported (respects TR_CPU_MAX like tr_kernels_tier,
 * since both read the same tr_cpu()). */
static const kvpack_kernels *kvpack_kernels_tier(const char *name) {
    if (strcmp(name, "scalar") == 0) return &KVPACK_SCALAR;
#ifdef KVPACK_X86
    if (strcmp(name, "avx2") == 0) return tr_cpu()->avx2 ? &KVPACK_AVX2 : NULL;
    if (strcmp(name, "avx512") == 0) return tr_cpu()->avx512f ? &KVPACK_AVX512 : NULL;
#endif
    return NULL;
}

static const kvpack_kernels *kvpack_kernels_get(void) {
    static const kvpack_kernels *active = NULL;
    if (active != NULL) return active;
    static const char *const order[] = {"avx512", "avx2"};
    for (size_t i = 0; i < sizeof order / sizeof order[0] && active == NULL; i++) active = kvpack_kernels_tier(order[i]);
    if (active == NULL) active = &KVPACK_SCALAR;
    return active;
}

/* Full round trip, for MEASURE 1: every position back to f32, packed region then plain tail. */
static void kvpack_decode_stream_full(const kvpack_kernels *pk, const kv_stream_packed *s, float *out) {
    for (int64_t b = 0; b < s->n_full_blocks; b++) {
        int64_t cursor = 0;
        for (int64_t r = 0; r < TR_ATTN_BLOCK; r++)
            pk->decode_row(s, b, r, &cursor, out + ((size_t)b * TR_ATTN_BLOCK + r) * HEAD_DIM);
    }
    if (s->tail_rows > 0)
        memcpy(out + (size_t)s->n_full_blocks * TR_ATTN_BLOCK * HEAD_DIM, s->tail,
               (size_t)s->tail_rows * HEAD_DIM * sizeof(float));
}

/* ---- attention over packed K/V: tr_attention_group specialized to n_q == 1 ------------------
 * first_n_pos == n_pos, j is always 0 (last_n_pos == n_pos, j0 always 0): same block sweep,
 * same dot_f32_x4/axpy_f32_x4 calls in the same order, keys/values decoded 4 (then 1) at a time
 * into `scratch` instead of read straight from memory. A block maps entirely to the packed region
 * or entirely to the plain tail (both aligned to TR_ATTN_BLOCK), so the branch is per block. */
static void kvpack_attention_decode(const kvpack_kernels *pk, const tr_kernels *K, const float *q,
                                    const kv_stream_packed *kp, const kv_stream_packed *vp, int64_t n_pos,
                                    int64_t head_dim, float scale, float *scores, float *out) {
    int64_t full_positions = kp->n_full_blocks * TR_ATTN_BLOCK;
    float scratch[TR_ATTN_X * HEAD_DIM];

    for (int64_t t0 = 0; t0 < n_pos; t0 += TR_ATTN_BLOCK) {
        int64_t t1 = t0 + TR_ATTN_BLOCK < n_pos ? t0 + TR_ATTN_BLOCK : n_pos;
        int64_t t = t0;
        if (t0 < full_positions) {
            int64_t block = t0 / TR_ATTN_BLOCK;
            int64_t cursor = 0;
            for (; t + TR_ATTN_X <= t1; t += TR_ATTN_X) {
                for (int j = 0; j < TR_ATTN_X; j++) pk->decode_row(kp, block, (t - t0) + j, &cursor, scratch + j * HEAD_DIM);
                K->dot_f32_x4(q, scratch, head_dim, head_dim, scores + t);
                for (int j = 0; j < TR_ATTN_X; j++) scores[t + j] *= scale;
            }
            for (; t < t1; t++) {
                pk->decode_row(kp, block, t - t0, &cursor, scratch);
                scores[t] = K->dot_f32(q, scratch, head_dim) * scale;
            }
        } else {
            const float *tail = kp->tail + (t0 - full_positions) * head_dim;
            for (; t + TR_ATTN_X <= t1; t += TR_ATTN_X) {
                K->dot_f32_x4(q, tail + (t - t0) * head_dim, head_dim, head_dim, scores + t);
                for (int j = 0; j < TR_ATTN_X; j++) scores[t + j] *= scale;
            }
            for (; t < t1; t++) scores[t] = K->dot_f32(q, tail + (t - t0) * head_dim, head_dim) * scale;
        }
    }

    tr_softmax(scores, n_pos);
    for (int64_t d = 0; d < head_dim; d++) out[d] = 0.0f;

    for (int64_t t0 = 0; t0 < n_pos; t0 += TR_ATTN_BLOCK) {
        int64_t t1 = t0 + TR_ATTN_BLOCK < n_pos ? t0 + TR_ATTN_BLOCK : n_pos;
        int64_t t = t0;
        if (t0 < full_positions) {
            int64_t block = t0 / TR_ATTN_BLOCK;
            int64_t cursor = 0;
            for (; t + TR_ATTN_X <= t1; t += TR_ATTN_X) {
                for (int j = 0; j < TR_ATTN_X; j++) pk->decode_row(vp, block, (t - t0) + j, &cursor, scratch + j * HEAD_DIM);
                K->axpy_f32_x4(out, scratch, head_dim, scores + t, head_dim);
            }
            for (; t < t1; t++) {
                pk->decode_row(vp, block, t - t0, &cursor, scratch);
                K->axpy_f32(out, scratch, scores[t], head_dim);
            }
        } else {
            const float *tail = vp->tail + (t0 - full_positions) * head_dim;
            for (; t + TR_ATTN_X <= t1; t += TR_ATTN_X) K->axpy_f32_x4(out, tail + (t - t0) * head_dim, head_dim, scores + t, head_dim);
            for (; t < t1; t++) K->axpy_f32(out, tail + (t - t0) * head_dim, scores[t], head_dim);
        }
    }
}

/* ---- dumps: probe/<run>/{k,v,q}_L%02d_H%02d.bin ---------------------------------------------- */

#define DEFAULT_PROBE_DIR \
    "C:/Users/marce/AppData/Local/Temp/claude/C--Users-marce-desktop-trochilus/" \
    "3d7a6f19-9d45-4e25-8db2-10960a06adbd/scratchpad/probe"

static const char *probe_dir(void) {
    const char *e = getenv("KVPACK_PROBE_DIR");
    return (e != NULL && e[0] != '\0') ? e : DEFAULT_PROBE_DIR;
}

static void build_path(char *out, size_t outsz, const char *run, char kind, int L, int H) {
    snprintf(out, outsz, "%s/%s/%c_L%02d_H%02d.bin", probe_dir(), run, kind, L, H);
}

typedef struct {
    const char *name;
    int64_t expected_n;
} run_info;

static const run_info RUNS[] = {
    {"prose2k", 1912}, {"prose4k", 4010}, {"code2k", 1982}, {"code4k", 3987}, {"synth2k", 2064}, {"synth4k", 4016},
};
#define N_RUNS ((int)(sizeof(RUNS) / sizeof(RUNS[0])))
/* Representative contexts for MEASURE 3 (~2048 and ~4000); --run all times every run instead. */
static const char *const TIME_DEFAULT_RUNS[] = {"prose2k", "prose4k"};
#define N_TIME_DEFAULT ((int)(sizeof(TIME_DEFAULT_RUNS) / sizeof(TIME_DEFAULT_RUNS[0])))

/* n_pos derived from file size (self-checking), not trusted blindly from RUNS[]. */
static float *load_f32_kv(const char *path, int64_t *n_pos_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "kvpack: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int64_t n_pos = sz / (int64_t)(HEAD_DIM * sizeof(float));
    float *buf = (float *)tr_alloc_aligned((size_t)n_pos * HEAD_DIM * sizeof(float), 64);
    if (!buf) { fclose(f); return NULL; }
    size_t want = (size_t)n_pos * HEAD_DIM;
    size_t got = fread(buf, sizeof(float), want, f);
    fclose(f);
    if (got != want) { fprintf(stderr, "kvpack: short read %s\n", path); tr_free_aligned(buf); return NULL; }
    *n_pos_out = n_pos;
    return buf;
}

typedef struct {
    int64_t n_pos;
    float q[HEAD_DIM];
} q_record;

/* Records with n_pos > max_n are skipped (the capture's last step, see file header). Returns the
 * count kept, <= cap. */
static int load_q_records(const char *path, int64_t max_n, q_record *out, int cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "kvpack: cannot open %s\n", path); return -1; }
    int n = 0;
    while (n < cap) {
        int64_t npos;
        if (fread(&npos, sizeof npos, 1, f) != 1) break;
        float q[HEAD_DIM];
        if (fread(q, sizeof(float), HEAD_DIM, f) != (size_t)HEAD_DIM) break;
        if (npos <= max_n) {
            out[n].n_pos = npos;
            memcpy(out[n].q, q, sizeof q);
            n++;
        }
    }
    fclose(f);
    return n;
}

/* ---- synthetic edge-case block: every float class, MEASURE 1 ------------------------------- */

/* One block (TR_ATTN_BLOCK positions x HEAD_DIM channels). Channels 0-3: +0/-0/+subnormal/
 * -subnormal at row 0 (code 0, no escape), baseline (e=120) elsewhere. Channels 4-7: +inf/-inf/
 * NaN(payload A)/NaN(payload B) at row 0 (forced escapes), baseline elsewhere. Channel 8: e==1
 * everywhere (the smallest normal exponent, must NOT be treated as subnormal). Channel 9: e==254
 * everywhere (the largest normal exponent, must NOT be treated as inf/NaN). Channel 10: 14
 * distinct exponents across rows 0..13 (base..base+13, code 14 is the top non-escape code, no
 * escape). Channel 11: 15 distinct exponents across rows 0..14 (row 14 must escape: base+14 does
 * not fit in code 1..14). Every other cell: baseline (e=120, random sign/mantissa) so no
 * unintended escape sneaks in. Exactly 5 escapes total (channels 4,5,6,7,11), asserted below. */
static void build_synthetic_block(float block[TR_ATTN_BLOCK][HEAD_DIM]) {
    uint64_t rng = 0xC0FFEEu;
    for (int r = 0; r < TR_ATTN_BLOCK; r++)
        for (int d = 0; d < HEAD_DIM; d++) {
            uint32_t bits = (uint32_t)rng_next(&rng);
            uint32_t sign = bits & 1u;
            uint32_t mant = (bits >> 1) & 0x7FFFFFu;
            block[r][d] = b2f((sign << 31) | (120u << 23) | mant);
        }

    block[0][0] = b2f(0x00000000u);
    block[0][1] = b2f(0x80000000u);
    block[0][2] = b2f(0x00000001u);
    block[0][3] = b2f(0x80100000u);
    block[0][4] = b2f(0x7F800000u);
    block[0][5] = b2f(0xFF800000u);
    block[0][6] = b2f(0x7FC00001u);
    block[0][7] = b2f(0xFFA00002u);

    for (int r = 0; r < TR_ATTN_BLOCK; r++) {
        uint32_t bits = (uint32_t)rng_next(&rng);
        block[r][8] = b2f((bits & 0x80000000u) | (1u << 23) | ((bits >> 1) & 0x7FFFFFu));
        bits = (uint32_t)rng_next(&rng);
        block[r][9] = b2f((bits & 0x80000000u) | (254u << 23) | ((bits >> 1) & 0x7FFFFFu));
        uint32_t e10 = (uint32_t)(r < 14 ? 10 + r : 10);
        bits = (uint32_t)rng_next(&rng);
        block[r][10] = b2f((bits & 0x80000000u) | (e10 << 23) | ((bits >> 1) & 0x7FFFFFu));
        uint32_t e11 = (uint32_t)(r < 15 ? 10 + r : 10);
        bits = (uint32_t)rng_next(&rng);
        block[r][11] = b2f((bits & 0x80000000u) | (e11 << 23) | ((bits >> 1) & 0x7FFFFFu));
    }
}

/* ---- MEASURE 1: round trip ------------------------------------------------------------------ */

static void test_roundtrip(const char *only_run) {
    const kvpack_kernels *tiers[3];
    int n_tiers = 0;
    tiers[n_tiers++] = &KVPACK_SCALAR;
#ifdef KVPACK_X86
    if (tr_cpu()->avx2) tiers[n_tiers++] = &KVPACK_AVX2;
    if (tr_cpu()->avx512f) tiers[n_tiers++] = &KVPACK_AVX512;
#endif
    printf("-- MEASURE 1: round trip (%d decode tiers: ", n_tiers);
    for (int i = 0; i < n_tiers; i++) printf("%s%s", tiers[i]->tier, i + 1 < n_tiers ? ", " : ")\n");

    for (int ri = 0; ri < N_RUNS; ri++) {
        if (only_run && strcmp(RUNS[ri].name, only_run) != 0) continue;
        uint64_t sum_bytes = 0, sum_values = 0, sum_exc = 0, sum_packed_values = 0;
        int checked = 0;
        for (int L = 0; L < N_LAYER; L++) {
            for (int H = 0; H < N_HEAD; H++) {
                for (int kind = 0; kind < 2; kind++) {
                    char path[512];
                    build_path(path, sizeof path, RUNS[ri].name, kind ? 'v' : 'k', L, H);
                    int64_t n_pos = 0;
                    float *f32 = load_f32_kv(path, &n_pos);
                    TR_CHECK(f32 != NULL);
                    if (!f32) continue;
                    if (L == 0 && H == 0 && kind == 0) TR_CHECK_EQ_INT(n_pos, RUNS[ri].expected_n);

                    kv_stream_packed pk_s;
                    TR_CHECK_EQ_INT(kvpack_encode(f32, n_pos, &pk_s), 0);

                    float *round = (float *)tr_alloc_aligned((size_t)n_pos * HEAD_DIM * sizeof(float), 64);
                    kvpack_decode_stream_full(kvpack_kernels_get(), &pk_s, round);
                    int bad = memcmp(f32, round, (size_t)n_pos * HEAD_DIM * sizeof(float)) != 0;
                    TR_CHECK(!bad);

                    /* cross-tier: L00_H00 of every run, every available tier vs the original */
                    if (L == 0 && H == 0) {
                        for (int ti = 0; ti < n_tiers; ti++) {
                            kvpack_decode_stream_full(tiers[ti], &pk_s, round);
                            int tier_bad = memcmp(f32, round, (size_t)n_pos * HEAD_DIM * sizeof(float)) != 0;
                            TR_CHECK(!tier_bad);
                        }
                    }

                    sum_bytes += kvpack_bytes(&pk_s);
                    sum_values += (uint64_t)n_pos * HEAD_DIM;
                    sum_exc += (pk_s.n_full_blocks > 0 && pk_s.exc_off) ? pk_s.exc_off[pk_s.n_full_blocks] : 0;
                    sum_packed_values += (uint64_t)pk_s.n_full_blocks * TR_ATTN_BLOCK * HEAD_DIM;
                    checked++;

                    tr_free_aligned(round);
                    kvpack_free(&pk_s);
                    tr_free_aligned(f32);
                }
            }
        }
        double bytes_per_value = sum_values ? (double)sum_bytes / (double)sum_values : 0.0;
        double f32_bytes = (double)sum_values * sizeof(float);
        double escape_pct = sum_packed_values ? 100.0 * (double)sum_exc / (double)sum_packed_values : 0.0;
        printf("  roundtrip %-8s streams=%-4d bytes/value=%.4f  ratio(f32/packed)=%.4fx  escapes=%llu/%llu (%.5f%%)\n",
               RUNS[ri].name, checked, bytes_per_value, sum_bytes ? f32_bytes / (double)sum_bytes : 0.0,
               (unsigned long long)sum_exc, (unsigned long long)sum_packed_values, escape_pct);
    }

    /* synthetic block: every float class */
    float block[TR_ATTN_BLOCK][HEAD_DIM];
    build_synthetic_block(block);
    kv_stream_packed pk_s;
    TR_CHECK_EQ_INT(kvpack_encode(&block[0][0], TR_ATTN_BLOCK, &pk_s), 0);
    TR_CHECK_EQ_INT(pk_s.n_full_blocks, 1);
    TR_CHECK_EQ_INT(pk_s.tail_rows, 0);
    float round[TR_ATTN_BLOCK][HEAD_DIM];
    kvpack_decode_stream_full(kvpack_kernels_get(), &pk_s, &round[0][0]);
    TR_CHECK(memcmp(block, round, sizeof block) == 0);
    uint32_t total_exc = pk_s.exc_off[1];
    TR_CHECK(total_exc > 0);           /* the brief's literal ask: the escape branch ran */
    TR_CHECK_EQ_INT(total_exc, 5);     /* exact, by construction: channels 4,5,6,7,11 */
    for (int ti = 0; ti < n_tiers; ti++) {
        kvpack_decode_stream_full(tiers[ti], &pk_s, &round[0][0]);
        TR_CHECK(memcmp(block, round, sizeof block) == 0);
    }
    printf("  roundtrip synthetic-block  every float class memcmp==0, exceptions=%u (expected 5)\n", total_exc);
    kvpack_free(&pk_s);
}

/* ---- MEASURE 2: bits, packed attention vs tr_attention_group on the same F32 rows ------------ */

static void test_bits(const char *only_run) {
    const kvpack_kernels *pk = kvpack_kernels_get();
    const tr_kernels *K = tr_kernels_get();
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    printf("-- MEASURE 2: bits (kvpack tier %s, engine tier %s)\n", pk->tier, K->tier);

    for (int ri = 0; ri < N_RUNS; ri++) {
        if (only_run && strcmp(RUNS[ri].name, only_run) != 0) continue;
        int64_t bad = 0, checked = 0, skipped = 0;
        for (int L = 0; L < N_LAYER; L++) {
            for (int H = 0; H < N_HEAD; H++) {
                char kpath[512], vpath[512], qpath[512];
                build_path(kpath, sizeof kpath, RUNS[ri].name, 'k', L, H);
                build_path(vpath, sizeof vpath, RUNS[ri].name, 'v', L, H);
                build_path(qpath, sizeof qpath, RUNS[ri].name, 'q', L, H);

                int64_t n_pos_k = 0, n_pos_v = 0;
                float *kf = load_f32_kv(kpath, &n_pos_k);
                float *vf = load_f32_kv(vpath, &n_pos_v);
                TR_CHECK(kf != NULL && vf != NULL);
                if (!kf || !vf) { tr_free_aligned(kf); tr_free_aligned(vf); continue; }
                TR_CHECK_EQ_INT(n_pos_k, n_pos_v);

                q_record recs[32];
                int n_recs = load_q_records(qpath, n_pos_k, recs, 32);
                TR_CHECK(n_recs > 0);

                kv_stream_packed kp, vp;
                TR_CHECK_EQ_INT(kvpack_encode(kf, n_pos_k, &kp), 0);
                TR_CHECK_EQ_INT(kvpack_encode(vf, n_pos_v, &vp), 0);

                static float scores_ref[MAX_N_POS], scores_pk[MAX_N_POS];
                for (int i = 0; i < n_recs; i++) {
                    float out_ref[HEAD_DIM], out_pk[HEAD_DIM];
                    tr_attention_group(recs[i].q, HEAD_DIM, kf, vf, 1, recs[i].n_pos, HEAD_DIM, scale, scores_ref,
                                       recs[i].n_pos, out_ref, HEAD_DIM);
                    kvpack_attention_decode(pk, K, recs[i].q, &kp, &vp, recs[i].n_pos, HEAD_DIM, scale, scores_pk,
                                            out_pk);
                    if (memcmp(out_ref, out_pk, sizeof out_ref) != 0) bad++;
                    checked++;
                }
                skipped += 17 - n_recs > 0 ? 0 : 0; /* records >N are silently filtered by load_q_records */

                kvpack_free(&kp);
                kvpack_free(&vp);
                tr_free_aligned(kf);
                tr_free_aligned(vf);
            }
        }
        TR_CHECK_EQ_INT(bad, 0);
        printf("  bits %-8s queries=%-5lld mismatches=%lld\n", RUNS[ri].name, (long long)checked, (long long)bad);
        (void)skipped;
    }
}

/* ---- MEASURE 3: time, F32 vs packed, 8 threads, the engine's pool ---------------------------- */

typedef struct {
    const float *q_per_head;
    const float *k_f32_per_head, *v_f32_per_head;
    const kv_stream_packed *k_pk, *v_pk;
    float *scores_per_worker;
    float *out_per_head;
    int64_t n_pos, head_dim;
    float scale;
    int use_packed;
    const kvpack_kernels *pk;
    const tr_kernels *K;
} layer_attn_ctx;

static void layer_attn_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const layer_attn_ctx *c = (const layer_attn_ctx *)ctx_;
    float *scores = c->scores_per_worker + (int64_t)worker * MAX_N_POS;
    for (int64_t h = begin; h < end; h++) {
        const float *q = c->q_per_head + h * c->head_dim;
        float *out = c->out_per_head + h * c->head_dim;
        if (c->use_packed) {
            kvpack_attention_decode(c->pk, c->K, q, &c->k_pk[h], &c->v_pk[h], c->n_pos, c->head_dim, c->scale, scores,
                                    out);
        } else {
            const float *keys = c->k_f32_per_head + h * c->n_pos * c->head_dim;
            const float *values = c->v_f32_per_head + h * c->n_pos * c->head_dim;
            tr_attention_group(q, c->head_dim, keys, values, 1, c->n_pos, c->head_dim, c->scale, scores, c->n_pos,
                               out, c->head_dim);
        }
    }
}

typedef struct {
    const unsigned char *base;
    size_t block;
} flush_ctx;

static volatile uint64_t g_sink;

static void ram_flush_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const flush_ctx *c = (const flush_ctx *)ctx_;
    uint64_t s = 0;
    for (int64_t i = begin; i < end; i++) {
        const unsigned char *p = c->base + (size_t)i * c->block;
        for (size_t k = 0; k + 64 <= c->block; k += 64) {
            uint64_t w[8];
            memcpy(w, p + k, 64);
            s += w[0] + w[1] + w[2] + w[3] + w[4] + w[5] + w[6] + w[7];
        }
    }
    g_sink += s;
}

static void fill_arena(unsigned char *base, size_t bytes) {
    uint64_t s = 0x1234567u;
    for (size_t i = 0; i + 8 <= bytes; i += 8) {
        uint64_t v = rng_next(&s);
        memcpy(base + i, &v, 8);
    }
}

static void report_time(const char *run, const char *variant, int64_t n_pos, double *secs, int n, uint64_t bytes) {
    qsort(secs, (size_t)n, sizeof secs[0], cmp_double);
    double med = (n % 2) ? secs[n / 2] : 0.5 * (secs[n / 2 - 1] + secs[n / 2]);
    printf("  time %-8s %-7s N=%-5lld %8.3f ms  %7.2f GB/s  (min %7.3f max %7.3f spread %4.1f%%)\n", run, variant,
           (long long)n_pos, med * 1e3, (double)bytes / med / 1e9, secs[0] * 1e3, secs[n - 1] * 1e3,
           (secs[n - 1] - secs[0]) / med * 100.0);
}

static void test_time(const char *only_run, int n_runs) {
    tr_pool *pool = tr_pool_create(8);
    if (!pool) { fprintf(stderr, "kvpack: tr_pool_create failed\n"); TR_CHECK(0); return; }
    int T = tr_pool_size(pool);
    printf("-- MEASURE 3: time (%d threads, kvpack tier %s, engine tier %s)\n", T, kvpack_kernels_get()->tier,
           tr_kernels_get()->tier);

    enum { FLUSH_SLOTS = 4 };
    size_t flush_bytes = 64u * 1024u * 1024u;
    unsigned char *flush_arena = (unsigned char *)tr_alloc_aligned(flush_bytes * FLUSH_SLOTS, 4096);
    TR_CHECK(flush_arena != NULL);
    if (flush_arena) fill_arena(flush_arena, flush_bytes * FLUSH_SLOTS);

    float *scores_workers = (float *)tr_alloc_aligned((size_t)T * MAX_N_POS * sizeof(float), 64);
    float *out_scratch = (float *)tr_alloc_aligned((size_t)N_HEAD * HEAD_DIM * sizeof(float), 64);
    TR_CHECK(scores_workers != NULL && out_scratch != NULL);

    const kvpack_kernels *pk = kvpack_kernels_get();
    const tr_kernels *K = tr_kernels_get();
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    const char *const *run_list = TIME_DEFAULT_RUNS;
    int n_run_list = N_TIME_DEFAULT;
    static const char *one_run[1];
    if (only_run) {
        if (strcmp(only_run, "all") == 0) {
            static const char *all_names[N_RUNS];
            for (int i = 0; i < N_RUNS; i++) all_names[i] = RUNS[i].name;
            run_list = all_names;
            n_run_list = N_RUNS;
        } else {
            one_run[0] = only_run;
            run_list = one_run;
            n_run_list = 1;
        }
    }

    for (int ri = 0; ri < n_run_list; ri++) {
        const run_info *run = NULL;
        for (int i = 0; i < N_RUNS; i++) if (strcmp(RUNS[i].name, run_list[ri]) == 0) run = &RUNS[i];
        if (!run) continue;
        int64_t N = run->expected_n;
        size_t layer_floats = (size_t)N * HEAD_DIM;

        float *k_f32 = (float *)tr_alloc_aligned((size_t)N_LAYER * N_HEAD * layer_floats * sizeof(float), 64);
        float *v_f32 = (float *)tr_alloc_aligned((size_t)N_LAYER * N_HEAD * layer_floats * sizeof(float), 64);
        float *q_all = (float *)tr_alloc_aligned((size_t)N_LAYER * N_HEAD * HEAD_DIM * sizeof(float), 64);
        kv_stream_packed *k_pk = (kv_stream_packed *)calloc((size_t)N_LAYER * N_HEAD, sizeof(kv_stream_packed));
        kv_stream_packed *v_pk = (kv_stream_packed *)calloc((size_t)N_LAYER * N_HEAD, sizeof(kv_stream_packed));
        TR_CHECK(k_f32 && v_f32 && q_all && k_pk && v_pk);
        if (!k_f32 || !v_f32 || !q_all || !k_pk || !v_pk) goto next_run;

        for (int L = 0; L < N_LAYER; L++) {
            for (int H = 0; H < N_HEAD; H++) {
                char path[512];
                int64_t got_n;
                build_path(path, sizeof path, run->name, 'k', L, H);
                float *kbuf = load_f32_kv(path, &got_n);
                TR_CHECK(kbuf != NULL && got_n == N);
                if (kbuf) { memcpy(k_f32 + ((size_t)L * N_HEAD + H) * layer_floats, kbuf, layer_floats * sizeof(float)); tr_free_aligned(kbuf); }

                build_path(path, sizeof path, run->name, 'v', L, H);
                float *vbuf = load_f32_kv(path, &got_n);
                TR_CHECK(vbuf != NULL && got_n == N);
                if (vbuf) { memcpy(v_f32 + ((size_t)L * N_HEAD + H) * layer_floats, vbuf, layer_floats * sizeof(float)); tr_free_aligned(vbuf); }

                build_path(path, sizeof path, run->name, 'q', L, H);
                q_record recs[32];
                int n_recs = load_q_records(path, N, recs, 32);
                TR_CHECK(n_recs > 0 && recs[n_recs - 1].n_pos == N);
                if (n_recs > 0) memcpy(q_all + ((size_t)L * N_HEAD + H) * HEAD_DIM, recs[n_recs - 1].q, HEAD_DIM * sizeof(float));

                TR_CHECK_EQ_INT(kvpack_encode(k_f32 + ((size_t)L * N_HEAD + H) * layer_floats, N, &k_pk[L * N_HEAD + H]), 0);
                TR_CHECK_EQ_INT(kvpack_encode(v_f32 + ((size_t)L * N_HEAD + H) * layer_floats, N, &v_pk[L * N_HEAD + H]), 0);
            }
        }

        uint64_t f32_bytes = (uint64_t)N_LAYER * N_HEAD * (uint64_t)N * HEAD_DIM * sizeof(float) * 2;
        uint64_t packed_bytes = 0;
        for (int i = 0; i < N_LAYER * N_HEAD; i++) packed_bytes += kvpack_bytes(&k_pk[i]) + kvpack_bytes(&v_pk[i]);
        printf("  time %-8s N=%-5lld f32=%.1f MiB packed=%.1f MiB ratio=%.4fx\n", run->name, (long long)N,
               (double)f32_bytes / (1024.0 * 1024.0), (double)packed_bytes / (1024.0 * 1024.0),
               (double)f32_bytes / (double)packed_bytes);

        for (int variant = 0; variant <= 1; variant++) {
            double secs[MAX_TIME_RUNS];
            int cap = n_runs < MAX_TIME_RUNS ? n_runs : MAX_TIME_RUNS;
            int slot = 0;
            for (int r = -1; r < cap; r++) {
                double sum = 0.0;
                for (int L = 0; L < N_LAYER; L++) {
                    layer_attn_ctx ctx;
                    memset(&ctx, 0, sizeof ctx);
                    ctx.q_per_head = q_all + (size_t)L * N_HEAD * HEAD_DIM;
                    ctx.head_dim = HEAD_DIM;
                    ctx.n_pos = N;
                    ctx.scale = scale;
                    ctx.use_packed = variant;
                    ctx.pk = pk;
                    ctx.K = K;
                    ctx.scores_per_worker = scores_workers;
                    ctx.out_per_head = out_scratch;
                    if (variant == 0) {
                        ctx.k_f32_per_head = k_f32 + (size_t)L * N_HEAD * layer_floats;
                        ctx.v_f32_per_head = v_f32 + (size_t)L * N_HEAD * layer_floats;
                    } else {
                        ctx.k_pk = &k_pk[L * N_HEAD];
                        ctx.v_pk = &v_pk[L * N_HEAD];
                    }
                    double t0 = tr_time_sec();
                    tr_parallel_for(pool, N_HEAD, 1, layer_attn_body, &ctx);
                    sum += tr_time_sec() - t0;

                    if (flush_arena) {
                        flush_ctx fc;
                        fc.base = flush_arena + (size_t)(slot % FLUSH_SLOTS) * flush_bytes;
                        fc.block = (flush_bytes / (size_t)T) & ~(size_t)4095;
                        tr_parallel_for(pool, T, 1, ram_flush_body, &fc);
                        slot++;
                    }
                }
                if (r >= 0) secs[r] = sum;
            }
            g_sink += f2b(out_scratch[0]);
            report_time(run->name, variant ? "packed" : "f32", N, secs, cap, variant ? packed_bytes : f32_bytes);
        }

    next_run:
        for (int i = 0; i < N_LAYER * N_HEAD; i++) {
            if (k_pk) kvpack_free(&k_pk[i]);
            if (v_pk) kvpack_free(&v_pk[i]);
        }
        free(k_pk);
        free(v_pk);
        tr_free_aligned(k_f32);
        tr_free_aligned(v_f32);
        tr_free_aligned(q_all);
    }

    tr_free_aligned(scores_workers);
    tr_free_aligned(out_scratch);
    tr_free_aligned(flush_arena);
    tr_pool_destroy(pool);
    printf("  (machine may be shared: times are noisy; re-measure on a still machine before concluding)\n");
}

/* ---- main ------------------------------------------------------------------------------------ */

static void usage(void) {
    fprintf(stderr, "usage: bench_kvpack [roundtrip|bits|time|all] [--run <name>|all] [--runs N]\n");
}

int main(int argc, char **argv) {
    const char *phase = "all";
    const char *only_run = NULL;
    int n_runs = 11;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) n_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--run") == 0 && i + 1 < argc) only_run = argv[++i];
        else if (strcmp(argv[i], "roundtrip") == 0 || strcmp(argv[i], "bits") == 0 || strcmp(argv[i], "time") == 0 ||
                 strcmp(argv[i], "all") == 0)
            phase = argv[i];
        else { usage(); return 2; }
    }
    if (n_runs < 3) n_runs = 3;
    if (n_runs > MAX_TIME_RUNS) n_runs = MAX_TIME_RUNS;

    tr_kernels_init();
    char cpu_line[512];
    tr_cpu_describe(tr_cpu(), cpu_line, sizeof cpu_line);
    printf("bench_kvpack %s, built %s %s, engine kernels %s, kvpack kernels %s\n%s\nprobe dir: %s\n", phase, __DATE__,
           __TIME__, tr_kernels_get()->tier, kvpack_kernels_get()->tier, cpu_line, probe_dir());
#ifdef KVPACK_MUTATE_BASE_OFF_BY_ONE
    printf("*** KVPACK_MUTATE_BASE_OFF_BY_ONE is ON: the decoder is deliberately wrong ***\n");
#endif

    if (strcmp(phase, "roundtrip") == 0 || strcmp(phase, "all") == 0) test_roundtrip(only_run);
    if (strcmp(phase, "bits") == 0 || strcmp(phase, "all") == 0) test_bits(only_run);
    if (strcmp(phase, "time") == 0 || strcmp(phase, "all") == 0) test_time(only_run, n_runs);

    TR_TEST_EXIT();
}
