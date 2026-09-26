/* kernels.h — CPU compute kernels and their runtime dispatch.
 *
 * Numerics contract (the reason every SIMD or assembly variant can be checked
 * bit for bit against scalar C):
 *
 *   - Scalar C is the definition. It is compiled with -ffp-contract=off, so no
 *     multiply-add is fused behind our back; variants use separate multiply and
 *     add instructions, never FMA, unless the scalar definition calls fma().
 *   - A reduction over n elements (dot product, sum of squares) keeps
 *     TR_LANES = 16 running accumulators: element k goes into lane k % 16,
 *     processed in increasing k. The lanes are then combined in a fixed
 *     pairwise tree: ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), same for 8..15,
 *     and the two halves added last. One AVX-512 register, two AVX2
 *     registers or four NEON registers hold exactly these 16 lanes.
 *   - An element-wise operation (scale, add, rope, silu) has no order to keep:
 *     every variant is exact by construction.
 *   - Results never depend on the thread count: work is split by output row
 *     or by token, never inside a reduction.
 *
 * "Bit identical" covers every non-NaN input; a NaN input gives a NaN in every
 * variant, but its payload may differ (x86 picks the NaN of the first operand).
 * A variant that cannot meet the contract does not enter the dispatch table. */
#ifndef TR_KERNELS_H
#define TR_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#include "../format/gguf.h"
#include "../base/threads.h"

#define TR_LANES 16
/* Tokens per block in tr_matmul: a block of activations stays in L2 while every weight
 * row is dotted against it, so the weight is read once per block instead of once per
 * token. Changes speed only, never a result; -DTR_MATMUL_TILE=<n> to measure another. */
#ifndef TR_MATMUL_TILE
#define TR_MATMUL_TILE 16
#endif
/* Input rows a dot_row_x4 kernel handles in one pass over the weight row. */
#define TR_DOT_TOKENS 4
/* Input rows a dot_row2_x8 kernel handles in one pass over its two weight rows. */
#define TR_DOT_TOKENS_WIDE 8
/* Phase-major matmul (the table's pm_* entries): weight rows per panel (one per SIMD lane), the widest
 * tile of input rows, and the floats of a tile's scratch (one lane vector per phase and input row). */
#define TR_PM_ROWS 16
#define TR_PM_TILE_MAX 24
#define TR_PM_PART (16 * TR_PM_TILE_MAX * TR_PM_ROWS)
/* Floats added after each phase's run in a panel and in an interleaved tile: phases a power of two of
 * bytes apart fall in one set of the L1 (8 ways) and the 16 stores of a step evict each other. */
#define TR_PM_PAD 16
/* floats of a panel of n columns, and of an interleaved tile of T input rows */
#define TR_PM_PANEL_FLOATS(n) (16 * ((int64_t)(n) + TR_PM_PAD))
#define TR_PM_XIL_FLOATS(n, T) (16 * ((int64_t)(n) / 16 * (T) + TR_PM_PAD))
/* Cache positions a dot_f32_x4 or axpy_f32_x4 kernel handles per load of the query or the output. */
#define TR_ATTN_X 4
/* Cache positions per block in tr_attention_group: a block of keys (then of values) stays in the
 * cache while every query of the group uses it. Changes speed only, never a result. */
#define TR_ATTN_BLOCK 64

/* A 2-D weight in ggml layout: `rows` output rows, each `cols` input elements,
 * stored as consecutive rows of type `type` (row bytes = tr_row_bytes). */
typedef struct {
    tr_type type;
    int64_t rows;
    int64_t cols;
    const void *data;
} tr_mat;

/* Bytes of one row of n elements; 0 if n is not a multiple of the block size. */
size_t tr_row_bytes(tr_type type, int64_t n);
/* 1 if the matmul and dequant kernels support this type. */
int tr_kernels_support(tr_type type);

/* Kernel table. Filled once by tr_kernels_init from tr_cpu(); read-only afterwards. */
typedef struct {
    const char *tier;   /* "scalar", "avx2", "avx512", "neon" ... */
    /* sum_k a[k] * b[k] under the lane contract */
    float (*dot_f32)(const float *a, const float *b, int64_t n);
    /* y[k] = y[k] + a * x[k], element-wise (x and y do not overlap) */
    void (*axpy_f32)(float *y, const float *x, float a, int64_t n);
    /* TR_ATTN_X dots of a with b, b+stride, ...: out[j] = dot_f32(a, b + j*stride, n), each under
     * the lane contract, with a loaded once for all of them (one query against 4 keys) */
    void (*dot_f32_x4)(const float *a, const float *b, int64_t stride, int64_t n, float *out);
    /* TR_ATTN_X axpys on the same y, one after the other: axpy_f32(y, x + j*stride, a[j], n) for
     * j = 0, 1, 2, 3 in this order, with y loaded and stored once for all of them */
    void (*axpy_f32_x4)(float *y, const float *x, int64_t stride, const float *a, int64_t n);
    /* A tile of a prompt's attention, TR_ATTN_X queries against TR_ATTN_X keys: out[i*out_stride +
     * j] = dot_f32(a + i*a_stride, b + j*stride, n) * scale for i, j < TR_ATTN_X, each dot under
     * the lane contract and then one multiply, as tr_attention_group scales a score */
    void (*dot_f32_4x4)(const float *a, int64_t a_stride, const float *b, int64_t stride, int64_t n, float scale,
                        float *out, int64_t out_stride);
    /* TR_ATTN_X outputs against the same TR_ATTN_X rows: axpy_f32_x4(y + i*y_stride, x, stride,
     * a + i*a_stride, n) for i < TR_ATTN_X, each output's four additions in order */
    void (*axpy_f32_4x4)(float *y, int64_t y_stride, const float *x, int64_t stride, const float *a, int64_t a_stride,
                         int64_t n);
    /* dot of one quantized row (n elements) with f32 x, under the lane contract
     * applied to the dequantized weights: w_k = d_block * q_k computed as a
     * float product first, then w_k * x_k into lane k % 16 */
    float (*dot_row[TR_TYPE_COUNT])(const void *row, const float *x, int64_t n);
    /* TR_DOT_TOKENS dots of the same row with consecutive input rows x, x+stride, ...:
     * out[j] = dot_row(row, x + j*stride, n), each under the lane contract, with the row
     * read and decoded once for all of them (fewer bytes and conversions per element, same
     * numbers). NULL for a type without a variant: the caller loops over dot_row. */
    void (*dot_row_x4[TR_TYPE_COUNT])(const void *row, const float *x, int64_t stride, int64_t n, float *out);
    /* Two weight rows against the same input row (a decode token): out[0] = dot_row(row0, x, n)
     * and out[1] = dot_row(row1, x, n), bit for bit, each row with its own chain of adds. NULL:
     * the caller calls dot_row twice. */
    void (*dot_row2[TR_TYPE_COUNT])(const void *row0, const void *row1, const float *x, int64_t n, float *out);
    /* Two weight rows against the same TR_DOT_TOKENS input rows: out[j] = dot_row(row0, x +
     * j*stride, n) and out[TR_DOT_TOKENS + j] = dot_row(row1, x + j*stride, n), each input
     * element loaded once for both rows (half the loads of two dot_row_x4). NULL: the caller
     * uses dot_row_x4. */
    void (*dot_row2_x4[TR_TYPE_COUNT])(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n,
                                       float *out);
    /* The same against TR_DOT_TOKENS_WIDE input rows: out[j] = dot_row(row0, x + j*stride, n) and
     * out[TR_DOT_TOKENS_WIDE + j] = dot_row(row1, x + j*stride, n), each weight decoded once for all
     * of them. NULL: the caller uses dot_row2_x4. */
    void (*dot_row2_x8[TR_TYPE_COUNT])(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n,
                                       float *out);
    /* The prompt's matmul in phase-major order (docs/MEASUREMENTS.md §Phase-major). The lane contract's
     * lane p (elements k = p, p + 16, ... in increasing k, then the tree) is a sequence in time, not a place
     * in a register: TR_PM_ROWS weight rows can run it side by side, one row a SIMD lane, and each lane
     * adds exactly dot_row's products in dot_row's order. Three steps, n a multiple of 16:
     * pm_panel: TR_PM_ROWS consecutive rows (row_bytes apart), each weight dequantized as dot_row
     *   computes it, laid out panel[p * (n + TR_PM_PAD) + m * TR_PM_ROWS + r] = w_r[16m + p]
     *   (TR_PM_PANEL_FLOATS(n) floats);
     * pm_interleave: T input rows (stride floats apart) laid out xil[p * (n/16 * T + TR_PM_PAD) + m * T + t]
     *   = x_t[16m + p] (TR_PM_XIL_FLOATS(n, T) floats);
     * pm_tile: the panel's rows against the T interleaved rows, T one of 24, 16, 8, 4 (TR_PM_TILE_MAX):
     *   y[t * y_stride + r] = dot_row(row r, x_t, n) bit for bit; part is TR_PM_PART floats of scratch.
     * NULL where the tier has none: the caller keeps dot_row2_x8 and the others. */
    void (*pm_panel[TR_TYPE_COUNT])(const void *rows, size_t row_bytes, int64_t n, float *panel);
    void (*pm_interleave)(const float *x, int64_t stride, int64_t n, int T, float *xil);
    void (*pm_tile)(const float *panel, const float *xil, int64_t n, int T, float *part, float *y, int64_t y_stride);
    /* decode one row of n elements to f32 */
    void (*dequant_row[TR_TYPE_COUNT])(const void *row, float *out, int64_t n);
    /* y[i] = tr_expf(x[i]) for i < n, element-wise; y may be x (in place). The exponential of the
     * softmax and of the SiLU, several lanes at a time: every lane is tr_expf's bits. */
    void (*expf_f32)(const float *x, float *y, int64_t n);
} tr_kernels;

/* Selects the best tier this CPU supports. Idempotent. */
void tr_kernels_init(void);
/* The active table (tr_kernels_init must have run). */
const tr_kernels *tr_kernels_get(void);
/* The table of a named tier, or NULL if not compiled in or not supported by this
 * CPU. Tests use it to compare every tier against "scalar". */
const tr_kernels *tr_kernels_tier(const char *tier);

/* ---- operations built on the table (all deterministic across thread counts) ---- */

/* y[t*rows + r] = row r of w  ·  x[t*cols ..] for t < n_tokens; parallel over (t, r).
 * Each y element is one dot_row call, whatever the thread count or the visiting order:
 * several tokens are visited in blocks (every weight row against a block of TR_MATMUL_TILE
 * tokens), so a weight is read once per block instead of once per token. */
void tr_matmul(tr_pool *pool, const tr_mat *w, const float *x, int64_t n_tokens, float *y);
/* The same over input rows grouped by weight (MoE experts): group g holds rows
 * [offsets[g], offsets[g+1]) of x and multiplies them by w[g]; every w[g] has the same
 * type, rows and cols; offsets is non-decreasing with offsets[0] = 0, and empty groups
 * are skipped. y[p*rows + r] = row r of w[g] · x[p*cols ..] for the group g of row p. */
void tr_matmul_grouped(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const float *x,
                       float *y);
/* Scratch of the prompt's phase-major matmul (the table's pm_* entries), owned by the caller, built once
 * by tr_pm_scratch_init and reused by every call: the interleaved input rows (as many floats as the
 * largest x a call passes), a panel and a tile's part per pool worker, and the call's plan (its items,
 * token chunks and tiles). A zeroed struct is "none". */
typedef struct {
    float *xil;
    int64_t max_x;              /* floats of xil */
    float *work;                /* n_workers * (TR_PM_ROWS * max_cols + TR_PM_PART) floats */
    int64_t max_cols;
    int n_workers;
    int64_t *plan;              /* item, chunk and tile tables, and each worker's last panel */
    int64_t max_groups, max_tokens;
} tr_pm_scratch;
/* 0 on success; the sizes are the largest a call will pass: groups, input rows (tokens), columns and
 * input floats (tokens x columns). The same struct is freed by tr_pm_scratch_free (zeroed after). */
int tr_pm_scratch_init(tr_pm_scratch *s, int n_workers, int64_t max_groups, int64_t max_tokens, int64_t max_cols,
                       int64_t max_x);
void tr_pm_scratch_free(tr_pm_scratch *s);
uint64_t tr_pm_scratch_bytes(int n_workers, int64_t max_groups, int64_t max_tokens, int64_t max_cols, int64_t max_x);
/* tr_matmul_grouped and tr_matmul, the same y bit for bit, taking the phase-major road where it applies:
 * the tier has pm_panel for w's type and pm_tile, rows and cols are multiples of TR_PM_ROWS and 16, the
 * call fits s; a group of fewer than 4 input rows keeps the x4 and x8 kernels. s == NULL: the old road. */
void tr_matmul_grouped_s(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const float *x,
                         float *y, const tr_pm_scratch *s);
void tr_matmul_s(tr_pool *pool, const tr_mat *w, const float *x, int64_t n_tokens, float *y, const tr_pm_scratch *s);
/* 1 if no group of tr_matmul_grouped's has more than one input row: a decode token's projections
 * (one group, one row) and its experts (n_expert groups, the used ones one row each, the rest
 * empty), whose call is only weight rows to stream and runs balanced at its tail. Until
 * 2026-09-26 the test was "rows == groups", never true for the experts' 64 groups and 8 rows
 * (docs/LESSONS.md #192). */
int tr_matmul_one_row_per_group(const int64_t *offsets, int64_t n_groups);
/* out = dequantized row `row` of w (cols elements): an embedding lookup. */
void tr_get_row(const tr_mat *w, int64_t row, float *out);
/* x[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i], mean via the lane contract, f32. */
void tr_rmsnorm(float *x, const float *weight, int64_t n, float eps);
/* RoPE table for positions 0..n_pos-1, half = head_dim / 2 entries per position:
 * cos_t[p*half + i] = (float)cos(inv_freq[i] * p), same for sin_t, with
 * inv_freq[i] = 1 / theta^(2i/head_dim) and the angle computed in double. Not hot:
 * built once per session, so the forward pass does no trigonometry. */
void tr_rope_table(float *cos_t, float *sin_t, int64_t n_pos, int64_t head_dim, float theta);
/* In place, NeoX layout (first half / second half), n_heads heads of head_dim (even)
 * elements, one position: x' = x*cos_p + rotate_half(x)*sin_p, where cos_p and sin_p are the
 * head_dim/2 entries of that position in the tr_rope_table tables. */
void tr_rope_neox(float *x, int64_t n_heads, int64_t head_dim, const float *cos_p, const float *sin_p);
/* exp(x) correctly rounded to float (to nearest, ties to even) for every float: the same bits
 * on every platform, and no C library call inside (expf.c). The exponential of the softmax and
 * of the SiLU. */
float tr_expf(float x);
/* Which way tr_expf settles x, for the tests: a border case (NaN, overflow, underflow), the
 * fast path, the table of exceptions, or none of them (a NaN; bench_expf --check proves that
 * no float gets there). */
enum { TR_EXPF_SPECIAL, TR_EXPF_FAST, TR_EXPF_TABLE, TR_EXPF_UNPROVEN };
int tr_expf_path(float x);
/* The table of exceptions, for the tests: entry i (0 <= i < n) is the bits of an argument and
 * the bits of its exp, computed at 200 bits (tools/gen_expf_table.py). */
int tr_expf_n_exceptions(void);
void tr_expf_exception(int i, uint32_t *x_bits, uint32_t *y_bits);
/* In place softmax over n logits (max-subtracted, tr_expf, sum via the lane contract). */
void tr_softmax(float *x, int64_t n);
/* x[i] = silu(x[i]) * y[i], silu(v) = v / (1 + tr_expf(-v)); element-wise, split over the pool
 * (p == NULL: serial). */
void tr_swiglu(tr_pool *pool, float *x, const float *y, int64_t n);
/* Causal attention of one query head over n_pos cached positions. `keys` and `values`
 * hold n_pos slots of `stride` floats; this head reads head_dim floats at `offset` in
 * each slot. scores[t] = dot_f32(q, k_t) * scale, softmax over t < n_pos, then
 * out = 0 + scores[0]*v_0 + scores[1]*v_1 + ... in increasing t (one axpy_f32 per t).
 * scores: n_pos floats of scratch owned by the caller. */
void tr_attention_head(const float *q, const float *keys, const float *values, int64_t stride, int64_t offset,
                       int64_t n_pos, int64_t head_dim, float scale, float *scores, float *out);
/* Causal attention of n_q consecutive tokens of one query head, a group: query j, at
 * q + j*q_stride, sees the first first_n_pos + j positions of keys and values (head_dim floats
 * per position, the positions in a row) and writes head_dim floats at out + j*out_stride.
 * Every output is bit for bit tr_attention_head's for that query: the same dot_f32 per
 * position, the same softmax of the same row, the same axpy_f32 in increasing position. What
 * changes is the order in memory: a block of TR_ATTN_BLOCK positions meets every query of the
 * group while it sits in the cache, so the keys and the values come from memory once per group
 * instead of once per query (docs/MEASUREMENTS.md "Prefill su prompt lunghi"). A decode token is a
 * group of one, and runs as tr_attention_head: position by position, one ascending stream the
 * prefetcher follows (docs/MEASUREMENTS.md "The decode's attention, one position at a time").
 * scores: n_q rows of scratch owned by the caller, row j at scores + j*score_stride and at
 * least first_n_pos + j floats long. */
void tr_attention_group(const float *q, int64_t q_stride, const float *keys, const float *values, int64_t n_q,
                        int64_t first_n_pos, int64_t head_dim, float scale, float *scores, int64_t score_stride,
                        float *out, int64_t out_stride);

#endif
