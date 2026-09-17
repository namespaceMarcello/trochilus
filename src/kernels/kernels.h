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
    /* dot of one quantized row (n elements) with f32 x, under the lane contract
     * applied to the dequantized weights: w_k = d_block * q_k computed as a
     * float product first, then w_k * x_k into lane k % 16 */
    float (*dot_row[TR_TYPE_COUNT])(const void *row, const float *x, int64_t n);
    /* TR_DOT_TOKENS dots of the same row with consecutive input rows x, x+stride, ...:
     * out[j] = dot_row(row, x + j*stride, n), each under the lane contract, with the row
     * read and decoded once for all of them (fewer bytes and conversions per element, same
     * numbers). NULL for a type without a variant: the caller loops over dot_row. */
    void (*dot_row_x4[TR_TYPE_COUNT])(const void *row, const float *x, int64_t stride, int64_t n, float *out);
    /* decode one row of n elements to f32 */
    void (*dequant_row[TR_TYPE_COUNT])(const void *row, float *out, int64_t n);
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
/* In place softmax over n logits (max-subtracted, sum via the lane contract). */
void tr_softmax(float *x, int64_t n);
/* x[i] = silu(x[i]) * y[i]; element-wise, split over the pool (p == NULL: serial). */
void tr_swiglu(tr_pool *pool, float *x, const float *y, int64_t n);
/* Causal attention of one query head over n_pos cached positions. `keys` and `values`
 * hold n_pos slots of `stride` floats; this head reads head_dim floats at `offset` in
 * each slot. scores[t] = dot_f32(q, k_t) * scale, softmax over t < n_pos, then
 * out = 0 + scores[0]*v_0 + scores[1]*v_1 + ... in increasing t (one axpy_f32 per t).
 * scores: n_pos floats of scratch owned by the caller. */
void tr_attention_head(const float *q, const float *keys, const float *values, int64_t stride, int64_t offset,
                       int64_t n_pos, int64_t head_dim, float scale, float *scores, float *out);

#endif
