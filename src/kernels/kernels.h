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
/* Q4_K's integer definition (docs/ARCHITECTURE.md §Q4_K in integers; the table's q4x_* entries): the widest
 * tile of prepared input rows, the bytes of a 16-row panel of n columns, the largest |X| of an input element
 * (the two balanced 16-bit digits' reach, 32295 + 64591 * 32295) and the digits' base. */
#define TR_Q4X_TILE_MAX 4
/* The most prepared input rows a q4x_dot_xt call takes (a short verify pass's run). */
#define TR_Q4X_XT_MAX 3
#define TR_Q4X_PANEL_BYTES(n) ((size_t)(n) / 256 * 9728)
#define TR_Q4X_XMAX 2085998640.0
#define TR_Q4X_BASE 64591
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
/* n F32 values at data rewritten in place as n bfloat16 (their top halves, the first 2n bytes) when every one has
 * a zero low half, so that nothing is lost: 1 then, and a BF16 row gives dot_row's bits of the F32 row it came from
 * (the same products in the same lanes, widened by a shift). 0, the data untouched, when one value does not fit.
 * An F32 tensor of a model converted from bf16 fits whole (OLMoE's routers: half their bytes). */
int tr_f32_to_bf16_exact(void *data, int64_t n);
/* Bytes of one input row of n columns prepared for Q4_K's integer definition (the table's q4x_prep; a
 * multiple of 64: the rounded tail is never written, so two prepared rows compare equal only past what their
 * buffers held before, LESSONS #246). */
size_t tr_q4x_bytes(int64_t n);

/* ---- The output head by a bound (head_bound.c; docs/MEASUREMENTS.md §The head's argmax by a bound) ----
 * A Q8_0 or Q4_K head of n columns (a multiple of 256, at most TR_HB_COLS_MAX) is split at load into two planes, the
 * same bytes: the high plane holds a row's scales and its codes' top bits, the low plane the rest.
 *   Q8_0, a block of 32 b: hi = [nb x 2 B: the f16 scales][nb x 16 B: byte i = u_i | u_(i+16) << 4, u = (q >> 4) + 8],
 *     lo = [nb x 16 B: byte i = (q_i & 15) | (q_(i+16) & 15) << 4];
 *   Q4_K, a super-block: hi = [16 B: d, dmin and the twelve scale bytes][64 B: q_k >> 2 at byte 32 (k / 128) + k % 32,
 *     shift 2 ((k % 128) / 32)][32 B: bit 1 of q_k at byte k / 8, bit k % 8], lo = [32 B: bit 0, the same way].
 * A token h is prepared once (tr_hb_prep_build): per block of 32, delta = max|h| / 32639 and h ~ delta (256 X + Xl),
 * two int8 digits, with the block's terms, where the float sums' margin (2^-16 of the terms' magnitudes), h's own
 * rounding (sum |h - delta (256 X + Xl)| at the codes' largest) and the engine's Q4_K rounding of h (2^-30 of a
 * super-block's max|h| an element) are folded in. A row's bound from its high plane
 *   Q8_0: U = sum_b d_b (a_b I_b + be_b) + |d_b| g_b,                      I_b = sum_k u_k (256 X_k + Xl_k)
 *   Q4_K: U = sum_j d sc_j (a_j I_j + be_j) + |d| sc_j g_j - dmin m_j H_j + |dmin| m_j mu_j,   I_j over q >> 1
 * is at or above the engine's own value of the row (dot_row, q4x_dot2) for every h of finite |h| <= 2^64: only the
 * rows whose bound reaches the best exact score found can be the argmax. Every tier's float order: I_b as eight int32
 * lanes of four elements (exact), lane l of block b (Q4_K: sub-block) into chain b % 4 as acc + (float)P * coef, coef
 * = d a (Q4_K: (d sc) a), the side terms of block b into lane b % 8 in the order written above, then lane l = ((c0 +
 * c2) + (c1 + c3)) + side and the eight lanes ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)). */
#define TR_HB_COLS_MAX 16384
/* The rows of the largest bounds computed first: the best of their scores is the threshold (1.02-1.1x the least rows
 * computed; the best bound's row alone 1.2-2.7x: docs/MEASUREMENTS.md, LESSONS #345) */
#define TR_HB_TOP 4
typedef struct {
    int8_t *X, *Xl;             /* n each */
    float *a, *be, *g, *H, *mu; /* n / 32 each (H, mu: Q4_K's) */
} tr_hb_prep;

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
    /* The same for t input rows, t 2 or 3 (a short verify pass's group, too few for dot_row_x4):
     * out[j] = dot_row(row, x + j*stride, n) for j < t, bit for bit, the row decoded once for all
     * of them. NULL: the caller calls dot_row for each input row. */
    void (*dot_row_xt[TR_TYPE_COUNT])(const void *row, const float *x, int64_t stride, int64_t n, int t, float *out);
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
     * pm_tile: the panel's rows against the T interleaved rows, T from 4 to TR_PM_TILE_MAX:
     *   y[t * y_stride + r] = dot_row(row r, x_t, n) bit for bit; part is TR_PM_PART floats of scratch.
     * NULL where the tier has none: the caller keeps dot_row2_x8 and the others. */
    void (*pm_panel[TR_TYPE_COUNT])(const void *rows, size_t row_bytes, int64_t n, float *panel);
    void (*pm_interleave)(const float *x, int64_t stride, int64_t n, int T, float *xil);
    void (*pm_tile)(const float *panel, const float *xil, int64_t n, int T, float *part, float *y, int64_t y_stride);
    /* Q4_K's integer definition (docs/ARCHITECTURE.md §Q4_K in integers), a Q4_K row against an input row x
     * of n columns (a multiple of 256): per super-block s of x, sh_s the largest shift with max|x_k| 2^sh_s
     * <= TR_Q4X_XMAX (0 for an all-zero block), X_k = nearest-even(x_k 2^sh_s), T_s = sum_k sc_j q_k X_k and
     * M_s = sum_j m_j sum_{k in j} X_k (exact integers), v_s = fl64(fl64(d_s T_s) - fl64(dmin_s M_s)); the
     * result fl32(sum_s v_s 2^-sh_s), the sum in f64 in increasing s from +0, and NaN when a block of x
     * holds a NaN or an infinity. Every sum inside a block is exact: any order, width, split or device
     * gives these bits. dot_row[TR_TYPE_Q4_K] computes it from the floats; these from an input row
     * prepared once (its digits, the windows' token terms, the sub-blocks' sums and the shifts):
     * q4x_prep: x (n floats) -> xq (tr_q4x_bytes(n) bytes, 64-byte aligned);
     * q4x_dot2: out[0], out[1] = rows row0 and row1 against the prepared row xq (the decode's road);
     * q4x_dot_xt: rows row0 and row1 against T prepared rows xq[0], ..., xq[T - 1] (anywhere: a group's rows
     *   read in place from the rows a pass prepared once), T from 2 to TR_Q4X_XT_MAX: y[t * y_stride + r] = row
     *   r against input row t, each weight decoded once for the T of them (a short verify pass's groups);
     * q4x_panel: TR_PM_ROWS consecutive rows (row_bytes apart) -> panel (TR_Q4X_PANEL_BYTES(n) bytes,
     *   64-byte aligned); next, when not NULL, is the TR_PM_ROWS rows the caller builds after these (the
     *   same row_bytes and n), which a tier may bring toward its caches while it builds (never read);
     * q4x_tile: the panel against T prepared rows xq[0], ..., xq[T - 1], T from 1 to TR_Q4X_TILE_MAX:
     *   y[t * y_stride + r] = the panel's row r against input row t;
     * q4x_tile_max: the widest tile the planner gives q4x_tile, 3 or 4 (AVX2's T = 4 spills: docs/MEASUREMENTS.md
     *   §AVX2's own Q4_K tile). */
    void (*q4x_prep)(const float *x, int64_t n, void *xq);
    void (*q4x_dot2)(const void *row0, const void *row1, const void *xq, int64_t n, float *out);
    void (*q4x_dot_xt)(const void *row0, const void *row1, const void *const *xq, int64_t n, int T, float *y,
                       int64_t y_stride);
    void (*q4x_panel)(const void *rows, size_t row_bytes, int64_t n, void *panel, const void *next);
    void (*q4x_tile)(const void *panel, const void *const *xq, int64_t n, int T, float *y, int64_t y_stride);
    int q4x_tile_max;
    /* The head's planes (above): hb_bounds writes U[r] = the bound of row r < rows (their high planes consecutive,
     * tr_hb_hi_bytes apart) against p, every tier the scalar's bits; hb_rebuild a row's original bytes from its two
     * planes. NULL for a type without planes. */
    void (*hb_bounds[TR_TYPE_COUNT])(const unsigned char *hi, int64_t rows, const tr_hb_prep *p, int64_t n, float *U);
    void (*hb_rebuild[TR_TYPE_COUNT])(const unsigned char *hi, const unsigned char *lo, int64_t n, unsigned char *row);
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
    unsigned char *xq;          /* Q4_K's prepared input rows (the table's q4x_prep), the same memory as xil: a call
                                 * takes one road or the other */
    int64_t xq_bytes;           /* max_tokens * tr_q4x_bytes(max_cols) */
    float *work;                /* n_workers * (TR_PM_PANEL_FLOATS(max_cols) + TR_PM_PART) floats: panels with pads */
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
 * call fits s; a group of fewer than 4 input rows goes by rows, 2 or 3 of them through dot_row_xt. A short
 * pass (no group of 4 input rows, one of 2 or 3: a verify pass of 2-3 rows) takes the road on any tier,
 * items cut by weight rows and balanced; one input row a group stays on tr_matmul_grouped's. A Q4_K weight takes its
 * integer road where the call fits s: every input row prepared once (q4x_prep), then the W16 panel and
 * tiles for a group of 4 input rows or more where the tier has them, a smaller group's runs of 2-3 input
 * rows by q4x_dot_xt and a lone input row by q4x_dot2; each output is dot_row[TR_TYPE_Q4_K]'s. s == NULL:
 * the old road (for Q4_K, dot_row from the floats). */
void tr_matmul_grouped_s(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const float *x,
                         float *y, const tr_pm_scratch *s);
void tr_matmul_s(tr_pool *pool, const tr_mat *w, const float *x, int64_t n_tokens, float *y, const tr_pm_scratch *s);
/* Q4_K's integer road on rows prepared once for several calls (docs/MEASUREMENTS.md §The prep once a row): a
 * pass's distinct input rows (the tokens) prepared by tr_q4x_prepare into the caller's buffer (n *
 * tr_q4x_bytes(cols) bytes, 64-byte aligned; never s->xq, which a phase-major call between two readers
 * overwrites), then every call reading them: q, k and v from the same rows, gate and up from the tokens'
 * rows through a map, each group's input row p reading prepared row map[p] (NULL: row p) without a copy.
 * tr_q4x_road: 1 when a call of these weights and groups can take the road on pool and s;
 * tr_matmul_q4x_prepared: tr_matmul_grouped_s's y, bit for bit, from the prepared rows; 0 (y untouched) when the
 * road cannot take the call, whose caller then gathers the floats and takes tr_matmul_grouped_s. */
void tr_q4x_prepare(tr_pool *pool, const float *x, int64_t n, int64_t cols, void *xq);
int tr_q4x_road(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const tr_pm_scratch *s);
int tr_matmul_q4x_prepared(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const void *xq,
                           const int64_t *map, float *y, const tr_pm_scratch *s);
/* tr_matmul_q4x_prepared over n_w (1..TR_Q4X_MATS) weights of one shape on the same groups and prepared rows (q, k and
 * v; gate and up), in one call of the pool: each thread runs its items on w[0] into y[0], then the same items on w[1]
 * into y[1], and so on (docs/MEASUREMENTS.md §Piece 4). y[m] is tr_matmul_q4x_prepared's on w[m], bit for bit; 0 (every
 * y untouched) when the road cannot take one of them or their shapes differ. tr_q4x_fused_calls: the calls of n_w > 1
 * taken since the process started (tests). */
#define TR_Q4X_MATS 3
int tr_matmul_q4x_prepared_n(tr_pool *pool, const tr_mat *const *w, int n_w, const int64_t *offsets, int64_t n_groups,
                             const void *xq, const int64_t *map, float *const *y, const tr_pm_scratch *s);
uint64_t tr_q4x_fused_calls(void);
/* 1 if no group of tr_matmul_grouped's has more than one input row: a decode token's projections
 * (one group, one row) and its experts (n_expert groups, the used ones one row each, the rest
 * empty), whose call is only weight rows to stream and runs balanced at its tail. Until
 * 2026-09-26 the test was "rows == groups", never true for the experts' 64 groups and 8 rows
 * (docs/LESSONS.md #192). */
int tr_matmul_one_row_per_group(const int64_t *offsets, int64_t n_groups);
/* The idle workers' hints for the next call of these weights and groups (threads.h tr_pool_hint): one input row a
 * group only (a decode token's calls; any other call, and a pool whose hints are off, posts nothing). Each chunk
 * but the calling thread's gets the first bytes its thread will read in that call, as tr_matmul_grouped_s (s) or
 * tr_matmul_q4x_prepared (s), or tr_matmul and tr_matmul_grouped (s NULL) split it: at most max_bytes, at most
 * half its region, never past its first group's matrix. Called by the pool's dispatcher before the serial steps
 * that precede the call; it moves no data the call reads. */
void tr_matmul_hint(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const tr_pm_scratch *s,
                    size_t max_bytes);
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
/* tr_swiglu on n_rows rows of n, split by whole rows, each row then prepared for Q4_K's integer road into xq (row r
 * at r * tr_q4x_bytes(n), the table's q4x_prep) while it is in its core's cache: the down's rows prepared in the
 * call that makes them (docs/MEASUREMENTS.md §The prep once a row). x's floats are tr_swiglu's, xq's bytes are
 * q4x_prep's of them. */
void tr_swiglu_prepare(tr_pool *pool, float *x, const float *y, int64_t n_rows, int64_t n, void *xq);
/* The first index of the largest x[i] (0 <= i < n), the serial scan `if (x[i] > x[best]) best = i` from best = 0: a
 * NaN never wins, ties go to the lower index, and index 0 stands when nothing is larger (a NaN there, or every value
 * -inf). Split over the pool (NULL: serial) by chunks of TR_PM_ROWS-aligned rows, each worker the logits the head's
 * matmul left in its cache: each chunk from (-inf, none) with strict >, eight interleaved lanes, the chunks merged in
 * order from (x[0], 0) with the same strict >: the scan's index by construction (docs/MEASUREMENTS.md §The argmax in
 * parallel). */
int32_t tr_argmax_f32(tr_pool *pool, const float *x, int64_t n);

/* ---- the output head by a bound (the table's hb_* and their layout above) ---- */

/* 1 when a head of this type and n columns has planes: Q8_0 or Q4_K, n a multiple of 256 up to TR_HB_COLS_MAX */
int tr_hb_supports(tr_type type, int64_t n);
/* bytes of a row's high and low plane (0 without planes); together tr_row_bytes(type, n) */
size_t tr_hb_hi_bytes(tr_type type, int64_t n);
size_t tr_hb_lo_bytes(tr_type type, int64_t n);
/* the planes of one row (scalar: once, at load) */
void tr_hb_planes(tr_type type, const void *row, int64_t n, unsigned char *hi, unsigned char *lo);
/* bytes of a prepared token (64-byte aligned), and a tr_hb_prep over buf (64-byte aligned) */
size_t tr_hb_prep_bytes(int64_t n);
tr_hb_prep tr_hb_prep_view(void *buf, int64_t n);
/* h (n floats) prepared for type's bound into p (the scalar definition, every tier's): 0, or -1 (p undefined) when h
 * holds a NaN, an infinity or an |h| above 2^64 */
int tr_hb_prep_build(tr_type type, const float *h, int64_t n, const tr_hb_prep *p);

/* A head held as its two planes: rows of cols, row r's at hi + r * tr_hb_hi_bytes and lo + r * tr_hb_lo_bytes */
typedef struct {
    tr_type type;
    int64_t rows, cols;
    const unsigned char *hi, *lo;
} tr_hb_head;
/* A session's memory for its head calls: the prepared token, a bound and a candidate a row, the prepared input rows of
 * Q4_K's road, two rebuilt rows and a cache line of results a worker */
typedef struct {
    int n_workers;
    int64_t max_tokens;
    void *prep_buf;
    tr_hb_prep prep;
    float *U;
    int32_t *ids;
    unsigned char *xq;
    size_t xq_row;
    unsigned char *rows;
    size_t row_stride;
    unsigned char *workers;
} tr_hb_scratch;
/* for a pool of up to n_workers and calls of up to max_tokens input rows: 0, or -1 (nothing held) */
int tr_hb_scratch_init(tr_hb_scratch *s, const tr_hb_head *w, int n_workers, int64_t max_tokens);
void tr_hb_scratch_free(tr_hb_scratch *s);
typedef struct {
    int64_t rows; /* computed exactly: the TR_HB_TOP of the largest bounds and those reaching the threshold */
    int full;     /* 1: every row computed into y (h, a bound or a score not finite) */
} tr_hb_result;
/* y[t * rows + r] = row r against x + t * cols (t < n_tokens <= max_tokens), from the planes: tr_matmul_s's bits on
 * the original rows (dot_row's, Q4_K's integer definition) */
void tr_hb_logits(tr_pool *pool, const tr_hb_head *w, const float *x, int64_t n_tokens, float *y, tr_hb_scratch *s);
/* tr_argmax_f32 over the first n of the logits tr_hb_logits would give h, by construction: region 1 the bounds and
 * each worker's TR_HB_TOP largest; those rows exactly, the best score the threshold; region 2 the rows whose bound
 * reaches it, exactly, the lowest row of the largest. When h, a bound or a score is not finite, every row into y
 * (rows floats) and tr_argmax_f32 (res->full); else y untouched. */
int32_t tr_hb_argmax(tr_pool *pool, const tr_hb_head *w, const float *h, int64_t n, tr_hb_scratch *s, float *y,
                     tr_hb_result *res);
/* The idle workers' hint for tr_hb_argmax over n rows (tr_matmul_hint's for the bound's region 1): each the first
 * half of its region of the high plane, at most max_bytes. Moves no data the call reads. */
void tr_hb_hint(tr_pool *pool, const tr_hb_head *w, int64_t n, size_t max_bytes);
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
