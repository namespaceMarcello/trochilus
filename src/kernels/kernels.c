/* kernels.c — scalar CPU kernels. Defines the numbers (kernels.h): every SIMD
 * or assembly variant added later must reproduce this arithmetic bit for bit.
 * No FMA (built with -ffp-contract=off), no state except the kernel table. */
#include "kernels.h"
#include "kernels_internal.h"

#include <math.h>
#include <string.h>

#define half_to_float tr_half_to_float
#define lane_combine tr_lane_combine
#define q8_0_block_scale tr_q8_0_block_scale

/* ---- the 16-lane reduction contract -------------------------------------- */
/* hot: begin */

static float k_dot_f32(const float *a, const float *b, int64_t n) {
    float lane[TR_LANES] = {0};
    for (int64_t k = 0; k < n; k++) lane[k % TR_LANES] += a[k] * b[k];
    return lane_combine(lane);
}

/* Plain sum under the same lane contract (used by softmax). */
static float lane_sum(const float *x, int64_t n) {
    float lane[TR_LANES] = {0};
    for (int64_t k = 0; k < n; k++) lane[k % TR_LANES] += x[k];
    return lane_combine(lane);
}

/* ---- element-wise --------------------------------------------------------- */

static void k_axpy_f32(float *y, const float *x, float a, int64_t n) {
    for (int64_t k = 0; k < n; k++) y[k] = y[k] + a * x[k];
}

/* ---- 4 cache positions per call: the definition is 4 calls ------------------ */

static void k_dot_f32_x4(const float *a, const float *b, int64_t stride, int64_t n, float *out) {
    for (int j = 0; j < TR_ATTN_X; j++) out[j] = k_dot_f32(a, b + j * stride, n);
}

static void k_axpy_f32_x4(float *y, const float *x, int64_t stride, const float *a, int64_t n) {
    for (int j = 0; j < TR_ATTN_X; j++) k_axpy_f32(y, x + j * stride, a[j], n);
}

/* ---- F32 ------------------------------------------------------------------ */

static void k_dequant_f32(const void *row, float *out, int64_t n) {
    memcpy(out, row, (size_t)n * sizeof(float));
}

static float k_dot_row_f32(const void *row, const float *x, int64_t n) {
    return k_dot_f32((const float *)row, x, n);
}

/* ---- F16 ------------------------------------------------------------------ */

static void k_dequant_f16(const void *row, float *out, int64_t n) {
    const uint16_t *r = (const uint16_t *)row;
    for (int64_t i = 0; i < n; i++) out[i] = half_to_float(r[i]);
}

static float k_dot_row_f16(const void *row, const float *x, int64_t n) {
    const uint16_t *r = (const uint16_t *)row;
    float lane[TR_LANES] = {0};
    for (int64_t k = 0; k < n; k++) {
        float w = half_to_float(r[k]);
        lane[k % TR_LANES] += w * x[k];
    }
    return lane_combine(lane);
}

/* ---- Q8_0: 32 elements/block, 2-byte f16 scale then 32 int8 values -------- */

static void k_dequant_q8_0(const void *row, float *out, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        float scale = q8_0_block_scale(blk);
        const int8_t *qs = (const int8_t *)(blk + TR_Q8_0_SCALE_BYTES);
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j++)
            out[b * TR_Q8_0_BLOCK_ELEMS + j] = scale * (float)qs[j];
    }
}

static float k_dot_row_q8_0(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    float lane[TR_LANES] = {0};
    int64_t k = 0;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        float scale = q8_0_block_scale(blk);
        const int8_t *qs = (const int8_t *)(blk + TR_Q8_0_SCALE_BYTES);
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j++, k++) {
            float w = scale * (float)qs[j];
            lane[k % TR_LANES] += w * x[k];
        }
    }
    return lane_combine(lane);
}

/* One weight row against TR_DOT_TOKENS input rows: the block scale and the decoded
 * weight are computed once and used for all of them, each accumulating in its own
 * lanes, so out[t] is exactly k_dot_row_q8_0(row, x + t*stride, n). */
static void k_dot_row_x4_q8_0(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q8_0_BLOCK_ELEMS;
    float lane[TR_DOT_TOKENS][TR_LANES] = {{0}};
    int64_t k = 0;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        float scale = q8_0_block_scale(blk);
        const int8_t *qs = (const int8_t *)(blk + TR_Q8_0_SCALE_BYTES);
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j++, k++) {
            float w = scale * (float)qs[j];
            for (int t = 0; t < TR_DOT_TOKENS; t++) lane[t][k % TR_LANES] += w * x[t * stride + k];
        }
    }
    for (int t = 0; t < TR_DOT_TOKENS; t++) out[t] = lane_combine(lane[t]);
}

/* ---- Q4_K: 256 elements/block in 8 sub-blocks of 32 (kernels_internal.h) ---- */

static void k_dequant_q4_k(const void *row, float *out, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        float scale[8], min[8];
        tr_q4_k_scales(blk, scale, min);
        float *o = out + b * TR_Q4_K_BLOCK_ELEMS;
        for (int i = 0; i < TR_Q4_K_BLOCK_ELEMS; i++)
            o[i] = scale[i >> 5] * (float)tr_q4_k_quant(qs, i) - min[i >> 5];
    }
}

/* The weight is the dequantized one, element by element (never d * sum(q*x) - m * sum(x),
 * which rounds otherwise): dot_row(row, x) is dot_f32(dequant_row(row), x) bit for bit. */
static float k_dot_row_q4_k(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    float lane[TR_LANES] = {0};
    int64_t k = 0;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        float scale[8], min[8];
        tr_q4_k_scales(blk, scale, min);
        for (int i = 0; i < TR_Q4_K_BLOCK_ELEMS; i++, k++) {
            float w = scale[i >> 5] * (float)tr_q4_k_quant(qs, i) - min[i >> 5];
            lane[k % TR_LANES] += w * x[k];
        }
    }
    return lane_combine(lane);
}

static void k_dot_row_x4_q4_k(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q4_K_BLOCK_ELEMS;
    float lane[TR_DOT_TOKENS][TR_LANES] = {{0}};
    int64_t k = 0;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
        float scale[8], min[8];
        tr_q4_k_scales(blk, scale, min);
        for (int i = 0; i < TR_Q4_K_BLOCK_ELEMS; i++, k++) {
            float w = scale[i >> 5] * (float)tr_q4_k_quant(qs, i) - min[i >> 5];
            for (int t = 0; t < TR_DOT_TOKENS; t++) lane[t][k % TR_LANES] += w * x[t * stride + k];
        }
    }
    for (int t = 0; t < TR_DOT_TOKENS; t++) out[t] = lane_combine(lane[t]);
}

/* ---- Q6_K: 256 elements/block in 16 sub-blocks of 16 (kernels_internal.h) ---- */

static void k_dequant_q6_k(const void *row, float *out, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        float scale[16];
        tr_q6_k_scales(blk, scale);
        float *o = out + b * TR_Q6_K_BLOCK_ELEMS;
        for (int i = 0; i < TR_Q6_K_BLOCK_ELEMS; i++) o[i] = scale[i >> 4] * (float)tr_q6_k_quant(blk, i);
    }
}

/* the dequantized weight in every product, as Q4_K: dot_row is dot_f32(dequant_row) bit for bit */
static float k_dot_row_q6_k(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    float lane[TR_LANES] = {0};
    int64_t k = 0;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        float scale[16];
        tr_q6_k_scales(blk, scale);
        for (int i = 0; i < TR_Q6_K_BLOCK_ELEMS; i++, k++) {
            float w = scale[i >> 4] * (float)tr_q6_k_quant(blk, i);
            lane[k % TR_LANES] += w * x[k];
        }
    }
    return lane_combine(lane);
}

static void k_dot_row_x4_q6_k(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const unsigned char *p = (const unsigned char *)row;
    int64_t nb = n / TR_Q6_K_BLOCK_ELEMS;
    float lane[TR_DOT_TOKENS][TR_LANES] = {{0}};
    int64_t k = 0;
    for (int64_t b = 0; b < nb; b++) {
        const unsigned char *blk = p + (size_t)b * TR_Q6_K_BLOCK_BYTES;
        float scale[16];
        tr_q6_k_scales(blk, scale);
        for (int i = 0; i < TR_Q6_K_BLOCK_ELEMS; i++, k++) {
            float w = scale[i >> 4] * (float)tr_q6_k_quant(blk, i);
            for (int t = 0; t < TR_DOT_TOKENS; t++) lane[t][k % TR_LANES] += w * x[t * stride + k];
        }
    }
    for (int t = 0; t < TR_DOT_TOKENS; t++) out[t] = lane_combine(lane[t]);
}

static void k_dot_row_x4_f32(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const float *w = (const float *)row;
    float lane[TR_DOT_TOKENS][TR_LANES] = {{0}};
    for (int64_t k = 0; k < n; k++)
        for (int t = 0; t < TR_DOT_TOKENS; t++) lane[t][k % TR_LANES] += w[k] * x[t * stride + k];
    for (int t = 0; t < TR_DOT_TOKENS; t++) out[t] = lane_combine(lane[t]);
}

static void k_dot_row_x4_f16(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const uint16_t *r = (const uint16_t *)row;
    float lane[TR_DOT_TOKENS][TR_LANES] = {{0}};
    for (int64_t k = 0; k < n; k++) {
        float w = half_to_float(r[k]);
        for (int t = 0; t < TR_DOT_TOKENS; t++) lane[t][k % TR_LANES] += w * x[t * stride + k];
    }
    for (int t = 0; t < TR_DOT_TOKENS; t++) out[t] = lane_combine(lane[t]);
}

/* hot: end */

/* ---- type table and dispatch ---------------------------------------------- */

size_t tr_row_bytes(tr_type type, int64_t n) {
    const tr_type_info *ti = tr_type_get((uint32_t)type);
    if (ti == NULL || ti->block_elems == 0) return 0;
    if (n % (int64_t)ti->block_elems != 0) return 0;
    return (size_t)(n / (int64_t)ti->block_elems) * (size_t)ti->block_bytes;
}

int tr_kernels_support(tr_type type) {
    return type == TR_TYPE_F32 || type == TR_TYPE_F16 || type == TR_TYPE_Q8_0 || type == TR_TYPE_Q4_K ||
           type == TR_TYPE_Q6_K;
}

static tr_kernels g_scalar_kernels;          /* global-ok: kernel tables depend only on the CPU */
static int g_scalar_built = 0;               /* global-ok: same */
static const tr_kernels *g_active = NULL;    /* global-ok: tier chosen for the CPU (or forced by tests) */

static void build_scalar_table(tr_kernels *k) {
    memset(k, 0, sizeof *k);
    k->tier = "scalar";
    k->dot_f32 = k_dot_f32;
    k->axpy_f32 = k_axpy_f32;
    k->dot_f32_x4 = k_dot_f32_x4;
    k->axpy_f32_x4 = k_axpy_f32_x4;
    k->dot_row[TR_TYPE_F32] = k_dot_row_f32;
    k->dot_row[TR_TYPE_F16] = k_dot_row_f16;
    k->dot_row[TR_TYPE_Q8_0] = k_dot_row_q8_0;
    k->dot_row[TR_TYPE_Q4_K] = k_dot_row_q4_k;
    k->dot_row[TR_TYPE_Q6_K] = k_dot_row_q6_k;
    k->dot_row_x4[TR_TYPE_F32] = k_dot_row_x4_f32;
    k->dot_row_x4[TR_TYPE_F16] = k_dot_row_x4_f16;
    k->dot_row_x4[TR_TYPE_Q8_0] = k_dot_row_x4_q8_0;
    k->dot_row_x4[TR_TYPE_Q4_K] = k_dot_row_x4_q4_k;
    k->dot_row_x4[TR_TYPE_Q6_K] = k_dot_row_x4_q6_k;
    k->dequant_row[TR_TYPE_F32] = k_dequant_f32;
    k->dequant_row[TR_TYPE_F16] = k_dequant_f16;
    k->dequant_row[TR_TYPE_Q8_0] = k_dequant_q8_0;
    k->dequant_row[TR_TYPE_Q4_K] = k_dequant_q4_k;
    k->dequant_row[TR_TYPE_Q6_K] = k_dequant_q6_k;
}

const tr_kernels *tr_kernels_scalar(void) {
    if (!g_scalar_built) {
        build_scalar_table(&g_scalar_kernels);
        g_scalar_built = 1;
    }
    return &g_scalar_kernels;
}

void tr_kernels_init(void) {
    if (g_active != NULL) return; /* idempotent */
    /* Fastest first; tr_kernels_tier returns NULL for what this CPU (or TR_CPU_MAX) lacks. */
    static const char *const order[] = {"avx512", "avx2"};
    for (size_t i = 0; i < sizeof order / sizeof order[0] && g_active == NULL; i++)
        g_active = tr_kernels_tier(order[i]);
    if (g_active == NULL) g_active = tr_kernels_scalar();
}

const tr_kernels *tr_kernels_get(void) {
    return g_active;
}

void tr_kernels_set_active(const tr_kernels *k) {
    g_active = k;
    if (k == NULL) tr_kernels_init();
}

const tr_kernels *tr_kernels_tier(const char *tier) {
    if (tier == NULL) return NULL;
    if (strcmp(tier, "scalar") == 0) return tr_kernels_scalar();
    return tr_kernels_x86_tier(tier);
}

/* ---- operations built on the table ---------------------------------------- */

void tr_rope_table(float *cos_t, float *sin_t, int64_t n_pos, int64_t head_dim, float theta) {
    int64_t half = head_dim / 2;
    double theta_d = (double)theta;
    for (int64_t i = 0; i < half; i++) {
        double inv_freq = 1.0 / pow(theta_d, (2.0 * (double)i) / (double)head_dim);
        for (int64_t p = 0; p < n_pos; p++) {
            double angle = inv_freq * (double)p;
            cos_t[p * half + i] = (float)cos(angle);
            sin_t[p * half + i] = (float)sin(angle);
        }
    }
}

/* hot: begin */

/* Index space of a matmul job: p * rows + r (input row p, weight row r). */
typedef struct {
    const tr_mat *w;          /* [n_groups], all with the same type, rows and cols */
    const int64_t *offsets;   /* [n_groups + 1] */
    int64_t n_groups, rows, cols;
    const float *x;
    float *y;
    float (*dot_row)(const void *row, const float *x, int64_t n);
    void (*dot_row_x4)(const void *row, const float *x, int64_t stride, int64_t n, float *out);
    void (*dot_row2)(const void *row0, const void *row1, const float *x, int64_t n, float *out);
    void (*dot_row2_x4)(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n, float *out);
    void (*dot_row2_x8)(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n, float *out);
    size_t row_bytes;
} matmul_ctx;

/* The group of input row p: the last g with offsets[g] <= p (an empty group before it
 * shares its offset, and only the last one of them has offsets[g+1] > p). */
static int64_t matmul_group_of(const matmul_ctx *c, int64_t p) {
    int64_t lo = 0, hi = c->n_groups - 1;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo + 1) / 2;
        if (c->offsets[mid] <= p) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

/* weight rows [r0, r1) against input row p: two at a time where the tier has dot_row2 (the
 * decode's road), each result dot_row's */
static void matmul_rows(const matmul_ctx *c, int64_t p, int64_t r0, int64_t r1) {
    const unsigned char *base = (const unsigned char *)c->w[matmul_group_of(c, p)].data;
    const float *xp = c->x + p * c->cols;
    float *yp = c->y + p * c->rows;
    int64_t r = r0;
    if (c->dot_row2 != NULL)
        for (; r + 2 <= r1; r += 2)
            c->dot_row2(base + (size_t)r * c->row_bytes, base + (size_t)(r + 1) * c->row_bytes, xp, c->cols, yp + r);
    for (; r < r1; r++) yp[r] = c->dot_row(base + (size_t)r * c->row_bytes, xp, c->cols);
}

/* every weight row against input rows [p0, p1), group by group, in blocks of TR_MATMUL_TILE.
 * Inside a block the rows go TR_DOT_TOKENS at a time through dot_row_x4 where the type has
 * one: the weight is read and decoded once for all of them, and each result is the dot_row
 * of its own input row, bit for bit. */
static void matmul_tiled(const matmul_ctx *c, int64_t p0, int64_t p1) {
    int64_t g = matmul_group_of(c, p0);
    for (int64_t p = p0; p < p1;) {
        while (c->offsets[g + 1] <= p) g++;
        int64_t run_end = c->offsets[g + 1] < p1 ? c->offsets[g + 1] : p1;
        const unsigned char *base = (const unsigned char *)c->w[g].data;
        for (int64_t b = p; b < run_end; b += TR_MATMUL_TILE) {
            int64_t b_end = b + TR_MATMUL_TILE < run_end ? b + TR_MATMUL_TILE : run_end;
            int64_t r = 0;
            /* two rows at a time where the tier has the kernel: each input element loaded once
             * for both; eight tokens a call where it has that one too, then four; the tokens past
             * the last whole group go one dot at a time */
            if (c->dot_row2_x4 != NULL) {
                for (; r + 2 <= c->rows; r += 2) {
                    const void *row0 = base + (size_t)r * c->row_bytes, *row1 = base + (size_t)(r + 1) * c->row_bytes;
                    int64_t q = b;
                    float out[2 * TR_DOT_TOKENS_WIDE];
                    if (c->dot_row2_x8 != NULL) {
                        for (; q + TR_DOT_TOKENS_WIDE <= b_end; q += TR_DOT_TOKENS_WIDE) {
                            c->dot_row2_x8(row0, row1, c->x + q * c->cols, c->cols, c->cols, out);
                            for (int j = 0; j < TR_DOT_TOKENS_WIDE; j++) {
                                c->y[(q + j) * c->rows + r] = out[j];
                                c->y[(q + j) * c->rows + r + 1] = out[TR_DOT_TOKENS_WIDE + j];
                            }
                        }
                    }
                    for (; q + TR_DOT_TOKENS <= b_end; q += TR_DOT_TOKENS) {
                        c->dot_row2_x4(row0, row1, c->x + q * c->cols, c->cols, c->cols, out);
                        for (int j = 0; j < TR_DOT_TOKENS; j++) {
                            c->y[(q + j) * c->rows + r] = out[j];
                            c->y[(q + j) * c->rows + r + 1] = out[TR_DOT_TOKENS + j];
                        }
                    }
                    for (; q < b_end; q++) {
                        if (c->dot_row2 != NULL) {
                            c->dot_row2(row0, row1, c->x + q * c->cols, c->cols, out);
                            c->y[q * c->rows + r] = out[0];
                            c->y[q * c->rows + r + 1] = out[1];
                        } else {
                            c->y[q * c->rows + r] = c->dot_row(row0, c->x + q * c->cols, c->cols);
                            c->y[q * c->rows + r + 1] = c->dot_row(row1, c->x + q * c->cols, c->cols);
                        }
                    }
                }
            }
            for (; r < c->rows; r++) {
                const void *row = base + (size_t)r * c->row_bytes;
                int64_t q = b;
                if (c->dot_row_x4 != NULL) {
                    float out[TR_DOT_TOKENS];
                    for (; q + TR_DOT_TOKENS <= b_end; q += TR_DOT_TOKENS) {
                        c->dot_row_x4(row, c->x + q * c->cols, c->cols, c->cols, out);
                        for (int j = 0; j < TR_DOT_TOKENS; j++) c->y[(q + j) * c->rows + r] = out[j];
                    }
                }
                for (; q < b_end; q++) c->y[q * c->rows + r] = c->dot_row(row, c->x + q * c->cols, c->cols);
            }
        }
        p = run_end;
    }
}

/* A chunk may start or end inside an input row: those rows go one dot at a time, the
 * whole input rows in between tiled. */
static void matmul_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const matmul_ctx *c = (const matmul_ctx *)ctx_;
    int64_t p_begin = begin / c->rows, r_begin = begin % c->rows;
    int64_t p_end = end / c->rows, r_end = end % c->rows;
    if (p_begin == p_end) {
        matmul_rows(c, p_begin, r_begin, r_end);
        return;
    }
    int64_t p_full = p_begin;
    if (r_begin != 0) {
        matmul_rows(c, p_begin, r_begin, c->rows);
        p_full++;
    }
    if (p_full < p_end) matmul_tiled(c, p_full, p_end);
    if (r_end != 0) matmul_rows(c, p_end, 0, r_end);
}

void tr_matmul_grouped(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const float *x,
                       float *y) {
    if (n_groups <= 0 || offsets[n_groups] <= 0 || w[0].rows <= 0) return;
    matmul_ctx ctx;
    ctx.w = w;
    ctx.offsets = offsets;
    ctx.n_groups = n_groups;
    ctx.rows = w[0].rows;
    ctx.cols = w[0].cols;
    ctx.x = x;
    ctx.y = y;
    const tr_kernels *k = tr_kernels_get();
    ctx.dot_row = k->dot_row[w[0].type];
    ctx.dot_row_x4 = k->dot_row_x4[w[0].type];
    ctx.dot_row2 = k->dot_row2[w[0].type];
    ctx.dot_row2_x4 = k->dot_row2_x4[w[0].type];
    ctx.dot_row2_x8 = k->dot_row2_x8[w[0].type];
    ctx.row_bytes = tr_row_bytes(w[0].type, w[0].cols);

    int64_t n = offsets[n_groups] * ctx.rows;
    /* Keep chunks worth threading: roughly a few thousand scalar multiplies
     * worth of rows per chunk, never less than one row. */
    int64_t min_chunk = ctx.cols > 0 ? (4096 / ctx.cols) + 1 : 1;
    tr_parallel_for(pool, n, min_chunk, matmul_body, &ctx);
}

void tr_matmul(tr_pool *pool, const tr_mat *w, const float *x, int64_t n_tokens, float *y) {
    const int64_t offsets[2] = {0, n_tokens};
    tr_matmul_grouped(pool, w, offsets, 1, x, y);
}

void tr_get_row(const tr_mat *w, int64_t row, float *out) {
    const tr_kernels *k = tr_kernels_get();
    size_t rb = tr_row_bytes(w->type, w->cols);
    const unsigned char *base = (const unsigned char *)w->data;
    const void *rowp = base + (size_t)row * rb;
    k->dequant_row[w->type](rowp, out, w->cols);
}

void tr_rmsnorm(float *x, const float *weight, int64_t n, float eps) {
    const tr_kernels *k = g_active != NULL ? g_active : tr_kernels_scalar();
    float sumsq = k->dot_f32(x, x, n);
    float mean = sumsq / (float)n;
    float inv = 1.0f / sqrtf(mean + eps);
    for (int64_t i = 0; i < n; i++) {
        float t = x[i] * inv;
        x[i] = t * weight[i];
    }
}

void tr_rope_neox(float *x, int64_t n_heads, int64_t head_dim, const float *cos_p, const float *sin_p) {
    int64_t half = head_dim / 2;
    for (int64_t h = 0; h < n_heads; h++) {
        float *base = x + h * head_dim;
        for (int64_t i = 0; i < half; i++) {
            float x1 = base[i];
            float x2 = base[i + half];
            base[i] = x1 * cos_p[i] - x2 * sin_p[i];
            base[i + half] = x2 * cos_p[i] + x1 * sin_p[i];
        }
    }
}

void tr_softmax(float *x, int64_t n) {
    if (n <= 0) return;
    float m = x[0];
    for (int64_t i = 1; i < n; i++)
        if (x[i] > m) m = x[i];
    for (int64_t i = 0; i < n; i++) x[i] = tr_expf(x[i] - m);
    float sum = lane_sum(x, n);
    for (int64_t i = 0; i < n; i++) x[i] /= sum;
}

typedef struct {
    float *x;
    const float *y;
} swiglu_ctx;

static void swiglu_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    swiglu_ctx *ctx = (swiglu_ctx *)ctx_;
    float *x = ctx->x;
    const float *y = ctx->y;
    for (int64_t i = begin; i < end; i++) {
        float v = x[i];
        float silu = v / (1.0f + tr_expf(-v));
        x[i] = silu * y[i];
    }
}

void tr_swiglu(tr_pool *pool, float *x, const float *y, int64_t n) {
    swiglu_ctx ctx;
    ctx.x = x;
    ctx.y = y;
    /* one tr_expf per element, a few ns: a few hundred per chunk are what waking a thread costs */
    tr_parallel_for(pool, n, 256, swiglu_body, &ctx);
}

void tr_attention_head(const float *q, const float *keys, const float *values, int64_t stride, int64_t offset,
                       int64_t n_pos, int64_t head_dim, float scale, float *scores, float *out) {
    const tr_kernels *k = g_active != NULL ? g_active : tr_kernels_scalar();
    for (int64_t t = 0; t < n_pos; t++) scores[t] = k->dot_f32(q, keys + t * stride + offset, head_dim) * scale;
    tr_softmax(scores, n_pos);
    for (int64_t d = 0; d < head_dim; d++) out[d] = 0.0f;
    for (int64_t t = 0; t < n_pos; t++) k->axpy_f32(out, values + t * stride + offset, scores[t], head_dim);
}

/* Block by block: a block of keys meets every query of the group that sees it, and fills its
 * piece of their score rows; then the softmax of every row; then the values block by block. A
 * query sees a block up to its own position, so the last blocks of a group are cut query by
 * query. Inside a block the positions go 4 at a time, in increasing order. */
void tr_attention_group(const float *q, int64_t q_stride, const float *keys, const float *values, int64_t n_q,
                        int64_t first_n_pos, int64_t head_dim, float scale, float *scores, int64_t score_stride,
                        float *out, int64_t out_stride) {
    /* One query (a decode token) goes position by position: tr_attention_head's single dot and
     * axpy read each key and value row as one ascending stream, which the prefetcher follows;
     * the x4 kernels read four rows 512 bytes apart a cache line of each in turn, and a lone query
     * shares nothing across them. Same bits (the x4 kernels are four calls by contract), the
     * decode's attention 1.10-1.13x faster (docs/MEASUREMENTS.md "The decode's attention, one
     * position at a time"). */
    if (n_q == 1) {
        tr_attention_head(q, keys, values, head_dim, 0, first_n_pos, head_dim, scale, scores, out);
        return;
    }
    const tr_kernels *k = g_active != NULL ? g_active : tr_kernels_scalar();
    int64_t last_n_pos = first_n_pos + n_q - 1;
    for (int64_t t0 = 0; t0 < last_n_pos; t0 += TR_ATTN_BLOCK) {
        int64_t t1 = t0 + TR_ATTN_BLOCK < last_n_pos ? t0 + TR_ATTN_BLOCK : last_n_pos;
        int64_t j0 = t0 + 1 > first_n_pos ? t0 + 1 - first_n_pos : 0; /* the first query that sees t0 */
        for (int64_t j = j0; j < n_q; j++) {
            const float *qj = q + j * q_stride;
            float *row = scores + j * score_stride;
            int64_t n_pos = first_n_pos + j, end = t1 < n_pos ? t1 : n_pos, t = t0;
            for (; t + TR_ATTN_X <= end; t += TR_ATTN_X) {
                k->dot_f32_x4(qj, keys + t * head_dim, head_dim, head_dim, row + t);
                for (int x = 0; x < TR_ATTN_X; x++) row[t + x] = row[t + x] * scale;
            }
            for (; t < end; t++) row[t] = k->dot_f32(qj, keys + t * head_dim, head_dim) * scale;
        }
    }
    for (int64_t j = 0; j < n_q; j++) {
        float *oj = out + j * out_stride;
        tr_softmax(scores + j * score_stride, first_n_pos + j);
        for (int64_t d = 0; d < head_dim; d++) oj[d] = 0.0f;
    }
    for (int64_t t0 = 0; t0 < last_n_pos; t0 += TR_ATTN_BLOCK) {
        int64_t t1 = t0 + TR_ATTN_BLOCK < last_n_pos ? t0 + TR_ATTN_BLOCK : last_n_pos;
        int64_t j0 = t0 + 1 > first_n_pos ? t0 + 1 - first_n_pos : 0;
        for (int64_t j = j0; j < n_q; j++) {
            float *oj = out + j * out_stride;
            const float *row = scores + j * score_stride;
            int64_t n_pos = first_n_pos + j, end = t1 < n_pos ? t1 : n_pos, t = t0;
            for (; t + TR_ATTN_X <= end; t += TR_ATTN_X)
                k->axpy_f32_x4(oj, values + t * head_dim, head_dim, row + t, head_dim);
            for (; t < end; t++) k->axpy_f32(oj, values + t * head_dim, row[t], head_dim);
        }
    }
}
/* hot: end */
