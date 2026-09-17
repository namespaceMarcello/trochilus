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

/* hot: end */

/* ---- type table and dispatch ---------------------------------------------- */

size_t tr_row_bytes(tr_type type, int64_t n) {
    const tr_type_info *ti = tr_type_get((uint32_t)type);
    if (ti == NULL || ti->block_elems == 0) return 0;
    if (n % (int64_t)ti->block_elems != 0) return 0;
    return (size_t)(n / (int64_t)ti->block_elems) * (size_t)ti->block_bytes;
}

int tr_kernels_support(tr_type type) {
    return type == TR_TYPE_F32 || type == TR_TYPE_F16 || type == TR_TYPE_Q8_0;
}

static tr_kernels g_scalar_kernels;          /* global-ok: kernel tables depend only on the CPU */
static int g_scalar_built = 0;               /* global-ok: same */
static const tr_kernels *g_active = NULL;    /* global-ok: tier chosen for the CPU (or forced by tests) */

static void build_scalar_table(tr_kernels *k) {
    memset(k, 0, sizeof *k);
    k->tier = "scalar";
    k->dot_f32 = k_dot_f32;
    k->axpy_f32 = k_axpy_f32;
    k->dot_row[TR_TYPE_F32] = k_dot_row_f32;
    k->dot_row[TR_TYPE_F16] = k_dot_row_f16;
    k->dot_row[TR_TYPE_Q8_0] = k_dot_row_q8_0;
    k->dequant_row[TR_TYPE_F32] = k_dequant_f32;
    k->dequant_row[TR_TYPE_F16] = k_dequant_f16;
    k->dequant_row[TR_TYPE_Q8_0] = k_dequant_q8_0;
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

typedef struct {
    const tr_mat *w;
    const float *x;
    float *y;
    const tr_kernels *k;
    size_t row_bytes;
} matmul_ctx;

static void matmul_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    matmul_ctx *ctx = (matmul_ctx *)ctx_;
    int64_t rows = ctx->w->rows;
    int64_t cols = ctx->w->cols;
    const unsigned char *base = (const unsigned char *)ctx->w->data;
    float (*dot_row)(const void *, const float *, int64_t) = ctx->k->dot_row[ctx->w->type];
    for (int64_t idx = begin; idx < end; idx++) {
        int64_t t = idx / rows;
        int64_t r = idx % rows;
        const void *row = base + (size_t)r * ctx->row_bytes;
        ctx->y[t * rows + r] = dot_row(row, ctx->x + t * cols, cols);
    }
}

void tr_matmul(tr_pool *pool, const tr_mat *w, const float *x, int64_t n_tokens, float *y) {
    matmul_ctx ctx;
    ctx.w = w;
    ctx.x = x;
    ctx.y = y;
    ctx.k = tr_kernels_get();
    ctx.row_bytes = tr_row_bytes(w->type, w->cols);

    int64_t n = n_tokens * w->rows;
    /* Keep chunks worth threading: roughly a few thousand scalar multiplies
     * worth of rows per chunk, never less than one row. */
    int64_t min_chunk = w->cols > 0 ? (4096 / w->cols) + 1 : 1;
    tr_parallel_for(pool, n, min_chunk, matmul_body, &ctx);
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
    for (int64_t i = 0; i < n; i++) x[i] = expf(x[i] - m);
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
        float silu = v / (1.0f + expf(-v));
        x[i] = silu * y[i];
    }
}

void tr_swiglu(tr_pool *pool, float *x, const float *y, int64_t n) {
    swiglu_ctx ctx;
    ctx.x = x;
    ctx.y = y;
    /* expf is ~20 ns per element: a few hundred per chunk outweigh waking a thread */
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
/* hot: end */
