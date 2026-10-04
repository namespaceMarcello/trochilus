/* kernels.c — scalar CPU kernels. Defines the numbers (kernels.h): every SIMD
 * or assembly variant added later must reproduce this arithmetic bit for bit.
 * No FMA (built with -ffp-contract=off), no state except the kernel table. */
#include "kernels.h"
#include "kernels_internal.h"
#include "../base/platform.h"

#include <math.h>
#include <stdatomic.h>
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

/* the definition of every tier's expf_f32 */
static void k_expf_f32(const float *x, float *y, int64_t n) {
    for (int64_t k = 0; k < n; k++) y[k] = tr_expf(x[k]);
}

/* ---- 4 cache positions per call: the definition is 4 calls ------------------ */

static void k_dot_f32_x4(const float *a, const float *b, int64_t stride, int64_t n, float *out) {
    for (int j = 0; j < TR_ATTN_X; j++) out[j] = k_dot_f32(a, b + j * stride, n);
}

static void k_axpy_f32_x4(float *y, const float *x, int64_t stride, const float *a, int64_t n) {
    for (int j = 0; j < TR_ATTN_X; j++) k_axpy_f32(y, x + j * stride, a[j], n);
}

/* ---- 4 queries against 4 cache positions: the definition is 16 dots, 4 x4 axpys ------ */

static void k_dot_f32_4x4(const float *a, int64_t a_stride, const float *b, int64_t stride, int64_t n, float scale,
                          float *out, int64_t out_stride) {
    for (int i = 0; i < TR_ATTN_X; i++)
        for (int j = 0; j < TR_ATTN_X; j++)
            out[i * out_stride + j] = k_dot_f32(a + i * a_stride, b + j * stride, n) * scale;
}

static void k_axpy_f32_4x4(float *y, int64_t y_stride, const float *x, int64_t stride, const float *a, int64_t a_stride,
                           int64_t n) {
    for (int i = 0; i < TR_ATTN_X; i++) k_axpy_f32_x4(y + i * y_stride, x, stride, a + i * a_stride, n);
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

/* ---- BF16: an F32 matrix whose values all fit (tr_f32_to_bf16_exact), half the bytes, the same products ---- */

static void k_dequant_bf16(const void *row, float *out, int64_t n) {
    const uint16_t *r = (const uint16_t *)row;
    for (int64_t i = 0; i < n; i++) out[i] = tr_bf16_to_float(r[i]);
}

static float k_dot_row_bf16(const void *row, const float *x, int64_t n) {
    const uint16_t *r = (const uint16_t *)row;
    float lane[TR_LANES] = {0};
    for (int64_t k = 0; k < n; k++) {
        float w = tr_bf16_to_float(r[k]);
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

/* The same against t input rows, t 2 or 3 (kernels.h dot_row_xt) */
static void k_dot_row_xt_q8_0(const void *row, const float *x, int64_t stride, int64_t n, int t, float *out) {
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
            for (int u = 0; u < t; u++) lane[u][k % TR_LANES] += w * x[u * stride + k];
        }
    }
    for (int u = 0; u < t; u++) out[u] = lane_combine(lane[u]);
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

/* ---- Q4_K's integer definition (kernels.h q4x_*, kernels_internal.h for the layouts) ---- */

/* the input row x prepared: per super-block its shift, X, the digits and the sub-blocks' sums; then each
 * window's token term per digit, sum over its 32 pairs (a, a+32) of C (V_a + V_a+32) + V_a V_a+32 */
static void k_q4x_prep(const float *x, int64_t n, void *xq) {
    const tr_q4x_view v = tr_q4x_view_of(xq, n);
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        const float *xs = x + s * TR_Q4_K_BLOCK_ELEMS;
        float m = 0.0f;
        int bad = 0;
        for (int k = 0; k < TR_Q4_K_BLOCK_ELEMS; k++) {
            const float a = fabsf(xs[k]);
            if (!(a <= 3.40282347e38f)) bad = 1; /* a NaN or an infinity */
            else if (a > m) m = a;
        }
        const int sh = bad ? 0 : tr_q4x_shift(m);
        v.scale[s] = bad ? (double)NAN : tr_pow2(-sh);
        /* x 2^sh is exact in double, and adding and taking away 1.5 2^52 rounds it to the nearest integer,
         * ties to even (|x 2^sh| <= TR_Q4X_XMAX < 2^51) */
        const double f = bad ? 0.0 : tr_pow2(sh), magic = 6755399441055744.0;
        for (int j = 0; j < 8; j++) {
            int64_t b = 0;
            for (int i = 0; i < 32; i++) {
                const int64_t k = s * TR_Q4_K_BLOCK_ELEMS + 32 * j + i;
                const int32_t X = bad ? 0 : (int32_t)(((double)x[k] * f + magic) - magic);
                tr_q4x_digits(X, &v.v0[k], &v.v1[k]);
                b += X;
            }
            v.bf[8 * s + j] = (double)b;
        }
    }
    for (int64_t w = 0; w < n / 64; w++)
        for (int d = 0; d < 2; d++) {
            const int16_t *vd = d ? v.v1 : v.v0;
            int64_t e = 0;
            for (int i = 0; i < 32; i++) {
                const int64_t a = vd[64 * w + i], c = vd[64 * w + 32 + i];
                e += TR_Q4X_C * (a + c) + a * c;
            }
            v.tok[2 * w + d] = (int32_t)(uint32_t)(uint64_t)e;
        }
}

/* v_s 2^-sh_s of the block at blk against super-block s of a prepared row: T and M exact, then the f64
 * steps (the scale is a power of two, or NaN) */
static double q4x_block(const unsigned char *blk, const tr_q4x_view *v, int64_t s) {
    int sc[8], m[8];
    tr_q4_k_sc_m(blk, sc, m);
    uint16_t hd, hm;
    memcpy(&hd, blk, 2);
    memcpy(&hm, blk + 2, 2);
    const double d = (double)half_to_float(hd), dmin = (double)half_to_float(hm);
    const unsigned char *qs = blk + TR_Q4_K_QS_OFFSET;
    int64_t T = 0, M = 0;
    for (int j = 0; j < 8; j++) {
        int64_t S0 = 0, S1 = 0;
        for (int i = 0; i < 32; i++) {
            const int64_t k = s * TR_Q4_K_BLOCK_ELEMS + 32 * j + i, q = tr_q4_k_quant(qs, 32 * j + i);
            S0 += q * v->v0[k];
            S1 += q * v->v1[k];
        }
        T += (int64_t)sc[j] * (S0 + (int64_t)TR_Q4X_BASE * S1);
        M += (int64_t)m[j] * (int64_t)v->bf[8 * s + j];
    }
    const double val = d * (double)T - dmin * (double)M;
    return val * v->scale[s];
}

/* The definition from the floats: each super-block of x prepared on the stack and summed in order. */
static float k_dot_row_q4_k(const void *row, const float *x, int64_t n) {
    const unsigned char *p = (const unsigned char *)row;
    double buf[1152 / sizeof(double)]; /* tr_q4x_bytes(256) */
    const tr_q4x_view v = tr_q4x_view_of(buf, TR_Q4_K_BLOCK_ELEMS);
    double y = 0.0;
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        k_q4x_prep(x + s * TR_Q4_K_BLOCK_ELEMS, TR_Q4_K_BLOCK_ELEMS, buf);
        y = y + q4x_block(p + (size_t)s * TR_Q4_K_BLOCK_BYTES, &v, 0);
    }
    return (float)y;
}

static void k_q4x_dot2(const void *row0, const void *row1, const void *xq, int64_t n, float *out) {
    const unsigned char *p0 = (const unsigned char *)row0, *p1 = (const unsigned char *)row1;
    const tr_q4x_view v = tr_q4x_view_of(xq, n);
    double y0 = 0.0, y1 = 0.0;
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        y0 = y0 + q4x_block(p0 + (size_t)s * TR_Q4_K_BLOCK_BYTES, &v, s);
        y1 = y1 + q4x_block(p1 + (size_t)s * TR_Q4_K_BLOCK_BYTES, &v, s);
    }
    out[0] = (float)y0;
    out[1] = (float)y1;
}

/* The pairs against T prepared rows, one input row after the other: the definition of the tiers' kernels */
static void k_q4x_dot_xt(const void *row0, const void *row1, const void *const *xq, int64_t n, int T, float *y,
                         int64_t y_stride) {
    for (int t = 0; t < T; t++) {
        float o[2];
        k_q4x_dot2(row0, row1, xq[t], n, o);
        y[(int64_t)t * y_stride] = o[0];
        y[(int64_t)t * y_stride + 1] = o[1];
    }
}

/* the lane of row r in the panel's doubles: the even rows first */
static int q4x_eo(int r) { return (r & 1) ? 8 + r / 2 : r / 2; }

static void k_q4x_panel(const void *rows, size_t row_bytes, int64_t n, void *panel, const void *next) {
    (void)next;
    const unsigned char *base = (const unsigned char *)rows;
    for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
        unsigned char *sb = (unsigned char *)panel + (size_t)s * TR_Q4X_SB_BYTES;
        double cst[TR_Q4X_CONST_BYTES / sizeof(double)];
        int sc[TR_PM_ROWS][8], m[TR_PM_ROWS][8];
        for (int r = 0; r < TR_PM_ROWS; r++) {
            const unsigned char *blk = base + (size_t)r * row_bytes + (size_t)s * TR_Q4_K_BLOCK_BYTES;
            tr_q4_k_sc_m(blk, sc[r], m[r]);
            uint16_t hd, hm;
            memcpy(&hd, blk, 2);
            memcpy(&hm, blk + 2, 2);
            cst[q4x_eo(r)] = (double)half_to_float(hd);
            cst[16 + q4x_eo(r)] = (double)half_to_float(hm);
            for (int j = 0; j < 8; j++) cst[32 + 16 * j + q4x_eo(r)] = (double)m[r][j];
        }
        memcpy(sb, cst, sizeof cst);
        for (int c = 0; c < 4; c++) {
            unsigned char *win = sb + TR_Q4X_CONST_BYTES + (size_t)c * TR_Q4X_WIN_BYTES;
            for (int r = 0; r < TR_PM_ROWS; r++) {
                const unsigned char *q = base + (size_t)r * row_bytes + (size_t)s * TR_Q4_K_BLOCK_BYTES +
                                         TR_Q4_K_QS_OFFSET + 32 * c;
                int64_t e = 0;
                for (int u = 0; u < 16; u++)
                    for (int h = 0; h < 2; h++) {
                        const int16_t wl = (int16_t)(sc[r][2 * c] * (q[2 * u + h] & 15) + TR_Q4X_C);
                        const int16_t wh = (int16_t)(sc[r][2 * c + 1] * (q[2 * u + h] >> 4) + TR_Q4X_C);
                        memcpy(win + 64 + 128 * u + 4 * r + 2 * h, &wl, 2);
                        memcpy(win + 128 + 128 * u + 4 * r + 2 * h, &wh, 2);
                        e += (int64_t)wl * wh;
                    }
                const uint32_t neg = (uint32_t)(uint64_t)(-e) + ((r & 1) ? 0u : 0x80000000u);
                memcpy(win + 4 * r, &neg, 4);
            }
        }
    }
}

/* The definition from the panel's own bytes, a row and an input row at a time: w = word - TR_Q4X_C, T and
 * M exact, the f64 steps (the panel's row terms are the SIMD tiles' business, checked with the panel). */
static void k_q4x_tile(const void *panel, const void *const *xq, int64_t n, int T, float *y, int64_t y_stride) {
    for (int t = 0; t < T; t++) {
        const tr_q4x_view v = tr_q4x_view_of(xq[t], n);
        for (int r = 0; r < TR_PM_ROWS; r++) {
            double acc = 0.0;
            for (int64_t s = 0; s < n / TR_Q4_K_BLOCK_ELEMS; s++) {
                const unsigned char *sb = (const unsigned char *)panel + (size_t)s * TR_Q4X_SB_BYTES;
                double cst[TR_Q4X_CONST_BYTES / sizeof(double)];
                memcpy(cst, sb, sizeof cst);
                int64_t Ts = 0, Ms = 0;
                for (int c = 0; c < 4; c++) {
                    const unsigned char *win = sb + TR_Q4X_CONST_BYTES + (size_t)c * TR_Q4X_WIN_BYTES;
                    for (int u = 0; u < 16; u++)
                        for (int h = 0; h < 2; h++) {
                            int16_t wl, wh;
                            memcpy(&wl, win + 64 + 128 * u + 4 * r + 2 * h, 2);
                            memcpy(&wh, win + 128 + 128 * u + 4 * r + 2 * h, 2);
                            const int64_t a = s * TR_Q4_K_BLOCK_ELEMS + 64 * c + 2 * u + h, b = a + 32;
                            const int64_t Xa = v.v0[a] + (int64_t)TR_Q4X_BASE * v.v1[a];
                            const int64_t Xb = v.v0[b] + (int64_t)TR_Q4X_BASE * v.v1[b];
                            Ts += (int64_t)(wl - TR_Q4X_C) * Xa + (int64_t)(wh - TR_Q4X_C) * Xb;
                        }
                }
                for (int j = 0; j < 8; j++) Ms += (int64_t)cst[32 + 16 * j + q4x_eo(r)] * (int64_t)v.bf[8 * s + j];
                const double val = cst[q4x_eo(r)] * (double)Ts - cst[16 + q4x_eo(r)] * (double)Ms;
                acc = acc + val * v.scale[s];
            }
            y[(int64_t)t * y_stride + r] = (float)acc;
        }
    }
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

static void k_dot_row_x4_bf16(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    const uint16_t *r = (const uint16_t *)row;
    float lane[TR_DOT_TOKENS][TR_LANES] = {{0}};
    for (int64_t k = 0; k < n; k++) {
        float w = tr_bf16_to_float(r[k]);
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

/* BF16 is not a file's type here: the table's BF16 entries serve the F32 matrices narrowed at load
 * (tr_f32_to_bf16_exact), and a GGUF tensor stored as BF16 is still refused. */
int tr_kernels_support(tr_type type) {
    return type == TR_TYPE_F32 || type == TR_TYPE_F16 || type == TR_TYPE_Q8_0 || type == TR_TYPE_Q4_K ||
           type == TR_TYPE_Q6_K;
}

/* Bytes, not typed pointers: the same memory is read as floats and written as halves. Value i's half goes to bytes
 * [2i, 2i + 2), inside value i / 2's bytes, which the loop has already read. */
int tr_f32_to_bf16_exact(void *data, int64_t n) {
    unsigned char *p = (unsigned char *)data;
    for (int64_t i = 0; i < n; i++) {
        uint32_t v;
        memcpy(&v, p + 4 * i, sizeof v);
        if ((v & 0xFFFFu) != 0) return 0;
    }
    for (int64_t i = 0; i < n; i++) {
        uint32_t v;
        memcpy(&v, p + 4 * i, sizeof v);
        uint16_t b = (uint16_t)(v >> 16);
        memcpy(p + 2 * i, &b, sizeof b);
    }
    return 1;
}

size_t tr_q4x_bytes(int64_t n) {
    const size_t b = 4 * (size_t)n + (size_t)n / 8 + (size_t)n / 4 + (size_t)n / 32;
    return (b + 63) / 64 * 64;
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
    k->dot_f32_4x4 = k_dot_f32_4x4;
    k->axpy_f32_4x4 = k_axpy_f32_4x4;
    k->dot_row[TR_TYPE_F32] = k_dot_row_f32;
    k->dot_row[TR_TYPE_F16] = k_dot_row_f16;
    k->dot_row[TR_TYPE_BF16] = k_dot_row_bf16;
    k->dot_row[TR_TYPE_Q8_0] = k_dot_row_q8_0;
    k->dot_row[TR_TYPE_Q4_K] = k_dot_row_q4_k;
    k->dot_row[TR_TYPE_Q6_K] = k_dot_row_q6_k;
    k->dot_row_x4[TR_TYPE_F32] = k_dot_row_x4_f32;
    k->dot_row_x4[TR_TYPE_F16] = k_dot_row_x4_f16;
    k->dot_row_x4[TR_TYPE_BF16] = k_dot_row_x4_bf16;
    k->dot_row_x4[TR_TYPE_Q8_0] = k_dot_row_x4_q8_0;
    k->dot_row_xt[TR_TYPE_Q8_0] = k_dot_row_xt_q8_0;
    k->dot_row_x4[TR_TYPE_Q6_K] = k_dot_row_x4_q6_k;
    k->q4x_prep = k_q4x_prep;
    k->q4x_dot2 = k_q4x_dot2;
    k->q4x_dot_xt = k_q4x_dot_xt;
    k->q4x_panel = k_q4x_panel;
    k->q4x_tile = k_q4x_tile;
    k->q4x_tile_max = TR_Q4X_TILE_MAX;
    k->dequant_row[TR_TYPE_F32] = k_dequant_f32;
    k->dequant_row[TR_TYPE_F16] = k_dequant_f16;
    k->dequant_row[TR_TYPE_BF16] = k_dequant_bf16;
    k->dequant_row[TR_TYPE_Q8_0] = k_dequant_q8_0;
    k->dequant_row[TR_TYPE_Q4_K] = k_dequant_q4_k;
    k->dequant_row[TR_TYPE_Q6_K] = k_dequant_q6_k;
    k->expf_f32 = k_expf_f32;
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

#if defined(TR_POOL_TRACE)
/* the weights a grouped call reads, the groups with input rows: the pool trace's note (threads.h) */
static uint64_t matmul_used_bytes(const int64_t *offsets, int64_t n_groups, int64_t rows, size_t row_bytes) {
    int64_t used = 0;
    for (int64_t g = 0; g < n_groups; g++) used += offsets[g + 1] > offsets[g];
    return (uint64_t)used * (uint64_t)rows * row_bytes;
}
#endif

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
    TR_TRACE_NOTE(matmul_used_bytes(offsets, n_groups, ctx.rows, ctx.row_bytes), ctx.rows, ctx.cols, offsets[n_groups]);
    /* One input row per group (a decode token, its experts' gathered rows): only weight rows to
     * stream, balanced at the tail. More rows share tiles, which want long chunks. */
    if (tr_matmul_one_row_per_group(offsets, n_groups)) tr_parallel_for_balanced(pool, n, min_chunk, matmul_body, &ctx);
    else tr_parallel_for(pool, n, min_chunk, matmul_body, &ctx);
}

/* ---- the prompt's matmul in phase-major order (kernels.h pm_*, tr_matmul_grouped_s) ----------------
 * An item is (group g, 16 weight rows rg, a chunk of g's input rows): the worker builds the rows' panel
 * (kept while its next item has the same rows) and runs every tile of the chunk. Chunks are at most
 * PM_CHUNK input rows and tiles 4..TR_PM_TILE_MAX, both cut evenly (the thinker's tiling on OLMoE's
 * real routing: no padding, 1.7% of the work in tiles under 12). A group of fewer than 4 input rows has
 * one chunk per 16 rows and keeps dot_row2 and dot_row. Each output is one pm_tile or one dot_row of its
 * own, whatever the thread: the same y at every thread count. */
#define PM_CHUNK 256
#define PM_MIN_ROWS 4
/* the narrowest tile_max a road plans with (AVX2's Q4_K tiles of 3): the plan's room for tiles */
#define PM_ROOM_TILE 3

/* the plan's tables inside s->plan */
typedef struct {
    int64_t *item0;       /* [n_groups + 1] first item of each group */
    int64_t *nchunk;      /* [n_groups] chunks of each group */
    int64_t *chunk0;      /* [n_groups + 1] first chunk of each group */
    int64_t *tile0;       /* [n_chunks + 1] first tile of each chunk */
    int64_t *tile_p;      /* [n_tiles] first input row of each tile */
    int64_t *tile_t;      /* [n_tiles] its width */
    int64_t *last;        /* [n_workers][2] the group and rows of the worker's panel, -1: none */
} pm_plan;

static int64_t pm_max_chunks(const tr_pm_scratch *s) { return s->max_groups + s->max_tokens / PM_CHUNK + 1; }
static int64_t pm_max_tiles(const tr_pm_scratch *s) { return s->max_tokens / PM_ROOM_TILE + pm_max_chunks(s) + 1; }

static pm_plan pm_plan_of(const tr_pm_scratch *s) {
    pm_plan p;
    int64_t *q = s->plan;
    p.item0 = q;
    q += s->max_groups + 1;
    p.nchunk = q;
    q += s->max_groups;
    p.chunk0 = q;
    q += s->max_groups + 1;
    p.tile0 = q;
    q += pm_max_chunks(s) + 1;
    p.tile_p = q;
    q += pm_max_tiles(s);
    p.tile_t = q;
    q += pm_max_tiles(s);
    p.last = q;
    return p;
}

typedef struct {
    const tr_mat *w;
    const int64_t *offsets;
    int64_t n_groups, rows, cols;
    const float *x;
    float *y;
    const tr_kernels *k;
    const tr_pm_scratch *s;
    pm_plan plan;
    size_t row_bytes;
} pm_ctx;

/* tile i's interleaved rows: where its input rows sit in x, plus the pads of the tiles before it */
static float *pm_xil_of(const pm_ctx *c, int64_t i) {
    return c->s->xil + c->plan.tile_p[i] * c->cols + i * 16 * TR_PM_PAD;
}

static void pm_interleave_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const pm_ctx *c = (const pm_ctx *)ctx_;
    for (int64_t i = begin; i < end; i++) {
        int64_t p0 = c->plan.tile_p[i];
        c->k->pm_interleave(c->x + p0 * c->cols, c->cols, c->cols, (int)c->plan.tile_t[i], pm_xil_of(c, i));
    }
}

/* input rows [p_begin, p_end) of group g against its 16 weight rows from r0: two or three of them with each
 * weight row decoded once for all (dot_row_xt), else one dot_row2 (or two dot_row) per pair of rows and input
 * row: the same bits as a tile */
static void pm_rows_by_dot(const pm_ctx *c, int64_t g, int64_t r0, int64_t p_begin, int64_t p_end) {
    const unsigned char *rows = (const unsigned char *)c->w[g].data + (size_t)r0 * c->row_bytes;
    const int64_t t = p_end - p_begin;
    if ((t == 2 || t == 3) && c->k->dot_row_xt[c->w[g].type] != NULL) {
        const float *xp = c->x + p_begin * c->cols;
        float *yp = c->y + p_begin * c->rows + r0;
        float out[3];
        for (int r = 0; r < TR_PM_ROWS; r++) {
            c->k->dot_row_xt[c->w[g].type](rows + (size_t)r * c->row_bytes, xp, c->cols, c->cols, (int)t, out);
            for (int64_t j = 0; j < t; j++) yp[j * c->rows + r] = out[j];
        }
        return;
    }
    for (int64_t p = p_begin; p < p_end; p++) {
        const float *xp = c->x + p * c->cols;
        float *yp = c->y + p * c->rows + r0;
        for (int r = 0; r < TR_PM_ROWS; r += 2) {
            const unsigned char *a = rows + (size_t)r * c->row_bytes, *b = a + c->row_bytes;
            if (c->k->dot_row2[c->w[g].type] != NULL) {
                c->k->dot_row2[c->w[g].type](a, b, xp, c->cols, yp + r);
            } else {
                yp[r] = c->k->dot_row[c->w[g].type](a, xp, c->cols);
                yp[r + 1] = c->k->dot_row[c->w[g].type](b, xp, c->cols);
            }
        }
    }
}

/* The plan of both roads, cut evenly (piece i of L in n: [i*L/n, (i+1)*L/n)): a group of min_rows input rows
 * or more in chunks of at most PM_CHUNK, each in tiles of at most tile_max; a smaller group in one chunk with no
 * tiles. Items are (group, TR_PM_ROWS weight rows, chunk); every worker's panel is forgotten. The tiles fit the
 * plan's room (pm_max_tiles: max_tokens / PM_ROOM_TILE + the chunks) while tile_max >= PM_ROOM_TILE: a chunk of
 * len holds ceil(len / tile_max) <= len / PM_ROOM_TILE + 1 tiles. Returns the items; *n_tiles the tiles. */
static int64_t pm_plan_build(pm_plan *pl, const int64_t *offsets, int64_t n_groups, int64_t rows, int64_t min_rows,
                             int64_t tile_max, int n_workers, int64_t *n_tiles) {
    int64_t n_items = 0, n_chunks = 0, nt_all = 0;
    for (int64_t g = 0; g < n_groups; g++) {
        int64_t cg = offsets[g + 1] - offsets[g];
        int64_t nch = cg >= min_rows ? (cg + PM_CHUNK - 1) / PM_CHUNK : 1;
        pl->item0[g] = n_items;
        pl->nchunk[g] = nch;
        pl->chunk0[g] = n_chunks;
        if (cg > 0) n_items += (rows / TR_PM_ROWS) * nch;
        if (cg >= min_rows) {
            for (int64_t ch = 0; ch < nch; ch++) {
                int64_t c0 = offsets[g] + ch * cg / nch, c1 = offsets[g] + (ch + 1) * cg / nch, len = c1 - c0;
                int64_t nt = (len + tile_max - 1) / tile_max;
                pl->tile0[n_chunks++] = nt_all;
                for (int64_t t = 0; t < nt; t++) {
                    pl->tile_p[nt_all] = c0 + t * len / nt;
                    pl->tile_t[nt_all] = c0 + (t + 1) * len / nt - pl->tile_p[nt_all];
                    nt_all++;
                }
            }
        } else {
            pl->tile0[n_chunks++] = nt_all; /* its one chunk has no tiles */
        }
    }
    pl->item0[n_groups] = n_items;
    pl->chunk0[n_groups] = n_chunks;
    pl->tile0[n_chunks] = nt_all;
    for (int i = 0; i < 2 * n_workers; i++) pl->last[i] = -1;
    *n_tiles = nt_all;
    return n_items;
}

/* 1 when the worker's panel (last: its group and rows) is not group g's rows rg; then it is marked as theirs,
 * and the caller builds it */
static int pm_panel_stale(int64_t *last, int64_t g, int64_t rg) {
    if (last[0] == g && last[1] == rg) return 0;
    last[0] = g;
    last[1] = rg;
    return 1;
}

static void pm_item_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const pm_ctx *c = (const pm_ctx *)ctx_;
    const pm_plan *pl = &c->plan;
    /* a worker the scratch has no panel for (a call with no pool from inside a larger pool's body runs
     * on that body's worker): its items go by dot_row2 */
    const int own = worker < c->s->n_workers;
    float *panel = own ? c->s->work + (size_t)worker * (size_t)(TR_PM_PANEL_FLOATS(c->s->max_cols) + TR_PM_PART)
                       : NULL;
    float *part = own ? panel + (size_t)TR_PM_PANEL_FLOATS(c->s->max_cols) : NULL;
    int64_t *last = own ? pl->last + 2 * worker : NULL;
    int64_t g = 0, lo = 0, hi = c->n_groups - 1;
    while (lo < hi) { /* the last group whose first item is <= begin */
        int64_t mid = lo + (hi - lo + 1) / 2;
        if (pl->item0[mid] <= begin) lo = mid;
        else hi = mid - 1;
    }
    g = lo;
    for (int64_t it = begin; it < end; it++) {
        while (pl->item0[g + 1] <= it) g++;
        int64_t local = it - pl->item0[g], nch = pl->nchunk[g];
        int64_t rg = local / nch, ch = local % nch;
        int64_t r0 = rg * TR_PM_ROWS, p_begin = c->offsets[g], n_in = c->offsets[g + 1] - p_begin;
        const unsigned char *rows = (const unsigned char *)c->w[g].data + (size_t)r0 * c->row_bytes;
        if (n_in < PM_MIN_ROWS || !own) { /* the chunk's input rows, cut as the plan cuts them */
            pm_rows_by_dot(c, g, r0, p_begin + ch * n_in / nch, p_begin + (ch + 1) * n_in / nch);
            continue;
        }
        if (pm_panel_stale(last, g, rg)) c->k->pm_panel[c->w[g].type](rows, c->row_bytes, c->cols, panel);
        int64_t ci = pl->chunk0[g] + ch;
        for (int64_t i = pl->tile0[ci]; i < pl->tile0[ci + 1]; i++) {
            int64_t p0 = pl->tile_p[i];
            c->k->pm_tile(panel, pm_xil_of(c, i), c->cols, (int)pl->tile_t[i], part, c->y + p0 * c->rows + r0,
                          c->rows);
        }
    }
}

/* 1 when the call took the phase-major road. A group of PM_MIN_ROWS input rows or more runs tiles. A short
 * pass (no such group, one of 2 or 3: a verify pass of 2-3 rows) is taken too, on a tier with no tiles as
 * well: every group by rows (pm_rows_by_dot) in items cut by weight rows and balanced, a group's input rows
 * against each weight row while it is in cache (docs/MEASUREMENTS.md §The short verify pass at its bytes).
 * One input row a group (a decode token) stays on tr_matmul_grouped's balanced road. */
static int matmul_phase_major(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups,
                              const float *x, float *y, const tr_pm_scratch *s) {
    const tr_kernels *k = tr_kernels_get();
    const tr_type type = w[0].type;
    const int64_t rows = w[0].rows, cols = w[0].cols, n_in = offsets[n_groups];
    if (s == NULL || s->plan == NULL) return 0;
    /* a row must be whole blocks of its type too (a Q4_K row of 272 columns has 0 bytes) */
    if (rows % TR_PM_ROWS != 0 || cols % 16 != 0 || tr_row_bytes(type, cols) == 0 || cols > s->max_cols ||
        n_groups > s->max_groups || n_in > s->max_tokens || n_in * cols > s->max_x)
        return 0;
    int64_t most = 0;
    for (int64_t g = 0; g < n_groups; g++)
        if (offsets[g + 1] - offsets[g] > most) most = offsets[g + 1] - offsets[g];
    if (most < 2) return 0;
    if (most >= PM_MIN_ROWS && (k->pm_tile == NULL || k->pm_interleave == NULL || k->pm_panel[type] == NULL)) return 0;

    pm_ctx c;
    c.w = w;
    c.offsets = offsets;
    c.n_groups = n_groups;
    c.rows = rows;
    c.cols = cols;
    c.x = x;
    c.y = y;
    c.k = k;
    c.s = s;
    c.plan = pm_plan_of(s);
    c.row_bytes = tr_row_bytes(type, cols);
    int64_t n_tiles;
    const int64_t n_items = pm_plan_build(&c.plan, offsets, n_groups, rows, PM_MIN_ROWS, TR_PM_TILE_MAX, s->n_workers, &n_tiles);

    if (n_tiles > 0) tr_parallel_for(pool, n_tiles, 1, pm_interleave_body, &c);
    TR_TRACE_NOTE(matmul_used_bytes(offsets, n_groups, rows, c.row_bytes), rows, cols, n_in);
    tr_parallel_for_balanced(pool, n_items, 1, pm_item_body, &c);
    return 1;
}

/* ---- Q4_K's integer road (kernels.h q4x_*) --------------------------------------------------------------
 * Every input row prepared once (its digits, sums and shifts); then items as the phase-major road cuts them,
 * (group, 16 weight rows, a chunk of the group's input rows): a group of Q4X_MIN_TILE_ROWS input rows or more
 * builds the rows' W16 panel (kept while the worker's next item has the same rows) and runs tiles of at most
 * TR_Q4X_TILE_MAX rows cut evenly; a smaller group (a decode token, the 2-3 rows of a verify pass) goes two
 * weight rows at a time: runs of 2-3 input rows through q4x_dot_xt, each weight decoded once for the run (2
 * streams a thread where the panel reads 16: from RAM at 8 threads 1.16x the panel at 3 rows, docs/MEASUREMENTS.md
 * §A Q4_K weight row decoded once for 2-3 tokens), a lone input row through q4x_dot2. A tier without
 * q4x_dot_xt pairs a group of 2 a token after the other (two q4x_dot2 streams cost less than a panel built for
 * two tokens, three do not: §The Q4_K short passes). A short group's panel is built from the rows the previous
 * item brought ahead (q4x_panel's next). Every output is the definition's, whatever the thread or the tile. */
#define Q4X_MIN_TILE_ROWS 4
#define Q4X_MIN_TILE_ROWS_DOT2 3

typedef struct {
    const tr_mat *w;
    const int64_t *offsets;
    int64_t n_groups, rows, cols;
    float *y;
    const tr_kernels *k;
    const tr_pm_scratch *s;
    pm_plan plan;
    size_t row_bytes, xq_row;   /* a weight row's bytes, a prepared input row's */
    int64_t min_rows;           /* a group of fewer input rows goes by rows */
    const unsigned char *xq;    /* the prepared rows: s->xq, or the caller's (tr_matmul_q4x_prepared) */
    const int64_t *map;         /* input row p reads prepared row map[p]; NULL: row p */
    int n_w;                    /* 1, or the weights of tr_matmul_q4x_prepared_n: each item on ws[m] into ys[m] */
    int64_t n_items;            /* one weight's items (the plan's) */
    const tr_mat *ws[TR_Q4X_MATS];
    float *ys[TR_Q4X_MATS];
} q4x_ctx;

/* the prepared row input row p reads */
static const void *q4x_row(const q4x_ctx *c, int64_t p) {
    return c->xq + (size_t)(c->map != NULL ? c->map[p] : p) * c->xq_row;
}

typedef struct {
    const tr_kernels *k;
    const float *x;
    int64_t cols;
    unsigned char *xq;
    size_t xq_row;
} q4x_prep_ctx;

static void q4x_prep_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const q4x_prep_ctx *c = (const q4x_prep_ctx *)ctx_;
    for (int64_t p = begin; p < end; p++) c->k->q4x_prep(c->x + p * c->cols, c->cols, c->xq + (size_t)p * c->xq_row);
}

/* input rows [p_begin, p_end) of group g against its TR_PM_ROWS weight rows from r0, two a call: runs of 3
 * input rows, of 2 when 2 or 4 are left (q4x_dot_xt), a lone one by q4x_dot2 */
static void q4x_rows(const q4x_ctx *c, int64_t g, int64_t r0, int64_t p_begin, int64_t p_end) {
    const unsigned char *rows = (const unsigned char *)c->w[g].data + (size_t)r0 * c->row_bytes;
    int64_t p = p_begin;
    if (c->k->q4x_dot_xt != NULL)
        while (p_end - p >= 2) {
            const int T = p_end - p == 2 || p_end - p == 4 ? 2 : 3;
            const void *xq[TR_Q4X_XT_MAX];
            for (int t = 0; t < T; t++) xq[t] = q4x_row(c, p + t);
            float *yp = c->y + p * c->rows + r0;
            for (int r = 0; r < TR_PM_ROWS; r += 2)
                c->k->q4x_dot_xt(rows + (size_t)r * c->row_bytes, rows + (size_t)(r + 1) * c->row_bytes, xq, c->cols, T,
                                 yp + r, c->rows);
            p += T;
        }
    for (; p < p_end; p++) {
        const void *xq = q4x_row(c, p);
        float *yp = c->y + p * c->rows + r0;
        for (int r = 0; r < TR_PM_ROWS; r += 2)
            c->k->q4x_dot2(rows + (size_t)r * c->row_bytes, rows + (size_t)(r + 1) * c->row_bytes, xq, c->cols, yp + r);
    }
}

static void q4x_items(const q4x_ctx *c, int64_t begin, int64_t end, int worker) {
    const pm_plan *pl = &c->plan;
    /* a worker the scratch has no panel for (a call with no pool from inside a larger pool's body): by rows */
    const int own = worker < c->s->n_workers && c->k->q4x_panel != NULL && c->k->q4x_tile != NULL;
    unsigned char *panel = own ? (unsigned char *)(c->s->work + (size_t)worker * (size_t)(TR_PM_PANEL_FLOATS(c->s->max_cols) +
                                                                                           TR_PM_PART))
                               : NULL;
    int64_t *last = own ? pl->last + 2 * worker : NULL;
    int64_t g = 0, lo = 0, hi = c->n_groups - 1;
    while (lo < hi) { /* the last group whose first item is <= begin */
        int64_t mid = lo + (hi - lo + 1) / 2;
        if (pl->item0[mid] <= begin) lo = mid;
        else hi = mid - 1;
    }
    g = lo;
    for (int64_t it = begin; it < end; it++) {
        while (pl->item0[g + 1] <= it) g++;
        int64_t local = it - pl->item0[g], nch = pl->nchunk[g];
        int64_t rg = local / nch, ch = local % nch;
        int64_t r0 = rg * TR_PM_ROWS, p_begin = c->offsets[g], n_in = c->offsets[g + 1] - p_begin;
        if (n_in < c->min_rows || !own) {
            q4x_rows(c, g, r0, p_begin + ch * n_in / nch, p_begin + (ch + 1) * n_in / nch);
            continue;
        }
        if (pm_panel_stale(last, g, rg)) {
            /* a short group (one tile an item, over soon): the item after this one, when it has other rows,
             * brought ahead while this panel is built */
            const void *next = NULL;
            if (n_in <= TR_Q4X_TILE_MAX && it + 1 < pl->item0[c->n_groups]) {
                int64_t g2 = g;
                while (pl->item0[g2 + 1] <= it + 1) g2++;
                const int64_t rg2 = (it + 1 - pl->item0[g2]) / pl->nchunk[g2];
                if (g2 != g || rg2 != rg)
                    next = (const unsigned char *)c->w[g2].data + (size_t)(rg2 * TR_PM_ROWS) * c->row_bytes;
            }
            c->k->q4x_panel((const unsigned char *)c->w[g].data + (size_t)r0 * c->row_bytes, c->row_bytes, c->cols, panel,
                            next);
        }
        int64_t ci = pl->chunk0[g] + ch;
        for (int64_t i = pl->tile0[ci]; i < pl->tile0[ci + 1]; i++) {
            const int64_t p0 = pl->tile_p[i];
            const int T = (int)pl->tile_t[i];
            const void *xq[TR_Q4X_TILE_MAX];
            for (int t = 0; t < T; t++) xq[t] = q4x_row(c, p0 + t);
            c->k->q4x_tile(panel, xq, c->cols, T, c->y + p0 * c->rows + r0, c->rows);
        }
    }
}

/* tr_matmul_q4x_prepared_n's items: every weight's in one range, w[0]'s first, so the pool balances them as finely as
 * one weight's and a thread's range crosses one or two weights (one region of rows where it had one a weight). Each
 * piece of a weight marks the thread's panel stale first: the weights share their groups and rows. */
static void q4x_item_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const q4x_ctx *c = (const q4x_ctx *)ctx_;
    if (c->n_w <= 1) {
        q4x_items(c, begin, end, worker);
        return;
    }
    q4x_ctx cm = *c;
    for (int64_t it = begin; it < end;) {
        const int64_t m = it / c->n_items, stop = end < (m + 1) * c->n_items ? end : (m + 1) * c->n_items;
        cm.w = c->ws[m];
        cm.y = c->ys[m];
        if (worker < c->s->n_workers) c->plan.last[2 * worker] = c->plan.last[2 * worker + 1] = -1;
        q4x_items(&cm, it - m * c->n_items, stop - m * c->n_items, worker);
        it = stop;
    }
}

/* 1 when Q4_K's integer road can take a call of these weights and groups on s (whatever holds its prepared rows) */
static int q4x_road(const tr_kernels *k, const tr_mat *w, const int64_t *offsets, int64_t n_groups,
                    const tr_pm_scratch *s) {
    const int64_t rows = w[0].rows, cols = w[0].cols;
    return w[0].type == TR_TYPE_Q4_K && s != NULL && s->plan != NULL && k->q4x_prep != NULL && k->q4x_dot2 != NULL &&
           rows % TR_PM_ROWS == 0 && tr_row_bytes(TR_TYPE_Q4_K, cols) != 0 && cols <= s->max_cols &&
           n_groups <= s->max_groups && offsets[n_groups] <= s->max_tokens;
}

/* the call's items over the prepared rows c->xq (read through c->map) */
static void q4x_run(tr_pool *pool, q4x_ctx *c, const tr_mat *w, const int64_t *offsets, int64_t n_groups, float *y,
                    const tr_pm_scratch *s) {
    const tr_kernels *k = c->k;
    c->w = w;
    c->offsets = offsets;
    c->n_groups = n_groups;
    c->rows = w[0].rows;
    c->cols = w[0].cols;
    c->y = y;
    c->s = s;
    c->plan = pm_plan_of(s);
    c->row_bytes = tr_row_bytes(TR_TYPE_Q4_K, c->cols);
    c->xq_row = tr_q4x_bytes(c->cols);
    c->min_rows = k->q4x_dot_xt != NULL ? Q4X_MIN_TILE_ROWS : Q4X_MIN_TILE_ROWS_DOT2;
    /* tiles of at most the tier's q4x_tile_max (3 or 4: within the plan's room, ceil(len / 3) tiles a chunk at
     * most); a group under min_rows is one chunk, so an item holds its 2-3 input rows on one thread */
    int64_t n_tiles;
    const int64_t n_items = pm_plan_build(&c->plan, offsets, n_groups, c->rows, c->min_rows, k->q4x_tile_max,
                                          s->n_workers, &n_tiles);
    (void)n_tiles;
    TR_TRACE_NOTE(matmul_used_bytes(offsets, n_groups, c->rows, c->row_bytes) * (uint64_t)c->n_w, c->rows, c->cols,
                  offsets[n_groups]);
    c->n_items = n_items;
    tr_parallel_for_balanced(pool, n_items * c->n_w, 1, q4x_item_body, c);
}

void tr_q4x_prepare(tr_pool *pool, const float *x, int64_t n, int64_t cols, void *xq) {
    q4x_prep_ctx pc;
    pc.k = tr_kernels_get();
    pc.x = x;
    pc.cols = cols;
    pc.xq = (unsigned char *)xq;
    pc.xq_row = tr_q4x_bytes(cols);
    tr_parallel_for(pool, n, 1, q4x_prep_body, &pc);
}

/* 1 when the call took the integer road: its input rows prepared into s->xq, then the items */
static int matmul_q4x(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const float *x,
                      float *y, const tr_pm_scratch *s) {
    const tr_kernels *k = tr_kernels_get();
    const int64_t n_in = offsets[n_groups];
    if (!q4x_road(k, w, offsets, n_groups, s) || s->xq == NULL ||
        (uint64_t)n_in * tr_q4x_bytes(w[0].cols) > (uint64_t)s->xq_bytes)
        return 0;
    tr_q4x_prepare(pool, x, n_in, w[0].cols, s->xq);
    q4x_ctx c;
    c.k = k;
    c.xq = s->xq;
    c.map = NULL;
    c.n_w = 1;
    q4x_run(pool, &c, w, offsets, n_groups, y, s);
    return 1;
}

int tr_matmul_q4x_prepared(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const void *xq,
                           const int64_t *map, float *y, const tr_pm_scratch *s) {
    if (n_groups <= 0 || offsets[n_groups] <= 0 || w[0].rows <= 0) return 1;
    if (pool != NULL && s != NULL && tr_pool_size(pool) > s->n_workers) return 0;
    const tr_kernels *k = tr_kernels_get();
    if (!q4x_road(k, w, offsets, n_groups, s)) return 0;
    q4x_ctx c;
    c.k = k;
    c.xq = (const unsigned char *)xq;
    c.map = map;
    c.n_w = 1;
    q4x_run(pool, &c, w, offsets, n_groups, y, s);
    return 1;
}

static _Atomic uint64_t g_q4x_fused_calls; /* global-ok: a count of the process's calls, read by tests only */

uint64_t tr_q4x_fused_calls(void) {
    return atomic_load_explicit(&g_q4x_fused_calls, memory_order_relaxed);
}

int tr_matmul_q4x_prepared_n(tr_pool *pool, const tr_mat *const *w, int n_w, const int64_t *offsets, int64_t n_groups,
                             const void *xq, const int64_t *map, float *const *y, const tr_pm_scratch *s) {
    if (n_w < 1 || n_w > TR_Q4X_MATS) return 0;
    if (n_w == 1) return tr_matmul_q4x_prepared(pool, w[0], offsets, n_groups, xq, map, y[0], s);
    if (n_groups <= 0 || offsets[n_groups] <= 0 || w[0][0].rows <= 0) return 1;
    if (pool != NULL && s != NULL && tr_pool_size(pool) > s->n_workers) return 0;
    const tr_kernels *k = tr_kernels_get();
    for (int m = 0; m < n_w; m++)
        if (!q4x_road(k, w[m], offsets, n_groups, s) || w[m][0].rows != w[0][0].rows || w[m][0].cols != w[0][0].cols)
            return 0;
    q4x_ctx c;
    c.k = k;
    c.xq = (const unsigned char *)xq;
    c.map = map;
    c.n_w = n_w;
    for (int m = 0; m < n_w; m++) {
        c.ws[m] = w[m];
        c.ys[m] = y[m];
    }
    atomic_fetch_add_explicit(&g_q4x_fused_calls, 1, memory_order_relaxed);
    q4x_run(pool, &c, w[0], offsets, n_groups, y[0], s);
    return 1;
}

int tr_q4x_road(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const tr_pm_scratch *s) {
    if (pool != NULL && s != NULL && tr_pool_size(pool) > s->n_workers) return 0;
    return q4x_road(tr_kernels_get(), w, offsets, n_groups, s);
}

/* A decode call's indices, as the call splits them over the pool: on Q4_K's road the items of TR_PM_ROWS rows of
 * each non-empty group (q4x_run), else tr_matmul_grouped's (input row, weight row) pairs; one input row a group,
 * so the k-th input row is the k-th non-empty group's and an index is (that group, its row r). Chunk t's first
 * index is where its thread reads first (run_chunk: its region's first block). */
void tr_matmul_hint(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const tr_pm_scratch *s,
                    size_t max_bytes) {
    if (tr_pool_hints(pool) == 0 || max_bytes == 0 || n_groups <= 0 || offsets[n_groups] <= 0 || w[0].rows <= 0 ||
        !tr_matmul_one_row_per_group(offsets, n_groups))
        return;
    if (s != NULL && tr_pool_size(pool) > s->n_workers) s = NULL;
    const int64_t rows = w[0].rows;
    const size_t rb = tr_row_bytes(w[0].type, w[0].cols);
    const int64_t per = q4x_road(tr_kernels_get(), w, offsets, n_groups, s) ? TR_PM_ROWS : 1; /* rows an index */
    const int64_t n = offsets[n_groups] * (rows / per);
    const int64_t min_chunk = per > 1 ? 1 : w[0].cols > 0 ? 4096 / w[0].cols + 1 : 1; /* as the call's */
    int64_t b, e, g = -1, k = -1; /* g: the k-th non-empty group */
    const int chunks = tr_pool_region(pool, n, min_chunk, 0, &b, &e);
    for (int t = 1; t < chunks; t++) {
        tr_pool_region(pool, n, min_chunk, t, &b, &e);
        const int64_t kt = b / (rows / per), r0 = b % (rows / per) * per;
        while (k < kt) {
            g++;
            if (offsets[g + 1] > offsets[g]) k++;
        }
        if (w[g].data == NULL) continue;
        size_t bytes = (size_t)((e - b) * per / 2) * rb, left = (size_t)(rows - r0) * rb;
        if (bytes > left) bytes = left;
        if (bytes > max_bytes) bytes = max_bytes;
        tr_pool_hint(pool, t, (const unsigned char *)w[g].data + (size_t)r0 * rb, bytes);
    }
}

void tr_matmul_grouped_s(tr_pool *pool, const tr_mat *w, const int64_t *offsets, int64_t n_groups, const float *x,
                         float *y, const tr_pm_scratch *s) {
    if (n_groups <= 0 || offsets[n_groups] <= 0 || w[0].rows <= 0) return;
    if (pool != NULL && s != NULL && tr_pool_size(pool) > s->n_workers) s = NULL;
    if (matmul_q4x(pool, w, offsets, n_groups, x, y, s)) return;
    if (matmul_phase_major(pool, w, offsets, n_groups, x, y, s)) return;
    tr_matmul_grouped(pool, w, offsets, n_groups, x, y);
}

void tr_matmul_s(tr_pool *pool, const tr_mat *w, const float *x, int64_t n_tokens, float *y, const tr_pm_scratch *s) {
    const int64_t offsets[2] = {0, n_tokens};
    tr_matmul_grouped_s(pool, w, offsets, 1, x, y, s);
}

/* hot: end */

/* the input buffer: the interleaved floats, or as many bytes as the prepared Q4_K rows of the largest call */
static uint64_t pm_input_bytes(const tr_pm_scratch *t, int64_t max_cols, int64_t max_x) {
    const uint64_t xil = (uint64_t)(max_x + pm_max_tiles(t) * 16 * TR_PM_PAD) * sizeof(float);
    const uint64_t xq = (uint64_t)t->max_tokens * (uint64_t)tr_q4x_bytes(max_cols);
    return xil > xq ? xil : xq;
}

uint64_t tr_pm_scratch_bytes(int n_workers, int64_t max_groups, int64_t max_tokens, int64_t max_cols, int64_t max_x) {
    tr_pm_scratch t;
    t.max_groups = max_groups;
    t.max_tokens = max_tokens;
    uint64_t plan = (uint64_t)((max_groups + 1) * 3 + pm_max_chunks(&t) + 1 + 2 * pm_max_tiles(&t) + 2 * n_workers);
    return pm_input_bytes(&t, max_cols, max_x) +
           (uint64_t)n_workers * (uint64_t)(TR_PM_PANEL_FLOATS(max_cols) + TR_PM_PART) * sizeof(float) +
           plan * sizeof(int64_t);
}

int tr_pm_scratch_init(tr_pm_scratch *s, int n_workers, int64_t max_groups, int64_t max_tokens, int64_t max_cols,
                       int64_t max_x) {
    memset(s, 0, sizeof *s);
    if (n_workers < 1) n_workers = 1;
    s->n_workers = n_workers;
    s->max_groups = max_groups;
    s->max_tokens = max_tokens;
    s->max_cols = max_cols;
    s->max_x = max_x;
    uint64_t plan = (uint64_t)((max_groups + 1) * 3 + pm_max_chunks(s) + 1 + 2 * pm_max_tiles(s) + 2 * n_workers);
    s->xil = (float *)tr_alloc_aligned((size_t)pm_input_bytes(s, max_cols, max_x), 64);
    s->xq = (unsigned char *)s->xil;
    s->xq_bytes = max_tokens * (int64_t)tr_q4x_bytes(max_cols);
    s->work = (float *)tr_alloc_aligned(
        (size_t)n_workers * (size_t)(TR_PM_PANEL_FLOATS(max_cols) + TR_PM_PART) * sizeof(float), 64);
    s->plan = (int64_t *)tr_alloc_aligned((size_t)plan * sizeof(int64_t), 64);
    if (s->xil == NULL || s->work == NULL || s->plan == NULL) {
        tr_pm_scratch_free(s);
        return -1;
    }
    return 0;
}

void tr_pm_scratch_free(tr_pm_scratch *s) {
    tr_free_aligned(s->xil);
    tr_free_aligned(s->work);
    tr_free_aligned(s->plan);
    memset(s, 0, sizeof *s);
}

/* hot: begin */

int tr_matmul_one_row_per_group(const int64_t *offsets, int64_t n_groups) {
    for (int64_t g = 0; g < n_groups; g++)
        if (offsets[g + 1] - offsets[g] > 1) return 0;
    return 1;
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

/* x[i] = tr_expf(x[i] - m): the difference rounded to float first, then the table's expf_f32,
 * which is tr_expf lane by lane (question 70) */
void tr_softmax(float *x, int64_t n) {
    if (n <= 0) return;
    const tr_kernels *k = g_active != NULL ? g_active : tr_kernels_scalar();
    float m = x[0];
    for (int64_t i = 1; i < n; i++)
        if (x[i] > m) m = x[i];
    for (int64_t i = 0; i < n; i++) x[i] = x[i] - m;
    k->expf_f32(x, x, n);
    float sum = lane_sum(x, n);
    for (int64_t i = 0; i < n; i++) x[i] /= sum;
}

typedef struct {
    float *x;
    const float *y;
} swiglu_ctx;

/* Elements per call of expf_f32 in tr_swiglu: the exponentials of a chunk sit on the stack. */
#define SWIGLU_CHUNK 64

static void swiglu_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    swiglu_ctx *ctx = (swiglu_ctx *)ctx_;
    const tr_kernels *k = g_active != NULL ? g_active : tr_kernels_scalar();
    float *x = ctx->x;
    const float *y = ctx->y;
    float e[SWIGLU_CHUNK];
    for (int64_t i0 = begin; i0 < end; i0 += SWIGLU_CHUNK) {
        int64_t len = end - i0 < SWIGLU_CHUNK ? end - i0 : SWIGLU_CHUNK;
        for (int64_t i = 0; i < len; i++) e[i] = -x[i0 + i];
        k->expf_f32(e, e, len); /* e[i] = tr_expf(-v) */
        for (int64_t i = 0; i < len; i++) {
            float silu = x[i0 + i] / (1.0f + e[i]);
            x[i0 + i] = silu * y[i0 + i];
        }
    }
}

void tr_swiglu(tr_pool *pool, float *x, const float *y, int64_t n) {
    swiglu_ctx ctx;
    ctx.x = x;
    ctx.y = y;
    /* one tr_expf per element, a few ns: a few hundred per chunk are what waking a thread costs */
    tr_parallel_for(pool, n, 256, swiglu_body, &ctx);
}

typedef struct {
    swiglu_ctx sw;
    int64_t n;
    unsigned char *xq;
    size_t xq_row;
} swiglu_prep_ctx;

/* rows [begin, end): each row's swiglu (swiglu_body's elements, so tr_swiglu's bits), then the row prepared while
 * it is in this core's cache */
static void swiglu_prep_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    swiglu_prep_ctx *c = (swiglu_prep_ctx *)ctx_;
    const tr_kernels *k = tr_kernels_get();
    for (int64_t r = begin; r < end; r++) {
        swiglu_body(&c->sw, r * c->n, (r + 1) * c->n, worker);
        k->q4x_prep(c->sw.x + r * c->n, c->n, c->xq + (size_t)r * c->xq_row);
    }
}

#define TR_ARGMAX_CHUNKS 64

typedef struct {
    const float *x;
    int64_t n, n_chunks;
    float best_v[TR_ARGMAX_CHUNKS];
    int64_t best_i[TR_ARGMAX_CHUNKS]; /* -1: nothing in the chunk above -inf */
} argmax_ctx;

/* chunk c's first element, on a TR_PM_ROWS row (the head's items) */
static int64_t argmax_bound(const argmax_ctx *c, int64_t ch) {
    return ch >= c->n_chunks ? c->n : ch * c->n / c->n_chunks / TR_PM_ROWS * TR_PM_ROWS;
}

static void argmax_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    argmax_ctx *c = (argmax_ctx *)ctx_;
    for (int64_t ch = begin; ch < end; ch++) {
        const int64_t lo = argmax_bound(c, ch), hi = argmax_bound(c, ch + 1);
        /* lane l: the chunk's elements lo + l, lo + l + 8, ... in order; each keeps the first of its largest */
        float bv[8];
        int64_t bi[8];
        for (int l = 0; l < 8; l++) {
            bv[l] = -INFINITY;
            bi[l] = -1;
        }
        int64_t i = lo;
        for (; i + 8 <= hi; i += 8)
            for (int l = 0; l < 8; l++)
                if (c->x[i + l] > bv[l]) {
                    bv[l] = c->x[i + l];
                    bi[l] = i + l;
                }
        for (; i < hi; i++)
            if (c->x[i] > bv[(i - lo) & 7]) {
                bv[(i - lo) & 7] = c->x[i];
                bi[(i - lo) & 7] = i;
            }
        /* the chunk's largest, the lowest index among the lanes that hold it */
        float v = -INFINITY;
        int64_t idx = -1;
        for (int l = 0; l < 8; l++)
            if (bi[l] >= 0 && (idx < 0 || bv[l] > v || (bv[l] == v && bi[l] < idx))) {
                v = bv[l];
                idx = bi[l];
            }
        c->best_v[ch] = v;
        c->best_i[ch] = idx;
    }
}

int32_t tr_argmax_f32(tr_pool *pool, const float *x, int64_t n) {
    if (n <= 0) return 0;
    argmax_ctx c;
    c.x = x;
    c.n = n;
    const int64_t workers = pool != NULL ? tr_pool_size(pool) : 1;
    c.n_chunks = workers < TR_ARGMAX_CHUNKS ? workers : TR_ARGMAX_CHUNKS;
    if (c.n_chunks > n / TR_PM_ROWS) c.n_chunks = n / TR_PM_ROWS > 0 ? n / TR_PM_ROWS : 1;
    tr_parallel_for(pool, c.n_chunks, 1, argmax_body, &c);
    int64_t best = 0;
    float bv = x[0];
    for (int64_t ch = 0; ch < c.n_chunks; ch++)
        if (c.best_i[ch] >= 0 && c.best_v[ch] > bv) {
            bv = c.best_v[ch];
            best = c.best_i[ch];
        }
    return (int32_t)best;
}

void tr_swiglu_prepare(tr_pool *pool, float *x, const float *y, int64_t n_rows, int64_t n, void *xq) {
    swiglu_prep_ctx c;
    c.sw.x = x;
    c.sw.y = y;
    c.n = n;
    c.xq = (unsigned char *)xq;
    c.xq_row = tr_q4x_bytes(n);
    tr_parallel_for(pool, n_rows, 1, swiglu_prep_body, &c);
}

void tr_attention_head(const float *q, const float *keys, const float *values, int64_t stride, int64_t offset,
                       int64_t n_pos, int64_t head_dim, float scale, float *scores, float *out) {
    const tr_kernels *k = g_active != NULL ? g_active : tr_kernels_scalar();
    for (int64_t t = 0; t < n_pos; t++) scores[t] = k->dot_f32(q, keys + t * stride + offset, head_dim) * scale;
    tr_softmax(scores, n_pos);
    for (int64_t d = 0; d < head_dim; d++) out[d] = 0.0f;
    for (int64_t t = 0; t < n_pos; t++) k->axpy_f32(out, values + t * stride + offset, scores[t], head_dim);
}

/* Positions [t, end) of one query's row of scores (t a multiple of 4): 4 at a time, then one by one. */
static inline void attn_dots(const tr_kernels *k, const float *q, const float *keys, int64_t t, int64_t end,
                             int64_t head_dim, float scale, float *row) {
    for (; t + TR_ATTN_X <= end; t += TR_ATTN_X) {
        k->dot_f32_x4(q, keys + t * head_dim, head_dim, head_dim, row + t);
        for (int x = 0; x < TR_ATTN_X; x++) row[t + x] = row[t + x] * scale;
    }
    for (; t < end; t++) row[t] = k->dot_f32(q, keys + t * head_dim, head_dim) * scale;
}

/* Positions [t, end) of one query's weighted sum of values, in increasing position. */
static inline void attn_values(const tr_kernels *k, float *out, const float *values, int64_t t, int64_t end,
                               int64_t head_dim, const float *row) {
    for (; t + TR_ATTN_X <= end; t += TR_ATTN_X) k->axpy_f32_x4(out, values + t * head_dim, head_dim, row + t, head_dim);
    for (; t < end; t++) k->axpy_f32(out, values + t * head_dim, row[t], head_dim);
}

/* Block by block: a block of keys meets every query of the group that sees it, and fills its
 * piece of their score rows; then the softmax of every row; then the values block by block. A
 * query sees a block up to its own position, so the last blocks of a group are cut query by
 * query. Inside a block the queries go 4 at a time: a tile of 4 queries against 4 positions
 * (dot_f32_4x4, axpy_f32_4x4) wherever the quad's first query, the one with the shortest
 * context, sees all 4; each query's positions past the tiles, and the queries past the last
 * quad, 4 positions at a time and then one by one. Every query's positions go in increasing
 * order, so a score and a sum are the same operations whichever way they are grouped. */
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
        int64_t j = j0;
        for (; j + TR_ATTN_X <= n_q; j += TR_ATTN_X) {
            int64_t end = t1 < first_n_pos + j ? t1 : first_n_pos + j, t = t0;
            for (; t + TR_ATTN_X <= end; t += TR_ATTN_X)
                k->dot_f32_4x4(q + j * q_stride, q_stride, keys + t * head_dim, head_dim, head_dim, scale,
                               scores + j * score_stride + t, score_stride);
            for (int64_t i = j; i < j + TR_ATTN_X; i++)
                attn_dots(k, q + i * q_stride, keys, t, t1 < first_n_pos + i ? t1 : first_n_pos + i, head_dim, scale,
                          scores + i * score_stride);
        }
        for (; j < n_q; j++)
            attn_dots(k, q + j * q_stride, keys, t0, t1 < first_n_pos + j ? t1 : first_n_pos + j, head_dim, scale,
                      scores + j * score_stride);
    }
    for (int64_t j = 0; j < n_q; j++) {
        float *oj = out + j * out_stride;
        tr_softmax(scores + j * score_stride, first_n_pos + j);
        for (int64_t d = 0; d < head_dim; d++) oj[d] = 0.0f;
    }
    for (int64_t t0 = 0; t0 < last_n_pos; t0 += TR_ATTN_BLOCK) {
        int64_t t1 = t0 + TR_ATTN_BLOCK < last_n_pos ? t0 + TR_ATTN_BLOCK : last_n_pos;
        int64_t j0 = t0 + 1 > first_n_pos ? t0 + 1 - first_n_pos : 0;
        int64_t j = j0;
        for (; j + TR_ATTN_X <= n_q; j += TR_ATTN_X) {
            int64_t end = t1 < first_n_pos + j ? t1 : first_n_pos + j, t = t0;
            for (; t + TR_ATTN_X <= end; t += TR_ATTN_X)
                k->axpy_f32_4x4(out + j * out_stride, out_stride, values + t * head_dim, head_dim,
                                scores + j * score_stride + t, score_stride, head_dim);
            for (int64_t i = j; i < j + TR_ATTN_X; i++)
                attn_values(k, out + i * out_stride, values, t, t1 < first_n_pos + i ? t1 : first_n_pos + i, head_dim,
                            scores + i * score_stride);
        }
        for (; j < n_q; j++)
            attn_values(k, out + j * out_stride, values, t0, t1 < first_n_pos + j ? t1 : first_n_pos + j, head_dim,
                        scores + j * score_stride);
    }
}
/* hot: end */
