/* kv.c — allocation and writes of the KV cache. See kv.h for the layout and why. */
#include "kv.h"

#include <string.h>

#include "../base/platform.h"

/* floats of one stream: n_ctx rows of head_dim, up to whole pages */
static int64_t stream_floats(int64_t head_dim, int64_t n_ctx) {
    const int64_t page = TR_KV_PAGE / (int64_t)sizeof(float);
    return (n_ctx * head_dim + page - 1) / page * page;
}

uint64_t tr_kv_bytes(int64_t n_layers, int64_t n_head_kv, int64_t head_dim, int64_t n_ctx) {
    if (n_layers <= 0 || n_head_kv <= 0 || head_dim <= 0 || n_ctx <= 0) return 0;
    return (uint64_t)n_layers * (uint64_t)n_head_kv * (uint64_t)stream_floats(head_dim, n_ctx) * 2 * sizeof(float);
}

int tr_kv_init(tr_kv *kv, int64_t n_layers, int64_t n_head_kv, int64_t head_dim, int64_t n_ctx) {
    memset(kv, 0, sizeof *kv);
    uint64_t bytes = tr_kv_bytes(n_layers, n_head_kv, head_dim, n_ctx);
    if (bytes == 0) return -1;
    kv->k = (float *)tr_alloc_aligned((size_t)(bytes / 2), TR_KV_PAGE);
    kv->v = (float *)tr_alloc_aligned((size_t)(bytes / 2), TR_KV_PAGE);
    if (kv->k == NULL || kv->v == NULL) {
        tr_kv_free(kv);
        return -1;
    }
    kv->n_layers = n_layers;
    kv->n_head_kv = n_head_kv;
    kv->head_dim = head_dim;
    kv->n_ctx = n_ctx;
    kv->stream = stream_floats(head_dim, n_ctx);
    return 0;
}

void tr_kv_free(tr_kv *kv) {
    if (kv == NULL) return;
    tr_free_aligned(kv->k);
    tr_free_aligned(kv->v);
    memset(kv, 0, sizeof *kv);
}

/* hot: begin */

/* Head by head: the destination of a head is one stream, n_tok * head_dim floats in a row. */
void tr_kv_write(tr_kv *kv, int64_t layer, int64_t pos0, int64_t n_tok, const float *k, const float *v) {
    int64_t head_dim = kv->head_dim, n_kv = kv->n_head_kv * head_dim;
    size_t head_bytes = (size_t)head_dim * sizeof(float);
    for (int64_t h = 0; h < kv->n_head_kv; h++) {
        size_t head0 = ((size_t)layer * (size_t)kv->n_head_kv + (size_t)h) * (size_t)kv->stream;
        float *dst_k = kv->k + head0 + (size_t)pos0 * (size_t)head_dim;
        float *dst_v = kv->v + head0 + (size_t)pos0 * (size_t)head_dim;
        for (int64_t i = 0; i < n_tok; i++) {
            memcpy(dst_k + i * head_dim, k + i * n_kv + h * head_dim, head_bytes);
            memcpy(dst_v + i * head_dim, v + i * n_kv + h * head_dim, head_bytes);
        }
    }
}

/* The byte goes through a volatile store: no one reads it before the row's write, only the page matters. */
void tr_kv_touch(tr_kv *kv, int64_t s0, int64_t s1, int64_t lo, int64_t hi) {
    size_t stream = (size_t)kv->stream, row = (size_t)kv->head_dim * sizeof(float);
    int64_t half = kv->n_layers * kv->n_head_kv;
    for (int64_t s = s0; s < s1 && lo < hi; s++) {
        float *base = s < half ? kv->k + (size_t)s * stream : kv->v + (size_t)(s - half) * stream;
        unsigned char *p = (unsigned char *)base + (size_t)lo * row, *end = (unsigned char *)base + (size_t)hi * row;
        while (p < end) {
            *(volatile unsigned char *)p = 0;
            p += TR_KV_PAGE - (size_t)((uintptr_t)p & (TR_KV_PAGE - 1));
        }
    }
}

typedef struct {
    tr_kv *kv;
    int64_t lo, hi;
} touch_ctx;

static void touch_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const touch_ctx *c = (const touch_ctx *)ctx_;
    tr_kv_touch(c->kv, begin, end, c->lo, c->hi);
}

int tr_kv_touch_pass(tr_kv *kv, tr_pool *pool, int64_t lo, int64_t hi) {
    if (lo >= hi || !tr_kv_fresh_page(kv, lo, hi)) return 0;
    touch_ctx c = {kv, lo, hi};
    tr_parallel_for(pool, tr_kv_streams(kv), 1, touch_body, &c);
    return 1;
}

/* hot: end */
