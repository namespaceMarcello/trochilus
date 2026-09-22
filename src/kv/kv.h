/* kv.h — the KV cache of one session.
 *
 * Layout [layer][kv head][position][head_dim], K and V in two blocks: the positions of one
 * head are contiguous, so an attention head walks its keys and then its values as two plain
 * streams, which is what the RAM serves at full bandwidth. With the positions outermost
 * ([layer][position][kv head]) a head reads head_dim floats out of every n_head_kv * head_dim:
 * a new page at every position and a prefetcher that sees nothing to follow. Measured on one
 * decode token's attention, 8 threads: 28-35 GB/s that way, 46-47 GB/s this way, of the 54 the
 * RAM gives (tests/bench_mem.c, docs/MEASUREMENTS.md "Decode a contesto lungo").
 *
 * The layout decides where a number lives, never its value: attention makes the same kernel
 * calls on the same floats in the same order, so the logits do not change by a bit. */
#ifndef TR_KV_H
#define TR_KV_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float *k, *v; /* [n_layers][n_head_kv][n_ctx][head_dim] each */
    int64_t n_layers, n_head_kv, head_dim, n_ctx;
} tr_kv;

/* Bytes tr_kv_init allocates (K and V together), for the memory guard. */
uint64_t tr_kv_bytes(int64_t n_layers, int64_t n_head_kv, int64_t head_dim, int64_t n_ctx);

/* 0 on success; -1 on a shape that is not positive or when memory runs out, and kv is then
 * empty and safe to free. Contents are not initialized: a position is read only after it was
 * written. */
int tr_kv_init(tr_kv *kv, int64_t n_layers, int64_t n_head_kv, int64_t head_dim, int64_t n_ctx);
void tr_kv_free(tr_kv *kv);

/* hot: begin */

/* Position 0 of one head of one layer; position t is t * head_dim floats further on. */
static inline const float *tr_kv_keys(const tr_kv *kv, int64_t layer, int64_t head) {
    return kv->k + ((size_t)layer * (size_t)kv->n_head_kv + (size_t)head) * (size_t)kv->n_ctx * (size_t)kv->head_dim;
}

static inline const float *tr_kv_values(const tr_kv *kv, int64_t layer, int64_t head) {
    return kv->v + ((size_t)layer * (size_t)kv->n_head_kv + (size_t)head) * (size_t)kv->n_ctx * (size_t)kv->head_dim;
}

/* Stores n_tok tokens of one layer at positions pos0 .. pos0 + n_tok - 1. k and v hold one row
 * per token with its heads one after the other, [n_tok][n_head_kv * head_dim], the way the
 * projections leave them. A position written before is overwritten (tr_session_rewind). The
 * caller keeps 0 <= pos0 and pos0 + n_tok <= n_ctx. */
void tr_kv_write(tr_kv *kv, int64_t layer, int64_t pos0, int64_t n_tok, const float *k, const float *v);

/* hot: end */

#endif
