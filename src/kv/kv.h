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
 * Every head's stream starts a page (the blocks are page-aligned, a stream padded to whole
 * pages), so the positions whose rows enter a fresh page are the same in every stream
 * (tr_kv_fresh_page): a pass faults the pages it is about to write over the pool, in one call
 * (tr_kv_touch), instead of one at a time inside tr_kv_write.
 *
 * The layout decides where a number lives, never its value: attention makes the same kernel
 * calls on the same floats in the same order, so the logits do not change by a bit. */
#ifndef TR_KV_H
#define TR_KV_H

#include <stddef.h>
#include <stdint.h>

#include "../base/threads.h"

/* Bytes of the pages the cache is laid out on: the smallest page of the platforms the engine runs
 * on (a larger page holds several, and is touched more than once). */
#define TR_KV_PAGE 4096

typedef struct {
    float *k, *v; /* [n_layers][n_head_kv][stream] each: a stream holds n_ctx positions of head_dim floats */
    int64_t n_layers, n_head_kv, head_dim, n_ctx;
    int64_t stream; /* floats from a stream's position 0 to the next stream's: n_ctx * head_dim up to a page */
} tr_kv;

/* Bytes tr_kv_init allocates (K and V together, every stream padded to whole pages), for the memory
 * guard. */
uint64_t tr_kv_bytes(int64_t n_layers, int64_t n_head_kv, int64_t head_dim, int64_t n_ctx);

/* 0 on success; -1 on a shape that is not positive or when memory runs out, and kv is then
 * empty and safe to free. Contents are not initialized: a position is read only after it was
 * written. */
int tr_kv_init(tr_kv *kv, int64_t n_layers, int64_t n_head_kv, int64_t head_dim, int64_t n_ctx);
void tr_kv_free(tr_kv *kv);

/* hot: begin */

/* Position 0 of one head of one layer; position t is t * head_dim floats further on. */
static inline const float *tr_kv_keys(const tr_kv *kv, int64_t layer, int64_t head) {
    return kv->k + ((size_t)layer * (size_t)kv->n_head_kv + (size_t)head) * (size_t)kv->stream;
}

static inline const float *tr_kv_values(const tr_kv *kv, int64_t layer, int64_t head) {
    return kv->v + ((size_t)layer * (size_t)kv->n_head_kv + (size_t)head) * (size_t)kv->stream;
}

/* Stores n_tok tokens of one layer at positions pos0 .. pos0 + n_tok - 1. k and v hold one row
 * per token with its heads one after the other, [n_tok][n_head_kv * head_dim], the way the
 * projections leave them. A position written before is overwritten (tr_session_rewind). The
 * caller keeps 0 <= pos0 and pos0 + n_tok <= n_ctx. */
void tr_kv_write(tr_kv *kv, int64_t layer, int64_t pos0, int64_t n_tok, const float *k, const float *v);

/* The cache's streams, a head's positions each: K's (layer, head) in order, then V's. */
static inline int64_t tr_kv_streams(const tr_kv *kv) {
    return 2 * kv->n_layers * kv->n_head_kv;
}

/* 1 when the rows of positions [lo, hi) reach a page that no position below lo reaches (lo == 0: a
 * stream's first page), 0 when every page they lie on already holds an earlier position; the same
 * answer for every stream, each starting a page. The caller keeps 0 <= lo < hi <= n_ctx. */
static inline int tr_kv_fresh_page(const tr_kv *kv, int64_t lo, int64_t hi) {
    int64_t row = kv->head_dim * (int64_t)sizeof(float);
    return lo == 0 || (hi * row - 1) / TR_KV_PAGE > (lo * row - 1) / TR_KV_PAGE;
}

/* The pages of positions [lo, hi) of streams [s0, s1) faulted in before their rows are written: a
 * zero byte at position lo's first byte and at every TR_KV_PAGE boundary up to position hi, on
 * whichever thread runs this. The cache is allocated untouched, so without it each stream enters
 * a fresh page every TR_KV_PAGE / (head_dim * 4) positions inside tr_kv_write, one fault at a
 * time on the thread that writes (docs/MEASUREMENTS.md §The KV's pages touched in time). Every
 * byte written belongs to a position in [lo, hi): the caller passes only positions no result reads
 * before their next write, and keeps 0 <= lo <= hi <= n_ctx, 0 <= s0 <= s1 <= tr_kv_streams. */
void tr_kv_touch(tr_kv *kv, int64_t s0, int64_t s1, int64_t lo, int64_t hi);

/* The pages positions [lo, hi) enter, faulted in over the pool before a pass writes them: when they
 * reach a fresh page (tr_kv_fresh_page), one call touches exactly them (tr_kv_touch), the streams
 * split over the pool's workers; 1 when it touched, 0 when every page they lie on already held an
 * earlier position. Never past hi: pages present past the streams' ends slow the attention's reads
 * by 1.5-2.3% (docs/MEASUREMENTS.md §The KV's pages touched in time). The caller keeps 0 <= lo <=
 * hi <= n_ctx and passes only positions no result reads before their next write. */
int tr_kv_touch_pass(tr_kv *kv, tr_pool *pool, int64_t lo, int64_t hi);

/* hot: end */

#endif
