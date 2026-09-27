/* test_kv.c — the KV cache (src/kv/kv.h): where a number lives, and that a write puts every
 * number there and nowhere else.
 *
 * The layout is the contract attention relies on: the positions of one head of one layer are
 * contiguous, head_dim floats each, heads after heads, layers after layers. The engine's
 * logits cannot see a layout (tests/test_prefill.c, test_spec.c and the oracles see a wrong
 * index instead), so the layout itself is checked here. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/kv/kv.h"

/* a value that names its own coordinates: no two slots of K or V hold the same one */
static float tag(int is_v, int64_t layer, int64_t pos, int64_t head, int64_t d) {
    return (float)(((layer * 64 + pos) * 8 + head) * 16 + d) + (is_v ? 0.5f : 0.0f);
}

/* ---- tr_kv_touch: the zero bytes it writes, and no other ---------------------------------------------------
 * The bytes it must leave come from a rule of their own, byte by byte: inside positions [lo, hi) of a stream in
 * [s0, s1) a byte is zero when it is the window's first or sits on a TR_KV_PAGE boundary; every other byte of
 * the cache keeps the fill. The counters say which of the touch's branches the cases reached. */
static int touch_start_on_page, touch_start_off_page, touch_inner_page, touch_end_on_page, touch_empty;

static int touch_case(tr_kv *kv, int64_t s0, int64_t s1, int64_t lo, int64_t hi) {
    int64_t half = kv->n_layers * kv->n_head_kv;
    size_t stream = (size_t)kv->stream * sizeof(float), row = (size_t)kv->head_dim * sizeof(float);
    memset(kv->k, 0xA5, (size_t)half * stream);
    memset(kv->v, 0xA5, (size_t)half * stream);
    tr_kv_touch(kv, s0, s1, lo, hi);
    if (s0 >= s1 || lo >= hi) touch_empty++;
    int wrong = 0;
    for (int64_t s = 0; s < 2 * half; s++) {
        const unsigned char *base = (const unsigned char *)(s < half ? kv->k : kv->v) + (size_t)(s % half) * stream;
        const unsigned char *start = base + (size_t)lo * row, *end = base + (size_t)hi * row;
        int in = s >= s0 && s < s1 && lo < hi;
        if (in && (uintptr_t)start % TR_KV_PAGE == 0) touch_start_on_page++;
        if (in && (uintptr_t)start % TR_KV_PAGE != 0) touch_start_off_page++;
        if (in && (uintptr_t)end % TR_KV_PAGE == 0) touch_end_on_page++;
        for (size_t b = 0; b < stream; b++) {
            const unsigned char *a = base + b;
            int page = (uintptr_t)a % TR_KV_PAGE == 0, inside = in && a >= start && a < end;
            if (inside && a > start && page) touch_inner_page++;
            if (*a != (inside && (a == start || page) ? 0x00 : 0xA5)) wrong++;
        }
    }
    return wrong;
}

/* the cases on one shape: the whole cache, a window, one position across K and V, nothing, the last position,
 * and (rows that divide a page) windows from a page's first position and up to one */
static void touch_shape(int64_t n_layers, int64_t n_head_kv, int64_t head_dim, int64_t n_ctx) {
    tr_kv kv;
    TR_CHECK(tr_kv_init(&kv, n_layers, n_head_kv, head_dim, n_ctx) == 0);
    if (kv.k == NULL) return;
    int64_t streams = tr_kv_streams(&kv), half = streams / 2;
    TR_CHECK_EQ_INT(streams, 2 * n_layers * n_head_kv);
    TR_CHECK_EQ_INT(touch_case(&kv, 0, streams, 0, n_ctx), 0);
    TR_CHECK_EQ_INT(touch_case(&kv, 0, streams, 8, 72), 0);
    TR_CHECK_EQ_INT(touch_case(&kv, half - 1, half + 1, 13, 14), 0);
    TR_CHECK_EQ_INT(touch_case(&kv, 0, streams, 20, 20), 0);
    TR_CHECK_EQ_INT(touch_case(&kv, 1, 1, 0, n_ctx), 0);
    TR_CHECK_EQ_INT(touch_case(&kv, streams - 1, streams, n_ctx - 1, n_ctx), 0);
    int64_t row = head_dim * (int64_t)sizeof(float), per_page = TR_KV_PAGE / row;
    if (TR_KV_PAGE % row == 0 && 2 * per_page + 5 <= n_ctx) {
        TR_CHECK_EQ_INT(touch_case(&kv, 0, 1, per_page, 2 * per_page + 5), 0);
        TR_CHECK_EQ_INT(touch_case(&kv, 0, 1, 1, per_page), 0);
    }
    tr_kv_free(&kv);
}

/* ---- tr_kv_touch_pass: a pass's own pages, over a pool, never past its end ---------------------------------------
 * It must touch exactly what tr_kv_touch(every stream, lo, hi) touches when the positions reach a fresh page, and
 * nothing when they do not: every byte from position hi on keeps the fill. */
static int pass_touched, pass_skipped;

static int pass_case(tr_kv *kv, tr_pool *pool, int64_t lo, int64_t hi) {
    int64_t half = kv->n_layers * kv->n_head_kv;
    size_t stream = (size_t)kv->stream * sizeof(float), row = (size_t)kv->head_dim * sizeof(float);
    memset(kv->k, 0xA5, (size_t)half * stream);
    memset(kv->v, 0xA5, (size_t)half * stream);
    int touched = tr_kv_touch_pass(kv, pool, lo, hi), wrong = touched != tr_kv_fresh_page(kv, lo, hi);
    if (touched) pass_touched++;
    else pass_skipped++;
    for (int64_t s = 0; s < 2 * half; s++) {
        const unsigned char *base = (const unsigned char *)(s < half ? kv->k : kv->v) + (size_t)(s % half) * stream;
        for (size_t b = 0; b < stream; b++) {
            int zero = touched && b >= (size_t)lo * row && b < (size_t)hi * row &&
                       (b == (size_t)lo * row || (uintptr_t)(base + b) % TR_KV_PAGE == 0);
            if (base[b] != (zero ? 0x00 : 0xA5)) wrong++;
        }
    }
    return wrong;
}

static void pass_shape(tr_pool *pool) {
    tr_kv kv;
    TR_CHECK(tr_kv_init(&kv, 2, 3, 128, 100) == 0); /* 8 positions a page */
    if (kv.k == NULL) return;
    TR_CHECK_EQ_INT(pass_case(&kv, pool, 0, 10), 0);  /* a prompt: pages 0 and 1 */
    TR_CHECK_EQ_INT(pass_case(&kv, pool, 10, 11), 0); /* page 1 holds 8 and 9: nothing */
    TR_CHECK_EQ_INT(pass_case(&kv, pool, 15, 17), 0); /* 16 starts page 2 */
    TR_CHECK_EQ_INT(pass_case(&kv, pool, 16, 16), 0); /* no position */
    TR_CHECK_EQ_INT(pass_case(&kv, pool, 57, 63), 0); /* inside page 7 */
    TR_CHECK_EQ_INT(pass_case(&kv, pool, 95, 100), 0); /* 96 starts the last page */
    tr_kv_free(&kv);
}

/* ---- tr_kv_fresh_page against the pages themselves ------------------------------------------------------------
 * Stream by stream, at its real addresses: positions [lo, hi) enter a fresh page when one of the pages their bytes
 * lie on holds no byte of positions [0, lo). The answer must be tr_kv_fresh_page's for every stream, which holds
 * only while every stream starts a page (the blocks page-aligned, a stream whole pages). */
static int fresh_yes, fresh_no;

static int fresh_shape(int64_t n_layers, int64_t n_head_kv, int64_t head_dim, int64_t n_ctx) {
    tr_kv kv;
    TR_CHECK(tr_kv_init(&kv, n_layers, n_head_kv, head_dim, n_ctx) == 0);
    if (kv.k == NULL) return 1;
    TR_CHECK(kv.stream >= n_ctx * head_dim && kv.stream * (int64_t)sizeof(float) % TR_KV_PAGE == 0);
    int64_t half = n_layers * n_head_kv, row = head_dim * (int64_t)sizeof(float);
    int wrong = 0;
    for (int64_t lo = 0; lo < n_ctx; lo++)
        for (int64_t hi = lo + 1; hi <= n_ctx; hi = hi < lo + 12 ? hi + 1 : (hi < n_ctx ? n_ctx : n_ctx + 1)) {
            int want = tr_kv_fresh_page(&kv, lo, hi);
            if (want) fresh_yes++;
            else fresh_no++;
            for (int64_t s = 0; s < 2 * half; s++) {
                uintptr_t base = (uintptr_t)((s < half ? kv.k : kv.v) + (size_t)(s % half) * (size_t)kv.stream);
                uintptr_t a = base + (uintptr_t)(lo * row), b = base + (uintptr_t)(hi * row);
                int fresh = 0;
                for (uintptr_t page = a / TR_KV_PAGE * TR_KV_PAGE; page < b; page += TR_KV_PAGE)
                    if (page >= a || lo == 0) fresh = 1; /* no earlier byte of this stream on the page */
                if (fresh != want) wrong++;
            }
        }
    tr_kv_free(&kv);
    return wrong;
}

int main(void) {
    const int64_t n_layers = 3, n_head_kv = 4, head_dim = 8, n_ctx = 16, n_kv = n_head_kv * head_dim;
    const float untouched = -1.0f;

    /* a stream of 16 positions of 8 floats is padded to a page: 1024 floats */
    TR_CHECK_EQ_INT(tr_kv_bytes(n_layers, n_head_kv, head_dim, n_ctx), 3 * 4 * 1024 * 2 * sizeof(float));
    TR_CHECK_EQ_INT(tr_kv_bytes(0, n_head_kv, head_dim, n_ctx), 0);

    tr_kv bad;
    TR_CHECK(tr_kv_init(&bad, n_layers, n_head_kv, 0, n_ctx) == -1);
    TR_CHECK(bad.k == NULL && bad.v == NULL);
    tr_kv_free(&bad);

    tr_kv kv;
    TR_CHECK(tr_kv_init(&kv, n_layers, n_head_kv, head_dim, n_ctx) == 0);
    if (kv.k == NULL) TR_TEST_EXIT();
    size_t n_elems = (size_t)(n_layers * n_head_kv) * (size_t)kv.stream;
    for (size_t i = 0; i < n_elems; i++) kv.k[i] = kv.v[i] = untouched;

    /* the layout: one head's positions in a row, then the next head, then the next layer, each head's
     * stream starting a page */
    TR_CHECK_EQ_INT(kv.stream, 1024);
    TR_CHECK((uintptr_t)kv.k % TR_KV_PAGE == 0 && (uintptr_t)kv.v % TR_KV_PAGE == 0);
    TR_CHECK(tr_kv_keys(&kv, 0, 0) == kv.k);
    TR_CHECK(tr_kv_values(&kv, 0, 0) == kv.v);
    for (int64_t L = 0; L < n_layers; L++)
        for (int64_t h = 0; h < n_head_kv; h++) {
            const float *expect_k = kv.k + (L * n_head_kv + h) * kv.stream;
            TR_CHECK(tr_kv_keys(&kv, L, h) == expect_k);
            TR_CHECK(tr_kv_values(&kv, L, h) == kv.v + (expect_k - kv.k));
        }

    /* rows as the projections leave them: [token][head][d] */
    float *k = (float *)malloc((size_t)(n_ctx * n_kv) * sizeof(float));
    float *v = (float *)malloc((size_t)(n_ctx * n_kv) * sizeof(float));
    TR_CHECK(k != NULL && v != NULL);
    if (k == NULL || v == NULL) TR_TEST_EXIT();

    /* layer 1 written in three passes of 5, 1 and 4 tokens; layers 0 and 2 stay untouched */
    const int64_t L1 = 1, split[4] = {0, 5, 6, 10};
    for (int p = 0; p < 3; p++) {
        int64_t pos0 = split[p], n_tok = split[p + 1] - split[p];
        for (int64_t i = 0; i < n_tok; i++)
            for (int64_t h = 0; h < n_head_kv; h++)
                for (int64_t d = 0; d < head_dim; d++) {
                    k[i * n_kv + h * head_dim + d] = tag(0, L1, pos0 + i, h, d);
                    v[i * n_kv + h * head_dim + d] = tag(1, L1, pos0 + i, h, d);
                }
        tr_kv_write(&kv, L1, pos0, n_tok, k, v);
    }
    int wrong = 0;
    for (int64_t L = 0; L < n_layers; L++)
        for (int64_t h = 0; h < n_head_kv; h++)
            for (int64_t t = 0; t < n_ctx; t++)
                for (int64_t d = 0; d < head_dim; d++) {
                    int written = L == L1 && t < 10;
                    float want_k = written ? tag(0, L, t, h, d) : untouched;
                    float want_v = written ? tag(1, L, t, h, d) : untouched;
                    /* position t of a head is t * head_dim floats after its position 0 */
                    if (tr_kv_keys(&kv, L, h)[t * head_dim + d] != want_k) wrong++;
                    if (tr_kv_values(&kv, L, h)[t * head_dim + d] != want_v) wrong++;
                }
    TR_CHECK_EQ_INT(wrong, 0);

    /* after a rewind the same positions are written again: the new values replace the old
     * ones, and the positions before and after the pass keep theirs */
    for (int64_t i = 0; i < 2; i++)
        for (int64_t j = 0; j < n_kv; j++) {
            k[i * n_kv + j] = 1000.0f + (float)(i * n_kv + j);
            v[i * n_kv + j] = 2000.0f + (float)(i * n_kv + j);
        }
    tr_kv_write(&kv, L1, 4, 2, k, v);
    wrong = 0;
    for (int64_t h = 0; h < n_head_kv; h++)
        for (int64_t t = 0; t < n_ctx; t++)
            for (int64_t d = 0; d < head_dim; d++) {
                float want_k = t >= 10 ? untouched : tag(0, L1, t, h, d);
                float want_v = t >= 10 ? untouched : tag(1, L1, t, h, d);
                if (t == 4 || t == 5) {
                    want_k = 1000.0f + (float)((t - 4) * n_kv + h * head_dim + d);
                    want_v = 2000.0f + (float)((t - 4) * n_kv + h * head_dim + d);
                }
                if (tr_kv_keys(&kv, L1, h)[t * head_dim + d] != want_k) wrong++;
                if (tr_kv_values(&kv, L1, h)[t * head_dim + d] != want_v) wrong++;
            }
    TR_CHECK_EQ_INT(wrong, 0);

    /* the last layer, last position: the write ends where the last stream's positions end, and the page's
     * padding after them keeps its fill */
    tr_kv_write(&kv, n_layers - 1, n_ctx - 1, 1, k, v);
    TR_CHECK(tr_kv_keys(&kv, n_layers - 1, n_head_kv - 1)[n_ctx * head_dim - 1] == k[n_kv - 1]);
    TR_CHECK(tr_kv_values(&kv, n_layers - 1, n_head_kv - 1)[n_ctx * head_dim - 1] == v[n_kv - 1]);
    TR_CHECK(kv.k[n_elems - (size_t)(kv.stream - n_ctx * head_dim)] == untouched);

    free(k);
    free(v);
    tr_kv_free(&kv);
    TR_CHECK(kv.k == NULL && kv.v == NULL);

    /* the engine's rows (512 B, 8 to a page) and rows of 32 B (128 to a page) */
    touch_shape(2, 3, 128, 100);
    touch_shape(1, 2, 8, 700);
    TR_CHECK(touch_start_on_page > 0 && touch_start_off_page > 0);
    TR_CHECK(touch_inner_page > 0 && touch_end_on_page > 0 && touch_empty > 0);

    /* rows that divide a page (512, 32, 16 B) and rows that straddle pages (12, 400 B) */
    TR_CHECK_EQ_INT(fresh_shape(2, 3, 128, 100), 0);
    TR_CHECK_EQ_INT(fresh_shape(1, 2, 8, 700), 0);
    TR_CHECK_EQ_INT(fresh_shape(1, 1, 4, 33), 0);
    TR_CHECK_EQ_INT(fresh_shape(2, 1, 3, 100), 0);
    TR_CHECK_EQ_INT(fresh_shape(1, 2, 100, 50), 0);
    TR_CHECK(fresh_yes > 0 && fresh_no > 0);

    /* a pass's pages, on the calling thread and over three */
    pass_shape(NULL);
    tr_pool *pool = tr_pool_create(3);
    TR_CHECK(pool != NULL);
    if (pool != NULL) {
        pass_shape(pool);
        tr_pool_destroy(pool);
    }
    TR_CHECK(pass_touched > 0 && pass_skipped > 0);
    TR_TEST_EXIT();
}
