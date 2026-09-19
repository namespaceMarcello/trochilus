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

int main(void) {
    const int64_t n_layers = 3, n_head_kv = 4, head_dim = 8, n_ctx = 16, n_kv = n_head_kv * head_dim;
    const float untouched = -1.0f;

    TR_CHECK_EQ_INT(tr_kv_bytes(n_layers, n_head_kv, head_dim, n_ctx), 3 * 4 * 8 * 16 * 2 * sizeof(float));
    TR_CHECK_EQ_INT(tr_kv_bytes(0, n_head_kv, head_dim, n_ctx), 0);

    tr_kv bad;
    TR_CHECK(tr_kv_init(&bad, n_layers, n_head_kv, 0, n_ctx) == -1);
    TR_CHECK(bad.k == NULL && bad.v == NULL);
    tr_kv_free(&bad);

    tr_kv kv;
    TR_CHECK(tr_kv_init(&kv, n_layers, n_head_kv, head_dim, n_ctx) == 0);
    if (kv.k == NULL) TR_TEST_EXIT();
    size_t n_elems = (size_t)(n_layers * n_head_kv * n_ctx * head_dim);
    for (size_t i = 0; i < n_elems; i++) kv.k[i] = kv.v[i] = untouched;

    /* the layout: one head's positions in a row, then the next head, then the next layer */
    TR_CHECK(tr_kv_keys(&kv, 0, 0) == kv.k);
    TR_CHECK(tr_kv_values(&kv, 0, 0) == kv.v);
    for (int64_t L = 0; L < n_layers; L++)
        for (int64_t h = 0; h < n_head_kv; h++) {
            const float *expect_k = kv.k + (L * n_head_kv + h) * n_ctx * head_dim;
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

    /* the last layer, last position: the write ends exactly where the cache ends */
    tr_kv_write(&kv, n_layers - 1, n_ctx - 1, 1, k, v);
    TR_CHECK(kv.k[n_elems - 1] == k[n_kv - 1]);
    TR_CHECK(kv.v[n_elems - 1] == v[n_kv - 1]);

    free(k);
    free(v);
    tr_kv_free(&kv);
    TR_CHECK(kv.k == NULL && kv.v == NULL);
    TR_TEST_EXIT();
}
