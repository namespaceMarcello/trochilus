/* test_session.c — tr_session_rewind: after rewinding to n and evaluating more tokens,
 * the logits are bit-identical to a fresh session that evaluated the same sequence.
 * `trochilus chat` relies on it to reuse the cache across turns.
 *
 * Runs on a tiny synthetic OLMoE (synth_olmoe.h). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/format/gguf.h"
#include "../src/models/model.h"
#include "synth_olmoe.h"

enum { VOCAB = 16 };

/* logits of a fresh session after evaluating seq[0..n), one eval call per chunk of `step` */
static int fresh_logits(tr_model *model, const int32_t *seq, int64_t n, int64_t step, float *out) {
    tr_session *s = tr_session_create(model, 0, NULL, 0);
    if (s == NULL) return -1;
    int rc = 0;
    for (int64_t i = 0; i < n && rc == 0; i += step) rc = tr_session_eval(s, seq + i, n - i < step ? n - i : step);
    if (rc == 0) memcpy(out, tr_session_logits(s), VOCAB * sizeof(float));
    tr_session_free(s);
    return rc;
}

int main(int argc, char **argv) {
    /* 2 layers, n_embd 16, 4 heads (2 kv), n_ff 8, 4 experts (2 used), vocab 16, context 32 */
    static const synth_params P = {2, 16, 4, 2, 8, 4, 2, VOCAB, 32};
    char path[512];
    TR_CHECK(synth_write(&P, argc > 0 ? argv[0] : "", "test_session_tmp.gguf", path, sizeof path) == 0);

    tr_pool *pool = tr_pool_create(2);
    char err[256];
    tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
    TR_CHECK(model != NULL);
    tr_session *s = model != NULL ? tr_session_create(model, 0, err, sizeof err) : NULL;
    TR_CHECK(s != NULL);
    if (s == NULL) {
        tr_model_free(model);
        tr_pool_destroy(pool);
        remove(path);
        TR_TEST_EXIT();
    }

    static const int32_t a[12] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};
    static const int32_t b[6] = {9, 7, 9, 3, 2, 3};
    int32_t seq[11];
    float want[VOCAB], got[VOCAB];

    /* evaluate a, go back to 5 tokens, continue with b */
    TR_CHECK(tr_session_eval(s, a, 12) == 0);
    TR_CHECK(tr_session_rewind(s, 13) == -1);
    TR_CHECK(tr_session_rewind(s, -1) == -1);
    TR_CHECK_EQ_INT(tr_session_pos(s), 12);
    TR_CHECK(tr_session_rewind(s, 5) == 0);
    TR_CHECK_EQ_INT(tr_session_pos(s), 5);
    TR_CHECK(tr_session_eval(s, b, 6) == 0);
    memcpy(seq, a, 5 * sizeof(int32_t));
    memcpy(seq + 5, b, 6 * sizeof(int32_t));
    memcpy(got, tr_session_logits(s), sizeof got);
    TR_CHECK(fresh_logits(model, seq, 11, 11, want) == 0);
    TR_CHECK(memcmp(got, want, sizeof got) == 0);
    TR_CHECK(fresh_logits(model, seq, 11, 1, want) == 0);     /* token by token: same numbers */
    TR_CHECK(memcmp(got, want, sizeof got) == 0);

    /* back to the start */
    TR_CHECK(tr_session_rewind(s, 0) == 0);
    TR_CHECK(tr_session_eval(s, b, 6) == 0);
    memcpy(got, tr_session_logits(s), sizeof got);
    TR_CHECK(fresh_logits(model, b, 6, 6, want) == 0);
    TR_CHECK(memcmp(got, want, sizeof got) == 0);

    /* rewinding to the current position changes nothing */
    TR_CHECK(tr_session_rewind(s, 6) == 0);
    TR_CHECK(tr_session_eval(s, a, 1) == 0);
    memcpy(seq, b, 6 * sizeof(int32_t));
    seq[6] = a[0];
    memcpy(got, tr_session_logits(s), sizeof got);
    TR_CHECK(fresh_logits(model, seq, 7, 7, want) == 0);
    TR_CHECK(memcmp(got, want, sizeof got) == 0);

    tr_session_free(s);
    tr_model_free(model);
    tr_pool_destroy(pool);
    remove(path);
    TR_TEST_EXIT();
}
