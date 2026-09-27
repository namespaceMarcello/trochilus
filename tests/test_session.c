/* test_session.c — tr_session_rewind: after rewinding to n and evaluating more tokens,
 * the logits are bit-identical to a fresh session that evaluated the same sequence.
 * `trochilus chat` relies on it to reuse the cache across turns.
 *
 * And the cache's pages a pass enters, touched by the pool before its layers (olmoe.c kv_touch_pass):
 * on a model whose heads have 128 dimensions (a page holds 8 positions, as the real model's), the
 * same logits as a session whose writes fault their own pages (TR_KV_TOUCH=0), pass by pass, over
 * a prompt, a decode, a rewind and a pass that ends the context; one touch a pass that enters a
 * fresh page, and none on a pass that does not.
 *
 * Runs on a tiny synthetic OLMoE (synth_olmoe.h). */
#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* setenv/unsetenv */
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/base/prof.h"
#include "../src/format/gguf.h"
#include "../src/models/model.h"
#include "synth_olmoe.h"

enum { VOCAB = 16 };

/* NULL clears the variable */
static void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value != NULL ? value : "");
#else
    if (value != NULL) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

/* both sessions evaluate tok[from, to) in passes of `step`; the number of passes whose last logits differ */
static int touch_run(tr_session *on, tr_session *off, const int32_t *tok, int64_t from, int64_t to, int64_t step) {
    int differ = 0;
    for (int64_t i = from; i < to; i += step) {
        int64_t n = to - i < step ? to - i : step;
        TR_CHECK(tr_session_eval(on, tok + i, n) == 0);
        TR_CHECK(tr_session_eval(off, tok + i, n) == 0);
        if (memcmp(tr_session_logits(on), tr_session_logits(off), VOCAB * sizeof(float)) != 0) differ++;
    }
    return differ;
}

static void touch_same_bits(tr_pool *pool, const char *argv0) {
    enum { CTX = 200 };
    /* 2 layers, n_embd 128, 1 head of 128, n_ff 8, 4 experts (2 used), vocab 16 */
    static const synth_params P = {2, 128, 1, 1, 8, 4, 2, VOCAB, 32, TR_TYPE_F32};
    char path[512], err[256];
    TR_CHECK(synth_write(&P, argv0, "test_session_touch_tmp.gguf", path, sizeof path) == 0);
    tr_model *model = tr_model_load(path, pool, err, sizeof err);
    TR_CHECK(model != NULL);
    if (model == NULL) {
        remove(path);
        return;
    }
    set_env("TR_KV_TOUCH", "0");
    tr_session *off = tr_session_create(model, CTX, 0, NULL, 0);
    set_env("TR_KV_TOUCH", NULL);
    tr_session *on = tr_session_create(model, CTX, 0, NULL, 0);
    TR_CHECK(on != NULL && off != NULL);
    if (on != NULL && off != NULL) {
        tr_session_prof(on)->enabled = tr_session_prof(off)->enabled = 1;
        int32_t tok[CTX], other[CTX];
        for (int i = 0; i < CTX; i++) {
            tok[i] = (int32_t)((i * 7 + 3) % VOCAB);
            other[i] = (int32_t)((i * 5 + 1) % VOCAB);
        }
        /* a page every 8 positions: the touches at 0 (a prompt of 10), then at 16, 24, ..., 128 (15 decode passes),
         * none after a rewind to 60 until the pass at 136, then 144, ..., 184 (7 in all), and the last pass's 192 */
        TR_CHECK_EQ_INT(touch_run(on, off, tok, 0, 10, 10), 0);
        TR_CHECK_EQ_INT(touch_run(on, off, tok, 10, 130, 1), 0);
        TR_CHECK(tr_session_rewind(on, 60) == 0 && tr_session_rewind(off, 60) == 0);
        TR_CHECK_EQ_INT(touch_run(on, off, other, 60, 75, 15), 0);
        TR_CHECK_EQ_INT(touch_run(on, off, other, 75, 190, 1), 0);
        TR_CHECK_EQ_INT(touch_run(on, off, tok, 190, 200, 10), 0);
        TR_CHECK_EQ_INT(tr_session_prof(on)->acc[TR_PHASE_PREFILL][TR_PROF_KV_TOUCH].calls, 1 + 15 + 7 + 1);
        TR_CHECK_EQ_INT(tr_session_prof(off)->acc[TR_PHASE_PREFILL][TR_PROF_KV_TOUCH].calls, 0);
    }
    tr_session_free(on);
    tr_session_free(off);
    tr_model_free(model);
    remove(path);
}

/* logits of a fresh session after evaluating seq[0..n), one eval call per chunk of `step` */
static int fresh_logits(tr_model *model, const int32_t *seq, int64_t n, int64_t step, float *out) {
    tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
    if (s == NULL) return -1;
    int rc = 0;
    for (int64_t i = 0; i < n && rc == 0; i += step) rc = tr_session_eval(s, seq + i, n - i < step ? n - i : step);
    if (rc == 0) memcpy(out, tr_session_logits(s), VOCAB * sizeof(float));
    tr_session_free(s);
    return rc;
}

int main(int argc, char **argv) {
    /* 2 layers, n_embd 16, 4 heads (2 kv), n_ff 8, 4 experts (2 used), vocab 16, context 32 */
    static const synth_params P = {2, 16, 4, 2, 8, 4, 2, VOCAB, 32, TR_TYPE_F32};
    char path[512];
    TR_CHECK(synth_write(&P, argc > 0 ? argv[0] : "", "test_session_tmp.gguf", path, sizeof path) == 0);

    tr_pool *pool = tr_pool_create(2);
    char err[256];
    tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
    TR_CHECK(model != NULL);
    tr_session *s = model != NULL ? tr_session_create(model, 0, 0, err, sizeof err) : NULL;
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

    touch_same_bits(pool, argc > 0 ? argv[0] : "");

    tr_session_free(s);
    tr_model_free(model);
    tr_pool_destroy(pool);
    remove(path);
    TR_TEST_EXIT();
}
