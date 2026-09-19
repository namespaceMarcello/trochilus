/* test_model_prof.c — the profiler wired into a session (src/models/model.h
 * tr_session_prof, src/models/olmoe.c forward_one): disabled by default, and
 * turning it on must never change a computed value.
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

/* ------------------------------------------------------------------ test */

int main(int argc, char **argv) {
    /* 1 layer, n_embd 8, 2 heads (1 kv), n_ff 4, 2 experts (1 used), vocab 4, context 8 */
    static const synth_params P = {1, 8, 2, 1, 4, 2, 1, 4, 8, TR_TYPE_F32};
    char path[512];
    TR_CHECK(synth_write(&P, argc > 0 ? argv[0] : "", "test_model_prof_tmp.gguf", path, sizeof path) == 0);

    tr_pool *pool = tr_pool_create(1);
    TR_CHECK(pool != NULL);
    if (pool == NULL) { remove(path); TR_TEST_EXIT(); }

    char err[256];
    tr_model *model = tr_model_load(path, pool, err, sizeof err);
    TR_CHECK(model != NULL);
    if (model == NULL) {
        fprintf(stderr, "load failed: %s\n", err);
        tr_pool_destroy(pool);
        remove(path);
        TR_TEST_EXIT();
    }
    const tr_model_info *info = tr_model_get_info(model);

    tr_session *sess = tr_session_create(model, 0, 0, err, sizeof err);
    TR_CHECK(sess != NULL);
    if (sess == NULL) {
        fprintf(stderr, "session failed: %s\n", err);
        tr_model_free(model);
        tr_pool_destroy(pool);
        remove(path);
        TR_TEST_EXIT();
    }

    /* disabled by default (docs/ARCHITETTURA.md §Profilazione) */
    tr_prof *prof = tr_session_prof(sess);
    TR_CHECK(prof != NULL);
    TR_CHECK(prof->enabled == 0);

    int32_t tokens[3] = {0, 1, 2};

    /* baseline: profiler left disabled */
    TR_CHECK(tr_session_eval(sess, tokens, 3) == 0);
    size_t logits_bytes = (size_t)info->vocab_size * sizeof(float);
    float *logits_off = (float *)malloc(logits_bytes);
    TR_CHECK(logits_off != NULL);
    if (logits_off != NULL) memcpy(logits_off, tr_session_logits(sess), logits_bytes);

    /* a fresh session with the profiler enabled must produce bit-identical
     * logits: the profiler must never change a computed value. */
    tr_session *sess2 = tr_session_create(model, 0, 0, err, sizeof err);
    TR_CHECK(sess2 != NULL);
    if (sess2 != NULL) {
        tr_prof *prof2 = tr_session_prof(sess2);
        prof2->enabled = 1;
        prof2->phase = TR_PHASE_PREFILL;
        TR_CHECK(tr_session_eval(sess2, tokens, 3) == 0);

        if (logits_off != NULL)
            TR_CHECK(memcmp(logits_off, tr_session_logits(sess2), logits_bytes) == 0);

        /* it recorded real work: the 3 tokens ran as one forward pass (one TOKEN
         * zone, MoE zones once per layer of the pass: 1 layer), and it counted
         * bytes of weights touched (the material for MiB/token, GB/s). */
        TR_CHECK_EQ_INT(prof2->tokens[TR_PHASE_PREFILL], 3);
        TR_CHECK_EQ_INT(prof2->acc[TR_PHASE_PREFILL][TR_PROF_TOKEN].calls, 1);
        TR_CHECK_EQ_INT(prof2->acc[TR_PHASE_PREFILL][TR_PROF_ROUTER].calls, 1);
        TR_CHECK_EQ_INT(prof2->acc[TR_PHASE_PREFILL][TR_PROF_EXPERT_GATE_UP].calls, 1);
        TR_CHECK_EQ_INT(prof2->acc[TR_PHASE_DECODE][TR_PROF_TOKEN].calls, 0);
        TR_CHECK(prof2->weight_bytes_touched[TR_PHASE_PREFILL] > 0);

        /* bytes per zone: token i of the pass reads i + 1 cached positions of K and of V, once
         * per query head (2 heads of 4 floats: 8 floats a position), so 1 + 2 + 3 positions;
         * and what the zones read adds up to the totals of the phase */
        const tr_prof_acc *acc = prof2->acc[TR_PHASE_PREFILL];
        uint64_t kv_pass = (uint64_t)(1 + 2 + 3) * 8 * 2 * sizeof(float);
        TR_CHECK_EQ_INT(prof2->kv_bytes_read[TR_PHASE_PREFILL], kv_pass);
        TR_CHECK_EQ_INT(acc[TR_PROF_ATTENTION].bytes, kv_pass);
        /* F32 weights: wq 8x8, wk and wv 4x8 (1 kv head), one expert of 3 matrices 4x8 */
        TR_CHECK_EQ_INT(acc[TR_PROF_QKV_PROJ].bytes, (64 + 32 + 32) * sizeof(float));
        TR_CHECK_EQ_INT(acc[TR_PROF_EXPERT_DOWN].bytes * 2, acc[TR_PROF_EXPERT_GATE_UP].bytes);
        uint64_t zones = 0;
        for (int z = 0; z < TR_PROF_ZONE_COUNT; z++) zones += acc[z].bytes;
        TR_CHECK_EQ_INT(zones, prof2->weight_bytes_touched[TR_PHASE_PREFILL] + kv_pass);

        /* a decode pass at position 3 reads 4 positions */
        prof2->phase = TR_PHASE_DECODE;
        TR_CHECK(tr_session_eval(sess2, tokens, 1) == 0);
        TR_CHECK_EQ_INT(prof2->kv_bytes_read[TR_PHASE_DECODE], (uint64_t)4 * 8 * 2 * sizeof(float));
        TR_CHECK_EQ_INT(prof2->kv_bytes_read[TR_PHASE_PREFILL], kv_pass);

        tr_session_free(sess2);
    }

    free(logits_off);
    tr_session_free(sess);
    tr_model_free(model);
    tr_pool_destroy(pool);
    remove(path);
    TR_TEST_EXIT();
}
