/* test_prefill.c — tokens run many per forward pass give the same logits, bit for bit, as
 * the same tokens run one per pass: for every n_batch, every split across eval calls, every
 * thread count, f32 and Q8_0 weights, and after a rewind. The decode that follows reads
 * the cache those passes wrote, so identical logits there mean an identical cache.
 *
 * Runs on a synthetic OLMoE (synth_olmoe.h) with 8 experts, 3 used: short passes leave
 * experts without tokens, long ones give them uneven groups. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/format/gguf.h"
#include "../src/models/model.h"
#include "synth_olmoe.h"

enum { VOCAB = 48, N_PROMPT = 45, N_SEQ = 49, REWIND_TO = 7 };

static int32_t seq[N_SEQ];

/* logits after every position of seq, one token per pass and per call, one thread */
static int reference(const char *path, float *ref) {
    char err[256];
    tr_pool *pool = tr_pool_create(1);
    tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
    tr_session *s = model != NULL ? tr_session_create(model, 0, 1, err, sizeof err) : NULL;
    int rc = s != NULL ? 0 : -1;
    for (int64_t i = 0; i < N_SEQ && rc == 0; i++) {
        rc = tr_session_eval(s, seq + i, 1);
        if (rc == 0) memcpy(ref + i * VOCAB, tr_session_logits(s), VOCAB * sizeof(float));
    }
    if (rc != 0) fprintf(stderr, "reference failed: %s\n", s == NULL ? err : "eval");
    tr_session_free(s);
    tr_model_free(model);
    tr_pool_destroy(pool);
    return rc;
}

/* The prompt in eval calls of `call` tokens, passes of at most n_batch, then one token per
 * call to the end, then a rewind into the prompt and the rest as one call. Returns the
 * number of logits rows that differ from ref (-1 if a call fails). */
static int run_case(tr_model *model, int64_t n_batch, int64_t call, const float *ref) {
    tr_session *s = tr_session_create(model, 0, n_batch, NULL, 0);
    if (s == NULL) return -1;
    int bad = 0;
    for (int64_t i = 0; i < N_PROMPT && bad >= 0; i += call) {
        int64_t n = N_PROMPT - i < call ? N_PROMPT - i : call;
        if (tr_session_eval(s, seq + i, n) != 0) bad = -1;
        else bad += memcmp(tr_session_logits(s), ref + (i + n - 1) * VOCAB, VOCAB * sizeof(float)) != 0;
    }
    for (int64_t i = N_PROMPT; i < N_SEQ && bad >= 0; i++) {
        if (tr_session_eval(s, seq + i, 1) != 0) bad = -1;
        else bad += memcmp(tr_session_logits(s), ref + i * VOCAB, VOCAB * sizeof(float)) != 0;
    }
    if (bad >= 0) {
        if (tr_session_rewind(s, REWIND_TO) != 0 || tr_session_eval(s, seq + REWIND_TO, N_SEQ - REWIND_TO) != 0)
            bad = -1;
        else
            bad += memcmp(tr_session_logits(s), ref + (N_SEQ - 1) * VOCAB, VOCAB * sizeof(float)) != 0;
    }
    tr_session_free(s);
    return bad;
}

int main(int argc, char **argv) {
    /* 2 layers, n_embd 64, 4 heads (2 kv), n_ff 64, 8 experts (3 used), vocab 48, context 64 */
    static const synth_params params[2] = {
        {2, 64, 4, 2, 64, 8, 3, VOCAB, 64, TR_TYPE_F32},
        {2, 64, 4, 2, 64, 8, 3, VOCAB, 64, TR_TYPE_Q8_0},
    };
    static const int threads[3] = {1, 3, 8};
    static const int64_t batches[5] = {1, 2, 5, 16, 64};
    static const int64_t calls[3] = {1, 7, N_PROMPT};
    for (int i = 0; i < N_SEQ; i++) seq[i] = (int32_t)((i * 29 + (i * i) % 7 + 11) % VOCAB);

    static float ref[N_SEQ * VOCAB];
    char path[512], err[256];
    int cases = 0;
    for (int pi = 0; pi < 2; pi++) {
        const char *name = params[pi].type == TR_TYPE_Q8_0 ? "q8_0" : "f32";
        TR_CHECK(synth_write(&params[pi], argc > 0 ? argv[0] : "", "test_prefill_tmp.gguf", path, sizeof path) == 0);
        TR_CHECK(reference(path, ref) == 0);
        /* the logits move from position to position: equal rows below are not trivially equal */
        TR_CHECK(memcmp(ref, ref + VOCAB, VOCAB * sizeof(float)) != 0);

        for (int t = 0; t < 3; t++) {
            tr_pool *pool = tr_pool_create(threads[t]);
            tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
            TR_CHECK(model != NULL);
            for (int b = 0; b < 5 && model != NULL; b++) {
                for (int c = 0; c < 3; c++) {
                    int bad = run_case(model, batches[b], calls[c], ref);
                    TR_CHECK_EQ_INT(bad, 0);
                    if (bad != 0)
                        printf("  %s, %d threads, n_batch %lld, calls of %lld: %d rows differ\n", name, threads[t],
                               (long long)batches[b], (long long)calls[c], bad);
                    cases++;
                }
            }
            tr_model_free(model);
            tr_pool_destroy(pool);
        }
        remove(path);
    }
    printf("  %d cases (f32/q8_0, 1/3/8 threads, n_batch 1..64, calls of 1/7/%d, rewind): logits identical to one token per pass\n",
           cases, N_PROMPT);
    TR_TEST_EXIT();
}
