/* test_phase.c — threads per phase (src/models/model.h): a long pass runs on the whole pool, a
 * short one on fewer threads, and no logit moves by a bit whatever the width.
 *   - the widths a pool is measured on and the choice among them, on made-up timings: the
 *     choice must not depend on the clock of the machine that runs the test;
 *   - the schedule on a real session: prompt wide, the first one-token passes in turn on every
 *     width, then the chosen one, also for the short passes of several rows, and the same
 *     measurement again after TR_DECODE_TUNE_AGAIN passes on the choice;
 *   - a forced width, clamped to the pool, and back to measuring;
 *   - every logit of every pass identical to one token per pass on one thread, f32 and Q8_0.
 *
 * Synthetic OLMoE big enough for a pool of 8 to split every matmul (as tests/test_hot.c). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/format/gguf.h"
#include "../src/models/model.h"
#include "../src/models/model_internal.h"
#include "synth_olmoe.h"

enum { VOCAB = 64, POOL = 8, N_PROMPT = 20, N_ONE = 12, N_ROW_PASSES = 4 };
/* short passes of several rows: two, the most that still counts as short, one more, many */
static const int64_t row_pass[N_ROW_PASSES] = {2, TR_DECODE_ROWS, TR_DECODE_ROWS + 1, 9};
enum { N_SEQ = N_PROMPT + N_ONE + 2 + TR_DECODE_ROWS + TR_DECODE_ROWS + 1 + 9 };

static int32_t seq[N_SEQ];

static void test_widths(void) {
    static const struct { int pool, n, w[TR_DECODE_TUNE_WIDTHS]; } cases[] = {
        {16, 3, {16, 8, 4}}, {8, 3, {8, 4, 2}}, {12, 3, {12, 6, 3}}, {6, 3, {6, 3, 1}}, {5, 3, {5, 2, 1}},
        {4, 3, {4, 2, 1}},   {3, 2, {3, 1, 0}}, {2, 2, {2, 1, 0}},   {1, 1, {1, 0, 0}}, {0, 1, {1, 0, 0}},
        {-4, 1, {1, 0, 0}},  {64, 3, {64, 32, 16}},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int w[TR_DECODE_TUNE_WIDTHS] = {0, 0, 0};
        int n = tr_decode_tune_widths(cases[i].pool, w);
        TR_CHECK_EQ_INT(n, cases[i].n);
        for (int j = 0; j < cases[i].n && j < n; j++) TR_CHECK_EQ_INT(w[j], cases[i].w[j]);
    }
}

static void test_pick(void) {
    static const struct { int n; double sec[TR_DECODE_TUNE_WIDTHS]; int want; } cases[] = {
        {3, {1.00, 1.00, 1.00}, 0},    /* flat: the widest */
        {3, {1.10, 1.00, 0.995}, 1},   /* 16 loses 10%, 8 and 4 are one: the wider of the two */
        {3, {1.00, 1.10, 1.30}, 0},    /* bound by compute: every thread counts */
        {3, {1.05, 1.04, 1.00}, 2},    /* only the narrowest is within the margin */
        {3, {1.005, 1.20, 1.00}, 0},   /* a noisy middle does not hide a wide width that is as fast */
        {3, {0.0325, 0.0296, 0.0291}, 2},  /* the reference machine at 512 tokens: 4 is 1.8% ahead of 8... */
        {3, {0.0416, 0.0406, 0.0421}, 1},  /* ...and at 2048 it is 8, with 16 only 2.5% behind: not a tie */
        {2, {1.20, 1.00, 0.0}, 1},
        {2, {1.00, 1.20, 0.0}, 0},
        {1, {1.00, 0.0, 0.0}, 0},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
        TR_CHECK_EQ_INT(tr_decode_tune_pick(cases[i].sec, cases[i].n), cases[i].want);
}

/* logits after every position of seq, one token per pass, one thread */
static int reference(const char *path, float *ref) {
    char err[256];
    tr_pool *pool = tr_pool_create(1);
    tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
    tr_session *s = model != NULL ? tr_session_create(model, 0, 1, err, sizeof err) : NULL;
    int rc = s != NULL ? 0 : -1;
    for (int64_t i = 0; i < N_SEQ && rc == 0; i++) {
        rc = tr_session_eval(s, seq + i, 1);
        if (rc == 0) memcpy(ref + i * VOCAB, tr_session_logits(s), VOCAB * sizeof(float));
        if (rc == 0) TR_CHECK_EQ_INT(tr_session_last_threads(s), 1);
    }
    if (s != NULL) TR_CHECK_EQ_INT(tr_session_decode_threads(s), 1); /* one thread: nothing to measure */
    if (rc != 0) fprintf(stderr, "reference failed: %s\n", s == NULL ? err : "eval");
    tr_session_free(s);
    tr_model_free(model);
    tr_pool_destroy(pool);
    return rc;
}

/* The whole sequence on a pool of POOL threads: prompt in one pass, N_ONE one-token passes, then
 * the passes of row_pass rows. `forced` > 0 is the width asked for, 0 lets the session measure.
 * Checks the threads of every pass and every row of logits against ref. */
static void run_case(tr_model *model, tr_pool *pool, int forced, const float *ref) {
    int widths[TR_DECODE_TUNE_WIDTHS];
    int n_widths = tr_decode_tune_widths(POOL, widths);
    int want_forced = forced > POOL ? POOL : forced;
    tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
    TR_CHECK(s != NULL);
    if (s == NULL) return;
    TR_CHECK_EQ_INT(tr_session_last_threads(s), 0);
    TR_CHECK_EQ_INT(tr_session_decode_threads(s), want_forced);

    int bad = 0;
    int64_t pos = 0;
    TR_CHECK(tr_session_eval(s, seq, N_PROMPT) == 0);
    TR_CHECK_EQ_INT(tr_session_last_threads(s), POOL);
    pos += N_PROMPT;
    bad += memcmp(tr_session_logits(s), ref + (pos - 1) * VOCAB, VOCAB * sizeof(float)) != 0;

    for (int k = 0; k < N_ONE; k++, pos++) {
        TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0);
        bad += memcmp(tr_session_logits(s), ref + pos * VOCAB, VOCAB * sizeof(float)) != 0;
        int measuring = forced <= 0 && k < n_widths * TR_DECODE_TUNE_ROUNDS;
        int want = forced > 0 ? want_forced : measuring ? widths[k % n_widths] : tr_session_decode_threads(s);
        TR_CHECK_EQ_INT(tr_session_last_threads(s), want);
        /* the session says 0 until the last measured pass is in, then one of the widths, for good */
        if (forced <= 0 && k + 1 < n_widths * TR_DECODE_TUNE_ROUNDS) TR_CHECK_EQ_INT(tr_session_decode_threads(s), 0);
    }
    int chosen = tr_session_decode_threads(s);
    int is_a_width = 0;
    for (int j = 0; j < n_widths; j++) is_a_width |= chosen == widths[j];
    if (forced > 0) TR_CHECK_EQ_INT(chosen, want_forced);
    else TR_CHECK(is_a_width);

    for (int p = 0; p < N_ROW_PASSES; p++) {
        int64_t n = row_pass[p];
        TR_CHECK(tr_session_eval_rows(s, seq + pos, n, n) == 0);
        pos += n;
        TR_CHECK_EQ_INT(tr_session_last_threads(s), n <= TR_DECODE_ROWS ? chosen : POOL);
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), chosen);
        for (int64_t j = 0; j < n; j++)
            bad += memcmp(tr_session_logits_back(s, j), ref + (pos - 1 - j) * VOCAB, VOCAB * sizeof(float)) != 0;
    }
    TR_CHECK_EQ_INT(pos, N_SEQ);

    /* After TR_DECODE_TUNE_AGAIN one-token passes on its choice the session measures again, width
     * by width, and goes on answering with the old choice until the new one is in. The same
     * position over and over (rewind), so the context of the test stays small. */
    if (forced <= 0) {
        int n_probes = n_widths * TR_DECODE_TUNE_ROUNDS;
        int first_probe = TR_DECODE_TUNE_AGAIN - (N_ONE - n_probes);
        for (int k = 0; k < first_probe + n_probes + 3; k++) {
            TR_CHECK(tr_session_rewind(s, N_PROMPT) == 0 && tr_session_eval(s, seq + N_PROMPT, 1) == 0);
            bad += memcmp(tr_session_logits(s), ref + N_PROMPT * VOCAB, VOCAB * sizeof(float)) != 0;
            int probe = k - first_probe;
            if (probe < 0) TR_CHECK_EQ_INT(tr_session_last_threads(s), chosen);
            else if (probe < n_probes) TR_CHECK_EQ_INT(tr_session_last_threads(s), widths[probe % n_widths]);
            else TR_CHECK_EQ_INT(tr_session_last_threads(s), tr_session_decode_threads(s));
            if (probe + 1 < n_probes) TR_CHECK_EQ_INT(tr_session_decode_threads(s), chosen);
        }
        int again = tr_session_decode_threads(s);
        is_a_width = 0;
        for (int j = 0; j < n_widths; j++) is_a_width |= again == widths[j];
        TR_CHECK(is_a_width);
    }
    TR_CHECK_EQ_INT(bad, 0);
    if (bad != 0) printf("  forced %d: %d rows of logits differ from one thread\n", forced, bad);
    /* a pass leaves the pool whole for whoever uses it next */
    TR_CHECK_EQ_INT(tr_pool_active(pool), POOL);
    tr_session_free(s);
}

/* A short pass of several rows before the session has finished measuring: the whole pool. */
static void test_rows_before_measured(tr_model *model, const float *ref) {
    tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
    TR_CHECK(s != NULL);
    if (s == NULL) return;
    TR_CHECK(tr_session_eval(s, seq, N_PROMPT) == 0);
    TR_CHECK(tr_session_eval_rows(s, seq + N_PROMPT, 3, 3) == 0);
    TR_CHECK_EQ_INT(tr_session_last_threads(s), POOL);
    TR_CHECK_EQ_INT(tr_session_decode_threads(s), 0);
    TR_CHECK(memcmp(tr_session_logits(s), ref + (N_PROMPT + 2) * VOCAB, VOCAB * sizeof(float)) == 0);
    /* a failed eval measures nothing and changes nothing */
    int32_t out_of_range = VOCAB;
    TR_CHECK(tr_session_eval(s, &out_of_range, 1) != 0);
    TR_CHECK_EQ_INT(tr_session_decode_threads(s), 0);
    tr_session_free(s);
}

int main(int argc, char **argv) {
    /* 2 layers, n_embd 256, 4 heads (2 kv), n_ff 512, 8 experts (2 used), vocab 64, context 64 */
    static const synth_params params[2] = {
        {2, 256, 4, 2, 512, 8, 2, VOCAB, 64, TR_TYPE_F32},
        {2, 256, 4, 2, 512, 8, 2, VOCAB, 64, TR_TYPE_Q8_0},
    };
    static const int forced[] = {0, 1, 2, 3, 5, POOL, 99};
    for (int i = 0; i < N_SEQ; i++) seq[i] = (int32_t)((i * 29 + (i * i) % 7 + 11) % VOCAB);

    test_widths();
    test_pick();

    static float ref[N_SEQ * VOCAB];
    char path[512], err[256];
    for (int pi = 0; pi < 2; pi++) {
        TR_CHECK(synth_write(&params[pi], argc > 0 ? argv[0] : "", "test_phase_tmp.gguf", path, sizeof path) == 0);
        TR_CHECK(reference(path, ref) == 0);
        TR_CHECK(memcmp(ref, ref + VOCAB, VOCAB * sizeof(float)) != 0);

        tr_pool *pool = tr_pool_create(POOL);
        tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
        TR_CHECK(model != NULL);
        if (model != NULL) {
            for (size_t f = 0; f < sizeof forced / sizeof forced[0]; f++) {
                tr_model_set_decode_threads(model, forced[f]);
                run_case(model, pool, forced[f], ref);
            }
            tr_model_set_decode_threads(model, -1); /* back to measuring */
            run_case(model, pool, 0, ref);
            test_rows_before_measured(model, ref);
        }
        tr_model_free(model);
        tr_pool_destroy(pool);

        /* no pool at all: one thread, nothing to measure, same logits */
        model = tr_model_load(path, NULL, err, sizeof err);
        tr_session *s = model != NULL ? tr_session_create(model, 0, 0, NULL, 0) : NULL;
        TR_CHECK(s != NULL);
        if (s != NULL) {
            TR_CHECK(tr_session_eval(s, seq, N_PROMPT) == 0 && tr_session_eval(s, seq + N_PROMPT, 1) == 0);
            TR_CHECK_EQ_INT(tr_session_last_threads(s), 1);
            TR_CHECK_EQ_INT(tr_session_decode_threads(s), 1);
            TR_CHECK(memcmp(tr_session_logits(s), ref + N_PROMPT * VOCAB, VOCAB * sizeof(float)) == 0);
        }
        tr_session_free(s);
        tr_model_free(model);
        remove(path);
    }
    printf("  f32/q8_0, pool of %d: prompt wide, one-token passes measured on %d widths then kept, "
           "forced 1..%d and beyond: every logit identical to one thread\n", POOL, TR_DECODE_TUNE_WIDTHS, POOL);
    TR_TEST_EXIT();
}
