/* test_spec.c — speculation on the prompt changes the speed, never the tokens.
 *
 * Five things are checked on a synthetic OLMoE (synth_olmoe.h):
 *   1. tr_lookup_draft picks the continuation of the most recent earlier occurrence of the
 *      tail, longest n-gram first, and never reads past the context.
 *   2. tr_session_eval_rows keeps one row of logits per position, and each row is identical
 *      bit for bit to the logits that position gives when the tokens run one per pass.
 *   3. a rejected draft leaves no trace: after eval_rows on wrong tokens and a rewind, the
 *      next token gives the logits of a session that never saw them.
 *   4. greedy generation with any draft size, under either draft policy, emits the same
 *      tokens as with no speculation, and the logits it ends on are the logits of those
 *      tokens. Two vocabularies are used: the model's, where the drafts are almost always
 *      right, and a narrow one (the argmax is taken over the first 8 logits), where the
 *      tokens vary and drafts are often wrong, so both the accepting and the rejecting
 *      branch run.
 *   5. the adaptive policy's k_cur, on the model's own vocabulary (mostly right, occasionally
 *      not): left alone it still dips below n_draft at every real rejection, and starting from
 *      a shrunk state it climbs all the way back to n_draft — see test_adaptive_* below.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/format/gguf.h"
#include "../src/gen/greedy.h"
#include "../src/gen/lookup.h"
#include "../src/models/model.h"
#include "synth_olmoe.h"

enum { VOCAB = 48, NARROW = 8, N_PROMPT = 40, N_GEN = 40, N_REF = N_GEN + TR_GREEDY_DRAFT_MAX, CTX = 256 };

/* ---- 1. the drafter, no model ------------------------------------------------ */

static void test_lookup(void) {
    int32_t out[8];
    /* "1 2 3 4 5 1 2 3" -> the tail "1 2 3" occurred at 0 and was followed by 4 5 1 2 3 */
    static const int32_t a[] = {1, 2, 3, 4, 5, 1, 2, 3};
    TR_CHECK_EQ_INT((int)tr_lookup_draft(a, 8, out, 4), 4);
    TR_CHECK_EQ_INT(out[0], 4);
    TR_CHECK_EQ_INT(out[1], 5);
    TR_CHECK_EQ_INT(out[2], 1);
    TR_CHECK_EQ_INT(out[3], 2);
    /* max_draft caps it, and 0 asks for nothing */
    TR_CHECK_EQ_INT((int)tr_lookup_draft(a, 8, out, 2), 2);
    TR_CHECK_EQ_INT((int)tr_lookup_draft(a, 8, out, 0), 0);

    /* the most recent occurrence wins: tail "7 8" appears at 0 and at 4 */
    static const int32_t b[] = {7, 8, 100, 0, 7, 8, 200, 0, 7, 8};
    TR_CHECK_EQ_INT((int)tr_lookup_draft(b, 10, out, 3), 3);
    TR_CHECK_EQ_INT(out[0], 200);

    /* the longest n-gram wins over a shorter one that matches later */
    static const int32_t c[] = {1, 2, 3, 4, 9, 3, 4, 5, 6, 1, 2, 3, 4};
    TR_CHECK_EQ_INT((int)tr_lookup_draft(c, 13, out, 2), 2);
    TR_CHECK_EQ_INT(out[0], 9); /* from "1 2 3 4" at 0, not from "3 4" at 5 */

    /* nothing repeats: no draft. Also: a context too short to hold a tail and a match. */
    static const int32_t d[] = {1, 2, 3, 4, 5};
    TR_CHECK_EQ_INT((int)tr_lookup_draft(d, 5, out, 4), 0);
    TR_CHECK_EQ_INT((int)tr_lookup_draft(d, 1, out, 4), 0);
    TR_CHECK_EQ_INT((int)tr_lookup_draft(d, 2, out, 4), 0);

    /* one token repeated: "0 0 0 0" proposes 0, and never reads past the end */
    static const int32_t e[] = {0, 0, 0, 0};
    TR_CHECK_EQ_INT((int)tr_lookup_draft(e, 4, out, 8), 1);
    TR_CHECK_EQ_INT(out[0], 0);
}

/* ---- the model --------------------------------------------------------------- */

static int32_t argmax(const float *x, int64_t n) {
    int32_t best = 0;
    for (int64_t i = 1; i < n; i++)
        if (x[i] > x[best]) best = (int32_t)i;
    return best;
}

/* logits after every position of seq, one token per pass: the reference for everything. */
static int reference(tr_model *model, const int32_t *seq, int64_t n, float *ref) {
    tr_session *s = tr_session_create(model, CTX, 1, NULL, 0);
    if (s == NULL) return -1;
    int rc = 0;
    for (int64_t i = 0; i < n && rc == 0; i++) {
        rc = tr_session_eval(s, seq + i, 1);
        if (rc == 0) memcpy(ref + i * VOCAB, tr_session_logits(s), VOCAB * sizeof(float));
    }
    tr_session_free(s);
    return rc;
}

/* 2. every row of one pass over seq[0..n-1] equals the reference row of its own position. */
static int rows_match(tr_model *model, const int32_t *seq, int64_t n, int64_t n_logits, const float *ref) {
    tr_session *s = tr_session_create(model, CTX, 64, NULL, 0);
    if (s == NULL) return -1;
    int bad = 0;
    if (tr_session_eval_rows(s, seq, n, n_logits) != 0) bad = -1;
    for (int64_t j = 0; j < n_logits && bad >= 0; j++) {
        const float *row = tr_session_logits_back(s, j); /* back = j is position n - 1 - j */
        if (row == NULL) bad = -1;
        else bad += memcmp(row, ref + (n - 1 - j) * VOCAB, VOCAB * sizeof(float)) != 0;
    }
    if (bad >= 0) bad += tr_session_logits_back(s, n_logits) != NULL; /* nothing past the kept rows */
    tr_session_free(s);
    return bad;
}

/* 3. tokens the model would not have chosen, run and then rewound, leave no trace. */
static int reject_leaves_no_trace(tr_model *model, const int32_t *seq, int64_t n_prompt, const float *ref) {
    tr_session *s = tr_session_create(model, CTX, 64, NULL, 0);
    if (s == NULL) return -1;
    int bad = 0;
    if (tr_session_eval(s, seq, n_prompt) != 0) return -1;

    /* three tokens the model does not choose: its own pick, shifted */
    int32_t wrong[3];
    wrong[0] = (argmax(tr_session_logits(s), VOCAB) + 1) % VOCAB;
    wrong[1] = (wrong[0] + 7) % VOCAB;
    wrong[2] = (wrong[0] + 13) % VOCAB;
    if (tr_session_eval_rows(s, wrong, 3, 3) != 0) bad = -1;
    /* the first row is the model's own choice after wrong[0]: it must not be wrong[1],
     * otherwise this case would not be testing a rejection at all */
    else if (argmax(tr_session_logits_back(s, 2), VOCAB) == wrong[1]) bad = -1;
    else if (tr_session_rewind(s, n_prompt) != 0) bad = -1;
    else if (tr_session_logits_back(s, 0) != NULL) bad = -1; /* a rewind invalidates the rows */
    /* now the real token: the logits must be the ones of a session that never saw `wrong` */
    else if (tr_session_eval(s, seq + n_prompt, 1) != 0) bad = -1;
    else bad = memcmp(tr_session_logits(s), ref + n_prompt * VOCAB, VOCAB * sizeof(float)) != 0;
    tr_session_free(s);
    return bad;
}

/* 4. greedy generation, speculative when n_draft > 0. Writes the first N_GEN tokens to out,
 * the whole history to hist (it can overshoot by up to n_draft), the final logits to last. */
static int generate_with_batch(tr_model *model, const int32_t *prompt, int64_t vocab, int64_t n_draft,
                               tr_draft_policy policy, int64_t n_want, int64_t n_batch, int32_t *out,
                               int32_t *hist, int64_t *n_hist, float *last, tr_greedy *g_out) {
    tr_session *s = tr_session_create(model, CTX, n_batch, NULL, 0);
    if (s == NULL) return -1;
    memcpy(hist, prompt, N_PROMPT * sizeof(int32_t));
    int rc = tr_session_eval(s, prompt, N_PROMPT);
    tr_greedy g;
    if (rc == 0) rc = tr_greedy_init(&g, s, vocab, tr_session_n_ctx(s), n_draft, policy, hist, N_PROMPT);
    int64_t produced = 0;
    int32_t step[1 + TR_GREEDY_DRAFT_MAX];
    while (rc == 0 && produced < n_want) {
        int64_t got = tr_greedy_step(&g, step);
        if (got < 0) { rc = -1; break; }
        for (int64_t j = 0; j < got && produced < n_want; j++) out[produced++] = step[j];
    }
    if (rc == 0) {
        memcpy(last, tr_session_logits(s), VOCAB * sizeof(float));
        *n_hist = g.n_hist;
        *g_out = g;
    }
    tr_session_free(s);
    return rc;
}

static int generate_with(tr_model *model, const int32_t *prompt, int64_t vocab, int64_t n_draft,
                         tr_draft_policy policy, int64_t n_want, int32_t *out, int32_t *hist, int64_t *n_hist,
                         float *last, tr_greedy *g_out) {
    return generate_with_batch(model, prompt, vocab, n_draft, policy, n_want, 64, out, hist, n_hist, last, g_out);
}

/* 5. the k_cur rule, direct: a step drafts min(k_cur, n_draft, room); after the step, a full
 * accept (a == k) grows k_cur by one (capped at n_draft), anything less shrinks it to max(1, a),
 * and an empty draft (k == 0, nothing to propose) leaves k_cur untouched. Runs its own session
 * because what is asserted here is g.k_cur and g.n_drafted/g.n_steps, not the tokens. */
enum { N_ADAPT = 1500, ADAPT_CTX = 2048 };

static void test_adaptive_shrinks_on_reject(tr_model *model, const int32_t *prompt) {
    tr_session *s = tr_session_create(model, ADAPT_CTX, 64, NULL, 0);
    TR_CHECK(s != NULL);
    if (s == NULL) return;
    static int32_t hist[N_PROMPT + N_ADAPT + TR_GREEDY_DRAFT_MAX];
    memcpy(hist, prompt, N_PROMPT * sizeof(int32_t));
    tr_greedy g;
    int rc = tr_session_eval(s, prompt, N_PROMPT);
    if (rc == 0)
        rc = tr_greedy_init(&g, s, VOCAB, tr_session_n_ctx(s), TR_GREEDY_DRAFT_MAX, TR_DRAFT_ADAPTIVE, hist,
                            N_PROMPT);
    TR_CHECK(rc == 0);
    if (rc == 0) {
        int32_t step[1 + TR_GREEDY_DRAFT_MAX];
        int64_t produced = 0;
        int64_t min_k_cur = g.k_cur;
        while (produced < N_ADAPT) {
            int64_t got = tr_greedy_step(&g, step);
            TR_CHECK(got > 0);
            if (got < 0) break;
            produced += got;
            if (g.k_cur < min_k_cur) min_k_cur = g.k_cur;
        }
        /* this prompt's near-periodic tail is a good draft most of the time (the plain-greedy
         * comparison above accepts most of it) but not always: every real rejection must pull
         * k_cur below the n_draft it started at, so the average step drafts fewer tokens than
         * the fixed policy would (which always asks for n_draft, room permitting). */
        double mean = g.n_steps > 0 ? (double)g.n_drafted / (double)g.n_steps : 0.0;
        TR_CHECK(min_k_cur < TR_GREEDY_DRAFT_MAX);
        TR_CHECK(mean < (double)TR_GREEDY_DRAFT_MAX);
        if (min_k_cur >= TR_GREEDY_DRAFT_MAX || mean >= (double)TR_GREEDY_DRAFT_MAX)
            printf("  adaptive/reject: k_cur dipped to %lld, mean draft %.2f, %llu steps\n", (long long)min_k_cur,
                   mean, (unsigned long long)g.n_steps);
    }
    tr_session_free(s);
}

static void test_adaptive_grows_on_repeat(tr_model *model, const int32_t *prompt) {
    tr_session *s = tr_session_create(model, ADAPT_CTX, 64, NULL, 0);
    TR_CHECK(s != NULL);
    if (s == NULL) return;
    static int32_t hist[N_PROMPT + N_ADAPT + TR_GREEDY_DRAFT_MAX];
    memcpy(hist, prompt, N_PROMPT * sizeof(int32_t));
    tr_greedy g;
    int rc = tr_session_eval(s, prompt, N_PROMPT);
    if (rc == 0)
        rc = tr_greedy_init(&g, s, VOCAB, tr_session_n_ctx(s), TR_GREEDY_DRAFT_MAX, TR_DRAFT_ADAPTIVE, hist,
                            N_PROMPT);
    TR_CHECK(rc == 0);
    if (rc == 0) {
        g.k_cur = 1; /* as if a rejection had just shrunk it: the rule must recover from here */
        int32_t step[1 + TR_GREEDY_DRAFT_MAX];
        int64_t produced = 0;
        int64_t max_k_cur = g.k_cur;
        while (produced < N_ADAPT) {
            int64_t got = tr_greedy_step(&g, step);
            TR_CHECK(got > 0);
            if (got < 0) break;
            produced += got;
            if (g.k_cur > max_k_cur) max_k_cur = g.k_cur;
        }
        /* on the real vocabulary this prompt's near-periodic tail is right often enough that,
         * one accepted step at a time, k_cur must climb all the way back to n_draft at some
         * point in the run (whether or not a later rejection pulls it back down again). */
        TR_CHECK_EQ_INT((int)max_k_cur, TR_GREEDY_DRAFT_MAX);
        if ((int)max_k_cur != TR_GREEDY_DRAFT_MAX)
            printf("  adaptive/repeat: k_cur peaked at %lld (ended %lld), expected a peak of %d, %llu steps\n",
                   (long long)max_k_cur, (long long)g.k_cur, TR_GREEDY_DRAFT_MAX, (unsigned long long)g.n_steps);
    }
    tr_session_free(s);
}

/* The pause after a draft where nothing was accepted: that step must be followed by steps that
 * draft nothing at all (a wrong draft costs more than half a pass, docs/MEASUREMENTS.md). Checked on
 * every step of a real run, and the test fails if no such rejection ever happened, so it cannot
 * pass for the wrong reason (LEZIONI #43, #50, #55). */
static int64_t test_adaptive_pauses_after_a_wrong_draft(tr_model *model, const int32_t *prompt) {
    tr_session *s = tr_session_create(model, ADAPT_CTX, 64, NULL, 0);
    TR_CHECK(s != NULL);
    if (s == NULL) return 0;
    static int32_t hist[N_PROMPT + N_ADAPT + TR_GREEDY_DRAFT_MAX];
    memcpy(hist, prompt, N_PROMPT * sizeof(int32_t));
    tr_greedy g;
    int64_t all_wrong = 0;
    int rc = tr_session_eval(s, prompt, N_PROMPT);
    if (rc == 0)
        rc = tr_greedy_init(&g, s, VOCAB, tr_session_n_ctx(s), TR_GREEDY_DRAFT_MAX, TR_DRAFT_ADAPTIVE, hist,
                            N_PROMPT);
    TR_CHECK(rc == 0);
    if (rc == 0) {
        int32_t step[1 + TR_GREEDY_DRAFT_MAX];
        int64_t produced = 0, paused_after = 0;
        uint64_t prev_drafted = 0, prev_accepted = 0;
        int expect_pause = 0;
        while (produced < N_ADAPT) {
            int64_t got = tr_greedy_step(&g, step);
            TR_CHECK(got > 0);
            if (got < 0) break;
            produced += got;
            uint64_t k = g.n_drafted - prev_drafted, a = g.n_accepted - prev_accepted;
            prev_drafted = g.n_drafted;
            prev_accepted = g.n_accepted;
            if (expect_pause) {
                TR_CHECK_EQ_INT((int)k, 0);
                if (k == 0) paused_after++;
                expect_pause = 0;
            }
            if (k > 0 && a == 0) {
                all_wrong++;
                expect_pause = 1; /* the next step must not draft */
                TR_CHECK_EQ_INT((int)g.k_cur, 0);
                TR_CHECK(g.cool > 0);
            }
        }
        /* whether a given model and prompt ever produce a fully wrong draft is their business;
         * that at least one case in the suite does is checked once, in main. */
        if (all_wrong > 0)
            printf("  adaptive/pause: %lld drafts fully rejected, %lld followed by a step with no draft\n",
                   (long long)all_wrong, (long long)paused_after);
    }
    tr_session_free(s);
    return all_wrong;
}

/* The pause doubles only up to its cap: 1, 3, 7, 15, then 16 for good (greedy.h). Every draft
 * here is forced to fail: before each step the history the drafter reads is rewritten so that
 * the tail was once followed by a token the model will not choose (its real choice comes from a
 * shadow session fed the same tokens). The session never sees that history, so the tokens are
 * still the model's own; what is measured is how many steps pass between two drafts
 * (docs/LESSONS.md #60: the cap let 15 double to 31 before it applied). */
static void test_adaptive_pause_is_capped(tr_model *model, const int32_t *prompt) {
    enum { MAX_STEPS = 200, N_PAUSES = 6 };
    static const int64_t want[N_PAUSES] = {1, 3, 7, 15, 16, 16};
    tr_session *s = tr_session_create(model, ADAPT_CTX, 64, NULL, 0);
    tr_session *shadow = tr_session_create(model, ADAPT_CTX, 64, NULL, 0);
    TR_CHECK(s != NULL && shadow != NULL);
    static int32_t hist[N_PROMPT + MAX_STEPS + TR_GREEDY_DRAFT_MAX + 1];
    memcpy(hist, prompt, N_PROMPT * sizeof(int32_t));
    tr_greedy g;
    int rc = s != NULL && shadow != NULL ? 0 : -1;
    if (rc == 0) rc = tr_session_eval(s, prompt, N_PROMPT);
    if (rc == 0) rc = tr_session_eval(shadow, prompt, N_PROMPT);
    if (rc == 0)
        rc = tr_greedy_init(&g, s, VOCAB, tr_session_n_ctx(s), TR_GREEDY_DRAFT_MAX, TR_DRAFT_ADAPTIVE, hist,
                            N_PROMPT);
    TR_CHECK(rc == 0);

    int64_t pauses[N_PAUSES], n_pauses = 0, pause = -1; /* -1: no draft has failed yet */
    uint64_t prev_drafted = 0;
    for (int step_i = 0; rc == 0 && step_i < MAX_STEPS && n_pauses < N_PAUSES; step_i++) {
        int32_t next = g.next, step[1 + TR_GREEDY_DRAFT_MAX];
        if (tr_session_eval(shadow, &next, 1) != 0) { rc = -1; break; }
        int32_t choice = argmax(tr_session_logits(shadow), VOCAB);
        int32_t wrong = (choice + 1) % VOCAB, filler = (next + 1) % VOCAB;
        /* "F F F next wrong F F ... F" + next: the 4-gram tail "F F F next" occurred once, at 0 */
        for (int64_t i = 0; i < g.n_hist; i++) hist[i] = filler;
        hist[3] = next;
        hist[4] = wrong;

        int64_t got = tr_greedy_step(&g, step);
        TR_CHECK_EQ_INT((int)got, 1);              /* nothing drafted here is ever right */
        if (got != 1) { rc = -1; break; }
        TR_CHECK_EQ_INT(step[0], next);
        TR_CHECK_EQ_INT(g.next, choice);           /* and the tokens are still the model's */
        uint64_t k = g.n_drafted - prev_drafted;
        prev_drafted = g.n_drafted;
        if (k > 0) {
            if (pause >= 0) pauses[n_pauses++] = pause;
            pause = 0;
        } else if (pause >= 0) {
            pause++;
        }
    }
    TR_CHECK(rc == 0);
    TR_CHECK_EQ_INT((int)n_pauses, N_PAUSES);
    TR_CHECK_EQ_INT((int)g.n_accepted, 0);
    for (int64_t i = 0; i < n_pauses; i++) {
        TR_CHECK_EQ_INT((int)pauses[i], (int)want[i]);
        if (pauses[i] != want[i])
            printf("  adaptive/cap: pause %lld lasted %lld steps, expected %lld\n", (long long)i + 1,
                   (long long)pauses[i], (long long)want[i]);
    }
    tr_session_free(shadow);
    tr_session_free(s);
}

int main(int argc, char **argv) {
    int64_t pause_rule_seen = 0; /* drafts fully rejected across the suite: the pause rule needs one */

    test_lookup();

    /* 2 layers, n_embd 64, 4 heads (2 kv), n_ff 64, 8 experts (3 used), vocab 48 */
    static const synth_params params[2] = {
        {2, 64, 4, 2, 64, 8, 3, VOCAB, CTX, TR_TYPE_F32},
        {2, 64, 4, 2, 64, 8, 3, VOCAB, CTX, TR_TYPE_Q8_0},
    };
    static const int64_t drafts[5] = {1, 2, 3, 5, TR_GREEDY_DRAFT_MAX};
    static const int threads[2] = {1, 4};
    static const int64_t vocabs[2] = {VOCAB, NARROW};
    static const tr_draft_policy policies[2] = {TR_DRAFT_FIXED, TR_DRAFT_ADAPTIVE};

    static int32_t prompt[N_PROMPT];
    for (int i = 0; i < N_PROMPT; i++) prompt[i] = (int32_t)(((i % 13) * 3 + (i / 13) % 5) % VOCAB);

    static float ref[(N_PROMPT + N_REF) * VOCAB], last_plain[VOCAB], last_spec[VOCAB];
    static int32_t plain[N_REF], spec[N_GEN];
    static int32_t hist_plain[CTX], hist_spec[CTX], seq[N_PROMPT + N_REF];
    char path[512], err[256];
    uint64_t accepted_total = 0, rejected_total = 0;
    int cases = 0;

    for (int pi = 0; pi < 2; pi++) {
        const char *name = params[pi].type == TR_TYPE_Q8_0 ? "q8_0" : "f32";
        TR_CHECK(synth_write(&params[pi], argc > 0 ? argv[0] : "", "test_spec_tmp.gguf", path, sizeof path) == 0);

        for (int t = 0; t < 2; t++) {
            tr_pool *pool = tr_pool_create(threads[t]);
            tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
            TR_CHECK(model != NULL);
            if (model == NULL) { tr_pool_destroy(pool); continue; }

            for (int v = 0; v < 2; v++) {
                tr_greedy g;
                int64_t n_hist_plain = 0, n_hist_spec = 0;
                /* the plain greedy run is the reference for the tokens; policy does not matter
                 * when n_draft = 0 (nothing is ever ahead to draft) */
                TR_CHECK(generate_with(model, prompt, vocabs[v], 0, TR_DRAFT_FIXED, N_REF, plain, hist_plain,
                                       &n_hist_plain, last_plain, &g) == 0);
                TR_CHECK(n_hist_plain == N_PROMPT + N_REF);
                memcpy(seq, hist_plain, (size_t)(N_PROMPT + N_REF) * sizeof(int32_t));
                TR_CHECK(reference(model, seq, N_PROMPT + N_REF, ref) == 0);

                if (v == 0) {
                    /* 2 and 3 do not depend on the vocabulary: once per model is enough */
                    for (int64_t k = 1; k <= TR_LOGIT_ROWS_MAX; k *= 2) {
                        int bad = rows_match(model, seq, 60, k, ref);
                        TR_CHECK_EQ_INT(bad, 0);
                        if (bad != 0)
                            printf("  %s, %d threads, %lld rows: %d differ\n", name, threads[t], (long long)k, bad);
                    }
                    TR_CHECK_EQ_INT(reject_leaves_no_trace(model, seq, N_PROMPT, ref), 0);
                    test_adaptive_shrinks_on_reject(model, prompt);
                    test_adaptive_grows_on_repeat(model, prompt);
                    pause_rule_seen += test_adaptive_pauses_after_a_wrong_draft(model, prompt);
                    test_adaptive_pause_is_capped(model, prompt);
                }

                for (int pp = 0; pp < 2; pp++) {
                    for (int d = 0; d < 5; d++) {
                        TR_CHECK(generate_with(model, prompt, vocabs[v], drafts[d], policies[pp], N_GEN, spec,
                                               hist_spec, &n_hist_spec, last_spec, &g) == 0);
                        int same = memcmp(plain, spec, N_GEN * sizeof(int32_t)) == 0;
                        TR_CHECK(same);
                        if (!same)
                            printf("  %s, %d threads, vocab %lld, %s, draft %lld: tokens differ\n", name,
                                   threads[t], (long long)vocabs[v],
                                   policies[pp] == TR_DRAFT_ADAPTIVE ? "adaptive" : "fixed", (long long)drafts[d]);
                        /* the whole history is greedy, overshoot included, and the session ends
                         * on the logits of its last token: the cache survived every rejection */
                        TR_CHECK(n_hist_spec <= N_PROMPT + N_REF);
                        TR_CHECK(memcmp(hist_spec, seq, (size_t)n_hist_spec * sizeof(int32_t)) == 0);
                        TR_CHECK(memcmp(last_spec, ref + (n_hist_spec - 1) * VOCAB, VOCAB * sizeof(float)) == 0);
                        TR_CHECK(g.n_drafted > 0);
                        accepted_total += g.n_accepted;
                        rejected_total += g.n_drafted - g.n_accepted;
                        cases++;
                    }
                }

                /* a small -b leaves room for fewer rows: the draft is lowered, not refused */
                TR_CHECK(generate_with_batch(model, prompt, vocabs[v], TR_GREEDY_DRAFT_MAX, TR_DRAFT_ADAPTIVE,
                                             N_GEN, 4, spec, hist_spec, &n_hist_spec, last_spec, &g) == 0);
                TR_CHECK_EQ_INT((int)g.n_draft, 3);
                TR_CHECK(memcmp(plain, spec, N_GEN * sizeof(int32_t)) == 0);
                cases++;
            }
            tr_model_free(model);
            tr_pool_destroy(pool);
        }
        remove(path);
    }
    /* a run where nothing is ever accepted, or nothing ever rejected, would prove half of it */
    TR_CHECK(accepted_total > 0);
    TR_CHECK(rejected_total > 0);
    /* and the pause rule is only proved if some case really did produce a draft with nothing
     * accepted: without one, test_adaptive_pauses_after_a_wrong_draft checked nothing */
    TR_CHECK(pause_rule_seen > 0);
    printf("  %d cases (f32/q8_0, 1/4 threads, vocab %d/%d, fixed/adaptive, draft 1..%d): same tokens as plain "
           "greedy, %llu drafted tokens accepted, %llu rejected\n",
           cases, VOCAB, NARROW, TR_GREEDY_DRAFT_MAX, (unsigned long long)accepted_total,
           (unsigned long long)rejected_total);
    TR_TEST_EXIT();
}
