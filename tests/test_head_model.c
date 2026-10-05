/* test_head_model.c — the head by a bound through the engine (olmoe.c read_head, head_argmax; model.h
 * tr_session_set_greedy), on synthetic OLMoEs with a Q8_0 and a Q4_K head (synth_olmoe.h).
 *
 * Two loads of each file: one holds the head as its planes (the default), one as its original rows (TR_HEAD_BOUND=0,
 * today's road). Branches, each counted (zero fails the test, LESSONS #43):
 *   greedy   a greedy session on the planes gives the scan's token of the original rows' logits at every position
 *            of a prompt and 40 generated tokens; the profile counts the rows the bound computed;
 *   lazy     its logits, asked after the token, are the original rows' bit for bit (computed then, from the planes),
 *            and the token stays the same after them;
 *   full     a session on the planes that is not greedy: every logit at every position the original rows' bits;
 *   verify   a pass keeping 4 rows of logits (a speculative check): every row the original rows' bits;
 *   rewind   after a rewind the next token is again the scan's. */
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

enum { VOCAB = 1000, CTX = 96, PROMPT = 20, GEN = 40 };

static long n_greedy, n_lazy, n_full, n_verify, n_rewind;

static void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value != NULL ? value : "");
#else
    if (value != NULL) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static int32_t scan(const float *x) {
    int32_t best = 0;
    for (int32_t i = 1; i < VOCAB; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static void run_type(tr_type type, const char *argv0, tr_pool *pool) {
    const synth_params P = {2, 256, 4, 2, 256, 4, 2, VOCAB, CTX, type};
    char path[512], err[256];
    TR_CHECK(synth_write(&P, argv0, type == TR_TYPE_Q8_0 ? "test_head_q8_tmp.gguf" : "test_head_q4_tmp.gguf", path,
                         sizeof path) == 0);
    set_env("TR_HEAD_BOUND", NULL);
    tr_model *mp = tr_model_load(path, pool, err, sizeof err);
    set_env("TR_HEAD_BOUND", "0");
    tr_model *mo = tr_model_load(path, pool, err, sizeof err);
    set_env("TR_HEAD_BOUND", NULL);
    TR_CHECK(mp != NULL && mo != NULL);
    tr_session *g = mp ? tr_session_create(mp, CTX, 0, err, sizeof err) : NULL;  /* planes, greedy */
    tr_session *f = mp ? tr_session_create(mp, CTX, 0, err, sizeof err) : NULL;  /* planes, every logit */
    tr_session *o = mo ? tr_session_create(mo, CTX, 0, err, sizeof err) : NULL;  /* original rows */
    TR_CHECK(g != NULL && f != NULL && o != NULL);
    if (g == NULL || f == NULL || o == NULL) goto out;
    tr_session_set_greedy(g, 1);
    tr_session_set_greedy(o, 1); /* no planes: no effect */
    tr_session_prof(g)->enabled = 1;

    int32_t tok[PROMPT + GEN];
    uint32_t seed = 7u + (uint32_t)type;
    for (int i = 0; i < PROMPT; i++) tok[i] = (int32_t)((seed = seed * 1664525u + 1013904223u) >> 8) % VOCAB;
    /* the prompt one token a pass, then the generated tokens: a greedy token at every position */
    for (int i = 0; i < PROMPT + GEN - 1; i++) {
        TR_CHECK(tr_session_eval(g, tok + i, 1) == 0 && tr_session_eval(f, tok + i, 1) == 0 &&
                 tr_session_eval(o, tok + i, 1) == 0);
        const float *lo = tr_session_logits(o);
        const int32_t want = scan(lo);
        const int32_t got = tr_session_argmax(g, 0, VOCAB);
        TR_CHECK(got == want);
        n_greedy++;
        TR_CHECK(memcmp(tr_session_logits(f), lo, VOCAB * sizeof(float)) == 0);
        n_full++;
        if (i % 3 == 0) {
            TR_CHECK(memcmp(tr_session_logits(g), lo, VOCAB * sizeof(float)) == 0);
            TR_CHECK(tr_session_argmax(g, 0, VOCAB) == want);
            n_lazy++;
        }
        if (i + 1 >= PROMPT) tok[i + 1] = want;
    }
    const tr_prof *pr = tr_session_prof(g);
    uint64_t rows = 0, calls = 0;
    for (int ph = 0; ph < TR_PHASE_COUNT; ph++) {
        rows += pr->acc[ph][TR_PROF_LM_HEAD].rows;
        calls += pr->acc[ph][TR_PROF_LM_HEAD].calls;
    }
    printf("test_head_model: %s head: %llu rows computed over %llu head calls (%d rows a head)\n",
           type == TR_TYPE_Q8_0 ? "Q8_0" : "Q4_K", (unsigned long long)rows, (unsigned long long)calls, VOCAB);
    TR_CHECK(rows > 0);
    /* one head call a pass, one more where the logits were asked: a greedy token never computes them */
    TR_CHECK(calls == (uint64_t)(PROMPT + GEN - 1) + (uint64_t)((PROMPT + GEN - 1 + 2) / 3));

    /* a speculative check's pass: 4 rows of logits, from the planes, the original rows' bits */
    TR_CHECK(tr_session_rewind(g, PROMPT) == 0 && tr_session_rewind(o, PROMPT) == 0);
    TR_CHECK(tr_session_eval_rows(g, tok + PROMPT, 4, 4) == 0 && tr_session_eval_rows(o, tok + PROMPT, 4, 4) == 0);
    for (int b = 0; b < 4; b++) {
        const float *a = tr_session_logits_back(g, b), *w = tr_session_logits_back(o, b);
        TR_CHECK(a != NULL && w != NULL && memcmp(a, w, VOCAB * sizeof(float)) == 0);
        TR_CHECK(tr_session_argmax(g, b, VOCAB) == scan(w));
        n_verify++;
    }
    /* a rewind forgets the pass's token: the next one is the scan's again */
    TR_CHECK(tr_session_rewind(g, PROMPT + 1) == 0 && tr_session_rewind(o, PROMPT + 1) == 0);
    TR_CHECK(tr_session_eval(g, tok + PROMPT + 1, 1) == 0 && tr_session_eval(o, tok + PROMPT + 1, 1) == 0);
    TR_CHECK(tr_session_argmax(g, 0, VOCAB) == scan(tr_session_logits(o)));
    n_rewind++;
out:
    tr_session_free(g);
    tr_session_free(f);
    tr_session_free(o);
    tr_model_free(mp);
    tr_model_free(mo);
    remove(path);
}

int main(int argc, char **argv) {
    tr_pool *pool = tr_pool_create(4);
    TR_CHECK(pool != NULL);
    run_type(TR_TYPE_Q8_0, argc > 0 ? argv[0] : "", pool);
    run_type(TR_TYPE_Q4_K, argc > 0 ? argv[0] : "", pool);
    printf("test_head_model: greedy %ld, lazy %ld, full %ld, verify %ld, rewind %ld\n", n_greedy, n_lazy, n_full, n_verify,
           n_rewind);
    TR_CHECK(n_greedy > 0 && n_lazy > 0 && n_full > 0 && n_verify > 0 && n_rewind > 0);
    tr_pool_destroy(pool);
    TR_TEST_EXIT();
}
