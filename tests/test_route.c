/* test_route.c — the routing trace (src/models/model.h, docs/MEASUREMENTS.md domande 13-15): does the
 * next layer's router already know which experts it will pick, before or after the current
 * layer's own experts run?
 *
 * Each test says which branch it exercises and counts that the branch was actually taken
 * (CLAUDE.md, LEZIONI #43 #50 #54 #78):
 *   pred_out   synth_zero_attn_out: attention never moves x, so layer L+1's router sees exactly
 *              layer L's output; pred_out's top n_used must equal chosen[L+1] every time. A
 *              negative control on the plain model must differ at least once, or the knob above
 *              proves nothing.
 *   pred_in    synth_router_from_next: model B's layer L router equals model A's layer L+1
 *              router (12 tensors apart); A's pred_in[t][0] must equal B's chosen[t][0].
 *   structure  ids in range and distinct within a row, chosen increasing, last layer's
 *              predictions all 0xFFFF, expert_bytes/layer_bytes positive and consistent.
 *   shape      a prompt of 20 tokens then one-token passes gives the same trace, byte for byte,
 *              as one token per pass -- with a pool of 8 and with no pool.
 *   on_vs_off  turning the trace on moves no logit; a max_tokens below the run length stops
 *              recording there but never the logits.
 *   margins    (trace version 2) the id of every token; the router's probability of the last
 *              expert chosen and of the best one left out, exact at layer 0 against the same
 *              weights with one more expert per token (left out by 2 == last taken by 3).
 *   mask       tr_model_set_expert_mask: a masked expert is never chosen, layer 1 by its own row
 *              and not layer 0's; the run really wanted the masked experts and the output moved;
 *              refused when a layer keeps too few (the mask in force stays); cleared and empty
 *              masks give the model's own logits, byte for byte.
 *
 * Seen red: tools/mutate_route.sh.
 * Synthetic OLMoE as tests/test_phase.c (tests/synth_olmoe.h), 2 layers. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/format/gguf.h"
#include "../src/models/model.h"
#include "synth_olmoe.h"

enum { VOCAB = 64, N_PROMPT = 20, N_GEN = 10, N_TOTAL = N_PROMPT + N_GEN };
static int32_t seq[N_TOTAL];

/* 2 layers, n_embd 256, 4 heads (2 kv), n_ff 512, 8 experts (2 used), vocab 64, context 64: big
 * enough for a pool of 8 to split every matmul (as tests/test_phase.c). */
static const synth_params ROUTE_PARAMS = {2, 256, 4, 2, 512, 8, 2, VOCAB, 64, TR_TYPE_F32};

/* True if every one of a[0..n) is found somewhere in b[0..n): with both sides exactly n long and
 * internally distinct (route_token's and top_experts' own taken[] guarantee it), containment one
 * way already proves the two sets are equal. */
static int set_eq_u16(const uint16_t *a, const uint16_t *b, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int found = 0;
        for (int64_t j = 0; j < n; j++)
            if (a[i] == b[j]) { found = 1; break; }
        if (!found) return 0;
    }
    return 1;
}

static int trace_eq(const tr_route_trace *a, const tr_route_trace *b) {
    if (a->n_tokens != b->n_tokens || a->n_layers != b->n_layers || a->n_used != b->n_used ||
        a->n_pred != b->n_pred)
        return 0;
    size_t n_chosen = (size_t)(a->n_tokens * a->n_layers * a->n_used);
    size_t n_pred = (size_t)(a->n_tokens * a->n_layers * a->n_pred);
    return memcmp(a->chosen, b->chosen, n_chosen * sizeof(uint16_t)) == 0 &&
           memcmp(a->pred_in, b->pred_in, n_pred * sizeof(uint16_t)) == 0 &&
           memcmp(a->pred_out, b->pred_out, n_pred * sizeof(uint16_t)) == 0 &&
           memcmp(a->tokens, b->tokens, (size_t)a->n_tokens * sizeof(int32_t)) == 0 &&
           memcmp(a->margins, b->margins, (size_t)(a->n_tokens * a->n_layers * 2) * sizeof(float)) == 0;
}

/* pred_out exact on synth_zero_attn_out, and the negative control on the plain model. */
static void test_pred_out(const char *argv0) {
    char path_n[512], path_z[512], err[256];
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_normal.gguf", path_n, sizeof path_n) == 0);
    synth_zero_attn_out = 1;
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_zero.gguf", path_z, sizeof path_z) == 0);
    synth_zero_attn_out = 0;

    tr_model *mn = tr_model_load(path_n, NULL, err, sizeof err);
    tr_model *mz = tr_model_load(path_z, NULL, err, sizeof err);
    TR_CHECK(mn != NULL && mz != NULL);
    if (mn != NULL && mz != NULL) {
        tr_session *sn = tr_session_create(mn, 0, 0, NULL, 0);
        tr_session *sz = tr_session_create(mz, 0, 0, NULL, 0);
        TR_CHECK(sn != NULL && sz != NULL);
        if (sn != NULL && sz != NULL) {
            TR_CHECK(tr_session_route_trace_begin(sn, N_TOTAL) == 0);
            TR_CHECK(tr_session_route_trace_begin(sz, N_TOTAL) == 0);
            TR_CHECK(tr_session_eval(sn, seq, N_TOTAL) == 0);
            TR_CHECK(tr_session_eval(sz, seq, N_TOTAL) == 0);
            const tr_route_trace *trn = tr_session_route_trace(sn);
            const tr_route_trace *trz = tr_session_route_trace(sz);
            TR_CHECK(trn != NULL && trz != NULL);
            if (trn != NULL && trz != NULL) {
                int64_t exact = 0, checked = 0, differ = 0;
                for (int64_t t = 0; t < trz->n_tokens; t++)
                    for (int64_t L = 0; L + 1 < trz->n_layers; L++) {
                        const uint16_t *pout = trz->pred_out + (t * trz->n_layers + L) * trz->n_pred;
                        const uint16_t *ch = trz->chosen + (t * trz->n_layers + (L + 1)) * trz->n_used;
                        if (set_eq_u16(ch, pout, trz->n_used)) exact++;
                        checked++;
                    }
                TR_CHECK_EQ_INT(exact, checked); /* every comparison matches when attention can't move x */
                TR_CHECK(checked > 0);

                for (int64_t t = 0; t < trn->n_tokens; t++)
                    for (int64_t L = 0; L + 1 < trn->n_layers; L++) {
                        const uint16_t *pout = trn->pred_out + (t * trn->n_layers + L) * trn->n_pred;
                        const uint16_t *ch = trn->chosen + (t * trn->n_layers + (L + 1)) * trn->n_used;
                        if (!set_eq_u16(ch, pout, trn->n_used)) differ++;
                    }
                TR_CHECK(differ > 0); /* negative control: the plain model must differ somewhere */
            }
        }
        tr_session_free(sn);
        tr_session_free(sz);
    }
    tr_model_free(mn);
    tr_model_free(mz);
    remove(path_n);
    remove(path_z);
}

/* pred_in exact against a model whose own router is the normal model's next layer, 12 tensors
 * ahead (synth_router_from_next). */
static void test_pred_in(const char *argv0) {
    char path_a[512], path_b[512], err[256];
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_a.gguf", path_a, sizeof path_a) == 0);
    synth_router_from_next = 1;
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_b.gguf", path_b, sizeof path_b) == 0);
    synth_router_from_next = 0;

    tr_model *ma = tr_model_load(path_a, NULL, err, sizeof err);
    tr_model *mb = tr_model_load(path_b, NULL, err, sizeof err);
    TR_CHECK(ma != NULL && mb != NULL);
    if (ma != NULL && mb != NULL) {
        tr_session *sa = tr_session_create(ma, 0, 0, NULL, 0);
        tr_session *sb = tr_session_create(mb, 0, 0, NULL, 0);
        TR_CHECK(sa != NULL && sb != NULL);
        if (sa != NULL && sb != NULL) {
            TR_CHECK(tr_session_route_trace_begin(sa, N_TOTAL) == 0);
            TR_CHECK(tr_session_route_trace_begin(sb, N_TOTAL) == 0);
            TR_CHECK(tr_session_eval(sa, seq, N_TOTAL) == 0);
            TR_CHECK(tr_session_eval(sb, seq, N_TOTAL) == 0);
            const tr_route_trace *tra = tr_session_route_trace(sa);
            const tr_route_trace *trb = tr_session_route_trace(sb);
            TR_CHECK(tra != NULL && trb != NULL);
            if (tra != NULL && trb != NULL) {
                int64_t matches = 0, checked = 0;
                for (int64_t t = 0; t < tra->n_tokens; t++) {
                    const uint16_t *pin = tra->pred_in + (t * tra->n_layers + 0) * tra->n_pred;
                    const uint16_t *ch0 = trb->chosen + (t * trb->n_layers + 0) * trb->n_used;
                    if (set_eq_u16(ch0, pin, trb->n_used)) matches++;
                    checked++;
                }
                TR_CHECK_EQ_INT(matches, checked);
                TR_CHECK(checked > 0);
            }
        }
        tr_session_free(sa);
        tr_session_free(sb);
    }
    tr_model_free(ma);
    tr_model_free(mb);
    remove(path_a);
    remove(path_b);
}

/* Structural invariants of a trace on a plain model. */
static void test_structure(const char *argv0) {
    char path[512], err[256];
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_struct.gguf", path, sizeof path) == 0);
    tr_model *m = tr_model_load(path, NULL, err, sizeof err);
    TR_CHECK(m != NULL);
    if (m != NULL) {
        tr_session *s = tr_session_create(m, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        if (s != NULL) {
            TR_CHECK(tr_session_route_trace_begin(s, N_TOTAL) == 0);
            TR_CHECK(tr_session_eval(s, seq, N_TOTAL) == 0);
            const tr_route_trace *tr = tr_session_route_trace(s);
            TR_CHECK(tr != NULL);
            if (tr != NULL) {
                TR_CHECK(tr->expert_bytes > 0);
                TR_CHECK(tr->layer_bytes > 0);
                TR_CHECK(tr->expert_bytes * tr->n_expert < tr->layer_bytes);

                int64_t checked_chosen = 0, checked_pred = 0, checked_sentinel = 0;
                for (int64_t t = 0; t < tr->n_tokens; t++) {
                    for (int64_t L = 0; L < tr->n_layers; L++) {
                        const uint16_t *ch = tr->chosen + (t * tr->n_layers + L) * tr->n_used;
                        for (int64_t i = 0; i < tr->n_used; i++) {
                            TR_CHECK(ch[i] < tr->n_expert);
                            if (i > 0) TR_CHECK(ch[i] > ch[i - 1]); /* increasing, hence distinct */
                            checked_chosen++;
                        }

                        const uint16_t *pin = tr->pred_in + (t * tr->n_layers + L) * tr->n_pred;
                        const uint16_t *pout = tr->pred_out + (t * tr->n_layers + L) * tr->n_pred;
                        if (L + 1 == tr->n_layers) {
                            for (int64_t k = 0; k < tr->n_pred; k++) {
                                TR_CHECK_EQ_INT(pin[k], 0xFFFF);
                                TR_CHECK_EQ_INT(pout[k], 0xFFFF);
                                checked_sentinel++;
                            }
                        } else {
                            for (int64_t k = 0; k < tr->n_pred; k++) {
                                TR_CHECK(pin[k] < tr->n_expert);
                                TR_CHECK(pout[k] < tr->n_expert);
                                for (int64_t k2 = 0; k2 < k; k2++) {
                                    TR_CHECK(pin[k2] != pin[k]);
                                    TR_CHECK(pout[k2] != pout[k]);
                                }
                                checked_pred++;
                            }
                        }
                    }
                }
                TR_CHECK(checked_chosen > 0);
                TR_CHECK(checked_pred > 0);
                TR_CHECK(checked_sentinel > 0);
            }
        }
        tr_session_free(s);
    }
    tr_model_free(m);
    remove(path);
}

/* A prompt of 20 tokens then one-token passes gives the same trace as one token per pass,
 * with a pool of 8 and with no pool. */
static void test_shape_of_passes(const char *argv0) {
    char path[512], err[256];
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_shape.gguf", path, sizeof path) == 0);

    int64_t checked_pools = 0;
    for (int use_pool = 0; use_pool < 2; use_pool++) {
        tr_pool *pool = use_pool ? tr_pool_create(8) : NULL;
        TR_CHECK(!use_pool || pool != NULL);
        tr_model *m = tr_model_load(path, pool, err, sizeof err);
        TR_CHECK(m != NULL);
        if (m != NULL) {
            tr_session *s_batched = tr_session_create(m, 0, 0, NULL, 0);
            tr_session *s_single = tr_session_create(m, 0, 0, NULL, 0);
            TR_CHECK(s_batched != NULL && s_single != NULL);
            if (s_batched != NULL && s_single != NULL) {
                TR_CHECK(tr_session_route_trace_begin(s_batched, N_TOTAL) == 0);
                TR_CHECK(tr_session_route_trace_begin(s_single, N_TOTAL) == 0);

                TR_CHECK(tr_session_eval(s_batched, seq, N_PROMPT) == 0);
                for (int64_t i = N_PROMPT; i < N_TOTAL; i++)
                    TR_CHECK(tr_session_eval(s_batched, seq + i, 1) == 0);
                for (int64_t i = 0; i < N_TOTAL; i++)
                    TR_CHECK(tr_session_eval(s_single, seq + i, 1) == 0);

                const tr_route_trace *tb = tr_session_route_trace(s_batched);
                const tr_route_trace *ts = tr_session_route_trace(s_single);
                TR_CHECK(tb != NULL && ts != NULL);
                if (tb != NULL && ts != NULL) {
                    TR_CHECK(trace_eq(tb, ts));
                    checked_pools++;
                }
            }
            tr_session_free(s_batched);
            tr_session_free(s_single);
        }
        tr_model_free(m);
        tr_pool_destroy(pool);
    }
    TR_CHECK_EQ_INT(checked_pools, 2);
    remove(path);
}

/* Turning the trace on moves no logit; a max_tokens below the run length stops recording there. */
static void test_on_vs_off(const char *argv0) {
    char path[512], err[256];
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_onoff.gguf", path, sizeof path) == 0);
    tr_model *m = tr_model_load(path, NULL, err, sizeof err);
    TR_CHECK(m != NULL);
    if (m != NULL) {
        tr_session *soff = tr_session_create(m, 0, 0, NULL, 0);
        tr_session *son = tr_session_create(m, 0, 0, NULL, 0);
        TR_CHECK(soff != NULL && son != NULL);
        if (soff != NULL && son != NULL) {
            const int64_t max_tokens = 5; /* smaller than N_TOTAL */
            TR_CHECK(tr_session_route_trace_begin(son, max_tokens) == 0);
            int64_t identical = 0, checked = 0;
            for (int64_t i = 0; i < N_TOTAL; i++) {
                TR_CHECK(tr_session_eval(soff, seq + i, 1) == 0);
                TR_CHECK(tr_session_eval(son, seq + i, 1) == 0);
                if (memcmp(tr_session_logits(soff), tr_session_logits(son), VOCAB * sizeof(float)) == 0)
                    identical++;
                checked++;
            }
            TR_CHECK_EQ_INT(identical, checked);
            TR_CHECK(checked > 0);
            const tr_route_trace *tr = tr_session_route_trace(son);
            TR_CHECK(tr != NULL);
            if (tr != NULL) TR_CHECK_EQ_INT(tr->n_tokens, max_tokens);
        }
        tr_session_free(soff);
        tr_session_free(son);
    }
    tr_model_free(m);
    remove(path);
}

/* One traced run of the whole sequence in one pass on a model from `path` (NULL pool); the last
 * token's logits into `logits` when not NULL. The caller frees the session (the trace lives in
 * it) and the model. Returns the trace, or NULL. */
static const tr_route_trace *traced_run(const char *path, const unsigned char *off, int expect_mask_rc,
                                        tr_model **model_out, tr_session **sess_out, float *logits) {
    char err[256];
    *sess_out = NULL;
    *model_out = tr_model_load(path, NULL, err, sizeof err);
    TR_CHECK(*model_out != NULL);
    if (*model_out == NULL) return NULL;
    if (off != NULL) TR_CHECK_EQ_INT(tr_model_set_expert_mask(*model_out, off), expect_mask_rc);
    *sess_out = tr_session_create(*model_out, 0, 0, NULL, 0);
    TR_CHECK(*sess_out != NULL);
    if (*sess_out == NULL) return NULL;
    TR_CHECK(tr_session_route_trace_begin(*sess_out, N_TOTAL) == 0);
    TR_CHECK(tr_session_eval(*sess_out, seq, N_TOTAL) == 0);
    if (logits != NULL) memcpy(logits, tr_session_logits(*sess_out), VOCAB * sizeof(float));
    return tr_session_route_trace(*sess_out);
}

/* Version 2 of the trace: the id of every token, and the router's margin. The margin is checked
 * exactly through a second model with the same weights and one more expert per token (the
 * synthetic weights do not depend on n_used): at layer 0, whose input is the same in the two,
 * the best expert the first model leaves out IS the last one the second model takes. */
static void test_tokens_margins(const char *argv0) {
    static const synth_params USED3 = {2, 256, 4, 2, 512, 8, 3, VOCAB, 64, TR_TYPE_F32};
    char path2[512], path3[512];
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_used2.gguf", path2, sizeof path2) == 0);
    TR_CHECK(synth_write(&USED3, argv0, "route_used3.gguf", path3, sizeof path3) == 0);
    tr_model *m2, *m3;
    tr_session *s2, *s3;
    const tr_route_trace *t2 = traced_run(path2, NULL, 0, &m2, &s2, NULL);
    const tr_route_trace *t3 = traced_run(path3, NULL, 0, &m3, &s3, NULL);
    TR_CHECK(t2 != NULL && t3 != NULL);
    if (t2 != NULL && t3 != NULL) {
        int64_t tokens_ok = 0, ordered = 0, crossed = 0, not_the_same_number = 0;
        TR_CHECK_EQ_INT(t2->n_tokens, N_TOTAL);
        for (int64_t t = 0; t < t2->n_tokens; t++) {
            TR_CHECK_EQ_INT(t2->tokens[t], seq[t]);
            tokens_ok++;
            for (int64_t L = 0; L < t2->n_layers; L++) {
                const float *mg = t2->margins + (t * t2->n_layers + L) * 2;
                TR_CHECK(mg[0] <= 1.0f && mg[0] >= mg[1] && mg[1] > 0.0f);
                ordered++;
            }
            const float *a = t2->margins + t * t2->n_layers * 2; /* layer 0 */
            const float *b = t3->margins + t * t3->n_layers * 2;
            TR_CHECK(memcmp(&a[1], &b[0], sizeof(float)) == 0); /* left out by 2 == last taken by 3 */
            crossed++;
            not_the_same_number += a[0] != b[0];
        }
        TR_CHECK_EQ_INT(tokens_ok, N_TOTAL);
        TR_CHECK(ordered > 0 && crossed > 0);
        TR_CHECK(not_the_same_number > 0); /* or the cross-check above compares a number with itself */
    }
    tr_session_free(s2);
    tr_session_free(s3);
    tr_model_free(m2);
    tr_model_free(m3);
    remove(path2);
    remove(path3);
}

/* The expert mask (tr_model_set_expert_mask, measurement only): a masked expert is never chosen,
 * each layer by its own row; an empty mask and a cleared mask are the model itself, byte for
 * byte; a mask that leaves a layer fewer experts than a token uses is refused and changes nothing. */
static void test_mask(const char *argv0) {
    enum { N_LAYERS = 2, N_EXPERT = 8 };
    char path[512];
    TR_CHECK(synth_write(&ROUTE_PARAMS, argv0, "route_mask.gguf", path, sizeof path) == 0);
    static float ref[VOCAB], got[VOCAB];
    tr_model *m;
    tr_session *s;

    const tr_route_trace *plain = traced_run(path, NULL, 0, &m, &s, ref);
    /* layer 0 loses experts 0-2, layer 1 loses 5-7: different rows on purpose */
    unsigned char off[N_LAYERS * N_EXPERT] = {1, 1, 1, 0, 0, 0, 0, 0, /**/ 0, 0, 0, 0, 0, 1, 1, 1};
    int64_t would_have_used = 0;
    if (plain != NULL)
        for (int64_t t = 0; t < plain->n_tokens; t++)
            for (int64_t L = 0; L < N_LAYERS; L++)
                for (int64_t k = 0; k < plain->n_used; k++)
                    would_have_used += off[L * N_EXPERT + plain->chosen[(t * N_LAYERS + L) * plain->n_used + k]];
    TR_CHECK(would_have_used > 0); /* the mask takes away experts this run really used */
    tr_session_free(s);
    tr_model_free(m);

    const tr_route_trace *masked = traced_run(path, off, 0, &m, &s, got);
    int64_t checked = 0;
    if (masked != NULL)
        for (int64_t t = 0; t < masked->n_tokens; t++)
            for (int64_t L = 0; L < N_LAYERS; L++)
                for (int64_t k = 0; k < masked->n_used; k++) {
                    TR_CHECK_EQ_INT(off[L * N_EXPERT + masked->chosen[(t * N_LAYERS + L) * masked->n_used + k]], 0);
                    checked++;
                }
    TR_CHECK(checked > 0);
    TR_CHECK(memcmp(ref, got, sizeof ref) != 0); /* and the output moved: it is not the model's own */

    if (m != NULL && s != NULL) {
        static float masked_logits[VOCAB];
        memcpy(masked_logits, got, sizeof got);
        /* refused: layer 1 would keep one expert, a token uses two; the mask in force stays */
        unsigned char too_many[N_LAYERS * N_EXPERT] = {0, 0, 0, 0, 0, 0, 0, 0, /**/ 1, 1, 1, 1, 1, 1, 1, 0};
        TR_CHECK_EQ_INT(tr_model_set_expert_mask(m, too_many), -1);
        TR_CHECK(tr_session_rewind(s, 0) == 0 && tr_session_eval(s, seq, N_TOTAL) == 0);
        TR_CHECK(memcmp(tr_session_logits(s), masked_logits, sizeof masked_logits) == 0);
        /* cleared, then empty: the model itself again */
        TR_CHECK_EQ_INT(tr_model_set_expert_mask(m, NULL), 0);
        TR_CHECK(tr_session_rewind(s, 0) == 0 && tr_session_eval(s, seq, N_TOTAL) == 0);
        TR_CHECK(memcmp(tr_session_logits(s), ref, sizeof ref) == 0);
        unsigned char none[N_LAYERS * N_EXPERT] = {0};
        TR_CHECK_EQ_INT(tr_model_set_expert_mask(m, none), 0);
        TR_CHECK(tr_session_rewind(s, 0) == 0 && tr_session_eval(s, seq, N_TOTAL) == 0);
        TR_CHECK(memcmp(tr_session_logits(s), ref, sizeof ref) == 0);
    }
    tr_session_free(s);
    tr_model_free(m);
    remove(path);
}

int main(int argc, char **argv) {
    const char *argv0 = argc > 0 ? argv[0] : "";
    for (int i = 0; i < N_TOTAL; i++) seq[i] = (int32_t)((i * 29 + (i * i) % 7 + 11) % VOCAB);

    test_pred_out(argv0);
    test_pred_in(argv0);
    test_structure(argv0);
    test_shape_of_passes(argv0);
    test_on_vs_off(argv0);
    test_tokens_margins(argv0);
    test_mask(argv0);

    printf("  route trace: pred_out exact when attention can't move x (and differs when it can), "
           "pred_in exact against a next-layer-seeded model, ids in range, chosen increasing, "
           "last layer sentinel, same trace whatever the batching or the pool, no logit moves; "
           "token ids and margins (exact against a model with one more expert per token); "
           "expert mask: never chosen, each layer its own row, empty and cleared = the model\n");
    TR_TEST_EXIT();
}
