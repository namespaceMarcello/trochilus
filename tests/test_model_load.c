/* test_model_load.c — what src/models/olmoe.c refuses at load and at eval, and the rules a file
 * alone does not show, on synthetic OLMoE files (tests/synth_olmoe.h) malformed one way at a time.
 *
 * Branches pinned (docs/MEASUREMENTS.md §Generated mutations: each case kills survivors of
 * tools/mutate_auto.py on olmoe.c, seen red there):
 *  - load: a missing key, each zero hyperparameter (with key_length written, so that the derived
 *    one never masks it), more experts used than there are and exactly as many, a tensor with a
 *    wrong dimension (rows, columns, experts), a missing tensor, an unsupported type -- each
 *    refused with its own message, never by a later check's;
 *  - session: n_ctx 0 takes the file's context, or 4096 when the file says 0;
 *  - eval: no tokens, or more than the context holds, is -1 and moves nothing;
 *  - the expert mask may leave exactly n_used experts, not fewer;
 *  - eval_rows refuses n_logits below 1, above n, above the rows kept, and above 1 over more than
 *    one pass, and moves nothing;
 *  - the router's ties go to the lower expert id (a zero router: every score equal);
 *  - an expert part stored in another type than the layer's gate (real GGUFs mix them): up in F16
 *    gives the bits of the same values in F32, each part read as its own type;
 *  - the route trace: of a pass exactly n_batch long (under ASan: nothing read past its rows), of
 *    0 tokens, of a 1-layer model (the tokens are recorded), and refused above 65535 experts (its
 *    ids are 16 bits, 0xFFFF the "no prediction" mark). */
#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* setenv */
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/format/gguf.h"
#include "../src/models/model.h"
#include "synth_olmoe.h"

static void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static const char *self = "";
static char path[512];
static tr_pool *pool;

static void knobs_off(void) {
    synth_drop = NULL;
    synth_set_key[0] = synth_set_key[1] = NULL;
    synth_reshape = NULL;
    synth_reshape_dim = 0;
    synth_retype = NULL;
    synth_f16_as_f32 = NULL;
    synth_zero_router = 0;
}

/* The file the knobs describe, loaded. want_err NULL: it must load, and the model is returned;
 * otherwise it must be refused with a message that contains want_err. */
static tr_model *load_case(const char *what, const synth_params *P, const char *want_err) {
    char err[256] = "";
    if (synth_write(P, self, "test_model_load_tmp.gguf", path, sizeof path) != 0) {
        fprintf(stderr, "%s: could not write the file\n", what);
        TR_CHECK(0);
        return NULL;
    }
    tr_model *m = tr_model_load(path, pool, err, sizeof err);
    remove(path);
    if (want_err == NULL) {
        if (m == NULL) fprintf(stderr, "%s: refused: %s\n", what, err);
        TR_CHECK(m != NULL);
        return m;
    }
    if (m != NULL || strstr(err, want_err) == NULL) {
        fprintf(stderr, "%s: %s, expected a refusal with \"%s\"\n", what, m != NULL ? "loaded" : err, want_err);
        TR_CHECK(m == NULL && strstr(err, want_err) != NULL);
    }
    tr_model_free(m);
    return NULL;
}

/* logits of the last of `n` tokens, one pass, on a model written from P with the current knobs */
static int logits_of(const synth_params *P, const int32_t *tok, int64_t n, float *out, int64_t vocab) {
    tr_model *m = load_case("logits", P, NULL);
    tr_session *s = m != NULL ? tr_session_create(m, 0, 0, NULL, 0) : NULL;
    int rc = s != NULL && tr_session_eval(s, tok, n) == 0 ? 0 : -1;
    if (rc == 0) memcpy(out, tr_session_logits(s), (size_t)vocab * sizeof(float));
    tr_session_free(s);
    tr_model_free(m);
    return rc;
}

/* key_length written (so a zero head count or embedding cannot zero it too), and one key set */
static void refuse_zero(const synth_params *P, const char *key) {
    knobs_off();
    synth_set_key[0] = "olmoe.attention.key_length";
    synth_set_value[0] = P->n_embd / P->n_head;
    synth_set_key[1] = key;
    synth_set_value[1] = 0;
    load_case(key, P, "invalid (zero) hyperparameter");
}

int main(int argc, char **argv) {
    self = argc > 0 ? argv[0] : "";
    pool = tr_pool_create(2);
    TR_CHECK(pool != NULL);
    if (pool == NULL) TR_TEST_EXIT();

    /* 2 layers, n_embd 16, 4 heads (2 kv, head_dim 4), n_ff 8, 4 experts (2 used), vocab 16, context 32 */
    const synth_params P = {2, 16, 4, 2, 8, 4, 2, 16, 32, TR_TYPE_F32};

    /* ---- load: refusals, each with its own message ---- */
    knobs_off();
    synth_drop = "olmoe.block_count";
    load_case("no block_count", &P, "missing metadata key 'olmoe.block_count'");
    /* one zero at a time: every operand of the check stands alone once */
    refuse_zero(&P, "olmoe.embedding_length");
    refuse_zero(&P, "olmoe.attention.head_count");
    refuse_zero(&P, "olmoe.expert_count");
    refuse_zero(&P, "olmoe.attention.key_length");

    knobs_off();
    synth_set_key[0] = "olmoe.expert_used_count";
    synth_set_value[0] = 5;
    load_case("5 of 4 experts used", &P, "exceeds olmoe.expert_count");
    synth_set_value[0] = 4;
    tr_model_free(load_case("4 of 4 experts used", &P, NULL));

    knobs_off();
    synth_reshape = "blk.0.attn_q.weight";
    synth_reshape_dim = 1;
    load_case("attn_q one row more", &P, "tensor 'blk.0.attn_q.weight' has shape");
    synth_reshape = "blk.1.ffn_gate_exps.weight";
    synth_reshape_dim = 2;
    load_case("gate_exps one expert more", &P, "tensor 'blk.1.ffn_gate_exps.weight' has shape");
    synth_reshape = "token_embd.weight";
    synth_reshape_dim = 0;
    load_case("token_embd one column more", &P, "tensor 'token_embd.weight' has unexpected shape");

    knobs_off();
    synth_drop = "blk.1.attn_output.weight";
    load_case("no attn_output", &P, "missing tensor 'blk.1.attn_output.weight'");
    knobs_off();
    synth_retype = "blk.0.attn_k.weight";
    synth_retype_to = TR_TYPE_BF16;
    load_case("attn_k in BF16", &P, "tensor 'blk.0.attn_k.weight' has unsupported type 30");

    /* ---- session and eval on a good file ---- */
    knobs_off();
    tr_model *m = load_case("good file", &P, NULL);
    if (m != NULL) {
        char err[256];
        tr_session *s = tr_session_create(m, 0, 0, err, sizeof err);
        TR_CHECK(s != NULL);
        if (s != NULL) {
            TR_CHECK_EQ_INT(tr_session_n_ctx(s), 32);
            int32_t tok[33];
            for (int i = 0; i < 33; i++) tok[i] = i % 16;
            TR_CHECK(tr_session_eval(s, tok, 0) == -1);
            TR_CHECK(tr_session_eval(s, tok, 33) == -1);
            TR_CHECK_EQ_INT(tr_session_pos(s), 0);
            TR_CHECK(tr_session_eval(s, tok, 30) == 0);
            TR_CHECK(tr_session_eval(s, tok, 3) == -1);
            TR_CHECK_EQ_INT(tr_session_pos(s), 30);
            TR_CHECK(tr_session_eval(s, tok, 2) == 0);
            TR_CHECK_EQ_INT(tr_session_pos(s), 32);
            tr_session_free(s);
        }

        /* the mask may leave exactly n_used experts in a layer, not fewer */
        unsigned char off[2 * 4] = {1, 1, 0, 0, 0, 1, 0, 1};
        TR_CHECK_EQ_INT(tr_model_set_expert_mask(m, off), 0);
        off[2] = 1;
        TR_CHECK_EQ_INT(tr_model_set_expert_mask(m, off), -1);
        TR_CHECK_EQ_INT(tr_model_set_expert_mask(m, NULL), 0);

        /* eval_rows: each bound alone (32 positions, 32 a pass, 16 rows kept) */
        s = tr_session_create(m, 0, 0, err, sizeof err);
        TR_CHECK(s != NULL);
        if (s != NULL) {
            int32_t tok[20];
            for (int i = 0; i < 20; i++) tok[i] = i % 16;
            TR_CHECK(tr_session_eval_rows(s, tok, 2, 0) == -1);   /* below 1 */
            TR_CHECK(tr_session_eval_rows(s, tok, 2, 3) == -1);   /* above n */
            TR_CHECK(tr_session_eval_rows(s, tok, 20, 17) == -1); /* above the 16 rows kept */
            TR_CHECK_EQ_INT(tr_session_pos(s), 0);
            TR_CHECK(tr_session_eval_rows(s, tok, 20, 16) == 0);
            tr_session_free(s);
        }
        s = tr_session_create(m, 0, 4, err, sizeof err);
        TR_CHECK(s != NULL);
        if (s != NULL) {
            int32_t five[5] = {1, 2, 3, 4, 5};
            TR_CHECK(tr_session_eval_rows(s, five, 5, 2) == -1);  /* two rows over two passes of 4 */
            TR_CHECK_EQ_INT(tr_session_pos(s), 0);
            TR_CHECK(tr_session_eval_rows(s, five, 4, 2) == 0);
            tr_session_free(s);
        }

        /* the route trace of a pass exactly n_batch long: every token recorded */
        s = tr_session_create(m, 0, 4, err, sizeof err);
        TR_CHECK(s != NULL);
        if (s != NULL) {
            TR_CHECK(tr_session_route_trace_begin(s, 8) == 0);
            int32_t four[4] = {1, 2, 3, 4};
            TR_CHECK(tr_session_eval(s, four, 4) == 0);
            const tr_route_trace *tr = tr_session_route_trace(s);
            TR_CHECK(tr != NULL && tr->n_tokens == 4);
            tr_session_free(s);
        }

        /* a trace of 0 tokens: begun, and records nothing */
        s = tr_session_create(m, 0, 0, err, sizeof err);
        TR_CHECK(s != NULL);
        if (s != NULL) {
            TR_CHECK(tr_session_route_trace_begin(s, 0) == 0);
            int32_t two[2] = {1, 2};
            TR_CHECK(tr_session_eval(s, two, 2) == 0);
            const tr_route_trace *tr = tr_session_route_trace(s);
            TR_CHECK(tr != NULL && tr->n_tokens == 0 && tr->max_tokens == 0);
            tr_session_free(s);
        }
        tr_model_free(m);
    }

    /* a 1-layer model: its one layer is the first and the last, and the trace keeps the tokens */
    knobs_off();
    synth_params P1 = P;
    P1.layers = 1;
    m = load_case("one layer", &P1, NULL);
    if (m != NULL) {
        tr_session *s = tr_session_create(m, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        if (s != NULL) {
            TR_CHECK(tr_session_route_trace_begin(s, 8) == 0);
            int32_t three[3] = {5, 6, 7};
            TR_CHECK(tr_session_eval(s, three, 3) == 0);
            const tr_route_trace *tr = tr_session_route_trace(s);
            TR_CHECK(tr != NULL && tr->n_tokens == 3);
            if (tr != NULL && tr->n_tokens == 3)
                for (int i = 0; i < 3; i++) TR_CHECK_EQ_INT(tr->tokens[i], three[i]);
            tr_session_free(s);
        }
        tr_model_free(m);
    }

    /* the trace's ids are 16 bits and 0xFFFF marks "no prediction": 65535 experts are traced,
     * 65536 are refused. Tiny experts (2 x 1), read buffered: aligned for direct reads, 65536
     * units would take 800 MB of slots */
    set_env("TR_EXPERT_DIRECT", "0");
    for (uint32_t n_expert = 65535; n_expert <= 65536; n_expert++) {
        const synth_params PE = {1, 2, 1, 1, 1, n_expert, 1, 4, 8, TR_TYPE_F32};
        knobs_off();
        m = load_case("65535 or 65536 experts", &PE, NULL);
        if (m == NULL) continue;
        tr_session *s = tr_session_create(m, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        if (s != NULL) TR_CHECK_EQ_INT(tr_session_route_trace_begin(s, 1), n_expert <= 65535 ? 0 : -1);
        tr_session_free(s);
        tr_model_free(m);
    }
    set_env("TR_EXPERT_DIRECT", "");

    /* mixed types per part: up in F16 against the same values in F32 (and both against the plain
     * F32 model, so that the up matrix is seen to matter at all) */
    {
        int32_t tok[5] = {3, 1, 4, 1, 5};
        float mixed[16], same[16], plain[16];
        knobs_off();
        synth_retype = "blk.0.ffn_up_exps.weight";
        synth_retype_to = TR_TYPE_F16;
        int ok = logits_of(&P, tok, 5, mixed, 16) == 0;
        knobs_off();
        synth_f16_as_f32 = "blk.0.ffn_up_exps.weight";
        ok = ok && logits_of(&P, tok, 5, same, 16) == 0;
        knobs_off();
        ok = ok && logits_of(&P, tok, 5, plain, 16) == 0;
        TR_CHECK(ok);
        if (ok) {
            TR_CHECK(memcmp(mixed, same, sizeof mixed) == 0);
            TR_CHECK(memcmp(mixed, plain, sizeof mixed) != 0);
        }
    }

    /* a file whose context is 0: the session takes 4096 */
    knobs_off();
    synth_params P0 = P;
    P0.ctx = 0;
    m = load_case("context 0", &P0, NULL);
    if (m != NULL) {
        tr_session *s = tr_session_create(m, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        if (s != NULL) TR_CHECK_EQ_INT(tr_session_n_ctx(s), 4096);
        tr_session_free(s);
        tr_model_free(m);
    }

    /* a zero router: every expert scores the same, and the tie goes to the lower id, so every
     * token of every layer takes experts 0 and 1 */
    knobs_off();
    synth_zero_router = 1;
    m = load_case("zero router", &P, NULL);
    if (m != NULL) {
        tr_session *s = tr_session_create(m, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        if (s != NULL) {
            TR_CHECK(tr_session_route_trace_begin(s, 8) == 0);
            int32_t six[6] = {3, 1, 4, 1, 5, 9};
            TR_CHECK(tr_session_eval(s, six, 6) == 0);
            const tr_route_trace *tr = tr_session_route_trace(s);
            int64_t checked = 0;
            for (int64_t i = 0; tr != NULL && i < tr->n_tokens * tr->n_layers; i++, checked++) {
                TR_CHECK_EQ_INT(tr->chosen[i * 2], 0);
                TR_CHECK_EQ_INT(tr->chosen[i * 2 + 1], 1);
            }
            TR_CHECK_EQ_INT(checked, 6 * 2);
            tr_session_free(s);
        }
        tr_model_free(m);
    }
    knobs_off();

    tr_pool_destroy(pool);
    TR_TEST_EXIT();
}
