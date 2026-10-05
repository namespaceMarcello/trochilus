/* test_stream.c — the OLMoE engine takes its experts from the shared store (src/memory/experts.h)
 * instead of holding every one of them resident, and gives byte-identical logits whatever the
 * budget (docs/ARCHITECTURE.md Esperti M1, lot 2).
 *
 * Every check forces its budget explicitly (never automatic: automatic depends on the machine's
 * free RAM, and this suite must give the same answer everywhere). Synthetic models from
 * tests/synth_olmoe.h: 2 layers, 8 experts, 2 used (n_units = 16, min_slots = 8 + 2 = 10), F32
 * and Q8_0, deterministic weights (same params -> the same file, every time), so a "dry run" of
 * a pass tells us exactly how many real reads that same pass needs again, which lets a failure
 * be pointed at an exact read without depending on what the router actually chose.
 *
 * Each test says which branch it exercises and counts that it was actually taken (CLAUDE.md,
 * LESSONS #43 #50 #54 #78):
 *   content      a resident model's tr_experts_part(layer, expert, part) bytes match exactly the
 *                named GGUF tensor's own bytes at that expert's offset, read independently with
 *                tr_gguf_read_range straight off the file: a load-time mix-up (parts swapped, an
 *                offset off by one tensor) gives the same *wrong* answer under every budget, so
 *                only a check against the file itself, never through olmoe_load's own
 *                bookkeeping, can catch it
 *   reference    one-token passes, a whole prompt in one pass, and tr_session_eval_rows agree on
 *                a forced-resident model (F32 and Q8_0); the same tokens under a min-slots
 *                (10/16) budget (no pool, and a pool of 8) and a 13-slot budget give
 *                byte-identical logits; the min store shows misses > 0 and evictions > 0 (the
 *                branch really ran)
 *   resident     a forced-resident load: misses == n_units at load, evictions == 0, unchanged
 *                after a full run (nothing read again)
 *   budget_gate  one byte under the minimum refuses (message names MiB); tr_expert_budget_plan
 *                on made-up numbers: resident, partial, refused under the minimum, and the 2 GiB
 *                reserve floor overriding total/10
 *   failure      a wrapped reader fails the k-th real read, chosen (via a dry run) to land inside
 *                a one-token pass, and inside the SECOND internal pass of a prompt longer than
 *                n_batch: eval returns -1, tr_session_pos is exactly what it was before that
 *                eval call; the reader restored, the same eval retried gives the reference logits;
 *                and in a session that had logits, the failed eval leaves them as they were
 *   once_per_prompt  a prompt of 36 tokens in passes of 12, on the smallest store: the units
 *                read must be at most n_units + n_slots, not passes x n_units
 *   rewind       rewind to 0 and re-evaluate the same tokens under the min store: identical
 *                logits (the store has no notion of position)
 *   direct       a real model loaded with the direct path (docs/ARCHITECTURE.md Esperti M1, Step
 *                C) gives logits byte-identical to TR_EXPERT_DIRECT=0's buffered path, resident
 *                and at --expert-budget min; skipped, with a message, if tr_file_open_direct
 *                cannot actually read this filesystem
 *   mem_available TR_MEM_AVAILABLE_MIB (Step D) too small refuses with the minimum named in MiB;
 *                a value placed inside the window a bigger synthetic model opens between "not
 *                enough to be fully resident" and "not enough to run at all" forces the plan's
 *                partial branch on a real load, logits still identical to a forced-resident one;
 *                TR_MEM_TOTAL_MIB (tools/machines.sh) of 1 TiB places the window at its tenth's
 *                reserve, partial where the real total would load resident; tr_model_load_plan in
 *                the window: a session of 4 positions leaves no fewer slots than the default
 *                context and one of 1024 fewer only by its working memory (the KV is not set
 *                aside), the 4-position store's logits the resident's; kv_room: a 1024-position
 *                session's passes give the store's slots to the KV's pages as they are written
 *                (floor or ceil of their bytes in slots), the logits a resident's, every slot
 *                taken again at free, a context too long for the room refused by name
 *   progress    tr_model_load_progress (model.h tr_progress) at a resident budget: total ==
 *                the file's dense + expert tensor bytes, one report per dense tensor and per
 *                expert unit, done strictly growing and ending on total; at the smallest store
 *                (really partial): total == dense bytes only, one report per dense tensor
 *   arrival      a layer's misses read by the store's I/O thread while its present experts compute
 *                (olmoe_refresh_experts, arrival_layout, olmoe_take_late): 6 layers, 4 of 8 experts used,
 *                at the I/O thread's smallest store and 6 slots more, Q4_K (its integer roads) and F32,
 *                pool of 4: one-token passes and a whole prompt's rows give the resident's logits in every
 *                arm (each late unit as it lands, two waves, --ab arrive's arm B on the calling thread);
 *                the first two counted arrived > 0, the third 0; misses, evictions and bytes equal in all
 *
 * Seen red: tools/mutate_stream.sh; progress: tools/mutate_bar.sh.
 */
#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* setenv/unsetenv (test_mem_available, test_direct) */
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "synth_olmoe.h"
#include "../src/base/platform.h"
#include "../src/kernels/kernels.h"
#include "../src/models/model.h"
#include "../src/models/model_internal.h"
#include "../src/memory/experts.h"
#include "../src/kv/kv.h"

/* NULL clears the variable; portable enough for a test (both platforms declare stdlib's
 * putenv-family functions differently, so each gets its own call rather than one shared one). */
static void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value != NULL ? value : "");
#else
    if (value != NULL) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

enum {
    N_LAYERS = 2, N_EMBD = 256, N_HEAD = 4, N_HEAD_KV = 2, N_FF = 512, N_EXPERT = 8, N_USED = 2,
    VOCAB = 64, CTX = 64
};
static const synth_params P_F32 = {N_LAYERS, N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_EXPERT, N_USED,
                                    VOCAB, CTX, TR_TYPE_F32};
static const synth_params P_Q8_0 = {N_LAYERS, N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_EXPERT, N_USED,
                                     VOCAB, CTX, TR_TYPE_Q8_0};

enum { N_UNITS = N_LAYERS * N_EXPERT, MIN_SLOTS = N_EXPERT + N_USED }; /* 16, 10 */
#define HUGE_BUDGET ((uint64_t)1 << 40) /* forced: always above every expert's real need */

enum { N_PROMPT = 12 }; /* enough distinct tokens that 2 layers x 8 experts touches more than the
                         * 10 slots of the min store, so an eviction is not left to chance */
static int32_t prompt_token(int i) {
    return (int32_t)((i * 11 + 5) % VOCAB);
}

/* ---- small helpers shared by every test ---- */

static tr_model *load_budget(const char *path, tr_pool *pool, uint64_t budget) {
    char err[256];
    tr_model *m = tr_model_load_budget(path, pool, budget, err, sizeof err);
    TR_CHECK(m != NULL);
    if (m == NULL) fprintf(stderr, "  load failed: %s\n", err);
    return m;
}

static tr_session *create_session(tr_model *m, int64_t n_batch) {
    if (m == NULL) return NULL;
    char err[256];
    tr_session *s = tr_session_create(m, CTX, n_batch, err, sizeof err);
    TR_CHECK(s != NULL);
    if (s == NULL) fprintf(stderr, "  session failed: %s\n", err);
    return s;
}

/* Evaluates tokens[0..n) of a fresh model at `budget` (one pass, via tr_session_eval_rows) and
 * checks every row against ref[i], byte for byte. want_slots >= 0: also checks the store landed
 * on exactly that many slots (not just that it happened to give the right answer). */
static void run_and_compare(const char *path, tr_pool *pool, uint64_t budget, int64_t want_slots,
                            const int32_t *tokens, int64_t n, float ref[][VOCAB], int *checked) {
    tr_model *m = load_budget(path, pool, budget);
    if (m == NULL) return;
    if (want_slots >= 0) {
        tr_experts_stats st;
        TR_CHECK(tr_model_expert_stats(m, &st) == 0);
        TR_CHECK_EQ_INT(st.n_slots, want_slots);
    }
    tr_session *s = create_session(m, 0);
    if (s != NULL) {
        TR_CHECK(tr_session_eval_rows(s, tokens, n, n) == 0);
        for (int64_t i = 0; i < n; i++) {
            TR_CHECK(memcmp(tr_session_logits_back(s, n - 1 - i), ref[i], VOCAB * sizeof(float)) == 0);
            (*checked)++;
        }
        tr_session_free(s);
    }
    tr_model_free(m);
}

/* ---- content: the store's bytes for (layer, expert, part) are exactly the named GGUF tensor's,
 * read independently of olmoe_load's own offset bookkeeping ---- */

static void check_part(tr_gguf *g, const tr_experts *ex, int64_t layer, int64_t expert, int part,
                       const char *tensor_name, int64_t in_dim, int64_t out_dim, int *checked) {
    const tr_gguf_tensor *t = tr_gguf_find_tensor(g, tensor_name);
    TR_CHECK(t != NULL);
    if (t == NULL) return;
    size_t n = (size_t)out_dim * tr_row_bytes(t->type, in_dim);
    unsigned char *want = (unsigned char *)malloc(n);
    TR_CHECK(want != NULL);
    if (want == NULL) return;
    TR_CHECK(tr_gguf_read_range(g, t, (uint64_t)expert * (uint64_t)n, want, n) == 0);
    const unsigned char *got = (const unsigned char *)tr_experts_part(ex, layer, expert, part);
    TR_CHECK(got != NULL);
    if (got != NULL) TR_CHECK(memcmp(got, want, n) == 0);
    free(want);
    (*checked)++;
}

static void test_content(const char *argv0) {
    char path[512];
    TR_CHECK(synth_write(&P_F32, argv0, "stream_content.gguf", path, sizeof path) == 0);
    int checked = 0;

    tr_model *m = load_budget(path, NULL, HUGE_BUDGET); /* resident: every part pointer valid */
    if (m != NULL) {
        tr_experts *ex = tr_model_experts(m);
        TR_CHECK(ex != NULL);
        char gerr[256];
        tr_gguf *g = tr_gguf_open(path, gerr, sizeof gerr);
        TR_CHECK(g != NULL);
        if (ex != NULL && g != NULL) {
            char name[64];
            for (int64_t L = 0; L < N_LAYERS; L++) {
                for (int64_t e = 0; e < N_EXPERT; e += 3) { /* a few experts is enough */
                    snprintf(name, sizeof name, "blk.%lld.ffn_gate_exps.weight", (long long)L);
                    check_part(g, ex, L, e, 0, name, N_EMBD, N_FF, &checked);
                    snprintf(name, sizeof name, "blk.%lld.ffn_up_exps.weight", (long long)L);
                    check_part(g, ex, L, e, 1, name, N_EMBD, N_FF, &checked);
                    snprintf(name, sizeof name, "blk.%lld.ffn_down_exps.weight", (long long)L);
                    check_part(g, ex, L, e, 2, name, N_FF, N_EMBD, &checked);
                }
            }
        }
        if (g != NULL) tr_gguf_close(g);
        tr_model_free(m);
    }
    remove(path);
    TR_CHECK(checked > 0);
}

/* ---- reference: every pass shape agrees, and every budget matches it ---- */

static void test_reference_one(const char *argv0, const synth_params *P, const char *file, int *checked) {
    char path[512];
    TR_CHECK(synth_write(P, argv0, file, path, sizeof path) == 0);

    int32_t tokens[N_PROMPT];
    for (int i = 0; i < N_PROMPT; i++) tokens[i] = prompt_token(i);

    tr_model *m = load_budget(path, NULL, HUGE_BUDGET);
    if (m != NULL) {
        tr_session *s = create_session(m, 0);
        if (s != NULL) {
            float ref[N_PROMPT][VOCAB];
            for (int i = 0; i < N_PROMPT; i++) {
                TR_CHECK(tr_session_eval(s, tokens + i, 1) == 0); /* one-token passes: canonical */
                memcpy(ref[i], tr_session_logits(s), sizeof ref[i]);
            }
            (*checked)++;

            TR_CHECK(tr_session_rewind(s, 0) == 0);
            TR_CHECK(tr_session_eval_rows(s, tokens, N_PROMPT, N_PROMPT) == 0); /* several rows, one pass */
            for (int i = 0; i < N_PROMPT; i++)
                TR_CHECK(memcmp(tr_session_logits_back(s, N_PROMPT - 1 - i), ref[i], sizeof ref[i]) == 0);
            (*checked)++;

            TR_CHECK(tr_session_rewind(s, 0) == 0);
            TR_CHECK(tr_session_eval(s, tokens, N_PROMPT) == 0); /* the whole prompt, one pass */
            TR_CHECK(memcmp(tr_session_logits(s), ref[N_PROMPT - 1], sizeof ref[0]) == 0);
            (*checked)++;

            tr_experts_stats st;
            TR_CHECK(tr_model_expert_stats(m, &st) == 0);
            TR_CHECK_EQ_INT(st.n_slots, N_UNITS);
            TR_CHECK_EQ_INT(st.evictions, 0);
            (*checked)++;

            /* min store (10 of 16 units): no pool, then a pool of 8 -- same logits either way */
            run_and_compare(path, NULL, UINT64_MAX, MIN_SLOTS, tokens, N_PROMPT, ref, checked);
            tr_pool *pool8 = tr_pool_create(8);
            run_and_compare(path, pool8, UINT64_MAX, MIN_SLOTS, tokens, N_PROMPT, ref, checked);

            /* the min store really evicted, not just fit by luck (LESSONS: same result never
             * proves which branch ran) */
            {
                tr_model *m2 = load_budget(path, pool8, UINT64_MAX);
                if (m2 != NULL) {
                    tr_session *s2 = create_session(m2, 0);
                    if (s2 != NULL) {
                        TR_CHECK(tr_session_eval_rows(s2, tokens, N_PROMPT, N_PROMPT) == 0);
                        tr_experts_stats mst;
                        TR_CHECK(tr_model_expert_stats(m2, &mst) == 0);
                        TR_CHECK_EQ_INT(mst.n_slots, MIN_SLOTS);
                        TR_CHECK(mst.misses > 0);
                        TR_CHECK(mst.evictions > 0);
                        (*checked)++;
                        tr_session_free(s2);
                    }
                    tr_model_free(m2);
                }
            }
            tr_pool_destroy(pool8);

            /* a budget in between: 13 of 16 units */
            run_and_compare(path, NULL, (uint64_t)13 * st.slot_bytes, 13, tokens, N_PROMPT, ref, checked);

            tr_session_free(s);
        }
        tr_model_free(m);
    }
    remove(path);
}

static void test_reference(const char *argv0) {
    int checked = 0;
    test_reference_one(argv0, &P_F32, "stream_ref_f32.gguf", &checked);
    test_reference_one(argv0, &P_Q8_0, "stream_ref_q8.gguf", &checked);
    TR_CHECK(checked > 0);
}

/* ---- resident: budget covers every unit ---- */

static void test_resident(const char *argv0) {
    char path[512];
    TR_CHECK(synth_write(&P_F32, argv0, "stream_resident.gguf", path, sizeof path) == 0);
    int checked = 0;

    tr_model *m = load_budget(path, NULL, HUGE_BUDGET);
    if (m != NULL) {
        tr_experts_stats before;
        TR_CHECK(tr_model_expert_stats(m, &before) == 0);
        TR_CHECK_EQ_INT(before.n_slots, N_UNITS);
        TR_CHECK_EQ_INT(before.misses, N_UNITS); /* tr_experts_load_all at load: one read per unit */
        TR_CHECK_EQ_INT(before.evictions, 0);
        checked++;

        tr_session *s = create_session(m, 0);
        if (s != NULL) {
            int32_t tokens[N_PROMPT];
            for (int i = 0; i < N_PROMPT; i++) tokens[i] = prompt_token(i);
            TR_CHECK(tr_session_eval(s, tokens, N_PROMPT) == 0);
            for (int i = 0; i < 4; i++) {
                int32_t t = prompt_token(N_PROMPT + i);
                TR_CHECK(tr_session_eval(s, &t, 1) == 0);
            }
            tr_experts_stats after;
            TR_CHECK(tr_model_expert_stats(m, &after) == 0);
            TR_CHECK_EQ_INT(after.misses, before.misses);
            TR_CHECK_EQ_INT(after.evictions, before.evictions);
            checked++;
            tr_session_free(s);
        }
        tr_model_free(m);
    }
    remove(path);
    TR_CHECK(checked > 0);
}

/* ---- budget_gate: the create-time refusal, and the pure plan function on made-up numbers ---- */

static void test_budget_gate(const char *argv0) {
    char path[512];
    TR_CHECK(synth_write(&P_F32, argv0, "stream_gate.gguf", path, sizeof path) == 0);
    int checked = 0;

    tr_model *m = load_budget(path, NULL, UINT64_MAX);
    if (m != NULL) {
        tr_experts_stats st;
        TR_CHECK(tr_model_expert_stats(m, &st) == 0);
        TR_CHECK_EQ_INT(st.n_slots, MIN_SLOTS);
        uint64_t min_bytes = (uint64_t)st.n_slots * st.slot_bytes;
        tr_model_free(m);

        char err[256];
        tr_model *bad = tr_model_load_budget(path, NULL, min_bytes - 1, err, sizeof err);
        TR_CHECK(bad == NULL);
        TR_CHECK(strstr(err, "MiB") != NULL);
        checked++;
    }
    remove(path);

    uint64_t gib = (uint64_t)1024 * 1024 * 1024, half_gib = gib / 2;
    uint64_t budget;

    /* resident: 90 GiB free, 15 GiB needed, reserve = max(2, 10) GiB -> 75 GiB spare, plenty */
    TR_CHECK(tr_expert_budget_plan(90 * gib, 100 * gib, 5 * gib, 10 * gib, 1 * gib, 1 * gib, &budget) == 0);
    TR_CHECK_EQ_INT(budget, 10 * gib);
    checked++;

    /* partial: only 20 GiB free -> resident (needs 15 + reserve 10 = 25) does not fit; what is
     * left after reserve, dense and the session's own allowance is the budget */
    TR_CHECK(tr_expert_budget_plan(20 * gib, 100 * gib, 5 * gib, 10 * gib, 1 * gib, 1 * gib, &budget) == 0);
    TR_CHECK_EQ_INT(budget, 4 * gib);
    checked++;

    /* refused: same numbers, but the minimum store needs more than the 4 GiB left */
    TR_CHECK(tr_expert_budget_plan(20 * gib, 100 * gib, 5 * gib, 10 * gib, 1 * gib, 5 * gib, &budget) == -1);
    checked++;

    /* the 2 GiB reserve floor, not total / 10: total/10 alone (1 GiB) would still leave the 4.5
     * GiB request resident (4.5 - 3 = 1.5 GiB spare >= 1 GiB); the 2 GiB floor pushes it off
     * resident instead (1.5 GiB spare < 2 GiB), so the budget is the smaller, partial figure */
    TR_CHECK(tr_expert_budget_plan(9 * half_gib, 20 * half_gib, 0, 6 * half_gib, 0, half_gib, &budget) == 0);
    TR_CHECK_EQ_INT(budget, 5 * half_gib);
    checked++;

    TR_CHECK(checked > 0);
}

/* ---- failure: a wrapped reader fails an exact real read, chosen with a deterministic dry run ---- */

typedef struct {
    tr_experts_read_fn real;
    tr_experts_readv_fn realv;
    void *real_ctx;
    int64_t calls, fail_at; /* 1-based over both doors' calls; 0 never fails */
} fail_reader_ctx;

static int fail_reader(void *ctx, void *buf, size_t n, uint64_t offset) {
    fail_reader_ctx *f = (fail_reader_ctx *)ctx;
    f->calls++;
    if (f->calls == f->fail_at) return -1;
    return f->real(f->real_ctx, buf, n, offset);
}

/* a run's requests, counted one by one as the store counts them: the call fails when one of them is the one */
static int fail_readerv(void *ctx, const tr_readv_req *req, int n, void *scratch) {
    fail_reader_ctx *f = (fail_reader_ctx *)ctx;
    int fail = 0;
    for (int k = 0; k < n; k++)
        if (++f->calls == f->fail_at) fail = 1;
    if (fail) return -1;
    return f->realv(f->real_ctx, req, n, scratch);
}

/* How many real reads a fresh min-store model consumes evaluating tokens[0..n) (the store's
 * requests: 3 a lone miss, 3 a run; src/memory/experts.c): deterministic, since the file and the store's starting state are always
 * the same, so this is exactly the read count the same pass will reproduce later. */
static int64_t count_reads(const char *path, int64_t n_batch, const int32_t *tokens, int64_t n) {
    tr_model *m = load_budget(path, NULL, UINT64_MAX);
    if (m == NULL) return -1;
    tr_session *s = create_session(m, n_batch);
    int64_t reads = -1;
    if (s != NULL && tr_session_eval(s, tokens, n) == 0) {
        tr_experts_stats st;
        tr_model_expert_stats(m, &st);
        reads = (int64_t)st.requests; /* both doors' calls: a run is one a part */
    }
    if (s != NULL) tr_session_free(s);
    tr_model_free(m);
    return reads;
}

static void test_failure(const char *argv0) {
    char path[512];
    TR_CHECK(synth_write(&P_F32, argv0, "stream_fail.gguf", path, sizeof path) == 0);
    int32_t tokens[N_PROMPT];
    for (int i = 0; i < N_PROMPT; i++) tokens[i] = prompt_token(i);
    int checked = 0;

    float ref[N_PROMPT][VOCAB];
    {
        tr_model *m = load_budget(path, NULL, UINT64_MAX);
        if (m != NULL) {
            tr_session *s = create_session(m, 0);
            if (s != NULL) {
                for (int i = 0; i < N_PROMPT; i++) {
                    TR_CHECK(tr_session_eval(s, tokens + i, 1) == 0);
                    memcpy(ref[i], tr_session_logits(s), sizeof ref[i]);
                }
                tr_session_free(s);
            }
            tr_model_free(m);
        }
    }

    /* case 1: a one-token pass -- the very first read the store ever needs */
    {
        tr_model *m = load_budget(path, NULL, UINT64_MAX);
        if (m != NULL) {
            tr_session *s = create_session(m, 0);
            if (s != NULL) {
                tr_experts *ex = tr_model_experts(m);
                TR_CHECK(ex != NULL);
                fail_reader_ctx frc = {0};
                if (ex != NULL) {
                    tr_experts_get_reader(ex, &frc.real, &frc.realv, &frc.real_ctx);
                    frc.fail_at = 1;
                    tr_experts_set_reader(ex, fail_reader, frc.realv != NULL ? fail_readerv : NULL, &frc);
                }

                TR_CHECK(tr_session_eval(s, tokens, 1) == -1);
                TR_CHECK_EQ_INT(tr_session_pos(s), 0);
                checked++;

                if (ex != NULL) tr_experts_set_reader(ex, frc.real, frc.realv, frc.real_ctx);
                TR_CHECK(tr_session_eval(s, tokens, 1) == 0);
                TR_CHECK(memcmp(tr_session_logits(s), ref[0], sizeof ref[0]) == 0);
                checked++;

                tr_session_free(s);
            }
            tr_model_free(m);
        }
    }

    /* case 2: the SECOND internal pass of a prompt longer than n_batch */
    {
        enum { N_BATCH = 3 }; /* < N_PROMPT: at least two internal passes */
        int64_t reads1 = count_reads(path, N_BATCH, tokens, N_BATCH);
        TR_CHECK(reads1 > 0);

        tr_model *m = load_budget(path, NULL, UINT64_MAX);
        if (m != NULL) {
            tr_session *s = create_session(m, N_BATCH);
            if (s != NULL) {
                tr_experts *ex = tr_model_experts(m);
                TR_CHECK(ex != NULL);
                fail_reader_ctx frc = {0};
                if (ex != NULL) {
                    tr_experts_get_reader(ex, &frc.real, &frc.realv, &frc.real_ctx);
                    frc.fail_at = reads1 + 1; /* the first read of the second pass */
                    tr_experts_set_reader(ex, fail_reader, frc.realv != NULL ? fail_readerv : NULL, &frc);
                }

                TR_CHECK(tr_session_eval(s, tokens, N_PROMPT) == -1);
                TR_CHECK_EQ_INT(tr_session_pos(s), 0); /* the whole eval, not just the failed pass */
                checked++;

                if (ex != NULL) tr_experts_set_reader(ex, frc.real, frc.realv, frc.real_ctx);
                TR_CHECK(tr_session_eval(s, tokens, N_PROMPT) == 0);
                TR_CHECK(memcmp(tr_session_logits(s), ref[N_PROMPT - 1], sizeof ref[0]) == 0);
                checked++;

                tr_session_free(s);
            }
            tr_model_free(m);
        }
    }

    /* case 3: the same failure in a session that already had logits: the failed eval leaves them
     * as they were (the session is exactly as before the call), not emptied by the passes that ran
     * before the failure. Pass by pass (a route trace keeps it off the layer-major path), so the
     * first pass is the one the dry run counts after the same first token. */
    {
        enum { N_BATCH = 3 };
        int64_t reads = -1;
        tr_model *m = load_budget(path, NULL, UINT64_MAX);
        if (m != NULL) {
            tr_session *s = create_session(m, N_BATCH);
            tr_experts_stats a, b;
            if (s != NULL && tr_session_eval(s, tokens, 1) == 0) {
                tr_model_expert_stats(m, &a);
                if (tr_session_eval(s, tokens + 1, N_BATCH) == 0) {
                    tr_model_expert_stats(m, &b);
                    reads = (int64_t)(b.misses - a.misses) * TR_EXPERT_PARTS;
                }
            }
            if (s != NULL) tr_session_free(s);
            tr_model_free(m);
        }
        TR_CHECK(reads > 0);

        m = load_budget(path, NULL, UINT64_MAX);
        if (m != NULL) {
            tr_session *s = create_session(m, N_BATCH);
            if (s != NULL) {
                TR_CHECK(tr_session_route_trace_begin(s, N_PROMPT) == 0);
                TR_CHECK(tr_session_eval(s, tokens, 1) == 0);
                tr_experts *ex = tr_model_experts(m);
                fail_reader_ctx frc = {0};
                if (ex != NULL) {
                    tr_experts_get_reader(ex, &frc.real, &frc.realv, &frc.real_ctx);
                    frc.fail_at = reads + 1; /* the first read of the second pass */
                    tr_experts_set_reader(ex, fail_reader, frc.realv != NULL ? fail_readerv : NULL, &frc);
                }
                TR_CHECK(tr_session_eval(s, tokens + 1, N_PROMPT - 1) == -1);
                TR_CHECK_EQ_INT(tr_session_pos(s), 1);
                const float *kept = tr_session_logits(s);
                TR_CHECK(kept != NULL && memcmp(kept, ref[0], sizeof ref[0]) == 0);
                checked++;
                if (ex != NULL) tr_experts_set_reader(ex, frc.real, frc.realv, frc.real_ctx);
                tr_session_free(s);
            }
            tr_model_free(m);
        }
    }

    TR_CHECK(checked > 0);
    remove(path);
}

/* ---- rewind: the store has no notion of position ---- */

/* ---- once_per_prompt: a prompt longer than n_batch reads every unit ONCE, not once per
 * internal pass. The prompt runs in passes of n_batch tokens and each pass walked every layer,
 * so under a store too small to hold the model each pass read the whole table again (3.5x the
 * bytes on the real model at 2048 tokens: docs/MEASUREMENTS.md domanda 47, docs/LESSONS.md #99). The
 * layer-major order of the prefill -- every pass of one layer, then the next layer -- reads each
 * unit once per prompt instead. Exercises that order; without it the count is passes x units. */
static void test_once_per_prompt(const char *argv0) {
    char path[512];
    TR_CHECK(synth_write(&P_F32, argv0, "stream_once.gguf", path, sizeof path) == 0);
    /* Long enough that a single pass already asks for (nearly) every expert of a layer, like a
     * real prompt does: otherwise a pass touches a handful of units and re-reading them costs
     * too little to show. 36 tokens in passes of 12 = three passes. */
    enum { N_LONG = 36, N_BATCH = 12 };
    int32_t tokens[N_LONG];
    for (int i = 0; i < N_LONG; i++) tokens[i] = prompt_token(i);
    int checked = 0;
    tr_model *m = load_budget(path, NULL, UINT64_MAX); /* the smallest store: a layer at a time */
    if (m != NULL) {
        tr_session *s = create_session(m, N_BATCH);
        tr_experts_stats before, after;
        if (s != NULL && tr_model_expert_stats(m, &before) == 0) {
            TR_CHECK(tr_session_eval(s, tokens, N_LONG) == 0);
            TR_CHECK(tr_model_expert_stats(m, &after) == 0);
            uint64_t read = after.misses - before.misses;
            /* the branch really ran: the store had to fetch something */
            TR_CHECK(read > 0);
            /* every unit at most once for the whole prompt, plus one store's worth of slack */
            TR_CHECK(read <= (uint64_t)after.n_units + (uint64_t)after.n_slots);
            if (read > (uint64_t)after.n_units + (uint64_t)after.n_slots)
                fprintf(stderr, "  %llu units read for %d tokens in passes of %d, %lld in the table\n",
                        (unsigned long long)read, (int)N_LONG, (int)N_BATCH, (long long)after.n_units);
            checked++;
        }
        if (s != NULL) tr_session_free(s);
        tr_model_free(m);
    }
    TR_CHECK(checked > 0);
    remove(path);
}

static void test_rewind(const char *argv0) {
    char path[512];
    TR_CHECK(synth_write(&P_F32, argv0, "stream_rewind.gguf", path, sizeof path) == 0);
    int32_t tokens[N_PROMPT];
    for (int i = 0; i < N_PROMPT; i++) tokens[i] = prompt_token(i);
    int checked = 0;

    tr_model *m = load_budget(path, NULL, UINT64_MAX);
    if (m != NULL) {
        tr_session *s = create_session(m, 0);
        if (s != NULL) {
            float first[N_PROMPT][VOCAB];
            for (int i = 0; i < N_PROMPT; i++) {
                TR_CHECK(tr_session_eval(s, tokens + i, 1) == 0);
                memcpy(first[i], tr_session_logits(s), sizeof first[i]);
            }
            TR_CHECK(tr_session_rewind(s, 0) == 0);
            for (int i = 0; i < N_PROMPT; i++) {
                TR_CHECK(tr_session_eval(s, tokens + i, 1) == 0);
                TR_CHECK(memcmp(tr_session_logits(s), first[i], sizeof first[i]) == 0);
                checked++;
            }
            tr_session_free(s);
        }
        tr_model_free(m);
    }
    TR_CHECK(checked > 0);
    remove(path);
}

/* ---- direct: the unbuffered path (docs/ARCHITECTURE.md Esperti M1, Step C) gives the same
 * logits as the buffered one, if this filesystem can actually serve it ---- */

static void test_direct(const char *argv0) {
    char path[512];
    TR_CHECK(synth_write(&P_F32, argv0, "stream_direct.gguf", path, sizeof path) == 0);
    int32_t tokens[N_PROMPT];
    for (int i = 0; i < N_PROMPT; i++) tokens[i] = prompt_token(i);
    int checked = 0;

    /* olmoe_load probes tr_file_open_direct itself before trusting it (a trial aligned read):
     * ask the same question here first, so a filesystem that refuses it (docker bind mounts, in
     * particular) skips this test with a message instead of silently comparing buffered to
     * buffered, which would prove nothing. */
    tr_file *probe_file = tr_file_open_direct(path, NULL, 0);
    int direct_ok = 0;
    if (probe_file != NULL) {
        unsigned char *probe = (unsigned char *)tr_alloc_aligned(TR_FILE_DIRECT_ALIGN, TR_FILE_DIRECT_ALIGN);
        direct_ok = probe != NULL && tr_file_pread(probe_file, probe, TR_FILE_DIRECT_ALIGN, 0) == 0;
        tr_free_aligned(probe);
        tr_file_close(probe_file);
    }
    if (!direct_ok) {
        printf("  test_direct: SKIPPED, tr_file_open_direct cannot read '%s' on this filesystem\n", path);
        remove(path);
        return;
    }
    printf("  test_direct: this filesystem serves aligned reads, comparing buffered and direct\n");

    set_env("TR_EXPERT_DIRECT", "0");
    float ref_resident[N_PROMPT][VOCAB], ref_min[N_PROMPT][VOCAB];
    {
        tr_model *m = load_budget(path, NULL, HUGE_BUDGET);
        if (m != NULL) {
            tr_experts_stats st;
            TR_CHECK(tr_model_expert_stats(m, &st) == 0);
            TR_CHECK(!st.direct);
            tr_session *s = create_session(m, 0);
            if (s != NULL) {
                TR_CHECK(tr_session_eval_rows(s, tokens, N_PROMPT, N_PROMPT) == 0);
                for (int i = 0; i < N_PROMPT; i++)
                    memcpy(ref_resident[i], tr_session_logits_back(s, N_PROMPT - 1 - i), sizeof ref_resident[i]);
                tr_session_free(s);
                checked++;
            }
            tr_model_free(m);
        }
    }
    {
        tr_model *m = load_budget(path, NULL, UINT64_MAX); /* "min" */
        if (m != NULL) {
            tr_experts_stats st;
            TR_CHECK(tr_model_expert_stats(m, &st) == 0);
            TR_CHECK(!st.direct);
            tr_session *s = create_session(m, 0);
            if (s != NULL) {
                TR_CHECK(tr_session_eval_rows(s, tokens, N_PROMPT, N_PROMPT) == 0);
                for (int i = 0; i < N_PROMPT; i++)
                    memcpy(ref_min[i], tr_session_logits_back(s, N_PROMPT - 1 - i), sizeof ref_min[i]);
                tr_session_free(s);
                checked++;
            }
            tr_model_free(m);
        }
    }

    set_env("TR_EXPERT_DIRECT", NULL); /* the default: try direct first */
    {
        tr_model *m = load_budget(path, NULL, HUGE_BUDGET);
        if (m != NULL) {
            tr_experts_stats st;
            TR_CHECK(tr_model_expert_stats(m, &st) == 0);
            TR_CHECK(st.direct); /* the probe above already proved this filesystem allows it */
            tr_session *s = create_session(m, 0);
            if (s != NULL) {
                TR_CHECK(tr_session_eval_rows(s, tokens, N_PROMPT, N_PROMPT) == 0);
                for (int i = 0; i < N_PROMPT; i++)
                    TR_CHECK(memcmp(tr_session_logits_back(s, N_PROMPT - 1 - i), ref_resident[i], sizeof ref_resident[i]) ==
                             0);
                tr_session_free(s);
                checked++;
            }
            tr_model_free(m);
        }
    }
    {
        tr_model *m = load_budget(path, NULL, UINT64_MAX);
        if (m != NULL) {
            tr_experts_stats st;
            TR_CHECK(tr_model_expert_stats(m, &st) == 0);
            TR_CHECK(st.direct);
            tr_session *s = create_session(m, 0);
            if (s != NULL) {
                TR_CHECK(tr_session_eval_rows(s, tokens, N_PROMPT, N_PROMPT) == 0);
                for (int i = 0; i < N_PROMPT; i++)
                    TR_CHECK(memcmp(tr_session_logits_back(s, N_PROMPT - 1 - i), ref_min[i], sizeof ref_min[i]) == 0);
                tr_session_free(s);
                checked++;
            }
            tr_model_free(m);
        }
    }

    TR_CHECK(checked > 0);
    remove(path);
}

/* ---- mem_available: TR_MEM_AVAILABLE_MIB (docs/ARCHITECTURE.md Esperti M1, Step D) ---- */

/* Big enough that its expert weights alone (F32, not Q8_0: at this size Q8_0's per-byte cost in
 * synth_olmoe.h would dominate the test) clear 512 MiB, the margin the window below keeps (the plan's
 * flat session floor until 2026-10-03; now the session's own bytes, olmoe.c session_bytes), with room
 * to spare: only then does a
 * window of RAM exist that is enough to run partially but not enough to be fully resident, for
 * TR_MEM_AVAILABLE_MIB to force. n_layers = 32, n_expert = 8: min_slots (8 + 2 = 10) is a small
 * fraction (~3.9%) of n_units (256), which is what keeps that window open (the window's width is
 * resident_bytes * (1 - min_slots/n_units) - 512 MiB, independent of the real machine's own RAM). */
enum { BIG_LAYERS = 32, BIG_EMBD = 640, BIG_HEAD = 8, BIG_HEAD_KV = 2, BIG_FF = 320, BIG_EXPERT = 8,
       BIG_USED = 2, BIG_VOCAB = 64, BIG_CTX = 64 };
static const synth_params P_BIG = {BIG_LAYERS, BIG_EMBD, BIG_HEAD, BIG_HEAD_KV, BIG_FF, BIG_EXPERT,
                                    BIG_USED, BIG_VOCAB, BIG_CTX, TR_TYPE_F32};

static void test_mem_available(const char *argv0) {
    char path[512];
    TR_CHECK(synth_write(&P_BIG, argv0, "stream_mem_available.gguf", path, sizeof path) == 0);
    enum { N_TOK = 4 };
    int32_t tokens[N_TOK];
    for (int i = 0; i < N_TOK; i++) tokens[i] = prompt_token(i);
    int checked = 0;

    /* dense_bytes, resident_bytes and min_bytes: derived from two real loads (never
     * hand-recomputed from the tensor list), like test_budget_gate derives min_bytes. */
    uint64_t dense_bytes = 0, resident_bytes = 0, min_bytes = 0;
    float ref[N_TOK][BIG_VOCAB];
    {
        tr_model *m = load_budget(path, NULL, UINT64_MAX); /* "min" */
        if (m != NULL) {
            tr_experts_stats st;
            TR_CHECK(tr_model_expert_stats(m, &st) == 0);
            min_bytes = (uint64_t)st.n_slots * st.slot_bytes;
            dense_bytes = tr_model_get_info(m)->weight_bytes - min_bytes;
            tr_model_free(m);
        }
    }
    {
        tr_model *m = load_budget(path, NULL, HUGE_BUDGET); /* resident: also the reference logits */
        if (m != NULL) {
            resident_bytes = tr_model_get_info(m)->weight_bytes - dense_bytes;
            tr_session *s = create_session(m, 0);
            if (s != NULL) {
                TR_CHECK(tr_session_eval_rows(s, tokens, N_TOK, N_TOK) == 0);
                for (int i = 0; i < N_TOK; i++)
                    memcpy(ref[i], tr_session_logits_back(s, N_TOK - 1 - i), sizeof ref[i]);
                tr_session_free(s);
                checked++;
            }
            tr_model_free(m);
        }
    }
    TR_CHECK(dense_bytes > 0 && min_bytes > 0 && resident_bytes > min_bytes);

    /* too small: refuses, with the minimum named in MiB (Step D says the plan sees `available`
     * in place of the real one; the message comes from the same "not enough memory" path as an
     * actually short machine) */
    set_env("TR_MEM_AVAILABLE_MIB", "1");
    {
        char err[256];
        tr_model *bad = tr_model_load_budget(path, NULL, 0, err, sizeof err);
        TR_CHECK(bad == NULL);
        TR_CHECK(strstr(err, "MiB") != NULL);
        if (bad != NULL) tr_model_free(bad);
        checked++;
    }

    /* the window: [reserve + dense + 512 MiB + min_bytes, reserve + dense + resident_bytes).
     * session_allowance is the session's own bytes (olmoe.c session_bytes: a few MiB at this context
     * length), so the window's middle leaves the store (512 MiB + min + resident) / 2 minus them: more
     * than the minimum and less than resident, since resident > 512 MiB + min (BIG_* sized for it).
     * reserve depends on the real machine's total RAM (tr_mem_guard's own
     * rule, model.c tr_mem_guard / tr_expert_budget_plan), read here only to place the value the
     * plan will see; the guard downstream still checks the real available RAM (Step D). */
    uint64_t mib = 1024 * 1024, gib = 1024 * mib, two_gib = 2 * gib, floor_512mib = 512 * mib;
    TR_CHECK(resident_bytes > floor_512mib + min_bytes); /* BIG_* was sized to guarantee this */
    if (resident_bytes > floor_512mib + min_bytes) {
        tr_meminfo mi;
        TR_CHECK(tr_mem_info(&mi) == 0);
        if (mi.total_bytes > 0) {
            uint64_t reserve = mi.total_bytes / 10;
            if (reserve < two_gib) reserve = two_gib;
            uint64_t lo_mib = (reserve + dense_bytes + floor_512mib + min_bytes) / mib + 1;
            uint64_t hi_mib = (reserve + dense_bytes + resident_bytes) / mib;
            TR_CHECK(hi_mib > lo_mib); /* the window this model was sized to open */
            printf("  test_mem_available: dense %.1f MiB, min %.1f MiB, resident %.1f MiB, window "
                   "[%llu, %llu) MiB\n",
                   (double)dense_bytes / mib, (double)min_bytes / mib, (double)resident_bytes / mib,
                   (unsigned long long)lo_mib, (unsigned long long)hi_mib);
            if (hi_mib > lo_mib) {
                char avail[32];
                snprintf(avail, sizeof avail, "%llu", (unsigned long long)((lo_mib + hi_mib) / 2));
                set_env("TR_MEM_AVAILABLE_MIB", avail);

                tr_model *m = load_budget(path, NULL, 0); /* automatic */
                if (m != NULL) {
                    tr_experts_stats st;
                    TR_CHECK(tr_model_expert_stats(m, &st) == 0);
                    uint64_t slab_bytes = (uint64_t)st.n_slots * st.slot_bytes;
                    TR_CHECK(slab_bytes > min_bytes);       /* more than the bare minimum forced it gave */
                    TR_CHECK(slab_bytes < resident_bytes);  /* proves the override forced partial, not luck */
                    tr_session *s = create_session(m, 0);
                    if (s != NULL) {
                        TR_CHECK(tr_session_eval_rows(s, tokens, N_TOK, N_TOK) == 0);
                        for (int i = 0; i < N_TOK; i++)
                            TR_CHECK(memcmp(tr_session_logits_back(s, N_TOK - 1 - i), ref[i], sizeof ref[i]) == 0);
                        tr_session_free(s);
                        checked++;
                    }
                    tr_model_free(m);
                }
            }
        }
    }

    /* TR_MEM_TOTAL_MIB: the plan's reserve is a tenth of the total it names. 1 TiB: a reserve of ~102 GiB,
     * so the window placed with it loads partial, where the real total (any machine under ~1000 GiB)
     * would leave the same available resident: only the override's branch passes */
    int total_seen = 0;
    if (dense_bytes > 0 && resident_bytes > floor_512mib + min_bytes) {
        uint64_t reserve = 1024 * gib / 10;
        uint64_t lo_mib = (reserve + dense_bytes + floor_512mib + min_bytes) / mib + 1;
        uint64_t hi_mib = (reserve + dense_bytes + resident_bytes) / mib;
        TR_CHECK(hi_mib > lo_mib);
        if (hi_mib > lo_mib) {
            char avail[32];
            snprintf(avail, sizeof avail, "%llu", (unsigned long long)((lo_mib + hi_mib) / 2));
            set_env("TR_MEM_AVAILABLE_MIB", avail);
            set_env("TR_MEM_TOTAL_MIB", "1048576");
            tr_model *m = load_budget(path, NULL, 0); /* automatic */
            TR_CHECK(m != NULL);
            if (m != NULL) {
                tr_experts_stats st;
                TR_CHECK(tr_model_expert_stats(m, &st) == 0);
                uint64_t slab_bytes = (uint64_t)st.n_slots * st.slot_bytes;
                TR_CHECK(slab_bytes > min_bytes);
                TR_CHECK(slab_bytes < resident_bytes); /* the emulated total's reserve, not the real one's */
                printf("  test_mem_available: TR_MEM_TOTAL_MIB 1 TiB, available %s MiB: %.1f MiB of slots\n", avail,
                       (double)slab_bytes / mib);
                tr_model_free(m);
                total_seen++;
            }
            set_env("TR_MEM_TOTAL_MIB", NULL);
        }
    }
    TR_CHECK(total_seen > 0);

    /* tr_model_load_plan: the plan sets aside the session of plan_ctx positions, not the default context's
     * (64 here, the model's trained length). At the same available RAM a 4-position session leaves the
     * experts more slots and a 1024-position one fewer, and the 4-position store's logits are the
     * resident's to the bit */
    int plan_seen = 0, kv_room_seen = 0;
    if (dense_bytes > 0 && resident_bytes > floor_512mib + min_bytes) {
        tr_meminfo mi;
        TR_CHECK(tr_mem_info(&mi) == 0);
        uint64_t reserve = mi.total_bytes / 10;
        if (reserve < two_gib) reserve = two_gib;
        uint64_t lo_mib = (reserve + dense_bytes + floor_512mib + min_bytes) / mib + 1;
        uint64_t hi_mib = (reserve + dense_bytes + resident_bytes) / mib;
        if (mi.total_bytes > 0 && hi_mib > lo_mib) {
            char avail[32];
            snprintf(avail, sizeof avail, "%llu", (unsigned long long)((lo_mib + hi_mib) / 2));
            set_env("TR_MEM_AVAILABLE_MIB", avail);
            static const int64_t plans[3] = {4, 0, 1024};
            int64_t slots[3] = {0, 0, 0};
            uint64_t slot_bytes = 0;
            int same = 0;
            for (int k = 0; k < 3; k++) {
                char err[256];
                tr_model *m = tr_model_load_plan(path, NULL, 0, plans[k], NULL, err, sizeof err);
                TR_CHECK(m != NULL);
                if (m == NULL) continue;
                tr_experts_stats st;
                TR_CHECK(tr_model_expert_stats(m, &st) == 0);
                slots[k] = st.n_slots;
                slot_bytes = st.slot_bytes;
                TR_CHECK_EQ_INT(st.evict, TR_EXPERTS_EVICT_HOT); /* ds4's eviction, the default */
                if (k == 0) {
                    tr_session *s = tr_session_create(m, plans[0], 0, err, sizeof err);
                    TR_CHECK(s != NULL);
                    if (s != NULL) {
                        TR_CHECK(tr_session_eval_rows(s, tokens, N_TOK, N_TOK) == 0);
                        same = 1;
                        for (int i = 0; i < N_TOK; i++)
                            same &= memcmp(tr_session_logits_back(s, N_TOK - 1 - i), ref[i], sizeof ref[i]) == 0;
                        TR_CHECK(same);
                        tr_session_free(s);
                    }
                }
                tr_model_free(m);
            }
            printf("  test_mem_available: plans of 4, the default and 1024 positions: %lld, %lld, %lld slots\n",
                   (long long)slots[0], (long long)slots[1], (long long)slots[2]);
            /* from 64 positions to 1024 the KV takes this many slots more; the plan no longer sets it aside (the
             * store gives its slots back as the KV grows, kv_room below): only the working memory differs */
            uint64_t kv_more = tr_kv_bytes(BIG_LAYERS, BIG_HEAD_KV, BIG_EMBD / BIG_HEAD, 1024) -
                               tr_kv_bytes(BIG_LAYERS, BIG_HEAD_KV, BIG_EMBD / BIG_HEAD, BIG_CTX);
            int64_t kv_slots = slot_bytes > 0 ? (int64_t)(kv_more / slot_bytes) : 0;
            TR_CHECK(kv_slots > 2);
            TR_CHECK(slots[0] >= slots[1] && slots[1] >= slots[2]);
            TR_CHECK(slots[1] - slots[2] < kv_slots);
            if (same && slots[0] >= slots[1] && kv_slots > 2 && slots[1] - slots[2] < kv_slots) plan_seen++;
            /* TR_EXPERT_EVICT=lru: the LRU, for the measurement that compares them */
            set_env("TR_EXPERT_EVICT", "lru");
            char err[256];
            tr_model *m = tr_model_load_plan(path, NULL, 0, 4, NULL, err, sizeof err);
            TR_CHECK(m != NULL);
            if (m != NULL) {
                tr_experts_stats st;
                TR_CHECK(tr_model_expert_stats(m, &st) == 0);
                TR_CHECK_EQ_INT(st.evict, TR_EXPERTS_EVICT_LRU);
                tr_model_free(m);
            }
            set_env("TR_EXPERT_EVICT", NULL);

            /* kv_room: the KV's pages from the store's room (olmoe.c kv_hold). A session of 1024 positions written
             * 32 at a time: after each pass the store has given back the slots of the KV's pages so far (floor or
             * ceil of them: the room is the slots made and less than one more), never one taken again, the
             * logits a resident store's to the bit; freed, every slot taken again. A context whose KV would
             * leave the store under its minimum is refused, named */
            enum { KV_CHUNK = 32, KV_CHUNKS = 8, KV_CTX = 1024 };
            int32_t kv_toks[KV_CHUNK * KV_CHUNKS];
            for (int i = 0; i < KV_CHUNK * KV_CHUNKS; i++) kv_toks[i] = prompt_token(i);
            static float kv_ref[KV_CHUNKS][BIG_VOCAB];
            int have_ref = 0;
            tr_model *r = load_budget(path, NULL, HUGE_BUDGET);
            if (r != NULL) {
                tr_session *s = tr_session_create(r, KV_CTX, 0, err, sizeof err);
                TR_CHECK(s != NULL);
                if (s != NULL) {
                    have_ref = 1;
                    for (int c = 0; c < KV_CHUNKS; c++) {
                        have_ref &= tr_session_eval_rows(s, kv_toks + c * KV_CHUNK, KV_CHUNK, 1) == 0;
                        memcpy(kv_ref[c], tr_session_logits_back(s, 0), sizeof kv_ref[c]);
                    }
                    tr_session_free(s);
                }
                tr_experts_stats st;
                TR_CHECK(tr_model_expert_stats(r, &st) == 0);
                TR_CHECK_EQ_INT(st.slots_given, 0); /* a forced budget shares no room */
                tr_model_free(r);
            }
            TR_CHECK(have_ref);
            tr_model *km = tr_model_load_plan(path, NULL, 0, KV_CTX, NULL, err, sizeof err);
            TR_CHECK(km != NULL);
            if (km != NULL && have_ref) {
                tr_experts_stats st0, st;
                TR_CHECK(tr_model_expert_stats(km, &st0) == 0);
                TR_CHECK(st0.n_slots_made < st0.n_units && st0.n_slots == st0.n_slots_made);
                tr_session *s = tr_session_create(km, KV_CTX, 0, err, sizeof err);
                TR_CHECK(s != NULL);
                int same_kv = 1, given_ok = 1, falling = 1;
                memset(&st, 0, sizeof st);
                if (s != NULL) {
                    int64_t last = st0.n_slots;
                    for (int c = 0; c < KV_CHUNKS; c++) {
                        TR_CHECK(tr_session_eval_rows(s, kv_toks + c * KV_CHUNK, KV_CHUNK, 1) == 0);
                        same_kv &= memcmp(tr_session_logits_back(s, 0), kv_ref[c], sizeof kv_ref[c]) == 0;
                        TR_CHECK(tr_model_expert_stats(km, &st) == 0);
                        uint64_t held = tr_kv_bytes(BIG_LAYERS, BIG_HEAD_KV, BIG_EMBD / BIG_HEAD,
                                                    (int64_t)(c + 1) * KV_CHUNK);
                        uint64_t given = (uint64_t)(st.n_slots_made - st.n_slots);
                        given_ok &= given >= held / st.slot_bytes && given <= (held + st.slot_bytes - 1) / st.slot_bytes;
                        falling &= st.n_slots <= last && st.slots_taken == 0;
                        last = st.n_slots;
                    }
                    TR_CHECK(same_kv);
                    TR_CHECK(given_ok);
                    TR_CHECK(falling);
                    TR_CHECK(st.slots_given >= 2);
                    printf("  test_mem_available: kv_room, %lld slots made, %llu given to 256 positions' KV, %llu "
                           "units moved\n",
                           (long long)st.n_slots_made, (unsigned long long)st.slots_given,
                           (unsigned long long)st.moved);
                    tr_session_free(s);
                }
                tr_experts_stats st1;
                TR_CHECK(tr_model_expert_stats(km, &st1) == 0);
                TR_CHECK_EQ_INT(st1.n_slots, st1.n_slots_made); /* freed: every slot taken again */
                TR_CHECK_EQ_INT(st1.slots_taken, st1.slots_given);
                tr_session *big = tr_session_create(km, (int64_t)1 << 24, 0, err, sizeof err);
                int refused = big == NULL && strstr(err, "context of") != NULL;
                TR_CHECK(refused);
                if (big != NULL) tr_session_free(big);
                if (same_kv && given_ok && falling && st.slots_given >= 2 && st1.n_slots == st1.n_slots_made &&
                    refused)
                    kv_room_seen++;
            }
            if (km != NULL) tr_model_free(km);
        }
    }
    TR_CHECK(plan_seen > 0);
    TR_CHECK(kv_room_seen > 0);

    set_env("TR_MEM_AVAILABLE_MIB", NULL);
    TR_CHECK(checked > 0);
    remove(path);
}

/* ---- progress: what tr_model_load_progress reports, against the file's own tensor sizes ---- */

typedef struct {
    int64_t calls;
    uint64_t done, total, first_total;
    int monotone, same_total;
} progress_log;

static void progress_record(void *ctx, uint64_t done, uint64_t total) {
    progress_log *lg = (progress_log *)ctx;
    if (lg->calls == 0) lg->first_total = total;
    else if (done <= lg->done) lg->monotone = 0;
    if (total != lg->first_total) lg->same_total = 0;
    lg->done = done;
    lg->total = total;
    lg->calls++;
}

static void test_progress(const char *argv0) {
    char path[512], err[256];
    TR_CHECK(synth_write(&P_Q8_0, argv0, "stream_progress.gguf", path, sizeof path) == 0);
    int checked = 0;

    /* the file's own split, read straight off its directory: dense tensors and expert tensors */
    uint64_t dense = 0, experts = 0;
    int64_t n_dense = 0;
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g != NULL);
    if (g != NULL) {
        for (uint64_t i = 0; i < tr_gguf_tensor_count(g); i++) {
            const tr_gguf_tensor *t = tr_gguf_tensor_at(g, i);
            if (strstr(t->name, "_exps.") != NULL) {
                experts += t->n_bytes;
            } else {
                dense += t->n_bytes;
                n_dense++;
            }
        }
        tr_gguf_close(g);
    }
    TR_CHECK(dense > 0 && experts > 0);

    /* resident (budget above every unit): every dense tensor and every expert unit, one report each */
    progress_log lg = {0, 0, 0, 0, 1, 1};
    tr_progress pr = {progress_record, &lg};
    tr_model *m = tr_model_load_progress(path, NULL, HUGE_BUDGET, &pr, err, sizeof err);
    TR_CHECK(m != NULL);
    if (m != NULL) {
        TR_CHECK_EQ_INT(lg.first_total, dense + experts);
        TR_CHECK(lg.same_total);
        TR_CHECK(lg.monotone);
        TR_CHECK_EQ_INT(lg.done, lg.total); /* the last report lands on the total */
        TR_CHECK_EQ_INT(lg.calls, n_dense + N_UNITS);
        checked++;
        tr_model_free(m);
    }

    /* partial (the smallest store, 10 of 16 slots): the experts are read while running, so the
     * load reports the dense tensors only */
    progress_log lp = {0, 0, 0, 0, 1, 1};
    pr.ctx = &lp;
    m = tr_model_load_progress(path, NULL, UINT64_MAX, &pr, err, sizeof err);
    TR_CHECK(m != NULL);
    if (m != NULL) {
        tr_experts_stats st;
        TR_CHECK(tr_model_expert_stats(m, &st) == 0);
        TR_CHECK(st.n_slots < st.n_units); /* really partial */
        TR_CHECK_EQ_INT(lp.first_total, dense);
        TR_CHECK(lp.same_total);
        TR_CHECK(lp.monotone);
        TR_CHECK_EQ_INT(lp.done, lp.total);
        TR_CHECK_EQ_INT(lp.calls, n_dense);
        checked++;
        tr_model_free(m);
    }

    /* no progress asked for: the same load, nothing called */
    m = tr_model_load_progress(path, NULL, HUGE_BUDGET, NULL, err, sizeof err);
    TR_CHECK(m != NULL);
    tr_model_free(m);
    remove(path);
    TR_CHECK_EQ_INT(checked, 2);
}

/* ---- arrival: a layer's misses read by the store's I/O thread while its present experts compute ---- */

/* 6 layers of 8 experts, 4 used: 48 units, the I/O thread from 2 x 8 + 4 = 20 slots (tr_experts_prefetch_start) */
enum { AR_LAYERS = 6, AR_USED = 4, AR_ARMS = 3 };
static const char *const AR_ARM_NAME[AR_ARMS] = {"arrival", "two waves", "on the calling thread"};
static int64_t g_ar_arrived, g_ar_waves2, g_ar_same_bytes;

static void test_arrival_one(const char *argv0, tr_type type, int64_t embd, int64_t ff, const char *file) {
    const synth_params P = {AR_LAYERS, embd, N_HEAD, N_HEAD_KV, ff, N_EXPERT, AR_USED, VOCAB, CTX, type};
    char path[512];
    TR_CHECK(synth_write(&P, argv0, file, path, sizeof path) == 0);
    int32_t tokens[N_PROMPT];
    for (int i = 0; i < N_PROMPT; i++) tokens[i] = prompt_token(i);

    float ref[N_PROMPT][VOCAB];
    uint64_t slot_bytes = 0;
    tr_model *m = load_budget(path, NULL, HUGE_BUDGET);
    tr_session *s = create_session(m, 0);
    if (s != NULL) {
        for (int i = 0; i < N_PROMPT; i++) {
            TR_CHECK(tr_session_eval(s, tokens + i, 1) == 0); /* one-token passes, resident: the reference */
            memcpy(ref[i], tr_session_logits(s), sizeof ref[i]);
        }
        tr_experts_stats st;
        TR_CHECK(tr_model_expert_stats(m, &st) == 0);
        slot_bytes = st.slot_bytes;
        tr_session_free(s);
    }
    tr_model_free(m);
    if (slot_bytes == 0) return;

    /* each budget, each arm on a fresh model: the decode's logits and a whole prompt's rows the reference's, and
     * the store's reads the same in every arm (the arrival order moves no byte) */
    static const int64_t slots_list[] = {2 * N_EXPERT + AR_USED, 2 * N_EXPERT + AR_USED + 6};
    tr_pool *pool = tr_pool_create(4);
    for (size_t b = 0; b < sizeof slots_list / sizeof slots_list[0]; b++) {
        tr_experts_stats arm_st[AR_ARMS];
        memset(arm_st, 0, sizeof arm_st);
        for (int arm = 0; arm < AR_ARMS; arm++) {
            m = load_budget(path, pool, (uint64_t)slots_list[b] * slot_bytes);
            s = create_session(m, 0);
            if (s == NULL) {
                tr_model_free(m);
                continue;
            }
            int sw_arrive = tr_session_ab_switch(s, "arrive"), sw_waves = tr_session_ab_switch(s, "waves");
            TR_CHECK(sw_arrive > 0 && sw_waves > 0);
            tr_session_ab_set(s, sw_arrive, arm == 2);
            tr_session_ab_set(s, sw_waves, arm == 1);
            int same = 1;
            for (int i = 0; i < N_PROMPT; i++) {
                TR_CHECK(tr_session_eval(s, tokens + i, 1) == 0);
                same &= memcmp(tr_session_logits(s), ref[i], sizeof ref[i]) == 0;
            }
            TR_CHECK(tr_session_rewind(s, 0) == 0);
            TR_CHECK(tr_session_eval_rows(s, tokens, N_PROMPT, N_PROMPT) == 0); /* several rows a group */
            for (int i = 0; i < N_PROMPT; i++)
                same &= memcmp(tr_session_logits_back(s, N_PROMPT - 1 - i), ref[i], sizeof ref[i]) == 0;
            if (!same)
                fprintf(stderr, "  arrival: %s, %lld slots: logits differ\n", AR_ARM_NAME[arm], (long long)slots_list[b]);
            TR_CHECK(same);
            TR_CHECK(tr_model_expert_stats(m, &arm_st[arm]) == 0);
            TR_CHECK_EQ_INT(arm_st[arm].n_slots, slots_list[b]);
            if (arm < 2) TR_CHECK(arm_st[arm].arrived > 0); /* the branch ran: misses read under the compute */
            else TR_CHECK_EQ_INT(arm_st[arm].arrived, 0);
            if (same && arm == 0 && arm_st[arm].arrived > 0) g_ar_arrived++;
            if (same && arm == 1 && arm_st[arm].arrived > 0) g_ar_waves2++;
            tr_session_free(s);
            tr_model_free(m);
        }
        for (int arm = 1; arm < AR_ARMS; arm++) {
            TR_CHECK_EQ_INT(arm_st[arm].misses, arm_st[0].misses);
            TR_CHECK_EQ_INT(arm_st[arm].evictions, arm_st[0].evictions);
            TR_CHECK_EQ_INT(arm_st[arm].bytes_read, arm_st[0].bytes_read);
        }
        if (arm_st[0].misses > 0 && arm_st[0].bytes_read == arm_st[2].bytes_read) g_ar_same_bytes++;
    }
    tr_pool_destroy(pool);
    remove(path);
}

static void test_arrival(const char *argv0) {
    test_arrival_one(argv0, TR_TYPE_Q4_K, 256, 512, "stream_arrival_q4k.gguf"); /* Q4_K's integer roads */
    test_arrival_one(argv0, TR_TYPE_F32, 64, 128, "stream_arrival_f32.gguf");  /* the roads that read floats */
    TR_CHECK(g_ar_arrived > 0);
    TR_CHECK(g_ar_waves2 > 0);
    TR_CHECK(g_ar_same_bytes > 0);
}

int main(int argc, char **argv) {
    const char *argv0 = argc > 0 ? argv[0] : "";
    test_content(argv0);
    test_reference(argv0);
    test_resident(argv0);
    test_budget_gate(argv0);
    test_failure(argv0);
    test_once_per_prompt(argv0);
    test_rewind(argv0);
    test_direct(argv0);
    test_mem_available(argv0);
    test_progress(argv0);
    test_arrival(argv0);
    TR_TEST_EXIT();
}
