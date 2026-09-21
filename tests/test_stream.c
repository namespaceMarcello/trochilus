/* test_stream.c — the OLMoE engine takes its experts from the shared store (src/memory/experts.h)
 * instead of holding every one of them resident, and gives byte-identical logits whatever the
 * budget (docs/ARCHITETTURA.md Esperti M1, lot 2).
 *
 * Every check forces its budget explicitly (never automatic: automatic depends on the machine's
 * free RAM, and this suite must give the same answer everywhere). Synthetic models from
 * tests/synth_olmoe.h: 2 layers, 8 experts, 2 used (n_units = 16, min_slots = 8 + 2 = 10), F32
 * and Q8_0, deterministic weights (same params -> the same file, every time), so a "dry run" of
 * a pass tells us exactly how many real reads that same pass needs again, which lets a failure
 * be pointed at an exact read without depending on what the router actually chose.
 *
 * Each test says which branch it exercises and counts that it was actually taken (CLAUDE.md,
 * LEZIONI #43 #50 #54 #78):
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
 *                eval call; the reader restored, the same eval retried gives the reference logits
 *   once_per_prompt  a prompt of 36 tokens in passes of 12, on the smallest store: the units
 *                read must be at most n_units + n_slots, not passes x n_units
 *   rewind       rewind to 0 and re-evaluate the same tokens under the min store: identical
 *                logits (the store has no notion of position)
 *   direct       a real model loaded with the direct path (docs/ARCHITETTURA.md Esperti M1, Step
 *                C) gives logits byte-identical to TR_EXPERT_DIRECT=0's buffered path, resident
 *                and at --expert-budget min; skipped, with a message, if tr_file_open_direct
 *                cannot actually read this filesystem
 *   mem_available TR_MEM_AVAILABLE_MIB (Step D) too small refuses with the minimum named in MiB;
 *                a value placed inside the window a bigger synthetic model opens between "not
 *                enough to be fully resident" and "not enough to run at all" forces the plan's
 *                partial branch on a real load, logits still identical to a forced-resident one
 *
 * Seen red: tools/mutate_stream.sh.
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

            /* the min store really evicted, not just fit by luck (LEZIONI: same result never
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
    void *real_ctx;
    int64_t calls, fail_at; /* 1-based; 0 never fails */
} fail_reader_ctx;

static int fail_reader(void *ctx, void *buf, size_t n, uint64_t offset) {
    fail_reader_ctx *f = (fail_reader_ctx *)ctx;
    f->calls++;
    if (f->calls == f->fail_at) return -1;
    return f->real(f->real_ctx, buf, n, offset);
}

/* How many real reads a fresh min-store model consumes evaluating tokens[0..n) (3 per miss,
 * src/memory/experts.c): deterministic, since the file and the store's starting state are always
 * the same, so this is exactly the read count the same pass will reproduce later. */
static int64_t count_reads(const char *path, int64_t n_batch, const int32_t *tokens, int64_t n) {
    tr_model *m = load_budget(path, NULL, UINT64_MAX);
    if (m == NULL) return -1;
    tr_session *s = create_session(m, n_batch);
    int64_t reads = -1;
    if (s != NULL && tr_session_eval(s, tokens, n) == 0) {
        tr_experts_stats st;
        tr_model_expert_stats(m, &st);
        reads = (int64_t)st.misses * TR_EXPERT_PARTS;
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
                    tr_experts_get_reader(ex, &frc.real, &frc.real_ctx);
                    frc.fail_at = 1;
                    tr_experts_set_reader(ex, fail_reader, &frc);
                }

                TR_CHECK(tr_session_eval(s, tokens, 1) == -1);
                TR_CHECK_EQ_INT(tr_session_pos(s), 0);
                checked++;

                if (ex != NULL) tr_experts_set_reader(ex, frc.real, frc.real_ctx);
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
                    tr_experts_get_reader(ex, &frc.real, &frc.real_ctx);
                    frc.fail_at = reads1 + 1; /* the first read of the second pass */
                    tr_experts_set_reader(ex, fail_reader, &frc);
                }

                TR_CHECK(tr_session_eval(s, tokens, N_PROMPT) == -1);
                TR_CHECK_EQ_INT(tr_session_pos(s), 0); /* the whole eval, not just the failed pass */
                checked++;

                if (ex != NULL) tr_experts_set_reader(ex, frc.real, frc.real_ctx);
                TR_CHECK(tr_session_eval(s, tokens, N_PROMPT) == 0);
                TR_CHECK(memcmp(tr_session_logits(s), ref[N_PROMPT - 1], sizeof ref[0]) == 0);
                checked++;

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
 * bytes on the real model at 2048 tokens: docs/MISURE.md domanda 47, docs/LEZIONI.md #99). The
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

/* ---- direct: the unbuffered path (docs/ARCHITETTURA.md Esperti M1, Step C) gives the same
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

/* ---- mem_available: TR_MEM_AVAILABLE_MIB (docs/ARCHITETTURA.md Esperti M1, Step D) ---- */

/* Big enough that its expert weights alone (F32, not Q8_0: at this size Q8_0's per-byte cost in
 * synth_olmoe.h would dominate the test) clear the automatic plan's fixed 512 MiB session floor
 * (model.c tr_expert_budget_plan, olmoe.c session_allowance) with room to spare: only then does a
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
     * session_allowance is >= the flat 512 MiB (plus a KV allowance too small to matter at this
     * context length), so this slightly overshoots the true lower edge -- always inward, never
     * past the upper one. reserve depends on the real machine's total RAM (tr_mem_guard's own
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

    set_env("TR_MEM_AVAILABLE_MIB", NULL);
    TR_CHECK(checked > 0);
    remove(path);
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
    TR_TEST_EXIT();
}
