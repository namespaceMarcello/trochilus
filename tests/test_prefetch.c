/* test_prefetch.c — reading the next layer's experts ahead while a prompt computes this one
 * (src/models/olmoe.c forward_prompt_layer_major, src/memory/experts.h tr_experts_prefetch) moves
 * only when the same bytes arrive, never which bytes the kernels see.
 *
 * A synthetic OLMoE (tests/synth_olmoe.h), 6 layers of 8 experts, 2 used: 48 units, so a partial
 * store can hold two layers and a margin (2 * 8 + 2 = 18 slots, the least that reads ahead). A
 * 96-token prompt in blocks of 32 takes the layer-major path under every partial budget; then 4
 * tokens are decoded one by one. The reference is the resident model (pass-major, no store
 * traffic at all).
 *
 * Branches, each counted (a case that never ran fails the test):
 *   same       at budgets of 24 slots (half) and 18 (the exact margin), with the I/O thread on
 *              (and seen reading: prefetched > 0) and off (TR_PREFETCH=0, prefetched == 0), with
 *              no pool and pools of 3 and 8: the prompt's logits and every decoded token's,
 *              byte for byte the reference's; and on and off read the same units (misses)
 *   refused    17 slots (one short of the margin) and the smallest store (10): no I/O thread even
 *              when asked for, prefetched == 0, logits identical
 *   rule       blocks of 2 tokens ask for at most 4 of 8 experts: nothing is read ahead
 *              (prefetched == 0) though the thread runs, logits identical
 *   fail       the I/O thread's first read fails (a reader that fails only off the test's own
 *              thread, and counts it): the prompt's eval is -1, the session's position has not
 *              moved and nothing is left in flight; the reader restored, the same eval gives the
 *              reference logits
 *   unused     the last layer's router all zeros (synth_zero_router_layer): it asks for 2 of 8
 *              experts, the layer before reads all 8 ahead. Exactly 6 more units read than
 *              without reading ahead (the cost of the rule, pinned), logits identical; (drained)
 *              when the prompt's eval returns nothing is in flight, not even those 6; (fail) a
 *              read ahead failing for one of them, asked by nobody, still fails the eval, and
 *              the retry gives the reference
 *
 * TSan: run it under build/linux-tsan (the I/O thread and the calling thread share the store).
 * Seen red: the mutations listed in the report of the change that wrote it (docs/STATUS.md). */
#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* setenv/unsetenv */
#endif
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "synth_olmoe.h"
#include "../src/base/threads.h"
#include "../src/models/model.h"
#include "../src/models/model_internal.h"
#include "../src/memory/experts.h"

enum { N_LAYERS = 6, N_EMBD = 64, N_FF = 64, N_EXPERT = 8, N_USED = 2, VOCAB = 64, CTX = 128 };
enum { N_PROMPT = 96, N_BATCH = 32, N_DECODE = 4 };
enum { MARGIN_SLOTS = 2 * N_EXPERT + N_USED, MIN_SLOTS = N_EXPERT + N_USED }; /* 18, 10 */
static const synth_params P = {N_LAYERS, N_EMBD, 4, 2, N_FF, N_EXPERT, N_USED, VOCAB, CTX, TR_TYPE_F32};
#define HUGE_BUDGET ((uint64_t)1 << 40)

static _Thread_local int t_test_thread; /* global-ok: 1 on the thread that runs the test */

static struct {
    int same_on, same_off, refused, rule, fail, drained, fail_unused;
} n; /* global-ok: this test's own branch counters */

static void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value != NULL ? value : "");
#else
    if (value != NULL) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static int32_t token(int i) {
    return (int32_t)((i * 13 + 7) % VOCAB);
}

/* the logits after the prompt, then after each decoded token */
typedef struct {
    float row[1 + N_DECODE][VOCAB];
} run_logits;

static uint64_t g_prompt_misses; /* global-ok: the last run's units read by the end of its prompt */

/* Loads at `budget` (TR_PREFETCH as the caller set it), runs the prompt in blocks of n_batch and
 * the decode; stats of the store after it. -1 if anything failed. */
static int run(const char *path, tr_pool *pool, uint64_t budget, int64_t n_batch, run_logits *out,
               tr_experts_stats *st, int *prefetching) {
    char err[256];
    tr_model *m = tr_model_load_budget(path, pool, budget, err, sizeof err);
    if (m == NULL) {
        fprintf(stderr, "  load failed: %s\n", err);
        return -1;
    }
    *prefetching = tr_experts_prefetching(tr_model_experts(m));
    tr_session *s = tr_session_create(m, CTX, n_batch, err, sizeof err);
    int rc = s != NULL ? 0 : -1;
    int32_t prompt[N_PROMPT];
    for (int i = 0; i < N_PROMPT; i++) prompt[i] = token(i);
    if (rc == 0) rc = tr_session_eval(s, prompt, N_PROMPT);
    if (rc == 0) memcpy(out->row[0], tr_session_logits(s), sizeof out->row[0]);
    tr_model_expert_stats(m, st);
    g_prompt_misses = st->misses;
    for (int k = 0; k < N_DECODE && rc == 0; k++) {
        int32_t t = token(N_PROMPT + k);
        rc = tr_session_eval(s, &t, 1);
        if (rc == 0) memcpy(out->row[1 + k], tr_session_logits(s), sizeof out->row[0]);
    }
    tr_model_expert_stats(m, st);
    tr_session_free(s);
    tr_model_free(m);
    return rc;
}

/* ---- a reader that fails the next read made off the test's thread (the I/O thread's) ---- */

typedef struct {
    tr_experts_read_fn real;
    void *real_ctx;
    atomic_int fail_io, failed;
    long long fail_at; /* >= 0: fail instead every I/O-thread read covering this file offset */
} io_fail_reader;

static int io_fail_read(void *ctx, void *buf, size_t len, uint64_t offset) {
    io_fail_reader *r = (io_fail_reader *)ctx;
    int fail = 0;
    if (!t_test_thread) {
        if (r->fail_at >= 0) fail = (uint64_t)r->fail_at >= offset && (uint64_t)r->fail_at < offset + len;
        else fail = atomic_load(&r->fail_io) > 0 && atomic_fetch_sub(&r->fail_io, 1) > 0;
    }
    if (fail) {
        atomic_fetch_add(&r->failed, 1);
        return -1;
    }
    return r->real(r->real_ctx, buf, len, offset);
}

static void reader_wrap(tr_experts *x, io_fail_reader *r, int fail_io, long long fail_at) {
    tr_experts_get_reader(x, &r->real, &r->real_ctx);
    atomic_init(&r->fail_io, fail_io);
    atomic_init(&r->failed, 0);
    r->fail_at = fail_at;
    tr_experts_set_reader(x, io_fail_read, r);
}

/* ---- unused: the last layer's router is all zeros, so it asks for 2 of its 8 experts while the
 * layer before it, asking for all 8, reads all 8 ahead ---- */
static void test_unused(const char *argv0, tr_pool *pool) {
    char path[512], err[256];
    synth_zero_router_layer = N_LAYERS - 1;
    TR_CHECK(synth_write(&P, argv0, "prefetch_unused.gguf", path, sizeof path) == 0);
    synth_zero_router_layer = -1;
    static run_logits ref, got;
    tr_experts_stats st;
    int prefetching = 0;
    set_env("TR_PREFETCH", NULL);
    TR_CHECK(run(path, NULL, HUGE_BUDGET, N_BATCH, &ref, &st, &prefetching) == 0);
    uint64_t budget = (uint64_t)24 * st.slot_bytes;

    /* the cost of the rule, exactly: by the end of the prompt 6 units were read ahead and never
     * asked for (the decode after it then runs on a store holding them: its own misses differ);
     * the same logits */
    uint64_t misses[2] = {0, 0};
    for (int on = 0; on <= 1; on++) {
        set_env("TR_PREFETCH", on ? NULL : "0");
        TR_CHECK(run(path, pool, budget, N_BATCH, &got, &st, &prefetching) == 0);
        TR_CHECK(memcmp(&got, &ref, sizeof ref) == 0);
        misses[on] = g_prompt_misses;
    }
    set_env("TR_PREFETCH", NULL);
    TR_CHECK_EQ_INT(misses[1] - misses[0], N_EXPERT - N_USED);

    /* drained: when the prompt's eval returns nothing is left in flight, so waiting again finds
     * nothing to take in, not even the 6 units nobody asked for */
    int32_t prompt[N_PROMPT];
    for (int i = 0; i < N_PROMPT; i++) prompt[i] = token(i);
    tr_model *m = tr_model_load_budget(path, pool, budget, err, sizeof err);
    tr_session *s = m != NULL ? tr_session_create(m, CTX, N_BATCH, err, sizeof err) : NULL;
    TR_CHECK(s != NULL);
    if (s != NULL) {
        tr_experts *x = tr_model_experts(m);
        TR_CHECK(tr_session_eval(s, prompt, N_PROMPT) == 0);
        tr_experts_stats a, b;
        tr_model_expert_stats(m, &a);
        TR_CHECK(tr_experts_prefetch_wait(x) == 0);
        tr_model_expert_stats(m, &b);
        TR_CHECK_EQ_INT(b.misses, a.misses);
        TR_CHECK_EQ_INT(b.prefetched, a.prefetched);
        if (b.misses == a.misses && a.prefetched > 0) n.drained++;
        tr_session_free(s);
    }
    tr_model_free(m);

    /* a read ahead that fails for a unit nobody asks for -- the last layer's expert 2 -- still fails
     * the eval, like any read of the model that fails; retried, the reference */
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    char name[64];
    snprintf(name, sizeof name, "blk.%d.ffn_up_exps.weight", N_LAYERS - 1);
    const tr_gguf_tensor *t = g != NULL ? tr_gguf_find_tensor(g, name) : NULL;
    TR_CHECK(t != NULL);
    uint64_t part_bytes = t != NULL ? t->n_bytes / N_EXPERT : 0;
    long long at = t != NULL ? (long long)(t->offset + 2 * part_bytes + part_bytes / 2) : -1;
    tr_gguf_close(g);
    m = tr_model_load_budget(path, pool, budget, err, sizeof err);
    s = m != NULL ? tr_session_create(m, CTX, N_BATCH, err, sizeof err) : NULL;
    TR_CHECK(s != NULL && at >= 0);
    if (s != NULL && at >= 0) {
        tr_experts *x = tr_model_experts(m);
        io_fail_reader r;
        reader_wrap(x, &r, 0, at);
        int rc = tr_session_eval(s, prompt, N_PROMPT);
        TR_CHECK_EQ_INT(rc, -1);
        TR_CHECK(atomic_load(&r.failed) > 0);
        TR_CHECK_EQ_INT(tr_session_pos(s), 0);
        tr_experts_set_reader(x, r.real, r.real_ctx);
        TR_CHECK(tr_session_eval(s, prompt, N_PROMPT) == 0);
        int same = memcmp(tr_session_logits(s), ref.row[0], sizeof ref.row[0]) == 0;
        TR_CHECK(same);
        if (rc == -1 && atomic_load(&r.failed) > 0 && same) n.fail_unused++;
        tr_session_free(s);
    }
    tr_model_free(m);
    remove(path);
}

int main(int argc, char **argv) {
    t_test_thread = 1;
    char path[512];
    TR_CHECK(synth_write(&P, argc > 0 ? argv[0] : "./test_prefetch", "prefetch.gguf", path, sizeof path) == 0);

    /* the reference, and the size of a slot */
    static run_logits ref, got;
    tr_experts_stats st;
    int prefetching = 0;
    set_env("TR_PREFETCH", NULL);
    TR_CHECK(run(path, NULL, HUGE_BUDGET, N_BATCH, &ref, &st, &prefetching) == 0);
    TR_CHECK(!prefetching); /* resident: nothing to read ahead */
    uint64_t slot_bytes = st.slot_bytes;
    TR_CHECK(slot_bytes > 0);

    tr_pool *pool3 = tr_pool_create(3), *pool8 = tr_pool_create(8);
    tr_pool *pools[3] = {NULL, pool3, pool8};

    /* same: on and off, two budgets that read ahead, three pools */
    static const int64_t reading[] = {24, MARGIN_SLOTS};
    for (size_t b = 0; b < sizeof reading / sizeof reading[0]; b++) {
        for (int pi = 0; pi < 3; pi++) {
            uint64_t misses[2] = {0, 0}, prompt_misses[2] = {0, 0};
            for (int on = 0; on <= 1; on++) {
                set_env("TR_PREFETCH", on ? NULL : "0");
                int rc = run(path, pools[pi], (uint64_t)reading[b] * slot_bytes, N_BATCH, &got, &st, &prefetching);
                TR_CHECK(rc == 0);
                TR_CHECK_EQ_INT(st.n_slots, reading[b]);
                TR_CHECK_EQ_INT(prefetching, on);
                int same = rc == 0 && memcmp(&got, &ref, sizeof ref) == 0;
                TR_CHECK(same);
                if (!same)
                    fprintf(stderr, "  %lld slots, pool %d, prefetch %d: logits differ\n", (long long)reading[b], pi,
                            on);
                misses[on] = st.misses;
                prompt_misses[on] = g_prompt_misses;
                if (on) {
                    TR_CHECK(st.prefetched > 0);
                    if (same && st.prefetched > 0) n.same_on++;
                } else {
                    TR_CHECK_EQ_INT(st.prefetched, 0);
                    if (same) n.same_off++;
                }
            }
            TR_CHECK_EQ_INT(prompt_misses[1], prompt_misses[0]); /* the same units read, ahead or on demand */
            TR_CHECK_EQ_INT(misses[1], misses[0]);
        }
    }

    /* refused: one slot short of the margin, and the smallest store */
    set_env("TR_PREFETCH", NULL);
    static const int64_t too_few[] = {MARGIN_SLOTS - 1, MIN_SLOTS};
    for (size_t b = 0; b < sizeof too_few / sizeof too_few[0]; b++) {
        int rc = run(path, pool3, (uint64_t)too_few[b] * slot_bytes, N_BATCH, &got, &st, &prefetching);
        TR_CHECK(rc == 0);
        TR_CHECK_EQ_INT(st.n_slots, too_few[b]);
        TR_CHECK(!prefetching);
        TR_CHECK_EQ_INT(st.prefetched, 0);
        TR_CHECK(rc == 0 && memcmp(&got, &ref, sizeof ref) == 0);
        n.refused++;
    }

    /* rule: blocks of 2 tokens use at most 4 of 8 experts a layer, never enough to read ahead */
    {
        int rc = run(path, pool3, (uint64_t)24 * slot_bytes, 2, &got, &st, &prefetching);
        TR_CHECK(rc == 0);
        TR_CHECK(prefetching);
        TR_CHECK_EQ_INT(st.prefetched, 0);
        TR_CHECK(rc == 0 && memcmp(&got, &ref, sizeof ref) == 0);
        if (rc == 0 && prefetching && st.prefetched == 0) n.rule++;
    }

    /* fail: the I/O thread's first read fails; the eval is -1 and the session has not moved; the
     * same eval again, the reader restored, is the reference */
    {
        char err[256];
        tr_model *m = tr_model_load_budget(path, pool3, (uint64_t)24 * slot_bytes, err, sizeof err);
        TR_CHECK(m != NULL);
        tr_session *s = m != NULL ? tr_session_create(m, CTX, N_BATCH, err, sizeof err) : NULL;
        TR_CHECK(s != NULL);
        if (s != NULL) {
            tr_experts *x = tr_model_experts(m);
            TR_CHECK(tr_experts_prefetching(x));
            io_fail_reader r;
            reader_wrap(x, &r, 1, -1);
            int32_t prompt[N_PROMPT];
            for (int i = 0; i < N_PROMPT; i++) prompt[i] = token(i);
            int rc = tr_session_eval(s, prompt, N_PROMPT);
            TR_CHECK_EQ_INT(rc, -1);
            TR_CHECK_EQ_INT(atomic_load(&r.failed), 1);
            TR_CHECK_EQ_INT(tr_session_pos(s), 0);
            /* the failed eval left nothing in flight: waiting again takes nothing in, and the
             * failure was already reported to the eval, not left for the next wait */
            tr_experts_stats a, b;
            tr_model_expert_stats(m, &a);
            TR_CHECK(tr_experts_prefetch_wait(x) == 0);
            tr_model_expert_stats(m, &b);
            TR_CHECK_EQ_INT(b.misses, a.misses);
            tr_experts_set_reader(x, r.real, r.real_ctx);
            TR_CHECK(tr_session_eval(s, prompt, N_PROMPT) == 0);
            int same = memcmp(tr_session_logits(s), ref.row[0], sizeof ref.row[0]) == 0;
            TR_CHECK(same);
            if (rc == -1 && atomic_load(&r.failed) == 1 && same) n.fail++;
            tr_session_free(s);
        }
        tr_model_free(m);
    }

    test_unused(argc > 0 ? argv[0] : "./test_prefetch", pool3);

    tr_pool_destroy(pool8);
    tr_pool_destroy(pool3);
    set_env("TR_PREFETCH", NULL);
    remove(path);

    TR_CHECK(n.same_on > 0);
    TR_CHECK(n.same_off > 0);
    TR_CHECK(n.refused > 0);
    TR_CHECK(n.rule > 0);
    TR_CHECK(n.fail > 0);
    TR_CHECK(n.drained > 0);
    TR_CHECK(n.fail_unused > 0);
    TR_TEST_EXIT();
}
