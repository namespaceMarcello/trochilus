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
 *              byte for byte the reference's, under ds4's eviction (the default) and the LRU; and
 *              under the LRU on and off read the same units (misses). Under ds4's they need not: the
 *              read ahead never evicts the computing layer, a call on demand may evict its coolest
 *              units once a prompt's routings made the hotness unequal (LESSONS #287); (heat) under
 *              ds4's the prompt's passes added their routings (hot_extra > 0), under the LRU nothing
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
 *   pass       the prompt in one pass (forward_pass reads each next layer ahead, in short requests),
 *              on and off at both budgets and three pools: the reference logits, the same units
 *   cancel     that one pass on the zero-router model, requests of 3 units, the last layer's reads
 *              slowed; the read ahead returns once its first run is taken, so 0 and 1 are asked
 *              while 0-2 are in flight: 5 dropped unread, 1 read unasked; the reference logits;
 *              (pass_drained) that 1 taken in before the eval returns; (major_kept) layer-major
 *              blocks drop nothing and take the store's runs (as many requests as on demand): all 6;
 *              (fail_pass) fail above, the prompt in one pass
 *   lru        layer-major and one pass under TR_EXPERT_EVICT=lru at the exact margin: the reference
 *              (ds4's eviction is the default; the LRU's victims are another walk)
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
#include "../src/base/platform.h" /* tr_iov */
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
    int same_on, same_off, same_lru, heat, refused, rule, fail, drained, fail_unused, pass_on, pass_off, cancel, pass_drained, fail_pass,
        major_kept, lru;
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
    tr_experts_readv_fn realv;
    void *real_ctx;
    atomic_int fail_io, failed;
    long long fail_at; /* >= 0: fail instead every I/O-thread read covering this file offset */
    long long slow_from; /* >= 0: every I/O-thread read from this file offset on takes slow_sec first */
    double slow_sec;
} io_fail_reader;

/* a request off the test's thread over [offset, offset + len): fail it? counted when it does */
static int io_fail_now(io_fail_reader *r, uint64_t offset, uint64_t len) {
    int fail = 0;
    if (!t_test_thread && r->slow_from >= 0 && offset >= (uint64_t)r->slow_from) {
        double t0 = tr_time_sec();
        while (tr_time_sec() - t0 < r->slow_sec) {
        }
    }
    if (!t_test_thread) {
        if (r->fail_at >= 0) fail = (uint64_t)r->fail_at >= offset && (uint64_t)r->fail_at < offset + len;
        else fail = atomic_load(&r->fail_io) > 0 && atomic_fetch_sub(&r->fail_io, 1) > 0;
    }
    if (fail) atomic_fetch_add(&r->failed, 1);
    return fail;
}

static int io_fail_read(void *ctx, void *buf, size_t len, uint64_t offset) {
    io_fail_reader *r = (io_fail_reader *)ctx;
    return io_fail_now(r, offset, len) ? -1 : r->real(r->real_ctx, buf, len, offset);
}

/* a run's requests one by one through io_fail_now: the call fails when any of them does */
static int io_fail_readv(void *ctx, const tr_readv_req *req, int n_req, void *scratch) {
    io_fail_reader *r = (io_fail_reader *)ctx;
    int fail = 0;
    for (int k = 0; k < n_req; k++) {
        uint64_t len = 0;
        for (int i = 0; i < req[k].cnt; i++) len += req[k].iov[i].n;
        if (io_fail_now(r, req[k].offset, len)) fail = 1;
    }
    return fail ? -1 : r->realv(r->real_ctx, req, n_req, scratch);
}

static void reader_wrap(tr_experts *x, io_fail_reader *r, int fail_io, long long fail_at) {
    tr_experts_get_reader(x, &r->real, &r->realv, &r->real_ctx);
    atomic_init(&r->fail_io, fail_io);
    atomic_init(&r->failed, 0);
    r->fail_at = fail_at;
    r->slow_from = -1;
    r->slow_sec = 0.0;
    tr_experts_set_reader(x, io_fail_read, r->realv != NULL ? io_fail_readv : NULL, r);
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
        tr_experts_set_reader(x, r.real, r.realv, r.real_ctx);
        TR_CHECK(tr_session_eval(s, prompt, N_PROMPT) == 0);
        int same = memcmp(tr_session_logits(s), ref.row[0], sizeof ref.row[0]) == 0;
        TR_CHECK(same);
        if (rc == -1 && atomic_load(&r.failed) > 0 && same) n.fail_unused++;
        tr_session_free(s);
    }
    tr_model_free(m);

    /* cancel: the prompt in one pass (n_batch = N_PROMPT), every next layer read ahead in requests
     * of 3 units (TR_AHEAD_RUN_KIB=48: a part is 16 KiB), the last layer's reads slowed to 0.2 s a
     * request. The read ahead returns once the I/O thread has taken its first run (0-2), so it is in
     * flight, far longer than a layer's compute, when the last layer asks for experts 0 and 1 (a
     * zero router: the lowest ids win the ties): the other 5 are dropped unread, expert 2 is read and
     * not asked, so exactly 1 more unit than on demand; the reference logits, and expert 2 taken in
     * before the eval returns (pass_drained). major: the same in layer-major blocks (N_BATCH), which
     * never drop and take the store's own runs (as many requests as on demand): all 6 read, as the
     * rule above pins */
    static const int64_t cancel_batch[] = {N_PROMPT, N_BATCH};
    for (size_t cb = 0; cb < sizeof cancel_batch / sizeof cancel_batch[0]; cb++) {
        const int pass = cancel_batch[cb] == N_PROMPT;
        uint64_t pass_misses[2] = {0, 0}, off_requests = 0;
        for (int on = 0; on <= 1; on++) {
            set_env("TR_PREFETCH", on ? NULL : "0");
            set_env("TR_AHEAD_RUN_KIB", "48");
            g = tr_gguf_open(path, err, sizeof err);
            long long last = -1; /* where the last layer's experts start: its reads are the slow ones */
            static const char *const parts[3] = {"gate", "up", "down"};
            for (int p = 0; p < 3 && g != NULL; p++) {
                snprintf(name, sizeof name, "blk.%d.ffn_%s_exps.weight", N_LAYERS - 1, parts[p]);
                t = tr_gguf_find_tensor(g, name);
                if (t != NULL && (last < 0 || (long long)t->offset < last)) last = (long long)t->offset;
            }
            tr_gguf_close(g);
            m = tr_model_load_budget(path, pool, budget, err, sizeof err);
            s = m != NULL ? tr_session_create(m, CTX, cancel_batch[cb], err, sizeof err) : NULL;
            TR_CHECK(s != NULL && last >= 0);
            if (s != NULL && last >= 0) {
                tr_experts *x = tr_model_experts(m);
                io_fail_reader r;
                reader_wrap(x, &r, 0, -1);
                r.slow_from = last;
                r.slow_sec = 0.2;
                TR_CHECK(tr_session_eval(s, prompt, N_PROMPT) == 0);
                int same = memcmp(tr_session_logits(s), ref.row[0], sizeof ref.row[0]) == 0;
                TR_CHECK(same);
                tr_experts_stats a, b;
                tr_model_expert_stats(m, &a);
                pass_misses[on] = a.misses;
                TR_CHECK(tr_experts_prefetch_wait(x) == 0);
                tr_model_expert_stats(m, &b);
                TR_CHECK_EQ_INT(b.misses, a.misses);
                if (on) {
                    /* a dropped unit is absent: the last layer holds what it asked plus what was read unasked */
                    int present = 0;
                    for (int e = 0; e < N_EXPERT; e++) present += tr_experts_part(x, N_LAYERS - 1, e, 0) != NULL;
                    uint64_t unasked = a.misses - pass_misses[0];
                    TR_CHECK_EQ_INT(present, N_USED + unasked);
                    TR_CHECK(a.prefetched > 0);
                    TR_CHECK_EQ_INT(a.cancelled, pass ? N_EXPERT - 3 : 0);
                    TR_CHECK_EQ_INT(unasked, pass ? 3 - N_USED : N_EXPERT - N_USED);
                    /* layer-major: the store's runs, one a layer, as many requests as on demand (the
                     * pass's short ones would be three a layer here) */
                    if (!pass) TR_CHECK_EQ_INT(a.requests, off_requests);
                    if (same && present == N_USED + (int)unasked) {
                        if (pass && a.cancelled == N_EXPERT - 3 && unasked == 3 - N_USED) {
                            n.cancel++;
                            if (b.misses == a.misses) n.pass_drained++;
                        }
                        if (!pass && a.cancelled == 0 && unasked == N_EXPERT - N_USED && a.requests == off_requests)
                            n.major_kept++;
                    }
                } else {
                    TR_CHECK_EQ_INT(a.prefetched, 0);
                    TR_CHECK_EQ_INT(a.cancelled, 0);
                    off_requests = a.requests;
                }
                tr_experts_set_reader(x, r.real, r.realv, r.real_ctx);
                tr_session_free(s);
            }
            tr_model_free(m);
        }
    }
    set_env("TR_AHEAD_RUN_KIB", NULL);
    set_env("TR_PREFETCH", NULL);
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

    /* same: on and off, two budgets that read ahead, three pools, ds4's eviction and the LRU */
    static const int64_t reading[] = {24, MARGIN_SLOTS};
    for (int lru = 0; lru <= 1; lru++) {
        set_env("TR_EXPERT_EVICT", lru ? "lru" : NULL);
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
                    /* the prompt's passes told the store their routings (olmoe_refresh_experts): more than
                     * one a unit; the LRU keeps no hotness */
                    if (lru) TR_CHECK_EQ_INT(st.hot_extra, 0);
                    else TR_CHECK(st.hot_extra > 0);
                    if (!lru && st.hot_extra > 0) n.heat++;
                    if (on) {
                        TR_CHECK(st.prefetched > 0);
                        if (same && st.prefetched > 0) n.same_on++;
                    } else {
                        TR_CHECK_EQ_INT(st.prefetched, 0);
                        if (same) n.same_off++;
                    }
                }
                TR_CHECK_EQ_INT(prompt_misses[1], prompt_misses[0]); /* the same units read, ahead or on demand */
                if (lru) TR_CHECK_EQ_INT(misses[1], misses[0]);
                if (lru && misses[1] == misses[0]) n.same_lru++;
            }
        }
    }
    set_env("TR_EXPERT_EVICT", NULL);

    /* pass: the whole prompt in one pass (n_batch = N_PROMPT), each next layer read ahead in short
     * requests while this one computes and what it does not ask dropped (forward_pass); on and off
     * the same logits and the prompt's same units read (every layer asks for all 8) */
    for (size_t b = 0; b < sizeof reading / sizeof reading[0]; b++) {
        for (int pi = 0; pi < 3; pi++) {
            uint64_t prompt_misses[2] = {0, 0};
            for (int on = 0; on <= 1; on++) {
                set_env("TR_PREFETCH", on ? NULL : "0");
                int rc = run(path, pools[pi], (uint64_t)reading[b] * slot_bytes, N_PROMPT, &got, &st, &prefetching);
                TR_CHECK(rc == 0);
                int same = rc == 0 && memcmp(&got, &ref, sizeof ref) == 0;
                TR_CHECK(same);
                prompt_misses[on] = g_prompt_misses;
                if (on) {
                    TR_CHECK(st.prefetched > 0);
                    if (same && st.prefetched > 0) n.pass_on++;
                } else {
                    TR_CHECK_EQ_INT(st.prefetched, 0);
                    if (same) n.pass_off++;
                }
            }
            TR_CHECK_EQ_INT(prompt_misses[1], prompt_misses[0]);
        }
    }

    /* lru: the same under the LRU (TR_EXPERT_EVICT=lru), at the exact margin, layer-major and in one
     * pass: the default is ds4's eviction since 10-03, and the LRU's victims are another walk */
    set_env("TR_PREFETCH", NULL);
    set_env("TR_EXPERT_EVICT", "lru");
    static const int64_t lru_batch[] = {N_BATCH, N_PROMPT};
    for (size_t lb = 0; lb < sizeof lru_batch / sizeof lru_batch[0]; lb++) {
        int rc = run(path, pool3, (uint64_t)MARGIN_SLOTS * slot_bytes, lru_batch[lb], &got, &st, &prefetching);
        TR_CHECK(rc == 0);
        TR_CHECK_EQ_INT(st.evict, TR_EXPERTS_EVICT_LRU);
        TR_CHECK(st.prefetched > 0);
        int same = rc == 0 && memcmp(&got, &ref, sizeof ref) == 0;
        TR_CHECK(same);
        if (same && st.prefetched > 0 && st.evict == TR_EXPERTS_EVICT_LRU) n.lru++;
    }
    set_env("TR_EXPERT_EVICT", NULL);

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

    /* fail: the I/O thread's first read fails, in a layer-major prompt and in one pass; the eval is
     * -1 and the session has not moved; the same eval again, the reader restored, is the reference */
    static const int64_t fail_batch[] = {N_BATCH, N_PROMPT};
    for (size_t fb = 0; fb < sizeof fail_batch / sizeof fail_batch[0]; fb++) {
        char err[256];
        tr_model *m = tr_model_load_budget(path, pool3, (uint64_t)24 * slot_bytes, err, sizeof err);
        TR_CHECK(m != NULL);
        tr_session *s = m != NULL ? tr_session_create(m, CTX, fail_batch[fb], err, sizeof err) : NULL;
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
            tr_experts_set_reader(x, r.real, r.realv, r.real_ctx);
            TR_CHECK(tr_session_eval(s, prompt, N_PROMPT) == 0);
            int same = memcmp(tr_session_logits(s), ref.row[0], sizeof ref.row[0]) == 0;
            TR_CHECK(same);
            if (rc == -1 && atomic_load(&r.failed) == 1 && same) *(fail_batch[fb] == N_PROMPT ? &n.fail_pass : &n.fail) += 1;
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
    TR_CHECK(n.same_lru > 0);
    TR_CHECK(n.heat > 0);
    TR_CHECK(n.refused > 0);
    TR_CHECK(n.rule > 0);
    TR_CHECK(n.fail > 0);
    TR_CHECK(n.drained > 0);
    TR_CHECK(n.fail_unused > 0);
    TR_CHECK(n.pass_on > 0);
    TR_CHECK(n.pass_off > 0);
    TR_CHECK(n.cancel > 0);
    TR_CHECK(n.pass_drained > 0);
    TR_CHECK(n.fail_pass > 0);
    TR_CHECK(n.major_kept > 0);
    TR_CHECK(n.lru > 0);
    TR_TEST_EXIT();
}
