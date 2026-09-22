/* test_phase.c — threads per phase (src/models/model.h): a long pass runs on the whole pool, a
 * short one on a measured width, and no logit moves by a bit whatever the width.
 *
 * Each test says which branch of the rewrite it exercises, and counts that it was actually taken
 * (CLAUDE.md, LESSONS #43 #50 #54 #78):
 *   widths      tr_decode_tune_widths: narrowest first, a width under 4 threads dropped unless
 *               dropping it would leave nothing (the pool itself < 4)
 *   stats       tr_decode_tune_stats: center is the fastest pass, spread the noise around it --
 *               one disturbed pass does not move either, more passes only tighten the spread
 *   pick        tr_decode_tune_pick: the margin comes from each width's OWN spread, paired with
 *               the fastest width's (never a spread borrowed from a third width), flat and
 *               compute-bound curves, n = 1 and n = 2
 *   debounce    tr_decode_tune_debounce: two votes in a row switch, a third width replaces the
 *               one pending, current == 0 adopts outright
 *   session     a fake clock (tr_session_set_tune_clock, tests only) drives a real session
 *               through a clear winner (no extension), a pairwise tie (extension all the way to
 *               TR_DECODE_TUNE_ROUNDS_MAX), a milestone crossing (and a rewind that lowers it
 *               back down, so crossing again re-arms again), and debounce across two
 *               re-measurements of the session itself
 *   history     tr_session_decode_history: every measurement leaves position, width in effect and
 *               pick (a pick the debounce held back is seen once); past TR_DECODE_TUNE_HISTORY
 *               measurements the oldest stay and the last slot holds the newest
 *   keep        forced widths, clamped to the pool, and back to measuring after; rows of a short
 *               pass before the first choice and after it; no pool at all; a failed eval changes
 *               nothing; every logit row identical to one thread, f32 and Q8_0
 *
 * Seen red: tools/mutate_tune.sh.
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

/* ---- a made-up clock, for an exact schedule (session tests below) ----
 * Both timestamps of a probe (session_eval, model.c) are read while the pool is still set to the
 * probed width, and only their difference is used, so a clock that lets the FIRST of the pair
 * pass through unchanged and adds a scheduled amount on the SECOND makes a probe's recorded time
 * exactly that amount, whichever width it probed. Each width has its own list, consumed in the
 * order that width is probed; a long, explicit list makes a whole session's schedule -- including
 * a later re-measurement with a different balance -- readable as a table, not a formula. */
enum { FAKE_SEQ_MAX = 128 };
typedef struct {
    double v[FAKE_SEQ_MAX];
    int n, i;
} fake_seq;

static fake_seq mk_seq(const double *v, int n) {
    fake_seq sq;
    sq.n = n;
    sq.i = 0;
    for (int i = 0; i < n; i++) sq.v[i] = v[i];
    return sq;
}

static double fake_seq_next(fake_seq *sq) {
    double v = sq->v[sq->i < sq->n ? sq->i : sq->n - 1];
    if (sq->i < sq->n) sq->i++;
    return v;
}

typedef struct {
    tr_pool *pool;
    int narrow_w, wide_w;
    fake_seq narrow, wide;
    int have_t0;
    double t;
} fake_clock;

static double fake_clock_now(void *ctx) {
    fake_clock *fc = (fake_clock *)ctx;
    if (!fc->have_t0) {
        fc->have_t0 = 1;
        return fc->t;
    }
    fc->have_t0 = 0;
    int w = tr_pool_active(fc->pool);
    fc->t += fake_seq_next(w == fc->narrow_w ? &fc->narrow : &fc->wide);
    return fc->t;
}

static void test_widths(void) {
    static const struct {
        int pool, n, w[TR_DECODE_TUNE_WIDTHS], dropped;
    } cases[] = {
        {16, 3, {4, 8, 16}, 0}, {12, 2, {6, 12, 0}, 1}, {8, 2, {4, 8, 0}, 1},
        {6, 1, {6, 0, 0}, 1},   {5, 1, {5, 0, 0}, 1},   {4, 1, {4, 0, 0}, 1},
        {3, 1, {3, 0, 0}, 1},   {2, 1, {2, 0, 0}, 1},   {1, 1, {1, 0, 0}, 1},
        {0, 1, {1, 0, 0}, 1},   {-4, 1, {1, 0, 0}, 1},  {64, 3, {16, 32, 64}, 0},
    };
    int dropped_cases = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int w[TR_DECODE_TUNE_WIDTHS] = {0, 0, 0};
        int n = tr_decode_tune_widths(cases[i].pool, w);
        TR_CHECK_EQ_INT(n, cases[i].n);
        for (int j = 0; j < cases[i].n && j < n; j++) TR_CHECK_EQ_INT(w[j], cases[i].w[j]);
        dropped_cases += cases[i].dropped;
    }
    TR_CHECK_EQ_INT(dropped_cases, 10); /* every case but the whole pools (16 and 64) drops a width under 4 */
}

static void test_stats(void) {
    /* a disturbed pass out of three does not move center or spread, wherever it sits */
    static const double perms[3][3] = {
        {1.50, 1.00, 1.01},
        {1.00, 1.50, 1.01},
        {1.00, 1.01, 1.50},
    };
    int disturbed_ok = 0;
    for (int i = 0; i < 3; i++) {
        double center, spread;
        tr_decode_tune_stats(perms[i], 3, &center, &spread);
        TR_CHECK(center > 0.999 && center < 1.001);
        TR_CHECK(spread > 0.0099 && spread < 0.0101);
        disturbed_ok++;
    }
    TR_CHECK_EQ_INT(disturbed_ok, 3);

    /* more passes tighten the spread, never widen it -- unlike a plain (max - min) / min range */
    static const double s[7] = {1.00, 1.50, 1.02, 1.20, 1.01, 1.30, 2.00};
    double c2, prev_spread;
    tr_decode_tune_stats(s, 2, &c2, &prev_spread);
    int tightened = 0;
    for (int n = 3; n <= 7; n++) {
        double center, spread, lo = s[0], hi = s[0];
        for (int i = 1; i < n; i++) {
            if (s[i] < lo) lo = s[i];
            if (s[i] > hi) hi = s[i];
        }
        double range = (hi - lo) / lo;
        tr_decode_tune_stats(s, n, &center, &spread);
        TR_CHECK(center > 0.999 && center < 1.001); /* 1.00 is always the minimum of the prefix */
        TR_CHECK(spread <= prev_spread + 1e-12);
        if (spread < prev_spread - 1e-12) tightened++;
        prev_spread = spread;
        if (n == 7) TR_CHECK(spread < range); /* the range grew with the new max; the spread did not */
    }
    TR_CHECK(tightened > 0);
}

static void test_pick(void) {
    int need_more;

    /* the margin comes from the spread: the same centers, two different spreads, two answers */
    static const double c2[2] = {1.02, 1.00};
    static const double s_wide[2] = {0.03, 0.03}, s_narrow[2] = {0.005, 0.005};
    int margin_from_spread = 0;
    TR_CHECK_EQ_INT(tr_decode_tune_pick(c2, s_wide, 2, &need_more), 0);
    TR_CHECK_EQ_INT(need_more, 1);
    margin_from_spread++;
    TR_CHECK_EQ_INT(tr_decode_tune_pick(c2, s_narrow, 2, &need_more), 1);
    TR_CHECK_EQ_INT(need_more, 0);
    margin_from_spread++;
    TR_CHECK_EQ_INT(margin_from_spread, 2);

    /* pairwise: a wide width's own noise (0.50) must not become the narrow width's margin -- a
     * global max spread would pick the narrow (index 0); pairwise picks the actual best */
    static const double c3[3] = {1.04, 1.00, 1.02}; /* narrow, best (middle), wide */
    static const double s3[3] = {0.01, 0.01, 0.50};
    int pairwise = tr_decode_tune_pick(c3, s3, 3, &need_more);
    TR_CHECK_EQ_INT(pairwise, 1);
    TR_CHECK_EQ_INT(need_more, 0);

    /* flat: no reason to add threads, the narrowest wins outright */
    static const double c_flat[3] = {1.00, 1.00, 1.00}, s_flat[3] = {0, 0, 0};
    TR_CHECK_EQ_INT(tr_decode_tune_pick(c_flat, s_flat, 3, &need_more), 0);
    TR_CHECK_EQ_INT(need_more, 0);

    /* compute-bound: every thread counts, the widest wins outright */
    static const double c_cb[3] = {1.30, 1.10, 1.00}, s_cb[3] = {0, 0, 0};
    TR_CHECK_EQ_INT(tr_decode_tune_pick(c_cb, s_cb, 3, &need_more), 2);
    TR_CHECK_EQ_INT(need_more, 0);

    /* n = 1: nothing to compare against but itself */
    static const double c1[1] = {1.23}, s1[1] = {0.9};
    TR_CHECK_EQ_INT(tr_decode_tune_pick(c1, s1, 1, &need_more), 0);
    TR_CHECK_EQ_INT(need_more, 0);

    /* n = 2, both ways: outright, no ambiguity */
    static const double c2a[2] = {1.20, 1.00}, s2a[2] = {0, 0};
    TR_CHECK_EQ_INT(tr_decode_tune_pick(c2a, s2a, 2, &need_more), 1);
    TR_CHECK_EQ_INT(need_more, 0);
    static const double c2b[2] = {1.00, 1.20}, s2b[2] = {0, 0};
    TR_CHECK_EQ_INT(tr_decode_tune_pick(c2b, s2b, 2, &need_more), 0);
    TR_CHECK_EQ_INT(need_more, 0);
}

static void test_debounce(void) {
    int pending;

    /* a sequence that would end on 8 without debounce (no two consecutive votes) stays 4 */
    int current = 4;
    static const int raws[] = {8, 4, 8, 4, 8};
    pending = 0;
    int stays = 0;
    for (size_t i = 0; i < sizeof raws / sizeof raws[0]; i++) {
        current = tr_decode_tune_debounce(current, &pending, raws[i]);
        TR_CHECK_EQ_INT(current, 4);
        stays++;
    }
    TR_CHECK_EQ_INT(stays, 5);

    /* two votes in a row for the same width switch to it */
    current = 4;
    pending = 0;
    TR_CHECK_EQ_INT(tr_decode_tune_debounce(current, &pending, 8), 4);
    current = tr_decode_tune_debounce(4, &pending, 8); /* second 8 in a row */
    TR_CHECK_EQ_INT(current, 8);

    /* a third width replaces the one pending, it does not accumulate towards it */
    current = 4;
    pending = 0;
    current = tr_decode_tune_debounce(current, &pending, 8);
    TR_CHECK_EQ_INT(current, 4);
    TR_CHECK_EQ_INT(pending, 8);
    current = tr_decode_tune_debounce(current, &pending, 5); /* a third width, not 8 again */
    TR_CHECK_EQ_INT(current, 4);
    TR_CHECK_EQ_INT(pending, 5);

    /* no choice yet: adopts the first raw outright, whatever was pending before */
    pending = 8;
    TR_CHECK_EQ_INT(tr_decode_tune_debounce(0, &pending, 7), 7);
    TR_CHECK_EQ_INT(pending, 0);
}

/* Session schedule, pool 8 (widths {4, 8}): a clear winner needs no extension (6 probes), a
 * pairwise tie extends one round at a time up to TR_DECODE_TUNE_ROUNDS_MAX and the narrower of
 * the two stands. */
static void test_session_extend(tr_model *model, tr_pool *pool) {
    /* (a) clear winner: narrow always cheaper, wide always dearer, no noise -- 6 probes, no more */
    {
        fake_clock fc = {0};
        fc.pool = pool;
        fc.narrow_w = 4;
        fc.wide_w = 8;
        static const double narrow[] = {1.00, 1.00, 1.00};
        static const double wide[] = {2.00, 2.00, 2.00};
        fc.narrow = mk_seq(narrow, 3);
        fc.wide = mk_seq(wide, 3);
        tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        tr_session_set_tune_clock(s, fake_clock_now, &fc);
        static const int want[6] = {4, 8, 4, 8, 4, 8};
        int clear_winner_probes = 0;
        for (int i = 0; i < 6; i++) {
            TR_CHECK(tr_session_eval(s, seq + i, 1) == 0);
            TR_CHECK_EQ_INT(tr_session_last_threads(s), want[i]);
            TR_CHECK_EQ_INT(tr_session_decode_threads(s), i < 5 ? 0 : 4);
            clear_winner_probes++;
        }
        TR_CHECK_EQ_INT(clear_winner_probes, 6);
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 4);
        tr_session_free(s);
    }

    /* (b) pairwise tie: narrow stays within its own margin of the wide (best) every round, so the
     * measurement extends all the way to TR_DECODE_TUNE_ROUNDS_MAX, then the narrower stands */
    {
        fake_clock fc = {0};
        fc.pool = pool;
        fc.narrow_w = 4;
        fc.wide_w = 8;
        static const double narrow[] = {1.02, 1.06, 1.08, 1.07, 1.09, 1.10}; /* min 1.02, second 1.06 always */
        static const double wide[] = {1.00, 1.00, 1.00, 1.00, 1.00, 1.00};
        fc.narrow = mk_seq(narrow, 6);
        fc.wide = mk_seq(wide, 6);
        tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        tr_session_set_tune_clock(s, fake_clock_now, &fc);
        int extension_probes = 0;
        for (int i = 0; i < 12; i++) {
            TR_CHECK(tr_session_eval(s, seq + i, 1) == 0);
            TR_CHECK_EQ_INT(tr_session_last_threads(s), i % 2 == 0 ? 4 : 8);
            TR_CHECK_EQ_INT(tr_session_decode_threads(s), i < 11 ? 0 : 4);
            if (i >= 6) extension_probes++;
        }
        TR_CHECK_EQ_INT(extension_probes, 6); /* 2 * (TR_DECODE_TUNE_ROUNDS_MAX - TR_DECODE_TUNE_ROUNDS) */
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 4); /* the narrower stands */
        tr_session_free(s);
    }
}

/* Crossing pos 32 (N_PROMPT=20, synth context 64) re-arms a measurement; a rewind below 32, then
 * crossing again, re-arms a second time -- a rewind lowers the remembered context class too. */
static void test_session_milestone(tr_model *model, tr_pool *pool) {
    fake_clock fc = {0};
    fc.pool = pool;
    fc.narrow_w = 4;
    fc.wide_w = 8;
    static const double narrow[] = {1.00, 1.00, 1.00};
    static const double wide[] = {2.00, 2.00, 2.00};
    fc.narrow = mk_seq(narrow, 3);
    fc.wide = mk_seq(wide, 3);
    tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
    TR_CHECK(s != NULL);
    tr_session_set_tune_clock(s, fake_clock_now, &fc);

    TR_CHECK(tr_session_eval(s, seq, N_PROMPT) == 0); /* a long pass, not a probe: pos = 20 */
    int64_t pos = N_PROMPT;
    for (int i = 0; i < 6; i++, pos++) TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0); /* first measurement: pos 26 */
    TR_CHECK_EQ_INT(tr_session_decode_threads(s), 4);

    int rearm_seen = 0;
    for (int cross = 0; cross < 2; cross++) {
        while (pos < 32) { /* steady, on the choice already in effect */
            TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0);
            pos++;
        }
        /* pos is now 32; that last eval was still steady -- the re-arm is decided only after it returns */
        TR_CHECK_EQ_INT(tr_session_last_threads(s), 4);
        TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0); /* probe 1 of the re-armed measurement: also 4, ambiguous */
        pos++;
        TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0); /* probe 2: the tell -- steady never shows 8 here */
        pos++;
        TR_CHECK_EQ_INT(tr_session_last_threads(s), 8);
        rearm_seen++;
        for (int i = 0; i < 4; i++, pos++) TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0); /* finish the measurement */
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 4);
        if (cross == 0) {
            TR_CHECK(tr_session_rewind(s, N_PROMPT) == 0);
            pos = N_PROMPT;
            TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0); /* pos 21, below 32: lowers the remembered class */
            pos++;
        }
    }
    TR_CHECK_EQ_INT(rearm_seen, 2);
    tr_session_free(s);
}

/* One steady one-token eval that stays inside [32, 64) via a rewind when it would leave that
 * range, so a milestone crossing never confounds the count of kept passes being tested. */
static int step_kept(tr_session *s, int *k, int64_t *pos) {
    if (*pos >= 60) {
        if (tr_session_rewind(s, 40) != 0) return -1;
        *pos = 40;
    }
    int rc = tr_session_eval(s, seq + (*k % N_SEQ), 1);
    (*k)++;
    if (rc == 0) (*pos)++;
    return rc;
}

/* The history of the session's choices (tr_session_decode_history): `want_n` measurements so far,
 * the newest ended at `pos` with `width` in effect and `picked` by the measurement. Returns 1 if
 * that entry shows a pick the debounce held back. */
static int check_history(const tr_session *s, int want_n, int64_t pos, int width, int picked) {
    const tr_decode_choice *h = NULL;
    int n = tr_session_decode_history(s, &h);
    TR_CHECK_EQ_INT(n, want_n);
    if (n != want_n || n < 1 || n > TR_DECODE_TUNE_HISTORY) return 0;
    TR_CHECK_EQ_INT(h[n - 1].pos, pos);
    TR_CHECK_EQ_INT(h[n - 1].width, width);
    TR_CHECK_EQ_INT(h[n - 1].picked, picked);
    return h[n - 1].picked != h[n - 1].width;
}

/* More measurements than the history keeps, pool 8: every crossing of 32 is one measurement (a
 * rewind below it lowers the class, test_session_milestone), the first TR_DECODE_TUNE_HISTORY - 1
 * stay where they are and the last slot holds the newest, told apart by its pick: every
 * measurement wants 8 but the last, which wants 4 and is held back. */
static void test_session_history_overflow(tr_model *model, tr_pool *pool) {
    enum { CROSSINGS = TR_DECODE_TUNE_HISTORY + 2, N_MEAS = CROSSINGS + 1 };
    fake_clock fc = {0};
    fc.pool = pool;
    fc.narrow_w = 4;
    fc.wide_w = 8;
    fc.narrow.n = fc.wide.n = N_MEAS * TR_DECODE_TUNE_ROUNDS;
    TR_CHECK(fc.narrow.n <= FAKE_SEQ_MAX);
    for (int i = 0; i < fc.narrow.n; i++) {
        int last = i >= (N_MEAS - 1) * TR_DECODE_TUNE_ROUNDS;
        fc.narrow.v[i] = last ? 1.00 : 2.00;
        fc.wide.v[i] = last ? 2.00 : 1.00;
    }
    tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
    TR_CHECK(s != NULL);
    if (s == NULL) return;
    tr_session_set_tune_clock(s, fake_clock_now, &fc);

    int64_t pos = 0;
    for (; pos < 30; pos++) TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0);
    check_history(s, 1, 6, 8, 8);
    for (int c = 0; c < CROSSINGS; c++) {
        TR_CHECK(tr_session_rewind(s, 30) == 0);
        /* 31, 32 (the class grows: armed), then the six probes */
        for (pos = 30; pos < 32 + 2 * TR_DECODE_TUNE_ROUNDS; pos++) TR_CHECK(tr_session_eval(s, seq + pos, 1) == 0);
    }
    const tr_decode_choice *h = NULL;
    int n = tr_session_decode_history(s, &h);
    TR_CHECK_EQ_INT(n, N_MEAS);
    int overflowed = n - TR_DECODE_TUNE_HISTORY;
    TR_CHECK(overflowed > 0);
    TR_CHECK_EQ_INT(h[0].pos, 6); /* the oldest stays */
    TR_CHECK_EQ_INT(h[0].picked, 8);
    TR_CHECK_EQ_INT(h[TR_DECODE_TUNE_HISTORY - 2].picked, 8);
    TR_CHECK_EQ_INT(h[TR_DECODE_TUNE_HISTORY - 1].pos, 32 + 2 * TR_DECODE_TUNE_ROUNDS); /* the newest, in the last slot */
    TR_CHECK_EQ_INT(h[TR_DECODE_TUNE_HISTORY - 1].width, 8);
    TR_CHECK_EQ_INT(h[TR_DECODE_TUNE_HISTORY - 1].picked, 4);
    tr_session_free(s);
}

/* Debounce inside a session, pool 8: a switch needs a width to win twice (raw == pending), and a
 * pending switch is also confirmed after TR_DECODE_TUNE_REMEASURE_KEPT kept passes force another
 * measurement -- or dropped, with no further probe, if that one agrees with the width already in
 * effect instead. */
static void test_session_debounce(tr_model *model, tr_pool *pool) {
    static const double favor_wide_n[] = {2.00, 2.00, 2.00}, favor_wide_w[] = {1.00, 1.00, 1.00};
    static const double favor_narrow_n[] = {1.00, 1.00, 1.00}, favor_narrow_w[] = {2.00, 2.00, 2.00};
    int switched = 0, held = 0, held_back = 0;

    /* variant A: the third measurement agrees with the width still pending (4) -- switches */
    {
        fake_clock fc = {0};
        fc.pool = pool;
        fc.narrow_w = 4;
        fc.wide_w = 8;
        double narrow[9], wide[9];
        memcpy(narrow, favor_wide_n, sizeof favor_wide_n);
        memcpy(narrow + 3, favor_narrow_n, sizeof favor_narrow_n);
        memcpy(narrow + 6, favor_narrow_n, sizeof favor_narrow_n); /* phase 3: 4 again */
        memcpy(wide, favor_wide_w, sizeof favor_wide_w);
        memcpy(wide + 3, favor_narrow_w, sizeof favor_narrow_w);
        memcpy(wide + 6, favor_narrow_w, sizeof favor_narrow_w);
        fc.narrow = mk_seq(narrow, 9);
        fc.wide = mk_seq(wide, 9);
        tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        tr_session_set_tune_clock(s, fake_clock_now, &fc);

        int k = 0;
        int64_t pos = 0;
        for (int i = 0; i < 6; i++) {
            TR_CHECK(tr_session_eval(s, seq + k, 1) == 0);
            k++;
            pos++;
        }
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 8); /* first measurement, from unset: adopted outright */
        held_back += check_history(s, 1, pos, 8, 8);
        while (pos < 32) {
            TR_CHECK(step_kept(s, &k, &pos) == 0);
        }
        for (int i = 0; i < 6; i++) {
            TR_CHECK(tr_session_eval(s, seq + (k % N_SEQ), 1) == 0);
            k++;
            pos++;
        }
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 8); /* the re-measure at 32 wants 4, stays 8: pending */
        held_back += check_history(s, 2, pos, 8, 4);

        int kept_steady = 0;
        for (int i = 0; i < TR_DECODE_TUNE_REMEASURE_KEPT - 1; i++) {
            TR_CHECK(step_kept(s, &k, &pos) == 0);
            TR_CHECK_EQ_INT(tr_session_last_threads(s), 8);
            kept_steady++;
        }
        TR_CHECK_EQ_INT(kept_steady, TR_DECODE_TUNE_REMEASURE_KEPT - 1);
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 8); /* still not confirmed */

        TR_CHECK(step_kept(s, &k, &pos) == 0); /* the 128th kept pass: re-arms */
        TR_CHECK_EQ_INT(tr_session_last_threads(s), 8);
        for (int i = 0; i < 6; i++) {
            TR_CHECK(tr_session_eval(s, seq + (k % N_SEQ), 1) == 0);
            k++;
            pos++;
        }
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 4); /* raw agrees with the pending vote: switches */
        held_back += check_history(s, 3, pos, 4, 4);
        switched++;
        tr_session_free(s);
    }
    TR_CHECK_EQ_INT(switched, 1);
    TR_CHECK_EQ_INT(held_back, 1); /* the history shows the pick the debounce held back, once */

    /* variant B: the third measurement agrees with the width already in effect (8) instead --
     * pending is cleared and no further probe follows, even after many more kept passes */
    {
        fake_clock fc = {0};
        fc.pool = pool;
        fc.narrow_w = 4;
        fc.wide_w = 8;
        double narrow[9], wide[9];
        memcpy(narrow, favor_wide_n, sizeof favor_wide_n);
        memcpy(narrow + 3, favor_narrow_n, sizeof favor_narrow_n);
        memcpy(narrow + 6, favor_wide_n, sizeof favor_wide_n); /* phase 3: 8 again */
        memcpy(wide, favor_wide_w, sizeof favor_wide_w);
        memcpy(wide + 3, favor_narrow_w, sizeof favor_narrow_w);
        memcpy(wide + 6, favor_wide_w, sizeof favor_wide_w);
        fc.narrow = mk_seq(narrow, 9);
        fc.wide = mk_seq(wide, 9);
        tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
        TR_CHECK(s != NULL);
        tr_session_set_tune_clock(s, fake_clock_now, &fc);

        int k = 0;
        int64_t pos = 0;
        for (int i = 0; i < 6; i++) {
            TR_CHECK(tr_session_eval(s, seq + k, 1) == 0);
            k++;
            pos++;
        }
        while (pos < 32) TR_CHECK(step_kept(s, &k, &pos) == 0);
        for (int i = 0; i < 6; i++) {
            TR_CHECK(tr_session_eval(s, seq + (k % N_SEQ), 1) == 0);
            k++;
            pos++;
        }
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 8); /* stays 8, pending 4 */
        for (int i = 0; i < TR_DECODE_TUNE_REMEASURE_KEPT; i++) TR_CHECK(step_kept(s, &k, &pos) == 0); /* re-arms */
        for (int i = 0; i < 6; i++) {
            TR_CHECK(tr_session_eval(s, seq + (k % N_SEQ), 1) == 0);
            k++;
            pos++;
        }
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 8); /* agrees with current: pending cleared, no switch */

        int steady_after = 0;
        for (int i = 0; i < 300; i++) {
            TR_CHECK(step_kept(s, &k, &pos) == 0);
            TR_CHECK_EQ_INT(tr_session_last_threads(s), 8); /* never a probe: pending was cleared */
            steady_after++;
        }
        TR_CHECK_EQ_INT(steady_after, 300);
        TR_CHECK_EQ_INT(tr_session_decode_threads(s), 8);
        held++;
        tr_session_free(s);
    }
    TR_CHECK_EQ_INT(held, 1);
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
 * the passes of row_pass rows. forced > 0 is the width asked for, 0 lets the session measure --
 * with a fake clock (a clear winner) for an exact schedule, or with the real one (use_real_clock)
 * for a loose sanity check only. Checks the threads of every pass and every row of logits against
 * ref. */
static void run_case(tr_model *model, tr_pool *pool, int forced, int use_real_clock, const float *ref) {
    int widths[TR_DECODE_TUNE_WIDTHS];
    int n_widths = tr_decode_tune_widths(POOL, widths);
    int want_forced = forced > POOL ? POOL : forced;
    tr_session *s = tr_session_create(model, 0, 0, NULL, 0);
    TR_CHECK(s != NULL);
    if (s == NULL) return;
    TR_CHECK_EQ_INT(tr_session_last_threads(s), 0);
    TR_CHECK_EQ_INT(tr_session_decode_threads(s), want_forced);

    fake_clock fc = {0};
    if (forced <= 0 && !use_real_clock) {
        fc.pool = pool;
        fc.narrow_w = widths[0];
        fc.wide_w = widths[n_widths - 1];
        static const double narrow_v[] = {1.00, 1.00, 1.00};
        static const double wide_v[] = {2.00, 2.00, 2.00};
        fc.narrow = mk_seq(narrow_v, 3);
        fc.wide = mk_seq(wide_v, 3);
        tr_session_set_tune_clock(s, fake_clock_now, &fc);
    }

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
        if (forced > 0) {
            TR_CHECK_EQ_INT(tr_session_last_threads(s), want_forced);
        } else if (use_real_clock) {
            int t = tr_session_last_threads(s);
            int is_width = t == POOL;
            for (int j = 0; j < n_widths; j++) is_width |= t == widths[j];
            TR_CHECK(is_width);
        } else {
            int want = measuring ? widths[k % n_widths] : tr_session_decode_threads(s);
            TR_CHECK_EQ_INT(tr_session_last_threads(s), want);
            if (k + 1 < n_widths * TR_DECODE_TUNE_ROUNDS) TR_CHECK_EQ_INT(tr_session_decode_threads(s), 0);
        }
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
    test_stats();
    test_pick();
    test_debounce();

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
            if (pi == 0) {
                test_session_extend(model, pool);
                test_session_milestone(model, pool);
                test_session_debounce(model, pool);
                test_session_history_overflow(model, pool);
            }
            for (size_t f = 0; f < sizeof forced / sizeof forced[0]; f++) {
                tr_model_set_decode_threads(model, forced[f]);
                run_case(model, pool, forced[f], 0, ref);
            }
            tr_model_set_decode_threads(model, -1); /* back to measuring */
            run_case(model, pool, 0, 0, ref);        /* fake clock: exact schedule, after being forced and back */
            run_case(model, pool, 0, 1, ref);        /* real clock: only the loose invariants */
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
    printf("  f32/q8_0, pool of %d: prompt wide, one-token passes measured then kept, forced 1..%d "
           "and beyond, extension and re-measurement, debounce: every logit identical to one thread\n",
           POOL, POOL);
    TR_TEST_EXIT();
}
