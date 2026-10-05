/* proof_argmax.c — CBMC: lemma C, the argmax by the bound is the scan's row, on the real tr_hb_argmax (make proof;
 * docs/MEASUREMENTS.md §The head's bound proved). Every score S[r] any finite float, every bound U[r] any float at or
 * above it (an infinity included) or a NaN; the result is the scan's row, the first of the largest scores (kernels.c
 * tr_argmax_f32 over finite scores). Every function is one harness (cbmc --function <name>):
 *   argmax_q8_0        one worker, six rows, any prefix of them;
 *   argmax_q8_0_pool   two workers, five rows, each region cut at any row, a side each;
 *   argmax_q8_0_steal  the same cut, worker 0 taking both sides, the later first (a worker's several ranges, in any
 *                      order, as the balanced pool's stolen blocks);
 *   argmax_q4_k_pool   two workers through Q4_K's road (two rows a call), five rows.
 * The two-worker harnesses take S and U as the integers 0 to 10 (U at or above S, or NaN): the code never computes
 * with a score or a bound, it only compares them (>, >=, ==, x != x, against each other, -infinity and 0 with an index
 * guard), so its result depends only on their order and NaNs, and eleven values realize every order of ten floats
 * (equal ones too: -0 == +0 is one value). argmax_q8_0 checks every float, NaN and infinities among them; the floats
 * in the pooled model ran past 30 minutes (LESSONS #351).
 * Real: tr_hb_argmax, the two regions' bodies, the top list, the workers' merge, the full path's tr_hb_logits, the
 * scratch, the prep on h = 0. Stubbed, each only as the lemma needs: the kernel table (a row's bound is U[r], its exact
 * score S[r], read back from the rebuilt row's first bytes), the pool, the allocator, and tr_argmax_f32 by its rule
 * over finite scores, the first of the largest (kernels.c; tests/test_head_bound.c compares the two on real heads).
 * For every number of rows the argument is written in MEASUREMENTS; five rows past TR_HB_TOP = 4 reach every branch. */
#include "proof.h"

enum { NR = 6, COLS = 256 };
static float S[NR], Ub[NR];
static const unsigned char *g_hi;
static size_t g_hi_b;
static int g_two;

/* a row's index from its high plane's address, by comparison (a division of a free offset is a large circuit) */
static int32_t row_of(const unsigned char *hi) {
    int32_t r = -1;
    for (int32_t i = 0; i < NR; i++)
        if (hi == g_hi + (size_t)i * g_hi_b) r = i;
    __CPROVER_assert(r >= 0, "a row's own address");
    return r;
}
static int32_t rebuilt(const void *row) {
    int32_t r;
    memcpy(&r, row, sizeof r);
    return r;
}

static void stub_bounds(const unsigned char *hi, int64_t rows, const tr_hb_prep *p, int64_t n, float *U) {
    (void)p, (void)n;
    if (rows == 0) return; /* an empty range may point one past the last row */
    const int32_t r0 = row_of(hi);
    for (int64_t r = 0; r < rows; r++) U[r] = Ub[r0 + r];
}
static void stub_rebuild(const unsigned char *hi, const unsigned char *lo, int64_t n, unsigned char *row) {
    (void)lo, (void)n;
    const int32_t r = row_of(hi);
    memcpy(row, &r, sizeof r);
}
static float stub_dot_row(const void *row, const float *x, int64_t n) {
    (void)x, (void)n;
    return S[rebuilt(row)];
}
static void stub_dot2(const void *r0, const void *r1, const void *xq, int64_t n, float *out) {
    (void)xq, (void)n;
    out[0] = S[rebuilt(r0)];
    out[1] = S[rebuilt(r1)];
}
static void stub_prep(const float *x, int64_t n, void *xq) { (void)x, (void)n, (void)xq; }

static tr_kernels g_k;
const tr_kernels *tr_kernels_get(void) { return &g_k; }

/* the pool: NULL one call; otherwise two workers and the range cut at any row, and either each worker takes one side
 * (g_steal 0) or worker 0 takes both, the later first, as a worker that steals a block does (g_steal 1). Every chunk
 * to either worker in either order at once, or three chunks, ran past 30 minutes or out of the VM's memory */
static int g_steal;
int tr_pool_size(const tr_pool *p) { return p != NULL ? 2 : 1; }
static void run(tr_pool *p, int64_t n, tr_range_fn fn, void *ctx) {
    if (p == NULL) {
        fn(ctx, 0, n, 0);
        return;
    }
    const int64_t k = nondet_int();
    __CPROVER_assume(0 <= k && k <= n);
    if (g_steal) {
        fn(ctx, k, n, 0);
        fn(ctx, 0, k, 0);
    } else {
        fn(ctx, 0, k, 0);
        fn(ctx, k, n, 1);
    }
}
void tr_parallel_for(tr_pool *p, int64_t n, int64_t min_chunk, tr_range_fn fn, void *ctx) {
    (void)min_chunk;
    run(p, n, fn, ctx);
}
void tr_parallel_for_balanced(tr_pool *p, int64_t n, int64_t min_chunk, tr_range_fn fn, void *ctx) {
    (void)min_chunk;
    run(p, n, fn, ctx);
}
void *tr_alloc_aligned(size_t n, size_t align) {
    (void)align;
    return malloc(n);
}
void tr_free_aligned(void *p) { free(p); }
size_t tr_row_bytes(tr_type t, int64_t n) { return t == TR_TYPE_Q8_0 ? (size_t)n / 32 * 34 : (size_t)n / 256 * 144; }
size_t tr_q4x_bytes(int64_t n) { return (size_t)n; }
void tr_q4x_prepare(tr_pool *pool, const float *x, int64_t n, int64_t cols, void *xq) {
    (void)pool, (void)x, (void)n, (void)cols, (void)xq;
}

/* the scan's rule over finite scores: the first of the largest */
static int32_t scan(const float *y, int64_t n) {
    int32_t b = 0;
    for (int32_t r = 1; r < n; r++)
        if (y[r] > y[b]) b = r;
    return b;
}
int32_t tr_argmax_f32(tr_pool *pool, const float *x, int64_t n) {
    (void)pool;
    return scan(x, n);
}

/* nr rows of the head (at most NR), the call over all of them or (prefix) over any first n: two workers take five,
 * past TR_HB_TOP still, and all of them (one worker's harness covers the prefixes) */
static void check(tr_type t, int pooled, int nr, int prefix) {
    g_k.hb_bounds[t] = stub_bounds;
    g_k.hb_rebuild[t] = stub_rebuild;
    g_k.dot_row[TR_TYPE_Q8_0] = stub_dot_row;
    g_k.q4x_dot2 = stub_dot2;
    g_k.q4x_prep = stub_prep;
    static unsigned char hi[NR * 144], lo[NR * 128];
    static float h[COLS];
    g_hi = hi;
    g_hi_b = tr_hb_hi_bytes(t, COLS);
    for (int r = 0; r < nr; r++) {
        if (pooled) { /* the order abstraction: S and U as integers 0..10 or U NaN (the header says why it is enough) */
            const int si = nondet_int(), ui = nondet_int(), nan = nondet_int() & 1;
            __CPROVER_assume(0 <= si && si <= 10 && si <= ui && ui <= 10);
            S[r] = (float)si;
            Ub[r] = nan ? NAN : (float)ui;
        } else {
            S[r] = nondet_float();
            Ub[r] = nondet_float();
            __CPROVER_assume(S[r] - S[r] == 0.0f && (Ub[r] != Ub[r] || Ub[r] >= S[r])); /* S finite */
        }
    }
    const tr_hb_head w = {t, nr, COLS, hi, lo};
    tr_hb_scratch s;
    __CPROVER_assume(tr_hb_scratch_init(&s, &w, pooled ? 2 : 1, 1) == 0);
    int64_t n = nr;
    if (prefix) {
        n = nondet_int();
        __CPROVER_assume(1 <= n && n <= nr);
    }
    float y[NR];
    tr_hb_result res;
    const int32_t got = tr_hb_argmax(pooled ? (tr_pool *)(void *)&g_two : NULL, &w, h, n, &s, y, &res);
    __CPROVER_assert(got == scan(S, n), "the bound's argmax is the scan's row");
}

void argmax_q8_0(void) { check(TR_TYPE_Q8_0, 0, NR, 1); }
void argmax_q8_0_pool(void) { check(TR_TYPE_Q8_0, 1, NR - 1, 0); }
void argmax_q8_0_steal(void) {
    g_steal = 1;
    check(TR_TYPE_Q8_0, 1, NR - 1, 0);
}
void argmax_q4_k_pool(void) { check(TR_TYPE_Q4_K, 1, NR - 1, 0); }
