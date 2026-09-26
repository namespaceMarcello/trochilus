/* test_expf.c — tr_expf (src/kernels/expf.c): the borders, the two ways a result is settled, and
 * the softmax and the SiLU that are defined on it.
 *
 * The proof that tr_expf is the correctly rounded exp is tests/bench_expf.c --check, every float,
 * in `make check`. This is the quick part, and it says which branch it exercises: tr_expf_path
 * tells which way an argument went, and a count fails the test if a branch was never taken.
 *
 *   special   NaN, the infinities, the two thresholds and the floats next to them
 *   fast      one float every 1021 of the 2^32, against (float)exp((double)x): the reference
 *             that bench_expf and tools/expf_hard_cases.py show to be the correctly rounded value
 *   table     every entry of the table of exceptions takes the table, gives the value written
 *             there, and that value is the reference too
 *   softmax, SiLU   tr_softmax and tr_swiglu are their definitions with tr_expf, bit for bit,
 *             under every tier of the table (they call its expf_f32)
 *   expf_f32  every vector tier against tr_expf on all the 2^32 floats, in place and not, with
 *             tails of every length (question 70)
 *
 * Seen red: tools/mutate_expf.sh (a wrong constant, the margin of the rounding test, a wrong
 * entry of the table, the thresholds, a softmax on the library's expf). */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "test.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static uint32_t to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static uint32_t reference_bits(float x) {
    volatile double d = exp((double)x);
    return to_bits((float)d);
}

static void test_special(void) {
    const float overflow_x = 0x1.62e42ep+6f; /* the largest float whose exp is finite */
    int special = 0;
    TR_CHECK(tr_expf(from_bits(0x7FC00000u)) != tr_expf(from_bits(0x7FC00000u))); /* NaN */
    TR_CHECK(tr_expf(from_bits(0xFFC00001u)) != tr_expf(from_bits(0xFFC00001u)));
    TR_CHECK_EQ_INT(to_bits(tr_expf(INFINITY)), 0x7F800000u);
    TR_CHECK_EQ_INT(to_bits(tr_expf(-INFINITY)), 0u);
    TR_CHECK_EQ_INT(to_bits(tr_expf(3.4028234664e38f)), 0x7F800000u);
    TR_CHECK_EQ_INT(to_bits(tr_expf(-3.4028234664e38f)), 0u);
    special += tr_expf_path(from_bits(0x7FC00000u)) == TR_EXPF_SPECIAL;
    special += tr_expf_path(INFINITY) == TR_EXPF_SPECIAL;
    special += tr_expf_path(-INFINITY) == TR_EXPF_SPECIAL;
    TR_CHECK_EQ_INT(special, 3);

    /* the overflow threshold: finite on it, infinite on the next float, and the branch changes there */
    TR_CHECK_EQ_INT(tr_expf_path(overflow_x), TR_EXPF_FAST);
    TR_CHECK(tr_expf(overflow_x) < INFINITY);
    TR_CHECK_EQ_INT(to_bits(tr_expf(overflow_x)), reference_bits(overflow_x));
    TR_CHECK_EQ_INT(tr_expf_path(nextafterf(overflow_x, INFINITY)), TR_EXPF_SPECIAL);
    TR_CHECK_EQ_INT(to_bits(tr_expf(nextafterf(overflow_x, INFINITY))), 0x7F800000u);
    /* the underflow threshold: below -104 the branch answers zero, above it the fast path does,
     * down to the last argument whose exp is not zero */
    TR_CHECK_EQ_INT(tr_expf_path(-104.0f), TR_EXPF_FAST);
    TR_CHECK_EQ_INT(to_bits(tr_expf(-104.0f)), 0u);
    TR_CHECK_EQ_INT(tr_expf_path(nextafterf(-104.0f, -INFINITY)), TR_EXPF_SPECIAL);
    TR_CHECK_EQ_INT(to_bits(tr_expf(nextafterf(-104.0f, -INFINITY))), 0u);
    TR_CHECK_EQ_INT(to_bits(tr_expf(-103.97f)), 1u);           /* the smallest subnormal */
    TR_CHECK_EQ_INT(to_bits(tr_expf(-103.98f)), 0u);
    TR_CHECK_EQ_INT(to_bits(tr_expf(-87.4f)), reference_bits(-87.4f)); /* a subnormal result */
    TR_CHECK(tr_expf(-87.4f) > 0.0f && tr_expf(-87.4f) < 0x1p-126f);

    TR_CHECK_EQ_INT(to_bits(tr_expf(0.0f)), 0x3F800000u);
    TR_CHECK_EQ_INT(to_bits(tr_expf(-0.0f)), 0x3F800000u);
    TR_CHECK_EQ_INT(to_bits(tr_expf(from_bits(1u))), 0x3F800000u);          /* subnormal arguments */
    TR_CHECK_EQ_INT(to_bits(tr_expf(from_bits(0x80000001u))), 0x3F800000u);
    TR_CHECK_EQ_INT(to_bits(tr_expf(1.0f)), 0x402DF854u);                    /* e */
}

static void test_fast(void) {
    uint64_t fast = 0, bad = 0, unproven = 0;
    for (uint64_t u = 0; u < (1ull << 32); u += 1021) {
        float x = from_bits((uint32_t)u);
        if (x != x) continue;
        int path = tr_expf_path(x);
        fast += path == TR_EXPF_FAST;
        unproven += path == TR_EXPF_UNPROVEN;
        bad += to_bits(tr_expf(x)) != reference_bits(x);
    }
    TR_CHECK_EQ_INT(bad, 0);
    TR_CHECK_EQ_INT(unproven, 0);
    TR_CHECK(fast > 2000000); /* the fast path is what this loop exercises: about half of the floats are in range */
}

static void test_table(void) {
    int n = tr_expf_n_exceptions(), taken = 0;
    for (int i = 0; i < n; i++) {
        uint32_t xb, yb;
        tr_expf_exception(i, &xb, &yb);
        float x = from_bits(xb);
        taken += tr_expf_path(x) == TR_EXPF_TABLE;
        TR_CHECK_EQ_INT(to_bits(tr_expf(x)), yb);
        TR_CHECK_EQ_INT(yb, reference_bits(x));
        /* the neighbours of an exception are ordinary arguments */
        TR_CHECK_EQ_INT(to_bits(tr_expf(nextafterf(x, INFINITY))), reference_bits(nextafterf(x, INFINITY)));
        TR_CHECK_EQ_INT(to_bits(tr_expf(nextafterf(x, -INFINITY))), reference_bits(nextafterf(x, -INFINITY)));
    }
    TR_CHECK(n > 0);
    TR_CHECK_EQ_INT(taken, n); /* the table branch: every entry, or an entry is dead weight */
}

/* the lane contract of kernels.h, written again here: the sum of a softmax */
static float ref_lane_sum(const float *x, int64_t n) {
    float lane[TR_LANES] = {0};
    for (int64_t k = 0; k < n; k++) lane[k % TR_LANES] += x[k];
    float lo = ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
    float hi = ((lane[8] + lane[9]) + (lane[10] + lane[11])) + ((lane[12] + lane[13]) + (lane[14] + lane[15]));
    return lo + hi;
}

static float pseudo(unsigned *seed, float span) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)(*seed >> 8 & 0xFFFF) / 65536.0f - 0.5f) * span;
}

/* On Windows the library's expf gives the same bits and this cannot tell which one ran (there
 * tools/lint.py refuses expf( in the hot zone); on Linux glibc's differs on one argument in
 * 25 000, and the rows below take 100 000 exponentials. */
static void test_softmax_and_silu_tier(void) {
    enum { N = 4001, ROWS = 15 };
    static float x[N], got[N], want[N], gate[N];
    unsigned seed = 11u;
    int bad = 0;
    for (int row = 0; row < ROWS; row++) {
        int64_t n = row == 0 ? 1 : row == 1 ? 17 : N;
        for (int64_t i = 0; i < n; i++) {
            x[i] = pseudo(&seed, 24.0f);
            gate[i] = pseudo(&seed, 4.0f);
        }
        float m = x[0];
        for (int64_t i = 1; i < n; i++)
            if (x[i] > m) m = x[i];
        for (int64_t i = 0; i < n; i++) want[i] = tr_expf(x[i] - m);
        float sum = ref_lane_sum(want, n);
        for (int64_t i = 0; i < n; i++) want[i] /= sum;
        memcpy(got, x, sizeof got);
        tr_softmax(got, n);
        for (int64_t i = 0; i < n; i++) bad += to_bits(got[i]) != to_bits(want[i]);

        for (int64_t i = 0; i < n; i++) want[i] = x[i] / (1.0f + tr_expf(-x[i])) * gate[i];
        memcpy(got, x, sizeof got);
        tr_swiglu(NULL, got, gate, n);
        for (int64_t i = 0; i < n; i++) bad += to_bits(got[i]) != to_bits(want[i]);
    }
    TR_CHECK_EQ_INT(bad, 0);
}

static const char *const tier_names[] = {"scalar", "avx2", "avx512"};
enum { N_TIER_NAMES = sizeof tier_names / sizeof tier_names[0] };

/* tr_softmax and tr_swiglu take the active table's expf_f32: the same bits under every tier */
static void test_softmax_and_silu(void) {
    int tiers = 0;
    for (int t = 0; t < N_TIER_NAMES; t++) {
        const tr_kernels *k = tr_kernels_tier(tier_names[t]);
        if (k == NULL) continue;
        tr_kernels_set_active(k);
        test_softmax_and_silu_tier();
        tiers++;
    }
    tr_kernels_set_active(NULL);
    TR_CHECK(tiers > 0);
}

/* ---- every tier's expf_f32 against tr_expf on all the 2^32 floats (question 70) ----------------- *
 * Blocks of 2^16 consecutive bit patterns over a pool. Each block goes through every tier in two
 * calls whose split point moves with the block (a first call of 2^16 - (b mod 17) elements, then the
 * rest: tails of 0..16 and calls shorter than one register); odd blocks in place (y = x, as the
 * softmax calls it), even blocks out of place. Every lane is compared with tr_expf bit for bit (NaN
 * included: both return x + x). Branches: the settled lanes, the subnormal grid, the lanes handed to
 * tr_expf (about 1 in 33 000), the specials, the tails. Seen red: lane 3 of the AVX-512 tier one ulp
 * up (268 376 318 floats), the unsettled lanes keeping the fast guess instead of tr_expf (10 422 avx2,
 * 10 204 avx512: only this exhaustive loop sees it, the softmax rows do not). */
enum { ALL_BLOCK = 1 << 16, ALL_WORKERS = 4 };
static float all_x[ALL_WORKERS][ALL_BLOCK], all_y[ALL_WORKERS][ALL_BLOCK];
static uint32_t all_ref[ALL_WORKERS][ALL_BLOCK];

typedef struct {
    const tr_kernels *tiers[N_TIER_NAMES];
    int n_tiers;
    int64_t bad[ALL_WORKERS][N_TIER_NAMES];
    int64_t seen[ALL_WORKERS][N_TIER_NAMES];
} all_ctx;

static void all_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    all_ctx *c = (all_ctx *)ctx_;
    float *x = all_x[worker], *y = all_y[worker];
    uint32_t *ref = all_ref[worker];
    for (int64_t b = begin; b < end; b++) {
        for (int64_t i = 0; i < ALL_BLOCK; i++) {
            x[i] = from_bits((uint32_t)(b * ALL_BLOCK + i));
            ref[i] = to_bits(tr_expf(x[i]));
        }
        int64_t split = ALL_BLOCK - b % 17;
        for (int t = 0; t < c->n_tiers; t++) {
            if (b & 1) {
                memcpy(y, x, sizeof all_x[0]);
                c->tiers[t]->expf_f32(y, y, split);
                c->tiers[t]->expf_f32(y + split, y + split, ALL_BLOCK - split);
            } else {
                c->tiers[t]->expf_f32(x, y, split);
                c->tiers[t]->expf_f32(x + split, y + split, ALL_BLOCK - split);
            }
            int64_t bad = 0;
            for (int64_t i = 0; i < ALL_BLOCK; i++) bad += to_bits(y[i]) != ref[i];
            c->bad[worker][t] += bad;
            c->seen[worker][t] += ALL_BLOCK;
        }
    }
}

static void test_all_floats_every_tier(void) {
    static all_ctx c;
    memset(&c, 0, sizeof c);
    for (int t = 0; t < N_TIER_NAMES; t++) {
        const tr_kernels *k = tr_kernels_tier(tier_names[t]);
        if (k != NULL && t > 0) c.tiers[c.n_tiers++] = k; /* scalar is tr_expf itself */
    }
    if (c.n_tiers == 0) {
        printf("test_expf: no vector tier on this CPU, the exhaustive vector check is skipped\n");
        return;
    }
    tr_pool *pool = tr_pool_create(ALL_WORKERS);
    TR_CHECK(pool != NULL);
    if (pool == NULL) return;
    tr_parallel_for(pool, (int64_t)1 << 16, 1, all_body, &c);
    tr_pool_destroy(pool);
    for (int t = 0; t < c.n_tiers; t++) {
        int64_t bad = 0, seen = 0;
        for (int w = 0; w < ALL_WORKERS; w++) {
            bad += c.bad[w][t];
            seen += c.seen[w][t];
        }
        printf("test_expf: %s expf_f32, %lld of %lld floats different from tr_expf\n", c.tiers[t]->tier,
               (long long)bad, (long long)seen);
        TR_CHECK(seen == (int64_t)1 << 32);
        TR_CHECK(bad == 0);
    }
}

int main(void) {
    test_special();
    test_fast();
    test_table();
    test_softmax_and_silu();
    test_all_floats_every_tier();
    TR_TEST_EXIT();
}
