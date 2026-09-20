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
 *   softmax, SiLU   tr_softmax and tr_swiglu are their definitions with tr_expf, bit for bit
 *
 * Seen red: tools/mutate_expf.sh (a wrong constant, the margin of the rounding test, a wrong
 * entry of the table, the thresholds, a softmax on the library's expf). */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "test.h"
#include "../src/kernels/kernels.h"

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
static void test_softmax_and_silu(void) {
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

int main(void) {
    test_special();
    test_fast();
    test_table();
    test_softmax_and_silu();
    TR_TEST_EXIT();
}
