/* proof_prep.c — CBMC: the token's prep, the facts lemmas A and B lean on, on the real code (make proof;
 * docs/MEASUREMENTS.md §The head's bound proved). Every function is one harness (cbmc --function <name>):
 *   digits  for every h code q in [-HB_LIM, HB_LIM]: hb_digits gives X and Xl with q = 256 X + Xl, both within int8
 *           (the conversion check: no cast out of range);
 *   up      for every double v of |v| <= FLT_MAX: hb_up(v) is the least float at or above v (the prep's g and mu are
 *           rounded up, never down), with nextafterf's model below;
 *   next_float  that model, against its definition, for every finite float;
 *   margin  for every n the head takes (a multiple of 256 up to TR_HB_COLS_MAX) and both types: hb_margin(n) is at least
 *           (D + 16) 2^-24, D the most roundings on any term's path to a row's value, counted here path by path from
 *           the code (lemma B: the engine's sum and the bound's, each term's product, its chain or lane, the folds). */
#include "proof.h"

/* CBMC 5.95 has no body for nextafterf (a call would return any float): the IEEE definition, by the bits, for finite
 * x toward +-infinity; next_float below checks it against its specification. */
float nextafterf(float x, float y) {
    if (x != x || y != y) return x + y;
    if (x == y) return y;
    if (x == 0.0f) return y > 0.0f ? 0x1p-149f : -0x1p-149f;
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    b = (x < y) == (x > 0.0f) ? b + 1 : b - 1;
    memcpy(&x, &b, sizeof b);
    return x;
}

/* the model above, for every finite float f and both directions: past f on the right side, no float in between */
void next_float(void) {
    const float f = nondet_float(), z = nondet_float();
    __CPROVER_assume(fabsf(f) < FLT_MAX); /* finite, below the largest */
    const float up = nextafterf(f, INFINITY), dn = nextafterf(f, -INFINITY);
    __CPROVER_assert(up > f && dn < f, "the next float is past f");
    __CPROVER_assert(!(z > f && z < up) && !(z < f && z > dn), "no float lies between f and the next");
}

void digits(void) {
    const long q = nondet_long();
    __CPROVER_assume(-HB_LIM <= q && q <= HB_LIM);
    int8_t X, Xl;
    hb_digits(q, &X, &Xl);
    __CPROVER_assert(256 * X + Xl == q, "q = 256 X + Xl");
}

void up(void) {
    const double v = nondet_double();
    __CPROVER_assume(fabs(v) <= FLT_MAX);
    const float f = hb_up(v);
    __CPROVER_assert((double)f >= v, "hb_up(v) >= v");
    __CPROVER_assert((double)nextafterf(f, -INFINITY) < v, "no float below hb_up(v) is at or above v");
}

static int64_t max3(int64_t a, int64_t b, int64_t c) {
    const int64_t m = a > b ? a : b;
    return m > c ? m : c;
}

void margin(void) {
    const int64_t n = 256 * (int64_t)nondet_int();
    __CPROVER_assume(n >= 256 && n <= TR_HB_COLS_MAX);
    const int64_t nb = n / 32, ns = n / 256;
    /* Q8_0. The engine (kernels.c k_dot_row_q8_0): w = d q exact (11 by 8 bits), one product w x, the lane's n / 16
     * terms after a first one added to 0 (exact), the 16 lanes' tree of depth 4. The bound (hb_bounds_q8_0): a chain
     * term (float)P coef, P exact below 2^21, coef = d a one rounding, the product one; nb / 4 terms a chain; then
     * (c0 + c2), + (c1 + c3), + side, and the eight lanes' tree of depth 3. A side term d be or |d| g, one product; nb / 4
     * terms a lane; then + into its lane and the tree. */
    const int64_t q8 = max3(1 + (n / 16 - 1) + 4, 2 + (nb / 4 - 1) + 6, 1 + (nb / 4 - 1) + 4);
    __CPROVER_assert(hb_margin(0, n) >= (double)(q8 + 16) * 0x1p-24, "Q8_0: the margin covers every path");
    /* Q4_K. The engine (q4x_block, k_q4x_dot2): integers exact, doubles (2^-53 a step, inside the 16), one float
     * rounding at the end. The bound (hb_bounds_q4_k): a chain term (float)P coef, coef = (d sc) a two roundings, the
     * product one; two sub-blocks a super-block into each chain: 2 ns terms; the fold of depth 6. A side term (d sc) be,
     * (|d| sc) g, (dmin m) H or (|dmin| m) mu, two roundings; four a super-block into lane j: 4 ns terms; then + into its
     * lane and the tree. */
    const int64_t q4 = max3(1, 3 + (2 * ns - 1) + 6, 2 + (4 * ns - 1) + 4);
    __CPROVER_assert(hb_margin(1, n) >= (double)(q4 + 16) * 0x1p-24, "Q4_K: the margin covers every path");
}
