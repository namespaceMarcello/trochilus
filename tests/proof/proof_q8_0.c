/* proof_q8_0.c — CBMC: the Q8_0 head's planes and the bound's integer part, on the real code (make proof;
 * docs/MEASUREMENTS.md §The head's bound proved, lemma A). Every function is one harness (cbmc --function <name>):
 *   planes_q8_0  for every row of two blocks: the rebuilt row is the row byte for byte (the exact scores read it), and
 *                the u the bound reads from the high plane gives each code q = 16 (u - 8) + lo with lo in [0, 15];
 *   bound_q8_0   for every block of codes, every element e and the h codes Q = 32639, -32513, 300 and -1000 (the
 *                largest, digits of both signs, the two digits unequal) at e and 0 elsewhere: lane l of the real
 *                hb_bounds_q8_0 (hb_lane_q8_0) gives u_e Q when e is one of its four elements and 0 otherwise, u_e =
 *                (q_e + 128) >> 4 the row's own code. A lane is linear in its digits (P += u (256 X + Xl)), so for
 *                every h lane l gives sum over its own four elements of u_e Q_e: the code reads the right bits of the
 *                right element against that element's digits. (A sum over every Q at once is an adder tree's
 *                equivalence, beyond the solver in minutes: LESSONS #351.) |P| < 2^21, (float)P exact: proof_enum.c.
 *                Not here, by reading: hb_bounds_q8_0's block offsets, chains, side lanes and fold.
 * The per-element inequality: proof_enum.c; the digits: proof_prep.c. */
#include "proof.h"

static int s8(unsigned char c) { return c < 128 ? c : c - 256; }

void planes_q8_0(void) {
    enum { N = 64, NB = 2 };
    unsigned char row[NB * 34], hi[NB * 18], lo[NB * 16], back[NB * 34];
    for (int i = 0; i < NB * 34; i++) row[i] = nondet_uchar();
    tr_hb_planes(TR_TYPE_Q8_0, row, N, hi, lo);
    hb_rebuild_q8_0(hi, lo, N, back);
    for (int i = 0; i < NB * 34; i++) __CPROVER_assert(back[i] == row[i], "the rebuilt row is the row");
    for (int b = 0; b < NB; b++)
        for (int i = 0; i < 32; i++) {
            const int q = s8(row[34 * b + 2 + i]), u = hb_u_q8_0(hi + 2 * NB + 16 * b, i), l = q - 16 * (u - 8);
            __CPROVER_assert(0 <= l && l <= 15, "q = 16 (u - 8) + lo, lo in [0, 15]");
        }
}

void bound_q8_0(void) {
    enum { N = 32 };
    unsigned char row[34], hi[18], lo[16];
    for (int i = 0; i < 34; i++) row[i] = nondet_uchar();
    tr_hb_planes(TR_TYPE_Q8_0, row, N, hi, lo);
    static const long Qs[4] = {HB_LIM, -32513, 300, -1000};
    static int8_t X[32], Xl[32];
    for (int e = 0; e < 32; e++)
        for (int v = 0; v < 4; v++) {
            hb_digits(Qs[v], &X[e], &Xl[e]);
            const long want = (long)((s8(row[2 + e]) + 128) >> 4) * Qs[v];
            for (int l = 0; l < 8; l++)
                __CPROVER_assert(hb_lane_q8_0(hi + 2, X, Xl, l) == (e / 4 == l ? want : 0),
                                 "lane l gives u_e Q when e is one of its four elements, else 0");
            X[e] = Xl[e] = 0;
        }
}
