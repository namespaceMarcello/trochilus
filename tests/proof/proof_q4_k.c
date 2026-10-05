/* proof_q4_k.c — CBMC: the Q4_K head's planes and the bound's integer part, on the real code (make proof;
 * docs/MEASUREMENTS.md §The head's bound proved, lemma A). Every function is one harness (cbmc --function <name>):
 *   planes_q4_k  for every super-block: the rebuilt super-block is the super-block byte for byte (the exact scores read
 *                it), and the hv the bound reads from the high plane gives each code (tr_q4_k_quant, the engine's own
 *                reading) q = 2 hv + bit with bit in {0, 1};
 *   bound_q4_k   for every super-block of codes, every element e (a free index) and the h codes of proof_q8_0.c at e
 *                and 0 elsewhere: sub-block j's lane l of the real hb_bounds_q4_k (hb_lane_q4_k) gives hv_e Q when e is
 *                one of its four elements and 0 otherwise, hv_e = q_e >> 1 from the engine's reading of the codes. A
 *                lane is linear in its digits, so for every h each lane gives the sum over its own four elements
 *                (proof_q8_0.c says why one element at a time). |P| < 2^21 and so (float)P exact: proof_enum.c.
 *                Not here, by reading: hb_bounds_q4_k's super-block offsets, chains, side lanes and fold.
 * The per-element inequality: proof_enum.c; the digits: proof_prep.c. */
#include "proof.h"

void planes_q4_k(void) {
    enum { N = 256 };
    unsigned char row[144], hi[112], lo[32], back[144];
    for (int i = 0; i < 144; i++) row[i] = nondet_uchar();
    tr_hb_planes(TR_TYPE_Q4_K, row, N, hi, lo);
    hb_rebuild_q4_k(hi, lo, N, back);
    for (int i = 0; i < 144; i++) __CPROVER_assert(back[i] == row[i], "the rebuilt super-block is the super-block");
    for (int k = 0; k < 256; k++) {
        const int q = (int)tr_q4_k_quant(row + TR_Q4_K_QS_OFFSET, k), bit = q - 2 * hb_hv_q4_k(hi, k);
        __CPROVER_assert(bit == 0 || bit == 1, "q = 2 hv + bit, bit in {0, 1}");
    }
}

void bound_q4_k(void) {
    enum { N = 256 };
    unsigned char row[144], hi[112], lo[32];
    for (int i = 0; i < 144; i++) row[i] = nondet_uchar();
    tr_hb_planes(TR_TYPE_Q4_K, row, N, hi, lo);
    static const long Qs[4] = {HB_LIM, -32513, 300, -1000};
    const int e = nondet_int(), v = nondet_int();
    __CPROVER_assume(0 <= e && e < 256 && 0 <= v && v < 4);
    int8_t Xe, Xle, X[256], Xl[256];
    hb_digits(Qs[v], &Xe, &Xle);
    for (int k = 0; k < 256; k++) {
        X[k] = k == e ? Xe : 0;
        Xl[k] = k == e ? Xle : 0;
    }
    const long want = (long)(tr_q4_k_quant(row + TR_Q4_K_QS_OFFSET, e) >> 1) * Qs[v];
    for (int j = 0; j < 8; j++)
        for (int l = 0; l < 8; l++)
            __CPROVER_assert(hb_lane_q4_k(hi, X, Xl, j, l) == (e / 4 == 8 * j + l ? want : 0),
                             "sub-block j's lane l gives hv_e Q when e is one of its four elements, else 0");
}
