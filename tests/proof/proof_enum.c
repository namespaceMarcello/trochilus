/* proof_enum.c — every case, enumerated: the per-element inequalities lemma A sums (make proof; docs/MEASUREMENTS.md
 * §The head's bound proved). A plain C program: the domains are small, so each fact is checked on every value of it.
 *   Q8_0: for every code q in [-128, 127], its top nibble u = (q + 128) >> 4 (what the bound reads: proof_q8_0.c) and
 *         every h code Q in [-32639, 32639]: |q Q - (16 u Q - 120.5 Q)| <= 7.5 |Q| (both sides: d of either sign);
 *   Q4_K: for every code q in [0, 15], hv = q >> 1 (proof_q4_k.c) and every Q: |q Q - (2 hv Q + Q / 2)| <= |Q| / 2;
 *   lanes: a lane's |P| at most 4 * 15 * 32639 (Q8_0) and 4 * 7 * 32639 (Q4_K), below 2^21: (float)P is exact.
 * Doubled to stay in integers. Prints the cases checked; exit 1 at the first that fails. */
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const long lim = 32639;
    long cases = 0;
    for (long q = -128; q <= 127; q++) {
        const long u = (q + 128) >> 4;
        for (long Q = -lim; Q <= lim; Q++, cases++)
            if (labs(2 * q * Q - (32 * u * Q - 241 * Q)) > 15 * labs(Q)) {
                printf("proof_enum: Q8_0 fails at q %ld, Q %ld\n", q, Q);
                return 1;
            }
    }
    for (long q = 0; q <= 15; q++)
        for (long Q = -lim; Q <= lim; Q++, cases++)
            if (labs(2 * q * Q - (4 * (q >> 1) * Q + Q)) > labs(Q)) {
                printf("proof_enum: Q4_K fails at q %ld, Q %ld\n", q, Q);
                return 1;
            }
    if (4 * 15 * lim >= 1L << 21 || 4 * 7 * lim >= 1L << 21) {
        printf("proof_enum: a lane's P reaches 2^21\n");
        return 1;
    }
    printf("proof_enum: %ld cases, every one holds\n", cases);
    return 0;
}
