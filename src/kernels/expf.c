/* expf.c — tr_expf: exp(x) correctly rounded to float, on every float (docs/MISURE.md question 37).
 *
 * The softmax of the attention and of the router, and the SiLU of the experts, take one
 * exponential per element. The C library's expf was the one kernel of the hot zone the engine
 * did not own: 30 ns a call with MinGW-w64 (x87 instructions), and not the same bits on two
 * platforms (glibc's is not the correctly rounded value for 170 648 floats). This one is the
 * definition: C, scalar, no library call inside, and the same float on every platform because
 * there is only one right answer, the float nearest to exp(x), ties to even.
 *
 *   exp(x) = 2^(k/64) * exp(r),   k = round(x * 64/ln2),   r = x - k * ln2/64,   |r| <= ln2/128
 *
 * in double: 2^(k/64) is one of the 64 values of 2^(j/64) times a power of two, exp(r) a Taylor
 * polynomial of degree 5. The double y this gives is within 2^-51.7 of exp(x) (the table entry
 * 2^-53, the last sum 2^-53, the polynomial 2^-54.7, the reduction far below: k * hi is exact),
 * so rounding it to float is right unless y falls that close to the middle between two floats.
 * The rounding test asks exactly that: if y * (1 - 2^-50) and y * (1 + 2^-50) round to the same
 * float, exp(x) rounds there too, subnormal results included. The arguments that fail it are
 * TR_EXPF_N_EXCEPTIONS out of 4 billion, and sit in a table with their exp computed at 200 bits
 * (tools/gen_expf_table.py, which writes every constant used here: none comes from a C library).
 *
 * The proof is exhaustive, not this argument: tests/bench_expf.c --check compares all the 2^32
 * floats with the reference, in `make check` under gcc and clang. An argument that failed the
 * rounding test without being in the table would give a NaN, which that check sees: a rounding
 * nobody proved never leaves this function as a number.
 *
 * Every operation below is an IEEE double operation with one correct result, so the bits do
 * not depend on the compiler, as long as it does not fuse a multiply with an add
 * (-ffp-contract=off, Makefile) and the rounding mode is to nearest. */
#include "kernels.h"

#include <stdint.h>

#include "expf_table.h"

typedef union {
    float f;
    uint32_t u;
} f32_word;

typedef union {
    double d;
    uint64_t u;
} f64_word;

/* k >> 6 below is floor(k / 64) for a negative k too */
_Static_assert((-1 >> 1) == -1, "tr_expf needs an arithmetic right shift of signed integers");

/* hot: begin */
static inline float expf_core(float x, int *path) {
    f32_word in, a, b, out;
    *path = TR_EXPF_SPECIAL;
    if (x != x) return x + x; /* NaN in, quiet NaN out */
    if (x > TR_EXPF_OVERFLOW_X) {
        out.u = 0x7F800000u; /* +infinity */
        return out.f;
    }
    if (x < TR_EXPF_UNDERFLOW_X) return 0.0f;
    *path = TR_EXPF_FAST;
    double xd = (double)x;
    /* round to the nearest integer without a library call: exact for |z| < 2^51 */
    double kd = (xd * TR_EXPF_INV_LN2_64 + 0x1.8p52) - 0x1.8p52;
    int64_t k = (int64_t)kd;
    double r = (xd - kd * TR_EXPF_LN2_64_HI) - kd * TR_EXPF_LN2_64_LO;
    double p = r + r * r * (0.5 + r * (TR_EXPF_C3 + r * (TR_EXPF_C4 + r * TR_EXPF_C5)));
    double t = tr_expf_pow2[k & 63];
    f64_word scale;
    scale.u = (uint64_t)((k >> 6) + 1023) << 52; /* 2^floor(k / 64), exact */
    double y = (t + t * p) * scale.d;
    /* the fast path is trusted where an error of TR_EXPF_MARGIN cannot move the rounding */
    a.f = (float)(y * (1.0 - TR_EXPF_MARGIN));
    b.f = (float)(y * (1.0 + TR_EXPF_MARGIN));
    if (a.u == b.u) return a.f;
    in.f = x;
    for (int i = 0; i < TR_EXPF_N_EXCEPTIONS; i++)
        if (tr_expf_exceptions[i][0] == in.u) {
            *path = TR_EXPF_TABLE;
            out.u = tr_expf_exceptions[i][1];
            return out.f;
        }
    *path = TR_EXPF_UNPROVEN;
    out.u = 0x7FC00000u; /* NaN: bench_expf --check proves no float gets here */
    return out.f;
}

float tr_expf(float x) {
    int path;
    return expf_core(x, &path);
}
/* hot: end */

int tr_expf_path(float x) {
    int path;
    (void)expf_core(x, &path);
    return path;
}

int tr_expf_n_exceptions(void) {
    return TR_EXPF_N_EXCEPTIONS;
}

void tr_expf_exception(int i, uint32_t *x_bits, uint32_t *y_bits) {
    *x_bits = tr_expf_exceptions[i][0];
    *y_bits = tr_expf_exceptions[i][1];
}
