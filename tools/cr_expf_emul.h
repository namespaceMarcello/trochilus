/* cr_expf_emul.h — for a MEASUREMENT build only, never for the engine (docs/MEASUREMENTS.md question 37).
 *
 * Included in front of the source files that call expf (tools/expf_quality.sh finds them; in
 * front of every file it would come before their _GNU_SOURCE), it turns each expf of the engine
 * into the correctly rounded value, computed the slow way: the double precision exp rounded to
 * float, which
 * tests/bench_expf.c and tools/expf_hard_cases.py show to be the correctly rounded exp on all
 * 2^32 floats. The binary it makes answers one question before anybody writes an expf of our
 * own: what changes in the logits and in the tokens on Linux, where glibc's expf is not the
 * correctly rounded value for 170 648 arguments (tools/expf_quality.sh).
 *
 *   make BUILD=build/linux-crexpf CC=gcc EXTRA_CFLAGS="-include tools/cr_expf_emul.h" \
 *        -W src/kernels/kernels.c build/linux-crexpf/src/kernels/kernels.o */
#ifndef TR_CR_EXPF_EMUL_H
#define TR_CR_EXPF_EMUL_H

#include <math.h>

static inline float tr_cr_expf_emul(float x) {
    return (float)exp((double)x);
}
#define expf tr_cr_expf_emul

#endif
