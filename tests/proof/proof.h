/* proof.h — what every CBMC harness in tests/proof/ shares (make proof; docs/MEASUREMENTS.md §The head's bound proved).
 * The real src/kernels/head_bound.c is compiled into the harness, its static functions included: the proof runs on the
 * engine's own code, not on a model of it. The x86 tiers are left out (each gives the scalar definition's bits, a test
 * checks it: tests/test_head_bound.c, branch tiers), so the system headers come first and __x86_64__ is dropped after.
 * Nondeterministic inputs are functions without a body: CBMC gives them every value of their type. */
#ifndef TR_PROOF_H
#define TR_PROOF_H
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#undef __x86_64__
#include "../../src/kernels/head_bound.c"

unsigned char nondet_uchar(void);
int nondet_int(void);
long nondet_long(void);
float nondet_float(void);
double nondet_double(void);
#endif
