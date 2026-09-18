/* model_internal.h — helpers shared between model.c and each architecture's
 * tr_arch_vtable implementation (src/models/<arch>.c). Not part of the public
 * interface in model.h. */
#ifndef TR_MODEL_INTERNAL_H
#define TR_MODEL_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

/* Safety of the machine (docs/ARCHITETTURA.md): refuses if `available - needed`
 * would leave less than max(2 GiB, 10% of total RAM). 0 on success, -1 with a
 * message in err on refusal. If tr_mem_info cannot query the OS, allows the
 * allocation (nothing to check against) and logs a warning. */
int tr_mem_guard(uint64_t needed_bytes, char *err, size_t err_len);

/* Threads per phase (model.h): the widths a pool of pool_size threads is measured on, widest
 * first and distinct (the whole pool, half, a quarter), and the choice among them. Both are
 * here for the tests: the choice must not depend on the clock of the machine that runs them. */
#define TR_DECODE_TUNE_WIDTHS 3
int tr_decode_tune_widths(int pool_size, int width[TR_DECODE_TUNE_WIDTHS]);
/* best_sec[i]: the fastest one-token pass seen on width[i] threads. Returns the index of the
 * widest width within TR_DECODE_TUNE_MARGIN of the fastest: on the flat part of the curve more
 * threads cost nothing on a one-token pass and help the passes with more rows or more context. */
int tr_decode_tune_pick(const double *best_sec, int n_widths);

#endif
