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

#endif
