/* model_internal.h — helpers shared between model.c and each architecture's
 * tr_arch_vtable implementation (src/models/<arch>.c). Not part of the public
 * interface in model.h. */
#ifndef TR_MODEL_INTERNAL_H
#define TR_MODEL_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "model.h" /* TR_DECODE_TUNE_WIDTHS, TR_DECODE_TUNE_ROUNDS_MAX: the tuner's shape */
#include "../memory/experts.h" /* tr_experts: tr_model_experts, tests only */

/* Safety of the machine (docs/ARCHITETTURA.md): refuses if `available - needed`
 * would leave less than max(2 GiB, 10% of total RAM). 0 on success, -1 with a
 * message in err on refusal. If tr_mem_info cannot query the OS, allows the
 * allocation (nothing to check against) and logs a warning. */
int tr_mem_guard(uint64_t needed_bytes, char *err, size_t err_len);

/* The automatic expert budget (docs/ARCHITETTURA.md Esperti M1, tr_model_load_budget): resident
 * (*budget = all_experts) when dense + all_experts passes the same rule tr_mem_guard checks
 * (available - needed >= reserve, reserve = max(2 GiB, total / 10)); otherwise *budget is what
 * available leaves after reserve, dense and session_allowance, capped to all_experts. -1 (budget
 * untouched) if that leftover is under min_bytes. Pure and total/available are parameters (not
 * read from the machine here) so tests drive it with made-up numbers. */
int tr_expert_budget_plan(uint64_t available, uint64_t total, uint64_t dense, uint64_t all_experts,
                          uint64_t session_allowance, uint64_t min_bytes, uint64_t *budget);

/* Tests only: the model's shared expert store, or NULL if its architecture keeps none. */
tr_experts *tr_model_experts(tr_model *m);

/* Threads per phase (model.h): the widths a pool of pool_size threads is probed on, narrowest
 * first, dropping any under 4 threads (a single width left: nothing to measure). Pure and
 * deterministic, so the tests do not depend on the clock of the machine that runs them. */
int tr_decode_tune_widths(int pool_size, int width[TR_DECODE_TUNE_WIDTHS]);

/* sec[0..n-1]: one width's own probe passes, n >= 2. center is the fastest of them (noise only
 * adds time); spread is (second fastest - fastest) / fastest, how noisy this width's passes were. */
void tr_decode_tune_stats(const double *sec, int n, double *center, double *spread);

/* center[i]/spread[i]: width i's stats, index 0 the narrowest, n-1 the widest. Returns the
 * narrowest width whose center is within a pairwise margin of the fastest (best) width's own:
 * margin = max(spread[i], spread[best]), never a spread borrowed from a third width. candidate
 * == best: *need_more = 0. Otherwise *need_more = 1: measure candidate and best for one more
 * round each and call again, up to TR_DECODE_TUNE_ROUNDS_MAX passes per width; past that the
 * candidate returned stands. */
int tr_decode_tune_pick(const double *center, const double *spread, int n, int *need_more);

/* Hysteresis on tr_decode_tune_pick's raw choice (a width, not an index): a new width must win
 * two measurements in a row (or agree with the one still pending) before the session switches to
 * it, so one noisy measurement cannot flip it back and forth. current == 0: no choice yet, adopt
 * raw outright. *pending persists between calls (0 when nothing is waiting). */
int tr_decode_tune_debounce(int current, int *pending, int raw);

/* Substitutes the clock a session's tuner reads both timestamps of a probe from (NULL, the
 * default set by tr_session_create: tr_time_sec()). Tests only, so a session's tuning schedule
 * can be driven by a made-up clock instead of the real one. */
void tr_session_set_tune_clock(tr_session *s, double (*now)(void *ctx), void *ctx);

#endif
