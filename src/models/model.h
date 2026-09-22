/* model.h — loading a model and running it.
 *
 * A tr_model holds weights and configuration and is immutable after load, so
 * several sessions (conversations) can share it. A tr_session owns its KV cache
 * and scratch buffers. Nothing here is global: two models can live in one process.
 * Each architecture ("olmoe", ...) implements tr_arch_vtable in src/models/<arch>.c
 * and is listed in src/models/model.c.
 *
 * Not quite immutable any more (docs/ARCHITECTURE.md Esperti M1): an architecture with a shared
 * expert store (src/memory/experts.h) refreshes which units are resident on every forward pass,
 * under a budget smaller than every expert. Like the pool, this makes one evaluation at a time
 * per model the rule -- already true today, since sessions of the same model already share it. */
#ifndef TR_MODEL_H
#define TR_MODEL_H

#include <stdint.h>
#include <stddef.h>

#include "../base/prof.h"
#include "../base/threads.h"
#include "../format/gguf.h"
#include "../memory/experts.h" /* tr_experts_stats: tr_model_expert_stats */

typedef struct {
    const char *arch;           /* general.architecture */
    int64_t vocab_size;
    int64_t n_ctx_train;        /* context the model was trained with */
    int64_t n_layers;
    int64_t n_embd;
    uint64_t weight_bytes;      /* tensor data loaded in memory */
} tr_model_info;

typedef struct tr_model tr_model;
typedef struct tr_session tr_session;

/* Most token positions of one pass whose logits a session can keep (tr_session_eval_rows).
 * The buffer costs this many rows of vocab floats, so it stays small on purpose. */
#define TR_LOGIT_ROWS_MAX 16

/* Opens `path` and loads it, keeping `pool` for compute. Dense tensors always land fully in
 * RAM; an architecture with a shared expert store (src/memory/experts.h) gets `expert_budget`
 * bytes for it: 0 asks for the automatic plan (resident if dense plus every expert passes the
 * memory guard, otherwise as much as the guard, the dense weights and a session's headroom
 * leave); UINT64_MAX asks for the smallest store that can run at all ("min"). NULL on failure
 * with a message in err. tr_model_load is this with 0, except that TR_EXPERT_BUDGET_MIB in the
 * environment (a number of MiB, or "min"), read once at load like TR_DECODE_ROWS, forces it --
 * for measurements, and for running the existing gates under a small store.
 * MEASUREMENT ONLY: TR_MEM_AVAILABLE_MIB in the environment, read once at load, replaces the RAM
 * the automatic plan (expert_budget == 0) sees as available, so its partial-store branch can be
 * exercised without a machine actually short on RAM; the memory guard that refuses an unsafe load
 * still queries the real machine, so safety never depends on it. TR_EXPERT_DIRECT=0 forces the
 * expert store to read through the ordinary buffered handle instead of trying an unbuffered one
 * first (docs/ARCHITECTURE.md Esperti M1, Step C), for the measurement that compares the two. */
tr_model *tr_model_load_budget(const char *path, tr_pool *pool, uint64_t expert_budget, char *err, size_t err_len);
tr_model *tr_model_load(const char *path, tr_pool *pool, char *err, size_t err_len);
void tr_model_free(tr_model *m);
const tr_model_info *tr_model_get_info(const tr_model *m);
/* Snapshot of the shared expert store's counters (src/memory/experts.h); -1 if this model's
 * architecture keeps no such store. */
int tr_model_expert_stats(const tr_model *m, tr_experts_stats *out);
/* MEASUREMENT ONLY (docs/MEASUREMENTS.md domanda 44): switches experts off. off is [n_layers][n_expert],
 * 1 = this expert is never chosen by its layer's router, which then takes the best of the
 * others; the softmax still runs over all of them. The output is no longer the model's own. NULL
 * clears the mask. -1 (nothing changes) if a layer would be left with fewer experts than a token
 * uses, or memory runs out. Copies off. Not during an evaluation. */
int tr_model_set_expert_mask(tr_model *m, const unsigned char *off);

/* A conversation with room for n_ctx tokens (n_ctx <= 0: the training context, at most
 * 4096). n_batch is the most tokens run in one forward pass (<= 0: 512; never more than
 * n_ctx): it sets the scratch memory and the speed of a long prompt, never a result. */
tr_session *tr_session_create(tr_model *m, int64_t n_ctx, int64_t n_batch, char *err, size_t err_len);
void tr_session_free(tr_session *s);
/* Appends n tokens and runs them, in forward passes of at most n_batch tokens. On success
 * the logits of the last token are available through tr_session_logits; they and the
 * cache are bit-identical for every n_batch and every way of splitting the tokens across
 * eval calls. -1 if the context is full or a token id is out of range (the session is
 * then unchanged). */
int tr_session_eval(tr_session *s, const int32_t *tokens, int64_t n);
/* Like tr_session_eval, but keeps the logits of the last n_logits tokens instead of the
 * last one: tr_session_logits_back(s, j) is then valid for 0 <= j < n_logits. Needs the
 * tokens to fit one pass, so 1 <= n_logits <= n <= n_batch and n_logits <= TR_LOGIT_ROWS_MAX;
 * -1 otherwise (the session is unchanged). Every row is bit-identical to the logits the
 * same token would give evaluated one at a time: that is what makes speculation exact. */
int tr_session_eval_rows(tr_session *s, const int32_t *tokens, int64_t n, int64_t n_logits);
/* vocab_size logits of the last evaluated token, valid until the next eval. */
const float *tr_session_logits(const tr_session *s);
/* vocab_size logits of the token `back` positions before the last one (back = 0: the last).
 * Only the rows the previous eval kept are valid; NULL for any other `back`. */
const float *tr_session_logits_back(const tr_session *s, int64_t back);
/* Number of tokens already in the KV cache. */
int64_t tr_session_pos(const tr_session *s);
/* Tokens the session has room for (what tr_session_create settled on). */
int64_t tr_session_n_ctx(const tr_session *s);
/* Most rows tr_session_eval_rows can keep: min(n_batch, TR_LOGIT_ROWS_MAX). A small -b
 * therefore caps how many tokens one pass can verify. */
int64_t tr_session_max_logit_rows(const tr_session *s);
/* Forgets every token after the first n (0 <= n <= pos): the next eval continues from
 * position n exactly as if only those n tokens had been evaluated. The logits are
 * not valid again until the next eval. -1 if n is out of range (nothing changes). */
int tr_session_rewind(tr_session *s, int64_t n);
/* The session's own profiler (docs/ARCHITECTURE.md §Profilazione), disabled by
 * default: set ->enabled and ->phase to turn it on. Never NULL. */
tr_prof *tr_session_prof(tr_session *s);

/* ---- routing trace (docs/MEASUREMENTS.md domande 13-15: does the next layer's router already know
 * what it will choose, and how much of the model would still need to come from disk) ----
 * Off by default and free when off: nothing changes until tr_session_route_trace_begin is
 * called. Once begun, every forward pass records its tokens' routing in evaluation order until
 * max_tokens is reached (recording then stops, but evaluation goes on); a rewind never removes
 * what was already recorded. For token t at layer L < n_layers - 1:
 *   pred_in[t][L]  layer L+1's router (its own gate_inp) applied to layer L's own FFN input
 *                  (ffn_norm_L(x), read right where layer L's own router reads it) -- known
 *                  before layer L's experts run.
 *   pred_out[t][L] layer L+1's router applied to layer L+1's own ffn_norm over layer L's output
 *                  x (after L's residual add, but before L+1's own attention runs) -- known
 *                  only after layer L's experts ran.
 * Both rank experts with route_token's own rule (softmax, repeated argmax, ties to the lower
 * id), keeping the first n_pred = min(TR_ROUTE_TRACE_PRED, n_expert), best first (never
 * re-sorted by id, unlike `chosen`). The last layer has no L+1: its predictions are 0xFFFF. */
#define TR_ROUTE_TRACE_PRED 16
typedef struct {
    int64_t n_tokens;     /* tokens recorded so far (evaluation order; a rewind removes none) */
    int64_t max_tokens, n_layers, n_expert, n_used, n_pred;
    int64_t expert_bytes; /* one expert of one layer: its gate + up + down rows, bytes as stored */
    int64_t layer_bytes;  /* every tensor of one layer, bytes as stored */
    const uint16_t *chosen;   /* [n_tokens][n_layers][n_used], increasing expert id */
    const uint16_t *pred_in;  /* [n_tokens][n_layers][n_pred], best first; 0xFFFF on the last layer */
    const uint16_t *pred_out; /* same shape as pred_in */
    const int32_t *tokens;    /* [n_tokens]: the id of every token recorded */
    const float *margins;     /* [n_tokens][n_layers][2]: the router's probability (softmax over every
                               * expert, before any renormalization) of the last expert chosen and
                               * of the best one left out: how close the choice was */
} tr_route_trace;

/* Allocates room for at most max_tokens tokens (0 ok: nothing is ever recorded); must be called
 * before the session's first eval. 0 on success, -1 out of memory or n_expert > 65535 (an id
 * would not fit a uint16_t). Replaces any trace already begun on this session. */
int tr_session_route_trace_begin(tr_session *s, int64_t max_tokens);
/* NULL if tr_session_route_trace_begin was never called on this session. */
const tr_route_trace *tr_session_route_trace(const tr_session *s);

/* ---- threads per phase (docs/MEASUREMENTS.md "Thread per fase") ----
 * A long pass (a prompt) is bound by compute and runs on the whole pool. A short pass, of at
 * most TR_DECODE_ROWS tokens (decoding, a short draft to verify), is bound by reading the
 * weights: once the memory bus is full more threads only add waiting, and how many fill it is
 * a fact of the machine, not of the model. So it is measured, not configured, and the margin is
 * the noise measured in the passes themselves, never a constant (docs/LESSONS.md #88: a fixed
 * 1% was tuned on a machine contaminated by stray load and picked a different width on a quiet
 * one). The widths probed are the whole pool, half and a quarter (tr_pool_set_active: the first
 * slots, distinct cores), dropping any under 4 threads, narrowest first.
 * A session times TR_DECODE_TUNE_ROUNDS one-token passes on every width in turn; tr_decode_tune_stats
 * turns a width's own passes into a center (the fastest one: noise only adds time) and a spread
 * (the gap to the second fastest). tr_decode_tune_pick keeps the narrowest width whose center is
 * within its own pairwise margin of the fastest width's — never a spread borrowed from a third
 * width. Too close to call: the session times one more round on just that pair, up to
 * TR_DECODE_TUNE_ROUNDS_MAX passes on any one width, then the narrower of the two stands.
 * tr_decode_tune_debounce then asks a new width to win twice in a row before the session actually
 * switches. The answer moves with the context (on the reference machine 4 threads are as fast as
 * 8 at 512 tokens and lose 4% at 2048) and with the heat, so the session measures again once
 * tr_session_pos crosses the next power of two from TR_DECODE_TUNE_REMEASURE_POS up (recomputed
 * from the current position every time, so a rewind lowers it back down too), or after
 * TR_DECODE_TUNE_REMEASURE_KEPT one-token passes if a switch is still waiting on its second vote.
 * Until the first choice is in, other short passes use the whole pool. The width changes the
 * speed, never a logit.
 * TR_DECODE_ROWS in the environment, read at load, moves where a short pass ends, for
 * measurements (0: no pass is short, which is the engine before threads per phase). */
#define TR_DECODE_ROWS 4                  /* what the dense kernel covers with one read of a weight row */
#define TR_DECODE_TUNE_WIDTHS 3           /* widths probed: the whole pool, half, a quarter */
#define TR_DECODE_TUNE_ROUNDS 3           /* one-token passes timed on every width before the first choice */
#define TR_DECODE_TUNE_ROUNDS_MAX 6       /* passes any one width gets at most; past it the candidate stands */
#define TR_DECODE_TUNE_REMEASURE_POS 32   /* first context length asking for a re-measure, then every doubling */
#define TR_DECODE_TUNE_REMEASURE_KEPT 128 /* kept passes on an unconfirmed switch before forcing a fresh measurement */

/* Forces the threads of every short pass of every session of this model (clamped to the
 * pool), measuring nothing; n_threads <= 0 goes back to measuring. */
void tr_model_set_decode_threads(tr_model *m, int n_threads);
/* Threads the short passes of this session run on: the forced number, or the measured one, or
 * 0 until the session has measured for the first time. */
int tr_session_decode_threads(const tr_session *s);
/* Threads the last eval of this session ran on (0 before the first). */
int tr_session_last_threads(const tr_session *s);
/* What every measurement of this session decided, oldest first: the position it ended at, the
 * width in effect after it and the width it picked (they differ when tr_decode_tune_debounce
 * held the pick back for a second vote). Returns how many measurements have finished; *out has
 * the first TR_DECODE_TUNE_HISTORY of them, and past that many its last slot holds the newest. */
#define TR_DECODE_TUNE_HISTORY 32
typedef struct {
    int64_t pos;
    int width, picked;
} tr_decode_choice;
int tr_session_decode_history(const tr_session *s, const tr_decode_choice **out);

/* ---- implemented once per architecture ----
 * model.c wraps the architecture's own objects: struct tr_model is
 * { const tr_arch_vtable *vt; void *impl; } and struct tr_session likewise, so an
 * architecture only ever sees its own `impl` pointers. Same contracts as above. */
typedef struct {
    const char *arch;
    /* takes ownership of g, also on failure; expert_budget: see tr_model_load_budget. path: the
     * same file g was opened from, kept only so an architecture with a shared expert store can
     * open a second, unbuffered handle on it (docs/ARCHITECTURE.md Esperti M1, Step C) -- not
     * retained beyond this call. */
    void *(*load)(const char *path, tr_gguf *g, tr_pool *pool, uint64_t expert_budget, char *err, size_t err_len);
    void (*free)(void *model);
    const tr_model_info *(*info)(const void *model);
    int (*expert_stats)(const void *model, tr_experts_stats *out); /* -1: no store */
    void *(*experts)(const void *model); /* the store itself, or NULL: tr_model_experts, tests only */
    void *(*session_create)(void *model, int64_t n_ctx, int64_t n_batch, char *err, size_t err_len);
    void (*session_free)(void *session);
    int (*eval)(void *session, const int32_t *tokens, int64_t n, int64_t n_logits);
    const float *(*logits)(const void *session, int64_t back);
    int64_t (*pos)(const void *session);
    int64_t (*n_ctx)(const void *session);
    int64_t (*max_logit_rows)(const void *session);
    int (*rewind)(void *session, int64_t n);
    tr_prof *(*prof)(void *session);
    int (*route_trace_begin)(void *session, int64_t max_tokens);
    const tr_route_trace *(*route_trace)(const void *session);
    int (*set_expert_mask)(void *model, const unsigned char *off);
} tr_arch_vtable;

#endif
