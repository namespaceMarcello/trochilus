/* model.h — loading a model and running it.
 *
 * A tr_model holds weights and configuration and is immutable after load, so
 * several sessions (conversations) can share it. A tr_session owns its KV cache
 * and scratch buffers. Nothing here is global: two models can live in one process.
 * Each architecture ("olmoe", ...) implements tr_arch_vtable in src/models/<arch>.c
 * and is listed in src/models/model.c. */
#ifndef TR_MODEL_H
#define TR_MODEL_H

#include <stdint.h>
#include <stddef.h>

#include "../base/prof.h"
#include "../base/threads.h"
#include "../format/gguf.h"

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

/* Opens `path`, loads every weight into memory (M0: no streaming yet) and
 * keeps `pool` for compute. NULL on failure with a message in err. */
tr_model *tr_model_load(const char *path, tr_pool *pool, char *err, size_t err_len);
void tr_model_free(tr_model *m);
const tr_model_info *tr_model_get_info(const tr_model *m);

/* A conversation with room for n_ctx tokens (n_ctx <= 0: the training context). */
tr_session *tr_session_create(tr_model *m, int64_t n_ctx, char *err, size_t err_len);
void tr_session_free(tr_session *s);
/* Appends n tokens and runs them. On success the logits of the last token are
 * available through tr_session_logits. -1 if the context is full or a token id
 * is out of range (the session is then unchanged). */
int tr_session_eval(tr_session *s, const int32_t *tokens, int64_t n);
/* vocab_size logits of the last evaluated token, valid until the next eval. */
const float *tr_session_logits(const tr_session *s);
/* Number of tokens already in the KV cache. */
int64_t tr_session_pos(const tr_session *s);
/* Forgets every token after the first n (0 <= n <= pos): the next eval continues from
 * position n exactly as if only those n tokens had been evaluated. The logits are
 * not valid again until the next eval. -1 if n is out of range (nothing changes). */
int tr_session_rewind(tr_session *s, int64_t n);
/* The session's own profiler (docs/ARCHITETTURA.md §Profilazione), disabled by
 * default: set ->enabled and ->phase to turn it on. Never NULL. */
tr_prof *tr_session_prof(tr_session *s);

/* ---- implemented once per architecture ----
 * model.c wraps the architecture's own objects: struct tr_model is
 * { const tr_arch_vtable *vt; void *impl; } and struct tr_session likewise, so an
 * architecture only ever sees its own `impl` pointers. Same contracts as above. */
typedef struct {
    const char *arch;
    void *(*load)(tr_gguf *g, tr_pool *pool, char *err, size_t err_len);   /* takes ownership of g, also on failure */
    void (*free)(void *model);
    const tr_model_info *(*info)(const void *model);
    void *(*session_create)(void *model, int64_t n_ctx, char *err, size_t err_len);
    void (*session_free)(void *session);
    int (*eval)(void *session, const int32_t *tokens, int64_t n);
    const float *(*logits)(const void *session);
    int64_t (*pos)(const void *session);
    int (*rewind)(void *session, int64_t n);
    tr_prof *(*prof)(void *session);
} tr_arch_vtable;

#endif
