/* model.c — architecture registry and the public wrappers of model.h.
 *
 * struct tr_model and struct tr_session wrap an architecture's own opaque
 * `impl` behind its tr_arch_vtable, so this file never looks inside a model's
 * weights or a session's KV cache: only the architecture in src/models/<arch>.c
 * does. Adding an architecture means writing its vtable and listing it below. */
#include "model.h"
#include "model_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../base/platform.h"

extern const tr_arch_vtable tr_olmoe_vtable;

static const tr_arch_vtable *const g_registry[] = {
    &tr_olmoe_vtable,
};
#define TR_REGISTRY_LEN (sizeof g_registry / sizeof g_registry[0])

struct tr_model {
    const tr_arch_vtable *vt;
    void *impl;
};

struct tr_session {
    const tr_arch_vtable *vt;
    void *impl;
};

int tr_mem_guard(uint64_t needed_bytes, char *err, size_t err_len) {
    tr_meminfo mi;
    if (tr_mem_info(&mi) != 0) {
        tr_log(TR_LOG_WARN, "could not query system memory; skipping the memory guard");
        return 0;
    }

    uint64_t two_gib = (uint64_t)2 * 1024 * 1024 * 1024;
    uint64_t min_free = mi.total_bytes / 10;
    if (min_free < two_gib) min_free = two_gib;

    int refuse = needed_bytes > mi.available_bytes || mi.available_bytes - needed_bytes < min_free;
    if (refuse) {
        double gib = 1024.0 * 1024.0 * 1024.0;
        double left = needed_bytes > mi.available_bytes
                           ? 0.0
                           : (double)(mi.available_bytes - needed_bytes) / gib;
        if (err != NULL) {
            snprintf(err, err_len,
                     "not enough memory: need %.2f GiB, only %.2f GiB available, "
                     "would leave %.2f GiB free (minimum %.2f GiB)",
                     (double)needed_bytes / gib, (double)mi.available_bytes / gib, left,
                     (double)min_free / gib);
        }
        return -1;
    }
    return 0;
}

tr_model *tr_model_load(const char *path, tr_pool *pool, char *err, size_t err_len) {
    char local_err[256];
    if (err == NULL) {
        err = local_err;
        err_len = sizeof local_err;
    }

    tr_gguf *g = tr_gguf_open(path, err, err_len);
    if (g == NULL) return NULL;

    const char *arch = NULL;
    if (tr_gguf_get_str(g, "general.architecture", &arch) != 0 || arch == NULL) {
        snprintf(err, err_len, "missing metadata key general.architecture");
        tr_gguf_close(g);
        return NULL;
    }

    const tr_arch_vtable *vt = NULL;
    for (size_t i = 0; i < TR_REGISTRY_LEN; i++) {
        if (strcmp(g_registry[i]->arch, arch) == 0) {
            vt = g_registry[i];
            break;
        }
    }
    if (vt == NULL) {
        snprintf(err, err_len, "unsupported architecture '%s'", arch);
        tr_gguf_close(g);
        return NULL;
    }

    void *impl = vt->load(g, pool, err, err_len); /* takes ownership of g, also on failure */
    if (impl == NULL) return NULL;

    tr_model *m = (tr_model *)malloc(sizeof *m);
    if (m == NULL) {
        vt->free(impl);
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    m->vt = vt;
    m->impl = impl;
    return m;
}

void tr_model_free(tr_model *m) {
    if (m == NULL) return;
    m->vt->free(m->impl);
    free(m);
}

const tr_model_info *tr_model_get_info(const tr_model *m) {
    return m->vt->info(m->impl);
}

tr_session *tr_session_create(tr_model *m, int64_t n_ctx, int64_t n_batch, char *err, size_t err_len) {
    char local_err[256];
    if (err == NULL) {
        err = local_err;
        err_len = sizeof local_err;
    }

    void *impl = m->vt->session_create(m->impl, n_ctx, n_batch, err, err_len);
    if (impl == NULL) return NULL;

    tr_session *s = (tr_session *)malloc(sizeof *s);
    if (s == NULL) {
        m->vt->session_free(impl);
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    s->vt = m->vt;
    s->impl = impl;
    return s;
}

void tr_session_free(tr_session *s) {
    if (s == NULL) return;
    s->vt->session_free(s->impl);
    free(s);
}

int tr_session_eval(tr_session *s, const int32_t *tokens, int64_t n) {
    return s->vt->eval(s->impl, tokens, n);
}

const float *tr_session_logits(const tr_session *s) {
    return s->vt->logits(s->impl);
}

int64_t tr_session_pos(const tr_session *s) {
    return s->vt->pos(s->impl);
}

int tr_session_rewind(tr_session *s, int64_t n) {
    return s->vt->rewind(s->impl, n);
}

tr_prof *tr_session_prof(tr_session *s) {
    return s->vt->prof(s->impl);
}
