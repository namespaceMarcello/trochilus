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
    tr_pool *pool;      /* not owned; NULL: everything runs on the calling thread */
    int decode_threads; /* forced width of the short passes; 0: every session measures its own */
    int decode_rows;    /* a pass of at most this many tokens is a short one */
};

struct tr_session {
    const tr_arch_vtable *vt;
    void *impl;
    const tr_model *model;
    /* threads per phase (model.h): the widths still being measured, then the one kept */
    int width[TR_DECODE_TUNE_WIDTHS];
    double best_sec[TR_DECODE_TUNE_WIDTHS]; /* fastest one-token pass on width[i] */
    int n_widths;
    int measuring;    /* the next one-token passes are timed, width by width */
    int n_probes;     /* one-token passes timed so far in this measurement */
    int n_kept;       /* one-token passes run on the choice since it was made */
    int chosen;       /* 0 until the first measurement is over */
    int last_threads; /* of the last eval that ran */
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
    m->pool = pool;
    m->decode_threads = 0;
    m->decode_rows = TR_DECODE_ROWS;
    /* for measurements: where a short pass ends (0: no pass is short, every thread always) */
    const char *rows = getenv("TR_DECODE_ROWS");
    if (rows != NULL && atoi(rows) >= 0) m->decode_rows = atoi(rows);
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
    s->model = m;
    s->n_widths = tr_decode_tune_widths(m->pool != NULL ? tr_pool_size(m->pool) : 1, s->width);
    for (int i = 0; i < TR_DECODE_TUNE_WIDTHS; i++) s->best_sec[i] = 0.0;
    s->measuring = s->n_widths > 1; /* one width: nothing to measure */
    s->n_probes = 0;
    s->n_kept = 0;
    s->chosen = s->n_widths == 1 ? s->width[0] : 0;
    s->last_threads = 0;
    return s;
}

void tr_session_free(tr_session *s) {
    if (s == NULL) return;
    s->vt->session_free(s->impl);
    free(s);
}

int tr_decode_tune_widths(int pool_size, int width[TR_DECODE_TUNE_WIDTHS]) {
    int n = 0;
    if (pool_size < 1) pool_size = 1;
    for (int halvings = 0; halvings < TR_DECODE_TUNE_WIDTHS; halvings++) {
        int w = pool_size >> halvings;
        if (w < 1) w = 1;
        if (n == 0 || w < width[n - 1]) width[n++] = w;
    }
    return n;
}

int tr_decode_tune_pick(const double *best_sec, int n_widths) {
    double fastest = best_sec[0];
    for (int i = 1; i < n_widths; i++)
        if (best_sec[i] < fastest) fastest = best_sec[i];
    for (int i = 0; i < n_widths; i++)
        if (best_sec[i] <= fastest * (1.0 + TR_DECODE_TUNE_MARGIN)) return i;
    return 0;
}

/* hot: begin */
/* Every eval goes through here: the pass gets the threads of its phase (model.h), and the pool
 * goes back to whole before returning, for whoever uses it next. */
static int session_eval(tr_session *s, const int32_t *tokens, int64_t n, int64_t n_logits) {
    const tr_model *m = s->model;
    int size = m->pool != NULL ? tr_pool_size(m->pool) : 1;
    int threads = size, probe = 0, forced = m->decode_threads > 0;
    if (n <= m->decode_rows) {
        if (forced) threads = m->decode_threads < size ? m->decode_threads : size;
        else if (n == 1 && s->measuring) {
            probe = 1;
            threads = s->width[s->n_probes % s->n_widths];
        } else if (s->chosen > 0) threads = s->chosen;
    }

    tr_pool_set_active(m->pool, threads);
    double t0 = probe ? tr_time_sec() : 0.0;
    int rc = s->vt->eval(s->impl, tokens, n, n_logits);
    if (probe && rc == 0) {
        /* the fastest of the rounds: whatever else the machine does can only add time */
        double sec = tr_time_sec() - t0;
        int i = s->n_probes % s->n_widths;
        if (s->n_probes < s->n_widths || sec < s->best_sec[i]) s->best_sec[i] = sec;
        if (++s->n_probes == s->n_widths * TR_DECODE_TUNE_ROUNDS) {
            s->chosen = s->width[tr_decode_tune_pick(s->best_sec, s->n_widths)];
            s->measuring = 0;
            s->n_kept = 0;
        }
    } else if (rc == 0 && n == 1 && threads == s->chosen && !forced && s->n_widths > 1 &&
               ++s->n_kept == TR_DECODE_TUNE_AGAIN) {
        s->measuring = 1; /* the context has grown and the machine has warmed up: ask again */
        s->n_probes = 0;
    }
    tr_pool_set_active(m->pool, size);
    if (rc == 0) s->last_threads = threads;
    return rc;
}
/* hot: end */

int tr_session_eval(tr_session *s, const int32_t *tokens, int64_t n) {
    return session_eval(s, tokens, n, 1);
}

int tr_session_eval_rows(tr_session *s, const int32_t *tokens, int64_t n, int64_t n_logits) {
    return session_eval(s, tokens, n, n_logits);
}

const float *tr_session_logits_back(const tr_session *s, int64_t back) {
    return s->vt->logits(s->impl, back);
}

const float *tr_session_logits(const tr_session *s) {
    return s->vt->logits(s->impl, 0);
}

int64_t tr_session_pos(const tr_session *s) {
    return s->vt->pos(s->impl);
}

int64_t tr_session_n_ctx(const tr_session *s) {
    return s->vt->n_ctx(s->impl);
}

int64_t tr_session_max_logit_rows(const tr_session *s) {
    return s->vt->max_logit_rows(s->impl);
}

int tr_session_rewind(tr_session *s, int64_t n) {
    return s->vt->rewind(s->impl, n);
}

tr_prof *tr_session_prof(tr_session *s) {
    return s->vt->prof(s->impl);
}

void tr_model_set_decode_threads(tr_model *m, int n_threads) {
    m->decode_threads = n_threads > 0 ? n_threads : 0;
}

int tr_session_decode_threads(const tr_session *s) {
    int size = s->model->pool != NULL ? tr_pool_size(s->model->pool) : 1;
    if (s->model->decode_threads > 0) return s->model->decode_threads < size ? s->model->decode_threads : size;
    return s->chosen;
}

int tr_session_last_threads(const tr_session *s) {
    return s->last_threads;
}
