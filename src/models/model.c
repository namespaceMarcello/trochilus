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
    tr_gpu *gpu;        /* owned: opened by tr_model_set_gpu, closed by tr_model_free */
    int gpu_on;         /* new sessions run the decode's attention on gpu */
};

struct tr_session {
    const tr_arch_vtable *vt;
    void *impl;
    const tr_model *model;
    /* threads per phase (model.h): the widths still being measured, then the one kept */
    int width[TR_DECODE_TUNE_WIDTHS]; /* narrowest first */
    int n_widths;
    double sec[TR_DECODE_TUNE_WIDTHS][TR_DECODE_TUNE_ROUNDS_MAX]; /* width[i]'s own passes, this measurement */
    int n_pass[TR_DECODE_TUNE_WIDTHS]; /* passes recorded so far on width[i], this measurement */
    int measuring;        /* the next one-token passes are timed, width by width */
    int extending;        /* 0: round-robin over every width; 1: only pair[0]/pair[1], undecided */
    int pair[2];           /* the two widths (indices into width[]) still undecided, once extending */
    int next_in_pair;      /* which of pair[] the next probe in extending mode goes to (0 or 1) */
    int n_probes;          /* one-token probes run in the round-robin so far, this measurement */
    int rounds;            /* rounds completed this measurement, round-robin plus extension */
    int64_t last_milestone; /* context-length class last seen (a power of two >= REMEASURE_POS, or 0) */
    int n_kept;            /* one-token passes run on the choice since the measurement last armed */
    int chosen;            /* 0 until the first measurement is over; else the width in effect */
    int pending;           /* tr_decode_tune_debounce's state: a width waiting for a second vote */
    int last_threads;      /* of the last eval that ran */
    tr_decode_choice history[TR_DECODE_TUNE_HISTORY]; /* what every measurement decided, oldest first */
    int n_history;         /* measurements finished; past the array the newest takes the last slot */
    double (*tune_now)(void *ctx); /* a probe's clock; NULL (the default): tr_time_sec() */
    void *tune_now_ctx;
};

#ifdef TR_DRAFT_PROBE
/* tools/draft_probe.c: the architecture's session behind s (a diagnostic build only). */
void *tr_draft_probe_session(tr_session *s);
void *tr_draft_probe_session(tr_session *s) {
    return s->impl;
}
#endif

/* defined near tr_decode_tune_widths below; tr_session_create arms the first measurement too */
static void tune_arm(tr_session *s);

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

/* TR_EXPERT_BUDGET_MIB in the environment, read once here like TR_DECODE_ROWS: "min" or a
 * number of MiB. Forces the expert budget outright, whatever the caller asked tr_model_load_budget
 * for -- for measurements and for running the existing gates under a small store. Absent or
 * unparseable: budget is left untouched. */
static uint64_t apply_expert_budget_env(uint64_t budget) {
    const char *mib = getenv("TR_EXPERT_BUDGET_MIB");
    if (mib == NULL) return budget;
    if (strcmp(mib, "min") == 0) return UINT64_MAX;
    long long v = atoll(mib);
    return v > 0 ? (uint64_t)v * 1024 * 1024 : budget;
}

tr_model *tr_model_load_progress(const char *path, tr_pool *pool, uint64_t expert_budget, const tr_progress *progress,
                                 char *err, size_t err_len) {
    char local_err[256];
    if (err == NULL) {
        err = local_err;
        err_len = sizeof local_err;
    }
    expert_budget = apply_expert_budget_env(expert_budget);

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

    /* takes ownership of g, also on failure */
    void *impl = vt->load(path, g, pool, expert_budget, progress, err, err_len);
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
    m->gpu = NULL;
    m->gpu_on = 0;
    /* for measurements: where a short pass ends (0: no pass is short, every thread always) */
    const char *rows = getenv("TR_DECODE_ROWS");
    if (rows != NULL && atoi(rows) >= 0) m->decode_rows = atoi(rows);
    return m;
}

tr_model *tr_model_load_budget(const char *path, tr_pool *pool, uint64_t expert_budget, char *err, size_t err_len) {
    return tr_model_load_progress(path, pool, expert_budget, NULL, err, err_len);
}

tr_model *tr_model_load(const char *path, tr_pool *pool, char *err, size_t err_len) {
    return tr_model_load_budget(path, pool, 0, err, err_len);
}

int tr_model_expert_stats(const tr_model *m, tr_experts_stats *out) {
    return m->vt->expert_stats(m->impl, out);
}

int tr_model_set_expert_mask(tr_model *m, const unsigned char *off) {
    return m->vt->set_expert_mask(m->impl, off);
}

int tr_model_set_gpu(tr_model *m, int on, char *err, size_t err_len) {
    if (on && m->gpu == NULL) {
        m->gpu = tr_gpu_open(err, err_len);
        if (m->gpu == NULL) return -1;
    }
    m->gpu_on = on && m->gpu != NULL;
    m->vt->set_gpu(m->impl, m->gpu_on ? m->gpu : NULL);
    return 0;
}

const char *tr_model_gpu_name(const tr_model *m) {
    return m->gpu_on ? tr_gpu_name(m->gpu) : NULL;
}

int64_t tr_session_gpu_tokens(const tr_session *s) {
    return s->vt->gpu_tokens(s->impl);
}

tr_experts *tr_model_experts(tr_model *m) {
    return (tr_experts *)m->vt->experts(m->impl);
}

int tr_expert_budget_plan(uint64_t available, uint64_t total, uint64_t dense, uint64_t all_experts,
                          uint64_t session_allowance, uint64_t min_bytes, uint64_t *budget) {
    uint64_t two_gib = (uint64_t)2 * 1024 * 1024 * 1024;
    uint64_t reserve = total / 10;
    if (reserve < two_gib) reserve = two_gib;

    /* resident, exactly today's guard: available - (dense + all_experts) >= reserve */
    uint64_t needed_resident = dense + all_experts;
    if (needed_resident <= available && available - needed_resident >= reserve) {
        *budget = all_experts;
        return 0;
    }

    uint64_t used = reserve + dense + session_allowance;
    uint64_t have = used < available ? available - used : 0;
    if (have < min_bytes) return -1;
    *budget = have < all_experts ? have : all_experts;
    return 0;
}

void tr_model_free(tr_model *m) {
    if (m == NULL) return;
    m->vt->free(m->impl);
    tr_gpu_close(m->gpu); /* after the model: its sessions are gone, nothing uses the device */
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
    s->chosen = s->n_widths == 1 ? s->width[0] : 0;
    s->pending = 0;
    s->last_milestone = 0;
    s->last_threads = 0;
    s->n_history = 0;
    s->tune_now = NULL;
    s->tune_now_ctx = NULL;
    tune_arm(s);
    s->measuring = s->n_widths > 1; /* one width: nothing to measure */
    return s;
}

void tr_session_free(tr_session *s) {
    if (s == NULL) return;
    s->vt->session_free(s->impl);
    free(s);
}

void tr_session_set_tune_clock(tr_session *s, double (*now)(void *ctx), void *ctx) {
    s->tune_now = now;
    s->tune_now_ctx = ctx;
}

int tr_decode_tune_widths(int pool_size, int width[TR_DECODE_TUNE_WIDTHS]) {
    if (pool_size < 1) pool_size = 1;
    int tmp[TR_DECODE_TUNE_WIDTHS];
    int m = 0;
    for (int halvings = 0; halvings < TR_DECODE_TUNE_WIDTHS; halvings++) {
        int w = pool_size >> halvings;
        if (w < 1) w = 1;
        if (m == 0 || w < tmp[m - 1]) tmp[m++] = w;
    }
    /* tmp is widest first; a width under 4 threads reads a weight row without filling it, so
     * it is never worth probing, unless dropping it would leave nothing (the pool itself < 4). */
    int n = 0;
    for (int i = m - 1; i >= 0; i--)
        if (tmp[i] >= 4) width[n++] = tmp[i];
    if (n == 0) width[n++] = tmp[0];
    return n;
}

void tr_decode_tune_stats(const double *sec, int n, double *center, double *spread) {
    double best = sec[0] < sec[1] ? sec[0] : sec[1];
    double second = sec[0] < sec[1] ? sec[1] : sec[0];
    for (int i = 2; i < n; i++) {
        if (sec[i] < best) { second = best; best = sec[i]; }
        else if (sec[i] < second) second = sec[i];
    }
    *center = best;
    *spread = (second - best) / best;
}

int tr_decode_tune_pick(const double *center, const double *spread, int n, int *need_more) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (center[i] < center[best]) best = i;
    int candidate = best;
    for (int i = 0; i < n; i++) {
        double margin = spread[i] > spread[best] ? spread[i] : spread[best];
        if (center[i] <= center[best] * (1.0 + margin)) { candidate = i; break; }
    }
    *need_more = candidate != best;
    return candidate;
}

int tr_decode_tune_debounce(int current, int *pending, int raw) {
    if (current == 0) { *pending = 0; return raw; }
    if (raw == current) { *pending = 0; return current; }
    if (raw == *pending) { *pending = 0; return raw; }
    *pending = raw;
    return current;
}

static void tune_arm(tr_session *s) {
    s->measuring = 1;
    s->extending = 0;
    s->n_probes = 0;
    s->rounds = 0;
    s->n_kept = 0;
    for (int i = 0; i < s->n_widths; i++) s->n_pass[i] = 0;
}

/* Every width gets stats from its passes so far; on a pairwise tie the session times one more
 * round on just the candidate and the fastest, up to TR_DECODE_TUNE_ROUNDS_MAX; otherwise the
 * measurement is over and the raw choice goes through the debounce. */
static void tune_decide(tr_session *s) {
    double center[TR_DECODE_TUNE_WIDTHS], spread[TR_DECODE_TUNE_WIDTHS];
    for (int i = 0; i < s->n_widths; i++) tr_decode_tune_stats(s->sec[i], s->n_pass[i], &center[i], &spread[i]);
    int need_more = 0;
    int candidate = tr_decode_tune_pick(center, spread, s->n_widths, &need_more);
    if (need_more && s->rounds < TR_DECODE_TUNE_ROUNDS_MAX) {
        int best = 0;
        for (int i = 1; i < s->n_widths; i++)
            if (center[i] < center[best]) best = i;
        s->pair[0] = candidate;
        s->pair[1] = best;
        s->next_in_pair = 0;
        s->extending = 1;
    } else {
        s->chosen = tr_decode_tune_debounce(s->chosen, &s->pending, s->width[candidate]);
        s->measuring = 0;
        int slot = s->n_history < TR_DECODE_TUNE_HISTORY ? s->n_history : TR_DECODE_TUNE_HISTORY - 1;
        s->history[slot].pos = s->vt->pos(s->impl);
        s->history[slot].width = s->chosen;
        s->history[slot].picked = s->width[candidate];
        s->n_history++;
    }
}

/* The context-length class of pos: 0 below TR_DECODE_TUNE_REMEASURE_POS, else the largest power
 * of two <= pos from there up. A pure function of pos, so a rewind lowers it back down too. */
static int64_t tune_milestone(int64_t pos) {
    int64_t ms = TR_DECODE_TUNE_REMEASURE_POS;
    if (pos < ms) return 0;
    while (ms * 2 <= pos) ms *= 2;
    return ms;
}

/* A probe's clock (both timestamps): tr_time_sec() unless a test has substituted a deterministic
 * one (tr_session_set_tune_clock), so the tuning schedule in a test does not depend on the real
 * clock of the machine that runs it. */
static double tune_clock(const tr_session *s) {
    return s->tune_now != NULL ? s->tune_now(s->tune_now_ctx) : tr_time_sec();
}

/* hot: begin */
/* Every eval goes through here: the pass gets the threads of its phase (model.h), and the pool
 * goes back to whole before returning, for whoever uses it next. */
static int session_eval(tr_session *s, const int32_t *tokens, int64_t n, int64_t n_logits) {
    const tr_model *m = s->model;
    int size = m->pool != NULL ? tr_pool_size(m->pool) : 1;
    int threads = size, probe = 0, probe_w = 0, forced = m->decode_threads > 0;
    if (n <= m->decode_rows) {
        if (forced) threads = m->decode_threads < size ? m->decode_threads : size;
        else if (n == 1 && s->measuring) {
            probe = 1;
            probe_w = s->extending ? s->pair[s->next_in_pair] : s->n_probes % s->n_widths;
            threads = s->width[probe_w];
        } else if (s->chosen > 0) threads = s->chosen;
    }

    tr_pool_set_active(m->pool, threads);
    double t0 = probe ? tune_clock(s) : 0.0;
    int rc = s->vt->eval(s->impl, tokens, n, n_logits);

    if (rc == 0) {
        if (probe) {
            /* record the pass; tune_decide turns the pile into a center and a spread */
            s->sec[probe_w][s->n_pass[probe_w]++] = tune_clock(s) - t0;
            if (!s->extending) {
                if (++s->n_probes == s->n_widths * TR_DECODE_TUNE_ROUNDS) {
                    s->rounds = TR_DECODE_TUNE_ROUNDS;
                    tune_decide(s);
                }
            } else if ((s->next_in_pair ^= 1) == 0) {
                s->rounds++;
                tune_decide(s);
            }
        } else if (n == 1 && threads == s->chosen && !forced && s->n_widths > 1) {
            s->n_kept++;
        }

        if (!forced && s->n_widths > 1) {
            int64_t fm = tune_milestone(s->vt->pos(s->impl));
            if (!s->measuring && (fm > s->last_milestone ||
                                   (s->pending != 0 && s->n_kept >= TR_DECODE_TUNE_REMEASURE_KEPT))) {
                tune_arm(s);
            }
            s->last_milestone = fm;
        }
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

int tr_session_route_trace_begin(tr_session *s, int64_t max_tokens) {
    return s->vt->route_trace_begin(s->impl, max_tokens);
}

const tr_route_trace *tr_session_route_trace(const tr_session *s) {
    return s->vt->route_trace(s->impl);
}

void tr_model_set_decode_threads(tr_model *m, int n_threads) {
    m->decode_threads = n_threads > 0 ? n_threads : 0;
}

int tr_session_decode_threads(const tr_session *s) {
    int size = s->model->pool != NULL ? tr_pool_size(s->model->pool) : 1;
    if (s->model->decode_threads > 0) return s->model->decode_threads < size ? s->model->decode_threads : size;
    return s->chosen;
}

int tr_session_decode_history(const tr_session *s, const tr_decode_choice **out) {
    *out = s->history;
    return s->n_history;
}

int tr_session_last_threads(const tr_session *s) {
    return s->last_threads;
}
