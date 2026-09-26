/* olmoe.c — tr_arch_vtable for "olmoe" (src/models/model.h).
 *
 * Reads hyperparameters and every weight tensor from GGUF metadata (M0: no
 * streaming, the whole model is loaded), then runs forward passes over blocks of
 * tokens: embedding -> N x [attention with GQA and NeoX RoPE, softmax-then-topk MoE
 * FFN, tokens grouped by expert] -> output norm -> logits of the last token.
 * Matches transformers' OlmoeForCausalLM (tools/.venv/.../modeling_olmoe.py):
 * q_norm/k_norm over the whole projection (not per head), router softmax over
 * all experts before top-k, ties broken to the lower expert id, optional
 * norm_topk_prob rescaling. */
#include "model.h"
#include "model_internal.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../base/platform.h"
#include "../kernels/kernels.h"
#include "../kv/kv.h"
#include "../memory/experts.h"

#ifdef TR_ATTN_PROBE
/* A diagnostic build only (make attn-probe, tools/attn_probe.c): sees every decode token's
 * attention, head by head. The engine is never built with it. */
void tr_attn_probe(int64_t layer, int64_t head, const float *q, const float *keys, const float *values,
                   int64_t n_pos, int64_t head_dim, float scale);
void tr_attn_probe_out(int64_t layer, int64_t head, const float *out, int64_t n_pos, int64_t head_dim);
#endif

#ifdef TR_DRAFT_PROBE
/* A diagnostic build only (make draft-probe, tools/draft_probe.c): a draft's routing (the top k of
 * the used experts, rescaled) and a decode token's attention over a subset of positions (a window,
 * or those the exact session's previous token attended to most). The probe returns 1 when it
 * computed the head itself. The engine is never built with it. */
void tr_draft_probe_route(float *sel_w, int64_t n_used);
int tr_draft_probe_attention(int64_t layer, int64_t head, const float *q, const float *keys, const float *values,
                             int64_t n_pos, int64_t head_dim, float scale, float *scores, int64_t score_stride,
                             float *out);
#endif

/* ---- weights ---------------------------------------------------------- */

typedef struct {
    float *attn_norm;   /* [n_embd] */
    float *ffn_norm;     /* [n_embd] */
    float *q_norm;        /* [n_qkv] */
    float *k_norm;          /* [n_kv] */
    tr_mat wq, wk, wv, wo;  /* wq/wk/wv: cols=n_embd; wo: cols=n_qkv, rows=n_embd */
    tr_mat gate_inp;         /* router: rows=n_expert, cols=n_embd */

    /* stacked expert tensors, raw ggml layout [in, out, n_expert], seen as one tr_mat per
     * expert (byte offset e * out * row_bytes(in)): [n_expert] each, one allocation */
    tr_mat *exps;
    const tr_mat *gate_exps; /* rows=n_ff, cols=n_embd */
    const tr_mat *up_exps;    /* rows=n_ff, cols=n_embd */
    const tr_mat *down_exps;   /* rows=n_embd, cols=n_ff */
} olmoe_layer;

/* Owns the buffers behind a session's tr_route_trace, defined near tr_session_route_trace_begin
 * below (docs/MEASUREMENTS.md domande 13-15). Opaque here: forward_pass only ever sees s->trace != NULL. */
typedef struct olmoe_route_trace olmoe_route_trace;

typedef struct {
    tr_pool *pool; /* not owned */
    tr_model_info info;

    int64_t n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, n_qkv, n_kv;
    int64_t n_expert, n_expert_used, vocab, n_ctx_train;
    float rms_eps, rope_freq_base, clamp;
    int have_clamp, norm_topk;

    float *output_norm; /* [n_embd] */
    tr_mat token_embd;    /* rows=vocab, cols=n_embd */
    tr_mat output;          /* rows=vocab, cols=n_embd */
    olmoe_layer *layers;      /* [n_layers] */

    tr_gguf *gguf;       /* kept open for the model's life: tr_experts reads straight from it
                          * (buffered), unless experts_file is not NULL */
    tr_file *experts_file; /* NULL: experts read through gguf's own buffered handle; else a second,
                            * unbuffered handle on the same path this model owns and closes (Step C,
                            * docs/ARCHITECTURE.md): tr_experts then reads with tr_file_alignment */
    tr_experts *experts; /* shared store for every layer's gate/up/down experts (Esperti M1) */
    unsigned char *expert_off; /* [n_layers][n_expert], 1: never chosen (tr_model_set_expert_mask,
                                * measurement only); NULL: every expert can be chosen */
    tr_gpu *gpu; /* not owned (tr_model_set_gpu): new sessions run the decode's attention there */

    /* every tr_alloc_aligned buffer owned by this model, freed on destroy */
    void **owned;
    size_t n_owned, cap_owned;
} olmoe_model;

/* Tokens in one forward pass when the caller does not choose (llama.cpp's n_ubatch). */
#define OLMOE_DEFAULT_BATCH 512
/* Consecutive tokens of a head whose attention runs as one group (tr_attention_group): the keys
 * and the values come from memory once per group. Measured at 4, 16 and 64 on a prompt of 4000:
 * the same speed (docs/MEASUREMENTS.md "Prefill su prompt lunghi"); 16 rows of scores stay in L2. */
#define OLMOE_ATTN_QUERIES 16
/* The work of a pass that belongs to one token alone (norms, RoPE, the write into the cache, the
 * router's choice, the rows copied for the experts) is split over the pool by token, at least
 * this many to a chunk: a decode token or a short pass stays on the calling thread. */
#define OLMOE_TOKENS_PER_CHUNK 8

typedef struct {
    olmoe_model *m; /* not owned */
    int64_t n_ctx;
    int64_t n_batch; /* most tokens in one forward pass (B below) */
    int64_t pos;
    tr_prof prof; /* disabled by default (zero-initialized in olmoe_session_create) */

    tr_kv kv; /* [n_layers][n_head_kv][n_ctx][head_dim]: one head's positions in a row (src/kv/kv.h) */
    /* The same keys and values in VRAM, and a decode token's attention run there with the same
     * bits (src/backend/gpu_attn.h); NULL: the CPU. Every pass writes both caches, so the CPU's is
     * always whole: a driver error sets gpu_failed and the session goes on without the GPU. */
    tr_gpu_attn *gpu;
    int gpu_failed;
    int gpu_warm;       /* this pass is the last of a multi-token eval: keep the GPU awake (gpu_attn.h) */
    int64_t gpu_tokens; /* decode tokens whose attention ran on the GPU */

    /* RoPE cos/sin per position: [n_ctx][head_dim / 2] each (tr_rope_table) */
    float *rope_cos, *rope_sin;

    /* scratch of one forward pass, reused every call; U = n_expert_used */
    float *x, *normed, *attn_out, *ffn_out; /* [B][n_embd] */
    float *q, *attn_concat;                 /* [B][n_qkv] */
    float *k, *v;                           /* [B][n_kv] */
    float *router;                          /* [B][n_expert] */
    int64_t *sel_id;  /* [B][U] experts of each token, increasing id */
    float *sel_w;     /* [B][U] their weights */
    int64_t *place;   /* [B][U] row of each (token, slot) in the grouped buffers below */
    int64_t *offsets; /* [n_expert + 1] grouped rows of each expert */
    int64_t *acquire_ids; /* [n_expert] scratch: this pass's non-empty experts of the current layer,
                           * for tr_experts_acquire (olmoe_refresh_experts, outside the hot zone) */
    unsigned char *taken; /* [n_workers][n_expert] scratch of the router's choice, a row per pool worker */
    float *xg;        /* [B*U][n_embd] expert inputs, grouped by expert */
    float *h1, *h2;   /* [B*U][n_ff] gate and up outputs */
    float *h3;        /* [B*U][n_embd] down outputs */
    float *logits;    /* [n_logits_max][vocab], oldest kept position first */
    int64_t n_logits_max; /* rows the buffer holds (TR_LOGIT_ROWS_MAX, never more than B) */
    int64_t n_logits;     /* rows the last pass filled */
    float *scores;    /* attention scores, a group of rows per pool worker: [n_workers][OLMOE_ATTN_QUERIES][score_stride] */
    int64_t score_stride; /* floats from a row of scores to the next: n_ctx and some (score_row_stride) */
    int64_t n_workers;
    tr_pm_scratch pm; /* the prompt's phase-major matmul (kernels.h tr_matmul_grouped_s) */

    olmoe_route_trace *trace; /* NULL: tracing off (docs/MEASUREMENTS.md domande 13-15) */

    /* [n_ctx][n_embd]: the whole prompt's hidden state, layer by layer (docs/MEASUREMENTS.md "Il
     * prefill legge il modello una volta per passata"). NULL when the expert store is fully
     * resident (nothing to save a re-read for); non-NULL only makes forward_prompt_layer_major
     * reachable, it does not force it (olmoe_eval still needs n > n_batch). */
    float *x_all;
    /* The layer whose units the next olmoe_refresh_experts may read ahead once it has acquired
     * its own (tr_experts_prefetch), -1: none. Set by forward_prompt_layer_major for the first
     * block of each layer but the last; used and cleared by olmoe_refresh_experts. */
    int64_t prefetch_layer;
} olmoe_session;

/* Reading ahead during a layer-major prompt (olmoe_refresh_experts): the next layer's missing
 * units are read while this one computes, when this layer's first block alone asked for at least
 * n_expert - n_expert / OLMOE_PREFETCH_SPARE_DIV of its experts. Measured on OLMoE-1B-7B (docs/
 * MEASUREMENTS.md): a 512-token block asks for nearly every expert of every layer, so the next
 * layer will too, and a unit read ahead is almost never one the prompt does not use. A short or
 * narrow block leaves the next layer's reads on demand, as without reading ahead. */
#define OLMOE_PREFETCH_SPARE_DIV 16

/* ---- loading helpers ---------------------------------------------------- */

static int track(olmoe_model *m, void *p) {
    if (p == NULL) return 0;
    if (m->n_owned == m->cap_owned) {
        size_t newcap = m->cap_owned ? m->cap_owned * 2 : 16;
        void **grown = (void **)realloc(m->owned, newcap * sizeof(void *));
        if (grown == NULL) {
            tr_free_aligned(p);
            return -1;
        }
        m->owned = grown;
        m->cap_owned = newcap;
    }
    m->owned[m->n_owned++] = p;
    return 0;
}

/* The load's progress (model.h tr_progress): bytes read so far against the load's total, handed
 * to the caller after every tensor read at load and every expert unit tr_experts_load_all reads. */
typedef struct {
    const tr_progress *to; /* NULL: nobody asked */
    uint64_t done, total;
} load_progress;

static void progress_add(load_progress *lp, uint64_t bytes) {
    if (lp->to == NULL || lp->to->fn == NULL) return;
    lp->done += bytes;
    lp->to->fn(lp->to->ctx, lp->done, lp->total);
}

static void progress_unit(void *ctx, uint64_t unit_bytes) {
    progress_add((load_progress *)ctx, unit_bytes);
}

static int req_u32(tr_gguf *g, const char *key, uint32_t *out, char *err, size_t err_len) {
    if (tr_gguf_get_u32(g, key, out) != 0) {
        snprintf(err, err_len, "missing metadata key '%s'", key);
        return -1;
    }
    return 0;
}

static int check_shape(const tr_gguf_tensor *t, uint32_t n_dims, uint64_t ne0, uint64_t ne1, uint64_t ne2,
                        char *err, size_t err_len) {
    int ok = t->n_dims == n_dims && t->ne[0] == ne0 &&
             (n_dims < 2 || t->ne[1] == ne1) &&
             (n_dims < 3 || t->ne[2] == ne2);
    if (!ok) {
        snprintf(err, err_len,
                 "tensor '%s' has shape [%" PRIu64 " x %" PRIu64 " x %" PRIu64 "] (%u dims), "
                 "expected [%" PRIu64 " x %" PRIu64 " x %" PRIu64 "] (%u dims)",
                 t->name, t->ne[0], t->ne[1], t->ne[2], t->n_dims, ne0, ne1, ne2, n_dims);
        return -1;
    }
    return 0;
}

/* Loads a 1-D weight, dequantized to an owned float array regardless of its
 * on-disk type (norms are elementwise weights, tr_rmsnorm wants plain float). */
static float *read_vec(tr_gguf *g, olmoe_model *m, load_progress *lp, const char *name, int64_t len, char *err,
                       size_t err_len) {
    const tr_gguf_tensor *t = tr_gguf_find_tensor(g, name);
    if (t == NULL) {
        snprintf(err, err_len, "missing tensor '%s'", name);
        return NULL;
    }
    if (!tr_kernels_support(t->type)) {
        snprintf(err, err_len, "tensor '%s' has unsupported type %u", name, (unsigned)t->type);
        return NULL;
    }
    if (check_shape(t, 1, (uint64_t)len, 1, 1, err, err_len) != 0) return NULL;

    void *raw = tr_alloc_aligned((size_t)t->n_bytes, 64);
    if (raw == NULL) {
        snprintf(err, err_len, "out of memory loading '%s'", name);
        return NULL;
    }
    if (tr_gguf_read(g, t, raw) != 0) {
        snprintf(err, err_len, "failed reading tensor '%s'", name);
        tr_free_aligned(raw);
        return NULL;
    }
    progress_add(lp, t->n_bytes);
    float *out = (float *)tr_alloc_aligned((size_t)len * sizeof(float), 64);
    if (out == NULL) {
        snprintf(err, err_len, "out of memory loading '%s'", name);
        tr_free_aligned(raw);
        return NULL;
    }
    tr_kernels_get()->dequant_row[t->type](raw, out, len);
    tr_free_aligned(raw);
    if (track(m, out) != 0) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    return out;
}

/* Loads a 2-D matmul weight, kept in its on-disk type (F32/F16/Q8_0). */
static int read_mat(tr_gguf *g, olmoe_model *m, load_progress *lp, const char *name, int64_t cols, int64_t rows,
                    tr_mat *out, char *err, size_t err_len) {
    const tr_gguf_tensor *t = tr_gguf_find_tensor(g, name);
    if (t == NULL) {
        snprintf(err, err_len, "missing tensor '%s'", name);
        return -1;
    }
    if (!tr_kernels_support(t->type)) {
        snprintf(err, err_len, "tensor '%s' has unsupported type %u", name, (unsigned)t->type);
        return -1;
    }
    if (check_shape(t, 2, (uint64_t)cols, (uint64_t)rows, 1, err, err_len) != 0) return -1;

    void *raw = tr_alloc_aligned((size_t)t->n_bytes, 64);
    if (raw == NULL) {
        snprintf(err, err_len, "out of memory loading '%s'", name);
        return -1;
    }
    if (tr_gguf_read(g, t, raw) != 0) {
        snprintf(err, err_len, "failed reading tensor '%s'", name);
        tr_free_aligned(raw);
        return -1;
    }
    progress_add(lp, t->n_bytes);
    if (track(m, raw) != 0) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    out->type = t->type;
    out->rows = rows;
    out->cols = cols;
    out->data = raw;
    return 0;
}

/* Validates a stacked expert weight's shape and type [ne0=in, ne1=out, ne2=n_expert] like
 * read_mat/read_vec, but never reads it: the shared store (src/memory/experts.h) reads it later,
 * part by part, straight off disk (Esperti M1). The caller records t->offset and t->type. */
static const tr_gguf_tensor *find_experts(tr_gguf *g, const char *name, int64_t ne0, int64_t ne1, int64_t ne2,
                                          char *err, size_t err_len) {
    const tr_gguf_tensor *t = tr_gguf_find_tensor(g, name);
    if (t == NULL) {
        snprintf(err, err_len, "missing tensor '%s'", name);
        return NULL;
    }
    if (!tr_kernels_support(t->type)) {
        snprintf(err, err_len, "tensor '%s' has unsupported type %u", name, (unsigned)t->type);
        return NULL;
    }
    if (check_shape(t, 3, (uint64_t)ne0, (uint64_t)ne1, (uint64_t)ne2, err, err_len) != 0) return NULL;
    return t;
}

/* One tr_mat view per expert of a stacked tensor: shape only, data = NULL until the shared store
 * fills it in for the layer's turn (olmoe_refresh_experts, below the hot zone). */
static void expert_shapes(tr_mat *exps, tr_type type, int64_t in_dim, int64_t out_dim, int64_t n_expert) {
    for (int64_t e = 0; e < n_expert; e++) {
        exps[e].type = type;
        exps[e].rows = out_dim;
        exps[e].cols = in_dim;
        exps[e].data = NULL;
    }
}

/* The expert store's only door to disk (src/memory/experts.h): positional reads straight on the
 * model's own GGUF file, kept open for the model's life. */
static int model_experts_read(void *ctx, void *buf, size_t n, uint64_t offset) {
    return tr_file_pread((const tr_file *)ctx, buf, n, offset);
}

/* ---- vtable: free ------------------------------------------------------- */

static void olmoe_free(void *model) {
    olmoe_model *m = (olmoe_model *)model;
    if (m == NULL) return;
    tr_experts_free(m->experts);
    tr_file_close(m->experts_file);
    free(m->expert_off);
    for (size_t i = 0; i < m->n_owned; i++) tr_free_aligned(m->owned[i]);
    free(m->owned);
    if (m->layers != NULL)
        for (int64_t L = 0; L < m->n_layers; L++) free(m->layers[L].exps);
    free(m->layers);
    tr_gguf_close(m->gguf);
    free(m);
}

/* ---- vtable: load -------------------------------------------------------- */

static void *olmoe_load(const char *path, tr_gguf *g, tr_pool *pool, uint64_t expert_budget,
                        const tr_progress *progress, char *err, size_t err_len) {
    tr_kernels_init();
    load_progress lp = {progress, 0, 0}; /* total: set once the store knows whether it is resident */

    olmoe_model *m = (olmoe_model *)calloc(1, sizeof *m);
    if (m == NULL) {
        snprintf(err, err_len, "out of memory");
        tr_gguf_close(g);
        return NULL;
    }
    m->pool = pool;
    m->gguf = g; /* kept open for the model's life; olmoe_free closes it, not this function */

    uint32_t block_count = 0, n_embd32 = 0, n_ff32 = 0, n_head32 = 0, n_head_kv32 = 0;
    uint32_t n_expert32 = 0, n_expert_used32 = 0, key_length = 0, ctx_length = 0;
    float rms_eps = 0.0f, rope_freq_base = 10000.0f, clamp = 0.0f;
    int have_clamp = 0, norm_topk = 0, norm_topk_b = 0;

    if (req_u32(g, "olmoe.block_count", &block_count, err, err_len) != 0) goto fail;
    if (req_u32(g, "olmoe.embedding_length", &n_embd32, err, err_len) != 0) goto fail;
    if (req_u32(g, "olmoe.feed_forward_length", &n_ff32, err, err_len) != 0) goto fail;
    if (req_u32(g, "olmoe.attention.head_count", &n_head32, err, err_len) != 0) goto fail;
    if (req_u32(g, "olmoe.attention.head_count_kv", &n_head_kv32, err, err_len) != 0) goto fail;
    if (req_u32(g, "olmoe.expert_count", &n_expert32, err, err_len) != 0) goto fail;
    if (req_u32(g, "olmoe.expert_used_count", &n_expert_used32, err, err_len) != 0) goto fail;

    if (tr_gguf_get_u32(g, "olmoe.attention.key_length", &key_length) != 0) {
        if (n_head32 == 0) {
            snprintf(err, err_len, "olmoe.attention.head_count is 0");
            goto fail;
        }
        key_length = n_embd32 / n_head32;
    }
    if (tr_gguf_get_f32(g, "olmoe.attention.layer_norm_rms_epsilon", &rms_eps) != 0) {
        snprintf(err, err_len, "missing metadata key 'olmoe.attention.layer_norm_rms_epsilon'");
        goto fail;
    }
    if (tr_gguf_get_f32(g, "olmoe.rope.freq_base", &rope_freq_base) != 0) rope_freq_base = 10000.0f;
    if (tr_gguf_get_u32(g, "olmoe.context_length", &ctx_length) != 0) ctx_length = 4096;
    if (tr_gguf_get_bool(g, "olmoe.expert_weights_norm", &norm_topk_b) == 0) norm_topk = norm_topk_b;
    if (tr_gguf_get_f32(g, "olmoe.attention.clamp_kqv", &clamp) == 0) have_clamp = 1;

    if (block_count == 0 || n_embd32 == 0 || n_ff32 == 0 || n_head32 == 0 || n_head_kv32 == 0 ||
        n_expert32 == 0 || n_expert_used32 == 0 || key_length == 0) {
        snprintf(err, err_len, "invalid (zero) hyperparameter in metadata");
        goto fail;
    }
    if (n_expert_used32 > n_expert32) {
        snprintf(err, err_len, "olmoe.expert_used_count (%u) exceeds olmoe.expert_count (%u)", n_expert_used32,
                 n_expert32);
        goto fail;
    }
    if (n_head32 % n_head_kv32 != 0) {
        snprintf(err, err_len, "olmoe.attention.head_count (%u) is not a multiple of head_count_kv (%u)", n_head32,
                 n_head_kv32);
        goto fail;
    }

    m->n_layers = block_count;
    m->n_embd = n_embd32;
    m->n_ff = n_ff32;
    m->n_head = n_head32;
    m->n_head_kv = n_head_kv32;
    m->head_dim = key_length;
    m->n_qkv = m->n_head * m->head_dim;
    m->n_kv = m->n_head_kv * m->head_dim;
    m->n_expert = n_expert32;
    m->n_expert_used = n_expert_used32;
    m->n_ctx_train = ctx_length;
    m->rms_eps = rms_eps;
    m->rope_freq_base = rope_freq_base;
    m->clamp = clamp;
    m->have_clamp = have_clamp;
    m->norm_topk = norm_topk;

    /* Shapes and file offsets of every expert tensor, no reads yet (Esperti M1): the shared
     * store (src/memory/experts.h) reads these straight off disk later, part by part, under
     * whatever budget this load settles on. Part 0 = gate, 1 = up, 2 = down, per layer, since a
     * GGUF file may quantize layers differently (part sizes are not assumed uniform). */
    int64_t n_units = m->n_layers * m->n_expert;
    size_t *part_bytes_tbl =
        (size_t *)tr_alloc_aligned(sizeof(size_t) * (size_t)(m->n_layers * TR_EXPERT_PARTS), 64);
    uint64_t *part_offset_tbl =
        (uint64_t *)tr_alloc_aligned(sizeof(uint64_t) * (size_t)(m->n_layers * TR_EXPERT_PARTS), 64);
    tr_type *part_type_tbl =
        (tr_type *)tr_alloc_aligned(sizeof(tr_type) * (size_t)(m->n_layers * TR_EXPERT_PARTS), 64);
    if (part_bytes_tbl == NULL || part_offset_tbl == NULL || part_type_tbl == NULL ||
        track(m, part_bytes_tbl) != 0 || track(m, part_offset_tbl) != 0 || track(m, part_type_tbl) != 0) {
        snprintf(err, err_len, "out of memory");
        goto fail;
    }
    uint64_t all_experts_bytes = 0; /* this file's real expert bytes, unpadded */
    for (int64_t L = 0; L < m->n_layers; L++) {
        char name[64];
        const tr_gguf_tensor *t;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_gate_exps.weight", L);
        t = find_experts(g, name, m->n_embd, m->n_ff, m->n_expert, err, err_len);
        if (t == NULL) goto fail;
        part_type_tbl[L * TR_EXPERT_PARTS + 0] = t->type;
        part_bytes_tbl[L * TR_EXPERT_PARTS + 0] = (size_t)m->n_ff * tr_row_bytes(t->type, m->n_embd);
        part_offset_tbl[L * TR_EXPERT_PARTS + 0] = t->offset;
        all_experts_bytes += t->n_bytes;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_up_exps.weight", L);
        t = find_experts(g, name, m->n_embd, m->n_ff, m->n_expert, err, err_len);
        if (t == NULL) goto fail;
        part_type_tbl[L * TR_EXPERT_PARTS + 1] = t->type;
        part_bytes_tbl[L * TR_EXPERT_PARTS + 1] = (size_t)m->n_ff * tr_row_bytes(t->type, m->n_embd);
        part_offset_tbl[L * TR_EXPERT_PARTS + 1] = t->offset;
        all_experts_bytes += t->n_bytes;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_down_exps.weight", L);
        t = find_experts(g, name, m->n_ff, m->n_embd, m->n_expert, err, err_len);
        if (t == NULL) goto fail;
        part_type_tbl[L * TR_EXPERT_PARTS + 2] = t->type;
        part_bytes_tbl[L * TR_EXPERT_PARTS + 2] = (size_t)m->n_embd * tr_row_bytes(t->type, m->n_ff);
        part_offset_tbl[L * TR_EXPERT_PARTS + 2] = t->offset;
        all_experts_bytes += t->n_bytes;
    }

    /* Step C (docs/ARCHITECTURE.md): read the experts without the operating system's page cache
     * when the platform and filesystem allow it, so the budget planned below is not spent twice
     * (docs/MEASUREMENTS.md M1 point 4: a resident store's own copy plus the OS's page cache copy of the
     * same bytes). TR_EXPERT_DIRECT=0 forces the buffered path outright, for the measurement that
     * compares the two. A resident store does not need this, but takes the same path anyway --
     * one path either way (docs/ARCHITECTURE.md). */
    uint64_t read_align = 1;
    void *experts_read_ctx = (void *)tr_gguf_file(g);
    {
        const char *want_direct = getenv("TR_EXPERT_DIRECT");
        if (want_direct == NULL || strcmp(want_direct, "0") != 0) {
            tr_file *df = tr_file_open_direct(path, NULL, 0);
            if (df != NULL) {
                /* some filesystems accept the unbuffered open but cannot actually service an
                 * aligned read on it (docker bind mounts, in particular): a trial read at the
                 * file's own start catches that before the store is built to depend on it. */
                unsigned char *probe = (unsigned char *)tr_alloc_aligned(TR_FILE_DIRECT_ALIGN, TR_FILE_DIRECT_ALIGN);
                int probe_ok = probe != NULL && tr_file_pread(df, probe, TR_FILE_DIRECT_ALIGN, 0) == 0;
                tr_free_aligned(probe);
                if (probe_ok) {
                    m->experts_file = df; /* owned from here on: olmoe_free closes it */
                    read_align = (uint64_t)tr_file_alignment(df);
                    experts_read_ctx = (void *)df;
                } else {
                    tr_file_close(df);
                }
            }
        }
    }

    /* Safety of the machine, and the expert budget (docs/ARCHITECTURE.md Esperti M1). Every
     * tensor in an OLMoE GGUF file is a model weight, so the sum over the whole directory minus
     * the expert tensors just sized above is the (exact) dense total. slot_bytes pads every
     * layer's parts to the biggest layer's needs, so resident_bytes (every unit, padded) is what
     * the store would actually allocate, not the raw all_experts_bytes above. */
    uint64_t slot_bytes = tr_experts_slot_bytes(part_bytes_tbl, m->n_layers, read_align);
    int64_t min_slots = tr_experts_min_slots(m->n_expert, m->n_expert_used);
    uint64_t min_bytes = (uint64_t)min_slots * slot_bytes;
    uint64_t resident_bytes = (uint64_t)n_units * slot_bytes;
    uint64_t dense_bytes;
    {
        uint64_t total_bytes = 0;
        uint64_t tcount = tr_gguf_tensor_count(g);
        for (uint64_t i = 0; i < tcount; i++) total_bytes += tr_gguf_tensor_at(g, i)->n_bytes;
        dense_bytes = total_bytes - all_experts_bytes;
    }

    uint64_t budget;
    if (expert_budget == UINT64_MAX) {
        budget = min_bytes; /* "min": exactly tr_experts_min_slots slots */
    } else if (expert_budget != 0) {
        budget = expert_budget; /* forced; above resident_bytes clamps to resident in tr_experts_create */
    } else {
        uint64_t session_allowance = tr_kv_bytes(m->n_layers, m->n_head_kv, m->head_dim,
                                                 m->n_ctx_train < 4096 ? m->n_ctx_train : 4096) +
                                     (uint64_t)512 * 1024 * 1024;
        tr_meminfo mi;
        if (tr_mem_info(&mi) != 0) {
            tr_log(TR_LOG_WARN, "could not query system memory; the expert store will be resident");
            budget = resident_bytes;
        } else {
            /* TR_MEM_AVAILABLE_MIB in the environment, read once here like TR_EXPERT_BUDGET_MIB
             * (model.c): substitutes the RAM the plan below sees as available, MEASUREMENT ONLY
             * (model.h tr_model_load_budget), so the partial branch can be exercised without a
             * machine actually short on RAM. tr_mem_guard just below still queries the real
             * machine, so safety never depends on this. */
            uint64_t available = mi.available_bytes;
            const char *avail_mib = getenv("TR_MEM_AVAILABLE_MIB");
            if (avail_mib != NULL) {
                long long v = atoll(avail_mib);
                if (v > 0) available = (uint64_t)v * 1024 * 1024;
            }
            if (tr_expert_budget_plan(available, mi.total_bytes, dense_bytes, resident_bytes, session_allowance,
                                      min_bytes, &budget) != 0) {
                uint64_t two_gib = (uint64_t)2 * 1024 * 1024 * 1024;
                uint64_t reserve = mi.total_bytes / 10;
                if (reserve < two_gib) reserve = two_gib;
                uint64_t used = reserve + dense_bytes + session_allowance;
                uint64_t have = used < available ? available - used : 0;
                double mib = 1024.0 * 1024.0;
                snprintf(err, err_len,
                         "not enough memory for the expert store: %.2f MiB short of the %.2f MiB minimum",
                         (double)(min_bytes - have) / mib, (double)min_bytes / mib);
                goto fail;
            }
        }
    }

    {
        uint64_t budget_slots = budget / slot_bytes;
        uint64_t slab_bytes = (budget_slots < (uint64_t)n_units ? budget_slots : (uint64_t)n_units) * slot_bytes;
        if (tr_mem_guard(dense_bytes + slab_bytes, err, err_len) != 0) goto fail;
        m->info.weight_bytes = dense_bytes + slab_bytes;
    }

    {
        tr_experts_config ecfg;
        memset(&ecfg, 0, sizeof ecfg);
        ecfg.n_layers = m->n_layers;
        ecfg.n_expert = m->n_expert;
        ecfg.n_used = m->n_expert_used;
        ecfg.part_bytes = part_bytes_tbl;
        ecfg.part_offset = part_offset_tbl;
        ecfg.budget_bytes = budget;
        ecfg.read = model_experts_read;
        ecfg.read_ctx = experts_read_ctx;
        ecfg.read_align = read_align;
        m->experts = tr_experts_create(&ecfg, err, err_len);
        if (m->experts == NULL) goto fail;
        /* what this load reads (model.h tr_progress): every dense tensor, and every expert unit
         * only when the store is resident -- counted from the same part sizes tr_experts_load_all
         * reports unit by unit, so the last report lands exactly on the total */
        int resident = tr_experts_resident(m->experts);
        uint64_t experts_at_load = 0;
        if (resident)
            for (int64_t i = 0; i < m->n_layers * TR_EXPERT_PARTS; i++)
                experts_at_load += (uint64_t)part_bytes_tbl[i] * (uint64_t)m->n_expert;
        lp.total = dense_bytes + experts_at_load;
        /* resident: fill every slot now, so forward_pass's acquire calls are hits from the very
         * first token and the reader is never called again (one path afterwards either way). */
        if (resident && tr_experts_load_all(m->experts, progress_unit, &lp) != 0) {
            snprintf(err, err_len, "failed reading the experts at load");
            goto fail;
        }
        /* partial: an I/O thread reads the next layer's units while a prompt computes this one
         * (forward_prompt_layer_major). TR_PREFETCH=0, read once here, keeps every read on
         * demand: the A/B measurement only (model.h). A store too small for it stays as it is. */
        const char *want_prefetch = getenv("TR_PREFETCH");
        if (!resident && (want_prefetch == NULL || strcmp(want_prefetch, "0") != 0) &&
            tr_experts_prefetch_start(m->experts) < 0)
            tr_log(TR_LOG_WARN, "could not start the expert store's I/O thread: every read stays on demand");
    }

    /* token_embd.weight: vocab size comes from its shape, not metadata. */
    {
        const tr_gguf_tensor *te = tr_gguf_find_tensor(g, "token_embd.weight");
        if (te == NULL) {
            snprintf(err, err_len, "missing tensor 'token_embd.weight'");
            goto fail;
        }
        if (!tr_kernels_support(te->type)) {
            snprintf(err, err_len, "tensor 'token_embd.weight' has unsupported type %u", (unsigned)te->type);
            goto fail;
        }
        if (te->n_dims != 2 || te->ne[0] != (uint64_t)m->n_embd) {
            snprintf(err, err_len, "tensor 'token_embd.weight' has unexpected shape");
            goto fail;
        }
        m->vocab = (int64_t)te->ne[1];
        if (m->vocab <= 0) {
            snprintf(err, err_len, "token_embd.weight has zero vocabulary");
            goto fail;
        }

        void *raw = tr_alloc_aligned((size_t)te->n_bytes, 64);
        if (raw == NULL) {
            snprintf(err, err_len, "out of memory");
            goto fail;
        }
        if (tr_gguf_read(g, te, raw) != 0) {
            snprintf(err, err_len, "failed reading tensor 'token_embd.weight'");
            tr_free_aligned(raw);
            goto fail;
        }
        progress_add(&lp, te->n_bytes);
        if (track(m, raw) != 0) {
            snprintf(err, err_len, "out of memory");
            goto fail;
        }
        m->token_embd.type = te->type;
        m->token_embd.rows = m->vocab;
        m->token_embd.cols = m->n_embd;
        m->token_embd.data = raw;
    }

    m->output_norm = read_vec(g, m, &lp, "output_norm.weight", m->n_embd, err, err_len);
    if (m->output_norm == NULL) goto fail;

    if (read_mat(g, m, &lp, "output.weight", m->n_embd, m->vocab, &m->output, err, err_len) != 0) goto fail;

    m->layers = (olmoe_layer *)calloc((size_t)m->n_layers, sizeof(olmoe_layer));
    if (m->layers == NULL) {
        snprintf(err, err_len, "out of memory");
        goto fail;
    }

    for (int64_t L = 0; L < m->n_layers; L++) {
        olmoe_layer *layer = &m->layers[L];
        char name[64];

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_norm.weight", L);
        layer->attn_norm = read_vec(g, m, &lp, name, m->n_embd, err, err_len);
        if (layer->attn_norm == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_q.weight", L);
        if (read_mat(g, m, &lp, name, m->n_embd, m->n_qkv, &layer->wq, err, err_len) != 0) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_k.weight", L);
        if (read_mat(g, m, &lp, name, m->n_embd, m->n_kv, &layer->wk, err, err_len) != 0) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_v.weight", L);
        if (read_mat(g, m, &lp, name, m->n_embd, m->n_kv, &layer->wv, err, err_len) != 0) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_output.weight", L);
        if (read_mat(g, m, &lp, name, m->n_qkv, m->n_embd, &layer->wo, err, err_len) != 0) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_q_norm.weight", L);
        layer->q_norm = read_vec(g, m, &lp, name, m->n_qkv, err, err_len);
        if (layer->q_norm == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_k_norm.weight", L);
        layer->k_norm = read_vec(g, m, &lp, name, m->n_kv, err, err_len);
        if (layer->k_norm == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_norm.weight", L);
        layer->ffn_norm = read_vec(g, m, &lp, name, m->n_embd, err, err_len);
        if (layer->ffn_norm == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_gate_inp.weight", L);
        if (read_mat(g, m, &lp, name, m->n_embd, m->n_expert, &layer->gate_inp, err, err_len) != 0) goto fail;

        layer->exps = (tr_mat *)calloc((size_t)m->n_expert * 3, sizeof(tr_mat));
        if (layer->exps == NULL) {
            snprintf(err, err_len, "out of memory");
            goto fail;
        }

        /* shapes and type only, already validated above: data comes from the shared store,
         * refreshed layer by layer as the forward pass runs (olmoe_refresh_experts) */
        expert_shapes(layer->exps, part_type_tbl[L * TR_EXPERT_PARTS + 0], m->n_embd, m->n_ff, m->n_expert);
        layer->gate_exps = layer->exps;

        expert_shapes(layer->exps + m->n_expert, part_type_tbl[L * TR_EXPERT_PARTS + 1], m->n_embd, m->n_ff,
                      m->n_expert);
        layer->up_exps = layer->exps + m->n_expert;

        expert_shapes(layer->exps + 2 * m->n_expert, part_type_tbl[L * TR_EXPERT_PARTS + 2], m->n_ff, m->n_embd,
                      m->n_expert);
        layer->down_exps = layer->exps + 2 * m->n_expert;
    }

    m->info.arch = "olmoe";
    m->info.vocab_size = m->vocab;
    m->info.n_ctx_train = m->n_ctx_train;
    m->info.n_layers = m->n_layers;
    m->info.n_embd = m->n_embd;

    return m; /* g stays open: m->gguf, closed by olmoe_free */

fail:
    olmoe_free(m); /* closes m->gguf */
    return NULL;
}

static const tr_model_info *olmoe_info(const void *model) {
    const olmoe_model *m = (const olmoe_model *)model;
    return &m->info;
}

static int olmoe_expert_stats(const void *model, tr_experts_stats *out) {
    const olmoe_model *m = (const olmoe_model *)model;
    tr_experts_get_stats(m->experts, out);
    return 0;
}

/* model.h: tr_model_set_gpu (the device is the wrapper's; sessions created afterwards use it) */
static void olmoe_set_gpu(void *model, tr_gpu *gpu) {
    ((olmoe_model *)model)->gpu = gpu;
}

static int64_t olmoe_gpu_tokens(const void *session) {
    return ((const olmoe_session *)session)->gpu_tokens;
}

/* Measurement only (model.h: tr_model_set_expert_mask). */
static int olmoe_set_expert_mask(void *model, const unsigned char *off) {
    olmoe_model *m = (olmoe_model *)model;
    if (off == NULL) {
        free(m->expert_off);
        m->expert_off = NULL;
        return 0;
    }
    for (int64_t L = 0; L < m->n_layers; L++) {
        int64_t left = 0;
        for (int64_t e = 0; e < m->n_expert; e++) left += off[L * m->n_expert + e] == 0;
        if (left < m->n_expert_used) return -1;
    }
    size_t n = (size_t)(m->n_layers * m->n_expert);
    unsigned char *copy = (unsigned char *)malloc(n);
    if (copy == NULL) return -1;
    for (size_t i = 0; i < n; i++) copy[i] = off[i] != 0;
    free(m->expert_off);
    m->expert_off = copy;
    return 0;
}

/* tests only (model_internal.h: tr_model_experts) */
static void *olmoe_experts(const void *model) {
    const olmoe_model *m = (const olmoe_model *)model;
    return m->experts;
}

/* ---- vtable: session ------------------------------------------------------ */

static void olmoe_route_trace_free(olmoe_route_trace *tr); /* below, after the hot zone */

#ifdef TR_DRAFT_PROBE
/* tools/draft_probe.c: the session's cache, overwritten with the exact session's cut to 16 bits. */
tr_kv *tr_draft_probe_kv(void *session);
tr_kv *tr_draft_probe_kv(void *session) {
    return &((olmoe_session *)session)->kv;
}
#endif

static void olmoe_session_free(void *session) {
    olmoe_session *s = (olmoe_session *)session;
    if (s == NULL) return;
    olmoe_route_trace_free(s->trace);
    tr_gpu_attn_free(s->gpu);
    tr_kv_free(&s->kv);
    tr_free_aligned(s->rope_cos);
    tr_free_aligned(s->rope_sin);
    tr_free_aligned(s->x);
    tr_free_aligned(s->normed);
    tr_free_aligned(s->attn_out);
    tr_free_aligned(s->ffn_out);
    tr_free_aligned(s->q);
    tr_free_aligned(s->attn_concat);
    tr_free_aligned(s->k);
    tr_free_aligned(s->v);
    tr_free_aligned(s->router);
    tr_free_aligned(s->sel_id);
    tr_free_aligned(s->sel_w);
    tr_free_aligned(s->place);
    tr_free_aligned(s->offsets);
    tr_free_aligned(s->acquire_ids);
    tr_free_aligned(s->taken);
    tr_free_aligned(s->xg);
    tr_free_aligned(s->h1);
    tr_free_aligned(s->h2);
    tr_free_aligned(s->h3);
    tr_free_aligned(s->logits);
    tr_free_aligned(s->scores);
    tr_pm_scratch_free(&s->pm);
    tr_free_aligned(s->x_all);
    free(s);
}

static float *alloc_f32(int64_t n) {
    return (float *)tr_alloc_aligned((size_t)n * sizeof(float), 64);
}

/* Floats from a row of attention scores to the next: n_ctx rounded up to 4 KiB of floats, plus
 * one cache line. With rows a multiple of 4 KiB apart, position t of every row of a group would
 * fall in the same set of the L1 cache. */
static int64_t score_row_stride(int64_t n_ctx) {
    return (n_ctx + 1023) / 1024 * 1024 + TR_LANES;
}

static void *olmoe_session_create(void *model, int64_t n_ctx, int64_t n_batch, char *err, size_t err_len) {
    olmoe_model *m = (olmoe_model *)model;

    int64_t actual_ctx = n_ctx;
    if (actual_ctx <= 0) actual_ctx = m->n_ctx_train < 4096 ? m->n_ctx_train : 4096;
    if (actual_ctx <= 0) actual_ctx = 4096;
    int64_t B = n_batch > 0 ? n_batch : OLMOE_DEFAULT_BATCH;
    if (B > actual_ctx) B = actual_ctx;

    int64_t n_workers = m->pool != NULL ? tr_pool_size(m->pool) : 1;
    int64_t U = m->n_expert_used;
    uint64_t kv_bytes = tr_kv_bytes(m->n_layers, m->n_head_kv, m->head_dim, actual_ctx);
    uint64_t rope_elems = (uint64_t)actual_ctx * (uint64_t)(m->head_dim / 2);
    uint64_t scratch_f32 = (uint64_t)B * (uint64_t)(m->n_embd * 4 + m->n_qkv * 2 + m->n_kv * 2 + m->n_expert + U) +
                           (uint64_t)B * (uint64_t)U * (uint64_t)(m->n_embd * 2 + m->n_ff * 2) +
                           (uint64_t)(n_workers * OLMOE_ATTN_QUERIES * score_row_stride(actual_ctx) + m->vocab) +
                           rope_elems * 2;
    uint64_t scratch_i64 = (uint64_t)B * (uint64_t)U * 2 + (uint64_t)m->n_expert + 1 + (uint64_t)m->n_expert;
    /* the phase-major matmul's scratch: the largest input a prompt's call passes (the experts' gathered rows),
     * the widest row, a panel per worker */
    const int64_t pm_cols = m->n_embd > m->n_ff ? (m->n_embd > m->n_qkv ? m->n_embd : m->n_qkv)
                                                : (m->n_ff > m->n_qkv ? m->n_ff : m->n_qkv);
    const int64_t pm_x = B * U * pm_cols > B * pm_cols ? B * U * pm_cols : B * pm_cols;
    uint64_t scratch_bytes = scratch_f32 * sizeof(float) + scratch_i64 * sizeof(int64_t) +
                             (uint64_t)(n_workers * m->n_expert) +
                             tr_pm_scratch_bytes((int)n_workers, m->n_expert, B * U, pm_cols, pm_x);

    /* Layer-major prefill (docs/MEASUREMENTS.md "Il prefill legge il modello una volta per passata")
     * only pays for itself, and only helps, when the store cannot hold the whole table: ask it
     * (tr_experts_get_stats), never guess from the budget the caller passed. Resident stores
     * (n_slots >= n_units) keep x_all NULL, so olmoe_eval takes today's pass-major loop. */
    int store_partial = 0;
    uint64_t x_all_bytes = 0;
    if (m->experts != NULL) {
        tr_experts_stats est;
        tr_experts_get_stats(m->experts, &est);
        store_partial = est.n_slots < est.n_units;
        if (store_partial) x_all_bytes = (uint64_t)actual_ctx * (uint64_t)m->n_embd * sizeof(float);
    }

    if (tr_mem_guard(kv_bytes + scratch_bytes + x_all_bytes, err, err_len) != 0) return NULL;

    olmoe_session *s = (olmoe_session *)calloc(1, sizeof *s);
    if (s == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    s->m = m;
    s->n_ctx = actual_ctx;
    s->n_batch = B;
    s->pos = 0;
    s->prefetch_layer = -1;

    int kv_rc = tr_kv_init(&s->kv, m->n_layers, m->n_head_kv, m->head_dim, actual_ctx);
    s->rope_cos = (float *)tr_alloc_aligned((size_t)rope_elems * sizeof(float), 64);
    s->rope_sin = (float *)tr_alloc_aligned((size_t)rope_elems * sizeof(float), 64);
    s->n_workers = n_workers;
    s->x = alloc_f32(B * m->n_embd);
    s->normed = alloc_f32(B * m->n_embd);
    s->attn_out = alloc_f32(B * m->n_embd);
    s->ffn_out = alloc_f32(B * m->n_embd);
    s->q = alloc_f32(B * m->n_qkv);
    s->attn_concat = alloc_f32(B * m->n_qkv);
    s->k = alloc_f32(B * m->n_kv);
    s->v = alloc_f32(B * m->n_kv);
    s->router = alloc_f32(B * m->n_expert);
    s->sel_id = (int64_t *)tr_alloc_aligned((size_t)(B * U) * sizeof(int64_t), 64);
    s->sel_w = alloc_f32(B * U);
    s->place = (int64_t *)tr_alloc_aligned((size_t)(B * U) * sizeof(int64_t), 64);
    s->offsets = (int64_t *)tr_alloc_aligned((size_t)(m->n_expert + 1) * sizeof(int64_t), 64);
    s->acquire_ids = (int64_t *)tr_alloc_aligned((size_t)m->n_expert * sizeof(int64_t), 64);
    s->taken = (unsigned char *)tr_alloc_aligned((size_t)(n_workers * m->n_expert), 64);
    s->xg = alloc_f32(B * U * m->n_embd);
    s->h1 = alloc_f32(B * U * m->n_ff);
    s->h2 = alloc_f32(B * U * m->n_ff);
    s->h3 = alloc_f32(B * U * m->n_embd);
    s->n_logits_max = B < TR_LOGIT_ROWS_MAX ? B : TR_LOGIT_ROWS_MAX;
    s->n_logits = 0;
    s->logits = alloc_f32(s->n_logits_max * m->vocab);
    s->score_stride = score_row_stride(actual_ctx);
    s->scores = alloc_f32(n_workers * OLMOE_ATTN_QUERIES * s->score_stride);
    if (store_partial) s->x_all = alloc_f32(actual_ctx * m->n_embd);
    int pm_rc = tr_pm_scratch_init(&s->pm, (int)n_workers, m->n_expert, B * U, pm_cols, pm_x);

    if (kv_rc != 0 || pm_rc != 0 || s->rope_cos == NULL || s->rope_sin == NULL || s->x == NULL ||
        s->normed == NULL || s->attn_out == NULL || s->ffn_out == NULL || s->q == NULL || s->attn_concat == NULL ||
        s->k == NULL || s->v == NULL || s->router == NULL || s->sel_id == NULL || s->sel_w == NULL ||
        s->place == NULL || s->offsets == NULL || s->acquire_ids == NULL || s->taken == NULL || s->xg == NULL ||
        s->h1 == NULL || s->h2 == NULL || s->h3 == NULL || s->logits == NULL || s->scores == NULL ||
        (store_partial && s->x_all == NULL)) {
        if (err != NULL) snprintf(err, err_len, "out of memory allocating session");
        olmoe_session_free(s);
        return NULL;
    }
    tr_rope_table(s->rope_cos, s->rope_sin, actual_ctx, m->head_dim, m->rope_freq_base);
    /* The GPU is a faster road to the same bits, never a requirement: a device that cannot take
     * this session (shape, VRAM) leaves it on the CPU; the command line's gpu: line says where the
     * decode ran, the reason is a debug line. */
    if (m->gpu != NULL && m->n_head_kv == m->n_head) {
        char why[256] = "";
        s->gpu = tr_gpu_attn_create(m->gpu, m->n_layers, m->n_head, m->head_dim, actual_ctx, why, sizeof why);
        if (s->gpu == NULL) tr_log(TR_LOG_DEBUG, "gpu: the attention stays on the CPU (%s)", why);
    }
    return s;
}

/* ---- forward pass --------------------------------------------------------- */
/* hot: begin */

static void clamp_inplace(float *x, int64_t n, float c) {
    for (int64_t i = 0; i < n; i++) {
        if (x[i] < -c) x[i] = -c;
        else if (x[i] > c) x[i] = c;
    }
}

/* Bytes of a matmul weight actually touched for one token: every output row,
 * each row_bytes(type, cols) long (docs/ARCHITECTURE.md §Profiling). */
static uint64_t mat_bytes(const tr_mat *w) {
    return (uint64_t)w->rows * (uint64_t)tr_row_bytes(w->type, w->cols);
}

/* One layer's attention for every token of a pass, one item per (query head, token) with the
 * head first: a chunk holds whole heads, so the tokens at the end of a pass (the longest
 * contexts) are spread over every chunk. Token i sits at position pos0 + i and attends to
 * positions 0..pos0+i, all already in the cache; an item writes only its own slice of
 * attn_concat. The keys and the values of a head are contiguous in the cache (src/kv/kv.h).
 * The tokens of a head that fall in a chunk run in groups of OLMOE_ATTN_QUERIES
 * (tr_attention_group): every token gets the bits it would get alone, and the head's keys and
 * values are read once per group instead of once per token. The rows of scores are the
 * worker's. */
typedef struct {
    const float *q;
    const tr_kv *kv;
    float *scores, *out;
    float scale;
    int64_t layer, n_qkv, head_dim, group, n_tok, pos0, score_stride;
} attn_ctx;

static void attn_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const attn_ctx *c = (const attn_ctx *)ctx_;
    float *scores = c->scores + (int64_t)worker * OLMOE_ATTN_QUERIES * c->score_stride;
    int64_t idx = begin;
    while (idx < end) {
        /* the tokens [i0, i1) of head h: what this chunk holds of the head */
        int64_t h = idx / c->n_tok, i0 = idx % c->n_tok;
        int64_t i1 = (h + 1) * c->n_tok <= end ? c->n_tok : end - h * c->n_tok;
        const float *keys = tr_kv_keys(c->kv, c->layer, h / c->group);
        const float *values = tr_kv_values(c->kv, c->layer, h / c->group);
        for (int64_t i = i0; i < i1; i += OLMOE_ATTN_QUERIES) {
            int64_t n_q = i1 - i < OLMOE_ATTN_QUERIES ? i1 - i : OLMOE_ATTN_QUERIES;
#ifdef TR_ATTN_PROBE
            if (c->n_tok == 1)
                tr_attn_probe(c->layer, h, c->q + h * c->head_dim, keys, values, c->pos0 + 1, c->head_dim, c->scale);
#endif
#ifdef TR_DRAFT_PROBE
            if (c->n_tok == 1 && tr_draft_probe_attention(c->layer, h, c->q + h * c->head_dim, keys, values, c->pos0 + 1,
                                                          c->head_dim, c->scale, scores, c->score_stride,
                                                          c->out + h * c->head_dim))
                continue;
#endif
            tr_attention_group(c->q + i * c->n_qkv + h * c->head_dim, c->n_qkv, keys, values, n_q, c->pos0 + i + 1,
                               c->head_dim, c->scale, scores, c->score_stride,
                               c->out + i * c->n_qkv + h * c->head_dim, c->n_qkv);
#ifdef TR_ATTN_PROBE
            if (c->n_tok == 1) tr_attn_probe_out(c->layer, h, c->out + h * c->head_dim, c->pos0 + 1, c->head_dim);
#endif
        }
        idx = h * c->n_tok + i1;
    }
}

/* Bytes of K and V the attention of a pass pulls from the cache, for the profiler: a group of
 * OLMOE_ATTN_QUERIES tokens reads the positions its last token sees, once per query head. A
 * decode token is a group of one, every position up to its own. (A chunk border inside a head
 * cuts one group in two: not counted.) */
static uint64_t attn_kv_bytes(int64_t n_tok, int64_t pos0, int64_t n_head, int64_t head_dim) {
    uint64_t positions = 0;
    for (int64_t i = 0; i < n_tok; i += OLMOE_ATTN_QUERIES)
        positions += (uint64_t)(pos0 + (i + OLMOE_ATTN_QUERIES < n_tok ? i + OLMOE_ATTN_QUERIES : n_tok));
    return positions * (uint64_t)(n_head * head_dim) * 2 * sizeof(float);
}

/* ---- the work of one token alone, split over the pool by token ------------------------ */
/* Each body runs the same calls on the same numbers as the loop over the tokens it replaces, so
 * the thread count changes nothing (tests/test_prefill.c, tests/test_hot.c). */

typedef struct {
    const tr_mat *w;
    const int32_t *tokens;
    float *x;
    int64_t n_embd;
} embed_ctx;

static void embed_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const embed_ctx *c = (const embed_ctx *)ctx_;
    for (int64_t i = begin; i < end; i++) tr_get_row(c->w, c->tokens[i], c->x + i * c->n_embd);
}

/* dst = rmsnorm(src) token by token; dst == src norms in place */
typedef struct {
    const float *src, *weight;
    float *dst;
    int64_t n;
    float eps;
} norm_ctx;

static void norm_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const norm_ctx *c = (const norm_ctx *)ctx_;
    if (c->dst != c->src)
        memcpy(c->dst + begin * c->n, c->src + begin * c->n, (size_t)((end - begin) * c->n) * sizeof(float));
    for (int64_t i = begin; i < end; i++) tr_rmsnorm(c->dst + i * c->n, c->weight, c->n, c->eps);
}

typedef struct {
    float *q, *k;
    const float *rope_cos, *rope_sin;
    int64_t n_qkv, n_kv, n_head, n_head_kv, head_dim, pos0;
} rope_ctx;

static void rope_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const rope_ctx *c = (const rope_ctx *)ctx_;
    int64_t half = c->head_dim / 2;
    for (int64_t i = begin; i < end; i++) {
        const float *cos_p = c->rope_cos + (c->pos0 + i) * half, *sin_p = c->rope_sin + (c->pos0 + i) * half;
        tr_rope_neox(c->q + i * c->n_qkv, c->n_head, c->head_dim, cos_p, sin_p);
        tr_rope_neox(c->k + i * c->n_kv, c->n_head_kv, c->head_dim, cos_p, sin_p);
    }
}

/* a chunk of tokens goes into the cache as a pass of its own: distinct positions, no overlap */
typedef struct {
    tr_kv *kv;
    const float *k, *v;
    int64_t layer, pos0, n_kv;
} kv_write_ctx;

static void kv_write_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const kv_write_ctx *c = (const kv_write_ctx *)ctx_;
    tr_kv_write(c->kv, c->layer, c->pos0 + begin, end - begin, c->k + begin * c->n_kv, c->v + begin * c->n_kv);
}

/* y[j] += x[j] over the elements of tokens [begin, end) */
typedef struct {
    float *y;
    const float *x;
    int64_t n;
} add_ctx;

static void add_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const add_ctx *c = (const add_ctx *)ctx_;
    for (int64_t j = begin * c->n; j < end * c->n; j++) c->y[j] += c->x[j];
}

/* Each token's expert outputs weighted and summed in increasing expert id, then added to
 * the residual stream: one item per token, the same additions in the same order whatever
 * the pass size. */
typedef struct {
    const tr_kernels *K;
    const float *h3, *sel_w;
    const int64_t *place;
    float *ffn_out, *x;
    int64_t n_embd, n_used;
} mix_ctx;

static void mix_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const mix_ctx *c = (const mix_ctx *)ctx_;
    int64_t n_embd = c->n_embd;
    for (int64_t i = begin; i < end; i++) {
        float *out = c->ffn_out + i * n_embd;
        float *xi = c->x + i * n_embd;
        for (int64_t d = 0; d < n_embd; d++) out[d] = 0.0f;
        for (int64_t slot = 0; slot < c->n_used; slot++) {
            int64_t j = i * c->n_used + slot;
            c->K->axpy_f32(out, c->h3 + c->place[j] * n_embd, c->sel_w[j], n_embd);
        }
        for (int64_t d = 0; d < n_embd; d++) xi[d] += out[d];
    }
}

/* Repeated argmax over router[0..n_expert) (already a softmax), ties to the lower id: the k
 * best experts, best first, into ids (and, if not NULL, their scores into vals). taken is
 * scratch of n_expert bytes, set here: cleared, or a copy of `off` (NULL: none), the experts
 * switched off for a measurement (tr_model_set_expert_mask), which are then never chosen.
 * Shared by route_token (which sorts its own selection
 * by id afterward) and the route trace's predictions (kept best-first, docs/MEASUREMENTS.md domande
 * 13-15: tr_session_route_trace_begin below, after the hot zone). */
static void top_experts(const float *router, int64_t n_expert, const unsigned char *off, unsigned char *taken,
                        int64_t k, int64_t *ids, float *vals) {
    for (int64_t e = 0; e < n_expert; e++) taken[e] = off != NULL ? off[e] : 0;
    for (int64_t slot = 0; slot < k; slot++) {
        int64_t best = -1;
        float best_val = 0.0f;
        for (int64_t e = 0; e < n_expert; e++) {
            if (taken[e]) continue;
            if (best == -1 || router[e] > best_val) {
                best = e;
                best_val = router[e];
            }
        }
        taken[best] = 1;
        ids[slot] = best;
        if (vals != NULL) vals[slot] = best_val;
    }
}

/* Router of token i: softmax over all experts, top-k (ties to the lower id), optional
 * renormalization, then the selection sorted by increasing expert id (the order in which
 * contributions are added). Writes n_used entries of sel_id and sel_w. */
static void route_token(const olmoe_model *m, float *router, const unsigned char *off, unsigned char *taken,
                        int64_t *sel_id, float *sel_w) {
    int64_t n_expert = m->n_expert, n_used = m->n_expert_used;
    tr_softmax(router, n_expert);
    top_experts(router, n_expert, off, taken, n_used, sel_id, sel_w);
    if (m->norm_topk) {
        float sum = 0.0f;
        for (int64_t i = 0; i < n_used; i++) sum += sel_w[i];
        for (int64_t i = 0; i < n_used; i++) sel_w[i] /= sum;
    }
#ifdef TR_DRAFT_PROBE
    tr_draft_probe_route(sel_w, n_used); /* sel_w is still in decreasing order here */
#endif
    for (int64_t i = 1; i < n_used; i++) {
        int64_t id = sel_id[i];
        float w = sel_w[i];
        int64_t j = i - 1;
        while (j >= 0 && sel_id[j] > id) {
            sel_id[j + 1] = sel_id[j];
            sel_w[j + 1] = sel_w[j];
            j--;
        }
        sel_id[j + 1] = id;
        sel_w[j + 1] = w;
    }
}

/* The router's choice of tokens [begin, end); the scratch is the worker's row of `taken`. */
typedef struct {
    const olmoe_model *m;
    float *router, *sel_w;
    const unsigned char *off; /* this layer's row of the expert mask, or NULL */
    unsigned char *taken;
    int64_t *sel_id;
} route_ctx;

static void route_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const route_ctx *c = (const route_ctx *)ctx_;
    int64_t n_expert = c->m->n_expert, n_used = c->m->n_expert_used;
    for (int64_t i = begin; i < end; i++)
        route_token(c->m, c->router + i * n_expert, c->off, c->taken + (int64_t)worker * n_expert,
                    c->sel_id + i * n_used, c->sel_w + i * n_used);
}

/* Rows [begin, end) of the (token, slot) pairs copied to their place in the grouped buffer:
 * every pair has a row of its own, so no two items write the same floats. */
typedef struct {
    const float *normed;
    const int64_t *place;
    float *xg;
    int64_t n_embd, n_used;
} gather_ctx;

static void gather_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    const gather_ctx *c = (const gather_ctx *)ctx_;
    for (int64_t j = begin; j < end; j++)
        memcpy(c->xg + c->place[j] * c->n_embd, c->normed + (j / c->n_used) * c->n_embd,
               (size_t)c->n_embd * sizeof(float));
}

/* Defined below, after the hot zone (docs/MEASUREMENTS.md domande 13-15): records layer L's routing
 * of this block's tokens into s->trace, called once per (layer, block) under one `if` in
 * forward_layer. x: this layer's hidden state for the block (s->x in pass-major, a slice of
 * s->x_all in layer-major). */
static void route_trace_record(olmoe_session *s, int64_t L, const int32_t *tokens, int64_t n_tok, const float *x);

/* Defined below, after the hot zone (docs/ARCHITECTURE.md Esperti M1): acquires layer L's
 * non-empty experts from the shared store (reading the missing ones from disk) and refreshes
 * every expert's tr_mat.data for this layer, called once per layer from forward_layer. Allocates
 * nothing: it only touches s->acquire_ids, sized at session create. -1 on a failed read. */
static int olmoe_refresh_experts(olmoe_model *m, olmoe_session *s, int64_t L);

/* Defined below, after the hot zone: waits for every unit the store is still reading ahead and
 * counts the wait and the bytes in the profile's weight_read. 0, or -1 if a read ahead failed
 * (tr_experts_prefetch_wait). Allocates nothing. */
static int olmoe_prefetch_drain(olmoe_model *m, olmoe_session *s);

/* Token embeddings of n_tok tokens into x ([n_tok][n_embd]). */
static void forward_embed(olmoe_model *m, olmoe_session *s, const int32_t *tokens, int64_t n_tok, float *x) {
    tr_prof *prof = &s->prof;
    tr_pool *pool = m->pool;
    const int64_t per_chunk = OLMOE_TOKENS_PER_CHUNK;

    uint64_t t = tr_prof_begin(prof);
    embed_ctx ec;
    ec.w = &m->token_embd;
    ec.tokens = tokens;
    ec.x = x;
    ec.n_embd = m->n_embd;
    tr_parallel_for(pool, n_tok, per_chunk, embed_body, &ec);
    tr_prof_end(prof, TR_PROF_EMBED, t);
    tr_prof_count(prof, TR_PROF_EMBED,
                  (uint64_t)n_tok * (uint64_t)tr_row_bytes(m->token_embd.type, m->token_embd.cols), 0);
}

/* One layer over n_tok tokens (1 <= n_tok <= n_batch) at positions pos0..pos0+n_tok-1, in place
 * on x ([n_tok][n_embd]: read as this layer's input, written as its output). Every value is
 * computed with the same kernel call as when the tokens run one per pass: a matmul element is
 * one dot_row, norms/RoPE/routing/mixing are per token, and a token's attention reads only cache
 * positions up to its own. So the logits and the cache are bit-identical for every split of the
 * input into passes and for every order the layers and blocks of a prompt run in (forward_pass,
 * forward_prompt_layer_major: tests/test_prefill.c). -1 if the shared expert store fails to read
 * this layer's missing units from disk: the cache and s->pos are untouched by this call
 * (olmoe_eval restores s->pos across the whole eval on failure), so a later eval of the same
 * tokens can retry. */
static int forward_layer(olmoe_model *m, olmoe_session *s, int64_t L, const int32_t *tokens, int64_t n_tok,
                         int64_t pos0, float *x) {
    tr_prof *prof = &s->prof;
    const tr_kernels *K = tr_kernels_get();
    tr_pool *pool = m->pool;
    int64_t n_embd = m->n_embd, n_qkv = m->n_qkv, n_kv = m->n_kv;
    int64_t n_head = m->n_head, n_head_kv = m->n_head_kv, head_dim = m->head_dim;
    int64_t n_ff = m->n_ff, n_expert = m->n_expert, n_used = m->n_expert_used;
    int64_t n_rows = n_tok * n_used; /* (token, slot) pairs = rows of the grouped expert buffers */
    float scale = 1.0f / sqrtf((float)head_dim);

    /* the jobs of one token alone (see the bodies above): the context changes zone by zone */
    const int64_t per_chunk = OLMOE_TOKENS_PER_CHUNK;
    norm_ctx nc;
    nc.eps = m->rms_eps;
    uint64_t t;

    olmoe_layer *layer = &m->layers[L];

    /* ---- attention ---- */
    t = tr_prof_begin(prof);
    nc.src = x;
    nc.dst = s->normed;
    nc.weight = layer->attn_norm;
    nc.n = n_embd;
    tr_parallel_for(pool, n_tok, per_chunk, norm_body, &nc);
    tr_prof_end(prof, TR_PROF_ATTN_NORM, t);

    t = tr_prof_begin(prof);
    tr_matmul_s(pool, &layer->wq, s->normed, n_tok, s->q, &s->pm);
    tr_matmul_s(pool, &layer->wk, s->normed, n_tok, s->k, &s->pm);
    tr_matmul_s(pool, &layer->wv, s->normed, n_tok, s->v, &s->pm);
    tr_prof_end(prof, TR_PROF_QKV_PROJ, t);
    tr_prof_count(prof, TR_PROF_QKV_PROJ, mat_bytes(&layer->wq) + mat_bytes(&layer->wk) + mat_bytes(&layer->wv),
                  0);

    t = tr_prof_begin(prof);
    nc.src = nc.dst = s->q;
    nc.weight = layer->q_norm;
    nc.n = n_qkv;
    tr_parallel_for(pool, n_tok, per_chunk, norm_body, &nc);
    nc.src = nc.dst = s->k;
    nc.weight = layer->k_norm;
    nc.n = n_kv;
    tr_parallel_for(pool, n_tok, per_chunk, norm_body, &nc);
    tr_prof_end(prof, TR_PROF_QK_NORM, t);

    if (m->have_clamp) {
        clamp_inplace(s->q, n_tok * n_qkv, m->clamp);
        clamp_inplace(s->k, n_tok * n_kv, m->clamp);
        clamp_inplace(s->v, n_tok * n_kv, m->clamp);
    }

    t = tr_prof_begin(prof);
    rope_ctx rc;
    rc.q = s->q;
    rc.k = s->k;
    rc.rope_cos = s->rope_cos;
    rc.rope_sin = s->rope_sin;
    rc.n_qkv = n_qkv;
    rc.n_kv = n_kv;
    rc.n_head = n_head;
    rc.n_head_kv = n_head_kv;
    rc.head_dim = head_dim;
    rc.pos0 = pos0;
    tr_parallel_for(pool, n_tok, per_chunk, rope_body, &rc);
    tr_prof_end(prof, TR_PROF_ROPE, t);

    t = tr_prof_begin(prof);
    kv_write_ctx wc;
    wc.kv = &s->kv;
    wc.k = s->k;
    wc.v = s->v;
    wc.layer = L;
    wc.pos0 = pos0;
    wc.n_kv = n_kv;
    tr_parallel_for(pool, n_tok, per_chunk, kv_write_body, &wc);
    /* the VRAM cache gets the same rows; a decode token's own row goes with its attention call */
    if (s->gpu != NULL && !s->gpu_failed && n_tok > 1 &&
        (tr_gpu_attn_write(s->gpu, L, pos0, n_tok, s->k, s->v) != 0 ||
         (s->gpu_warm && tr_gpu_attn_warm(s->gpu, TR_GPU_WARM_MAX_NS) != 0)))
        s->gpu_failed = 1;
    tr_prof_end(prof, TR_PROF_KV_WRITE, t);

    t = tr_prof_begin(prof);
    /* one decode token: on the GPU when the session has one, the same bits (gpu_attn.h) */
    int on_gpu = s->gpu != NULL && !s->gpu_failed && n_tok == 1;
    if (on_gpu && tr_gpu_attn_decode(s->gpu, L, pos0, s->q, s->k, s->v, scale, s->attn_concat) != 0) {
        s->gpu_failed = 1;
        on_gpu = 0;
    }
    if (!on_gpu) {
        attn_ctx ac;
        ac.q = s->q;
        ac.kv = &s->kv;
        ac.scores = s->scores;
        ac.out = s->attn_concat;
        ac.scale = scale;
        ac.layer = L;
        ac.n_qkv = n_qkv;
        ac.head_dim = head_dim;
        ac.group = n_head / n_head_kv;
        ac.n_tok = n_tok;
        ac.pos0 = pos0;
        ac.score_stride = s->score_stride;
        /* an item costs ~30 ns per cached position: short contexts keep several per chunk; a
         * decode token's heads, one stream of cached positions each, balanced at the tail */
        TR_TRACE_NOTE(attn_kv_bytes(n_tok, pos0, n_head, head_dim), -1, pos0 + n_tok);
        if (n_tok == 1) tr_parallel_for_balanced(pool, n_head, 1 + 256 / (pos0 + 1), attn_body, &ac);
        else tr_parallel_for(pool, n_head * n_tok, 1 + 256 / (pos0 + n_tok), attn_body, &ac);
    } else if (L == m->n_layers - 1) {
        s->gpu_tokens++;
    }
    tr_prof_end(prof, TR_PROF_ATTENTION, t);
    if (prof->enabled) tr_prof_count_kv(prof, TR_PROF_ATTENTION, attn_kv_bytes(n_tok, pos0, n_head, head_dim));

    t = tr_prof_begin(prof);
    tr_matmul_s(pool, &layer->wo, s->attn_concat, n_tok, s->attn_out, &s->pm);
    add_ctx dc;
    dc.y = x;
    dc.x = s->attn_out;
    dc.n = n_embd;
    tr_parallel_for(pool, n_tok, per_chunk, add_body, &dc);
    tr_prof_end(prof, TR_PROF_ATTN_OUT_PROJ, t);
    tr_prof_count(prof, TR_PROF_ATTN_OUT_PROJ, mat_bytes(&layer->wo), 0);

    /* ---- MoE FFN ---- */
    t = tr_prof_begin(prof);
    nc.src = x;
    nc.dst = s->normed;
    nc.weight = layer->ffn_norm;
    nc.n = n_embd;
    tr_parallel_for(pool, n_tok, per_chunk, norm_body, &nc);
    tr_prof_end(prof, TR_PROF_FFN_NORM, t);

    /* route every token, then a counting sort of the (token, slot) pairs by expert:
     * offsets[e] counts, then starts, place[] takes each pair's row, and the shift
     * at the end turns the advanced starts back into starts */
    t = tr_prof_begin(prof);
    tr_matmul(pool, &layer->gate_inp, s->normed, n_tok, s->router);
    route_ctx oc;
    oc.m = m;
    oc.router = s->router;
    oc.off = m->expert_off != NULL ? m->expert_off + L * n_expert : NULL;
    oc.sel_w = s->sel_w;
    oc.taken = s->taken;
    oc.sel_id = s->sel_id;
    tr_parallel_for(pool, n_tok, per_chunk, route_body, &oc);
    for (int64_t e = 0; e <= n_expert; e++) s->offsets[e] = 0;
    for (int64_t j = 0; j < n_rows; j++) s->offsets[s->sel_id[j] + 1]++;
    for (int64_t e = 0; e < n_expert; e++) s->offsets[e + 1] += s->offsets[e];
    for (int64_t j = 0; j < n_rows; j++) s->place[j] = s->offsets[s->sel_id[j]]++;
    for (int64_t e = n_expert; e > 0; e--) s->offsets[e] = s->offsets[e - 1];
    s->offsets[0] = 0;
    tr_prof_end(prof, TR_PROF_ROUTER, t);
    tr_prof_count(prof, TR_PROF_ROUTER, mat_bytes(&layer->gate_inp), 0);

    /* the shared store's turn (Esperti M1): acquire this layer's non-empty experts (missing
     * ones read from disk here) and refresh every expert's data pointer, NULL for the ones
     * not in RAM -- a wrong read then crashes instead of reading stale bytes. */
    if (olmoe_refresh_experts(m, s, L) != 0) return -1;

    /* experts in stages, each one job over every (token, slot) row grouped by expert */
    t = tr_prof_begin(prof);
    gather_ctx gc;
    gc.normed = s->normed;
    gc.place = s->place;
    gc.xg = s->xg;
    gc.n_embd = n_embd;
    gc.n_used = n_used;
    tr_parallel_for(pool, n_rows, per_chunk * n_used, gather_body, &gc);
    tr_prof_end(prof, TR_PROF_EXPERT_GATHER, t);

    t = tr_prof_begin(prof);
    tr_matmul_grouped_s(pool, layer->gate_exps, s->offsets, n_expert, s->xg, s->h1, &s->pm);
    tr_matmul_grouped_s(pool, layer->up_exps, s->offsets, n_expert, s->xg, s->h2, &s->pm);
    tr_prof_end(prof, TR_PROF_EXPERT_GATE_UP, t);

    t = tr_prof_begin(prof);
    tr_swiglu(pool, s->h1, s->h2, n_rows * n_ff);
    tr_prof_end(prof, TR_PROF_EXPERT_ACT, t);

    t = tr_prof_begin(prof);
    tr_matmul_grouped_s(pool, layer->down_exps, s->offsets, n_expert, s->h1, s->h3, &s->pm);
    tr_prof_end(prof, TR_PROF_EXPERT_DOWN, t);
    if (prof->enabled) {
        for (int64_t e = 0; e < n_expert; e++)
            if (s->offsets[e + 1] > s->offsets[e]) {
                tr_prof_count(prof, TR_PROF_EXPERT_GATE_UP,
                              mat_bytes(&layer->gate_exps[e]) + mat_bytes(&layer->up_exps[e]), 0);
                tr_prof_count(prof, TR_PROF_EXPERT_DOWN, mat_bytes(&layer->down_exps[e]), 0);
            }
    }

    t = tr_prof_begin(prof);
    mix_ctx mc;
    mc.K = K;
    mc.h3 = s->h3;
    mc.sel_w = s->sel_w;
    mc.place = s->place;
    mc.ffn_out = s->ffn_out;
    mc.x = x;
    mc.n_embd = n_embd;
    mc.n_used = n_used;
    tr_parallel_for(pool, n_tok, per_chunk, mix_body, &mc);
    tr_prof_end(prof, TR_PROF_EXPERT_MIX, t);

    if (s->trace != NULL) route_trace_record(s, L, tokens, n_tok, x);
    return 0;
}

/* Logits of the last n_logits rows of x ([n_tok][n_embd]; 0: none). One row per kept position:
 * every row is its own dot_row over the same weights, so a row is bit-identical to the logits
 * that token gives alone (tests/test_spec.c). */
static void forward_logits(olmoe_model *m, olmoe_session *s, float *x, int64_t n_tok, int64_t n_logits) {
    if (n_logits <= 0) return;
    tr_prof *prof = &s->prof;
    tr_pool *pool = m->pool;
    int64_t n_embd = m->n_embd;
    float *rows = x + (n_tok - n_logits) * n_embd;

    uint64_t t = tr_prof_begin(prof);
    for (int64_t i = 0; i < n_logits; i++) tr_rmsnorm(rows + i * n_embd, m->output_norm, n_embd, m->rms_eps);
    tr_prof_end(prof, TR_PROF_OUTPUT_NORM, t);

    t = tr_prof_begin(prof);
    tr_matmul_s(pool, &m->output, rows, n_logits, s->logits, &s->pm);
    tr_prof_end(prof, TR_PROF_LM_HEAD, t);
    tr_prof_count(prof, TR_PROF_LM_HEAD, mat_bytes(&m->output), 0);
    s->n_logits = n_logits;
}

/* One forward pass over n_tok tokens (1 <= n_tok <= n_batch): embed, every layer in order, then
 * logits of the last n_logits (0: none). -1 on a failed layer (see forward_layer). */
static int forward_pass(olmoe_model *m, olmoe_session *s, const int32_t *tokens, int64_t n_tok, int64_t n_logits) {
    tr_prof *prof = &s->prof;
    uint64_t t_pass = tr_prof_begin(prof);
    const int64_t pos0 = s->pos;

    forward_embed(m, s, tokens, n_tok, s->x);
    for (int64_t L = 0; L < m->n_layers; L++) {
        if (forward_layer(m, s, L, tokens, n_tok, pos0, s->x) != 0) return -1;
    }
    forward_logits(m, s, s->x, n_tok, n_logits);

    tr_prof_end(prof, TR_PROF_TOKEN, t_pass);
    if (prof->enabled) prof->tokens[prof->phase] += (uint64_t)n_tok;
    return 0;
}

/* The whole prompt (n > n_batch), layer by layer instead of pass by pass (docs/MEASUREMENTS.md "Il
 * prefill legge il modello una volta per passata"): every block is embedded into its own slice
 * of s->x_all, then for each layer every block runs forward_layer in turn before the next layer
 * starts, so olmoe_refresh_experts sees every block's need for that layer before moving on --
 * under a partial expert store this reads each layer's units once for the whole prompt instead
 * of once per block. Same kernel calls, same arguments as forward_pass on the same tokens (only
 * the order blocks and layers run in changes), so the logits and the cache are bit-identical to
 * splitting the same prompt into passes of n_batch (tests/test_prefill.c). Only reachable when
 * s->x_all != NULL and n > s->n_batch (olmoe_eval); n_logits > 1 is never asked of this path
 * (olmoe_eval already requires n <= s->n_batch for that, so this function is always n_logits <=
 * 1 in practice, but takes any n_logits <= n like forward_pass). */
static int forward_prompt_layer_major(olmoe_model *m, olmoe_session *s, const int32_t *tokens, int64_t n,
                                      int64_t n_logits) {
    tr_prof *prof = &s->prof;
    uint64_t t_pass = tr_prof_begin(prof);
    const int64_t pos_base = s->pos;
    int64_t n_embd = m->n_embd;

    for (int64_t off = 0; off < n; off += s->n_batch) {
        int64_t len = n - off < s->n_batch ? n - off : s->n_batch;
        forward_embed(m, s, tokens + off, len, s->x_all + off * n_embd);
        if (prof->enabled) prof->tokens[prof->phase] += (uint64_t)len; /* once per block, not per layer */
    }

    /* the first block of each layer but the last, once it has its own units, may start reading
     * the next layer's (olmoe_refresh_experts); whatever is still in flight at the end, or when a
     * layer fails, is waited for before returning, so nothing is read behind the caller's back */
    for (int64_t L = 0; L < m->n_layers; L++) {
        for (int64_t off = 0; off < n; off += s->n_batch) {
            int64_t len = n - off < s->n_batch ? n - off : s->n_batch;
            s->prefetch_layer = off == 0 && L + 1 < m->n_layers ? L + 1 : -1;
            if (forward_layer(m, s, L, tokens + off, len, pos_base + off, s->x_all + off * n_embd) != 0) {
                s->prefetch_layer = -1;
                olmoe_prefetch_drain(m, s);
                return -1;
            }
        }
    }
    s->prefetch_layer = -1;
    if (olmoe_prefetch_drain(m, s) != 0) return -1;

    forward_logits(m, s, s->x_all, n, n_logits);

    tr_prof_end(prof, TR_PROF_TOKEN, t_pass);
    return 0;
}

/* Passes of at most n_batch tokens; only the last one computes logits. More than one row of
 * logits needs them all in that pass, so the whole call must fit one pass. Under a partial
 * expert store and a prompt longer than one pass, the whole prompt runs layer by layer instead
 * (forward_prompt_layer_major) so every layer's experts are read once for the prompt, not once
 * per pass; s->x_all being non-NULL is what makes that path reachable at all (only allocated at
 * session create when the store cannot hold every unit). A failed call restores s->pos to what
 * it was when this function was called: the cache past that position is never read, so the
 * session is exactly as it was before this call, and the same tokens can be evaluated again. */
static int olmoe_eval(void *session, const int32_t *tokens, int64_t n, int64_t n_logits) {
    olmoe_session *s = (olmoe_session *)session;
    olmoe_model *m = s->m;

    if (n <= 0) return -1;
    if (n_logits < 1 || n_logits > n || n_logits > s->n_logits_max) return -1;
    if (n_logits > 1 && n > s->n_batch) return -1;
    if (s->pos + n > s->n_ctx) return -1;
    for (int64_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= m->vocab) return -1;
    }

    int64_t pos0 = s->pos;
    /* The route trace numbers its rows from tr->pub.n_tokens, which only advances once a token has
     * been through its LAST layer: in layer-major order every block of a layer would write over the
     * same rows. A trace is measurement only (--route-trace), so it takes the pass-major path and
     * keeps its rows right, instead of the trace growing a second way of counting. */
    if (s->x_all != NULL && n > s->n_batch && s->trace == NULL) {
        s->gpu_warm = 0; /* a prompt read from disk is slow: the GPU is not kept awake through it */
        if (forward_prompt_layer_major(m, s, tokens, n, n_logits) != 0) {
            s->pos = pos0;
            return -1;
        }
        s->pos += n;
        return 0;
    }

    for (int64_t off = 0; off < n; off += s->n_batch) {
        int64_t len = n - off < s->n_batch ? n - off : s->n_batch;
        /* the last pass of a prompt or of a speculative check wakes the GPU for the decode after it */
        s->gpu_warm = off + len == n;
        if (forward_pass(m, s, tokens + off, len, off + len == n ? n_logits : 0) != 0) {
            s->pos = pos0;
            return -1;
        }
        s->pos += len;
    }
    return 0;
}
/* hot: end */

/* Outside the hot zone (docs/ARCHITECTURE.md Esperti M1): allocates nothing, using only
 * s->acquire_ids (sized at session create). Called once per layer from forward_pass. */
static int olmoe_refresh_experts(olmoe_model *m, olmoe_session *s, int64_t L) {
    olmoe_layer *layer = &m->layers[L];
    int64_t n_expert = m->n_expert;

    int64_t n_ids = 0;
    for (int64_t e = 0; e < n_expert; e++)
        if (s->offsets[e + 1] > s->offsets[e]) s->acquire_ids[n_ids++] = e;

    tr_experts_stats before, after;
    tr_experts_get_stats(m->experts, &before);
    uint64_t t = tr_prof_begin(&s->prof);
    int rc = tr_experts_acquire(m->experts, L, s->acquire_ids, n_ids);
    tr_prof_end(&s->prof, TR_PROF_WEIGHT_READ, t);
    tr_experts_get_stats(m->experts, &after);
    uint64_t read_bytes = after.bytes_read - before.bytes_read;
    tr_prof_count(&s->prof, TR_PROF_WEIGHT_READ, read_bytes, read_bytes);

    /* this layer's units in hand: read the next layer's ahead while this one computes, if this
     * block asked for nearly every expert (OLMOE_PREFETCH_SPARE_DIV) */
    if (rc == 0 && s->prefetch_layer >= 0 && n_ids >= n_expert - n_expert / OLMOE_PREFETCH_SPARE_DIV)
        tr_experts_prefetch(m->experts, s->prefetch_layer, L);
    s->prefetch_layer = -1;

    /* every expert of the layer, present or not: a stale pointer from a past layer's turn must
     * never survive, so a wrong read crashes instead of reading someone else's bytes. */
    for (int64_t e = 0; e < n_expert; e++) {
        layer->exps[e].data = tr_experts_part(m->experts, L, e, 0);
        layer->exps[n_expert + e].data = tr_experts_part(m->experts, L, e, 1);
        layer->exps[2 * n_expert + e].data = tr_experts_part(m->experts, L, e, 2);
    }
    return rc;
}

static int olmoe_prefetch_drain(olmoe_model *m, olmoe_session *s) {
    if (!tr_experts_prefetching(m->experts)) return 0;
    tr_experts_stats before, after;
    tr_experts_get_stats(m->experts, &before);
    uint64_t t = tr_prof_begin(&s->prof);
    int rc = tr_experts_prefetch_wait(m->experts);
    tr_prof_end(&s->prof, TR_PROF_WEIGHT_READ, t);
    tr_experts_get_stats(m->experts, &after);
    uint64_t read_bytes = after.bytes_read - before.bytes_read;
    tr_prof_count(&s->prof, TR_PROF_WEIGHT_READ, read_bytes, read_bytes);
    return rc;
}

static const float *olmoe_logits(const void *session, int64_t back) {
    const olmoe_session *s = (const olmoe_session *)session;
    if (back < 0 || back >= s->n_logits) return NULL;
    return s->logits + (s->n_logits - 1 - back) * s->m->vocab;
}

static int64_t olmoe_pos(const void *session) {
    const olmoe_session *s = (const olmoe_session *)session;
    return s->pos;
}

static int64_t olmoe_n_ctx(const void *session) {
    const olmoe_session *s = (const olmoe_session *)session;
    return s->n_ctx;
}

static int64_t olmoe_max_logit_rows(const void *session) {
    const olmoe_session *s = (const olmoe_session *)session;
    return s->n_logits_max;
}

/* The cache rows past n are simply overwritten by the next eval: attention only ever
 * reads positions below the current one. */
static int olmoe_rewind(void *session, int64_t n) {
    olmoe_session *s = (olmoe_session *)session;
    if (n < 0 || n > s->pos) return -1;
    s->pos = n;
    s->n_logits = 0; /* the kept rows describe positions that may no longer be in the cache */
    return 0;
}

static tr_prof *olmoe_prof(void *session) {
    olmoe_session *s = (olmoe_session *)session;
    return &s->prof;
}

/* ---- routing trace (docs/MEASUREMENTS.md domande 13-15) -------------------------------------------
 * Outside the hot zone: route_trace_record (called from inside it, under one `if`) allocates
 * nothing and only touches the scratch begun here. */

struct olmoe_route_trace {
    tr_route_trace pub;                    /* returned as-is by olmoe_route_trace */
    uint16_t *chosen, *pred_in, *pred_out; /* owned; pub.chosen/pred_in/pred_out alias these */
    int32_t *tokens;                       /* owned; pub.tokens */
    float *margins;                        /* owned; pub.margins */
    float *router;                         /* [B][n_expert] scratch: a prediction's raw scores */
    float *normed;                         /* [B][n_embd] scratch: next layer's ffn_norm over this layer's x */
    unsigned char *taken;                  /* [n_expert] scratch for top_experts, one row (serial) */
};

static void olmoe_route_trace_free(olmoe_route_trace *tr) {
    if (tr == NULL) return;
    tr_free_aligned(tr->chosen);
    tr_free_aligned(tr->pred_in);
    tr_free_aligned(tr->pred_out);
    tr_free_aligned(tr->tokens);
    tr_free_aligned(tr->margins);
    tr_free_aligned(tr->router);
    tr_free_aligned(tr->normed);
    tr_free_aligned(tr->taken);
    free(tr);
}

/* One expert's gate + up + down rows, and every tensor of one layer, bytes as stored
 * (mat_bytes above): every layer has the same shapes, so layer 0 stands for all. */
static void route_trace_sizes(const olmoe_model *m, int64_t *expert_bytes, int64_t *layer_bytes) {
    const olmoe_layer *l0 = &m->layers[0];
    int64_t eb = (int64_t)mat_bytes(&l0->gate_exps[0]) + (int64_t)mat_bytes(&l0->up_exps[0]) +
                 (int64_t)mat_bytes(&l0->down_exps[0]);
    int64_t lb = (int64_t)mat_bytes(&l0->wq) + (int64_t)mat_bytes(&l0->wk) + (int64_t)mat_bytes(&l0->wv) +
                 (int64_t)mat_bytes(&l0->wo) + (int64_t)mat_bytes(&l0->gate_inp) + m->n_expert * eb +
                 (m->n_embd + m->n_embd + m->n_qkv + m->n_kv) * (int64_t)sizeof(float);
    *expert_bytes = eb;
    *layer_bytes = lb;
}

static int olmoe_route_trace_begin(void *session, int64_t max_tokens) {
    olmoe_session *s = (olmoe_session *)session;
    olmoe_model *m = s->m;
    if (m->n_expert > 65535) return -1;
    if (max_tokens < 0) max_tokens = 0;

    olmoe_route_trace *tr = (olmoe_route_trace *)calloc(1, sizeof *tr);
    if (tr == NULL) return -1;

    int64_t n_expert = m->n_expert, n_used = m->n_expert_used, n_layers = m->n_layers;
    int64_t n_pred = n_expert < TR_ROUTE_TRACE_PRED ? n_expert : TR_ROUTE_TRACE_PRED;
    int64_t B = s->n_batch;

    if (max_tokens > 0) {
        tr->chosen = (uint16_t *)tr_alloc_aligned((size_t)(max_tokens * n_layers * n_used) * sizeof(uint16_t), 64);
        tr->pred_in = (uint16_t *)tr_alloc_aligned((size_t)(max_tokens * n_layers * n_pred) * sizeof(uint16_t), 64);
        tr->pred_out = (uint16_t *)tr_alloc_aligned((size_t)(max_tokens * n_layers * n_pred) * sizeof(uint16_t), 64);
        tr->tokens = (int32_t *)tr_alloc_aligned((size_t)max_tokens * sizeof(int32_t), 64);
        tr->margins = alloc_f32(max_tokens * n_layers * 2);
    }
    tr->router = alloc_f32(B * n_expert);
    tr->normed = alloc_f32(B * m->n_embd);
    tr->taken = (unsigned char *)tr_alloc_aligned((size_t)n_expert, 64);

    if ((max_tokens > 0 && (tr->chosen == NULL || tr->pred_in == NULL || tr->pred_out == NULL ||
                            tr->tokens == NULL || tr->margins == NULL)) ||
        tr->router == NULL || tr->normed == NULL || tr->taken == NULL) {
        olmoe_route_trace_free(tr);
        return -1;
    }

    tr->pub.n_tokens = 0;
    tr->pub.max_tokens = max_tokens;
    tr->pub.n_layers = n_layers;
    tr->pub.n_expert = n_expert;
    tr->pub.n_used = n_used;
    tr->pub.n_pred = n_pred;
    route_trace_sizes(m, &tr->pub.expert_bytes, &tr->pub.layer_bytes);
    tr->pub.chosen = tr->chosen;
    tr->pub.pred_in = tr->pred_in;
    tr->pub.pred_out = tr->pred_out;
    tr->pub.tokens = tr->tokens;
    tr->pub.margins = tr->margins;

    olmoe_route_trace_free(s->trace);
    s->trace = tr;
    return 0;
}

static const tr_route_trace *olmoe_route_trace_get(const void *session) {
    const olmoe_session *s = (const olmoe_session *)session;
    return s->trace != NULL ? &s->trace->pub : NULL;
}

/* Records layer L's routing into the trace for the tokens of this pass that still fit under
 * max_tokens (evaluation order; recording then stops, but the pass still runs to completion).
 * Called once per layer from forward_pass, right after that layer's own residual add, while
 * s->normed still holds this layer's own FFN input (ffn_norm_L(x), untouched since this layer's
 * own router read it: the gather that follows only copies from it) and s->x holds this layer's
 * own output (x after the residual). Uses only the scratch tr_session_route_trace_begin
 * allocated: nothing here allocates. */
static void route_trace_record(olmoe_session *s, int64_t L, const int32_t *tokens, int64_t n_tok, const float *x) {
    olmoe_route_trace *tr = s->trace;
    const olmoe_model *m = s->m;
    int64_t n_expert = m->n_expert, n_used = m->n_expert_used, n_pred = tr->pub.n_pred;
    int64_t n_layers = m->n_layers, n_embd = m->n_embd;

    int64_t rec = tr->pub.max_tokens - tr->pub.n_tokens;
    if (rec > n_tok) rec = n_tok;
    if (rec < 0) rec = 0;
    int64_t base = tr->pub.n_tokens;

    /* s->router still holds this layer's softmax (route_token ran it in place and nothing has
     * written there since): the margin is the probability of the last expert chosen and of the
     * best one left out, among the experts the mask, if any, leaves on. */
    const unsigned char *off = m->expert_off != NULL ? m->expert_off + L * n_expert : NULL;
    for (int64_t i = 0; i < rec; i++) {
        uint16_t *row = tr->chosen + (base + i) * n_layers * n_used + L * n_used;
        const float *prob = s->router + i * n_expert;
        float p_last = 1.0f, p_next = 0.0f;
        for (int64_t e = 0; e < n_expert; e++) tr->taken[e] = off != NULL ? off[e] : 0;
        for (int64_t slot = 0; slot < n_used; slot++) {
            int64_t id = s->sel_id[i * n_used + slot];
            row[slot] = (uint16_t)id;
            tr->taken[id] = 1;
            if (prob[id] < p_last) p_last = prob[id];
        }
        for (int64_t e = 0; e < n_expert; e++)
            if (!tr->taken[e] && prob[e] > p_next) p_next = prob[e];
        tr->margins[((base + i) * n_layers + L) * 2] = p_last;
        tr->margins[((base + i) * n_layers + L) * 2 + 1] = p_next;
        if (L == 0) tr->tokens[base + i] = tokens[i];
    }

    if (rec > 0 && L + 1 < n_layers) {
        const olmoe_layer *next = &m->layers[L + 1];
        int64_t ids[TR_ROUTE_TRACE_PRED];

        /* pred_in: the next layer's router applied to this layer's own FFN input -- known
         * before this layer's experts run. */
        tr_matmul(m->pool, &next->gate_inp, s->normed, rec, tr->router);
        for (int64_t i = 0; i < rec; i++) {
            float *scores = tr->router + i * n_expert;
            tr_softmax(scores, n_expert);
            top_experts(scores, n_expert, NULL, tr->taken, n_pred, ids, NULL);
            uint16_t *out = tr->pred_in + (base + i) * n_layers * n_pred + L * n_pred;
            for (int64_t k = 0; k < n_pred; k++) out[k] = (uint16_t)ids[k];
        }

        /* pred_out: the next layer's own ffn_norm and router applied to this layer's own
         * output -- known only after this layer's experts ran and added their residual. */
        memcpy(tr->normed, x, (size_t)rec * n_embd * sizeof(float));
        for (int64_t i = 0; i < rec; i++) tr_rmsnorm(tr->normed + i * n_embd, next->ffn_norm, n_embd, m->rms_eps);
        tr_matmul(m->pool, &next->gate_inp, tr->normed, rec, tr->router);
        for (int64_t i = 0; i < rec; i++) {
            float *scores = tr->router + i * n_expert;
            tr_softmax(scores, n_expert);
            top_experts(scores, n_expert, NULL, tr->taken, n_pred, ids, NULL);
            uint16_t *out = tr->pred_out + (base + i) * n_layers * n_pred + L * n_pred;
            for (int64_t k = 0; k < n_pred; k++) out[k] = (uint16_t)ids[k];
        }
    } else if (rec > 0) {
        for (int64_t i = 0; i < rec; i++) {
            uint16_t *pin = tr->pred_in + (base + i) * n_layers * n_pred + L * n_pred;
            uint16_t *pout = tr->pred_out + (base + i) * n_layers * n_pred + L * n_pred;
            for (int64_t k = 0; k < n_pred; k++) { pin[k] = 0xFFFF; pout[k] = 0xFFFF; }
        }
    }

    if (L == n_layers - 1) tr->pub.n_tokens += rec;
}

const tr_arch_vtable tr_olmoe_vtable = {
    "olmoe",
    olmoe_load,
    olmoe_free,
    olmoe_info,
    olmoe_expert_stats,
    olmoe_experts,
    olmoe_session_create,
    olmoe_session_free,
    olmoe_eval,
    olmoe_logits,
    olmoe_pos,
    olmoe_n_ctx,
    olmoe_max_logit_rows,
    olmoe_rewind,
    olmoe_prof,
    olmoe_route_trace_begin,
    olmoe_route_trace_get,
    olmoe_set_expert_mask,
    olmoe_set_gpu,
    olmoe_gpu_tokens,
};
