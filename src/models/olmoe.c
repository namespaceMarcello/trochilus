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

    /* every tr_alloc_aligned buffer owned by this model, freed on destroy */
    void **owned;
    size_t n_owned, cap_owned;
} olmoe_model;

/* Tokens in one forward pass when the caller does not choose (llama.cpp's n_ubatch). */
#define OLMOE_DEFAULT_BATCH 512
/* Consecutive tokens of a head whose attention runs as one group (tr_attention_group): the keys
 * and the values come from memory once per group. Measured at 4, 16 and 64 on a prompt of 4000:
 * the same speed (docs/MISURE.md "Prefill su prompt lunghi"); 16 rows of scores stay in L2. */
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
} olmoe_session;

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
static float *read_vec(tr_gguf *g, olmoe_model *m, const char *name, int64_t len, char *err, size_t err_len) {
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
static int read_mat(tr_gguf *g, olmoe_model *m, const char *name, int64_t cols, int64_t rows, tr_mat *out,
                     char *err, size_t err_len) {
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

/* Loads a 3-D stacked expert weight [ne0=in, ne1=out, ne2=n_expert], kept raw. */
static const void *read_experts(tr_gguf *g, olmoe_model *m, const char *name, int64_t ne0, int64_t ne1,
                                 int64_t ne2, tr_type *type_out, char *err, size_t err_len) {
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
    if (track(m, raw) != 0) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    *type_out = t->type;
    return raw;
}

/* One tr_mat per expert over a stacked tensor: exps[e] = rows [e*out_dim, (e+1)*out_dim). */
static void expert_views(tr_mat *exps, const void *base, tr_type type, int64_t in_dim, int64_t out_dim,
                         int64_t n_expert) {
    size_t rb = tr_row_bytes(type, in_dim);
    for (int64_t e = 0; e < n_expert; e++) {
        exps[e].type = type;
        exps[e].rows = out_dim;
        exps[e].cols = in_dim;
        exps[e].data = (const unsigned char *)base + (size_t)e * (size_t)out_dim * rb;
    }
}

/* ---- vtable: free ------------------------------------------------------- */

static void olmoe_free(void *model) {
    olmoe_model *m = (olmoe_model *)model;
    if (m == NULL) return;
    for (size_t i = 0; i < m->n_owned; i++) tr_free_aligned(m->owned[i]);
    free(m->owned);
    if (m->layers != NULL)
        for (int64_t L = 0; L < m->n_layers; L++) free(m->layers[L].exps);
    free(m->layers);
    free(m);
}

/* ---- vtable: load -------------------------------------------------------- */

static void *olmoe_load(tr_gguf *g, tr_pool *pool, char *err, size_t err_len) {
    tr_kernels_init();

    olmoe_model *m = (olmoe_model *)calloc(1, sizeof *m);
    if (m == NULL) {
        snprintf(err, err_len, "out of memory");
        tr_gguf_close(g);
        return NULL;
    }
    m->pool = pool;

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

    /* Safety of the machine: estimate before loading any weight. Every tensor
     * in an OLMoE GGUF file is a model weight, so the sum over the whole
     * directory is the (exact, not just estimated) total to be loaded. */
    {
        uint64_t total_bytes = 0;
        uint64_t tcount = tr_gguf_tensor_count(g);
        for (uint64_t i = 0; i < tcount; i++) total_bytes += tr_gguf_tensor_at(g, i)->n_bytes;
        if (tr_mem_guard(total_bytes, err, err_len) != 0) goto fail;
        m->info.weight_bytes = total_bytes;
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
        if (track(m, raw) != 0) {
            snprintf(err, err_len, "out of memory");
            goto fail;
        }
        m->token_embd.type = te->type;
        m->token_embd.rows = m->vocab;
        m->token_embd.cols = m->n_embd;
        m->token_embd.data = raw;
    }

    m->output_norm = read_vec(g, m, "output_norm.weight", m->n_embd, err, err_len);
    if (m->output_norm == NULL) goto fail;

    if (read_mat(g, m, "output.weight", m->n_embd, m->vocab, &m->output, err, err_len) != 0) goto fail;

    m->layers = (olmoe_layer *)calloc((size_t)m->n_layers, sizeof(olmoe_layer));
    if (m->layers == NULL) {
        snprintf(err, err_len, "out of memory");
        goto fail;
    }

    for (int64_t L = 0; L < m->n_layers; L++) {
        olmoe_layer *layer = &m->layers[L];
        char name[64];

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_norm.weight", L);
        layer->attn_norm = read_vec(g, m, name, m->n_embd, err, err_len);
        if (layer->attn_norm == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_q.weight", L);
        if (read_mat(g, m, name, m->n_embd, m->n_qkv, &layer->wq, err, err_len) != 0) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_k.weight", L);
        if (read_mat(g, m, name, m->n_embd, m->n_kv, &layer->wk, err, err_len) != 0) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_v.weight", L);
        if (read_mat(g, m, name, m->n_embd, m->n_kv, &layer->wv, err, err_len) != 0) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_output.weight", L);
        if (read_mat(g, m, name, m->n_qkv, m->n_embd, &layer->wo, err, err_len) != 0) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_q_norm.weight", L);
        layer->q_norm = read_vec(g, m, name, m->n_qkv, err, err_len);
        if (layer->q_norm == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".attn_k_norm.weight", L);
        layer->k_norm = read_vec(g, m, name, m->n_kv, err, err_len);
        if (layer->k_norm == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_norm.weight", L);
        layer->ffn_norm = read_vec(g, m, name, m->n_embd, err, err_len);
        if (layer->ffn_norm == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_gate_inp.weight", L);
        if (read_mat(g, m, name, m->n_embd, m->n_expert, &layer->gate_inp, err, err_len) != 0) goto fail;

        layer->exps = (tr_mat *)calloc((size_t)m->n_expert * 3, sizeof(tr_mat));
        if (layer->exps == NULL) {
            snprintf(err, err_len, "out of memory");
            goto fail;
        }
        tr_type type;
        const void *raw;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_gate_exps.weight", L);
        raw = read_experts(g, m, name, m->n_embd, m->n_ff, m->n_expert, &type, err, err_len);
        if (raw == NULL) goto fail;
        expert_views(layer->exps, raw, type, m->n_embd, m->n_ff, m->n_expert);
        layer->gate_exps = layer->exps;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_up_exps.weight", L);
        raw = read_experts(g, m, name, m->n_embd, m->n_ff, m->n_expert, &type, err, err_len);
        if (raw == NULL) goto fail;
        expert_views(layer->exps + m->n_expert, raw, type, m->n_embd, m->n_ff, m->n_expert);
        layer->up_exps = layer->exps + m->n_expert;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_down_exps.weight", L);
        raw = read_experts(g, m, name, m->n_ff, m->n_embd, m->n_expert, &type, err, err_len);
        if (raw == NULL) goto fail;
        expert_views(layer->exps + 2 * m->n_expert, raw, type, m->n_ff, m->n_embd, m->n_expert);
        layer->down_exps = layer->exps + 2 * m->n_expert;
    }

    m->info.arch = "olmoe";
    m->info.vocab_size = m->vocab;
    m->info.n_ctx_train = m->n_ctx_train;
    m->info.n_layers = m->n_layers;
    m->info.n_embd = m->n_embd;

    tr_gguf_close(g);
    return m;

fail:
    olmoe_free(m);
    tr_gguf_close(g);
    return NULL;
}

static const tr_model_info *olmoe_info(const void *model) {
    const olmoe_model *m = (const olmoe_model *)model;
    return &m->info;
}

/* ---- vtable: session ------------------------------------------------------ */

static void olmoe_session_free(void *session) {
    olmoe_session *s = (olmoe_session *)session;
    if (s == NULL) return;
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
    tr_free_aligned(s->taken);
    tr_free_aligned(s->xg);
    tr_free_aligned(s->h1);
    tr_free_aligned(s->h2);
    tr_free_aligned(s->h3);
    tr_free_aligned(s->logits);
    tr_free_aligned(s->scores);
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
    uint64_t scratch_i64 = (uint64_t)B * (uint64_t)U * 2 + (uint64_t)m->n_expert + 1;
    uint64_t scratch_bytes = scratch_f32 * sizeof(float) + scratch_i64 * sizeof(int64_t) +
                             (uint64_t)(n_workers * m->n_expert);
    if (tr_mem_guard(kv_bytes + scratch_bytes, err, err_len) != 0) return NULL;

    olmoe_session *s = (olmoe_session *)calloc(1, sizeof *s);
    if (s == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    s->m = m;
    s->n_ctx = actual_ctx;
    s->n_batch = B;
    s->pos = 0;

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

    if (kv_rc != 0 || s->rope_cos == NULL || s->rope_sin == NULL || s->x == NULL ||
        s->normed == NULL || s->attn_out == NULL || s->ffn_out == NULL || s->q == NULL || s->attn_concat == NULL ||
        s->k == NULL || s->v == NULL || s->router == NULL || s->sel_id == NULL || s->sel_w == NULL ||
        s->place == NULL || s->offsets == NULL || s->taken == NULL || s->xg == NULL || s->h1 == NULL ||
        s->h2 == NULL || s->h3 == NULL || s->logits == NULL || s->scores == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory allocating session");
        olmoe_session_free(s);
        return NULL;
    }
    tr_rope_table(s->rope_cos, s->rope_sin, actual_ctx, m->head_dim, m->rope_freq_base);
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
 * each row_bytes(type, cols) long (docs/ARCHITETTURA.md §Profilazione). */
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
            tr_attention_group(c->q + i * c->n_qkv + h * c->head_dim, c->n_qkv, keys, values, n_q, c->pos0 + i + 1,
                               c->head_dim, c->scale, scores, c->score_stride,
                               c->out + i * c->n_qkv + h * c->head_dim, c->n_qkv);
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

/* Router of token i: softmax over all experts, top-k (ties to the lower id), optional
 * renormalization, then the selection sorted by increasing expert id (the order in which
 * contributions are added). Writes n_used entries of sel_id and sel_w. */
static void route_token(const olmoe_model *m, float *router, unsigned char *taken, int64_t *sel_id, float *sel_w) {
    int64_t n_expert = m->n_expert, n_used = m->n_expert_used;
    tr_softmax(router, n_expert);
    for (int64_t e = 0; e < n_expert; e++) taken[e] = 0;
    for (int64_t slot = 0; slot < n_used; slot++) {
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
        sel_id[slot] = best;
        sel_w[slot] = best_val;
    }
    if (m->norm_topk) {
        float sum = 0.0f;
        for (int64_t i = 0; i < n_used; i++) sum += sel_w[i];
        for (int64_t i = 0; i < n_used; i++) sel_w[i] /= sum;
    }
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
    unsigned char *taken;
    int64_t *sel_id;
} route_ctx;

static void route_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const route_ctx *c = (const route_ctx *)ctx_;
    int64_t n_expert = c->m->n_expert, n_used = c->m->n_expert_used;
    for (int64_t i = begin; i < end; i++)
        route_token(c->m, c->router + i * n_expert, c->taken + (int64_t)worker * n_expert, c->sel_id + i * n_used,
                    c->sel_w + i * n_used);
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

/* One forward pass over n_tok tokens (1 <= n_tok <= n_batch) at positions pos..pos+n_tok-1.
 * Every value is computed with the same kernel call as when the tokens run one per pass:
 * a matmul element is one dot_row, norms/RoPE/routing/mixing are per token, and a token's
 * attention reads only cache positions up to its own. So the logits and the cache are
 * bit-identical for every split of the input into passes (tests/test_prefill.c).
 * Logits of the last n_logits tokens (0: none), one row each, oldest first. */
static void forward_pass(olmoe_model *m, olmoe_session *s, const int32_t *tokens, int64_t n_tok, int64_t n_logits) {
    tr_prof *prof = &s->prof;
    uint64_t t_pass = tr_prof_begin(prof);

    const tr_kernels *K = tr_kernels_get();
    tr_pool *pool = m->pool;
    const int64_t pos0 = s->pos;
    int64_t n_embd = m->n_embd, n_qkv = m->n_qkv, n_kv = m->n_kv;
    int64_t n_head = m->n_head, n_head_kv = m->n_head_kv, head_dim = m->head_dim;
    int64_t n_ff = m->n_ff, n_expert = m->n_expert, n_used = m->n_expert_used;
    int64_t n_rows = n_tok * n_used; /* (token, slot) pairs = rows of the grouped expert buffers */
    float scale = 1.0f / sqrtf((float)head_dim);

    /* the jobs of one token alone (see the bodies above): the context changes zone by zone */
    const int64_t per_chunk = OLMOE_TOKENS_PER_CHUNK;
    norm_ctx nc;
    nc.eps = m->rms_eps;

    uint64_t t = tr_prof_begin(prof);
    embed_ctx ec;
    ec.w = &m->token_embd;
    ec.tokens = tokens;
    ec.x = s->x;
    ec.n_embd = n_embd;
    tr_parallel_for(pool, n_tok, per_chunk, embed_body, &ec);
    tr_prof_end(prof, TR_PROF_EMBED, t);
    tr_prof_count(prof, TR_PROF_EMBED,
                  (uint64_t)n_tok * (uint64_t)tr_row_bytes(m->token_embd.type, m->token_embd.cols), 0);

    for (int64_t L = 0; L < m->n_layers; L++) {
        olmoe_layer *layer = &m->layers[L];

        /* ---- attention ---- */
        t = tr_prof_begin(prof);
        nc.src = s->x;
        nc.dst = s->normed;
        nc.weight = layer->attn_norm;
        nc.n = n_embd;
        tr_parallel_for(pool, n_tok, per_chunk, norm_body, &nc);
        tr_prof_end(prof, TR_PROF_ATTN_NORM, t);

        t = tr_prof_begin(prof);
        tr_matmul(pool, &layer->wq, s->normed, n_tok, s->q);
        tr_matmul(pool, &layer->wk, s->normed, n_tok, s->k);
        tr_matmul(pool, &layer->wv, s->normed, n_tok, s->v);
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
        tr_prof_end(prof, TR_PROF_KV_WRITE, t);

        t = tr_prof_begin(prof);
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
        /* an item costs ~30 ns per cached position: short contexts keep several per chunk */
        tr_parallel_for(pool, n_head * n_tok, 1 + 256 / (pos0 + n_tok), attn_body, &ac);
        tr_prof_end(prof, TR_PROF_ATTENTION, t);
        if (prof->enabled) tr_prof_count_kv(prof, TR_PROF_ATTENTION, attn_kv_bytes(n_tok, pos0, n_head, head_dim));

        t = tr_prof_begin(prof);
        tr_matmul(pool, &layer->wo, s->attn_concat, n_tok, s->attn_out);
        add_ctx dc;
        dc.y = s->x;
        dc.x = s->attn_out;
        dc.n = n_embd;
        tr_parallel_for(pool, n_tok, per_chunk, add_body, &dc);
        tr_prof_end(prof, TR_PROF_ATTN_OUT_PROJ, t);
        tr_prof_count(prof, TR_PROF_ATTN_OUT_PROJ, mat_bytes(&layer->wo), 0);

        /* ---- MoE FFN ---- */
        t = tr_prof_begin(prof);
        nc.src = s->x;
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
        tr_matmul_grouped(pool, layer->gate_exps, s->offsets, n_expert, s->xg, s->h1);
        tr_matmul_grouped(pool, layer->up_exps, s->offsets, n_expert, s->xg, s->h2);
        tr_prof_end(prof, TR_PROF_EXPERT_GATE_UP, t);

        t = tr_prof_begin(prof);
        tr_swiglu(pool, s->h1, s->h2, n_rows * n_ff);
        tr_prof_end(prof, TR_PROF_EXPERT_ACT, t);

        t = tr_prof_begin(prof);
        tr_matmul_grouped(pool, layer->down_exps, s->offsets, n_expert, s->h1, s->h3);
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
        mc.x = s->x;
        mc.n_embd = n_embd;
        mc.n_used = n_used;
        tr_parallel_for(pool, n_tok, per_chunk, mix_body, &mc);
        tr_prof_end(prof, TR_PROF_EXPERT_MIX, t);
    }

    if (n_logits > 0) {
        /* One row per kept position: every row is its own dot_row over the same weights, so
         * a row is bit-identical to the logits that token gives alone (tests/test_spec.c). */
        float *rows = s->x + (n_tok - n_logits) * n_embd;
        t = tr_prof_begin(prof);
        for (int64_t i = 0; i < n_logits; i++) tr_rmsnorm(rows + i * n_embd, m->output_norm, n_embd, m->rms_eps);
        tr_prof_end(prof, TR_PROF_OUTPUT_NORM, t);

        t = tr_prof_begin(prof);
        tr_matmul(pool, &m->output, rows, n_logits, s->logits);
        tr_prof_end(prof, TR_PROF_LM_HEAD, t);
        tr_prof_count(prof, TR_PROF_LM_HEAD, mat_bytes(&m->output), 0);
        s->n_logits = n_logits;
    }

    tr_prof_end(prof, TR_PROF_TOKEN, t_pass);
    if (prof->enabled) prof->tokens[prof->phase] += (uint64_t)n_tok;
}

/* Passes of at most n_batch tokens; only the last one computes logits. More than one row of
 * logits needs them all in that pass, so the whole call must fit one pass. */
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

    for (int64_t off = 0; off < n; off += s->n_batch) {
        int64_t len = n - off < s->n_batch ? n - off : s->n_batch;
        forward_pass(m, s, tokens + off, len, off + len == n ? n_logits : 0);
        s->pos += len;
    }
    return 0;
}
/* hot: end */

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

const tr_arch_vtable tr_olmoe_vtable = {
    "olmoe",
    olmoe_load,
    olmoe_free,
    olmoe_info,
    olmoe_session_create,
    olmoe_session_free,
    olmoe_eval,
    olmoe_logits,
    olmoe_pos,
    olmoe_n_ctx,
    olmoe_max_logit_rows,
    olmoe_rewind,
    olmoe_prof,
};
