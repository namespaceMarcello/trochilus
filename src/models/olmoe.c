/* olmoe.c — tr_arch_vtable for "olmoe" (src/models/model.h).
 *
 * Reads hyperparameters and every weight tensor from GGUF metadata (M0: no
 * streaming, the whole model is loaded), then runs a per-token forward pass:
 * embedding -> N x [attention with GQA and NeoX RoPE, softmax-then-topk MoE
 * FFN] -> output norm -> logits. Matches transformers' OlmoeForCausalLM
 * (tools/.venv/.../modeling_olmoe.py): q_norm/k_norm over the whole
 * projection (not per head), router softmax over all experts before top-k,
 * ties broken to the lower expert id, optional norm_topk_prob rescaling. */
#include "model.h"
#include "model_internal.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../base/platform.h"
#include "../kernels/kernels.h"

/* ---- weights ---------------------------------------------------------- */

typedef struct {
    float *attn_norm;   /* [n_embd] */
    float *ffn_norm;     /* [n_embd] */
    float *q_norm;        /* [n_qkv] */
    float *k_norm;          /* [n_kv] */
    tr_mat wq, wk, wv, wo;  /* wq/wk/wv: cols=n_embd; wo: cols=n_qkv, rows=n_embd */
    tr_mat gate_inp;         /* router: rows=n_expert, cols=n_embd */

    /* stacked expert tensors, raw ggml layout [in, out, n_expert]; sliced by
     * expert_slice() at forward time (byte offset e * out * row_bytes(in)). */
    tr_type ffn_gate_type, ffn_up_type, ffn_down_type;
    const void *ffn_gate; /* per expert: rows=n_ff, cols=n_embd */
    const void *ffn_up;    /* per expert: rows=n_ff, cols=n_embd */
    const void *ffn_down;   /* per expert: rows=n_embd, cols=n_ff */
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

typedef struct {
    olmoe_model *m; /* not owned */
    int64_t n_ctx;
    int64_t pos;
    tr_prof prof; /* disabled by default (zero-initialized in olmoe_session_create) */

    float *k_cache; /* [n_layers][n_ctx][n_kv] */
    float *v_cache;  /* [n_layers][n_ctx][n_kv] */

    /* RoPE cos/sin per position: [n_ctx][head_dim / 2] each (tr_rope_table) */
    float *rope_cos, *rope_sin;

    /* per-token scratch, reused every call */
    float *x, *normed, *q, *k, *v, *attn_concat, *attn_out;
    float *router, *ffn_out, *logits;
    float *h1, *h2;  /* gate and up outputs: [n_expert_used][n_ff] */
    float *h3;       /* down outputs: [n_expert_used][n_embd] */
    float *scores;   /* attention scores, one row per pool worker: [n_workers][n_ctx] */
    int64_t n_workers;
    unsigned char *taken;
    int64_t *sel_id;
    float *sel_w;
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

/* hot: begin */
static tr_mat expert_slice(const void *base, tr_type type, int64_t in_dim, int64_t out_dim, int64_t e) {
    size_t rb = tr_row_bytes(type, in_dim);
    tr_mat mm;
    mm.type = type;
    mm.rows = out_dim;
    mm.cols = in_dim;
    mm.data = (const unsigned char *)base + (size_t)e * (size_t)out_dim * rb;
    return mm;
}
/* hot: end */

/* ---- vtable: free ------------------------------------------------------- */

static void olmoe_free(void *model) {
    olmoe_model *m = (olmoe_model *)model;
    if (m == NULL) return;
    for (size_t i = 0; i < m->n_owned; i++) tr_free_aligned(m->owned[i]);
    free(m->owned);
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

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_gate_exps.weight", L);
        layer->ffn_gate =
            read_experts(g, m, name, m->n_embd, m->n_ff, m->n_expert, &layer->ffn_gate_type, err, err_len);
        if (layer->ffn_gate == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_up_exps.weight", L);
        layer->ffn_up = read_experts(g, m, name, m->n_embd, m->n_ff, m->n_expert, &layer->ffn_up_type, err, err_len);
        if (layer->ffn_up == NULL) goto fail;

        snprintf(name, sizeof name, "blk.%" PRId64 ".ffn_down_exps.weight", L);
        layer->ffn_down =
            read_experts(g, m, name, m->n_ff, m->n_embd, m->n_expert, &layer->ffn_down_type, err, err_len);
        if (layer->ffn_down == NULL) goto fail;
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
    tr_free_aligned(s->k_cache);
    tr_free_aligned(s->v_cache);
    tr_free_aligned(s->rope_cos);
    tr_free_aligned(s->rope_sin);
    tr_free_aligned(s->x);
    tr_free_aligned(s->normed);
    tr_free_aligned(s->q);
    tr_free_aligned(s->k);
    tr_free_aligned(s->v);
    tr_free_aligned(s->attn_concat);
    tr_free_aligned(s->attn_out);
    tr_free_aligned(s->router);
    tr_free_aligned(s->h1);
    tr_free_aligned(s->h2);
    tr_free_aligned(s->h3);
    tr_free_aligned(s->ffn_out);
    tr_free_aligned(s->scores);
    tr_free_aligned(s->logits);
    tr_free_aligned(s->taken);
    tr_free_aligned(s->sel_id);
    tr_free_aligned(s->sel_w);
    free(s);
}

static void *olmoe_session_create(void *model, int64_t n_ctx, char *err, size_t err_len) {
    olmoe_model *m = (olmoe_model *)model;

    int64_t actual_ctx = n_ctx;
    if (actual_ctx <= 0) actual_ctx = m->n_ctx_train < 4096 ? m->n_ctx_train : 4096;
    if (actual_ctx <= 0) actual_ctx = 4096;

    int64_t n_workers = m->pool != NULL ? tr_pool_size(m->pool) : 1;
    int64_t n_used = m->n_expert_used;
    uint64_t kv_elems = (uint64_t)m->n_layers * (uint64_t)actual_ctx * (uint64_t)m->n_kv;
    uint64_t kv_bytes = kv_elems * 2 * sizeof(float);
    uint64_t rope_elems = (uint64_t)actual_ctx * (uint64_t)(m->head_dim / 2);
    uint64_t scratch_elems = (uint64_t)(m->n_embd * 4 + m->n_qkv * 2 + m->n_kv * 2 + m->n_expert * 2 +
                                         n_used * (m->n_ff * 2 + m->n_embd) + n_workers * actual_ctx + m->vocab) +
                             rope_elems * 2;
    uint64_t scratch_bytes = scratch_elems * sizeof(float);
    if (tr_mem_guard(kv_bytes + scratch_bytes, err, err_len) != 0) return NULL;

    olmoe_session *s = (olmoe_session *)calloc(1, sizeof *s);
    if (s == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    s->m = m;
    s->n_ctx = actual_ctx;
    s->pos = 0;

    s->k_cache = (float *)tr_alloc_aligned((size_t)kv_elems * sizeof(float), 64);
    s->v_cache = (float *)tr_alloc_aligned((size_t)kv_elems * sizeof(float), 64);
    s->rope_cos = (float *)tr_alloc_aligned((size_t)rope_elems * sizeof(float), 64);
    s->rope_sin = (float *)tr_alloc_aligned((size_t)rope_elems * sizeof(float), 64);
    s->n_workers = n_workers;
    s->x =(float *)tr_alloc_aligned((size_t)m->n_embd * sizeof(float), 64);
    s->normed = (float *)tr_alloc_aligned((size_t)m->n_embd * sizeof(float), 64);
    s->q = (float *)tr_alloc_aligned((size_t)m->n_qkv * sizeof(float), 64);
    s->k = (float *)tr_alloc_aligned((size_t)m->n_kv * sizeof(float), 64);
    s->v = (float *)tr_alloc_aligned((size_t)m->n_kv * sizeof(float), 64);
    s->attn_concat = (float *)tr_alloc_aligned((size_t)m->n_qkv * sizeof(float), 64);
    s->attn_out = (float *)tr_alloc_aligned((size_t)m->n_embd * sizeof(float), 64);
    s->router = (float *)tr_alloc_aligned((size_t)m->n_expert * sizeof(float), 64);
    s->h1 = (float *)tr_alloc_aligned((size_t)(n_used * m->n_ff) * sizeof(float), 64);
    s->h2 = (float *)tr_alloc_aligned((size_t)(n_used * m->n_ff) * sizeof(float), 64);
    s->h3 = (float *)tr_alloc_aligned((size_t)(n_used * m->n_embd) * sizeof(float), 64);
    s->ffn_out = (float *)tr_alloc_aligned((size_t)m->n_embd * sizeof(float), 64);
    s->scores = (float *)tr_alloc_aligned((size_t)(n_workers * actual_ctx) * sizeof(float), 64);
    s->logits = (float *)tr_alloc_aligned((size_t)m->vocab * sizeof(float), 64);
    s->taken = (unsigned char *)tr_alloc_aligned((size_t)m->n_expert * sizeof(unsigned char), 64);
    s->sel_id = (int64_t *)tr_alloc_aligned((size_t)m->n_expert * sizeof(int64_t), 64);
    s->sel_w = (float *)tr_alloc_aligned((size_t)m->n_expert * sizeof(float), 64);

    if (s->k_cache == NULL || s->v_cache == NULL || s->rope_cos == NULL || s->rope_sin == NULL ||
        s->x == NULL || s->normed == NULL || s->q == NULL ||
        s->k == NULL || s->v == NULL || s->attn_concat == NULL || s->attn_out == NULL || s->router == NULL ||
        s->h1 == NULL || s->h2 == NULL || s->h3 == NULL || s->ffn_out == NULL || s->scores == NULL ||
        s->logits == NULL || s->taken == NULL || s->sel_id == NULL || s->sel_w == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory allocating session");
        olmoe_session_free(s);
        return NULL;
    }
    tr_rope_table(s->rope_cos, s->rope_sin, actual_ctx, m->head_dim, m->rope_freq_base);
    return s;
}

/* ---- forward pass --------------------------------------------------------- */
/* hot: begin */

static float *kv_slot(float *cache, int64_t n_ctx, int64_t n_kv, int64_t layer, int64_t pos) {
    return cache + ((size_t)layer * (size_t)n_ctx + (size_t)pos) * (size_t)n_kv;
}

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

/* One layer's attention, split by query head: each head writes only its own slice of
 * attn_concat and uses the score row of the worker running it. */
typedef struct {
    const float *q, *keys, *values;
    float *scores, *out;
    float scale;
    int64_t n_ctx, n_kv, head_dim, group, n_pos;
} attn_ctx;

static void attn_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const attn_ctx *c = (const attn_ctx *)ctx_;
    float *scores = c->scores + (int64_t)worker * c->n_ctx;
    for (int64_t h = begin; h < end; h++) {
        tr_attention_head(c->q + h * c->head_dim, c->keys, c->values, c->n_kv, (h / c->group) * c->head_dim,
                          c->n_pos, c->head_dim, c->scale, scores, c->out + h * c->head_dim);
    }
}

static void forward_one(olmoe_model *m, olmoe_session *s, int32_t token, int64_t pos) {
    tr_prof *prof = &s->prof;
    uint64_t t_token = tr_prof_begin(prof);

    const tr_kernels *K = tr_kernels_get();
    tr_pool *pool = m->pool;
    const float *rope_cos = s->rope_cos + pos * (m->head_dim / 2);
    const float *rope_sin = s->rope_sin + pos * (m->head_dim / 2);
    int64_t n_embd = m->n_embd, n_qkv = m->n_qkv, n_kv = m->n_kv;
    int64_t n_head = m->n_head, n_head_kv = m->n_head_kv, head_dim = m->head_dim;
    int64_t n_ff = m->n_ff, n_expert = m->n_expert, n_used = m->n_expert_used;
    int64_t group = n_head / n_head_kv;
    float scale = 1.0f / sqrtf((float)head_dim);

    uint64_t t = tr_prof_begin(prof);
    tr_get_row(&m->token_embd, token, s->x);
    tr_prof_end(prof, TR_PROF_EMBED, t);
    tr_prof_count(prof, (uint64_t)tr_row_bytes(m->token_embd.type, m->token_embd.cols), 0);

    for (int64_t L = 0; L < m->n_layers; L++) {
        olmoe_layer *layer = &m->layers[L];

        /* ---- attention ---- */
        t = tr_prof_begin(prof);
        memcpy(s->normed, s->x, (size_t)n_embd * sizeof(float));
        tr_rmsnorm(s->normed, layer->attn_norm, n_embd, m->rms_eps);
        tr_prof_end(prof, TR_PROF_ATTN_NORM, t);

        t = tr_prof_begin(prof);
        tr_matmul(pool, &layer->wq, s->normed, 1, s->q);
        tr_matmul(pool, &layer->wk, s->normed, 1, s->k);
        tr_matmul(pool, &layer->wv, s->normed, 1, s->v);
        tr_prof_end(prof, TR_PROF_QKV_PROJ, t);
        tr_prof_count(prof, mat_bytes(&layer->wq) + mat_bytes(&layer->wk) + mat_bytes(&layer->wv), 0);

        t = tr_prof_begin(prof);
        tr_rmsnorm(s->q, layer->q_norm, n_qkv, m->rms_eps);
        tr_rmsnorm(s->k, layer->k_norm, n_kv, m->rms_eps);
        tr_prof_end(prof, TR_PROF_QK_NORM, t);

        if (m->have_clamp) {
            clamp_inplace(s->q, n_qkv, m->clamp);
            clamp_inplace(s->k, n_kv, m->clamp);
            clamp_inplace(s->v, n_kv, m->clamp);
        }

        t = tr_prof_begin(prof);
        tr_rope_neox(s->q, n_head, head_dim, rope_cos, rope_sin);
        tr_rope_neox(s->k, n_head_kv, head_dim, rope_cos, rope_sin);
        tr_prof_end(prof, TR_PROF_ROPE, t);

        t = tr_prof_begin(prof);
        memcpy(kv_slot(s->k_cache, s->n_ctx, n_kv, L, pos), s->k, (size_t)n_kv * sizeof(float));
        memcpy(kv_slot(s->v_cache, s->n_ctx, n_kv, L, pos), s->v, (size_t)n_kv * sizeof(float));
        tr_prof_end(prof, TR_PROF_KV_WRITE, t);

        t = tr_prof_begin(prof);
        attn_ctx ac;
        ac.q = s->q;
        ac.keys = kv_slot(s->k_cache, s->n_ctx, n_kv, L, 0);
        ac.values = kv_slot(s->v_cache, s->n_ctx, n_kv, L, 0);
        ac.scores = s->scores;
        ac.out = s->attn_concat;
        ac.scale = scale;
        ac.n_ctx = s->n_ctx;
        ac.n_kv = n_kv;
        ac.head_dim = head_dim;
        ac.group = group;
        ac.n_pos = pos + 1;
        /* a head costs ~30 ns per cached position: short contexts keep several heads per chunk */
        tr_parallel_for(pool, n_head, 1 + 256 / (pos + 1), attn_body, &ac);
        tr_prof_end(prof, TR_PROF_ATTENTION, t);

        t = tr_prof_begin(prof);
        tr_matmul(pool, &layer->wo, s->attn_concat, 1, s->attn_out);
        for (int64_t i = 0; i < n_embd; i++) s->x[i] += s->attn_out[i];
        tr_prof_end(prof, TR_PROF_ATTN_OUT_PROJ, t);
        tr_prof_count(prof, mat_bytes(&layer->wo), 0);

        /* ---- MoE FFN ---- */
        t = tr_prof_begin(prof);
        memcpy(s->normed, s->x, (size_t)n_embd * sizeof(float));
        tr_rmsnorm(s->normed, layer->ffn_norm, n_embd, m->rms_eps);
        tr_prof_end(prof, TR_PROF_FFN_NORM, t);

        t = tr_prof_begin(prof);
        tr_matmul(pool, &layer->gate_inp, s->normed, 1, s->router);
        tr_softmax(s->router, n_expert);

        for (int64_t e = 0; e < n_expert; e++) s->taken[e] = 0;
        for (int64_t slot = 0; slot < n_used; slot++) {
            int64_t best = -1;
            float best_val = 0.0f;
            for (int64_t e = 0; e < n_expert; e++) {
                if (s->taken[e]) continue;
                if (best == -1 || s->router[e] > best_val) {
                    best = e;
                    best_val = s->router[e];
                }
            }
            s->taken[best] = 1;
            s->sel_id[slot] = best;
            s->sel_w[slot] = best_val;
        }
        if (m->norm_topk) {
            float sum = 0.0f;
            for (int64_t i = 0; i < n_used; i++) sum += s->sel_w[i];
            for (int64_t i = 0; i < n_used; i++) s->sel_w[i] /= sum;
        }
        /* sort selected experts by increasing id: contributions are added in
         * that order (the task's documented accumulation order). */
        for (int64_t i = 1; i < n_used; i++) {
            int64_t id = s->sel_id[i];
            float w = s->sel_w[i];
            int64_t j = i - 1;
            while (j >= 0 && s->sel_id[j] > id) {
                s->sel_id[j + 1] = s->sel_id[j];
                s->sel_w[j + 1] = s->sel_w[j];
                j--;
            }
            s->sel_id[j + 1] = id;
            s->sel_w[j + 1] = w;
        }
        tr_prof_end(prof, TR_PROF_ROUTER, t);
        tr_prof_count(prof, mat_bytes(&layer->gate_inp), 0);

        /* experts in stages, each stage over all selected experts: the activations of
         * every expert then run as one parallel job (slot i: h1/h2 at i*n_ff, h3 at i*n_embd) */
        t = tr_prof_begin(prof);
        for (int64_t slot = 0; slot < n_used; slot++) {
            int64_t e = s->sel_id[slot];
            tr_mat gate_e = expert_slice(layer->ffn_gate, layer->ffn_gate_type, n_embd, n_ff, e);
            tr_mat up_e = expert_slice(layer->ffn_up, layer->ffn_up_type, n_embd, n_ff, e);
            tr_matmul(pool, &gate_e, s->normed, 1, s->h1 + slot * n_ff);
            tr_matmul(pool, &up_e, s->normed, 1, s->h2 + slot * n_ff);
            tr_prof_count(prof, mat_bytes(&gate_e) + mat_bytes(&up_e), 0);
        }
        tr_prof_end(prof, TR_PROF_EXPERT_GATE_UP, t);

        t = tr_prof_begin(prof);
        tr_swiglu(pool, s->h1, s->h2, n_used * n_ff);
        tr_prof_end(prof, TR_PROF_EXPERT_ACT, t);

        t = tr_prof_begin(prof);
        for (int64_t slot = 0; slot < n_used; slot++) {
            tr_mat down_e = expert_slice(layer->ffn_down, layer->ffn_down_type, n_ff, n_embd, s->sel_id[slot]);
            tr_matmul(pool, &down_e, s->h1 + slot * n_ff, 1, s->h3 + slot * n_embd);
            tr_prof_count(prof, mat_bytes(&down_e), 0);
        }
        tr_prof_end(prof, TR_PROF_EXPERT_DOWN, t);

        t = tr_prof_begin(prof);
        for (int64_t i = 0; i < n_embd; i++) s->ffn_out[i] = 0.0f;
        for (int64_t slot = 0; slot < n_used; slot++)
            K->axpy_f32(s->ffn_out, s->h3 + slot * n_embd, s->sel_w[slot], n_embd);
        tr_prof_end(prof, TR_PROF_EXPERT_MIX, t);
        for (int64_t i = 0; i < n_embd; i++) s->x[i] += s->ffn_out[i];
    }

    t = tr_prof_begin(prof);
    tr_rmsnorm(s->x, m->output_norm, n_embd, m->rms_eps);
    tr_prof_end(prof, TR_PROF_OUTPUT_NORM, t);

    t = tr_prof_begin(prof);
    tr_matmul(pool, &m->output, s->x, 1, s->logits);
    tr_prof_end(prof, TR_PROF_LM_HEAD, t);
    tr_prof_count(prof, mat_bytes(&m->output), 0);

    tr_prof_end(prof, TR_PROF_TOKEN, t_token);
    if (prof->enabled) prof->tokens[prof->phase]++;
}

static int olmoe_eval(void *session, const int32_t *tokens, int64_t n) {
    olmoe_session *s = (olmoe_session *)session;
    olmoe_model *m = s->m;

    if (n <= 0) return -1;
    if (s->pos + n > s->n_ctx) return -1;
    for (int64_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= m->vocab) return -1;
    }

    for (int64_t i = 0; i < n; i++) {
        forward_one(m, s, tokens[i], s->pos);
        s->pos++;
    }
    return 0;
}
/* hot: end */

static const float *olmoe_logits(const void *session) {
    const olmoe_session *s = (const olmoe_session *)session;
    return s->logits;
}

static int64_t olmoe_pos(const void *session) {
    const olmoe_session *s = (const olmoe_session *)session;
    return s->pos;
}

/* The cache rows past n are simply overwritten by the next eval: attention only ever
 * reads positions below the current one. */
static int olmoe_rewind(void *session, int64_t n) {
    olmoe_session *s = (olmoe_session *)session;
    if (n < 0 || n > s->pos) return -1;
    s->pos = n;
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
    olmoe_rewind,
    olmoe_prof,
};
