/* synth_olmoe.h — writes a synthetic OLMoE GGUF (f32, deterministic weights) of any
 * size, so model tests need no external fixture or Python tool. */
#ifndef TR_TEST_SYNTH_OLMOE_H
#define TR_TEST_SYNTH_OLMOE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/format/gguf.h"

typedef struct {
    uint32_t layers, n_embd, n_head, n_head_kv, n_ff, n_expert, n_used, vocab, ctx;
} synth_params;

typedef struct {
    uint8_t *data;
    size_t len, cap;
} synth_buf;

static void synth_put(synth_buf *b, const void *p, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2;
        b->data = realloc(b->data, b->cap);
        if (b->data == NULL) abort();
    }
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static void synth_u32(synth_buf *b, uint32_t v) { synth_put(b, &v, 4); }
static void synth_u64(synth_buf *b, uint64_t v) { synth_put(b, &v, 8); }
static void synth_str(synth_buf *b, const char *s) { synth_u64(b, strlen(s)); synth_put(b, s, strlen(s)); }
static void synth_pad(synth_buf *b, size_t align) { uint8_t z = 0; while (b->len % align) synth_put(b, &z, 1); }

static void synth_kv_u32(synth_buf *b, const char *key, uint32_t v) {
    synth_str(b, key);
    synth_u32(b, TR_GGUF_UINT32);
    synth_u32(b, v);
}

/* Small pseudo-random weights in [-0.1, 0.1]: same file every run, no NaN/Inf. */
static void synth_tensor(synth_buf *hdr, synth_buf *data, const char *name, int ndims, uint64_t ne[3], uint32_t seed) {
    synth_pad(data, 32);
    synth_str(hdr, name);
    synth_u32(hdr, (uint32_t)ndims);
    uint64_t n = 1;
    for (int i = 0; i < ndims; i++) {
        synth_u64(hdr, ne[i]);
        n *= ne[i];
    }
    synth_u32(hdr, (uint32_t)TR_TYPE_F32);
    synth_u64(hdr, data->len);
    for (uint64_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        float v = ((float)((seed >> 16) & 0xFFFFu) / 65535.0f - 0.5f) * 0.2f;
        synth_put(data, &v, sizeof v);
    }
}

/* The whole file in memory; the caller frees .data. */
static synth_buf synth_olmoe(const synth_params *P) {
    uint64_t head_dim = P->n_embd / P->n_head, n_qkv = P->n_embd, n_kv = P->n_head_kv * head_dim;
    synth_buf hdr = {0}, data = {0};
    uint64_t n_kv_count = 0, n_tensors = 0;
    char name[64];

    synth_u32(&hdr, 0x46554747u); /* "GGUF" */
    synth_u32(&hdr, 3);
    synth_u64(&hdr, 0); /* tensors, patched below */
    synth_u64(&hdr, 0); /* keys, patched below */

    synth_str(&hdr, "general.architecture");
    synth_u32(&hdr, TR_GGUF_STRING);
    synth_str(&hdr, "olmoe");
    n_kv_count++;
    synth_kv_u32(&hdr, "olmoe.block_count", P->layers), n_kv_count++;
    synth_kv_u32(&hdr, "olmoe.embedding_length", P->n_embd), n_kv_count++;
    synth_kv_u32(&hdr, "olmoe.feed_forward_length", P->n_ff), n_kv_count++;
    synth_kv_u32(&hdr, "olmoe.attention.head_count", P->n_head), n_kv_count++;
    synth_kv_u32(&hdr, "olmoe.attention.head_count_kv", P->n_head_kv), n_kv_count++;
    synth_kv_u32(&hdr, "olmoe.expert_count", P->n_expert), n_kv_count++;
    synth_kv_u32(&hdr, "olmoe.expert_used_count", P->n_used), n_kv_count++;
    synth_kv_u32(&hdr, "olmoe.context_length", P->ctx), n_kv_count++;
    synth_str(&hdr, "olmoe.attention.layer_norm_rms_epsilon");
    synth_u32(&hdr, TR_GGUF_FLOAT32);
    float eps = 1e-5f;
    synth_put(&hdr, &eps, sizeof eps);
    n_kv_count++;

    uint32_t seed = 1;
#define T(nm, nd, a, b, c) do { uint64_t ne[3] = {(a), (b), (c)}; synth_tensor(&hdr, &data, (nm), (nd), ne, seed++); n_tensors++; } while (0)
    T("token_embd.weight", 2, P->n_embd, P->vocab, 0);
    T("output_norm.weight", 1, P->n_embd, 0, 0);
    T("output.weight", 2, P->n_embd, P->vocab, 0);
    for (uint32_t L = 0; L < P->layers; L++) {
#define NM(suffix) (snprintf(name, sizeof name, "blk.%u.%s", (unsigned)L, (suffix)), name)
        T(NM("attn_norm.weight"), 1, P->n_embd, 0, 0);
        T(NM("attn_q.weight"), 2, P->n_embd, n_qkv, 0);
        T(NM("attn_k.weight"), 2, P->n_embd, n_kv, 0);
        T(NM("attn_v.weight"), 2, P->n_embd, n_kv, 0);
        T(NM("attn_output.weight"), 2, n_qkv, P->n_embd, 0);
        T(NM("attn_q_norm.weight"), 1, n_qkv, 0, 0);
        T(NM("attn_k_norm.weight"), 1, n_kv, 0, 0);
        T(NM("ffn_norm.weight"), 1, P->n_embd, 0, 0);
        T(NM("ffn_gate_inp.weight"), 2, P->n_embd, P->n_expert, 0);
        T(NM("ffn_gate_exps.weight"), 3, P->n_embd, P->n_ff, P->n_expert);
        T(NM("ffn_up_exps.weight"), 3, P->n_embd, P->n_ff, P->n_expert);
        T(NM("ffn_down_exps.weight"), 3, P->n_ff, P->n_embd, P->n_expert);
#undef NM
    }
#undef T

    memcpy(hdr.data + 8, &n_tensors, 8);
    memcpy(hdr.data + 16, &n_kv_count, 8);
    synth_pad(&hdr, 32);
    synth_buf out = {0};
    synth_put(&out, hdr.data, hdr.len);
    synth_put(&out, data.data, data.len);
    free(hdr.data);
    free(data.data);
    return out;
}

/* Writes the model next to the test binary (tmpfile() needs write access to the drive
 * root on Windows); path receives the file name. Returns 0 on success. */
static int synth_write(const synth_params *P, const char *argv0, const char *file, char *path, size_t path_len) {
    const char *sl = strrchr(argv0, '/'), *bs = strrchr(argv0, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    snprintf(path, path_len, "%.*s/%s", sl ? (int)(sl - argv0) : 1, sl ? argv0 : ".", file);
    synth_buf b = synth_olmoe(P);
    FILE *f = fopen(path, "wb");
    int ok = f != NULL && fwrite(b.data, 1, b.len, f) == b.len;
    if (f != NULL) fclose(f);
    free(b.data);
    return ok ? 0 : -1;
}

#endif
