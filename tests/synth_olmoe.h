/* synth_olmoe.h — writes a synthetic OLMoE GGUF (deterministic weights) of any size, so
 * model tests need no external fixture or Python tool. Norms are f32; matrices are f32 or
 * Q8_0 (then every matrix row length, n_embd and n_ff, must be a multiple of 32). */
#ifndef TR_TEST_SYNTH_OLMOE_H
#define TR_TEST_SYNTH_OLMOE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/format/gguf.h"

typedef struct {
    uint32_t layers, n_embd, n_head, n_head_kv, n_ff, n_expert, n_used, vocab, ctx;
    tr_type type; /* of the 2-D and 3-D tensors: TR_TYPE_F32, TR_TYPE_F16 or TR_TYPE_Q8_0 */
} synth_params;

/* Set to 1 before synth_write: the router's matrix is F32 whatever `type` is, as in every real
 * GGUF, so one model holds weights of two types (tests/test_tier_used.c). */
static int synth_f32_router = 0;

/* Set to 1 before synth_write: every attn_output.weight (the attention's output projection) is
 * all zeros, so a layer's residual add never moves x: layer L+1's own ffn_norm then sees exactly
 * layer L's output, whatever layer L+1's own attention would otherwise have added
 * (tests/test_route.c: makes the route trace's pred_out exact). */
static int synth_zero_attn_out = 0;

/* Set to 1 before synth_write: every layer's ffn_gate_inp is generated with the seed the NEXT
 * layer's would get in the normal sequence (12 tensors per layer apart), so layer L's own router
 * equals the normal model's layer L+1 router; seed++ still advances by exactly one per tensor,
 * so every other tensor keeps its normal seed (tests/test_route.c: makes the route trace's
 * pred_in exact against a second, differently-seeded model). */
static int synth_router_from_next = 0;

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

/* Small pseudo-random weights, same file every run, no NaN/Inf: f32 in [-0.1, 0.1], f16 with a
 * random sign, exponent bits 8..11 (2^-7 .. 2^-4) and a random mantissa, or Q8_0 blocks with an
 * f16 scale in [2^-10, 2^-9) (exponent bits 5, random mantissa) and random quants in
 * [-127, 127]. zero: every byte 0 instead (a Q8_0 block's scale 0 also zeroes its quants). */
static void synth_tensor(synth_buf *hdr, synth_buf *data, const char *name, int ndims, uint64_t ne[3], uint32_t seed,
                         tr_type type, int zero) {
    synth_pad(data, 32);
    synth_str(hdr, name);
    synth_u32(hdr, (uint32_t)ndims);
    uint64_t n = 1;
    for (int i = 0; i < ndims; i++) {
        synth_u64(hdr, ne[i]);
        n *= ne[i];
    }
    synth_u32(hdr, (uint32_t)type);
    synth_u64(hdr, data->len);
    if (type == TR_TYPE_Q8_0) {
        for (uint64_t b = 0; b < n / 32; b++) {
            uint16_t d = 0;
            if (!zero) {
                seed = seed * 1103515245u + 12345u;
                d = (uint16_t)((5u << 10) | ((seed >> 16) & 0x3FFu));
            }
            synth_put(data, &d, sizeof d);
            for (int k = 0; k < 32; k++) {
                int8_t q = 0;
                if (!zero) {
                    seed = seed * 1103515245u + 12345u;
                    q = (int8_t)((int)((seed >> 16) % 255u) - 127);
                }
                synth_put(data, &q, 1);
            }
        }
        return;
    }
    if (type == TR_TYPE_F16) {
        for (uint64_t i = 0; i < n; i++) {
            uint16_t h = 0;
            if (!zero) {
                seed = seed * 1103515245u + 12345u;
                uint32_t r = seed >> 16;
                h = (uint16_t)(((r & 1u) << 15) | ((8u + ((r >> 1) & 3u)) << 10) | ((r >> 3) & 0x3FFu));
            }
            synth_put(data, &h, sizeof h);
        }
        return;
    }
    for (uint64_t i = 0; i < n; i++) {
        if (zero) {
            float v = 0.0f;
            synth_put(data, &v, sizeof v);
            continue;
        }
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
#define T(nm, nd, a, b, c) do { \
        uint64_t ne[3] = {(a), (b), (c)}; \
        const char *nm_ = (nm); \
        tr_type tt = (nd) >= 2 ? P->type : TR_TYPE_F32; \
        if (synth_f32_router && strstr(nm_, "ffn_gate_inp") != NULL) tt = TR_TYPE_F32; \
        uint32_t tensor_seed = seed; \
        if (synth_router_from_next && strstr(nm_, "ffn_gate_inp") != NULL) tensor_seed = seed + 12; \
        int tensor_zero = synth_zero_attn_out && strstr(nm_, "attn_output.weight") != NULL; \
        synth_tensor(&hdr, &data, nm_, (nd), ne, tensor_seed, tt, tensor_zero); \
        seed++; \
        n_tensors++; \
    } while (0)
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
