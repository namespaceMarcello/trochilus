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
#include "../src/kernels/kernels_internal.h"

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

/* Set to 1 before synth_write: the file also carries a byte-level tokenizer (pre "olmo", no
 * merges) and OLMoE-0125-Instruct's chat template, so `trochilus run` and `chat` take it
 * (tests/test_cli.c). Token b < 256 is byte b in GPT-2's byte alphabet; 256 <|endoftext|> (BOS
 * and EOS), 257 <|system|>, 258 <|user|>, 259 <|assistant|>, all control; the ids above are
 * unused pads. P->vocab must be at least SYNTH_TOK_MIN_VOCAB. */
static int synth_with_tokenizer = 0;
#define SYNTH_TOK_MIN_VOCAB 260u

/* Set to 1 before synth_write: every ffn_gate_inp.weight is all zeros, so every expert gets the
 * same router score and the choice rests on the tie rule alone (tests/test_model_load.c). */
static int synth_zero_router = 0;

/* Set to a layer before synth_write (-1: none): only that layer's ffn_gate_inp.weight is all zeros,
 * so every token of that layer takes experts 0..n_used-1 and the others go unused
 * (tests/test_prefetch.c: units read ahead that nobody asks for). */
static int synth_zero_router_layer = -1;
static int synth_is_zero_router_layer(const char *name) {
    char want[64];
    if (synth_zero_router_layer < 0) return 0;
    snprintf(want, sizeof want, "blk.%d.ffn_gate_inp.weight", synth_zero_router_layer);
    return strcmp(name, want) == 0;
}

/* Malformed files (tests/test_model_load.c), all off by default: synth_drop, a key or a tensor of
 * that name left out; synth_set_key[i], a u32 key written with synth_set_value[i] instead, or
 * added after the others when the file has no such key; synth_reshape, a tensor whose dimension
 * synth_reshape_dim is one larger; synth_retype, a tensor stored as synth_retype_to (all bytes
 * zero when that is not F32, F16 or Q8_0). */
static const char *synth_drop = NULL;
static const char *synth_set_key[2] = {NULL, NULL};
static uint32_t synth_set_value[2];
static int synth_set_done[2];
static const char *synth_reshape = NULL;
static int synth_reshape_dim = 0;
static const char *synth_retype = NULL;
static tr_type synth_retype_to = TR_TYPE_F32;
/* A tensor written F32 whose every value is exactly the F16 value the same seed gives as F16, so
 * that an F16 copy of it must give the same bits (tests/test_model_load.c: mixed types per part). */
static const char *synth_f16_as_f32 = NULL;

/* the next F16 weight of a seed's sequence (synth_tensor) */
static uint16_t synth_f16_bits(uint32_t *seed) {
    *seed = *seed * 1103515245u + 12345u;
    uint32_t r = *seed >> 16;
    return (uint16_t)(((r & 1u) << 15) | ((8u + ((r >> 1) & 3u)) << 10) | ((r >> 3) & 0x3FFu));
}

/* The template's exact bytes: the engine recognizes a template by its length and hash
 * (src/tokenizer/chat.c), never by parsing it. */
static const char SYNTH_OLMOE_TEMPLATE[] =
    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + "
    "message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + "
    "'\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + "
    "message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token "
    "}}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% "
    "endfor %}";

/* The template synth_with_tokenizer writes: SYNTH_OLMOE_TEMPLATE unless a test points it elsewhere
 * (tests/test_cli.c: the same length with one byte changed must be refused). */
static const char *synth_chat_template = SYNTH_OLMOE_TEMPLATE;

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

/* 1 if written, 0 if synth_drop left it out */
static uint64_t synth_kv_u32(synth_buf *b, const char *key, uint32_t v) {
    if (synth_drop != NULL && strcmp(key, synth_drop) == 0) return 0;
    for (int i = 0; i < 2; i++)
        if (synth_set_key[i] != NULL && strcmp(key, synth_set_key[i]) == 0) {
            v = synth_set_value[i];
            synth_set_done[i] = 1;
        }
    synth_str(b, key);
    synth_u32(b, TR_GGUF_UINT32);
    synth_u32(b, v);
    return 1;
}

static void synth_kv_str(synth_buf *b, const char *key, const char *s) {
    synth_str(b, key);
    synth_u32(b, TR_GGUF_STRING);
    synth_str(b, s);
}

static void synth_kv_bool(synth_buf *b, const char *key, int v) {
    uint8_t x = (uint8_t)(v != 0);
    synth_str(b, key);
    synth_u32(b, TR_GGUF_BOOL);
    synth_put(b, &x, 1);
}

static void synth_kv_array(synth_buf *b, const char *key, uint32_t elem_type, uint64_t n) {
    synth_str(b, key);
    synth_u32(b, TR_GGUF_ARRAY);
    synth_u32(b, elem_type);
    synth_u64(b, n);
}

/* synth_with_tokenizer's keys; returns how many it wrote. */
static uint64_t synth_tokenizer_kv(synth_buf *b, uint32_t vocab) {
    static const char *const special[4] = {"<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"};
    uint64_t n = 0;
    synth_kv_str(b, "tokenizer.ggml.model", "gpt2"), n++;
    synth_kv_str(b, "tokenizer.ggml.pre", "olmo"), n++;
    synth_kv_array(b, "tokenizer.ggml.tokens", TR_GGUF_STRING, vocab), n++;
    uint32_t next = 256;
    for (uint32_t id = 0; id < vocab; id++) {
        char s[32];
        if (id < 256) {
            /* GPT-2's byte alphabet: printable bytes stand for themselves, the others map in
             * order to U+0100.. (src/tokenizer/tokenizer.c byte_alphabet), written as UTF-8 */
            int printable = (id >= 33 && id <= 126) || (id >= 161 && id <= 172) || (id >= 174 && id <= 255);
            uint32_t cp = printable ? id : next++;
            if (cp < 0x80) {
                s[0] = (char)cp;
                s[1] = 0;
            } else {
                s[0] = (char)(0xC0 | (cp >> 6));
                s[1] = (char)(0x80 | (cp & 0x3F));
                s[2] = 0;
            }
        } else if (id < 260) {
            snprintf(s, sizeof s, "%s", special[id - 256]);
        } else {
            snprintf(s, sizeof s, "<|pad_%u|>", (unsigned)id);
        }
        synth_str(b, s);
    }
    synth_kv_array(b, "tokenizer.ggml.token_type", TR_GGUF_INT32, vocab), n++;
    for (uint32_t id = 0; id < vocab; id++) synth_u32(b, id < 256 ? 1u : id < 260 ? 3u : 5u);
    synth_kv_array(b, "tokenizer.ggml.merges", TR_GGUF_STRING, 0), n++;
    n += synth_kv_u32(b, "tokenizer.ggml.bos_token_id", 256);
    n += synth_kv_u32(b, "tokenizer.ggml.eos_token_id", 256);
    synth_kv_bool(b, "tokenizer.ggml.add_bos_token", 0), n++;
    synth_kv_str(b, "tokenizer.chat_template", synth_chat_template), n++;
    return n;
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
    if (type != TR_TYPE_F32 && type != TR_TYPE_F16 && type != TR_TYPE_Q8_0) {
        const tr_type_info *ti = tr_type_get((uint32_t)type);
        uint8_t z = 0;
        for (uint64_t i = 0; i < n / ti->block_elems * ti->block_bytes; i++) synth_put(data, &z, 1);
        return;
    }
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
            uint16_t h = zero ? 0 : synth_f16_bits(&seed);
            synth_put(data, &h, sizeof h);
        }
        return;
    }
    if (synth_f16_as_f32 != NULL && strcmp(name, synth_f16_as_f32) == 0) {
        for (uint64_t i = 0; i < n; i++) {
            float v = tr_half_to_float(synth_f16_bits(&seed));
            synth_put(data, &v, sizeof v);
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
    synth_set_done[0] = synth_set_done[1] = 0;

    synth_u32(&hdr, 0x46554747u); /* "GGUF" */
    synth_u32(&hdr, 3);
    synth_u64(&hdr, 0); /* tensors, patched below */
    synth_u64(&hdr, 0); /* keys, patched below */

    synth_str(&hdr, "general.architecture");
    synth_u32(&hdr, TR_GGUF_STRING);
    synth_str(&hdr, "olmoe");
    n_kv_count++;
    n_kv_count += synth_kv_u32(&hdr, "olmoe.block_count", P->layers);
    n_kv_count += synth_kv_u32(&hdr, "olmoe.embedding_length", P->n_embd);
    n_kv_count += synth_kv_u32(&hdr, "olmoe.feed_forward_length", P->n_ff);
    n_kv_count += synth_kv_u32(&hdr, "olmoe.attention.head_count", P->n_head);
    n_kv_count += synth_kv_u32(&hdr, "olmoe.attention.head_count_kv", P->n_head_kv);
    n_kv_count += synth_kv_u32(&hdr, "olmoe.expert_count", P->n_expert);
    n_kv_count += synth_kv_u32(&hdr, "olmoe.expert_used_count", P->n_used);
    n_kv_count += synth_kv_u32(&hdr, "olmoe.context_length", P->ctx);
    synth_str(&hdr, "olmoe.attention.layer_norm_rms_epsilon");
    synth_u32(&hdr, TR_GGUF_FLOAT32);
    float eps = 1e-5f;
    synth_put(&hdr, &eps, sizeof eps);
    n_kv_count++;
    if (synth_with_tokenizer) n_kv_count += synth_tokenizer_kv(&hdr, P->vocab);
    for (int i = 0; i < 2; i++)
        if (synth_set_key[i] != NULL && !synth_set_done[i]) n_kv_count += synth_kv_u32(&hdr, synth_set_key[i], 0);

    uint32_t seed = 1;
#define T(nm, nd, a, b, c) do { \
        uint64_t ne[3] = {(a), (b), (c)}; \
        const char *nm_ = (nm); \
        if (synth_drop != NULL && strcmp(nm_, synth_drop) == 0) { \
            seed++; \
            break; \
        } \
        tr_type tt = (nd) >= 2 ? P->type : TR_TYPE_F32; \
        if (synth_f32_router && strstr(nm_, "ffn_gate_inp") != NULL) tt = TR_TYPE_F32; \
        if (synth_retype != NULL && strcmp(nm_, synth_retype) == 0) tt = synth_retype_to; \
        if (synth_reshape != NULL && strcmp(nm_, synth_reshape) == 0) ne[synth_reshape_dim]++; \
        uint32_t tensor_seed = seed; \
        if (synth_router_from_next && strstr(nm_, "ffn_gate_inp") != NULL) tensor_seed = seed + 12; \
        int tensor_zero = (synth_zero_attn_out && strstr(nm_, "attn_output.weight") != NULL) || \
                          (synth_zero_router && strstr(nm_, "ffn_gate_inp") != NULL) || \
                          synth_is_zero_router_layer(nm_); \
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
