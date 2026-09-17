/* llamacpp_logits.c — the reference side of tools/compare_llamacpp.py, not part of the engine.
 *
 * Runs token ids through llama.cpp (ref/llama.cpp, built by tools/build_llamacpp.sh in the
 * container) on the CPU and writes the logits after each token exactly like `trochilus logits`
 * (vocab_size float32 per position, no header). With -n it then continues greedily and prints
 * "tokens: id,id,..." like `trochilus generate`.
 *
 *   llamacpp_logits -m <file.gguf> --tokens id,id,... --out <file> [-n count] [-t threads] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

static int parse_ids(const char *s, llama_token **out, int *n) {
    int cap = 64;
    *n = 0;
    *out = (llama_token *)malloc((size_t)cap * sizeof(llama_token));
    while (*out != NULL && *s != '\0') {
        char *end;
        long v = strtol(s, &end, 10);
        if (end == s) return -1;
        if (*n == cap) {
            cap *= 2;
            *out = (llama_token *)realloc(*out, (size_t)cap * sizeof(llama_token));
            if (*out == NULL) return -1;
        }
        (*out)[(*n)++] = (llama_token)v;
        s = *end == ',' ? end + 1 : end;
    }
    return *out != NULL && *n > 0 ? 0 : -1;
}

static llama_token argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return (llama_token)best;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *ids_str = NULL, *out_path = NULL;
    int n_gen = 0, n_threads = 8;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) ids_str = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
    }
    llama_token *ids;
    int n;
    if (model_path == NULL || ids_str == NULL || out_path == NULL || parse_ids(ids_str, &ids, &n) != 0) {
        fprintf(stderr, "usage: llamacpp_logits -m <file.gguf> --tokens id,id,... --out <file> [-n count] [-t threads]\n");
        return 2;
    }

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (model == NULL) return 1;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(n + n_gen + 16);
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx == NULL) return 1;

    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) return 1;

    FILE *out = fopen(out_path, "wb");
    if (out == NULL) return 1;
    for (int i = 0; i < n; i++)
        if (fwrite(llama_get_logits_ith(ctx, i), sizeof(float), (size_t)vocab, out) != (size_t)vocab) return 1;
    fclose(out);

    if (n_gen > 0) {
        llama_token next = argmax(llama_get_logits_ith(ctx, n - 1), vocab);
        printf("tokens:");
        for (int k = 0; k < n_gen; k++) {
            printf(k ? ",%d" : " %d", next);
            if (k + 1 == n_gen) break;
            struct llama_batch one = llama_batch_init(1, 0, 1);
            one.token[0] = next;
            one.pos[0] = n + k;
            one.n_seq_id[0] = 1;
            one.seq_id[0][0] = 0;
            one.logits[0] = 1;
            one.n_tokens = 1;
            if (llama_decode(ctx, one) != 0) return 1;
            next = argmax(llama_get_logits_ith(ctx, 0), vocab);
            llama_batch_free(one);
        }
        printf("\n");
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(ids);
    return 0;
}
