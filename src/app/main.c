/* main.c — command line entry point.
 *
 *   trochilus cpu                             what this machine offers (cores, instruction sets, RAM)
 *   trochilus inspect <file>                  GGUF metadata and tensor directory
 *   trochilus generate -m <f> --tokens .. -n <n> [-t threads]   greedy-decode n tokens
 *   trochilus logits -m <f> --tokens .. --out <f> [-t threads]  logits after each token
 *
 * Exit codes: 0 success, 1 runtime error, 2 usage, 3 context full before all tokens were generated. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "../base/platform.h"
#include "../base/cpu.h"
#include "../base/prof.h"
#include "../base/threads.h"
#include "../format/gguf.h"
#include "../kernels/kernels.h"
#include "../models/model.h"

static void usage(void) {
    fprintf(stderr,
            "usage:\n"
            "  trochilus cpu               describe this machine\n"
            "  trochilus inspect <file>    print GGUF metadata and tensors\n"
            "  trochilus generate -m <file.gguf> (--tokens id,id,... | -p <n>) -n <count>\n"
            "                     [-t threads] [-c context] [--profile] [--profile-json <file>]\n"
            "  trochilus logits -m <file.gguf> --tokens id,id,... --out <file> [-t threads]\n");
}

static void print_bytes(uint64_t n) {
    static const char *unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = (double)n;
    int u = 0;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; u++; }
    printf(u ? "%.2f %s" : "%.0f %s", v, unit[u]);
}

static int cmd_cpu(void) {
    char line[512];
    tr_cpu_describe(tr_cpu(), line, sizeof line);
    printf("cpu: %s\n", line);
    tr_meminfo m;
    if (tr_mem_info(&m) == 0) {
        printf("ram: ");
        print_bytes(m.total_bytes);
        printf(" total, ");
        print_bytes(m.available_bytes);
        printf(" available\n");
    }
    return 0;
}

static const char *vtype_name(tr_gguf_vtype t) {
    static const char *names[] = {"u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool",
                                  "str", "arr", "u64", "i64", "f64"};
    return (unsigned)t < sizeof names / sizeof names[0] ? names[t] : "?";
}

static void print_value(const tr_gguf_kv *kv) {
    switch (kv->type) {
    case TR_GGUF_UINT8: case TR_GGUF_UINT16: case TR_GGUF_UINT32: case TR_GGUF_UINT64:
        printf("%" PRIu64, kv->v.u); break;
    case TR_GGUF_INT8: case TR_GGUF_INT16: case TR_GGUF_INT32: case TR_GGUF_INT64:
        printf("%" PRId64, kv->v.i); break;
    case TR_GGUF_FLOAT32: case TR_GGUF_FLOAT64:
        printf("%g", kv->v.f); break;
    case TR_GGUF_BOOL:
        printf("%s", kv->v.u ? "true" : "false"); break;
    case TR_GGUF_STRING: {
        size_t len = strlen(kv->v.s);
        printf("\"%.*s%s\"", len > 80 ? 80 : (int)len, kv->v.s, len > 80 ? "..." : "");
        break;
    }
    case TR_GGUF_ARRAY:
        printf("[%s x %" PRIu64 "]", vtype_name(kv->arr_type), kv->arr_len);
        if (kv->arr_type == TR_GGUF_STRING && kv->arr_len > 0)
            printf(" first \"%.40s\"", ((char **)kv->arr)[0]);
        break;
    }
}

static int cmd_inspect(const char *path) {
    char err[512];
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    if (g == NULL) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    printf("gguf v%u, %" PRIu64 " metadata entries, %" PRIu64 " tensors\n",
           tr_gguf_version(g), tr_gguf_kv_count(g), tr_gguf_tensor_count(g));
    for (uint64_t i = 0; i < tr_gguf_kv_count(g); i++) {
        const tr_gguf_kv *kv = tr_gguf_kv_at(g, i);
        printf("  %-48s %-4s ", kv->key, vtype_name(kv->type));
        print_value(kv);
        printf("\n");
    }

    uint64_t total = 0, by_type[TR_TYPE_COUNT] = {0};
    for (uint64_t i = 0; i < tr_gguf_tensor_count(g); i++) {
        const tr_gguf_tensor *t = tr_gguf_tensor_at(g, i);
        char shape[96];
        int k = 0;
        for (uint32_t d = 0; d < t->n_dims && k < (int)sizeof shape; d++)
            k += snprintf(shape + k, sizeof shape - (size_t)k, d ? " x %" PRIu64 : "%" PRIu64, t->ne[d]);
        printf("  %-40s %-7s %-24s ", t->name, tr_type_get(t->type)->name, shape);
        print_bytes(t->n_bytes);
        printf("\n");
        total += t->n_bytes;
        by_type[t->type] += t->n_bytes;
    }
    printf("tensor data: ");
    print_bytes(total);
    for (int ty = 0; ty < TR_TYPE_COUNT; ty++) {
        if (by_type[ty] == 0) continue;
        printf(", %s ", tr_type_get((uint32_t)ty)->name);
        print_bytes(by_type[ty]);
    }
    printf("\n");
    tr_gguf_close(g);
    return 0;
}

/* ---- generate / logits: shared helpers ------------------------------------ */

/* Parses a comma-separated list of ids, e.g. "3,11,29", into a malloc'd array. */
static int parse_tokens(const char *s, int32_t **out, int64_t *out_n) {
    int64_t cap = 16, n = 0;
    int32_t *arr = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (arr == NULL) return -1;

    const char *p = s;
    while (*p != '\0') {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) {
            free(arr);
            return -1;
        }
        if (n == cap) {
            cap *= 2;
            int32_t *grown = (int32_t *)realloc(arr, (size_t)cap * sizeof(int32_t));
            if (grown == NULL) {
                free(arr);
                return -1;
            }
            arr = grown;
        }
        arr[n++] = (int32_t)v;
        p = end;
        if (*p == ',') p++;
        else if (*p != '\0') {
            free(arr);
            return -1;
        }
    }
    if (n == 0) {
        free(arr);
        return -1;
    }
    *out = arr;
    *out_n = n;
    return 0;
}

static int32_t argmax_f32(const float *x, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return (int32_t)best;
}

/* Deterministic synthetic prompt for -p <n>: no tokenizer needed, so scenarios
 * can ask for a prompt of any length within the model's vocabulary. */
static void synth_prompt(int32_t *out, int64_t n, int64_t vocab_size) {
    for (int64_t i = 0; i < n; i++) out[i] = (int32_t)((i * 7919 + 13) % vocab_size);
}

/* ---- profiling: --profile-json --------------------------------------------- */

static void write_json_string(FILE *out, const char *s) {
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; p++) {
        if (*p == '"' || *p == '\\') fputc('\\', out);
        fputc(*p, out);
    }
    fputc('"', out);
}

/* {"engine": <tr_prof_write_json output>, "model", "n_prompt", "n_gen", "threads"
 * (actual pool size), "cpu", "kernel_tier"} — the raw material for tools/profile_suite.py. */
static int write_profile_json(const char *path, tr_prof *prof, const char *model_path, int64_t n_prompt,
                               int64_t n_gen, int actual_threads) {
    FILE *out = fopen(path, "w");
    if (out == NULL) return -1;

    fprintf(out, "{\"engine\":");
    tr_prof_write_json(prof, out);

    char cpu_line[512];
    tr_cpu_describe(tr_cpu(), cpu_line, sizeof cpu_line);

    fprintf(out, ",\"model\":");
    write_json_string(out, model_path);
    fprintf(out, ",\"n_prompt\":%" PRId64 ",\"n_gen\":%" PRId64 ",\"threads\":%d,\"cpu\":", n_prompt, n_gen,
            actual_threads);
    write_json_string(out, cpu_line);
    fprintf(out, ",\"kernel_tier\":");
    write_json_string(out, tr_kernels_get()->tier);
    fprintf(out, "}\n");

    fclose(out);
    return 0;
}

/* ---- generate -------------------------------------------------------------- */

static int cmd_generate(int argc, char **argv) {
    const char *model_path = NULL, *tokens_str = NULL, *profile_json_path = NULL;
    int64_t n_gen = -1, n_prompt_synth = -1, n_ctx = 0;
    int n_threads = 0, do_profile = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens_str = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) n_prompt_synth = atoll(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_gen = atoll(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) n_ctx = atoll(argv[++i]);
        else if (strcmp(argv[i], "--profile") == 0) do_profile = 1;
        else if (strcmp(argv[i], "--profile-json") == 0 && i + 1 < argc) profile_json_path = argv[++i];
        else {
            fprintf(stderr, "generate: unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    /* exactly one of --tokens / -p */
    if (model_path == NULL || n_gen < 0 || (tokens_str == NULL) == (n_prompt_synth < 0)) {
        fprintf(stderr,
                "usage: trochilus generate -m <file.gguf> (--tokens id,id,... | -p <n>) -n <count>\n"
                "                   [-t threads] [-c context] [--profile] [--profile-json <file>]\n");
        return 2;
    }

    char err[256];
    tr_pool *pool = tr_pool_create(n_threads);
    if (pool == NULL) {
        fprintf(stderr, "generate: could not create thread pool\n");
        return 1;
    }
    tr_model *model = tr_model_load(model_path, pool, err, sizeof err);
    if (model == NULL) {
        fprintf(stderr, "error: %s\n", err);
        tr_pool_destroy(pool);
        return 1;
    }
    /* -c: KV cache size in tokens, 0 = model default. A small context keeps the cache and
     * the memory guard small when profiling a real model. */
    tr_session *sess = tr_session_create(model, n_ctx, err, sizeof err);
    if (sess == NULL) {
        fprintf(stderr, "error: %s\n", err);
        tr_model_free(model);
        tr_pool_destroy(pool);
        return 1;
    }
    const tr_model_info *info = tr_model_get_info(model);

    /* the prompt: parsed ids, or a deterministic synthetic one of n_prompt_synth
     * ids (no tokenizer needed), for scenarios that want a long prompt. */
    int32_t *prompt = NULL;
    int64_t n_prompt = 0;
    if (tokens_str != NULL) {
        if (parse_tokens(tokens_str, &prompt, &n_prompt) != 0) {
            fprintf(stderr, "generate: invalid --tokens\n");
            tr_session_free(sess);
            tr_model_free(model);
            tr_pool_destroy(pool);
            return 2;
        }
    } else {
        n_prompt = n_prompt_synth;
        prompt = (int32_t *)malloc((size_t)(n_prompt > 0 ? n_prompt : 1) * sizeof(int32_t));
        if (prompt == NULL) {
            fprintf(stderr, "generate: out of memory\n");
            tr_session_free(sess);
            tr_model_free(model);
            tr_pool_destroy(pool);
            return 1;
        }
        synth_prompt(prompt, n_prompt, info->vocab_size);
    }

    tr_prof *prof = tr_session_prof(sess);
    if (do_profile || profile_json_path != NULL) prof->enabled = 1;
    prof->phase = TR_PHASE_PREFILL;

    double t0 = tr_time_sec();
    if (tr_session_eval(sess, prompt, n_prompt) != 0) {
        fprintf(stderr, "generate: prompt evaluation failed (context full or token id out of range)\n");
        free(prompt);
        tr_session_free(sess);
        tr_model_free(model);
        tr_pool_destroy(pool);
        return 1;
    }
    double t1 = tr_time_sec();
    prof->phase = TR_PHASE_DECODE;

    int32_t *generated = (int32_t *)malloc((size_t)n_gen * sizeof(int32_t));
    if (generated == NULL && n_gen > 0) {
        fprintf(stderr, "generate: out of memory\n");
        free(prompt);
        tr_session_free(sess);
        tr_model_free(model);
        tr_pool_destroy(pool);
        return 1;
    }

    int64_t produced = 0;
    int context_full = 0;
    for (int64_t i = 0; i < n_gen; i++) {
        const float *logits = tr_session_logits(sess);
        uint64_t ts = tr_prof_begin(prof);
        int32_t next = argmax_f32(logits, info->vocab_size);
        tr_prof_end(prof, TR_PROF_SAMPLE, ts);
        generated[i] = next;
        produced++;
        if (i + 1 < n_gen) {
            if (tr_session_eval(sess, &next, 1) != 0) {
                fprintf(stderr, "generate: context full while generating\n");
                context_full = 1;
                break;
            }
        }
    }
    double t2 = tr_time_sec();

    printf("tokens:");
    for (int64_t i = 0; i < produced; i++) printf(i ? ",%d" : " %d", generated[i]);
    printf("\n");

    double prompt_secs = t1 - t0, gen_secs = t2 - t1;
    fprintf(stderr, "prompt: %" PRId64 " tokens in %.4fs (%.2f tok/s)\n", n_prompt, prompt_secs,
            prompt_secs > 0 ? (double)n_prompt / prompt_secs : 0.0);
    fprintf(stderr, "generate: %" PRId64 " tokens in %.4fs (%.2f tok/s)\n", produced, gen_secs,
            gen_secs > 0 ? (double)produced / gen_secs : 0.0);

    if (do_profile) tr_prof_print(prof, stderr);
    /* fewer tokens than asked is a failure, not a success with short output (LEZIONI #17) */
    int rc = context_full ? 3 : 0;
    if (profile_json_path != NULL &&
        write_profile_json(profile_json_path, prof, model_path, n_prompt, produced, tr_pool_size(pool)) != 0) {
        fprintf(stderr, "generate: could not write --profile-json '%s'\n", profile_json_path);
        rc = 1;
    }

    free(generated);
    free(prompt);
    tr_session_free(sess);
    tr_model_free(model);
    tr_pool_destroy(pool);
    return rc;
}

/* ---- logits ----------------------------------------------------------------- */

static int cmd_logits(int argc, char **argv) {
    const char *model_path = NULL, *tokens_str = NULL, *out_path = NULL;
    int n_threads = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens_str = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else {
            fprintf(stderr, "logits: unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (model_path == NULL || tokens_str == NULL || out_path == NULL) {
        fprintf(stderr, "usage: trochilus logits -m <file.gguf> --tokens id,id,... --out <file> [-t threads]\n");
        return 2;
    }

    int32_t *tokens = NULL;
    int64_t n_tokens = 0;
    if (parse_tokens(tokens_str, &tokens, &n_tokens) != 0) {
        fprintf(stderr, "logits: invalid --tokens\n");
        return 2;
    }

    char err[256];
    tr_pool *pool = tr_pool_create(n_threads);
    if (pool == NULL) {
        fprintf(stderr, "logits: could not create thread pool\n");
        free(tokens);
        return 1;
    }
    tr_model *model = tr_model_load(model_path, pool, err, sizeof err);
    if (model == NULL) {
        fprintf(stderr, "error: %s\n", err);
        free(tokens);
        tr_pool_destroy(pool);
        return 1;
    }
    tr_session *sess = tr_session_create(model, 0, err, sizeof err);
    if (sess == NULL) {
        fprintf(stderr, "error: %s\n", err);
        free(tokens);
        tr_model_free(model);
        tr_pool_destroy(pool);
        return 1;
    }

    FILE *out = fopen(out_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "logits: could not open '%s' for writing\n", out_path);
        free(tokens);
        tr_session_free(sess);
        tr_model_free(model);
        tr_pool_destroy(pool);
        return 1;
    }

    const tr_model_info *info = tr_model_get_info(model);
    int rc = 0;
    for (int64_t i = 0; i < n_tokens; i++) {
        if (tr_session_eval(sess, &tokens[i], 1) != 0) {
            fprintf(stderr, "logits: evaluation failed at token %" PRId64 "\n", i);
            rc = 1;
            break;
        }
        const float *logits = tr_session_logits(sess);
        if (fwrite(logits, sizeof(float), (size_t)info->vocab_size, out) != (size_t)info->vocab_size) {
            fprintf(stderr, "logits: write failed\n");
            rc = 1;
            break;
        }
    }

    fclose(out);
    free(tokens);
    tr_session_free(sess);
    tr_model_free(model);
    tr_pool_destroy(pool);
    return rc;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "cpu") == 0) return cmd_cpu();
    if (argc >= 3 && strcmp(argv[1], "inspect") == 0) return cmd_inspect(argv[2]);
    if (argc >= 2 && strcmp(argv[1], "generate") == 0) return cmd_generate(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "logits") == 0) return cmd_logits(argc - 2, argv + 2);
    usage();
    return 2;
}
