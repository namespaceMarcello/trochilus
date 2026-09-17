/* main.c — command line entry point.
 *
 *   trochilus cpu                             what this machine offers (cores, instruction sets, RAM)
 *   trochilus inspect <file>                  GGUF metadata and tensor directory
 *   trochilus generate -m <f> --tokens .. -n <n> [-t threads]   greedy-decode n tokens
 *   trochilus logits -m <f> --tokens .. --out <f> [-t threads]  logits after each token
 *   trochilus tokenize -m <f> (-p <text> | -f <file> | --batch <file>)   text to token ids
 *   trochilus run -m <f> (-p <text> | -f <file>) [-n max]      text in, greedy text out
 *   trochilus chat -m <f> [-s system]                           conversation in the model's chat template
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
#include "../tokenizer/chat.h"
#include "../tokenizer/tokenizer.h"

static void usage(void) {
    fprintf(stderr,
            "usage:\n"
            "  trochilus cpu               describe this machine\n"
            "  trochilus inspect <file>    print GGUF metadata and tensors\n"
            "  trochilus generate -m <file.gguf> (--tokens id,id,... | -p <n>) -n <count>\n"
            "                     [-t threads] [-c context] [--profile] [--profile-json <file>]\n"
            "  trochilus logits -m <file.gguf> --tokens id,id,... --out <file> [-t threads]\n"
            "  trochilus tokenize -m <file.gguf> (-p <text> | -f <file> | --batch <file>)\n"
            "                     [--no-parse-special] [--no-add-special] [--pieces | --decode]\n"
            "  trochilus run -m <file.gguf> (-p <text> | -f <file>) [-n <max tokens>]\n"
            "                [-t threads] [-c context] [--no-parse-special]\n"
            "  trochilus chat -m <file.gguf> [-s <system prompt>] [-n <max tokens per reply>]\n"
            "                 [-t threads] [-c context]\n");
}

static void print_bytes(uint64_t n) {
    static const char *const unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
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
    static const char *const names[] = {"u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool",
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

/* The whole file, through the platform layer (UTF-8 paths on Windows too).
 * malloc'd with a NUL after the last byte; NULL on failure with a message in err. */
static char *read_file(const char *path, size_t *len, char *err, size_t err_len) {
    tr_file *f = tr_file_open(path, err, err_len);
    if (f == NULL) return NULL;
    int64_t size = tr_file_size(f);
    char *buf = NULL;
    if (size < 0 || (uint64_t)size >= SIZE_MAX) {
        snprintf(err, err_len, "cannot size '%s'", path);
    } else if ((buf = (char *)malloc((size_t)size + 1)) == NULL) {
        snprintf(err, err_len, "out of memory reading '%s'", path);
    } else if (size > 0 && tr_file_pread(f, buf, (size_t)size, 0) != 0) {
        snprintf(err, err_len, "cannot read '%s'", path);
        free(buf);
        buf = NULL;
    } else {
        buf[size] = '\0';
        *len = (size_t)size;
    }
    tr_file_close(f);
    return buf;
}

static tr_tokenizer *load_tokenizer(const char *path) {
    char err[256];
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    if (g == NULL) {
        fprintf(stderr, "error: %s\n", err);
        return NULL;
    }
    tr_tokenizer *tok = tr_tokenizer_load(g, err, sizeof err);
    tr_gguf_close(g);   /* the tokenizer keeps no pointer into the file */
    if (tok == NULL) fprintf(stderr, "error: %s\n", err);
    return tok;
}

typedef struct {
    FILE *out;
    int first;
} pieces_out;

static void print_unit(void *ctx, const char *bytes, size_t len, int32_t added_id) {
    pieces_out *po = (pieces_out *)ctx;
    if (!po->first) fputc(' ', po->out);
    po->first = 0;
    if (added_id >= 0) {
        fprintf(po->out, "#%d", added_id);
        return;
    }
    for (size_t i = 0; i < len; i++) fprintf(po->out, "%02x", (uint8_t)bytes[i]);
}

enum { OUT_IDS, OUT_PIECES, OUT_DECODE };

/* One output line for one text. 0, or -1 (with a message) if memory ran out or the
 * text is too long. */
static int tokenize_one(const tr_tokenizer *tok, const char *text, size_t len, int flags, int mode) {
    if (mode == OUT_PIECES) {
        pieces_out po = {stdout, 1};
        if (tr_tokenizer_split(tok, text, len, flags, print_unit, &po) != 0) {
            fprintf(stderr, "error: tokenize failed (out of memory or text too long)\n");
            return -1;
        }
        fputc('\n', stdout);
        return 0;
    }
    int32_t *ids;
    size_t n;
    if (tr_tokenizer_encode(tok, text, len, flags, &ids, &n) != 0) {
        fprintf(stderr, "error: tokenize failed (out of memory or text too long)\n");
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        if (mode == OUT_IDS) {
            printf(i ? ",%d" : "%d", ids[i]);
        } else {
            size_t pl;
            const char *p = tr_tokenizer_piece(tok, ids[i], &pl);
            for (size_t k = 0; k < pl; k++) printf("%02x", (uint8_t)p[k]);
        }
    }
    fputc('\n', stdout);
    free(ids);
    return 0;
}

static int cmd_tokenize(int argc, char **argv) {
    const char *model_path = NULL, *text = NULL, *file = NULL, *batch = NULL;
    int flags = TR_TOK_ADD_SPECIAL | TR_TOK_PARSE_SPECIAL, mode = OUT_IDS;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) text = argv[++i];
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) file = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = argv[++i];
        else if (strcmp(argv[i], "--no-parse-special") == 0) flags &= ~TR_TOK_PARSE_SPECIAL;
        else if (strcmp(argv[i], "--no-add-special") == 0) flags &= ~TR_TOK_ADD_SPECIAL;
        else if (strcmp(argv[i], "--pieces") == 0) mode = OUT_PIECES;
        else if (strcmp(argv[i], "--decode") == 0) mode = OUT_DECODE;
        else {
            model_path = NULL;
            break;
        }
    }
    if (model_path == NULL || (text != NULL) + (file != NULL) + (batch != NULL) != 1) {
        fprintf(stderr,
                "usage: trochilus tokenize -m <file.gguf> (-p <text> | -f <file> | --batch <file>)\n"
                "                          [--no-parse-special] [--no-add-special] [--pieces | --decode]\n"
                "  --batch: records of <decimal byte length>\\n<bytes>; one output line per record\n");
        return 2;
    }
    tr_stdout_binary();     /* ids and hex lines end in a bare LF on every OS */
    tr_tokenizer *tok = load_tokenizer(model_path);
    if (tok == NULL) return 1;

    int rc = 0;
    if (text != NULL) {
        if (tokenize_one(tok, text, strlen(text), flags, mode) != 0) rc = 1;
    } else {
        char err[256];
        size_t size = 0;
        char *buf = read_file(file != NULL ? file : batch, &size, err, sizeof err);
        if (buf == NULL) {
            fprintf(stderr, "error: %s\n", err);
            rc = 1;
        } else if (file != NULL) {
            if (tokenize_one(tok, buf, size, flags, mode) != 0) rc = 1;
        } else {
            size_t pos = 0;
            while (pos < size && rc == 0) {
                uint64_t rec = 0;
                size_t digits = 0;
                while (pos < size && buf[pos] >= '0' && buf[pos] <= '9' && digits < 18) {
                    rec = rec * 10 + (uint64_t)(buf[pos++] - '0');
                    digits++;
                }
                if (digits == 0 || pos >= size || buf[pos] != '\n' || rec > size - pos - 1) {
                    fprintf(stderr, "error: malformed batch record at byte %zu\n", pos);
                    rc = 1;
                    break;
                }
                pos++;
                if (tokenize_one(tok, buf + pos, (size_t)rec, flags, mode) != 0) rc = 1;
                pos += (size_t)rec;
            }
        }
        free(buf);
    }
    tr_tokenizer_free(tok);
    return rc;
}

static int cmd_run(int argc, char **argv) {
    const char *model_path = NULL, *text = NULL, *file = NULL;
    int64_t n_max = 256, n_ctx = 0;
    int n_threads = 0, flags = TR_TOK_ADD_SPECIAL | TR_TOK_PARSE_SPECIAL;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) text = argv[++i];
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) file = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_max = atoll(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) n_ctx = atoll(argv[++i]);
        else if (strcmp(argv[i], "--no-parse-special") == 0) flags &= ~TR_TOK_PARSE_SPECIAL;
        else {
            model_path = NULL;
            break;
        }
    }
    if (model_path == NULL || (text != NULL) == (file != NULL) || n_max < 0) {
        fprintf(stderr, "usage: trochilus run -m <file.gguf> (-p <text> | -f <file>) [-n <max tokens>]\n"
                        "                     [-t threads] [-c context] [--no-parse-special]\n");
        return 2;
    }

    char err[256];
    size_t text_len = text != NULL ? strlen(text) : 0;
    char *file_text = NULL;
    if (file != NULL && (file_text = read_file(file, &text_len, err, sizeof err)) == NULL) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    tr_stdout_binary();     /* generated bytes reach a file or pipe unchanged */
    tr_tokenizer *tok = load_tokenizer(model_path);
    if (tok == NULL) {
        free(file_text);
        return 1;
    }
    int32_t *prompt = NULL;
    size_t n_prompt = 0;
    if (tr_tokenizer_encode(tok, file_text != NULL ? file_text : text, text_len, flags, &prompt, &n_prompt) != 0) {
        fprintf(stderr, "run: could not tokenize the prompt (out of memory or too long)\n");
        free(file_text);
        tr_tokenizer_free(tok);
        return 1;
    }
    free(file_text);
    if (n_prompt == 0) {
        fprintf(stderr, "run: the prompt has no tokens\n");
        tr_tokenizer_free(tok);
        return 2;
    }

    int rc = 1;
    tr_pool *pool = tr_pool_create(n_threads);
    tr_model *model = NULL;
    tr_session *sess = NULL;
    if (pool == NULL) {
        fprintf(stderr, "run: could not create thread pool\n");
        goto done;
    }
    if ((model = tr_model_load(model_path, pool, err, sizeof err)) == NULL ||
        (sess = tr_session_create(model, n_ctx, err, sizeof err)) == NULL) {
        fprintf(stderr, "error: %s\n", err);
        goto done;
    }
    const tr_model_info *info = tr_model_get_info(model);

    double t0 = tr_time_sec();
    if (tr_session_eval(sess, prompt, (int64_t)n_prompt) != 0) {
        fprintf(stderr, "run: prompt evaluation failed (context full or token id out of range)\n");
        goto done;
    }
    double t1 = tr_time_sec();
    /* greedy until the end-of-sequence token, any control token (never content), or n_max */
    int64_t produced = 0;
    int context_full = 0;
    int32_t eos = tr_tokenizer_eos(tok);
    while (produced < n_max) {
        int32_t next = argmax_f32(tr_session_logits(sess), info->vocab_size);
        produced++;
        if (next == eos || tr_tokenizer_type(tok, next) == TR_TOKEN_CONTROL) break;
        size_t pl;
        const char *piece = tr_tokenizer_piece(tok, next, &pl);
        fwrite(piece, 1, pl, stdout);
        fflush(stdout);
        if (produced < n_max && tr_session_eval(sess, &next, 1) != 0) {
            context_full = 1;
            break;
        }
    }
    double t2 = tr_time_sec();
    printf("\n");
    if (context_full) fprintf(stderr, "run: context full\n");
    fprintf(stderr, "prompt: %zu tokens in %.4fs (%.2f tok/s)\n", n_prompt, t1 - t0,
            t1 > t0 ? (double)n_prompt / (t1 - t0) : 0.0);
    fprintf(stderr, "generate: %" PRId64 " tokens in %.4fs (%.2f tok/s)\n", produced, t2 - t1,
            t2 > t1 ? (double)produced / (t2 - t1) : 0.0);
    rc = context_full ? 3 : 0;

done:
    free(prompt);
    tr_session_free(sess);
    tr_model_free(model);
    tr_pool_destroy(pool);
    tr_tokenizer_free(tok);
    return rc;
}

/* ---- chat -------------------------------------------------------------------- */

typedef struct {
    tr_chat_msg *m;     /* contents are malloc'd and owned here; roles are literals */
    size_t n, cap;
} conversation;

/* Appends a message and takes ownership of content. 0, or -1 (content freed) if memory ran out. */
static int conv_push(conversation *c, const char *role, char *content, size_t len) {
    if (c->n == c->cap) {
        size_t cap = c->cap > 0 ? c->cap * 2 : 16;
        tr_chat_msg *m = (tr_chat_msg *)realloc(c->m, cap * sizeof(tr_chat_msg));
        if (m == NULL) {
            free(content);
            return -1;
        }
        c->m = m;
        c->cap = cap;
    }
    c->m[c->n].role = role;
    c->m[c->n].content = content;
    c->m[c->n].content_len = len;
    c->n++;
    return 0;
}

/* Drops messages from index `keep` on. */
static void conv_truncate(conversation *c, size_t keep) {
    while (c->n > keep) free((char *)c->m[--c->n].content);
}

static char *copy_bytes(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1);
    if (p != NULL) {
        memcpy(p, s, n);
        p[n] = '\0';
    }
    return p;
}

/* Length of the prefix of s[0..n) that ends on a whole UTF-8 character: a character split
 * across two tokens is printed once both halves are there. */
static size_t utf8_whole_prefix(const char *s, size_t n) {
    for (size_t back = 1; back <= 3 && back <= n; back++) {
        uint8_t c = (uint8_t)s[n - back];
        if ((c & 0xC0) == 0x80) continue;
        size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? n - back : n;
    }
    return n;
}

/* For the oracle: records of --batch are a flag byte ('1': add the generation prompt) then
 * messages separated by 0x1E, each "role 0x1F content". One line per record: the rendered
 * text in hex, or with --ids its token ids (special tokens parsed, none added). */
static int cmd_chat_template(int argc, char **argv) {
    const char *model_path = NULL, *batch = NULL;
    int ids_mode = 0;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = argv[++i];
        else if (strcmp(argv[i], "--ids") == 0) ids_mode = 1;
        else {
            model_path = NULL;
            break;
        }
    }
    if (model_path == NULL || batch == NULL) {
        fprintf(stderr, "usage: trochilus chat-template -m <file.gguf> --batch <file> [--ids]\n");
        return 2;
    }
    tr_stdout_binary();
    char err[256];
    tr_gguf *g = tr_gguf_open(model_path, err, sizeof err);
    if (g == NULL) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    tr_tokenizer *tok = tr_tokenizer_load(g, err, sizeof err);
    const tr_chat_template *tpl = tok != NULL ? tr_chat_template_find(g, err, sizeof err) : NULL;
    tr_gguf_close(g);
    size_t size = 0;
    char *buf = tpl != NULL ? read_file(batch, &size, err, sizeof err) : NULL;
    if (buf == NULL) {
        fprintf(stderr, "error: %s\n", err);
        tr_tokenizer_free(tok);
        return 1;
    }

    int rc = 0;
    size_t pos = 0;
    tr_chat_msg msgs[64];
    char roles[64][32];
    while (pos < size && rc == 0) {
        uint64_t rec = 0;
        size_t digits = 0;
        while (pos < size && buf[pos] >= '0' && buf[pos] <= '9' && digits < 18) {
            rec = rec * 10 + (uint64_t)(buf[pos++] - '0');
            digits++;
        }
        if (digits == 0 || pos >= size || buf[pos] != '\n' || rec == 0 || rec > size - pos - 1) {
            fprintf(stderr, "error: malformed batch record at byte %zu\n", pos);
            rc = 1;
            break;
        }
        const char *r = buf + pos + 1, *end = r + rec;
        pos += 1 + (size_t)rec;
        int gen = r[0] == '1';
        size_t n = 0;
        for (const char *p = r + 1; p < end && rc == 0;) {
            const char *stop = (const char *)memchr(p, 0x1E, (size_t)(end - p));
            if (stop == NULL) stop = end;
            const char *sep = (const char *)memchr(p, 0x1F, (size_t)(stop - p));
            if (sep == NULL || sep - p >= 32 || n == 64) {
                fprintf(stderr, "error: malformed message in batch record\n");
                rc = 1;
                break;
            }
            memcpy(roles[n], p, (size_t)(sep - p));
            roles[n][sep - p] = '\0';
            msgs[n].role = roles[n];
            msgs[n].content = sep + 1;
            msgs[n].content_len = (size_t)(stop - sep - 1);
            n++;
            p = stop + 1;
        }
        char *text;
        size_t len;
        if (rc != 0 || tr_chat_render(tpl, tok, msgs, n, gen, &text, &len) != 0) {
            rc = 1;
            break;
        }
        if (ids_mode) rc = tokenize_one(tok, text, len, TR_TOK_PARSE_SPECIAL, OUT_IDS) != 0;
        else {
            for (size_t k = 0; k < len; k++) printf("%02x", (uint8_t)text[k]);
            fputc('\n', stdout);
        }
        free(text);
    }
    free(buf);
    tr_tokenizer_free(tok);
    return rc;
}

static int cmd_chat(int argc, char **argv) {
    const char *model_path = NULL, *system = NULL;
    int64_t n_max = 1024, n_ctx = 0;
    int n_threads = 0;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) system = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_max = atoll(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) n_ctx = atoll(argv[++i]);
        else {
            model_path = NULL;
            break;
        }
    }
    if (model_path == NULL || n_max <= 0) {
        fprintf(stderr, "usage: trochilus chat -m <file.gguf> [-s <system prompt>] [-n <max tokens per reply>]\n"
                        "                      [-t threads] [-c context]\n");
        return 2;
    }

    char err[256];
    tr_gguf *g = tr_gguf_open(model_path, err, sizeof err);
    if (g == NULL) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    tr_tokenizer *tok = tr_tokenizer_load(g, err, sizeof err);
    const tr_chat_template *tpl = tok != NULL ? tr_chat_template_find(g, err, sizeof err) : NULL;
    tr_gguf_close(g);
    if (tpl == NULL) {
        fprintf(stderr, "error: %s\n", err);
        tr_tokenizer_free(tok);
        return 1;
    }

    int rc = 1;
    conversation conv = {0};
    int32_t *hist = NULL;      /* the tokens in the session's cache */
    size_t n_hist = 0, hist_cap = 0;
    tr_pool *pool = tr_pool_create(n_threads);
    tr_model *model = NULL;
    tr_session *sess = NULL;
    if (pool == NULL) {
        fprintf(stderr, "chat: could not create thread pool\n");
        goto done;
    }
    if ((model = tr_model_load(model_path, pool, err, sizeof err)) == NULL) {
        fprintf(stderr, "error: %s\n", err);
        goto done;
    }
    const tr_model_info *info = tr_model_get_info(model);
    /* Without -c the chat takes the training context, and halves it (down to 512 tokens)
     * while the memory guard refuses the cache: a shorter conversation beats no conversation. */
    int64_t ctx = n_ctx > 0 ? n_ctx : info->n_ctx_train;
    while ((sess = tr_session_create(model, ctx, err, sizeof err)) == NULL && n_ctx <= 0 && ctx / 2 >= 512)
        ctx /= 2;
    if (sess == NULL) {
        fprintf(stderr, "error: %s\n", err);
        goto done;
    }
    if (n_ctx <= 0 && ctx < info->n_ctx_train)
        fprintf(stderr, "(not enough free RAM for %" PRId64 " tokens of conversation: using %" PRId64
                        "; close other programs for longer conversations)\n", info->n_ctx_train, ctx);
    size_t keep = 0;
    if (system != NULL) {
        char *content = copy_bytes(system, strlen(system));
        if (content == NULL || conv_push(&conv, "system", content, strlen(system)) != 0) goto oom;
        keep = 1;
    }
    tr_stdout_binary();
    fprintf(stderr, "trochilus chat (%s). Write a message and press Enter; /reset starts over, "
                    "/exit or an empty input (Ctrl+Z, Ctrl+D) quits.\n", tr_chat_template_name(tpl));
    int32_t eos = tr_tokenizer_eos(tok);

    for (;;) {
        printf("\n> ");
        fflush(stdout);
        size_t line_len;
        char *line = tr_stdin_line(&line_len);
        if (line == NULL) break;
        if (strcmp(line, "/exit") == 0) {
            free(line);
            break;
        }
        if (strcmp(line, "/reset") == 0) {
            free(line);
            conv_truncate(&conv, keep);
            tr_session_rewind(sess, 0);
            n_hist = 0;
            printf("(new conversation)\n");
            continue;
        }
        if (line_len == 0) {
            free(line);
            continue;
        }
        if (conv_push(&conv, "user", line, line_len) != 0) goto oom;

        /* the whole conversation, as transformers would tokenize it; only the part that
         * differs from what the cache holds is evaluated */
        char *text;
        size_t text_len, n_ids;
        int32_t *ids;
        if (tr_chat_render(tpl, tok, conv.m, conv.n, 1, &text, &text_len) != 0) goto oom;
        int enc = tr_tokenizer_encode(tok, text, text_len, TR_TOK_PARSE_SPECIAL, &ids, &n_ids);
        free(text);
        if (enc != 0 || n_ids == 0) goto oom;
        size_t p = 0;
        while (p < n_hist && p < n_ids && hist[p] == ids[p]) p++;
        if (p == n_ids) p--;    /* at least one token to evaluate, for fresh logits */
        tr_session_rewind(sess, (int64_t)p);
        n_hist = p;
        if (n_ids + (size_t)n_max > hist_cap) {     /* room for the prompt and the whole reply */
            int32_t *h = (int32_t *)realloc(hist, (n_ids + (size_t)n_max) * sizeof(int32_t));
            if (h == NULL) {
                free(ids);
                goto oom;
            }
            hist = h;
            hist_cap = n_ids + (size_t)n_max;
        }
        size_t n_new = n_ids - p;
        double t0 = tr_time_sec();
        if (tr_session_eval(sess, ids + p, (int64_t)n_new) != 0) {
            fprintf(stderr, "(context full: the conversation is %zu tokens; /reset to start over)\n", n_ids);
            free(ids);
            conv_truncate(&conv, conv.n - 1);
            continue;
        }
        memcpy(hist, ids, n_ids * sizeof(int32_t));
        n_hist = n_ids;
        free(ids);

        double t1 = tr_time_sec();
        char *reply = NULL;
        size_t reply_len = 0, reply_cap = 0, printed = 0;
        int64_t produced = 0;
        int full = 0;
        for (;;) {
            int32_t next = argmax_f32(tr_session_logits(sess), info->vocab_size);
            if (next == eos || tr_tokenizer_type(tok, next) == TR_TOKEN_CONTROL) break;
            size_t pl;
            const char *piece = tr_tokenizer_piece(tok, next, &pl);
            if (reply_len + pl + 1 > reply_cap) {
                size_t cap = (reply_len + pl + 1) * 2;
                char *r = (char *)realloc(reply, cap);
                if (r == NULL) {
                    free(reply);
                    goto oom;
                }
                reply = r;
                reply_cap = cap;
            }
            memcpy(reply + reply_len, piece, pl);
            reply_len += pl;
            size_t whole = utf8_whole_prefix(reply, reply_len);
            fwrite(reply + printed, 1, whole - printed, stdout);
            fflush(stdout);
            printed = whole;
            if (++produced >= n_max) break;
            if (tr_session_eval(sess, &next, 1) != 0) {
                full = 1;
                break;
            }
            hist[n_hist++] = next;
        }
        double t2 = tr_time_sec();
        fwrite(reply + printed, 1, reply_len - printed, stdout);
        printf("\n");
        fflush(stdout);
        fprintf(stderr, "[%" PRId64 " tokens, %.1f tok/s; %zu new prompt tokens in %.2fs]%s\n", produced,
                t2 > t1 ? (double)produced / (t2 - t1) : 0.0, n_new, t1 - t0,
                full ? " (context full: /reset to start over)" : "");
        char *content = reply != NULL ? reply : copy_bytes("", 0);
        if (content == NULL || conv_push(&conv, "assistant", content, reply_len) != 0) goto oom;
    }
    rc = 0;
    goto done;

oom:
    fprintf(stderr, "chat: out of memory\n");
done:
    conv_truncate(&conv, 0);
    free(conv.m);
    free(hist);
    tr_session_free(sess);
    tr_model_free(model);
    tr_pool_destroy(pool);
    tr_tokenizer_free(tok);
    return rc;
}

static int real_main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "cpu") == 0) return cmd_cpu();
    if (argc >= 3 && strcmp(argv[1], "inspect") == 0) return cmd_inspect(argv[2]);
    if (argc >= 2 && strcmp(argv[1], "generate") == 0) return cmd_generate(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "logits") == 0) return cmd_logits(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "tokenize") == 0) return cmd_tokenize(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "run") == 0) return cmd_run(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "chat") == 0) return cmd_chat(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "chat-template") == 0) return cmd_chat_template(argc - 2, argv + 2);
    usage();
    return 2;
}

#ifdef _WIN32
/* Windows hands main() its arguments in the local code page, so a prompt or path with
 * accents arrives mangled (LEZIONI #13): take them as UTF-16 and convert. Linked with
 * -municode (MinGW) so that wmain is the entry point. */
int wmain(int argc, wchar_t **wargv) {
    char **argv = tr_utf8_argv(argc, wargv);
    if (argv == NULL) {
        fprintf(stderr, "error: could not convert the command line to UTF-8\n");
        return 1;
    }
    tr_console_utf8();
    int rc = real_main(argc, argv);
    tr_utf8_argv_free(argc, argv);
    return rc;
}
#else
int main(int argc, char **argv) {
    return real_main(argc, argv);
}
#endif
