/* resident_probe.c — diagnostic only, never in the engine: how far a draft that reads only the experts
 * the machine already holds in RAM agrees with the exact greedy chain (docs/MEASUREMENTS.md §A draft
 * from the bytes the machine holds, question 94).
 *
 * `make resident-probe` builds build/residentprobe/resident_probe: the engine's core compiled with
 * -DTR_DRAFT_PROBE (the router's hook in src/models/olmoe.c) and this main. One session S of one model.
 * Pass 1 runs the exact greedy chain from the prompt for N + K tokens. Pass 2 rewinds to the prompt's
 * end and, at every generated row g (S holding the exact cache before it), runs each variant's draft
 * chain from the exact token of row g, feeding its own tokens, for up to K tokens or to its first
 * token unlike the exact chain's; then rewinds and runs row g exactly (its argmax must be pass 1's).
 * The resident set of row g comes from a file of N x n_layers x n_expert bytes (1 = resident), written
 * by tools/spec_replay.py --dump-resident from the same run's route trace: the store before row g.
 * The variants (a draft never reads an expert the store lacks):
 *   sub   every layer's router takes the best n_used of its resident experts (tr_model_set_expert_mask);
 *         a layer holding fewer than n_used residents makes the row's sub -1 (not run)
 *   skip  the exact routing, the absent experts' weights 0
 *   norm  skip, the present weights scaled to the sum of all n_used
 *
 *   resident_probe -m <gguf> --tokens <prompt ids> -n N --chain K --resident <file> [-t T]
 *                  [--n-expert 64 --n-used 8]   (the model's: a run where the router's differ fails)
 *                  [--routes-out <file>]        the exact rows' routing, N x n_layers x n_used uint16
 *                                               (tools/spec_replay.py --routes compares it with the trace)
 *
 * Writes one line a row: g, the exact token after row g, the exact top-2 margin there, the resident
 * share of row g's exact routing (units of n_layers x n_used), then each variant's matched length
 * (0..K: how many of the exact chain's next tokens its chain reproduced). tools/spec_replay.py
 * --chain reads it. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/models/model.h"

int tr_draft_probe_attention(int64_t layer, int64_t head, const float *q, const float *keys, const float *values,
                             int64_t n_pos, int64_t head_dim, float scale, float *scores, int64_t score_stride,
                             float *out);
void tr_draft_probe_route(const int64_t *sel_id, float *sel_w, int64_t n_used);

enum { MODE_NONE, MODE_COUNT, MODE_SKIP, MODE_NORM };

static int g_mode;                  /* what the router's hook does in the running evaluation */
static const unsigned char *g_res;  /* this row's resident set, [n_layers][n_expert] */
static int64_t g_n_expert, g_layer; /* the hook's layer: one call a layer in a one-token evaluation */
static int64_t g_n_used, g_bad_used; /* the model's experts a token, and calls that disagreed */
static int64_t g_present, g_named;  /* MODE_COUNT: the exact routing's resident units, and all */
static uint16_t *g_routes;          /* MODE_COUNT: this row's routing, [n_layers][n_used] as the hook saw it */

int tr_draft_probe_attention(int64_t layer, int64_t head, const float *q, const float *keys, const float *values,
                             int64_t n_pos, int64_t head_dim, float scale, float *scores, int64_t score_stride,
                             float *out) {
    (void)layer, (void)head, (void)q, (void)keys, (void)values, (void)n_pos, (void)head_dim, (void)scale;
    (void)scores, (void)score_stride, (void)out;
    return 0;
}

void tr_draft_probe_route(const int64_t *sel_id, float *sel_w, int64_t n_used) {
    const unsigned char *res = g_res != NULL ? g_res + g_layer * g_n_expert : NULL;
    g_layer++;
    if (n_used != g_n_used) g_bad_used++;
    if (g_mode == MODE_NONE || res == NULL) return;
    float all = 0.0f, kept = 0.0f;
    for (int64_t i = 0; i < n_used; i++) {
        all += sel_w[i];
        if (res[sel_id[i]]) kept += sel_w[i];
        g_present += res[sel_id[i]] != 0;
    }
    g_named += n_used;
    if (g_mode == MODE_COUNT) {
        if (g_routes != NULL)
            for (int64_t i = 0; i < n_used; i++) g_routes[(g_layer - 1) * n_used + i] = (uint16_t)sel_id[i];
        return;
    }
    for (int64_t i = 0; i < n_used; i++) {
        if (!res[sel_id[i]]) sel_w[i] = 0.0f;
        else if (g_mode == MODE_NORM && kept > 0.0f) sel_w[i] *= all / kept;
    }
}

static int32_t argmax2(const float *l, int64_t n, float *margin) {
    int64_t a = 0, b = -1;
    for (int64_t i = 1; i < n; i++) {
        if (l[i] > l[a]) {
            b = a;
            a = i;
        } else if (b < 0 || l[i] > l[b]) {
            b = i;
        }
    }
    if (margin != NULL) *margin = l[a] - l[b];
    return (int32_t)a;
}

static int32_t *read_ids(const char *path, int64_t *n) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    int64_t cap = 1 << 16, k = 0;
    int32_t *ids = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    long v;
    while (ids != NULL && k < cap && fscanf(f, "%ld%*[, \r\n]", &v) >= 1) ids[k++] = (int32_t)v;
    fclose(f);
    *n = k;
    return ids;
}

/* One token through S with the hook in `mode`; its argmax (and margin), -1 on failure. */
static int32_t step(tr_session *S, int32_t tok, int mode, int64_t vocab, int64_t n_layers, float *margin) {
    g_mode = mode;
    g_layer = 0;
    int rc = tr_session_eval(S, &tok, 1);
    g_mode = MODE_NONE;
    if (rc != 0 || g_layer != n_layers || g_bad_used != 0) {
        fprintf(stderr, "resident_probe: eval failed or the hook saw %" PRId64 " layers\n", g_layer);
        return -1;
    }
    return argmax2(tr_session_logits(S), vocab, margin);
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *tok_path = NULL, *res_path = NULL, *routes_path = NULL;
    int64_t steps = 64, chain = 8, n_expert = 64, n_used = 8;
    int threads = 8;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "-m") == 0) model_path = argv[i + 1];
        else if (strcmp(argv[i], "--tokens") == 0) tok_path = argv[i + 1];
        else if (strcmp(argv[i], "--resident") == 0) res_path = argv[i + 1];
        else if (strcmp(argv[i], "-n") == 0) steps = atoll(argv[i + 1]);
        else if (strcmp(argv[i], "--chain") == 0) chain = atoll(argv[i + 1]);
        else if (strcmp(argv[i], "-t") == 0) threads = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--n-expert") == 0) n_expert = atoll(argv[i + 1]);
        else if (strcmp(argv[i], "--n-used") == 0) n_used = atoll(argv[i + 1]);
        else if (strcmp(argv[i], "--routes-out") == 0) routes_path = argv[i + 1];
    }
    if (model_path == NULL || tok_path == NULL || res_path == NULL || steps < 1 || chain < 1) {
        fprintf(stderr, "usage: resident_probe -m <gguf> --tokens <prompt ids> -n N --chain K --resident <file>\n");
        return 2;
    }
    int64_t n_prompt = 0;
    int32_t *ids = read_ids(tok_path, &n_prompt);
    if (ids == NULL || n_prompt < 1) {
        fprintf(stderr, "resident_probe: no ids in %s\n", tok_path);
        return 1;
    }
    tr_kernels_init();
    tr_pool *pool = tr_pool_create(threads);
    char err[512];
    tr_model *m = tr_model_load(model_path, pool, err, sizeof err);
    if (m == NULL) {
        fprintf(stderr, "resident_probe: %s\n", err);
        return 1;
    }
    tr_model_set_decode_threads(m, threads);
    const tr_model_info *info = tr_model_get_info(m);
    int64_t vocab = info->vocab_size, n_layers = info->n_layers;
    int64_t row = n_layers * n_expert;
    g_n_expert = n_expert;
    g_n_used = n_used;
    unsigned char *res = (unsigned char *)malloc((size_t)(steps * row));
    unsigned char *off = (unsigned char *)malloc((size_t)row);
    int32_t *gen = (int32_t *)malloc((size_t)(steps + chain + 1) * sizeof(int32_t));
    float *margin = (float *)malloc((size_t)(steps + chain + 1) * sizeof(float));
    uint16_t *routes = (uint16_t *)malloc((size_t)(steps * n_layers * n_used) * sizeof(uint16_t));
    FILE *f = fopen(res_path, "rb");
    if (res == NULL || off == NULL || gen == NULL || margin == NULL || routes == NULL || f == NULL ||
        fread(res, 1, (size_t)(steps * row), f) != (size_t)(steps * row)) {
        fprintf(stderr, "resident_probe: %s must hold %" PRId64 " rows of %" PRId64 " bytes\n", res_path, steps, row);
        return 1;
    }
    fclose(f);
    tr_session *S = tr_session_create(m, n_prompt + steps + chain + 8, 512, err, sizeof err);
    if (S == NULL || tr_session_eval(S, ids, n_prompt) != 0) {
        fprintf(stderr, "resident_probe: %s\n", S == NULL ? err : "prompt eval failed");
        return 1;
    }
    /* pass 1: the exact greedy chain; gen[g] is the input of row g, margin[g] the margin that chose it */
    gen[0] = argmax2(tr_session_logits(S), vocab, &margin[0]);
    for (int64_t g = 0; g < steps + chain; g++) {
        gen[g + 1] = step(S, gen[g], MODE_NONE, vocab, n_layers, &margin[g + 1]);
        if (gen[g + 1] < 0) return 1;
    }
    if (tr_session_rewind(S, n_prompt) != 0) return 1;

    printf("row\texact\tmargin\tresident\tsub\tskip\tnorm\n");
    int64_t sub_skipped = 0;
    for (int64_t g = 0; g < steps; g++) {
        int64_t pos = n_prompt + g;
        const unsigned char *r = res + g * row;
        int sub_ok = 1;
        for (int64_t L = 0; L < n_layers; L++) {
            int64_t have = 0;
            for (int64_t e = 0; e < n_expert; e++) {
                off[L * n_expert + e] = r[L * n_expert + e] == 0;
                have += r[L * n_expert + e] != 0;
            }
            if (have < n_used) sub_ok = 0;
        }
        int64_t got[3];
        for (int v = 0; v < 3; v++) {
            got[v] = 0;
            if (v == 0 && (!sub_ok || tr_model_set_expert_mask(m, off) != 0)) {
                got[v] = -1;
                sub_skipped++;
                continue;
            }
            int mode = v == 0 ? MODE_NONE : v == 1 ? MODE_SKIP : MODE_NORM;
            g_res = r;
            int32_t t = gen[g];
            for (int64_t j = 0; j < chain; j++) {
                int32_t d = step(S, t, mode, vocab, n_layers, NULL);
                if (d < 0) return 1;
                if (d != gen[g + 1 + j]) break;
                got[v]++;
                t = d;
            }
            g_res = NULL;
            if (v == 0) tr_model_set_expert_mask(m, NULL);
            if (tr_session_rewind(S, pos) != 0) return 1;
        }
        /* row g exactly, counting its routing's resident units: the cache stays the exact one */
        g_res = r;
        g_routes = routes + g * n_layers * n_used;
        g_present = g_named = 0;
        int32_t e = step(S, gen[g], MODE_COUNT, vocab, n_layers, NULL);
        g_res = NULL;
        g_routes = NULL;
        if (e != gen[g + 1]) {
            fprintf(stderr, "resident_probe: row %" PRId64 " gave %d, pass 1 gave %d\n", g, (int)e, (int)gen[g + 1]);
            return 1;
        }
        printf("%" PRId64 "\t%d\t%.4f\t%.4f\t%" PRId64 "\t%" PRId64 "\t%" PRId64 "\n", g, (int)gen[g + 1],
               (double)margin[g + 1], (double)g_present / (double)g_named, got[0], got[1], got[2]);
        fflush(stdout);
    }
    fprintf(stderr, "resident_probe: %" PRId64 " rows, sub not run on %" PRId64 "\n", steps, sub_skipped);
    if (routes_path != NULL) {
        FILE *o = fopen(routes_path, "wb");
        size_t n = (size_t)(steps * n_layers * n_used);
        if (o == NULL || fwrite(routes, sizeof(uint16_t), n, o) != n || fclose(o) != 0) {
            fprintf(stderr, "resident_probe: cannot write %s\n", routes_path);
            return 1;
        }
    }
    free(routes);
    tr_session_free(S);
    tr_model_free(m);
    tr_pool_destroy(pool);
    free(res);
    free(off);
    free(gen);
    free(margin);
    free(ids);
    return 0;
}
