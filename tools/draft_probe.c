/* draft_probe.c — diagnostic only, never in the engine: how often a draft read from the exact
 * model's own bits picks the exact greedy token (docs/MEASUREMENTS.md §The engine read as entangled
 * pairs, pair 1: a coarse model inside the exact one).
 *
 * `make draft-probe` builds build/draftprobe/draft_probe: the engine's core compiled with
 * -DTR_DRAFT_PROBE (the hooks in src/models/olmoe.c and model.c) and this main. Two sessions of one
 * model: E runs the exact model; D holds E's cache cut to its high 16 bits (a bf16, read at the
 * midpoint of what it stands for). At every position D runs the same token once per draft variant,
 * rewound after each, then takes E's row for that position, cut: every draft starts from the exact
 * state, as a draft that reads the high planes of the exact cache would. A variant is:
 *   - the weights: exact, or every Q8_0 code cut to its high nibble and read as 16 * hi + 8 (the
 *     scale kept): the high plane of a Q8_0 block, 18 of its 34 bytes;
 *   - the routing: the top k of the used experts, rescaled to the used experts' sum (0: all);
 *   - the attention of a decode token: the first SINKS positions and the last `window` (0: all),
 *     a selection (sel, succ), or every position with the history from a copy of kvbits bits.
 *
 *   draft_probe -m <q8_0.gguf> --tokens <file of comma-separated ids> --ctx C -n N [-t T] [--text 1]
 *
 * Starts from the first C tokens of the file (C - 1 evaluated as a prompt), then follows E's greedy
 * tokens for N steps, or with --text 1 the file's own tokens (a greedy continuation can fall into a
 * loop that inflates every agreement: question 53's code trajectory). Writes to stdout one line a step: the position, E's token, E's top-2 margin
 * (a logit difference: nats), each variant's token, then each variant's own top-2 margin (what a
 * draft knows of its own doubt). tools/draft_probe_report.py reads it. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"
#include "../src/kv/kv.h"
#include "../src/models/model.h"

void *tr_draft_probe_session(tr_session *s);
tr_kv *tr_draft_probe_kv(void *session);
void tr_draft_probe_route(float *sel_w, int64_t n_used);
int tr_draft_probe_attention(int64_t layer, int64_t head, const float *q, const float *keys, const float *values,
                             int64_t n_pos, int64_t head_dim, float scale, float *scores, int64_t score_stride,
                             float *out);

typedef struct {
    const char *name;
    int cut;    /* Q8_0 codes cut to the high nibble */
    int topk;   /* 0: every used expert */
    int window; /* 0: every position */
    int sel;    /* > 0: the SINKS, the last RECENT and the `sel` positions E's previous token attended to most */
    int succ;   /* with sel: each selected position's successor too (a copying head moves one on) */
    int kvbits; /* > 0: the history's K and V read from a copy of kvbits bits, the last RECENT kept */
} variant;

/* A draft whose cache bytes do not grow with the context reads a fixed set of positions. The
 * window (win256) keeps the last ones; sel192 and nx96 keep, besides SINKS and the last RECENT,
 * those the exact session's previous token weighted most, head by head (its scores q.k, known to
 * the exact pass that verified it): 192 of them, or 96 and the position after each. 260 at most,
 * as win256. kv4 and kv8 read every position, but the history from a copy written once as each
 * row leaves the last RECENT (question 75, as KIVI): K per channel over groups of KV_GROUP
 * positions, V per position, each group's min and step, 4 or 8 bits a value. planes_top4 {1, 4},
 * planes_win256 {1, 0, 256} and planes_sel192 {1, 0, 0, 192} were measured on 2026-09-26
 * (docs/MEASUREMENTS.md §The draft's two levers) and are left out of the runs. */
static const variant VARIANTS[] = {
    {"planes", 1, 0, 0, 0, 0, 0},
    {"planes_nx96", 1, 0, 0, 96, 1, 0},
    {"planes_kv4", 1, 0, 0, 0, 0, 4},
    {"planes_kv8", 1, 0, 0, 0, 0, 8},
};
enum {
    N_VARIANTS = sizeof VARIANTS / sizeof VARIANTS[0],
    SINKS = 4,
    RECENT = 64,
    KV_GROUP = 32,
    SEL_MAX = 192,
    MAX_LAYERS = 64,
    MAX_HEADS = 64,
    ROW_MAX = 1 << 16,
    Q8_BLOCK = 34
};

static int g_topk, g_window; /* the running variant's routing and attention; read by the hooks */
static int g_sel, g_succ;     /* the running variant's selection */
static int g_kvbits;          /* the running variant's copy of the history: bits a value, 0 none */
static int g_record;          /* 1 while E runs a decode token: its selection is written */

/* E's selection at position q, head by head, best first: slot q & 1 (D at q + 1 reads it). */
static int32_t g_selected[2][MAX_LAYERS][MAX_HEADS][SEL_MAX];
static int g_n_selected[2][MAX_LAYERS][MAX_HEADS];

typedef struct {
    float s;
    int32_t i;
} scored;

static int by_score(const void *a, const void *b) {
    float x = ((const scored *)a)->s, y = ((const scored *)b)->s;
    return x < y ? 1 : x > y ? -1 : 0;
}

static _Thread_local scored *t_scored;
static _Thread_local unsigned char *t_mark;
static _Thread_local int64_t t_scored_cap;

static void grow_scratch(int64_t n) {
    if (t_scored_cap >= n) return;
    free(t_scored);
    free(t_mark);
    t_scored = (scored *)malloc((size_t)n * sizeof(scored));
    t_mark = (unsigned char *)malloc((size_t)n);
    if (t_scored == NULL || t_mark == NULL) abort();
    t_scored_cap = n;
}

/* E at position n_pos - 1: the SEL_MAX candidates with the largest q.k among those D's next token
 * would not read anyway (past the SINKS, before its last RECENT). */
static void record_selection(int64_t layer, int64_t head, const float *q, const float *keys, int64_t n_pos,
                             int64_t head_dim) {
    if (layer >= MAX_LAYERS || head >= MAX_HEADS) abort();
    int slot = (int)((n_pos - 1) & 1);
    int64_t end = n_pos + 1 - RECENT, n = 0;
    grow_scratch(n_pos + 1);
    for (int64_t t = SINKS; t < end; t++) {
        const float *k = keys + t * head_dim;
        float s = 0.0f;
        for (int64_t d = 0; d < head_dim; d++) s += q[d] * k[d];
        t_scored[n].s = s;
        t_scored[n].i = (int32_t)t;
        n++;
    }
    qsort(t_scored, (size_t)n, sizeof(scored), by_score);
    int m = n < SEL_MAX ? (int)n : SEL_MAX;
    for (int j = 0; j < m; j++) g_selected[slot][layer][head][j] = t_scored[j].i;
    g_n_selected[slot][layer][head] = m;
}

void tr_draft_probe_route(float *sel_w, int64_t n_used) {
    if (g_topk <= 0 || g_topk >= n_used) return;
    float all = 0.0f, kept = 0.0f;
    for (int64_t i = 0; i < n_used; i++) all += sel_w[i];
    for (int64_t i = 0; i < g_topk; i++) kept += sel_w[i];
    for (int64_t i = 0; i < g_topk; i++) sel_w[i] *= all / kept;
    for (int64_t i = g_topk; i < n_used; i++) sel_w[i] = 0.0f;
}

static _Thread_local float *t_keys, *t_values; /* a head's window, gathered */
static _Thread_local int64_t t_cap;

/* D at position n_pos - 1: the SINKS, E's selection at n_pos - 2 (and successors), the last RECENT. */
static int selected_attention(int64_t layer, int64_t head, const float *q, const float *keys, const float *values,
                              int64_t n_pos, int64_t head_dim, float scale, float *scores, int64_t score_stride,
                              float *out) {
    int slot = (int)((n_pos - 2) & 1);
    int m = g_n_selected[slot][layer][head];
    if (m > g_sel) m = g_sel;
    grow_scratch(n_pos + 1);
    memset(t_mark, 0, (size_t)n_pos);
    for (int64_t t = 0; t < SINKS && t < n_pos; t++) t_mark[t] = 1;
    for (int64_t t = n_pos - RECENT < 0 ? 0 : n_pos - RECENT; t < n_pos; t++) t_mark[t] = 1;
    for (int j = 0; j < m; j++) {
        int32_t t = g_selected[slot][layer][head][j];
        if (t < n_pos) t_mark[t] = 1;
        if (g_succ && t + 1 < n_pos) t_mark[t + 1] = 1;
    }
    int64_t n = 0;
    for (int64_t t = 0; t < n_pos; t++) n += t_mark[t];
    if (t_cap < n * head_dim) {
        free(t_keys);
        free(t_values);
        t_keys = (float *)malloc((size_t)(n * head_dim) * sizeof(float));
        t_values = (float *)malloc((size_t)(n * head_dim) * sizeof(float));
        if (t_keys == NULL || t_values == NULL) abort();
        t_cap = n * head_dim;
    }
    size_t row = (size_t)head_dim * sizeof(float);
    int64_t j = 0;
    for (int64_t t = 0; t < n_pos; t++)
        if (t_mark[t]) {
            memcpy(t_keys + j * head_dim, keys + t * head_dim, row);
            memcpy(t_values + j * head_dim, values + t * head_dim, row);
            j++;
        }
    tr_attention_group(q, head_dim, t_keys, t_values, 1, n, head_dim, scale, scores, score_stride, out, head_dim);
    return 1;
}

/* x[0], x[stride], ... through a copy of `bits` bits: the min and the step (max - min) / (2^bits - 1)
 * of its first n_seen values, the first n_cut replaced by their nearest level */
static void quantize_group(float *x, int64_t n_seen, int64_t n_cut, int64_t stride, int bits) {
    float mn = x[0], mx = x[0];
    for (int64_t i = 1; i < n_seen; i++) {
        float v = x[i * stride];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    float step = (mx - mn) / (float)((1 << bits) - 1);
    if (!(step > 0.0f)) return; /* a flat group is its min: nothing to cut */
    for (int64_t i = 0; i < n_cut; i++) {
        float level = (x[i * stride] - mn) / step;
        x[i * stride] = mn + (float)(int)(level + 0.5f) * step;
    }
}

/* D at position n_pos - 1, the history from its copy of g_kvbits bits: the rows before the last
 * RECENT through the copy (K per channel over groups of KV_GROUP positions, V per position), the
 * last RECENT as they are. The copy is rebuilt here at every token; the bytes are the same as one
 * written once a row, since a row's copy never changes after it is written. */
static int quantized_attention(const float *q, const float *keys, const float *values, int64_t n_pos,
                               int64_t head_dim, float scale, float *scores, int64_t score_stride, float *out) {
    if (t_cap < n_pos * head_dim) {
        free(t_keys);
        free(t_values);
        t_keys = (float *)malloc((size_t)(n_pos * head_dim) * sizeof(float));
        t_values = (float *)malloc((size_t)(n_pos * head_dim) * sizeof(float));
        if (t_keys == NULL || t_values == NULL) abort();
        t_cap = n_pos * head_dim;
    }
    memcpy(t_keys, keys, (size_t)(n_pos * head_dim) * sizeof(float));
    memcpy(t_values, values, (size_t)(n_pos * head_dim) * sizeof(float));
    int64_t old = n_pos - RECENT;
    for (int64_t g = 0; g < old; g += KV_GROUP) {
        int64_t seen = n_pos - g < KV_GROUP ? n_pos - g : KV_GROUP;
        int64_t cut = old - g < KV_GROUP ? old - g : KV_GROUP;
        for (int64_t d = 0; d < head_dim; d++) quantize_group(t_keys + g * head_dim + d, seen, cut, head_dim, g_kvbits);
    }
    for (int64_t t = 0; t < old; t++) quantize_group(t_values + t * head_dim, head_dim, head_dim, 1, g_kvbits);
    tr_attention_group(q, head_dim, t_keys, t_values, 1, n_pos, head_dim, scale, scores, score_stride, out, head_dim);
    return 1;
}

int tr_draft_probe_attention(int64_t layer, int64_t head, const float *q, const float *keys, const float *values,
                             int64_t n_pos, int64_t head_dim, float scale, float *scores, int64_t score_stride,
                             float *out) {
    if (g_record) {
        record_selection(layer, head, q, keys, n_pos, head_dim);
        return 0; /* E's own attention stays the engine's */
    }
    if (g_kvbits > 0) {
        if (n_pos <= RECENT) return 0;
        return quantized_attention(q, keys, values, n_pos, head_dim, scale, scores, score_stride, out);
    }
    if (g_sel > 0) {
        if (n_pos <= SINKS + RECENT + (g_succ ? 2 : 1) * g_sel) return 0;
        return selected_attention(layer, head, q, keys, values, n_pos, head_dim, scale, scores, score_stride, out);
    }
    int64_t w = g_window;
    if (w <= 0 || n_pos <= SINKS + w) return 0;
    int64_t n = SINKS + w;
    if (t_cap < n * head_dim) {
        free(t_keys);
        free(t_values);
        t_keys = (float *)malloc((size_t)(n * head_dim) * sizeof(float));
        t_values = (float *)malloc((size_t)(n * head_dim) * sizeof(float));
        if (t_keys == NULL || t_values == NULL) abort();
        t_cap = n * head_dim;
    }
    size_t sink = (size_t)(SINKS * head_dim) * sizeof(float), tail = (size_t)(w * head_dim) * sizeof(float);
    memcpy(t_keys, keys, sink);
    memcpy(t_keys + SINKS * head_dim, keys + (n_pos - w) * head_dim, tail);
    memcpy(t_values, values, sink);
    memcpy(t_values + SINKS * head_dim, values + (n_pos - w) * head_dim, tail);
    tr_attention_group(q, head_dim, t_keys, t_values, 1, n, head_dim, scale, scores, score_stride, out, head_dim);
    return 1;
}

/* ---- the high plane of Q8_0: the tier's own kernels on rows cut on the fly ---- */

static const tr_kernels *g_exact;
static tr_kernels g_draft;
static _Thread_local unsigned char t_row[2][ROW_MAX];

static const void *cut_row(const void *row, int64_t n, int which) {
    const unsigned char *s = (const unsigned char *)row;
    unsigned char *d = t_row[which];
    for (int64_t b = 0; b < n / 32; b++, s += Q8_BLOCK, d += Q8_BLOCK) {
        d[0] = s[0];
        d[1] = s[1];
        for (int j = 2; j < Q8_BLOCK; j++) d[j] = (unsigned char)((s[j] & 0xF0) | 0x08);
    }
    return t_row[which];
}

static float d_dot_row(const void *row, const float *x, int64_t n) {
    return g_exact->dot_row[TR_TYPE_Q8_0](cut_row(row, n, 0), x, n);
}
static void d_dot_row_x4(const void *row, const float *x, int64_t stride, int64_t n, float *out) {
    g_exact->dot_row_x4[TR_TYPE_Q8_0](cut_row(row, n, 0), x, stride, n, out);
}
static void d_dot_row2(const void *r0, const void *r1, const float *x, int64_t n, float *out) {
    g_exact->dot_row2[TR_TYPE_Q8_0](cut_row(r0, n, 0), cut_row(r1, n, 1), x, n, out);
}
static void d_dot_row2_x4(const void *r0, const void *r1, const float *x, int64_t stride, int64_t n, float *out) {
    g_exact->dot_row2_x4[TR_TYPE_Q8_0](cut_row(r0, n, 0), cut_row(r1, n, 1), x, stride, n, out);
}
static void d_dot_row2_x8(const void *r0, const void *r1, const float *x, int64_t stride, int64_t n, float *out) {
    g_exact->dot_row2_x8[TR_TYPE_Q8_0](cut_row(r0, n, 0), cut_row(r1, n, 1), x, stride, n, out);
}
static void d_dequant(const void *row, float *out, int64_t n) {
    g_exact->dequant_row[TR_TYPE_Q8_0](cut_row(row, n, 0), out, n);
}

static void build_draft_table(void) {
    g_exact = tr_kernels_get();
    g_draft = *g_exact;
    const int q = TR_TYPE_Q8_0;
    if (g_exact->dot_row[q] != NULL) g_draft.dot_row[q] = d_dot_row;
    if (g_exact->dot_row_x4[q] != NULL) g_draft.dot_row_x4[q] = d_dot_row_x4;
    if (g_exact->dot_row2[q] != NULL) g_draft.dot_row2[q] = d_dot_row2;
    if (g_exact->dot_row2_x4[q] != NULL) g_draft.dot_row2_x4[q] = d_dot_row2_x4;
    if (g_exact->dot_row2_x8[q] != NULL) g_draft.dot_row2_x8[q] = d_dot_row2_x8;
    if (g_exact->dequant_row[q] != NULL) g_draft.dequant_row[q] = d_dequant;
}

/* ---- the cache: D's positions [p0, p1) = E's, cut to the high 16 bits, read at the midpoint ---- */

static void cut_kv(tr_kv *d, const tr_kv *e, int64_t p0, int64_t p1) {
    for (int64_t L = 0; L < e->n_layers; L++)
        for (int64_t h = 0; h < e->n_head_kv; h++)
            for (int kv = 0; kv < 2; kv++) {
                const float *src = kv ? tr_kv_values(e, L, h) : tr_kv_keys(e, L, h);
                float *dst = (float *)(kv ? tr_kv_values(d, L, h) : tr_kv_keys(d, L, h));
                for (int64_t i = p0 * e->head_dim; i < p1 * e->head_dim; i++) {
                    uint32_t u;
                    memcpy(&u, &src[i], 4);
                    u = (u & 0xFFFF0000u) | 0x8000u;
                    memcpy(&dst[i], &u, 4);
                }
            }
}

static int32_t argmax2(const float *l, int64_t n, float *margin, int32_t *second) {
    int64_t a = 0, b = -1;
    for (int64_t i = 1; i < n; i++) {
        if (l[i] > l[a]) {
            b = a;
            a = i;
        } else if (b < 0 || l[i] > l[b]) {
            b = i;
        }
    }
    *margin = l[a] - l[b];
    if (second != NULL) *second = (int32_t)b;
    return (int32_t)a;
}

static int32_t *read_ids(const char *path, int64_t *n) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    int64_t cap = 1 << 16, k = 0;
    int32_t *ids = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    long v;
    while (ids != NULL && fscanf(f, "%ld%*[, \r\n]", &v) >= 1) {
        if (k == cap) {
            cap *= 2;
            ids = (int32_t *)realloc(ids, (size_t)cap * sizeof(int32_t));
            if (ids == NULL) break;
        }
        ids[k++] = (int32_t)v;
    }
    fclose(f);
    *n = k;
    return ids;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *tok_path = NULL;
    int64_t ctx = 512, steps = 64;
    int threads = 8, text = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "-m") == 0) model_path = argv[i + 1];
        else if (strcmp(argv[i], "--tokens") == 0) tok_path = argv[i + 1];
        else if (strcmp(argv[i], "--ctx") == 0) ctx = atoll(argv[i + 1]);
        else if (strcmp(argv[i], "-n") == 0) steps = atoll(argv[i + 1]);
        else if (strcmp(argv[i], "-t") == 0) threads = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--text") == 0) text = atoi(argv[i + 1]);
    }
    if (model_path == NULL || tok_path == NULL || ctx < 2 || steps < 1) {
        fprintf(stderr, "usage: draft_probe -m <q8_0.gguf> --tokens <ids file> --ctx C -n N [-t T]\n");
        return 2;
    }
    int64_t n_ids = 0;
    int32_t *ids = read_ids(tok_path, &n_ids);
    if (ids == NULL || n_ids < ctx + (text ? steps : 0)) {
        fprintf(stderr, "draft_probe: %s has %" PRId64 " ids, need %" PRId64 "\n", tok_path, n_ids,
                ctx + (text ? steps : 0));
        return 1;
    }
    tr_kernels_init();
    build_draft_table();
    tr_pool *pool = tr_pool_create(threads);
    char err[512];
    tr_model *m = tr_model_load(model_path, pool, err, sizeof err);
    if (m == NULL) {
        fprintf(stderr, "draft_probe: %s\n", err);
        return 1;
    }
    tr_model_set_decode_threads(m, threads);
    int64_t vocab = tr_model_get_info(m)->vocab_size;
    tr_session *E = tr_session_create(m, ctx + steps + 8, 512, err, sizeof err);
    tr_session *D = E != NULL ? tr_session_create(m, ctx + steps + 8, 512, err, sizeof err) : NULL;
    if (D == NULL) {
        fprintf(stderr, "draft_probe: %s\n", err);
        return 1;
    }
    /* E's last prompt token alone, as a decode token: its selection exists for D's first step */
    if (tr_session_eval(E, ids, ctx - 2) != 0 || tr_session_eval(D, ids, ctx - 1) != 0) {
        fprintf(stderr, "draft_probe: prompt eval failed\n");
        return 1;
    }
    g_record = 1;
    if (tr_session_eval(E, ids + ctx - 2, 1) != 0) return 1;
    g_record = 0;
    tr_kv *kv_e = tr_draft_probe_kv(tr_draft_probe_session(E));
    tr_kv *kv_d = tr_draft_probe_kv(tr_draft_probe_session(D));
    cut_kv(kv_d, kv_e, 0, ctx - 1);

    printf("pos\texact\tmargin");
    for (int v = 0; v < N_VARIANTS; v++) printf("\t%s", VARIANTS[v].name);
    for (int v = 0; v < N_VARIANTS; v++) printf("\t%s_m", VARIANTS[v].name);
    for (int v = 0; v < N_VARIANTS; v++) printf("\t%s_2", VARIANTS[v].name);
    printf("\n");
    int32_t tok = ids[ctx - 1];
    for (int64_t step = 0; step < steps; step++) {
        int64_t p = ctx - 1 + step;
        float margin, own[N_VARIANTS];
        g_record = 1;
        if (tr_session_eval(E, &tok, 1) != 0) return 1;
        g_record = 0;
        int32_t g = argmax2(tr_session_logits(E), vocab, &margin, NULL);
        int32_t got[N_VARIANTS], second[N_VARIANTS];
        for (int v = 0; v < N_VARIANTS; v++) {
            tr_kernels_set_active(VARIANTS[v].cut ? &g_draft : g_exact);
            g_topk = VARIANTS[v].topk;
            g_window = VARIANTS[v].window;
            g_sel = VARIANTS[v].sel;
            g_succ = VARIANTS[v].succ;
            g_kvbits = VARIANTS[v].kvbits;
            if (tr_session_eval(D, &tok, 1) != 0) return 1;
            got[v] = argmax2(tr_session_logits(D), vocab, &own[v], &second[v]);
            if (v + 1 < N_VARIANTS && tr_session_rewind(D, p) != 0) return 1;
        }
        tr_kernels_set_active(g_exact);
        g_topk = g_window = g_sel = g_succ = g_kvbits = 0;
        cut_kv(kv_d, kv_e, p, p + 1); /* the last variant's row p replaced by E's, cut */
        printf("%" PRId64 "\t%d\t%.4f", p, (int)g, (double)margin);
        for (int v = 0; v < N_VARIANTS; v++) printf("\t%d", (int)got[v]);
        for (int v = 0; v < N_VARIANTS; v++) printf("\t%.4f", (double)own[v]);
        for (int v = 0; v < N_VARIANTS; v++) printf("\t%d", (int)second[v]);
        printf("\n");
        fflush(stdout);
        tok = text ? ids[ctx + step] : g;
    }
    tr_session_free(D);
    tr_session_free(E);
    tr_model_free(m);
    tr_pool_destroy(pool);
    free(ids);
    return 0;
}
