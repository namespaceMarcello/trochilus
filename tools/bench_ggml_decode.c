/* bench_ggml_decode.c -- the decode's matmul (one token, bound by memory) raced kernel against
 * kernel: llama.cpp's ggml (MUL_MAT, MUL_MAT_ID; ref/llama.cpp commit b49650a, built in
 * ref/llama.cpp/build-trochilus/) against trochilus's tr_matmul and tr_matmul_grouped, in one
 * binary, on the same weight bytes, the same cores and the same token (docs/ORIGINS.md §Every
 * piece, row 2; docs/MEASUREMENTS.md question 65). A one-off comparison tool like
 * tools/bench_ggml.c: not part of the engine, not run by `make check`.
 *
 * A decode token reads ~1.2 GB of weights, far more than the 2 x 32 MiB of L3, so every case lays
 * its weight out as copies filling ~1 GiB and a timed pass runs one matmul per copy (ggml: one
 * graph with a node per copy, as a model's graph has a node per weight; trochilus: one tr_matmul
 * per copy): every copy comes from RAM. GB/s = the weight bytes of a pass / its time; the token's
 * vector and the outputs are a few KiB and are not counted. Beside them, "read" is a plain read of
 * the same bytes by trochilus's pool (tests/bench_mem.c's loop): the ceiling of that case.
 *
 * Cases (tests/bench_peak.c's shapes, OLMoE-1B-7B's, one token):
 *   1024x2048   an expert's gate/up, as a dense matrix
 *   2048x1024   an expert's down
 *   2048x2048   an attention projection
 *   50304x2048  the output head
 *   experts     64 experts of 1024x2048, 8 distinct ones chosen at random per node: MUL_MAT_ID,
 *               and tr_matmul_grouped as olmoe.c calls it (one gathered row per chosen expert)
 * Weights Q8_0 or Q4_K; "q4_k-repack" puts ggml's Q4_K through its "repack" buffer type (what
 * llama.cpp uses on AVX2/AVX-512; trochilus reads the plain Q4_K bytes as always).
 *
 * Fairness: the same thread counts, on the same cores (ggml's pool pinned with strict_cpu to the
 * logical processors of trochilus's first T slots), alternating: R rounds of (trochilus, read, ggml),
 * each running one untimed pass and P timed ones, each pool created before its turn and destroyed
 * after (two live pools would spin against each other). Median of the R x P passes, and the spread
 * (max - min) / median: a difference smaller than the spread is not a difference.
 * The weights are one quantized block of 1024 rows of random floats repeated over the rows and
 * the copies: equal bytes change nothing for the memory. Before timing, the first node of ggml is
 * checked against trochilus's: both must compute the same product, within the error of ggml's
 * int8 activations (a relative error above 2% stops the run).
 *
 * Usage: bench_ggml_decode q8_0|q4_k|q4_k-repack [--rounds R]   (one type a run: under 60 s)
 * Build and run: tools/bench_ggml_decode.sh (inside trochilus-dev).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"

#define GIB ((size_t)1 << 30)
#define UNIT_ROWS 1024
#define N_EXPERT 64
#define TOP_K 8
#define PASSES 2       /* timed passes per turn */
#define MAX_ROUNDS 16
#define MAX_SAMPLES (MAX_ROUNDS * PASSES)

static int n_rounds = 5;

static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;

static uint64_t rng_next(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng * 0x2545F4914F6CDD1DULL;
}

static float rand_uniform(void) { /* [-1, 1) */
    return (float)((rng_next() >> 40) & 0xFFFF) / 32768.0f - 1.0f;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ---- the plain read: the ceiling of every case ------------------------------------ */

typedef struct {
    volatile uint64_t sum;
    char pad[56];
} sink_t;
static sink_t sinks[64];

typedef struct {
    const unsigned char *base;
    size_t share;
} read_ctx;

static void read_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const read_ctx *c = (const read_ctx *)ctx_;
    uint64_t a = 0, b = 0, cc = 0, d = 0;
    for (int64_t i = begin; i < end; i++) {
        const unsigned char *p = c->base + (size_t)i * c->share;
        for (size_t k = 0; k < c->share; k += 64) {
            uint64_t w[8];
            memcpy(w, p + k, 64);
            a += w[0] + w[4];
            b += w[1] + w[5];
            cc += w[2] + w[6];
            d += w[3] + w[7];
        }
    }
    sinks[worker & 63].sum += a ^ b ^ cc ^ d;
}

/* ---- one case ------------------------------------------------------------------- */

typedef struct {
    const char *name;
    int64_t rows, cols;
    int moe;
} shape;

typedef struct {
    const shape *sh;
    enum ggml_type gtype;
    tr_type ttype;
    int repack;
    size_t row_bytes, mat_bytes;   /* one matrix (one expert for the experts case) */
    int n_copies;                  /* weight tensors (of N_EXPERT matrices for the experts case) */
    int n_nodes;                   /* matmuls per pass */
    unsigned char *arena;          /* trochilus's copies, n_copies * (moe ? N_EXPERT : 1) matrices */
    float *x, *xg, *y;             /* the token; gathered rows (experts); trochilus's output */
    tr_mat *mats;                  /* [n_nodes * N_EXPERT] (experts) or [n_nodes] (dense) */
    int64_t *offsets;              /* [n_nodes * (N_EXPERT + 1)] (experts) */
    int32_t *ids;                  /* [n_nodes * TOP_K] (experts) */
    struct ggml_context *ctx_w, *ctx_g;
    struct ggml_backend_buffer *buf_w, *buf_g;
    struct ggml_cgraph *gf;
    struct ggml_tensor **results;
} bench;

static ggml_backend_buffer_type_t find_repack_buft(void) {
    typedef ggml_backend_buffer_type_t *(*get_extra_bufts_t)(ggml_backend_dev_t);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) return NULL;
    get_extra_bufts_t fn = (get_extra_bufts_t)ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(),
                                                                                "ggml_backend_dev_get_extra_bufts");
    if (!fn) return NULL;
    for (ggml_backend_buffer_type_t *b = fn(cpu_dev); b && *b; b++) {
        const char *name = ggml_backend_buft_name(*b);
        if (name && strstr(name, "REPACK")) return *b;
    }
    return NULL;
}

static void die(const char *what) {
    fprintf(stderr, "bench_ggml_decode: %s\n", what);
    exit(1);
}

/* one matrix of sh's shape: UNIT_ROWS quantized rows of random floats, repeated */
static unsigned char *make_matrix(const bench *b) {
    int64_t cols = b->sh->cols, rows = b->sh->rows;
    float *f = malloc((size_t)UNIT_ROWS * (size_t)cols * sizeof(float));
    unsigned char *unit = malloc((size_t)UNIT_ROWS * b->row_bytes);
    unsigned char *m = malloc(b->mat_bytes);
    if (!f || !unit || !m) die("out of memory (matrix)");
    for (size_t i = 0; i < (size_t)UNIT_ROWS * (size_t)cols; i++) f[i] = rand_uniform();
    ggml_quantize_chunk(b->gtype, f, unit, 0, UNIT_ROWS, cols, NULL);
    for (int64_t r = 0; r < rows; r += UNIT_ROWS) {
        int64_t n = rows - r < UNIT_ROWS ? rows - r : UNIT_ROWS;
        memcpy(m + (size_t)r * b->row_bytes, unit, (size_t)n * b->row_bytes);
    }
    free(unit);
    free(f);
    return m;
}

static int setup(bench *b, const shape *sh, enum ggml_type gtype, tr_type ttype, int repack) {
    memset(b, 0, sizeof *b);
    b->sh = sh;
    b->gtype = gtype;
    b->ttype = ttype;
    b->repack = repack;
    b->row_bytes = ggml_row_size(gtype, sh->cols);
    if (b->row_bytes != tr_row_bytes(ttype, sh->cols)) die("row size differs between ggml and trochilus");
    b->mat_bytes = b->row_bytes * (size_t)sh->rows;
    size_t tensor_bytes = b->mat_bytes * (sh->moe ? N_EXPERT : 1);
    b->n_copies = (int)(GIB / tensor_bytes);
    if (b->n_copies < 1) b->n_copies = 1;
    /* experts: N_EXPERT / TOP_K nodes per copy read, on average, as much as the copy holds */
    b->n_nodes = sh->moe ? b->n_copies * (N_EXPERT / TOP_K) : b->n_copies;

    ggml_backend_buffer_type_t wbuft = ggml_backend_cpu_buffer_type();
    if (repack) {
        wbuft = find_repack_buft();
        if (!wbuft) {
            printf("%-11s %-11s -- no repack buffer type on this CPU, skipped\n", sh->name, "q4_k-repack");
            return 0;
        }
    }

    unsigned char *m = make_matrix(b);
    size_t n_mats = (size_t)b->n_copies * (sh->moe ? N_EXPERT : 1);
    b->arena = tr_alloc_aligned(n_mats * b->mat_bytes, 4096);
    if (!b->arena) die("out of memory (arena)");
    for (size_t i = 0; i < n_mats; i++) memcpy(b->arena + i * b->mat_bytes, m, b->mat_bytes);

    /* ggml's weights: n_copies tensors in one buffer of wbuft (repack transforms on set) */
    struct ggml_init_params wip = {ggml_tensor_overhead() * (size_t)(b->n_copies + 1), NULL, true};
    b->ctx_w = ggml_init(wip);
    struct ggml_tensor **w = malloc(sizeof(*w) * (size_t)b->n_copies);
    if (!b->ctx_w || !w) die("out of memory (ggml weights)");
    for (int c = 0; c < b->n_copies; c++)
        w[c] = sh->moe ? ggml_new_tensor_3d(b->ctx_w, gtype, sh->cols, sh->rows, N_EXPERT)
                       : ggml_new_tensor_2d(b->ctx_w, gtype, sh->cols, sh->rows);
    b->buf_w = ggml_backend_alloc_ctx_tensors_from_buft(b->ctx_w, wbuft);
    if (!b->buf_w) die("ggml weight buffer");
    /* one whole tensor per set: the repack buffer transforms a tensor only when set at once */
    unsigned char *whole = sh->moe ? malloc(tensor_bytes) : m;
    if (!whole) die("out of memory (staging)");
    if (sh->moe)
        for (int e = 0; e < N_EXPERT; e++) memcpy(whole + (size_t)e * b->mat_bytes, m, b->mat_bytes);
    for (int c = 0; c < b->n_copies; c++) ggml_backend_tensor_set(w[c], whole, 0, tensor_bytes);
    if (whole != m) free(whole);
    free(m);

    /* the token, the gathered rows (the same token once per chosen expert), trochilus's output */
    b->x = tr_alloc_aligned((size_t)sh->cols * sizeof(float), 64);
    b->xg = tr_alloc_aligned((size_t)TOP_K * (size_t)sh->cols * sizeof(float), 64);
    b->y = tr_alloc_aligned((size_t)TOP_K * (size_t)sh->rows * sizeof(float), 64);
    if (!b->x || !b->xg || !b->y) die("out of memory (vectors)");
    for (int64_t i = 0; i < sh->cols; i++) b->x[i] = rand_uniform();
    for (int k = 0; k < TOP_K; k++) memcpy(b->xg + (size_t)k * (size_t)sh->cols, b->x, (size_t)sh->cols * sizeof(float));

    /* trochilus's matrices per node; the experts' choices (distinct, sorted as olmoe.c groups
     * them: ours are laid out by expert, ggml's by slot, and the check maps one onto the other) */
    size_t per_node = sh->moe ? N_EXPERT : 1;
    b->mats = malloc(sizeof(tr_mat) * per_node * (size_t)b->n_nodes);
    if (!b->mats) die("out of memory (mats)");
    if (sh->moe) {
        b->offsets = malloc(sizeof(int64_t) * (N_EXPERT + 1) * (size_t)b->n_nodes);
        b->ids = malloc(sizeof(int32_t) * TOP_K * (size_t)b->n_nodes);
        if (!b->offsets || !b->ids) die("out of memory (ids)");
    }
    for (int n = 0; n < b->n_nodes; n++) {
        int c = n % b->n_copies;
        for (size_t e = 0; e < per_node; e++) {
            tr_mat *t = &b->mats[(size_t)n * per_node + e];
            t->type = ttype;
            t->rows = sh->rows;
            t->cols = sh->cols;
            t->data = b->arena + ((size_t)c * per_node + e) * b->mat_bytes;
        }
        if (sh->moe) {
            int chosen[N_EXPERT] = {0};
            int32_t *ids = b->ids + (size_t)n * TOP_K;
            for (int k = 0; k < TOP_K; k++) {
                int e;
                do e = (int)(rng_next() % N_EXPERT); while (chosen[e]);
                chosen[e] = 1;
                ids[k] = e;
            }
            int64_t *off = b->offsets + (size_t)n * (N_EXPERT + 1);
            off[0] = 0;
            for (int e = 0; e < N_EXPERT; e++) off[e + 1] = off[e] + chosen[e];
        }
    }

    /* ggml's graph: one node per copy (dense) or n_nodes MUL_MAT_ID nodes (experts) */
    struct ggml_init_params gip = {ggml_tensor_overhead() * (size_t)(3 * b->n_nodes + 8) +
                                       ggml_graph_overhead_custom((size_t)(4 * b->n_nodes + 16), false),
                                   NULL, true};
    b->ctx_g = ggml_init(gip);
    if (!b->ctx_g) die("ggml graph context");
    struct ggml_tensor *gx = sh->moe ? ggml_new_tensor_3d(b->ctx_g, GGML_TYPE_F32, sh->cols, 1, 1)
                                     : ggml_new_tensor_2d(b->ctx_g, GGML_TYPE_F32, sh->cols, 1);
    struct ggml_tensor **gids = NULL;
    b->results = malloc(sizeof(*b->results) * (size_t)b->n_nodes);
    if (sh->moe) gids = malloc(sizeof(*gids) * (size_t)b->n_nodes);
    if (!b->results || (sh->moe && !gids)) die("out of memory (graph)");
    b->gf = ggml_new_graph_custom(b->ctx_g, (size_t)(4 * b->n_nodes + 16), false);
    for (int n = 0; n < b->n_nodes; n++) {
        struct ggml_tensor *wc = w[n % b->n_copies];
        if (sh->moe) {
            gids[n] = ggml_new_tensor_2d(b->ctx_g, GGML_TYPE_I32, TOP_K, 1);
            b->results[n] = ggml_mul_mat_id(b->ctx_g, wc, gx, gids[n]);
        } else {
            b->results[n] = ggml_mul_mat(b->ctx_g, wc, gx);
        }
        ggml_build_forward_expand(b->gf, b->results[n]);
    }
    b->buf_g = ggml_backend_alloc_ctx_tensors_from_buft(b->ctx_g, ggml_backend_cpu_buffer_type());
    if (!b->buf_g) die("ggml graph buffer");
    ggml_backend_tensor_set(gx, b->x, 0, ggml_nbytes(gx));
    if (sh->moe)
        for (int n = 0; n < b->n_nodes; n++)
            ggml_backend_tensor_set(gids[n], b->ids + (size_t)n * TOP_K, 0, ggml_nbytes(gids[n]));
    free(gids);
    free(w);
    return 1;
}

static void teardown(bench *b) {
    ggml_backend_buffer_free(b->buf_g);
    ggml_backend_buffer_free(b->buf_w);
    ggml_free(b->ctx_g);
    ggml_free(b->ctx_w);
    free(b->results);
    free(b->ids);
    free(b->offsets);
    free(b->mats);
    tr_free_aligned(b->y);
    tr_free_aligned(b->xg);
    tr_free_aligned(b->x);
    tr_free_aligned(b->arena);
}

/* one pass of trochilus: every node's matmul */
static void pass_ours(const bench *b, tr_pool *pool) {
    for (int n = 0; n < b->n_nodes; n++) {
        if (b->sh->moe)
            tr_matmul_grouped(pool, b->mats + (size_t)n * N_EXPERT, b->offsets + (size_t)n * (N_EXPERT + 1), N_EXPERT,
                              b->xg, b->y);
        else
            tr_matmul(pool, &b->mats[n], b->x, 1, b->y);
    }
}

static double pass_bytes(const bench *b) {
    return (double)b->n_nodes * (double)(b->sh->moe ? TOP_K : 1) * (double)b->mat_bytes;
}

/* ggml's first node against trochilus's: the largest difference over the largest value */
static double check(bench *b, tr_pool *pool, struct ggml_cplan *cplan) {
    if (ggml_graph_compute(b->gf, cplan) != GGML_STATUS_SUCCESS) die("ggml compute failed");
    if (b->sh->moe)
        tr_matmul_grouped(pool, b->mats, b->offsets, N_EXPERT, b->xg, b->y);
    else
        tr_matmul(pool, &b->mats[0], b->x, 1, b->y);
    int64_t rows = b->sh->rows, n_out = rows * (b->sh->moe ? TOP_K : 1);
    float *g = malloc(sizeof(float) * (size_t)n_out);
    if (!g) die("out of memory (check)");
    ggml_backend_tensor_get(b->results[0], g, 0, sizeof(float) * (size_t)n_out);
    double max_diff = 0.0, max_ref = 0.0;
    for (int k = 0; k < (b->sh->moe ? TOP_K : 1); k++) {
        /* ggml's slot k holds expert ids[k]; ours holds it at that expert's place */
        int64_t place = b->sh->moe ? b->offsets[b->ids[k]] : 0;
        for (int64_t r = 0; r < rows; r++) {
            double ref = b->y[place * rows + r], v = g[(int64_t)k * rows + r];
            if (fabs(ref) > max_ref) max_ref = fabs(ref);
            if (!(fabs(v - ref) <= max_diff)) max_diff = fabs(v - ref);
        }
    }
    free(g);
    return max_ref > 0.0 ? max_diff / max_ref : 1.0;
}

static void stats(double *gbs, int n, double *med, double *spread) {
    qsort(gbs, (size_t)n, sizeof gbs[0], cmp_double);
    *med = (n % 2) ? gbs[n / 2] : 0.5 * (gbs[n / 2 - 1] + gbs[n / 2]);
    *spread = (gbs[n - 1] - gbs[0]) / *med;
}

static struct ggml_threadpool *ggml_pool(int T) {
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(T);
    const tr_cpu_info *cpu = tr_cpu();
    if (cpu->n_slots >= T) {
        memset(tpp.cpumask, 0, sizeof tpp.cpumask);
        for (int i = 0; i < T; i++) tpp.cpumask[cpu->slot[i].lcpu] = true;
        tpp.strict_cpu = true;
    }
    struct ggml_threadpool *tp = ggml_threadpool_new(&tpp);
    if (!tp) die("ggml threadpool");
    return tp;
}

static void run(const shape *sh, const char *tname, enum ggml_type gtype, tr_type ttype, int repack) {
    bench b;
    if (!setup(&b, sh, gtype, ttype, repack)) return;
    double bytes = pass_bytes(&b);
    size_t arena_bytes = (size_t)b.n_copies * (sh->moe ? N_EXPERT : 1) * b.mat_bytes;

    static const int want[] = {1, 2, 4, 8, 16};
    int cores = tr_cpu()->physical_cores;
    for (size_t ti = 0; ti < sizeof want / sizeof want[0]; ti++) {
        int T = want[ti];
        if (T > cores) break;
        double ours[MAX_SAMPLES], theirs[MAX_SAMPLES], reads[MAX_SAMPLES];
        int n = 0;
        double err = -1.0;
        for (int r = 0; r < n_rounds; r++) {
            /* trochilus */
            tr_pool *pool = tr_pool_create(T);
            if (!pool) die("tr_pool_create");
            pass_ours(&b, pool);
            for (int p = 0; p < PASSES; p++) {
                double t0 = tr_time_sec();
                pass_ours(&b, pool);
                ours[n + p] = bytes / (tr_time_sec() - t0) / 1e9;
            }
            /* the plain read of the same bytes, same pool */
            read_ctx rc = {b.arena, (arena_bytes / 64) & ~(size_t)63};
            tr_parallel_for(pool, 64, 1, read_body, &rc);
            for (int p = 0; p < PASSES; p++) {
                double t0 = tr_time_sec();
                tr_parallel_for(pool, 64, 1, read_body, &rc);
                reads[n + p] = (double)rc.share * 64.0 / (tr_time_sec() - t0) / 1e9;
            }
            /* ggml, on the same cores */
            struct ggml_threadpool *tp = ggml_pool(T);
            struct ggml_cplan cplan = ggml_graph_plan(b.gf, T, tp);
            uint8_t *work = cplan.work_size ? malloc(cplan.work_size) : NULL;
            if (cplan.work_size && !work) die("out of memory (ggml work)");
            cplan.work_data = work;
            if (r == 0) err = check(&b, pool, &cplan);
            else if (ggml_graph_compute(b.gf, &cplan) != GGML_STATUS_SUCCESS) die("ggml compute failed");
            for (int p = 0; p < PASSES; p++) {
                double t0 = tr_time_sec();
                ggml_graph_compute(b.gf, &cplan);
                theirs[n + p] = bytes / (tr_time_sec() - t0) / 1e9;
            }
            free(work);
            ggml_threadpool_free(tp);
            tr_pool_destroy(pool);
            n += PASSES;
            if (err > 0.02) {
                printf("%-11s %-11s t=%-2d  MISMATCH: ggml differs from trochilus by %.3g of the largest value\n",
                       sh->name, tname, T, err);
                exit(1);
            }
        }
        double mo, so, mt, st, mr, sr;
        stats(ours, n, &mo, &so);
        stats(theirs, n, &mt, &st);
        stats(reads, n, &mr, &sr);
        printf("%-11s %-11s t=%-2d  ours %6.2f GB/s (%4.1f%%)  ggml %6.2f (%4.1f%%)  read %6.2f (%4.1f%%)"
               "  ggml/ours %.2f  ours/read %.2f  err %.1e\n",
               sh->name, tname, T, mo, so * 100, mt, st * 100, mr, sr * 100, mt / mo, mo / mr, err);
        fflush(stdout);
    }
    teardown(&b);
}

int main(int argc, char **argv) {
    const char *which = argc > 1 ? argv[1] : "";
    for (int i = 2; i + 1 < argc; i++)
        if (strcmp(argv[i], "--rounds") == 0) n_rounds = atoi(argv[i + 1]);
    if (n_rounds < 1 || n_rounds > MAX_ROUNDS) die("--rounds between 1 and 16");
    enum ggml_type gtype;
    tr_type ttype;
    int repack = 0;
    if (strcmp(which, "q8_0") == 0) gtype = GGML_TYPE_Q8_0, ttype = TR_TYPE_Q8_0;
    else if (strcmp(which, "q4_k") == 0) gtype = GGML_TYPE_Q4_K, ttype = TR_TYPE_Q4_K;
    else if (strcmp(which, "q4_k-repack") == 0) gtype = GGML_TYPE_Q4_K, ttype = TR_TYPE_Q4_K, repack = 1;
    else {
        fprintf(stderr, "usage: bench_ggml_decode q8_0|q4_k|q4_k-repack [--rounds R]\n");
        return 2;
    }

    tr_kernels_init();
    char desc[256];
    tr_cpu_describe(tr_cpu(), desc, sizeof desc);
    printf("%s\ntrochilus tier %s; ggml avx512 %d, avx512_vnni %d; %d rounds x %d passes, ~1 GiB per pass\n\n", desc,
           tr_kernels_get()->tier, ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni(), n_rounds, PASSES);

    static const shape shapes[] = {
        {"1024x2048", 1024, 2048, 0},
        {"2048x1024", 2048, 1024, 0},
        {"2048x2048", 2048, 2048, 0},
        {"50304x2048", 50304, 2048, 0},
        {"experts", 1024, 2048, 1},
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) run(&shapes[s], which, gtype, ttype, repack);
    return 0;
}
