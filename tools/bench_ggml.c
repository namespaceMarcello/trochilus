/* bench_ggml.c -- measures llama.cpp's ggml CPU matmul kernels (MUL_MAT, MUL_MAT_ID) in isolation,
 * on the same shapes as trochilus's tests/bench_peak.c matmul lines, so their GFLOP/s per core can
 * be compared against tr_matmul's. Links statically against ref/llama.cpp (commit b49650a)'s ggml,
 * built in ref/llama.cpp/build-trochilus/ (GGML_NATIVE, GGML_LLAMAFILE, GGML_CPU_REPACK all ON).
 * This is a one-off comparison tool: it is not part of the engine, not linked into
 * build/trochilus, and not run by `make check`.
 *
 * Cases (activations are F32, random in [-1,1]; ggml quantizes them to the weight's vec_dot type
 * as part of every call -- that conversion is included in the timing, since it is part of the
 * op's real cost):
 *   mul_mat        weights Q8_0: 1024x2048x64, 2048x1024x64, 2048x2048x512 tokens (OLMoE's
 *                  gate/up, down, and an attention projection)
 *   mul_mat_id     the MoE path: 64 experts of 1024x2048 Q8_0, 512 tokens, top-8 distinct experts
 *                  per token, chosen with a seeded RNG (OLMoE's gate/up shape)
 *   mul_mat/_id    the 1024x2048x64 mul_mat and the mul_mat_id case again, weights Q4_K
 *   mul_mat/_id    the same two Q4_K cases once more, through the CPU backend's "repack" extra
 *                  buffer type (what llama-bench uses for Q4_K on AVX2/AVX-512), when the running
 *                  CPU has a registered repack kernel for Q4_K (it does on AVX2/AVX-512)
 *
 * Per case: 1 thread and min(16, nproc) threads, one warm-up call, a calibrated repeat count
 * (targets >= 0.1 s per timed sample), median of 15 timed samples and their spread
 * (max-min)/median. GFLOP/s = 2 * rows * cols * tokens (mul_mat_id: tokens * top_k) / seconds.
 *
 * Build and run: tools/bench_ggml.sh (compiles and runs this file inside the trochilus-dev
 * container, against the static libs in ref/llama.cpp/build-trochilus/).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#define N_RUNS 15

static uint64_t g_rng = 0;

static uint64_t rng_next(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t x = g_rng;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static float rand_uniform(void) { /* [-1, 1) */
    return (float)((rng_next() >> 40) & 0xFFFF) / 32768.0f - 1.0f;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The CPU backend's extra buffer types (repack among them) are reached only through the public
 * proc-address indirection -- llama-model.cpp and llama-adapter.cpp use the same pattern to pick
 * a repack buffer type for a weight at load time. */
typedef ggml_backend_buffer_type_t *(*get_extra_bufts_t)(ggml_backend_dev_t);

static ggml_backend_buffer_type_t find_repack_buft(void) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) return NULL;
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    get_extra_bufts_t fn =
        (get_extra_bufts_t)ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (!fn) return NULL;
    ggml_backend_buffer_type_t *bufts = fn(cpu_dev);
    if (!bufts) return NULL;
    for (; *bufts; bufts++) {
        const char *name = ggml_backend_buft_name(*bufts);
        if (name && strstr(name, "REPACK")) return *bufts;
    }
    return NULL;
}

typedef struct {
    const char *name;
    enum ggml_type wtype;
    int repack;
    int64_t rows, cols, tokens, n_expert, top_k;
} bench_case;

static void run_case(const bench_case *bc, int max_threads) {
    int is_moe = bc->n_expert > 1;
    size_t w_elems = (size_t)bc->rows * (size_t)bc->cols * (size_t)bc->n_expert;

    float *wf = malloc(w_elems * sizeof(float));
    if (!wf) { fprintf(stderr, "out of memory (weight f32)\n"); exit(1); }
    for (size_t i = 0; i < w_elems; i++) wf[i] = rand_uniform();

    size_t row_bytes = ggml_row_size(bc->wtype, bc->cols);
    size_t q_bytes = row_bytes * (size_t)bc->rows * (size_t)bc->n_expert;
    void *qbuf = malloc(q_bytes);
    if (!qbuf) { fprintf(stderr, "out of memory (weight quantized)\n"); exit(1); }
    ggml_quantize_chunk(bc->wtype, wf, qbuf, 0, bc->rows * bc->n_expert, bc->cols, NULL);
    free(wf);

    ggml_backend_buffer_type_t wbuft = ggml_backend_cpu_buffer_type();
    if (bc->repack) {
        ggml_backend_buffer_type_t rb = find_repack_buft();
        if (!rb) {
            printf("%-34s %-6s  -- no CPU repack buffer type registered for this weight, skipping\n",
                   bc->name, ggml_type_name(bc->wtype));
            free(qbuf);
            return;
        }
        wbuft = rb;
    }

    struct ggml_init_params wip = {
        .mem_size = 4096 + ggml_tensor_overhead() * 2, .mem_buffer = NULL, .no_alloc = true};
    struct ggml_context *ctx_w = ggml_init(wip);
    struct ggml_tensor *w = is_moe ? ggml_new_tensor_3d(ctx_w, bc->wtype, bc->cols, bc->rows, bc->n_expert)
                                   : ggml_new_tensor_2d(ctx_w, bc->wtype, bc->cols, bc->rows);
    ggml_set_name(w, "weight");

    struct ggml_backend_buffer *buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, wbuft);
    if (!buf_w) { fprintf(stderr, "%s: weight buffer alloc failed\n", bc->name); exit(1); }
    ggml_backend_tensor_set(w, qbuf, 0, ggml_nbytes(w)); /* repack buft: transforms in place here */
    free(qbuf);

    struct ggml_init_params gip = {.mem_size = 4 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true};
    struct ggml_context *ctx_g = ggml_init(gip);

    struct ggml_tensor *x, *ids = NULL, *result;
    if (is_moe) {
        x = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, bc->cols, 1, bc->tokens);
        ids = ggml_new_tensor_2d(ctx_g, GGML_TYPE_I32, bc->top_k, bc->tokens);
        result = ggml_mul_mat_id(ctx_g, w, x, ids);
    } else {
        x = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, bc->cols, bc->tokens);
        result = ggml_mul_mat(ctx_g, w, x);
    }
    struct ggml_cgraph *gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, result);

    struct ggml_backend_buffer *buf_g =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx_g, ggml_backend_cpu_buffer_type());
    if (!buf_g) { fprintf(stderr, "%s: graph buffer alloc failed\n", bc->name); exit(1); }

    size_t x_n = (size_t)bc->cols * (size_t)(is_moe ? bc->tokens : bc->tokens);
    float *xf = malloc(x_n * sizeof(float));
    if (!xf) { fprintf(stderr, "out of memory (activations)\n"); exit(1); }
    for (size_t i = 0; i < x_n; i++) xf[i] = rand_uniform();
    ggml_backend_tensor_set(x, xf, 0, ggml_nbytes(x));
    free(xf);

    if (is_moe) {
        int32_t *idbuf = malloc(sizeof(int32_t) * (size_t)bc->top_k * (size_t)bc->tokens);
        int32_t *pool = malloc(sizeof(int32_t) * (size_t)bc->n_expert);
        if (!idbuf || !pool) { fprintf(stderr, "out of memory (ids)\n"); exit(1); }
        for (int64_t t = 0; t < bc->tokens; t++) {
            for (int64_t i = 0; i < bc->n_expert; i++) pool[i] = (int32_t)i;
            for (int64_t k = 0; k < bc->top_k; k++) {
                int64_t j = k + (int64_t)(rng_next() % (uint64_t)(bc->n_expert - k));
                int32_t tmp = pool[k];
                pool[k] = pool[j];
                pool[j] = tmp;
                idbuf[t * bc->top_k + k] = pool[k];
            }
        }
        ggml_backend_tensor_set(ids, idbuf, 0, ggml_nbytes(ids));
        free(idbuf);
        free(pool);
    }

    double flop_per_call = is_moe ? 2.0 * (double)bc->rows * (double)bc->cols * (double)bc->tokens *
                                         (double)bc->top_k
                                   : 2.0 * (double)bc->rows * (double)bc->cols * (double)bc->tokens;

    int thread_counts[2] = {1, max_threads};
    for (int tci = 0; tci < 2; tci++) {
        int nt = thread_counts[tci];
        if (tci == 1 && nt == thread_counts[0]) continue; /* max_threads == 1: nothing new to show */

        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(nt);
        struct ggml_threadpool *tp = ggml_threadpool_new(&tpp);
        struct ggml_cplan cplan = ggml_graph_plan(gf, nt, tp);
        uint8_t *work = NULL;
        if (cplan.work_size > 0) {
            work = malloc(cplan.work_size);
            if (!work) { fprintf(stderr, "out of memory (work buffer)\n"); exit(1); }
            cplan.work_data = work;
        }

        if (ggml_graph_compute(gf, &cplan) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: warm-up compute failed\n", bc->name);
            exit(1);
        }

        int calls = 1;
        double t = 0.0;
        for (;;) {
            double t0 = now_sec();
            for (int k = 0; k < calls; k++) ggml_graph_compute(gf, &cplan);
            t = now_sec() - t0;
            if (t >= 0.10 || calls >= (1 << 22)) break;
            calls *= 2;
        }

        double g[N_RUNS];
        for (int r = 0; r < N_RUNS; r++) {
            double t0 = now_sec();
            for (int k = 0; k < calls; k++) ggml_graph_compute(gf, &cplan);
            double dt = now_sec() - t0;
            g[r] = flop_per_call * (double)calls / dt / 1e9;
        }
        qsort(g, N_RUNS, sizeof g[0], cmp_double);
        double median = g[N_RUNS / 2];
        double spread = (g[N_RUNS - 1] - g[0]) / median;
        printf("%-34s %-6s %7d %12.1f %7.1f%%\n", bc->name, ggml_type_name(bc->wtype), nt, median,
               spread * 100.0);

        free(work);
        ggml_threadpool_free(tp);
    }

    ggml_backend_buffer_free(buf_g);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_g);
    ggml_free(ctx_w);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    g_rng = 0x9E3779B97F4A7C15ULL;

    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    int max_threads = (int)(nproc > 16 ? 16 : (nproc < 1 ? 1 : nproc));

    printf("nproc %ld, using 1 and %d threads\n", nproc, max_threads);
    printf("avx2 %d, avx512f %d, avx512_vnni %d, llamafile %d\n", ggml_cpu_has_avx2(),
           ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni(), ggml_cpu_has_llamafile());
    printf("\n%-34s %-6s %7s %12s %7s\n", "case", "type", "threads", "GFLOP/s", "spread");

    bench_case cases[] = {
        {"mul_mat 1024x2048x64", GGML_TYPE_Q8_0, 0, 1024, 2048, 64, 1, 1},
        {"mul_mat 2048x1024x64", GGML_TYPE_Q8_0, 0, 2048, 1024, 64, 1, 1},
        {"mul_mat 2048x2048x512", GGML_TYPE_Q8_0, 0, 2048, 2048, 512, 1, 1},
        {"mul_mat_id 64x(1024x2048) top8", GGML_TYPE_Q8_0, 0, 1024, 2048, 512, 64, 8},
        {"mul_mat 1024x2048x64", GGML_TYPE_Q4_K, 0, 1024, 2048, 64, 1, 1},
        {"mul_mat_id 64x(1024x2048) top8", GGML_TYPE_Q4_K, 0, 1024, 2048, 512, 64, 8},
        {"mul_mat 1024x2048x64 -repack", GGML_TYPE_Q4_K, 1, 1024, 2048, 64, 1, 1},
        {"mul_mat_id 64x(1024x2048) top8 -repack", GGML_TYPE_Q4_K, 1, 1024, 2048, 512, 64, 8},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) run_case(&cases[i], max_threads);

    return 0;
}
