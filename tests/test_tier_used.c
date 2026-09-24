/* test_tier_used.c — the kernels of the active tier are the ones a model runs.
 *
 * Which branch this exercises: the dispatch. Every other kernel test compares NUMBERS, tier
 * against scalar, and a tier whose table silently falls back to scalar's function gives the same
 * numbers many times slower: for two days the F32 weight rows (the router's matrix of every
 * GGUF) and the F16 ones ran on the scalar loop in every tier, and every test was green
 * (docs/LESSONS.md #78; the fourth green check that did not see its branch, after #43, #50, #54).
 * Identical logits between tiers say the tiers agree, not that a tier is used. Two halves:
 *
 *   the table    for every tier this CPU has, every entry the hot zone goes through is a function
 *                of the tier's own, not scalar's: dot_f32, axpy_f32, their x4, and dot_row and
 *                dot_row_x4 of EVERY weight type the engine supports (tr_kernels_support). A new
 *                type enters the engine with its tier kernels, or this goes red. A tier is handed
 *                out under its own name, exactly when the CPU has it, and tr_kernels_init takes
 *                the fastest.
 *   the ops      tr_rmsnorm and tr_attention_head reduce through the active table.
 *   the engine   a model with each weight type, and an F32 router beside it as in every real
 *                GGUF, runs with the active table wrapped in counters: every product of every
 *                matrix must come through the active table's entry for its type. The count is
 *                exact, rows x tokens by the model's shapes: a matrix that took another road
 *                (scalar called directly, a private copy of a table) leaves it short.
 *
 * tools/tier_check.sh runs this under TR_CPU_MAX=scalar and avx2 too, so the engine half sees
 * every tier as the active one. */
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test.h"
#include "../src/base/cpu.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"
#include "../src/models/model.h"
#include "synth_olmoe.h"

/* ---- the table -------------------------------------------------------------------------- */

static void test_table(void) {
    static const char *const tiers[] = {"avx2", "avx512"};
    const tr_kernels *S = tr_kernels_tier("scalar");
    TR_CHECK(S != NULL);
    if (S == NULL) return;
    for (size_t t = 0; t < sizeof tiers / sizeof tiers[0]; t++) {
        const tr_kernels *K = tr_kernels_tier(tiers[t]);
        if (K == NULL) {
            printf("  tier %-8s not available on this CPU: skipped\n", tiers[t]);
            continue;
        }
        int missing = 0;
        missing += K->dot_f32 == S->dot_f32;
        missing += K->axpy_f32 == S->axpy_f32;
        missing += K->dot_f32_x4 == S->dot_f32_x4;
        missing += K->axpy_f32_x4 == S->axpy_f32_x4;
        if (missing != 0) printf("  tier %s: %d of dot_f32, axpy_f32 and their x4 are scalar's\n", K->tier, missing);
        int types = 0;
        for (int type = 0; type < TR_TYPE_COUNT; type++) {
            if (!tr_kernels_support((tr_type)type)) continue;
            /* F16 rows convert with F16C: an AVX2 CPU without it (none is known) keeps scalar's */
            if (type == TR_TYPE_F16 && !tr_cpu()->f16c) continue;
            types++;
            if (K->dot_row[type] == NULL || K->dot_row[type] == S->dot_row[type]) {
                printf("  tier %s: dot_row of weight type %d is scalar's\n", K->tier, type);
                missing++;
            }
            if (K->dot_row_x4[type] == NULL || K->dot_row_x4[type] == S->dot_row_x4[type]) {
                printf("  tier %s: dot_row_x4 of weight type %d is scalar's or missing\n", K->tier, type);
                missing++;
            }
        }
        /* AVX-512 takes two rows at a time in every quantized type (kernels_x86.c): an entry left
         * out gives the same numbers slower, and nothing else would notice */
        if (strcmp(K->tier, "avx512") == 0) {
            static const tr_type row2[] = {TR_TYPE_Q8_0, TR_TYPE_Q4_K, TR_TYPE_Q6_K};
            for (size_t i = 0; i < sizeof row2 / sizeof row2[0]; i++) {
                if (K->dot_row2_x4[row2[i]] == NULL) {
                    printf("  tier avx512: dot_row2_x4 of weight type %d is missing\n", (int)row2[i]);
                    missing++;
                }
            }
        }
        TR_CHECK(strcmp(K->tier, tiers[t]) == 0); /* asked for one tier, handed another */
        TR_CHECK(types >= 3); /* F32, F16, Q8_0 today: a loop over nothing proves nothing */
        TR_CHECK_EQ_INT(missing, 0);
        if (missing == 0)
            printf("  tier %-8s every hot entry is its own: dot and axpy with their x4, dot_row and dot_row_x4 of %d "
                   "weight types\n",
                   K->tier, types);
    }

    /* a tier exists exactly when the CPU (capped by TR_CPU_MAX) has it, and tr_kernels_init takes
     * the fastest one there is: the same numbers from a slower tier would pass every other test */
    const tr_cpu_info *cpu = tr_cpu();
    TR_CHECK((tr_kernels_tier("avx2") != NULL) == (cpu->avx2 != 0));
    TR_CHECK((tr_kernels_tier("avx512") != NULL) == (cpu->avx512f != 0));
    const char *best = tr_kernels_tier("avx512") != NULL ? "avx512" : tr_kernels_tier("avx2") != NULL ? "avx2" : "scalar";
    TR_CHECK(tr_kernels_get() != NULL && strcmp(tr_kernels_get()->tier, best) == 0);
    printf("  active tier %s, the fastest this CPU offers\n", best);
}

/* ---- the operations built on the table ---------------------------------------------------- */

static const tr_kernels *g_under;
static atomic_ullong n_dot, n_axpy;
static float counted_dot_f32(const float *a, const float *b, int64_t n) {
    atomic_fetch_add(&n_dot, 1);
    return g_under->dot_f32(a, b, n);
}
static void counted_axpy_f32(float *y, const float *x, float a, int64_t n) {
    atomic_fetch_add(&n_axpy, 1);
    g_under->axpy_f32(y, x, a, n);
}

/* tr_rmsnorm and tr_attention_head reduce through the active table, not through scalar's: one
 * dot for a norm, one dot and one axpy per position for a head. The output of the head is a
 * buffer of exactly head_dim floats on the heap, so a write past it is ASan's. */
static void test_ops(void) {
    static tr_kernels counting;
    g_under = tr_kernels_get();
    counting = *g_under;
    counting.dot_f32 = counted_dot_f32;
    counting.axpy_f32 = counted_axpy_f32;
    enum { N = 40, HEAD = 8, POS = 3 };
    float x[N], w[N], keys[POS * HEAD], values[POS * HEAD], q[HEAD], scores[POS];
    for (int i = 0; i < N; i++) {
        x[i] = (float)(i % 7) - 3.0f;
        w[i] = 1.0f + (float)i / N;
    }
    for (int i = 0; i < POS * HEAD; i++) {
        keys[i] = (float)(i % 5) / 5;
        values[i] = (float)(i % 3) - 1;
    }
    for (int i = 0; i < HEAD; i++) q[i] = (float)i / HEAD;
    float *out = (float *)malloc(HEAD * sizeof(float));
    TR_CHECK(out != NULL);
    if (out == NULL) return;
    atomic_store(&n_dot, 0);
    atomic_store(&n_axpy, 0);
    tr_kernels_set_active(&counting);
    tr_rmsnorm(x, w, N, 1e-5f);
    TR_CHECK_EQ_INT(atomic_load(&n_dot), 1);
    tr_attention_head(q, keys, values, HEAD, 0, POS, HEAD, 0.5f, scores, out);
    tr_kernels_set_active(NULL);
    TR_CHECK_EQ_INT(atomic_load(&n_dot), 1 + POS);
    TR_CHECK_EQ_INT(atomic_load(&n_axpy), POS);
    free(out);
}

/* ---- the engine ------------------------------------------------------------------------- */

static const tr_kernels *g_real;                                       /* the tier under the counters */
static atomic_ullong n_row[TR_TYPE_COUNT], n_x4[TR_TYPE_COUNT], n_r2[TR_TYPE_COUNT]; /* calls, by weight type */
static atomic_ullong n_dot_x4, n_axpy_x4;

#define COUNTED(type, name) \
    static float row_##name(const void *row, const float *x, int64_t n) { \
        atomic_fetch_add(&n_row[type], 1); \
        return g_real->dot_row[type](row, x, n); \
    } \
    static void x4_##name(const void *row, const float *x, int64_t stride, int64_t n, float *out) { \
        atomic_fetch_add(&n_x4[type], 1); \
        g_real->dot_row_x4[type](row, x, stride, n, out); \
    } \
    static void r2_##name(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n, float *out) { \
        atomic_fetch_add(&n_r2[type], 1); \
        g_real->dot_row2_x4[type](row0, row1, x, stride, n, out); \
    }
COUNTED(TR_TYPE_F32, f32)
COUNTED(TR_TYPE_F16, f16)
COUNTED(TR_TYPE_Q8_0, q8_0)
COUNTED(TR_TYPE_Q4_K, q4_k)
COUNTED(TR_TYPE_Q6_K, q6_k)

static void counted_dot_f32_x4(const float *a, const float *b, int64_t stride, int64_t n, float *out) {
    atomic_fetch_add(&n_dot_x4, 1);
    g_real->dot_f32_x4(a, b, stride, n, out);
}

static void counted_axpy_f32_x4(float *y, const float *x, int64_t stride, const float *a, int64_t n) {
    atomic_fetch_add(&n_axpy_x4, 1);
    g_real->axpy_f32_x4(y, x, stride, a, n);
}

/* 2 layers, n_embd 64, 4 heads (2 kv), n_ff 64, 8 experts (3 used), vocab 48, context 32 */
enum { LAYERS = 2, N_EMBD = 64, N_HEAD = 4, N_HEAD_KV = 2, N_FF = 64, N_EXPERT = 8, N_USED = 3, VOCAB = 48, CTX = 32 };
enum { N_PROMPT = 11, N_SINGLE = 2 }; /* a pass of 11 tokens (x4 and a tail of 3), then one token a pass */

static void test_engine(const char *argv0, tr_type type, const char *name, long long n_embd, long long n_ff) {
    static tr_kernels counting;
    g_real = tr_kernels_get();
    counting = *g_real;
    counting.dot_row[TR_TYPE_F32] = row_f32;
    counting.dot_row[TR_TYPE_F16] = row_f16;
    counting.dot_row[TR_TYPE_Q8_0] = row_q8_0;
    counting.dot_row[TR_TYPE_Q4_K] = row_q4_k;
    counting.dot_row[TR_TYPE_Q6_K] = row_q6_k;
    if (g_real->dot_row_x4[TR_TYPE_F32] != NULL) counting.dot_row_x4[TR_TYPE_F32] = x4_f32;
    if (g_real->dot_row_x4[TR_TYPE_F16] != NULL) counting.dot_row_x4[TR_TYPE_F16] = x4_f16;
    if (g_real->dot_row_x4[TR_TYPE_Q8_0] != NULL) counting.dot_row_x4[TR_TYPE_Q8_0] = x4_q8_0;
    if (g_real->dot_row_x4[TR_TYPE_Q4_K] != NULL) counting.dot_row_x4[TR_TYPE_Q4_K] = x4_q4_k;
    if (g_real->dot_row_x4[TR_TYPE_Q6_K] != NULL) counting.dot_row_x4[TR_TYPE_Q6_K] = x4_q6_k;
    if (g_real->dot_row2_x4[TR_TYPE_F32] != NULL) counting.dot_row2_x4[TR_TYPE_F32] = r2_f32;
    if (g_real->dot_row2_x4[TR_TYPE_F16] != NULL) counting.dot_row2_x4[TR_TYPE_F16] = r2_f16;
    if (g_real->dot_row2_x4[TR_TYPE_Q8_0] != NULL) counting.dot_row2_x4[TR_TYPE_Q8_0] = r2_q8_0;
    if (g_real->dot_row2_x4[TR_TYPE_Q4_K] != NULL) counting.dot_row2_x4[TR_TYPE_Q4_K] = r2_q4_k;
    if (g_real->dot_row2_x4[TR_TYPE_Q6_K] != NULL) counting.dot_row2_x4[TR_TYPE_Q6_K] = r2_q6_k;
    counting.dot_f32_x4 = counted_dot_f32_x4;
    counting.axpy_f32_x4 = counted_axpy_f32_x4;
    for (int i = 0; i < TR_TYPE_COUNT; i++) {
        atomic_store(&n_row[i], 0);
        atomic_store(&n_x4[i], 0);
        atomic_store(&n_r2[i], 0);
    }
    atomic_store(&n_dot_x4, 0);
    atomic_store(&n_axpy_x4, 0);

    const synth_params P = {LAYERS, (uint32_t)n_embd, N_HEAD, N_HEAD_KV, (uint32_t)n_ff, N_EXPERT, N_USED, VOCAB, CTX, type};
    char path[512], err[256];
    synth_f32_router = 1;
    TR_CHECK(synth_write(&P, argv0, "test_tier_used_tmp.gguf", path, sizeof path) == 0);
    tr_pool *pool = tr_pool_create(3); /* chunk borders fall inside input rows: both matmul paths */
    tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
    tr_session *s = model != NULL ? tr_session_create(model, 0, 0, err, sizeof err) : NULL;
    TR_CHECK(s != NULL);
    if (s != NULL) {
        int32_t tok[N_PROMPT + N_SINGLE];
        for (int i = 0; i < N_PROMPT + N_SINGLE; i++) tok[i] = (int32_t)((i * 7 + 3) % VOCAB);
        tr_kernels_set_active(&counting);
        TR_CHECK(tr_session_eval(s, tok, N_PROMPT) == 0);
        for (int i = 0; i < N_SINGLE; i++) TR_CHECK(tr_session_eval(s, tok + N_PROMPT + i, 1) == 0);
        tr_kernels_set_active(NULL);

        /* products a token asks of the matrices of `type`, layer by layer: q, k, v, the output
         * projection, and gate, up and down of each expert it uses; the router's are F32; the
         * logits are one row of the vocabulary per call */
        long long tokens = N_PROMPT + N_SINGLE, evals = 1 + N_SINGLE;
        long long n_kv = (long long)N_HEAD_KV * (n_embd / N_HEAD);
        long long of_type = tokens * LAYERS * (n_embd + 2 * n_kv + n_embd + N_USED * (2 * n_ff + n_embd)) + evals * VOCAB;
        long long of_router = tokens * LAYERS * N_EXPERT;
        for (int i = 0; i < TR_TYPE_COUNT; i++) {
            long long want = (i == (int)type ? of_type : 0) + (i == TR_TYPE_F32 ? of_router : 0);
            long long got = (long long)atomic_load(&n_row[i]) + 4 * (long long)atomic_load(&n_x4[i]) +
                            8 * (long long)atomic_load(&n_r2[i]);
            TR_CHECK_EQ_INT(got, want);
            if (got != want) printf("  %s model, weight type %d: %lld products through the active table, %lld wanted\n",
                                    name, i, got, want);
        }
        /* a pass of 11 tokens must take the widest road the tier has (two rows at once, else x4),
         * and so must attention; every matrix of these models has an even number of rows */
        if (g_real->dot_row2_x4[type] != NULL) {
            TR_CHECK(atomic_load(&n_r2[type]) > 0);
            TR_CHECK_EQ_INT(atomic_load(&n_x4[type]), 0); /* no pair of rows left to the one-row road */
        }
        else if (g_real->dot_row_x4[type] != NULL) TR_CHECK(atomic_load(&n_x4[type]) > 0);
        /* the router's x4 road at n_embd 64 only: at 256 tr_matmul_grouped's chunks (4096 / cols + 1
         * products) hold fewer than 4 whole tokens of its 8 rows, and every product goes one dot at a time */
        if (g_real->dot_row_x4[TR_TYPE_F32] != NULL && n_embd == N_EMBD) TR_CHECK(atomic_load(&n_x4[TR_TYPE_F32]) > 0);
        TR_CHECK(atomic_load(&n_dot_x4) > 0);
        TR_CHECK(atomic_load(&n_axpy_x4) > 0);
        printf("  %-5s model on tier %-7s %lld products of its type and %lld of the F32 router, all through the "
               "active table\n",
               name, g_real->tier, of_type, of_router);
    } else {
        fprintf(stderr, "setup failed: %s\n", err);
    }
    tr_session_free(s);
    tr_model_free(model);
    tr_pool_destroy(pool);
    remove(path);
}

int main(int argc, char **argv) {
    tr_kernels_init();
    const char *argv0 = argc > 0 ? argv[0] : "";
    test_table();
    test_ops();
    test_engine(argv0, TR_TYPE_F32, "f32", N_EMBD, N_FF);
    test_engine(argv0, TR_TYPE_F16, "f16", N_EMBD, N_FF);
    test_engine(argv0, TR_TYPE_Q8_0, "q8_0", N_EMBD, N_FF);
    test_engine(argv0, TR_TYPE_Q4_K, "q4_k", 256, 256);   /* rows of whole 256-element blocks */
    test_engine(argv0, TR_TYPE_Q6_K, "q6_k", 256, 256);
    TR_TEST_EXIT();
}
