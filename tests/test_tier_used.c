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
 *                of the tier's own, not scalar's: dot_f32, axpy_f32, their x4 and 4x4, dot_row and
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
 * A router whose values fit bfloat16 (as every real OLMoE GGUF's) is kept BF16 at load: its products go through
 * the BF16 entries, counted exactly, and every pass's logits are the bits of the same model loaded with it F32.
 *
 * tools/tier_check.sh runs this under TR_CPU_MAX=scalar and avx2 too, so the engine half sees
 * every tier as the active one. */
#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* setenv/unsetenv */
#endif
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test.h"
#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"
#include "../src/models/model.h"
#include "synth_olmoe.h"

/* NULL clears the variable */
static void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value != NULL ? value : "");
#else
    if (value != NULL) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

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
        missing += K->dot_f32_4x4 == S->dot_f32_4x4;
        missing += K->axpy_f32_4x4 == S->axpy_f32_4x4;
        if (missing != 0)
            printf("  tier %s: %d of dot_f32, axpy_f32, their x4 and 4x4 are scalar's\n", K->tier, missing);
        int types = 0;
        for (int type = 0; type < TR_TYPE_COUNT; type++) {
            /* BF16: no file's type, the F32 matrices narrowed at load (tr_f32_to_bf16_exact) */
            if (!tr_kernels_support((tr_type)type) && type != TR_TYPE_BF16) continue;
            /* F16 rows convert with F16C: an AVX2 CPU without it (none is known) keeps scalar's */
            if (type == TR_TYPE_F16 && !tr_cpu()->f16c) continue;
            /* Q4_K's dot_row is its definition from the floats, off the hot path: the engine's Q4_K products
             * go through the q4x entries, checked below */
            if (type == TR_TYPE_Q4_K) continue;
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
            static const tr_type row2[] = {TR_TYPE_Q8_0, TR_TYPE_Q6_K};
            for (size_t i = 0; i < sizeof row2 / sizeof row2[0]; i++) {
                if (K->dot_row2_x4[row2[i]] == NULL) {
                    printf("  tier avx512: dot_row2_x4 of weight type %d is missing\n", (int)row2[i]);
                    missing++;
                }
            }
            /* and against eight tokens at a time */
            for (size_t i = 0; i < sizeof row2 / sizeof row2[0]; i++) {
                if (K->dot_row2_x8[row2[i]] == NULL) {
                    printf("  tier avx512: dot_row2_x8 of weight type %d is missing\n", (int)row2[i]);
                    missing++;
                }
            }
        }
        /* a short verify pass's groups of 2 and 3 input rows (dot_row_xt): Q8_0's is the tier's own */
        if (K->dot_row_xt[TR_TYPE_Q8_0] == NULL || K->dot_row_xt[TR_TYPE_Q8_0] == S->dot_row_xt[TR_TYPE_Q8_0]) {
            printf("  tier %s: dot_row_xt of Q8_0 is scalar's or missing\n", K->tier);
            missing++;
        }
        /* Q4_K's integer road: every tier prepares and pairs with its own kernels; AVX-512 with the byte
         * permutes, VNNI and DQ has its W16 panel and tiles too (elsewhere the road goes by rows) */
        const tr_cpu_info *ci = tr_cpu();
        const int w16 = strcmp(K->tier, "avx512") == 0 && ci->avx512bw && ci->avx512vl && ci->avx512dq &&
                        ci->avx512vnni && ci->avx512vbmi && ci->f16c;
        if (K->q4x_prep == S->q4x_prep || K->q4x_dot2 == S->q4x_dot2 || K->q4x_dot_xt == NULL ||
            K->q4x_dot_xt == S->q4x_dot_xt) {
            printf("  tier %s: q4x_prep, q4x_dot2 or q4x_dot_xt is scalar's or missing\n", K->tier);
            missing++;
        }
        if (w16 && (K->q4x_panel == NULL || K->q4x_panel == S->q4x_panel || K->q4x_tile == NULL ||
                    K->q4x_tile == S->q4x_tile)) {
            printf("  tier %s: the W16 panel or tile is missing or scalar's\n", K->tier);
            missing++;
        }
        TR_CHECK(strcmp(K->tier, tiers[t]) == 0); /* asked for one tier, handed another */
        TR_CHECK(types >= 3); /* F32, F16, Q8_0 today: a loop over nothing proves nothing */
        TR_CHECK_EQ_INT(missing, 0);
        if (missing == 0)
            printf("  tier %-8s every hot entry is its own: dot and axpy with their x4, dot_row and dot_row_x4 of %d "
                   "weight types, Q4_K's q4x %s\n",
                   K->tier, types, w16 ? "with the W16 panel and tiles" : "by rows");
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
static atomic_ullong n_row[TR_TYPE_COUNT], n_x4[TR_TYPE_COUNT], n_r2[TR_TYPE_COUNT], n_r8[TR_TYPE_COUNT]; /* calls, by type */
static atomic_ullong n_p2[TR_TYPE_COUNT];   /* dot_row2 calls (two rows, one token), by type */
static atomic_ullong n_xt[TR_TYPE_COUNT];   /* dot_row_xt products (one row, 2 or 3 tokens), by type */
static atomic_ullong n_dot_x4, n_axpy_x4, n_dot_4x4, n_axpy_4x4;

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
    } \
    static void r8_##name(const void *row0, const void *row1, const float *x, int64_t stride, int64_t n, float *out) { \
        atomic_fetch_add(&n_r8[type], 1); \
        g_real->dot_row2_x8[type](row0, row1, x, stride, n, out); \
    } \
    static void p2_##name(const void *row0, const void *row1, const float *x, int64_t n, float *out) { \
        atomic_fetch_add(&n_p2[type], 1); \
        g_real->dot_row2[type](row0, row1, x, n, out); \
    } \
    static void xt_##name(const void *row, const float *x, int64_t stride, int64_t n, int t, float *out) { \
        atomic_fetch_add(&n_xt[type], (unsigned long long)t); \
        g_real->dot_row_xt[type](row, x, stride, n, t, out); \
    }
COUNTED(TR_TYPE_F32, f32)
COUNTED(TR_TYPE_F16, f16)
COUNTED(TR_TYPE_BF16, bf16)
COUNTED(TR_TYPE_Q8_0, q8_0)
COUNTED(TR_TYPE_Q4_K, q4_k)
COUNTED(TR_TYPE_Q6_K, q6_k)

/* the prompt's phase-major road (kernels.h pm_*): panels by type, and the products of every tile,
 * TR_PM_ROWS rows by T input rows (a tile has no type: the models here have one quantized type) */
static atomic_ullong n_pm_panel[TR_TYPE_COUNT], n_pm_products;
#define COUNTED_PANEL(type, name) \
    static void pmp_##name(const void *rows, size_t row_bytes, int64_t n, float *panel) { \
        atomic_fetch_add(&n_pm_panel[type], 1); \
        g_real->pm_panel[type](rows, row_bytes, n, panel); \
    }
COUNTED_PANEL(TR_TYPE_Q8_0, q8_0)
COUNTED_PANEL(TR_TYPE_Q4_K, q4_k)
COUNTED_PANEL(TR_TYPE_Q6_K, q6_k)
static void counted_pm_tile(const float *panel, const float *xil, int64_t n, int T, float *part, float *y,
                            int64_t y_stride) {
    atomic_fetch_add(&n_pm_products, (unsigned long long)(TR_PM_ROWS * T));
    g_real->pm_tile(panel, xil, n, T, part, y, y_stride);
}

/* Q4_K's integer road (kernels.h q4x_*): the prepared rows, the pairs (two products a call), the runs of the
 * short passes' pairs (2 T products a call) and the tiles' products */
static atomic_ullong n_q4x_prep, n_q4x_dot2, n_q4x_xt, n_q4x_products;
static void counted_q4x_prep(const float *x, int64_t n, void *xq) {
    atomic_fetch_add(&n_q4x_prep, 1);
    g_real->q4x_prep(x, n, xq);
}
static void counted_q4x_dot2(const void *row0, const void *row1, const void *xq, int64_t n, float *out) {
    atomic_fetch_add(&n_q4x_dot2, 1);
    g_real->q4x_dot2(row0, row1, xq, n, out);
}
static void counted_q4x_dot_xt(const void *row0, const void *row1, const void *const *xq, int64_t n, int T, float *y,
                               int64_t y_stride) {
    atomic_fetch_add(&n_q4x_xt, 1);
    atomic_fetch_add(&n_q4x_products, (unsigned long long)(2 * T));
    g_real->q4x_dot_xt(row0, row1, xq, n, T, y, y_stride);
}
static void counted_q4x_tile(const void *panel, const void *const *xq, int64_t n, int T, float *y, int64_t y_stride) {
    atomic_fetch_add(&n_q4x_products, (unsigned long long)(TR_PM_ROWS * T));
    g_real->q4x_tile(panel, xq, n, T, y, y_stride);
}

static void counted_dot_f32_x4(const float *a, const float *b, int64_t stride, int64_t n, float *out) {
    atomic_fetch_add(&n_dot_x4, 1);
    g_real->dot_f32_x4(a, b, stride, n, out);
}

static void counted_axpy_f32_x4(float *y, const float *x, int64_t stride, const float *a, int64_t n) {
    atomic_fetch_add(&n_axpy_x4, 1);
    g_real->axpy_f32_x4(y, x, stride, a, n);
}

static void counted_dot_f32_4x4(const float *a, int64_t a_stride, const float *b, int64_t stride, int64_t n,
                                float scale, float *out, int64_t out_stride) {
    atomic_fetch_add(&n_dot_4x4, 1);
    g_real->dot_f32_4x4(a, a_stride, b, stride, n, scale, out, out_stride);
}

static void counted_axpy_f32_4x4(float *y, int64_t y_stride, const float *x, int64_t stride, const float *a,
                                 int64_t a_stride, int64_t n) {
    atomic_fetch_add(&n_axpy_4x4, 1);
    g_real->axpy_f32_4x4(y, y_stride, x, stride, a, a_stride, n);
}

/* 2 layers, n_embd 64, 4 heads (2 kv), n_ff 64, 8 experts (3 used), vocab 48, context 32 */
enum { LAYERS = 2, N_EMBD = 64, N_HEAD = 4, N_HEAD_KV = 2, N_FF = 64, N_EXPERT = 8, N_USED = 3, VOCAB = 48, CTX = 40 };
/* a pass of 29 tokens (three threads' chunks of ~9.7 whole tokens: 8 at a time, then fewer), then
 * one token a pass */
enum { N_PROMPT = 29, N_SINGLE = 2, N_SHORT = 3 };

/* the hints the pool's workers have taken (threads.h tr_pool_hint_counts) */
static uint64_t pool_hints(const tr_pool *p) {
    uint64_t sum = 0;
    for (int w = 1; p != NULL && w < tr_pool_size(p); w++) {
        uint64_t h, l;
        tr_pool_hint_counts(p, w, &h, &l);
        sum += h;
    }
    return sum;
}

static void test_engine(const char *argv0, tr_type type, const char *name, long long n_embd, long long n_ff,
                        int bf16_router, int n_head_kv) {
    static tr_kernels counting;
    g_real = tr_kernels_get();
    counting = *g_real;
    counting.dot_row[TR_TYPE_F32] = row_f32;
    counting.dot_row[TR_TYPE_F16] = row_f16;
    counting.dot_row[TR_TYPE_BF16] = row_bf16;
    if (g_real->dot_row_x4[TR_TYPE_BF16] != NULL) counting.dot_row_x4[TR_TYPE_BF16] = x4_bf16;
    if (g_real->dot_row2_x4[TR_TYPE_BF16] != NULL) counting.dot_row2_x4[TR_TYPE_BF16] = r2_bf16;
    if (g_real->dot_row2_x8[TR_TYPE_BF16] != NULL) counting.dot_row2_x8[TR_TYPE_BF16] = r8_bf16;
    if (g_real->dot_row2[TR_TYPE_BF16] != NULL) counting.dot_row2[TR_TYPE_BF16] = p2_bf16;
    if (g_real->dot_row_xt[TR_TYPE_BF16] != NULL) counting.dot_row_xt[TR_TYPE_BF16] = xt_bf16;
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
    if (g_real->dot_row2_x8[TR_TYPE_F32] != NULL) counting.dot_row2_x8[TR_TYPE_F32] = r8_f32;
    if (g_real->dot_row2_x8[TR_TYPE_F16] != NULL) counting.dot_row2_x8[TR_TYPE_F16] = r8_f16;
    if (g_real->dot_row2_x8[TR_TYPE_Q8_0] != NULL) counting.dot_row2_x8[TR_TYPE_Q8_0] = r8_q8_0;
    if (g_real->dot_row2_x8[TR_TYPE_Q4_K] != NULL) counting.dot_row2_x8[TR_TYPE_Q4_K] = r8_q4_k;
    if (g_real->dot_row2_x8[TR_TYPE_Q6_K] != NULL) counting.dot_row2_x8[TR_TYPE_Q6_K] = r8_q6_k;
    if (g_real->dot_row2[TR_TYPE_F32] != NULL) counting.dot_row2[TR_TYPE_F32] = p2_f32;
    if (g_real->dot_row2[TR_TYPE_F16] != NULL) counting.dot_row2[TR_TYPE_F16] = p2_f16;
    if (g_real->dot_row2[TR_TYPE_Q8_0] != NULL) counting.dot_row2[TR_TYPE_Q8_0] = p2_q8_0;
    if (g_real->dot_row2[TR_TYPE_Q4_K] != NULL) counting.dot_row2[TR_TYPE_Q4_K] = p2_q4_k;
    if (g_real->dot_row2[TR_TYPE_Q6_K] != NULL) counting.dot_row2[TR_TYPE_Q6_K] = p2_q6_k;
    if (g_real->dot_row_xt[TR_TYPE_F32] != NULL) counting.dot_row_xt[TR_TYPE_F32] = xt_f32;
    if (g_real->dot_row_xt[TR_TYPE_F16] != NULL) counting.dot_row_xt[TR_TYPE_F16] = xt_f16;
    if (g_real->dot_row_xt[TR_TYPE_Q8_0] != NULL) counting.dot_row_xt[TR_TYPE_Q8_0] = xt_q8_0;
    if (g_real->dot_row_xt[TR_TYPE_Q4_K] != NULL) counting.dot_row_xt[TR_TYPE_Q4_K] = xt_q4_k;
    if (g_real->dot_row_xt[TR_TYPE_Q6_K] != NULL) counting.dot_row_xt[TR_TYPE_Q6_K] = xt_q6_k;
    counting.dot_f32_x4 = counted_dot_f32_x4;
    counting.axpy_f32_x4 = counted_axpy_f32_x4;
    counting.dot_f32_4x4 = counted_dot_f32_4x4;
    counting.axpy_f32_4x4 = counted_axpy_f32_4x4;
    if (g_real->pm_panel[TR_TYPE_Q8_0] != NULL) counting.pm_panel[TR_TYPE_Q8_0] = pmp_q8_0;
    if (g_real->pm_panel[TR_TYPE_Q4_K] != NULL) counting.pm_panel[TR_TYPE_Q4_K] = pmp_q4_k;
    if (g_real->pm_panel[TR_TYPE_Q6_K] != NULL) counting.pm_panel[TR_TYPE_Q6_K] = pmp_q6_k;
    if (g_real->pm_tile != NULL) counting.pm_tile = counted_pm_tile;
    counting.q4x_prep = counted_q4x_prep;
    counting.q4x_dot2 = counted_q4x_dot2;
    if (g_real->q4x_dot_xt != NULL) counting.q4x_dot_xt = counted_q4x_dot_xt;
    if (g_real->q4x_tile != NULL) counting.q4x_tile = counted_q4x_tile;
    atomic_store(&n_q4x_prep, 0);
    atomic_store(&n_q4x_dot2, 0);
    atomic_store(&n_q4x_xt, 0);
    atomic_store(&n_q4x_products, 0);
    atomic_store(&n_pm_products, 0);
    for (int i = 0; i < TR_TYPE_COUNT; i++) {
        atomic_store(&n_row[i], 0);
        atomic_store(&n_x4[i], 0);
        atomic_store(&n_r2[i], 0);
        atomic_store(&n_r8[i], 0);
        atomic_store(&n_p2[i], 0);
        atomic_store(&n_xt[i], 0);
        atomic_store(&n_pm_panel[i], 0);
    }
    atomic_store(&n_dot_x4, 0);
    atomic_store(&n_axpy_x4, 0);
    atomic_store(&n_dot_4x4, 0);
    atomic_store(&n_axpy_4x4, 0);

    const synth_params P = {LAYERS, (uint32_t)n_embd, N_HEAD, (uint32_t)n_head_kv, (uint32_t)n_ff, N_EXPERT, N_USED, VOCAB,
                            CTX, type};
    char path[512], err[256];
    synth_f32_router = 1;
    synth_bf16_router = bf16_router; /* the router's values fit bf16: the engine keeps it BF16 */
    TR_CHECK(synth_write(&P, argv0, "test_tier_used_tmp.gguf", path, sizeof path) == 0);
    synth_bf16_router = 0;
    const tr_type router = bf16_router ? TR_TYPE_BF16 : TR_TYPE_F32;
    tr_pool *pool = tr_pool_create(3); /* chunk borders fall inside input rows: both matmul paths */
    tr_model *model = pool != NULL ? tr_model_load(path, pool, err, sizeof err) : NULL;
    tr_session *s = model != NULL ? tr_session_create(model, 0, 0, err, sizeof err) : NULL;
    TR_CHECK(s != NULL);
    int32_t tok[N_PROMPT + N_SINGLE + N_SHORT];
    for (int i = 0; i < N_PROMPT + N_SINGLE + N_SHORT; i++) tok[i] = (int32_t)((i * 7 + 3) % VOCAB);
    static float logits[2][1 + N_SINGLE + 1][VOCAB]; /* each pass's last logit row, the engine's road and prep's arm B */
    tr_pool_set_hints(pool, 2); /* off as a pool starts: on here, to see the engine's hints taken */
    const uint64_t hints0 = pool_hints(pool);
    if (s != NULL) {
        tr_kernels_set_active(&counting);
        TR_CHECK(tr_session_eval(s, tok, N_PROMPT) == 0);
        memcpy(logits[0][0], tr_session_logits(s), sizeof logits[0][0]);
        for (int i = 0; i < N_SINGLE; i++) {
            TR_CHECK(tr_session_eval(s, tok + N_PROMPT + i, 1) == 0);
            memcpy(logits[0][1 + i], tr_session_logits(s), sizeof logits[0][0]);
        }
        /* and a short pass, a verify pass of 3 rows: its groups of 2 and 3 through dot_row_xt */
        TR_CHECK(tr_session_eval(s, tok + N_PROMPT + N_SINGLE, N_SHORT) == 0);
        memcpy(logits[0][1 + N_SINGLE], tr_session_logits(s), sizeof logits[0][0]);
        tr_kernels_set_active(NULL);

        /* products a token asks of the matrices of `type`, layer by layer: q, k, v, the output
         * projection, and gate, up and down of each expert it uses; the router's are F32, or BF16 where its
         * values fit; the logits are one row of the vocabulary per call */
        long long tokens = N_PROMPT + N_SINGLE + N_SHORT, evals = 1 + N_SINGLE + 1;
        long long n_kv = (long long)n_head_kv * (n_embd / N_HEAD);
        long long of_type = tokens * LAYERS * (n_embd + 2 * n_kv + n_embd + N_USED * (2 * n_ff + n_embd)) + evals * VOCAB;
        long long of_router = tokens * LAYERS * N_EXPERT;
        for (int i = 0; i < TR_TYPE_COUNT; i++) {
            long long want = (i == (int)type ? of_type : 0) + (i == (int)router ? of_router : 0);
            long long got = (long long)atomic_load(&n_row[i]) + 4 * (long long)atomic_load(&n_x4[i]) +
                            8 * (long long)atomic_load(&n_r2[i]) + 16 * (long long)atomic_load(&n_r8[i]) +
                            2 * (long long)atomic_load(&n_p2[i]) + (long long)atomic_load(&n_xt[i]) +
                            (i == (int)type ? (long long)atomic_load(&n_pm_products) : 0) +
                            (i == TR_TYPE_Q4_K ? 2 * (long long)atomic_load(&n_q4x_dot2) +
                                                     (long long)atomic_load(&n_q4x_products)
                                               : 0);
            TR_CHECK_EQ_INT(got, want);
            if (got != want) printf("  %s model, weight type %d: %lld products through the active table, %lld wanted\n",
                                    name, i, got, want);
        }
        /* a pass of N_PROMPT tokens must take the widest road the tier has: phase-major (a panel of the type,
         * tiles of the prompt's rows), else two rows at once, else x4; and so must attention; every
         * matrix of these models has an even number of rows */
        const int pm = g_real->pm_tile != NULL && g_real->pm_panel[type] != NULL;
        if (pm) {
            TR_CHECK(atomic_load(&n_pm_panel[type]) > 0);
            TR_CHECK(atomic_load(&n_pm_products) > 0);
        } else if (g_real->dot_row2_x8[type] != NULL) {
            TR_CHECK(atomic_load(&n_r8[type]) > 0);
        }
        /* the short pass's groups of 2 and 3 input rows, where the type has the kernel (Q8_0: every tier) */
        if (g_real->dot_row_xt[type] != NULL) TR_CHECK(atomic_load(&n_xt[type]) > 0);
        /* and a one-token pass must take the decode's pair where the tier has it */
        if (g_real->dot_row2[type] != NULL) TR_CHECK(atomic_load(&n_p2[type]) > 0);
        /* Q4_K: every call prepares its rows; the one-token passes pair rows, the short pass runs its groups of
         * 2-3 rows through the runs' pairs, and the prompt's pass runs tiles where the tier has the W16 panel */
        if (type == TR_TYPE_Q4_K) {
            /* each pass's token rows prepared once for q, k and v and once for gate and up (a grouped row reads its
             * token's through the map), o's and down's rows by their own calls, and a logit row a pass
             * (docs/MEASUREMENTS.md §The prep once a row): 412 here, where every call preparing its own rows
             * made 888 */
            TR_CHECK_EQ_INT(atomic_load(&n_q4x_prep), (3 + N_USED) * tokens * LAYERS + evals);
            TR_CHECK(atomic_load(&n_q4x_dot2) > 0);
            TR_CHECK(atomic_load(&n_q4x_xt) > 0);
            if (g_real->q4x_panel != NULL && g_real->q4x_tile != NULL) TR_CHECK(atomic_load(&n_q4x_products) > 0);
            TR_CHECK_EQ_INT(atomic_load(&n_row[type]), 0); /* nothing left to the definition from the floats */
            printf("  q4_k  model: %llu rows prepared, %llu pairs, %llu runs of 2-3 rows, %llu run and tile products\n",
                   (unsigned long long)atomic_load(&n_q4x_prep), (unsigned long long)atomic_load(&n_q4x_dot2),
                   (unsigned long long)atomic_load(&n_q4x_xt), (unsigned long long)atomic_load(&n_q4x_products));
        }
        printf("  %-5s model: %llu phase-major products in %llu panels, %llu calls of 2 rows x 8 tokens, %llu of "
               "2 x 4, %llu of 1 x 4, %llu products of 1 x 2-3, %llu dots\n",
               name, (unsigned long long)atomic_load(&n_pm_products), (unsigned long long)atomic_load(&n_pm_panel[type]),
               (unsigned long long)atomic_load(&n_r8[type]), (unsigned long long)atomic_load(&n_r2[type]),
               (unsigned long long)atomic_load(&n_x4[type]), (unsigned long long)atomic_load(&n_xt[type]),
               (unsigned long long)atomic_load(&n_row[type]));
        if (pm) {
            TR_CHECK_EQ_INT(atomic_load(&n_x4[type]), 0); /* no pair of rows left to the one-row road */
        } else if (g_real->dot_row2_x4[type] != NULL) {
            TR_CHECK(atomic_load(&n_r2[type]) > 0);
            TR_CHECK_EQ_INT(atomic_load(&n_x4[type]), 0); /* no pair of rows left to the one-row road */
        }
        else if (g_real->dot_row_x4[type] != NULL) TR_CHECK(atomic_load(&n_x4[type]) > 0);
        /* the router's x4 road at n_embd 64 only: at 256 tr_matmul_grouped's chunks (4096 / cols + 1
         * products) hold fewer than 4 whole tokens of its 8 rows, and every product goes one dot at a time */
        if (g_real->dot_row_x4[router] != NULL && n_embd == N_EMBD) TR_CHECK(atomic_load(&n_x4[router]) > 0);
        TR_CHECK(atomic_load(&n_dot_x4) > 0);
        TR_CHECK(atomic_load(&n_axpy_x4) > 0);
        /* the prompt's tiles: a pass of 29 tokens has groups of 16 with quads that see 4 positions */
        TR_CHECK(atomic_load(&n_dot_4x4) > 0);
        TR_CHECK(atomic_load(&n_axpy_4x4) > 0);
        /* the one-token passes hinted their next calls to the idle workers (threads.h tr_pool_hint), and the
         * workers took them (the counts move as each hint starts) */
        uint64_t hints1 = pool_hints(pool);
        for (double t0 = tr_time_sec(); hints1 == hints0 && tr_time_sec() - t0 < 1.0;) hints1 = pool_hints(pool);
        TR_CHECK(hints1 > hints0);
        /* Q4_K: the same passes on a second session with a switch at arm B (generate --ab): prep, every call
         * preparing its own rows and the experts' gathered, 888 rows here; act, the swiglu apart and the down
         * preparing its own rows, the same 412; idle, no hint to the idle workers (none taken), the same 412;
         * fuse, q, k, v and gate, up one call each (tr_matmul_q4x_prepared_n never taken; act and idle take it,
         * prep's floats cannot), the same 412; every pass's logits the bits of the first session's */
        static const char *const sw_names[4] = {"prep", "act", "idle", "fuse"};
        const long long sw_preps[4] = {(4 + 3 * N_USED) * tokens * LAYERS + evals, (3 + N_USED) * tokens * LAYERS + evals,
                                       (3 + N_USED) * tokens * LAYERS + evals, (3 + N_USED) * tokens * LAYERS + evals};
        for (int si = 0; si < 4 && type == TR_TYPE_Q4_K; si++) {
            tr_session *s2 = tr_session_create(model, 0, 0, err, sizeof err);
            const int sw = s2 != NULL ? tr_session_ab_switch(s2, sw_names[si]) : -1;
            TR_CHECK(s2 != NULL && sw > 0);
            if (s2 != NULL && sw > 0) {
                tr_session_ab_set(s2, sw, 1);
                atomic_store(&n_q4x_prep, 0);
                const double settle = tr_time_sec(); /* the first session's last hints taken before counting */
                while (tr_time_sec() - settle < 0.02) {
                }
                const uint64_t h_before = pool_hints(pool), f_before = tr_q4x_fused_calls();
                tr_kernels_set_active(&counting);
                TR_CHECK(tr_session_eval(s2, tok, N_PROMPT) == 0);
                memcpy(logits[1][0], tr_session_logits(s2), sizeof logits[1][0]);
                for (int i = 0; i < N_SINGLE; i++) {
                    TR_CHECK(tr_session_eval(s2, tok + N_PROMPT + i, 1) == 0);
                    memcpy(logits[1][1 + i], tr_session_logits(s2), sizeof logits[1][0]);
                }
                TR_CHECK(tr_session_eval(s2, tok + N_PROMPT + N_SINGLE, N_SHORT) == 0);
                memcpy(logits[1][1 + N_SINGLE], tr_session_logits(s2), sizeof logits[1][0]);
                tr_kernels_set_active(NULL);
                TR_CHECK_EQ_INT(atomic_load(&n_q4x_prep), sw_preps[si]);
                TR_CHECK(memcmp(logits[0], logits[1], sizeof logits[0]) == 0);
                const double t1 = tr_time_sec();
                while (tr_time_sec() - t1 < 0.02) {
                }
                if (si == 2) TR_CHECK_EQ_INT(pool_hints(pool), h_before);
                if (si == 0 || si == 3) TR_CHECK_EQ_INT(tr_q4x_fused_calls(), f_before);
                else /* gate and up a layer, and q, k and v where they share a shape (no GQA) */
                    TR_CHECK_EQ_INT(tr_q4x_fused_calls() - f_before, (n_head_kv == N_HEAD ? 2 : 1) * LAYERS * evals);
                printf("  q4_k  model, %s's arm B: %llu rows prepared, every pass's logits the engine's bits\n",
                       sw_names[si], (unsigned long long)atomic_load(&n_q4x_prep));
            }
            tr_session_free(s2);
        }
        /* a router kept BF16 gives, pass by pass, the bits of the same model loaded with it F32 (TR_BF16_EXACT=0) */
        if (bf16_router) {
            set_env("TR_BF16_EXACT", "0");
            tr_model *m32 = tr_model_load(path, pool, err, sizeof err);
            set_env("TR_BF16_EXACT", NULL);
            tr_session *s32 = m32 != NULL ? tr_session_create(m32, 0, 0, err, sizeof err) : NULL;
            TR_CHECK(s32 != NULL);
            if (s32 != NULL) {
                TR_CHECK(tr_session_eval(s32, tok, N_PROMPT) == 0);
                memcpy(logits[1][0], tr_session_logits(s32), sizeof logits[1][0]);
                for (int i = 0; i < N_SINGLE; i++) {
                    TR_CHECK(tr_session_eval(s32, tok + N_PROMPT + i, 1) == 0);
                    memcpy(logits[1][1 + i], tr_session_logits(s32), sizeof logits[1][0]);
                }
                TR_CHECK(tr_session_eval(s32, tok + N_PROMPT + N_SINGLE, N_SHORT) == 0);
                memcpy(logits[1][1 + N_SINGLE], tr_session_logits(s32), sizeof logits[1][0]);
                TR_CHECK(memcmp(logits[0], logits[1], sizeof logits[0]) == 0);
                printf("  %-5s model: every pass's logits with the BF16 router the bits of the router kept F32\n", name);
            }
            tr_session_free(s32);
            tr_model_free(m32);
        }
        printf("  %-5s model on tier %-7s %lld products of its type and %lld of the %s router, all through the "
               "active table\n",
               name, g_real->tier, of_type, of_router, bf16_router ? "BF16" : "F32");
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
    test_engine(argv0, TR_TYPE_F32, "f32", N_EMBD, N_FF, 0, N_HEAD_KV);
    test_engine(argv0, TR_TYPE_F16, "f16", N_EMBD, N_FF, 0, N_HEAD_KV);
    test_engine(argv0, TR_TYPE_Q8_0, "q8_0", N_EMBD, N_FF, 0, N_HEAD_KV);
    test_engine(argv0, TR_TYPE_Q8_0, "q8_0", N_EMBD, N_FF, 1, N_HEAD_KV); /* a router whose values fit bf16: kept BF16 */
    test_engine(argv0, TR_TYPE_Q4_K, "q4_k", 256, 256, 0, N_HEAD_KV);   /* rows of whole 256-element blocks; GQA */
    test_engine(argv0, TR_TYPE_Q4_K, "q4_k", 256, 256, 1, N_HEAD);      /* as the real model: router BF16, no GQA */
    test_engine(argv0, TR_TYPE_Q6_K, "q6_k", 256, 256, 0, N_HEAD_KV);
    TR_TEST_EXIT();
}
