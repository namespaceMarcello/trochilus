/* test_hot.c — the hot zone contract (docs/ARCHITETTURA.md §Zona calda) on a model big
 * enough for the pool to split real work:
 *   - generating tokens performs zero memory allocations and frees (counted by linking
 *     the engine's allocator calls through --wrap, see the Makefile);
 *   - logits are bit-identical for every pool size, with and without the profiler. */
#include <stdatomic.h>

#include "test.h"
#include "synth_olmoe.h"
#include "../src/models/model.h"

static atomic_ulong g_allocs;

#ifndef TR_NO_WRAP
void *__real_malloc(size_t n);
void *__real_calloc(size_t n, size_t m);
void *__real_realloc(void *p, size_t n);
void __real_free(void *p);
void *__wrap_malloc(size_t n) { atomic_fetch_add(&g_allocs, 1); return __real_malloc(n); }
void *__wrap_calloc(size_t n, size_t m) { atomic_fetch_add(&g_allocs, 1); return __real_calloc(n, m); }
void *__wrap_realloc(void *p, size_t n) { atomic_fetch_add(&g_allocs, 1); return __real_realloc(p, n); }
void __wrap_free(void *p) { atomic_fetch_add(&g_allocs, 1); __real_free(p); }
#if defined(_WIN32)
void *__real__aligned_malloc(size_t n, size_t a);
void __real__aligned_free(void *p);
void *__wrap__aligned_malloc(size_t n, size_t a) { atomic_fetch_add(&g_allocs, 1); return __real__aligned_malloc(n, a); }
void __wrap__aligned_free(void *p) { atomic_fetch_add(&g_allocs, 1); __real__aligned_free(p); }
#else
int __real_posix_memalign(void **p, size_t a, size_t n);
int __wrap_posix_memalign(void **p, size_t a, size_t n) { atomic_fetch_add(&g_allocs, 1); return __real_posix_memalign(p, a, n); }
#endif
#endif

/* 2 layers, n_embd 256, 4 heads (2 kv), n_ff 512, 8 experts (2 used), vocab 64, context 32:
 * every matmul has hundreds of rows, so pools of 2+ threads really split them. */
static const synth_params P = {2, 256, 4, 2, 512, 8, 2, 64, 32, TR_TYPE_F32};
/* one token per pass, then a short pass of 4 rows, then a pass long enough for the pool to split
 * the work that belongs to one token (norms, RoPE, cache write, routing, expert rows):
 * ThreadSanitizer runs this test (make check) */
enum { N_TOKENS = 32, SHORT_PASS = 4, LAST_PASS = 20 };

/* Generates N_TOKENS with a pool of `threads`; logits of the last token go to out. expert_budget:
 * see tr_model_load_budget (0: automatic, same as tr_model_load). Returns the allocations
 * counted while generating (after the first token). */
static unsigned long run(const char *path, int threads, int profile, uint64_t expert_budget, float *out) {
    tr_pool *pool = tr_pool_create(threads);
    char err[256];
    tr_model *model = pool ? tr_model_load_budget(path, pool, expert_budget, err, sizeof err) : NULL;
    tr_session *s = model ? tr_session_create(model, 0, 0, err, sizeof err) : NULL;
    TR_CHECK(s != NULL);
    unsigned long allocs = 0;
    if (s != NULL) {
        tr_session_prof(s)->enabled = profile;
        int32_t tok[N_TOKENS];
        for (int i = 0; i < N_TOKENS; i++) tok[i] = (i * 7 + 3) % (int32_t)P.vocab;
        TR_CHECK(tr_session_eval(s, tok, 1) == 0);
        unsigned long before = atomic_load(&g_allocs);
        for (int i = 1; i < N_TOKENS - LAST_PASS - SHORT_PASS; i++) TR_CHECK(tr_session_eval(s, tok + i, 1) == 0);
        TR_CHECK(tr_session_eval(s, tok + N_TOKENS - LAST_PASS - SHORT_PASS, SHORT_PASS) == 0);
        TR_CHECK(tr_session_eval(s, tok + N_TOKENS - LAST_PASS, LAST_PASS) == 0);
        allocs = atomic_load(&g_allocs) - before;
        memcpy(out, tr_session_logits(s), P.vocab * sizeof(float));
        tr_session_free(s);
    } else {
        fprintf(stderr, "setup failed: %s\n", err);
    }
    tr_model_free(model);
    tr_pool_destroy(pool);
    return allocs;
}

int main(int argc, char **argv) {
    char path[512];
    TR_CHECK(synth_write(&P, argc > 0 ? argv[0] : "", "test_hot_tmp.gguf", path, sizeof path) == 0);

#ifndef TR_NO_WRAP
    /* the counter must see an allocation, or a zero below proves nothing */
    static void *volatile probe; /* volatile: the compiler may not drop the pair */
    unsigned long c0 = atomic_load(&g_allocs);
    probe = malloc(16);
    free(probe);
    TR_CHECK_EQ_INT((int)(atomic_load(&g_allocs) - c0), 2);
#else
    printf("  allocation count skipped: this linker has no --wrap\n");
#endif

    static const int threads[] = {1, 2, 3, 8};
    float ref[64], got[64];
    run(path, 1, 0, 0, ref);
    for (size_t t = 0; t < sizeof threads / sizeof threads[0]; t++) {
        for (int profile = 0; profile < 2; profile++) {
            unsigned long allocs = run(path, threads[t], profile, 0, got);
#ifndef TR_NO_WRAP
            TR_CHECK_EQ_INT((int)allocs, 0);
#endif
            TR_CHECK(memcmp(ref, got, sizeof ref) == 0);
            if (allocs != 0 || memcmp(ref, got, sizeof ref) != 0)
                printf("  threads %d profile %d: %lu allocations, logits %s\n", threads[t], profile, allocs,
                       memcmp(ref, got, sizeof ref) == 0 ? "identical" : "DIFFERENT");
        }
    }
    printf("  %d tokens (one a pass, then passes of %d and %d), pools of 1/2/3/8 threads, profiler off/on: no "
           "allocations, identical logits\n",
           N_TOKENS, SHORT_PASS, LAST_PASS);

    /* the shared expert store's disk reads (Esperti M1) must also allocate nothing: the smallest
     * store that can run ("min"), so most tokens miss and evict, not just the resident path above */
    float got_min[64];
    unsigned long allocs_min = run(path, 8, 0, UINT64_MAX, got_min);
#ifndef TR_NO_WRAP
    TR_CHECK_EQ_INT((int)allocs_min, 0);
#endif
    TR_CHECK(memcmp(ref, got_min, sizeof ref) == 0);
    printf("  same, under the smallest expert store (--expert-budget min): no allocations, identical logits\n");

    remove(path);
    TR_TEST_EXIT();
}
