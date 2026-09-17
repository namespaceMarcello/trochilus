/* test_base.c — tests for src/base/platform.c, threads.c, cpu.c. */

#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE
#endif

#include "test.h"

#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/base/cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static void ensure_dir(const char *path) {
#if defined(_WIN32)
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

/* ---- tr_file ----------------------------------------------------------- */

static void test_file(void) {
    ensure_dir("build");
    ensure_dir("build/tests");
    const char *path = "build/tests/test_base_tmp.bin";

    unsigned char data[4096];
    for (size_t i = 0; i < sizeof data; i++) data[i] = (unsigned char)(i * 37u + 11u);

    FILE *wf = fopen(path, "wb");
    TR_CHECK(wf != NULL);
    if (wf != NULL) {
        TR_CHECK(fwrite(data, 1, sizeof data, wf) == sizeof data);
        fclose(wf);
    }

    char err[256];
    err[0] = '\0';
    tr_file *f = tr_file_open(path, err, sizeof err);
    TR_CHECK(f != NULL);
    if (f == NULL) return;

    TR_CHECK_EQ_INT(tr_file_size(f), (int64_t)sizeof data);

    unsigned char buf[64];

    /* pread at the start */
    memset(buf, 0, sizeof buf);
    TR_CHECK_EQ_INT(tr_file_pread(f, buf, sizeof buf, 0), 0);
    TR_CHECK(memcmp(buf, data, sizeof buf) == 0);

    /* pread in the middle */
    memset(buf, 0, sizeof buf);
    TR_CHECK_EQ_INT(tr_file_pread(f, buf, sizeof buf, 1000), 0);
    TR_CHECK(memcmp(buf, data + 1000, sizeof buf) == 0);

    /* pread ending exactly at EOF */
    memset(buf, 0, sizeof buf);
    size_t last_off = sizeof data - sizeof buf;
    TR_CHECK_EQ_INT(tr_file_pread(f, buf, sizeof buf, last_off), 0);
    TR_CHECK(memcmp(buf, data + last_off, sizeof buf) == 0);

    /* pread spanning past EOF -> -1 */
    TR_CHECK_EQ_INT(tr_file_pread(f, buf, sizeof buf, sizeof data - 10), -1);

    /* pread entirely past EOF -> -1 */
    TR_CHECK_EQ_INT(tr_file_pread(f, buf, sizeof buf, sizeof data + 100), -1);

    tr_file_close(f);

    /* opening a nonexistent file fails and reports an error */
    err[0] = '\0';
    tr_file *bad = tr_file_open("build/tests/does_not_exist_xyz.bin", err, sizeof err);
    TR_CHECK(bad == NULL);
    TR_CHECK(err[0] != '\0');
    if (bad != NULL) tr_file_close(bad);
}

/* Positional reads on the same tr_file from several threads at once must not
 * corrupt each other (this is the whole point of the OVERLAPPED-offset /
 * pread design). Each chunk owns a distinct slice of the output, so this is
 * race-free without atomics, same as the parallel_for coverage checks below. */
typedef struct {
    const tr_file *f;
    const unsigned char *expected;
    int64_t stride;
    int *ok; /* one slot per chunk index; only the owning chunk writes it */
} pread_ctx;

static void pread_fn(void *ctx_, int64_t begin, int64_t end, int worker) {
    pread_ctx *ctx = ctx_;
    unsigned char buf[64];
    int chunk_ok = 1;
    for (int64_t i = begin; i < end; i++) {
        uint64_t off = (uint64_t)(i * ctx->stride);
        if (tr_file_pread(ctx->f, buf, sizeof buf, off) != 0) { chunk_ok = 0; continue; }
        if (memcmp(buf, ctx->expected + off, sizeof buf) != 0) chunk_ok = 0;
    }
    ctx->ok[worker] = chunk_ok;
}

static void test_file_concurrent(tr_pool *p) {
    ensure_dir("build");
    ensure_dir("build/tests");
    const char *path = "build/tests/test_base_tmp2.bin";

    unsigned char data[8192];
    for (size_t i = 0; i < sizeof data; i++) data[i] = (unsigned char)(i * 13u + 7u);

    FILE *wf = fopen(path, "wb");
    TR_CHECK(wf != NULL);
    if (wf == NULL) return;
    TR_CHECK(fwrite(data, 1, sizeof data, wf) == sizeof data);
    fclose(wf);

    tr_file *f = tr_file_open(path, NULL, 0);
    TR_CHECK(f != NULL);
    if (f == NULL) return;

    int pool_size = tr_pool_size(p);
    int *ok = malloc(sizeof(int) * (size_t)pool_size);
    for (int i = 0; i < pool_size; i++) ok[i] = -1;

    int64_t stride = 64;
    int64_t n = (int64_t)(sizeof data / stride) - 1; /* keep every read in-bounds */
    pread_ctx ctx = { f, data, stride, ok };
    tr_parallel_for(p, n, 1, pread_fn, &ctx);

    int all_ok = 1;
    for (int i = 0; i < pool_size; i++) {
        if (ok[i] == 0) all_ok = 0;
    }
    TR_CHECK(all_ok);

    free(ok);
    tr_file_close(f);
}

/* ---- aligned alloc ------------------------------------------------------ */

static void test_alloc(void) {
    size_t aligns[] = { 16, 64, 4096 };
    for (size_t i = 0; i < sizeof aligns / sizeof aligns[0]; i++) {
        size_t a = aligns[i];
        void *p = tr_alloc_aligned(1024, a);
        TR_CHECK(p != NULL);
        if (p != NULL) {
            TR_CHECK_EQ_INT((int64_t)((uintptr_t)p % a), 0);
            memset(p, 0xAB, 1024);
            tr_free_aligned(p);
        }
    }
}

/* ---- time / mem --------------------------------------------------------- */

static void test_time(void) {
    double t0 = tr_time_sec();
    volatile long sum = 0;
    for (long i = 0; i < 2000000; i++) sum += i;
    double t1 = tr_time_sec();
    TR_CHECK(t1 >= t0);
    TR_CHECK(t1 - t0 < 30.0); /* sanity: not stuck, not running backwards */
    (void)sum;
}

static void test_mem(void) {
    tr_meminfo mi;
    int rc = tr_mem_info(&mi);
    TR_CHECK_EQ_INT(rc, 0);
    if (rc == 0) {
        TR_CHECK(mi.total_bytes > 0);
        TR_CHECK(mi.available_bytes <= mi.total_bytes);
    }
}

/* ---- thread pool --------------------------------------------------------- */

static void test_pool_sizes(void) {
    int sizes[] = { 1, 2, 3, 8 };
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        tr_pool *p = tr_pool_create(sizes[i]);
        TR_CHECK(p != NULL);
        if (p != NULL) {
            TR_CHECK_EQ_INT(tr_pool_size(p), sizes[i]);
            tr_pool_destroy(p);
        }
    }

    int nonpositive[] = { 0, -1, -5 };
    for (size_t i = 0; i < sizeof nonpositive / sizeof nonpositive[0]; i++) {
        tr_pool *p = tr_pool_create(nonpositive[i]);
        TR_CHECK(p != NULL);
        if (p != NULL) {
            TR_CHECK_EQ_INT(tr_pool_size(p), tr_cpu()->physical_cores);
            tr_pool_destroy(p);
        }
    }
}

/* Coverage context: count[i] must end up == 1 for every index (no gaps, no
 * double writes) and worker_id[i] must be a valid worker for that pool.
 * Race-free because tr_parallel_for hands each chunk a disjoint [begin, end). */
typedef struct {
    unsigned char *count;
    unsigned char *worker_id;
} cov_ctx;

static void cov_fn(void *ctx_, int64_t begin, int64_t end, int worker) {
    cov_ctx *ctx = ctx_;
    for (int64_t i = begin; i < end; i++) {
        ctx->count[i]++;
        ctx->worker_id[i] = (unsigned char)worker;
    }
}

static void check_coverage(int64_t n, const unsigned char *count, const unsigned char *worker_id,
                            int pool_size) {
    int covered_once = 1;
    int workers_in_range = 1;
    for (int64_t i = 0; i < n; i++) {
        if (count[i] != 1) covered_once = 0;
        if (worker_id[i] >= pool_size) workers_in_range = 0;
    }
    TR_CHECK(covered_once);
    TR_CHECK(workers_in_range);
}

static void test_parallel_coverage(tr_pool *p) {
    int64_t ns[] = { 0, 1, 7, 1000, 100003 };
    int64_t min_chunks[] = { 1, 64, 1000000 };
    int pool_size = tr_pool_size(p);

    for (size_t ni = 0; ni < sizeof ns / sizeof ns[0]; ni++) {
        for (size_t mi = 0; mi < sizeof min_chunks / sizeof min_chunks[0]; mi++) {
            int64_t n = ns[ni];
            int64_t min_chunk = min_chunks[mi];

            if (n == 0) {
                /* Nothing should run; just check it does not crash or hang. */
                cov_ctx ctx = { NULL, NULL };
                tr_parallel_for(p, n, min_chunk, cov_fn, &ctx);
                continue;
            }

            unsigned char *count = calloc((size_t)n, 1);
            unsigned char *worker_id = calloc((size_t)n, 1);
            TR_CHECK(count != NULL && worker_id != NULL);
            if (count == NULL || worker_id == NULL) { free(count); free(worker_id); continue; }

            cov_ctx ctx = { count, worker_id };
            tr_parallel_for(p, n, min_chunk, cov_fn, &ctx);

            check_coverage(n, count, worker_id, pool_size);

            free(count);
            free(worker_id);
        }
    }
}

static void test_parallel_null_pool(void) {
    int64_t n = 500;
    unsigned char *count = calloc((size_t)n, 1);
    unsigned char *worker_id = calloc((size_t)n, 1);
    TR_CHECK(count != NULL && worker_id != NULL);
    if (count != NULL && worker_id != NULL) {
        cov_ctx ctx = { count, worker_id };
        tr_parallel_for(NULL, n, 1, cov_fn, &ctx);
        check_coverage(n, count, worker_id, 1);
        for (int64_t i = 0; i < n; i++) TR_CHECK(worker_id[i] == 0);
    }
    free(count);
    free(worker_id);
}

/* Nested parallel_for: outer_fn runs on every outer chunk (one per worker id,
 * since min_chunk=1 and n == pool size) and, from inside that body, launches
 * a fresh inner parallel_for over its own local array. The nested call must
 * run (not hang) and must cover its range exactly once -- and it does so
 * serially per the documented contract, so its own worker id is whatever the
 * outer chunk's was. Each outer chunk writes only its own ok[worker] slot. */
typedef struct {
    tr_pool *pool;
    int *ok;
} nest_ctx;

typedef struct {
    unsigned char *marks;
} inner_ctx;

static void inner_fn(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)worker;
    inner_ctx *ic = ctx_;
    for (int64_t i = begin; i < end; i++) ic->marks[i]++;
}

static void outer_fn(void *ctx_, int64_t begin, int64_t end, int worker) {
    (void)begin;
    (void)end;
    nest_ctx *nc = ctx_;

    unsigned char local[50];
    memset(local, 0, sizeof local);
    inner_ctx ic = { local };
    tr_parallel_for(nc->pool, (int64_t)(sizeof local), 1, inner_fn, &ic);

    int ok = 1;
    for (size_t i = 0; i < sizeof local; i++) {
        if (local[i] != 1) { ok = 0; break; }
    }
    nc->ok[worker] = ok;
}

static void test_parallel_nested(tr_pool *p) {
    int pool_size = tr_pool_size(p);
    int *ok = malloc(sizeof(int) * (size_t)pool_size);
    for (int i = 0; i < pool_size; i++) ok[i] = -1;

    nest_ctx nc = { p, ok };
    int64_t n = pool_size; /* one chunk per worker id, so every worker runs a nested call */
    tr_parallel_for(p, n, 1, outer_fn, &nc);

    int all_ok = 1;
    for (int i = 0; i < pool_size; i++) {
        if (ok[i] == 0) all_ok = 0;
    }
    TR_CHECK(all_ok);

    free(ok);
}

static void test_parallel_repeated(tr_pool *p) {
    const int64_t n = 37;
    unsigned char count[37];
    unsigned char worker_id[37];
    int pool_size = tr_pool_size(p);

    for (int iter = 0; iter < 1000; iter++) {
        memset(count, 0, sizeof count);
        memset(worker_id, 0, sizeof worker_id);
        cov_ctx ctx = { count, worker_id };
        tr_parallel_for(p, n, 1, cov_fn, &ctx);
        check_coverage(n, count, worker_id, pool_size);
    }
}

/* Regression for two dispatch races found on 2026-09-17 (docs/LEZIONI.md): a worker
 * that starts after the first job, and a worker left on a stale generation, both only
 * showed up when the number of chunks changes from one call to the next. 20000 calls
 * with n cycling through 1..64 on fresh pools of several sizes, each must cover every
 * index exactly once and return (a hang fails the test by timeout). */
static void test_parallel_varying_chunks(void) {
    static const int sizes[] = {2, 3, 8, 16};
    unsigned char count[64], worker_id[64];
    for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        tr_pool *p = tr_pool_create(sizes[si]);
        TR_CHECK(p != NULL);
        if (p == NULL) continue;
        for (int iter = 0; iter < 5000; iter++) {
            int64_t n = 1 + (iter * 7) % 64;
            memset(count, 0, sizeof count);
            cov_ctx ctx = { count, worker_id };
            tr_parallel_for(p, n, 1, cov_fn, &ctx);
            check_coverage(n, count, worker_id, tr_pool_size(p));
        }
        tr_pool_destroy(p);
    }
}

/* ---- cpu ----------------------------------------------------------------- */

static void test_cpu(void) {
    const tr_cpu_info *c = tr_cpu();
    TR_CHECK(c->logical_cores >= 1);
    TR_CHECK(c->physical_cores >= 1);
    TR_CHECK(c->physical_cores <= c->logical_cores);

    char buf[256];
    tr_cpu_describe(c, buf, sizeof buf);
    TR_CHECK(buf[0] != '\0');
    printf("tr_cpu_describe: %s\n", buf);

    /* TR_CPU_MAX is read once and cached inside tr_cpu(): by the time this
     * process runs, tr_cpu() has already been called (tr_pool_create above),
     * so a different TR_CPU_MAX value cannot be exercised in-process. Manually
     * verified instead: `TR_CPU_MAX=scalar ./test_base` and checking that the
     * printed tr_cpu_describe line carries no SIMD flags after "threads,". */
}

int main(void) {
    test_file();
    test_alloc();
    test_time();
    test_mem();
    test_pool_sizes();

    tr_pool *p = tr_pool_create(0);
    TR_CHECK(p != NULL);
    if (p != NULL) {
        test_file_concurrent(p);
        test_parallel_coverage(p);
        test_parallel_null_pool();
        test_parallel_nested(p);
        test_parallel_repeated(p);
        tr_pool_destroy(p);
    }

    test_parallel_varying_chunks();
    test_cpu();

    TR_TEST_EXIT();
}
