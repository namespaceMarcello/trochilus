/* test_base.c — tests for src/base/platform.c, threads.c, cpu.c. */

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* sched_getcpu, to see where a pinned worker really ran */
#endif
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

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

/* Temporary files go next to this binary, as in the other tests: the gcc, clang and ASan builds
 * of this test then never share a file, and can run at once. */
static char g_dir[512] = ".";

static const char *tmp_path(char *out, size_t n, const char *name) {
    snprintf(out, n, "%s/%s", g_dir, name);
    return out;
}

/* ---- tr_file ----------------------------------------------------------- */

static void test_file(void) {
    char path_buf[600];
    const char *path = tmp_path(path_buf, sizeof path_buf, "test_base_tmp.bin");

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
    tr_file *bad = tr_file_open(tmp_path(path_buf, sizeof path_buf, "does_not_exist_xyz.bin"), err, sizeof err);
    TR_CHECK(bad == NULL);
    TR_CHECK(err[0] != '\0');
    if (bad != NULL) tr_file_close(bad);
}

/* ---- tr_file_open_direct: unbuffered reads, and the short-read-at-EOF rule (platform.h Step A,
 * docs/ARCHITECTURE.md Esperti M1). Skips with a message, rather than failing, when this
 * filesystem cannot actually serve an aligned read on it (docker bind mounts, in particular) --
 * proven with a trial read, not merely a successful open, since some filesystems accept
 * FILE_FLAG_NO_BUFFERING/O_DIRECT at open and then cannot service a read on it. */
static void test_file_direct(void) {
    char path_buf[600];
    const char *path = tmp_path(path_buf, sizeof path_buf, "test_base_direct.bin");

    /* not a multiple of TR_FILE_DIRECT_ALIGN: the file's real end falls inside its last aligned
     * sector, exactly the case Step A must tolerate. */
    enum { SIZE = 3 * TR_FILE_DIRECT_ALIGN + 777 };
    unsigned char *data = (unsigned char *)malloc(SIZE);
    TR_CHECK(data != NULL);
    if (data == NULL) return;
    for (int i = 0; i < SIZE; i++) data[i] = (unsigned char)(i * 53u + 7u);

    FILE *wf = fopen(path, "wb");
    TR_CHECK(wf != NULL);
    if (wf != NULL) {
        TR_CHECK(fwrite(data, 1, SIZE, wf) == (size_t)SIZE);
        fclose(wf);
    }

    char err[256];
    tr_file *buffered = tr_file_open(path, err, sizeof err);
    TR_CHECK(buffered != NULL);
    if (buffered != NULL) {
        TR_CHECK_EQ_INT(tr_file_alignment(buffered), 1);
        tr_file_close(buffered);
    }

    err[0] = '\0';
    tr_file *f = tr_file_open_direct(path, err, sizeof err);
    unsigned char *probe = f != NULL ? (unsigned char *)tr_alloc_aligned(TR_FILE_DIRECT_ALIGN, TR_FILE_DIRECT_ALIGN)
                                     : NULL;
    int works = f != NULL && probe != NULL && tr_file_pread(f, probe, TR_FILE_DIRECT_ALIGN, 0) == 0 &&
                memcmp(probe, data, TR_FILE_DIRECT_ALIGN) == 0;
    tr_free_aligned(probe);
    if (!works) {
        printf("  test_file_direct: SKIPPED, '%s' cannot serve an aligned read on this filesystem%s%s\n", path,
               err[0] != '\0' ? ": " : "", err);
        if (f != NULL) tr_file_close(f);
        free(data);
        remove(path);
        return;
    }
    TR_CHECK_EQ_INT(tr_file_alignment(f), TR_FILE_DIRECT_ALIGN);

    /* the last sector: its aligned range runs past the real end of the file -- must still
     * succeed, and return exactly the real bytes that exist (the rest of buf is unspecified) */
    {
        int64_t lo = (SIZE / TR_FILE_DIRECT_ALIGN) * TR_FILE_DIRECT_ALIGN; /* 3 * ALIGN, < SIZE */
        size_t want = 2 * TR_FILE_DIRECT_ALIGN; /* [lo, lo + want) ends well past SIZE */
        unsigned char *buf = (unsigned char *)tr_alloc_aligned(want, TR_FILE_DIRECT_ALIGN);
        TR_CHECK(buf != NULL);
        if (buf != NULL) {
            TR_CHECK_EQ_INT(tr_file_pread(f, buf, want, (uint64_t)lo), 0);
            TR_CHECK(memcmp(buf, data + lo, (size_t)(SIZE - lo)) == 0);
            tr_free_aligned(buf);
        }
    }

    /* entirely past the real end of the file: still not an error on a direct handle */
    {
        uint64_t lo = 4 * (uint64_t)TR_FILE_DIRECT_ALIGN; /* >= SIZE */
        unsigned char *buf = (unsigned char *)tr_alloc_aligned(TR_FILE_DIRECT_ALIGN, TR_FILE_DIRECT_ALIGN);
        TR_CHECK(buf != NULL);
        if (buf != NULL) {
            TR_CHECK_EQ_INT(tr_file_pread(f, buf, TR_FILE_DIRECT_ALIGN, lo), 0);
            tr_free_aligned(buf);
        }
    }

    tr_file_close(f);
    free(data);
    remove(path);
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
    char path_buf[600];
    const char *path = tmp_path(path_buf, sizeof path_buf, "test_base_tmp2.bin");

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

/* An empty range runs no body, on a pool or not (a mutation that let n == 0 through lived:
 * tools/mutate_auto.py, 2026-09-22). */
static int g_empty_calls;
static void empty_fn(void *ctx, int64_t begin, int64_t end, int worker) {
    (void)ctx; (void)begin; (void)end; (void)worker;
    g_empty_calls++;
}
static void test_parallel_empty(tr_pool *p) {
    g_empty_calls = 0;
    tr_parallel_for(p, 0, 1, empty_fn, NULL);
    tr_parallel_for(NULL, 0, 1, empty_fn, NULL);
    tr_parallel_for(p, -3, 1, empty_fn, NULL);
    TR_CHECK_EQ_INT(g_empty_calls, 0);
    tr_parallel_for(p, 1, 0, empty_fn, NULL); /* min_chunk 0 counts as 1: one call */
    TR_CHECK_EQ_INT(g_empty_calls, 1);
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

/* Regression for two dispatch races found on 2026-09-17 (docs/LESSONS.md): a worker
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

/* A pool narrowed to w threads gives work to workers 0..w-1 only and really uses all w of them,
 * still covers every index once, and takes a width out of range as "every thread". A generation
 * switches width thousands of times (prompt wide, decode narrow), so the width changes on every
 * call here: that is where a worker left on a stale job would show (a hang fails by timeout). */
static void test_pool_active(void) {
    static const int widths[] = {3, 8, 1, 5, 2, 8, 4};
    unsigned char count[64], worker_id[64];
    tr_pool *p = tr_pool_create(8);
    TR_CHECK(p != NULL);
    if (p == NULL) return;
    TR_CHECK_EQ_INT(tr_pool_active(p), 8);

    for (int iter = 0; iter < 5000; iter++) {
        int w = widths[iter % 7];
        int64_t n = 1 + (iter * 7) % 64;
        tr_pool_set_active(p, w);
        TR_CHECK_EQ_INT(tr_pool_active(p), w);
        memset(count, 0, sizeof count);
        memset(worker_id, 0, sizeof worker_id);
        cov_ctx ctx = { count, worker_id };
        tr_parallel_for(p, n, 1, cov_fn, &ctx);
        check_coverage(n, count, worker_id, w);
        int last = 0;
        for (int64_t i = 0; i < n; i++)
            if (worker_id[i] > last) last = worker_id[i];
        TR_CHECK_EQ_INT(last, (n < w ? (int)n : w) - 1);
    }

    static const int all[] = {0, -3, 9, 99};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        tr_pool_set_active(p, 2);
        tr_pool_set_active(p, all[i]);
        TR_CHECK_EQ_INT(tr_pool_active(p), 8);
    }
    TR_CHECK_EQ_INT(tr_pool_size(p), 8); /* the width never changes the pool itself */
    tr_pool_set_active(NULL, 3);          /* no pool: nothing to narrow */
    TR_CHECK_EQ_INT(tr_pool_active(NULL), 1);
    tr_pool_destroy(p);
}

/* tr_parallel_for_balanced: every index once whatever the width, n and min_chunk, and the help
 * really happens. The witness: worker 1 holds its first block for 5 ms, so the other workers
 * finish their own chunks and take the rest of worker 1's; indices of chunk 1 run by another
 * worker are counted, and a call where nobody helped fails the test (`helped`, the branch of
 * run_chunk past a thread's own chunk). Red with TR_POOL_BLOCKS=1 (the static split) and with the
 * help loop cut to the own chunk. */
typedef struct {
    unsigned char *count;
    unsigned char *worker_id;
    atomic_int held; /* worker 1 has held its first block */
} slow_ctx;

static void slow_fn(void *ctx_, int64_t begin, int64_t end, int worker) {
    slow_ctx *ctx = ctx_;
    if (worker == 1 && !atomic_exchange(&ctx->held, 1)) {
        double t0 = tr_time_sec();
        while (tr_time_sec() - t0 < 0.005) { }
    }
    for (int64_t i = begin; i < end; i++) {
        ctx->count[i]++;
        ctx->worker_id[i] = (unsigned char)worker;
    }
}

static void test_parallel_balanced(void) {
    static const int widths[] = {8, 3, 2, 8, 5};
    static const int64_t min_chunks[] = {1, 3, 64};
    unsigned char count[3000], worker_id[3000];
    tr_pool *p = tr_pool_create(8);
    TR_CHECK(p != NULL);
    if (p == NULL) return;
    for (int iter = 0; iter < 3000; iter++) {
        int w = widths[iter % 5];
        int64_t n = 1 + (iter * 13) % 3000, min_chunk = min_chunks[iter % 3];
        tr_pool_set_active(p, w);
        memset(count, 0, (size_t)n);
        cov_ctx ctx = { count, worker_id };
        tr_parallel_for_balanced(p, n, min_chunk, cov_fn, &ctx);
        check_coverage(n, count, worker_id, w);
    }
    tr_pool_set_active(p, 8);
    int helped = 0;
    for (int round = 0; round < 3; round++) {
        const int64_t n = 2048; /* 8 chunks of 256, blocks of 4 */
        memset(count, 0, (size_t)n);
        slow_ctx sc;
        sc.count = count;
        sc.worker_id = worker_id;
        atomic_init(&sc.held, 0);
        tr_parallel_for_balanced(p, n, 1, slow_fn, &sc);
        check_coverage(n, count, worker_id, 8);
        for (int64_t i = n / 8; i < 2 * n / 8; i++)
            if (worker_id[i] != 1) helped++;
    }
    TR_CHECK(helped > 0);
    printf("  balanced: %d indices of the held chunk run by the other workers over 3 calls\n", helped);
    tr_pool_destroy(p);
}

/* ---- thread placement ---------------------------------------------------- */

/* Every slot names a different logical processor, and the first physical_cores of them
 * come before any SMT sibling: that ordering is what makes the pin worth its 30%
 * (docs/MEASUREMENTS.md "Dove vanno i thread"). */
static void test_cpu_slots(void) {
    const tr_cpu_info *c = tr_cpu();
    TR_CHECK(c->n_slots >= 0 && c->n_slots <= TR_CPU_MAX_SLOTS);
    if (c->n_slots == 0) {
        printf("thread placement: no topology on this platform, nothing is pinned\n");
        return;
    }
    TR_CHECK(c->n_slots <= c->logical_cores);
    TR_CHECK(c->n_slots >= c->physical_cores || c->physical_cores > TR_CPU_MAX_SLOTS);
    for (int i = 0; i < c->n_slots; i++)
        for (int j = i + 1; j < c->n_slots; j++)
            TR_CHECK(!(c->slot[i].group == c->slot[j].group && c->slot[i].lcpu == c->slot[j].lcpu));
}

/* Where each worker really ran, one index per worker. */
typedef struct {
    int *cpu;
    int n;
} where_ctx;

static int current_cpu(void) {
#if defined(_WIN32)
    return (int)GetCurrentProcessorNumber();
#elif defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

static void where_fn(void *ctx_, int64_t begin, int64_t end, int worker) {
    where_ctx *ctx = ctx_;
    for (int64_t i = begin; i < end; i++)
        if (i < ctx->n) ctx->cpu[i] = current_cpu();
    (void)worker;
}

/* The pin is only worth something if the threads really are where cpu.c put them: ask the
 * OS, from inside the pool, which processor each chunk ran on. */
static void test_pool_pinned(void) {
    const tr_cpu_info *c = tr_cpu();
    if (c->n_slots == 0 || current_cpu() < 0) return; /* platform without placement */

    int n = c->physical_cores < c->n_slots ? c->physical_cores : c->n_slots;
    if (n > 8) n = 8; /* enough to catch a doubled-up core without a long test */
    tr_pool *p = tr_pool_create(n);
    TR_CHECK(p != NULL);
    if (p == NULL) return;

    int cpu_of[8];
    where_ctx ctx = { cpu_of, n };
    for (int i = 0; i < n; i++) cpu_of[i] = -1;
    tr_parallel_for(p, n, 1, where_fn, &ctx); /* one index per worker */
    tr_parallel_for(p, n, 1, where_fn, &ctx); /* again: every worker has started by now */

    for (int i = 0; i < n; i++) {
        /* chunk i is run by worker i, which is pinned to slot i: with TR_POOL_PIN=2 (the
         * default) that is any processor of slot i's physical core, with 1 exactly slot i's */
        int on_its_core = 0;
        for (int s = 0; s < (int)c->slot[i].n_core; s++)
            if (cpu_of[i] == (int)c->slot[i].core[s]) on_its_core = 1;
        TR_CHECK(on_its_core);
        if (!on_its_core)
            printf("  worker %d ran on cpu %d, its core is %d (+%d siblings)\n", i, cpu_of[i],
                   (int)c->slot[i].lcpu, (int)c->slot[i].n_core - 1);
        /* whatever the mode, no two workers may share a physical core */
        for (int j = i + 1; j < n; j++)
            for (int s = 0; s < (int)c->slot[i].n_core; s++)
                TR_CHECK(cpu_of[j] != (int)c->slot[i].core[s]);
    }
    tr_pool_destroy(p);
}

/* The affinity of the calling thread as one comparable value; ok = 0 where it cannot be read. */
typedef struct {
    int ok;
    unsigned group;
    unsigned char mask[128];
} thread_mask;

static thread_mask current_mask(void) {
    thread_mask m;
    memset(&m, 0, sizeof m);
#if defined(_WIN32)
    GROUP_AFFINITY ga;
    if (GetThreadGroupAffinity(GetCurrentThread(), &ga)) {
        m.ok = 1;
        m.group = ga.Group;
        memcpy(m.mask, &ga.Mask, sizeof ga.Mask);
    }
#elif defined(__linux__)
    cpu_set_t s;
    if (sizeof s <= sizeof m.mask && sched_getaffinity(0, sizeof s, &s) == 0) {
        m.ok = 1;
        memcpy(m.mask, &s, sizeof s);
    }
#endif
    return m;
}

static int same_mask(const thread_mask *a, const thread_mask *b) {
    return a->ok && b->ok && a->group == b->group && memcmp(a->mask, b->mask, sizeof a->mask) == 0;
}

/* Two pools alive at once: whatever the order they are destroyed in, the calling thread ends
 * where it started, and stays on its slot while one of them is still alive. Each pool used to
 * save "where the caller was" for itself, so the second saved the first one's pin and, destroyed
 * last, left the caller on one core for the rest of the process (docs/LESSONS.md #61). */
static void test_pool_caller_affinity_any_order(void) {
    thread_mask before = current_mask();
    if (!before.ok || tr_cpu()->n_slots == 0) return; /* platform without placement */

    for (int order = 0; order < 2; order++) {
        tr_pool *a = tr_pool_create(2), *b = tr_pool_create(2);
        TR_CHECK(a != NULL && b != NULL);
        if (a == NULL || b == NULL) {
            tr_pool_destroy(a);
            tr_pool_destroy(b);
            return;
        }
        thread_mask pinned = current_mask();
        tr_pool_destroy(order == 0 ? a : b);
        thread_mask one_left = current_mask();
        TR_CHECK(same_mask(&one_left, &pinned)); /* still worker 0 of the pool that is alive */
        tr_pool_destroy(order == 0 ? b : a);
        thread_mask after = current_mask();
        TR_CHECK(same_mask(&after, &before));
        if (!same_mask(&after, &before))
            printf("  two pools destroyed %s: the caller is left pinned\n",
                   order == 0 ? "in creation order" : "in reverse order");
    }
}

#if defined(__linux__)
static void mask_fn(void *ctx_, int64_t begin, int64_t end, int worker) {
    thread_mask *masks = ctx_;
    (void)begin;
    (void)end;
    masks[worker] = current_mask();
}

/* Runs in a process of its own (test_pool_oversubscribed below): tr_cpu() is detected once,
 * so the process has to be confined to two processors before anything asks for it. */
static int oversubscribed_child(void) {
    cpu_set_t allowed, two;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return 0;
    CPU_ZERO(&two);
    int n = 0;
    for (int c = 0; c < CPU_SETSIZE && n < 2; c++)
        if (CPU_ISSET(c, &allowed)) {
            CPU_SET(c, &two);
            n++;
        }
    if (n < 2 || sched_setaffinity(0, sizeof two, &two) != 0) {
        printf("  oversubscribed pool: fewer than 2 processors, skipped\n");
        return 0;
    }
    setenv("TR_POOL_PIN", "1", 1); /* one processor per thread: the caller's pin is one cpu, not both */
    thread_mask process = current_mask();

    if (tr_cpu()->n_slots == 0) { /* no topology in /sys: nothing is pinned, nothing to inherit */
        printf("  oversubscribed pool: no topology on this machine, skipped\n");
        return 0;
    }
    TR_CHECK_EQ_INT(tr_cpu()->n_slots, 2);
    tr_pool *p = tr_pool_create(4);
    TR_CHECK(p != NULL);
    if (p == NULL) return 1;
    thread_mask masks[4];
    memset(masks, 0, sizeof masks);
    tr_parallel_for(p, 4, 1, mask_fn, masks);
    tr_parallel_for(p, 4, 1, mask_fn, masks); /* again: every worker has started by now */
    /* workers 0 and 1 have a slot each; 2 and 3 have none and belong to the scheduler, on
     * every processor the process may use, not on the one the caller was pinned to */
    TR_CHECK(!same_mask(&masks[0], &process));
    TR_CHECK(!same_mask(&masks[1], &process));
    TR_CHECK(!same_mask(&masks[0], &masks[1]));
    TR_CHECK(same_mask(&masks[2], &process));
    TR_CHECK(same_mask(&masks[3], &process));
    tr_pool_destroy(p);
    return tr_test_failures != 0;
}
#endif

/* More threads than processors: the threads without a slot are left to the scheduler. On Linux
 * a new thread inherits its creator's affinity, and the creator had just pinned itself to slot
 * 0: they all piled up on the caller's core (docs/LESSONS.md #62). */
static void test_pool_oversubscribed(const char *argv0) {
#if defined(__linux__)
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "\"%s\" --oversubscribed", argv0);
    TR_CHECK_EQ_INT(system(cmd), 0);
#else
    (void)argv0; /* Windows gives a new thread the affinity of the process, not of its creator */
#endif
}

/* ---- tr_thread and tr_monitor ----------------------------------------------
 * One item at a time handed to a thread of our own through a monitor, 1000 times: every item
 * arrives, in order, once; the thread finishes slowly after its last item, and tr_thread_join
 * returns only after it has (under TSan, the plain `finished` read after the join is also the
 * proof that join orders it). tr_thread_join(NULL) and tr_monitor_free(NULL) do nothing. */

typedef struct {
    tr_monitor *m;
    int item, stop, received, in_order, finished;
    int64_t sum;
} handoff;

static void handoff_consumer(void *arg) {
    handoff *h = (handoff *)arg;
    tr_monitor_lock(h->m);
    for (;;) {
        while (h->item == 0 && !h->stop) tr_monitor_wait(h->m);
        if (h->item == 0) break; /* stop, and nothing left */
        if (h->item != h->received + 1) h->in_order = 0;
        h->sum += h->item;
        h->received++;
        h->item = 0;
        tr_monitor_broadcast(h->m);
    }
    tr_monitor_unlock(h->m);
    double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < 0.05) {
    }
    h->finished = 1;
}

static int g_thread_cases = 0;

static void test_thread_monitor(void) {
    handoff h;
    memset(&h, 0, sizeof h);
    h.in_order = 1;
    h.m = tr_monitor_create();
    TR_CHECK(h.m != NULL);
    if (h.m == NULL) return;
    tr_thread *t = tr_thread_start(handoff_consumer, &h);
    TR_CHECK(t != NULL);
    if (t == NULL) {
        tr_monitor_free(h.m);
        return;
    }
    for (int k = 1; k <= 1000; k++) {
        tr_monitor_lock(h.m);
        while (h.item != 0) tr_monitor_wait(h.m);
        h.item = k;
        tr_monitor_broadcast(h.m);
        tr_monitor_unlock(h.m);
    }
    tr_monitor_lock(h.m);
    while (h.item != 0) tr_monitor_wait(h.m);
    h.stop = 1;
    tr_monitor_broadcast(h.m);
    tr_monitor_unlock(h.m);
    tr_thread_join(t);
    TR_CHECK_EQ_INT(h.finished, 1); /* join waited for the thread's slow finish */
    TR_CHECK_EQ_INT(h.received, 1000);
    TR_CHECK_EQ_INT(h.sum, 500500);
    TR_CHECK(h.in_order);
    tr_monitor_free(h.m);
    tr_thread_join(NULL);
    tr_monitor_free(NULL);
    g_thread_cases++;
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

int main(int argc, char **argv) {
    if (argc > 0) {
        const char *sl = strrchr(argv[0], '/'), *bs = strrchr(argv[0], '\\');
        if (bs != NULL && (sl == NULL || bs > sl)) sl = bs;
        if (sl != NULL) snprintf(g_dir, sizeof g_dir, "%.*s", (int)(sl - argv[0]), argv[0]);
    }
#if defined(__linux__)
    if (argc > 1 && strcmp(argv[1], "--oversubscribed") == 0) return oversubscribed_child();
#endif
    test_file();
    test_file_direct();
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
        test_parallel_empty(p);
        test_parallel_nested(p);
        test_parallel_repeated(p);
        tr_pool_destroy(p);
    }

    test_parallel_varying_chunks();
    test_pool_active();
    test_parallel_balanced();
    test_cpu_slots();
    test_pool_pinned();
    test_pool_caller_affinity_any_order();
    test_pool_oversubscribed(argc > 0 ? argv[0] : "");
    test_thread_monitor();
    TR_CHECK(g_thread_cases > 0);
    test_cpu();

    TR_TEST_EXIT();
}
