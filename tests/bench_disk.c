/* bench_disk.c — what the disk of this machine gives to a reader of experts (docs/MISURE.md
 * question 16, the ground of milestone M1: experts read from the disk when they are needed).
 *
 * An expert that is not in RAM is three matrices somewhere in a file of several GB, wanted now,
 * at a position nobody can guess from the one before. So: blocks as big as one expert matrix
 * (and as big as three, as if an expert's matrices were stored together), at random aligned
 * positions of a big file, with 1, 2, 4, 8 and 16 readers at once.
 *
 *   bench_disk <big file> [--block <bytes>] [--runs N]
 *
 *   direct   the file opened so that the system keeps no copy (FILE_FLAG_NO_BUFFERING, O_DIRECT,
 *            F_NOCACHE): what the disk itself gives, every time, also the first
 *   cached   the engine's own tr_file_pread, through the system's cache, the same blocks twice:
 *            `first` may come from the disk or from a copy the system already has, `again`
 *            comes from RAM. What relying on the system's cache would give at best.
 *
 * Before any number: the direct reader and tr_file_pread return the same bytes at the same
 * positions, or the benchmark stops (a fast reader of the wrong bytes measures nothing).
 * Every line is the median of N runs (default 5) after one warm-up, with min, max and spread
 * (max - min) / median. The whole program stops launching runs after 50 s (docs/ARCHITETTURA.md,
 * safety of the machine: a benchmark run stays under 60 s) and says so. Reads only. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* O_DIRECT */
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/platform.h"
#include "../src/base/threads.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

enum { MAX_READERS = 16, MAX_BLOCKS = 256, ALIGN = 4096, MAX_RUNS = 15 };
#define RUN_BYTES (128u << 20) /* a run reads about this much */
#define TIME_LIMIT_SEC 50.0

/* ---- a file the system keeps no copy of ---- */
typedef struct {
#if defined(_WIN32)
    HANDLE h;
    HANDLE ev[MAX_READERS]; /* one event per reader: a read waits on its own */
#else
    int fd;
#endif
} raw_file;

static int raw_open(raw_file *f, const char *path) {
#if defined(_WIN32)
    wchar_t wpath[1024];
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024) == 0) return -1;
    f->h = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                       FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (f->h == INVALID_HANDLE_VALUE) return -1;
    for (int i = 0; i < MAX_READERS; i++) {
        f->ev[i] = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (f->ev[i] == NULL) return -1;
    }
    return 0;
#elif defined(__APPLE__)
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) return -1;
    return fcntl(f->fd, F_NOCACHE, 1) == -1 ? -1 : 0;
#else
    f->fd = open(path, O_RDONLY | O_DIRECT);
    return f->fd < 0 ? -1 : 0;
#endif
}

static void raw_close(raw_file *f) {
#if defined(_WIN32)
    for (int i = 0; i < MAX_READERS; i++)
        if (f->ev[i] != NULL) CloseHandle(f->ev[i]);
    if (f->h != INVALID_HANDLE_VALUE) CloseHandle(f->h);
#else
    if (f->fd >= 0) close(f->fd);
#endif
}

/* n and offset multiples of ALIGN, buf aligned to ALIGN */
static int raw_read(raw_file *f, int reader, void *buf, size_t n, uint64_t offset) {
#if defined(_WIN32)
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)(offset & 0xFFFFFFFFu);
        ov.OffsetHigh = (DWORD)(offset >> 32);
        ov.hEvent = f->ev[reader];
        DWORD got = 0;
        if (!ReadFile(f->h, p, (DWORD)n, &got, &ov)) {
            if (GetLastError() != ERROR_IO_PENDING) return -1;
            if (!GetOverlappedResult(f->h, &ov, &got, TRUE)) return -1;
        }
        if (got == 0 || got % ALIGN != 0) return -1;
        p += got;
        offset += got;
        n -= got;
    }
    return 0;
#else
    (void)reader;
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        ssize_t got = pread(f->fd, p, n, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return -1;
        p += got;
        offset += (uint64_t)got;
        n -= (size_t)got;
    }
    return 0;
#endif
}

/* ---- Step F (docs/LEZIONI.md #94): tr_file_pread creates and destroys a Windows event on every
 * call (platform.c); measured against the same read through one event kept alive across every
 * call, on a small block the OS already has cached, so the event's own cost is what shows up, not
 * the disk's. POSIX pread(2) has no per-call kernel object, so there is nothing to measure there. */
#if defined(_WIN32)
static double bench_pread_default(const tr_file *file, void *buf, uint64_t offset, int iters) {
    for (int i = 0; i < iters; i++)
        if (tr_file_pread(file, buf, ALIGN, offset) != 0) return -1.0;
    double t0 = tr_time_sec();
    for (int i = 0; i < iters; i++)
        if (tr_file_pread(file, buf, ALIGN, offset) != 0) return -1.0;
    return (tr_time_sec() - t0) / iters;
}

/* Same read, ReadFile directly with one OVERLAPPED/event kept alive across every call instead of
 * tr_file_pread's own fresh CreateEventW/CloseHandle per call: isolates that one cost. */
static double bench_pread_reused_event(const char *path, void *buf, uint64_t offset, int iters) {
    wchar_t wpath[1024];
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024) == 0) return -1.0;
    HANDLE h = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1.0;
    HANDLE ev = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (ev == NULL) {
        CloseHandle(h);
        return -1.0;
    }

    double per_call = -1.0;
    for (int pass = 0; pass < 2; pass++) { /* pass 0: warm-up, pass 1: timed */
        double t0 = tr_time_sec();
        int ok_all = 1;
        for (int i = 0; i < iters; i++) {
            ResetEvent(ev);
            OVERLAPPED ov;
            memset(&ov, 0, sizeof ov);
            ov.Offset = (DWORD)(offset & 0xFFFFFFFFu);
            ov.OffsetHigh = (DWORD)(offset >> 32);
            ov.hEvent = ev;
            DWORD got = 0;
            BOOL ok = ReadFile(h, buf, ALIGN, &got, &ov);
            if (!ok) {
                if (GetLastError() != ERROR_IO_PENDING || !GetOverlappedResult(h, &ov, &got, TRUE)) {
                    ok_all = 0;
                    break;
                }
            }
            if (got != ALIGN) {
                ok_all = 0;
                break;
            }
        }
        if (!ok_all) {
            per_call = -1.0;
            break;
        }
        if (pass == 1) per_call = (tr_time_sec() - t0) / iters;
    }

    CloseHandle(ev);
    CloseHandle(h);
    return per_call;
}

/* ~14 us is 1% of a 2 MiB read at this disk's own measured speed (docs/MISURE.md): only above
 * that bar would the per-call event be worth removing from tr_file_pread's hot path. */
#define EVENT_OVERHEAD_BAR_SEC 14e-6

static void bench_event_overhead(const char *path, tr_file *file) {
    enum { ITERS = 20000 };
    void *buf = tr_alloc_aligned(ALIGN, ALIGN);
    if (buf == NULL) return;
    double per_default = bench_pread_default(file, buf, 0, ITERS);
    double per_reused = bench_pread_reused_event(path, buf, 0, ITERS);
    if (per_default >= 0.0 && per_reused >= 0.0) {
        double diff = per_default - per_reused;
        printf("event overhead: tr_file_pread %.2f us/call, one event reused %.2f us/call, "
               "difference %.2f us/call (bar %.2f us)\n",
               per_default * 1e6, per_reused * 1e6, diff * 1e6, EVENT_OVERHEAD_BAR_SEC * 1e6);
    } else {
        printf("event overhead: could not measure (a read failed)\n");
    }
    tr_free_aligned(buf);
}
#endif

/* ---- one run: n_blocks blocks at offs[], each reader into its own buffer ---- */
typedef struct {
    raw_file *raw;       /* direct, or */
    const tr_file *file; /* through the system's cache */
    size_t block;
    const uint64_t *offs;
    void *buf[MAX_READERS];
    uint64_t *sums; /* first + last 8 bytes of every block: the bytes really arrived */
    int failed;
} run_ctx;

static void read_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    run_ctx *c = (run_ctx *)ctx_;
    for (int64_t i = begin; i < end; i++) {
        int rc = c->raw != NULL ? raw_read(c->raw, worker, c->buf[worker], c->block, c->offs[i])
                                : tr_file_pread(c->file, c->buf[worker], c->block, c->offs[i]);
        if (rc != 0) {
            c->failed = 1;
            continue;
        }
        uint64_t a, b;
        memcpy(&a, c->buf[worker], 8);
        memcpy(&b, (const unsigned char *)c->buf[worker] + c->block - 8, 8);
        c->sums[i] = a + b;
    }
}

static uint32_t rng_state = 1;
static uint32_t rng(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return rng_state >> 8;
}

static void pick_offsets(uint64_t *offs, int n, uint64_t file_size, size_t block) {
    uint64_t slots = (file_size - block) / ALIGN;
    for (int i = 0; i < n; i++) {
        uint64_t r = ((uint64_t)rng() << 24) ^ rng();
        offs[i] = (r % slots) * ALIGN;
    }
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double t_start;
static int out_of_time(void) { return tr_time_sec() - t_start > TIME_LIMIT_SEC; }

static void print_line(const char *what, size_t block, int readers, int n_blocks, double *sec, int n) {
    if (n == 0) return;
    qsort(sec, (size_t)n, sizeof sec[0], cmp_double);
    double med = (n % 2) ? sec[n / 2] : 0.5 * (sec[n / 2 - 1] + sec[n / 2]);
    double mb = (double)block * n_blocks / 1e6;
    /* the fastest run has the smallest time: min MB/s comes from sec[n - 1] */
    printf("%-14s block %8zu  readers %2d: median %8.1f MB/s (%6.2f ms a block)  min %8.1f  max %8.1f  spread %5.1f%%  n=%d\n",
           what, block, readers, mb / med, med / n_blocks * 1e3, mb / sec[n - 1], mb / sec[0],
           (sec[n - 1] - sec[0]) / med * 100.0, n);
}

int main(int argc, char **argv) {
    const char *path = NULL;
    /* one Q8_0 expert matrix of OLMoE-1B-7B (1024 x 2048 weights, 34 bytes per 32), and three */
    size_t blocks[2] = {2228224, 3 * 2228224};
    int n_sizes = 2, runs = 5, bad = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            blocks[0] = ((size_t)atoll(argv[++i]) + ALIGN - 1) / ALIGN * ALIGN;
            n_sizes = 1;
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (path == NULL && argv[i][0] != '-') path = argv[i];
        else bad = 1;
    }
    if (bad || path == NULL || runs < 1 || runs > MAX_RUNS || blocks[0] < ALIGN) {
        fprintf(stderr, "usage: bench_disk <big file> [--block <bytes>] [--runs N]  (N <= %d)\n", MAX_RUNS);
        return 2;
    }

    char err[256];
    tr_file *file = tr_file_open(path, err, sizeof err);
    if (file == NULL) {
        fprintf(stderr, "bench_disk: %s\n", err);
        return 1;
    }
    uint64_t size = (uint64_t)tr_file_size(file);
    raw_file raw;
    memset(&raw, 0, sizeof raw);
    if (size < 64u * blocks[n_sizes - 1] || raw_open(&raw, path) != 0) {
        fprintf(stderr, "bench_disk: '%s' is too small (64 blocks wanted) or cannot be opened without the system's cache\n", path);
        return 1;
    }
    printf("file: %s, %.2f GiB\n", path, (double)size / (1u << 30));

    static uint64_t offs[MAX_BLOCKS], sums[MAX_BLOCKS], sums_cached[MAX_BLOCKS];
    static const int readers[] = {1, 2, 4, 8, 16};
    run_ctx c;
    memset(&c, 0, sizeof c);
    for (int i = 0; i < MAX_READERS; i++) {
        c.buf[i] = tr_alloc_aligned(blocks[n_sizes - 1], ALIGN);
        if (c.buf[i] == NULL) {
            fprintf(stderr, "bench_disk: out of memory\n");
            return 1;
        }
    }
    t_start = tr_time_sec();

    /* the same bytes from the two readers, or nothing to measure */
    {
        tr_pool *pool = tr_pool_create(4);
        if (pool == NULL) return 1;
        c.block = blocks[0];
        c.offs = offs;
        pick_offsets(offs, 32, size, c.block);
        c.raw = &raw, c.file = NULL, c.sums = sums;
        tr_parallel_for(pool, 32, 1, read_body, &c);
        c.raw = NULL, c.file = file, c.sums = sums_cached;
        tr_parallel_for(pool, 32, 1, read_body, &c);
        tr_pool_destroy(pool);
        if (c.failed || memcmp(sums, sums_cached, 32 * sizeof sums[0]) != 0) {
            fprintf(stderr, "bench_disk: the direct reader and tr_file_pread disagree, or a read failed\n");
            return 1;
        }
        printf("direct and cached readers agree on 32 blocks\n");
    }

#if defined(_WIN32)
    bench_event_overhead(path, file);
#endif

    int stopped = 0;
    for (int si = 0; si < n_sizes && !stopped; si++) {
        c.block = blocks[si];
        int n_blocks = (int)(RUN_BYTES / c.block);
        if (n_blocks > MAX_BLOCKS) n_blocks = MAX_BLOCKS;
        if (n_blocks < 8) n_blocks = 8;
        for (size_t qi = 0; qi < sizeof readers / sizeof readers[0] && !stopped; qi++) {
            tr_pool *pool = tr_pool_create(readers[qi]);
            if (pool == NULL) break;
            double sec[MAX_RUNS];
            int n = 0;
            c.raw = &raw, c.file = NULL, c.sums = sums;
            for (int r = -1; r < runs && !(stopped = out_of_time()); r++) {
                pick_offsets(offs, n_blocks, size, c.block); /* new positions every run */
                double t0 = tr_time_sec();
                tr_parallel_for(pool, n_blocks, 1, read_body, &c);
                if (r >= 0) sec[n++] = tr_time_sec() - t0;
            }
            print_line("direct", c.block, readers[qi], n_blocks, sec, n);
            tr_pool_destroy(pool);
        }
    }

    /* through the system's cache, 8 readers: the same blocks twice */
    for (int si = 0; si < n_sizes && !stopped; si++) {
        c.block = blocks[si];
        int n_blocks = (int)(RUN_BYTES / c.block);
        if (n_blocks > MAX_BLOCKS) n_blocks = MAX_BLOCKS;
        if (n_blocks < 8) n_blocks = 8;
        tr_pool *pool = tr_pool_create(8);
        if (pool == NULL) break;
        double first[MAX_RUNS], again[MAX_RUNS];
        int n = 0;
        c.raw = NULL, c.file = file, c.sums = sums;
        for (int r = 0; r < runs && !(stopped = out_of_time()); r++) {
            pick_offsets(offs, n_blocks, size, c.block);
            double t0 = tr_time_sec();
            tr_parallel_for(pool, n_blocks, 1, read_body, &c);
            double t1 = tr_time_sec();
            tr_parallel_for(pool, n_blocks, 1, read_body, &c);
            again[n] = tr_time_sec() - t1;
            first[n++] = t1 - t0;
        }
        print_line("cached first", c.block, 8, n_blocks, first, n);
        print_line("cached again", c.block, 8, n_blocks, again, n);
        tr_pool_destroy(pool);
    }
    if (stopped) printf("stopped at the %.0f s limit: the lines above are what fitted\n", TIME_LIMIT_SEC);
    if (c.failed) fprintf(stderr, "bench_disk: a read failed\n");

    for (int i = 0; i < MAX_READERS; i++) tr_free_aligned(c.buf[i]);
    raw_close(&raw);
    tr_file_close(file);
    return c.failed ? 1 : 0;
}
