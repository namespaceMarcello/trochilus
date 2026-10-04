/* bench_disk_misses.c — a decode token's misses, read the way each engine reads them (docs/MEASUREMENTS.md
 * §The three machines; R1 phase 3, the 8 GB machine; CLAUDE.md "every piece against colibri and ds4").
 *
 * Under one token's units the store's decode reads each layer's missing (layer, expert) units from the disk while
 * the token waits. The same units, the same bytes, read as:
 *   ours      a unit's three parts (gate, up, down: three GGUF tensors) in flight together (tr_file_preadv_n on a
 *             direct handle), the units one after the other: the store's on-demand read today
 *   ours-all  every unit of the layer in flight at once: one thread a unit, each its three parts together
 *   colibri   one request a unit for its three parts back to back (its own merged format, ref/colibri
 *             c/olmoe.c:660 load_expert_merged: emulated as the unit's bytes read in one piece from where its gate
 *             part starts), through the system's cache (its default), the units one after the other
 *   colibri-d the same on a direct handle (its DIRECT=1)
 *   ds4       the layer's parts as tasks of a pool of 9 threads, each one part in one direct read (ref/ds4
 *             ds4_metal.m:13334-13400, ds4_cuda.cu:4509-4525)
 *   mmap      llama.cpp's: the file mapped (ref/llama.cpp src/llama-mmap.cpp:576-622), the layer's parts touched
 *             by 4 compute threads, a byte a page, the pages faulted in from the disk by the system
 *
 *   bench_disk_misses <model.gguf> <fresh copy of it> [--runs N] [--layers L] [--used K]
 *   bench_disk_misses <model.gguf> <model.gguf> --run <k> [--runs N]
 *
 * --run k: a prompt's reads instead (a whole layer, its experts k at a time in runs: the store's readv, three
 * requests of k parts each), with 1, 2 or 4 runs in flight at once, 3, 6 or 12 requests (docs/LESSONS.md #269:
 * the engine's own request shape, at k = 4, ~8.5 MiB a request, and k = 32, ~68 MiB).
 *
 * A run reads L layers x K units (default 4 x 8, a quarter of an OLMoE token's 128) at random. The direct
 * patterns read the model; the two that go through the system's cache (colibri, mmap) read the fresh copy, made
 * without the system's cache (xcopy /J on Windows: no page of it is held in RAM), and never a unit twice, so
 * every byte they get comes from the disk. Median of N runs (default 5) after one warm-up: ms a layer and GB/s,
 * with min and max. Before any number every pattern's first 8 bytes of every unit agree with a direct read of the
 * same file. The program stops launching runs after 50 s (a benchmark run stays under 60 s). Reads only. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/format/gguf.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif

enum { PARTS = 3, MAX_LAYERS = 128, MAX_USED = 64, MAX_RUNS = 15, DS4_THREADS = 9, MMAP_THREADS = 4, N_PAT = 6 };
#define TIME_LIMIT_SEC 50.0
#define PAGE ((uint64_t)TR_FILE_DIRECT_ALIGN)

static const char *const PAT_NAME[N_PAT] = {"ours", "ours-all", "colibri", "colibri-d", "ds4", "mmap"};
static const char *const PART_NAME[PARTS] = {"gate", "up", "down"};

/* ---- the model's geometry: where each layer's expert e of part p sits ---- */
static int64_t n_layers, n_expert;
static uint64_t part_off[MAX_LAYERS][PARTS], part_bytes[MAX_LAYERS][PARTS];

static int geometry(const char *path) {
    char err[256];
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    if (g == NULL) {
        fprintf(stderr, "bench_disk_misses: %s\n", err);
        return -1;
    }
    n_layers = 0;
    for (int64_t L = 0; L < MAX_LAYERS; L++) {
        int found = 0;
        for (int p = 0; p < PARTS; p++) {
            char name[96];
            snprintf(name, sizeof name, "blk.%lld.ffn_%s_exps.weight", (long long)L, PART_NAME[p]);
            const tr_gguf_tensor *t = tr_gguf_find_tensor(g, name);
            if (t == NULL || t->n_dims != 3) continue;
            n_expert = (int64_t)t->ne[2];
            part_off[L][p] = t->offset;
            part_bytes[L][p] = t->n_bytes / t->ne[2];
            found++;
        }
        if (found == 0) break;
        if (found != PARTS) {
            fprintf(stderr, "bench_disk_misses: layer %lld has %d of its 3 expert tensors\n", (long long)L, found);
            tr_gguf_close(g);
            return -1;
        }
        n_layers++;
    }
    tr_gguf_close(g);
    if (n_layers == 0 || n_expert <= 0) {
        fprintf(stderr, "bench_disk_misses: no expert tensors in %s\n", path);
        return -1;
    }
    return 0;
}

/* a unit's part p: [lo, hi) aligned, and where its own bytes start inside */
static uint64_t unit_lo(int64_t L, int64_t e, int p) {
    return (part_off[L][p] + part_bytes[L][p] * (uint64_t)e) / PAGE * PAGE;
}
static uint64_t unit_hi(int64_t L, int64_t e, int p) {
    return (part_off[L][p] + part_bytes[L][p] * (uint64_t)(e + 1) + PAGE - 1) / PAGE * PAGE;
}
static uint64_t unit_bytes(int64_t L) {
    return part_bytes[L][0] + part_bytes[L][1] + part_bytes[L][2];
}

/* ---- one run: the layers and units, the buffers, the files ---- */
typedef struct {
    int64_t layer[MAX_LAYERS], id[MAX_LAYERS][MAX_USED]; /* the run's L layers and K units each */
    int L, K;
    const tr_file *direct, *fresh_direct, *fresh_buffered;
    const unsigned char *map; /* the fresh copy mapped, read only */
    unsigned char *buf[MAX_USED]; /* a unit's three parts, aligned, one buffer a thread */
    size_t buf_bytes;
    void *scratch[MAX_USED];
    uint64_t first8[MAX_LAYERS][MAX_USED]; /* each unit's gate part's first 8 bytes, as the pattern read them */
    int64_t cur; /* the layer index in this run a parallel body works on */
    int failed;
    _Atomic uint64_t touched; /* mmap: a sum of the bytes touched, so the reads are not optimized away */
} run_ctx;

static void keep_first8(run_ctx *c, int64_t li, int64_t k, const unsigned char *gate_lo, int64_t L, int64_t e) {
    uint64_t shift = part_off[L][0] + part_bytes[L][0] * (uint64_t)e - unit_lo(L, e, 0);
    memcpy(&c->first8[li][k], gate_lo + shift, 8);
}

/* ours: a unit's three parts, one request each, in flight together, into buffer w */
static int read_unit_ours(run_ctx *c, const tr_file *f, int64_t li, int64_t k, int w) {
    int64_t L = c->layer[li], e = c->id[li][k];
    tr_iov iov[PARTS];
    tr_readv_req req[PARTS];
    unsigned char *at = c->buf[w];
    for (int p = 0; p < PARTS; p++) {
        iov[p].base = at;
        iov[p].n = (size_t)(unit_hi(L, e, p) - unit_lo(L, e, p));
        req[p].iov = &iov[p];
        req[p].cnt = 1;
        req[p].offset = unit_lo(L, e, p);
        at += iov[p].n;
    }
    if (tr_file_preadv_n(f, req, PARTS, c->scratch[w]) != 0) return -1;
    keep_first8(c, li, k, c->buf[w], L, e);
    return 0;
}

static void ours_all_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    run_ctx *c = (run_ctx *)ctx_;
    for (int64_t k = begin; k < end; k++)
        if (read_unit_ours(c, c->direct, c->cur, k, worker) != 0) c->failed = 1;
}

/* colibri: the unit's bytes in one request from where its gate part starts (its merged format's shape) */
static int read_unit_colibri(run_ctx *c, const tr_file *f, int direct, int64_t li, int64_t k) {
    int64_t L = c->layer[li], e = c->id[li][k];
    uint64_t start = part_off[L][0] + part_bytes[L][0] * (uint64_t)e;
    uint64_t lo = direct ? start / PAGE * PAGE : start;
    uint64_t hi = direct ? (start + unit_bytes(L) + PAGE - 1) / PAGE * PAGE : start + unit_bytes(L);
    if (tr_file_pread(f, c->buf[0], (size_t)(hi - lo), lo) != 0) return -1;
    memcpy(&c->first8[li][k], c->buf[0] + (start - lo), 8);
    return 0;
}

/* ds4: task t of the layer is unit t / 3's part t % 3, one direct read */
static void ds4_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    run_ctx *c = (run_ctx *)ctx_;
    for (int64_t t = begin; t < end; t++) {
        int64_t k = t / PARTS;
        int p = (int)(t % PARTS);
        int64_t L = c->layer[c->cur], e = c->id[c->cur][k];
        uint64_t lo = unit_lo(L, e, p), hi = unit_hi(L, e, p);
        if (tr_file_pread(c->direct, c->buf[worker], (size_t)(hi - lo), lo) != 0) {
            c->failed = 1;
            continue;
        }
        if (p == 0) keep_first8(c, c->cur, k, c->buf[worker], L, e);
    }
}

/* mmap: the layer's units' parts as one list of pages, split among the threads, a byte a page touched */
static void mmap_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    run_ctx *c = (run_ctx *)ctx_;
    (void)worker;
    uint64_t sum = 0;
    for (int64_t t = begin; t < end; t++) {
        int64_t k = t / PARTS;
        int p = (int)(t % PARTS);
        int64_t L = c->layer[c->cur], e = c->id[c->cur][k];
        uint64_t s = part_off[L][p] + part_bytes[L][p] * (uint64_t)e;
        for (uint64_t o = s; o < s + part_bytes[L][p]; o += PAGE) sum += c->map[o];
        if (p == 0) memcpy(&c->first8[c->cur][k], c->map + s, 8);
    }
    atomic_fetch_add(&c->touched, sum);
}

/* one pattern over the whole run: 0, or -1 on a failed read */
static int run_pattern(int pat, run_ctx *c, tr_pool *pool_k, tr_pool *pool_ds4, tr_pool *pool_mmap) {
    c->failed = 0;
    for (int64_t li = 0; li < c->L; li++) {
        c->cur = li;
        switch (pat) {
        case 0:
            for (int64_t k = 0; k < c->K; k++)
                if (read_unit_ours(c, c->direct, li, k, 0) != 0) return -1;
            break;
        case 1:
            tr_parallel_for(pool_k, c->K, 1, ours_all_body, c);
            break;
        case 2:
            for (int64_t k = 0; k < c->K; k++)
                if (read_unit_colibri(c, c->fresh_buffered, 0, li, k) != 0) return -1;
            break;
        case 3:
            for (int64_t k = 0; k < c->K; k++)
                if (read_unit_colibri(c, c->direct, 1, li, k) != 0) return -1;
            break;
        case 4:
            tr_parallel_for(pool_ds4, c->K * PARTS, 1, ds4_body, c);
            break;
        case 5:
            tr_parallel_for(pool_mmap, c->K * PARTS, 1, mmap_body, c);
            break;
        }
        if (c->failed) return -1;
    }
    return 0;
}

/* ---- --run k: a prompt's runs (a layer's experts k at a time, the store's readv: three requests of k parts each)
 * with 1, 2 or 4 runs in flight at once: 3, 6 or 12 requests (docs/LESSONS.md #269: the engine's own shape) ---- */
static uint32_t rng(void);
static int cmp_double(const void *a, const void *b);

typedef struct {
    const tr_file *f;
    int64_t layer, k, n_runs;
    unsigned char *buf[4];
    void *scratch[4];
    int failed;
} runs_ctx;

static void runs_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    runs_ctx *c = (runs_ctx *)ctx_;
    for (int64_t r = begin; r < end; r++) {
        int64_t e0 = r * c->k, k = e0 + c->k <= n_expert ? c->k : n_expert - e0;
        tr_iov iov[PARTS];
        tr_readv_req req[PARTS];
        unsigned char *at = c->buf[worker];
        for (int p = 0; p < PARTS; p++) {
            uint64_t lo = unit_lo(c->layer, e0, p), hi = unit_hi(c->layer, e0 + k - 1, p);
            iov[p].base = at;
            iov[p].n = (size_t)(hi - lo);
            req[p].iov = &iov[p];
            req[p].cnt = 1;
            req[p].offset = lo;
            at += iov[p].n;
        }
        if (tr_file_preadv_n(c->f, req, PARTS, c->scratch[worker]) != 0) c->failed = 1;
    }
}

static int bench_runs(const tr_file *direct, int64_t k, int runs) {
    runs_ctx c;
    memset(&c, 0, sizeof c);
    c.f = direct;
    c.k = k;
    c.n_runs = (n_expert + k - 1) / k;
    size_t bytes = 0;
    for (int p = 0; p < PARTS; p++) bytes += (size_t)(part_bytes[0][p] * (uint64_t)k + 2 * PAGE);
    for (int w = 0; w < 4; w++) {
        c.buf[w] = (unsigned char *)tr_alloc_aligned(bytes, (size_t)PAGE);
        size_t s = tr_file_preadv_n_scratch(bytes, PARTS);
        c.scratch[w] = s > 0 ? malloc(s) : NULL;
        if (c.buf[w] == NULL || (s > 0 && c.scratch[w] == NULL)) return 1;
    }
    static const int inflight[3] = {1, 2, 4};
    tr_pool *pool[3];
    for (int i = 0; i < 3; i++)
        if ((pool[i] = tr_pool_create(inflight[i])) == NULL) return 1;
    double ms[3][MAX_RUNS + 1];
    double layer_bytes = (double)unit_bytes(0) * (double)n_expert;
    double t_start = tr_time_sec();
    int done = 0;
    for (int r = 0; r <= runs; r++) {
        if (tr_time_sec() - t_start > TIME_LIMIT_SEC) {
            printf("stopped after %d runs: 50 s reached\n", done);
            break;
        }
        for (int i0 = 0; i0 < 3; i0++) {
            int i = (i0 + r) % 3;
            c.layer = (int64_t)(rng() % (uint32_t)n_layers);
            double t0 = tr_time_sec();
            tr_parallel_for(pool[i], c.n_runs, 1, runs_body, &c);
            ms[i][r] = (tr_time_sec() - t0) * 1e3;
            if (c.failed) {
                fprintf(stderr, "bench_disk_misses: a run's read failed\n");
                return 1;
            }
        }
        done = r;
    }
    if (done < 1) return 1;
    printf("a whole layer in runs of %lld experts (3 requests of %.2f MiB a run), %lld runs; median of %d after a "
           "warm-up\n",
           (long long)k, part_bytes[0][0] * (double)k / 1048576.0, (long long)c.n_runs, done);
    printf("%-14s %10s %8s %10s %10s\n", "in flight", "ms/layer", "GB/s", "min ms", "max ms");
    for (int i = 0; i < 3; i++) {
        double v[MAX_RUNS];
        for (int r = 1; r <= done; r++) v[r - 1] = ms[i][r];
        qsort(v, (size_t)done, sizeof v[0], cmp_double);
        double med = done % 2 ? v[done / 2] : (v[done / 2 - 1] + v[done / 2]) / 2;
        printf("%d runs (%2d req) %10.2f %8.2f %10.2f %10.2f\n", inflight[i], 3 * inflight[i], med,
               layer_bytes / (med * 1e-3) / 1e9, v[0], v[done - 1]);
    }
    return 0;
}

/* ---- the units: random for the direct patterns, never twice from the fresh copy ---- */
static uint32_t rng_state = 12345;
static uint32_t rng(void) {
    uint32_t v = rng_state;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    return rng_state = v;
}

static int64_t *fresh_order; /* every (layer, expert) unit once, shuffled */
static int64_t fresh_next, fresh_total;

/* the run's units: from the fresh copy's unused ones (cached patterns), or anywhere (direct ones) */
static int pick_units(run_ctx *c, int fresh) {
    for (int64_t li = 0; li < c->L; li++) {
        if (fresh) {
            /* a layer's K units: the next K unused units of one layer, taken from the shuffled order */
            if (fresh_next + c->K > fresh_total) return -1;
            int64_t u0 = fresh_order[fresh_next];
            c->layer[li] = u0 / n_expert;
            int64_t got = 0;
            for (int64_t i = fresh_next; i < fresh_total && got < c->K; i++) {
                if (fresh_order[i] / n_expert != c->layer[li]) continue;
                int64_t t = fresh_order[i];
                fresh_order[i] = fresh_order[fresh_next + got];
                fresh_order[fresh_next + got] = t;
                c->id[li][got++] = t % n_expert;
            }
            if (got < c->K) return -1;
            fresh_next += c->K;
        } else {
            c->layer[li] = (int64_t)(rng() % (uint32_t)n_layers);
            int64_t got = 0;
            while (got < c->K) {
                int64_t e = (int64_t)(rng() % (uint32_t)n_expert), dup = 0;
                for (int64_t j = 0; j < got; j++) dup |= c->id[li][j] == e;
                if (!dup) c->id[li][got++] = e;
            }
        }
    }
    return 0;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

#if defined(_WIN32)
static const unsigned char *map_file(const char *path, HANDLE *fh, HANDLE *mh) {
    wchar_t w[1024];
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 1024) == 0) return NULL;
    *fh = CreateFileW(w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (*fh == INVALID_HANDLE_VALUE) return NULL;
    *mh = CreateFileMappingW(*fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (*mh == NULL) return NULL;
    return (const unsigned char *)MapViewOfFile(*mh, FILE_MAP_READ, 0, 0, 0);
}
#else
static const unsigned char *map_file(const char *path, int64_t size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    void *p = mmap(NULL, (size_t)size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return p == MAP_FAILED ? NULL : (const unsigned char *)p;
}
#endif

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: bench_disk_misses <model.gguf> <fresh copy of it> [--runs N] [--layers L] [--used K]\n");
        return 2;
    }
    const char *path = argv[1], *fresh_path = argv[2];
    int runs = 5, L = 4, K = 8;
    int64_t run_k = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--runs") == 0) runs = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--layers") == 0) L = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--used") == 0) K = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--run") == 0) run_k = atoll(argv[i + 1]);
    }
    if (runs < 1 || runs > MAX_RUNS || L < 1 || L > MAX_LAYERS || K < 1 || K > MAX_USED) {
        fprintf(stderr, "bench_disk_misses: --runs 1..%d, --layers 1..%d, --used 1..%d\n", MAX_RUNS, MAX_LAYERS,
                MAX_USED);
        return 2;
    }
    if (geometry(path) != 0) return 1;
    if (L > n_layers || K > n_expert) {
        fprintf(stderr, "bench_disk_misses: the model has %lld layers of %lld experts\n", (long long)n_layers,
                (long long)n_expert);
        return 2;
    }
    char err[256];
    tr_file *direct = tr_file_open_direct(path, err, sizeof err);
    tr_file *fresh_direct = tr_file_open_direct(fresh_path, err, sizeof err);
    tr_file *fresh_buffered = tr_file_open(fresh_path, err, sizeof err);
    if (direct == NULL || fresh_direct == NULL || fresh_buffered == NULL ||
        tr_file_size(direct) != tr_file_size(fresh_buffered)) {
        fprintf(stderr, "bench_disk_misses: cannot open both files directly and buffered, or their sizes differ\n");
        return 1;
    }
    if (run_k > 0) return bench_runs(direct, run_k < n_expert ? run_k : n_expert, runs);
    run_ctx *c = (run_ctx *)calloc(1, sizeof *c);
    if (c == NULL) return 1;
    c->L = L;
    c->K = K;
    c->direct = direct;
    c->fresh_direct = fresh_direct;
    c->fresh_buffered = fresh_buffered;
    uint64_t most = 0;
    for (int64_t l = 0; l < n_layers; l++) {
        uint64_t b = unit_bytes(l) + PARTS * 2 * PAGE;
        if (b > most) most = b;
    }
    c->buf_bytes = (size_t)most;
    int n_buf = K > DS4_THREADS ? K : DS4_THREADS; /* a buffer a thread of the widest pool */
    for (int w = 0; w < n_buf; w++) {
        c->buf[w] = (unsigned char *)tr_alloc_aligned(c->buf_bytes, (size_t)PAGE);
        size_t s = tr_file_preadv_n_scratch(c->buf_bytes, PARTS);
        c->scratch[w] = s > 0 ? malloc(s) : NULL;
        if (c->buf[w] == NULL || (s > 0 && c->scratch[w] == NULL)) return 1;
    }
#if defined(_WIN32)
    HANDLE fh = NULL, mh = NULL;
    c->map = map_file(fresh_path, &fh, &mh);
#else
    c->map = map_file(fresh_path, tr_file_size(fresh_buffered));
#endif
    if (c->map == NULL) {
        fprintf(stderr, "bench_disk_misses: cannot map %s\n", fresh_path);
        return 1;
    }
    tr_pool *pool_k = tr_pool_create(K), *pool_ds4 = tr_pool_create(DS4_THREADS), *pool_mmap = tr_pool_create(MMAP_THREADS);
    if (pool_k == NULL || pool_ds4 == NULL || pool_mmap == NULL) return 1;
    fresh_total = n_layers * n_expert;
    fresh_order = (int64_t *)malloc(sizeof(int64_t) * (size_t)fresh_total);
    if (fresh_order == NULL) return 1;
    for (int64_t u = 0; u < fresh_total; u++) fresh_order[u] = u;
    for (int64_t u = fresh_total - 1; u > 0; u--) {
        int64_t j = (int64_t)(rng() % (uint32_t)(u + 1)), t = fresh_order[u];
        fresh_order[u] = fresh_order[j];
        fresh_order[j] = t;
    }

    /* every pattern's bytes against a direct read of the same file, on one run's units, before any number */
    {
        uint64_t ref[MAX_LAYERS][MAX_USED];
        if (pick_units(c, 0) != 0 || run_pattern(0, c, pool_k, pool_ds4, pool_mmap) != 0) return 1;
        memcpy(ref, c->first8, sizeof ref);
        for (int pat = 1; pat < N_PAT; pat++) {
            if (pat == 2 || pat == 5) continue; /* the cached ones read the fresh copy: checked below */
            if (run_pattern(pat, c, pool_k, pool_ds4, pool_mmap) != 0 || memcmp(ref, c->first8, sizeof ref) != 0) {
                fprintf(stderr, "bench_disk_misses: %s read other bytes than a direct read\n", PAT_NAME[pat]);
                return 1;
            }
        }
        for (int pat = 2; pat <= 5; pat += 3) {
            if (pick_units(c, 1) != 0) return 1;
            const tr_file *keep = c->direct;
            c->direct = fresh_direct;
            int rc = run_pattern(0, c, pool_k, pool_ds4, pool_mmap);
            c->direct = keep;
            memcpy(ref, c->first8, sizeof ref);
            if (rc != 0 || run_pattern(pat, c, pool_k, pool_ds4, pool_mmap) != 0 ||
                memcmp(ref, c->first8, sizeof ref) != 0) {
                fprintf(stderr, "bench_disk_misses: %s read other bytes than a direct read of the copy\n",
                        PAT_NAME[pat]);
                return 1;
            }
        }
        printf("checked: every pattern's units start with the bytes a direct read finds\n");
    }

    double ms[N_PAT][MAX_RUNS + 1];
    double bytes_run = 0;
    for (int64_t li = 0; li < L; li++) bytes_run += (double)unit_bytes(li) * K; /* every layer the same size */
    double t_start = tr_time_sec();
    int done = 0;
    for (int r = 0; r <= runs; r++) {
        if (tr_time_sec() - t_start > TIME_LIMIT_SEC) {
            printf("stopped after %d runs: 50 s reached\n", done);
            break;
        }
        for (int pat = 0; pat < N_PAT; pat++) {
            /* the rounds rotate the patterns' order */
            int p = (pat + r) % N_PAT;
            if (pick_units(c, p == 2 || p == 5) != 0) {
                fprintf(stderr, "bench_disk_misses: the fresh copy has no unused units left\n");
                return 1;
            }
            double t0 = tr_time_sec();
            if (run_pattern(p, c, pool_k, pool_ds4, pool_mmap) != 0) {
                fprintf(stderr, "bench_disk_misses: %s: a read failed\n", PAT_NAME[p]);
                return 1;
            }
            ms[p][r] = (tr_time_sec() - t0) * 1e3 / L;
        }
        done = r;
    }
    if (done < 1) return 1;
    printf("%lld layers x %lld experts, a unit %.3f MiB (%s %.3f, %s %.3f, %s %.3f); a run: %d layers x %d units, "
           "%.1f MiB; median of %d runs after a warm-up\n",
           (long long)n_layers, (long long)n_expert, unit_bytes(0) / 1048576.0, PART_NAME[0],
           part_bytes[0][0] / 1048576.0, PART_NAME[1], part_bytes[0][1] / 1048576.0, PART_NAME[2],
           part_bytes[0][2] / 1048576.0, L, K, bytes_run / 1048576.0, done);
    printf("%-10s %10s %8s %10s %10s\n", "pattern", "ms/layer", "GB/s", "min ms", "max ms");
    for (int p = 0; p < N_PAT; p++) {
        double v[MAX_RUNS];
        for (int r = 1; r <= done; r++) v[r - 1] = ms[p][r];
        qsort(v, (size_t)done, sizeof v[0], cmp_double);
        double med = done % 2 ? v[done / 2] : (v[done / 2 - 1] + v[done / 2]) / 2;
        printf("%-10s %10.2f %8.2f %10.2f %10.2f\n", PAT_NAME[p], med, bytes_run / L / (med * 1e-3) / 1e9, v[0],
               v[done - 1]);
    }
    printf("(touched %llu)\n", (unsigned long long)atomic_load(&c->touched));
    return 0;
}
