/* bench_q4x.c — Q4_K's integer kernels (kernels.h q4x_*) timed the way a short verify pass runs them
 * (docs/MEASUREMENTS.md §A Q4_K weight row decoded once for 2-3 tokens: question 78's road (c)).
 *
 * A group of 2-3 input rows (a verify pass's dense matrices, an expert two or three of its rows chose) goes by
 * items of 16 weight rows. Three ways to run an item against T prepared rows:
 *   dot2    T x 8 calls of q4x_dot2, a token after the other (each weight row decoded T times);
 *   xt      8 calls of q4x_dot_xt (each weight decoded once for the T tokens);
 *   panel   the W16 panel of the 16 rows (the next item's rows brought ahead), then one tile of T (the road of a
 *           group of 3 before road (c)).
 *
 *   bench_q4x [--runs N] [--ms M]       one core: two rows and T prepared rows hot (n = 1024 and 2048): T calls of
 *                                       q4x_dot2 against one q4x_dot_xt (ns a call and their ratio); then an item
 *                                       hot (16 rows of 2048 and T prepared rows, L1 and L2) the three ways
 *   bench_q4x --ram P [--runs N] [--mib M]
 *                                       P threads on distinct cores, M MiB (default 1024) of Q4_K rows of 2048 in
 *                                       items of 16 rows, one contiguous chunk of items a thread: T = 1 (the
 *                                       decode's dot2), then each way at T = 2 and 3, the prepared rows the same
 *                                       for every item (a verify pass: every group's rows are among its 2-3) or
 *                                       the next T of a pool of 48 (new tokens every item, docs/LESSONS.md #219);
 *                                       a plain read of the same bytes is the ceiling. GB/s of weight bytes.
 *   bench_q4x --ends P [--runs N]       what lies past a thread's region (docs/MEASUREMENTS.md question 84): P
 *                                       threads, sequences of 4 calls from RAM, each thread one region a call,
 *                                       the decode's kernel (dot2, T = 1) and a plain read; the regions laid out
 *                                       five ways: next (a call's regions side by side), guard (the same, the
 *                                       page after each region dropped), own (a thread's regions of the 4 calls
 *                                       one after the other), ownrev (the same in reverse), next0 (next with no
 *                                       page between the regions). Two shapes: the dense call (2048 rows of
 *                                       2048) and the experts' (8 x 1024 rows of 2048).
 *   bench_q4x --idle P [--runs N] [--w us,..] [--x KiB,..] [--kinds spin,l2,ram]
 *                                       the idle workers (docs/MEASUREMENTS.md §The idle workers): a serial
 *                                       step of W us on the calling thread, then a call from RAM (the two
 *                                       shapes), the other threads bringing their regions' first X KiB in
 *                                       while they wait (tr_pool_hint, T0 and T2) or not, raced step set by
 *                                       step set.
 * Medians of N runs, the lines alternated run by run, spread (max - min) / median. Every way's outputs are
 * compared with scalar's dot_row (L1) or with dot2's (RAM) before timing; a mismatch fails the run. The active
 * tier's kernels; run natively on a still machine (tools/measure_guard.lib). */
#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE /* madvise, MAP_ANONYMOUS */
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif

#define MAX_RUNS 31
#define ROWS TR_PM_ROWS
#define NMAX 2048
#define POOL 48 /* prepared rows the rotating mode cycles through */

static volatile float g_sink;

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(g_rng >> 33);
}

/* plausible Q4_K blocks: small positive d and dmin, random scale bytes and quants */
static void fill_q4_k(unsigned char *row, int64_t nb) {
    for (int64_t b = 0; b < nb; b++) {
        unsigned char *blk = row + (size_t)b * TR_Q4_K_BLOCK_BYTES;
        uint16_t d = (uint16_t)(0x2C00 | (rnd() & 0x3FF)), dmin = (uint16_t)(0x2800 | (rnd() & 0x3FF));
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (int i = 4; i < TR_Q4_K_BLOCK_BYTES; i++) blk[i] = (unsigned char)rnd();
    }
}

static unsigned char *aligned64(size_t bytes, void **base) {
    *base = malloc(bytes + 64);
    return *base == NULL ? NULL : (unsigned char *)(((uintptr_t)*base + 63) & ~(uintptr_t)63);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_spread(double *v, int n, double *spread) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    double med = v[n / 2];
    *spread = med > 0 ? (v[n - 1] - v[0]) / med : 0;
    return med;
}

/* ---- an item: 16 weight rows against T prepared rows, one of three ways ------------------------------------ */

enum { WAY_DOT2, WAY_XT, WAY_PANEL, N_WAYS };
static const char *const WAY_NAME[N_WAYS] = {"dot2", "xt", "panel"};

typedef struct {
    const tr_kernels *K;
    int64_t n;
    size_t rb, xb; /* a weight row's bytes, a prepared row's */
} shape;

/* the T prepared rows xq, xq + xb, ... as the kernels take them */
static void rows_of(const unsigned char *xq, size_t xb, int T, const void **xp) {
    for (int t = 0; t < T; t++) xp[t] = xq + (size_t)t * xb;
}

/* y[t * ROWS + r] */
static void run_item(const shape *sh, int way, int T, const unsigned char *rows, const unsigned char *xq,
                     unsigned char *panel, const void *next, float *y) {
    const tr_kernels *K = sh->K;
    const void *xp[TR_Q4X_TILE_MAX > TR_Q4X_XT_MAX ? TR_Q4X_TILE_MAX : TR_Q4X_XT_MAX];
    rows_of(xq, sh->xb, T, xp);
    switch (way) {
    case WAY_DOT2:
        for (int t = 0; t < T; t++)
            for (int r = 0; r < ROWS; r += 2)
                K->q4x_dot2(rows + (size_t)r * sh->rb, rows + (size_t)(r + 1) * sh->rb, xq + (size_t)t * sh->xb, sh->n,
                            y + t * ROWS + r);
        break;
    case WAY_XT:
        for (int r = 0; r < ROWS; r += 2)
            K->q4x_dot_xt(rows + (size_t)r * sh->rb, rows + (size_t)(r + 1) * sh->rb, xp, sh->n, T, y + r, ROWS);
        break;
    default:
        K->q4x_panel(rows, sh->rb, sh->n, panel, next);
        K->q4x_tile(panel, xp, sh->n, T, y, ROWS);
        break;
    }
}

static int way_available(const tr_kernels *K, int way, int T) {
    if (way == WAY_XT) return T >= 2 && K->q4x_dot_xt != NULL;
    if (way == WAY_PANEL) return T >= 2 && K->q4x_panel != NULL && K->q4x_tile != NULL;
    return 1;
}

/* ---- one core ------------------------------------------------------------------------------------------- */

/* ns a call: xt (way 1) or T calls of dot2 (way 0) on two rows */
static double time_pair(const shape *sh, int way, int T, const unsigned char *rows, const unsigned char *xq,
                        long calls) {
    float y[2 * TR_Q4X_XT_MAX];
    const void *xp[TR_Q4X_XT_MAX];
    rows_of(xq, sh->xb, T, xp);
    double t0 = tr_time_sec();
    for (long c = 0; c < calls; c++) {
        if (way == WAY_XT) sh->K->q4x_dot_xt(rows, rows + sh->rb, xp, sh->n, T, y, 2);
        else
            for (int t = 0; t < T; t++) sh->K->q4x_dot2(rows, rows + sh->rb, xq + (size_t)t * sh->xb, sh->n, y + 2 * t);
    }
    double t1 = tr_time_sec();
    g_sink = y[0] + y[2 * T - 1];
    return (t1 - t0) * 1e9 / (double)calls;
}

static double time_item(const shape *sh, int way, int T, const unsigned char *rows, const unsigned char *xq,
                        unsigned char *panel, long calls) {
    float y[ROWS * TR_Q4X_TILE_MAX];
    double t0 = tr_time_sec();
    for (long c = 0; c < calls; c++) run_item(sh, way, T, rows, xq, panel, NULL, y);
    double t1 = tr_time_sec();
    g_sink = y[0] + y[ROWS * T - 1];
    return (t1 - t0) * 1e9 / (double)calls;
}

static int l1_lines(const tr_kernels *K, int runs, double run_ms) {
    static unsigned char rows[ROWS * (NMAX / 256) * TR_Q4_K_BLOCK_BYTES];
    /* T prepared rows of n, tr_q4x_bytes(n) apart (a multiple of 64) */
    static unsigned char xq[TR_Q4X_XT_MAX * (NMAX / 256) * 1152] __attribute__((aligned(64)));
    static unsigned char panel[NMAX / 256 * 9728] __attribute__((aligned(64)));
    static float x[TR_Q4X_XT_MAX][NMAX];
    const tr_kernels *S = tr_kernels_tier("scalar");
    int bad = 0;
    long checked = 0;
    /* the pair lines: n 1024 and 2048, T 2 and 3; then the item lines at 2048 */
    enum { N_PAIR = 8, N_ITEM = 6 };
    typedef struct {
        int64_t n;
        int T, way, item;
        long calls;
        double ns[MAX_RUNS];
    } line;
    line L[N_PAIR + N_ITEM];
    int nl = 0;
    static const int64_t ns_of[2] = {1024, 2048};
    for (int ni = 0; ni < 2; ni++)
        for (int T = 2; T <= 3; T++)
            for (int way = WAY_DOT2; way <= WAY_XT; way++) L[nl++] = (line){ns_of[ni], T, way, 0, 0, {0}};
    for (int T = 2; T <= 3; T++)
        for (int way = 0; way < N_WAYS; way++)
            if (way_available(K, way, T)) L[nl++] = (line){NMAX, T, way, 1, 0, {0}};
    fill_q4_k(rows, ROWS * NMAX / 256);
    for (int t = 0; t < TR_Q4X_XT_MAX; t++)
        for (int i = 0; i < NMAX; i++) x[t][i] = (float)rnd() * 0x1p-30f - 1.0f;
    /* exactness: every line's outputs against scalar's dot_row, then its calls for ~run_ms */
    for (int l = 0; l < nl; l++) {
        const shape sh = {K, L[l].n, tr_row_bytes(TR_TYPE_Q4_K, L[l].n), tr_q4x_bytes(L[l].n)};
        const unsigned char *r = rows; /* rows of n: consecutive rows of sh.rb bytes in the same buffer */
        for (int t = 0; t < L[l].T; t++) K->q4x_prep(x[t], L[l].n, xq + (size_t)t * sh.xb);
        float y[ROWS * TR_Q4X_TILE_MAX];
        const int nr = L[l].item ? ROWS : 2;
        if (L[l].item) run_item(&sh, L[l].way, L[l].T, r, xq, panel, NULL, y);
        else if (L[l].way == WAY_XT) {
            const void *xp[TR_Q4X_XT_MAX];
            rows_of(xq, sh.xb, L[l].T, xp);
            K->q4x_dot_xt(r, r + sh.rb, xp, sh.n, L[l].T, y, ROWS);
        }
        else
            for (int t = 0; t < L[l].T; t++) K->q4x_dot2(r, r + sh.rb, xq + (size_t)t * sh.xb, sh.n, y + t * ROWS);
        for (int t = 0; t < L[l].T; t++)
            for (int rr = 0; rr < nr; rr++) {
                const float want = S->dot_row[TR_TYPE_Q4_K](r + (size_t)rr * sh.rb, x[t], sh.n);
                checked++;
                if (memcmp(&want, &y[t * ROWS + rr], sizeof want) != 0) {
                    printf("MISMATCH %s T=%d n=%lld row %d token %d\n", WAY_NAME[L[l].way], L[l].T, (long long)sh.n, rr, t);
                    bad = 1;
                }
            }
        double ns = L[l].item ? time_item(&sh, L[l].way, L[l].T, r, xq, panel, 200)
                              : time_pair(&sh, L[l].way, L[l].T, r, xq, 2000);
        L[l].calls = (long)(run_ms * 1e6 / ns) + 1;
    }
    if (bad || checked == 0) {
        printf("bench_q4x: exactness check failed (%ld outputs compared)\n", checked);
        return 1;
    }
    printf("# every line's outputs equal scalar's dot_row bit for bit: %ld compared\n", checked);
    for (int run = 0; run < runs; run++)
        for (int l = 0; l < nl; l++) {
            const shape sh = {K, L[l].n, tr_row_bytes(TR_TYPE_Q4_K, L[l].n), tr_q4x_bytes(L[l].n)};
            for (int t = 0; t < L[l].T; t++) K->q4x_prep(x[t], L[l].n, xq + (size_t)t * sh.xb);
            L[l].ns[run] = L[l].item ? time_item(&sh, L[l].way, L[l].T, rows, xq, panel, L[l].calls)
                                     : time_pair(&sh, L[l].way, L[l].T, rows, xq, L[l].calls);
        }
    printf("%-34s %9s %7s %9s\n", "line (tier's kernels)", "ns", "spread", "vs dot2");
    double base = 0;
    for (int l = 0; l < nl; l++) {
        double sp, ns = median_spread(L[l].ns, runs, &sp);
        if (L[l].way == WAY_DOT2) base = ns;
        char name[64];
        snprintf(name, sizeof name, "%s %s T=%d n=%lld", L[l].item ? "item of 16 rows," : "2 rows,", WAY_NAME[L[l].way],
                 L[l].T, (long long)L[l].n);
        printf("%-34s %9.1f %6.1f%% %8.3fx\n", name, ns, sp * 100, ns / base);
    }
    return 0;
}

/* the prep alone: one input row of 2048 prepared again and again in L1 (ns a call), its bytes scalar's first
 * (docs/MEASUREMENTS.md §The prep once a row: what the prep costs before its bit arithmetic) */
static int prep_line(const tr_kernels *K, int runs, double run_ms) {
    static float x[NMAX];
    static unsigned char xk[(NMAX / 256) * 1152] __attribute__((aligned(64)));
    static unsigned char xs[(NMAX / 256) * 1152] __attribute__((aligned(64)));
    const tr_kernels *S = tr_kernels_tier("scalar");
    for (int i = 0; i < NMAX; i++) x[i] = (float)rnd() * 0x1p-30f - 1.0f;
    K->q4x_prep(x, NMAX, xk);
    S->q4x_prep(x, NMAX, xs);
    if (memcmp(xk, xs, tr_q4x_bytes(NMAX)) != 0) {
        printf("bench_q4x: the prep's bytes differ from scalar's\n");
        return 1;
    }
    double ns[MAX_RUNS];
    long calls = 1000;
    for (int run = -1; run < runs; run++) {
        double t0 = tr_time_sec();
        for (long c = 0; c < calls; c++) K->q4x_prep(x, NMAX, xk);
        double t = (tr_time_sec() - t0) * 1e9 / (double)calls;
        if (run < 0) calls = (long)(run_ms * 1e6 / t) + 1; /* the first run sizes the others */
        else ns[run] = t;
    }
    g_sink = (float)xk[5];
    double sp, med = median_spread(ns, runs, &sp);
    printf("%-34s %9.1f %6.1f%%\n", "prep, one row of 2048", med, sp * 100);
    return 0;
}

/* ---- from RAM ------------------------------------------------------------------------------------------- */

#define RAM_UNIT_ROWS 1024
#define RAM_PASSES 2

/* every byte read once: four chains of 8-byte xors */
static float read_bytes(const unsigned char *p, size_t bytes) {
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (size_t i = 0; i + 32 <= bytes; i += 32) {
        uint64_t v[4];
        memcpy(v, p + i, sizeof v);
        a ^= v[0];
        b ^= v[1];
        c ^= v[2];
        d ^= v[3];
    }
    return (float)((a ^ b ^ c ^ d) & 0xFFFF);
}

typedef struct {
    shape sh;
    const unsigned char *w;   /* n_items * ROWS rows */
    const unsigned char *xq;  /* POOL prepared rows */
    int way, T, rotate;       /* way -1: the plain read */
    float *y;                 /* n_items * ROWS * T */
    unsigned char *panels;    /* one a worker */
    size_t panel_bytes;
    int64_t n_items;
    volatile float sink[64];
} ram_job;

static void ram_body(void *ctx, int64_t begin, int64_t end, int worker) {
    ram_job *m = (ram_job *)ctx;
    const size_t item_bytes = (size_t)ROWS * m->sh.rb;
    if (m->way < 0) {
        m->sink[worker & 63] = read_bytes(m->w + (size_t)begin * item_bytes, (size_t)(end - begin) * item_bytes);
        return;
    }
    unsigned char *panel = m->panels + (size_t)worker * m->panel_bytes;
    for (int64_t it = begin; it < end; it++) {
        const unsigned char *rows = m->w + (size_t)it * item_bytes;
        const int64_t first = m->rotate ? (it * 7) % (POOL - m->T + 1) : 0;
        const void *next = it + 1 < end ? rows + item_bytes : NULL;
        run_item(&m->sh, m->way, m->T, rows, m->xq + (size_t)first * m->sh.xb, panel, next,
                 m->y + (size_t)it * ROWS * (size_t)m->T);
    }
}

static int ram_lines(const tr_kernels *K, int threads, int runs, int mib) {
    const shape sh = {K, NMAX, tr_row_bytes(TR_TYPE_Q4_K, NMAX), tr_q4x_bytes(NMAX)};
    const int64_t unit_bytes = (int64_t)RAM_UNIT_ROWS * (int64_t)sh.rb;
    const int64_t units = ((int64_t)mib << 20) / unit_bytes;
    const int64_t n_items = units * RAM_UNIT_ROWS / ROWS;
    const size_t bytes = (size_t)n_items * ROWS * sh.rb;
    void *wb = NULL, *xb = NULL, *pb = NULL;
    unsigned char *w = aligned64(bytes, &wb), *xq = aligned64((size_t)POOL * sh.xb, &xb);
    unsigned char *panels = aligned64((size_t)threads * TR_Q4X_PANEL_BYTES(NMAX), &pb);
    float *y = (float *)malloc((size_t)n_items * ROWS * TR_Q4X_XT_MAX * sizeof(float));
    float *ref = (float *)malloc((size_t)n_items * ROWS * TR_Q4X_XT_MAX * sizeof(float));
    static float x[NMAX];
    if (!w || !xq || !panels || !y || !ref) {
        printf("ram: no memory for %zu bytes\n", bytes);
        free(wb), free(xb), free(pb), free(y), free(ref);
        return 1;
    }
    fill_q4_k(w, RAM_UNIT_ROWS * NMAX / 256);
    for (int64_t u = 1; u < units; u++) memcpy(w + (size_t)u * (size_t)unit_bytes, w, (size_t)unit_bytes);
    for (int p = 0; p < POOL; p++) {
        for (int i = 0; i < NMAX; i++) x[i] = (float)rnd() * 0x1p-30f - 1.0f;
        K->q4x_prep(x, NMAX, xq + (size_t)p * sh.xb);
    }
    /* the lines: the read, dot2 at T = 1, then at T = 2 and 3 each way, same rows and rotating */
    typedef struct {
        int way, T, rotate;
        double gbs[MAX_RUNS * RAM_PASSES];
    } line;
    static line L[2 + 2 * 2 * N_WAYS];
    int nl = 0;
    L[nl++] = (line){-1, 0, 0, {0}};
    L[nl++] = (line){WAY_DOT2, 1, 0, {0}};
    for (int T = 2; T <= 3; T++)
        for (int rot = 0; rot < 2; rot++)
            for (int way = 0; way < N_WAYS; way++)
                if (way_available(K, way, T)) L[nl++] = (line){way, T, rot, {0}};
    tr_pool *pool = tr_pool_create(threads);
    ram_job m;
    memset(&m, 0, sizeof m);
    m.sh = sh;
    m.w = w;
    m.xq = xq;
    m.y = y;
    m.panels = panels;
    m.panel_bytes = TR_Q4X_PANEL_BYTES(NMAX);
    m.n_items = n_items;
    int bad = 0;
    for (int round = 0; round < runs; round++)
        for (int l = 0; l < nl; l++) {
            m.way = L[l].way;
            m.T = L[l].T;
            m.rotate = L[l].rotate;
            tr_parallel_for(pool, n_items, 1, ram_body, &m); /* untimed */
            if (round == 0 && m.way >= 0) {
                const size_t ob = (size_t)n_items * ROWS * (size_t)m.T * sizeof(float);
                if (m.way == WAY_DOT2) memcpy(ref, y, ob);
                else if (memcmp(ref, y, ob) != 0) {
                    printf("ram: %s T=%d%s differs from dot2\n", WAY_NAME[m.way], m.T, m.rotate ? " rotating" : "");
                    bad = 1;
                }
            }
            for (int pass = 0; pass < RAM_PASSES; pass++) {
                double t0 = tr_time_sec();
                tr_parallel_for(pool, n_items, 1, ram_body, &m);
                L[l].gbs[round * RAM_PASSES + pass] = (double)bytes / ((tr_time_sec() - t0) * 1e9);
            }
        }
    tr_pool_destroy(pool);
    printf("ram: %d threads, %.2f GiB of Q4_K rows of %d in items of %d rows, %d rounds x %d passes\n", threads,
           (double)bytes / (1 << 30), NMAX, ROWS, runs, RAM_PASSES);
    printf("%-34s %8s %7s %9s\n", "line", "GB/s", "spread", "vs read");
    double read = 1;
    for (int l = 0; l < nl; l++) {
        double sp, g = median_spread(L[l].gbs, runs * RAM_PASSES, &sp);
        if (L[l].way < 0) read = g;
        char name[64];
        if (L[l].way < 0) snprintf(name, sizeof name, "read (ceiling)");
        else snprintf(name, sizeof name, "%s T=%d%s", WAY_NAME[L[l].way], L[l].T, L[l].rotate ? " new rows" : "");
        printf("%-34s %8.1f %6.1f%% %8.3fx\n", name, g, sp * 100, g / read);
    }
    free(wb), free(xb), free(pb), free(y), free(ref);
    return bad;
}

/* ---- question 84: what lies past a thread's region ------------------------------------------------------ */
/* Present memory past the KV's streams' ends slowed their reads 1-2.4% (docs/MEASUREMENTS.md §The KV's pages
 * touched in time); a decode token's weight calls end a thread's region ~3000 times, each in present memory: the
 * next thread's rows (the dense calls), an expert nobody chose (the experts' calls). A layout maps (set, call k,
 * thread t) to a slot of an arena, KP = K * P slots a set:
 *   next     set * KP + k * P + t              a call's regions side by side, as a matrix's rows over the pool
 *   guard    as next, the page after every region dropped (decommitted): nothing present past any end
 *   own      set * KP + t * K + k              a thread's regions of the K calls one after the other
 *   ownrev   set * KP + t * K + (K - 1 - k)    the same slots, a thread's calls in reverse: past its end the
 *                                              region it read one call before (in its L2), never the next one
 *   next0    as next, on slots of a region alone (own's arena)
 * next and guard run on two arenas of region-and-page slots, the pages after the regions dropped in one of them
 * (the other's written), the roles swapped every round; own, ownrev and next0 on one arena of region slots.
 * The RAM's speed drifts by 10-30% over seconds on this machine, so two layouts race set by set: a pair of sets
 * (one of each, the order by a hash of the pair's index), each set K calls from RAM (the second arm reads the
 * sets half an arena away), a round's value the median of its pairs' time ratios (a preempted set moves one pair). */

#define ENDS_K 4
#define ENDS_ARENA ((size_t)512 << 20)
#define ENDS_MAXP 64
#define ENDS_PAGE 4096
#define ENDS_MAX_SETS 64
enum { L_NEXT, L_GUARD, L_OWN, L_OWNREV, L_NEXT0, N_LAYOUTS };
static const char *const LAYOUT_NAME[N_LAYOUTS] = {"next", "guard", "own", "ownrev", "next0"};

static unsigned char *arena_map(size_t bytes) {
#if defined(_WIN32)
    return (unsigned char *)VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : (unsigned char *)p;
#endif
}

/* [p, p + bytes), page-aligned: no page present until its next write */
static int arena_drop(unsigned char *p, size_t bytes) {
#if defined(_WIN32)
    return VirtualFree(p, bytes, MEM_DECOMMIT) && VirtualAlloc(p, bytes, MEM_COMMIT, PAGE_READWRITE) == p ? 0 : 1;
#else
    return madvise(p, bytes, MADV_DONTNEED);
#endif
}

static void arena_unmap(unsigned char *p, size_t bytes) {
#if defined(_WIN32)
    (void)bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

/* the OS's own work after a drop or a refill (zeroing the freed pages) done before the next timing */
static void ends_settle(void) {
#if defined(_WIN32)
    Sleep(30);
#else
    const double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < 0.03) {
    }
#endif
}

typedef struct {
    shape sh;
    const unsigned char *region[ENDS_MAXP]; /* this call's region of each thread */
    int64_t rows;                           /* a region's rows */
    const unsigned char *xq;
    int plain;
    float *y; /* P * rows: every thread's outputs of the last call */
    volatile float sink[64];
} ends_job;

static void ends_body(void *ctx, int64_t begin, int64_t end, int worker) {
    ends_job *m = (ends_job *)ctx;
    for (int64_t t = begin; t < end; t++) {
        const unsigned char *w = m->region[t];
        if (m->plain) {
            m->sink[worker & 63] = read_bytes(w, (size_t)m->rows * m->sh.rb);
            continue;
        }
        float *y = m->y + t * m->rows;
        for (int64_t r = 0; r < m->rows; r += 2)
            m->sh.K->q4x_dot2(w + (size_t)r * m->sh.rb, w + (size_t)(r + 1) * m->sh.rb, m->xq, m->sh.n, y + r);
    }
}

static size_t ends_slot(int layout, int64_t set, int k, int t, int P) {
    const int64_t kp = (int64_t)ENDS_K * P;
    if (layout == L_OWN) return (size_t)(set * kp + (int64_t)t * ENDS_K + k);
    if (layout == L_OWNREV) return (size_t)(set * kp + (int64_t)t * ENDS_K + (ENDS_K - 1 - k));
    return (size_t)(set * kp + (int64_t)k * P + t);
}

typedef struct {
    int layout;
    unsigned char *arena;
    size_t slot;
} ends_arm;

/* one set's K calls back to back: seconds */
static double ends_set(tr_pool *pool, ends_job *m, const ends_arm *a, int64_t s, int P) {
    const double t0 = tr_time_sec();
    for (int k = 0; k < ENDS_K; k++) {
        for (int t = 0; t < P; t++) m->region[t] = a->arena + ends_slot(a->layout, s, k, t, P) * a->slot;
        tr_parallel_for(pool, P, 1, ends_body, m);
    }
    return tr_time_sec() - t0;
}

/* every set of both arms, a pair at a time (arm 0's set i, arm 1's set i + sets / 2), the pair's order by a hash
 * of i: ratio[i] = arm 0's time / arm 1's (above 1: arm 1 faster); ta, tb: the pairs' times */
static void ends_race(tr_pool *pool, ends_job *m, const ends_arm *arm, int64_t sets, int P, double *ratio, double *ta,
                      double *tb) {
    for (int64_t i = 0; i < sets; i++) {
        const int64_t j = (i + sets / 2) % sets;
        const int first = (int)(((uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull) >> 63);
        double t[2];
        t[first] = ends_set(pool, m, &arm[first], first ? j : i, P);
        t[!first] = ends_set(pool, m, &arm[!first], first ? i : j, P);
        ratio[i] = t[0] / t[1];
        ta[i] = t[0];
        tb[i] = t[1];
    }
}

/* median and the standard error of the mean */
static double med_se(const double *v, int n, double *se) {
    double c[MAX_RUNS], s = 0, s2 = 0, sp;
    for (int i = 0; i < n; i++) {
        c[i] = v[i];
        s += v[i];
        s2 += v[i] * v[i];
    }
    const double mean = s / n, var = n > 1 ? (s2 - n * mean * mean) / (n - 1) : 0;
    *se = var > 0 ? sqrt(var / n) : 0;
    return median_spread(c, n, &sp);
}

/* the page after every region of an arena of region-and-page slots: dropped, or written back */
static int ends_tails(unsigned char *arena, size_t n_slots, size_t slot, size_t region, int drop) {
    for (size_t s = 0; s < n_slots; s++) {
        unsigned char *page = arena + s * slot + region;
        if (drop) {
            if (arena_drop(page, ENDS_PAGE) != 0) return 1;
        } else memset(page, 0x5A, ENDS_PAGE);
    }
    return 0;
}

static int ends_shape(const tr_kernels *K, tr_pool *pool, int P, int runs, int64_t rows, const char *what) {
    const shape sh = {K, NMAX, tr_row_bytes(TR_TYPE_Q4_K, NMAX), tr_q4x_bytes(NMAX)};
    const size_t region = (size_t)rows * sh.rb; /* a multiple of the page: rows a multiple of 32 */
    const size_t paged = region + ENDS_PAGE;
    const int64_t kp = (int64_t)ENDS_K * P;
    int64_t sets = (int64_t)(ENDS_ARENA / (paged * (size_t)kp));
    if (sets > ENDS_MAX_SETS) sets = ENDS_MAX_SETS;
    const size_t n_slots = (size_t)(sets * kp);
    /* arenas 0 and 1: region-and-page slots (next and guard); 2: region slots */
    const size_t slot_of[3] = {paged, paged, region};
    unsigned char *arena[3];
    for (int a = 0; a < 3; a++) arena[a] = arena_map(n_slots * slot_of[a]);
    void *ub = NULL, *xb = NULL;
    unsigned char *unit = aligned64(region, &ub), *xq = aligned64(sh.xb, &xb);
    float *y = (float *)malloc((size_t)P * (size_t)rows * sizeof(float)), *ref = (float *)malloc((size_t)rows * sizeof(float));
    static float x[NMAX];
    int bad = arena[0] == NULL || arena[1] == NULL || arena[2] == NULL || unit == NULL || xq == NULL || y == NULL ||
              ref == NULL || sets < 4 || region % ENDS_PAGE != 0;
    if (bad) printf("ends: no memory, or the arena holds fewer than 4 sets\n");
    else {
        fill_q4_k(unit, rows * NMAX / 256);
        for (int a = 0; a < 3; a++)
            for (size_t s = 0; s < n_slots; s++) {
                memcpy(arena[a] + s * slot_of[a], unit, region);
                if (slot_of[a] > region) memset(arena[a] + s * slot_of[a] + region, 0x5A, ENDS_PAGE);
            }
        for (int i = 0; i < NMAX; i++) x[i] = (float)rnd() * 0x1p-30f - 1.0f;
        K->q4x_prep(x, NMAX, xq);
        for (int64_t r = 0; r < rows; r += 2)
            K->q4x_dot2(unit + (size_t)r * sh.rb, unit + (size_t)(r + 1) * sh.rb, xq, NMAX, ref + r);
    }

    ends_job m;
    memset(&m, 0, sizeof m);
    m.sh = sh;
    m.rows = rows;
    m.xq = xq;
    m.y = y;
    /* the races: arm 0 against arm 1 */
    static const int race[][2] = {{L_NEXT, L_GUARD}, {L_OWNREV, L_OWN}, {L_NEXT0, L_OWN}};
    enum { N_RACES = sizeof race / sizeof race[0] };
    static double rr[N_RACES][2][MAX_RUNS], ga[N_RACES][2][MAX_RUNS], gb[N_RACES][2][MAX_RUNS];
    double ratio[2 * ENDS_MAX_SETS], ta[2 * ENDS_MAX_SETS], tb[2 * ENDS_MAX_SETS], sp;
    const double set_bytes = (double)kp * (double)region;
    int guard_arena = 1;
    for (int round = 0; round < runs && !bad; round++)
        for (int q = 0; q < N_RACES && !bad; q++) {
            ends_arm arm[2];
            for (int e = 0; e < 2; e++) {
                arm[e].layout = race[q][e];
                arm[e].arena = arena[2];
                arm[e].slot = slot_of[2];
            }
            if (q == 0) { /* next on one arena, guard on the other: the roles swapped every round */
                if (round > 0) bad |= ends_tails(arena[guard_arena], n_slots, paged, region, 0);
                guard_arena = round & 1;
                bad |= ends_tails(arena[guard_arena], n_slots, paged, region, 1);
                ends_settle();
                arm[0].arena = arena[!guard_arena];
                arm[1].arena = arena[guard_arena];
                arm[0].slot = arm[1].slot = paged;
            }
            for (int h = 0; h < 2 && !bad; h++) {
                m.plain = (h + round) & 1;
                ends_race(pool, &m, arm, sets, P, ratio, ta, tb); /* untimed: then every set from RAM */
                for (int t = 0; t < P && !m.plain && !bad; t++)
                    bad = memcmp(y + (size_t)t * (size_t)rows, ref, (size_t)rows * sizeof(float)) != 0;
                ends_race(pool, &m, arm, sets, P, ratio, ta, tb);
                ends_race(pool, &m, arm, sets, P, ratio + sets, ta + sets, tb + sets);
                rr[q][m.plain][round] = median_spread(ratio, (int)(2 * sets), &sp);
                ga[q][m.plain][round] = set_bytes / (median_spread(ta, (int)(2 * sets), &sp) * 1e9);
                gb[q][m.plain][round] = set_bytes / (median_spread(tb, (int)(2 * sets), &sp) * 1e9);
            }
            if (bad) printf("ends: %s against %s: outputs differ from one thread's dot2, or a page was not dropped\n",
                            LAYOUT_NAME[race[q][1]], LAYOUT_NAME[race[q][0]]);
        }
    if (!bad) {
        printf("ends: %s, %d threads, %lld rows of %d a region (%.0f KiB), %d calls a set, %lld sets, %d rounds; "
               "a round: the median pair\n",
               what, P, (long long)rows, NMAX, (double)region / 1024, ENDS_K, (long long)sets, runs);
        for (int h = 0; h < 2; h++)
            for (int q = 0; q < N_RACES; q++) {
                double se, sa, sb;
                const double r = med_se(rr[q][h], runs, &se), a = med_se(ga[q][h], runs, &sa),
                             b = med_se(gb[q][h], runs, &sb);
                printf("  %-5s %-7s against %-7s %.4f +- %.4f   (%6.2f and %6.2f GB/s)\n", h ? "read" : "dot2",
                       LAYOUT_NAME[race[q][1]], LAYOUT_NAME[race[q][0]], r, se, b, a);
            }
    }
    g_sink = y != NULL ? y[0] + m.sink[0] : 0;
    for (int a = 0; a < 3; a++)
        if (arena[a] != NULL) arena_unmap(arena[a], n_slots * slot_of[a]);
    free(ub), free(xb), free(y), free(ref);
    return bad;
}

static int ends_lines(const tr_kernels *K, int P, int runs) {
    if (P > ENDS_MAXP) P = ENDS_MAXP;
    tr_pool *pool = tr_pool_create(P);
    if (pool == NULL) return 1;
    const int r = ends_shape(K, pool, P, runs, 2048 / P, "the dense call (2048 rows)") |
                  ends_shape(K, pool, P, runs, 8 * 1024 / P, "the experts' call (8 x 1024 rows)");
    tr_pool_destroy(pool);
    return r;
}

/* ---- piece 3, B: the idle workers ------------------------------------------------------------------------ */
/* Between two weight calls the calling thread runs serial steps (a norm, the routing, the KV's write) while the
 * other threads spin, and the RAM rests (docs/MEASUREMENTS.md §The idle workers). A step here: the hints for the
 * next call posted (tr_pool_hint: chunks 1..P-1, each its region's first X bytes), a serial step of W us on the
 * calling thread, then the call from RAM (dot2, T = 1, a region a thread, the regions side by side as a matrix's
 * rows over the pool). The serial step of three kinds: spin (the clock alone), l2 (sums over 64 KiB in L2), ram
 * (reads a cold buffer, going on where it stopped). Arm A with the pool's hints off, arm B on (T0 or T2), raced set
 * by set as the ends lines are (docs/LESSONS.md #254); each set S steps over consecutive calls, every call's
 * regions read again only after the whole arena. */

#define IDLE_ARENA ((size_t)512 << 20)
#define IDLE_COLD ((size_t)256 << 20)
#define IDLE_L2 ((size_t)64 << 10)
#define IDLE_MAX_SETS 64
#define IDLE_LIST 8
enum { K_SPIN, K_L2, K_RAM, N_KINDS };
static const char *const KIND_NAME[N_KINDS] = {"spin", "l2", "ram"};

typedef struct {
    int kind;
    double w; /* seconds */
    const unsigned char *l2;
    const unsigned char *cold;
    size_t cold_at;
    uint64_t sink;
} idle_serial;

static void idle_serial_run(idle_serial *s) {
    const double t0 = tr_time_sec();
    if (s->kind == K_SPIN) {
        while (tr_time_sec() - t0 < s->w) {
        }
        return;
    }
    size_t at = 0;
    do {
        if (s->kind == K_L2) {
            s->sink += (uint64_t)read_bytes(s->l2 + at, 4096);
            at = (at + 4096) % IDLE_L2;
        } else {
            s->sink += (uint64_t)read_bytes(s->cold + s->cold_at, 4096);
            s->cold_at = (s->cold_at + 4096) % IDLE_COLD;
        }
    } while (tr_time_sec() - t0 < s->w);
}

typedef struct {
    int level; /* the pool's hints: 0 off */
    size_t x;  /* bytes a hint */
} idle_arm;

/* S steps over calls c0.. of the arena: seconds */
static double idle_set(tr_pool *pool, ends_job *m, idle_serial *ss, const unsigned char *arena, size_t call_bytes,
                       size_t region, int64_t c0, int S, int P, const idle_arm *a) {
    tr_pool_set_hints(pool, a->level);
    const double t0 = tr_time_sec();
    for (int i = 0; i < S; i++) {
        const unsigned char *call = arena + (size_t)(c0 + i) * call_bytes;
        for (int t = 0; t < P; t++) m->region[t] = call + (size_t)t * region;
        for (int t = 1; t < P; t++) tr_pool_hint(pool, t, m->region[t], a->x);
        idle_serial_run(ss);
        tr_parallel_for(pool, P, 1, ends_body, m);
    }
    return tr_time_sec() - t0;
}

static int parse_list(const char *s, double *v) {
    int n = 0;
    while (s != NULL && *s != '\0' && n < IDLE_LIST) {
        v[n++] = atof(s);
        s = strchr(s, ',');
        if (s != NULL) s++;
    }
    return n;
}

static int idle_shape(const tr_kernels *K, tr_pool *pool, int P, int runs, int64_t rows, int S, const char *what,
                      const double *ws, int nw, const double *xs, int nx, const int *kinds, int nk) {
    const shape sh = {K, NMAX, tr_row_bytes(TR_TYPE_Q4_K, NMAX), tr_q4x_bytes(NMAX)};
    const size_t region = (size_t)rows * sh.rb, call_bytes = region * (size_t)P;
    const int64_t n_calls = (int64_t)(IDLE_ARENA / call_bytes);
    int64_t sets = n_calls / S;
    if (sets > IDLE_MAX_SETS) sets = IDLE_MAX_SETS;
    unsigned char *arena = arena_map((size_t)(sets * S) * call_bytes), *cold = arena_map(IDLE_COLD);
    void *ub = NULL, *xb = NULL, *lb = NULL;
    unsigned char *unit = aligned64(region, &ub), *xq = aligned64(sh.xb, &xb), *l2 = aligned64(IDLE_L2, &lb);
    float *y = (float *)malloc((size_t)P * (size_t)rows * sizeof(float)), *ref = (float *)malloc((size_t)rows * sizeof(float));
    static float x[NMAX];
    int bad = arena == NULL || cold == NULL || unit == NULL || xq == NULL || l2 == NULL || y == NULL || ref == NULL ||
              sets < 4;
    if (bad) printf("idle: no memory, or the arena holds fewer than 4 sets\n");
    else {
        fill_q4_k(unit, rows * NMAX / 256);
        for (int64_t r = 0; r < sets * S * P; r++) memcpy(arena + (size_t)r * region, unit, region);
        memset(cold, 0x3C, IDLE_COLD);
        memset(l2, 0x5A, IDLE_L2);
        for (int i = 0; i < NMAX; i++) x[i] = (float)rnd() * 0x1p-30f - 1.0f;
        K->q4x_prep(x, NMAX, xq);
        for (int64_t r = 0; r < rows; r += 2)
            K->q4x_dot2(unit + (size_t)r * sh.rb, unit + (size_t)(r + 1) * sh.rb, xq, NMAX, ref + r);
        printf("idle: %s, %d threads, %.0f KiB a region, %d steps a set, %lld sets, %d rounds; B/A above 1: B faster; "
               "lines: brought a hint (of X)\n",
               what, P, (double)region / 1024, S, (long long)sets, runs);
    }
    ends_job m;
    memset(&m, 0, sizeof m);
    m.sh = sh;
    m.rows = rows;
    m.xq = xq;
    m.y = y;
    idle_serial ss;
    memset(&ss, 0, sizeof ss);
    ss.l2 = l2;
    ss.cold = cold;
    double ratio[2 * IDLE_MAX_SETS], ta[2 * IDLE_MAX_SETS], tb[2 * IDLE_MAX_SETS], sp;
    for (int ik = 0; ik < nk && !bad; ik++)
        for (int iw = 0; iw < nw && !bad; iw++)
            for (int ix = 0; ix < nx && !bad; ix++)
                for (int level = 1; level <= 2 && !bad; level++) {
                    ss.kind = kinds[ik];
                    ss.w = ws[iw] * 1e-6;
                    idle_arm arm[2] = {{0, 0}, {level, (size_t)(xs[ix] * 1024)}};
                    double rr[MAX_RUNS], sa[MAX_RUNS], sb[MAX_RUNS];
                    uint64_t h0 = 0, l0 = 0;
                    for (int t = 1; t < P; t++) {
                        uint64_t h, l;
                        tr_pool_hint_counts(pool, t, &h, &l);
                        h0 += h;
                        l0 += l;
                    }
                    for (int round = 0; round < runs && !bad; round++) {
                        for (int pass = 0; pass < 3; pass++) /* the first untimed: every set from RAM after it */
                            for (int64_t i = 0; i < sets; i++) {
                                const int64_t j = (i + sets / 2) % sets;
                                const int first = (int)(((uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull) >> 63);
                                double t[2];
                                t[first] = idle_set(pool, &m, &ss, arena, call_bytes, region, (first ? j : i) * S, S,
                                                    P, &arm[first]);
                                t[!first] = idle_set(pool, &m, &ss, arena, call_bytes, region, (first ? i : j) * S, S,
                                                     P, &arm[!first]);
                                if (pass == 0) continue;
                                const int64_t k = (pass - 1) * sets + i;
                                ratio[k] = t[0] / t[1];
                                ta[k] = t[0];
                                tb[k] = t[1];
                            }
                        for (int t = 0; t < P && !bad; t++)
                            bad = memcmp(y + (size_t)t * (size_t)rows, ref, (size_t)rows * sizeof(float)) != 0;
                        rr[round] = median_spread(ratio, (int)(2 * sets), &sp);
                        sa[round] = median_spread(ta, (int)(2 * sets), &sp) / S * 1e6;
                        sb[round] = median_spread(tb, (int)(2 * sets), &sp) / S * 1e6;
                    }
                    if (bad) {
                        printf("idle: the call's outputs differ from one thread's dot2\n");
                        break;
                    }
                    uint64_t h1 = 0, l1 = 0;
                    for (int t = 1; t < P; t++) {
                        uint64_t h, l;
                        tr_pool_hint_counts(pool, t, &h, &l);
                        h1 += h;
                        l1 += l;
                    }
                    double se, s1, s2;
                    const double r = med_se(rr, runs, &se), a = med_se(sa, runs, &s1), b = med_se(sb, runs, &s2);
                    printf("  %-4s W %4.1f us  X %4.0f KiB  %s: B/A %.4f +- %.4f   (A %7.2f, B %7.2f us a step)   "
                           "lines %5.1f of %4.0f\n",
                           KIND_NAME[ss.kind], ws[iw], xs[ix], level == 1 ? "T0" : "T2", r, se, a, b,
                           h1 > h0 ? (double)(l1 - l0) / (double)(h1 - h0) : 0.0, xs[ix] * 16);
                    fflush(stdout);
                }
    tr_pool_set_hints(pool, 0);
    g_sink = y != NULL ? y[0] + m.sink[0] + (float)(ss.sink & 1) : 0;
    if (arena != NULL) arena_unmap(arena, (size_t)(sets * S) * call_bytes);
    if (cold != NULL) arena_unmap(cold, IDLE_COLD);
    free(ub), free(xb), free(lb), free(y), free(ref);
    return bad;
}

static int idle_lines(const tr_kernels *K, int P, int runs, const char *wl, const char *xl, const char *kl) {
    double ws[IDLE_LIST], xs[IDLE_LIST];
    int kinds[N_KINDS], nk = 0;
    const int nw = parse_list(wl != NULL ? wl : "1.5,3,6", ws), nx = parse_list(xl != NULL ? xl : "8,16,32,64", xs);
    for (int k = 0; k < N_KINDS; k++)
        if (kl == NULL || strstr(kl, KIND_NAME[k]) != NULL) kinds[nk++] = k;
    if (P > ENDS_MAXP) P = ENDS_MAXP;
    tr_pool *pool = tr_pool_create(P);
    if (pool == NULL) return 1;
    const int r = idle_shape(K, pool, P, runs, 2048 / P, 8, "the dense call (2048 rows)", ws, nw, xs, nx, kinds, nk) |
                  idle_shape(K, pool, P, runs, 8 * 1024 / P, 2, "the experts' call (8 x 1024 rows)", ws, nw, xs, nx,
                             kinds, nk);
    tr_pool_destroy(pool);
    return r;
}

int main(int argc, char **argv) {
    int runs = 11, ram = 0, mib = 1024, ends = 0, idle = 0;
    const char *wl = NULL, *xl = NULL, *kl = NULL;
    double run_ms = 5;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ms") == 0 && i + 1 < argc) run_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--ram") == 0 && i + 1 < argc) ram = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mib") == 0 && i + 1 < argc) mib = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ends") == 0 && i + 1 < argc) ends = atoi(argv[++i]);
        else if (strcmp(argv[i], "--idle") == 0 && i + 1 < argc) idle = atoi(argv[++i]);
        else if (strcmp(argv[i], "--w") == 0 && i + 1 < argc) wl = argv[++i];
        else if (strcmp(argv[i], "--x") == 0 && i + 1 < argc) xl = argv[++i];
        else if (strcmp(argv[i], "--kinds") == 0 && i + 1 < argc) kl = argv[++i];
        else {
            fprintf(stderr, "usage: bench_q4x [--runs N] [--ms M] [--ram P] [--mib M] [--ends P] [--idle P [--w us,..] "
                            "[--x KiB,..] [--kinds spin,l2,ram]]\n");
            return 2;
        }
    }
    if (runs < 3) runs = 3;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
    if (mib < 64) mib = 64;
    tr_kernels_init();
    const tr_kernels *K = tr_kernels_get();
    if (K->q4x_dot_xt == NULL) {
        printf("bench_q4x: tier %s has no q4x_dot_xt: nothing to measure\n", K->tier);
        return 0;
    }
    printf("# tier %s, %d runs, median and spread (max-min)/median\n", K->tier, runs);
    if (ram > 0) return ram_lines(K, ram, runs, mib);
    if (ends > 0) return ends_lines(K, ends, runs);
    if (idle > 0) return idle_lines(K, idle, runs, wl, xl, kl);
    tr_pool *pool = tr_pool_create(1); /* pins this thread to the first slot */
    const int r = l1_lines(K, runs, run_ms) | prep_line(K, runs, run_ms);
    tr_pool_destroy(pool);
    return r;
}
