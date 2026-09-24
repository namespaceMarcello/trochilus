/* test_gpu_attn.c — the GPU attention (src/backend/gpu_attn.c) against tr_attention_group, bit for bit.
 *
 * Skipped, with one line and exit 0, where tr_gpu_open fails: no driver, no device of compute
 * capability 8.0, the Linux container. Otherwise, on synthetic heads, the branches it covers:
 *   - both ways the engine fills the VRAM cache: rows of prefill passes (tr_gpu_attn_write in pieces
 *     of 1 to 512 rows, then single rows), and decode tokens (tr_gpu_attn_decode appends its own row,
 *     three decodes in a row per case)
 *   - n_pos 1..4096 across every border of the kernels: 128-position blocks, 32-position tiles, the
 *     32-row pieces of kv_write; past 4096 (a second session): the second chunk of attn_v's sum and a
 *     head maximum over more than 32 block maxima; 2 layers of 16 heads, and 3 heads with a context
 *     of 300 (a cap that is not the context)
 *   - data: random; wide scores (exponentials that underflow to 0 and to subnormals); values of -0 (the
 *     sum must start at +0); equal scores; scores that land exactly on arguments of tr_expf's
 *     exception table
 *   - every row a decode must not read yet, its own included, holds NaN until the decode appends it:
 *     a read past the token, or an append to the wrong place, poisons the output
 *   - each output compared as bits with tr_attention_group per head under the scalar table (the
 *     definition) and under the active tier (the engine's)
 *   - the refusals: head_dim 64, a position or a layer outside the cache
 *   - the warm-up of a prompt's last pass (tr_gpu_attn_warm): the cases run with its warp napping,
 *     and 16 calls in a row leave one warp, not 16 queued ones (a free after them under 0.5 s)
 * Counters fail the test when a branch never ran: decodes, poisoned rows, arguments through the
 * exception table, underflows, subnormal exponentials. Seen red under every mutation of
 * tools/mutate_gpu.sh. One run is well under 60 s. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/backend/gpu_attn.h"
#include "../src/base/platform.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

#define HD 128
#define N_DECODE 3   /* decode tokens at the end of every case */
#define POISON 40    /* NaN rows from the first decode's position on */
#define MAX_HEADS 16

enum { RANDOM, WIDE, NEGZERO, EQUAL, EXCEPTION, N_KIND };
static const char *const kind_name[N_KIND] = {"random", "wide", "negzero", "equal", "exception"};

typedef struct {
    int64_t decodes, poisoned, table, underflow, subnormal, compared, bad_scalar, bad_active;
    int printed;
} tally;

typedef struct {
    int64_t n_head, n_ctx;
    float *kh[MAX_HEADS], *vh[MAX_HEADS]; /* one head's positions in a row, as the engine's cache */
    float *krow, *vrow;                   /* the same as rows [pos][n_head * 128], as tr_kv_write takes them */
    float *nan_rows, *q, *out, *ref, *scores;
} bufs;

static float rnd(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(int32_t)(*s >> 32) * (1.0f / 2147483648.0f);
}

static uint32_t bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static int bufs_init(bufs *B, int64_t n_head, int64_t n_ctx) {
    memset(B, 0, sizeof *B);
    B->n_head = n_head;
    B->n_ctx = n_ctx;
    size_t row = (size_t)n_head * HD, all = (size_t)n_ctx * row;
    for (int64_t h = 0; h < n_head; h++) {
        B->kh[h] = malloc((size_t)n_ctx * HD * sizeof(float));
        B->vh[h] = malloc((size_t)n_ctx * HD * sizeof(float));
        if (B->kh[h] == NULL || B->vh[h] == NULL) return -1;
    }
    B->krow = malloc(all * sizeof(float));
    B->vrow = malloc(all * sizeof(float));
    B->nan_rows = malloc(POISON * row * sizeof(float));
    B->q = malloc(row * sizeof(float));
    B->out = malloc(row * sizeof(float));
    B->ref = malloc(row * sizeof(float));
    B->scores = malloc((size_t)n_ctx * sizeof(float));
    if (B->krow == NULL || B->vrow == NULL || B->nan_rows == NULL || B->q == NULL || B->out == NULL ||
        B->ref == NULL || B->scores == NULL)
        return -1;
    for (size_t i = 0; i < POISON * row; i++) B->nan_rows[i] = from_bits(0x7FC00000u);
    return 0;
}

static void bufs_free(bufs *B) {
    for (int64_t h = 0; h < B->n_head; h++) {
        free(B->kh[h]);
        free(B->vh[h]);
    }
    free(B->krow);
    free(B->vrow);
    free(B->nan_rows);
    free(B->q);
    free(B->out);
    free(B->ref);
    free(B->scores);
}

/* a key (only its dim 0 set) whose score against q = (q0, 0, ...) is exactly the float `target`, as the
 * engine computes a score: the dot under the lane contract, times scale */
static int find_key(uint32_t target, float scale, float *q0_out, float *k0_out) {
    static const float q0s[] = {1.0f, 1.25f, 1.5f, 1.75f, 3.0f, 5.0f, 7.0f, 0.75f};
    const tr_kernels *kt = tr_kernels_scalar();
    float q[HD] = {0}, key[HD] = {0};
    for (size_t i = 0; i < sizeof q0s / sizeof q0s[0]; i++) {
        q[0] = q0s[i];
        float guess = from_bits(target) / (q0s[i] * scale), up = guess, down = guess;
        for (int step = 0; step < 256; step++) {
            for (int side = 0; side < 2; side++) {
                key[0] = side ? up : down;
                if (bits(kt->dot_f32(q, key, HD) * scale) == target) {
                    *q0_out = q0s[i];
                    *k0_out = key[0];
                    return 0;
                }
            }
            up = nextafterf(up, INFINITY);
            down = nextafterf(down, -INFINITY);
        }
    }
    return -1;
}

/* the keys and values of a case, positions 0..n-1; for EXCEPTION head 0 is set up so that its scores hit
 * the exception table (q0v: the q head 0 must use) */
static void make_data(bufs *B, int n, int kind, float scale, uint64_t *seed, float *q0v) {
    for (int64_t h = 0; h < B->n_head; h++) {
        float *k = B->kh[h], *v = B->vh[h];
        for (size_t i = 0; i < (size_t)n * HD; i++) {
            k[i] = rnd(seed);
            v[i] = rnd(seed);
        }
        if (kind == EQUAL)
            for (int p = 1; p < n; p++) memcpy(k + (size_t)p * HD, k, HD * sizeof(float));
        if (kind == NEGZERO) /* dim 0 of every head, and the whole of the last head, are -0 */
            for (int p = 0; p < n; p++)
                for (int d = 0; d < HD; d++)
                    if (d == 0 || h == B->n_head - 1) v[(size_t)p * HD + d] = -0.0f;
    }
    *q0v = 1.0f;
    if (kind == EXCEPTION) {
        /* key 0 scores 0, the maximum; keys 2, 4, ... score exactly the negative exception arguments; the
         * others score below 0 */
        float *k = B->kh[0];
        int placed = 0;
        for (int i = 0; i < tr_expf_n_exceptions() && 2 * (placed + 1) < n; i++) {
            uint32_t xb, yb;
            tr_expf_exception(i, &xb, &yb);
            float qv, kv;
            if (!(xb & 0x80000000u) || find_key(xb, scale, &qv, &kv) != 0) continue;
            if (placed > 0 && qv != *q0v) continue; /* one q for the head: the targets that share it */
            *q0v = qv;
            placed++;
            memset(k + (size_t)(2 * placed) * HD, 0, HD * sizeof(float));
            k[(size_t)(2 * placed) * HD] = kv;
        }
        memset(k, 0, HD * sizeof(float));
        for (int p = 1; p < n; p++)
            if (p % 2 == 1 || p > 2 * placed) k[(size_t)p * HD] = -fabsf(k[(size_t)p * HD]);
    }
    for (int p = 0; p < n; p++)
        for (int64_t h = 0; h < B->n_head; h++) {
            memcpy(B->krow + ((size_t)p * (size_t)B->n_head + (size_t)h) * HD, B->kh[h] + (size_t)p * HD,
                   HD * sizeof(float));
            memcpy(B->vrow + ((size_t)p * (size_t)B->n_head + (size_t)h) * HD, B->vh[h] + (size_t)p * HD,
                   HD * sizeof(float));
        }
}

/* the reference for n_pos positions under table k, head by head, into B->ref */
static void reference(bufs *B, const tr_kernels *k, int n_pos, float scale) {
    tr_kernels_set_active(k);
    for (int64_t h = 0; h < B->n_head; h++)
        tr_attention_group(B->q + h * HD, HD, B->kh[h], B->vh[h], 1, n_pos, HD, scale, B->scores, n_pos,
                           B->ref + h * HD, HD);
}

static int64_t differ(const bufs *B, tally *t, const char *what, int kind, int n_pos) {
    int64_t bad = 0;
    for (int64_t i = 0; i < B->n_head * HD; i++) {
        if (bits(B->out[i]) == bits(B->ref[i])) continue;
        if (bad == 0 && t->printed < 6) {
            t->printed++;
            fprintf(stderr, "  %s %s n_pos %d head %d dim %d: gpu %08X cpu %08X\n", kind_name[kind], what, n_pos,
                    (int)(i / HD), (int)(i % HD), (unsigned)bits(B->out[i]), (unsigned)bits(B->ref[i]));
        }
        bad++;
    }
    return bad;
}

/* which branches of the softmax the positions of this decode go through, on the CPU's side */
static void count_branches(bufs *B, int n_pos, float scale, tally *t) {
    const tr_kernels *k = tr_kernels_scalar();
    for (int64_t h = 0; h < B->n_head; h++) {
        float m = -INFINITY;
        for (int p = 0; p < n_pos; p++) {
            B->scores[p] = k->dot_f32(B->q + h * HD, B->kh[h] + (size_t)p * HD, HD) * scale;
            if (B->scores[p] > m) m = B->scores[p];
        }
        for (int p = 0; p < n_pos; p++) {
            float x = B->scores[p] - m, e = tr_expf(x);
            t->table += tr_expf_path(x) == TR_EXPF_TABLE;
            t->underflow += x < -104.0f;
            t->subnormal += e != 0.0f && e < 1.17549435e-38f;
        }
    }
}

/* one case: positions 0..n-1 of `layer`, the last N_DECODE of them decoded */
static void run_case(tr_gpu_attn *a, bufs *B, int64_t layer, int n, int kind, uint64_t *seed, const tr_kernels *active,
                     tally *t) {
    const float scale = 1.0f / sqrtf((float)HD);
    size_t row = (size_t)B->n_head * HD;
    float q0v;
    make_data(B, n, kind, scale, seed, &q0v);
    int d0 = n > N_DECODE ? n - N_DECODE : 0;          /* the first decoded position */
    int bulk = d0 > 4 ? d0 - 4 : 0;                    /* rows [0, bulk) in pieces, [bulk, d0) one by one */
    static const int piece[] = {1, 37, 512, 200, 31, 512, 33, 512};
    int p = 0;
    for (int i = 0; p < bulk; i++) {
        int m = piece[i % (int)(sizeof piece / sizeof piece[0])];
        if (m > bulk - p) m = bulk - p;
        TR_CHECK(tr_gpu_attn_write(a, layer, p, m, B->krow + (size_t)p * row, B->vrow + (size_t)p * row) == 0);
        p += m;
    }
    for (; p < d0; p++) TR_CHECK(tr_gpu_attn_write(a, layer, p, 1, B->krow + (size_t)p * row, B->vrow + (size_t)p * row) == 0);
    int np = B->n_ctx - d0 < POISON ? (int)(B->n_ctx - d0) : POISON;
    TR_CHECK(tr_gpu_attn_write(a, layer, d0, np, B->nan_rows, B->nan_rows) == 0);
    t->poisoned += np;
    for (int pos = d0; pos < n; pos++) {
        for (size_t i = 0; i < row; i++) B->q[i] = (kind == WIDE ? 100.0f : 1.0f) * rnd(seed);
        if (kind == EXCEPTION) {
            memset(B->q, 0, HD * sizeof(float));
            B->q[0] = q0v;
        }
        TR_CHECK(tr_gpu_attn_decode(a, layer, pos, B->q, B->krow + (size_t)pos * row, B->vrow + (size_t)pos * row, scale,
                                    B->out) == 0);
        t->decodes++;
        reference(B, tr_kernels_scalar(), pos + 1, scale);
        t->bad_scalar += differ(B, t, "vs scalar", kind, pos + 1);
        reference(B, active, pos + 1, scale);
        t->bad_active += differ(B, t, "vs active tier", kind, pos + 1);
        t->compared += B->n_head;
        count_branches(B, pos + 1, scale, t);
    }
}

static void run_sizes(tr_gpu_attn *a, bufs *B, int64_t n_layers, const int *sizes, int n_sizes, const int *kinds,
                      int n_kinds, uint64_t *seed, const tr_kernels *active, tally *t) {
    int c = 0;
    for (int ki = 0; ki < n_kinds; ki++)
        for (int si = 0; si < n_sizes; si++) {
            if (kinds[ki] == EXCEPTION && sizes[si] < 16) continue;
            run_case(a, B, c % n_layers, sizes[si], kinds[ki], seed, active, t);
            c++;
        }
}

int main(void) {
    char err[512];
    double t0 = tr_time_sec();
    tr_gpu *g = tr_gpu_open(err, sizeof err);
    double t_open = tr_time_sec() - t0;
    if (g == NULL) {
        printf("test_gpu_attn: skipped (%s)\n", err);
        return 0;
    }
    tr_kernels_init();
    const tr_kernels *active = tr_kernels_get();
    tally t;
    memset(&t, 0, sizeof t);
    uint64_t seed = 0x5EED5EEDu;

    /* the refusals */
    TR_CHECK(tr_gpu_attn_create(g, 1, 16, 64, 100, err, sizeof err) == NULL && strstr(err, "head_dim") != NULL);
    TR_CHECK(tr_gpu_attn_bytes(2, 16, HD, 4000) ==
             2ull * 2 * 16 * 4096 * HD * 4 + 2ull * 16 * 4096 * 4 + 16ull * 32 * 4 + 256);

    /* 2 layers of 16 heads, context 4096: the model's shape */
    t0 = tr_time_sec();
    tr_gpu_attn *a = tr_gpu_attn_create(g, 2, 16, HD, 4096, err, sizeof err);
    double t_create = tr_time_sec() - t0;
    if (a == NULL) fprintf(stderr, "tr_gpu_attn_create: %s\n", err);
    TR_CHECK(a != NULL);
    bufs B;
    TR_CHECK(bufs_init(&B, 16, 4096) == 0);
    double t_first = 0.0, t_write = 0.0;
    if (a != NULL && B.scores != NULL) {
        const float scale = 1.0f / sqrtf((float)HD);
        size_t row = 16 * HD;
        float q0v;
        /* what a cold call and a pass of 512 rows cost, once, before the cases */
        make_data(&B, 600, RANDOM, scale, &seed, &q0v);
        t0 = tr_time_sec();
        TR_CHECK(tr_gpu_attn_write(a, 0, 0, 512, B.krow, B.vrow) == 0);
        t_write = tr_time_sec() - t0;
        t0 = tr_time_sec();
        TR_CHECK(tr_gpu_attn_decode(a, 0, 512, B.q, B.krow + 512 * row, B.vrow + 512 * row, scale, B.out) == 0);
        t_first = tr_time_sec() - t0;
        TR_CHECK(tr_gpu_attn_decode(a, 0, 4096, B.q, B.krow, B.vrow, scale, B.out) == -1);
        TR_CHECK(tr_gpu_attn_decode(a, 2, 0, B.q, B.krow, B.vrow, scale, B.out) == -1);
        TR_CHECK(tr_gpu_attn_write(a, 0, 4095, 2, B.krow, B.vrow) == -1);
        TR_CHECK(tr_gpu_attn_write(a, 1, 0, 0, B.krow, B.vrow) == 0);
        /* the prompt's last pass wakes the GPU (tr_gpu_attn_warm): the cases below start with that
         * warp still napping, and must give the same bits */
        TR_CHECK(tr_gpu_attn_warm(a, TR_GPU_WARM_MAX_NS) == 0);
        TR_CHECK(tr_gpu_attn_warm(NULL, 1) == -1);

        static const int sizes[] = {1,   2,   3,   15,  16,  17,  31,   32,   33,   63,   64,   65,   127,  128,
                                    129, 191, 192, 193, 255, 256, 257,  511,  512,  513,  1000, 1025, 2047, 2048,
                                    2049, 3999, 4000, 4094, 4095, 4096};
        static const int kinds[] = {RANDOM, WIDE, NEGZERO, EQUAL, EXCEPTION};
        run_sizes(a, &B, 2, sizes, (int)(sizeof sizes / sizeof sizes[0]), kinds, N_KIND, &seed, active, &t);
    }
    tr_gpu_attn_free(a);
    bufs_free(&B);

    /* 3 heads, context 300: another number of heads, a cap (384) past the context */
    tr_gpu_attn *a3 = tr_gpu_attn_create(g, 1, 3, HD, 300, err, sizeof err);
    TR_CHECK(a3 != NULL);
    bufs B3;
    TR_CHECK(bufs_init(&B3, 3, 300) == 0);
    if (a3 != NULL && B3.scores != NULL) {
        static const int sizes[] = {1, 5, 127, 128, 129, 200, 299, 300};
        static const int kinds[] = {RANDOM, EXCEPTION, NEGZERO};
        run_sizes(a3, &B3, 1, sizes, 8, kinds, 3, &seed, active, &t);
    }
    /* a prompt's last pass asks for a warm-up after every layer, and no decode may follow (logits -b,
     * a session freed after its prompt): the warps must not queue up. Queued, 16 of them napped
     * 200 ms each in a row and the free below waited 3.2 s (docs/LESSONS.md #159). */
    double t_free = 0.0;
    if (a3 != NULL) {
        for (int i = 0; i < 16; i++) TR_CHECK(tr_gpu_attn_warm(a3, TR_GPU_WARM_MAX_NS) == 0);
        double t_f0 = tr_time_sec();
        tr_gpu_attn_free(a3);
        t_free = tr_time_sec() - t_f0;
        TR_CHECK(t_free < 0.5);
        a3 = NULL;
    }
    tr_gpu_attn_free(a3);
    bufs_free(&B3);

    /* past 4096 positions: the second chunk of the sum, more than 32 block maxima */
    tr_gpu_attn *a2 = tr_gpu_attn_create(g, 1, 2, HD, 4300, err, sizeof err);
    TR_CHECK(a2 != NULL);
    bufs B2;
    TR_CHECK(bufs_init(&B2, 2, 4300) == 0);
    if (a2 != NULL && B2.scores != NULL) {
        static const int sizes[] = {4097, 4099, 4225, 4300};
        static const int kinds[] = {RANDOM, WIDE};
        run_sizes(a2, &B2, 1, sizes, 4, kinds, 2, &seed, active, &t);
    }
    tr_gpu_attn_free(a2);
    bufs_free(&B2);
    tr_gpu_close(g);
    tr_kernels_set_active(NULL);

    TR_CHECK(t.decodes > 0);
    TR_CHECK(t.poisoned > 0);
    TR_CHECK(t.table > 0);
    TR_CHECK(t.underflow > 0);
    TR_CHECK(t.subnormal > 0);
    TR_CHECK_EQ_INT(t.bad_scalar, 0);
    TR_CHECK_EQ_INT(t.bad_active, 0);
    printf("test_gpu_attn: %lld decodes (%lld head outputs) as bits against scalar and %s: %lld and %lld floats "
           "differ; exercised %lld exception-table arguments, %lld underflows, %lld subnormal e, %lld poisoned rows\n",
           (long long)t.decodes, (long long)t.compared, active->tier, (long long)t.bad_scalar, (long long)t.bad_active,
           (long long)t.table, (long long)t.underflow, (long long)t.subnormal, (long long)t.poisoned);
    printf("test_gpu_attn: open %.0f ms, create (2 x 16 x 4096) %.1f ms, a write of 512 rows %.2f ms, a first "
           "decode %.2f ms, a free after 16 warm-ups %.0f ms\n",
           1e3 * t_open, 1e3 * t_create, 1e3 * t_write, 1e3 * t_first, 1e3 * t_free);
    TR_TEST_EXIT();
}
