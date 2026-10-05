/* test_head_bound.c — the output head's argmax by a bound (src/kernels/head_bound.c; kernels.h §The output head by a
 * bound; docs/MEASUREMENTS.md §The head's argmax by a bound).
 *
 * Branches, each counted (a count of zero fails the test, LESSONS #43):
 *   planes     tr_hb_planes then every tier's hb_rebuild gives the original row byte for byte;
 *   tiers      every tier's hb_bounds gives the scalar definition's bits (memcmp of the bounds);
 *   bound      every bound at or above the engine's own value of its row (tr_matmul_s on the original rows), on
 *              random heads, adversarial ones (negative scales, f16's largest, subnormal scales, saturated and all-zero
 *              codes, Q4_K's negative d and dmin, sc and m 0 and 63) and tight rows: h on the digits' grid (no rounding
 *              of h) and the low bits at their worst against h, where only the float sums' margin keeps the bound up
 *              (their least relative gap is printed: it must stay below 2^-8, or the rows are not tight);
 *   pruned     tr_hb_argmax computed fewer rows than the head (a row aligned with h stands out) and gave
 *              tr_argmax_f32's row over tr_matmul_s's logits, on pools of 1, 3 and 8, over every row and a prefix;
 *   tie        that row's copy at a lower row: the same bound and score, the lower row wins;
 *   exact th   rows of zeros (bound and score 0) and three decoys (a positive bound, a negative score): the threshold
 *              is a zero row's 0 and every row of zeros reaches it with its bound exactly (>=, not >);
 *   top        three large decoys above the aligned row's bound: the threshold from TR_HB_TOP rows reaches the aligned
 *              row's score and a quarter of the rows at most is computed (the best bound's row alone: every row);
 *   full       h with a NaN or an infinity, a row with a NaN or an infinite scale: every row computed, the scan's rule;
 *   logits     tr_hb_logits for 1 to 5 input rows against tr_matmul_s, bit for bit. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

enum { R = 300, NMAX = 2048, TMAX = 5, ALIGNED = 250, COPY = 77 };

static long n_planes, n_tiers, n_bound, n_tight, n_pruned, n_tie, n_full, n_logits, n_argmax, n_zero_th, n_top;
static double tight_gap = 1.0; /* the least (bound - value) / sum|w h| over the tight rows */

static uint32_t rnd(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}
static float rnd_normal(uint32_t *s) {
    float v = 0;
    for (int i = 0; i < 12; i++) v += (float)(rnd(s) & 0xFFFF) / 65536.0f;
    return v - 6.0f;
}

/* f16 bits of a normal scale near 2^(e - 15) */
static uint16_t half_bits(uint32_t *s, unsigned e, int neg) {
    return (uint16_t)((e << 10) | (rnd(s) & 1023u) | (neg ? 0x8000u : 0u));
}

/* h: 0 normal with outliers, 1 of one sign, 2 on the digits' grid (h = 2^-10 c, the largest |c| 32639 a block, so no
 * rounding of h) with a zero block, 3 blocks below 2^-64 and one near 2^60 */
static void make_h(float *x, int64_t n, int mode, uint32_t *s) {
    for (int64_t b = 0; b < n / 32; b++) {
        float *xb = x + 32 * b;
        for (int k = 0; k < 32; k++) {
            float v = rnd_normal(s) * ((rnd(s) & 63u) == 0 ? 8.0f : 1.0f);
            if (mode == 1) v = fabsf(v) + 0.01f;
            if (mode == 2) v = ldexpf((float)((int)(rnd(s) % 65279u) - 32639), -10);
            xb[k] = v;
        }
        if (mode == 2) xb[rnd(s) % 32] = ldexpf((rnd(s) & 1) ? 32639.0f : -32639.0f, -10);
        if (mode == 2 && b == 3) memset(xb, 0, 32 * sizeof *xb);
        /* a block 2^-30 below its super-block's others, still on its own grid: Q4_K's engine rounds it (2^-26) */
        if (mode == 2 && b % 8 == 5)
            for (int k = 0; k < 32; k++) xb[k] = ldexpf(xb[k], -30);
        if (mode == 3 && b % 5 == 1)
            for (int k = 0; k < 32; k++) xb[k] = ldexpf(xb[k], -80);
        if (mode == 3 && b == 2) xb[7] = 0x1p60f;
    }
}

static void make_q8_0(unsigned char *row, int64_t n, int adv, uint32_t *s) {
    for (int64_t b = 0; b < n / 32; b++) {
        unsigned char *blk = row + 34 * b;
        uint16_t d = half_bits(s, 6, (rnd(s) & 7u) == 0);
        const int kind = adv ? (int)(rnd(s) % 7) : 6;
        if (kind == 0) d = 0x7BFF;                      /* f16's largest */
        if (kind == 1) d = (uint16_t)(rnd(s) & 0x83FF); /* subnormal, either sign */
        memcpy(blk, &d, 2);
        for (int i = 0; i < 32; i++) {
            int q = (int)lrintf(rnd_normal(s) * 40.0f);
            if (kind == 2) q = -128;
            if (kind == 3) q = 127;
            if (kind == 4) q = 0;
            if (kind == 5) q = (i & 1) ? 127 : -128;
            q = q < -128 ? -128 : q > 127 ? 127 : q;
            blk[2 + i] = (unsigned char)(int8_t)q;
        }
    }
}

static void make_q4_k(unsigned char *row, int64_t n, int adv, uint32_t *s) {
    for (int64_t sb = 0; sb < n / 256; sb++) {
        unsigned char *blk = row + 144 * sb;
        const int kind = adv ? (int)(rnd(s) % 6) : 5;
        uint16_t d = half_bits(s, 5, kind == 0), dm = half_bits(s, 4, kind == 0 || kind == 1);
        if (kind == 2) d = 0x7BFF;
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dm, 2);
        for (int i = 4; i < 144; i++) blk[i] = (unsigned char)rnd(s);
        if (kind == 3) memset(blk + 4, 0xFF, 12);            /* sc and m 63 */
        if (kind == 4) memset(blk + 4, 0x00, 12);            /* sc and m 0 */
        if (kind == 1) memset(blk + 16, (rnd(s) & 1) ? 0xFF : 0x00, 128); /* codes all 15 or all 0 */
    }
}

/* a row tight against h on the grid: the top bits random, the low bits at their worst (Q8_0: the low nibble 15
 * where h > 0, else 0; Q4_K: bit 0 so), positive scales. Q4_K's variant 1 and 2: only sub-block 5 weighs, by its
 * sc (1) or its m (2): the block make_h puts 2^-30 below the others, whose rounding by the engine only the prep's
 * err_el terms cover */
static void make_tight(tr_type t, unsigned char *row, int64_t n, const float *x, uint32_t *s, int variant) {
    if (t == TR_TYPE_Q8_0) {
        for (int64_t b = 0; b < n / 32; b++) {
            unsigned char *blk = row + 34 * b;
            const uint16_t d = half_bits(s, 6, 0);
            memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) {
                const int u = (int)(rnd(s) % 16u), lo = x[32 * b + i] > 0 ? 15 : 0;
                blk[2 + i] = (unsigned char)(int8_t)(16 * (u - 8) + lo);
            }
        }
        return;
    }
    for (int64_t sb = 0; sb < n / 256; sb++) {
        unsigned char *blk = row + 144 * sb;
        const uint16_t d = half_bits(s, 5, 0), dm = half_bits(s, 4, 0);
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dm, 2);
        for (int i = 4; i < 16; i++) blk[i] = (unsigned char)rnd(s);
        if (variant != 0) { /* sc_5 or m_5 63 (tr_q4_k_sc_m), every other 0 */
            memset(blk + 4, 0, 12);
            blk[4 + (variant == 1 ? 1 : 5)] = 0xC0;
            blk[4 + 9] = variant == 1 ? 0x0F : 0xF0;
        }
        for (int k = 0; k < 256; k++) {
            const unsigned q = 2u * (rnd(s) % 8u) + (x[256 * sb + k] > 0 ? 1u : 0u);
            unsigned char *byte = blk + 16 + 32 * (k / 64) + k % 32;
            *byte = (unsigned char)((k & 32) ? ((*byte & 0x0F) | (q << 4)) : ((*byte & 0xF0) | q));
        }
    }
}

/* a decoy against h: the top bits add nothing (Q8_0 u = 8, Q4_K q >> 1 = 0, dmin 0, sc 63) and the low bit is on
 * where h < 0, so the bound is large and positive (d 15 delta sum of h's positive part, at the low bits' worst) and the
 * score negative; d16 its scale's f16 bits. d16 0: a row of zeros, bound and score exactly 0 */
static void make_decoy(tr_type t, unsigned char *row, int64_t n, const float *x, uint16_t d16) {
    if (t == TR_TYPE_Q8_0) {
        for (int64_t b = 0; b < n / 32; b++) {
            unsigned char *blk = row + 34 * b;
            memcpy(blk, &d16, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (unsigned char)(int8_t)(x[32 * b + i] < 0 ? 15 : 0);
        }
        return;
    }
    for (int64_t sb = 0; sb < n / 256; sb++) {
        unsigned char *blk = row + 144 * sb;
        const uint16_t dm = 0;
        memcpy(blk, &d16, 2);
        memcpy(blk + 2, &dm, 2);
        const unsigned char sc12[12] = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0x0F, 0x0F, 0x0F, 0x0F};
        memcpy(blk + 4, sc12, 12);
        for (int c = 0; c < 4; c++)
            for (int i = 0; i < 32; i++)
                blk[16 + 32 * c + i] = (unsigned char)((x[256 * sb + 64 * c + i] < 0 ? 1 : 0) |
                                                       ((x[256 * sb + 64 * c + 32 + i] < 0 ? 1 : 0) << 4));
    }
}

/* a row aligned with h: its score stands out, so the bound drops rows */
static void make_aligned(tr_type t, unsigned char *row, int64_t n, const float *x) {
    if (t == TR_TYPE_Q8_0) {
        for (int64_t b = 0; b < n / 32; b++) {
            unsigned char *blk = row + 34 * b;
            const uint16_t d = 0x1800; /* 2^-9 */
            memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (unsigned char)(int8_t)(x[32 * b + i] > 0 ? 100 : -100);
        }
        return;
    }
    for (int64_t sb = 0; sb < n / 256; sb++) {
        unsigned char *blk = row + 144 * sb;
        const uint16_t d = 0x1400, dm = 0x1000;
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dm, 2);
        const unsigned char sc12[12] = {63, 63, 63, 63, 0, 0, 0, 0, 15, 15, 15, 15};
        memcpy(blk + 4, sc12, 12);
        for (int c = 0; c < 4; c++)
            for (int i = 0; i < 32; i++)
                blk[16 + 32 * c + i] = (unsigned char)((x[256 * sb + 64 * c + i] > 0 ? 15 : 0) |
                                                       ((x[256 * sb + 64 * c + 32 + i] > 0 ? 15 : 0) << 4));
    }
}

/* the magnitude of row r's terms against x: sum |w_k x_k| from the dequantized row */
static double row_mag(tr_type t, const unsigned char *row, int64_t n, const float *x) {
    static float w[NMAX];
    tr_kernels_tier("scalar")->dequant_row[t](row, w, n);
    double m = 0;
    for (int64_t k = 0; k < n; k++) m += fabs((double)w[k] * (double)x[k]);
    return m;
}

/* a and b bit for bit, two NaNs equal whatever their payload (kernels.h: a NaN's payload may differ between roads,
 * compilers and flags; clang -O2 and gcc -O1 gave other ones, LESSONS #346) */
static int same_bits(const float *a, const float *b, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], sizeof a[i]) != 0 && !(a[i] != a[i] && b[i] != b[i])) {
            fprintf(stderr, "  element %lld: %.9g against %.9g\n", (long long)i, (double)a[i], (double)b[i]);
            return 0;
        }
    return 1;
}

static const tr_kernels *g_tiers[3];
static int g_n_tiers;

static void run_case(tr_type t, int64_t n, int head_mode, int h_mode, tr_pool *const *pools, uint32_t *s) {
    const size_t rb = tr_row_bytes(t, n), hb = tr_hb_hi_bytes(t, n), lb = tr_hb_lo_bytes(t, n);
    static unsigned char orig[R * (NMAX / 32) * 34], hi[R * (NMAX / 32) * 18], lo[R * (NMAX / 32) * 16];
    static unsigned char tmp[(NMAX / 32) * 34];
    static float x[TMAX * NMAX], y[TMAX * R], y2[TMAX * R], U[R], U0[R];
    TR_CHECK(hb + lb == rb && rb <= sizeof tmp);
    for (int t5 = 0; t5 < TMAX; t5++) make_h(x + t5 * n, n, t5 == 0 ? h_mode : (int)(rnd(s) % 3), s);
    if (h_mode == 4) x[5] = (head_mode & 1) ? NAN : INFINITY; /* the full path from h */
    for (int r = 0; r < R; r++) {
        unsigned char *row = orig + (size_t)r * rb;
        if (head_mode == 4) make_decoy(t, row, n, x, 0);
        else if (head_mode == 2 && r % 3 == 0) make_tight(t, row, n, x, s, h_mode == 2 ? (int)(r / 3 % 3) : 0);
        else if (t == TR_TYPE_Q8_0) make_q8_0(row, n, head_mode == 1, s);
        else make_q4_k(row, n, head_mode == 1, s);
    }
    /* head 4: rows of zeros and three small decoys, the threshold then a zero row's 0, which every row of zeros
     * reaches with its bound exactly; head 5: three large decoys above the aligned row's bound, so the threshold needs
     * the TR_HB_TOP rows, not the best bound's alone */
    for (int r = 100; r <= 200 && head_mode == 4; r += 50) make_decoy(t, orig + (size_t)r * rb, n, x, 0x1800);
    for (int r = 30; r <= 32 && head_mode == 5; r++) make_decoy(t, orig + (size_t)r * rb, n, x, 0x3000);
    if (h_mode != 4 && head_mode != 4) {
        make_aligned(t, orig + (size_t)ALIGNED * rb, n, x);
        memcpy(orig + (size_t)COPY * rb, orig + (size_t)ALIGNED * rb, rb);
    }
    if (head_mode == 3) { /* the full path from a row: a NaN or an infinite scale, at row 0 (the scan's own) or 11 */
        const uint16_t bad = (rnd(s) & 1) ? 0x7E00 : 0x7C00;
        memcpy(orig + ((rnd(s) & 1) ? 0 : 11) * rb, &bad, 2);
    }
    for (int r = 0; r < R; r++) tr_hb_planes(t, orig + (size_t)r * rb, n, hi + (size_t)r * hb, lo + (size_t)r * lb);
    const tr_hb_head w = {t, R, n, hi, lo};

    /* the engine's own values: tr_matmul_s on the original rows */
    tr_pm_scratch pm;
    TR_CHECK(tr_pm_scratch_init(&pm, 8, 1, TMAX, n, TMAX * n) == 0);
    const tr_mat m = {t, R, n, orig};
    tr_matmul_s(pools[2], &m, x, TMAX, y, &pm);

    static unsigned char prep_buf[2 * NMAX + 5 * 4 * (NMAX / 32)] __attribute__((aligned(64)));
    const tr_hb_prep p = tr_hb_prep_view(prep_buf, n);
    const int prep_ok = tr_hb_prep_build(t, x, n, &p) == 0;
    TR_CHECK(prep_ok == (h_mode != 4));

    for (int ti = 0; ti < g_n_tiers; ti++) {
        const tr_kernels *k = g_tiers[ti];
        for (int r = 0; r < R; r += 7) {
            k->hb_rebuild[t](hi + (size_t)r * hb, lo + (size_t)r * lb, n, tmp);
            TR_CHECK(memcmp(tmp, orig + (size_t)r * rb, rb) == 0);
            n_planes++;
        }
        if (!prep_ok) continue;
        k->hb_bounds[t](hi, R, &p, n, ti == 0 ? U0 : U);
        if (ti > 0) {
            const int same = same_bits(U, U0, R);
            TR_CHECK(same);
            if (!same) fprintf(stderr, "  tier %s: bounds differ from scalar (type %d, n %lld)\n", k->tier, (int)t, (long long)n);
            n_tiers++;
        }
    }
    if (prep_ok && head_mode != 3)
        for (int r = 0; r < R; r++) {
            const int ok = (double)U0[r] >= (double)y[r];
            if (!ok)
                fprintf(stderr, "  bound below the row: type %d n %lld head %d h %d row %d: %.9g < %.9g\n", (int)t,
                        (long long)n, head_mode, h_mode, r, (double)U0[r], (double)y[r]);
            TR_CHECK(ok);
            n_bound++;
            if (head_mode == 2 && h_mode == 2 && r % 3 == 0 && r != COPY) {
                const double gap = ((double)U0[r] - (double)y[r]) / row_mag(t, orig + (size_t)r * rb, n, x);
                if (gap < tight_gap) tight_gap = gap;
                n_tight++;
            }
        }

    /* the argmax, every tier and pool, over every row and a prefix */
    for (int ti = 0; ti < g_n_tiers; ti++) {
        tr_kernels_set_active(g_tiers[ti]);
        for (int pi = 0; pi < 3; pi++) {
            tr_hb_scratch hs;
            const int nw = pools[pi] != NULL ? tr_pool_size(pools[pi]) : 1;
            TR_CHECK(tr_hb_scratch_init(&hs, &w, nw, TMAX) == 0);
            for (int pre = 0; pre < 2; pre++) {
                const int64_t nn = pre ? R - 7 : R;
                tr_hb_result res;
                const int32_t got = tr_hb_argmax(pools[pi], &w, x, nn, &hs, y2, &res);
                const int32_t want = tr_argmax_f32(NULL, y, nn);
                if (got != want)
                    fprintf(stderr, "  argmax: tier %s pool %d type %d head %d h %d: %d, the scan %d\n", g_tiers[ti]->tier,
                            nw, (int)t, head_mode, h_mode, (int)got, (int)want);
                TR_CHECK(got == want);
                n_argmax++;
                if (res.full) n_full++;
                else if (res.rows < nn) n_pruned++;
                if (!res.full && want == COPY) n_tie++;
                if (!res.full && head_mode == 4 && got == want) n_zero_th++;
                if (!res.full && head_mode == 5 && res.rows <= R / 4) n_top++;
            }
            /* the logits from the planes, 1 to TMAX input rows */
            for (int T = 1; T <= TMAX; T++) {
                tr_hb_logits(pools[pi], &w, x, T, y2, &hs);
                tr_matmul_s(NULL, &m, x, T, y, &pm);
                TR_CHECK(same_bits(y2, y, (int64_t)T * R));
                n_logits++;
            }
            tr_matmul_s(pools[2], &m, x, TMAX, y, &pm); /* y back to every input row */
            tr_hb_scratch_free(&hs);
        }
        tr_kernels_set_active(NULL);
    }
    tr_pm_scratch_free(&pm);
}

int main(void) {
    tr_kernels_init();
    const char *names[3] = {"scalar", "avx2", "avx512"};
    for (int i = 0; i < 3; i++) {
        const tr_kernels *k = tr_kernels_tier(names[i]);
        if (k != NULL && k->hb_bounds[TR_TYPE_Q8_0] != NULL) g_tiers[g_n_tiers++] = k;
    }
    tr_pool *pools[3] = {NULL, tr_pool_create(3), tr_pool_create(8)};
    TR_CHECK(pools[1] != NULL && pools[2] != NULL);
    TR_CHECK(!tr_hb_supports(TR_TYPE_Q8_0, 2048 + 32) && !tr_hb_supports(TR_TYPE_Q6_K, 2048) &&
             tr_hb_supports(TR_TYPE_Q4_K, 256));
    uint32_t seed = 73;
    const tr_type types[2] = {TR_TYPE_Q8_0, TR_TYPE_Q4_K};
    const int64_t ns[2] = {256, 2048};
    for (int ti = 0; ti < 2; ti++)
        for (int ni = 0; ni < 2; ni++)
            for (int hm = 0; hm < 6; hm++)
                for (int xm = 0; xm < 5; xm++) run_case(types[ti], ns[ni], hm, xm, pools, &seed);
    printf("test_head_bound: tiers %d; planes %ld, tiers %ld, bounds %ld (tight %ld, least gap %.3g of sum|w h|), "
           "argmax %ld (pruned %ld, tie %ld, full %ld, threshold reached exactly %ld, past the decoys %ld), logits %ld\n",
           g_n_tiers, n_planes, n_tiers, n_bound, n_tight, tight_gap, n_argmax, n_pruned, n_tie, n_full, n_zero_th, n_top,
           n_logits);
    TR_CHECK(n_planes > 0 && n_bound > 0 && n_tight > 0 && n_pruned > 0 && n_tie > 0 && n_full > 0 && n_logits > 0);
    TR_CHECK(n_zero_th > 0 && n_top > 0);
    TR_CHECK(g_n_tiers < 2 || n_tiers > 0);
    TR_CHECK(tight_gap < 0x1p-8); /* the tight rows are tight: the margin is what keeps them above */
    tr_pool_destroy(pools[1]);
    tr_pool_destroy(pools[2]);
    TR_TEST_EXIT();
}
