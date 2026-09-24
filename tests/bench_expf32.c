/* bench_expf32.c — expf32: exp(x) correctly rounded to float with float32 and int32 operations
 * only (no double), for a GPU; the proof that it is tr_expf bit for bit on every float; its cost.
 *
 * Why: tr_expf (src/kernels/expf.c) computes in double, and a consumer GPU runs FP64 at 1/64 of
 * FP32, while the prefill of a 4000-token prompt takes about 2.5 G exponentials (softmax, SiLU).
 * A correctly rounded exp is one function: any correctly rounded expf is tr_expf, bit for bit.
 *
 * The algorithm. Every operation is an IEEE binary32 or a 32-bit integer one, rounded to nearest,
 * subnormals kept (no FTZ), nothing fused unless written as fma:
 *
 *   k = round(x 256/ln2), j = k mod 256, e = floor(k/256), r = x - k ln2/256, |r| <= 0.51 ln2/256
 *   exp(x) = 2^e y,  y = 2^(j/256) exp(r) in [0.998, 2.003]
 *
 *   1. r in two floats: rh = x - k LH is exact (one fma: the difference fits in 24 bits; without
 *      fma, LH in three pieces of 8 bits makes every product and difference exact), rl = -k LL,
 *      |rl| < 2^-21.7 (LH + LL = ln2/256 within 2^-61.3).
 *   2. q = exp(r) - 1 - r ~ r^2 (c2 + c3 r), minimax on |r| <= 0.51 ln2/256: error 2^-44.6.
 *   3. 2^(j/256) = th + tl, a table of 256 float pairs (2 KB, error 2^-49.2). th rh = ph + pl
 *      exactly (fma, or Dekker's product), th + ph = yh + e exactly (Fast2Sum), and the rest,
 *      th (rl + q) + tl (1 + r) + pl + e, in floats: yl.
 *   4. |yh + yl - y| < 2^-39.9 (the budget below). If yh + (yl - D) and yh + (yl + D), D = 2^-39,
 *      round to the same float u, so does y, and exp(x) is u 2^e: an integer add on the exponent
 *      field. Below 2^-126 the grid is 2^-149, that is g = 2^(-149-e) in the units of y: with
 *      m = 2^23 g the float sum m + y rounds on that grid (Fast2Sum again, margin D + 2^-22 g).
 *   5. What step 4 does not settle goes to the slow path: 2D over the float spacing of y, about 1
 *      argument in 33 000 away from 0 (21 125 of the 2 239 853 081 floats of [-104, 88.72]). Integers,
 *      64-bit fixed point on 32-bit limbs. x 2^80 exact, k16 = floor(x 16/ln2) by an exact test,
 *      r = x - k16 ln2/16 in [0, ln2/16) with ln2/16 to 2^-80, Taylor of degree 9 (error 2^-67),
 *      2^(i/16) from a table of 16; error under 3.4 units of 2^-62, rounding test with 8 units.
 *      Tables: 2 KB + 220 bytes. An argument the slow path cannot settle returns a NaN: the
 *      exhaustive check proves no float gets there, with or without the fast path in front.
 *
 * The budget of step 4, worst case, relative to y, fma variant (units of 2^-46): q evaluated at
 * the rounded r 7.7, q's two roundings 7.7, rl + q rounded 5.3, yl rounded 4.0, tl q left out 3.9,
 * polynomial 2.6, reduction (rl rounded, LL short) 2.3, tl rl + pl + e sums 1.0, table 0.25: 34.8,
 * 2^-40.9 relative, 2^-39.9 on y < 2.003. Without fma two more roundings: 2^-39.5. D = 2^-39 covers
 * both plus the rounding of yl +- D (2^-42.4); --error measures the real maximum.
 *
 * The proof is exhaustive, not that budget: every one of the 2^32 floats, both variants, compared
 * with tr_expf (itself proven by bench_expf --check against the correctly rounded value).
 *
 *   bench_expf32 [--threads T] [--slow-all] [--error] [--quick] [--no-timing]
 *
 *   (none)       every constant against its definition (long double and integers: what the
 *                exhaustive check cannot see, LESSONS #83), timing, then every float through
 *                expf32 (no fma, and fma when the CPU has it): wrong results, how each path is
 *                taken; exit 1 if a constant or a result is wrong, a path is never taken, or an
 *                argument is left unproven
 *   --slow-all   also the slow path alone on every float of [-104, 88.72]: it is a whole
 *                correctly rounded expf by itself
 *   --error      also the largest |yh + yl - y| of both variants, against D, and whether rh is
 *                exact everywhere (the library's double exp as reference, 2^-52)
 *   --quick      1/64 of the floats: a smoke test, not a proof
 *   --threads T  default 4, never more than 8 nor more than the physical cores
 *   --no-timing  skip the ns per call (one thread, a few seconds)
 *   --blocks A B only the blocks [A, B) of the 65536 blocks of 2^16 floats: slices of a busy
 *                machine; their "tally" lines add up to the whole run
 *
 * Mutations (tests seen red): -DEXPF32_MUTATE=n, 1 th[77] one ulp up, 2 D = 2^-45, 3 tl[77] one
 * ulp up (every result stays correctly rounded: only the constants check sees it), 4 the 1/6!
 * term of the slow path dropped, 5 the subnormal grid one binade off, 6 pl dropped (th rh no
 * longer exact), 7 the SIMD tiers skip the slow path (the unsettled lanes keep the fast guess).
 *
 * The fma variant also runs as SIMD tiers (AVX-512, AVX2: question 58), checked on every float like
 * the scalar ones and timed per element. */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/base/cpu.h"
#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/kernels.h"
#include "test.h"

_Static_assert(FLT_EVAL_METHOD == 0, "expf32 needs every float operation rounded to float");
_Static_assert((-1 >> 1) == -1, "expf32 needs an arithmetic right shift of signed integers");

#ifndef EXPF32_MUTATE
#define EXPF32_MUTATE 0
#endif

/* ---- generated by gen_consts.py (mpmath 1.3.0, 300 bits) ---- */
#define EXPF32_INVL 0x1.7154760000000p+8f /* 256/ln2 */
#define EXPF32_LH 0x1.62e4300000000p-9f /* ln2/256 rounded to float */
#define EXPF32_NLL 0x1.05c6100000000p-37f /* -(ln2/256 - LH) rounded to float */
#define EXPF32_LA 0x1.6200000000000p-9f /* LH = LA + LB + LC, 8 significant bits each */
#define EXPF32_LB 0x1.c800000000000p-18f
#define EXPF32_LC 0x1.8000000000000p-28f
#define EXPF32_C2 0x1.0000020000000p-1f /* 1/2 + 2^-24: minimax with C3 */
#define EXPF32_C3 0x1.5555560000000p-3f /* 1/6 */
#define EXPF32_X_NORM -0x1.5d589e0000000p+6f /* smallest float >= -126 ln2 */
static const float expf32_tab[256][2] = { /* 2^(j/256) = hi + lo */
    {0x1.0000000000000p+0f, 0.0f}, {0x1.00b1b00000000p+0f, -0x1.6950d00000000p-26f},
    {0x1.0163da0000000p+0f, 0x1.3f66660000000p-25f}, {0x1.0216820000000p+0f, -0x1.789fb00000000p-25f},
    {0x1.02c9a40000000p+0f, -0x1.887fa00000000p-28f}, {0x1.037d420000000p+0f, 0x1.c2377a0000000p-25f},
    {0x1.04315e0000000p+0f, 0x1.0dcff00000000p-25f}, {0x1.04e5f80000000p+0f, -0x1.a1356a0000000p-25f},
    {0x1.059b0e0000000p+0f, -0x1.9d4f520000000p-25f}, {0x1.0650a00000000p+0f, 0x1.c783f20000000p-25f},
    {0x1.0706b20000000p+0f, 0x1.3bbedc0000000p-25f}, {0x1.07bd420000000p+0f, 0x1.6e55060000000p-25f},
    {0x1.0874520000000p+0f, -0x1.e2990e0000000p-26f}, {0x1.092be00000000p+0f, -0x1.333f040000000p-25f},
    {0x1.09e3ec0000000p+0f, 0x1.58de700000000p-25f}, {0x1.0a9c7a0000000p+0f, -0x1.3831ba0000000p-26f},
    {0x1.0b55860000000p+0f, 0x1.9f31220000000p-25f}, {0x1.0c0f140000000p+0f, 0x1.791b220000000p-26f},
    {0x1.0cc9220000000p+0f, 0x1.6e48fe0000000p-25f}, {0x1.0d83b20000000p+0f, 0x1.9caef60000000p-27f},
    {0x1.0e3ec40000000p+0f, -0x1.a585cc0000000p-25f}, {0x1.0efa560000000p+0f, -0x1.02b1da0000000p-31f},
    {0x1.0fb66a0000000p+0f, 0x1.ffda640000000p-25f}, {0x1.1073020000000p+0f, 0x1.1ae4680000000p-25f},
    {0x1.11301e0000000p+0f, -0x1.fdb4960000000p-25f}, {0x1.11edba0000000p+0f, 0x1.6bc5560000000p-25f},
    {0x1.12abdc0000000p+0f, 0x1.b0c7300000000p-30f}, {0x1.136a820000000p+0f, -0x1.61bf6a0000000p-25f},
    {0x1.1429aa0000000p+0f, 0x1.d525bc0000000p-25f}, {0x1.14e95a0000000p+0f, -0x1.9619da0000000p-25f},
    {0x1.15a98c0000000p+0f, 0x1.14b1ca0000000p-25f}, {0x1.166a460000000p+0f, -0x1.71c7880000000p-25f},
    {0x1.172b840000000p+0f, -0x1.c157420000000p-27f}, {0x1.17ed480000000p+0f, 0x1.a56ef00000000p-26f},
    {0x1.18af940000000p+0f, -0x1.dcdc860000000p-26f}, {0x1.1972660000000p+0f, -0x1.f228b40000000p-26f},
    {0x1.1a35be0000000p+0f, 0x1.6df96e0000000p-25f}, {0x1.1af9a00000000p+0f, -0x1.fb1d780000000p-26f},
    {0x1.1bbe080000000p+0f, 0x1.0117340000000p-26f}, {0x1.1c82fa0000000p+0f, -0x1.5afc720000000p-25f},
    {0x1.1d48740000000p+0f, -0x1.d2e8ca0000000p-25f}, {0x1.1e0e760000000p+0f, -0x1.4bbfda0000000p-28f},
    {0x1.1ed5020000000p+0f, 0x1.7e6c8e0000000p-27f}, {0x1.1f9c180000000p+0f, 0x1.0e33940000000p-26f},
    {0x1.2063b80000000p+0f, 0x1.0c519a0000000p-25f}, {0x1.212be40000000p+0f, -0x1.50eafc0000000p-25f},
    {0x1.21f49a0000000p+0f, -0x1.d0446e0000000p-25f}, {0x1.22bdda0000000p+0f, 0x1.3c89680000000p-27f},
    {0x1.2387a60000000p+0f, 0x1.ceac480000000p-25f}, {0x1.2452000000000p+0f, -0x1.1f7afe0000000p-26f},
    {0x1.251ce40000000p+0f, 0x1.f654c80000000p-25f}, {0x1.25e8580000000p+0f, -0x1.dc26320000000p-25f},
    {0x1.26b4560000000p+0f, 0x1.789f380000000p-26f}, {0x1.2780e40000000p+0f, -0x1.7c441a0000000p-25f},
    {0x1.284dfe0000000p+0f, 0x1.f563800000000p-28f}, {0x1.291ba80000000p+0f, -0x1.4dc8920000000p-25f},
    {0x1.29e9e00000000p+0f, -0x1.5c04240000000p-25f}, {0x1.2ab8a60000000p+0f, 0x1.b443c40000000p-26f},
    {0x1.2b87fe0000000p+0f, -0x1.e4a4ce0000000p-25f}, {0x1.2c57e40000000p+0f, -0x1.a239340000000p-26f},
    {0x1.2d285a0000000p+0f, 0x1.b900c20000000p-26f}, {0x1.2df9620000000p+0f, -0x1.37d4ee0000000p-29f},
    {0x1.2ecafa0000000p+0f, 0x1.27c5ea0000000p-25f}, {0x1.2f9d240000000p+0f, 0x1.57b10e0000000p-25f},
    {0x1.306fe00000000p+0f, 0x1.4636e20000000p-25f}, {0x1.31432e0000000p+0f, 0x1.bdd6600000000p-25f},
    {0x1.3217100000000p+0f, -0x1.d993e80000000p-27f}, {0x1.32eb840000000p+0f, -0x1.15c5740000000p-26f},
    {0x1.33c08c0000000p+0f, -0x1.b37d200000000p-25f}, {0x1.3496260000000p+0f, 0x1.b8fe8c0000000p-26f},
    {0x1.356c560000000p+0f, -0x1.b5803c0000000p-30f}, {0x1.36431a0000000p+0f, 0x1.6f441e0000000p-27f},
    {0x1.371a740000000p+0f, -0x1.18aac60000000p-25f}, {0x1.37f2620000000p+0f, 0x1.8f3aa40000000p-27f},
    {0x1.38cae60000000p+0f, 0x1.a0bb0c0000000p-25f}, {0x1.39a4020000000p+0f, -0x1.23afc40000000p-26f},
    {0x1.3a7db40000000p+0f, -0x1.634c020000000p-25f}, {0x1.3b57fc0000000p+0f, -0x1.3930ba0000000p-32f},
    {0x1.3c32dc0000000p+0f, 0x1.89d4720000000p-27f}, {0x1.3d0e540000000p+0f, 0x1.3b785c0000000p-26f},
    {0x1.3dea640000000p+0f, 0x1.8246840000000p-25f}, {0x1.3ec70e0000000p+0f, -0x1.c75d160000000p-29f},
    {0x1.3fa4500000000p+0f, 0x1.2b20060000000p-26f}, {0x1.40822c0000000p+0f, 0x1.b3d0120000000p-27f},
    {0x1.4160a20000000p+0f, 0x1.f72e2a0000000p-28f}, {0x1.423fb20000000p+0f, 0x1.c251a20000000p-26f},
    {0x1.431f5e0000000p+0f, -0x1.abd5da0000000p-26f}, {0x1.43ffa40000000p+0f, -0x1.ed18b00000000p-30f},
    {0x1.44e0860000000p+0f, 0x1.8624b40000000p-30f}, {0x1.45c2040000000p+0f, 0x1.53e9180000000p-27f},
    {0x1.46a41e0000000p+0f, 0x1.a3a00a0000000p-25f}, {0x1.4786d60000000p+0f, 0x1.a2cc8e0000000p-26f},
    {0x1.486a2c0000000p+0f, -0x1.47d8660000000p-25f}, {0x1.494e1e0000000p+0f, 0x1.92aed20000000p-28f},
    {0x1.4a32b00000000p+0f, -0x1.e505840000000p-25f}, {0x1.4b17de0000000p+0f, 0x1.4db6fa0000000p-25f},
    {0x1.4bfdae0000000p+0f, -0x1.593abc0000000p-25f}, {0x1.4ce41c0000000p+0f, -0x1.fa0fba0000000p-26f},
    {0x1.4dcb2a0000000p+0f, -0x1.8088bc0000000p-26f}, {0x1.4eb2d80000000p+0f, 0x1.d8abfe0000000p-28f},
    {0x1.4f9b280000000p+0f, -0x1.2c5a6c0000000p-25f}, {0x1.5084180000000p+0f, -0x1.759c240000000p-29f},
    {0x1.516daa0000000p+0f, 0x1.67b3200000000p-27f}, {0x1.5257de0000000p+0f, 0x1.07e9de0000000p-25f},
    {0x1.5342b60000000p+0f, -0x1.2c56100000000p-25f}, {0x1.542e300000000p+0f, -0x1.612a5c0000000p-25f},
    {0x1.551a4c0000000p+0f, 0x1.4bb2420000000p-25f}, {0x1.56070e0000000p+0f, -0x1.0b77980000000p-27f},
    {0x1.56f4740000000p+0f, -0x1.295b040000000p-25f}, {0x1.57e27e0000000p+0f, -0x1.074ecc0000000p-26f},
    {0x1.58d12e0000000p+0f, -0x1.6d07000000000p-25f}, {0x1.59c0820000000p+0f, 0x1.ffc1f20000000p-26f},
    {0x1.5ab07e0000000p+0f, -0x1.5bd5ec0000000p-27f}, {0x1.5ba1200000000p+0f, -0x1.15e1800000000p-26f},
    {0x1.5c92680000000p+0f, 0x1.4b28d60000000p-25f}, {0x1.5d845a0000000p+0f, -0x1.ecce8e0000000p-25f},
    {0x1.5e76f20000000p+0f, -0x1.4a5bd60000000p-25f}, {0x1.5f6a320000000p+0f, 0x1.b9d6e20000000p-29f},
    {0x1.605e1c0000000p+0f, -0x1.a248fe0000000p-26f}, {0x1.6152ae0000000p+0f, 0x1.b37dbe0000000p-26f},
    {0x1.6247ec0000000p+0f, -0x1.f8b5500000000p-25f}, {0x1.633dd20000000p+0f, -0x1.736b020000000p-27f},
    {0x1.6434640000000p+0f, -0x1.66679c0000000p-25f}, {0x1.652ba00000000p+0f, -0x1.43704a0000000p-28f},
    {0x1.6623880000000p+0f, 0x1.2a91120000000p-27f}, {0x1.671c1c0000000p+0f, 0x1.c20cfe0000000p-26f},
    {0x1.68155e0000000p+0f, -0x1.766ad20000000p-25f}, {0x1.690f4c0000000p+0f, -0x1.cc2d580000000p-25f},
    {0x1.6a09e60000000p+0f, 0x1.9fcef40000000p-26f}, {0x1.6b05300000000p+0f, -0x1.62ba300000000p-26f},
    {0x1.6c01280000000p+0f, -0x1.5e84a80000000p-25f}, {0x1.6cfdce0000000p+0f, -0x1.15c4de0000000p-27f},
    {0x1.6dfb240000000p+0f, -0x1.cd72e80000000p-27f}, {0x1.6ef92a0000000p+0f, -0x1.e9b1460000000p-26f},
    {0x1.6ff7e00000000p+0f, -0x1.ab9ae00000000p-26f}, {0x1.70f7460000000p+0f, 0x1.bd0ba20000000p-26f},
    {0x1.71f75e0000000p+0f, 0x1.1d8bee0000000p-25f}, {0x1.72f8280000000p+0f, 0x1.bab4220000000p-26f},
    {0x1.73f9a40000000p+0f, 0x1.14b02e0000000p-25f}, {0x1.74fbd40000000p+0f, -0x1.4506800000000p-25f},
    {0x1.75feb60000000p+0f, -0x1.37b3060000000p-25f}, {0x1.77024c0000000p+0f, -0x1.ca923e0000000p-25f},
    {0x1.7806940000000p+0f, 0x1.fbcba80000000p-25f}, {0x1.790b940000000p+0f, -0x1.d4f8c20000000p-26f},
    {0x1.7a11480000000p+0f, -0x1.829fd00000000p-25f}, {0x1.7b17b00000000p+0f, 0x1.2ed9fc0000000p-25f},
    {0x1.7c1ed00000000p+0f, 0x1.30c1320000000p-28f}, {0x1.7d26a60000000p+0f, 0x1.7fc3780000000p-27f},
    {0x1.7e2f340000000p+0f, -0x1.2616340000000p-25f}, {0x1.7f38780000000p+0f, 0x1.2471240000000p-26f},
    {0x1.8042760000000p+0f, -0x1.783cbe0000000p-25f}, {0x1.814d2a0000000p+0f, 0x1.ba20dc0000000p-25f},
    {0x1.82589a0000000p+0f, -0x1.accc7c0000000p-26f}, {0x1.8364c20000000p+0f, -0x1.46be080000000p-28f},
    {0x1.8471a40000000p+0f, 0x1.88f1ec0000000p-26f}, {0x1.857f420000000p+0f, -0x1.0c149c0000000p-25f},
    {0x1.868d9a0000000p+0f, -0x1.2edb440000000p-26f}, {0x1.879cae0000000p+0f, -0x1.b396f20000000p-26f},
    {0x1.88ac7e0000000p+0f, -0x1.9d665a0000000p-26f}, {0x1.89bd0a0000000p+0f, 0x1.1e16040000000p-26f},
    {0x1.8ace540000000p+0f, 0x1.15506e0000000p-27f}, {0x1.8be05c0000000p+0f, -0x1.4a7a220000000p-26f},
    {0x1.8cf3220000000p+0f, -0x1.29576e0000000p-25f}, {0x1.8e06a60000000p+0f, -0x1.f799280000000p-28f},
    {0x1.8f1aea0000000p+0f, -0x1.baa2320000000p-26f}, {0x1.902fee0000000p+0f, -0x1.fafa6e0000000p-25f},
    {0x1.9145b00000000p+0f, 0x1.723ff80000000p-25f}, {0x1.925c360000000p+0f, -0x1.8aba040000000p-25f},
    {0x1.93737c0000000p+0f, -0x1.e647440000000p-25f}, {0x1.948b820000000p+0f, 0x1.6bf31c0000000p-25f},
    {0x1.95a44c0000000p+0f, 0x1.790a420000000p-25f}, {0x1.96bdda0000000p+0f, -0x1.6263d40000000p-26f},
    {0x1.97d82a0000000p+0f, -0x1.0d8d840000000p-31f}, {0x1.98f33e0000000p+0f, 0x1.1e88a80000000p-26f},
    {0x1.9a0f180000000p+0f, -0x1.e6bf080000000p-25f}, {0x1.9b2bb40000000p+0f, 0x1.aa7fc20000000p-25f},
    {0x1.9c49180000000p+0f, 0x1.51f8480000000p-27f}, {0x1.9d67420000000p+0f, -0x1.ad11ca0000000p-26f},
    {0x1.9e86320000000p+0f, -0x1.8737380000000p-26f}, {0x1.9fa5e80000000p+0f, 0x1.a0fe540000000p-25f},
    {0x1.a0c6680000000p+0f, -0x1.2886a60000000p-26f}, {0x1.a1e7ae0000000p+0f, 0x1.b1d7180000000p-25f},
    {0x1.a309be0000000p+0f, 0x1.8945a60000000p-25f}, {0x1.a42c980000000p+0f, 0x1.182b5e0000000p-30f},
    {0x1.a5503c0000000p+0f, -0x1.b83b540000000p-25f}, {0x1.a674a80000000p+0f, 0x1.5e8c0a0000000p-25f},
    {0x1.a799e20000000p+0f, -0x1.99e9940000000p-25f}, {0x1.a8bfe60000000p+0f, -0x1.87da340000000p-25f},
    {0x1.a9e6b60000000p+0f, -0x1.50c0480000000p-25f}, {0x1.ab0e520000000p+0f, 0x1.356eba0000000p-28f},
    {0x1.ac36bc0000000p+0f, -0x1.6064320000000p-31f}, {0x1.ad5ff40000000p+0f, -0x1.70f6220000000p-26f},
    {0x1.ae89fa0000000p+0f, -0x1.a94b140000000p-26f}, {0x1.afb4ce0000000p+0f, 0x1.88bcc00000000p-26f},
    {0x1.b0e0720000000p+0f, 0x1.31b6cc0000000p-25f}, {0x1.b20ce60000000p+0f, 0x1.93512a0000000p-25f},
    {0x1.b33a2c0000000p+0f, -0x1.ec3a820000000p-26f}, {0x1.b468420000000p+0f, -0x1.4916ca0000000p-25f},
    {0x1.b597280000000p+0f, 0x1.bcab280000000p-25f}, {0x1.b6c6e20000000p+0f, 0x1.3e38a60000000p-25f},
    {0x1.b7f7700000000p+0f, -0x1.a094380000000p-25f}, {0x1.b928d00000000p+0f, -0x1.bb16c40000000p-25f},
    {0x1.ba5b040000000p+0f, -0x1.ebdf360000000p-25f}, {0x1.bb8e0c0000000p+0f, -0x1.0cb21c0000000p-25f},
    {0x1.bcc1ea0000000p+0f, -0x1.f687c60000000p-25f}, {0x1.bdf69c0000000p+0f, 0x1.f9d1040000000p-27f},
    {0x1.bf2c260000000p+0f, -0x1.0a387e0000000p-26f}, {0x1.c062860000000p+0f, 0x1.41b33c0000000p-28f},
    {0x1.c199be0000000p+0f, -0x1.3d56b20000000p-27f}, {0x1.c2d1ce0000000p+0f, -0x1.8166b60000000p-26f},
    {0x1.c40ab60000000p+0f, -0x1.7c2c980000000p-39f}, {0x1.c544780000000p+0f, -0x1.c141380000000p-26f},
    {0x1.c67f120000000p+0f, 0x1.cafa2a0000000p-25f}, {0x1.c7ba880000000p+0f, 0x1.3119260000000p-25f},
    {0x1.c8f6da0000000p+0f, -0x1.7f230a0000000p-25f}, {0x1.ca34060000000p+0f, -0x1.15c7640000000p-25f},
    {0x1.cb720e0000000p+0f, -0x1.8837cc0000000p-27f}, {0x1.ccb0f20000000p+0f, 0x1.cda2ce0000000p-25f},
    {0x1.cdf0b60000000p+0f, -0x1.5447800000000p-25f}, {0x1.cf31560000000p+0f, -0x1.2915240000000p-26f},
    {0x1.d072d40000000p+0f, 0x1.40f1300000000p-25f}, {0x1.d1b5320000000p+0f, 0x1.61192e0000000p-25f},
    {0x1.d2f8700000000p+0f, 0x1.01b13e0000000p-25f}, {0x1.d43c8e0000000p+0f, 0x1.59543a0000000p-25f},
    {0x1.d5818e0000000p+0f, -0x1.822dbc0000000p-27f}, {0x1.d6c76e0000000p+0f, 0x1.0c5cda0000000p-25f},
    {0x1.d80e320000000p+0f, -0x1.26cf8e0000000p-25f}, {0x1.d955d80000000p+0f, -0x1.c013f20000000p-25f},
    {0x1.da9e600000000p+0f, 0x1.ed99420000000p-27f}, {0x1.dbe7ce0000000p+0f, -0x1.38af9e0000000p-25f},
    {0x1.dd32200000000p+0f, -0x1.9fc9740000000p-25f}, {0x1.de7d560000000p+0f, 0x1.0701960000000p-26f},
    {0x1.dfc9740000000p+0f, -0x1.908c940000000p-25f}, {0x1.e116760000000p+0f, 0x1.632fa20000000p-25f},
    {0x1.e264620000000p+0f, -0x1.614bda0000000p-25f}, {0x1.e3b3340000000p+0f, -0x1.3a447c0000000p-26f},
    {0x1.e502ee0000000p+0f, 0x1.e2cffe0000000p-26f}, {0x1.e653920000000p+0f, 0x1.19db5e0000000p-26f},
    {0x1.e7a5200000000p+0f, -0x1.0e2ce00000000p-26f}, {0x1.e8f7980000000p+0f, -0x1.0649180000000p-25f},
    {0x1.ea4afa0000000p+0f, 0x1.52486c0000000p-27f}, {0x1.eb9f480000000p+0f, 0x1.9f329c0000000p-26f},
    {0x1.ecf4820000000p+0f, 0x1.b1ccfe0000000p-25f}, {0x1.ee4aaa0000000p+0f, 0x1.0c42880000000p-27f},
    {0x1.efa1be0000000p+0f, 0x1.cc2b440000000p-25f}, {0x1.f0f9c20000000p+0f, -0x1.a4df6c0000000p-27f},
    {0x1.f252b40000000p+0f, -0x1.1288ae0000000p-25f}, {0x1.f3ac940000000p+0f, 0x1.1bae4e0000000p-25f},
    {0x1.f507660000000p+0f, -0x1.246eb00000000p-26f}, {0x1.f663280000000p+0f, -0x1.9deec20000000p-26f},
    {0x1.f7bfda0000000p+0f, 0x1.b397c20000000p-25f}, {0x1.f91d800000000p+0f, 0x1.121e440000000p-27f},
    {0x1.fa7c180000000p+0f, 0x1.9e90d80000000p-28f}, {0x1.fbdba40000000p+0f, -0x1.2da55e0000000p-25f},
    {0x1.fd3c220000000p+0f, 0x1.71ee3e0000000p-25f}, {0x1.fe9d960000000p+0f, 0x1.65447c0000000p-25f},
};
#define EXPF32_INV16 0x1.7154760000000p+4f /* 16/ln2 */
/* ln2/16 * 2^80, rounded: limbs low to high */
static const uint32_t expf32_c80[3] = {0xf79abc9eu, 0x217f7d1cu, 0x00000b17u};
/* 2^(i/16) * 2^63, rounded: {high, low} */
static const uint32_t expf32_t16[16][2] = {
    {0x80000000u, 0x00000000u}, {0x85aac367u, 0xcc487b15u},
    {0x8b95c1e3u, 0xea8bd6e7u}, {0x91c3d373u, 0xab11c336u},
    {0x9837f051u, 0x8db8a96fu}, {0x9ef53260u, 0x91a111aeu},
    {0xa5fed6a9u, 0xb15138eau}, {0xad583eeau, 0x42a14ac6u},
    {0xb504f333u, 0xf9de6484u}, {0xbd08a39fu, 0x580c36bfu},
    {0xc5672a11u, 0x5506daddu}, {0xce248c15u, 0x1f8480e4u},
    {0xd744fccau, 0xd69d6af4u}, {0xe0ccdeecu, 0x2a94e111u},
    {0xeac0c6e7u, 0xdd24392fu}, {0xf5257d15u, 0x2486cc2cu},
};
/* 2^63 / n!, rounded, n = 0..9: {high, low} */
static const uint32_t expf32_fac[10][2] = {
    {0x80000000u, 0x00000000u}, {0x80000000u, 0x00000000u},
    {0x40000000u, 0x00000000u}, {0x15555555u, 0x55555555u},
    {0x05555555u, 0x55555555u}, {0x01111111u, 0x11111111u},
    {0x002d82d8u, 0x2d82d82eu}, {0x00068068u, 0x06806807u},
    {0x0000d00du, 0x00d00d01u}, {0x0000171du, 0xe3a556c7u},
};

#define EXPF32_SHIFT 0x1.8p23f      /* 1.5 2^23: adding it rounds to an integer */
#define EXPF32_X_OVF 0x1.62e42ep+6f /* the largest float whose exp is finite (expf_table.h) */
#define EXPF32_X_UF -0x1.a0p+6f     /* below -104 exp(x) < 2^-150: zero */
#if EXPF32_MUTATE == 2
#define EXPF32_D 0x1p-45f
#else
#define EXPF32_D 0x1p-39f           /* what the fast path may be off by, in units of y */
#endif
#define EXPF32_SLOW_MARGIN 8u       /* the same for the slow path, in units of 2^-62 */

static inline uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static inline float u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* The fma variant: fused multiply-add rounded once, as a GPU's fma.rn.f32 */
#if defined(__x86_64__) || defined(__i386__)
#define FMA_TARGET __attribute__((target("fma")))
#else
#define FMA_TARGET
#endif

static inline FMA_TARGET float fma32(float a, float b, float c) {
    return __builtin_fmaf(a, b, c);
}

/* The table is read through these two: a mutated entry is what the function and check_constants()
 * both see */
static inline float tab_hi(int32_t j) {
#if EXPF32_MUTATE == 1
    if (j == 77) return u2f(f2u(expf32_tab[j][0]) + 1u);
#endif
    return expf32_tab[j][0];
}

static inline float tab_lo(int32_t j) {
#if EXPF32_MUTATE == 3
    if (j == 77) return u2f(f2u(expf32_tab[j][1]) + 1u);
#endif
    return expf32_tab[j][1];
}

/* ---- fast path -------------------------------------------------------------------------------- */

typedef struct {
    float yh, yl; /* y = 2^(j/256) exp(r) = yh + yl within 2^-39.9 */
    int32_t e;    /* exp(x) = 2^e y */
    float rh, kf; /* for --error: rh = x - kf LH must be exact */
} core32;

static inline FMA_TARGET core32 core_fma(float x) {
    core32 c;
    float t = fma32(x, EXPF32_INVL, EXPF32_SHIFT); /* 1.5 2^23 + round(x 256/ln2) */
    float kf = t - EXPF32_SHIFT;
    int32_t k = (int32_t)kf;
    int32_t j = k & 255;
    float rh = fma32(-kf, EXPF32_LH, x); /* exact: the difference fits in 24 bits */
    float rl = kf * EXPF32_NLL;          /* -k (ln2/256 - LH) */
    float rs = rh + rl;
    float s = fma32(rs * rs, fma32(EXPF32_C3, rs, EXPF32_C2), rl); /* rl + q */
    float th = tab_hi(j), tl = tab_lo(j);
    float ph = th * rh;
    float pl = fma32(th, rh, -ph); /* th rh = ph + pl exactly */
#if EXPF32_MUTATE == 6
    pl = 0.0f;
#endif
    float yh = th + ph;
    float e = (th - yh) + ph; /* th + ph = yh + e exactly: |th| > |ph| */
    float a = fma32(tl, rs, tl) + pl;
    a = a + e;
    c.yh = yh;
    c.yl = fma32(th, s, a);
    c.e = k >> 8;
    c.rh = rh;
    c.kf = kf;
    return c;
}

static inline core32 core_nofma(float x) {
    core32 c;
    float t = x * EXPF32_INVL + EXPF32_SHIFT;
    float kf = t - EXPF32_SHIFT;
    int32_t k = (int32_t)kf;
    int32_t j = k & 255;
    /* |k| < 2^16 times 8 bits: every product exact, and every difference fits in 24 bits */
    float rh = ((x - kf * EXPF32_LA) - kf * EXPF32_LB) - kf * EXPF32_LC;
    float rl = kf * EXPF32_NLL;
    float rs = rh + rl;
    float s = rs * rs * (EXPF32_C3 * rs + EXPF32_C2) + rl;
    float th = tab_hi(j), tl = tab_lo(j);
    /* th rh = ph + pl exactly: Veltkamp's split in 12-bit halves, Dekker's product */
    float ph = th * rh;
    float ct = th * 4097.0f, th1 = ct - (ct - th), th2 = th - th1;
    float cr = rh * 4097.0f, rh1 = cr - (cr - rh), rh2 = rh - rh1;
    float pl = (((th1 * rh1 - ph) + th1 * rh2) + th2 * rh1) + th2 * rh2;
#if EXPF32_MUTATE == 6
    pl = 0.0f;
#endif
    float yh = th + ph;
    float e = (th - yh) + ph;
    float a = (tl * rs + tl) + pl;
    a = a + e;
    c.yh = yh;
    c.yl = th * s + a;
    c.e = k >> 8;
    c.rh = rh;
    c.kf = kf;
    return c;
}

/* 1 when the rounding test settles x, and then *bits is exp(x) correctly rounded */
static inline int finish32(float x, core32 c, uint32_t *bits) {
    if (x >= EXPF32_X_NORM) { /* exp(x) > 2^-126: the grid of y is the float grid */
        float u1 = c.yh + (c.yl - EXPF32_D), u2 = c.yh + (c.yl + EXPF32_D);
        *bits = f2u(u1) + ((uint32_t)c.e << 23);
        return u1 == u2;
    }
    /* exp(x) < 2^-126: the grid 2^-149 is g = 2^(-149-e) in units of y, and in [m, 2m), m = 2^23 g,
     * the floats are exactly its multiples: m + y rounds y on it, and the bits of that float minus
     * those of m are the bits of the subnormal result */
#if EXPF32_MUTATE == 5
    float m = u2f((uint32_t)(2 - c.e) << 23);
#else
    float m = u2f((uint32_t)(1 - c.e) << 23);
#endif
    float g = u2f((uint32_t)(-22 - c.e) << 23);
    float s = m + c.yh;
    float e2 = (m - s) + c.yh;          /* m + yh = s + e2 exactly */
    float v = e2 + c.yl;                /* rounded: at most 2^-24 g, covered by the 2^-22 g below */
    float d2 = g * 0x1p-22f + EXPF32_D; /* exact: two powers of two */
    float u1 = s + (v - d2), u2 = s + (v + d2);
    *bits = f2u(u1) - f2u(m);
    return u1 == u2;
}

/* ---- slow path: integers only ------------------------------------------------------------------- */

typedef struct {
    uint32_t hi, lo;
} w64;
typedef struct {
    uint32_t w[3]; /* w[0] lowest; two's complement */
} w96;

/* the high word of a 32 x 32 product, in 16-bit halves (a GPU has it in one instruction) */
static inline uint32_t mulhi32(uint32_t a, uint32_t b) {
    uint32_t a0 = a & 0xFFFFu, a1 = a >> 16, b0 = b & 0xFFFFu, b1 = b >> 16;
    uint32_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint32_t mid = (p00 >> 16) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu); /* < 3 2^16 */
    return p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
}

static inline w64 add64(w64 a, w64 b) {
    w64 s;
    s.lo = a.lo + b.lo;
    s.hi = a.hi + b.hi + (s.lo < a.lo);
    return s;
}

/* floor(a b / 2^64), exact */
static inline w64 mulhi64(w64 a, w64 b) {
    uint32_t h00 = mulhi32(a.lo, b.lo);
    uint32_t l01 = a.lo * b.hi, h01 = mulhi32(a.lo, b.hi);
    uint32_t l10 = a.hi * b.lo, h10 = mulhi32(a.hi, b.lo);
    uint32_t l11 = a.hi * b.hi, h11 = mulhi32(a.hi, b.hi);
    uint32_t c1 = h00 + l01, k1 = c1 < l01; /* the column of 2^32, and its carry */
    c1 += l10;
    k1 += c1 < l10;
    uint32_t c2 = h01 + h10, k2 = c2 < h10; /* the column of 2^64 */
    c2 += l11;
    k2 += c2 < l11;
    c2 += k1;
    k2 += c2 < k1;
    w64 r = {h11 + k2, c2};
    return r;
}

static inline w96 add96(w96 a, w96 b) {
    w96 s;
    s.w[0] = a.w[0] + b.w[0];
    uint32_t c0 = s.w[0] < a.w[0];
    s.w[1] = a.w[1] + b.w[1];
    uint32_t c1 = s.w[1] < a.w[1];
    s.w[1] += c0;
    c1 += s.w[1] < c0;
    s.w[2] = a.w[2] + b.w[2] + c1;
    return s;
}

static inline w96 neg96(w96 a) {
    w96 n = {{~a.w[0], ~a.w[1], ~a.w[2]}}, one = {{1u, 0u, 0u}};
    return add96(n, one);
}

static inline int ge96(w96 a, w96 b) { /* unsigned */
    if (a.w[2] != b.w[2]) return a.w[2] > b.w[2];
    if (a.w[1] != b.w[1]) return a.w[1] > b.w[1];
    return a.w[0] >= b.w[0];
}

/* floor(v / 2^s), 39 <= s <= 64, when that fits in 32 bits */
static inline uint32_t shr96(w96 v, int s) {
    int t = s - 32;
    return t == 32 ? v.w[2] : (v.w[1] >> t) | (v.w[2] << (32 - t));
}

/* exp(x) correctly rounded, for -104 <= x <= 88.72: the bits, or a NaN and *unproven = 1 when the
 * margin does not settle it (the exhaustive check proves no float gets there) */
static uint32_t expf32_slow(float x, int *unproven) {
    *unproven = 0;
    /* exp(x) rounds to 1 on [-2^-25, 2^-24): exp(-2^-25) = 1 - 2^-25 + 2^-51 is above the middle
     * 1 - 2^-25, and below 2^-24 exp(x) < 1 + x + x^2 < 1 + 2^-24 */
    if (x >= -0x1p-25f && x < 0x1p-24f) return 0x3F800000u;
    uint32_t bx = f2u(x);
    uint32_t sh = ((bx >> 23) & 255u) - 70u; /* |x| 2^80 = m 2^sh, 32 <= sh <= 63 */
    uint32_t m = (bx & 0x7FFFFFu) | 0x800000u;
    w96 xx = {{0u, m << (sh - 32u), sh > 32u ? m >> (64u - sh) : 0u}};
    if (bx >> 31) xx = neg96(xx);
    /* k16 = floor(x 16/ln2): rounded in floats first, then fixed by the exact sign of r */
    float kf = (x * EXPF32_INV16 + EXPF32_SHIFT) - EXPF32_SHIFT;
    int32_t k = (int32_t)kf;
    uint32_t ak = (uint32_t)(k < 0 ? -k : k);
    w96 c = {{expf32_c80[0], expf32_c80[1], expf32_c80[2]}}, p;
    uint32_t h0 = mulhi32(ak, c.w[0]), h1 = mulhi32(ak, c.w[1]);
    p.w[0] = ak * c.w[0];
    p.w[1] = h0 + ak * c.w[1];
    p.w[2] = h1 + ak * c.w[2] + (p.w[1] < h0);
    w96 r = add96(xx, k > 0 ? neg96(p) : p); /* (x - k16 ln2/16) 2^80 */
    while (r.w[2] >> 31) {
        r = add96(r, c);
        k--;
    }
    while (ge96(r, c)) {
        r = add96(r, neg96(c));
        k++;
    }
    w64 rr = {(r.w[2] << 16) | (r.w[1] >> 16), (r.w[1] << 16) | (r.w[0] >> 16)}; /* r 2^64 */
    /* exp(r) 2^63, Horner: acc = 1/n! + r acc */
    w64 acc = {expf32_fac[9][0], expf32_fac[9][1]};
    for (int n = 8; n >= 0; n--) {
        w64 f = {expf32_fac[n][0], expf32_fac[n][1]};
#if EXPF32_MUTATE == 4
        if (n == 6) f.hi = f.lo = 0u;
#endif
        acc = add64(f, mulhi64(rr, acc));
    }
    int32_t i = k & 15, e = k >> 4;
    w64 t16 = {expf32_t16[i][0], expf32_t16[i][1]};
    w64 y = mulhi64(t16, acc); /* 2^(i/16) exp(r) 2^62, in [2^62, 2^63) */
    if (y.hi >> 31) {
        y.lo = (y.lo >> 1) | (y.hi << 31);
        y.hi >>= 1;
        e++;
    }
    if (!(y.hi >> 30)) {
        y.hi = (y.hi << 1) | (y.lo >> 31);
        y.lo <<= 1;
        e--;
    }
    /* exp(x) = y 2^(e - 62): drop s bits, 39 for a normal result, more on the grid 2^-149 */
    int s = e >= -126 ? 39 : -87 - e;
    if (s > 64) return 0u; /* below 2^-151: far from the middle 2^-150 */
    uint32_t half = 1u << (s - 33); /* 2^(s-1), in the middle word */
    w96 v = {{y.lo, y.hi, 0u}};
    w96 lo = {{0u - EXPF32_SLOW_MARGIN, half - 1u, 0u}}, hi = {{EXPF32_SLOW_MARGIN, half, 0u}};
    uint32_t n_lo = shr96(add96(v, lo), s), n_hi = shr96(add96(v, hi), s);
    if (n_lo != n_hi) {
        *unproven = 1;
        return 0x7FC00000u;
    }
    return e >= -126 ? ((uint32_t)(e + 126) << 23) + n_lo : n_lo;
}

/* ---- the whole function ------------------------------------------------------------------------- */

enum { P_SPECIAL, P_FAST, P_FAST_SUB, P_SLOW, P_SLOW_SUB, P_UNPROVEN, P_N };
static const char *const path_name[P_N] = {"special", "fast",        "fast, below 2^-126",
                                           "slow",    "slow, below 2^-126", "unproven"};

static inline int expf32_special(float x, float *y) {
    if (x != x) {
        *y = x + x; /* NaN in, quiet NaN out */
        return 1;
    }
    if (x > EXPF32_X_OVF) {
        *y = u2f(0x7F800000u);
        return 1;
    }
    if (x < EXPF32_X_UF) {
        *y = 0.0f;
        return 1;
    }
    return 0;
}

static inline float expf32_tail(float x, core32 c, int *path) {
    uint32_t bits;
    int sub = x < EXPF32_X_NORM;
    if (finish32(x, c, &bits)) {
        *path = sub ? P_FAST_SUB : P_FAST;
        return u2f(bits);
    }
    int unproven;
    bits = expf32_slow(x, &unproven);
    *path = unproven ? P_UNPROVEN : sub ? P_SLOW_SUB : P_SLOW;
    return u2f(bits);
}

static inline FMA_TARGET float expf32_fma_path(float x, int *path) {
    float y;
    *path = P_SPECIAL;
    if (expf32_special(x, &y)) return y;
    return expf32_tail(x, core_fma(x), path);
}

static inline float expf32_nofma_path(float x, int *path) {
    float y;
    *path = P_SPECIAL;
    if (expf32_special(x, &y)) return y;
    return expf32_tail(x, core_nofma(x), path);
}

static FMA_TARGET float expf32_fma(float x) {
    int path;
    return expf32_fma_path(x, &path);
}

static float expf32_nofma(float x) {
    int path;
    return expf32_nofma_path(x, &path);
}

static float expf32_slow_only(float x) {
    int unproven;
    return u2f(expf32_slow(x, &unproven));
}

/* ---- the fma variant in SIMD (question 58) ------------------------------------------------------- *
 * 16 (AVX-512) or 8 (AVX2) arguments at a time, lane by lane the operations of core_fma, finish32
 * and expf32_special in the same order, so the same bits; the table read by gathers; the lanes the
 * rounding test does not settle (about 1 in 33 000) through expf32_slow, one by one. Each tier
 * returns how many lanes took the slow path and counts the special ones: the exhaustive check wants
 * both branches taken. */
#if defined(__x86_64__) || defined(__i386__)
#define HAVE_EXPF32_SIMD 1
#include <immintrin.h>

static float simd_tab[512]; /* hi, lo pairs, read through tab_hi and tab_lo: mutations reach it */

static void simd_tab_init(void) {
    for (int32_t j = 0; j < 256; j++) {
        simd_tab[2 * j] = tab_hi(j);
        simd_tab[2 * j + 1] = tab_lo(j);
    }
}

/* the lanes whose bit in bad is set, through the slow path */
static int64_t simd_slow_lanes(const float *x, float *y, unsigned bad) {
    int64_t n = 0;
#if EXPF32_MUTATE == 7
    bad = 0; /* mutation: the unsettled lanes keep the fast path's guess */
#endif
    while (bad) {
        int l = __builtin_ctz(bad);
        int unproven;
        y[l] = u2f(expf32_slow(x[l], &unproven));
        bad &= bad - 1;
        n++;
    }
    return n;
}

__attribute__((target("avx512f,fma"))) static int64_t expf32_avx512(const float *x, float *y, int64_t n,
                                                                    int64_t *n_special) {
    const __m512 invl = _mm512_set1_ps(EXPF32_INVL), shift = _mm512_set1_ps(EXPF32_SHIFT);
    const __m512 lh = _mm512_set1_ps(EXPF32_LH), nll = _mm512_set1_ps(EXPF32_NLL);
    const __m512 c2 = _mm512_set1_ps(EXPF32_C2), c3 = _mm512_set1_ps(EXPF32_C3), d = _mm512_set1_ps(EXPF32_D);
    const __m512 xnorm = _mm512_set1_ps(EXPF32_X_NORM), xovf = _mm512_set1_ps(EXPF32_X_OVF);
    const __m512 xuf = _mm512_set1_ps(EXPF32_X_UF), g22 = _mm512_set1_ps(0x1p-22f);
    int64_t slow = 0;
    for (int64_t i = 0; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 kf = _mm512_sub_ps(_mm512_fmadd_ps(vx, invl, shift), shift);
        __m512i k = _mm512_cvttps_epi32(kf);
        __m512i j2 = _mm512_slli_epi32(_mm512_and_si512(k, _mm512_set1_epi32(255)), 1);
        __m512 rh = _mm512_fnmadd_ps(kf, lh, vx);
        __m512 rl = _mm512_mul_ps(kf, nll);
        __m512 rs = _mm512_add_ps(rh, rl);
        __m512 s = _mm512_fmadd_ps(_mm512_mul_ps(rs, rs), _mm512_fmadd_ps(c3, rs, c2), rl);
        __m512 th = _mm512_i32gather_ps(j2, simd_tab, 4);
        __m512 tl = _mm512_i32gather_ps(_mm512_add_epi32(j2, _mm512_set1_epi32(1)), simd_tab, 4);
        __m512 ph = _mm512_mul_ps(th, rh);
        __m512 pl = _mm512_fmsub_ps(th, rh, ph);
#if EXPF32_MUTATE == 6
        pl = _mm512_setzero_ps();
#endif
        __m512 yh = _mm512_add_ps(th, ph);
        __m512 e = _mm512_add_ps(_mm512_sub_ps(th, yh), ph);
        __m512 a = _mm512_add_ps(_mm512_add_ps(_mm512_fmadd_ps(tl, rs, tl), pl), e);
        __m512 yl = _mm512_fmadd_ps(th, s, a);
        __m512i ee = _mm512_srai_epi32(k, 8);
        __m512 u1 = _mm512_add_ps(yh, _mm512_sub_ps(yl, d)), u2 = _mm512_add_ps(yh, _mm512_add_ps(yl, d));
        __m512i bits = _mm512_add_epi32(_mm512_castps_si512(u1), _mm512_slli_epi32(ee, 23));
        __mmask16 ok = _mm512_cmp_ps_mask(u1, u2, _CMP_EQ_OQ);
        __mmask16 sub = _mm512_cmp_ps_mask(vx, xnorm, _CMP_LT_OQ);
        if (sub) {
#if EXPF32_MUTATE == 5
            __m512i mb = _mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(2), ee), 23);
#else
            __m512i mb = _mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(1), ee), 23);
#endif
            __m512 m = _mm512_castsi512_ps(mb);
            __m512 g = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(-22), ee), 23));
            __m512 s2 = _mm512_add_ps(m, yh);
            __m512 v = _mm512_add_ps(_mm512_add_ps(_mm512_sub_ps(m, s2), yh), yl);
            __m512 d2 = _mm512_add_ps(_mm512_mul_ps(g, g22), d);
            __m512 w1 = _mm512_add_ps(s2, _mm512_sub_ps(v, d2)), w2 = _mm512_add_ps(s2, _mm512_add_ps(v, d2));
            bits = _mm512_mask_mov_epi32(bits, sub, _mm512_sub_epi32(_mm512_castps_si512(w1), mb));
            ok = (__mmask16)((ok & ~sub) | (_mm512_cmp_ps_mask(w1, w2, _CMP_EQ_OQ) & sub));
        }
        __mmask16 nan = _mm512_cmp_ps_mask(vx, vx, _CMP_UNORD_Q);
        __mmask16 ovf = _mm512_cmp_ps_mask(vx, xovf, _CMP_GT_OQ);
        __mmask16 uf = _mm512_cmp_ps_mask(vx, xuf, _CMP_LT_OQ);
        bits = _mm512_mask_mov_epi32(bits, ovf, _mm512_set1_epi32(0x7F800000));
        bits = _mm512_mask_mov_epi32(bits, uf, _mm512_setzero_si512());
        bits = _mm512_mask_mov_epi32(bits, nan, _mm512_castps_si512(_mm512_add_ps(vx, vx)));
        __mmask16 spec = (__mmask16)(nan | ovf | uf);
        _mm512_storeu_si512((void *)(y + i), bits);
        *n_special += __builtin_popcount(spec);
        slow += simd_slow_lanes(x + i, y + i, (unsigned)(uint16_t)~(ok | spec));
    }
    return slow;
}

__attribute__((target("avx2,fma"))) static int64_t expf32_avx2(const float *x, float *y, int64_t n,
                                                              int64_t *n_special) {
    const __m256 invl = _mm256_set1_ps(EXPF32_INVL), shift = _mm256_set1_ps(EXPF32_SHIFT);
    const __m256 lh = _mm256_set1_ps(EXPF32_LH), nll = _mm256_set1_ps(EXPF32_NLL);
    const __m256 c2 = _mm256_set1_ps(EXPF32_C2), c3 = _mm256_set1_ps(EXPF32_C3), d = _mm256_set1_ps(EXPF32_D);
    const __m256 xnorm = _mm256_set1_ps(EXPF32_X_NORM), xovf = _mm256_set1_ps(EXPF32_X_OVF);
    const __m256 xuf = _mm256_set1_ps(EXPF32_X_UF), g22 = _mm256_set1_ps(0x1p-22f);
    int64_t slow = 0;
    for (int64_t i = 0; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 kf = _mm256_sub_ps(_mm256_fmadd_ps(vx, invl, shift), shift);
        __m256i k = _mm256_cvttps_epi32(kf);
        __m256i j2 = _mm256_slli_epi32(_mm256_and_si256(k, _mm256_set1_epi32(255)), 1);
        __m256 rh = _mm256_fnmadd_ps(kf, lh, vx);
        __m256 rl = _mm256_mul_ps(kf, nll);
        __m256 rs = _mm256_add_ps(rh, rl);
        __m256 s = _mm256_fmadd_ps(_mm256_mul_ps(rs, rs), _mm256_fmadd_ps(c3, rs, c2), rl);
        __m256 th = _mm256_i32gather_ps(simd_tab, j2, 4);
        __m256 tl = _mm256_i32gather_ps(simd_tab + 1, j2, 4);
        __m256 ph = _mm256_mul_ps(th, rh);
        __m256 pl = _mm256_fmsub_ps(th, rh, ph);
#if EXPF32_MUTATE == 6
        pl = _mm256_setzero_ps();
#endif
        __m256 yh = _mm256_add_ps(th, ph);
        __m256 e = _mm256_add_ps(_mm256_sub_ps(th, yh), ph);
        __m256 a = _mm256_add_ps(_mm256_add_ps(_mm256_fmadd_ps(tl, rs, tl), pl), e);
        __m256 yl = _mm256_fmadd_ps(th, s, a);
        __m256i ee = _mm256_srai_epi32(k, 8);
        __m256 u1 = _mm256_add_ps(yh, _mm256_sub_ps(yl, d)), u2 = _mm256_add_ps(yh, _mm256_add_ps(yl, d));
        __m256i bits = _mm256_add_epi32(_mm256_castps_si256(u1), _mm256_slli_epi32(ee, 23));
        __m256 ok = _mm256_cmp_ps(u1, u2, _CMP_EQ_OQ);
        __m256 sub = _mm256_cmp_ps(vx, xnorm, _CMP_LT_OQ);
        if (_mm256_movemask_ps(sub)) {
#if EXPF32_MUTATE == 5
            __m256i mb = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(2), ee), 23);
#else
            __m256i mb = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(1), ee), 23);
#endif
            __m256 m = _mm256_castsi256_ps(mb);
            __m256 g = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(-22), ee), 23));
            __m256 s2 = _mm256_add_ps(m, yh);
            __m256 v = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(m, s2), yh), yl);
            __m256 d2 = _mm256_add_ps(_mm256_mul_ps(g, g22), d);
            __m256 w1 = _mm256_add_ps(s2, _mm256_sub_ps(v, d2)), w2 = _mm256_add_ps(s2, _mm256_add_ps(v, d2));
            __m256i sbits = _mm256_sub_epi32(_mm256_castps_si256(w1), mb);
            bits = _mm256_blendv_epi8(bits, sbits, _mm256_castps_si256(sub));
            ok = _mm256_blendv_ps(ok, _mm256_cmp_ps(w1, w2, _CMP_EQ_OQ), sub);
        }
        __m256 nan = _mm256_cmp_ps(vx, vx, _CMP_UNORD_Q);
        __m256 ovf = _mm256_cmp_ps(vx, xovf, _CMP_GT_OQ);
        __m256 uf = _mm256_cmp_ps(vx, xuf, _CMP_LT_OQ);
        bits = _mm256_blendv_epi8(bits, _mm256_set1_epi32(0x7F800000), _mm256_castps_si256(ovf));
        bits = _mm256_blendv_epi8(bits, _mm256_setzero_si256(), _mm256_castps_si256(uf));
        bits = _mm256_blendv_epi8(bits, _mm256_castps_si256(_mm256_add_ps(vx, vx)), _mm256_castps_si256(nan));
        unsigned spec = (unsigned)_mm256_movemask_ps(_mm256_or_ps(nan, _mm256_or_ps(ovf, uf)));
        _mm256_storeu_si256((__m256i *)(void *)(y + i), bits);
        *n_special += __builtin_popcount(spec);
        slow += simd_slow_lanes(x + i, y + i, ~((unsigned)_mm256_movemask_ps(ok) | spec) & 0xFFu);
    }
    return slow;
}
#endif

/* ---- every float --------------------------------------------------------------------------------- */

#define MAX_WORKERS 8
#define BLOCK_BITS 16

typedef struct {
    int have_fma, have_avx512, have_avx2, slow_all, error, quick;
    int64_t first, count; /* the blocks of 2^16 floats to run: [first, first + count) */
} run_opts;

typedef struct {
    uint64_t wrong[2], path[2][P_N];
    uint32_t wrong_x[2];
    uint32_t sub_x[2][4]; /* the first arguments of the slow path below 2^-126 */
    uint64_t slow_n, slow_wrong, slow_unproven;
    uint32_t slow_wrong_x;
    double err[2]; /* largest |yh + yl - y| */
    float err_x[2];
    uint64_t rh_inexact[2];
    uint64_t simd_wrong[2], simd_slow[2], simd_spec[2]; /* [0] AVX-512, [1] AVX2 */
    uint32_t simd_wrong_x[2];
} tally32;
static tally32 tallies[MAX_WORKERS];
#if HAVE_EXPF32_SIMD
static float simd_x[MAX_WORKERS][1 << BLOCK_BITS], simd_y[2][MAX_WORKERS][1 << BLOCK_BITS];
#endif

static void measure(float x, double y_ref, core32 c, double *err, float *err_x, uint64_t *rh_inexact) {
    double y = y_ref * ldexp(1.0, -c.e);
    double d = fabs(((double)c.yh + (double)c.yl) - y); /* the sum is exact in double */
    if (d > *err) {
        *err = d;
        *err_x = x;
    }
    *rh_inexact += (double)c.rh != (double)x - (double)c.kf * (double)EXPF32_LH; /* exact in double */
}

static void bits_body(void *ctx, int64_t begin, int64_t end, int worker) {
    const run_opts *o = ctx;
    tally32 t; /* on the stack: two workers' tallies may share a cache line */
    memset(&t, 0, sizeof t);
    for (int64_t b = o->first + begin; b < o->first + end; b++) {
        if (o->quick && (b & 63) != 0) continue;
#if HAVE_EXPF32_SIMD
        const float *ys[2] = {simd_y[0][worker], simd_y[1][worker]};
        if (o->have_avx512 || o->have_avx2) {
            for (uint32_t i = 0; i < (1u << BLOCK_BITS); i++) simd_x[worker][i] = u2f(((uint32_t)b << BLOCK_BITS) | i);
            if (o->have_avx512)
                t.simd_slow[0] += (uint64_t)expf32_avx512(simd_x[worker], simd_y[0][worker], 1 << BLOCK_BITS,
                                                          (int64_t *)&t.simd_spec[0]);
            if (o->have_avx2)
                t.simd_slow[1] += (uint64_t)expf32_avx2(simd_x[worker], simd_y[1][worker], 1 << BLOCK_BITS,
                                                        (int64_t *)&t.simd_spec[1]);
        }
#endif
        for (uint32_t i = 0; i < (1u << BLOCK_BITS); i++) {
            uint32_t xb = ((uint32_t)b << BLOCK_BITS) | i;
            float x = u2f(xb);
            uint32_t want = f2u(tr_expf(x));
#if HAVE_EXPF32_SIMD
            for (int v = 0; v < 2; v++)
                if ((v == 0 ? o->have_avx512 : o->have_avx2) && f2u(ys[v][i]) != want && t.simd_wrong[v]++ == 0)
                    t.simd_wrong_x[v] = xb;
#endif
            int path;
            uint32_t got = f2u(expf32_nofma_path(x, &path));
            if (path == P_SLOW_SUB && t.path[0][path] < 4) t.sub_x[0][t.path[0][path]] = xb;
            t.path[0][path]++;
            if (got != want && t.wrong[0]++ == 0) t.wrong_x[0] = xb;
            if (o->have_fma) {
                got = f2u(expf32_fma_path(x, &path));
                if (path == P_SLOW_SUB && t.path[1][path] < 4) t.sub_x[1][t.path[1][path]] = xb;
                t.path[1][path]++;
                if (got != want && t.wrong[1]++ == 0) t.wrong_x[1] = xb;
            }
            if (!(x >= EXPF32_X_UF && x <= EXPF32_X_OVF)) continue;
            if (o->slow_all) {
                int unproven;
                got = expf32_slow(x, &unproven);
                t.slow_n++;
                t.slow_unproven += (uint64_t)unproven;
                if (got != want && t.slow_wrong++ == 0) t.slow_wrong_x = xb;
            }
            if (o->error) {
                double y_ref = exp((double)x);
                measure(x, y_ref, core_nofma(x), &t.err[0], &t.err_x[0], &t.rh_inexact[0]);
                if (o->have_fma) measure(x, y_ref, core_fma(x), &t.err[1], &t.err_x[1], &t.rh_inexact[1]);
            }
        }
    }
    tally32 *mine = &tallies[worker];
    for (int v = 0; v < 2; v++) {
        if (t.wrong[v] && !mine->wrong[v]) mine->wrong_x[v] = t.wrong_x[v];
        mine->wrong[v] += t.wrong[v];
        for (uint64_t k = 0; k < t.path[v][P_SLOW_SUB] && k < 4; k++)
            if (mine->path[v][P_SLOW_SUB] + k < 4) mine->sub_x[v][mine->path[v][P_SLOW_SUB] + k] = t.sub_x[v][k];
        for (int p = 0; p < P_N; p++) mine->path[v][p] += t.path[v][p];
        if (t.err[v] > mine->err[v]) {
            mine->err[v] = t.err[v];
            mine->err_x[v] = t.err_x[v];
        }
        mine->rh_inexact[v] += t.rh_inexact[v];
        if (t.simd_wrong[v] && !mine->simd_wrong[v]) mine->simd_wrong_x[v] = t.simd_wrong_x[v];
        mine->simd_wrong[v] += t.simd_wrong[v];
        mine->simd_slow[v] += t.simd_slow[v];
        mine->simd_spec[v] += t.simd_spec[v];
    }
    if (t.slow_wrong && !mine->slow_wrong) mine->slow_wrong_x = t.slow_wrong_x;
    mine->slow_n += t.slow_n;
    mine->slow_wrong += t.slow_wrong;
    mine->slow_unproven += t.slow_unproven;
}

/* ns per call, one thread, best of 7 */
static double cost_ns(float (*fn)(float), const float *xs, int n, float *sum) {
    enum { REPS = 64 };
    double best = 1e30;
    for (int r = 0; r < 7; r++) {
        double t0 = tr_time_sec();
        float acc = 0.0f;
        for (int k = 0; k < REPS; k++)
            for (int i = 0; i < n; i++) acc += fn(xs[i]);
        double dt = tr_time_sec() - t0;
        *sum += acc;
        if (dt < best) best = dt;
    }
    return best * 1e9 / ((double)n * REPS);
}

/* ---- the constants against their definitions ---------------------------------------------------- *
 * The exhaustive check cannot see a constant off by less than the margins absorb: a low part one ulp
 * off leaves every result correctly rounded (an equivalent mutant, docs/LESSONS.md #83). Each constant
 * is compared here with its definition: long double (64 bits) and integers, the harness's own means. */

static long double ulp_of(float f) { /* of a normal float */
    return ldexpl(1.0L, (int)((f2u(f) >> 23) & 255u) - 150);
}

/* got is not want rounded to nearest, want being known within tol */
static int off_rn(long double want, float got, long double tol) {
    return fabsl(want - (long double)got) > ulp_of(got) / 2 + tol;
}

static int check_constants(void) {
    const long double ln2 = logl(2.0L), l256 = ln2 / 256;
    int bad = 0;
    for (int32_t j = 0; j < 256; j++) {
        /* t within 2^-62 (exp2l: 64 bits); rest = t - th exactly in long double, so also within 2^-62 */
        long double t = exp2l((long double)j / 256.0L), rest = t - tab_hi(j);
        bad += off_rn(t, tab_hi(j), 0x1p-62L);
        bad += tab_lo(j) != 0.0f ? off_rn(rest, tab_lo(j), 0x1p-62L) : fabsl(rest) > 0x1p-62L;
    }
    bad += off_rn(256 / ln2, EXPF32_INVL, 0x1p-52L) + off_rn(l256, EXPF32_LH, 0x1p-70L) +
           off_rn(16 / ln2, EXPF32_INV16, 0x1p-56L);
    bad += fabsl(l256 - EXPF32_LH + EXPF32_NLL) > ulp_of(EXPF32_NLL) / 2 + 0x1p-70L; /* NLL = RN(LH - ln2/256) */
    bad += (double)EXPF32_LA + (double)EXPF32_LB + (double)EXPF32_LC != (double)EXPF32_LH;
    bad += (f2u(EXPF32_LA) & 0xFFFFu) != 0 || (f2u(EXPF32_LB) & 0xFFFFu) != 0 || (f2u(EXPF32_LC) & 0xFFFFu) != 0;
    bad += EXPF32_C2 != 0.5f + 0x1p-24f;
    bad += off_rn(1.0L / 6, EXPF32_C3, 0x1p-64L);
    bad += !(EXPF32_X_NORM > -126 * ln2) || !(nextafterf(EXPF32_X_NORM, -INFINITY) < -126 * ln2);
    long double max_mid = 0x1p128L - 0x1p103L; /* above it exp rounds to infinity */
    bad += !(expl(EXPF32_X_OVF) < max_mid) || !(expl(nextafterf(EXPF32_X_OVF, INFINITY)) >= max_mid);
    bad += !(expl(EXPF32_X_UF) < 0x1p-150L);
    uint64_t nf = 1; /* 2^63 / n!, rounded: exact in integers */
    for (int n = 0; n < 10; n++) {
        nf *= n > 1 ? (uint64_t)n : 1u;
        uint64_t q = ((uint64_t)1 << 63) / nf, r = ((uint64_t)1 << 63) % nf;
        q += 2 * r >= nf;
        bad += ((uint64_t)expf32_fac[n][0] << 32 | expf32_fac[n][1]) != q;
    }
    for (int i = 0; i < 16; i++) { /* 2^(i/16) 2^63, to 2 units (long double has 64 bits) */
        uint64_t t = (uint64_t)expf32_t16[i][0] << 32 | expf32_t16[i][1];
        bad += fabsl(ldexpl(exp2l(i / 16.0L), 63) - (long double)t) > 2.0L;
    }
    /* ln2/16 2^80 rounded, from ln2 2^128 = 0xb17217f7d1cf79ab c9e3b39803f2f6af (truncated) */
    uint64_t hi = 0xb17217f7d1cf79abu, lo = 0xc9e3b39803f2f6afu;
    uint64_t c_lo = (hi << 12 | lo >> 52) + (lo >> 51 & 1u), c_hi = hi >> 52; /* no carry: bit 51 is 0 here */
    bad += c_lo != ((uint64_t)expf32_c80[1] << 32 | expf32_c80[0]) || c_hi != expf32_c80[2];
    return bad;
}

#if HAVE_EXPF32_SIMD
/* ns per element of a SIMD tier over an array, one thread, best of 7 */
static double cost_simd_ns(int64_t (*fn)(const float *, float *, int64_t, int64_t *), const float *xs, int n,
                           float *sum) {
    enum { REPS = 64 };
    static float out[1 << 16];
    double best = 1e30;
    int64_t sp = 0;
    for (int r = 0; r < 7; r++) {
        double t0 = tr_time_sec();
        for (int k = 0; k < REPS; k++) fn(xs, out, n, &sp);
        double dt = tr_time_sec() - t0;
        *sum += out[n / 2];
        if (dt < best) best = dt;
    }
    return best * 1e9 / ((double)n * REPS);
}
#endif

static void timing(const run_opts *o) {
    int have_fma = o->have_fma;
    enum { N = 1 << 16 };
    static float soft[N], wide[N];
    for (int i = 0; i < N; i++) {
        float u = (float)((i * 2654435761u) >> 8 & 0xFFFF) / 65536.0f;
        soft[i] = -12.0f * u;            /* a softmax row: scores minus their maximum */
        wide[i] = -104.0f + 192.7f * u;  /* the whole range where exp is not 0 nor infinite */
    }
    const char *set_name[2] = {"-12..0", "-104..88.7"};
    const float *sets[2] = {soft, wide};
    for (int s = 0; s < 2; s++) {
        float sum = 0.0f;
        double ref = cost_ns(tr_expf, sets[s], N, &sum), nf = cost_ns(expf32_nofma, sets[s], N, &sum);
        double fm = have_fma ? cost_ns(expf32_fma, sets[s], N, &sum) : 0.0;
        double sl = cost_ns(expf32_slow_only, sets[s], N / 16, &sum);
        int slow = 0, path;
        for (int i = 0; i < N; i++) {
            (void)expf32_nofma_path(sets[s][i], &path);
            slow += path == P_SLOW || path == P_SLOW_SUB;
        }
        printf("cost, x in %s: tr_expf %.2f ns, expf32 %.2f ns without fma, %.2f with fma, slow path alone "
               "%.1f; %d of %d arguments take the slow path (sum %.1f)\n",
               set_name[s], ref, nf, fm, sl, slow, N, (double)sum);
#if HAVE_EXPF32_SIMD
        double v512 = o->have_avx512 ? cost_simd_ns(expf32_avx512, sets[s], N, &sum) : 0.0;
        double v256 = o->have_avx2 ? cost_simd_ns(expf32_avx2, sets[s], N, &sum) : 0.0;
        printf("  SIMD, fma variant, a value: AVX-512 %.3f ns (%.1fx tr_expf), AVX2 %.3f ns (%.1fx) (sum %.1f)\n", v512,
               v512 > 0 ? ref / v512 : 0.0, v256, v256 > 0 ? ref / v256 : 0.0, (double)sum);
#endif
    }
}

int main(int argc, char **argv) {
    run_opts o = {.first = 0, .count = (int64_t)1 << (32 - BLOCK_BITS)};
    int threads = 4, no_timing = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--slow-all") == 0) o.slow_all = 1;
        else if (strcmp(argv[i], "--error") == 0) o.error = 1;
        else if (strcmp(argv[i], "--quick") == 0) o.quick = 1;
        else if (strcmp(argv[i], "--blocks") == 0 && i + 2 < argc) {
            o.first = atoll(argv[++i]);
            o.count = atoll(argv[++i]) - o.first;
            if (o.first < 0 || o.count < 1 || o.first + o.count > ((int64_t)1 << (32 - BLOCK_BITS))) {
                fprintf(stderr, "--blocks A B: 0 <= A < B <= 65536\n");
                return 2;
            }
        }
        else if (strcmp(argv[i], "--no-timing") == 0) no_timing = 1;
        else {
            fprintf(stderr, "usage: bench_expf32 [--threads T] [--slow-all] [--error] [--quick] [--no-timing] [--blocks A B]\n");
            return 2;
        }
    }
    int cores = tr_cpu()->physical_cores;
    if (threads < 1) threads = 4;
    if (threads > cores) threads = cores;
    if (threads > MAX_WORKERS) threads = MAX_WORKERS;
#if defined(__x86_64__) || defined(__i386__)
    o.have_fma = tr_cpu()->fma;
#else
    o.have_fma = 1;
#endif
    char cpu_line[512];
    tr_cpu_describe(tr_cpu(), cpu_line, sizeof cpu_line);
    printf("bench_expf32, built %s %s, mutation %d, %d threads\n%s\n", __DATE__, __TIME__, EXPF32_MUTATE, threads,
           cpu_line);
    if (!o.have_fma) printf("no fma on this CPU: the fma variant is not run\n");
#if HAVE_EXPF32_SIMD
    simd_tab_init();
    o.have_avx512 = o.have_fma && tr_cpu()->avx512f;
    o.have_avx2 = o.have_fma && tr_cpu()->avx2;
#endif

    int bad_constants = check_constants();
    printf("constants: %d differ from their definitions (the table, the reduction, the polynomial, the "
           "borders, the slow path)\n", bad_constants);
    TR_CHECK(bad_constants == 0);
    if (!no_timing) timing(&o);

    tr_pool *pool = tr_pool_create(threads);
    if (pool == NULL) return 1;
    double t0 = tr_time_sec();
    tr_parallel_for(pool, o.count, 16, bits_body, &o);
    double secs = tr_time_sec() - t0;
    tr_pool_destroy(pool);
    tally32 sum;
    memset(&sum, 0, sizeof sum);
    for (int w = 0; w < threads; w++) {
        for (int v = 0; v < 2; v++) {
            if (tallies[w].wrong[v] && !sum.wrong[v]) sum.wrong_x[v] = tallies[w].wrong_x[v];
            sum.wrong[v] += tallies[w].wrong[v];
            for (uint64_t k = 0; k < tallies[w].path[v][P_SLOW_SUB] && k < 4; k++)
                if (sum.path[v][P_SLOW_SUB] + k < 4) sum.sub_x[v][sum.path[v][P_SLOW_SUB] + k] = tallies[w].sub_x[v][k];
            for (int p = 0; p < P_N; p++) sum.path[v][p] += tallies[w].path[v][p];
            if (tallies[w].err[v] > sum.err[v]) {
                sum.err[v] = tallies[w].err[v];
                sum.err_x[v] = tallies[w].err_x[v];
            }
            sum.rh_inexact[v] += tallies[w].rh_inexact[v];
            if (tallies[w].simd_wrong[v] && !sum.simd_wrong[v]) sum.simd_wrong_x[v] = tallies[w].simd_wrong_x[v];
            sum.simd_wrong[v] += tallies[w].simd_wrong[v];
            sum.simd_slow[v] += tallies[w].simd_slow[v];
            sum.simd_spec[v] += tallies[w].simd_spec[v];
        }
        if (tallies[w].slow_wrong && !sum.slow_wrong) sum.slow_wrong_x = tallies[w].slow_wrong_x;
        sum.slow_n += tallies[w].slow_n;
        sum.slow_wrong += tallies[w].slow_wrong;
        sum.slow_unproven += tallies[w].slow_unproven;
    }

    const char *var_name[2] = {"without fma", "with fma"};
    int whole = !o.quick && o.count == (int64_t)1 << (32 - BLOCK_BITS);
    if (o.count < (int64_t)1 << (32 - BLOCK_BITS))
        printf("blocks %lld..%lld of 65536 (a slice: the tally lines of all the slices add up to the proof), "
               "%.1f s\n", (long long)o.first, (long long)(o.first + o.count), secs);
    else
        printf("%s floats (%s), %.1f s\n", o.quick ? "1/64 of the" : "all the 2^32",
               o.quick ? "a smoke test, not a proof" : "the proof", secs);
    /* one line for the sum of slices (the scratch script adds them up and applies the checks) */
    printf("tally first=%lld count=%lld quick=%d fma=%d", (long long)o.first, (long long)o.count, o.quick, o.have_fma);
    for (int v = 0; v < 2; v++) {
        printf(" w%d=%llu wx%d=%08x p%d=", v, (unsigned long long)sum.wrong[v], v, (unsigned)sum.wrong_x[v], v);
        for (int p = 0; p < P_N; p++) printf("%llu%s", (unsigned long long)sum.path[v][p], p + 1 < P_N ? "," : "");
        printf(" e%d=%a ex%d=%a rh%d=%llu s%d=%08x,%08x,%08x,%08x", v, sum.err[v], v, (double)sum.err_x[v], v,
               (unsigned long long)sum.rh_inexact[v], v, (unsigned)sum.sub_x[v][0], (unsigned)sum.sub_x[v][1],
               (unsigned)sum.sub_x[v][2], (unsigned)sum.sub_x[v][3]);
    }
    printf(" sn=%llu sw=%llu swx=%08x su=%llu", (unsigned long long)sum.slow_n, (unsigned long long)sum.slow_wrong,
           (unsigned)sum.slow_wrong_x, (unsigned long long)sum.slow_unproven);
    printf(" vw=%llu,%llu vs=%llu,%llu vp=%llu,%llu\n", (unsigned long long)sum.simd_wrong[0],
           (unsigned long long)sum.simd_wrong[1], (unsigned long long)sum.simd_slow[0],
           (unsigned long long)sum.simd_slow[1], (unsigned long long)sum.simd_spec[0],
           (unsigned long long)sum.simd_spec[1]);
    const char *tier_name[2] = {"AVX-512", "AVX2"};
    for (int v = 0; v < 2; v++) {
        if (!(v == 0 ? o.have_avx512 : o.have_avx2)) continue;
        printf("expf32 SIMD %s (fma variant): %llu results differ from tr_expf", tier_name[v],
               (unsigned long long)sum.simd_wrong[v]);
        if (sum.simd_wrong[v]) printf(" (the first: x bits %08x)", (unsigned)sum.simd_wrong_x[v]);
        printf("; %llu lanes through the slow path, %llu special\n", (unsigned long long)sum.simd_slow[v],
               (unsigned long long)sum.simd_spec[v]);
        TR_CHECK(sum.simd_wrong[v] == 0);
        if (whole) { /* both of the tier's own branches taken */
            TR_CHECK(sum.simd_slow[v] > 0);
            TR_CHECK(sum.simd_spec[v] > 0);
        }
    }
    for (int v = 0; v < 2; v++) {
        if (v == 1 && !o.have_fma) continue;
        printf("expf32 %s: %llu results differ from tr_expf", var_name[v], (unsigned long long)sum.wrong[v]);
        if (sum.wrong[v]) printf(" (the first: x bits %08x)", (unsigned)sum.wrong_x[v]);
        printf("\n  paths:");
        uint64_t total = 0;
        for (int p = 0; p < P_N; p++) total += sum.path[v][p];
        for (int p = 0; p < P_N; p++)
            printf(" %s %llu (%.3g%%)%s", path_name[p], (unsigned long long)sum.path[v][p],
                   100.0 * (double)sum.path[v][p] / (double)total, p + 1 < P_N ? "," : "\n");
        if (sum.path[v][P_SLOW_SUB]) {
            printf("  slow path below 2^-126, the first arguments (bits):");
            for (uint64_t k = 0; k < sum.path[v][P_SLOW_SUB] && k < 4; k++) printf(" %08x", (unsigned)sum.sub_x[v][k]);
            printf("\n");
        }
        TR_CHECK(sum.wrong[v] == 0);
        TR_CHECK(sum.path[v][P_UNPROVEN] == 0);
        if (whole) { /* every branch taken at least once */
            TR_CHECK(sum.path[v][P_SPECIAL] > 0);
            TR_CHECK(sum.path[v][P_FAST] > 0);
            TR_CHECK(sum.path[v][P_FAST_SUB] > 0);
            TR_CHECK(sum.path[v][P_SLOW] > 0);
            TR_CHECK(sum.path[v][P_SLOW_SUB] > 0);
        }
        if (o.error) {
            printf("  largest |yh + yl - y| = 2^%.2f = %.3f D, at x = %a; rh inexact for %llu arguments\n",
                   log2(sum.err[v]), sum.err[v] / (double)EXPF32_D, (double)sum.err_x[v],
                   (unsigned long long)sum.rh_inexact[v]);
            TR_CHECK(sum.err[v] < 0.9 * (double)EXPF32_D); /* 0.1 D: the rounding of yl +- D */
            TR_CHECK(sum.rh_inexact[v] == 0);
        }
    }
    if (o.slow_all) {
        printf("slow path alone, every float of [-104, 88.72]: %llu arguments, %llu differ from tr_expf",
               (unsigned long long)sum.slow_n, (unsigned long long)sum.slow_wrong);
        if (sum.slow_wrong) printf(" (the first: x bits %08x)", (unsigned)sum.slow_wrong_x);
        printf(", %llu unproven\n", (unsigned long long)sum.slow_unproven);
        if (whole) TR_CHECK(sum.slow_n > 0);
        TR_CHECK(sum.slow_wrong == 0);
        TR_CHECK(sum.slow_unproven == 0);
    }
    TR_TEST_EXIT();
}
