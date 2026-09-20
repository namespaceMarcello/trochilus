/* dump_rope.c — the RoPE table this platform builds, written to a file.
 *
 * tr_rope_table takes pow, cos and sin from the C library, in double, and rounds to float. After
 * tr_expf it is the one place left where the engine's numbers come from a library that is not
 * the same code on two platforms (MinGW-w64, glibc). This writes the table, and the doubles it
 * is rounded from, so that two platforms can be compared entry by entry
 * (tools/rope_table_compare.py):
 *
 *   dump_rope <n_pos> <head_dim> <theta> <file>
 *
 * With n = n_pos * head_dim/2, the file holds: n floats of cosines and n of sines (the engine's
 * table, from tr_rope_table), then head_dim/2 doubles of pow(theta, 2i/head_dim), then n doubles
 * of cos(angle) and n of sin(angle), the same expressions tr_rope_table evaluates. A measurement
 * tool, not a test: make build/tests/dump_rope(.exe). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/kernels/kernels.h"

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: dump_rope <n_pos> <head_dim> <theta> <file>\n");
        return 2;
    }
    int64_t n_pos = atoll(argv[1]), head_dim = atoll(argv[2]);
    float theta = (float)atof(argv[3]);
    if (n_pos < 1 || head_dim < 2 || head_dim % 2 != 0) return 2;
    int64_t half = head_dim / 2;
    size_t n = (size_t)(n_pos * half);
    float *cos_t = malloc(n * sizeof(float)), *sin_t = malloc(n * sizeof(float));
    double *pow_d = malloc((size_t)half * sizeof(double));
    double *cos_d = malloc(n * sizeof(double)), *sin_d = malloc(n * sizeof(double));
    if (cos_t == NULL || sin_t == NULL || pow_d == NULL || cos_d == NULL || sin_d == NULL) return 1;
    tr_rope_table(cos_t, sin_t, n_pos, head_dim, theta);
    double theta_d = (double)theta;
    for (int64_t i = 0; i < half; i++) {
        volatile double pw = pow(theta_d, (2.0 * (double)i) / (double)head_dim);
        double inv_freq = 1.0 / pw;
        pow_d[i] = pw;
        for (int64_t p = 0; p < n_pos; p++) {
            double angle = inv_freq * (double)p;
            cos_d[p * half + i] = cos(angle);
            sin_d[p * half + i] = sin(angle);
        }
    }
    FILE *f = fopen(argv[4], "wb");
    if (f == NULL) return 1;
    int ok = fwrite(cos_t, sizeof(float), n, f) == n && fwrite(sin_t, sizeof(float), n, f) == n &&
             fwrite(pow_d, sizeof(double), (size_t)half, f) == (size_t)half &&
             fwrite(cos_d, sizeof(double), n, f) == n && fwrite(sin_d, sizeof(double), n, f) == n;
    ok = fclose(f) == 0 && ok;
    printf("dump_rope: %lld positions, head_dim %lld, theta %g: %zu entries\n", (long long)n_pos,
           (long long)head_dim, (double)theta, n);
    return ok ? 0 : 1;
}
