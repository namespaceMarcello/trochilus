/* dump_dequant.c — the scalar dequantization of raw blocks, written to a file, for
 * tools/check_dequant.py, which compares it bit for bit with gguf-py's (llama.cpp's Python
 * package, the reader our converters and oracles use):
 *
 *   dump_dequant <q8_0|q4_k> <blocks.bin> <out.f32>
 *
 * Reads whole blocks, writes one float per element in the machine's byte order. The scalar table
 * is the definition (kernels.h); the SIMD tiers are held to it by tests/test_kernels.c. A tool for
 * the gate, not a test: make build/tests/dump_dequant(.exe). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/format/gguf.h"
#include "../src/kernels/kernels.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: dump_dequant <q8_0|q4_k> <blocks.bin> <out.f32>\n");
        return 2;
    }
    tr_type type;
    if (strcmp(argv[1], "q8_0") == 0) type = TR_TYPE_Q8_0;
    else if (strcmp(argv[1], "q4_k") == 0) type = TR_TYPE_Q4_K;
    else {
        fprintf(stderr, "dump_dequant: unknown type %s\n", argv[1]);
        return 2;
    }
    const tr_type_info *ti = tr_type_get((uint32_t)type);
    const tr_kernels *k = tr_kernels_tier("scalar");
    FILE *f = fopen(argv[2], "rb");
    if (ti == NULL || k == NULL || k->dequant_row[type] == NULL || f == NULL) {
        fprintf(stderr, "dump_dequant: cannot read %s as %s\n", argv[2], argv[1]);
        if (f != NULL) fclose(f);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0 || len % (long)ti->block_bytes != 0) {
        fprintf(stderr, "dump_dequant: %s holds %ld bytes, not whole %s blocks\n", argv[2], len, argv[1]);
        fclose(f);
        return 1;
    }
    int64_t n = (int64_t)(len / (long)ti->block_bytes) * (int64_t)ti->block_elems;
    unsigned char *raw = malloc((size_t)len);
    float *out = malloc((size_t)n * sizeof(float));
    int ok = raw != NULL && out != NULL && fread(raw, 1, (size_t)len, f) == (size_t)len;
    fclose(f);
    if (ok) {
        k->dequant_row[type](raw, out, n);
        FILE *o = fopen(argv[3], "wb");
        ok = o != NULL && fwrite(out, sizeof(float), (size_t)n, o) == (size_t)n;
        if (o != NULL) ok = fclose(o) == 0 && ok;
    }
    free(raw);
    free(out);
    if (!ok) fprintf(stderr, "dump_dequant: could not write %s\n", argv[3]);
    return ok ? 0 : 1;
}
