/* attn_probe.c — diagnostic only, never in the engine: what one decode token's attention reads.
 *
 * Linked into the build of `make attn-probe` (build/probe/, olmoe.c compiled with
 * -DTR_ATTN_PROBE), which calls tr_attn_probe for every head of every decode token. With
 * TR_PROBE_DIR=<dir> set it appends, per layer and head, the query and its position count to
 * <dir>/q_L<l>_H<h>.bin (int64 n_pos, then head_dim floats); when n_pos equals TR_PROBE_KV_AT it
 * also writes that head's keys and values (n_pos * head_dim floats each) to <dir>/k_L<l>_H<h>.bin
 * and v_L<l>_H<h>.bin. The keys and values of an earlier token are a prefix of those: one dump at
 * the last token serves every query. After the attention, the head's output goes to
 * <dir>/o_L<l>_H<h>.bin like the query (int64 n_pos, then head_dim floats): the bits the replay of
 * tools/attn_skip_report.py must reproduce before its counts are believed.
 *
 * A head of a layer is one call on one worker, so no two calls write the same file at once. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void dump(const char *dir, const char *what, int64_t layer, int64_t head, const float *x, int64_t n) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s_L%02" PRId64 "_H%02" PRId64 ".bin", dir, what, layer, head);
    FILE *f = fopen(path, "wb");
    if (f == NULL) return;
    fwrite(x, sizeof(float), (size_t)n, f);
    fclose(f);
}

void tr_attn_probe(int64_t layer, int64_t head, const float *q, const float *keys, const float *values,
                   int64_t n_pos, int64_t head_dim, float scale) {
    (void)scale;
    const char *dir = getenv("TR_PROBE_DIR");
    if (dir == NULL) return;
    char path[1024];
    snprintf(path, sizeof path, "%s/q_L%02" PRId64 "_H%02" PRId64 ".bin", dir, layer, head);
    FILE *f = fopen(path, "ab");
    if (f == NULL) return;
    fwrite(&n_pos, sizeof n_pos, 1, f);
    fwrite(q, sizeof(float), (size_t)head_dim, f);
    fclose(f);
    const char *at = getenv("TR_PROBE_KV_AT");
    if (at != NULL && strtoll(at, NULL, 10) == n_pos) {
        dump(dir, "k", layer, head, keys, n_pos * head_dim);
        dump(dir, "v", layer, head, values, n_pos * head_dim);
    }
}

void tr_attn_probe_out(int64_t layer, int64_t head, const float *out, int64_t n_pos, int64_t head_dim) {
    const char *dir = getenv("TR_PROBE_DIR");
    if (dir == NULL) return;
    char path[1024];
    snprintf(path, sizeof path, "%s/o_L%02" PRId64 "_H%02" PRId64 ".bin", dir, layer, head);
    FILE *f = fopen(path, "ab");
    if (f == NULL) return;
    fwrite(&n_pos, sizeof n_pos, 1, f);
    fwrite(out, sizeof(float), (size_t)head_dim, f);
    fclose(f);
}
