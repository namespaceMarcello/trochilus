/* test_gguf.c — the GGUF reader against valid, truncated, corrupted and fuzzed files.
 *
 * Model files come from the internet. Whatever the bytes, tr_gguf_open must either
 * return a reader whose every tensor lies inside the file, or NULL with a message:
 * never crash, never read out of bounds (run under ASan to see the second part). */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "test.h"
#include "../src/format/gguf.h"

/* ------------------------------------------------------------ file builder */

typedef struct {
    uint8_t *data;
    size_t len, cap;
} buf;

static void put(buf *b, const void *p, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2;
        b->data = realloc(b->data, b->cap);
        if (b->data == NULL) abort();
    }
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static size_t put_u32(buf *b, uint32_t v) { size_t at = b->len; put(b, &v, 4); return at; }
static size_t put_u64(buf *b, uint64_t v) { size_t at = b->len; put(b, &v, 8); return at; }
static size_t put_str(buf *b, const char *s) { size_t at = put_u64(b, strlen(s)); put(b, s, strlen(s)); return at; }
static void pad_to(buf *b, size_t align) { uint8_t z = 0; while (b->len % align) put(b, &z, 1); }

static void patch_u32(buf *b, size_t at, uint32_t v) { memcpy(b->data + at, &v, 4); }
static void patch_u64(buf *b, size_t at, uint64_t v) { memcpy(b->data + at, &v, 8); }

/* Offsets of the fields the corruption cases patch. */
typedef struct {
    size_t magic, version, n_tensors, n_kv;
    size_t key0_len, kv_type_u8, kv_bool_value, align_value, arr_elem_type, arr_len, arch_value_str;
    size_t ta_name, ta_ndims, ta_ne0, ta_type, ta_offset;
    size_t tb_name, tb_ne0, flag_key;
} layout;

/* A valid file: 6 metadata entries, tensor "a" (f32, 4 elements) and tensor "b"
 * (q8_0, 2 rows of 32), data aligned to 32. */
static buf build_valid(layout *L) {
    buf b = {0};
    L->magic = put_u32(&b, 0x46554747u);
    L->version = put_u32(&b, 3);
    L->n_tensors = put_u64(&b, 2);
    L->n_kv = put_u64(&b, 6);

    L->key0_len = put_str(&b, "general.architecture");
    put_u32(&b, TR_GGUF_STRING);
    L->arch_value_str = put_str(&b, "test");

    put_str(&b, "general.alignment");
    put_u32(&b, TR_GGUF_UINT32);
    L->align_value = put_u32(&b, 32);

    put_str(&b, "test.u8");
    L->kv_type_u8 = put_u32(&b, TR_GGUF_UINT8);
    uint8_t seven = 7;
    put(&b, &seven, 1);

    L->flag_key = put_str(&b, "test.flag");
    put_u32(&b, TR_GGUF_BOOL);
    L->kv_bool_value = b.len;
    uint8_t one = 1;
    put(&b, &one, 1);

    put_str(&b, "test.words");
    put_u32(&b, TR_GGUF_ARRAY);
    L->arr_elem_type = put_u32(&b, TR_GGUF_STRING);
    L->arr_len = put_u64(&b, 2);
    put_str(&b, "a");
    put_str(&b, "bc");

    put_str(&b, "test.ints");
    put_u32(&b, TR_GGUF_ARRAY);
    put_u32(&b, TR_GGUF_INT32);
    put_u64(&b, 3);
    int32_t ints[3] = {1, -2, 3};
    put(&b, ints, sizeof ints);

    L->ta_name = put_str(&b, "a");
    L->ta_ndims = put_u32(&b, 1);
    L->ta_ne0 = put_u64(&b, 4);
    L->ta_type = put_u32(&b, TR_TYPE_F32);
    L->ta_offset = put_u64(&b, 0);

    L->tb_name = put_str(&b, "b");
    put_u32(&b, 2);
    L->tb_ne0 = put_u64(&b, 32);
    put_u64(&b, 2);
    put_u32(&b, TR_TYPE_Q8_0);
    put_u64(&b, 32);

    pad_to(&b, 32);
    float a[4] = {1.0f, -2.5f, 3.25f, 0.0f};
    put(&b, a, sizeof a);
    pad_to(&b, 32);
    for (int i = 0; i < 68; i++) { uint8_t v = (uint8_t)i; put(&b, &v, 1); }
    return b;
}

static char path[512];

static int write_file(const uint8_t *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    size_t w = n ? fwrite(p, 1, n, f) : 0;
    fclose(f);
    return w == n ? 0 : -1;
}

/* Opens path; if it opens, every tensor must be readable in full. Returns 1 if opened. */
static int open_and_touch(char *err, size_t err_len) {
    tr_gguf *g = tr_gguf_open(path, err, err_len);
    if (g == NULL) return 0;
    for (uint64_t i = 0; i < tr_gguf_tensor_count(g); i++) {
        const tr_gguf_tensor *t = tr_gguf_tensor_at(g, i);
        TR_CHECK(t->n_bytes <= (1u << 20));
        if (t->n_bytes > (1u << 20)) continue;
        uint8_t *tmp = malloc((size_t)t->n_bytes + 1);
        TR_CHECK(tmp != NULL && tr_gguf_read(g, t, tmp) == 0);
        free(tmp);
    }
    for (uint64_t i = 0; i < tr_gguf_kv_count(g); i++) {
        const tr_gguf_kv *kv = tr_gguf_kv_at(g, i);
        TR_CHECK(tr_gguf_find_kv(g, kv->key) == kv);
    }
    tr_gguf_close(g);
    return 1;
}

/* ------------------------------------------------------------ cases */

static void test_valid(const buf *b) {
    char err[256];
    TR_CHECK(write_file(b->data, b->len) == 0);
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g != NULL);
    if (g == NULL) { fprintf(stderr, "valid file rejected: %s\n", err); return; }

    TR_CHECK_EQ_INT(tr_gguf_version(g), 3);
    TR_CHECK_EQ_INT(tr_gguf_kv_count(g), 6);
    TR_CHECK_EQ_INT(tr_gguf_tensor_count(g), 2);

    const char *s = NULL;
    TR_CHECK(tr_gguf_get_str(g, "general.architecture", &s) == 0 && strcmp(s, "test") == 0);
    uint32_t u = 0;
    TR_CHECK(tr_gguf_get_u32(g, "test.u8", &u) == 0 && u == 7);
    TR_CHECK(tr_gguf_get_u32(g, "general.alignment", &u) == 0 && u == 32);
    int flag = 0;
    TR_CHECK(tr_gguf_get_bool(g, "test.flag", &flag) == 0 && flag == 1);
    TR_CHECK(tr_gguf_get_str(g, "test.u8", &s) == -1);
    TR_CHECK(tr_gguf_get_u32(g, "missing", &u) == -1);

    const tr_gguf_kv *words = tr_gguf_find_kv(g, "test.words");
    TR_CHECK(words != NULL && words->type == TR_GGUF_ARRAY && words->arr_type == TR_GGUF_STRING && words->arr_len == 2);
    if (words) TR_CHECK(strcmp(((char **)words->arr)[1], "bc") == 0);
    const tr_gguf_kv *ints = tr_gguf_find_kv(g, "test.ints");
    TR_CHECK(ints != NULL && ints->arr_len == 3);
    if (ints) { int32_t v; memcpy(&v, (char *)ints->arr + 4, 4); TR_CHECK_EQ_INT(v, -2); }

    /* the arena hands out 16-byte aligned memory, as it says: its block header is 24 bytes, and
     * data placed right after it came out at 8 mod 16 (docs/LESSONS.md #105) */
    int aligned_seen = 0;
    for (uint64_t i = 0; i < tr_gguf_kv_count(g); i++) {
        const tr_gguf_kv *kv = tr_gguf_kv_at(g, i);
        TR_CHECK_EQ_INT((uintptr_t)kv->key % 16, 0);
        if (kv->type == TR_GGUF_ARRAY) { TR_CHECK_EQ_INT((uintptr_t)kv->arr % 16, 0); aligned_seen++; }
    }
    TR_CHECK(aligned_seen == 2);

    const tr_gguf_tensor *ta = tr_gguf_find_tensor(g, "a");
    const tr_gguf_tensor *tb = tr_gguf_find_tensor(g, "b");
    TR_CHECK(ta != NULL && tb != NULL && tr_gguf_find_tensor(g, "c") == NULL);
    if (ta && tb) {
        TR_CHECK_EQ_INT(ta->n_bytes, 16);
        TR_CHECK_EQ_INT(tb->n_bytes, 68);
        TR_CHECK_EQ_INT(tb->ne[1], 2);
        TR_CHECK_EQ_INT(tb->n_elems, 64);
        float a[4];
        TR_CHECK(tr_gguf_read(g, ta, a) == 0 && a[1] == -2.5f && a[2] == 3.25f);
        uint8_t row[34];
        TR_CHECK(tr_gguf_read_range(g, tb, 34, row, 34) == 0 && row[0] == 34 && row[33] == 67);
        TR_CHECK(tr_gguf_read_range(g, tb, 35, row, 34) == -1);
    }
    tr_gguf_close(g);
}

static void test_truncated(const buf *b) {
    char err[256];
    int opened = 0;
    for (size_t n = 0; n < b->len; n++) {
        TR_CHECK(write_file(b->data, n) == 0);
        if (open_and_touch(err, sizeof err)) {
            fprintf(stderr, "truncated to %zu of %zu bytes and still opened\n", n, b->len);
            opened++;
        }
    }
    TR_CHECK_EQ_INT(opened, 0);
}

typedef void (*corrupt_fn)(buf *b, const layout *L);

static void c_magic(buf *b, const layout *L)       { patch_u32(b, L->magic, 0x46554746u); }
static void c_version(buf *b, const layout *L)     { patch_u32(b, L->version, 1); }
static void c_n_kv(buf *b, const layout *L)        { patch_u64(b, L->n_kv, 1ull << 40); }
static void c_n_tensors(buf *b, const layout *L)   { patch_u64(b, L->n_tensors, 1ull << 40); }
static void c_key_len(buf *b, const layout *L)     { patch_u64(b, L->key0_len, 1ull << 62); }
static void c_vtype(buf *b, const layout *L)       { patch_u32(b, L->kv_type_u8, 13); }
static void c_bool(buf *b, const layout *L)        { b->data[L->kv_bool_value] = 2; }
static void c_align(buf *b, const layout *L)       { patch_u32(b, L->align_value, 3); }
static void c_nested(buf *b, const layout *L)      { patch_u32(b, L->arr_elem_type, TR_GGUF_ARRAY); }
static void c_arr_len(buf *b, const layout *L)     { patch_u64(b, L->arr_len, 1ull << 40); }
static void c_nul(buf *b, const layout *L)         { b->data[L->arch_value_str + 8 + 2] = 0; }
static void c_ndims0(buf *b, const layout *L)      { patch_u32(b, L->ta_ndims, 0); }
static void c_ndims5(buf *b, const layout *L)      { patch_u32(b, L->ta_ndims, 5); }
static void c_ne_zero(buf *b, const layout *L)     { patch_u64(b, L->ta_ne0, 0); }
static void c_ne_huge(buf *b, const layout *L)     { patch_u64(b, L->ta_ne0, 1ull << 50); }
static void c_type(buf *b, const layout *L)        { patch_u32(b, L->ta_type, 4); }
static void c_row_len(buf *b, const layout *L)     { patch_u64(b, L->tb_ne0, 33); }
static void c_misalign(buf *b, const layout *L)    { patch_u64(b, L->ta_offset, 1); }
static void c_past_end(buf *b, const layout *L)    { patch_u64(b, L->ta_offset, 96); }
static void c_wrap(buf *b, const layout *L)        { patch_u64(b, L->ta_offset, UINT64_MAX - 31); }
static void c_dup_tensor(buf *b, const layout *L)  { b->data[L->tb_name + 8] = 'a'; }
static void c_dup_key(buf *b, const layout *L)     { memcpy(b->data + L->flag_key + 8, "test.ints", 9); }

static void test_corrupted(const layout *L, const buf *valid) {
    static const struct { const char *name; corrupt_fn fn; } cases[] = {
        {"magic", c_magic}, {"version", c_version}, {"n_kv", c_n_kv}, {"n_tensors", c_n_tensors},
        {"key length", c_key_len}, {"value type", c_vtype}, {"bool value", c_bool},
        {"alignment", c_align}, {"nested array", c_nested}, {"array length", c_arr_len},
        {"NUL in string", c_nul}, {"0 dims", c_ndims0}, {"5 dims", c_ndims5},
        {"zero dim", c_ne_zero}, {"huge dim", c_ne_huge}, {"unknown type", c_type},
        {"row not a block multiple", c_row_len}, {"misaligned offset", c_misalign},
        {"offset past end", c_past_end}, {"offset wraps", c_wrap},
        {"duplicate tensor", c_dup_tensor}, {"duplicate key", c_dup_key},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        buf b = {malloc(valid->len), valid->len, valid->len};
        memcpy(b.data, valid->data, valid->len);
        cases[i].fn(&b, L);
        TR_CHECK(write_file(b.data, b.len) == 0);
        char err[256] = {0};
        tr_gguf *g = tr_gguf_open(path, err, sizeof err);
        if (g != NULL) {
            fprintf(stderr, "corruption '%s' was accepted\n", cases[i].name);
            tr_gguf_close(g);
            TR_TEST_FAILED();
        } else {
            TR_CHECK(err[0] != 0);
        }
        free(b.data);
    }
}

static void test_fuzz(const buf *valid) {
    uint64_t state = 0x9E3779B97F4A7C15ull;
    char err[256];
    int opened = 0;
    for (int it = 0; it < 3000; it++) {
        buf b = {malloc(valid->len), valid->len, valid->len};
        memcpy(b.data, valid->data, valid->len);
        int flips = 1 + it % 4;
        for (int k = 0; k < flips; k++) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            size_t pos = (size_t)((state >> 33) % valid->len);
            b.data[pos] ^= (uint8_t)(1u << ((state >> 20) % 8));
        }
        TR_CHECK(write_file(b.data, b.len) == 0);
        opened += open_and_touch(err, sizeof err);
        free(b.data);
    }
    printf("fuzz: %d of 3000 mutated files opened (all tensors read in bounds)\n", opened);
}

/* ------------------------------------------------------------ what the valid file leaves out
 * Each case below was a mutation of src/format/gguf.c that the cases above let through
 * (tools/mutate_auto.py, 2026-09-22): a getter never called, a boundary never reached, a path
 * of the reader never taken. */

#define BIG_ARR (3u << 20)          /* > READ_CHUNK: rd() reads it straight into the array */
#define LONG_STR ((3u << 19) + 5u)  /* 1.5 MiB + 5: a string through the same path */

static int open_expect(const buf *b, int want_open, const char *what) {
    TR_CHECK(write_file(b->data, b->len) == 0);
    char err[256] = {0};
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    int ok = (g != NULL) == want_open;
    if (!ok) fprintf(stderr, "%s: %s (%s)\n", what, want_open ? "rejected" : "accepted", err);
    TR_CHECK(ok);
    if (!want_open) TR_CHECK(err[0] != 0);
    tr_gguf_close(g);
    return ok;
}

/* n_kv keys "k.<i>" (u32 value i), key `dup_of` repeated as the last one when >= 0 */
static buf build_keys(int n_kv, int dup_of) {
    buf b = {0};
    put_u32(&b, 0x46554747u);
    put_u32(&b, 3);
    put_u64(&b, 0);
    put_u64(&b, (uint64_t)n_kv);
    char key[32];
    for (int i = 0; i < n_kv; i++) {
        snprintf(key, sizeof key, "k.%d", i == n_kv - 1 && dup_of >= 0 ? dup_of : i);
        put_str(&b, key);
        put_u32(&b, TR_GGUF_UINT32);
        put_u32(&b, (uint32_t)i);
    }
    pad_to(&b, 32);
    return b;
}

static void test_more(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u);
    put_u32(&b, 3);
    put_u64(&b, 2);
    size_t n_kv_at = put_u64(&b, 0);
    uint64_t n_kv = 0;
    put_str(&b, "general.alignment"); put_u32(&b, TR_GGUF_UINT32); put_u32(&b, 64); n_kv++;
    double f64 = 2.5; float f32 = 0.25f; int32_t neg = -5; int16_t pos = 300; uint64_t big = 1ull << 33;
    put_str(&b, "t.f64"); put_u32(&b, TR_GGUF_FLOAT64); put(&b, &f64, 8); n_kv++;
    put_str(&b, "t.f32"); put_u32(&b, TR_GGUF_FLOAT32); put(&b, &f32, 4); n_kv++;
    put_str(&b, "t.neg"); put_u32(&b, TR_GGUF_INT32); put(&b, &neg, 4); n_kv++;
    put_str(&b, "t.pos"); put_u32(&b, TR_GGUF_INT16); put(&b, &pos, 2); n_kv++;
    put_str(&b, "t.big"); put_u32(&b, TR_GGUF_UINT64); put(&b, &big, 8); n_kv++;
    put_str(&b, "t.bools"); put_u32(&b, TR_GGUF_ARRAY); put_u32(&b, TR_GGUF_BOOL); put_u64(&b, 3);
    size_t bools_at = b.len;
    uint8_t bools[3] = {0, 1, 1};
    put(&b, bools, 3); n_kv++;
    put_str(&b, "t.f64s"); put_u32(&b, TR_GGUF_ARRAY); put_u32(&b, TR_GGUF_FLOAT64); put_u64(&b, 2);
    double f64s[2] = {1.5, -2.0};
    put(&b, f64s, sizeof f64s); n_kv++;
    put_str(&b, "t.big_arr"); put_u32(&b, TR_GGUF_ARRAY); put_u32(&b, TR_GGUF_UINT8); put_u64(&b, BIG_ARR);
    for (uint32_t i = 0; i < BIG_ARR; i++) { uint8_t v = (uint8_t)(i * 7u + (i >> 16)); put(&b, &v, 1); }
    n_kv++;
    put_str(&b, "t.long_str"); put_u32(&b, TR_GGUF_STRING); put_u64(&b, LONG_STR);
    for (uint32_t i = 0; i < LONG_STR; i++) { char c = (char)('a' + i % 26); put(&b, &c, 1); }
    n_kv++;
    patch_u64(&b, n_kv_at, n_kv);
    /* a 4-dim tensor, then one whose data ends exactly at the end of the file */
    put_str(&b, "four"); put_u32(&b, 4);
    put_u64(&b, 2); put_u64(&b, 1); put_u64(&b, 2); put_u64(&b, 1);
    put_u32(&b, TR_TYPE_F32); put_u64(&b, 0);
    put_str(&b, "last"); put_u32(&b, 1); put_u64(&b, 8); put_u32(&b, TR_TYPE_F32); put_u64(&b, 64);
    pad_to(&b, 64);
    for (int i = 0; i < 16; i++) { float v = (float)i; put(&b, &v, 4); }
    pad_to(&b, 64);
    for (int i = 0; i < 8; i++) { float v = 100.0f + (float)i; put(&b, &v, 4); }

    TR_CHECK(write_file(b.data, b.len) == 0);
    char err[256];
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g != NULL);
    if (g == NULL) fprintf(stderr, "test_more: rejected: %s\n", err);
    if (g != NULL) {
        float f = 0; uint32_t u32 = 0; uint64_t u64 = 0; int flag = -1; const char *s = NULL;
        TR_CHECK(tr_gguf_get_f32(g, "t.f64", &f) == 0 && f == 2.5f);
        TR_CHECK(tr_gguf_get_f32(g, "t.f32", &f) == 0 && f == 0.25f);
        TR_CHECK(tr_gguf_get_f32(g, "t.pos", &f) == -1);
        TR_CHECK(tr_gguf_get_f32(g, "missing", &f) == -1);
        TR_CHECK(tr_gguf_get_u32(g, "t.pos", &u32) == 0 && u32 == 300);
        TR_CHECK(tr_gguf_get_u32(g, "t.neg", &u32) == -1);
        TR_CHECK(tr_gguf_get_u64(g, "t.neg", &u64) == -1);
        TR_CHECK(tr_gguf_get_u32(g, "t.big", &u32) == -1);
        TR_CHECK(tr_gguf_get_u64(g, "t.big", &u64) == 0 && u64 == big);
        TR_CHECK(tr_gguf_get_u64(g, "t.f32", &u64) == -1);
        TR_CHECK(tr_gguf_get_bool(g, "t.pos", &flag) == -1 && flag == -1);
        TR_CHECK(tr_gguf_get_bool(g, "missing", &flag) == -1);
        TR_CHECK(tr_gguf_get_str(g, "t.pos", &s) == -1);
        TR_CHECK(tr_gguf_find_kv(g, NULL) == NULL && tr_gguf_find_tensor(g, NULL) == NULL);
        TR_CHECK(tr_gguf_kv_at(g, n_kv) == NULL && tr_gguf_kv_at(g, n_kv - 1) != NULL);
        TR_CHECK(tr_gguf_tensor_at(g, 2) == NULL && tr_gguf_tensor_at(g, 1) != NULL);

        const tr_gguf_kv *kv = tr_gguf_find_kv(g, "t.bools");
        TR_CHECK(kv != NULL && kv->arr_len == 3 && memcmp(kv->arr, bools, 3) == 0);
        kv = tr_gguf_find_kv(g, "t.f64s");
        TR_CHECK(kv != NULL && kv->arr_type == TR_GGUF_FLOAT64 && memcmp(kv->arr, f64s, sizeof f64s) == 0);
        kv = tr_gguf_find_kv(g, "t.big_arr");
        int bad = kv == NULL || kv->arr_len != BIG_ARR;
        for (uint32_t i = 0; !bad && i < BIG_ARR; i++) bad = ((uint8_t *)kv->arr)[i] != (uint8_t)(i * 7u + (i >> 16));
        TR_CHECK(!bad);
        TR_CHECK(tr_gguf_get_str(g, "t.long_str", &s) == 0 && strlen(s) == LONG_STR && s[0] == 'a' &&
                 s[LONG_STR - 1] == (char)('a' + (LONG_STR - 1) % 26));

        const tr_gguf_tensor *four = tr_gguf_find_tensor(g, "four"), *last = tr_gguf_find_tensor(g, "last");
        TR_CHECK(four != NULL && four->n_dims == 4 && four->ne[2] == 2 && four->n_elems == 4 && four->n_bytes == 16);
        TR_CHECK(last != NULL && last->n_bytes == 32);
        if (four != NULL && last != NULL) {
            float v[8];
            /* "four" has data after it: a range past its end is refused by the range check itself */
            TR_CHECK(tr_gguf_read_range(g, four, 4, v, 16) == -1);
            TR_CHECK(tr_gguf_read_range(g, four, 17, v, 0) == -1);
            TR_CHECK(tr_gguf_read_range(g, four, 16, v, 0) == 0);
            TR_CHECK(tr_gguf_read_range(g, four, 4, v, 12) == 0 && v[0] == 1.0f && v[2] == 3.0f);
            TR_CHECK(tr_gguf_read(g, last, v) == 0 && v[0] == 100.0f && v[7] == 107.0f);
        }
        tr_gguf_close(g);
    }

    /* a bool array with a 2 in it is refused, as a bool scalar is */
    b.data[bools_at + 1] = 2;
    open_expect(&b, 0, "bool array value 2");
    free(b.data);

    /* alignment 0 and 65536 (the largest allowed) */
    buf a = build_keys(1, -1);
    patch_u64(&a, 16, 1);
    buf al = {0};
    put(&al, a.data, 24);
    put_str(&al, "general.alignment"); put_u32(&al, TR_GGUF_UINT32);
    size_t al_value = put_u32(&al, 0);
    open_expect(&al, 0, "alignment 0");
    patch_u32(&al, al_value, 1u << 16);
    pad_to(&al, 1u << 16);
    open_expect(&al, 1, "alignment 65536");
    free(a.data);
    free(al.data);

    /* forty keys: all found through the probing; the last repeating the eighth is refused */
    buf k = build_keys(40, -1);
    open_expect(&k, 1, "40 distinct keys");
    TR_CHECK(write_file(k.data, k.len) == 0);
    tr_gguf *gk = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(gk != NULL);
    for (int i = 0; gk != NULL && i < 40; i++) {
        char key[32];
        uint32_t v = 0;
        snprintf(key, sizeof key, "k.%d", i);
        TR_CHECK(tr_gguf_get_u32(gk, key, &v) == 0 && v == (uint32_t)i);
    }
    tr_gguf_close(gk);
    free(k.data);
    k = build_keys(40, 7);
    open_expect(&k, 0, "key 7 twice, 32 keys apart");
    /* no error buffer at all: refused just the same, nothing written */
    TR_CHECK(tr_gguf_open(path, NULL, 0) == NULL);
    free(k.data);

    /* no tensors, and the last value ends exactly on the alignment: the reader's last read ends at
     * the last byte of the file */
    buf e = {0};
    put_u32(&e, 0x46554747u); put_u32(&e, 3); put_u64(&e, 0); put_u64(&e, 1);
    put_str(&e, "k"); put_u32(&e, TR_GGUF_STRING);
    size_t tail = e.len + 8;
    size_t n = (32 - tail % 32) % 32 + 32;
    put_u64(&e, n);
    for (size_t i = 0; i < n; i++) { char c = 'x'; put(&e, &c, 1); }
    TR_CHECK(e.len % 32 == 0);
    open_expect(&e, 1, "file ending on its last value");
    free(e.data);
}

int main(int argc, char **argv) {
    const char *self = argc > 0 ? argv[0] : "";
    const char *slash = strrchr(self, '/');
    const char *bslash = strrchr(self, '\\');
    if (bslash && (!slash || bslash > slash)) slash = bslash;
    int dir_len = slash ? (int)(slash - self) : 1;
    snprintf(path, sizeof path, "%.*s/test_gguf_tmp.gguf", dir_len, slash ? self : ".");

    layout L;
    buf valid = build_valid(&L);
    test_valid(&valid);
    test_truncated(&valid);
    test_corrupted(&L, &valid);
    test_fuzz(&valid);
    test_more();
    remove(path);
    free(valid.data);
    TR_TEST_EXIT();
}
