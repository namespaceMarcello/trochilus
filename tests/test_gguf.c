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
    size_t header_end, data_start;  /* where the tensor infos end, and the aligned data begins */
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

    L->header_end = b.len;
    pad_to(&b, 32);
    L->data_start = b.len;
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

/* A k-quant tensor (Q4_K: 144 bytes a block, Q6_K: 210): its bytes counted in blocks of 256
 * elements, a row read by its range; a row that is not whole blocks refused by name (the type table,
 * src/format/gguf.c). */
static void test_k_quant(tr_type type, const char *refusal, int bb) {
    int checked = 0;
    for (int bad = 0; bad < 2; bad++) {
        buf b = {0};
        put_u32(&b, 0x46554747u);
        put_u32(&b, 3);
        put_u64(&b, 1);                        /* tensors */
        put_u64(&b, 0);                        /* key-value pairs: alignment 32 by default */
        put_str(&b, "w");
        put_u32(&b, 2);
        put_u64(&b, bad ? 128 : 256);
        put_u64(&b, 3);
        put_u32(&b, type);
        put_u64(&b, 0);
        pad_to(&b, 32);
        for (int i = 0; i < 3 * bb; i++) { uint8_t v = (uint8_t)i; put(&b, &v, 1); }
        TR_CHECK(write_file(b.data, b.len) == 0);
        char err[256] = "";
        tr_gguf *g = tr_gguf_open(path, err, sizeof err);
        if (bad) {
            TR_CHECK(g == NULL && strstr(err, refusal) != NULL);
            checked++;
        } else {
            const tr_gguf_tensor *t = g != NULL ? tr_gguf_find_tensor(g, "w") : NULL;
            TR_CHECK(t != NULL);
            if (t != NULL) {
                TR_CHECK_EQ_INT(t->type, type);
                TR_CHECK_EQ_INT(t->n_elems, 768);
                TR_CHECK_EQ_INT(t->n_bytes, 3 * bb);
                uint8_t row[256];
                TR_CHECK(tr_gguf_read_range(g, t, 2 * (uint64_t)bb, row, (size_t)bb) == 0 && row[0] == (uint8_t)(2 * bb) &&
                         row[bb - 1] == (uint8_t)(3 * bb - 1));
                TR_CHECK(tr_gguf_read_range(g, t, 2 * (uint64_t)bb + 1, row, (size_t)bb) == -1);
                checked++;
            }
        }
        tr_gguf_close(g);
        free(b.data);
    }
    TR_CHECK_EQ_INT(checked, 2);
}

/* Every prefix of the valid file is rejected, and with a message of the part it ends in. Inside
 * the header every count, length and dimension is legitimate, so running out of bytes is the only
 * way it can fail: "unexpected end of file" from a read, or the checks that compare a length with
 * the bytes left before reading ("string length ... out of range", "array longer than file"). In
 * the header's padding "data section starts past end of file", inside the data "outside the file".
 * fail() keeps the last message, so a swallowed read failure shows when the garbage it leaves trips
 * some other check (a value type, a dimension): a wrong message, not just opened != 0 (mutate_auto:
 * several ignored "return -1"s only cascade to a different rejection). */
static void test_truncated(const buf *b, const layout *L) {
    static const char *header_msgs[3] = {"unexpected end of file", "out of range", "array longer than file"};
    char err[256];
    int opened = 0, right_msg = 0, seen[3] = {0, 0, 0}, in_pad = 0, in_data = 0;
    for (size_t n = 0; n < b->len; n++) {
        TR_CHECK(write_file(b->data, n) == 0);
        err[0] = 0;
        if (open_and_touch(err, sizeof err)) {
            fprintf(stderr, "truncated to %zu of %zu bytes and still opened\n", n, b->len);
            opened++;
            continue;
        }
        TR_CHECK(err[0] != 0);
        int right = 0;
        if (n < L->header_end) {
            for (int m = 0; m < 3; m++)
                if (strstr(err, header_msgs[m]) != NULL) { seen[m]++; right = 1; }
        } else if (n < L->data_start) {
            right = strstr(err, "data section starts past end of file") != NULL;
            in_pad += right;
        } else {
            right = strstr(err, "outside the file") != NULL;
            in_data += right;
        }
        /* 4 bytes, the magic alone: read whole (n == file_size), the file fails on the version
         * after it (gguf.c's rd: a bound off by one would fail on the magic, at @0) */
        if (n == 4 && strstr(err, "gguf @4: unexpected end of file") == NULL) right = 0;
        right_msg += right;
        if (!right) fprintf(stderr, "truncated to %zu of %zu bytes: rejected with '%s'\n", n, b->len, err);
    }
    TR_CHECK_EQ_INT(opened, 0);
    TR_CHECK_EQ_INT(right_msg, (int)b->len);
    /* every message above was met: the header's three, the padding, the data */
    TR_CHECK(seen[0] > 0 && seen[1] > 0 && seen[2] > 0 && in_pad > 0 && in_data > 0);
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

/* ------------------------------------------------------------ mutant-driven cases (2026-09-24)
 * Each case below kills one surviving mutant from tools/mutate_auto.py on src/format/gguf.c that
 * test_corrupted's whole-valid-file corruptions did not: patching one field of the full valid file
 * often leaves the *other* disjunct of an `||`, or a later size check, to reject the file anyway
 * (same NULL, different reason) -- a weakened check never gets exercised alone. These build
 * minimal, purpose-built files instead, so exactly one condition is ever true, or a truncation
 * lands exactly on a read with no redundant check behind it. Rejections whose real message
 * necessarily differs from the mutant's are checked by message (expect_reject_msg), not just by
 * NULL, for the same reason. */

static void expect_reject_msg(const buf *b, const char *must_contain, const char *what) {
    TR_CHECK(write_file(b->data, b->len) == 0);
    char err[256] = {0};
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    if (g != NULL) {
        fprintf(stderr, "%s: accepted, expected a message containing '%s'\n", what, must_contain);
        tr_gguf_close(g);
        TR_TEST_FAILED();
        return;
    }
    if (strstr(err, must_contain) == NULL) {
        fprintf(stderr, "%s: rejected with '%s', expected it to contain '%s'\n", what, err, must_contain);
        TR_TEST_FAILED();
    }
}

/* err given with err_len 0, or err_len given with err NULL: fail()'s and tr_gguf_open's own
 * "err && err_len" guards must both hold before either is touched (gguf.c:121,315). */
static void test_err_buffer(void) {
    buf b = {0};
    put_u32(&b, 0x46554746u);   /* bad magic: guaranteed to reach fail() */
    put_u32(&b, 3); put_u64(&b, 0); put_u64(&b, 0);
    TR_CHECK(write_file(b.data, b.len) == 0);
    tr_gguf *g = tr_gguf_open(path, NULL, 100);   /* err_len > 0 but err == NULL */
    TR_CHECK(g == NULL);
    tr_gguf_close(g);
    free(b.data);
}

/* An empty file: size 0, not negative -- the real rejection is running out of bytes reading the
 * magic, never "cannot get file size" (gguf.c:335). */
static void test_empty_file(void) {
    TR_CHECK(write_file(NULL, 0) == 0);
    char err[256] = {0};
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g == NULL);
    TR_CHECK(strstr(err, "cannot get file size") == NULL);
    TR_CHECK(strstr(err, "unexpected end of file") != NULL);
    tr_gguf_close(g);
}

/* n_kv/n_tensors exactly at the limit: allowed by the count check, so the real rejection (the
 * file has no room for even the first entry) always says "unexpected end of file", never
 * "... entries"/"... tensors" (gguf.c:345,346). */
static void test_count_limits(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 0);            /* n_tensors */
    put_u64(&b, 1ull << 20);   /* n_kv == MAX_KV exactly */
    expect_reject_msg(&b, "unexpected end of file", "n_kv == MAX_KV alone");
    free(b.data);

    buf t = {0};
    put_u32(&t, 0x46554747u); put_u32(&t, 3);
    put_u64(&t, 1ull << 20);   /* n_tensors == MAX_TENSORS exactly */
    put_u64(&t, 0);            /* n_kv */
    expect_reject_msg(&t, "unexpected end of file", "n_tensors == MAX_TENSORS alone");
    free(t.data);
}

/* is_signed's four types, each alone (gguf.c:439), and its ">= 0" boundary at exactly 0
 * (gguf.c:446); get_u32's boundary at exactly UINT32_MAX (gguf.c:452). */
static void test_int_types(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 0);
    size_t n_kv_at = put_u64(&b, 0);
    uint64_t n_kv = 0;
    int8_t i8 = 0;
    put_str(&b, "t.i8"); put_u32(&b, TR_GGUF_INT8); put(&b, &i8, 1); n_kv++;
    int64_t i64 = 9;
    put_str(&b, "t.i64"); put_u32(&b, TR_GGUF_INT64); put(&b, &i64, 8); n_kv++;
    uint64_t u32max = 0xFFFFFFFFull;
    put_str(&b, "t.u32max"); put_u32(&b, TR_GGUF_UINT64); put(&b, &u32max, 8); n_kv++;
    patch_u64(&b, n_kv_at, n_kv);
    pad_to(&b, 32);
    TR_CHECK(write_file(b.data, b.len) == 0);
    char err[256];
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g != NULL);
    if (g == NULL) fprintf(stderr, "test_int_types: rejected: %s\n", err);
    if (g != NULL) {
        uint64_t u = 12345;
        TR_CHECK(tr_gguf_get_u64(g, "t.i8", &u) == 0 && u == 0);
        u = 12345;
        TR_CHECK(tr_gguf_get_u64(g, "t.i64", &u) == 0 && u == 9);
        uint32_t u32 = 0;
        TR_CHECK(tr_gguf_get_u32(g, "t.u32max", &u32) == 0 && u32 == 0xFFFFFFFFu);
        tr_gguf_close(g);
    }
    free(b.data);
}

/* A nested array (element type ARRAY) alone, length 0: no bytes to mis-consume either way, so a
 * weakened `||` (gguf.c:218) would accept it outright instead of the file cascading into some
 * unrelated error. */
static void test_nested_array_alone(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 0); put_u64(&b, 1);
    put_str(&b, "w");
    put_u32(&b, TR_GGUF_ARRAY);
    put_u32(&b, TR_GGUF_ARRAY);   /* element type: itself an array */
    put_u64(&b, 0);               /* length 0 */
    pad_to(&b, 32);
    open_expect(&b, 0, "nested array alone, length 0");
    free(b.data);
}

/* MAX_ARRAY's exact boundary (gguf.c:219), told apart by the message, in a file of a few bytes: a
 * uint8 array of exactly MAX_ARRAY elements passes the length check and is refused by the next one
 * (its bytes are not in the file), one more element is refused by the length check itself. An
 * accepted MAX_ARRAY array would need 256 MiB on disk and in the arena for the same answer. */
#define MAX_ARRAY_LEN (1ull << 28)

static void test_max_array_boundary(void) {
    for (int over = 0; over < 2; over++) {
        buf b = {0};
        put_u32(&b, 0x46554747u); put_u32(&b, 3);
        put_u64(&b, 0); put_u64(&b, 1);
        put_str(&b, "w");
        put_u32(&b, TR_GGUF_ARRAY);
        put_u32(&b, TR_GGUF_UINT8);
        put_u64(&b, MAX_ARRAY_LEN + (uint64_t)over);
        pad_to(&b, 32);
        if (over) expect_reject_msg(&b, "array length", "uint8 array of MAX_ARRAY + 1");
        else expect_reject_msg(&b, "array longer than file", "uint8 array of MAX_ARRAY (past the length check)");
        free(b.data);
    }
}

/* MAX_STRING's exact boundary and one past it (gguf.c:166): both need real, NUL-free content, so
 * the string read either succeeds (boundary) or would have to be attempted at all (one past) --
 * a sparse/zero-filled file would trip the "string contains NUL" check for either mutant and
 * original alike, hiding the difference this is meant to show. */
#define MAX_STRING_LEN (64ull << 20)

static buf build_one_string_file(uint64_t len) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 0); put_u64(&b, 1);
    put_str(&b, "w");
    put_u32(&b, TR_GGUF_STRING);
    put_u64(&b, len);
    size_t at = b.len;
    if (b.len + len > b.cap) { b.cap = b.len + len; b.data = realloc(b.data, b.cap); TR_CHECK(b.data != NULL); }
    memset(b.data + at, 'a', (size_t)len);
    b.len += (size_t)len;
    return b;
}

static void test_max_string_boundary(void) {
    buf ok = build_one_string_file(MAX_STRING_LEN);
    pad_to(&ok, 32);
    TR_CHECK(write_file(ok.data, ok.len) == 0);
    char err[256] = {0};
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g != NULL);
    if (g == NULL) fprintf(stderr, "test_max_string_boundary: len==MAX_STRING rejected: %s\n", err);
    if (g != NULL) {
        const char *s = NULL;
        TR_CHECK(tr_gguf_get_str(g, "w", &s) == 0 && s != NULL && strlen(s) == MAX_STRING_LEN);
        tr_gguf_close(g);
    }
    free(ok.data);

    buf over = build_one_string_file(MAX_STRING_LEN + 1);
    pad_to(&over, 32);
    open_expect(&over, 0, "len == MAX_STRING + 1");
    free(over.data);
}

/* The array-longer-than-file boundary, exactly enough room: a string array whose per-element
 * minimum (8 bytes) exactly divides the remaining file (gguf.c:223), and a scalar array whose
 * byte size exactly equals the remaining file (gguf.c:232). One byte short of either is already
 * covered by test_corrupted/test_truncated; this is the other edge. The 16-byte key ends the
 * array's length field at byte 64, so 32 bytes of items end the file on the 32-byte alignment:
 * no padding after them, nothing left over for the check to round away. */
static void test_array_len_boundary(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 0); put_u64(&b, 1);
    char key16[17]; memset(key16, 'w', 16); key16[16] = 0;
    put_str(&b, key16);
    put_u32(&b, TR_GGUF_ARRAY);
    put_u32(&b, TR_GGUF_STRING);
    put_u64(&b, 4);
    TR_CHECK_EQ_INT(b.len, 64);
    for (int i = 0; i < 4; i++) put_u64(&b, 0);   /* 4 empty strings, 8 bytes each */
    TR_CHECK_EQ_INT(b.len, 96);   /* remaining at the check 96 - 64 = 32, exactly 4 x 8 */
    open_expect(&b, 1, "4 empty strings, exactly enough room");
    free(b.data);

    buf s = {0};
    put_u32(&s, 0x46554747u); put_u32(&s, 3);
    put_u64(&s, 0); put_u64(&s, 1);
    put_str(&s, key16);
    put_u32(&s, TR_GGUF_ARRAY);
    put_u32(&s, TR_GGUF_UINT8);
    put_u64(&s, 32);
    for (int i = 0; i < 32; i++) { uint8_t z = (uint8_t)i; put(&s, &z, 1); }
    TR_CHECK_EQ_INT(s.len, 96);   /* remaining at the check 96 - 64 = 32, exactly the array's bytes */
    open_expect(&s, 1, "32 uint8 elements, exactly enough room");
    free(s.data);
}

/* An array's element read fails with the file ending exactly at the array's own length field
 * (gguf.c:227,235): its item slots stay whatever the arena gave them (uninitialised for the
 * string array's pointers, unset for the scalar array's bytes) -- a swallowed failure must not
 * let the file complete anyway. The 16-byte key lands the truncation point on the default
 * 32-byte alignment, so nothing else is left for the reader to reject the file on. */
static void test_array_item_truncated(void) {
    char key16[17]; memset(key16, 'w', 16); key16[16] = 0;

    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 0); put_u64(&b, 1);
    put_str(&b, key16);
    put_u32(&b, TR_GGUF_ARRAY);
    put_u32(&b, TR_GGUF_STRING);
    put_u64(&b, 2);
    TR_CHECK_EQ_INT(b.len, 64);
    open_expect(&b, 0, "string array, items truncated");
    free(b.data);

    buf s = {0};
    put_u32(&s, 0x46554747u); put_u32(&s, 3);
    put_u64(&s, 0); put_u64(&s, 1);
    put_str(&s, key16);
    put_u32(&s, TR_GGUF_ARRAY);
    put_u32(&s, TR_GGUF_UINT8);
    put_u64(&s, 5);
    TR_CHECK_EQ_INT(s.len, 64);
    open_expect(&s, 0, "scalar array, bytes truncated");
    free(s.data);
}

/* A scalar value's byte is missing entirely (gguf.c:189): rd_scalar's local buffer is
 * zero-initialised, so a swallowed failure reads deterministically as 0 -- the file must still be
 * refused, not accepted with a bogus value. The 28-byte key lands the missing byte exactly on the
 * default alignment, so nothing else is left to reject the file on. */
static void test_scalar_value_truncated(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 0); put_u64(&b, 1);
    char key28[29]; memset(key28, 'w', 28); key28[28] = 0;
    put_str(&b, key28);
    put_u32(&b, TR_GGUF_UINT8);
    TR_CHECK_EQ_INT(b.len, 64);
    open_expect(&b, 0, "scalar value byte missing");
    free(b.data);
}

/* n_dims == 0 alone, and n_dims > MAX_DIMS alone (gguf.c:291): purpose-built one-tensor files
 * laid out exactly as the reader would consume them if the check did not fire, so a weakened `||`
 * accepts the file instead of the layout drifting into an unrelated rejection. */
static void test_ndims_alone(void) {
    buf z = {0};
    put_u32(&z, 0x46554747u); put_u32(&z, 3);
    put_u64(&z, 1); put_u64(&z, 0);
    put_str(&z, "t");
    put_u32(&z, 0);              /* n_dims = 0: alone, not > MAX_DIMS */
    put_u32(&z, TR_TYPE_F32);
    put_u64(&z, 0);
    pad_to(&z, 32);
    float v = 1.0f; put(&z, &v, 4);
    open_expect(&z, 0, "tensor with 0 dims, alone");
    free(z.data);

    buf o = {0};
    put_u32(&o, 0x46554747u); put_u32(&o, 3);
    put_u64(&o, 1); put_u64(&o, 0);
    put_str(&o, "t");
    put_u32(&o, 5);              /* n_dims = 5: alone, not == 0 */
    put_u64(&o, 2); put_u64(&o, 1); put_u64(&o, 1); put_u64(&o, 1);  /* only 4 are ever read */
    put_u32(&o, TR_TYPE_F32);
    put_u64(&o, 0);
    pad_to(&o, 32);
    float vv[2] = {1.0f, 2.0f}; put(&o, vv, 8);
    open_expect(&o, 0, "tensor with 5 dims, alone");
    free(o.data);
}

/* A dimension of 0, alone (gguf.c:297): the real code refuses it before the element-count
 * overflow check that follows would divide by it. */
static void test_ne_zero_alone(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 1); put_u64(&b, 0);
    put_str(&b, "t");
    put_u32(&b, 1);
    put_u64(&b, 0);              /* ne[0] = 0: alone, not > the per-dimension limit */
    put_u32(&b, TR_TYPE_F32);
    put_u64(&b, 0);
    pad_to(&b, 32);
    open_expect(&b, 0, "tensor dim 0 == 0, alone");
    free(b.data);
}

/* A dimension over the per-dimension limit, alone (gguf.c:297): the real message names the huge
 * dimension; a weakened `||` would only reject the file later, when its (equally huge) declared
 * byte size cannot fit the file -- a different message. */
static void test_ne_huge_alone(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 1); put_u64(&b, 0);
    put_str(&b, "t");
    put_u32(&b, 1);
    put_u64(&b, (1ull << 40) + 1);   /* one past the limit: alone, not == 0 */
    put_u32(&b, TR_TYPE_F32);
    put_u64(&b, 0);
    pad_to(&b, 32);
    expect_reject_msg(&b, "dim 0 =", "tensor dim 0 over the limit, alone");
    free(b.data);
}

/* The element-count overflow check's exact boundary (gguf.c:298): ne[0] chosen so the running
 * product lands exactly on floor(UINT64_MAX / ne[1]) -- the real code does not overflow multiplying
 * them, and rejects on the next check, the byte size (2^64 - 2^40 elements of 4 bytes); a check
 * weakened to >= would say "element count overflows". */
static void test_elem_count_boundary(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 1); put_u64(&b, 0);
    put_str(&b, "t");
    put_u32(&b, 2);
    put_u64(&b, 16777215ull);      /* ne[0] == floor(UINT64_MAX / ne[1]) */
    put_u64(&b, 1ull << 40);       /* ne[1]: the largest a single dimension may be */
    put_u32(&b, TR_TYPE_F32);
    put_u64(&b, 0);
    pad_to(&b, 32);
    expect_reject_msg(&b, "byte size overflows", "element count boundary (the element count must not overflow)");
    free(b.data);
}

/* The byte-size overflow check's exact boundary (gguf.c:309): ne[0] * ne[1] == 2^62 - 1, so
 * blocks lands exactly on floor(UINT64_MAX / block_bytes) for f32 (block_bytes 4). The real code
 * does not overflow computing n_bytes, and only rejects later because 2^64-ish bytes cannot fit
 * this tiny file; a check weakened to >= would say "byte size overflows". */
static void test_byte_size_boundary(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 1); put_u64(&b, 0);
    put_str(&b, "t");
    put_u32(&b, 2);
    put_u64(&b, 2147483647ull);    /* (2^31-1) * (2^31+1) == 2^62-1 == floor(UINT64_MAX/4) */
    put_u64(&b, 2147483649ull);
    put_u32(&b, TR_TYPE_F32);
    put_u64(&b, 0);
    pad_to(&b, 32);
    expect_reject_msg(&b, "outside the file", "byte size boundary (the byte size must not overflow)");
    free(b.data);
}

/* general.alignment present but not readable as a u32 (gguf.c:365): the check's first clause must
 * reject it without the later clauses ever reading the (then unset) local value. */
static void test_alignment_wrong_type(void) {
    buf b = {0};
    put_u32(&b, 0x46554747u); put_u32(&b, 3);
    put_u64(&b, 0); put_u64(&b, 1);
    put_str(&b, "general.alignment");
    put_u32(&b, TR_GGUF_STRING);   /* wrong type: get_u32 fails before any range check */
    put_str(&b, "x");
    pad_to(&b, 32);
    open_expect(&b, 0, "alignment key present but not a u32");
    free(b.data);
}

/* general.alignment 0 and 3, alone (gguf.c:365), by message: with no tensor after them nothing
 * else rejects the file, so a weakened clause either accepts it or divides by zero. */
static void test_alignment_values(void) {
    static const uint32_t bad[2] = {0, 3};
    for (int i = 0; i < 2; i++) {
        buf b = {0};
        put_u32(&b, 0x46554747u); put_u32(&b, 3);
        put_u64(&b, 0); put_u64(&b, 1);
        put_str(&b, "general.alignment");
        put_u32(&b, TR_GGUF_UINT32);
        put_u32(&b, bad[i]);
        pad_to(&b, 32);
        expect_reject_msg(&b, "must be a power of two", bad[i] == 0 ? "alignment 0, alone" : "alignment 3, alone");
        free(b.data);
    }
}

/* A string array cut inside its last item (gguf.c:227): the item's length is read and refused
 * against the bytes left ("string length 2 out of range"); a swallowed failure would go on to the
 * next key and fail there on "unexpected end of file" instead. */
static void test_string_item_cut(const buf *valid, const layout *L) {
    size_t bc = L->arr_len + 8 + (8 + 1);   /* test.words: after its length, "a", then "bc" */
    buf b = {malloc(bc + 8 + 1), bc + 8 + 1, bc + 8 + 1};
    memcpy(b.data, valid->data, b.len);     /* bc's length and one of its two bytes */
    TR_CHECK(b.data[bc] == 2 && b.data[bc + 8] == 'b');
    expect_reject_msg(&b, "string length 2 out of range", "test.words cut inside \"bc\"");
    free(b.data);
}

/* The last key cut inside its value type, and inside an array's length, with the file ending on
 * the 32-byte alignment right there (gguf.c:212,219). A read that failed and was taken for a
 * success would leave nothing else to read: no tensor, the data section starting at the file's
 * end, and the file would open. A 29-byte key ends at 61, 3 bytes short of a type; a 17-byte key
 * of type array with its element type ends at 57, 7 bytes short of the length. */
static void test_kv_cut_at_alignment(void) {
    for (int arr = 0; arr < 2; arr++) {
        buf b = {0};
        put_u32(&b, 0x46554747u); put_u32(&b, 3);
        put_u64(&b, 0); put_u64(&b, 1);
        char key[30];
        memset(key, 'k', sizeof key);
        key[arr ? 17 : 29] = 0;
        put_str(&b, key);
        if (arr) { put_u32(&b, TR_GGUF_ARRAY); put_u32(&b, TR_GGUF_UINT8); }
        TR_CHECK_EQ_INT(b.len, arr ? 57 : 61);
        pad_to(&b, 32);
        TR_CHECK_EQ_INT(b.len, 64);
        expect_reject_msg(&b, arr ? "gguf @57: unexpected end of file" : "gguf @61: unexpected end of file",
                          arr ? "array length cut at the alignment" : "value type cut at the alignment");
        free(b.data);
    }
}

/* The tensor data read fails after the file shrinks on disk post-open (gguf.c:488): tr_gguf_read
 * must propagate a real pread failure, not report success after a chunk it never got. On Windows
 * the reader's handle lets no writer in: the shrink itself is refused and the read gets its bytes. */
static void test_read_after_shrink(const buf *valid) {
    TR_CHECK(write_file(valid->data, valid->len) == 0);
    char err[256];
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g != NULL);
    if (g == NULL) return;
    const tr_gguf_tensor *t = tr_gguf_find_tensor(g, "b");
    TR_CHECK(t != NULL);
    if (t != NULL) {
        uint8_t buf68[68];
#ifdef _WIN32
        TR_CHECK(write_file(valid->data, 4) != 0);
        TR_CHECK(tr_gguf_read(g, t, buf68) == 0 && memcmp(buf68, valid->data + valid->len - 68, 68) == 0);
#else
        TR_CHECK(write_file(valid->data, 4) == 0);   /* same path, same inode: shrunk under the open fd */
        TR_CHECK(tr_gguf_read(g, t, buf68) == -1);
#endif
    }
    tr_gguf_close(g);
}

int main(int argc, char **argv) {
    char dir[480];
    tr_test_tmpdir(argc > 0 ? argv[0] : "", dir, sizeof dir);
    snprintf(path, sizeof path, "%s/test_gguf_tmp.gguf", dir);

    layout L;
    buf valid = build_valid(&L);
    test_valid(&valid);
    test_truncated(&valid, &L);
    test_corrupted(&L, &valid);
    test_fuzz(&valid);
    test_more();
    test_k_quant(TR_TYPE_Q4_K, "not a multiple of q4_k block 256", 144);
    test_k_quant(TR_TYPE_Q6_K, "not a multiple of q6_k block 256", 210);
    /* the cases written against mutate_auto's survivors (2026-09-24) */
    test_err_buffer();
    test_empty_file();
    test_count_limits();
    test_int_types();
    test_nested_array_alone();
    test_max_array_boundary();
    test_max_string_boundary();
    test_array_len_boundary();
    test_array_item_truncated();
    test_scalar_value_truncated();
    test_ndims_alone();
    test_ne_zero_alone();
    test_ne_huge_alone();
    test_elem_count_boundary();
    test_byte_size_boundary();
    test_alignment_wrong_type();
    test_alignment_values();
    test_string_item_cut(&valid, &L);
    test_kv_cut_at_alignment();
    test_read_after_shrink(&valid);
    remove(path);
    free(valid.data);
    TR_TEST_EXIT();
}
