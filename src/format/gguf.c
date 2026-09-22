/* gguf.c — GGUF v2/v3 reader.
 * The type table is derived from ds4 8db1d1d ds4.c gguf_types[] (MIT), modified:
 * adds tq1_0/tq2_0, fixes iq4_nl (32/18) and iq1_s (256/50) to ggml's sizes, exposes it through
 * tr_type_get. The parser is new. */
#include "gguf.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define GGUF_MAGIC 0x46554747u              /* "GGUF" read as little-endian u32 */
#define MAX_KV        (1u << 20)
#define MAX_TENSORS   (1u << 20)
#define MAX_STRING    (64ull << 20)         /* one metadata string */
#define MAX_ARRAY     (1ull << 28)          /* elements in one metadata array */
#define READ_CHUNK    (1u << 20)

static const tr_type_info type_table[TR_TYPE_COUNT] = {
    [TR_TYPE_F32]     = {"f32",      1,   4},
    [TR_TYPE_F16]     = {"f16",      1,   2},
    [TR_TYPE_Q4_0]    = {"q4_0",    32,  18},
    [TR_TYPE_Q4_1]    = {"q4_1",    32,  20},
    [TR_TYPE_Q5_0]    = {"q5_0",    32,  22},
    [TR_TYPE_Q5_1]    = {"q5_1",    32,  24},
    [TR_TYPE_Q8_0]    = {"q8_0",    32,  34},
    [TR_TYPE_Q8_1]    = {"q8_1",    32,  40},
    [TR_TYPE_Q2_K]    = {"q2_k",   256,  84},
    [TR_TYPE_Q3_K]    = {"q3_k",   256, 110},
    [TR_TYPE_Q4_K]    = {"q4_k",   256, 144},
    [TR_TYPE_Q5_K]    = {"q5_k",   256, 176},
    [TR_TYPE_Q6_K]    = {"q6_k",   256, 210},
    [TR_TYPE_Q8_K]    = {"q8_k",   256, 292},
    [TR_TYPE_IQ2_XXS] = {"iq2_xxs", 256,  66},
    [TR_TYPE_IQ2_XS]  = {"iq2_xs", 256,  74},
    [TR_TYPE_IQ3_XXS] = {"iq3_xxs", 256,  98},
    [TR_TYPE_IQ1_S]   = {"iq1_s",  256,  50},
    [TR_TYPE_IQ4_NL]  = {"iq4_nl",  32,  18},
    [TR_TYPE_IQ3_S]   = {"iq3_s",  256, 110},
    [TR_TYPE_IQ2_S]   = {"iq2_s",  256,  82},
    [TR_TYPE_IQ4_XS]  = {"iq4_xs", 256, 136},
    [TR_TYPE_I8]      = {"i8",       1,   1},
    [TR_TYPE_I16]     = {"i16",      1,   2},
    [TR_TYPE_I32]     = {"i32",      1,   4},
    [TR_TYPE_I64]     = {"i64",      1,   8},
    [TR_TYPE_F64]     = {"f64",      1,   8},
    [TR_TYPE_IQ1_M]   = {"iq1_m",  256,  56},
    [TR_TYPE_BF16]    = {"bf16",     1,   2},
    [TR_TYPE_TQ1_0]   = {"tq1_0",  256,  54},
    [TR_TYPE_TQ2_0]   = {"tq2_0",  256,  66},
    [TR_TYPE_MXFP4]   = {"mxfp4",   32,  17},
};

const tr_type_info *tr_type_get(uint32_t type) {
    if (type >= TR_TYPE_COUNT || type_table[type].name == NULL) return NULL;
    return &type_table[type];
}

/* ------------------------------------------------------------------ arena */

typedef struct arena_block {
    struct arena_block *next;
    size_t used, cap;
    /* data follows */
} arena_block;

typedef struct { arena_block *head; } arena;

/* Data starts past the header rounded up to 16: right after the 24-byte header it would sit at
 * 8 mod 16 (docs/LESSONS.md #105). 16-byte alignment holds where malloc gives 16. */
#define ARENA_HDR ((sizeof(arena_block) + 15) & ~(size_t)15)

static void *arena_alloc(arena *a, size_t n) {
    size_t align = 16;
    n = (n + align - 1) & ~(align - 1);
    if (a->head == NULL || a->head->cap - a->head->used < n) {
        size_t cap = n > READ_CHUNK ? n : READ_CHUNK;
        arena_block *b = malloc(ARENA_HDR + cap);
        if (b == NULL) return NULL;
        b->next = a->head; b->used = 0; b->cap = cap;
        a->head = b;
    }
    void *p = (char *)a->head + ARENA_HDR + a->head->used;
    a->head->used += n;
    return p;
}

static void arena_free(arena *a) {
    arena_block *b = a->head;
    while (b) { arena_block *n = b->next; free(b); b = n; }
    a->head = NULL;
}

/* ------------------------------------------------------------------ reader */

struct tr_gguf {
    tr_file *file;
    uint64_t file_size;
    uint32_t version;
    uint64_t n_kv, n_tensors;
    tr_gguf_kv *kv;
    tr_gguf_tensor *tensors;
    uint64_t alignment, data_offset;
    uint64_t *kv_index, *tensor_index;  /* open addressing: slot = i + 1, 0 = empty */
    uint64_t kv_cap, tensor_cap;
    arena mem;
};

typedef struct {
    const tr_file *f;
    uint64_t file_size;
    uint64_t pos;               /* absolute position of the next byte */
    uint8_t buf[READ_CHUNK];
    uint64_t buf_start;         /* absolute offset of buf[0] */
    size_t buf_len;
    char *err;
    size_t err_len;
} reader;

static int fail(reader *r, const char *fmt, ...) {
    if (r->err && r->err_len) {
        int k = snprintf(r->err, r->err_len, "gguf @%llu: ", (unsigned long long)r->pos);
        if (k < 0) k = 0;
        if ((size_t)k < r->err_len) {
            va_list ap; va_start(ap, fmt);
            vsnprintf(r->err + k, r->err_len - (size_t)k, fmt, ap);
            va_end(ap);
        }
    }
    return -1;
}

static int rd(reader *r, void *dst, uint64_t n) {
    if (n > r->file_size || r->pos > r->file_size - n) return fail(r, "unexpected end of file");
    uint8_t *out = dst;
    while (n > 0) {
        if (r->pos < r->buf_start || r->pos >= r->buf_start + r->buf_len) {
            uint64_t left = r->file_size - r->pos;
            size_t want = left < READ_CHUNK ? (size_t)left : READ_CHUNK;
            if (n >= READ_CHUNK) {                  /* large payload: read straight into dst */
                size_t direct = (size_t)(n - n % READ_CHUNK);
                if (tr_file_pread(r->f, out, direct, r->pos) != 0) return fail(r, "read error");
                out += direct; r->pos += direct; n -= direct;
                continue;
            }
            if (tr_file_pread(r->f, r->buf, want, r->pos) != 0) return fail(r, "read error");
            r->buf_start = r->pos; r->buf_len = want;
        }
        size_t off = (size_t)(r->pos - r->buf_start);
        size_t take = r->buf_len - off;
        if (take > n) take = (size_t)n;
        memcpy(out, r->buf + off, take);
        out += take; r->pos += take; n -= take;
    }
    return 0;
}

static int rd_u32(reader *r, uint32_t *v) { return rd(r, v, 4); }
static int rd_u64(reader *r, uint64_t *v) { return rd(r, v, 8); }

/* Reads a GGUF string into the arena, NUL-terminated. Interior NULs are rejected:
 * every string here is used as a C string (keys, names, tokens, templates). */
static int rd_str(reader *r, arena *a, char **out) {
    uint64_t len;
    if (rd_u64(r, &len) != 0) return -1;
    if (len > MAX_STRING || len > r->file_size - r->pos) return fail(r, "string length %llu out of range", (unsigned long long)len);
    char *s = arena_alloc(a, (size_t)len + 1);
    if (s == NULL) return fail(r, "out of memory");
    if (rd(r, s, len) != 0) return -1;
    if (memchr(s, 0, (size_t)len) != NULL) return fail(r, "string contains NUL");
    s[len] = 0;
    *out = s;
    return 0;
}

static int scalar_size(tr_gguf_vtype t) {
    switch (t) {
    case TR_GGUF_UINT8: case TR_GGUF_INT8: case TR_GGUF_BOOL: return 1;
    case TR_GGUF_UINT16: case TR_GGUF_INT16: return 2;
    case TR_GGUF_UINT32: case TR_GGUF_INT32: case TR_GGUF_FLOAT32: return 4;
    case TR_GGUF_UINT64: case TR_GGUF_INT64: case TR_GGUF_FLOAT64: return 8;
    default: return 0;
    }
}

static int rd_scalar(reader *r, tr_gguf_vtype t, tr_gguf_kv *kv) {
    uint8_t b[8] = {0};
    int n = scalar_size(t);
    if (rd(r, b, (uint64_t)n) != 0) return -1;
    switch (t) {
    case TR_GGUF_UINT8:  kv->v.u = b[0]; break;
    case TR_GGUF_BOOL:   if (b[0] > 1) return fail(r, "bool value %u", b[0]); kv->v.u = b[0]; break;
    case TR_GGUF_INT8:   kv->v.i = (int8_t)b[0]; break;
    case TR_GGUF_UINT16: { uint16_t x; memcpy(&x, b, 2); kv->v.u = x; break; }
    case TR_GGUF_INT16:  { int16_t x;  memcpy(&x, b, 2); kv->v.i = x; break; }
    case TR_GGUF_UINT32: { uint32_t x; memcpy(&x, b, 4); kv->v.u = x; break; }
    case TR_GGUF_INT32:  { int32_t x;  memcpy(&x, b, 4); kv->v.i = x; break; }
    case TR_GGUF_UINT64: { uint64_t x; memcpy(&x, b, 8); kv->v.u = x; break; }
    case TR_GGUF_INT64:  { int64_t x;  memcpy(&x, b, 8); kv->v.i = x; break; }
    case TR_GGUF_FLOAT32:{ float x;    memcpy(&x, b, 4); kv->v.f = x; break; }
    case TR_GGUF_FLOAT64:{ double x;   memcpy(&x, b, 8); kv->v.f = x; break; }
    default: return fail(r, "not a scalar type %d", (int)t);
    }
    return 0;
}

static int rd_kv(reader *r, arena *a, tr_gguf_kv *kv) {
    uint32_t t;
    if (rd_str(r, a, &kv->key) != 0) return -1;
    if (rd_u32(r, &t) != 0) return -1;
    if (t > TR_GGUF_FLOAT64) return fail(r, "key %s: value type %u", kv->key, t);
    kv->type = (tr_gguf_vtype)t;
    if (kv->type == TR_GGUF_STRING) return rd_str(r, a, &kv->v.s);
    if (kv->type != TR_GGUF_ARRAY) return rd_scalar(r, kv->type, kv);

    uint32_t et; uint64_t len;
    if (rd_u32(r, &et) != 0 || rd_u64(r, &len) != 0) return -1;
    if (et > TR_GGUF_FLOAT64 || et == TR_GGUF_ARRAY) return fail(r, "key %s: array element type %u", kv->key, et);
    if (len > MAX_ARRAY) return fail(r, "key %s: array length %llu", kv->key, (unsigned long long)len);
    kv->arr_type = (tr_gguf_vtype)et;
    kv->arr_len = len;
    if (kv->arr_type == TR_GGUF_STRING) {
        if (len > (r->file_size - r->pos) / 8) return fail(r, "key %s: array longer than file", kv->key);
        char **items = arena_alloc(a, (size_t)len * sizeof(char *));
        if (items == NULL && len) return fail(r, "out of memory");
        for (uint64_t i = 0; i < len; i++)
            if (rd_str(r, a, &items[i]) != 0) return -1;
        kv->arr = items;
        return 0;
    }
    uint64_t bytes = len * (uint64_t)scalar_size(kv->arr_type);
    if (bytes > r->file_size - r->pos) return fail(r, "key %s: array longer than file", kv->key);
    kv->arr = arena_alloc(a, (size_t)bytes + 1);
    if (kv->arr == NULL) return fail(r, "out of memory");
    if (rd(r, kv->arr, bytes) != 0) return -1;
    if (kv->arr_type == TR_GGUF_BOOL)
        for (uint64_t i = 0; i < len; i++)
            if (((uint8_t *)kv->arr)[i] > 1) return fail(r, "key %s: bool array value", kv->key);
    return 0;
}

/* ------------------------------------------------------------------ index */

static uint64_t hash_str(const char *s) {
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (uint8_t)*s; h *= 1099511628211ull; }
    return h;
}

static uint64_t cap_for(uint64_t n) {
    uint64_t c = 16;
    while (c < n * 2) c <<= 1;
    return c;
}

/* Inserts names[i] for all i; returns -1 on a duplicate (dup_out = its index). */
static int build_index(uint64_t *slots, uint64_t cap, uint64_t n,
                       const char *(*name_at)(const tr_gguf *, uint64_t), const tr_gguf *g, uint64_t *dup_out) {
    for (uint64_t i = 0; i < n; i++) {
        const char *name = name_at(g, i);
        uint64_t s = hash_str(name) & (cap - 1);
        while (slots[s]) {
            if (strcmp(name_at(g, slots[s] - 1), name) == 0) { *dup_out = i; return -1; }
            s = (s + 1) & (cap - 1);
        }
        slots[s] = i + 1;
    }
    return 0;
}

static const char *kv_name(const tr_gguf *g, uint64_t i) { return g->kv[i].key; }
static const char *tensor_name(const tr_gguf *g, uint64_t i) { return g->tensors[i].name; }

static uint64_t lookup(const uint64_t *slots, uint64_t cap, const char *name,
                       const char *(*name_at)(const tr_gguf *, uint64_t), const tr_gguf *g) {
    if (slots == NULL || name == NULL) return 0;
    uint64_t s = hash_str(name) & (cap - 1);
    while (slots[s]) {
        if (strcmp(name_at(g, slots[s] - 1), name) == 0) return slots[s];
        s = (s + 1) & (cap - 1);
    }
    return 0;
}

/* ------------------------------------------------------------------ open */

static int rd_tensor_info(reader *r, arena *a, tr_gguf_tensor *t) {
    uint32_t type;
    if (rd_str(r, a, &t->name) != 0) return -1;
    if (rd_u32(r, &t->n_dims) != 0) return -1;
    if (t->n_dims == 0 || t->n_dims > TR_GGUF_MAX_DIMS) return fail(r, "tensor %s: %u dims", t->name, t->n_dims);
    uint64_t n = 1;
    for (uint32_t d = 0; d < TR_GGUF_MAX_DIMS; d++) {
        t->ne[d] = 1;
        if (d >= t->n_dims) continue;
        if (rd_u64(r, &t->ne[d]) != 0) return -1;
        if (t->ne[d] == 0 || t->ne[d] > (1ull << 40)) return fail(r, "tensor %s: dim %u = %llu", t->name, d, (unsigned long long)t->ne[d]);
        if (n > UINT64_MAX / t->ne[d]) return fail(r, "tensor %s: element count overflows", t->name);
        n *= t->ne[d];
    }
    if (rd_u32(r, &type) != 0 || rd_u64(r, &t->offset) != 0) return -1;
    const tr_type_info *ti = tr_type_get(type);
    if (ti == NULL) return fail(r, "tensor %s: unknown type %u", t->name, type);
    if (t->ne[0] % ti->block_elems != 0) return fail(r, "tensor %s: row length %llu not a multiple of %s block %u",
                                                     t->name, (unsigned long long)t->ne[0], ti->name, ti->block_elems);
    t->type = (tr_type)type;
    t->n_elems = n;
    uint64_t blocks = n / ti->block_elems;
    if (blocks > UINT64_MAX / ti->block_bytes) return fail(r, "tensor %s: byte size overflows", t->name);
    t->n_bytes = blocks * ti->block_bytes;
    return 0;
}

tr_gguf *tr_gguf_open(const char *path, char *err, size_t err_len) {
    if (err && err_len) err[0] = 0;
    uint16_t probe = 1;
    if (*(uint8_t *)&probe != 1) {
        if (err && err_len) snprintf(err, err_len, "big-endian hosts are not supported");
        return NULL;
    }
    tr_gguf *g = calloc(1, sizeof *g);
    reader *r = malloc(sizeof *r);
    if (g == NULL || r == NULL) {
        free(g); free(r);
        if (err && err_len) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    memset(r, 0, offsetof(reader, buf));
    r->err = err; r->err_len = err_len;
    r->buf_start = 0; r->buf_len = 0;

    g->file = tr_file_open(path, err, err_len);
    if (g->file == NULL) goto bad;
    int64_t size = tr_file_size(g->file);
    if (size < 0) { fail(r, "cannot get file size"); goto bad; }
    g->file_size = (uint64_t)size;
    r->f = g->file; r->file_size = g->file_size;

    uint32_t magic;
    if (rd_u32(r, &magic) != 0) goto bad;
    if (magic != GGUF_MAGIC) { fail(r, "not a GGUF file (magic %08x)", magic); goto bad; }
    if (rd_u32(r, &g->version) != 0) goto bad;
    if (g->version != 2 && g->version != 3) { fail(r, "GGUF version %u not supported (2 or 3)", g->version); goto bad; }
    if (rd_u64(r, &g->n_tensors) != 0 || rd_u64(r, &g->n_kv) != 0) goto bad;
    if (g->n_kv > MAX_KV) { fail(r, "%llu metadata entries", (unsigned long long)g->n_kv); goto bad; }
    if (g->n_tensors > MAX_TENSORS) { fail(r, "%llu tensors", (unsigned long long)g->n_tensors); goto bad; }

    g->kv = calloc(g->n_kv ? g->n_kv : 1, sizeof *g->kv);
    g->tensors = calloc(g->n_tensors ? g->n_tensors : 1, sizeof *g->tensors);
    if (g->kv == NULL || g->tensors == NULL) { fail(r, "out of memory"); goto bad; }

    for (uint64_t i = 0; i < g->n_kv; i++)
        if (rd_kv(r, &g->mem, &g->kv[i]) != 0) goto bad;

    uint64_t dup;
    g->kv_cap = cap_for(g->n_kv);
    g->kv_index = calloc(g->kv_cap, sizeof(uint64_t));
    if (g->kv_index == NULL) { fail(r, "out of memory"); goto bad; }
    if (build_index(g->kv_index, g->kv_cap, g->n_kv, kv_name, g, &dup) != 0) { fail(r, "duplicate key %s", g->kv[dup].key); goto bad; }

    g->alignment = 32;
    const tr_gguf_kv *al = tr_gguf_find_kv(g, "general.alignment");
    if (al != NULL) {
        uint32_t a32;
        if (tr_gguf_get_u32(g, "general.alignment", &a32) != 0 || a32 == 0 || (a32 & (a32 - 1)) != 0 || a32 > (1u << 16)) {
            fail(r, "general.alignment must be a power of two <= 65536"); goto bad;
        }
        g->alignment = a32;
    }

    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (rd_tensor_info(r, &g->mem, &g->tensors[i]) != 0) goto bad;

    g->tensor_cap = cap_for(g->n_tensors);
    g->tensor_index = calloc(g->tensor_cap, sizeof(uint64_t));
    if (g->tensor_index == NULL) { fail(r, "out of memory"); goto bad; }
    if (build_index(g->tensor_index, g->tensor_cap, g->n_tensors, tensor_name, g, &dup) != 0) {
        fail(r, "duplicate tensor %s", g->tensors[dup].name); goto bad;
    }

    uint64_t pad = (g->alignment - r->pos % g->alignment) % g->alignment;
    g->data_offset = r->pos + pad;
    if (g->data_offset > g->file_size) { fail(r, "data section starts past end of file"); goto bad; }
    uint64_t data_size = g->file_size - g->data_offset;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        tr_gguf_tensor *t = &g->tensors[i];
        if (t->offset % g->alignment != 0) { fail(r, "tensor %s: offset %llu not aligned to %llu", t->name,
                                                  (unsigned long long)t->offset, (unsigned long long)g->alignment); goto bad; }
        if (t->offset > data_size || t->n_bytes > data_size - t->offset) {
            fail(r, "tensor %s: data [%llu, +%llu) outside the file", t->name,
                 (unsigned long long)t->offset, (unsigned long long)t->n_bytes);
            goto bad;
        }
        t->offset += g->data_offset;
    }
    free(r);
    return g;

bad:
    free(r);
    tr_gguf_close(g);
    return NULL;
}

void tr_gguf_close(tr_gguf *g) {
    if (g == NULL) return;
    if (g->file) tr_file_close(g->file);
    free(g->kv);
    free(g->tensors);
    free(g->kv_index);
    free(g->tensor_index);
    arena_free(&g->mem);
    free(g);
}

/* ------------------------------------------------------------------ queries */

uint32_t tr_gguf_version(const tr_gguf *g) { return g->version; }
uint64_t tr_gguf_kv_count(const tr_gguf *g) { return g->n_kv; }
const tr_gguf_kv *tr_gguf_kv_at(const tr_gguf *g, uint64_t i) { return i < g->n_kv ? &g->kv[i] : NULL; }
uint64_t tr_gguf_tensor_count(const tr_gguf *g) { return g->n_tensors; }
const tr_gguf_tensor *tr_gguf_tensor_at(const tr_gguf *g, uint64_t i) { return i < g->n_tensors ? &g->tensors[i] : NULL; }
const tr_file *tr_gguf_file(const tr_gguf *g) { return g->file; }

const tr_gguf_kv *tr_gguf_find_kv(const tr_gguf *g, const char *key) {
    uint64_t s = lookup(g->kv_index, g->kv_cap, key, kv_name, g);
    return s ? &g->kv[s - 1] : NULL;
}

const tr_gguf_tensor *tr_gguf_find_tensor(const tr_gguf *g, const char *name) {
    uint64_t s = lookup(g->tensor_index, g->tensor_cap, name, tensor_name, g);
    return s ? &g->tensors[s - 1] : NULL;
}

static int is_unsigned(tr_gguf_vtype t) {
    return t == TR_GGUF_UINT8 || t == TR_GGUF_UINT16 || t == TR_GGUF_UINT32 || t == TR_GGUF_UINT64;
}
static int is_signed(tr_gguf_vtype t) {
    return t == TR_GGUF_INT8 || t == TR_GGUF_INT16 || t == TR_GGUF_INT32 || t == TR_GGUF_INT64;
}

int tr_gguf_get_u64(const tr_gguf *g, const char *key, uint64_t *out) {
    const tr_gguf_kv *kv = tr_gguf_find_kv(g, key);
    if (kv == NULL) return -1;
    if (is_unsigned(kv->type)) { *out = kv->v.u; return 0; }
    if (is_signed(kv->type) && kv->v.i >= 0) { *out = (uint64_t)kv->v.i; return 0; }
    return -1;
}

int tr_gguf_get_u32(const tr_gguf *g, const char *key, uint32_t *out) {
    uint64_t v;
    if (tr_gguf_get_u64(g, key, &v) != 0 || v > UINT32_MAX) return -1;
    *out = (uint32_t)v;
    return 0;
}

int tr_gguf_get_f32(const tr_gguf *g, const char *key, float *out) {
    const tr_gguf_kv *kv = tr_gguf_find_kv(g, key);
    if (kv == NULL || (kv->type != TR_GGUF_FLOAT32 && kv->type != TR_GGUF_FLOAT64)) return -1;
    *out = (float)kv->v.f;
    return 0;
}

int tr_gguf_get_bool(const tr_gguf *g, const char *key, int *out) {
    const tr_gguf_kv *kv = tr_gguf_find_kv(g, key);
    if (kv == NULL || kv->type != TR_GGUF_BOOL) return -1;
    *out = (int)kv->v.u;
    return 0;
}

int tr_gguf_get_str(const tr_gguf *g, const char *key, const char **out) {
    const tr_gguf_kv *kv = tr_gguf_find_kv(g, key);
    if (kv == NULL || kv->type != TR_GGUF_STRING) return -1;
    *out = kv->v.s;
    return 0;
}

int tr_gguf_read(const tr_gguf *g, const tr_gguf_tensor *t, void *buf) {
    return tr_gguf_read_range(g, t, 0, buf, t->n_bytes);
}

int tr_gguf_read_range(const tr_gguf *g, const tr_gguf_tensor *t, uint64_t offset, void *buf, uint64_t n) {
    if (offset > t->n_bytes || n > t->n_bytes - offset) return -1;
    uint64_t done = 0;
    while (done < n) {                          /* size_t may be 32-bit: read in chunks */
        uint64_t step = n - done;
        if (step > (1ull << 30)) step = 1ull << 30;
        if (tr_file_pread(g->file, (char *)buf + done, (size_t)step, t->offset + offset + done) != 0) return -1;
        done += step;
    }
    return 0;
}
