/* tokenizer.c — byte-level BPE from GGUF metadata (see tokenizer.h).
 *
 * Load turns the file's strings into ids once: merges become a hash from a pair of
 * token ids to (rank, merged id), every token keeps the bytes it decodes to, and the
 * string-to-id table is dropped. Encoding then never looks at a string again. */
#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unicode.h"

/* ---- pre-tokenizer families ------------------------------------------------ */

typedef enum {
    /* GPT-2 ByteLevel regex:
     * 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ */
    PRE_GPT2
} pre_rules;

typedef struct {
    const char *pre;    /* tokenizer.ggml.pre */
    pre_rules rules;
    int nfc;            /* the family's tokenizer.json normalizes to NFC */
} family;

/* A family enters this table only with an oracle run against its tokenizer.json
 * (tools/tokenizer_oracle.py): the name alone does not say what HF does. */
static const family FAMILIES[] = {
    /* OLMo / OLMoE: GPT-NeoX tokenizer, NFC normalizer, ByteLevel without prefix space */
    {"olmo", PRE_GPT2, 1},
};

/* ---- the tokenizer --------------------------------------------------------- */

#define MAX_TOKENS ((size_t)1 << 24)
#define MAX_MERGES ((size_t)1 << 24)
#define EMPTY_KEY UINT64_MAX

struct tr_tokenizer {
    int32_t n_tokens;
    uint8_t *type;              /* [n_tokens] tr_token_type */
    char *bytes;                /* decoded bytes of every token, concatenated */
    uint32_t *off;              /* [n_tokens + 1] token i is bytes[off[i]..off[i+1]) */
    int32_t byte_token[256];    /* the token of each single byte */

    uint64_t *merge_key;        /* open addressing: left << 32 | right, EMPTY_KEY when free */
    int32_t *merge_rank;
    int32_t *merge_result;
    size_t merge_mask;

    int32_t *added;             /* control and user-defined tokens with text, by first byte */
    uint32_t added_start[257];  /* added[added_start[b]..added_start[b+1]) start with byte b,
                                   longest first */
    const family *fam;
    int32_t bos, eos;
    int add_bos, add_eos;
};

static int fail(char *err, size_t err_len, const char *msg, const char *detail) {
    if (err != NULL && err_len > 0) {
        if (detail != NULL) snprintf(err, err_len, "tokenizer: %s '%s'", msg, detail);
        else snprintf(err, err_len, "tokenizer: %s", msg);
    }
    return -1;
}

static uint64_t hash_bytes(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;    /* FNV-1a */
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t hash_u64(uint64_t x) {
    x ^= x >> 33;    /* splitmix64 finalizer */
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

static size_t pow2_at_least(size_t n) {
    size_t c = 16;
    while (c < n) c <<= 1;
    return c;
}

/* GPT-2 byte-level alphabet: printable bytes stand for themselves, the other 68 map
 * in order to U+0100.. so that every byte is a visible character. */
static void byte_alphabet(uint32_t *b2cp, int16_t *cp2b /* [324] */) {
    uint32_t next = 256;
    for (int i = 0; i < 324; i++) cp2b[i] = -1;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        b2cp[b] = printable ? (uint32_t)b : next++;
        cp2b[b2cp[b]] = (int16_t)b;
    }
}

/* string -> id, used only while loading */
typedef struct {
    int32_t *slot;      /* id + 1; 0 when free */
    size_t mask;
    char **str;
} str_index;

static int32_t str_index_find(const str_index *x, const char *s, size_t n) {
    size_t h = (size_t)hash_bytes(s, n) & x->mask;
    while (x->slot[h] != 0) {
        const char *c = x->str[x->slot[h] - 1];
        if (strlen(c) == n && memcmp(c, s, n) == 0) return x->slot[h] - 1;
        h = (h + 1) & x->mask;
    }
    return -1;
}

static void str_index_add(str_index *x, int32_t id) {
    const char *s = x->str[id];
    size_t n = strlen(s);
    size_t h = (size_t)hash_bytes(s, n) & x->mask;
    while (x->slot[h] != 0) {
        const char *c = x->str[x->slot[h] - 1];
        if (strcmp(c, s) == 0) return;      /* the first id with a string keeps it */
        h = (h + 1) & x->mask;
    }
    x->slot[h] = id + 1;
}

static int merge_find(const tr_tokenizer *t, int32_t left, int32_t right, int32_t *rank, int32_t *result) {
    uint64_t key = (uint64_t)(uint32_t)left << 32 | (uint32_t)right;
    size_t h = (size_t)hash_u64(key) & t->merge_mask;
    while (t->merge_key[h] != EMPTY_KEY) {
        if (t->merge_key[h] == key) {
            *rank = t->merge_rank[h];
            *result = t->merge_result[h];
            return 1;
        }
        h = (h + 1) & t->merge_mask;
    }
    return 0;
}

typedef struct {
    uint8_t first;
    uint32_t len;
    int32_t id;
} added_key;

/* by first byte, then longest first, then id */
static int cmp_added(const void *pa, const void *pb) {
    const added_key *a = (const added_key *)pa, *b = (const added_key *)pb;
    if (a->first != b->first) return a->first < b->first ? -1 : 1;
    if (a->len != b->len) return a->len > b->len ? -1 : 1;
    return a->id < b->id ? -1 : (a->id > b->id);
}

void tr_tokenizer_free(tr_tokenizer *t) {
    if (t == NULL) return;
    free(t->type);
    free(t->bytes);
    free(t->off);
    free(t->merge_key);
    free(t->merge_rank);
    free(t->merge_result);
    free(t->added);
    free(t);
}

static int get_token_id(const tr_gguf *g, const char *key, int32_t n_tokens, int32_t *out,
                        char *err, size_t err_len) {
    uint32_t v;
    *out = -1;
    if (tr_gguf_find_kv(g, key) == NULL) return 0;
    if (tr_gguf_get_u32(g, key, &v) != 0 || v >= (uint32_t)n_tokens) return fail(err, err_len, "token id out of range in", key);
    *out = (int32_t)v;
    return 0;
}

tr_tokenizer *tr_tokenizer_load(const tr_gguf *g, char *err, size_t err_len) {
    const char *model, *pre;
    if (tr_gguf_get_str(g, "tokenizer.ggml.model", &model) != 0) {
        fail(err, err_len, "missing tokenizer.ggml.model", NULL);
        return NULL;
    }
    if (strcmp(model, "gpt2") != 0) {
        fail(err, err_len, "unsupported tokenizer model", model);
        return NULL;
    }
    if (tr_gguf_get_str(g, "tokenizer.ggml.pre", &pre) != 0) {
        fail(err, err_len, "missing tokenizer.ggml.pre", NULL);
        return NULL;
    }
    const family *fam = NULL;
    for (size_t i = 0; i < sizeof(FAMILIES) / sizeof(FAMILIES[0]); i++)
        if (strcmp(FAMILIES[i].pre, pre) == 0) fam = &FAMILIES[i];
    if (fam == NULL) {
        fail(err, err_len, "unsupported pre-tokenizer (no oracle run yet)", pre);
        return NULL;
    }

    const tr_gguf_kv *tokens = tr_gguf_find_kv(g, "tokenizer.ggml.tokens");
    const tr_gguf_kv *types = tr_gguf_find_kv(g, "tokenizer.ggml.token_type");
    const tr_gguf_kv *merges = tr_gguf_find_kv(g, "tokenizer.ggml.merges");
    if (tokens == NULL || tokens->type != TR_GGUF_ARRAY || tokens->arr_type != TR_GGUF_STRING ||
        tokens->arr_len == 0 || tokens->arr_len > MAX_TOKENS) {
        fail(err, err_len, "tokenizer.ggml.tokens missing, empty, too long or not strings", NULL);
        return NULL;
    }
    if (types != NULL && (types->type != TR_GGUF_ARRAY || types->arr_type != TR_GGUF_INT32 ||
                          types->arr_len != tokens->arr_len)) {
        fail(err, err_len, "tokenizer.ggml.token_type is not an int32 array as long as the tokens", NULL);
        return NULL;
    }
    if (merges == NULL || merges->type != TR_GGUF_ARRAY || merges->arr_type != TR_GGUF_STRING ||
        merges->arr_len > MAX_MERGES) {
        fail(err, err_len, "tokenizer.ggml.merges missing, too long or not strings", NULL);
        return NULL;
    }

    tr_tokenizer *t = (tr_tokenizer *)calloc(1, sizeof(*t));
    if (t == NULL) {
        fail(err, err_len, "out of memory", NULL);
        return NULL;
    }
    t->fam = fam;
    t->n_tokens = (int32_t)tokens->arr_len;
    char **str = (char **)tokens->arr;
    size_t n = (size_t)t->n_tokens;
    str_index idx = {0};

    if (get_token_id(g, "tokenizer.ggml.bos_token_id", t->n_tokens, &t->bos, err, err_len) != 0 ||
        get_token_id(g, "tokenizer.ggml.eos_token_id", t->n_tokens, &t->eos, err, err_len) != 0)
        goto fail;
    if (tr_gguf_find_kv(g, "tokenizer.ggml.add_bos_token") != NULL &&
        tr_gguf_get_bool(g, "tokenizer.ggml.add_bos_token", &t->add_bos) != 0) {
        fail(err, err_len, "tokenizer.ggml.add_bos_token is not a bool", NULL);
        goto fail;
    }
    if (tr_gguf_find_kv(g, "tokenizer.ggml.add_eos_token") != NULL &&
        tr_gguf_get_bool(g, "tokenizer.ggml.add_eos_token", &t->add_eos) != 0) {
        fail(err, err_len, "tokenizer.ggml.add_eos_token is not a bool", NULL);
        goto fail;
    }
    if ((t->add_bos && t->bos < 0) || (t->add_eos && t->eos < 0)) {
        fail(err, err_len, "add_bos_token/add_eos_token set without the token id", NULL);
        goto fail;
    }

    /* types and decoded bytes */
    t->type = (uint8_t *)malloc(n);
    t->off = (uint32_t *)malloc((n + 1) * sizeof(uint32_t));
    if (t->type == NULL || t->off == NULL) goto oom;
    uint64_t total = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t ty = TR_TOKEN_NORMAL;
        if (types != NULL) memcpy(&ty, (const char *)types->arr + i * 4, 4);
        if (ty < TR_TOKEN_UNDEFINED || ty > TR_TOKEN_BYTE) {
            fail(err, err_len, "invalid token type in tokenizer.ggml.token_type", NULL);
            goto fail;
        }
        t->type[i] = (uint8_t)ty;
        total += strlen(str[i]);
    }
    if (total > UINT32_MAX) {
        fail(err, err_len, "token strings too long", NULL);
        goto fail;
    }
    t->bytes = (char *)malloc(total > 0 ? (size_t)total : 1);
    if (t->bytes == NULL) goto oom;

    uint32_t b2cp[256];
    int16_t cp2b[324];
    byte_alphabet(b2cp, cp2b);
    uint32_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const uint8_t *s = (const uint8_t *)str[i];
        size_t sl = strlen(str[i]);
        t->off[i] = at;
        switch (t->type[i]) {
        case TR_TOKEN_UNUSED:
            break;
        case TR_TOKEN_CONTROL:
        case TR_TOKEN_USER_DEFINED:
            memcpy(t->bytes + at, s, sl);
            at += (uint32_t)sl;
            break;
        default:
            /* byte-level: each alphabet character is one byte; anything else (a
             * malformed file) is kept as its UTF-8, as HF's ByteLevel decoder does */
            for (size_t k = 0; k < sl;) {
                uint32_t cp;
                int used = tr_utf8_decode(s + k, sl - k, &cp);
                if (cp < 324 && cp2b[cp] >= 0) t->bytes[at++] = (char)cp2b[cp];
                else {
                    memcpy(t->bytes + at, s + k, (size_t)used);
                    at += (uint32_t)used;
                }
                k += (size_t)used;
            }
        }
    }
    t->off[n] = at;

    /* string -> id for byte tokens and merges: normal tokens first, so a normal token
     * wins over an added token with the same text; unused tokens never take part */
    idx.mask = pow2_at_least(2 * n) - 1;
    idx.str = str;
    idx.slot = (int32_t *)calloc(idx.mask + 1, sizeof(int32_t));
    if (idx.slot == NULL) goto oom;
    for (size_t i = 0; i < n; i++)
        if (t->type[i] == TR_TOKEN_NORMAL) str_index_add(&idx, (int32_t)i);
    for (size_t i = 0; i < n; i++)
        if (t->type[i] != TR_TOKEN_NORMAL && t->type[i] != TR_TOKEN_UNUSED) str_index_add(&idx, (int32_t)i);

    /* -1 for a byte the vocabulary lacks: GPT-NeoX has no token for the 13 bytes that
     * never occur in valid UTF-8 (C0, C1, F5..FF) */
    for (int b = 0; b < 256; b++) {
        uint8_t u[4];
        int ul = tr_utf8_encode(b2cp[b], u);
        t->byte_token[b] = str_index_find(&idx, (const char *)u, (size_t)ul);
    }

    size_t n_merges = (size_t)merges->arr_len;
    size_t cap = pow2_at_least(2 * n_merges);
    t->merge_mask = cap - 1;
    t->merge_key = (uint64_t *)malloc(cap * sizeof(uint64_t));
    t->merge_rank = (int32_t *)malloc(cap * sizeof(int32_t));
    t->merge_result = (int32_t *)malloc(cap * sizeof(int32_t));
    if (t->merge_key == NULL || t->merge_rank == NULL || t->merge_result == NULL) goto oom;
    for (size_t i = 0; i < cap; i++) t->merge_key[i] = EMPTY_KEY;
    char **mstr = (char **)merges->arr;
    for (size_t i = 0; i < n_merges; i++) {
        const char *m = mstr[i];
        const char *space = m[0] != '\0' ? strchr(m + 1, ' ') : NULL;   /* as llama.cpp: find(' ', 1) */
        if (space == NULL || space[1] == '\0') {
            fail(err, err_len, "malformed merge", m);
            goto fail;
        }
        size_t ll = (size_t)(space - m), rl = strlen(space + 1);
        int32_t left = str_index_find(&idx, m, ll);
        int32_t right = str_index_find(&idx, space + 1, rl);
        char *joined = (char *)malloc(ll + rl + 1);
        if (joined == NULL) goto oom;
        memcpy(joined, m, ll);
        memcpy(joined + ll, space + 1, rl);
        int32_t result = str_index_find(&idx, joined, ll + rl);
        free(joined);
        if (left < 0 || right < 0 || result < 0) {
            fail(err, err_len, "merge names a token that is not in the vocabulary", m);
            goto fail;
        }
        uint64_t key = (uint64_t)(uint32_t)left << 32 | (uint32_t)right;
        size_t h = (size_t)hash_u64(key) & t->merge_mask;
        while (t->merge_key[h] != EMPTY_KEY && t->merge_key[h] != key) h = (h + 1) & t->merge_mask;
        if (t->merge_key[h] == EMPTY_KEY) {     /* a repeated pair keeps its first (lowest) rank */
            t->merge_key[h] = key;
            t->merge_rank[h] = (int32_t)i;
            t->merge_result[h] = result;
        }
    }
    free(idx.slot);
    idx.slot = NULL;

    /* added tokens matched in text */
    size_t n_added = 0;
    for (size_t i = 0; i < n; i++)
        if ((t->type[i] == TR_TOKEN_CONTROL || t->type[i] == TR_TOKEN_USER_DEFINED) && t->off[i + 1] > t->off[i])
            n_added++;
    t->added = (int32_t *)malloc((n_added > 0 ? n_added : 1) * sizeof(int32_t));
    added_key *keys = (added_key *)malloc((n_added > 0 ? n_added : 1) * sizeof(added_key));
    if (t->added == NULL || keys == NULL) {
        free(keys);
        goto oom;
    }
    n_added = 0;
    for (size_t i = 0; i < n; i++)
        if ((t->type[i] == TR_TOKEN_CONTROL || t->type[i] == TR_TOKEN_USER_DEFINED) && t->off[i + 1] > t->off[i]) {
            keys[n_added].first = (uint8_t)t->bytes[t->off[i]];
            keys[n_added].len = t->off[i + 1] - t->off[i];
            keys[n_added].id = (int32_t)i;
            n_added++;
        }
    qsort(keys, n_added, sizeof(added_key), cmp_added);
    memset(t->added_start, 0, sizeof(t->added_start));
    for (size_t i = 0; i < n_added; i++) {
        t->added[i] = keys[i].id;
        t->added_start[keys[i].first + 1]++;
    }
    for (int b = 0; b < 256; b++) t->added_start[b + 1] += t->added_start[b];
    free(keys);
    return t;

oom:
    fail(err, err_len, "out of memory", NULL);
fail:
    free(idx.slot);
    tr_tokenizer_free(t);
    return NULL;
}

/* ---- splitting ------------------------------------------------------------- */

typedef struct {
    uint8_t *norm;
    size_t norm_cap;
    uint32_t *cp, *off;
    uint8_t *cls;
    size_t cp_cap;
} split_scratch;

typedef int (*unit_fn)(void *ctx, const char *bytes, size_t len, int32_t added_id);

/* p with room for `need` elements (doubling), or NULL with p untouched. */
static void *grow(void *p, size_t *cap, size_t need, size_t elem) {
    if (need <= *cap) return p;
    size_t c = *cap > 0 ? *cap : 64;
    while (c < need) {
        if (c > SIZE_MAX / 2) return NULL;
        c *= 2;
    }
    if (c > SIZE_MAX / elem) return NULL;
    void *q = realloc(p, c * elem);
    if (q == NULL) return NULL;
    *cap = c;
    return q;
}

/* GPT-2 rules over code points cp[0..n) with classes cls; piece i..j is emitted
 * as the bytes off[i]..off[j] of text. */
static int split_gpt2(const uint8_t *text, const uint32_t *cp, const uint32_t *off, const uint8_t *cls, size_t n,
                      unit_fn fn, void *ctx) {
    size_t i = 0;
    while (i < n) {
        uint32_t c = cp[i];
        size_t j = i + 1;
        if (c == '\'' && i + 1 < n &&
            (cp[i + 1] == 's' || cp[i + 1] == 't' || cp[i + 1] == 'm' || cp[i + 1] == 'd')) {
            j = i + 2;
        } else if (c == '\'' && i + 2 < n &&
                   ((cp[i + 1] == 'r' && cp[i + 2] == 'e') || (cp[i + 1] == 'v' && cp[i + 2] == 'e') ||
                    (cp[i + 1] == 'l' && cp[i + 2] == 'l'))) {
            j = i + 3;
        } else {
            /* ` ?X+` for X in letters, numbers, other: the optional space only helps
             * when a run follows it, and a space is never a letter, number or other */
            size_t s = (c == ' ' && i + 1 < n) ? i + 1 : i;
            uint8_t k = cls[s];
            if (k != TR_UCLASS_SPACE) {
                j = s + 1;
                while (j < n && cls[j] == k) j++;
            } else {    /* cls[i] is a space too: s == i, or s == i + 1 after ' ' */
                size_t r = i;
                while (r < n && cls[r] == TR_UCLASS_SPACE) r++;
                /* \s+(?!\S) keeps the last space for the next word; a lone space
                 * before a non-space falls through to \s+ */
                j = (r < n && r - i >= 2) ? r - 1 : r;
            }
        }
        if (j <= i) j = i + 1;     /* every piece advances: a rule mistake cannot loop forever */
        if (fn(ctx, (const char *)text + off[i], off[j] - off[i], -1) != 0) return -1;
        i = j;
    }
    return 0;
}

static int split_segment(const tr_tokenizer *t, const uint8_t *seg, size_t len, split_scratch *sc,
                         unit_fn fn, void *ctx) {
    const uint8_t *text = seg;
    size_t tlen = len;
    if (t->fam->nfc) {
        uint8_t *nb = (uint8_t *)grow(sc->norm, &sc->norm_cap, TR_NFC_MAX_LEN(len), 1);
        if (nb == NULL) return -1;
        sc->norm = nb;
        if (tr_nfc(seg, len, sc->norm, &tlen) != 0) return -1;
        text = sc->norm;
    }
    /* the three per-code-point arrays share one capacity */
    size_t cap = sc->cp_cap, c1 = sc->cp_cap, c2 = sc->cp_cap;
    uint32_t *cp = (uint32_t *)grow(sc->cp, &cap, tlen + 1, sizeof(uint32_t));
    if (cp == NULL) return -1;
    sc->cp = cp;
    uint32_t *off = (uint32_t *)grow(sc->off, &c1, tlen + 1, sizeof(uint32_t));
    if (off == NULL) return -1;
    sc->off = off;
    uint8_t *cls = (uint8_t *)grow(sc->cls, &c2, tlen + 1, 1);
    if (cls == NULL) return -1;
    sc->cls = cls;
    sc->cp_cap = cap;

    size_t n = 0;
    for (size_t k = 0; k < tlen; n++) {
        off[n] = (uint32_t)k;
        k += (size_t)tr_utf8_decode(text + k, tlen - k, &cp[n]);
        cls[n] = (uint8_t)tr_uclass_of(cp[n]);
    }
    off[n] = (uint32_t)tlen;
    switch (t->fam->rules) {
    case PRE_GPT2:
        return split_gpt2(text, cp, off, cls, n, fn, ctx);
    }
    return -1;
}

static int walk(const tr_tokenizer *t, const char *text, size_t len, int flags, unit_fn fn, void *ctx) {
    if (len > TR_TOKENIZER_MAX_TEXT) return -1;
    const uint8_t *p = (const uint8_t *)text;
    split_scratch sc = {0};
    int rc = 0;
    size_t seg = 0, q = 0;
    while (q < len && rc == 0) {
        /* the longest added token starting at q */
        int32_t mid = -1;
        size_t mlen = 0;
        for (uint32_t k = t->added_start[p[q]]; k < t->added_start[p[q] + 1]; k++) {
            int32_t id = t->added[k];
            size_t tl = t->off[id + 1] - t->off[id];
            if (tl <= len - q && memcmp(p + q, t->bytes + t->off[id], tl) == 0) {
                mid = id;
                mlen = tl;
                break;
            }
        }
        if (mid < 0) {
            q++;
        } else if (t->type[mid] == TR_TOKEN_CONTROL && !(flags & TR_TOK_PARSE_SPECIAL)) {
            /* not parsed, but still matched: its text stays in the segment and no token
             * overlapping it can match, as in HF's matcher with split_special_tokens */
            q += mlen;
        } else {
            if (q > seg) rc = split_segment(t, p + seg, q - seg, &sc, fn, ctx);
            if (rc == 0) rc = fn(ctx, t->bytes + t->off[mid], mlen, mid);
            q += mlen;
            seg = q;
        }
    }
    if (rc == 0 && len > seg) rc = split_segment(t, p + seg, len - seg, &sc, fn, ctx);
    free(sc.norm);
    free(sc.cp);
    free(sc.off);
    free(sc.cls);
    return rc;
}

typedef struct {
    tr_tokenizer_unit_fn fn;
    void *ctx;
} public_ctx;

static int public_unit(void *ctx, const char *bytes, size_t len, int32_t added_id) {
    public_ctx *pc = (public_ctx *)ctx;
    pc->fn(pc->ctx, bytes, len, added_id);
    return 0;
}

int tr_tokenizer_split(const tr_tokenizer *t, const char *text, size_t len, int flags,
                       tr_tokenizer_unit_fn fn, void *ctx) {
    public_ctx pc = {fn, ctx};
    return walk(t, text, len, flags, public_unit, &pc);
}

/* ---- BPE ------------------------------------------------------------------- */

typedef struct {
    int32_t id;
    uint32_t prev, next;    /* neighbours; UINT32_MAX / n at the ends */
} bpe_sym;

typedef struct {
    int32_t rank, result;
    uint32_t left, right;
    int32_t left_id, right_id;
} bpe_pair;

typedef struct {
    const tr_tokenizer *t;
    bpe_sym *sym;
    size_t sym_cap;
    bpe_pair *heap;
    size_t heap_cap, heap_n;
    int32_t *ids;
    size_t n_ids, ids_cap;
} encoder;

/* lowest rank first; at equal rank the leftmost pair, as HF and llama.cpp */
static int pair_before(const bpe_pair *a, const bpe_pair *b) {
    return a->rank != b->rank ? a->rank < b->rank : a->left < b->left;
}

static void heap_push(encoder *e, const bpe_pair *p) {
    size_t i = e->heap_n++;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!pair_before(p, &e->heap[parent])) break;
        e->heap[i] = e->heap[parent];
        i = parent;
    }
    e->heap[i] = *p;
}

static bpe_pair heap_pop(encoder *e) {
    bpe_pair top = e->heap[0];
    bpe_pair last = e->heap[--e->heap_n];
    size_t i = 0, n = e->heap_n;
    for (;;) {
        size_t c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && pair_before(&e->heap[c + 1], &e->heap[c])) c++;
        if (!pair_before(&e->heap[c], &last)) break;
        e->heap[i] = e->heap[c];
        i = c;
    }
    if (n > 0) e->heap[i] = last;
    return top;
}

static void try_pair(encoder *e, uint32_t left, uint32_t right) {
    bpe_pair p;
    if (merge_find(e->t, e->sym[left].id, e->sym[right].id, &p.rank, &p.result)) {
        p.left = left;
        p.right = right;
        p.left_id = e->sym[left].id;
        p.right_id = e->sym[right].id;
        heap_push(e, &p);
    }
}

static int emit_id(encoder *e, int32_t id) {
    if (e->n_ids == e->ids_cap) {
        int32_t *q = (int32_t *)grow(e->ids, &e->ids_cap, e->n_ids + 1, sizeof(int32_t));
        if (q == NULL) return -1;
        e->ids = q;
    }
    e->ids[e->n_ids++] = id;
    return 0;
}

static int encode_unit(void *ctx, const char *bytes, size_t len, int32_t added_id) {
    encoder *e = (encoder *)ctx;
    if (added_id >= 0) return emit_id(e, added_id);
    const tr_tokenizer *t = e->t;
    if (len == 1) return t->byte_token[(uint8_t)bytes[0]] < 0 ? 0 : emit_id(e, t->byte_token[(uint8_t)bytes[0]]);

    bpe_sym *sym = (bpe_sym *)grow(e->sym, &e->sym_cap, len, sizeof(bpe_sym));
    if (sym == NULL) return -1;
    e->sym = sym;
    /* every merge removes one pair and adds at most two: 3n entries are enough */
    bpe_pair *heap = (bpe_pair *)grow(e->heap, &e->heap_cap, 3 * len, sizeof(bpe_pair));
    if (heap == NULL) return -1;
    e->heap = heap;
    e->heap_n = 0;

    /* a byte without a token is dropped before merging, and its neighbours become
     * adjacent: what HF's BPE does with an unknown symbol when there is no unk token */
    uint32_t n = 0;     /* len <= 3 * TR_TOKENIZER_MAX_TEXT fits */
    for (size_t i = 0; i < len; i++) {
        int32_t id = t->byte_token[(uint8_t)bytes[i]];
        if (id < 0) continue;
        sym[n].id = id;
        sym[n].prev = n - 1;    /* UINT32_MAX for the first */
        sym[n].next = n + 1;
        n++;
    }
    for (uint32_t i = 0; i + 1 < n; i++) try_pair(e, i, i + 1);
    while (e->heap_n > 0) {
        bpe_pair p = heap_pop(e);
        bpe_sym *l = &sym[p.left];
        if (l->id != p.left_id || l->next != p.right || sym[p.right].id != p.right_id) continue;    /* stale */
        bpe_sym *r = &sym[p.right];
        l->id = p.result;
        l->next = r->next;
        if (r->next < n) sym[r->next].prev = p.left;
        r->id = -1;
        if (l->prev != UINT32_MAX) try_pair(e, l->prev, p.left);
        if (l->next < n) try_pair(e, p.left, l->next);
    }
    for (uint32_t i = 0; i < n; i = sym[i].next)
        if (emit_id(e, sym[i].id) != 0) return -1;
    return 0;
}

int tr_tokenizer_encode(const tr_tokenizer *t, const char *text, size_t len, int flags,
                        int32_t **out, size_t *n_out) {
    encoder e = {0};
    e.t = t;
    int rc = 0;
    *out = NULL;
    *n_out = 0;
    if ((flags & TR_TOK_ADD_SPECIAL) && t->add_bos) rc = emit_id(&e, t->bos);
    if (rc == 0) rc = walk(t, text, len, flags, encode_unit, &e);
    if (rc == 0 && (flags & TR_TOK_ADD_SPECIAL) && t->add_eos) rc = emit_id(&e, t->eos);
    free(e.sym);
    free(e.heap);
    if (rc != 0) {
        free(e.ids);
        return -1;
    }
    *out = e.ids;
    *n_out = e.n_ids;
    return 0;
}

/* ---- queries --------------------------------------------------------------- */

const char *tr_tokenizer_piece(const tr_tokenizer *t, int32_t id, size_t *len) {
    if (id < 0 || id >= t->n_tokens) {
        *len = 0;
        return NULL;
    }
    *len = t->off[id + 1] - t->off[id];
    return t->bytes + t->off[id];
}

tr_token_type tr_tokenizer_type(const tr_tokenizer *t, int32_t id) {
    if (id < 0 || id >= t->n_tokens) return TR_TOKEN_UNDEFINED;
    return (tr_token_type)t->type[id];
}

int32_t tr_tokenizer_n_tokens(const tr_tokenizer *t) { return t->n_tokens; }
int32_t tr_tokenizer_bos(const tr_tokenizer *t) { return t->bos; }
int32_t tr_tokenizer_eos(const tr_tokenizer *t) { return t->eos; }
