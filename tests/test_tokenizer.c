/* test_tokenizer.c — the tokenizer on a synthetic GGUF: loading and its refusals, the
 * GPT-2 split rules, added tokens, BPE against a naive reference, round trips; and a file with no
 * chat template, whose refusal writes no message when the caller gives no buffer.
 * Branches: every metadata refusal of tr_tokenizer_load by its own message, and the edges it
 * accepts (token types 0 and 6, id 0 as merge side and result, bos/eos 0, a merge making a
 * user-defined token); a code point one past the byte alphabet; byte 0 alone and inside a piece;
 * an empty control token, never matched in a text of every byte value (a mutant that matches it
 * loops: a timeout); the "'re" rule at a segment's end over reused buffers; a 64-byte piece that
 * fills the symbol array. The out-of-memory paths are not reached (no failing allocator).
 * Agreement with Hugging Face on a real vocabulary is tools/tokenizer_oracle.py's job. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "../src/format/gguf.h"
#include "../src/tokenizer/chat.h"
#include "../src/tokenizer/tokenizer.h"
#include "../src/tokenizer/unicode.h"

enum {
    ID_AA = 256, ID_AAAA, ID_BC, ID_SPACE_T, ID_E_ACUTE, ID_AB,
    ID_BOS, ID_END, ID_END_BANG, ID_SP2, ID_SP3, ID_PAD, ID_BAR_GT,
    ID_N_ACUTE, ID_XY, ID_EMPTY, N_TOKENS
};

enum { TOK_STRINGS, TOK_MISSING, TOK_INT32, TOK_EMPTY };

/* how the synthetic file departs from the good one; all zero is the good one */
typedef struct {
    const char *model, *pre;    /* NULL: "gpt2", "olmo" */
    int drop_byte;          /* byte + 1 whose token is left out; 0: none */
    const char *drop_as;    /* the text left in its place; NULL: a string no byte maps to */
    int tokens_kind;        /* TOK_*: how tokenizer.ggml.tokens is written */
    int short_types;        /* token_type one element short */
    int types_u32;          /* token_type written as uint32 */
    int set_ab_type;        /* "ab" gets ab_type instead of normal */
    int32_t ab_type;
    int bos_out_of_range;
    int bos_zero;           /* bos_token_id 0 */
    int add_bos_without_bos;
    int eos_zero;           /* eos_token_id 0, and add_eos_token */
    int add_eos_without_eos;
    int no_merges;
    int merges_i32;         /* merges written as an int32 array */
    const char *extra_merge;
} opts;

/* GPT-2 byte-level character of byte b, as UTF-8 (written independently of tokenizer.c). */
static void byte_char(int b, char *out) {
    int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    uint32_t cp = (uint32_t)b;
    if (!printable) {
        int n = 0;
        for (int k = 0; k < b; k++)
            if (!((k >= 33 && k <= 126) || (k >= 161 && k <= 172) || (k >= 174 && k <= 255))) n++;
        cp = 256u + (uint32_t)n;
    }
    int len = tr_utf8_encode(cp, (uint8_t *)out);
    out[len] = '\0';
}

static const char *MERGES[] = {"a a", "aa aa", "b c", "a b", "\xc4\xa0 t", "\xc3\x83 \xc2\xa9"};
static const int32_t MERGE_IDS[][3] = {{'a', 'a', ID_AA}, {ID_AA, ID_AA, ID_AAAA}, {'b', 'c', ID_BC},
                                       {'a', 'b', ID_AB}, {' ', 't', ID_SPACE_T}, {0xC3, 0xA9, ID_E_ACUTE}};
#define N_MERGES (sizeof(MERGES) / sizeof(MERGES[0]))

typedef struct {
    uint8_t *data;
    size_t len, cap;
    int n_kv;               /* counted by the kv_ helpers as they write (LESSONS #113) */
} synth_buf;

static void synth_put(synth_buf *b, const void *p, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2;
        b->data = (uint8_t *)realloc(b->data, b->cap);
        if (b->data == NULL) abort();
    }
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static void synth_u32(synth_buf *b, uint32_t v) { synth_put(b, &v, 4); }
static void synth_u64(synth_buf *b, uint64_t v) { synth_put(b, &v, 8); }
static void synth_str(synth_buf *b, const char *s) { synth_u64(b, strlen(s)); synth_put(b, s, strlen(s)); }

static void kv_str(synth_buf *b, const char *key, const char *v) {
    synth_str(b, key);
    synth_u32(b, TR_GGUF_STRING);
    synth_str(b, v);
    b->n_kv++;
}

static void kv_bool(synth_buf *b, const char *key, int v) {
    uint8_t x = (uint8_t)v;
    synth_str(b, key);
    synth_u32(b, TR_GGUF_BOOL);
    synth_put(b, &x, 1);
    b->n_kv++;
}

static void kv_u32(synth_buf *b, const char *key, uint32_t v) {
    synth_str(b, key);
    synth_u32(b, TR_GGUF_UINT32);
    synth_u32(b, v);
    b->n_kv++;
}

/* the head of an array; its n elements follow */
static void kv_array(synth_buf *b, const char *key, uint32_t type, uint64_t n) {
    synth_str(b, key);
    synth_u32(b, TR_GGUF_ARRAY);
    synth_u32(b, type);
    synth_u64(b, n);
    b->n_kv++;
}

static char path[512];

static int write_gguf(const opts *o) {
    char bytes[256][5];
    const char *tokens[N_TOKENS];
    int32_t types[N_TOKENS];
    for (int i = 0; i < 256; i++) {
        byte_char(i, bytes[i]);
        tokens[i] = bytes[i];
        types[i] = TR_TOKEN_NORMAL;
    }
    if (o->drop_byte) tokens[o->drop_byte - 1] = o->drop_as != NULL ? o->drop_as : "\xe2\x80\xbd";
    tokens[ID_AA] = "aa";
    tokens[ID_AAAA] = "aaaa";
    tokens[ID_BC] = "bc";
    tokens[ID_SPACE_T] = "\xc4\xa0t";
    tokens[ID_E_ACUTE] = "\xc3\x83\xc2\xa9";
    tokens[ID_AB] = "ab";
    tokens[ID_BOS] = "<s>";
    tokens[ID_END] = "<|end|>";
    tokens[ID_END_BANG] = "<|end|>!";
    tokens[ID_SP2] = "  ";
    tokens[ID_SP3] = "   ";
    tokens[ID_PAD] = "[PAD]";
    tokens[ID_BAR_GT] = "|>";
    tokens[ID_N_ACUTE] = "\xc5\x84";    /* U+0144, one past the byte alphabet: kept as UTF-8 */
    tokens[ID_XY] = "xy";
    tokens[ID_EMPTY] = "";              /* last: its text starts at the end of all the others */
    for (int i = ID_AA; i < ID_BOS; i++) types[i] = TR_TOKEN_NORMAL;
    types[ID_BOS] = TR_TOKEN_CONTROL;
    types[ID_END] = TR_TOKEN_CONTROL;
    types[ID_END_BANG] = TR_TOKEN_USER_DEFINED;
    types[ID_SP2] = TR_TOKEN_USER_DEFINED;
    types[ID_SP3] = TR_TOKEN_USER_DEFINED;
    types[ID_PAD] = TR_TOKEN_UNUSED;
    types[ID_BAR_GT] = TR_TOKEN_USER_DEFINED;
    types[ID_N_ACUTE] = TR_TOKEN_NORMAL;
    types[ID_XY] = TR_TOKEN_UNUSED;
    types[ID_EMPTY] = TR_TOKEN_CONTROL;
    if (o->set_ab_type) types[ID_AB] = o->ab_type;

    uint64_t n_merges = N_MERGES + (o->extra_merge != NULL);
    synth_buf kv = {0};
    kv_str(&kv, "general.architecture", "olmoe");
    kv_str(&kv, "tokenizer.ggml.model", o->model != NULL ? o->model : "gpt2");
    kv_str(&kv, "tokenizer.ggml.pre", o->pre != NULL ? o->pre : "olmo");
    if (o->tokens_kind == TOK_STRINGS) {
        kv_array(&kv, "tokenizer.ggml.tokens", TR_GGUF_STRING, N_TOKENS);
        for (int i = 0; i < N_TOKENS; i++) synth_str(&kv, tokens[i]);
    } else if (o->tokens_kind == TOK_INT32) {
        kv_array(&kv, "tokenizer.ggml.tokens", TR_GGUF_INT32, N_TOKENS);
        synth_put(&kv, types, sizeof types);
    } else if (o->tokens_kind == TOK_EMPTY) {
        kv_array(&kv, "tokenizer.ggml.tokens", TR_GGUF_STRING, 0);
    }
    kv_array(&kv, "tokenizer.ggml.token_type", o->types_u32 ? TR_GGUF_UINT32 : TR_GGUF_INT32,
             (uint64_t)(N_TOKENS - o->short_types));
    synth_put(&kv, types, sizeof(int32_t) * (size_t)(N_TOKENS - o->short_types));
    if (!o->no_merges) {
        kv_array(&kv, "tokenizer.ggml.merges", o->merges_i32 ? TR_GGUF_INT32 : TR_GGUF_STRING, n_merges);
        for (size_t i = 0; i < N_MERGES; i++) {
            if (o->merges_i32) synth_u32(&kv, 0);
            else synth_str(&kv, MERGES[i]);
        }
        if (o->extra_merge != NULL) synth_str(&kv, o->extra_merge);
    }
    if (!o->add_bos_without_bos)
        kv_u32(&kv, "tokenizer.ggml.bos_token_id", o->bos_out_of_range ? N_TOKENS : o->bos_zero ? 0 : ID_BOS);
    kv_bool(&kv, "tokenizer.ggml.add_bos_token", 1);
    if (o->eos_zero) kv_u32(&kv, "tokenizer.ggml.eos_token_id", 0);
    if (o->eos_zero || o->add_eos_without_eos) kv_bool(&kv, "tokenizer.ggml.add_eos_token", 1);

    synth_buf b = {0};
    synth_put(&b, "GGUF", 4);
    synth_u32(&b, 3);
    synth_u64(&b, 0);
    synth_u64(&b, (uint64_t)kv.n_kv);
    synth_put(&b, kv.data, kv.len);
    free(kv.data);
    uint8_t zero = 0;
    while (b.len % 32 != 0) synth_put(&b, &zero, 1);     /* the (empty) data section starts aligned */

    FILE *f = fopen(path, "wb");
    int ok = f != NULL && fwrite(b.data, 1, b.len, f) == b.len;
    if (f != NULL) fclose(f);
    free(b.data);
    return ok ? 0 : -1;
}

static tr_tokenizer *load(const opts *o, char *err, size_t err_len) {
    err[0] = '\0';
    if (write_gguf(o) != 0) {
        snprintf(err, err_len, "could not write the test GGUF next to the test binary");
        return NULL;
    }
    tr_gguf *g = tr_gguf_open(path, err, err_len);
    if (g == NULL) return NULL;
    tr_tokenizer *t = tr_tokenizer_load(g, err, err_len);
    tr_gguf_close(g);
    return t;
}

static const opts GOOD = {0};

/* each malformed file is refused with its own message: a mutant that lets it through, or that
 * refuses it for another reason further on, fails the message check */
static void test_refusals(void) {
    static const struct {
        opts o;
        const char *expect;
    } cases[] = {
        {{.model = "llama"}, "unsupported tokenizer model 'llama'"},
        {{.pre = "gpt-2"}, "(no oracle run yet) 'gpt-2'"},
        {{.tokens_kind = TOK_MISSING}, "tokenizer.ggml.tokens missing"},
        {{.tokens_kind = TOK_INT32}, "tokenizer.ggml.tokens missing"},
        {{.tokens_kind = TOK_EMPTY}, "tokenizer.ggml.tokens missing"},
        {{.short_types = 1}, "token_type is not an int32 array"},
        {{.types_u32 = 1}, "token_type is not an int32 array"},
        {{.set_ab_type = 1, .ab_type = 9}, "invalid token type"},
        {{.set_ab_type = 1, .ab_type = TR_TOKEN_BYTE + 1}, "invalid token type"},
        {{.set_ab_type = 1, .ab_type = TR_TOKEN_UNDEFINED - 1}, "invalid token type"},
        {{.bos_out_of_range = 1}, "out of range"},
        {{.add_bos_without_bos = 1}, "set without the token id"},
        {{.add_eos_without_eos = 1}, "set without the token id"},
        {{.no_merges = 1}, "merges missing"},
        {{.merges_i32 = 1}, "merges missing"},
        {{.extra_merge = "a zz"}, "not in the vocabulary"},
        {{.extra_merge = "aaa a"}, "not in the vocabulary"},    /* only the left side is missing */
        {{.extra_merge = "a aaa"}, "not in the vocabulary"},    /* only the right side */
        {{.extra_merge = "x y"}, "not in the vocabulary"},      /* "xy" is an unused token */
        {{.extra_merge = "ab"}, "malformed merge"},
        {{.extra_merge = "a "}, "malformed merge"},
        {{.extra_merge = " x"}, "malformed merge"},             /* the separator is sought from byte 1 */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char err[256];
        tr_tokenizer *t = load(&cases[i].o, err, sizeof err);
        TR_CHECK(t == NULL);
        if (strstr(err, cases[i].expect) == NULL) {
            fprintf(stderr, "refusal %zu: expected '%s' in '%s'\n", i, cases[i].expect, err);
            tr_test_failures++;
        }
        tr_tokenizer_free(t);
    }

    /* a caller with no buffer: refused without writing, whatever length comes with it */
    char err[256] = "";
    opts o = {.model = "llama"};
    TR_CHECK(write_gguf(&o) == 0);
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g != NULL);
    if (g == NULL) return;
    TR_CHECK(tr_tokenizer_load(g, NULL, 16) == NULL);
    tr_gguf_close(g);
}

/* no tokenizer.chat_template: refused, with the reason when there is room for it, and without
 * touching a NULL buffer whatever length comes with it */
static void test_no_template(void) {
    char err[256] = "";
    TR_CHECK(write_gguf(&GOOD) == 0);
    tr_gguf *g = tr_gguf_open(path, err, sizeof err);
    TR_CHECK(g != NULL);
    if (g == NULL) return;
    TR_CHECK(tr_chat_template_find(g, err, sizeof err) == NULL);
    TR_CHECK(strstr(err, "no tokenizer.chat_template") != NULL);
    TR_CHECK(tr_chat_template_find(g, NULL, 16) == NULL);
    tr_gguf_close(g);
}

static int encode_eq(const tr_tokenizer *t, const char *text, size_t len, int flags, const int32_t *want, size_t n_want);

/* GPT-NeoX has no token for bytes that valid UTF-8 never contains: such a byte is dropped
 * before merging and its neighbours become adjacent, as in HF's BPE. */
static void test_missing_byte(void) {
    opts o = {.drop_byte = 0xFF + 1};
    char err[256];
    tr_tokenizer *t = load(&o, err, sizeof err);
    TR_CHECK(t != NULL);
    if (t == NULL) {
        fprintf(stderr, "load without a token for 0xff: %s\n", err);
        return;
    }
    static const int32_t merged[] = {ID_E_ACUTE}, none[] = {0};
    TR_CHECK(encode_eq(t, "\xc3\xff\xa9", 3, 0, merged, 1));    /* one piece of invalid bytes */
    TR_CHECK(encode_eq(t, "\xff", 1, 0, none, 0));
    tr_tokenizer_free(t);
}

static int encode_eq(const tr_tokenizer *t, const char *text, size_t len, int flags, const int32_t *want, size_t n_want) {
    int32_t *ids;
    size_t n;
    if (tr_tokenizer_encode(t, text, len, flags, &ids, &n) != 0) return 0;
    int ok = n == n_want && (n == 0 || memcmp(ids, want, n * sizeof(int32_t)) == 0);
    if (!ok) {
        fprintf(stderr, "encode '%.*s': got", (int)len, text);
        for (size_t i = 0; i < n; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, ", want");
        for (size_t i = 0; i < n_want; i++) fprintf(stderr, " %d", want[i]);
        fprintf(stderr, "\n");
    }
    free(ids);
    return ok;
}

#define ENC(t, text, flags, ...) \
    do { \
        static const int32_t want_[] = {__VA_ARGS__}; \
        TR_CHECK(encode_eq(t, text, sizeof(text) - 1, flags, want_, sizeof(want_) / sizeof(want_[0]))); \
    } while (0)

static void test_encode(const tr_tokenizer *t) {
    const int P = TR_TOK_PARSE_SPECIAL;
    ENC(t, "aaa", P, ID_AA, 'a');                     /* equal ranks: leftmost first */
    ENC(t, "aaaa", P, ID_AAAA);
    ENC(t, "aaaaaaa", P, ID_AAAA, ID_AA, 'a');
    ENC(t, "abc", P, 'a', ID_BC);                     /* rank, not position, decides */
    ENC(t, "abab", P, ID_AB, ID_AB);
    ENC(t, " tt", P, ID_SPACE_T, 't');
    ENC(t, "\xc3\xa9", P, ID_E_ACUTE);
    ENC(t, "e\xcc\x81", P, ID_E_ACUTE);               /* NFC first: e + U+0301 is U+00E9 */
    ENC(t, "a<|end|>b", P, 'a', ID_END, 'b');
    ENC(t, "a<|end|>b", 0, 'a', '<', '|', 'e', 'n', 'd', '|', '>', 'b');
    ENC(t, "<|end|>!", P, ID_END_BANG);               /* longest at the position */
    ENC(t, "<|end|>!", 0, ID_END_BANG);               /* user-defined: matched without parse-special */
    /* an unparsed control token still consumes its text: the user-defined "|>" inside it does
     * not match (HF with split_special_tokens), elsewhere it does */
    ENC(t, "x|>y", 0, 'x', ID_BAR_GT, 'y');
    ENC(t, "x  y", P, 'x', ID_SP2, 'y');
    ENC(t, "x    y", P, 'x', ID_SP3, ' ', 'y');         /* 3 spaces, then " y" */
    ENC(t, "[PAD]", P, '[', 'P', 'A', 'D', ']');      /* unused tokens are never matched */
    ENC(t, "\xff", P, 0xFF);
    ENC(t, "\0\0", P, 0, 0);                          /* token 0 inside a piece is not "no token" */
    ENC(t, "\0", P, 0);                               /* nor as a piece of one byte */
    ENC(t, "ab", TR_TOK_ADD_SPECIAL, ID_BOS, ID_AB);

    int32_t *ids = (int32_t *)1;
    size_t n = 7;
    TR_CHECK(tr_tokenizer_encode(t, "", 0, 0, &ids, &n) == 0 && ids == NULL && n == 0);
    TR_CHECK(tr_tokenizer_encode(t, "a", TR_TOKENIZER_MAX_TEXT + 1, 0, &ids, &n) == -1 && ids == NULL && n == 0);

    size_t len;
    TR_CHECK(tr_tokenizer_piece(t, -1, &len) == NULL && len == 0);
    TR_CHECK(tr_tokenizer_piece(t, N_TOKENS, &len) == NULL && len == 0);
    const char *p = tr_tokenizer_piece(t, ID_E_ACUTE, &len);
    TR_CHECK(len == 2 && memcmp(p, "\xc3\xa9", 2) == 0);
    p = tr_tokenizer_piece(t, ' ', &len);
    TR_CHECK(len == 1 && p[0] == ' ');
    p = tr_tokenizer_piece(t, ID_END, &len);
    TR_CHECK(len == 7 && memcmp(p, "<|end|>", 7) == 0);
    p = tr_tokenizer_piece(t, 0, &len);
    TR_CHECK(len == 1 && p[0] == '\0');              /* U+0100 is byte 0 */
    p = tr_tokenizer_piece(t, ID_N_ACUTE, &len);
    TR_CHECK(len == 2 && memcmp(p, "\xc5\x84", 2) == 0);
    tr_tokenizer_piece(t, ID_PAD, &len);
    TR_CHECK_EQ_INT(len, 0);
    tr_tokenizer_piece(t, ID_EMPTY, &len);
    TR_CHECK_EQ_INT(len, 0);
    TR_CHECK_EQ_INT(tr_tokenizer_type(t, 0), TR_TOKEN_NORMAL);
    TR_CHECK_EQ_INT(tr_tokenizer_type(t, ID_PAD), TR_TOKEN_UNUSED);
    TR_CHECK_EQ_INT(tr_tokenizer_type(t, N_TOKENS), TR_TOKEN_UNDEFINED);
    TR_CHECK_EQ_INT(tr_tokenizer_n_tokens(t), N_TOKENS);
    TR_CHECK_EQ_INT(tr_tokenizer_bos(t), ID_BOS);
    TR_CHECK_EQ_INT(tr_tokenizer_eos(t), -1);
}

/* files at the edges of what is accepted: each loads, where a mutant that moves an edge by one
 * refuses it */
static void test_accepts(void) {
    static const opts edges[] = {
        {.set_ab_type = 1, .ab_type = TR_TOKEN_UNDEFINED},      /* the lowest token type */
        {.set_ab_type = 1, .ab_type = TR_TOKEN_BYTE},           /* the highest */
        {.extra_merge = "| >"},             /* a merge whose result is a user-defined token */
    };
    char err[256];
    for (size_t i = 0; i < sizeof(edges) / sizeof(edges[0]); i++) {
        tr_tokenizer *t = load(&edges[i], err, sizeof err);
        TR_CHECK(t != NULL);
        if (t == NULL) fprintf(stderr, "edge %zu refused: %s\n", i, err);
        tr_tokenizer_free(t);
    }

    /* token 0 is "aa": the merge "a a" makes id 0, and "aa aa" joins id 0 with itself */
    opts zero = {.drop_byte = 1, .drop_as = "aa"};
    tr_tokenizer *t = load(&zero, err, sizeof err);
    TR_CHECK(t != NULL);
    if (t == NULL) fprintf(stderr, "token 0 as a merge refused: %s\n", err);
    else {
        ENC(t, "aa", 0, 0);
        ENC(t, "aaaa", 0, ID_AAAA);
        tr_tokenizer_free(t);
    }

    /* bos and eos both id 0, both added */
    opts ends = {.bos_zero = 1, .eos_zero = 1};
    t = load(&ends, err, sizeof err);
    TR_CHECK(t != NULL);
    if (t == NULL) fprintf(stderr, "bos and eos 0 refused: %s\n", err);
    else {
        ENC(t, "ab", TR_TOK_ADD_SPECIAL, 0, ID_AB, 0);
        ENC(t, "ab", 0, ID_AB);
        tr_tokenizer_free(t);
    }
}

/* pieces joined by '|', added tokens as #id */
typedef struct {
    char buf[512];
    size_t len;
} joined;

static void join_unit(void *ctx, const char *bytes, size_t len, int32_t added_id) {
    joined *j = (joined *)ctx;
    if (j->len > 0 && j->len < sizeof(j->buf) - 1) j->buf[j->len++] = '|';
    if (added_id >= 0) j->len += (size_t)snprintf(j->buf + j->len, sizeof(j->buf) - j->len, "#%d", added_id);
    else if (j->len + len < sizeof(j->buf)) {
        memcpy(j->buf + j->len, bytes, len);
        j->len += len;
    }
    j->buf[j->len < sizeof(j->buf) ? j->len : sizeof(j->buf) - 1] = '\0';
}

static void test_split(const tr_tokenizer *t) {
    static const struct {
        const char *in, *want;
    } cases[] = {
        {"Hello world", "Hello| world"},
        {"it's we'll've", "it|'s| we|'ll|'ve"},
        {"you're", "you|'re"},
        /* the second segment reuses the first's buffers, where an 'e' is left two places on:
         * the "'re" rule must not look past the end */
        {"xxe<|end|>'r", "xxe|#263|'|r"},
        {"IT'S", "IT|'|S"},                         /* contractions are case-sensitive in GPT-2 */
        {"''s", "''|s"},
        {"a\t\tb", "a|\t|\t|b"},
        {"x \n y", "x| \n| y"},
        {"a\t\t", "a|\t\t"},
        {"123456 12ab", "123456| 12|ab"},
        {"!!!abc !?x", "!!!|abc| !?|x"},
        {"a\xff b", "a|\xff| b"},
        {"\xff\xc3\xa9", "\xff|\xc3\xa9"},
        {"citt\xc3\xa0 \xc3\xa8", "citt\xc3\xa0| \xc3\xa8"},
        {"\xd9\xa3\xd9\xa4x", "\xd9\xa3\xd9\xa4|x"},  /* Arabic-Indic digits are numbers */
        {"a\xc2\xa0" "b", "a|\xc2\xa0|b"},            /* NBSP is \s */
        {" ", " "},
        {"\n\n", "\n\n"},
        {"a  b", "a|#265|b"},
        {"e\xcc\x81<|end|>e\xcc\x81", "\xc3\xa9|#263|\xc3\xa9"},
        {"", ""},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        joined j = {{0}, 0};
        TR_CHECK(tr_tokenizer_split(t, cases[i].in, strlen(cases[i].in), TR_TOK_PARSE_SPECIAL, join_unit, &j) == 0);
        if (strcmp(j.buf, cases[i].want) != 0) {
            fprintf(stderr, "split %zu: got '%s', want '%s'\n", i, j.buf, cases[i].want);
            tr_test_failures++;
        }
    }
}

static uint32_t rng_state = 12345;
static uint32_t rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

/* Naive BPE over token ids: merge the lowest-rank adjacent pair, leftmost first, until none. */
static size_t naive_bpe(const char *s, size_t n, int32_t *out) {
    for (size_t i = 0; i < n; i++) out[i] = (uint8_t)s[i];
    for (;;) {
        size_t best = n, best_rank = N_MERGES;
        for (size_t i = 0; i + 1 < n; i++)
            for (size_t r = 0; r < best_rank; r++)
                if (MERGE_IDS[r][0] == out[i] && MERGE_IDS[r][1] == out[i + 1]) {
                    best = i;
                    best_rank = r;
                    break;
                }
        if (best == n) return n;
        out[best] = MERGE_IDS[best_rank][2];
        memmove(out + best + 1, out + best + 2, (n - best - 2) * sizeof(int32_t));
        n--;
    }
}

static void test_bpe_reference(const tr_tokenizer *t) {
    char s[64];
    int32_t want[64];
    int bad = 0;
    for (int it = 0; it < 5000 && bad < 5; it++) {
        size_t n = rnd() % 48;
        for (size_t i = 0; i < n; i++) s[i] = "abc"[rnd() % 3];     /* letters only: one piece */
        size_t nw = naive_bpe(s, n, want);
        if (!encode_eq(t, s, n, 0, want, nw)) bad++;
    }
    TR_CHECK_EQ_INT(bad, 0);

    /* 64 bytes fill the symbol array exactly: the last merges have no right neighbour to
     * relink, and one past the array is not written (ASan sees it) */
    char a64[64];
    int32_t want16[16];
    memset(a64, 'a', sizeof a64);
    for (int i = 0; i < 16; i++) want16[i] = ID_AAAA;
    TR_CHECK(encode_eq(t, a64, sizeof a64, 0, want16, 16));

    /* a long piece must not be quadratic, and 4k a's are k "aaaa" */
    size_t big = 400000;
    char *a = (char *)malloc(big);
    TR_CHECK(a != NULL);
    if (a == NULL) return;
    memset(a, 'a', big);
    int32_t *ids;
    size_t n;
    TR_CHECK(tr_tokenizer_encode(t, a, big, 0, &ids, &n) == 0);
    TR_CHECK_EQ_INT(n, big / 4);
    int all = 1;
    for (size_t i = 0; i < n; i++) all &= ids[i] == ID_AAAA;
    TR_CHECK(all);
    free(ids);
    free(a);
}

/* every byte value in one text: an empty added token, if it were matched, would claim some byte
 * (its first is read past its text), the text holds it, and a match of length zero never advances */
static void test_all_bytes(const tr_tokenizer *t) {
    char text[256];
    uint8_t nfc[TR_NFC_MAX_LEN(256)], back[1024];
    size_t nfc_len, bl = 0, n;
    int32_t *ids;
    for (int i = 0; i < 256; i++) text[i] = (char)i;
    TR_CHECK(tr_nfc((const uint8_t *)text, sizeof text, nfc, &nfc_len) == 0);
    TR_CHECK(tr_tokenizer_encode(t, text, sizeof text, 0, &ids, &n) == 0);
    for (size_t i = 0; i < n; i++) {
        size_t pl;
        const char *p = tr_tokenizer_piece(t, ids[i], &pl);
        if (bl + pl <= sizeof(back)) memcpy(back + bl, p, pl);
        bl += pl;
    }
    free(ids);
    TR_CHECK(bl == nfc_len && memcmp(back, nfc, bl) == 0);
}

/* decode(encode(x)) == NFC(x), for any bytes, with and without parse-special */
static void test_round_trip(const tr_tokenizer *t) {
    static const char *atoms[] = {"a", "b", "c", "t", " ", "\t", "\n", "'", "s", "1", "!", "\xc3\xa9", "e",
                                  "\xcc\x81", "\xcc", "\xff", "<|end|>", "<|end|>!", "  ", "[PAD]", "\xe4\xb8\xad"};
    char text[256];
    uint8_t nfc[TR_NFC_MAX_LEN(256)], back[1024];
    int bad = 0;
    for (int it = 0; it < 3000 && bad < 5; it++) {
        size_t len = 0;
        size_t atoms_n = rnd() % 24;
        for (size_t k = 0; k < atoms_n; k++) {
            const char *atom = atoms[rnd() % (sizeof(atoms) / sizeof(atoms[0]))];
            size_t al = strlen(atom);
            if (len + al > sizeof(text)) break;
            memcpy(text + len, atom, al);
            len += al;
        }
        size_t nfc_len;
        if (tr_nfc((const uint8_t *)text, len, nfc, &nfc_len) != 0) {
            bad++;
            continue;
        }
        for (int flags = 0; flags <= TR_TOK_PARSE_SPECIAL; flags += TR_TOK_PARSE_SPECIAL) {
            int32_t *ids;
            size_t n, bl = 0;
            if (tr_tokenizer_encode(t, text, len, flags, &ids, &n) != 0) {
                bad++;
                continue;
            }
            for (size_t i = 0; i < n; i++) {
                size_t pl;
                const char *p = tr_tokenizer_piece(t, ids[i], &pl);
                if (bl + pl <= sizeof(back)) memcpy(back + bl, p, pl);
                bl += pl;
            }
            free(ids);
            if (bl != nfc_len || memcmp(back, nfc, bl) != 0) {
                fprintf(stderr, "round trip failed (flags %d) on '%.*s'\n", flags, (int)len, text);
                bad++;
            }
        }
    }
    TR_CHECK_EQ_INT(bad, 0);
}

int main(int argc, char **argv) {
    (void)argc;
    const char *sl = strrchr(argv[0], '/'), *bs = strrchr(argv[0], '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    snprintf(path, sizeof path, "%.*s/%s", sl ? (int)(sl - argv[0]) : 1, sl ? argv[0] : ".", "test_tokenizer.gguf");

    test_refusals();
    test_accepts();
    test_no_template();
    test_missing_byte();
    char err[256];
    tr_tokenizer *t = load(&GOOD, err, sizeof err);
    if (t == NULL) {
        fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }
    test_encode(t);
    test_split(t);
    test_bpe_reference(t);
    test_all_bytes(t);
    test_round_trip(t);
    tr_tokenizer_free(t);
    remove(path);
    TR_TEST_EXIT();
}
