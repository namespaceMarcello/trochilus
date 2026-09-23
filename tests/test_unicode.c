/* test_unicode.c -- tests for src/tokenizer/unicode.c: UTF-8 decode/encode on every valid
 * and invalid byte pattern, round-trip fidelity, the whole-character prefix the chat prints
 * (every lead byte cut short and whole), the \p{L} \p{N} \s classes on known
 * characters, and NFC vectors (basic composition, reordering, blocking, Hangul, invalid
 * bytes, and large combining-mark runs checked against an independent stable-sort
 * reference, not against unicode.c's own internal ccc table). */
#include "test.h"

#include "../src/tokenizer/unicode.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned next_rand(unsigned *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return *seed;
}

/* Encodes cps[0..n) as UTF-8 into buf (must have room for 4*n bytes); returns bytes written. */
static size_t encode_seq(const uint32_t *cps, size_t n, uint8_t *buf) {
    size_t len = 0;
    for (size_t i = 0; i < n; i++) len += (size_t)tr_utf8_encode(cps[i], buf + len);
    return len;
}

/* Decodes src[0..len) fully into cps (must have room for len entries); returns the count. */
static size_t decode_seq(const uint8_t *src, size_t len, uint32_t *cps) {
    size_t n = 0, i = 0;
    while (i < len) {
        uint32_t cp;
        int consumed = tr_utf8_decode(src + i, len - i, &cp);
        cps[n++] = cp;
        i += (size_t)consumed;
    }
    return n;
}

/* ---- UTF-8 decode/encode --------------------------------------------------------------- */

static void test_utf8_valid_forms(void) {
    struct { uint8_t bytes[4]; int len; uint32_t cp; } cases[] = {
        {{0x41, 0, 0, 0}, 1, 0x41},             /* 'A', 1 byte */
        {{0xC3, 0xA9, 0, 0}, 2, 0xE9},           /* e-acute, 2 bytes */
        {{0xE4, 0xB8, 0xAD, 0}, 3, 0x4E2D},      /* CJK "middle", 3 bytes */
        {{0xF0, 0x90, 0x80, 0x80}, 4, 0x10000},  /* U+10000, 4 bytes */
        /* the edges of each length, both sides (a boundary moved by one is otherwise unseen) */
        {{0x7F, 0, 0, 0}, 1, 0x7F},
        {{0xC2, 0x80, 0, 0}, 2, 0x80},
        {{0xDF, 0xBF, 0, 0}, 2, 0x7FF},
        {{0xE0, 0xA0, 0x80, 0}, 3, 0x800},
        {{0xEF, 0xBF, 0xBF, 0}, 3, 0xFFFF},
        {{0xF4, 0x8F, 0xBF, 0xBF}, 4, 0x10FFFF}, /* the last scalar value */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint32_t cp;
        int n = tr_utf8_decode(cases[i].bytes, (size_t)cases[i].len, &cp);
        TR_CHECK_EQ_INT(n, cases[i].len);
        TR_CHECK_EQ_INT(cp, cases[i].cp);

        uint8_t out[4];
        int m = tr_utf8_encode(cp, out);
        TR_CHECK_EQ_INT(m, cases[i].len);
        TR_CHECK(memcmp(out, cases[i].bytes, (size_t)cases[i].len) == 0);
    }
}

static void test_utf8_invalid_forms(void) {
    struct { uint8_t bytes[4]; size_t len; uint8_t first; } cases[] = {
        {{0xC0, 0x80, 0, 0}, 2, 0xC0},          /* overlong 2-byte (U+0000) */
        {{0xE0, 0x80, 0x80, 0}, 3, 0xE0},       /* overlong 3-byte (U+0000) */
        {{0xF0, 0x80, 0x80, 0x80}, 4, 0xF0},    /* overlong 4-byte (U+0000) */
        {{0xED, 0xA0, 0x80, 0}, 3, 0xED},       /* surrogate U+D800 */
        {{0xED, 0xBF, 0xBF, 0}, 3, 0xED},       /* surrogate U+DFFF, the last one */
        {{0xF4, 0x90, 0x80, 0x80}, 4, 0xF4},    /* U+110000, above U+10FFFF */
        {{0x80, 0, 0, 0}, 1, 0x80},             /* lone continuation byte */
        {{0xE2, 0x82, 0, 0}, 2, 0xE2},          /* truncated: needs 3 bytes, only 2 given */
        {{0xF0, 0, 0, 0}, 1, 0xF0},             /* truncated: needs 4 bytes, only 1 given */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint32_t cp;
        int n = tr_utf8_decode(cases[i].bytes, cases[i].len, &cp);
        TR_CHECK_EQ_INT(n, 1); /* only the first byte is ever consumed on invalid input */
        TR_CHECK_EQ_INT(cp, (uint32_t)TR_CP_INVALID_BASE + cases[i].first);
    }
}

static void test_utf8_roundtrip(void) {
    unsigned seed = 0xC0FFEEu;
    uint8_t src[4096];
    size_t len = 0;

    while (len + 4 <= sizeof(src)) {
        if (next_rand(&seed) & 1) {
            src[len++] = (uint8_t)(next_rand(&seed) & 0xFFu); /* raw random byte, often invalid */
        } else {
            uint32_t cp = next_rand(&seed) % 0x110000u;
            if (cp >= 0xD800u && cp <= 0xDFFFu) cp = 0x41; /* skip surrogates, keep it a real cp */
            len += (size_t)tr_utf8_encode(cp, src + len);
        }
    }

    uint32_t *cps = malloc(len * sizeof *cps);
    TR_CHECK(cps != NULL);
    if (cps == NULL) return;
    size_t n = decode_seq(src, len, cps);

    uint8_t *back = malloc(len);
    TR_CHECK(back != NULL);
    if (back != NULL) {
        size_t back_len = encode_seq(cps, n, back);
        TR_CHECK_EQ_INT((long long)back_len, (long long)len);
        TR_CHECK(back_len == len && memcmp(src, back, len) == 0);
        free(back);
    }
    free(cps);
}

/* tr_utf8_whole_prefix: every lead byte cut before its last continuation byte, and whole. Each
 * case is copied to a buffer of exactly its length, so a look one byte before it is an ASan
 * error. */
static void test_utf8_whole_prefix(void) {
    static const struct {
        const char *s;
        size_t want;
    } cases[] = {
        {"", 0},
        {"a", 1},
        {"\xC3", 0},                 /* a 2-byte lead alone */
        {"a\xC3", 1},
        {"\xC3\xA9", 2},             /* é whole */
        {"\xC0", 0},                 /* 0xC0 counts as a 2-byte lead (the lowest one) */
        {"\xE0\xA0", 0},             /* a 3-byte character one byte short (lowest 3-byte lead) */
        {"a\xE2\x82", 1},
        {"a\xE2\x82\xAC", 4},        /* € whole */
        {"\xF0\x9F\x98", 0},         /* a 4-byte character one byte short (lowest 4-byte lead) */
        {"\xF0\x9F\x98\x80", 4},     /* U+1F600 whole */
        {"\x80", 1},                 /* stray continuation bytes: nothing waits, all of it */
        {"\x80\x80\x80\x80", 4},
    };
    for (size_t k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        size_t n = strlen(cases[k].s);
        uint8_t *buf = (uint8_t *)malloc(n > 0 ? n : 1);
        TR_CHECK(buf != NULL);
        if (buf == NULL) return;
        memcpy(buf, cases[k].s, n);
        size_t got = tr_utf8_whole_prefix(buf, n);
        if (got != cases[k].want) fprintf(stderr, "whole prefix of case %zu: %zu, expected %zu\n", k, got, cases[k].want);
        TR_CHECK(got == cases[k].want);
        free(buf);
    }
}

/* ---- \p{L} \p{N} \s classes ------------------------------------------------------------- */

static void test_classes(void) {
    struct { uint32_t cp; tr_uclass want; } cases[] = {
        {0x41, TR_UCLASS_LETTER},              /* A */
        {0xE9, TR_UCLASS_LETTER},              /* e-acute */
        {0x4E2D, TR_UCLASS_LETTER},             /* CJK "middle" */
        {0x663, TR_UCLASS_NUMBER},              /* Arabic-Indic digit three */
        {0x216B, TR_UCLASS_NUMBER},             /* Roman numeral twelve */
        {0xB2, TR_UCLASS_NUMBER},               /* superscript two */
        {0xA0, TR_UCLASS_SPACE},                /* no-break space */
        {0x3000, TR_UCLASS_SPACE},              /* ideographic space */
        {0x2028, TR_UCLASS_SPACE},              /* line separator */
        {0x85, TR_UCLASS_SPACE},                /* next line (NEL) */
        {0x5F, TR_UCLASS_OTHER},                /* underscore */
        {0x20AC, TR_UCLASS_OTHER},              /* euro sign */
        {(uint32_t)TR_CP_INVALID_BASE + 0xFF, TR_UCLASS_OTHER},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        TR_CHECK_EQ_INT(tr_uclass_of(cases[i].cp), cases[i].want);
    }
}

/* ---- NFC: a helper that runs tr_nfc and returns the decoded output code points ---------- */

static size_t run_nfc(const uint8_t *src, size_t len, uint32_t *out_cps, uint8_t *dst_buf) {
    size_t dst_len = 0;
    int rc = tr_nfc(src, len, dst_buf, &dst_len);
    TR_CHECK_EQ_INT(rc, 0);
    return decode_seq(dst_buf, dst_len, out_cps);
}

/* Builds cps[0..n) as UTF-8 in src_buf, runs tr_nfc, decodes the result into out_cps.
 * src_buf and dst_buf must have room for TR_NFC_MAX_LEN(4*n) bytes each. */
static size_t nfc_of_cps(const uint32_t *cps, size_t n, uint8_t *src_buf, uint8_t *dst_buf,
                          uint32_t *out_cps) {
    size_t len = encode_seq(cps, n, src_buf);
    return run_nfc(src_buf, len, out_cps, dst_buf);
}

static int cps_equal(const uint32_t *a, size_t na, const uint32_t *b, size_t nb) {
    if (na != nb) return 0;
    return memcmp(a, b, na * sizeof(uint32_t)) == 0;
}

static void test_nfc_vectors(void) {
    uint8_t src[64], dst[192];
    uint32_t got[16];

    /* e + combining acute -> e-acute */
    {
        uint32_t in[] = {0x65, 0x301};
        uint32_t want[] = {0xE9};
        size_t n = nfc_of_cps(in, 2, src, dst, got);
        TR_CHECK(cps_equal(got, n, want, 1));
    }
    /* ANGSTROM SIGN -> LATIN CAPITAL LETTER A WITH RING ABOVE */
    {
        uint32_t in[] = {0x212B};
        uint32_t want[] = {0xC5};
        size_t n = nfc_of_cps(in, 1, src, dst, got);
        TR_CHECK(cps_equal(got, n, want, 1));
    }
    /* d-with-dot-above + dot-below -> d-with-dot-below, dot-above (composes with the mark
     * that reaches the starter, the other stays separate: no such double-dotted d exists) */
    {
        uint32_t in[] = {0x1E0B, 0x323};
        uint32_t want[] = {0x1E0D, 0x307};
        size_t n = nfc_of_cps(in, 2, src, dst, got);
        TR_CHECK(cps_equal(got, n, want, 2));
    }
    /* a + acute(230) + ogonek(202), wrong order: reorder brings ogonek first, composing with
     * 'a' into a-ogonek; the acute (no such composite exists) stays separate afterwards */
    {
        uint32_t in[] = {0x61, 0x301, 0x328};
        uint32_t want[] = {0x105, 0x301};
        size_t n = nfc_of_cps(in, 3, src, dst, got);
        TR_CHECK(cps_equal(got, n, want, 2));
    }
    /* blocked composition: o + candrabindu(230, does not compose with o) + diaeresis(230):
     * same ccc as the intervening mark blocks diaeresis from reaching the starter */
    {
        uint32_t in[] = {0x6F, 0x310, 0x308};
        size_t n = nfc_of_cps(in, 3, src, dst, got);
        TR_CHECK(cps_equal(got, n, in, 3)); /* unchanged */
    }
    /* text starting with a combining mark: it never becomes a starter, so it never composes */
    {
        uint32_t in[] = {0x301, 0x65};
        size_t n = nfc_of_cps(in, 2, src, dst, got);
        TR_CHECK(cps_equal(got, n, in, 2)); /* unchanged */
    }
    /* Hangul L + V -> LV syllable */
    {
        uint32_t in[] = {0x1100, 0x1161};
        uint32_t want[] = {0xAC00};
        size_t n = nfc_of_cps(in, 2, src, dst, got);
        TR_CHECK(cps_equal(got, n, want, 1));
    }
    /* just outside the ranges that compose algorithmically, nothing composes: an L after the
     * last one (U+1113), a V after the last one (U+1176), U+11A7 (TBase itself, "no trailing
     * consonant", not a T) and a T after the last one (U+11C3) */
    {
        uint32_t cases[4][2] = {{0x1113, 0x1161}, {0x1100, 0x1176}, {0xAC00, 0x11A7}, {0xAC00, 0x11C3}};
        for (int i = 0; i < 4; i++) {
            size_t n = nfc_of_cps(cases[i], 2, src, dst, got);
            TR_CHECK(cps_equal(got, n, cases[i], 2)); /* unchanged */
        }
    }
    /* Hangul LV + T -> LVT syllable */
    {
        uint32_t in[] = {0xAC00, 0x11A8};
        uint32_t want[] = {0xAC01};
        size_t n = nfc_of_cps(in, 2, src, dst, got);
        TR_CHECK(cps_equal(got, n, want, 1));
    }
    /* pure ASCII, unchanged */
    {
        uint32_t in[] = {0x48, 0x65, 0x6C, 0x6C, 0x6F};
        size_t n = nfc_of_cps(in, 5, src, dst, got);
        TR_CHECK(cps_equal(got, n, in, 5));
    }
    /* empty input */
    {
        size_t dst_len = 0xDEADu;
        int rc = tr_nfc(src, 0, dst, &dst_len);
        TR_CHECK_EQ_INT(rc, 0);
        TR_CHECK_EQ_INT((long long)dst_len, 0);
    }
    /* U+1D160: decomposes to 3 code points, 3x the UTF-8 byte length (the documented bound) */
    {
        uint8_t in_bytes[4];
        int in_len = tr_utf8_encode(0x1D160, in_bytes);
        TR_CHECK_EQ_INT(in_len, 4);
        uint8_t big_dst[TR_NFC_MAX_LEN(4)];
        size_t dst_len = 0;
        int rc = tr_nfc(in_bytes, (size_t)in_len, big_dst, &dst_len);
        TR_CHECK_EQ_INT(rc, 0);
        TR_CHECK_EQ_INT((long long)dst_len, (long long)TR_NFC_MAX_LEN(4));
        uint32_t want[] = {0x1D158, 0x1D165, 0x1D16E};
        uint32_t out_cps[8];
        size_t n = decode_seq(big_dst, dst_len, out_cps);
        TR_CHECK(cps_equal(out_cps, n, want, 3));
    }
    /* an invalid byte between a letter and a combining mark: no composition reaches across */
    {
        uint8_t in_bytes[8];
        size_t len = 0;
        in_bytes[len++] = 0x65;             /* 'e' */
        in_bytes[len++] = 0xFF;             /* invalid byte */
        len += (size_t)tr_utf8_encode(0x301, in_bytes + len); /* combining acute */
        uint8_t out_bytes[TR_NFC_MAX_LEN(8)];
        size_t dst_len = 0;
        int rc = tr_nfc(in_bytes, len, out_bytes, &dst_len);
        TR_CHECK_EQ_INT(rc, 0);
        TR_CHECK_EQ_INT((long long)dst_len, (long long)len);
        TR_CHECK(memcmp(in_bytes, out_bytes, len) == 0); /* byte-identical: nothing composed */
    }
}

/* ---- NFC: canonical ordering on long combining-mark runs, checked independently ---------- */

/* Seven marks spanning distinct combining classes (with a tie at 230), none of which compose
 * with 'X' or appear in any decomposition -- so tr_nfc on X + these marks can only reorder
 * them, never compose or grow them. Combining classes are Unicode facts, not read from
 * unicode.c's internal (non-public) table: this check is independent of it. */
static const uint32_t MARK_CP[7] = {0x334, 0x93C, 0x5B0, 0x327, 0x316, 0x300, 0x301};
static const int MARK_CCC[7] = {1, 7, 10, 202, 220, 230, 230};

static int mark_ccc(uint32_t cp) {
    for (int i = 0; i < 7; i++) {
        if (MARK_CP[i] == cp) return MARK_CCC[i];
    }
    return -1; /* not one of ours */
}

static void build_mark_run(uint32_t *seq, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) seq[i] = MARK_CP[next_rand(&seed) % 7];
}

/* Stable insertion sort by ccc, independent of unicode.c's own merge sort. */
static void insertion_sort_by_ccc(uint32_t *seq, size_t n) {
    for (size_t i = 1; i < n; i++) {
        uint32_t v = seq[i];
        int vccc = mark_ccc(v);
        size_t j = i;
        while (j > 0 && mark_ccc(seq[j - 1]) > vccc) {
            seq[j] = seq[j - 1];
            j--;
        }
        seq[j] = v;
    }
}

static void test_nfc_stable_reorder_small(void) {
    enum { N = 500 };
    uint32_t marks[N], reference[N];
    build_mark_run(marks, N, 0xA5A5u);
    memcpy(reference, marks, sizeof marks);
    insertion_sort_by_ccc(reference, N);

    uint32_t in[N + 1], want[N + 1];
    in[0] = 0x58; /* 'X', composes with none of the marks */
    memcpy(in + 1, marks, sizeof marks);
    want[0] = 0x58;
    memcpy(want + 1, reference, sizeof reference);

    uint8_t *src = malloc(4 * (N + 1));
    uint8_t *dst = malloc(TR_NFC_MAX_LEN(4 * (size_t)(N + 1)));
    uint32_t *got = malloc((N + 1) * sizeof(uint32_t));
    TR_CHECK(src != NULL && dst != NULL && got != NULL);
    if (src != NULL && dst != NULL && got != NULL) {
        size_t n = nfc_of_cps(in, N + 1, src, dst, got);
        TR_CHECK(cps_equal(got, n, want, N + 1));
    }
    free(src);
    free(dst);
    free(got);
}

static void test_nfc_stable_reorder_large(void) {
    enum { N = 100000 };
    uint32_t *marks = malloc(N * sizeof(uint32_t));
    uint32_t *in = malloc((N + 1) * sizeof(uint32_t));
    TR_CHECK(marks != NULL && in != NULL);
    if (marks == NULL || in == NULL) {
        free(marks);
        free(in);
        return;
    }
    build_mark_run(marks, N, 0x1234u);
    in[0] = 0x58;
    memcpy(in + 1, marks, N * sizeof(uint32_t));

    size_t src_cap = 4 * (size_t)(N + 1);
    uint8_t *src = malloc(src_cap);
    uint8_t *dst = malloc(TR_NFC_MAX_LEN(src_cap));
    uint32_t *got = malloc((N + 1) * sizeof(uint32_t));
    TR_CHECK(src != NULL && dst != NULL && got != NULL);
    if (src != NULL && dst != NULL && got != NULL) {
        size_t len = encode_seq(in, N + 1, src);

        clock_t t0 = clock();
        size_t n = run_nfc(src, len, got, dst);
        clock_t t1 = clock();
        double seconds = (double)(t1 - t0) / CLOCKS_PER_SEC;
        TR_CHECK(seconds < 5.0); /* O(k log k), not O(k^2): would take far longer if quadratic */

        TR_CHECK_EQ_INT((long long)n, (long long)(N + 1));
        if (n == (size_t)(N + 1)) {
            TR_CHECK_EQ_INT(got[0], 0x58u);

            int sorted = 1;
            for (size_t i = 2; i < n; i++) {
                if (mark_ccc(got[i - 1]) > mark_ccc(got[i])) sorted = 0;
            }
            TR_CHECK(sorted);

            /* Stability: the only tie is ccc 230 between U+0300 and U+0301 -- their relative
             * order among each other must be unchanged from the input. */
            uint32_t *ref230 = malloc(N * sizeof(uint32_t));
            uint32_t *got230 = malloc(N * sizeof(uint32_t));
            TR_CHECK(ref230 != NULL && got230 != NULL);
            if (ref230 != NULL && got230 != NULL) {
                size_t wr = 0, gw = 0;
                for (size_t i = 1; i < N + 1; i++) {
                    if (in[i] == 0x300 || in[i] == 0x301) ref230[wr++] = in[i];
                }
                for (size_t i = 1; i < n; i++) {
                    if (got[i] == 0x300 || got[i] == 0x301) got230[gw++] = got[i];
                }
                TR_CHECK(gw == wr);
                TR_CHECK(wr == 0 || memcmp(ref230, got230, wr * sizeof(uint32_t)) == 0);
            }
            free(ref230);
            free(got230);
        }
    }
    free(src);
    free(dst);
    free(got);
    free(marks);
    free(in);
}

int main(void) {
    test_utf8_valid_forms();
    test_utf8_invalid_forms();
    test_utf8_roundtrip();
    test_utf8_whole_prefix();
    test_classes();
    test_nfc_vectors();
    test_nfc_stable_reorder_small();
    test_nfc_stable_reorder_large();

    TR_TEST_EXIT();
}
