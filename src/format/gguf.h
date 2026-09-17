/* gguf.h — GGUF v3 reader: metadata, tensor directory, tensor reads.
 *
 * Type numbers are ggml's, so files written by llama.cpp tools open as they are.
 * Model files come from the internet: every count, length, dimension and offset
 * is validated against the file size before use, and a malformed file is an
 * error, never undefined behaviour. Only the header is kept in memory; tensor
 * data is read on demand with positional reads. */
#ifndef TR_GGUF_H
#define TR_GGUF_H

#include <stdint.h>
#include <stddef.h>

#include "../base/platform.h"

/* ggml tensor types (numbering is part of the file format). */
typedef enum {
    TR_TYPE_F32 = 0, TR_TYPE_F16 = 1, TR_TYPE_Q4_0 = 2, TR_TYPE_Q4_1 = 3,
    TR_TYPE_Q5_0 = 6, TR_TYPE_Q5_1 = 7, TR_TYPE_Q8_0 = 8, TR_TYPE_Q8_1 = 9,
    TR_TYPE_Q2_K = 10, TR_TYPE_Q3_K = 11, TR_TYPE_Q4_K = 12, TR_TYPE_Q5_K = 13,
    TR_TYPE_Q6_K = 14, TR_TYPE_Q8_K = 15, TR_TYPE_IQ2_XXS = 16, TR_TYPE_IQ2_XS = 17,
    TR_TYPE_IQ3_XXS = 18, TR_TYPE_IQ1_S = 19, TR_TYPE_IQ4_NL = 20, TR_TYPE_IQ3_S = 21,
    TR_TYPE_IQ2_S = 22, TR_TYPE_IQ4_XS = 23, TR_TYPE_I8 = 24, TR_TYPE_I16 = 25,
    TR_TYPE_I32 = 26, TR_TYPE_I64 = 27, TR_TYPE_F64 = 28, TR_TYPE_IQ1_M = 29,
    TR_TYPE_BF16 = 30, TR_TYPE_TQ1_0 = 34, TR_TYPE_TQ2_0 = 35, TR_TYPE_MXFP4 = 39,
    TR_TYPE_COUNT = 40
} tr_type;

typedef struct {
    const char *name;       /* "q8_0" */
    uint32_t block_elems;   /* elements per block */
    uint32_t block_bytes;   /* bytes per block */
} tr_type_info;

/* NULL for a number that is not a known type. */
const tr_type_info *tr_type_get(uint32_t type);

/* GGUF metadata value types (file format). */
typedef enum {
    TR_GGUF_UINT8 = 0, TR_GGUF_INT8 = 1, TR_GGUF_UINT16 = 2, TR_GGUF_INT16 = 3,
    TR_GGUF_UINT32 = 4, TR_GGUF_INT32 = 5, TR_GGUF_FLOAT32 = 6, TR_GGUF_BOOL = 7,
    TR_GGUF_STRING = 8, TR_GGUF_ARRAY = 9, TR_GGUF_UINT64 = 10, TR_GGUF_INT64 = 11,
    TR_GGUF_FLOAT64 = 12
} tr_gguf_vtype;

typedef struct {
    char *key;
    tr_gguf_vtype type;
    union {
        uint64_t u;     /* UINT8..UINT64, BOOL */
        int64_t i;      /* INT8..INT64 */
        double f;       /* FLOAT32, FLOAT64 */
        char *s;        /* STRING, NUL-terminated (may contain no interior NUL) */
    } v;
    /* ARRAY only */
    tr_gguf_vtype arr_type;     /* never ARRAY: nested arrays are rejected */
    uint64_t arr_len;
    void *arr;                  /* STRING: char *[arr_len]; otherwise arr_len packed
                                   little-endian values of arr_type's natural size */
} tr_gguf_kv;

#define TR_GGUF_MAX_DIMS 4

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t ne[TR_GGUF_MAX_DIMS];  /* ggml order: ne[0] is the row length; unused dims are 1 */
    tr_type type;
    uint64_t offset;                /* absolute offset in the file */
    uint64_t n_elems;
    uint64_t n_bytes;
} tr_gguf_tensor;

typedef struct tr_gguf tr_gguf;

/* Opens and validates the header. Returns NULL on failure with a message in err. */
tr_gguf *tr_gguf_open(const char *path, char *err, size_t err_len);
void tr_gguf_close(tr_gguf *g);

uint32_t tr_gguf_version(const tr_gguf *g);
uint64_t tr_gguf_kv_count(const tr_gguf *g);
const tr_gguf_kv *tr_gguf_kv_at(const tr_gguf *g, uint64_t i);
uint64_t tr_gguf_tensor_count(const tr_gguf *g);
const tr_gguf_tensor *tr_gguf_tensor_at(const tr_gguf *g, uint64_t i);

/* Lookups are O(1) average (hash index built at open). NULL if absent. */
const tr_gguf_kv *tr_gguf_find_kv(const tr_gguf *g, const char *key);
const tr_gguf_tensor *tr_gguf_find_tensor(const tr_gguf *g, const char *name);

/* Typed getters: 0 on success; -1 if the key is absent or its value does not
 * fit (an integer getter accepts any integer type whose value is in range). */
int tr_gguf_get_u32(const tr_gguf *g, const char *key, uint32_t *out);
int tr_gguf_get_u64(const tr_gguf *g, const char *key, uint64_t *out);
int tr_gguf_get_f32(const tr_gguf *g, const char *key, float *out);
int tr_gguf_get_bool(const tr_gguf *g, const char *key, int *out);
int tr_gguf_get_str(const tr_gguf *g, const char *key, const char **out);

/* Reads the whole tensor payload (t->n_bytes) into buf. Thread-safe. 0 on success. */
int tr_gguf_read(const tr_gguf *g, const tr_gguf_tensor *t, void *buf);
/* Reads n bytes starting `offset` bytes into the tensor payload (one expert of
 * a stacked expert tensor, for instance). Fails if the range leaves the tensor. */
int tr_gguf_read_range(const tr_gguf *g, const tr_gguf_tensor *t, uint64_t offset, void *buf, uint64_t n);

/* The underlying file, for readers that need their own I/O policy. */
const tr_file *tr_gguf_file(const tr_gguf *g);

#endif
