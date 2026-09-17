/* tokenizer.h — text to token ids and back, from a GGUF's tokenizer.* metadata.
 *
 * Supported: tokenizer.ggml.model "gpt2" (byte-level BPE) with the pre-tokenizers
 * listed in tokenizer.c. Anything else fails to load: the engine never tokenizes
 * approximately. The reference is Hugging Face tokenizers as transformers runs it;
 * tools/tokenizer_oracle.py compares ids, pieces, normalized text and decoding.
 *
 * Encoding, in order: added tokens written in the text are matched leftmost-longest
 * on the raw bytes; each stretch between them is normalized (NFC where the family's
 * tokenizer.json has it), split by the family's pre-tokenizer rules, and each piece
 * is BPE-merged by rank. Any bytes are accepted: invalid UTF-8 is kept byte for byte,
 * except a byte the vocabulary has no token for (GPT-NeoX lacks C0, C1 and F5..FF, which
 * valid UTF-8 never contains): it is dropped, as HF drops an unknown symbol.
 *
 * A tokenizer is immutable after load and holds no pointer into the GGUF, so one
 * can serve several threads and outlive the file it came from. */
#ifndef TR_TOKENIZER_H
#define TR_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "../format/gguf.h"

/* tokenizer.ggml.token_type values (file format). */
typedef enum {
    TR_TOKEN_UNDEFINED = 0,
    TR_TOKEN_NORMAL = 1,
    TR_TOKEN_UNKNOWN = 2,
    TR_TOKEN_CONTROL = 3,       /* never content: matched in text only with TR_TOK_PARSE_SPECIAL */
    TR_TOKEN_USER_DEFINED = 4,  /* added token that is text: always matched in text */
    TR_TOKEN_UNUSED = 5,        /* padding of the embedding table: never produced, decodes to nothing */
    TR_TOKEN_BYTE = 6
} tr_token_type;

typedef struct tr_tokenizer tr_tokenizer;

/* NULL on failure with a message in err (missing or malformed metadata, an
 * unsupported model or pre-tokenizer, a merge naming an absent token). */
tr_tokenizer *tr_tokenizer_load(const tr_gguf *g, char *err, size_t err_len);
void tr_tokenizer_free(tr_tokenizer *t);

enum {
    TR_TOK_ADD_SPECIAL = 1,     /* BOS/EOS as tokenizer.ggml.add_bos_token / add_eos_token say */
    TR_TOK_PARSE_SPECIAL = 2    /* control tokens written in the text become their id */
};

/* Longest text encode and split accept. */
#define TR_TOKENIZER_MAX_TEXT ((size_t)1 << 30)

/* Encodes text[0..len). *out is malloc'd (release with free; NULL when *n_out is 0).
 * 0 on success, -1 if memory ran out or len > TR_TOKENIZER_MAX_TEXT (then *out is
 * NULL and *n_out 0). */
int tr_tokenizer_encode(const tr_tokenizer *t, const char *text, size_t len, int flags,
                        int32_t **out, size_t *n_out);

/* Calls fn once per unit the encoder produces, in order: a piece of normalized text
 * handed to BPE (added_id < 0), or a matched added token (added_id >= 0, bytes = its
 * text). Concatenating the bytes gives the normalized text. For tests and the oracle.
 * 0 on success, -1 as for tr_tokenizer_encode. */
typedef void (*tr_tokenizer_unit_fn)(void *ctx, const char *bytes, size_t len, int32_t added_id);
int tr_tokenizer_split(const tr_tokenizer *t, const char *text, size_t len, int flags,
                       tr_tokenizer_unit_fn fn, void *ctx);

/* The bytes a token stands for: byte-level decoded for normal tokens, the literal
 * text for control and user-defined tokens, empty for unused ones. Valid while the
 * tokenizer lives. NULL (and *len 0) for an id out of range. */
const char *tr_tokenizer_piece(const tr_tokenizer *t, int32_t id, size_t *len);
/* TR_TOKEN_UNDEFINED for an id out of range. */
tr_token_type tr_tokenizer_type(const tr_tokenizer *t, int32_t id);

int32_t tr_tokenizer_n_tokens(const tr_tokenizer *t);
int32_t tr_tokenizer_bos(const tr_tokenizer *t);    /* -1 if the file names none */
int32_t tr_tokenizer_eos(const tr_tokenizer *t);    /* -1 if the file names none */

#endif
