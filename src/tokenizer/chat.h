/* chat.h — a conversation turned into prompt text, as the model's own chat template does it.
 *
 * A GGUF carries its template as Jinja (tokenizer.chat_template). The engine runs no Jinja:
 * each supported template is recognized by the exact bytes of its source (length and FNV-1a
 * hash) and rendered by C code written for it, which tools/tokenizer_oracle.py compares with
 * transformers' apply_chat_template. Any other template is refused, never approximated. */
#ifndef TR_CHAT_H
#define TR_CHAT_H

#include <stddef.h>

#include "../format/gguf.h"
#include "tokenizer.h"

typedef struct {
    const char *role;       /* "system", "user", "assistant", ... (NUL-terminated) */
    const char *content;
    size_t content_len;
} tr_chat_msg;

typedef struct tr_chat_template tr_chat_template;

/* The template of this file, or NULL with a message in err (none, or not supported). */
const tr_chat_template *tr_chat_template_find(const tr_gguf *g, char *err, size_t err_len);
const char *tr_chat_template_name(const tr_chat_template *tpl);

/* Renders msgs[0..n) like apply_chat_template(msgs, tokenize=False, add_generation_prompt).
 * bos_token and eos_token are the texts of tok's BOS and EOS tokens. *out is malloc'd and
 * NUL-terminated, *len its length without the NUL. 0, or -1 if memory ran out. */
int tr_chat_render(const tr_chat_template *tpl, const tr_tokenizer *tok, const tr_chat_msg *msgs, size_t n,
                   int add_generation_prompt, char **out, size_t *len);

#endif
