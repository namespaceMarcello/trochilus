/* chat.c — chat templates rendered in C (see chat.h). */
#include "chat.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *p;
    size_t len, cap;
    int oom;
} strbuf;

static void put(strbuf *b, const char *s, size_t n) {
    if (b->oom) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap > 0 ? b->cap : 256;
        while (cap < b->len + n + 1) cap *= 2;
        char *q = (char *)realloc(b->p, cap);
        if (q == NULL) {
            b->oom = 1;
            return;
        }
        b->p = q;
        b->cap = cap;
    }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void put_str(strbuf *b, const char *s) { put(b, s, strlen(s)); }

static void put_token(strbuf *b, const tr_tokenizer *tok, int32_t id) {
    size_t n;
    const char *s = tr_tokenizer_piece(tok, id, &n);
    if (s != NULL) put(b, s, n);
}

/* allenai OLMoE-1B-7B-0125-Instruct (tokenizer_config.json):
 * {{ bos_token }}{% for message in messages %}
 *   system: '<|system|>\n' + content + '\n'      user: '<|user|>\n' + content + '\n'
 *   assistant: '<|assistant|>\n' + content + eos_token, then '\n' unless it is the last message
 *   {% if loop.last and add_generation_prompt %}'<|assistant|>\n'{% endif %}
 * {% endfor %}   (other roles render nothing) */
static void render_olmoe(strbuf *b, const tr_tokenizer *tok, const tr_chat_msg *m, size_t n, int gen) {
    put_token(b, tok, tr_tokenizer_bos(tok));
    for (size_t i = 0; i < n; i++) {
        int last = i + 1 == n;
        if (strcmp(m[i].role, "system") == 0 || strcmp(m[i].role, "user") == 0) {
            put_str(b, m[i].role[0] == 's' ? "<|system|>\n" : "<|user|>\n");
            put(b, m[i].content, m[i].content_len);
            put_str(b, "\n");
        } else if (strcmp(m[i].role, "assistant") == 0) {
            put_str(b, "<|assistant|>\n");
            put(b, m[i].content, m[i].content_len);
            put_token(b, tok, tr_tokenizer_eos(tok));
            if (!last) put_str(b, "\n");
        }
        if (last && gen) put_str(b, "<|assistant|>\n");
    }
}

struct tr_chat_template {
    const char *name;
    size_t source_len;
    uint64_t source_fnv1a;
    void (*render)(strbuf *b, const tr_tokenizer *tok, const tr_chat_msg *m, size_t n, int gen);
};

static const tr_chat_template TEMPLATES[] = {
    {"olmoe-0125-instruct", 508, 0x4a0a5062e55744b7ull, render_olmoe},
};

const tr_chat_template *tr_chat_template_find(const tr_gguf *g, char *err, size_t err_len) {
    const char *src;
    if (tr_gguf_get_str(g, "tokenizer.chat_template", &src) != 0) {
        if (err != NULL && err_len > 0) snprintf(err, err_len, "chat: the file has no tokenizer.chat_template");
        return NULL;
    }
    size_t n = strlen(src);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)src[i];
        h *= 1099511628211ull;
    }
    for (size_t i = 0; i < sizeof(TEMPLATES) / sizeof(TEMPLATES[0]); i++)
        if (TEMPLATES[i].source_len == n && TEMPLATES[i].source_fnv1a == h) return &TEMPLATES[i];
    if (err != NULL && err_len > 0)
        snprintf(err, err_len, "chat: this model's chat template is not supported yet (%zu bytes, fnv1a %016llx)",
                 n, (unsigned long long)h);
    return NULL;
}

const char *tr_chat_template_name(const tr_chat_template *tpl) {
    return tpl->name;
}

int tr_chat_render(const tr_chat_template *tpl, const tr_tokenizer *tok, const tr_chat_msg *msgs, size_t n,
                   int add_generation_prompt, char **out, size_t *len) {
    strbuf b = {0};
    put(&b, "", 0);     /* an empty conversation still gives a NUL-terminated string */
    tpl->render(&b, tok, msgs, n, add_generation_prompt);
    if (b.oom) {
        free(b.p);
        *out = NULL;
        *len = 0;
        return -1;
    }
    *out = b.p;
    *len = b.len;
    return 0;
}
