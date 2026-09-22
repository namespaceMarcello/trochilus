/* lookup.h — draft tokens taken from the text already in the context (prompt lookup).
 *
 * Speculative decoding needs a guess of the next few tokens. The cheapest guess that
 * costs no second model: the last n-gram of the context is searched backwards, and
 * whatever followed its most recent earlier occurrence is proposed. On code, where the
 * model repeats file contents, names and boilerplate it has already seen, this hits often.
 *
 * The guess is only a guess: the caller verifies every drafted token against the model in
 * one forward pass and keeps the ones the model would have chosen anyway, so the output
 * never depends on what this file returns. Nothing here allocates.
 *
 * Idea from llama.cpp (examples/lookup, common/ngram-cache) and colibri (v4_ngram_draft),
 * see docs/ORIGINS.md; the code is new. */
#ifndef TR_LOOKUP_H
#define TR_LOOKUP_H

#include <stdint.h>

/* Longest and shortest n-gram tried when matching the tail of the context. */
#define TR_LOOKUP_NGRAM_MAX 4
#define TR_LOOKUP_NGRAM_MIN 2

/* Proposes the continuation of the most recent earlier occurrence of the tail of
 * ctx[0..n_ctx-1], longest n-gram first, and writes at most max_draft tokens to out.
 * Returns how many it wrote (0 when nothing matches, or when max_draft <= 0).
 * The earlier occurrence starts before the tail but may overlap it ("0 0 0 0" proposes 0):
 * what is proposed is always text already in the context, never read past its end. */
int64_t tr_lookup_draft(const int32_t *ctx, int64_t n_ctx, int32_t *out, int64_t max_draft);

#endif
