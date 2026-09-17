/* greedy.h — greedy generation, with or without speculation on the prompt.
 *
 * One step emits one token (plain greedy) or up to n_draft + 1 tokens in a single forward
 * pass (speculation): the draft comes from tr_lookup_draft, and a drafted token is kept only
 * if the model's own greedy choice at that position is the same token. The choice is read
 * from the logits of that very pass, which are bit-identical to the logits the token would
 * have given alone (model.h, tests/test_spec.c): the tokens emitted with any n_draft are the
 * same tokens, in the same order, as with n_draft = 0. Speculation is a speed knob, never a
 * result. Nothing here allocates. */
#ifndef TR_GREEDY_H
#define TR_GREEDY_H

#include <stdint.h>

#include "../models/model.h"
#include "lookup.h"

/* Draft tokens per step: one pass verifies n_draft + 1 positions. */
#define TR_GREEDY_DRAFT_MAX (TR_LOGIT_ROWS_MAX - 1)

typedef struct {
    tr_session *s;      /* prefilled: its logits must be the ones after hist[0..n_hist-1] */
    int64_t vocab;
    int64_t n_ctx;      /* the session's context, so a step never overruns it */
    int64_t n_draft;    /* 0: plain greedy; else 1..TR_GREEDY_DRAFT_MAX */

    int32_t *hist;      /* caller's buffer, prompt then generated, room for n_ctx tokens */
    int64_t n_hist;     /* tokens already in hist (= tokens evaluated by the session) */

    int32_t next;       /* token chosen from the current logits, not yet evaluated */
    int32_t draft[TR_GREEDY_DRAFT_MAX];

    uint64_t n_steps, n_drafted, n_accepted; /* what generate and run print after a --spec run */
} tr_greedy;

/* Sets up from a session that has just evaluated hist[0..n_hist-1]. -1 if the logits are not
 * available or n_draft is out of range. n_draft is lowered to what one pass of this session
 * can verify (tr_session_max_logit_rows, so a small -b caps it); g->n_draft says what it is. */
int tr_greedy_init(tr_greedy *g, tr_session *s, int64_t vocab, int64_t n_ctx, int64_t n_draft,
                   int32_t *hist, int64_t n_hist);

/* Emits the next tokens into out (room for at least n_draft + 1) and appends them to hist.
 * Returns how many were emitted (>= 1), or -1 if the context is full. */
int64_t tr_greedy_step(tr_greedy *g, int32_t *out);

#endif
