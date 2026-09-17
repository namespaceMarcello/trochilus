/* greedy.h — greedy generation, with or without speculation on the prompt.
 *
 * One step emits one token (plain greedy) or up to n_draft + 1 tokens in a single forward
 * pass (speculation): the draft comes from tr_lookup_draft, and a drafted token is kept only
 * if the model's own greedy choice at that position is the same token. The choice is read
 * from the logits of that very pass, which are bit-identical to the logits the token would
 * have given alone (model.h, tests/test_spec.c): the tokens emitted with any n_draft are the
 * same tokens, in the same order, as with n_draft = 0. Speculation is a speed knob, never a
 * result. Nothing here allocates.
 *
 * How many tokens to draft each step is itself tunable (docs/MISURE.md question 25):
 * TR_DRAFT_FIXED always asks for n_draft (still capped by room left in the context);
 * TR_DRAFT_ADAPTIVE tracks k_cur, which starts at n_draft, shrinks to the number actually
 * accepted after a partial rejection, and grows by one after a step where the whole draft was
 * accepted (capped at n_draft). A draft where nothing at all was accepted stops the drafting
 * for a few steps, and that pause doubles while the single-token probe that ends it keeps being
 * wrong (1, 3, 7, 15, capped at 16). A step where the lookup found nothing to draft leaves
 * everything as it was. This keeps the worst case (text with nothing to repeat) at plain greedy
 * speed, while a context that keeps matching still drafts the full n_draft. */
#ifndef TR_GREEDY_H
#define TR_GREEDY_H

#include <stdint.h>

#include "../models/model.h"
#include "lookup.h"

/* Draft tokens per step: one pass verifies n_draft + 1 positions. */
#define TR_GREEDY_DRAFT_MAX (TR_LOGIT_ROWS_MAX - 1)

/* Whether the draft length is fixed at n_draft or tracks k_cur (see the comment above). */
typedef enum { TR_DRAFT_FIXED = 0, TR_DRAFT_ADAPTIVE = 1 } tr_draft_policy;

typedef struct {
    tr_session *s;      /* prefilled: its logits must be the ones after hist[0..n_hist-1] */
    int64_t vocab;
    int64_t n_ctx;      /* the session's context, so a step never overruns it */
    int64_t n_draft;    /* 0: plain greedy; else 1..TR_GREEDY_DRAFT_MAX; the cap on k_cur */
    tr_draft_policy policy;
    int64_t k_cur;      /* current draft length target; TR_DRAFT_FIXED never moves it */
    int64_t cool;       /* steps left with no draft at all, after one that was all wrong */
    int64_t back;       /* how long that pause was last time: it doubles while the probes fail */

    int32_t *hist;      /* caller's buffer, prompt then generated, room for n_ctx tokens */
    int64_t n_hist;     /* tokens already in hist (= tokens evaluated by the session) */

    int32_t next;       /* token chosen from the current logits, not yet evaluated */
    int32_t draft[TR_GREEDY_DRAFT_MAX];

    uint64_t n_steps, n_drafted, n_accepted; /* what generate and run print after a --spec run */
} tr_greedy;

/* Sets up from a session that has just evaluated hist[0..n_hist-1]. -1 if the logits are not
 * available or n_draft is out of range. n_draft is lowered to what one pass of this session
 * can verify (tr_session_max_logit_rows, so a small -b caps it); g->n_draft says what it is.
 * k_cur starts at (the lowered) n_draft regardless of policy. */
int tr_greedy_init(tr_greedy *g, tr_session *s, int64_t vocab, int64_t n_ctx, int64_t n_draft,
                   tr_draft_policy policy, int32_t *hist, int64_t n_hist);

/* Emits the next tokens into out (room for at least n_draft + 1) and appends them to hist.
 * Returns how many were emitted (>= 1), or -1 if the context is full. */
int64_t tr_greedy_step(tr_greedy *g, int32_t *out);

#endif
