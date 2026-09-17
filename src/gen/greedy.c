/* greedy.c — see greedy.h. */
#include "greedy.h"

#include <stddef.h>

/* hot: begin */

/* First maximum wins, as everywhere else: the tie-break is part of the result. */
static int32_t argmax_f32(const float *x, int64_t n) {
    int32_t best = 0;
    float bv = x[0];
    for (int64_t i = 1; i < n; i++) {
        if (x[i] > bv) {
            bv = x[i];
            best = (int32_t)i;
        }
    }
    return best;
}

int tr_greedy_init(tr_greedy *g, tr_session *s, int64_t vocab, int64_t n_ctx, int64_t n_draft,
                   int32_t *hist, int64_t n_hist) {
    const float *logits = tr_session_logits(s);
    if (g == NULL || s == NULL || hist == NULL || logits == NULL) return -1;
    if (n_draft < 0 || n_draft > TR_GREEDY_DRAFT_MAX) return -1;
    /* one pass verifies n_draft + 1 positions, and a small -b leaves room for fewer rows */
    int64_t rows = tr_session_max_logit_rows(s);
    if (n_draft > rows - 1) n_draft = rows - 1;
    g->s = s;
    g->vocab = vocab;
    g->n_ctx = n_ctx;
    g->n_draft = n_draft;
    g->hist = hist;
    g->n_hist = n_hist;
    g->next = argmax_f32(logits, vocab);
    g->n_steps = g->n_drafted = g->n_accepted = 0;
    return 0;
}

int64_t tr_greedy_step(tr_greedy *g, int32_t *out) {
    tr_session *s = g->s;
    const int64_t pos = tr_session_pos(s); /* = g->n_hist */

    if (pos + 1 > g->n_ctx) return -1;

    /* The token already chosen belongs to the context the draft is read from, so that a
     * repeated line is found from its first token. n_hist only moves once the pass is done:
     * a failed step leaves the session and the history as they were. */
    g->hist[g->n_hist] = g->next;

    /* Room left decides how much can be verified in this pass. */
    int64_t room = g->n_ctx - pos - 1;
    int64_t want = g->n_draft < room ? g->n_draft : room;
    int64_t k = want > 0 ? tr_lookup_draft(g->hist, g->n_hist + 1, g->draft, want) : 0;

    int32_t buf[1 + TR_GREEDY_DRAFT_MAX];
    buf[0] = g->next;
    for (int64_t i = 0; i < k; i++) buf[1 + i] = g->draft[i];

    if (tr_session_eval_rows(s, buf, k + 1, k + 1) != 0) return -1;
    out[0] = g->next;
    g->n_hist++;

    /* Row `back` of the pass is the logits after buf[k - back]: back = k - a is the model's
     * own choice after the first a drafted tokens were accepted. */
    int64_t a = 0;
    while (a < k && argmax_f32(tr_session_logits_back(s, k - a), g->vocab) == g->draft[a]) {
        out[1 + a] = g->draft[a];
        g->hist[g->n_hist++] = g->draft[a];
        a++;
    }
    g->next = argmax_f32(tr_session_logits_back(s, k - a), g->vocab);

    /* Anything past the accepted prefix was never emitted: forget those cache rows. Must come
     * after the line above, because a rewind invalidates the logits. */
    if (a < k) tr_session_rewind(s, g->n_hist);

    g->n_steps++;
    g->n_drafted += (uint64_t)k;
    g->n_accepted += (uint64_t)a;
    return 1 + a;
}

/* hot: end */
