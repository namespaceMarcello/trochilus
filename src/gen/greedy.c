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
                   tr_draft_policy policy, int32_t *hist, int64_t n_hist) {
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
    g->policy = policy;
    g->k_cur = n_draft;
    g->cool = 0;
    g->back = 0;
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

    /* Room left decides how much can be verified in this pass; k_cur (== n_draft under
     * TR_DRAFT_FIXED) decides how much this step is willing to risk. */
    int64_t room = g->n_ctx - pos - 1;
    int64_t want = g->k_cur < g->n_draft ? g->k_cur : g->n_draft;
    if (room < want) want = room;
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

    /* Adaptive: a fully accepted draft grows k_cur by one (capped at n_draft, slow growth);
     * a partly accepted one shrinks it to what was actually accepted; a draft where nothing was
     * accepted stops drafting altogether for `cool` steps, and that pause doubles every time the
     * probe is wrong again (1, 3, 7, 15, capped at 16). The pause is the important half: an extra
     * row of a pass costs 15-22 ms against the 34 ms of the pass itself, because the drafted
     * token usually routes to other experts and the pass reads their weights too, so a draft
     * pays only above roughly half accepted (docs/MISURE.md §Speculazione dal prompt). A step
     * where the lookup proposed nothing although it was allowed to (k == 0 with want > 0) leaves
     * everything as it was: nothing was risked. */
    if (g->policy == TR_DRAFT_ADAPTIVE) {
        if (want == 0) {
            if (g->cool > 0 && --g->cool == 0) g->k_cur = 1; /* the pause is over: probe once */
        } else if (k > 0) {
            if (a == k) {
                g->k_cur = g->k_cur + 1 < g->n_draft ? g->k_cur + 1 : g->n_draft;
                g->back = 0;
            } else if (a > 0) {
                g->k_cur = a;
                g->back = 0;
            } else {
                g->k_cur = 0;
                g->back = g->back == 0 ? 1 : (g->back < 16 ? g->back * 2 + 1 : 16);
                g->cool = g->back;
            }
        }
    }

    g->n_steps++;
    g->n_drafted += (uint64_t)k;
    g->n_accepted += (uint64_t)a;
    return 1 + a;
}

/* hot: end */
