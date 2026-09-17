/* lookup.c — see lookup.h. Backwards scan of the context, no state and no allocation. */
#include "lookup.h"

#include <stddef.h>

/* hot: begin */

int64_t tr_lookup_draft(const int32_t *ctx, int64_t n_ctx, int32_t *out, int64_t max_draft) {
    if (ctx == NULL || out == NULL || max_draft <= 0) return 0;

    for (int64_t n = TR_LOOKUP_NGRAM_MAX; n >= TR_LOOKUP_NGRAM_MIN; n--) {
        if (n_ctx < n + 1) continue; /* the tail plus at least one token before it */
        const int32_t *tail = ctx + n_ctx - n;
        /* The most recent occurrence wins: the model repeats what it just did more often
         * than what it did at the top of the file. */
        for (int64_t i = n_ctx - n - 1; i >= 0; i--) {
            if (ctx[i + n - 1] != tail[n - 1]) continue; /* cheap reject on the last token */
            int64_t j = 0;
            while (j < n && ctx[i + j] == tail[j]) j++;
            if (j < n) continue;
            int64_t avail = n_ctx - (i + n); /* what followed that occurrence, at least 1 */
            int64_t k = avail < max_draft ? avail : max_draft;
            for (int64_t d = 0; d < k; d++) out[d] = ctx[i + n + d];
            return k;
        }
    }
    return 0;
}

/* hot: end */
