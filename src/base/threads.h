/* threads.h — persistent worker pool with a parallel-for over index ranges.
 *
 * No OpenMP: plain OS threads (pthreads, Win32 threads on Windows), so the core
 * links against nothing but libc. Workers are created once and sleep between
 * jobs; the calling thread always runs one chunk itself.
 *
 * Determinism contract: results must never depend on the number of threads.
 * A parallel_for body may only write to state owned by its own index range
 * (for example output rows [begin, end)); a reduction that sums across indices
 * is done serially after the parallel_for, in index order. */
#ifndef TR_THREADS_H
#define TR_THREADS_H

#include <stdint.h>

typedef struct tr_pool tr_pool;

/* Body of a parallel_for: processes indices [begin, end). `worker` is in
 * [0, tr_pool_size) and identifies the thread, for per-thread scratch buffers. */
typedef void (*tr_range_fn)(void *ctx, int64_t begin, int64_t end, int worker);

/* n_threads counts the calling thread. n_threads <= 0 means "physical cores"
 * (tr_cpu()->physical_cores; SMT siblings slow matmul down). Returns NULL on failure.
 *
 * Every thread of the pool, the calling one included, is pinned to a logical processor
 * of its own (tr_cpu()->slot, best placement first): left to itself Windows puts two of
 * 16 threads on one physical core and leaves cores idle, which costs 30% of the prefill
 * (docs/MISURE.md "Dove vanno i thread"). The calling thread is put back where it was by
 * tr_pool_destroy. Pinning is skipped where the platform or the topology does not allow
 * it, and TR_POOL_PIN=0 turns it off for A/B measurements. */
tr_pool *tr_pool_create(int n_threads);
void tr_pool_destroy(tr_pool *p);
int tr_pool_size(const tr_pool *p);

/* Splits [0, n) into at most tr_pool_size contiguous chunks, each holding at
 * least min_chunk indices (except when n < min_chunk: one chunk), runs fn on
 * every chunk and returns when all are done. A call made from inside a body
 * runs serially on the calling thread. p == NULL runs serially. Not reentrant
 * from two unrelated threads on the same pool: one job at a time per pool. */
void tr_parallel_for(tr_pool *p, int64_t n, int64_t min_chunk, tr_range_fn fn, void *ctx);

#endif
