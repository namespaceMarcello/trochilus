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
 * Every thread of the pool, the calling one included, is pinned to a physical core of its
 * own (tr_cpu()->slot, best placement first; free between the SMT siblings of that core, or
 * tied to the one logical processor with TR_POOL_PIN=1): left to itself Windows puts two of
 * 16 threads on one physical core and leaves cores idle, which costs 30% of the prefill
 * (docs/MEASUREMENTS.md "Dove vanno i thread"). Threads beyond the last slot are not pinned. The
 * calling thread is put back where it was when the last live pool it created is destroyed,
 * in any order, provided tr_pool_destroy runs on the thread that called tr_pool_create.
 * Pinning is skipped where the platform or the topology does not allow it, and TR_POOL_PIN=0
 * turns it off for A/B measurements.
 *
 * Two pools alive at once take the same slots, first core first: used at the same time they
 * share cores and leave others idle (docs/LESSONS.md #63, open). */
tr_pool *tr_pool_create(int n_threads);
void tr_pool_destroy(tr_pool *p);
int tr_pool_size(const tr_pool *p);

/* How many of the pool's threads the next tr_parallel_for calls use: the first n, which sit on
 * the first n slots of tr_cpu(), so on distinct physical cores spread over the last-level
 * caches. n outside [1, tr_pool_size] means all of them, which is how a pool starts. The
 * threads left out get no work and go to sleep. Work bound by memory bandwidth wants fewer
 * threads than work bound by compute (docs/MEASUREMENTS.md "Thread per fase"); by the determinism
 * contract above the width changes the speed of a result, never the result. Called by the
 * thread that calls tr_parallel_for, between two calls and never from inside a body. */
void tr_pool_set_active(tr_pool *p, int n);
int tr_pool_active(const tr_pool *p);

/* Splits [0, n) into at most tr_pool_size contiguous chunks, each holding at
 * least min_chunk indices (except when n < min_chunk: one chunk), runs fn on
 * every chunk and returns when all are done. A call made from inside a body
 * runs serially on the calling thread, with the worker id of the body it was made
 * from: that id belongs to the outer pool, so a body may only nest calls on its own
 * pool (on another pool the id can exceed that pool's size, docs/LESSONS.md #63).
 * p == NULL runs serially. Not reentrant from two unrelated threads on the same
 * pool: one job at a time per pool. */
void tr_parallel_for(tr_pool *p, int64_t n, int64_t min_chunk, tr_range_fn fn, void *ctx);

/* ---- one thread of one's own, and a monitor to talk to it ----
 * For work that is not a parallel_for: the expert store's I/O thread (src/memory/experts.c).
 * The owner starts the thread and joins it; nothing here is global. */
typedef struct tr_thread tr_thread;
/* Runs fn(arg) on a new thread, not pinned. NULL if the thread could not be made. */
tr_thread *tr_thread_start(void (*fn)(void *arg), void *arg);
/* Waits for fn to return and frees the thread. NULL: nothing. */
void tr_thread_join(tr_thread *t);

/* A mutex and one condition variable. wait: the lock held, releases it, sleeps until a
 * broadcast (or spuriously: always wait in a loop on the condition), and takes it back. */
typedef struct tr_monitor tr_monitor;
tr_monitor *tr_monitor_create(void); /* NULL on failure */
void tr_monitor_free(tr_monitor *m);
void tr_monitor_lock(tr_monitor *m);
void tr_monitor_unlock(tr_monitor *m);
void tr_monitor_wait(tr_monitor *m);
void tr_monitor_broadcast(tr_monitor *m);

#endif
