/* threads.c — persistent worker pool: one slot per worker, spin then sleep.
 *
 * Protocol (every shared field is either atomic or written only while its reader
 * cannot be reading it):
 *   - Worker i owns slot i. The dispatcher writes fn/ctx/begin/end only while the slot
 *     is IDLE, then publishes it by storing READY. The worker reads the fields only
 *     after seeing READY, runs the chunk, stores IDLE, decrements `remaining`.
 *   - Waiting: a worker spins (CPU pause) for up to spin_sec, then sleeps on cond_work
 *     after setting slot.sleeping under the lock; the dispatcher checks `sleeping`
 *     after publishing and broadcasts under the lock. The dispatcher waits for
 *     `remaining` the same way (spin, then cond_done with caller_sleeping).
 *     Atomics are sequentially consistent, so a store followed by the other side's
 *     load cannot miss a wake-up.
 *   - The calling thread runs chunk 0. A call from inside a body, or with p == NULL,
 *     runs serially (t_depth). */

#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE
#endif

#include "threads.h"
#include "cpu.h"
#include "platform.h"

#include <stdatomic.h>
#include <stdlib.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <pthread.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  include <immintrin.h>
#  define CPU_RELAX() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#  define CPU_RELAX() __asm__("yield")
#else
#  define CPU_RELAX() ((void)0)
#endif

enum { SLOT_IDLE = 0, SLOT_READY = 1 };

typedef struct {
    atomic_int state;
    atomic_int sleeping;
    tr_range_fn fn;
    void *ctx;
    int64_t begin, end;
    tr_pool *pool;
    int id;
} slot;

/* one slot per cache line: workers polling their state never share a line */
typedef union {
    slot s;
    char pad[64];
} slot_line;

struct tr_pool {
    int size; /* includes the calling thread */
    double spin_sec;
    slot_line *slots; /* size entries, slot 0 unused */
    atomic_int remaining;
    atomic_int caller_sleeping;
    atomic_int shutdown;

#if defined(_WIN32)
    HANDLE *workers;
    SRWLOCK lock;
    CONDITION_VARIABLE cond_work;
    CONDITION_VARIABLE cond_done;
#else
    pthread_t *workers;
    pthread_mutex_t lock;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
#endif
};

static _Thread_local int t_worker_id = -1;
static _Thread_local int t_depth = 0;

#if defined(_WIN32)
#  define POOL_LOCK(p)      AcquireSRWLockExclusive(&(p)->lock)
#  define POOL_UNLOCK(p)    ReleaseSRWLockExclusive(&(p)->lock)
#  define POOL_WAKE_ALL(cv) WakeAllConditionVariable(&(cv))
#  define POOL_WAIT(cv, p)  SleepConditionVariableSRW(&(cv), &(p)->lock, INFINITE, 0)
#else
#  define POOL_LOCK(p)      pthread_mutex_lock(&(p)->lock)
#  define POOL_UNLOCK(p)    pthread_mutex_unlock(&(p)->lock)
#  define POOL_WAKE_ALL(cv) pthread_cond_broadcast(&(cv))
#  define POOL_WAIT(cv, p)  pthread_cond_wait(&(cv), &(p)->lock)
#endif

/* hot: begin */
static void worker_loop(slot *w) {
    tr_pool *p = w->pool;
    t_worker_id = w->id;
    for (;;) {
        double t0 = tr_time_sec();
        for (unsigned i = 1; atomic_load(&w->state) != SLOT_READY && !atomic_load(&p->shutdown); i++) {
            CPU_RELAX();
            if ((i & 63u) == 0 && tr_time_sec() - t0 > p->spin_sec) {
                POOL_LOCK(p);
                atomic_store(&w->sleeping, 1);
                while (atomic_load(&w->state) != SLOT_READY && !atomic_load(&p->shutdown))
                    POOL_WAIT(p->cond_work, p);
                atomic_store(&w->sleeping, 0);
                POOL_UNLOCK(p);
            }
        }
        if (atomic_load(&w->state) != SLOT_READY) return; /* shutdown */

        t_depth++;
        w->fn(w->ctx, w->begin, w->end, w->id);
        t_depth--;
        atomic_store(&w->state, SLOT_IDLE);
        if (atomic_fetch_sub(&p->remaining, 1) == 1 && atomic_load(&p->caller_sleeping)) {
            POOL_LOCK(p);
            POOL_WAKE_ALL(p->cond_done);
            POOL_UNLOCK(p);
        }
    }
}
/* hot: end */

#if defined(_WIN32)
static DWORD WINAPI worker_main(LPVOID arg) {
    worker_loop((slot *)arg);
    return 0;
}
#else
static void *worker_main(void *arg) {
    worker_loop((slot *)arg);
    return NULL;
}
#endif

tr_pool *tr_pool_create(int n_threads) {
    const tr_cpu_info *cpu = tr_cpu();
    if (n_threads <= 0) n_threads = cpu->physical_cores;
    if (n_threads < 1) n_threads = 1;

    tr_pool *p = calloc(1, sizeof *p);
    if (p == NULL) return NULL;
    p->size = n_threads;
    p->slots = tr_alloc_aligned(sizeof(slot_line) * (size_t)n_threads, 64);
    p->workers = n_threads > 1 ? malloc(sizeof *p->workers * (size_t)(n_threads - 1)) : NULL;
    if (p->slots == NULL || (n_threads > 1 && p->workers == NULL)) {
        tr_free_aligned(p->slots);
        free(p->workers);
        free(p);
        return NULL;
    }

    /* Spinning only pays while every spinning thread has a core of its own.
     * TR_POOL_SPIN_US overrides the budget, for measurements. */
    p->spin_sec = n_threads <= cpu->logical_cores ? 0.002 : 0.0;
    const char *env = getenv("TR_POOL_SPIN_US");
    if (env != NULL) p->spin_sec = atof(env) * 1e-6;

    atomic_init(&p->remaining, 0);
    atomic_init(&p->caller_sleeping, 0);
    atomic_init(&p->shutdown, 0);
    for (int i = 0; i < n_threads; i++) {
        slot *w = &p->slots[i].s;
        atomic_init(&w->state, SLOT_IDLE);
        atomic_init(&w->sleeping, 0);
        w->fn = NULL;
        w->ctx = NULL;
        w->begin = w->end = 0;
        w->pool = p;
        w->id = i;
    }

#if defined(_WIN32)
    InitializeSRWLock(&p->lock);
    InitializeConditionVariable(&p->cond_work);
    InitializeConditionVariable(&p->cond_done);
#else
    pthread_mutex_init(&p->lock, NULL);
    pthread_cond_init(&p->cond_work, NULL);
    pthread_cond_init(&p->cond_done, NULL);
#endif

    for (int i = 1; i < n_threads; i++) {
#if defined(_WIN32)
        p->workers[i - 1] = CreateThread(NULL, 0, worker_main, &p->slots[i].s, 0, NULL);
        int failed = p->workers[i - 1] == NULL;
#else
        int failed = pthread_create(&p->workers[i - 1], NULL, worker_main, &p->slots[i].s) != 0;
#endif
        if (failed) {
            p->size = i; /* best effort: keep the threads that started */
            break;
        }
    }
    return p;
}

void tr_pool_destroy(tr_pool *p) {
    if (p == NULL) return;
    POOL_LOCK(p);
    atomic_store(&p->shutdown, 1);
    POOL_WAKE_ALL(p->cond_work);
    POOL_UNLOCK(p);

    for (int i = 0; i < p->size - 1; i++) {
#if defined(_WIN32)
        WaitForSingleObject(p->workers[i], INFINITE);
        CloseHandle(p->workers[i]);
#else
        pthread_join(p->workers[i], NULL);
#endif
    }
#if !defined(_WIN32)
    pthread_mutex_destroy(&p->lock);
    pthread_cond_destroy(&p->cond_work);
    pthread_cond_destroy(&p->cond_done);
#endif
    free(p->workers);
    tr_free_aligned(p->slots);
    free(p);
}

int tr_pool_size(const tr_pool *p) {
    return p->size;
}

/* hot: begin */
void tr_parallel_for(tr_pool *p, int64_t n, int64_t min_chunk, tr_range_fn fn, void *ctx) {
    if (n <= 0) return;
    if (min_chunk < 1) min_chunk = 1;

    int chunks = 1;
    if (p != NULL && t_depth == 0 && n >= min_chunk) {
        int64_t max_chunks = n / min_chunk;
        chunks = p->size < max_chunks ? p->size : (int)max_chunks;
    }
    if (chunks <= 1) {
        int worker = t_worker_id >= 0 ? t_worker_id : 0;
        t_depth++;
        fn(ctx, 0, n, worker);
        t_depth--;
        return;
    }

    int64_t base = n / chunks, rem = n % chunks;
    int64_t chunk0_end = base + (rem > 0 ? 1 : 0), offset = chunk0_end;
    atomic_store(&p->remaining, chunks - 1);
    for (int i = 1; i < chunks; i++) {
        slot *w = &p->slots[i].s;
        w->fn = fn;
        w->ctx = ctx;
        w->begin = offset;
        offset += base + (i < rem ? 1 : 0);
        w->end = offset;
        atomic_store(&w->state, SLOT_READY);
    }
    int wake = 0;
    for (int i = 1; i < chunks; i++) wake |= atomic_load(&p->slots[i].s.sleeping);
    if (wake) {
        POOL_LOCK(p);
        POOL_WAKE_ALL(p->cond_work);
        POOL_UNLOCK(p);
    }

    t_worker_id = 0;
    t_depth++;
    fn(ctx, 0, chunk0_end, 0);
    t_depth--;
    t_worker_id = -1;

    double t0 = tr_time_sec();
    for (unsigned i = 1; atomic_load(&p->remaining) != 0; i++) {
        CPU_RELAX();
        if ((i & 63u) == 0 && tr_time_sec() - t0 > p->spin_sec) {
            POOL_LOCK(p);
            atomic_store(&p->caller_sleeping, 1);
            while (atomic_load(&p->remaining) != 0) POOL_WAIT(p->cond_done, p);
            atomic_store(&p->caller_sleeping, 0);
            POOL_UNLOCK(p);
        }
    }
}
/* hot: end */
