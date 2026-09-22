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
 *     runs serially (t_depth).
 *   - A narrowed pool (tr_pool_set_active) publishes to the first `active` slots only: a
 *     worker past the width never sees READY, spins out its budget and sleeps until a wider
 *     call finds it sleeping and broadcasts. */

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
    int pin;                /* number of processors the worker pins itself to, 0: none */
    unsigned pin_group;
    unsigned short pin_cpu[TR_CPU_MAX_SMT];
} slot;

/* Slots on lines of their own: a worker polling its state never shares a line with another
 * slot. The slot is 72 bytes, so a union with char[64] packed them at 72-byte strides and put
 * worker k's state on the line of worker k-1's begin/end, which the dispatcher writes on every
 * call (docs/LESSONS.md #104). */
typedef struct {
    _Alignas(64) slot s;
} slot_line;
_Static_assert(sizeof(slot_line) % 64 == 0, "a slot_line must fill whole cache lines");

struct tr_pool {
    int size; /* includes the calling thread */
    int active; /* threads a parallel_for may use, 1..size: the dispatcher's own, no worker reads it */
    double spin_sec;
    slot_line *slots; /* size entries, slot 0 unused */
    atomic_int remaining;
    atomic_int caller_sleeping;
    atomic_int shutdown;
    int caller_pinned;           /* this pool counts in t_pin_depth of the thread that made it */
    tr_affinity caller_affinity; /* where the calling thread ran before any live pool pinned it */

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

static _Thread_local int t_worker_id = -1;  /* global-ok: per thread, not per model */
static _Thread_local int t_depth = 0;       /* global-ok: per thread, not per model */
/* Live pools that pinned this thread as their worker 0, and where it ran before the first of
 * them. Per thread and counted, not per pool: with two pools alive the second would save the
 * first one's pin as "before" and, destroyed last, leave the thread on one core for good. */
static _Thread_local int t_pin_depth = 0;       /* global-ok: per thread, not per model */
static _Thread_local tr_affinity t_pin_prev;    /* global-ok: per thread, not per model */

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

/* MinGW keeps _Thread_local in emulated TLS: a thread's first touch of each variable mallocs its
 * storage. Touched when a thread joins a pool, so that its first chunk of work, in the hot path,
 * allocates nothing (tests/test_hot.c; docs/LESSONS.md #106). Elsewhere a plain load and store. */
static void touch_tls(void) {
    volatile int *depth = &t_depth, *id = &t_worker_id;
    *depth = *depth;
    *id = *id;
}

/* A worker pins itself, once, before it ever waits for work: affinity applies to the
 * calling thread, and this is the only moment the new thread is outside the hot loop. */
static void worker_start(slot *w) {
    touch_tls();
    /* A worker without a slot belongs to the scheduler, wherever the caller could run before it
     * was pinned: on Linux a new thread inherits its creator's affinity, which by now is slot 0. */
    if (w->pin > 0) tr_thread_pin(w->pin_group, w->pin_cpu, w->pin, NULL);
    else tr_thread_affinity_restore(&w->pool->caller_affinity);
    worker_loop(w);
}

#if defined(_WIN32)
static DWORD WINAPI worker_main(LPVOID arg) {
    worker_start((slot *)arg);
    return 0;
}
#else
static void *worker_main(void *arg) {
    worker_start((slot *)arg);
    return NULL;
}
#endif

tr_pool *tr_pool_create(int n_threads) {
    const tr_cpu_info *cpu = tr_cpu();
    if (n_threads <= 0) n_threads = cpu->physical_cores;
    if (n_threads < 1) n_threads = 1;

    tr_pool *p = calloc(1, sizeof *p);
    if (p == NULL) return NULL;
    touch_tls(); /* the calling thread runs chunk 0 */
    p->size = n_threads;
    p->active = n_threads;
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

    /* One thread per slot, in the order cpu.c ranked them: physical cores first, spread
     * over the last-level caches. More threads than slots means the extra ones are left
     * to the scheduler rather than doubled onto a core that already has one (worker_start
     * undoes the affinity they inherit from the pinned caller).
     * TR_POOL_PIN: 0 no pin, 1 one logical processor per thread, 2 (the default) the whole
     * physical core, which keeps one thread per core but lets the scheduler pick the sibling
     * (docs/MEASUREMENTS.md §Il pin dei thread). */
    const char *pin_env = getenv("TR_POOL_PIN");
    int pin_mode = pin_env != NULL ? atoi(pin_env) : 2;
    if (cpu->n_slots <= 0) pin_mode = 0;

    atomic_init(&p->remaining, 0);
    atomic_init(&p->caller_sleeping, 0);
    atomic_init(&p->shutdown, 0);
    p->caller_affinity.valid = 0;
    for (int i = 0; i < n_threads; i++) {
        slot *w = &p->slots[i].s;
        atomic_init(&w->state, SLOT_IDLE);
        atomic_init(&w->sleeping, 0);
        w->fn = NULL;
        w->ctx = NULL;
        w->begin = w->end = 0;
        w->pool = p;
        w->id = i;
        w->pin = 0;
        w->pin_group = 0;
        if (pin_mode > 0 && i < cpu->n_slots) {
            const tr_cpu_slot *s = &cpu->slot[i];
            w->pin_group = s->group;
            w->pin = pin_mode >= 2 ? (int)s->n_core : 1;
            for (int j = 0; j < w->pin; j++) w->pin_cpu[j] = s->core[j];
        }
    }
    /* The calling thread runs chunk 0, so it takes slot 0 like any other worker; its
     * previous affinity comes back when the last pool that pinned it is destroyed. */
    if (p->slots[0].s.pin > 0) {
        tr_affinity had;
        had.valid = 0;
        if (tr_thread_pin(p->slots[0].s.pin_group, p->slots[0].s.pin_cpu, p->slots[0].s.pin, &had) == 0) {
            if (t_pin_depth == 0) t_pin_prev = had;
            t_pin_depth++;
            p->caller_pinned = 1;
        }
    }
    if (t_pin_depth > 0) p->caller_affinity = t_pin_prev; /* where a worker without a slot goes back to */

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
            p->active = i;
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
    /* The caller outlives the pool: a process that creates pools of different sizes (the
     * benchmarks do) must not be left pinned to one core by the first of them. While another
     * pool made by this thread is alive, the thread stays its worker 0, on slot 0. */
    if (p->caller_pinned && t_pin_depth > 0 && --t_pin_depth == 0) tr_thread_affinity_restore(&t_pin_prev);
    free(p->workers);
    tr_free_aligned(p->slots);
    free(p);
}

int tr_pool_size(const tr_pool *p) {
    return p->size;
}

void tr_pool_set_active(tr_pool *p, int n) {
    if (p == NULL) return;
    p->active = n >= 1 && n <= p->size ? n : p->size;
}

int tr_pool_active(const tr_pool *p) {
    return p != NULL ? p->active : 1;
}

/* hot: begin */
void tr_parallel_for(tr_pool *p, int64_t n, int64_t min_chunk, tr_range_fn fn, void *ctx) {
    if (n <= 0) return;
    if (min_chunk < 1) min_chunk = 1;

    int chunks = 1;
    if (p != NULL && t_depth == 0 && n >= min_chunk) {
        int64_t max_chunks = n / min_chunk;
        chunks = p->active < max_chunks ? p->active : (int)max_chunks;
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
