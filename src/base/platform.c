/* platform.c — Windows, Linux and macOS implementations of platform.h.
 *
 * No third-party code: only libc and OS APIs (Win32, POSIX, Mach). Windows
 * uses CreateFileW opened with FILE_FLAG_OVERLAPPED so tr_file_pread can pass
 * a fresh OVERLAPPED (offset + private event) per call, which is what makes
 * concurrent positional reads on the same handle thread-safe: the offset never
 * goes through a shared file pointer. Linux and macOS use pread(2), which is
 * positional and thread-safe by definition. */

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* sched_setaffinity and the CPU_* macros */
#endif
#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE
#endif

#include "platform.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#  include <fcntl.h>
#  include <io.h>
#else
#  include <dlfcn.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <sched.h>
#  include <time.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/uio.h>
#  if defined(__APPLE__)
#    include <mach/mach.h>
#    include <mach/mach_host.h>
#    include <sys/sysctl.h>
#  endif
#endif

struct tr_file {
#if defined(_WIN32)
    HANDLE handle;
#else
    int fd;
#endif
    int direct; /* opened with tr_file_open_direct: tr_file_pread applies its EOF rule */
};

static void set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (err == NULL || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

#if defined(_WIN32)

static wchar_t *utf8_to_utf16(const char *path) {
    int needed = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (needed <= 0) return NULL;
    wchar_t *w = malloc((size_t)needed * sizeof(wchar_t));
    if (w == NULL) return NULL;
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, w, needed) <= 0) {
        free(w);
        return NULL;
    }
    return w;
}

void tr_utf8_argv_free(int argc, char **argv) {
    if (argv == NULL) return;
    for (int i = 0; i < argc; i++) free(argv[i]);
    free(argv);
}

char **tr_utf8_argv(int argc, wchar_t **wargv) {
    char **argv = calloc((size_t)argc + 1, sizeof(char *));
    if (argv == NULL) return NULL;
    for (int i = 0; i < argc; i++) {
        int needed = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        if (needed <= 0 || (argv[i] = malloc((size_t)needed)) == NULL ||
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, argv[i], needed, NULL, NULL) <= 0) {
            tr_utf8_argv_free(argc, argv);
            return NULL;
        }
    }
    return argv;
}

void tr_console_utf8(void) {
    SetConsoleOutputCP(CP_UTF8);
}

void tr_stdout_binary(void) {
    fflush(stdout);
    _setmode(_fileno(stdout), _O_BINARY);
}

tr_file *tr_file_open(const char *path, char *err, size_t err_len) {
    wchar_t *wpath = utf8_to_utf16(path);
    if (wpath == NULL) {
        set_err(err, err_len, "invalid UTF-8 path");
        return NULL;
    }

    HANDLE h = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    free(wpath);
    if (h == INVALID_HANDLE_VALUE) {
        set_err(err, err_len, "CreateFileW failed: error %lu",
                (unsigned long)GetLastError());
        return NULL;
    }

    tr_file *f = calloc(1, sizeof *f);
    if (f == NULL) {
        set_err(err, err_len, "out of memory");
        CloseHandle(h);
        return NULL;
    }
    f->handle = h;
    f->direct = 0;
    return f;
}

tr_file *tr_file_open_direct(const char *path, char *err, size_t err_len) {
    wchar_t *wpath = utf8_to_utf16(path);
    if (wpath == NULL) {
        set_err(err, err_len, "invalid UTF-8 path");
        return NULL;
    }

    HANDLE h = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    free(wpath);
    if (h == INVALID_HANDLE_VALUE) {
        set_err(err, err_len, "CreateFileW (unbuffered) failed: error %lu", (unsigned long)GetLastError());
        return NULL;
    }

    tr_file *f = calloc(1, sizeof *f);
    if (f == NULL) {
        set_err(err, err_len, "out of memory");
        CloseHandle(h);
        return NULL;
    }
    f->handle = h;
    f->direct = 1;
    return f;
}

void tr_file_close(tr_file *f) {
    if (f == NULL) return;
    CloseHandle(f->handle);
    free(f);
}

int64_t tr_file_size(const tr_file *f) {
    LARGE_INTEGER size;
    if (!GetFileSizeEx(f->handle, &size)) return -1;
    return (int64_t)size.QuadPart;
}

int64_t tr_file_alignment(const tr_file *f) {
    return f->direct ? TR_FILE_DIRECT_ALIGN : 1;
}

int tr_file_pread(const tr_file *f, void *buf, size_t n, uint64_t offset) {
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        DWORD want = (n > 0x10000000u) ? 0x10000000u : (DWORD)n; /* 256 MiB cap per call */

        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)(offset & 0xFFFFFFFFu);
        ov.OffsetHigh = (DWORD)(offset >> 32);
        ov.hEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (ov.hEvent == NULL) return -1;

        DWORD got = 0;
        int eof = 0;
        BOOL ok = ReadFile(f->handle, p, want, &got, &ov);
        if (!ok) {
            DWORD gle = GetLastError();
            if (gle == ERROR_IO_PENDING) {
                BOOL waited = GetOverlappedResult(f->handle, &ov, &got, TRUE);
                DWORD wait_err = waited ? 0 : GetLastError();
                CloseHandle(ov.hEvent);
                if (!waited) {
                    if (f->direct && wait_err == ERROR_HANDLE_EOF) eof = 1;
                    else return -1;
                }
            } else {
                CloseHandle(ov.hEvent);
                if (f->direct && gle == ERROR_HANDLE_EOF) eof = 1;
                else return -1; /* includes ERROR_HANDLE_EOF for a buffered read starting past EOF */
            }
        } else {
            CloseHandle(ov.hEvent);
        }

        /* a direct handle's aligned range can run past the file's real end: the last, partial
         * sector then comes back short (eof, or got == 0), which is not an error (platform.h) */
        if (eof) return 0;
        if (got == 0) return f->direct ? 0 : -1; /* end of file before n bytes were read */
        /* short of a sector boundary: the real end; the rest, asked from here, is error 87 */
        if (f->direct && got % TR_FILE_DIRECT_ALIGN != 0) return 0;
        p += got;
        offset += got;
        n -= got;
    }
    return 0;
}

size_t tr_file_preadv_scratch(size_t n) {
    return (n / TR_FILE_DIRECT_ALIGN + 1) * sizeof(FILE_SEGMENT_ELEMENT);
}

size_t tr_file_preadv_n_scratch(size_t total, int n) {
    return (total / TR_FILE_DIRECT_ALIGN + (size_t)n) * sizeof(FILE_SEGMENT_ELEMENT);
}

/* One direct request of tr_file_preadv_n: issued (its pages listed in seg, then ReadFileScatter),
 * finished later. in_flight 0: nothing to wait for, rc is its result. */
typedef struct {
    OVERLAPPED ov;
    size_t total;
    uint64_t offset;
    int in_flight, rc;
} preadv_op;

static void preadv_issue(const tr_file *f, const tr_readv_req *r, FILE_SEGMENT_ELEMENT *seg, preadv_op *op) {
    size_t pages = 0;
    for (int i = 0; i < r->cnt; i++)
        for (size_t at = 0; at < r->iov[i].n; at += TR_FILE_DIRECT_ALIGN)
            seg[pages++].Buffer = PtrToPtr64((unsigned char *)r->iov[i].base + at);
    seg[pages].Buffer = NULL;
    op->total = pages * TR_FILE_DIRECT_ALIGN;
    op->offset = r->offset;
    op->in_flight = 0;
    op->rc = 0;
    if (op->total == 0) return;
    if (op->total > 0x7FFFF000u) {
        op->rc = -1;
        return;
    }
    memset(&op->ov, 0, sizeof op->ov);
    op->ov.Offset = (DWORD)(r->offset & 0xFFFFFFFFu);
    op->ov.OffsetHigh = (DWORD)(r->offset >> 32);
    op->ov.hEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (op->ov.hEvent == NULL) {
        op->rc = -1;
        return;
    }
    BOOL ok = ReadFileScatter(f->handle, seg, (DWORD)op->total, NULL, &op->ov);
    DWORD gle = ok ? 0 : GetLastError();
    if (ok || gle == ERROR_IO_PENDING) {
        op->in_flight = 1;
        return;
    }
    CloseHandle(op->ov.hEvent);
    op->rc = gle == ERROR_HANDLE_EOF ? 0 : -1; /* wholly past the real end: as tr_file_pread */
}

static int preadv_finish(const tr_file *f, preadv_op *op) {
    if (!op->in_flight) return op->rc;
    DWORD got = 0;
    BOOL ok = GetOverlappedResult(f->handle, &op->ov, &got, TRUE);
    DWORD gle = ok ? 0 : GetLastError();
    CloseHandle(op->ov.hEvent);
    if (!ok) return gle == ERROR_HANDLE_EOF ? 0 : -1;
    if (got == op->total) return 0;
    /* short: only the file's real end may stop it, as for tr_file_pread on a direct handle */
    int64_t size = tr_file_size(f);
    return size >= 0 && op->offset + got >= (uint64_t)size ? 0 : -1;
}

int tr_file_preadv_n(const tr_file *f, const tr_readv_req *req, int n, void *scratch) {
    if (n < 1 || n > TR_FILE_PREADV_MAX) return -1;
    if (!f->direct) { /* ReadFileScatter wants an unbuffered handle: a read a piece, a request after another */
        for (int k = 0; k < n; k++) {
            uint64_t offset = req[k].offset;
            for (int i = 0; i < req[k].cnt; i++) {
                if (tr_file_pread(f, req[k].iov[i].base, req[k].iov[i].n, offset) != 0) return -1;
                offset += req[k].iov[i].n;
            }
        }
        return 0;
    }
    /* every request in flight before the first is waited for; each one's pages listed after the last's */
    preadv_op op[TR_FILE_PREADV_MAX];
    FILE_SEGMENT_ELEMENT *seg = (FILE_SEGMENT_ELEMENT *)scratch;
    for (int k = 0; k < n; k++) {
        preadv_issue(f, &req[k], seg, &op[k]);
        seg += op[k].total / TR_FILE_DIRECT_ALIGN + 1;
    }
    int rc = 0;
    for (int k = 0; k < n; k++)
        if (preadv_finish(f, &op[k]) != 0) rc = -1;
    return rc;
}

int tr_file_preadv(const tr_file *f, const tr_iov *iov, int cnt, uint64_t offset, void *scratch) {
    tr_readv_req r = {iov, cnt, offset};
    return tr_file_preadv_n(f, &r, 1, scratch);
}

void *tr_alloc_aligned(size_t n, size_t align) {
    return _aligned_malloc(n, align);
}

void tr_free_aligned(void *p) {
    _aligned_free(p);
}

static size_t page_size(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize > 0 ? (size_t)si.dwPageSize : 4096;
}

void *tr_pages_alloc(size_t n) {
    return n > 0 ? VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) : NULL;
}

void tr_pages_free(void *p, size_t n) {
    (void)n;
    if (p != NULL) VirtualFree(p, 0, MEM_RELEASE);
}

int tr_pages_release(void *p, size_t n) {
    size_t pg = page_size();
    uintptr_t lo = ((uintptr_t)p + pg - 1) / pg * pg, hi = ((uintptr_t)p + n) / pg * pg;
    if (hi <= lo) return 0;
    return VirtualFree((void *)lo, (SIZE_T)(hi - lo), MEM_DECOMMIT) ? 0 : -1;
}

int tr_pages_commit(void *p, size_t n) {
    size_t pg = page_size();
    uintptr_t lo = (uintptr_t)p / pg * pg, hi = ((uintptr_t)p + n + pg - 1) / pg * pg;
    if (hi <= lo) return 0;
    return VirtualAlloc((void *)lo, (SIZE_T)(hi - lo), MEM_COMMIT, PAGE_READWRITE) != NULL ? 0 : -1;
}

double tr_time_sec(void) {
    static LARGE_INTEGER freq;   /* global-ok: the performance counter frequency is fixed at boot */
    static int have_freq = 0;    /* global-ok: same value from every thread */
    if (!have_freq) {
        QueryPerformanceFrequency(&freq); /* idempotent: every thread computes the same value */
        have_freq = 1;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#  define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

void tr_wait_until(double t) {
    double left = t - tr_time_sec() - 0.001;
    if (left > 0) {
        /* Sleep's tick is 1-15.6 ms; a high-resolution timer (Windows 10 1803+) wakes within ~0.5 */
        HANDLE h = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        LARGE_INTEGER due;
        due.QuadPart = -(LONGLONG)(left * 1e7); /* relative, in 100 ns */
        if (h != NULL && SetWaitableTimer(h, &due, 0, NULL, NULL, FALSE))
            WaitForSingleObject(h, INFINITE);
        else if (left > 0.016)
            Sleep((DWORD)((left - 0.016) * 1000.0));
        if (h != NULL) CloseHandle(h);
    }
    while (tr_time_sec() < t) YieldProcessor();
}

int tr_thread_pin(unsigned group, const unsigned short *lcpus, int n, tr_affinity *prev) {
    if (lcpus == NULL || n <= 0) return -1;
    GROUP_AFFINITY want, had;
    ZeroMemory(&want, sizeof want);
    ZeroMemory(&had, sizeof had);
    want.Group = (WORD)group;
    for (int i = 0; i < n; i++) {
        if (lcpus[i] >= sizeof(KAFFINITY) * 8) return -1; /* a group never has more than 64 */
        want.Mask |= (KAFFINITY)1 << lcpus[i];
    }
    /* SetThreadGroupAffinity, not SetThreadAffinityMask: the latter only reaches the
     * thread's current processor group, so on a machine with more than 64 processors it
     * cannot name half of them (llama.cpp has that limit, see docs/UPSTREAM.md). */
    if (!SetThreadGroupAffinity(GetCurrentThread(), &want, &had)) return -1;
    if (prev != NULL) {
        prev->valid = 1;
        prev->group = had.Group;
        memset(prev->mask, 0, sizeof prev->mask);
        memcpy(prev->mask, &had.Mask, sizeof had.Mask);
    }
    return 0;
}

int tr_thread_affinity_restore(const tr_affinity *prev) {
    if (prev == NULL || !prev->valid) return 0;
    GROUP_AFFINITY back;
    ZeroMemory(&back, sizeof back);
    back.Group = (WORD)prev->group;
    memcpy(&back.Mask, prev->mask, sizeof back.Mask);
    if (back.Mask == 0) return -1;
    return SetThreadGroupAffinity(GetCurrentThread(), &back, NULL) ? 0 : -1;
}

int tr_mem_info(tr_meminfo *out) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) {
        out->total_bytes = 0;
        out->available_bytes = 0;
        return -1;
    }
    out->total_bytes = ms.ullTotalPhys;
    out->available_bytes = ms.ullAvailPhys;
    return 0;
}

#else /* POSIX: Linux and macOS */

void tr_stdout_binary(void) {
}

tr_file *tr_file_open(const char *path, char *err, size_t err_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_err(err, err_len, "open failed: %s", strerror(errno));
        return NULL;
    }
    tr_file *f = calloc(1, sizeof *f);
    if (f == NULL) {
        set_err(err, err_len, "out of memory");
        close(fd);
        return NULL;
    }
    f->fd = fd;
    f->direct = 0;
    return f;
}

tr_file *tr_file_open_direct(const char *path, char *err, size_t err_len) {
#if defined(__APPLE__)
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_err(err, err_len, "open failed: %s", strerror(errno));
        return NULL;
    }
    if (fcntl(fd, F_NOCACHE, 1) == -1) {
        set_err(err, err_len, "F_NOCACHE failed: %s", strerror(errno));
        close(fd);
        return NULL;
    }
#else
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        set_err(err, err_len, "open (O_DIRECT) failed: %s", strerror(errno));
        return NULL;
    }
#endif
    tr_file *f = calloc(1, sizeof *f);
    if (f == NULL) {
        set_err(err, err_len, "out of memory");
        close(fd);
        return NULL;
    }
    f->fd = fd;
    f->direct = 1;
    return f;
}

void tr_file_close(tr_file *f) {
    if (f == NULL) return;
    close(f->fd);
    free(f);
}

int64_t tr_file_size(const tr_file *f) {
    struct stat st;
    if (fstat(f->fd, &st) != 0) return -1;
    return (int64_t)st.st_size;
}

int64_t tr_file_alignment(const tr_file *f) {
    return f->direct ? TR_FILE_DIRECT_ALIGN : 1;
}

int tr_file_pread(const tr_file *f, void *buf, size_t n, uint64_t offset) {
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        ssize_t got = pread(f->fd, p, n, (off_t)offset);
        if (got < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        /* a direct handle's aligned range can run past the file's real end: the last, partial
         * sector then comes back short, which is not an error (platform.h) */
        if (got == 0) return f->direct ? 0 : -1; /* end of file before n bytes were read */
        /* short of a sector boundary: the real end (an unaligned retry is EINVAL on some filesystems) */
        if (f->direct && (size_t)got % TR_FILE_DIRECT_ALIGN != 0) return 0;
        p += got;
        offset += (uint64_t)got;
        n -= (size_t)got;
    }
    return 0;
}

size_t tr_file_preadv_scratch(size_t n) {
    (void)n;
    return 0;
}

int tr_file_preadv(const tr_file *f, const tr_iov *iov, int cnt, uint64_t offset, void *scratch) {
    (void)scratch;
    enum { BATCH = 64 }; /* pieces handed to one preadv; IOV_MAX is at least 1024 */
    int i = 0;
    size_t done = 0; /* bytes of iov[i] already read */
    while (i < cnt) {
        if (iov[i].n == done) {
            i++, done = 0;
            continue;
        }
        struct iovec v[BATCH];
        int k = 0;
        for (int j = i; j < cnt && k < BATCH; j++, k++) {
            v[k].iov_base = (unsigned char *)iov[j].base + (j == i ? done : 0);
            v[k].iov_len = iov[j].n - (j == i ? done : 0);
        }
        ssize_t got = preadv(f->fd, v, k, (off_t)offset);
        if (got < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        /* the real end, as tr_file_pread: an error on a buffered file, not on a direct one */
        if (got == 0) return f->direct ? 0 : -1;
        if (f->direct && (size_t)got % TR_FILE_DIRECT_ALIGN != 0) return 0;
        offset += (uint64_t)got;
        for (size_t left = (size_t)got; left > 0;) {
            size_t room = iov[i].n - done;
            if (left < room) {
                done += left;
                left = 0;
            } else {
                left -= room;
                i++, done = 0;
            }
        }
    }
    return 0;
}

size_t tr_file_preadv_n_scratch(size_t total, int n) {
    (void)total, (void)n;
    return 0;
}

/* one after the other: a preadv waits for its bytes, and nothing here puts two in flight */
int tr_file_preadv_n(const tr_file *f, const tr_readv_req *req, int n, void *scratch) {
    if (n < 1 || n > TR_FILE_PREADV_MAX) return -1;
    for (int k = 0; k < n; k++)
        if (tr_file_preadv(f, req[k].iov, req[k].cnt, req[k].offset, scratch) != 0) return -1;
    return 0;
}

void *tr_alloc_aligned(size_t n, size_t align) {
    void *p = NULL;
    if (posix_memalign(&p, align, n) != 0) return NULL;
    return p;
}

void tr_free_aligned(void *p) {
    free(p);
}

static size_t page_size(void) {
    long pg = sysconf(_SC_PAGESIZE);
    return pg > 0 ? (size_t)pg : 4096;
}

void *tr_pages_alloc(size_t n) {
    if (n == 0) return NULL;
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

void tr_pages_free(void *p, size_t n) {
    if (p != NULL) munmap(p, n);
}

int tr_pages_release(void *p, size_t n) {
    size_t pg = page_size();
    uintptr_t lo = ((uintptr_t)p + pg - 1) / pg * pg, hi = ((uintptr_t)p + n) / pg * pg;
    if (hi <= lo) return 0;
#if defined(__APPLE__)
    /* macOS keeps MADV_DONTNEED's pages; MADV_FREE hands them to the system */
    return madvise((void *)lo, hi - lo, MADV_FREE) == 0 ? 0 : -1;
#else
    return madvise((void *)lo, hi - lo, MADV_DONTNEED) == 0 ? 0 : -1;
#endif
}

int tr_pages_commit(void *p, size_t n) {
    (void)p;
    (void)n;
    return 0;
}

double tr_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void tr_wait_until(double t) {
    for (;;) {
        double left = t - tr_time_sec() - 0.001;
        if (left <= 0) break;
        struct timespec ts;
        ts.tv_sec = (time_t)left;
        ts.tv_nsec = (long)((left - (double)ts.tv_sec) * 1e9);
        nanosleep(&ts, NULL); /* a signal cuts it short: the loop sleeps the rest */
    }
    while (tr_time_sec() < t) {
    }
}

#if defined(__linux__)

int tr_thread_pin(unsigned group, const unsigned short *lcpus, int n, tr_affinity *prev) {
    (void)group; /* Linux has one flat cpu space */
    cpu_set_t set;
    if (lcpus == NULL || n <= 0) return -1;
    if (prev != NULL) {
        cpu_set_t had;
        prev->valid = 0;
        if (sched_getaffinity(0, sizeof had, &had) == 0 && sizeof had <= sizeof prev->mask) {
            memset(prev->mask, 0, sizeof prev->mask);
            memcpy(prev->mask, &had, sizeof had);
            prev->group = 0;
            prev->valid = 1;
        }
    }
    CPU_ZERO(&set);
    for (int i = 0; i < n; i++) {
        if (lcpus[i] >= (unsigned short)CPU_SETSIZE) return -1;
        CPU_SET((int)lcpus[i], &set);
    }
    return sched_setaffinity(0, sizeof set, &set) == 0 ? 0 : -1;
}

int tr_thread_affinity_restore(const tr_affinity *prev) {
    if (prev == NULL || !prev->valid) return 0;
    cpu_set_t back;
    if (sizeof back > sizeof prev->mask) return -1;
    memcpy(&back, prev->mask, sizeof back);
    return sched_setaffinity(0, sizeof back, &back) == 0 ? 0 : -1;
}

#else /* macOS and any other Unix: no way to pin a thread to one core */

int tr_thread_pin(unsigned group, const unsigned short *lcpus, int n, tr_affinity *prev) {
    (void)group;
    (void)lcpus;
    (void)n;
    if (prev != NULL) prev->valid = 0;
    return -1;
}

int tr_thread_affinity_restore(const tr_affinity *prev) {
    return (prev == NULL || !prev->valid) ? 0 : -1;
}

#endif

#if defined(__APPLE__)

int tr_mem_info(tr_meminfo *out) {
    uint64_t total = 0;
    size_t len = sizeof total;
    if (sysctlbyname("hw.memsize", &total, &len, NULL, 0) != 0) {
        out->total_bytes = 0;
        out->available_bytes = 0;
        return -1;
    }

    vm_size_t page_size = 0;
    mach_port_t host = mach_host_self();
    if (host_page_size(host, &page_size) != KERN_SUCCESS) {
        out->total_bytes = total;
        out->available_bytes = 0;
        return -1;
    }

    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) != KERN_SUCCESS) {
        out->total_bytes = total;
        out->available_bytes = 0;
        return -1;
    }

    uint64_t available = (uint64_t)(vm_stat.free_count + vm_stat.inactive_count +
                                     vm_stat.purgeable_count) * (uint64_t)page_size;
    out->total_bytes = total;
    out->available_bytes = available;
    return 0;
}

#else /* Linux */

int tr_mem_info(tr_meminfo *out) {
    out->total_bytes = 0;
    out->available_bytes = 0;

    FILE *fp = fopen("/proc/meminfo", "r");
    if (fp == NULL) return -1;

    char line[256];
    int have_total = 0, have_avail = 0;
    while (fgets(line, sizeof line, fp) != NULL) {
        unsigned long long kb;
        if (!have_total && sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
            out->total_bytes = kb * 1024ull;
            have_total = 1;
        } else if (!have_avail && sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            out->available_bytes = kb * 1024ull;
            have_avail = 1;
        }
        if (have_total && have_avail) break;
    }
    fclose(fp);

    if (!have_total || !have_avail) {
        out->total_bytes = 0;
        out->available_bytes = 0;
        return -1;
    }
    return 0;
}

#endif /* __APPLE__ */

#endif /* POSIX */

/* The handle is the library itself, cast: tr_lib is never defined, only pointed to. */
tr_lib *tr_lib_open(const char *name) {
#if defined(_WIN32)
    /* no dialog box when the library is missing or broken: the caller reads NULL */
    UINT old = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
    HMODULE h = LoadLibraryA(name);
    SetErrorMode(old);
    return (tr_lib *)h;
#else
    return (tr_lib *)dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void *tr_lib_sym(tr_lib *lib, const char *name) {
    if (lib == NULL) return NULL;
#if defined(_WIN32)
    FARPROC p = GetProcAddress((HMODULE)lib, name);
    void *out;
    memcpy(&out, &p, sizeof out); /* a function pointer as data, without the cast warning */
    return out;
#else
    return dlsym((void *)lib, name);
#endif
}

void tr_lib_close(tr_lib *lib) {
    if (lib == NULL) return;
#if defined(_WIN32)
    FreeLibrary((HMODULE)lib);
#else
    dlclose((void *)lib);
#endif
}

/* Bytes up to '\n'; the line without "\n" or "\r\n". NULL at end of input. */
static char *read_line_bytes(FILE *in, size_t *len) {
    size_t cap = 256, n = 0;
    char *buf = malloc(cap);
    if (buf == NULL) return NULL;
    int c;
    while ((c = fgetc(in)) != EOF && c != '\n') {
        if (n + 1 == cap) {
            char *grown = realloc(buf, cap * 2);
            if (grown == NULL) {
                free(buf);
                return NULL;
            }
            buf = grown;
            cap *= 2;
        }
        buf[n++] = (char)c;
    }
    if (c == EOF && n == 0) {
        free(buf);
        return NULL;
    }
    if (n > 0 && buf[n - 1] == '\r') n--;
    buf[n] = '\0';
    *len = n;
    return buf;
}

char *tr_stdin_line(size_t *len) {
#if defined(_WIN32)
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode)) {
        size_t cap = 256, n = 0;
        wchar_t *w = malloc(cap * sizeof(wchar_t));
        if (w == NULL) return NULL;
        for (;;) {
            if (cap - n < 128) {
                wchar_t *grown = realloc(w, cap * 2 * sizeof(wchar_t));
                if (grown == NULL) {
                    free(w);
                    return NULL;
                }
                w = grown;
                cap *= 2;
            }
            DWORD got = 0;
            if (!ReadConsoleW(h, w + n, (DWORD)(cap - n - 1), &got, NULL) || got == 0) break;
            n += got;
            if (w[n - 1] == L'\n') break;
        }
        if (n == 0 || w[0] == 0x1A) {   /* end of input, or Ctrl+Z */
            free(w);
            return NULL;
        }
        while (n > 0 && (w[n - 1] == L'\n' || w[n - 1] == L'\r')) n--;
        int need = n > 0 ? WideCharToMultiByte(CP_UTF8, 0, w, (int)n, NULL, 0, NULL, NULL) : 0;
        char *out = need >= 0 ? malloc((size_t)need + 1) : NULL;
        if (out == NULL || (need > 0 && WideCharToMultiByte(CP_UTF8, 0, w, (int)n, out, need, NULL, NULL) != need)) {
            free(out);
            free(w);
            return NULL;
        }
        out[need] = '\0';
        free(w);
        *len = (size_t)need;
        return out;
    }
#endif
    return read_line_bytes(stdin, len);
}

static int g_log_level = TR_LOG_INFO;   /* global-ok: one log level for the process */

void tr_log_set_level(int level) {
    g_log_level = level;
}

void tr_log(int level, const char *fmt, ...) {
    if (level > g_log_level) return;

    const char *prefix;
    switch (level) {
        case TR_LOG_ERROR: prefix = "[error] "; break;
        case TR_LOG_WARN:  prefix = "[warn] ";  break;
        case TR_LOG_INFO:  prefix = "[info] ";  break;
        default:           prefix = "[debug] "; break;
    }

    fputs(prefix, stderr);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
