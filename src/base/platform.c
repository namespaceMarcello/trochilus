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
#  include <errno.h>
#  include <fcntl.h>
#  include <sched.h>
#  include <time.h>
#  include <unistd.h>
#  include <sys/stat.h>
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

    tr_file *f = malloc(sizeof *f);
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

    tr_file *f = malloc(sizeof *f);
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

void *tr_alloc_aligned(size_t n, size_t align) {
    return _aligned_malloc(n, align);
}

void tr_free_aligned(void *p) {
    _aligned_free(p);
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
    tr_file *f = malloc(sizeof *f);
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
    tr_file *f = malloc(sizeof *f);
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

void *tr_alloc_aligned(size_t n, size_t align) {
    void *p = NULL;
    if (posix_memalign(&p, align, n) != 0) return NULL;
    return p;
}

void tr_free_aligned(void *p) {
    free(p);
}

double tr_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
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
