/* platform.c — Windows, Linux and macOS implementations of platform.h.
 *
 * No third-party code: only libc and OS APIs (Win32, POSIX, Mach). Windows
 * uses CreateFileW opened with FILE_FLAG_OVERLAPPED so tr_file_pread can pass
 * a fresh OVERLAPPED (offset + private event) per call, which is what makes
 * concurrent positional reads on the same handle thread-safe: the offset never
 * goes through a shared file pointer. Linux and macOS use pread(2), which is
 * positional and thread-safe by definition. */

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
#else
#  include <errno.h>
#  include <fcntl.h>
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
        BOOL ok = ReadFile(f->handle, p, want, &got, &ov);
        if (!ok) {
            DWORD gle = GetLastError();
            if (gle == ERROR_IO_PENDING) {
                BOOL waited = GetOverlappedResult(f->handle, &ov, &got, TRUE);
                CloseHandle(ov.hEvent);
                if (!waited) return -1; /* includes ERROR_HANDLE_EOF */
            } else {
                CloseHandle(ov.hEvent);
                return -1; /* includes ERROR_HANDLE_EOF for a read that starts past EOF */
            }
        } else {
            CloseHandle(ov.hEvent);
        }

        if (got == 0) return -1; /* end of file before n bytes were read */
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
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    if (!have_freq) {
        QueryPerformanceFrequency(&freq); /* idempotent: every thread computes the same value */
        have_freq = 1;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
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

int tr_file_pread(const tr_file *f, void *buf, size_t n, uint64_t offset) {
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        ssize_t got = pread(f->fd, p, n, (off_t)offset);
        if (got < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (got == 0) return -1; /* end of file before n bytes were read */
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

static int g_log_level = TR_LOG_INFO;

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
