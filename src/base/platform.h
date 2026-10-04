/* platform.h — the only place that knows which operating system it runs on.
 *
 * Files are read with positional reads into caller buffers, never mapped: a
 * mapping keeps the whole model resident (see docs/ARCHITECTURE.md). Every
 * function here is safe to call from any thread unless stated otherwise. */
#ifndef TR_PLATFORM_H
#define TR_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

typedef struct tr_file tr_file;

/* Opens `path` (UTF-8, also on Windows) read-only. Returns NULL on failure and,
 * if err is not NULL, writes a message into err[0..err_len). */
tr_file *tr_file_open(const char *path, char *err, size_t err_len);
/* Like tr_file_open, but the operating system keeps no copy of what is read (FILE_FLAG_NO_BUFFERING,
 * O_DIRECT, F_NOCACHE). Every tr_file_pread on such a file must have offset, length AND buffer
 * address multiples of tr_file_alignment(). NULL when the file cannot be opened that way (some
 * filesystems refuse O_DIRECT), with a message in err if it is not NULL: the caller falls back to
 * tr_file_open. */
tr_file *tr_file_open_direct(const char *path, char *err, size_t err_len);
void tr_file_close(tr_file *f);
/* Size in bytes, or -1 on error. */
int64_t tr_file_size(const tr_file *f);
/* The alignment tr_file_pread needs on this handle: TR_FILE_DIRECT_ALIGN for one opened with
 * tr_file_open_direct, 1 for one opened with tr_file_open. */
int64_t tr_file_alignment(const tr_file *f);
#define TR_FILE_DIRECT_ALIGN 4096
/* Reads exactly n bytes starting at offset into buf, looping over short reads.
 * Returns 0 on success, -1 on error or end of file before n bytes -- except on a handle from
 * tr_file_open_direct, where a read whose aligned range runs past the real end of the file still
 * succeeds (the last, partial sector of the file is not an error): bytes past the real end of the
 * file are then left as whatever buf already held. A direct read that comes back short of a
 * sector boundary has met that end and the call returns there: asking for the rest from the
 * unaligned position it stopped at is refused on NTFS (docs/LESSONS.md #103). Positional:
 * concurrent calls on the same tr_file are allowed. */
int tr_file_pread(const tr_file *f, void *buf, size_t n, uint64_t offset);

/* One piece of a scattered read: n bytes into base. */
typedef struct tr_iov {
    void *base;
    size_t n;
} tr_iov;
/* Bytes of scratch a tr_file_preadv of n bytes needs (Windows lists every page of a direct read:
 * one FILE_SEGMENT_ELEMENT a page and a closing one); 0 where none is needed. */
size_t tr_file_preadv_scratch(size_t n);
/* The file's bytes from offset on into the pieces in order (iov[0] takes the first iov[0].n, then
 * iov[1], ...), with tr_file_pread's contract on the whole range: exact, or the direct file's real
 * end. On a direct file one request where the system allows it (ReadFileScatter, preadv): this
 * disk gives 1.9 GB/s to 2 MiB requests and 3.5 to 64 MiB ones, and a scattered request runs level
 * with a contiguous one (docs/MEASUREMENTS.md §The disk at its limit); a buffered Windows file takes
 * a read a piece. On a direct file every base and n is a multiple of tr_file_alignment and the
 * total under 2 GiB. scratch: tr_file_preadv_scratch(total) bytes, 8-byte aligned (NULL when 0).
 * Positional: concurrent calls with their own scratch are allowed. */
int tr_file_preadv(const tr_file *f, const tr_iov *iov, int cnt, uint64_t offset, void *scratch);

/* One request of tr_file_preadv_n: what one tr_file_preadv would read. */
typedef struct tr_readv_req {
    const tr_iov *iov;
    int cnt;
    uint64_t offset;
} tr_readv_req;
/* The most requests one tr_file_preadv_n takes. */
#define TR_FILE_PREADV_MAX 4
/* Bytes of scratch a tr_file_preadv_n of n requests and total bytes in all needs; 0 where none. */
size_t tr_file_preadv_n_scratch(size_t total, int n);
/* n tr_file_preadv's (1 <= n <= TR_FILE_PREADV_MAX), each with its contract. On a direct Windows file
 * every request is issued before the first is waited for: n in flight, where one gives this disk 3.27
 * GB/s at 68 MiB and two 3.52 (docs/MEASUREMENTS.md §The disk at its limit); elsewhere, and on a
 * buffered file, one after the other. 0 when all succeed; -1 when any fails, after every request
 * issued has ended (no buffer is written after the return). scratch: tr_file_preadv_n_scratch bytes,
 * 8-byte aligned (NULL when 0). Positional: concurrent calls with their own scratch are allowed. */
int tr_file_preadv_n(const tr_file *f, const tr_readv_req *req, int n, void *scratch);

/* Aligned allocation; align is a power of two >= sizeof(void *). Contents are
 * uninitialized. NULL on failure. Free with tr_free_aligned. */
void *tr_alloc_aligned(size_t n, size_t align);
void tr_free_aligned(void *p);

/* Memory whose pages go back to the system in part while the block stays (the expert store's slab: its
 * slots give their pages to a session's KV as it grows, src/memory/experts.h tr_experts_set_slots).
 * tr_pages_alloc: n bytes on whole pages, zero, NULL on failure; free with tr_pages_free(p, n).
 * tr_pages_release: the whole pages inside [p, p + n) given back, their bytes lost; 0, -1 on failure.
 * tr_pages_commit: every page [p, p + n) touches usable again, its bytes undefined; on Windows the
 * system commits them (-1 when its commit limit refuses), elsewhere they come back on first touch.
 * [p, p + n) lies inside one tr_pages_alloc block. */
void *tr_pages_alloc(size_t n);
void tr_pages_free(void *p, size_t n);
int tr_pages_release(void *p, size_t n);
int tr_pages_commit(void *p, size_t n);

/* Monotonic clock in seconds. */
double tr_time_sec(void);
/* Returns once tr_time_sec() has reached t (at once if it has): a high-resolution timer to about
 * 1 ms before t, then a spin. For measurement-only emulations (the expert store's slower disk,
 * TR_EXPERT_DISK_MBPS), never on the hot path. */
void tr_wait_until(double t);

/* Where a thread was allowed to run before it was pinned, so it can be put back.
 * Opaque: only tr_thread_pin and tr_thread_affinity_restore read it. */
typedef struct {
    int valid;
    unsigned short group;     /* Windows processor group */
    unsigned char mask[128];  /* the platform's own bitmap (KAFFINITY, cpu_set_t) */
} tr_affinity;

/* Pins the calling thread to the n logical processors listed in `lcpus`; `group` is the
 * Windows processor group they belong to and is ignored elsewhere. One processor ties the
 * thread down, a whole core's processors leave the scheduler the choice of sibling. When prev
 * is not NULL it receives the affinity the thread had, for tr_thread_affinity_restore.
 * Returns 0 if the pin was applied, -1 if the platform has no affinity or the call failed
 * (macOS: always -1). Only the calling thread is affected. */
int tr_thread_pin(unsigned group, const unsigned short *lcpus, int n, tr_affinity *prev);

/* Puts the calling thread back where `prev` says. 0 on success, -1 otherwise; a `prev`
 * that was never filled in (valid == 0) is a no-op and returns 0. */
int tr_thread_affinity_restore(const tr_affinity *prev);

typedef struct {
    uint64_t total_bytes;      /* physical RAM */
    uint64_t available_bytes;  /* what can be allocated now without swapping */
} tr_meminfo;
/* Returns 0 on success, -1 if the OS query failed (out is then zeroed). */
int tr_mem_info(tr_meminfo *out);

enum { TR_LOG_ERROR = 0, TR_LOG_WARN = 1, TR_LOG_INFO = 2, TR_LOG_DEBUG = 3 };
/* printf-style line to stderr, prefixed by the level; a newline is added.
 * Messages above the current level are dropped. Default level: TR_LOG_INFO. */
void tr_log(int level, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
void tr_log_set_level(int level);

/* stdout writes bytes as given: no "\n" -> "\r\n" translation on Windows (no-op elsewhere).
 * For commands whose output is data (token ids, generated text). */
void tr_stdout_binary(void);

/* One line of stdin as UTF-8, without its line ending (a Windows console is read as
 * UTF-16 and converted; a pipe or file is read as bytes). malloc'd, *len its length.
 * NULL at end of input (Ctrl+D, Ctrl+Z) or if memory ran out. */
char *tr_stdin_line(size_t *len);

/* A shared library opened at run time (LoadLibrary, dlopen): how the core reaches a GPU driver
 * without linking it, so a machine without one still starts. NULL when it is not there. */
typedef struct tr_lib tr_lib;
tr_lib *tr_lib_open(const char *name);
/* The address of an exported symbol, NULL if the library has none by that name. */
void *tr_lib_sym(tr_lib *lib, const char *name);
void tr_lib_close(tr_lib *lib);

#if defined(_WIN32)
#include <wchar.h>
/* Windows gives wmain its arguments in UTF-16: the same arguments as UTF-8 strings
 * (argv[argc] is NULL). NULL if memory ran out or an argument does not convert. */
char **tr_utf8_argv(int argc, wchar_t **wargv);
void tr_utf8_argv_free(int argc, char **argv);
/* Console output as UTF-8 (a Windows console starts in the OEM code page). */
void tr_console_utf8(void);
#endif

#endif
