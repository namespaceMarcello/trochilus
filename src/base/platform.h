/* platform.h — the only place that knows which operating system it runs on.
 *
 * Files are read with positional reads into caller buffers, never mapped: a
 * mapping keeps the whole model resident (see docs/ARCHITETTURA.md). Every
 * function here is safe to call from any thread unless stated otherwise. */
#ifndef TR_PLATFORM_H
#define TR_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

typedef struct tr_file tr_file;

/* Opens `path` (UTF-8, also on Windows) read-only. Returns NULL on failure and,
 * if err is not NULL, writes a message into err[0..err_len). */
tr_file *tr_file_open(const char *path, char *err, size_t err_len);
void tr_file_close(tr_file *f);
/* Size in bytes, or -1 on error. */
int64_t tr_file_size(const tr_file *f);
/* Reads exactly n bytes starting at offset into buf, looping over short reads.
 * Returns 0 on success, -1 on error or end of file before n bytes.
 * Positional: concurrent calls on the same tr_file are allowed. */
int tr_file_pread(const tr_file *f, void *buf, size_t n, uint64_t offset);

/* Aligned allocation; align is a power of two >= sizeof(void *). Contents are
 * uninitialized. NULL on failure. Free with tr_free_aligned. */
void *tr_alloc_aligned(size_t n, size_t align);
void tr_free_aligned(void *p);

/* Monotonic clock in seconds. */
double tr_time_sec(void);

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

#endif
