/* serve.c — `trochilus serve`: one process keeps the model, its pool and its expert store alive
 * between commands, so a second `run` or `chat` starts with the experts the first one left in RAM
 * instead of an empty store that rereads the model from disk (docs/MEASUREMENTS.md question 49:
 * 4.7 s of a 2048-token prompt under a partial budget, all of it disk).
 *
 * The client is the ordinary command line: `run`, `generate`, `logits` and `chat` look for a server
 * first (TR_SERVER=0: never) and, when one answers, have it run the very same command with the same
 * arguments, in the client's working directory, on the client's own streams: what reaches stdout
 * and stderr is what the command prints run alone, byte for byte (tests/test_serve.c compares
 * them). No server: the command runs in its own process, as before.
 *
 * One request at a time: the engine has one pool per process (docs/LESSONS.md #63), and a second
 * client waits for the first. The model is kept between requests and reused when the next one asks
 * for the same file, budget and threads; anything else loads anew. What a command changes on the
 * model (the decode width, an expert mask) is put back when it ends. After `--idle` minutes without
 * a request the server exits and gives its memory back. The server reads the environment it was
 * started with (TR_EXPERT_BUDGET_MIB, TR_EXPERT_DIRECT, ...), not the client's.
 *
 * The streams. POSIX: the client hands the server its own stdout and stderr (SCM_RIGHTS), and the
 * read end of a pipe it feeds from its stdin (the client keeps reading its terminal in the
 * foreground, so job control never stops a server started in the background of the same one).
 * Windows: the client creates three pipes, the server opens them as its standard streams for the
 * command, and the client copies between them and its own (stdin line by line through
 * tr_stdin_line, so a console's UTF-16 arrives as UTF-8).
 *
 * Endpoint: \\.\pipe\trochilus-<user> on Windows (remote clients refused), <dir>/trochilus.sock on
 * POSIX, <dir> = $XDG_RUNTIME_DIR or /tmp/trochilus-<uid> (mode 0700 and ours, or refused).
 * TR_SERVER_NAME replaces "trochilus" (the tests run servers of their own). */
#if defined(__linux__)
#define _GNU_SOURCE /* SCM_RIGHTS and the CMSG_* macros under -std=c11 */
#endif
#include "serve.h"
#include "bar.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../base/platform.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <errno.h>
#include <fcntl.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#define SERVE_MAGIC 0x31435254u        /* "TRC1" */
#define SERVE_MAX_FRAME (1u << 20)
#define SERVE_MAX_ARGS 1024

enum { REQ_RUN = 1, REQ_STOP = 2, REQ_STATUS = 3 };

/* ---------------------------------------------------------------- the kept pool and model */

static void file_stamp(const char *path, char *out, size_t n);

typedef struct {
    int serving;                 /* this process is a server: pools and models are kept */
    char key[4200];              /* full path of the kept model's file, its size and mtime: a file
                                  * converted again at the same path is loaded again */
    char build[64];              /* build_stamp of the server */
    uint64_t budget;
    int64_t threads;
    tr_pool *pool;
    tr_model *model;
    tr_experts_stats base;       /* the store's counters when the current request took the model */
    uint64_t loads, requests;
    int64_t idle_minutes;
    int in_request;              /* 1 while dispatch() runs a forwarded command (serve_one): a load
                                  * here must never draw the bar (finding 4, review) -- the server
                                  * holds the client's tty on POSIX and its own TR_BAR/NO_COLOR/TERM
                                  * say nothing about the client's terminal. 0 around the server's
                                  * own startup load, which keeps the normal rule. */
} serve_state;

static serve_state g_serve; /* global-ok: a server process keeps one pool and one model (LESSONS #63) */

static void full_path(const char *path, char *out, size_t n) {
#ifdef _WIN32
    /* the name the disk gives the file: one file named with another case ("Desktop", "desktop"),
     * a short 8.3 name or through a junction is one model, not a reload (docs/LESSONS.md #140) */
    wchar_t wp[2048], wf[2048];
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 2048) > 0) {
        HANDLE h = CreateFileW(wp, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                               OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            DWORD k = GetFinalPathNameByHandleW(h, wf, 2048, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
            CloseHandle(h);
            const wchar_t *s = wf;
            if (k > 4 && k < 2048 && wcsncmp(wf, L"\\\\?\\", 4) == 0 && wf[5] == L':') s = wf + 4;
            if (k > 0 && k < 2048 && WideCharToMultiByte(CP_UTF8, 0, s, -1, out, (int)n, NULL, NULL) > 0) return;
        }
    }
    char *full = _fullpath(NULL, path, 0);
#else
    char *full = realpath(path, NULL);
#endif
    snprintf(out, n, "%s", full != NULL ? full : path);
    free(full);
}

static void drop_model(void) {
    if (g_serve.model != NULL) tr_model_free(g_serve.model);
    g_serve.model = NULL;
    g_serve.key[0] = 0;
}

tr_pool *app_pool(int64_t n_threads) {
    if (!g_serve.serving) return tr_pool_create((int)n_threads);
    if (g_serve.pool != NULL && g_serve.threads == n_threads) return g_serve.pool;
    drop_model();   /* a model runs on the pool it was loaded with */
    if (g_serve.pool != NULL) tr_pool_destroy(g_serve.pool);
    g_serve.pool = tr_pool_create((int)n_threads);
    g_serve.threads = n_threads;
    return g_serve.pool;
}

/* The decode's attention on the GPU when the machine has one (src/backend/gpu_attn.h): the same
 * bits, so it is not a mode, only a faster road. TR_GPU=0 keeps it on the CPU; unset, no device or
 * no driver means nothing happens and nothing is said; TR_GPU=1 asks for it and says on stderr why
 * it could not open one. In a server, the server's own environment decides. */
static tr_model *with_gpu(tr_model *m) {
    const char *env = getenv("TR_GPU");
    if (m == NULL || (env != NULL && strcmp(env, "0") == 0)) return m;
    char why[256] = "";
    int asked = env != NULL && strcmp(env, "1") == 0;
    if (tr_model_set_gpu(m, 1, why, sizeof why) != 0) tr_log(asked ? TR_LOG_WARN : TR_LOG_DEBUG, "gpu: none (%s)", why);
    return m;
}

tr_model *app_model(const char *path, tr_pool *pool, uint64_t expert_budget, char *err, size_t err_len) {
    if (!g_serve.serving) return with_gpu(tr_bar_load(path, pool, expert_budget, err, err_len));
    char full[4096], stamp[64], key[sizeof g_serve.key];
    full_path(path, full, sizeof full);
    file_stamp(path, stamp, sizeof stamp);
    snprintf(key, sizeof key, "%s|%s", full, stamp);
    if (g_serve.model == NULL || strcmp(key, g_serve.key) != 0 || expert_budget != g_serve.budget) {
        drop_model();
        if (g_serve.in_request) {
            /* never draw for a load done while serving a forwarded command (finding 4): a bar
             * forced off, whatever this process's own TR_BAR/NO_COLOR/TERM/tty say. */
            tr_bar b;
            tr_bar_init(&b, 0, NULL, NULL, NULL, NULL);
            g_serve.model = with_gpu(tr_bar_load_with(&b, path, pool, expert_budget, err, err_len));
        } else {
            g_serve.model = with_gpu(tr_bar_load(path, pool, expert_budget, err, err_len));
        }
        if (g_serve.model == NULL) return NULL;
        snprintf(g_serve.key, sizeof g_serve.key, "%s", key);
        g_serve.budget = expert_budget;
        g_serve.loads++;
    }
    if (tr_model_expert_stats(g_serve.model, &g_serve.base) != 0) memset(&g_serve.base, 0, sizeof g_serve.base);
    return g_serve.model;
}

void app_release(tr_model *model, tr_pool *pool) {
    if (!g_serve.serving) {
        if (model != NULL) tr_model_free(model);
        if (pool != NULL) tr_pool_destroy(pool);
        return;
    }
    if (model != NULL) {        /* kept: what the command set on it goes back */
        tr_model_set_expert_mask(model, NULL);
        tr_model_set_decode_threads(model, 0);
    }
}

int app_expert_stats(const tr_model *model, tr_experts_stats *out) {
    if (tr_model_expert_stats(model, out) != 0) return -1;
    if (g_serve.serving && model == g_serve.model) {
        out->hits -= g_serve.base.hits;
        out->misses -= g_serve.base.misses;
        out->evictions -= g_serve.base.evictions;
        out->bytes_read -= g_serve.base.bytes_read;
        out->read_sec -= g_serve.base.read_sec;
        out->prefetched -= g_serve.base.prefetched;
        out->prefetch_wait_sec -= g_serve.base.prefetch_wait_sec;
    }
    return 0;
}

/* ---------------------------------------------------------------- connections and frames */

typedef struct {
#ifdef _WIN32
    HANDLE h;
    HANDLE ev;                   /* not NULL: h is overlapped (the server's end of the endpoint) */
#else
    int fd;
#endif
} conn;

/* all n bytes, or -1 */
static int conn_io(conn *c, void *buf, size_t n, int writing) {
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
#ifdef _WIN32
        DWORD chunk = n > (1u << 20) ? (1u << 20) : (DWORD)n, got = 0;
        BOOL ok;
        if (c->ev != NULL) {
            OVERLAPPED ov;
            memset(&ov, 0, sizeof ov);
            ov.hEvent = c->ev;
            ok = writing ? WriteFile(c->h, p, chunk, NULL, &ov) : ReadFile(c->h, p, chunk, NULL, &ov);
            if (!ok && GetLastError() == ERROR_IO_PENDING) ok = TRUE;
            if (ok) ok = GetOverlappedResult(c->h, &ov, &got, TRUE);
        } else {
            ok = writing ? WriteFile(c->h, p, chunk, &got, NULL) : ReadFile(c->h, p, chunk, &got, NULL);
        }
        if (!ok || got == 0) return -1;
#else
        ssize_t got = writing ? write(c->fd, p, n) : read(c->fd, p, n);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return -1;
#endif
        p += got;
        n -= (size_t)got;
    }
    return 0;
}

typedef struct {
    unsigned char *p;
    size_t n, cap;
    int bad;                     /* out of memory */
} msg;

static void put(msg *m, const void *d, size_t n) {
    if (m->bad) return;
    if (m->n + n > m->cap) {
        size_t cap = m->cap ? m->cap : 256;
        while (cap < m->n + n) cap *= 2;
        unsigned char *q = (unsigned char *)realloc(m->p, cap);
        if (q == NULL) {
            m->bad = 1;
            return;
        }
        m->p = q;
        m->cap = cap;
    }
    memcpy(m->p + m->n, d, n);
    m->n += n;
}

static void put_u32(msg *m, uint32_t v) {
    unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    put(m, b, 4);
}

static void put_str(msg *m, const char *s) {
    size_t n = strlen(s);
    put_u32(m, (uint32_t)n);
    put(m, s, n);
}

static uint32_t le32(const unsigned char *b) {
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

/* one frame: its length, then its bytes */
static int send_frame(conn *c, msg *m) {
    unsigned char len[4] = {(unsigned char)m->n, (unsigned char)(m->n >> 8), (unsigned char)(m->n >> 16),
                            (unsigned char)(m->n >> 24)};
    if (m->bad || m->n > SERVE_MAX_FRAME) return -1;
    return conn_io(c, len, 4, 1) == 0 && conn_io(c, m->p, m->n, 1) == 0 ? 0 : -1;
}

static int recv_frame(conn *c, msg *m) {
    unsigned char len[4];
    if (conn_io(c, len, 4, 0) != 0) return -1;
    uint32_t n = le32(len);
    if (n > SERVE_MAX_FRAME) return -1;
    unsigned char *q = (unsigned char *)realloc(m->p, n > 0 ? n : 1);
    if (q == NULL) return -1;
    m->p = q;
    m->cap = n > 0 ? n : 1;
    m->n = 0;
    if (n > 0 && conn_io(c, m->p, n, 0) != 0) return -1;
    m->n = n;
    return 0;
}

typedef struct {
    const unsigned char *p;
    size_t n, at;
    int bad;
} reader;

static uint32_t get_u32(reader *r) {
    if (r->bad || r->n - r->at < 4) {
        r->bad = 1;
        return 0;
    }
    uint32_t v = le32(r->p + r->at);
    r->at += 4;
    return v;
}

/* a string, NUL-terminated, malloc'd; NULL (and r->bad) when the frame is short or memory runs out */
static char *get_str(reader *r) {
    uint32_t n = get_u32(r);
    if (r->bad || n > r->n - r->at) {
        r->bad = 1;
        return NULL;
    }
    char *s = (char *)malloc((size_t)n + 1);
    if (s == NULL) {
        r->bad = 1;
        return NULL;
    }
    memcpy(s, r->p + r->at, n);
    s[n] = 0;
    r->at += n;
    return s;
}

/* the answer to a request: a code, and a text (an error, or the status line) */
static int send_reply(conn *c, int32_t code, const char *text) {
    msg m = {0};
    put_u32(&m, SERVE_MAGIC);
    put_u32(&m, (uint32_t)code);
    put_str(&m, text);
    int rc = send_frame(c, &m);
    free(m.p);
    return rc;
}

/* -1 when the frame is not a reply; *text malloc'd */
static int recv_reply(conn *c, int32_t *code, char **text) {
    msg m = {0};
    int rc = -1;
    if (recv_frame(c, &m) == 0) {
        reader r = {m.p, m.n, 0, 0};
        uint32_t magic = get_u32(&r);
        *code = (int32_t)get_u32(&r);
        *text = get_str(&r);
        if (!r.bad && magic == SERVE_MAGIC) rc = 0;
        else {                  /* the caller frees *text too: never hand it a freed one (#138) */
            free(*text);
            *text = NULL;
        }
    }
    free(m.p);
    return rc;
}

/* ---------------------------------------------------------------- the endpoint */

static int endpoint_name(char *out, size_t n) {
    const char *name = getenv("TR_SERVER_NAME");
    if (name == NULL || name[0] == 0) name = "trochilus";
    size_t len = strlen(name);
    if (len > 64) return -1;
    for (size_t i = 0; i < len; i++) {
        char ch = name[i];
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' ||
              ch == '_' || ch == '.'))
            return -1;
    }
#ifdef _WIN32
    char user[80] = "user";
    const char *u = getenv("USERNAME");
    size_t k = 0;
    for (; u != NULL && u[k] != 0 && k + 1 < sizeof user; k++) {
        char ch = u[k];
        user[k] = ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) ? ch : '_';
    }
    if (k > 0) user[k] = 0;
    return snprintf(out, n, "\\\\.\\pipe\\%s-%s", name, user) < (int)n ? 0 : -1;
#else
    char dir[512];
    const char *run = getenv("XDG_RUNTIME_DIR");
    if (run != NULL && run[0] == '/') {
        snprintf(dir, sizeof dir, "%s", run);
    } else {
        snprintf(dir, sizeof dir, "/tmp/trochilus-%lu", (unsigned long)getuid());
        mkdir(dir, 0700);
    }
    struct stat st;
    if (lstat(dir, &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != getuid() || (st.st_mode & 077) != 0)
        return -1;      /* somebody else's, or open to others: no endpoint there */
    int k = snprintf(out, n, "%s/%s.sock", dir, name);
    return k > 0 && (size_t)k < n && (size_t)k < sizeof(((struct sockaddr_un *)0)->sun_path) ? 0 : -1;
#endif
}

#ifdef _WIN32
static wchar_t *wide(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    wchar_t *w = n > 0 ? (wchar_t *)malloc((size_t)n * sizeof(wchar_t)) : NULL;
    if (w != NULL && MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n) != n) {
        free(w);
        w = NULL;
    }
    return w;
}

static char *narrow(const wchar_t *w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char *s = n > 0 ? (char *)malloc((size_t)n) : NULL;
    if (s != NULL && WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) != n) {
        free(s);
        s = NULL;
    }
    return s;
}
#endif

/* this process's working directory, malloc'd UTF-8 */
static char *cwd_get(void) {
#ifdef _WIN32
    wchar_t *w = _wgetcwd(NULL, 0);
    char *s = w != NULL ? narrow(w) : NULL;
    free(w);
    return s;
#else
    return getcwd(NULL, 0);
#endif
}

static int cwd_set(const char *dir) {
#ifdef _WIN32
    wchar_t *w = wide(dir);
    int rc = w != NULL && _wchdir(w) == 0 ? 0 : -1;
    free(w);
    return rc;
#else
    return chdir(dir);
#endif
}

/* "size:mtime" of a file, "" when it cannot be read */
static void file_stamp(const char *path, char *out, size_t n) {
    out[0] = 0;
#ifdef _WIN32
    wchar_t *w = wide(path);
    struct _stat64 st;
    if (w != NULL && _wstat64(w, &st) == 0) snprintf(out, n, "%lld:%lld", (long long)st.st_size, (long long)st.st_mtime);
    free(w);
#else
    struct stat st;
    if (stat(path, &st) == 0) snprintf(out, n, "%lld:%lld", (long long)st.st_size, (long long)st.st_mtime);
#endif
}

/* This executable's stamp. A client of another build (rebuilt since the server started) must not
 * have its command run by the old code: the server declines and the client runs it itself. */
static void build_stamp(char *out, size_t n) {
    out[0] = 0;
#ifdef _WIN32
    wchar_t w[1024];
    DWORD k = GetModuleFileNameW(NULL, w, 1024);
    struct _stat64 st;
    if (k > 0 && k < 1024 && _wstat64(w, &st) == 0)
        snprintf(out, n, "%lld:%lld", (long long)st.st_size, (long long)st.st_mtime);
#elif defined(__linux__)
    file_stamp("/proc/self/exe", out, n);
#elif defined(__APPLE__)
    char p[1024];
    uint32_t size = sizeof p;
    if (_NSGetExecutablePath(p, &size) == 0) file_stamp(p, out, n);
#endif
}

/* ---------------------------------------------------------------- the server's endpoint */

typedef struct {
#ifdef _WIN32
    HANDLE h, ev;
#else
    int fd;
    char path[600];
#endif
} listener;

#ifndef _WIN32
/* endpoint_name keeps the path shorter than sun_path */
static void unix_address(struct sockaddr_un *a, const char *path) {
    size_t n = strlen(path);
    memset(a, 0, sizeof *a);
    a->sun_family = AF_UNIX;
    if (n >= sizeof a->sun_path) n = sizeof a->sun_path - 1;
    memcpy(a->sun_path, path, n);
}
#endif

static int listen_open(listener *L, const char *name, char *err, size_t err_len) {
#ifdef _WIN32
    wchar_t *w = wide(name);
    L->ev = CreateEventW(NULL, TRUE, FALSE, NULL);
    L->h = w != NULL && L->ev != NULL
               ? CreateNamedPipeW(w, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
                                  PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 65536,
                                  65536, 0, NULL)
               : INVALID_HANDLE_VALUE;
    DWORD e = GetLastError();
    free(w);
    if (L->h == INVALID_HANDLE_VALUE) {
        if (L->ev != NULL) CloseHandle(L->ev);
        /* the first instance is taken: ACCESS_DENIED by the documentation, PIPE_BUSY when it is
         * also the only one allowed */
        snprintf(err, err_len,
                 e == ERROR_ACCESS_DENIED || e == ERROR_PIPE_BUSY ? "a server is already running (%s)"
                                                                  : "cannot listen on %s (error %lu)",
                 name, (unsigned long)e);
        return -1;
    }
    return 0;
#else
    struct sockaddr_un a;
    unix_address(&a, name);
    memcpy(L->path, a.sun_path, sizeof a.sun_path);
    L->fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (L->fd < 0) {
        snprintf(err, err_len, "socket: %s", strerror(errno));
        return -1;
    }
    if (bind(L->fd, (struct sockaddr *)&a, sizeof a) != 0) {
        int probe = errno == EADDRINUSE ? socket(AF_UNIX, SOCK_STREAM, 0) : -1;
        int alive = probe >= 0 && connect(probe, (struct sockaddr *)&a, sizeof a) == 0;
        if (probe >= 0) close(probe);
        /* a socket file nobody answers on is what a server that died leaves behind */
        if (probe < 0 || alive || unlink(name) != 0 || bind(L->fd, (struct sockaddr *)&a, sizeof a) != 0) {
            snprintf(err, err_len, alive ? "a server is already running (%s)" : "cannot listen on %s", name);
            close(L->fd);
            return -1;
        }
    }
    if (listen(L->fd, 8) != 0) {
        snprintf(err, err_len, "listen: %s", strerror(errno));
        close(L->fd);
        unlink(name);
        return -1;
    }
    return 0;
#endif
}

/* 1: a client, in *c; 0: nobody came for timeout_ms (<= 0: wait forever); -1: an error */
static int listen_accept(listener *L, int64_t timeout_ms, conn *c) {
#ifdef _WIN32
    for (;;) {
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        ov.hEvent = L->ev;
        ResetEvent(L->ev);
        if (!ConnectNamedPipe(L->h, &ov)) {
            DWORD e = GetLastError(), got;
            if (e == ERROR_NO_DATA) {   /* a client came and went before we looked: the next one */
                DisconnectNamedPipe(L->h);
                continue;
            }
            if (e == ERROR_IO_PENDING) {
                DWORD wait = timeout_ms <= 0 ? INFINITE : timeout_ms > 0x7FFFFFFF ? 0x7FFFFFFF : (DWORD)timeout_ms;
                if (WaitForSingleObject(L->ev, wait) == WAIT_TIMEOUT) {
                    CancelIo(L->h);
                    /* a client that connected while the wait was being cancelled is still served */
                    if (!GetOverlappedResult(L->h, &ov, &got, TRUE)) return 0;
                } else if (!GetOverlappedResult(L->h, &ov, &got, FALSE)) {
                    return -1;
                }
            } else if (e != ERROR_PIPE_CONNECTED) {
                return -1;
            }
        }
        c->h = L->h;
        c->ev = L->ev;
        return 1;
    }
#else
    for (;;) {
        struct pollfd p = {L->fd, POLLIN, 0};
        int wait = timeout_ms <= 0 ? -1 : timeout_ms > INT_MAX ? INT_MAX : (int)timeout_ms;
        int r = poll(&p, 1, wait);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return -1;
        if (r == 0) return 0;
        c->fd = accept(L->fd, NULL, NULL);
        if (c->fd < 0 && (errno == EINTR || errno == ECONNABORTED)) continue;
        return c->fd >= 0 ? 1 : -1;
    }
#endif
}

/* the client is done with: what was written reaches it, then the endpoint takes the next */
static void listen_hangup(listener *L, conn *c) {
#ifdef _WIN32
    (void)c;
    FlushFileBuffers(L->h);
    DisconnectNamedPipe(L->h);
#else
    (void)L;
    close(c->fd);
#endif
}

static void listen_close(listener *L) {
#ifdef _WIN32
    CloseHandle(L->h);
    CloseHandle(L->ev);
#else
    close(L->fd);
    unlink(L->path);
#endif
}

/* ---------------------------------------------------------------- the command's streams */

/* The server's own 0, 1, 2 while a command runs on the client's. On Windows the standard handles
 * follow the descriptors (SetStdHandle after every _dup2): tr_stdin_line asks GetStdHandle. */
typedef struct {
    int fd[3];
} saved_std;

#ifdef _WIN32
static const DWORD std_ids[3] = {STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
#endif

/* takes the client's streams as 0, 1, 2 (the handles or descriptors in `in`, owned from here on);
 * -1 and nothing changed when one cannot be taken */
#ifdef _WIN32
static int std_take(HANDLE in[3], saved_std *s) {
    int fds[3];
    /* text mode, as a process's own streams start: without _O_TEXT the CRT opens them binary, and
     * a command that prints text (generate) lost its "\r" through the server (tests/test_serve.c) */
    const int modes[3] = {_O_RDONLY | _O_TEXT, _O_WRONLY | _O_TEXT, _O_WRONLY | _O_TEXT};
    for (int k = 0; k < 3; k++) {
        fds[k] = _open_osfhandle((intptr_t)in[k], modes[k]);
        if (fds[k] < 0) {
            for (int j = 0; j < k; j++) _close(fds[j]);
            for (int j = k; j < 3; j++) CloseHandle(in[j]);
            return -1;
        }
    }
    fflush(stdout);
    fflush(stderr);
    for (int k = 0; k < 3; k++) {
        s->fd[k] = _dup(k);
        _dup2(fds[k], k);
        _close(fds[k]);
        SetStdHandle(std_ids[k], (HANDLE)_get_osfhandle(k));
    }
    clearerr(stdin);
    return 0;
}
#else
static int std_take(int in[3], saved_std *s) {
    fflush(stdout);
    fflush(stderr);
    for (int k = 0; k < 3; k++) {
        s->fd[k] = dup(k);
        dup2(in[k], k);
        close(in[k]);
    }
    clearerr(stdin);
    return 0;
}
#endif

/* gives the server its own streams back; closing the client's ends tells the client the command
 * printed everything */
static void std_restore(saved_std *s) {
    fflush(stdout);
    fflush(stderr);
    for (int k = 0; k < 3; k++) {
#ifdef _WIN32
        if (s->fd[k] >= 0) {
            _dup2(s->fd[k], k);
            _close(s->fd[k]);
        } else {
            _close(k);
        }
        /* the handle the descriptor holds now: the one saved before was closed by _dup2 */
        SetStdHandle(std_ids[k], (HANDLE)_get_osfhandle(k));
#else
        if (s->fd[k] >= 0) {
            dup2(s->fd[k], k);
            close(s->fd[k]);
        } else {
            close(k);
        }
#endif
    }
    clearerr(stdin);
}

#ifndef _WIN32
static int send_fds(int sock, const int *fds, int n) {
    char byte = 'F';
    struct iovec iov = {&byte, 1};
    union {
        struct cmsghdr h;
        char buf[CMSG_SPACE(3 * sizeof(int))];
    } u;
    memset(&u, 0, sizeof u);
    struct msghdr mh;
    memset(&mh, 0, sizeof mh);
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    mh.msg_control = u.buf;
    mh.msg_controllen = CMSG_SPACE((size_t)n * sizeof(int));
    struct cmsghdr *c = CMSG_FIRSTHDR(&mh);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN((size_t)n * sizeof(int));
    memcpy(CMSG_DATA(c), fds, (size_t)n * sizeof(int));
    ssize_t k;
    do k = sendmsg(sock, &mh, 0);
    while (k < 0 && errno == EINTR);
    return k == 1 ? 0 : -1;
}

static int recv_fds(int sock, int *fds, int n) {
    char byte;
    struct iovec iov = {&byte, 1};
    union {
        struct cmsghdr h;
        char buf[CMSG_SPACE(3 * sizeof(int))];
    } u;
    struct msghdr mh;
    memset(&mh, 0, sizeof mh);
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    mh.msg_control = u.buf;
    mh.msg_controllen = sizeof u.buf;
    ssize_t k;
    do k = recvmsg(sock, &mh, 0);
    while (k < 0 && errno == EINTR);
    struct cmsghdr *c = k == 1 ? CMSG_FIRSTHDR(&mh) : NULL;
    if (c == NULL || c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) return -1;
    if (c->cmsg_len != CMSG_LEN((size_t)n * sizeof(int))) {
        /* descriptors the kernel already installed here, however many: closed, not leaked. The
         * buffer's padding holds a fourth (CMSG_SPACE rounds 12 bytes up to 16): the room is what
         * the buffer holds, not the 3 asked for (docs/LESSONS.md #137) */
        size_t got = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        size_t room = (sizeof u.buf - CMSG_LEN(0)) / sizeof(int);
        for (size_t i = 0; i < got && i < room; i++) {
            int fd;
            memcpy(&fd, CMSG_DATA(c) + i * sizeof(int), sizeof fd);
            close(fd);
        }
        return -1;
    }
    memcpy(fds, CMSG_DATA(c), (size_t)n * sizeof(int));
    return 0;
}
#endif

/* ---------------------------------------------------------------- the server */

static void status_line(char *out, size_t n) {
    int k = snprintf(out, n, "server: %llu requests, %llu loads, idle exit after %lld min; ",
                     (unsigned long long)g_serve.requests, (unsigned long long)g_serve.loads,
                     (long long)g_serve.idle_minutes);
    if (k < 0 || (size_t)k >= n) return;
    tr_experts_stats st;
    if (g_serve.model == NULL) {
        snprintf(out + k, n - (size_t)k, "no model");
    } else if (tr_model_expert_stats(g_serve.model, &st) == 0) {
        snprintf(out + k, n - (size_t)k, "model %s, %lld of %lld expert units in RAM", g_serve.key, (long long)st.n_slots,
                 (long long)st.n_units);
    } else {
        snprintf(out + k, n - (size_t)k, "model %s", g_serve.key);
    }
}

/* One request on a connected client. 1 when it asked the server to stop. */
static int serve_one(conn *c, serve_command_fn dispatch) {
    msg m = {0};
    char **argv = NULL, *cwd = NULL, *pipes[3] = {NULL, NULL, NULL};
    uint32_t argc = 0;
    int stop = 0;
    if (recv_frame(c, &m) != 0) goto done;
    reader r = {m.p, m.n, 0, 0};
    uint32_t magic = get_u32(&r), kind = get_u32(&r);
    get_u32(&r);    /* flags: none yet */
    argc = get_u32(&r);
    if (r.bad || magic != SERVE_MAGIC || argc > SERVE_MAX_ARGS) {
        send_reply(c, 2, "not a request of this version of trochilus");
        goto done;
    }
    if (kind == REQ_STOP) {
        send_reply(c, 0, "");
        stop = 1;
        goto done;
    }
    if (kind == REQ_STATUS) {
        char line[5000];
        status_line(line, sizeof line);
        send_reply(c, 0, line);
        goto done;
    }
    argv = (char **)calloc((size_t)argc + 1, sizeof(char *));
    if (argv == NULL) goto done;
    for (uint32_t i = 0; i < argc; i++) argv[i] = get_str(&r);
    cwd = get_str(&r);
    uint32_t n_pipes = get_u32(&r);
    for (uint32_t i = 0; i < n_pipes && i < 3; i++) pipes[i] = get_str(&r);
    char *build = get_str(&r);
    if (r.bad || kind != REQ_RUN || argc < 1 || n_pipes > 3) {
        free(build);
        send_reply(c, 2, "malformed request");
        goto done;
    }
    int other_build = build[0] != 0 && g_serve.build[0] != 0 && strcmp(build, g_serve.build) != 0;
    free(build);
    if (other_build) {
        send_reply(c, 3, "the server runs another build of trochilus");
        goto done;
    }

    saved_std saved;
#ifdef _WIN32
    HANDLE h[3] = {INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE};
    for (int k = 0; k < 3 && n_pipes == 3; k++) {
        wchar_t *w = wide(pipes[k]);
        if (w != NULL) h[k] = CreateFileW(w, k == 0 ? GENERIC_READ : GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
        free(w);
    }
    if (h[0] == INVALID_HANDLE_VALUE || h[1] == INVALID_HANDLE_VALUE || h[2] == INVALID_HANDLE_VALUE) {
        for (int k = 0; k < 3; k++)
            if (h[k] != INVALID_HANDLE_VALUE) CloseHandle(h[k]);
        send_reply(c, 1, "the server could not open the client's streams");
        goto done;
    }
#else
    int h[3];
    if (recv_fds(c->fd, h, 3) != 0) {
        send_reply(c, 1, "the server did not receive the client's streams");
        goto done;
    }
#endif
    char *home = cwd_get();
    if (home == NULL || cwd_set(cwd) != 0) {
#ifdef _WIN32
        for (int k = 0; k < 3; k++) CloseHandle(h[k]);
#else
        for (int k = 0; k < 3; k++) close(h[k]);
#endif
        send_reply(c, 1, "the server cannot enter the client's working directory");
        free(home);
        goto done;
    }
    if (std_take(h, &saved) != 0) {
        cwd_set(home);
        free(home);
        goto done;
    }
    if (send_reply(c, 0, "") != 0) {   /* the client left before its command began */
        std_restore(&saved);
        cwd_set(home);
        free(home);
        goto done;
    }
    double t0 = tr_time_sec();
    g_serve.in_request = 1;
    int rc = dispatch((int)argc, argv);
    g_serve.in_request = 0;
    std_restore(&saved);
    cwd_set(home);
    free(home);
    g_serve.requests++;
    fprintf(stderr, "serve: %s: exit %d in %.2f s\n", argv[0], rc, tr_time_sec() - t0);
    send_reply(c, rc, "");

done:
    if (argv != NULL)
        for (uint32_t i = 0; i < argc; i++) free(argv[i]);
    free(argv);
    free(cwd);
    for (int k = 0; k < 3; k++) free(pipes[k]);
    free(m.p);
    return stop;
}

int serve_run(const serve_config *cfg, serve_command_fn dispatch) {
    char name[600], err[700];
    if (endpoint_name(name, sizeof name) != 0) {
        fprintf(stderr, "serve: no endpoint (TR_SERVER_NAME takes letters, digits, '-', '_', '.'; on POSIX the "
                        "socket's directory must be yours and closed to others)\n");
        return 1;
    }
#ifndef _WIN32
    /* started with 0, 1 or 2 closed, the endpoint would land there and a client's stream be
     * dup2'ed over it */
    for (int k = 0; k < 3; k++) {
        if (fcntl(k, F_GETFD) >= 0) continue;
        int nul = open("/dev/null", O_RDWR);
        if (nul >= 0 && nul != k) {
            dup2(nul, k);
            close(nul);
        }
    }
#endif
    listener L;
    if (listen_open(&L, name, err, sizeof err) != 0) {
        fprintf(stderr, "serve: %s\n", err);
        return 1;
    }
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);   /* a client gone mid-command: its writes fail, the server lives */
#endif
    setvbuf(stdin, NULL, _IONBF, 0);   /* nothing one request typed may stay buffered for the next */
    memset(&g_serve, 0, sizeof g_serve);
    g_serve.serving = 1;
    build_stamp(g_serve.build, sizeof g_serve.build);
    g_serve.idle_minutes = cfg->idle_minutes;
    /* tests only: the idle time in milliseconds, so the idle exit is seen in seconds */
    const char *idle_ms_env = getenv("TR_SERVE_IDLE_MS");
    int64_t idle_ms = idle_ms_env != NULL && idle_ms_env[0] != 0 ? strtoll(idle_ms_env, NULL, 10)
                                                                  : cfg->idle_minutes * 60000;
    int rc = 0;
    if (cfg->model_path != NULL) {
        tr_pool *pool = app_pool(cfg->n_threads);
        tr_model *model = pool != NULL ? app_model(cfg->model_path, pool, cfg->expert_budget, err, sizeof err) : NULL;
        if (model == NULL) {
            fprintf(stderr, "serve: %s\n", pool == NULL ? "could not create thread pool" : err);
            rc = 1;
        }
    }
    if (rc == 0) {
        fprintf(stderr, "serve: listening on %s (%s); idle exit after %lld min\n", name,
                g_serve.model != NULL ? g_serve.key : "no model yet", (long long)cfg->idle_minutes);
        /* idle: no command run, since the start or the last one; --status and --stop are no
         * use, and a script polling --status must not keep an unused server (and its RAM) up */
        double last = tr_time_sec();
        for (;;) {
            conn c;
            int64_t left = idle_ms - (int64_t)((tr_time_sec() - last) * 1000.0);
            int a = idle_ms > 0 && left <= 0 ? 0 : listen_accept(&L, idle_ms > 0 ? left : 0, &c);
            if (a == 0) {
                fprintf(stderr, "serve: idle for %lld ms, exiting\n", (long long)idle_ms);
                break;
            }
            if (a < 0) {
                fprintf(stderr, "serve: the endpoint failed\n");
                rc = 1;
                break;
            }
            uint64_t before = g_serve.requests;
            int stop = serve_one(&c, dispatch);
            if (g_serve.requests != before) last = tr_time_sec();
            listen_hangup(&L, &c);
            if (stop) {
                fprintf(stderr, "serve: stopped\n");
                break;
            }
        }
    }
    listen_close(&L);
    drop_model();
    if (g_serve.pool != NULL) tr_pool_destroy(g_serve.pool);
    memset(&g_serve, 0, sizeof g_serve);
    return rc;
}

/* ---------------------------------------------------------------- the client */

/* 0 connected; -1 no server */
static int client_connect(conn *c) {
    char name[600];
    if (endpoint_name(name, sizeof name) != 0) return -1;
#ifdef _WIN32
    wchar_t *w = wide(name);
    if (w == NULL) return -1;
    int announced = 0;
    for (;;) {
        HANDLE h = CreateFileW(w, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            c->h = h;
            c->ev = NULL;
            free(w);
            return 0;
        }
        if (GetLastError() != ERROR_PIPE_BUSY) {
            free(w);
            return -1;
        }
        if (!announced) fprintf(stderr, "(the server is busy with another command: waiting)\n");
        announced = 1;
        WaitNamedPipeW(w, 5000);
    }
#else
    struct sockaddr_un a;
    unix_address(&a, name);
    c->fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (c->fd < 0) return -1;
    if (connect(c->fd, (struct sockaddr *)&a, sizeof a) != 0) {
        close(c->fd);
        return -1;
    }
    return 0;
#endif
}

static void client_close(conn *c) {
#ifdef _WIN32
    CloseHandle(c->h);
#else
    close(c->fd);
#endif
}

/* a request of `kind` with argv (none for stop and status) and, on Windows, the client's pipes */
static int send_request(conn *c, uint32_t kind, int argc, char **argv, char *const *pipes) {
    msg m = {0};
    put_u32(&m, SERVE_MAGIC);
    put_u32(&m, kind);
    put_u32(&m, 0);
    put_u32(&m, (uint32_t)argc);
    for (int i = 0; i < argc; i++) put_str(&m, argv[i]);
    char *cwd = argc > 0 ? cwd_get() : NULL;
    put_str(&m, cwd != NULL ? cwd : "");
    free(cwd);
    put_u32(&m, pipes != NULL ? 3 : 0);
    for (int k = 0; pipes != NULL && k < 3; k++) put_str(&m, pipes[k]);
    char build[64];
    build_stamp(build, sizeof build);
    put_str(&m, build);
    int rc = send_frame(c, &m);
    free(m.p);
    return rc;
}

#ifdef _WIN32
typedef struct {
    HANDLE from, to;
} pump;

static DWORD WINAPI pump_main(void *arg) {
    pump *p = (pump *)arg;
    char buf[16384];
    DWORD got, put_n;
    while (ReadFile(p->from, buf, sizeof buf, &got, NULL) && got > 0)
        if (!WriteFile(p->to, buf, got, &put_n, NULL)) break;
    return 0;
}

/* The client's stdin into the command's; at its end the pipe closes. A console line by line
 * through tr_stdin_line (UTF-16 in, UTF-8 out, ReadConsoleW: no lock of the CRT's); a file or a
 * pipe as its bytes, through ReadFile: fgetc would hold the CRT's stdin lock while it blocks, and
 * the exit of a client whose chat ended before its input would wait on that lock. */
static DWORD WINAPI stdin_main(void *arg) {
    HANDLE to = (HANDLE)arg, from = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode, put_n;
    if (from != NULL && from != INVALID_HANDLE_VALUE && GetConsoleMode(from, &mode)) {
        for (;;) {
            size_t n;
            char *line = tr_stdin_line(&n);
            if (line == NULL) break;
            int ok = (n == 0 || WriteFile(to, line, (DWORD)n, &put_n, NULL)) && WriteFile(to, "\n", 1, &put_n, NULL);
            free(line);
            if (!ok) break;
        }
    } else if (from != NULL && from != INVALID_HANDLE_VALUE) {
        char buf[4096];
        DWORD got;
        while (ReadFile(from, buf, sizeof buf, &got, NULL) && got > 0)
            if (!WriteFile(to, buf, got, &put_n, NULL)) break;
    }
    CloseHandle(to);
    return 0;
}
#endif

int serve_forward(int argc, char **argv, int reads_stdin, int *rc) {
    const char *off = getenv("TR_SERVER");
    if (off != NULL && strcmp(off, "0") == 0) return 0;
    conn c;
    if (client_connect(&c) != 0) return 0;
    int32_t code = 1;
    char *text = NULL;
    int ran = 0;                /* 0: nothing ran (the server declined, or left before it began) */
#ifdef _WIN32
    char names[3][700];
    char *pipes[3] = {names[0], names[1], names[2]};
    HANDLE h[3] = {INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE};
    char base[600];
    endpoint_name(base, sizeof base);
    int made = 1;
    for (int k = 0; k < 3; k++) {
        snprintf(names[k], sizeof names[k], "%s-%lu-%d", base, (unsigned long)GetCurrentProcessId(), k);
        wchar_t *w = wide(names[k]);
        if (w != NULL)
            h[k] = CreateNamedPipeW(w, (k == 0 ? PIPE_ACCESS_OUTBOUND : PIPE_ACCESS_INBOUND) | FILE_FLAG_FIRST_PIPE_INSTANCE,
                                    PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 65536, 65536, 0, NULL);
        free(w);
        if (h[k] == INVALID_HANDLE_VALUE) made = 0;
    }
    if (made && send_request(&c, REQ_RUN, argc, argv, pipes) == 0 && recv_reply(&c, &code, &text) == 0 && code == 0) {
        ran = 1;
        *rc = 1;
        /* the server opened all three before it answered */
        for (int k = 0; k < 3; k++) ConnectNamedPipe(h[k], NULL);
        HANDLE in_thread = NULL;
        if (reads_stdin) {
            in_thread = CreateThread(NULL, 0, stdin_main, h[0], 0, NULL);
            if (in_thread != NULL) h[0] = INVALID_HANDLE_VALUE;   /* the thread closes it */
        }
        if (h[0] != INVALID_HANDLE_VALUE) {
            CloseHandle(h[0]);
            h[0] = INVALID_HANDLE_VALUE;
        }
        pump err_pump = {h[2], GetStdHandle(STD_ERROR_HANDLE)}, out_pump = {h[1], GetStdHandle(STD_OUTPUT_HANDLE)};
        HANDLE err_thread = CreateThread(NULL, 0, pump_main, &err_pump, 0, NULL);
        pump_main(&out_pump);
        if (err_thread != NULL) {
            WaitForSingleObject(err_thread, INFINITE);
            CloseHandle(err_thread);
        } else {
            pump_main(&err_pump);
        }
        if (in_thread != NULL) {        /* the command is over: a read still waiting is not needed */
            CancelSynchronousIo(in_thread);
            CloseHandle(in_thread);
        }
        free(text);
        text = NULL;
        if (recv_reply(&c, &code, &text) != 0) fprintf(stderr, "error: the server stopped during the command\n");
        else *rc = code;
    }
    for (int k = 0; k < 3; k++)
        if (h[k] != INVALID_HANDLE_VALUE) CloseHandle(h[k]);
#else
    signal(SIGPIPE, SIG_IGN);
    int p[2];
    if (pipe(p) != 0) {
        client_close(&c);
        return 0;
    }
    int fds[3] = {p[0], 1, 2};
    if (send_request(&c, REQ_RUN, argc, argv, NULL) == 0 && send_fds(c.fd, fds, 3) == 0 &&
        recv_reply(&c, &code, &text) == 0 && code == 0) {
        ran = 1;
        *rc = 1;
        close(p[0]);
        p[0] = -1;
        if (!reads_stdin) {
            close(p[1]);
            p[1] = -1;
        }
        /* copy stdin into the command's until the command ends (its reply arrives) */
        int done = 0;
        while (!done) {
            struct pollfd pf[2] = {{c.fd, POLLIN, 0}, {0, POLLIN, 0}};
            int r = poll(pf, p[1] >= 0 ? 2 : 1, -1);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) break;
            /* POLLNVAL: no stdin at all (fd 0 closed), the end of it */
            if (p[1] >= 0 && (pf[1].revents & POLLNVAL)) {
                close(p[1]);
                p[1] = -1;
            } else if (p[1] >= 0 && (pf[1].revents & (POLLIN | POLLHUP | POLLERR))) {
                char buf[4096];
                ssize_t n = read(0, buf, sizeof buf);
                conn in = {p[1]};
                if (n <= 0 || conn_io(&in, buf, (size_t)n, 1) != 0) {
                    close(p[1]);
                    p[1] = -1;
                }
            }
            if (pf[0].revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) done = 1;
        }
        free(text);
        text = NULL;
        if (!done || recv_reply(&c, &code, &text) != 0) fprintf(stderr, "error: the server stopped during the command\n");
        else *rc = code;
    }
    if (p[0] >= 0) close(p[0]);
    if (p[1] >= 0) close(p[1]);
#endif
    free(text);
    client_close(&c);
    /* nothing ran: another build, a directory the server cannot enter, a server leaving at that
     * moment. The command runs here, as with no server at all. */
    return ran;
}

static int simple_request(uint32_t kind, int print) {
    conn c;
    if (client_connect(&c) != 0) {
        fprintf(stderr, "no server is running\n");
        return 1;
    }
    int32_t code = 1;
    char *text = NULL;
    int rc = 1;
    if (send_request(&c, kind, 0, NULL, NULL) == 0 && recv_reply(&c, &code, &text) == 0) {
        if (print) printf("%s\n", text);
        rc = code == 0 ? 0 : 1;
    } else {
        fprintf(stderr, "error: the server did not answer\n");
    }
    free(text);
    client_close(&c);
    return rc;
}

int serve_stop(void) {
    return simple_request(REQ_STOP, 0);
}

int serve_status(void) {
    return simple_request(REQ_STATUS, 1);
}
