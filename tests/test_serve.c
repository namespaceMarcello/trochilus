/* test_serve.c — `trochilus serve` (src/app/serve.c) as a user runs it: the trochilus binary built
 * beside this test, a server of its own (TR_SERVER_NAME), a synthetic model with a tokenizer and
 * OLMoE's chat template (tests/synth_olmoe.h).
 *
 * Branches pinned, each by an output it must match or a count:
 *  - no server: a model command runs in its own process (the baselines below are taken so);
 *  - a server up, started in another directory: generate, logits --out, run -f, run with a missing
 *    file, chat on stdin, and a chat whose input is longer than one read of the client (4 KiB)
 *    give the stdout (and the logits file) and the exit code they give alone, byte for byte, with
 *    relative paths: the client's working directory is the one used;
 *  - the model is loaded once for every request with the same file, budget and threads (the same
 *    file named "./tests/..." too), and anew for another budget, another -t, another file (whose
 *    output is its own); the status line counts the expert units of the kept model;
 *  - an expert mask set by one request is gone in the next: its output is the unmasked one, and the
 *    masked one differs (else the check would prove nothing);
 *  - the experts: line counts this request only: hits + misses of a repeated command equal the
 *    count of the command alone;
 *  - a decode width forced by one request is gone in the next (its threads line is the one alone);
 *    a command's own usage error runs in the server (the count moves);
 *  - TR_SERVER=0 runs locally with a server up (the same output, the server's count does not move);
 *  - usage: --stop or --status with no server (1), both at once (2), --idle out of range (2),
 *    --status with an option (2), a second server on the same endpoint (1, "already running");
 *  - the endpoint's name: 64 characters with every edge of its classes (a z A Z 0 9 - _ .) is
 *    taken, 65 or any other character is refused ("no endpoint", and no client finds a server);
 *  - a raw client sends what the command line never does, and the server answers each and lives
 *    on: a wrong magic, too many arguments and exactly the most, an unknown kind, a truncated
 *    frame, a string longer than its frame, no command, four pipes and three, another build (3: the
 *    client would run it itself), no streams, a frame of exactly 1 MiB (answered) and of 2 GiB
 *    (closed), a client that connects and leaves;
 *  - serve -m: the model is loaded before the first request, which loads nothing; a file that
 *    does not load ends the server with its own error; --idle 0 never leaves by itself;
 *  - --stop ends the server; with no command for TR_SERVE_IDLE_MS it exits by itself, though
 *    --status is asked all along, and (POSIX) as well when nobody asks anything.
 * POSIX only (the Windows side has no descriptors to hand over, no socket file, no fork):
 *  - every server this test starts exits 0 (under ASan: with nothing leaked), and none outlives the
 *    test, a failed check included; a command that hangs fails after 20 s instead of hanging;
 *  - raw: `cpu` on streams handed over, refused by the dispatch; a working directory the server
 *    cannot enter; two or four descriptors instead of three, each closed in the server (Linux:
 *    its descriptor count); a frame shorter than its length, and a length cut short, answered by
 *    nothing but the end of the connection;
 *  - a client whose stdin is closed (its socket is then descriptor 0) is served;
 *  - a server of another build declines, and the client runs the command itself (same output, the
 *    server's count does not move); a server that answers in another protocol is not believed;
 *  - XDG_RUNTIME_DIR: a private directory holds the socket; one open to others, a file, or a path
 *    of 108 characters (sun_path holds 107 and the NUL) is refused, and 107 is taken;
 *  - a socket file left by a server that died is taken over. */
#if defined(__linux__)
#define _GNU_SOURCE /* setenv, realpath, and CMSG_SPACE for the raw client */
#endif
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#include <process.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>
#endif
#include "test.h"
#include "synth_olmoe.h"
#include "../src/base/platform.h"

static char bin[1300], bin_abs[1200], dir[512], dir_abs[1200];
static char out_path[700], err_path[700], in_path[700];
static char out[65536], err[65536];
static size_t out_len;
#define ERR_HAS(s) TR_CHECK(strstr(err, s) != NULL)
#define SERVE_MAGIC_T 0x31435254u   /* "TRC1", src/app/serve.c */

static void native_path(char *p) {
#ifdef _WIN32
    for (; *p != 0; p++)
        if (*p == '/') *p = '\\';
#else
    (void)p;
#endif
}

static void in_dir(char *path, size_t len, const char *name) {
    snprintf(path, len, "%s/%s", dir, name);
    native_path(path);
}

static void absolute(const char *path, char *outp, size_t n) {
#ifdef _WIN32
    char *full = _fullpath(NULL, path, 0);
#else
    char *full = realpath(path, NULL);
#endif
    snprintf(outp, n, "%s", full != NULL ? full : path);
    free(full);
}

static size_t slurp(const char *path, char *buf, size_t cap) {
    buf[0] = 0;
    FILE *f = fopen(path, "rb");
    if (f == NULL) return 0;
    size_t n = fread(buf, 1, cap - 1, f);
    buf[n] = 0;
    fclose(f);
    return n;
}

static void write_bytes(const char *path, const char *data, size_t len) {
    FILE *f = fopen(path, "wb");
    TR_CHECK(f != NULL);
    if (f == NULL) return;
    TR_CHECK(fwrite(data, 1, len, f) == len);
    fclose(f);
}

/* value NULL: the variable is removed */
static void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value != NULL ? value : "");
#else
    if (value != NULL) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
#endif
}

/* ---- processes (POSIX): a deadline on each, and no server outlives the test ---- */

#define DEADLINE 20.0           /* s: a command or a server that hangs fails the test, not the gate */

#ifndef _WIN32
/* pid's exit code (128 + the signal that ended it), or -2 when it still ran after `seconds`
 * (then it is killed) */
static int wait_pid(pid_t pid, double seconds) {
    double end = tr_time_sec() + seconds;
    for (;;) {
        int st = 0;
        pid_t r = waitpid(pid, &st, WNOHANG);
        if (r == pid) return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
        if (r < 0) return -1;
        if (tr_time_sec() > end) {
            kill(pid, SIGKILL);
            waitpid(pid, &st, 0);
            return -2;
        }
        sleep_ms(1);
    }
}

/* `sh -c cmd`, where cmd execs the program: the pid waited for is the program's */
static pid_t spawn_sh(const char *cmd) {
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }
    return pid;
}

static int run_sh(const char *cmd) {
    pid_t pid = spawn_sh(cmd);
    if (pid < 0) return -1;
    int rc = wait_pid(pid, DEADLINE);
    if (rc == -2) fprintf(stderr, "test_serve: still running after %.0f s, killed: %s\n", DEADLINE, cmd);
    return rc;
}

/* the servers (and the fake one) this test started and has not seen end */
static pid_t live[16];

static void track(pid_t pid) {
    for (int i = 0; i < 16; i++)
        if (live[i] == 0) {
            live[i] = pid;
            return;
        }
}

static void forget(pid_t pid) {
    for (int i = 0; i < 16; i++)
        if (live[i] == pid) live[i] = 0;
}

/* at exit, a failed check's too (TR_TEST_FAILFAST): the mutation runs counted servers left behind */
static void kill_live(void) {
    for (int i = 0; i < 16; i++)
        if (live[i] > 0) {
            kill(live[i], SIGKILL);
            waitpid(live[i], NULL, 0);
            live[i] = 0;
        }
}
#endif

/* ---- a raw client: frames the command line never sends (src/app/serve.c's protocol) ---- */

static char server_name[128];
static int cli(const char *fmt, ...);

typedef struct {
    unsigned char b[8192];
    size_t n;
} frame;

static void f_u32(frame *f, uint32_t v) {
    for (int k = 0; k < 4 && f->n < sizeof f->b; k++) f->b[f->n++] = (unsigned char)(v >> (8 * k));
}

static void f_str(frame *f, const char *s) {
    size_t n = strlen(s);
    f_u32(f, (uint32_t)n);
    for (size_t i = 0; i < n && f->n < sizeof f->b; i++) f->b[f->n++] = (unsigned char)s[i];
}

/* a request: kind, argv, cwd, pipe names (n_pipes of them, "" each), the build stamp */
static void f_request_in(frame *f, uint32_t magic, uint32_t kind, int argc, const char *const *argv, const char *cwd,
                         uint32_t n_pipes, const char *build) {
    f->n = 0;
    f_u32(f, magic);
    f_u32(f, kind);
    f_u32(f, 0);
    f_u32(f, (uint32_t)argc);
    for (int i = 0; i < argc; i++) f_str(f, argv[i]);
    f_str(f, cwd);
    f_u32(f, n_pipes);
    for (uint32_t i = 0; i < n_pipes; i++) f_str(f, "");
    f_str(f, build);
}

static void f_request(frame *f, uint32_t magic, uint32_t kind, int argc, const char *const *argv, uint32_t n_pipes,
                      const char *build) {
    f_request_in(f, magic, kind, argc, argv, ".", n_pipes, build);
}

#ifdef _WIN32
typedef HANDLE raw_conn;
static int raw_open(raw_conn *c) {
    char user[80] = "user", name[300];
    const char *u = getenv("USERNAME");
    size_t k = 0;
    for (; u != NULL && u[k] != 0 && k + 1 < sizeof user; k++) {
        char ch = u[k];
        user[k] = ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) ? ch : '_';
    }
    if (k > 0) user[k] = 0;
    snprintf(name, sizeof name, "\\\\.\\pipe\\%s-%s", server_name, user);
    for (int tries = 0; tries < 50; tries++) {
        *c = CreateFileA(name, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
        if (*c != INVALID_HANDLE_VALUE) return 0;
        if (GetLastError() != ERROR_PIPE_BUSY) return -1;
        WaitNamedPipeA(name, 200);
    }
    return -1;
}
static int raw_io(raw_conn c, void *buf, size_t n, int writing) {
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        DWORD got = 0;
        BOOL ok = writing ? WriteFile(c, p, (DWORD)n, &got, NULL) : ReadFile(c, p, (DWORD)n, &got, NULL);
        if (!ok || got == 0) return -1;
        p += got;
        n -= got;
    }
    return 0;
}
static void raw_close(raw_conn c) {
    CloseHandle(c);
}
#else
typedef int raw_conn;
/* the socket path src/app/serve.c derives from TR_SERVER_NAME and XDG_RUNTIME_DIR */
static int endpoint_path(char *path, size_t n) {
    char rundir[512];
    const char *run = getenv("XDG_RUNTIME_DIR");
    if (run != NULL && run[0] == '/') snprintf(rundir, sizeof rundir, "%s", run);
    else snprintf(rundir, sizeof rundir, "/tmp/trochilus-%lu", (unsigned long)getuid());
    int k = snprintf(path, n, "%s/%s.sock", rundir, server_name);
    return k > 0 && (size_t)k < n && (size_t)k < sizeof(((struct sockaddr_un *)0)->sun_path) ? 0 : -1;
}
static void unix_addr(struct sockaddr_un *a, const char *path) {
    memset(a, 0, sizeof *a);
    a->sun_family = AF_UNIX;
    memcpy(a->sun_path, path, strlen(path));
}
static int raw_open(raw_conn *c) {
    char path[700];
    if (endpoint_path(path, sizeof path) != 0) return -1;
    struct sockaddr_un a;
    unix_addr(&a, path);
    *c = socket(AF_UNIX, SOCK_STREAM, 0);
    if (*c < 0) return -1;
    if (connect(*c, (struct sockaddr *)&a, sizeof a) != 0) {
        close(*c);
        return -1;
    }
    /* a server that neither answers nor hangs up fails the read, not the test's run */
    struct timeval tv = {10, 0};
    setsockopt(*c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(*c, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    return 0;
}
static int raw_io(raw_conn c, void *buf, size_t n, int writing) {
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        ssize_t got = writing ? write(c, p, n) : read(c, p, n);
        if (got <= 0) return -1;
        p += got;
        n -= (size_t)got;
    }
    return 0;
}
static void raw_close(raw_conn c) {
    close(c);
}
/* n descriptors, as the command line hands its three streams over */
static int raw_send_fds(raw_conn c, const int *fds, int n) {
    char byte = 'F';
    struct iovec iov = {&byte, 1};
    union {
        struct cmsghdr h;
        char buf[CMSG_SPACE(4 * sizeof(int))];
    } u;
    memset(&u, 0, sizeof u);
    struct msghdr mh;
    memset(&mh, 0, sizeof mh);
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    mh.msg_control = u.buf;
    mh.msg_controllen = CMSG_SPACE((size_t)n * sizeof(int));
    struct cmsghdr *cm = CMSG_FIRSTHDR(&mh);
    cm->cmsg_level = SOL_SOCKET;
    cm->cmsg_type = SCM_RIGHTS;
    cm->cmsg_len = CMSG_LEN((size_t)n * sizeof(int));
    memcpy(CMSG_DATA(cm), fds, (size_t)n * sizeof(int));
    return sendmsg(c, &mh, 0) == 1 ? 0 : -1;
}
#endif

/* reads one reply frame: 0 with its code and text, -1 when the server closed without one */
static int raw_reply(raw_conn c, int32_t *code, char *text, size_t tn) {
    unsigned char len[4], body[4096];
    if (raw_io(c, len, 4, 0) != 0) return -1;
    uint32_t n = (uint32_t)len[0] | (uint32_t)len[1] << 8 | (uint32_t)len[2] << 16 | (uint32_t)len[3] << 24;
    if (n < 12 || n > sizeof body || raw_io(c, body, n, 0) != 0) return -1;
    *code = (int32_t)((uint32_t)body[4] | (uint32_t)body[5] << 8 | (uint32_t)body[6] << 16 | (uint32_t)body[7] << 24);
    uint32_t tl = (uint32_t)body[8] | (uint32_t)body[9] << 8 | (uint32_t)body[10] << 16 | (uint32_t)body[11] << 24;
    if (tl > n - 12 || tl >= tn) return -1;
    memcpy(text, body + 12, tl);
    text[tl] = 0;
    return 0;
}

/* one frame (its length first, `len` if not 0, else f->n), then the reply */
static int raw(const frame *f, uint32_t len, int32_t *code, char *text, size_t tn) {
    raw_conn c;
    if (raw_open(&c) != 0) return -2;
    unsigned char l[4];
    uint32_t n = len != 0 ? len : (uint32_t)f->n;
    for (int k = 0; k < 4; k++) l[k] = (unsigned char)(n >> (8 * k));
    int sent = raw_io(c, l, 4, 1) == 0 && (len != 0 || raw_io(c, (void *)f->b, f->n, 1) == 0);
#ifndef _WIN32
    shutdown(c, SHUT_WR);       /* no streams will follow: a server waiting for them sees the end */
#endif
    int rc = sent ? raw_reply(c, code, text, tn) : -1;
    raw_close(c);
    return rc;
}

/* a status request of exactly SERVE_MAX_FRAME (1 MiB) bytes: its header, then zeros */
static int raw_1mib(int32_t *code, char *text, size_t tn) {
    raw_conn c;
    if (raw_open(&c) != 0) return -2;
    frame f;
    f.n = 0;
    f_u32(&f, 1u << 20);
    f_u32(&f, SERVE_MAGIC_T);
    f_u32(&f, 3);               /* status */
    f_u32(&f, 0);
    f_u32(&f, 0);
    int ok = raw_io(c, f.b, f.n, 1) == 0;
    static unsigned char zeros[8192];
    for (size_t left = (1u << 20) - 16; ok && left > 0;) {
        size_t k = left < sizeof zeros ? left : sizeof zeros;
        ok = raw_io(c, zeros, k, 1) == 0;
        left -= k;
    }
    int rc = ok ? raw_reply(c, code, text, tn) : -1;
    raw_close(c);
    return rc;
}

#ifndef _WIN32
/* exactly these bytes, then the end of the client's writing: 0 when a reply came, -1 when the
 * server closed without one */
static int raw_bytes(const unsigned char *p, size_t n) {
    raw_conn c;
    if (raw_open(&c) != 0) return -2;
    int ok = raw_io(c, (void *)p, n, 1) == 0;
    shutdown(c, SHUT_WR);
    int32_t code;
    char text[1024];
    int rc = ok ? raw_reply(c, &code, text, sizeof text) : -2;
    raw_close(c);
    return rc;
}

/* a request, then n descriptors; the reply, and the end of the connection read (the server has
 * closed its side, and what it received, when this returns) */
static int raw_fds(const frame *f, const int *fds, int n, int32_t *code, char *text, size_t tn) {
    raw_conn c;
    if (raw_open(&c) != 0) return -2;
    unsigned char l[4] = {(unsigned char)f->n, (unsigned char)(f->n >> 8), 0, 0};
    int rc = raw_io(c, l, 4, 1) == 0 && raw_io(c, (void *)f->b, f->n, 1) == 0 && raw_send_fds(c, fds, n) == 0
                 ? raw_reply(c, code, text, tn)
                 : -1;
    unsigned char b;
    while (rc == 0 && read(c, &b, 1) > 0) {}
    raw_close(c);
    return rc;
}

#if defined(__linux__)
/* a status request answered and its connection closed by the server: the server is back between
 * requests when this returns */
static int raw_status_to_end(void) {
    frame st;
    st.n = 0;
    f_u32(&st, SERVE_MAGIC_T);
    f_u32(&st, 3);
    f_u32(&st, 0);
    f_u32(&st, 0);
    raw_conn c;
    if (raw_open(&c) != 0) return -1;
    unsigned char l[4] = {(unsigned char)st.n, 0, 0, 0};
    int32_t code = -1;
    char text[1024];
    int rc = raw_io(c, l, 4, 1) == 0 && raw_io(c, st.b, st.n, 1) == 0 ? raw_reply(c, &code, text, sizeof text) : -1;
    unsigned char b;
    while (rc == 0 && read(c, &b, 1) > 0) {}
    raw_close(c);
    return rc == 0 && code == 0 ? 0 : -1;
}

/* how many descriptors process pid holds */
static int fd_count(pid_t pid) {
    char p[64];
    snprintf(p, sizeof p, "/proc/%d/fd", (int)pid);
    DIR *d = opendir(p);
    if (d == NULL) return -1;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL)
        if (e->d_name[0] != '.') n++;
    closedir(d);
    return n;
}
#endif

/* a server that answers every request with a reply of another protocol ("XXXX" for "TRC1"),
 * n_conn times, then leaves */
static pid_t fake_server(int n_conn) {
    char path[700];
    if (endpoint_path(path, sizeof path) != 0) return -1;
    unlink(path);
    struct sockaddr_un a;
    unix_addr(&a, path);
    int L = socket(AF_UNIX, SOCK_STREAM, 0);
    if (L < 0 || bind(L, (struct sockaddr *)&a, sizeof a) != 0 || listen(L, 4) != 0) {
        if (L >= 0) close(L);
        return -1;
    }
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        for (int i = 0; i < n_conn; i++) {
            int c = accept(L, NULL, NULL);
            if (c < 0) _exit(1);
            unsigned char len[4], body[8192];
            if (raw_io(c, len, 4, 0) == 0) {
                uint32_t n = (uint32_t)len[0] | (uint32_t)len[1] << 8 | (uint32_t)len[2] << 16 | (uint32_t)len[3] << 24;
                raw_io(c, body, n < sizeof body ? n : sizeof body, 0);
            }
            static const unsigned char reply[16] = {12, 0, 0, 0, 'X', 'X', 'X', 'X', 0, 0, 0, 0, 0, 0, 0, 0};
            raw_io(c, (void *)reply, sizeof reply, 1);
            close(c);
        }
        _exit(0);
    }
    close(L);
    if (pid > 0) track(pid);
    return pid;
}
#endif

static int n_raw;               /* raw cases that got the answer they expect */

/* the server answers `f` with `want_code` and a text holding `want_text`, then still serves */
static void raw_expect(const char *what, const frame *f, int32_t want_code, const char *want_text) {
    int32_t code = -1;
    char text[1024] = "";
    int rc = raw(f, 0, &code, text, sizeof text);
    int ok = rc == 0 && code == want_code && strstr(text, want_text) != NULL;
    if (!ok) fprintf(stderr, "test_serve: raw %s: rc %d, code %d, '%s'\n", what, rc, (int)code, text);
    TR_CHECK(ok);
    TR_CHECK(cli("serve --status") == 0);   /* alive */
    n_raw += ok;
}

/* stdin of a command: none (/dev/null on POSIX), in_path, or closed (POSIX) */
enum { IN_NONE, IN_FILE, IN_CLOSED };

/* `bin <args> [< in_path] > out_path 2> err_path`; the exit code */
static int vcli(int in, const char *fmt, va_list ap) {
    char args[4096], cmd[8192];
    vsnprintf(args, sizeof args, fmt, ap);
#ifdef _WIN32
    snprintf(cmd, sizeof cmd, "%s %s%s%s > %s 2> %s", bin, args, in == IN_FILE ? " < " : "", in == IN_FILE ? in_path : "",
             out_path, err_path);
    int st = system(cmd);
#else
    snprintf(cmd, sizeof cmd, "exec %s %s %s%s > %s 2> %s", bin, args,
             in == IN_FILE ? "< " : in == IN_CLOSED ? "0<&-" : "< /dev/null", in == IN_FILE ? in_path : "", out_path,
             err_path);
    int st = run_sh(cmd);
#endif
    out_len = slurp(out_path, out, sizeof out);
    slurp(err_path, err, sizeof err);
    return st;
}

static int cli(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int rc = vcli(IN_NONE, fmt, ap);
    va_end(ap);
    return rc;
}

static int cli_stdin(const char *input, const char *fmt, ...) {
    write_bytes(in_path, input, strlen(input));
    va_list ap;
    va_start(ap, fmt);
    int rc = vcli(IN_FILE, fmt, ap);
    va_end(ap);
    return rc;
}

#ifndef _WIN32
static int cli_closed_stdin(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int rc = vcli(IN_CLOSED, fmt, ap);
    va_end(ap);
    return rc;
}
#endif

/* POSIX: its pid; Windows: 0 (started by `start`, found by its endpoint) */
typedef long server;

/* a server run by `exe` in dir_abs (not this process's directory), its output in `log`; `env`
 * is one NAME=value for it alone, or "" */
static server start_server_exe(const char *exe, const char *env, const char *args, const char *log) {
    char cmd[4096];
#ifdef _WIN32
    char set[256] = "";
    if (env[0] != 0) snprintf(set, sizeof set, "set %s&& ", env);
    snprintf(cmd, sizeof cmd, "%sstart \"\" /b /d \"%s\" \"%s\" serve %s > \"%s\" 2>&1", set, dir_abs, exe, args, log);
    TR_CHECK(system(cmd) == 0);
    return 0;
#else
    /* the log is opened here, where its relative path points, before the cd */
    snprintf(cmd, sizeof cmd, "exec < /dev/null > '%s' 2>&1; cd '%s' && %s exec '%s' serve %s", log, dir_abs, env, exe, args);
    pid_t pid = spawn_sh(cmd);
    TR_CHECK(pid > 0);
    if (pid > 0) track(pid);
    return pid;
#endif
}

static server start_server(const char *env, const char *args, const char *log) {
    return start_server_exe(bin_abs, env, args, log);
}

/* until `serve --status` says the server is up (want_up) or gone; 0 when it did within 20 s */
static int wait_server(int want_up) {
    double end = tr_time_sec() + DEADLINE;
    int rc = -1;
    while (tr_time_sec() < end) {
        rc = cli("serve --status");
        if ((rc == 0) == (want_up != 0)) return 0;
    }
    fprintf(stderr, "test_serve: the server is still %s after 20 s: status exit %d, '%s' '%s'\n",
            want_up ? "down" : "up", rc, out, err);
    return -1;
}

/* the server is gone from its endpoint and, on POSIX, exited 0 (under ASan: with no leak) */
static int server_ended(server s) {
    if (wait_server(0) != 0) return -1;
#ifdef _WIN32
    (void)s;
    return 0;
#else
    int rc = wait_pid((pid_t)s, DEADLINE);
    forget((pid_t)s);
    if (rc != 0) fprintf(stderr, "test_serve: the server exited %d\n", rc);
    return rc;
#endif
}

/* the status line's "<n> requests, <n> loads" */
static void status_counts(long long *requests, long long *loads) {
    *requests = *loads = -1;
    TR_CHECK(cli("serve --status") == 0);
    const char *s = strstr(out, "server: ");
    TR_CHECK(s != NULL && sscanf(s, "server: %lld requests, %lld loads", requests, loads) == 2);
}

/* hits + misses of the experts: line on stderr, -1 without one */
static long long expert_lookups(void) {
    const char *s = strstr(err, "experts: ");
    const char *h = s != NULL ? strstr(s, " hits, ") : NULL;
    if (h == NULL) return -1;
    while (h > s && h[-1] >= '0' && h[-1] <= '9') h--;
    long long hits = -1, misses = -1;
    if (sscanf(h, "%lld hits, %lld misses", &hits, &misses) != 2) return -1;
    return hits + misses;
}

/* A command's stdout, alone then through the server: the same bytes and the same exit code. */
typedef struct {
    const char *what, *input;   /* input: on stdin, or NULL */
    char args[2048];
    int rc;
    char *out;
    size_t len;
} baseline;

static int n_same;              /* comparisons that matched: the count the end of main checks */

static void take(baseline *b) {
    b->rc = b->input != NULL ? cli_stdin(b->input, "%s", b->args) : cli("%s", b->args);
    b->out = (char *)malloc(out_len + 1);
    TR_CHECK(b->out != NULL);
    if (b->out != NULL) memcpy(b->out, out, out_len + 1);
    b->len = out_len;
}

static void same_output(const baseline *b, int rc) {
    int ok = rc == b->rc && out_len == b->len && b->out != NULL && memcmp(out, b->out, out_len) == 0;
    if (!ok) fprintf(stderr, "test_serve: %s differs through the server (exit %d, %d alone)\n%s\n", b->what, rc, b->rc, err);
    TR_CHECK(ok);
    n_same += ok;
}

static void same_as(const baseline *b) {
    same_output(b, b->input != NULL ? cli_stdin(b->input, "%s", b->args) : cli("%s", b->args));
}

/* TR_SERVER_NAME for this process and the ones it starts */
static void use_name(const char *name) {
    snprintf(server_name, sizeof server_name, "%s", name);
    set_env("TR_SERVER_NAME", name);
}

/* `serve` in the foreground: 0 when it listened (and left, idle, after TR_SERVE_IDLE_MS), 1 when
 * it had no endpoint */
static int serve_briefly(void) {
    set_env("TR_SERVE_IDLE_MS", "200");
    int rc = cli("serve");
    set_env("TR_SERVE_IDLE_MS", NULL);
    return rc;
}

int main(int argc, char **argv) {
    const char *self = argc > 0 ? argv[0] : "";
    const char *sl = strrchr(self, '/'), *bs = strrchr(self, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    snprintf(dir, sizeof dir, "%.*s", sl ? (int)(sl - self) : 1, sl ? self : ".");
#ifdef _WIN32
    snprintf(bin, sizeof bin, "%s/../trochilus.exe", dir);
#else
    snprintf(bin, sizeof bin, "%s/../trochilus", dir);
    atexit(kill_live);
#endif
    native_path(bin);
    absolute(bin, bin_abs, sizeof bin_abs);
    absolute(dir, dir_abs, sizeof dir_abs);
    /* This process works from the directory above its own, every path it hands over is relative
     * ("tests/..."), and the server runs in dir_abs, where those paths do not exist: a command
     * finds its files only in the client's working directory. (argv[0] is absolute on Windows.) */
    char parent[1300];
    snprintf(parent, sizeof parent, "%s/..", dir_abs);
    native_path(parent);
#ifdef _WIN32
    TR_CHECK(_chdir(parent) == 0);
#else
    TR_CHECK(chdir(parent) == 0);
#endif
    snprintf(bin, sizeof bin, "%s", bin_abs);
    snprintf(dir, sizeof dir, "tests");
    in_dir(out_path, sizeof out_path, "test_serve_out.txt");
    in_dir(err_path, sizeof err_path, "test_serve_err.txt");
    in_dir(in_path, sizeof in_path, "test_serve_in.txt");
    char name[64];
#ifdef _WIN32
    int my_pid = _getpid();
#else
    int my_pid = (int)getpid();
#endif
    snprintf(name, sizeof name, "test-serve-%d", my_pid);
    use_name(name);
    set_env("TR_SERVER", "1");
    set_env("TR_SERVE_IDLE_MS", NULL);

    if (cli("cpu") != 0) {
#ifdef _WIN32
        printf("test_serve: SKIPPED, %s does not run here\n", bin);
        return 126;
#else
        fprintf(stderr, "test_serve: %s does not run\n", bin);
        return 1;
#endif
    }

    synth_params P = {2, 32, 2, 2, 32, 4, 2, SYNTH_TOK_MIN_VOCAB, 64, TR_TYPE_F32};
    synth_with_tokenizer = 1;
    char model[600], model2[600], dot_model[620];
    synth_params P2 = P;
    P2.layers = 1;              /* another file, and other weights: another output (a width of its own:
                                 * with the same n_embd the two share embedding and output weights and
                                 * wrote the same six tokens) */
    P2.n_embd = 64;
    /* beside the binary whatever TR_TEST_TMPDIR says: the paths below name them relative to it */
    if (synth_write_in(&P, dir_abs, "test_serve.gguf", model, sizeof model) != 0 ||
        synth_write_in(&P2, dir_abs, "test_serve2.gguf", model2, sizeof model2) != 0) {
        fprintf(stderr, "test_serve: could not write the model\n");
        return 1;
    }
    in_dir(model, sizeof model, "test_serve.gguf");     /* where synth_write put it, relative */
    in_dir(model2, sizeof model2, "test_serve2.gguf");
    snprintf(dot_model, sizeof dot_model, "./%s", model);   /* the same file, named otherwise */
    native_path(dot_model);
    char prompt[700], mask[700], lout[700], log[700], log2[700], log3[700], missing[700], missing_model[700];
    in_dir(prompt, sizeof prompt, "test_serve_prompt.txt");
    in_dir(mask, sizeof mask, "test_serve_mask.txt");
    in_dir(lout, sizeof lout, "test_serve_logits.bin");
    in_dir(log, sizeof log, "test_serve_server.log");
    in_dir(log2, sizeof log2, "test_serve_server2.log");
    in_dir(log3, sizeof log3, "test_serve_server3.log");
    in_dir(missing, sizeof missing, "test_serve_missing.txt");
    in_dir(missing_model, sizeof missing_model, "test_serve_missing.gguf");
    write_bytes(prompt, "hello world", 11);
    static const char mask_lines[] = "0 0\n1 3\n";
    write_bytes(mask, mask_lines, sizeof mask_lines - 1);
    remove(missing);
    remove(missing_model);
    /* the model is here, relative to this process, and not relative to the server's directory */
    {
        char there[1900];
        snprintf(there, sizeof there, "%s/%s", dir_abs, model);
        FILE *f = fopen(model, "rb"), *g = fopen(there, "rb");
        TR_CHECK(f != NULL && g == NULL);   /* here, and not from the server's directory */
        if (f != NULL) fclose(f);
        if (g != NULL) fclose(g);
    }

    /* usage, with no server */
    TR_CHECK(cli("serve --status") == 1);
    ERR_HAS("no server is running");
    TR_CHECK(cli("serve --stop") == 1);
    ERR_HAS("no server is running");
    TR_CHECK(cli("serve --stop --status") == 2);
    TR_CHECK(cli("serve --idle 10081") == 2);
    TR_CHECK(cli("serve --stop -m %s", model) == 2);
    TR_CHECK(cli("serve --status -t 4") == 2);

    /* the endpoint's name: 64 characters, with every edge of the classes it takes, is taken (the
     * server listens, then leaves idle); 65, or a character outside them, is refused, and no
     * client finds a server there */
    {
        char name64[80], name65[96];
        /* this process's own digits: gcc, clang and ASan run this test at once in the gate, and a
         * fixed name met the other's server ("already running", docs/LESSONS.md #141) */
        snprintf(name64, sizeof name64, "azAZ09-_.%0*d", 55, my_pid);
        snprintf(name65, sizeof name65, "%sx", name64);
        TR_CHECK(strlen(name64) == 64);
        use_name(name64);
        TR_CHECK(serve_briefly() == 0);
        ERR_HAS("serve: listening on");
        use_name(name65);
        TR_CHECK(serve_briefly() == 1);
        ERR_HAS("serve: no endpoint");
        static const char *const bad[] = {"a b", "a/b", "a:b", "a`b", "a{b", "a@b", "a[b"};
        for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
            use_name(bad[i]);
            int rc = serve_briefly();
            if (rc != 1 || strstr(err, "serve: no endpoint") == NULL) fprintf(stderr, "test_serve: name '%s' taken\n", bad[i]);
            TR_CHECK(rc == 1);
            ERR_HAS("serve: no endpoint");
            TR_CHECK(cli("serve --status") == 1);
            ERR_HAS("no server is running");
        }
        use_name(name);
    }

    /* the baselines, each in its own process */
    char long_chat[5100];       /* more than one read (4 KiB) of the client's stdin: empty lines, then a turn */
    memset(long_chat, '\n', 5000);
    snprintf(long_chat + 5000, sizeof long_chat - 5000, "hi\n");
    baseline b[8] = {{"generate", NULL, "", 0, NULL, 0},
                     {"logits", NULL, "", 0, NULL, 0},
                     {"run -f", NULL, "", 0, NULL, 0},
                     {"run, missing file", NULL, "", 0, NULL, 0},
                     {"chat", "hello\n/reset\nhi\n", "", 0, NULL, 0},
                     {"logits, masked", NULL, "", 0, NULL, 0},
                     {"chat, 5 KB of input", long_chat, "", 0, NULL, 0},
                     {"generate, another model", NULL, "", 0, NULL, 0}};
    snprintf(b[0].args, sizeof b[0].args, "generate -m %s --tokens 5,7 -n 6", model);
    snprintf(b[1].args, sizeof b[1].args, "logits -m %s --tokens 5,7,9 --out %s", model, lout);
    snprintf(b[2].args, sizeof b[2].args, "run -m %s -f %s -n 8", model, prompt);
    snprintf(b[3].args, sizeof b[3].args, "run -m %s -f %s -n 8", model, missing);
    snprintf(b[4].args, sizeof b[4].args, "chat -m %s -n 3", model);
    snprintf(b[5].args, sizeof b[5].args, "logits -m %s --tokens 5,7,9 --out %s --expert-mask %s", model, lout, mask);
    snprintf(b[6].args, sizeof b[6].args, "chat -m %s -n 3", model);
    snprintf(b[7].args, sizeof b[7].args, "generate -m %s --tokens 5,7 -n 6", model2);
    /* the logits files: plain and masked, which must differ (else the mask's reset proves nothing) */
    static char logits_alone[65536], logits_masked[65536], logits_served[65536];
    size_t n_logits = 0, n_masked = 0;
    char b0_threads[256] = "";   /* generate's threads line alone: no width forced */
    for (int k = 0; k < 8; k++) {
        take(&b[k]);
        if (k == 0) {
            const char *t = strstr(err, "threads: ");
            size_t n = t != NULL ? strcspn(t, "\r\n") : 0;
            snprintf(b0_threads, sizeof b0_threads, "%.*s", (int)n, t != NULL ? t : "");
            TR_CHECK(n > 0 && strstr(b0_threads, "decode") != NULL);
        }
        if (k == 1) n_logits = slurp(lout, logits_alone, sizeof logits_alone);
        if (k == 5) n_masked = slurp(lout, logits_masked, sizeof logits_masked);
    }
    TR_CHECK(b[0].rc == 0 && b[1].rc == 0 && b[2].rc == 0 && b[4].rc == 0 && b[5].rc == 0 && b[6].rc == 0 && b[7].rc == 0);
    TR_CHECK(b[3].rc == 1);
    TR_CHECK(n_logits > 0 && n_masked == n_logits && memcmp(logits_alone, logits_masked, n_logits) != 0);
    /* the long chat saw its last turn; the other model writes other tokens */
    TR_CHECK(b[6].len > b[4].len);
    TR_CHECK(b[7].len != b[0].len || memcmp(b[7].out, b[0].out, b[0].len) != 0);
    remove(lout);
    TR_CHECK(cli("run -m %s -f %s -n 8 --expert-budget min", model, prompt) == 0);
    long long lookups_alone = expert_lookups();
    TR_CHECK(lookups_alone > 0);

    /* the server, started elsewhere */
    remove(log);
    server main_server = start_server("", "--idle 1", log);
    TR_CHECK(wait_server(1) == 0);
    /* a second one on the same endpoint; were it to listen, it would leave in 3 s, not in 30 min */
    set_env("TR_SERVE_IDLE_MS", "3000");
    TR_CHECK(cli("serve") == 1);
    ERR_HAS("a server is already running");
    set_env("TR_SERVE_IDLE_MS", NULL);

    for (int k = 0; k < 5; k++) {
        same_as(&b[k]);
        if (k == 1)
            TR_CHECK(slurp(lout, logits_served, sizeof logits_served) == n_logits &&
                     memcmp(logits_alone, logits_served, n_logits) == 0);
    }
    same_as(&b[5]);
    TR_CHECK(slurp(lout, logits_served, sizeof logits_served) == n_masked &&
             memcmp(logits_masked, logits_served, n_masked) == 0);
    same_as(&b[1]);                 /* the mask of the request before is gone */
    TR_CHECK(slurp(lout, logits_served, sizeof logits_served) == n_logits &&
             memcmp(logits_alone, logits_served, n_logits) == 0);
    same_as(&b[6]);                 /* all of a stdin longer than one read of the client */
    long long requests, loads;
    status_counts(&requests, &loads);
    TR_CHECK(requests == 8);
    TR_CHECK(loads == 1);           /* one file, one budget, one pool: loaded once */
    TR_CHECK(strstr(out, "expert units in RAM") != NULL);   /* the kept model's store */

    /* the same file by another name: kept; another file: loaded, with its own output; the first
     * again, and another -t: loaded anew each time */
    TR_CHECK(cli("generate -m %s --tokens 5,7 -n 6", dot_model) == 0);
    same_output(&b[0], 0);
    status_counts(&requests, &loads);
    TR_CHECK(requests == 9 && loads == 1);
    same_as(&b[7]);
    same_as(&b[0]);
    status_counts(&requests, &loads);
    TR_CHECK(requests == 11 && loads == 3);
    TR_CHECK(cli("generate -m %s --tokens 5,7 -n 6 -t 2", model) == 0);
    status_counts(&requests, &loads);
    TR_CHECK(requests == 12 && loads == 4);

    /* another budget loads anew; its experts: line counts its own lookups, twice */
    TR_CHECK(cli("run -m %s -f %s -n 8 --expert-budget min", model, prompt) == 0);
    TR_CHECK(expert_lookups() == lookups_alone);
    TR_CHECK(cli("run -m %s -f %s -n 8 --expert-budget min", model, prompt) == 0);
    TR_CHECK(expert_lookups() == lookups_alone);
    status_counts(&requests, &loads);
    TR_CHECK(requests == 14 && loads == 5);

    /* TR_SERVER=0: in this process, the same output, the server untouched */
    set_env("TR_SERVER", "0");
    same_as(&b[0]);
    set_env("TR_SERVER", "1");
    status_counts(&requests, &loads);
    TR_CHECK(requests == 14);

    /* a width forced by one request is gone in the next: its threads line is the one alone */
    TR_CHECK(cli("generate -m %s --tokens 5,7 -n 6 --decode-threads 3", model) == 0);
    TR_CHECK(strstr(err, "3 decode") != NULL);
    TR_CHECK(cli("%s", b[0].args) == 0);
    TR_CHECK(strstr(err, b0_threads) != NULL);
    /* a command's own usage, through the server: the command ran there (the count moves) */
    TR_CHECK(cli("run") == 2);
    ERR_HAS("usage: trochilus run");
    status_counts(&requests, &loads);
    TR_CHECK(requests == 17);
#ifndef _WIN32
    /* a client with its stdin closed: its socket is descriptor 0, and it is still served */
    same_output(&b[0], cli_closed_stdin("%s", b[0].args));
    status_counts(&requests, &loads);
    TR_CHECK(requests == 18);
#endif

    /* frames the command line never sends: each answered as it should, none takes the server down */
    frame f;
    static const char *const run_argv[] = {"run"};
    const uint32_t magic = SERVE_MAGIC_T;
    f_request(&f, 0x12345678u, 1, 1, run_argv, 0, "");
    raw_expect("magic", &f, 2, "not a request of this version");
    f.n = 0;
    f_u32(&f, magic);
    f_u32(&f, 1);
    f_u32(&f, 0);
    f_u32(&f, 1025);
    raw_expect("argc", &f, 2, "not a request of this version");
    /* exactly the most arguments: read, then refused only for the streams it has not handed over */
    f.n = 0;
    f_u32(&f, magic);
    f_u32(&f, 1);
    f_u32(&f, 0);
    f_u32(&f, 1024);
    for (int i = 0; i < 1024; i++) f_str(&f, "");
    f_str(&f, ".");
    f_u32(&f, 0);
    f_str(&f, "");
    raw_expect("1024 arguments", &f, 1, "the client's streams");
    f_request(&f, magic, 99, 1, run_argv, 0, "");
    raw_expect("kind", &f, 2, "malformed request");
    f_request(&f, magic, 1, 1, run_argv, 0, "");
    f.n -= 3;
    raw_expect("truncated", &f, 2, "malformed request");
    /* an argument whose length runs past the frame (under ASan a read past it kills the server) */
    f.n = 0;
    f_u32(&f, magic);
    f_u32(&f, 1);
    f_u32(&f, 0);
    f_u32(&f, 1);
    f_u32(&f, 100);
    for (int i = 0; i < 3; i++) f.b[f.n++] = (unsigned char)"run"[i];
    raw_expect("a string longer than its frame", &f, 2, "malformed request");
    f_request(&f, magic, 1, 0, NULL, 0, "");
    raw_expect("no command", &f, 2, "malformed request");
    f_request(&f, magic, 1, 1, run_argv, 4, "");
    raw_expect("pipes", &f, 2, "malformed request");
    /* three pipes, the most: past the request's checks, refused only because nothing is open */
    f_request(&f, magic, 1, 1, run_argv, 3, "");
    raw_expect("three pipes", &f, 1, "the client's streams");
    f_request(&f, magic, 1, 1, run_argv, 0, "1:1");
    raw_expect("another build", &f, 3, "another build");
    f_request(&f, magic, 1, 1, run_argv, 0, "");
#ifdef _WIN32
    raw_expect("no streams", &f, 1, "could not open the client's streams");
#else
    raw_expect("no streams", &f, 1, "did not receive the client's streams");
#endif
    {
        int32_t code = -1;
        char text[1024] = "";
        /* a frame of exactly SERVE_MAX_FRAME: answered */
        int rc = raw_1mib(&code, text, sizeof text);
        if (rc != 0 || code != 0 || strstr(text, "server: ") == NULL)
            fprintf(stderr, "test_serve: raw 1 MiB: rc %d, code %d, '%s'\n", rc, (int)code, text);
        TR_CHECK(rc == 0 && code == 0 && strstr(text, "server: ") != NULL);
        n_raw += rc == 0 && code == 0;
        TR_CHECK(raw(&f, 0x7FFFFFFFu, &code, text, sizeof text) == -1);   /* 2 GiB: closed, no reply */
        TR_CHECK(cli("serve --status") == 0);
        raw_conn c;
        TR_CHECK(raw_open(&c) == 0);        /* came and went */
        raw_close(c);
        TR_CHECK(cli("serve --status") == 0);
    }
#ifndef _WIN32
    {
        /* a command no server runs, on streams handed over as the command line does */
        static const char *const cpu_argv[] = {"cpu"};
        char errfile[700];
        in_dir(errfile, sizeof errfile, "test_serve_raw_err.txt");
        f_request(&f, magic, 1, 1, cpu_argv, 0, "");
        int fds[3] = {open("/dev/null", O_RDONLY), open("/dev/null", O_WRONLY), open(errfile, O_WRONLY | O_CREAT | O_TRUNC, 0600)};
        raw_conn c;
        int32_t ack = -1, code = -1;
        char text[256];
        unsigned char l[4] = {(unsigned char)f.n, (unsigned char)(f.n >> 8), 0, 0};
        int ok = raw_open(&c) == 0;
        ok = ok && raw_io(c, l, 4, 1) == 0 && raw_io(c, f.b, f.n, 1) == 0 && raw_send_fds(c, fds, 3) == 0 &&
             raw_reply(c, &ack, text, sizeof text) == 0 && ack == 0 && raw_reply(c, &code, text, sizeof text) == 0 &&
             code == 2;
        if (ok) raw_close(c);
        slurp(errfile, err, sizeof err);
        ERR_HAS("error: a server runs only generate, logits, run and chat");
        TR_CHECK(ok);
        n_raw += ok;
        remove(errfile);

        /* a working directory the server cannot enter: the streams are closed, the client told */
        f_request_in(&f, magic, 1, 1, cpu_argv, "/nonexistent-test-serve", 0, "");
        char text2[256] = "";
        code = -1;
        ok = raw_fds(&f, fds, 3, &code, text2, sizeof text2) == 0 && code == 1 &&
             strstr(text2, "cannot enter the client's working directory") != NULL;
        if (!ok) fprintf(stderr, "test_serve: raw cwd: code %d, '%s'\n", (int)code, text2);
        TR_CHECK(ok);
        TR_CHECK(cli("serve --status") == 0);
        n_raw += ok;

        /* two descriptors, then four, instead of three: refused, and every one the kernel put in
         * the server is closed there (Linux: the server holds as many descriptors after as before) */
        f_request(&f, magic, 1, 1, cpu_argv, 0, "");
        int four[4] = {fds[0], fds[1], fds[2], fds[0]};
        for (int n = 2; n <= 4; n += 2) {
#if defined(__linux__)
            TR_CHECK(raw_status_to_end() == 0);     /* the server between requests: nothing open */
            int before = fd_count((pid_t)main_server);
#endif
            code = -1;
            text2[0] = 0;
            ok = raw_fds(&f, four, n, &code, text2, sizeof text2) == 0 && code == 1 &&
                 strstr(text2, "did not receive the client's streams") != NULL;
            if (!ok) fprintf(stderr, "test_serve: raw %d descriptors: code %d, '%s'\n", n, (int)code, text2);
            TR_CHECK(ok);
#if defined(__linux__)
            int after = fd_count((pid_t)main_server);
            if (after != before) fprintf(stderr, "test_serve: %d descriptors: the server holds %d, %d before\n", n, after, before);
            TR_CHECK(before > 0 && after == before);
#endif
            TR_CHECK(cli("serve --status") == 0);
            n_raw += ok;
        }
        for (int k = 0; k < 3; k++) close(fds[k]);

        /* a frame shorter than its length, and a length cut short, then the end: no reply, the
         * connection closed, the server alive */
        unsigned char shortf[20] = {64, 0, 0, 0};
        frame hs;
        f_request(&hs, magic, 3, 0, NULL, 0, "");
        memcpy(shortf + 4, hs.b, 16);
        TR_CHECK(raw_bytes(shortf, sizeof shortf) == -1);
        TR_CHECK(cli("serve --status") == 0);
        TR_CHECK(raw_bytes(shortf, 2) == -1);
        TR_CHECK(cli("serve --status") == 0);
    }
#endif

    TR_CHECK(cli("serve --stop") == 0);
    TR_CHECK(server_ended(main_server) == 0);
    slurp(log, err, sizeof err);
    ERR_HAS("serve: listening on");
    ERR_HAS("serve: stopped");

    /* serve -m: loaded before any request (the log names it), so the first request loads nothing;
     * the path is the server's own, relative to where it runs */
    remove(log3);
    server with_model = start_server("", "-m test_serve.gguf --idle 1", log3);
    TR_CHECK(wait_server(1) == 0);
    same_as(&b[0]);
    status_counts(&requests, &loads);
    if (requests != 1 || loads != 1) {
        fprintf(stderr, "test_serve: serve -m: %lld requests, %lld loads (1 and 1 wanted)\n", requests, loads);
        cli("serve --status");
        fprintf(stderr, "test_serve: status: %s%s", out, err);
        char l3[2048];
        slurp(log3, l3, sizeof l3);
        fprintf(stderr, "test_serve: its log: %s", l3);
    }
    TR_CHECK(requests == 1 && loads == 1);
    TR_CHECK(cli("serve --stop") == 0);
    TR_CHECK(server_ended(with_model) == 0);
    slurp(log3, err, sizeof err);
    ERR_HAS("test_serve.gguf|");
    TR_CHECK(strstr(err, "no model yet") == NULL);
    /* a model that does not load: the server ends with the load's own error (were it to listen,
     * it would leave in 200 ms) */
    set_env("TR_SERVE_IDLE_MS", "200");
    TR_CHECK(cli("serve -m %s", missing_model) == 1);
    set_env("TR_SERVE_IDLE_MS", NULL);
    ERR_HAS("serve: ");
    TR_CHECK(strstr(err, "thread pool") == NULL && strstr(err, "listening") == NULL);

    /* --idle 0: never leaves by itself */
    remove(log3);
    server forever = start_server("", "--idle 0", log3);
    TR_CHECK(wait_server(1) == 0);
    sleep_ms(300);
    TR_CHECK(cli("serve --status") == 0);
    TR_CHECK(cli("serve --stop") == 0);
    TR_CHECK(server_ended(forever) == 0);

    /* no command for 1500 ms: the server leaves by itself, though --status is asked all along
     * (only a command counts as use) */
    remove(log2);
    server idle = start_server("TR_SERVE_IDLE_MS=1500", "--idle 1", log2);
    TR_CHECK(wait_server(1) == 0);
    double t_up = tr_time_sec();
    TR_CHECK(server_ended(idle) == 0);
    double t_gone = tr_time_sec() - t_up;
    TR_CHECK(t_gone < 10.0);
    slurp(log2, err, sizeof err);
    ERR_HAS("serve: idle for 1500 ms, exiting");

#ifndef _WIN32
    /* and when nobody asks anything: the wait itself runs out */
    remove(log2);
    server quiet = start_server("TR_SERVE_IDLE_MS=800", "--idle 1", log2);
    TR_CHECK(wait_server(1) == 0);
    TR_CHECK(wait_pid((pid_t)quiet, DEADLINE) == 0);
    forget((pid_t)quiet);
    slurp(log2, err, sizeof err);
    ERR_HAS("serve: idle for 800 ms, exiting");
    TR_CHECK(strstr(err, "endpoint failed") == NULL);

    /* a server of another build (the same bytes, another time stamp) declines: the client runs the
     * command itself, with the same output, and the server's count does not move */
    {
        char other[1300];
        snprintf(other, sizeof other, "%s/test_serve_other_build", dir_abs);
        FILE *src = fopen(bin_abs, "rb"), *dst = fopen(other, "wb");
        int copied = src != NULL && dst != NULL;
        char buf[65536];
        size_t k;
        while (copied && (k = fread(buf, 1, sizeof buf, src)) > 0) copied = fwrite(buf, 1, k, dst) == k;
        if (src != NULL) fclose(src);
        if (dst != NULL) fclose(dst);
        struct utimbuf old = {946684800, 946684800};     /* 2000-01-01 */
        TR_CHECK(copied && chmod(other, 0700) == 0 && utime(other, &old) == 0);
        remove(log3);
        server other_build = start_server_exe(other, "", "--idle 1", log3);
        TR_CHECK(wait_server(1) == 0);
        same_as(&b[0]);
        status_counts(&requests, &loads);
        TR_CHECK(requests == 0 && loads == 0);
        TR_CHECK(cli("serve --stop") == 0);
        TR_CHECK(server_ended(other_build) == 0);
        remove(other);
    }

    /* a server that answers in another protocol is not believed: --status fails, and a command
     * runs in the client */
    {
        char fake_name[80];
        snprintf(fake_name, sizeof fake_name, "test-serve-fake-%d", my_pid);
        use_name(fake_name);
        pid_t fake = fake_server(2);
        TR_CHECK(fake > 0);
        TR_CHECK(cli("serve --status") == 1);
        ERR_HAS("the server did not answer");
        same_as(&b[0]);
        TR_CHECK(wait_pid(fake, DEADLINE) == 0);
        forget(fake);
        char path[700];
        if (endpoint_path(path, sizeof path) == 0) unlink(path);
        use_name(name);
    }

    /* a socket file a server left when it died, nobody listening: taken over */
    {
        char stale_name[80], path[700];
        snprintf(stale_name, sizeof stale_name, "test-serve-stale-%d", my_pid);
        use_name(stale_name);
        TR_CHECK(endpoint_path(path, sizeof path) == 0);
        struct sockaddr_un a;
        unix_addr(&a, path);
        int s = socket(AF_UNIX, SOCK_STREAM, 0);
        TR_CHECK(s >= 0 && bind(s, (struct sockaddr *)&a, sizeof a) == 0);
        close(s);
        struct stat st;
        TR_CHECK(stat(path, &st) == 0);     /* left behind */
        TR_CHECK(serve_briefly() == 0);
        ERR_HAS("serve: listening on");
        use_name(name);
    }

    /* XDG_RUNTIME_DIR: a private directory holds the socket; one open to others, or a file, is no
     * place for it; the path fits sun_path at 107 characters and not at 108 */
    {
        char xdg[128], file[160], deep[200];
        snprintf(xdg, sizeof xdg, "/tmp/test-serve-xdg-%d", my_pid);
        mkdir(xdg, 0700);
        TR_CHECK(chmod(xdg, 0700) == 0);
        set_env("XDG_RUNTIME_DIR", xdg);
        TR_CHECK(serve_briefly() == 0);
        ERR_HAS(xdg);
        TR_CHECK(chmod(xdg, 0755) == 0);
        TR_CHECK(serve_briefly() == 1);
        ERR_HAS("serve: no endpoint");
        TR_CHECK(chmod(xdg, 0700) == 0);
        snprintf(file, sizeof file, "%s/file", xdg);
        write_bytes(file, "", 0);
        TR_CHECK(chmod(file, 0600) == 0);
        set_env("XDG_RUNTIME_DIR", file);
        TR_CHECK(serve_briefly() == 1);
        ERR_HAS("serve: no endpoint");
        /* <deep>/<name>.sock of 108 characters, then 107 */
        for (int len = 108; len >= 107; len--) {
            int pad = len - (int)strlen(xdg) - 2 - (int)strlen(name) - 5;
            TR_CHECK(pad > 0);
            snprintf(deep, sizeof deep, "%s/%0*d", xdg, pad, 0);
            mkdir(deep, 0700);
            set_env("XDG_RUNTIME_DIR", deep);
            char sock[300];
            snprintf(sock, sizeof sock, "%s/%s.sock", deep, name);
            TR_CHECK((int)strlen(sock) == len);
            TR_CHECK(serve_briefly() == (len == 108 ? 1 : 0));
            ERR_HAS(len == 108 ? "serve: no endpoint" : "serve: listening on");
            rmdir(deep);
        }
        set_env("XDG_RUNTIME_DIR", NULL);
        remove(file);
        rmdir(xdg);
    }
#endif

#ifdef _WIN32
    TR_CHECK(n_same == 13);
    TR_CHECK(n_raw == 12);
#else
    TR_CHECK(n_same == 16);
    TR_CHECK(n_raw == 16);
#endif
    for (int k = 0; k < 8; k++) free(b[k].out);
    remove(lout);
    remove(out_path);
    remove(err_path);
    remove(in_path);
    TR_TEST_EXIT();
}
