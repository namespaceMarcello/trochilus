/* bar.c — the progress bar while a model loads (bar.h). The physics, the colours and the frame
 * are ported from build/progress-bar/preview.py, variant A (class Tilted, rainbow, label, term),
 * step for step and constant for constant, so the terminal shows what A.gif shows. */
#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* isatty under -std=c11 */
#endif
#include "bar.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../base/platform.h"

/* ENABLE_VIRTUAL_TERMINAL_PROCESSING: defined in bar.h (tr_bar_console_mode_plan is pure and built
 * on every platform, review finding 5). */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#define BAR_RISE 4.0                                  /* how much the tube's floor climbs */
#define BAR_DT (1.0 / (TR_BAR_FPS * TR_BAR_SUB))       /* one substep */
static const unsigned char BAR_TRACK[3] = {46, 46, 52}; /* the empty tube */
/* " ▁▂▃▄▅▆▇█": a space, then U+2581..U+2588 in UTF-8 */
static const char *const BAR_EIGHTHS[9] = {" ", "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84",
                                           "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88"};

/* ---- water (preview.py Tilted) ---- */

void tr_bar_water_init(tr_bar_water *w) {
    for (int i = 0; i < TR_BAR_W; i++) {
        w->bed[i] = BAR_RISE * (i + 0.5) / TR_BAR_W;
        w->d[i] = 0.0;
    }
    for (int i = 0; i < TR_BAR_W - 1; i++) w->f[i] = 0.0;
}

/* The volume progress p asks for: water up to a flat level (rise + 1) * p over the sloping floor. */
static double water_target(const tr_bar_water *w, double p) {
    double level = (BAR_RISE + 1.0) * p, sum = 0.0;
    for (int i = 0; i < TR_BAR_W; i++) sum += level - w->bed[i] > 0.0 ? level - w->bed[i] : 0.0;
    return sum;
}

void tr_bar_water_step(tr_bar_water *w, double p, double dt) {
    const double g = 1500.0, damp = 0.997;
    const int n = TR_BAR_W;
    /* virtual pipes: the flow between neighbours accelerates with their level difference */
    for (int i = 0; i < n - 1; i++)
        w->f[i] = w->f[i] * damp + dt * g * ((w->bed[i] + w->d[i]) - (w->bed[i + 1] + w->d[i + 1]));
    /* a column never gives more water than it holds: its outflows are scaled down together */
    for (int i = 0; i < n; i++) {
        double right = i < n - 1 && w->f[i] > 0.0 ? w->f[i] : 0.0;      /* max(f[i], 0) */
        double left = i > 0 && -w->f[i - 1] > 0.0 ? -w->f[i - 1] : 0.0; /* max(-f[i - 1], 0) */
        double out = right + left;
        if (out * dt > w->d[i] && out > 0.0) {
            double k = w->d[i] / (out * dt);
            if (i < n - 1 && w->f[i] > 0.0) w->f[i] *= k;
            if (i > 0 && w->f[i - 1] < 0.0) w->f[i - 1] *= k;
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        w->d[i] += dt * ((i > 0 ? w->f[i - 1] : 0.0) - (i < n - 1 ? w->f[i] : 0.0));
        if (w->d[i] < 0.0) w->d[i] = 0.0;
        sum += w->d[i];
    }
    /* the pour at the left: whatever volume progress asks for and the tube does not hold yet */
    double rate = 30.0 * dt < 1.0 ? 30.0 * dt : 1.0;
    w->d[0] += (water_target(w, p) - sum) * rate;
}

int tr_bar_cell_eighths(const tr_bar_water *w, int i) {
    /* the pour at the left can leave d[0] below 0 for a substep when progress goes back (seen in
     * tests/test_bar.c under a jumping schedule; preview.py would index EIGHTHS from its end) */
    double h = w->d[i] < 1.0 ? w->d[i] : 1.0;
    if (!(h > 0.0)) return 0;
    return (int)nearbyint(h * 8.0); /* round half to even, as Python's round: 0..8 */
}

/* ---- render (preview.py rainbow, label, term) ---- */

/* colorsys.hsv_to_rgb */
static void hsv_to_rgb(double h, double s, double v, double rgb[3]) {
    int i = (int)(h * 6.0);
    double f = h * 6.0 - i, p = v * (1.0 - s), q = v * (1.0 - s * f), t = v * (1.0 - s * (1.0 - f));
    double r, g, b;
    switch (i % 6) {
    case 0: r = v, g = t, b = p; break;
    case 1: r = q, g = v, b = p; break;
    case 2: r = p, g = v, b = t; break;
    case 3: r = p, g = q, b = v; break;
    case 4: r = t, g = p, b = v; break;
    default: r = v, g = p, b = q; break;
    }
    rgb[0] = r, rgb[1] = g, rgb[2] = b;
}

/* rainbow(x, t, light=0, shade): the hue runs along the tube and drifts with time */
static void rainbow(int x, double t, double shade, int out[3]) {
    double h = (double)x / TR_BAR_W * 0.85 - t * 0.12;
    h -= floor(h); /* Python's % 1.0: always in [0, 1) */
    double rgb[3];
    hsv_to_rgb(h, 0.85, 1.0, rgb);
    for (int k = 0; k < 3; k++) {
        double c = rgb[k] * shade;
        out[k] = (int)(255.0 * (c < 1.0 ? c : 1.0));
    }
}

typedef struct {
    char *out;
    size_t cap, len;
    int full;
} line_buf;

static void put(line_buf *lb, const char *fmt, ...) {
    if (lb->full) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(lb->out + lb->len, lb->cap - lb->len, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= lb->cap - lb->len) {
        lb->full = 1;
        return;
    }
    lb->len += (size_t)n;
}

tr_bar_fit tr_bar_layout(int width, const int label_len[TR_BAR_LABELS]) {
    tr_bar_fit fit = {TR_BAR_W, TR_BAR_LABEL_FULL};
    if (width <= 0) return fit; /* unknown: the layout preview.py draws */
    /* the last column stays empty: a terminal that wraps on writing it would take the next "\r"
     * to a new line */
    int avail = width - 1;
    static const int min_cells[TR_BAR_LABELS] = {1, 5, 10, 20}; /* by label, NONE..FULL */
    for (int k = TR_BAR_LABEL_FULL; k >= TR_BAR_LABEL_NONE; k--) {
        int cells = avail - label_len[k];
        if (cells > TR_BAR_W) cells = TR_BAR_W;
        if (cells >= min_cells[k]) {
            fit.cells = cells;
            fit.label = (tr_bar_label)k;
            return fit;
        }
    }
    fit.cells = 0; /* not even one cell: nothing is drawn */
    fit.label = TR_BAR_LABEL_NONE;
    return fit;
}

size_t tr_bar_render(const tr_bar_water *w, double t, uint64_t done, uint64_t total, int width, char *out,
                     size_t cap) {
    if (out == NULL || cap == 0) return 0;
    /* the labels, longest first to shortest (all ASCII: one byte, one column) */
    char label[TR_BAR_LABELS][96];
    double gib = 1024.0 * 1024.0 * 1024.0;
    int pct = (int)(total > 0 ? (done < total ? done : total) * 100 / total : 100);
    label[TR_BAR_LABEL_NONE][0] = 0;
    snprintf(label[TR_BAR_LABEL_PCT], sizeof label[0], " %3d%%", pct);
    snprintf(label[TR_BAR_LABEL_SHORT], sizeof label[0], " %3.1f/%.1f GiB  %3d%%", (double)done / gib,
             (double)total / gib, pct);
    snprintf(label[TR_BAR_LABEL_FULL], sizeof label[0], " reading the model  %3.1f/%.1f GiB  %3d%%",
             (double)done / gib, (double)total / gib, pct);
    int label_len[TR_BAR_LABELS];
    for (int k = 0; k < TR_BAR_LABELS; k++) label_len[k] = (int)strlen(label[k]);
    tr_bar_fit fit = tr_bar_layout(width, label_len);
    out[0] = 0;
    if (fit.cells == 0) return 0;

    line_buf lb = {out, cap, 0, 0};
    put(&lb, "\r");
    /* the track's background never changes across cells or frames: set once instead of in every
     * cell (review finding 2 -- it was ~40% of each frame's bytes, written up to 30 times a
     * second). The visible result is identical: every cell would have set the very same colour. */
    put(&lb, "\x1b[48;2;%d;%d;%dm", BAR_TRACK[0], BAR_TRACK[1], BAR_TRACK[2]);
    for (int j = 0; j < fit.cells; j++) {
        /* fewer cells than columns: each shows the column under its centre, colour included */
        int i = (int)((j + 0.5) * TR_BAR_W / fit.cells);
        double shine = 0.88 + 0.12 * sin(i * 0.55 - t * 7.0);
        int rgb[3];
        rainbow(i, t, shine, rgb);
        put(&lb, "\x1b[38;2;%d;%d;%dm%s", rgb[0], rgb[1], rgb[2], BAR_EIGHTHS[tr_bar_cell_eighths(w, i)]);
    }
    put(&lb, "\x1b[0m%s", label[fit.label]);
    if (lb.full) {
        out[0] = 0;
        return 0;
    }
    return lb.len;
}

/* ---- bar ---- */

int tr_bar_wanted(const char *env_bar, const char *no_color, const char *term, int (*terminal)(void *ctx),
                  void *ctx) {
    if (env_bar != NULL && strcmp(env_bar, "0") == 0) return 0;
    if (no_color != NULL && no_color[0] != 0) return 0;
    if (term != NULL && strcmp(term, "dumb") == 0) return 0;
    return terminal != NULL && terminal(ctx) != 0;
}

void tr_bar_init(tr_bar *b, int on, tr_bar_clock_fn clock, void *clock_ctx, tr_bar_write_fn write,
                 void *write_ctx) {
    memset(b, 0, sizeof *b);
    tr_bar_water_init(&b->water);
    b->on = on && clock != NULL && write != NULL;
    b->clock = clock;
    b->clock_ctx = clock_ctx;
    b->write = write;
    b->write_ctx = write_ctx;
    if (b->on) b->start = b->last_draw = clock(clock_ctx);
}

static double clock_real(void *ctx) {
    (void)ctx;
    return tr_time_sec();
}

/* One frame, one write: a frame split across writes could show half drawn, and on Windows a
 * UTF-8 character split across two console writes may not be put back together. */
static void write_stderr(void *ctx, const char *bytes, size_t n) {
    (void)ctx;
    fflush(stderr); /* whatever the C library still holds goes first */
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    while (n > 0) {
        DWORD wrote = 0;
        if (!WriteFile(h, bytes, (DWORD)n, &wrote, NULL) || wrote == 0) return;
        bytes += wrote;
        n -= wrote;
    }
#else
    fwrite(bytes, 1, n, stderr);
    fflush(stderr);
#endif
}

tr_bar_console_plan tr_bar_console_mode_plan(unsigned long before) {
    tr_bar_console_plan plan;
    if ((before & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0) {
        plan.to_set = before; /* already on: nothing to restore */
        plan.restore = 0;
    } else {
        plan.to_set = before | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        plan.restore = 1; /* tr_bar_close must put `before` back */
    }
    return plan;
}

/* stderr is a terminal that takes ANSI sequences. Windows: a console whose mode takes
 * ENABLE_VIRTUAL_TERMINAL_PROCESSING (before -> to set, and whether tr_bar_close must restore it:
 * tr_bar_console_mode_plan, checked directly by tests/test_bar.c, review finding 5). POSIX: isatty
 * alone passes a process started in the background of an interactive shell (job control leaves its
 * stderr on the tty), which would draw over the prompt and whose final clear would wipe the line
 * being typed; tcgetpgrp(2) == getpgrp() asks whether this process is in the tty's foreground
 * group, as job control tracks it (review finding 1). */
static int stderr_terminal(void *ctx) {
    tr_bar *b = (tr_bar *)ctx;
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode = 0;
    if (h == NULL || h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) return 0;
    tr_bar_console_plan plan = tr_bar_console_mode_plan(mode);
    if (plan.restore) {
        if (!SetConsoleMode(h, (DWORD)plan.to_set)) return 0;
        b->console_mode = mode;
        b->console_restore = 1;
    }
    return 1;
#else
    (void)b;
    if (!isatty(2)) return 0;
    return tcgetpgrp(2) == getpgrp();
#endif
}

/* stderr's width in columns; 0 when it cannot be told */
static int stderr_width(void) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &info)) return 0;
    return info.srWindow.Right - info.srWindow.Left + 1;
#else
    struct winsize ws;
    if (ioctl(2, TIOCGWINSZ, &ws) != 0) return 0;
    return ws.ws_col;
#endif
}

void tr_bar_init_stderr(tr_bar *b) {
    tr_bar_init(b, 0, NULL, NULL, NULL, NULL);
    if (!tr_bar_wanted(getenv("TR_BAR"), getenv("NO_COLOR"), getenv("TERM"), stderr_terminal, b)) return;
    unsigned long mode = b->console_mode;
    int restore = b->console_restore;
    tr_bar_init(b, 1, clock_real, NULL, write_stderr, NULL);
    b->console_mode = mode;
    b->console_restore = restore;
    /* an unknown width (0: the ioctl/GetConsoleScreenBufferInfo call failed, or a tty that itself
     * reports 0 columns) is read once here and would otherwise mean tr_bar_layout's "unknown: draw
     * TR_BAR_W cells and the full label", unbounded up to 81 columns on a real, narrow terminal
     * (review finding 7); 80 is the width assumed instead. */
    int w = stderr_width();
    b->width = w > 0 ? w : 80;
}

static void bar_clear(tr_bar *b) {
    if (!b->drawn) return;
    b->write(b->write_ctx, TR_BAR_CLEAR, sizeof TR_BAR_CLEAR - 1);
    b->drawn = 0;
    b->clears++;
}

void tr_bar_progress(void *ctx, uint64_t done, uint64_t total) {
    tr_bar *b = (tr_bar *)ctx;
    if (!b->on) return;
    if (done >= total) { /* the load's last report: nothing stays on the line */
        bar_clear(b);
        return;
    }
    double now = b->clock(b->clock_ctx);
    if (!b->drawn && now - b->start < TR_BAR_DELAY_SEC) return; /* too soon: nothing flashes */
    if (b->drawn && now - b->last_draw < 1.0 / TR_BAR_FPS) return;
    /* the physics catch up with the clock in fixed substeps (preview.py: SUB per frame), at most
     * TR_BAR_MAX_STEPS of them: after a long stall the water resumes instead of racing */
    int64_t due = now > b->start ? (int64_t)((now - b->start) / BAR_DT) : 0;
    int64_t n = due - b->steps;
    if (n > TR_BAR_MAX_STEPS) n = TR_BAR_MAX_STEPS;
    double p = (double)done / (double)total;
    for (int64_t k = 0; k < n; k++) tr_bar_water_step(&b->water, p, BAR_DT);
    if (n > 0) {
        b->steps = due; /* past the cap, the rest of the stall is skipped, not owed */
        b->steps_run += n;
    }
    size_t len = tr_bar_render(&b->water, now - b->start, done, total, b->width, b->line, sizeof b->line);
    if (len == 0) return;
    b->write(b->write_ctx, b->line, len);
    b->drawn = 1;
    b->last_draw = now;
    b->frames++;
}

void tr_bar_close(tr_bar *b) {
    if (b->on) bar_clear(b);
    b->on = 0;
#ifdef _WIN32
    if (b->console_restore) {
        SetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), (DWORD)b->console_mode);
        b->console_restore = 0;
    }
#endif
}

tr_model *tr_bar_load_with(tr_bar *b, const char *path, tr_pool *pool, uint64_t expert_budget, char *err,
                           size_t err_len) {
    tr_progress progress = {tr_bar_progress, b};
    tr_model *m = tr_model_load_progress(path, pool, expert_budget, b->on ? &progress : NULL, err, err_len);
    tr_bar_close(b);
    return m;
}

tr_model *tr_bar_load(const char *path, tr_pool *pool, uint64_t expert_budget, char *err, size_t err_len) {
    tr_bar b;
    tr_bar_init_stderr(&b);
    return tr_bar_load_with(&b, path, pool, expert_budget, err, err_len);
}
