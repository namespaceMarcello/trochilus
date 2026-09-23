/* bar.h — the command line's progress bar while a model loads: rainbow water poured into a tube
 * tilted down to the left, that pools and climbs to the right as the bytes arrive (variant A of
 * build/progress-bar/preview.py, class Tilted, approved as A.gif). The core draws nothing: it
 * reports bytes through tr_progress (src/models/model.h), and this file turns them into a line on
 * stderr.
 *
 * Three layers, the first two pure functions of their inputs so tests/test_bar.c runs them
 * without a terminal:
 *   water    the physics: virtual pipes between 40 columns, flow accelerating with the level
 *            difference, the missing volume poured in at the left (preview.py Tilted.step)
 *   render   one frame: 40 cells of eighth blocks in truecolor over the track, and the label
 *            "reading the model X.X/Y.Y GiB  NN%" (preview.py term, rainbow, label)
 *   bar      whether to draw at all (tr_bar_wanted), and when: at the first report, then at most
 *            TR_BAR_FPS times a second on the clock, the physics run in fixed substeps up to the
 *            clock's time; the line is cleared ("\r\x1b[2K") by the last report of a load
 *            (done == total) and by tr_bar_close, which every load ends with, failed or not.
 * Nothing is ever written to stdout.
 *
 * The block glyphs (U+2581..U+2588) are East Asian Width Ambiguous: a terminal that draws
 * ambiguous characters two columns wide breaks the layout this file computes for one. TR_BAR=0 is
 * the way out. */
#ifndef TR_BAR_H
#define TR_BAR_H

#include <stddef.h>
#include <stdint.h>

#include "../base/threads.h"
#include "../models/model.h"

#define TR_BAR_W 40          /* columns of the tube */
#define TR_BAR_FPS 30        /* frames a second, at most */
#define TR_BAR_SUB 16        /* physics substeps a frame: dt = 1 / (TR_BAR_FPS * TR_BAR_SUB) */
#define TR_BAR_MAX_STEPS (TR_BAR_FPS * TR_BAR_SUB) /* substeps one frame may catch up: 1 s */
#define TR_BAR_LINE_MAX 4096 /* one frame's bytes: 40 cells of at most 38 bytes, and the label */
#define TR_BAR_CLEAR "\r\x1b[2K"
/* no frame is drawn before this much time has passed since the load started: a load of a few
 * milliseconds must not flash one frame and a clear (review finding 8). */
#define TR_BAR_DELAY_SEC 0.2

/* ---- water: the physics (preview.py Tilted) ---- */
typedef struct {
    double bed[TR_BAR_W];    /* the tube's floor under each column, rising to the right */
    double d[TR_BAR_W];      /* water depth over the floor; a cell shows min(1, d) */
    double f[TR_BAR_W - 1];  /* flow from column i to column i + 1 */
} tr_bar_water;

void tr_bar_water_init(tr_bar_water *w);
/* One substep of dt seconds towards the volume progress p (0..1) asks for. */
void tr_bar_water_step(tr_bar_water *w, double p, double dt);
/* What column i shows, in eighths of a cell: 0..8 (round half to even, as preview.py's round). */
int tr_bar_cell_eighths(const tr_bar_water *w, int i);

/* ---- render: one frame ---- */
/* The line fitted to a terminal `width` columns wide, never wrapping: at most width - 1 columns
 * (the last one stays empty), the longest label that still leaves its minimum of cells, and as
 * many cells as are left, up to TR_BAR_W. Labels, longest first: FULL " reading the model  X.X/Y.Y
 * GiB  NN%" with at least 20 cells, SHORT " X.X/Y.Y GiB  NN%" with 10, PCT " NN%" with 5, NONE
 * with 1; not even 1 cell: 0 cells, nothing is drawn. width <= 0 (unknown): TR_BAR_W cells and
 * FULL, preview.py's layout. label_len: each label's length in columns. */
typedef enum {
    TR_BAR_LABEL_NONE, TR_BAR_LABEL_PCT, TR_BAR_LABEL_SHORT, TR_BAR_LABEL_FULL, TR_BAR_LABELS
} tr_bar_label;
typedef struct {
    int cells;
    tr_bar_label label;
} tr_bar_fit;
tr_bar_fit tr_bar_layout(int width, const int label_len[TR_BAR_LABELS]);

/* "\r", the cells (foreground: the rainbow at time t, seconds since the bar started; background:
 * the track; with fewer cells than TR_BAR_W columns, each shows the column under its centre),
 * "\x1b[0m" and the label for done of total bytes, laid out by tr_bar_layout for `width`, into
 * out (NUL terminated). Returns the bytes written, without the NUL; 0 if cap is too small or the
 * layout leaves no cell. */
size_t tr_bar_render(const tr_bar_water *w, double t, uint64_t done, uint64_t total, int width, char *out,
                     size_t cap);

/* ---- bar: whether, when, where ---- */
/* 1 when the bar is wanted: TR_BAR is not "0", NO_COLOR is unset or empty (no-color.org), TERM
 * is not "dumb", and then terminal(ctx) says stderr is a terminal that takes ANSI sequences --
 * asked last, since on Windows it switches them on. The strings are the environment's values
 * (NULL: unset), injected so a test decides without a real terminal. */
int tr_bar_wanted(const char *env_bar, const char *no_color, const char *term, int (*terminal)(void *ctx),
                  void *ctx);

typedef double (*tr_bar_clock_fn)(void *ctx);                           /* seconds, monotonic */
typedef void (*tr_bar_write_fn)(void *ctx, const char *bytes, size_t n); /* to the terminal */

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004 /* wincon.h's own value; defined here too so
                                                    * tests/test_bar.c reads it without <windows.h> */
#endif

/* Windows console mode before -> to set, and whether tr_bar_close must restore `before`: pure, so
 * tests/test_bar.c checks it without a console (review finding 5). Built on every platform (the
 * logic itself is not Windows-specific, only its caller is). */
typedef struct {
    unsigned long to_set;
    int restore;
} tr_bar_console_plan;
tr_bar_console_plan tr_bar_console_mode_plan(unsigned long before);

typedef struct {
    int on;                          /* 0: every call below does nothing */
    int width;                       /* the terminal's columns, read once at init; 0: unknown */
    tr_bar_water water;
    double start, last_draw;         /* clock readings */
    int64_t steps;                   /* physics substeps the clock has asked for since start */
    int drawn;                       /* a frame is on the line and not cleared yet */
    int64_t steps_run, frames, clears; /* substeps actually run, frames and clears written (tests) */
    tr_bar_clock_fn clock;
    void *clock_ctx;
    tr_bar_write_fn write;
    void *write_ctx;
    unsigned long console_mode;      /* Windows: the console's mode before the bar switched ANSI on */
    int console_restore;             /* 1: tr_bar_close puts console_mode back */
    char line[TR_BAR_LINE_MAX];
} tr_bar;

/* A bar drawing through write on clock, or doing nothing when on is 0 (tests inject both). */
void tr_bar_init(tr_bar *b, int on, tr_bar_clock_fn clock, void *clock_ctx, tr_bar_write_fn write,
                 void *write_ctx);
/* The real one: tr_bar_wanted on this process's environment and stderr, the monotonic clock
 * (tr_time_sec), stderr itself (one write per frame), and stderr's width in columns (POSIX:
 * TIOCGWINSZ; Windows: the console window's), read once. */
void tr_bar_init_stderr(tr_bar *b);
/* The tr_progress callback: ctx is the tr_bar. */
void tr_bar_progress(void *ctx, uint64_t done, uint64_t total);
/* Clears the line if a frame is still on it, and gives the console its mode back. Every load
 * ends here, failed or not; calling it again does nothing. */
void tr_bar_close(tr_bar *b);

/* tr_model_load_progress with b reporting (b->on == 0: no progress asked for at all), then
 * tr_bar_close(b) whatever the outcome. */
tr_model *tr_bar_load_with(tr_bar *b, const char *path, tr_pool *pool, uint64_t expert_budget, char *err,
                           size_t err_len);
/* The same with tr_bar_init_stderr: what the commands load through (src/app/serve.c app_model). */
tr_model *tr_bar_load(const char *path, tr_pool *pool, uint64_t expert_budget, char *err, size_t err_len);

#endif
