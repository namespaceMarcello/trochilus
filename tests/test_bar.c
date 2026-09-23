/* test_bar.c — the command line's progress bar (src/app/bar.c): its physics, its frame, whether
 * it draws, when, and that nothing stays on the line. No terminal is needed: the clock, the
 * writer and the environment are injected.
 *
 * Branches, each counted (a case that never ran fails the test, CLAUDE.md, LESSONS #43 #50):
 *   physics   the water under a steady load (reports at an even rate, as a real load gives them),
 *             under preview.py's own bursty schedule, at progress 0, and under a random, jumping
 *             one: never NaN nor infinite, every cell 0..8 eighths; the front (the rightmost cell
 *             showing water) never recedes more than 1 cell under the steady load, nor more than
 *             8 under the bursty one (the slosh A.gif shows: preview.py itself recedes 8 there);
 *             full at progress 1 once settled; empty at progress 0
 *   render    one frame of a fixed state, byte for byte against the golden string preview.py's
 *             own rainbow/EIGHTHS/term give (depths k/32: halves of an eighth round to even, as
 *             Python's round); too small a buffer gives 0
 *   layout    the line on a terminal 0 (unknown), 200, 78, 60, 40, 20, 5 and 1 columns wide: the
 *             cells and label tr_bar_layout picks for each, every frame at most width - 1 columns
 *             wide, 78 and wider the golden frame itself, a 40-column frame made of the golden
 *             frame's cells for the columns under its 21 cells, and nothing at all at width 1
 *   decide    off for TR_BAR=0, NO_COLOR set, TERM=dumb (the terminal never asked), and a
 *             non-terminal; on for a terminal with NO_COLOR empty, TERM unset or not dumb, TR_BAR=1
 *   console   tr_bar_console_mode_plan (the Windows console-mode switch, pure and built on every
 *             platform): VT off asks to set it and to restore the old mode; VT already on asks for
 *             neither
 *   draw      nothing is drawn before TR_BAR_DELAY_SEC has passed since the load started; past it
 *             the first report draws at once, then at most TR_BAR_FPS frames a second; the physics
 *             follow the clock in substeps of 1/(FPS*SUB) s, at most TR_BAR_MAX_STEPS in one frame
 *             after a stall; the last report (done == total) clears the line, and tr_bar_close
 *             adds nothing
 *   fail      a bar left drawn (no last report) is cleared by tr_bar_close, once; after it nothing
 *             is written
 *   quiet     a bar that is off, or that never drew, writes nothing at all
 *   load      tr_bar_load_with on a synthetic model: drawn and cleared by the last report; a load
 *             failing after it drew (a tensor missing late in the file) cleared by tr_bar_close; a
 *             bar that is off writes nothing and the model loads the same
 *
 * Seen red: tools/mutate_bar.sh (each mutation, the check it turns red). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "synth_olmoe.h"
#include "../src/app/bar.h"

static struct {
    int physics_steady, physics_bursty, physics_empty, physics_random;
    int render, render_small, layout, layout_frames, layout_columns;
    int decide_off_bar, decide_off_no_color, decide_off_dumb, decide_off_pipe, decide_on;
    int draw_delay, draw_first, draw_throttle, draw_steps, draw_stall, draw_end;
    int fail_close, quiet_off, quiet_never_drew;
    int load_ok, load_fail, load_off;
    int console_restore, console_no_restore;
} n; /* global-ok: this test's own branch counters */

/* ---- injected clock and writer ---- */

typedef struct {
    double now, tick; /* tick: added after every reading (0: the test moves the clock itself) */
} fake_clock;

static double fake_now(void *ctx) {
    fake_clock *c = (fake_clock *)ctx;
    double t = c->now;
    c->now += c->tick;
    return t;
}

typedef struct {
    int writes;
    size_t bytes;
    char last[TR_BAR_LINE_MAX];
    size_t last_len;
} capture;

static void capture_write(void *ctx, const char *bytes, size_t len) {
    capture *c = (capture *)ctx;
    c->writes++;
    c->bytes += len;
    c->last_len = len < sizeof c->last ? len : sizeof c->last;
    memcpy(c->last, bytes, c->last_len);
}

static int last_is_clear(const capture *c) {
    return c->last_len == strlen(TR_BAR_CLEAR) && memcmp(c->last, TR_BAR_CLEAR, c->last_len) == 0;
}

static int last_is_frame(const capture *c) {
    /* "\r" then the track's background, set once (review finding 2) before the first cell */
    return c->last_len > 8 && memcmp(c->last, "\r\x1b[48;2;", 8) == 0;
}

/* ---- physics ---- */

/* Every depth and flow finite, every cell 0..8: 1 if so. */
static int water_sane(const tr_bar_water *w) {
    for (int i = 0; i < TR_BAR_W; i++) {
        if (!isfinite(w->d[i])) return 0;
        if (i < TR_BAR_W - 1 && !isfinite(w->f[i])) return 0;
        int e = tr_bar_cell_eighths(w, i);
        if (e < 0 || e > 8) return 0;
    }
    return 1;
}

/* The rightmost cell showing water, + 1 (0: none). */
static int water_front(const tr_bar_water *w) {
    int f = 0;
    for (int i = 0; i < TR_BAR_W; i++)
        if (tr_bar_cell_eighths(w, i) >= 1) f = i + 1;
    return f;
}

/* preview.py progress_at: bursts and stalls, full at 4.2 s */
static double bursty(double t) {
    static const double k[8][2] = {{0, 0},      {0.6, 0.18}, {1.0, 0.2},  {1.8, 0.52},
                                   {2.3, 0.55}, {3.2, 0.86}, {3.5, 0.87}, {4.2, 1.0}};
    for (int i = 0; i < 7; i++)
        if (t <= k[i + 1][0]) return k[i][1] + (k[i + 1][1] - k[i][1]) * (t - k[i][0]) / (k[i + 1][0] - k[i][0]);
    return 1.0;
}

/* a load reporting 1200 times at an even rate, one every 4 ms: 4.8 s, as OLMoE's full load */
static double steady(double t) {
    double p = floor(t / 0.004) / 1200.0;
    return p < 1.0 ? p : 1.0;
}

/* Runs `seconds` of frames (TR_BAR_SUB substeps each) under sched; returns the largest recession of
 * the front below its best so far, or -1 if the water ever went insane. */
static int run_schedule(tr_bar_water *w, double (*sched)(double), double seconds, int *frames) {
    const double dt = 1.0 / (TR_BAR_FPS * TR_BAR_SUB);
    int best = 0, worst = 0;
    for (int k = 0; k < (int)(seconds * TR_BAR_FPS); k++) {
        double p = sched((double)k / TR_BAR_FPS);
        for (int s = 0; s < TR_BAR_SUB; s++) tr_bar_water_step(w, p, dt);
        if (!water_sane(w)) return -1;
        int fr = water_front(w);
        if (fr > best) best = fr;
        if (best - fr > worst) worst = best - fr;
        (*frames)++;
    }
    return worst;
}

static double one(double t) {
    (void)t;
    return 1.0;
}

static double zero(double t) {
    (void)t;
    return 0.0;
}

static void test_physics(void) {
    tr_bar_water w;
    int frames = 0;

    tr_bar_water_init(&w);
    TR_CHECK(water_sane(&w));
    TR_CHECK_EQ_INT(water_front(&w), 0);
    int back = run_schedule(&w, steady, 4.8, &frames);
    TR_CHECK(back >= 0);
    TR_CHECK(back <= 1); /* steady: the front only climbs (a thin tip may flicker by a cell) */
    TR_CHECK(run_schedule(&w, one, 10.0, &frames) >= 0);
    int full = 0;
    for (int i = 0; i < TR_BAR_W; i++) full += tr_bar_cell_eighths(&w, i) == 8;
    TR_CHECK_EQ_INT(full, TR_BAR_W); /* settled at progress 1: every cell full */
    if (back >= 0 && back <= 1 && full == TR_BAR_W && frames > 0) n.physics_steady++;

    tr_bar_water_init(&w);
    back = run_schedule(&w, bursty, 12.0, &frames);
    TR_CHECK(back >= 0);
    TR_CHECK(back <= 8); /* the stalls slosh the front back, as A.gif does */
    full = 0;
    for (int i = 0; i < TR_BAR_W; i++) full += tr_bar_cell_eighths(&w, i) == 8;
    TR_CHECK_EQ_INT(full, TR_BAR_W);
    if (back >= 0) n.physics_bursty++;

    tr_bar_water_init(&w);
    TR_CHECK(run_schedule(&w, zero, 2.0, &frames) == 0);
    TR_CHECK_EQ_INT(water_front(&w), 0); /* no water from nowhere */
    n.physics_empty++;

    /* random progress, jumping both ways every frame: still finite, still 0..8 */
    tr_bar_water_init(&w);
    uint32_t seed = 12345;
    const double dt = 1.0 / (TR_BAR_FPS * TR_BAR_SUB);
    int sane = 1;
    for (int k = 0; k < 20 * TR_BAR_FPS; k++) {
        seed = seed * 1103515245u + 12345u;
        double p = (double)((seed >> 8) & 0xFFFF) / 65535.0;
        for (int s = 0; s < TR_BAR_SUB; s++) tr_bar_water_step(&w, p, dt);
        sane &= water_sane(&w);
        n.physics_random++;
    }
    TR_CHECK(sane);
}

/* ---- render ---- */

/* preview.py's own rainbow, EIGHTHS, TRACK and term for depths ((i * 7) % 41) / 32 at t = 0.75,
 * 3 GiB + 12345 of 6.5 GiB (generated with preview.py, not with bar.c). preview.py sets the track's
 * background colour in every cell; bar.c sets it once, right after "\r" (review finding 2 -- the
 * same 40 repeats of "\x1b[48;2;46;46;52m" were ~40% of every frame's bytes). A terminal reads
 * either byte stream the same way: an SGR background stays in effect until the next one changes or
 * clears it, so the visible frame is identical either way; only this golden string's bytes moved. */
static const char GOLDEN[] =
    "\r"
    "\x1b[48;2;46;46;52m"
    "\x1b[38;2;250;37;152m "
    "\x1b[38;2;254;38;127m\xe2\x96\x82"
    "\x1b[38;2;250;37;98m\xe2\x96\x84"
    "\x1b[38;2;237;35;67m\xe2\x96\x85"
    "\x1b[38;2;221;33;38m\xe2\x96\x87"
    "\x1b[38;2;206;47;30m\xe2\x96\x88"
    "\x1b[38;2;195;66;29m "
    "\x1b[38;2;194;87;29m\xe2\x96\x82"
    "\x1b[38;2;201;112;30m\xe2\x96\x84"
    "\x1b[38;2;215;143;32m\xe2\x96\x86"
    "\x1b[38;2;231;179;34m\xe2\x96\x87"
    "\x1b[38;2;246;217;36m\xe2\x96\x88"
    "\x1b[38;2;254;252;38m "
    "\x1b[38;2;228;253;38m\xe2\x96\x82"
    "\x1b[38;2;193;243;36m\xe2\x96\x84"
    "\x1b[38;2;156;228;34m\xe2\x96\x86"
    "\x1b[38;2;122;212;31m\xe2\x96\x88"
    "\x1b[38;2;93;199;29m\xe2\x96\x88"
    "\x1b[38;2;69;193;29m\xe2\x96\x81"
    "\x1b[38;2;49;197;29m\xe2\x96\x82"
    "\x1b[38;2;31;208;33m\xe2\x96\x84"
    "\x1b[38;2;33;224;60m\xe2\x96\x86"
    "\x1b[38;2;36;240;90m\xe2\x96\x88"
    "\x1b[38;2;37;251;121m\xe2\x96\x88"
    "\x1b[38;2;38;254;150m\xe2\x96\x81"
    "\x1b[38;2;37;248;174m\xe2\x96\x83"
    "\x1b[38;2;35;235;190m\xe2\x96\x84"
    "\x1b[38;2;32;219;200m\xe2\x96\x86"
    "\x1b[38;2;30;198;204m\xe2\x96\x88"
    "\x1b[38;2;29;169;195m\xe2\x96\x88"
    "\x1b[38;2;29;147;194m\xe2\x96\x81"
    "\x1b[38;2;30;131;203m\xe2\x96\x83"
    "\x1b[38;2;32;117;217m\xe2\x96\x85"
    "\x1b[38;2;35;101;234m\xe2\x96\x86"
    "\x1b[38;2;37;80;248m\xe2\x96\x88"
    "\x1b[38;2;38;54;254m\xe2\x96\x88"
    "\x1b[38;2;48;37;252m\xe2\x96\x82"
    "\x1b[38;2;72;36;241m\xe2\x96\x83"
    "\x1b[38;2;92;33;226m\xe2\x96\x85"
    "\x1b[38;2;108;31;209m\xe2\x96\x87"
    "\x1b[0m reading the model  3.0/6.5 GiB   46%";

static void test_render(void) {
    tr_bar_water w;
    tr_bar_water_init(&w);
    for (int i = 0; i < TR_BAR_W; i++) w.d[i] = (double)((i * 7) % 41) / 32.0;
    char out[TR_BAR_LINE_MAX];
    uint64_t done = (uint64_t)3 * 1024 * 1024 * 1024 + 12345, total = 6979321856u;
    size_t len = tr_bar_render(&w, 0.75, done, total, 0, out, sizeof out);
    TR_CHECK_EQ_INT(len, sizeof GOLDEN - 1);
    TR_CHECK(len == sizeof GOLDEN - 1 && memcmp(out, GOLDEN, len) == 0);
    if (len != sizeof GOLDEN - 1 || memcmp(out, GOLDEN, len) != 0) {
        for (size_t i = 0; i < len && i < sizeof GOLDEN - 1; i++)
            if (out[i] != GOLDEN[i]) {
                fprintf(stderr, "  first difference at byte %zu\n", i);
                break;
            }
    }
    n.render++;

    TR_CHECK_EQ_INT(tr_bar_render(&w, 0.75, done, total, 0, out, 100), 0); /* too small: nothing */
    TR_CHECK_EQ_INT(out[0], 0);
    n.render_small++;
}

/* ---- layout: the line on a narrow terminal ---- */

/* Columns a frame takes on the terminal: "\r" and escape sequences take none, a UTF-8 character one. */
static int visible_columns(const char *s, size_t len) {
    int cols = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\r') continue;
        if (c == 0x1b) { /* ESC [ parameters final-letter */
            i++;
            while (i + 1 < len && !((s[i + 1] >= 'A' && s[i + 1] <= 'Z') || (s[i + 1] >= 'a' && s[i + 1] <= 'z')))
                i++;
            i++;
            continue;
        }
        if ((c & 0xC0) != 0x80) cols++;
    }
    return cols;
}

static void test_layout(void) {
    /* the golden frame's labels: FULL " reading the model  3.0/6.5 GiB   46%", SHORT
     * " 3.0/6.5 GiB   46%", PCT "  46%" */
    const int lens[TR_BAR_LABELS] = {0, 5, 18, 37};
    static const struct {
        int width, cells;
        tr_bar_label label;
    } want[] = {
        {0, 40, TR_BAR_LABEL_FULL},   /* unknown: today's layout */
        {200, 40, TR_BAR_LABEL_FULL}, /* wide: never more than TR_BAR_W cells */
        {78, 40, TR_BAR_LABEL_FULL},  /* 40 + 37 = 77 = 78 - 1: exactly fits */
        {60, 22, TR_BAR_LABEL_FULL},  /* the label stays, the tube shortens */
        {40, 21, TR_BAR_LABEL_SHORT}, /* 20 cells with FULL do not fit: the short label */
        {20, 14, TR_BAR_LABEL_PCT},   /* only the percentage */
        {5, 4, TR_BAR_LABEL_NONE},    /* only the tube */
        {1, 0, TR_BAR_LABEL_NONE},    /* nothing at all */
    };
    for (size_t k = 0; k < sizeof want / sizeof want[0]; k++) {
        tr_bar_fit fit = tr_bar_layout(want[k].width, lens);
        TR_CHECK_EQ_INT(fit.cells, want[k].cells);
        TR_CHECK_EQ_INT(fit.label, want[k].label);
        if (fit.cells != want[k].cells || fit.label != want[k].label)
            fprintf(stderr, "  width %d: %d cells, label %d\n", want[k].width, fit.cells, (int)fit.label);
        n.layout++;
    }

    /* the frames themselves: never wider than width - 1, and 78 or more is the golden frame */
    tr_bar_water w;
    tr_bar_water_init(&w);
    for (int i = 0; i < TR_BAR_W; i++) w.d[i] = (double)((i * 7) % 41) / 32.0;
    char out[TR_BAR_LINE_MAX];
    uint64_t done = (uint64_t)3 * 1024 * 1024 * 1024 + 12345, total = 6979321856u;
    static const int widths[] = {20, 40, 60, 78, 200, 5};
    for (size_t k = 0; k < sizeof widths / sizeof widths[0]; k++) {
        size_t len = tr_bar_render(&w, 0.75, done, total, widths[k], out, sizeof out);
        TR_CHECK(len > 0);
        int cols = visible_columns(out, len);
        TR_CHECK(cols <= widths[k] - 1);
        if (cols > widths[k] - 1) fprintf(stderr, "  width %d: the frame takes %d columns\n", widths[k], cols);
        if (widths[k] >= 78) TR_CHECK(len == sizeof GOLDEN - 1 && memcmp(out, GOLDEN, len) == 0);
        n.layout_frames++;
    }
    /* a narrow frame shows the columns under its cells: 21 cells, cell j is column (2j + 1) * 40 / 42 */
    size_t len = tr_bar_render(&w, 0.75, done, total, 40, out, sizeof out);
    char want40[TR_BAR_LINE_MAX];
    size_t at = 0;
    at += (size_t)snprintf(want40 + at, sizeof want40 - at, "\r");
    at += (size_t)snprintf(want40 + at, sizeof want40 - at, "\x1b[48;2;46;46;52m"); /* set once (finding 2) */
    const char *base = GOLDEN + 1;         /* right after "\r": the background sequence */
    base = strchr(base + 1, '\x1b');       /* past it: cell 0's own escape */
    for (int j = 0; j < 21; j++) {
        int col = (int)((j + 0.5) * TR_BAR_W / 21);
        const char *cell = base; /* the golden frame's cell for that column: one escape each now */
        for (int c = 0; c < col; c++) cell = strchr(cell + 1, '\x1b');
        const char *end = strchr(cell + 1, '\x1b');
        at += (size_t)snprintf(want40 + at, sizeof want40 - at, "%.*s", (int)(end - cell), cell);
    }
    at += (size_t)snprintf(want40 + at, sizeof want40 - at, "\x1b[0m 3.0/6.5 GiB   46%%");
    TR_CHECK_EQ_INT(len, at);
    TR_CHECK(len == at && memcmp(out, want40, len) == 0);
    n.layout_columns++;
    /* width 1: no cell fits, nothing drawn */
    TR_CHECK_EQ_INT(tr_bar_render(&w, 0.75, done, total, 1, out, sizeof out), 0);
}

/* ---- decide ---- */

typedef struct {
    int answer, asked;
} fake_terminal;

static int terminal_probe(void *ctx) {
    fake_terminal *t = (fake_terminal *)ctx;
    t->asked++;
    return t->answer;
}

static void test_decide(void) {
    fake_terminal tty = {1, 0}, pipe = {0, 0};

    TR_CHECK_EQ_INT(tr_bar_wanted("0", NULL, NULL, terminal_probe, &tty), 0);
    TR_CHECK_EQ_INT(tty.asked, 0); /* decided before the terminal was touched */
    n.decide_off_bar++;

    TR_CHECK_EQ_INT(tr_bar_wanted(NULL, "1", NULL, terminal_probe, &tty), 0);
    TR_CHECK_EQ_INT(tr_bar_wanted(NULL, "anything", "xterm-256color", terminal_probe, &tty), 0);
    TR_CHECK_EQ_INT(tty.asked, 0);
    n.decide_off_no_color++;

    TR_CHECK_EQ_INT(tr_bar_wanted(NULL, NULL, "dumb", terminal_probe, &tty), 0);
    TR_CHECK_EQ_INT(tty.asked, 0);
    n.decide_off_dumb++;

    TR_CHECK_EQ_INT(tr_bar_wanted(NULL, NULL, NULL, terminal_probe, &pipe), 0);
    TR_CHECK_EQ_INT(pipe.asked, 1); /* off because the terminal said no */
    TR_CHECK_EQ_INT(tr_bar_wanted(NULL, NULL, NULL, NULL, NULL), 0);
    n.decide_off_pipe++;

    TR_CHECK_EQ_INT(tr_bar_wanted(NULL, NULL, NULL, terminal_probe, &tty), 1);
    TR_CHECK_EQ_INT(tr_bar_wanted("1", "", "xterm-256color", terminal_probe, &tty), 1); /* NO_COLOR empty: unset */
    TR_CHECK_EQ_INT(tr_bar_wanted(NULL, NULL, "dumber", terminal_probe, &tty), 1);
    TR_CHECK_EQ_INT(tty.asked, 3);
    n.decide_on++;
}

/* ---- console mode (Windows, but pure and built everywhere: review finding 5) ---- */

static void test_console_mode(void) {
    /* VT off, some other bit set: switch it on, and remember to restore */
    tr_bar_console_plan on = tr_bar_console_mode_plan(0x1);
    TR_CHECK_EQ_INT(on.restore, 1);
    TR_CHECK_EQ_INT((int)on.to_set, 0x1 | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    if (on.restore == 1 && on.to_set == (unsigned long)(0x1 | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        n.console_restore++;

    /* VT already on: nothing to set, nothing to restore */
    tr_bar_console_plan already = tr_bar_console_mode_plan(0x1 | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    TR_CHECK_EQ_INT(already.restore, 0);
    TR_CHECK_EQ_INT((int)already.to_set, 0x1 | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    if (already.restore == 0 && already.to_set == (unsigned long)(0x1 | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        n.console_no_restore++;
}

/* ---- draw, fail, quiet ---- */

static void test_draw(void) {
    fake_clock clk = {100.0, 0.0};
    static capture cap; /* global-ok: large, this test's own */
    memset(&cap, 0, sizeof cap);
    static tr_bar b; /* global-ok: 4 KiB line, this test's own */
    tr_bar_init(&b, 1, fake_now, &clk, capture_write, &cap);
    const double dt = 1.0 / (TR_BAR_FPS * TR_BAR_SUB);

    tr_bar_progress(&b, 1, 1000000); /* too soon: a load of a few ms must not flash a frame */
    TR_CHECK_EQ_INT(b.frames, 0);
    TR_CHECK_EQ_INT(cap.writes, 0);
    clk.now = 100.0 + TR_BAR_DELAY_SEC - 0.001;
    tr_bar_progress(&b, 1, 1000000); /* still short of the delay */
    TR_CHECK_EQ_INT(b.frames, 0);
    TR_CHECK_EQ_INT(cap.writes, 0);
    if (b.frames == 0 && cap.writes == 0) n.draw_delay++;

    clk.now = 100.0 + TR_BAR_DELAY_SEC;
    tr_bar_progress(&b, 1, 1000000); /* the delay has passed: the first report draws at once */
    TR_CHECK_EQ_INT(b.frames, 1);
    TR_CHECK_EQ_INT(cap.writes, 1);
    TR_CHECK(last_is_frame(&cap));
    if (b.frames == 1 && cap.writes == 1 && last_is_frame(&cap)) n.draw_first++;

    /* one second of reports every millisecond: about TR_BAR_FPS frames, not a thousand */
    for (int k = 1; k <= 1000; k++) {
        clk.now = 100.0 + TR_BAR_DELAY_SEC + k * 0.001;
        tr_bar_progress(&b, 1 + (uint64_t)k * 100, 1000000);
    }
    TR_CHECK(b.frames >= TR_BAR_FPS - 1 && b.frames <= TR_BAR_FPS + 1);
    TR_CHECK_EQ_INT(cap.writes, b.frames);
    if (b.frames > 1 && b.frames <= TR_BAR_FPS + 1) n.draw_throttle++;

    /* the physics followed the clock, one substep per 1/(FPS*SUB) s, none skipped */
    int64_t due = (int64_t)((b.last_draw - 100.0) / dt);
    TR_CHECK(b.steps >= due - 1 && b.steps <= due + 1);
    TR_CHECK_EQ_INT(b.steps_run, b.steps);
    TR_CHECK(b.steps > TR_BAR_FPS * TR_BAR_SUB * 9 / 10);
    if (b.steps_run == b.steps && b.steps > 0) n.draw_steps++;

    /* a 10 s stall: the next frame runs at most TR_BAR_MAX_STEPS substeps, the rest is skipped */
    int64_t run_before = b.steps_run, frames_before = b.frames;
    clk.now += 10.0;
    tr_bar_progress(&b, 500000, 1000000);
    TR_CHECK_EQ_INT(b.frames, frames_before + 1);
    TR_CHECK_EQ_INT(b.steps_run - run_before, TR_BAR_MAX_STEPS);
    due = (int64_t)((clk.now - 100.0) / dt);
    TR_CHECK(b.steps >= due - 1 && b.steps <= due + 1);
    if (b.steps_run - run_before == TR_BAR_MAX_STEPS) n.draw_stall++;

    /* the last report clears the line; close adds nothing */
    int writes = cap.writes;
    clk.now += 1.0;
    tr_bar_progress(&b, 1000000, 1000000);
    TR_CHECK_EQ_INT(cap.writes, writes + 1);
    TR_CHECK(last_is_clear(&cap));
    TR_CHECK_EQ_INT(b.clears, 1);
    TR_CHECK_EQ_INT(b.drawn, 0);
    tr_bar_close(&b);
    TR_CHECK_EQ_INT(cap.writes, writes + 1);
    if (last_is_clear(&cap) && cap.writes == writes + 1) n.draw_end++;
}

static void test_fail_and_quiet(void) {
    static capture cap; /* global-ok: large, this test's own */
    static tr_bar b;    /* global-ok: 4 KiB line, this test's own */
    fake_clock clk = {5.0, 0.0};

    /* a load that fails after drawing: no last report, tr_bar_close clears, once */
    memset(&cap, 0, sizeof cap);
    tr_bar_init(&b, 1, fake_now, &clk, capture_write, &cap);
    clk.now += TR_BAR_DELAY_SEC; /* past the draw delay (finding 8): the report below must draw */
    tr_bar_progress(&b, 10, 100);
    TR_CHECK(last_is_frame(&cap));
    tr_bar_close(&b);
    TR_CHECK_EQ_INT(cap.writes, 2);
    TR_CHECK(last_is_clear(&cap));
    TR_CHECK_EQ_INT(b.clears, 1);
    tr_bar_close(&b);               /* again: nothing */
    clk.now += 1.0;
    tr_bar_progress(&b, 20, 100);   /* after close: nothing */
    TR_CHECK_EQ_INT(cap.writes, 2);
    if (cap.writes == 2 && last_is_clear(&cap)) n.fail_close++;

    /* off: nothing at all */
    memset(&cap, 0, sizeof cap);
    tr_bar_init(&b, 0, fake_now, &clk, capture_write, &cap);
    tr_bar_progress(&b, 10, 100);
    tr_bar_progress(&b, 100, 100);
    tr_bar_close(&b);
    TR_CHECK_EQ_INT(cap.writes, 0);
    n.quiet_off++;

    /* on, but the load ends at its first report (done == total): nothing was drawn, nothing to clear */
    memset(&cap, 0, sizeof cap);
    tr_bar_init(&b, 1, fake_now, &clk, capture_write, &cap);
    tr_bar_progress(&b, 100, 100);
    tr_bar_close(&b);
    TR_CHECK_EQ_INT(cap.writes, 0);
    TR_CHECK_EQ_INT(b.clears, 0);
    n.quiet_never_drew++;
}

/* ---- load ---- */

static const synth_params P = {2, 64, 4, 2, 128, 8, 2, 32, 64, TR_TYPE_F32};
#define HUGE_BUDGET ((uint64_t)1 << 40)

static void test_load(const char *argv0) {
    static capture cap; /* global-ok: large, this test's own */
    static tr_bar b;    /* global-ok: 4 KiB line, this test's own */
    char path[512], err[256];
    fake_clock clk = {0.0, 0.05}; /* every reading 50 ms later: every report but the last draws */

    TR_CHECK(synth_write(&P, argv0, "bar_ok.gguf", path, sizeof path) == 0);
    memset(&cap, 0, sizeof cap);
    tr_bar_init(&b, 1, fake_now, &clk, capture_write, &cap);
    tr_model *m = tr_bar_load_with(&b, path, NULL, HUGE_BUDGET, err, sizeof err);
    TR_CHECK(m != NULL);
    TR_CHECK(b.frames > 0);
    TR_CHECK_EQ_INT(b.clears, 1);
    TR_CHECK(last_is_clear(&cap));
    TR_CHECK_EQ_INT(cap.writes, b.frames + 1);
    if (m != NULL && b.frames > 0 && last_is_clear(&cap)) n.load_ok++;
    tr_model_free(m);

    /* off: the same model loads, and nothing is written */
    memset(&cap, 0, sizeof cap);
    tr_bar_init(&b, 0, fake_now, &clk, capture_write, &cap);
    m = tr_bar_load_with(&b, path, NULL, HUGE_BUDGET, err, sizeof err);
    TR_CHECK(m != NULL);
    TR_CHECK_EQ_INT(cap.writes, 0);
    if (m != NULL && cap.writes == 0) n.load_off++;
    tr_model_free(m);
    remove(path);

    /* layer 0's router is missing: the load fails after every expert and a few dense tensors were
     * reported (the bar drew) and before layer 1's were, so the last report (done == total) never
     * comes and only tr_bar_close can clear the line. (Layer 1's router, the last tensor read,
     * would not do: every other tensor of the file is read before it, done reaches the file's
     * total, and the last report clears the line before the load fails.) */
    synth_drop = "blk.0.ffn_gate_inp.weight";
    TR_CHECK(synth_write(&P, argv0, "bar_fail.gguf", path, sizeof path) == 0);
    synth_drop = NULL;
    memset(&cap, 0, sizeof cap);
    tr_bar_init(&b, 1, fake_now, &clk, capture_write, &cap);
    m = tr_bar_load_with(&b, path, NULL, HUGE_BUDGET, err, sizeof err);
    TR_CHECK(m == NULL);
    TR_CHECK(strstr(err, "blk.0.ffn_gate_inp.weight") != NULL);
    TR_CHECK(b.frames > 0);
    TR_CHECK_EQ_INT(b.clears, 1);
    TR_CHECK(last_is_clear(&cap));
    if (m == NULL && b.frames > 0 && last_is_clear(&cap)) n.load_fail++;
    tr_model_free(m);
    remove(path);
}

int main(int argc, char **argv) {
    const char *argv0 = argc > 0 ? argv[0] : "./test_bar";
    test_physics();
    test_render();
    test_layout();
    test_decide();
    test_console_mode();
    test_draw();
    test_fail_and_quiet();
    test_load(argv0);

    TR_CHECK(n.physics_steady > 0);
    TR_CHECK(n.physics_bursty > 0);
    TR_CHECK(n.physics_empty > 0);
    TR_CHECK(n.physics_random > 0);
    TR_CHECK(n.render > 0);
    TR_CHECK(n.render_small > 0);
    TR_CHECK(n.layout > 0);
    TR_CHECK(n.layout_frames > 0);
    TR_CHECK(n.layout_columns > 0);
    TR_CHECK(n.decide_off_bar > 0);
    TR_CHECK(n.decide_off_no_color > 0);
    TR_CHECK(n.decide_off_dumb > 0);
    TR_CHECK(n.decide_off_pipe > 0);
    TR_CHECK(n.decide_on > 0);
    TR_CHECK(n.console_restore > 0);
    TR_CHECK(n.console_no_restore > 0);
    TR_CHECK(n.draw_delay > 0);
    TR_CHECK(n.draw_first > 0);
    TR_CHECK(n.draw_throttle > 0);
    TR_CHECK(n.draw_steps > 0);
    TR_CHECK(n.draw_stall > 0);
    TR_CHECK(n.draw_end > 0);
    TR_CHECK(n.fail_close > 0);
    TR_CHECK(n.quiet_off > 0);
    TR_CHECK(n.quiet_never_drew > 0);
    TR_CHECK(n.load_ok > 0);
    TR_CHECK(n.load_fail > 0);
    TR_CHECK(n.load_off > 0);
    TR_TEST_EXIT();
}
