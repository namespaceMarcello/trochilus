#!/bin/sh
# mutate_bar.sh — do tests/test_bar.c and test_stream's progress case see a wrong progress bar
# (src/app/bar.c: physics, frame, decision, drawing, clearing) or a wrong load report
# (src/models/olmoe.c, src/models/model.c, src/memory/experts.c: tr_progress)? A test never seen
# red proves nothing (docs/LESSONS.md #43). The tree is copied and built once; each mutation is
# applied to a fresh copy of that build, which then recompiles only the mutated file. Linux
# container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_bar.sh
#
# One line per mutation: the verdict of each test run (a crash counts as RED) and the first checks
# that failed. "no mutation" must be green, every other line RED; a mutation that leaves any test
# green or unbuilt is a miss. Each test run is capped (timeout -k 5 120): a mutation that hangs a
# test counts as a miss, not a stuck script. The last line is a tally, "N of N red"; the script
# exits 1 unless the baseline was green and every mutant was caught. ONLY=<word> runs "no mutation"
# and the mutations with that word in the name.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PYBIN=${PY:-tools/.venv/bin/python}
[ -x "$PYBIN" ] || PYBIN=python3
cat > /tmp/mut.py <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit("mutation does not apply cleanly (%d occurrences): %s" % (s.count(old), old))
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
rm -rf /tmp/mutbase && mkdir -p /tmp/mutbase && cp -r Makefile src tests /tmp/mutbase/
(cd /tmp/mutbase && make -j4 BUILD=b CC=gcc b/tests/test_bar b/tests/test_stream > /tmp/mutbase.log 2>&1) ||
  { echo "the unmutated tree does not build"; tail -20 /tmp/mutbase.log; exit 1; }
BASELINE_OK=0
TOTAL=0
CAUGHT=0
# $1: name, $2: tests to run (bar, or core: test_bar and test_stream), $3: file, $4: text, $5: replacement
run() {
  [ -z "$ONLY" ] || case "$1" in "no mutation"|*"$ONLY"*) ;; *) return 0 ;; esac
  rm -rf /tmp/mut && cp -a /tmp/mutbase /tmp/mut
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$3" "$4" "$5"
  line="$1:"
  tests="test_bar"
  [ "$2" = core ] && tests="test_bar test_stream"
  # baseline ("no mutation"): every test must be green. A mutant is caught the usual mutation-
  # testing way: at least one of its tests goes RED (a "core" mutation in olmoe.c/model.c/experts.c
  # need not show in test_bar's synthetic model to be caught by test_stream's).
  all_green=1
  any_red=0
  for t in $tests; do
    if ! timeout -k 5 120 make BUILD=b CC=gcc b/tests/$t > /tmp/mut.log 2>&1; then
      line="$line $t=does-not-build"
      all_green=0
      any_red=1
      continue
    fi
    if timeout -k 5 120 ./b/tests/$t > /tmp/mut.out 2>&1; then
      line="$line $t=green"
    else
      # every check that failed, by line (a crash: the lines before it, and no "ok")
      line="$line $t=RED [$(grep -o 'test_[a-z]*\.c:[0-9]*: check failed' /tmp/mut.out | sed 's/: check failed//' |
        sort -u -t: -k2n | tr '\n' ' ')]"
      all_green=0
      any_red=1
    fi
  done
  echo "$line"
  cd /src
  if [ "$1" = "no mutation" ]; then
    [ "$all_green" = 1 ] && BASELINE_OK=1
  else
    TOTAL=$((TOTAL + 1))
    [ "$any_red" = 1 ] && CAUGHT=$((CAUGHT + 1))
  fi
}
B=src/app/bar.c
O=src/models/olmoe.c
X=src/memory/experts.c
run "no mutation" core $B "(int)nearbyint(h * 8.0)" "(int)nearbyint(h * 8.0)"

# the frame: preview.py's colours, glyphs and label
run "half an eighth rounds up, not to even" bar $B "(int)nearbyint(h * 8.0)" "(int)(h * 8.0 + 0.5)"
run "the hue drifts the other way" bar $B "- t * 0.12;" "+ t * 0.12;"
run "no shine" bar $B "double shine = 0.88 + 0.12 * sin(i * 0.55 - t * 7.0);" "double shine = 1.0;"
run "the label says GB" bar $B " reading the model  %3.1f/%.1f GiB  %3d%%" \
  " reading the model  %3.1f/%.1f GB  %3d%%"
run "the frame does not go back to the line's start" bar $B "    put(&lb, \"\\r\");
" ""
run "the track's background is never set" bar $B \
  "    put(&lb, \"\\x1b[48;2;%d;%d;%dm\", BAR_TRACK[0], BAR_TRACK[1], BAR_TRACK[2]);
" ""
run "the line may take the last column" bar $B "int avail = width - 1;" "int avail = width;"
run "the width is ignored" bar $B "if (width <= 0) return fit;" "return fit;"
run "the full label needs fewer cells" bar $B "{1, 5, 10, 20};" "{1, 5, 10, 2};"
run "more cells than columns on a wide terminal" bar $B "        if (cells > TR_BAR_W) cells = TR_BAR_W;
" ""
run "a narrow frame shows the first columns" bar $B "int i = (int)((j + 0.5) * TR_BAR_W / fit.cells);" "int i = j;"
run "no cell still draws" bar $B "    if (fit.cells == 0) return 0;
" ""
run "a short buffer gives a cut frame" bar $B "    if (lb.full) {
        out[0] = 0;
        return 0;
    }
" ""

# the physics
run "a negative depth is not clamped" bar $B "    if (!(h > 0.0)) return 0;
" ""
run "the water starts with some in" bar $B "        w->d[i] = 0.0;" "        w->d[i] = 0.5;"
run "the water starts undefined" bar $B "        w->d[i] = 0.0;" "        w->d[i] = NAN;"
run "the outflow limiter divides by zero" bar $B "double k = w->d[i] / (out * dt);" "double k = w->d[i] / (out * dt - out * dt);"
run "no outflow limiter" bar $B "if (out * dt > w->d[i] && out > 0.0) {" "if (0) {"
run "the level at progress 1 is a cell short" bar $B "double level = (BAR_RISE + 1.0) * p" "double level = BAR_RISE * p"
run "the pour starts above empty" bar $B "water_target(w, p) - sum" "water_target(w, p + 0.05) - sum"
run "the flow never damps" bar $B "damp = 0.997" "damp = 1.0"

# the decision
run "TR_BAR=0 ignored" bar $B "    if (env_bar != NULL && strcmp(env_bar, \"0\") == 0) return 0;
" ""
run "NO_COLOR ignored" bar $B "    if (no_color != NULL && no_color[0] != 0) return 0;
" ""
run "NO_COLOR empty counts as set" bar $B "no_color != NULL && no_color[0] != 0" "no_color != NULL"
run "TERM=dumb ignored" bar $B "    if (term != NULL && strcmp(term, \"dumb\") == 0) return 0;
" ""
run "a non-terminal draws" bar $B "return terminal != NULL && terminal(ctx) != 0;" "return terminal != NULL;"
run "no terminal to ask means yes" bar $B "return terminal != NULL && terminal(ctx) != 0;" \
  "return terminal == NULL || terminal(ctx) != 0;"
run "never on" bar $B "return terminal != NULL && terminal(ctx) != 0;" "return 0;"
run "any TERM starting with dumb" bar $B "strcmp(term, \"dumb\") == 0" "strncmp(term, \"dumb\", 4) == 0"
run "the terminal is asked first" bar $B "    if (env_bar != NULL && strcmp(env_bar, \"0\") == 0) return 0;" \
  "    int tty = terminal != NULL && terminal(ctx) != 0;
    if (env_bar != NULL && strcmp(env_bar, \"0\") == 0) return 0;
    (void)tty;"

# drawing and clearing
# ("the first report waits a frame", dropping "b->drawn &&" from the frame-rate check alone, is
# gone: TR_BAR_DELAY_SEC (0.2 s) is now always past 1 / TR_BAR_FPS by the time the first frame is
# eligible to draw, so that guard's exemption for drawn == 0 is never the reason the first frame
# draws -- the mutation went green under review finding 8, not a miss in the harness.)
run "no draw delay: a few ms flash a frame" bar $B \
  "    if (!b->drawn && now - b->start < TR_BAR_DELAY_SEC) return; /* too soon: nothing flashes */
" ""
run "the draw delay never ends" bar $B "now - b->start < TR_BAR_DELAY_SEC" "now - b->start < 1e9"
run "no frame rate limit" bar $B "if (b->drawn && now - b->last_draw < 1.0 / TR_BAR_FPS) return;" "if (b->drawn && 0) return;"
run "one substep a frame, whatever the clock" bar $B "int64_t n = due - b->steps;" "int64_t n = due > b->steps ? 1 : 0;"
run "no cap after a stall" bar $B "    if (n > TR_BAR_MAX_STEPS) n = TR_BAR_MAX_STEPS;
" ""
run "a stall is owed, not skipped" bar $B "b->steps = due;" "b->steps += n;"
run "the physics at half speed" bar $B "(int64_t)((now - b->start) / BAR_DT)" "(int64_t)((now - b->start) / (2 * BAR_DT))"
# ("the first frame runs a substep before any time passed", mutating due's "now <= start" fallback
# from 0 to 1, is gone too: that branch needed now == b->start exactly, which only the very first
# call could ever see, and the draw delay (finding 8) always intercepts that call before this line
# runs -- now > b->start is guaranteed by the time due is computed, so the fallback is unreachable.)
run "the frame after a long stall is dropped" bar $B "if (b->drawn && now - b->last_draw < 1.0 / TR_BAR_FPS) return;" \
  "if (b->drawn && (now - b->last_draw < 1.0 / TR_BAR_FPS || now - b->last_draw > 5.0)) return;"
run "frames not counted" bar $B "    b->frames++;
" ""
run "a clear leaves the bar marked drawn" bar $B "    b->drawn = 0;
    b->clears++;" "    b->clears++;"
run "clears not counted" bar $B "    b->drawn = 0;
    b->clears++;" "    b->drawn = 0;"
run "the last report leaves the bar" bar $B "        bar_clear(b);
        return;
    }
    double now" "        return;
    }
    double now"
run "close does not clear" bar $B "    if (b->on) bar_clear(b);" ""
run "close leaves the bar on" bar $B "    b->on = 0;
#ifdef _WIN32" "#ifdef _WIN32"
run "a clear with nothing drawn" bar $B "    if (!b->drawn) return;
    b->write(b->write_ctx, TR_BAR_CLEAR" "    b->write(b->write_ctx, TR_BAR_CLEAR"
run "a bar that is off draws" bar $B "b->on = on && clock != NULL && write != NULL;" "b->on = clock != NULL && write != NULL;"
run "the load asks for no progress" bar $B "b->on ? &progress : NULL" "NULL"
run "the load does not close the bar" bar $B "    tr_bar_close(b);
    return m;" "    return m;"
run "the load opens no file" bar $B "tr_model_load_progress(path, pool" "tr_model_load_progress(\"\", pool"

# the core's report (model.h tr_progress)
run "read_vec does not report" core $O "    progress_add(lp, t->n_bytes);
    float *out" "    float *out"
run "read_mat does not report" core $O "    progress_add(lp, t->n_bytes);
    if (track(m, raw) != 0) {" "    if (track(m, raw) != 0) {"
run "token_embd is not reported" core $O "        progress_add(&lp, te->n_bytes);
" ""
run "the experts counted at a partial load" core $O "        if (resident)
            for (int64_t i = 0;" "        if (1)
            for (int64_t i = 0;"
run "the experts left out of the total" core $O "lp.total = dense_bytes + experts_at_load;" "lp.total = dense_bytes;"
run "a unit reported with no bytes" core $O "progress_add((load_progress *)ctx, unit_bytes);" "progress_add((load_progress *)ctx, 0);"
run "the total fixed only after the experts" core $O "        lp.total = dense_bytes + experts_at_load;
        /* resident: fill every slot now, so forward_pass's acquire calls are hits from the very
         * first token and the reader is never called again (one path afterwards either way). */
        if (resident && tr_experts_load_all(m->experts, progress_unit, &lp) != 0) {
            snprintf(err, err_len, \"failed reading the experts at load\");
            goto fail;
        }" "        /* resident: fill every slot now */
        if (resident && tr_experts_load_all(m->experts, progress_unit, &lp) != 0) {
            snprintf(err, err_len, \"failed reading the experts at load\");
            goto fail;
        }
        lp.total = dense_bytes + experts_at_load;"
run "no report per unit" core $X "            if (on_unit != NULL) on_unit(ctx, unit_bytes);
" ""
run "a unit reported without its down part" core $X "for (int p = 0; p < TR_EXPERT_PARTS; p++)
            unit_bytes +=" "for (int p = 0; p < 2; p++)
            unit_bytes +="
run "a dense tensor reported with no bytes" core $O "    progress_add(lp, t->n_bytes);
    float *out" "    progress_add(lp, 0);
    float *out"
run "the total grows half way" core $O "lp->to->fn(lp->to->ctx, lp->done, lp->total);" \
  "lp->to->fn(lp->to->ctx, lp->done, lp->total + (lp->done > lp->total / 2));"
run "a missing router is not an error" core $O \
  "if (read_mat(g, m, &lp, name, m->n_embd, m->n_expert, &layer->gate_inp, err, err_len) != 0) goto fail;" \
  "read_mat(g, m, &lp, name, m->n_embd, m->n_expert, &layer->gate_inp, err, err_len);"
run "a missing matrix names no tensor" core $O "snprintf(err, err_len, \"missing tensor '%s'\", name);
        return -1;" "snprintf(err, err_len, \"missing tensor\");
        return -1;"
run "model.c drops the progress" core src/models/model.c "vt->load(path, g, pool, expert_budget, progress, err, err_len)" \
  "vt->load(path, g, pool, expert_budget, NULL, err, err_len)"

echo "$CAUGHT of $TOTAL red"
[ "$BASELINE_OK" = 1 ] || { echo "the unmutated baseline was not green on every test"; exit 1; }
[ "$CAUGHT" = "$TOTAL" ] || exit 1
}
main "$@"; exit
