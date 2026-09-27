#!/bin/sh
# mutate_tune.sh — does tests/test_phase.c see a wrong choice in threads per phase (the tuner
# rewritten for docs/LESSONS.md #88: pairwise margin, extension, milestone and kept-pass
# re-measurement, debounce)? A test never seen red proves nothing (docs/LESSONS.md #43): each
# mutation is applied to a copy of the tree, the copy is built and test_phase is run. Linux
# container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_tune.sh
#
# One line per mutation: which tests went red. "no mutation" must be green, every other line must
# be RED. About 30 minutes; ONLY=<word> runs "no mutation" and the mutations with that word in
# the name (-e ONLY=history: 10 minutes).
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PYBIN=${PY:-tools/.venv/bin/python}
cat > /tmp/mut.py <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit("mutation does not apply exactly once: " + old)
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
# $1: name, $2: file, $3: text, $4: replacement
run() {
  [ -z "$ONLY" ] || case "$1" in "no mutation"|*"$ONLY"*) ;; *) return 0 ;; esac
  # tools without .venv: the repo's own Python environment is 706 MiB over a Windows bind mount,
  # and the container has its own, so copying it made one mutation take minutes
  rm -rf /tmp/mut && mkdir -p /tmp/mut/tools && cp -r Makefile src tests bench /tmp/mut/ &&
    find tools -maxdepth 1 -mindepth 1 ! -name .venv ! -name __pycache__ -exec cp -r {} /tmp/mut/tools/ ';'
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  # ASan: a mutation that writes past the session's arrays must turn red, not pass by luck
  make BUILD=b CC=gcc EXTRA_CFLAGS="-fsanitize=address -fno-omit-frame-pointer" EXTRA_LDFLAGS=-fsanitize=address \
    b/tests/test_phase > /dev/null 2>&1 || { echo "$1: does not build"; cd /src; return; }
  if ./b/tests/test_phase > /dev/null 2>&1; then echo "$1: test_phase=green"; else echo "$1: test_phase=RED"; fi
  cd /src
}
M=src/models/model.c
run "no mutation" $M "if (tmp[i] >= 4) width[n++] = tmp[i];" "if (tmp[i] >= 4) width[n++] = tmp[i];"
run "widths: the floor of 4 threads is gone" $M "if (tmp[i] >= 4) width[n++] = tmp[i];" "if (tmp[i] >= 1) width[n++] = tmp[i];"
run "stats: spread is (max - min) / min, a range, not the gap to the second fastest" $M \
  "        if (sec[i] < best) { second = best; best = sec[i]; }
        else if (sec[i] < second) second = sec[i];" \
  "        if (sec[i] < best) best = sec[i];
        if (sec[i] > second) second = sec[i];"
run "pick: a global max spread, not each width's own paired with the best's" $M \
  "double margin = spread[i] > spread[best] ? spread[i] : spread[best];" \
  "double margin = spread[0]; for (int j = 1; j < n; j++) if (spread[j] > margin) margin = spread[j];"
run "pick: ignores the spread (margin 0)" $M \
  "double margin = spread[i] > spread[best] ? spread[i] : spread[best];" \
  "double margin = 0.0; (void)spread;"
run "debounce: returns the raw pick outright, no second vote needed" $M \
  "    *pending = raw;
    return current;
}" \
  "    *pending = raw;
    return raw;
}"
run "session: extension never entered, the first round-robin always stands" $M \
  "if (need_more && t->rounds < TR_DECODE_TUNE_ROUNDS_MAX) {" \
  "if (0 && need_more && t->rounds < TR_DECODE_TUNE_ROUNDS_MAX) {"
run "session: the context-class re-arm is gone" $M \
  "(fm > u->last_milestone || (u->pending != 0 && u->n_kept >= TR_DECODE_TUNE_REMEASURE_KEPT))" \
  "((u->pending != 0 && u->n_kept >= TR_DECODE_TUNE_REMEASURE_KEPT))"
run "session: the pending re-arm after TR_DECODE_TUNE_REMEASURE_KEPT is gone" $M \
  "(fm > u->last_milestone || (u->pending != 0 && u->n_kept >= TR_DECODE_TUNE_REMEASURE_KEPT))" \
  "(fm > u->last_milestone)"
run "history: the width in effect is the pick, a held-back pick is not seen" $M \
  "s->history[slot].width = t->chosen;" "s->history[slot].width = s->width[candidate];"
run "history: past the array the newest takes the first slot, not the last" $M \
  "s->n_history : TR_DECODE_TUNE_HISTORY - 1;" "s->n_history : 0;"
run "rows: every size of short pass takes the decode's measurement" $M \
  "n_logits == n ? &s->tune[n - 1]" "n_logits == n ? &s->tune[0]"
run "rows: a verify size's measurement enters the decode's history" $M \
  "if (t != &s->tune[0]) return;" ""
run "rows: a milestone re-arms the decode's measurement only" $M \
  "for (int i = 0; i < TR_DECODE_ROWS; i++) {
                tune_state *u" "for (int i = 0; i < 1; i++) {
                tune_state *u"
run "rows: a pass of several rows runs on the decode's width while its own size measures" $M \
  "else if (t != NULL && t->measuring) {" "else if (t != NULL && t->measuring && n == 1) {"
run "rows: a milestone re-arms every size but the largest" $M \
  "for (int i = 0; i < TR_DECODE_ROWS; i++) {
                tune_state *u" "for (int i = 0; i < TR_DECODE_ROWS - 1; i++) {
                tune_state *u"
run "rows: a verify size's kept passes are not counted" $M \
  "} else if (t != NULL && threads == t->chosen && s->n_widths - t->lo > 1) {" \
  "} else if (t == &s->tune[0] && threads == t->chosen && s->n_widths - t->lo > 1) {"
run "rows: a verify size probes the narrowest width too" $M \
  "t->lo = i > 0 && s->n_widths > 1 ? 1 : 0;" "t->lo = 0;"
run "rows: a verify size skips the widest width, not the narrowest" $M \
  "probe_w = t->extending ? t->pair[t->next_in_pair] : t->lo + t->n_probes % (s->n_widths - t->lo);" \
  "probe_w = t->extending ? t->pair[t->next_in_pair] : t->n_probes % (s->n_widths - t->lo);"
run "rows: a verify size's pick read from the widths it did not probe" $M \
  "t->chosen = tr_decode_tune_debounce(t->chosen, &t->pending, s->width[lo + candidate]);" \
  "t->chosen = tr_decode_tune_debounce(t->chosen, &t->pending, s->width[candidate]);"
run "rows: a pass past the sizes measured takes the whole pool, not the decode's width" $M \
  "else if (s->tune[0].chosen > 0) threads = s->tune[0].chosen;" ""
run "rows: the sizes' bound dropped (an environment's TR_DECODE_ROWS=9 writes past tune[])" $M \
  "n <= m->decode_rows && n <= TR_DECODE_ROWS && n_logits == n" "n <= m->decode_rows && n_logits == n"
run "rows: a short prompt (one logit row) probes its size too" $M \
  "n <= m->decode_rows && n <= TR_DECODE_ROWS && n_logits == n" "n <= m->decode_rows && n <= TR_DECODE_ROWS"
run "rows: what a size reports ignores the forced width" $M \
  "    if (s->model->decode_threads > 0) return s->model->decode_threads < size ? s->model->decode_threads : size;
    return s->tune[n - 1].chosen;" "    (void)size;
    return s->tune[n - 1].chosen;"
}
main "$@"; exit
