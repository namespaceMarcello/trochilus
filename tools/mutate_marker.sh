#!/bin/sh
# mutate_marker.sh — does tools/test_marker.sh see a wrong machine marker (tools/measure_guard.lib:
# taking, waiting for and giving back ~/.claude/macchina-ferma)? A test never seen red proves
# nothing (docs/LESSONS.md #43). Each mutation goes into a COPY of the library, which the test loads
# over the real one (MEASURE_GUARD_LIB): the real file is never touched (docs/LESSONS.md #116, #128).
# Every run has a cap (timeout kills its whole process group: a mutant that waits forever is RED by
# timeout), and what a run left behind is counted and killed.
#
#   sh tools/mutate_marker.sh        from the repo root, Git Bash or Linux; about 2 minutes
#
# One line per mutation: "no mutation" must be green, every other line RED.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PYBIN=${PY:-tools/.venv/Scripts/python.exe}
[ -x "$PYBIN" ] || PYBIN=tools/.venv/bin/python
[ -x "$PYBIN" ] || PYBIN=python3
T=$(mktemp -d)
TAG="mutate-marker-$$"
cat > "$T/mut.py" <<'EOF'
import sys
src, dst, old, new = sys.argv[1:5]
s = open(src, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit("mutation does not apply once: " + old)
open(dst, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
EOF
RED=0
TOTAL=0
# $1: name, $2: text, $3: replacement
run() {
  "$PYBIN" "$T/mut.py" tools/measure_guard.lib "$T/lib" "$2" "$3" || { echo "$1: does not apply"; exit 1; }
  RC=0
  MEASURE_GUARD_LIB="$T/lib" timeout -k 5 60 sh tools/test_marker.sh "$TAG" > "$T/out" 2>&1 || RC=$?
  LEFT=$(ps -ef 2> /dev/null | grep -v grep | grep -c "test_marker.sh $TAG" || true)
  for P in $(ps -ef 2> /dev/null | grep -v grep | grep "test_marker.sh $TAG" | awk '{print $2}'); do kill -9 "$P" 2> /dev/null || true; done
  if [ "$RC" = 0 ]; then V=green; else V=RED; fi
  [ "$RC" != 124 ] && [ "$RC" != 137 ] || V="RED (timeout)"
  echo "$1: $V${LEFT:+$([ "$LEFT" = 0 ] || echo ", $LEFT left behind, killed")} $(grep -m1 FAILED "$T/out" || true)"
  if [ "$1" = "no mutation" ]; then
    [ "$V" = green ] || { echo "mutate_marker: the unmutated library fails its test"; cat "$T/out"; exit 1; }
  else
    TOTAL=$((TOTAL + 1))
    [ "$V" = green ] || RED=$((RED + 1))
  fi
}
run "no mutation" '(set -C; echo' '(set -C; echo'
run "a leftover of ours is never taken" '{ ! marker_fresh || marker_leftover; }' '{ ! marker_fresh; }'
run "a live measurement of another checkout taken as a leftover" \
  '[ -n "$MARK_PID" ] && ! kill -0 "$MARK_PID" 2> /dev/null' '[ -n "$MARK_PID" ]'
run "a fresh marker taken as stale" '-mmin -$MACHINE_MARKER_MAX_MIN' '-mmin +$MACHINE_MARKER_MAX_MIN'
run "the marker written over another's" '(set -C; echo' '(echo'
run "the wait has no cap" 'if [ $WAITED -ge $MACHINE_MARKER_WAIT_MAX ]; then' 'if false; then'
run "measure_end removes a line that is not ours" '!= "$MEASURE_MARK_LINE" ]' '= "$MEASURE_MARK_LINE never" ]'
run "measure_end removes a marker it never took" '[ -z "$MEASURE_MARK_MINE" ] ||' 'false ||'
run "the guard does not refresh the marker" ' && touch -c "$MACHINE_MARKER"' ''
run "the guard passes on another's marker" \
  '[ "$(head -1 "$MACHINE_MARKER" 2> /dev/null)" = "$MEASURE_MARK_LINE" ] && touch' 'touch'
rm -rf "$T"
echo "== mutate_marker: $RED of $TOTAL mutations RED"
[ "$RED" = "$TOTAL" ]
}
main "$@"; exit
