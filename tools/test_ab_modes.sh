#!/bin/sh
# test_ab_modes.sh — does tools/ab_modes.sh stop on a run that measured nothing, with and without
# AB_WALL=1? (docs/LESSONS.md #56, #132: with AB_WALL=1 every run printed a wall_ms line, so a run
# the engine refused was recorded as a fast one and the comparison went on with rc 0)
#
#   sh tools/test_ab_modes.sh        from the repo root; part of `make check`, about a second
#
# The branches, each on fake modes that print (or do not print) the engine's lines on stderr:
#   failed        a mode that prints nothing: ab_modes stops, rc 1
#   failed+wall   the same with AB_WALL=1: still rc 1, no wall_ms line recorded
#   crashed       a mode that prints its prompt line, then exits 1: ab_modes stops, rc 1
#   measured      modes that print a prompt line: rc 0, a prefill median per mode
#   measured+wall the same with AB_WALL=1: rc 0, a wall_ms median per mode as well
# A counter per branch fails the test if a branch was never reached.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
# a caller's measuring session exports these (tools/measure_guard.lib): not for fake runs
unset AB_WALL AB_GUARD
AB=${AB_MODES:-tools/ab_modes.sh}
T=$(mktemp -d)
finish() { cleanup_children; rm -rf "$T"; }
trap finish EXIT
trap 'exit 130' INT TERM
FAIL=0
N_FAILED=0 N_FAILED_WALL=0 N_CRASHED=0 N_MEASURED=0 N_MEASURED_WALL=0
fail() { echo "test_ab_modes: FAILED: $*"; FAIL=1; }

GOOD="echo 'prompt: 8 tokens in 0.1 s (80.0 tok/s)' >&2"
BAD="true"

# failed: no line, no wall
RC=0
sh "$AB" 1 "a=$GOOD" "b=$BAD" > "$T/failed.txt" 2>&1 || RC=$?
if [ "$RC" = 1 ]; then N_FAILED=$((N_FAILED + 1)); else fail "failed: rc $RC, 1 wanted"; fi

# failed+wall: the wall line alone must not count as a measurement
RC=0
AB_WALL=1 sh "$AB" 1 "a=$GOOD" "b=$BAD" > "$T/failed-wall.txt" 2>&1 || RC=$?
if [ "$RC" = 1 ] && ! grep -q '^b wall_ms' "$T/failed-wall.txt"; then
  N_FAILED_WALL=$((N_FAILED_WALL + 1))
else
  fail "failed+wall: rc $RC (1 wanted), or a wall_ms line of the failed mode recorded"
  cat "$T/failed-wall.txt"
fi

# crashed: its line printed, then a failure
RC=0
sh "$AB" 1 "a=$GOOD" "b=$GOOD; exit 1" > "$T/crashed.txt" 2>&1 || RC=$?
if [ "$RC" = 1 ] && ! grep -q '^b prefill' "$T/crashed.txt"; then
  N_CRASHED=$((N_CRASHED + 1))
else
  fail "crashed: rc $RC (1 wanted), or the crashed run's prefill recorded"
  cat "$T/crashed.txt"
fi

# measured
RC=0
sh "$AB" 1 "a=$GOOD" "b=$GOOD" > "$T/measured.txt" 2>&1 || RC=$?
if [ "$RC" = 0 ] && [ "$(grep -c '^prefill .* median ' "$T/measured.txt")" = 2 ] &&
   ! grep -q 'wall_ms' "$T/measured.txt"; then
  N_MEASURED=$((N_MEASURED + 1))
else
  fail "measured: rc $RC (0 wanted), or not 2 prefill medians, or a wall_ms line without AB_WALL"
  cat "$T/measured.txt"
fi

# measured+wall
RC=0
AB_WALL=1 sh "$AB" 1 "a=$GOOD" "b=$GOOD" > "$T/measured-wall.txt" 2>&1 || RC=$?
if [ "$RC" = 0 ] && [ "$(grep -c '^wall_ms .* median ' "$T/measured-wall.txt")" = 2 ] &&
   [ "$(grep -c '^prefill .* median ' "$T/measured-wall.txt")" = 2 ]; then
  N_MEASURED_WALL=$((N_MEASURED_WALL + 1))
else
  fail "measured+wall: rc $RC (0 wanted), or not 2 wall_ms and 2 prefill medians"
  cat "$T/measured-wall.txt"
fi

for n in N_FAILED N_FAILED_WALL N_CRASHED N_MEASURED N_MEASURED_WALL; do
  eval "v=\$$n"
  [ "$v" -gt 0 ] || fail "branch $n never reached"
done
[ "$FAIL" = 0 ] || exit 1
echo "test_ab_modes: 5 branches, all passed"
}
main "$@"; exit
