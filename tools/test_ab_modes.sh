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
#   zero          a field whose median is 0 (the store's hits): rc 0, its median and every other one printed
#   stop          AB_STOP on steady modes: stops after AB_MIN kept rounds, rc 0
#   nostop        AB_STOP with a mode 40% apart run to run: every round runs
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
N_FAILED=0 N_FAILED_WALL=0 N_CRASHED=0 N_MEASURED=0 N_MEASURED_WALL=0 N_ZERO=0 N_STOP=0 N_NOSTOP=0
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

# zero: a field whose median is 0 (the store's hits on a budget below one token's experts, docs/LESSONS.md #270)
# must print its median and every field after it; it once divided by zero and the summary stopped there
ZA="$GOOD; echo 'experts: 8 of 64 units in RAM (1 MiB, direct), 0 hits, 5 misses, 3 MiB read in 0.5 s' >&2"
ZB="$GOOD; echo 'experts: 8 of 64 units in RAM (1 MiB, direct), 2 hits, 3 misses, 2 MiB read in 0.4 s' >&2"
RC=0
sh "$AB" 1 "a=$ZA" "b=$ZB" > "$T/zero.txt" 2>&1 || RC=$?
if [ "$RC" = 0 ] && [ "$(grep -c ' median ' "$T/zero.txt")" = 10 ] && grep -q '^hits  *a  *median  *0[.]00 ' "$T/zero.txt"; then
  N_ZERO=$((N_ZERO + 1))
else
  fail "zero: rc $RC (0 wanted), or not 10 medians (prefill, hits, misses, mib, read_s of two modes), or no hits median 0"
  cat "$T/zero.txt"
fi

# stop: AB_STOP=5 AB_MIN=2 on modes that print the same speed every time: the comparison stops after round 2
# of 9 (three runs a mode: rounds 0, 1, 2), rc 0, its medians printed
RC=0
AB_STOP=5 AB_MIN=2 sh "$AB" 9 "a=$GOOD" "b=$GOOD" > "$T/stop.txt" 2>&1 || RC=$?
if [ "$RC" = 0 ] && grep -q 'stopped after round 2 of 9' "$T/stop.txt" && [ "$(grep -c '^a prefill ' "$T/stop.txt")" = 3 ]; then
  N_STOP=$((N_STOP + 1))
else
  fail "stop: rc $RC (0 wanted), or not stopped after round 2, or not three runs of a"
  cat "$T/stop.txt"
fi

# nostop: one mode's speed jumps 80 <-> 120 run by run, 40% apart: AB_STOP=5 never holds, all 4 rounds run
echo 0 > "$T/c"
NOISY="n=\$(cat '$T/c'); echo \$((n + 1)) > '$T/c'; if [ \$((n % 2)) = 0 ]; then echo 'prompt: 8 tokens in 0.1 s (80.0 tok/s)' >&2; else echo 'prompt: 8 tokens in 0.1 s (120.0 tok/s)' >&2; fi"
RC=0
AB_STOP=5 AB_MIN=2 sh "$AB" 4 "a=$GOOD" "b=$NOISY" > "$T/nostop.txt" 2>&1 || RC=$?
if [ "$RC" = 0 ] && ! grep -q 'stopped after' "$T/nostop.txt" && [ "$(grep -c '^b prefill ' "$T/nostop.txt")" = 5 ]; then
  N_NOSTOP=$((N_NOSTOP + 1))
else
  fail "nostop: rc $RC (0 wanted), or stopped early on a noisy mode, or not five runs of b"
  cat "$T/nostop.txt"
fi

for n in N_FAILED N_FAILED_WALL N_CRASHED N_MEASURED N_MEASURED_WALL N_ZERO N_STOP N_NOSTOP; do
  eval "v=\$$n"
  [ "$v" -gt 0 ] || fail "branch $n never reached"
done
[ "$FAIL" = 0 ] || exit 1
echo "test_ab_modes: 8 branches, all passed"
}
main "$@"; exit
