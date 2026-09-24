#!/bin/sh
# test_marker.sh — does a measurement take the machine's marker, wait for another window's, and
# give back only its own? (tools/measure_guard.lib; the rule of 2026-09-23: whoever measures
# times writes ~/.claude/macchina-ferma, the other windows pause their work while it is there)
#
#   sh tools/test_marker.sh        from the repo root; part of `make check`, about 1 second
#
# The branches of measure_mark, measure_end and the AB_GUARD of measure_machine, each on a marker
# in a directory of its own (MACHINE_MARKER), never the real one:
#   free        no marker: taken at once, with our line; measure_end removes it
#   foreign     another window's fresh marker: measure_mark waits, and takes it once it is gone
#   gives up    another window's marker that stays: exit 5 after the cap (MACHINE_MARKER_WAIT_MAX)
#   live ours   a marker of this project whose pid runs (another checkout): waited for, not taken
#   stale       another window's marker older than 6 hours: taken at once
#   leftover    a marker of ours whose script is gone (its pid no longer runs): taken at once
#   guard       AB_GUARD passes on our marker and refreshes it, fails once the marker is not ours
#   not ours    measure_end leaves a marker whose line is no longer ours
#   refused     measure_end of a script that never took the marker leaves the one that is there
# A counter per branch fails the test if a branch was never reached.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/measure_guard.lib
# tools/mutate_marker.sh: a mutated copy of the library, loaded over the real one (the real file is
# never the one mutated, docs/LESSONS.md #116, #128)
[ -z "$MEASURE_GUARD_LIB" ] || . "$MEASURE_GUARD_LIB"
T=$(mktemp -d)
finish() { measure_end; rm -rf "$T"; }
trap finish EXIT
trap 'exit 130' INT TERM
# measure_machine asks Docker for the running containers: none, here
docker() { return 0; }
# a fifth of a second between tries (sleep takes fractions on GNU, BSD and Git Bash): at a poll of
# 1 s the waits below were the poll's, 10 s of the gate
MACHINE_MARKER_POLL=0.2
# a marker that should be taken at once and is not ends the test here (exit 5), not in 6 hours
MACHINE_MARKER_WAIT_MAX=8
FAILED=0
REACHED=0
fail() { echo "test_marker: FAILED: $*"; FAILED=1; }
reached() { REACHED=$((REACHED + 1)); }
# older than 6 hours, in a form BSD touch takes too
OLD=200001010000

# free
MACHINE_MARKER=$T/free/macchina-ferma
measure_mark free > /dev/null
[ "$(head -1 "$MACHINE_MARKER")" = "$MEASURE_MARK_LINE" ] || fail "free: the marker does not carry our line"
case $MEASURE_MARK_LINE in "trochilus: free (pid $$, since "*) reached ;; *) fail "free: line '$MEASURE_MARK_LINE'" ;; esac
measure_end
[ ! -e "$MACHINE_MARKER" ] || fail "free: measure_end left our marker"
MEASURE_MARK_MINE=""

# foreign: waits while it is there, takes it after
MACHINE_MARKER=$T/foreign/macchina-ferma
mkdir -p "$T/foreign"
echo "ds4: a bench" > "$MACHINE_MARKER"
( measure_mark foreign > "$T/foreign.log" ) &
W=$!
# waited for by its announcement, printed before its first sleep, not by a fixed sleep
N=0
while ! grep -q "another window measures times" "$T/foreign.log" 2> /dev/null && [ $N -lt 50 ]; do
  sleep 0.2; N=$((N + 1))
done
[ "$(head -1 "$MACHINE_MARKER")" = "ds4: a bench" ] || fail "foreign: another window's fresh marker was taken"
grep -q "another window measures times (ds4: a bench)" "$T/foreign.log" || fail "foreign: the wait was not announced"
kill -0 $W 2> /dev/null && reached || fail "foreign: measure_mark did not wait"
rm -f "$MACHINE_MARKER"
wait $W || fail "foreign: measure_mark failed once the marker was gone"
head -1 "$MACHINE_MARKER" | grep -q '^trochilus: foreign ' || fail "foreign: not taken once free"

# gives up: another window's marker that stays is waited for up to the cap, then exit 5 (a cap
# of 0: the loop that waits is foreign's, above; here the branch that stops)
echo "ds4: a long bench" > "$MACHINE_MARKER"
RC=0
( MACHINE_MARKER_WAIT_MAX=0; measure_mark capped > "$T/capped.log" ) || RC=$?
[ $RC = 5 ] && reached || fail "gives up: exit $RC after the cap, 5 expected"
grep -q "still measures times after 0 min (ds4: a long bench): not starting" "$T/capped.log" ||
  fail "gives up: '$(tail -1 "$T/capped.log")'"
[ "$(head -1 "$MACHINE_MARKER")" = "ds4: a long bench" ] || fail "gives up: the other window's marker was touched"

# live ours: a measurement of another checkout, whose pid (this shell's) runs
LIVE="trochilus: decode_context (pid $$, since 2026-09-23 03:00)"
echo "$LIVE" > "$MACHINE_MARKER"
RC=0
( MACHINE_MARKER_WAIT_MAX=0; measure_mark other > /dev/null ) || RC=$?
[ $RC = 5 ] && [ "$(head -1 "$MACHINE_MARKER")" = "$LIVE" ] && reached ||
  fail "live ours: a running measurement's marker was taken (exit $RC)"
rm -f "$MACHINE_MARKER"

# stale: another window's, older than 6 hours
MACHINE_MARKER=$T/stale/macchina-ferma
mkdir -p "$T/stale"
echo "colibri: a night" > "$MACHINE_MARKER"
touch -t $OLD "$MACHINE_MARKER"
measure_mark stale > /dev/null
[ "$(head -1 "$MACHINE_MARKER")" = "$MEASURE_MARK_LINE" ] && reached || fail "stale: a 7-hour-old marker was not taken"
rm -f "$MACHINE_MARKER"

# leftover: ours, fresh, from a script that died (the lock is ours by now): a pid that ran and ended
sh -c 'exit 0' &
DEAD=$!
wait $DEAD
MACHINE_MARKER=$T/leftover/macchina-ferma
mkdir -p "$T/leftover"
echo "trochilus: decode_context (pid $DEAD, since 2026-09-23 03:00)" > "$MACHINE_MARKER"
measure_mark leftover > /dev/null
[ "$(head -1 "$MACHINE_MARKER")" = "$MEASURE_MARK_LINE" ] && reached || fail "leftover: our dead script's marker was not taken"

# guard: passes and refreshes on our marker, fails on another's; the CPU check is not the one
# under test here, so it is replaced by `true`
touch -t $OLD "$MACHINE_MARKER"
measure_machine leftover > /dev/null
GUARD=$(echo "$AB_GUARD" | sed 's|sh tools/machine_still.sh 3.5 600 2|true|')
[ "$GUARD" != "$AB_GUARD" ] || fail "guard: the CPU check is not where this test expects it"
sh -c "$GUARD" && reached || fail "guard: fails on our own marker"
[ -n "$(find "$MACHINE_MARKER" -mmin -1)" ] || fail "guard: did not refresh our marker"
echo "ds4: a bench" > "$MACHINE_MARKER"
sh -c "$GUARD" && fail "guard: passes on another window's marker" || reached

# not ours: measure_end leaves another window's line alone
measure_end
[ "$(head -1 "$MACHINE_MARKER" 2> /dev/null)" = "ds4: a bench" ] && reached || fail "not ours: measure_end removed another window's marker"

# refused: a script stopped before it took the marker (by the lock, say) has no line and no flag;
# an empty marker would equal its empty line
MEASURE_MARK_MINE=""
MEASURE_MARK_LINE=""
: > "$MACHINE_MARKER"
measure_end
[ -e "$MACHINE_MARKER" ] && reached || fail "refused: measure_end removed a marker it never took"

[ $REACHED -eq 10 ] || fail "$REACHED of 10 branches reached"
[ $FAILED = 0 ] || exit 1
echo "== test_marker: the marker is taken, waited for, and given back only by its writer (10 branches)"
}
main "$@"; exit
