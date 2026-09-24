#!/bin/sh
# test_beside.sh -- tools/beside.sh: both commands' outcomes reach its exit code, the first
# command's output is shown, and a failing second command leaves no first one running.
#
#   sh tools/test_beside.sh        from the repo root; part of `make check`, about 3 s
#
# Four cases: both succeed (exit 0, the first's output shown); the first fails (its exit code and
# its output); the second fails while the first still runs (its exit code at once, and the
# first's process gone); and the same with beside.sh's trap removed, where the first's process
# must STILL be running: without that half the third would prove nothing (seen red at every run,
# as tools/test_cleanup.sh does; the leftover is then removed). The first command writes its pid
# to a file, and `kill -0` asks for it: MSYS's ps shows no arguments to tell a sleep by.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
T=build/test_beside
rm -rf $T
mkdir -p $T
fail() { echo "test_beside: $1"; exit 1; }

OUT=$(sh tools/beside.sh $T/a.log 'echo first-said-this' true)
RC=$?
[ $RC -eq 0 ] || fail "both succeeded, exit $RC"
echo "$OUT" | grep -q first-said-this || fail "the first command's output was not shown"

OUT=$(sh tools/beside.sh $T/b.log 'echo first-failed-here; exit 3' true)
RC=$?
[ $RC -eq 3 ] || fail "the first failed with 3, beside exited $RC"
echo "$OUT" | grep -q first-failed-here || fail "the failing first command's output was not shown"

# $1: beside script, $2: log; the first sleeps 30 s, the second fails after 1 s
second_fails() {
  sh $1 $2 "echo \$\$ > $T/pid; exec sleep 30" sh -c 'sleep 1; exit 4' > /dev/null 2>&1
}
START=$(date +%s)
second_fails tools/beside.sh $T/c.log
RC=$?
[ $RC -eq 4 ] || fail "the second failed with 4, beside exited $RC"
[ $(( $(date +%s) - START )) -lt 20 ] || fail "beside waited for the first command after the second failed"
[ -s $T/pid ] || fail "the first command never started"
sleep 1
if kill -0 "$(cat $T/pid)" 2> /dev/null; then kill "$(cat $T/pid)"; fail "the first command outlived beside"; fi

grep -v "trap cleanup_children EXIT" tools/beside.sh > $T/beside_notrap.sh
rm -f $T/pid
second_fails $T/beside_notrap.sh $T/d.log
[ -s $T/pid ] || fail "the first command never started (no trap)"
kill -0 "$(cat $T/pid)" 2> /dev/null || fail "without the trap the first command died too: the check above proves nothing"
kill "$(cat $T/pid)" 2> /dev/null || true
echo "== test_beside: exit codes and output of both commands, and no first command left when the second fails"
}
main "$@"; exit
