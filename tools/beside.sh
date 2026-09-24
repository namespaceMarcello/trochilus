#!/bin/sh
# beside.sh -- one command beside another: the first in the background, its output held in a file
# and shown when it ends, the second in the foreground; exits non-zero if either failed.
#
# The gate runs the C tests natively beside the container's gate instead of after it
# (docs/MEASUREMENTS.md §The gate): the two share nothing but the processors, and the native side
# waited ~35 s of every gate in a row. Whatever it started dies with it (tools/cleanup.lib): the
# second command failing, or a signal, stops the first. tools/test_beside.sh proves the three
# outcomes and that no child is left.
#
#   sh tools/beside.sh <log> '<first command>' <second command...>
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
[ $# -ge 3 ] || { echo "usage: sh tools/beside.sh <log> '<first command>' <second command...>"; exit 2; }
LOG=$1
FIRST=$2
shift 2
sh -c "$FIRST" > "$LOG" 2>&1 &
BESIDE=$!
cleanup_run "$@"
RC=$?
if [ $RC -ne 0 ]; then
    echo "beside: the second command failed (exit $RC); stopping the first ($FIRST)"
    exit $RC
fi
wait $BESIDE
RC1=$?
cat "$LOG"
[ $RC1 -eq 0 ] || { echo "beside: $FIRST failed (exit $RC1)"; exit $RC1; }
}
main "$@"; exit
