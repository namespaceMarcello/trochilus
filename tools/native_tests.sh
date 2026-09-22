#!/bin/sh
# native_tests.sh -- the C tests on this machine's own OS, not only in the Linux container.
#
# `make check` on Windows runs the tests in Docker, so that Smart App Control cannot make the gate
# flaky (docs/LESSONS.md #12); the Windows branches of src/base (files, threads, processor
# topology) then never ran in the gate. tr_file_pread on a direct handle failed at the end of the
# file on NTFS from the day it was written, with a test for exactly that case green in Linux
# (docs/LESSONS.md #103). Each binary given here runs natively: a failure fails the gate, a binary Smart App
# Control blocks is SKIPPED with a line that says so. Links are reproducible
# (-Wl,--no-insert-timestamp, test_base linked with src/base only), so a binary whose code did not
# change keeps the hash Smart App Control already cleared.
#
#   sh tools/native_tests.sh <test binary>...
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
passed=0
skipped=0
for t in "$@"; do
    out=$("$t" 2>&1)
    rc=$?
    if [ $rc -eq 126 ]; then
        echo "== native $t: SKIPPED, Smart App Control blocked the new exe (docs/LESSONS.md #12)"
        skipped=$((skipped + 1))
    elif [ $rc -ne 0 ]; then
        echo "$out"
        echo "native $t: FAILED (exit $rc)"
        exit 1
    else
        passed=$((passed + 1))
    fi
done
echo "== native tests: $passed passed, $skipped skipped by Smart App Control"
}
main "$@"; exit
