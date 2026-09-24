#!/bin/sh
# bench_native.sh — one premise bench of build/tests (bench_peak, bench_kvpack, ...) under the rules
# of every native measurement (tools/measure_guard.lib): the machine's marker, a still machine, the
# load declared before and after. Native, from a copy of the binary one byte longer (Smart App
# Control refuses a freshly built one: LESSONS #12).
#
#   sh tools/bench_native.sh <bench> [args...]    e.g. bench_peak, or bench_kvpack time --run all
#
# Results in build/bench_native/<bench>/ (bench.txt, load.txt, commit.txt); lines above 10% spread
# named in noisy.txt.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
[ -n "${1:-}" ] || { echo "usage: sh tools/bench_native.sh <bench> [args...]"; exit 2; }
B=$1
shift
OUT=build/bench_native/$B
mkdir -p $OUT
make WERROR=1 build/tests/$B.exe > /dev/null
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin $B
git rev-parse HEAD > $OUT/commit.txt
measure_machine $B
measure_still $B
measure_declare "before" > $OUT/load.txt
cat $OUT/load.txt
cp build/tests/$B.exe build/tests/${B}_run.exe
printf 'x' >> build/tests/${B}_run.exe
cleanup_run ./build/tests/${B}_run.exe "$@" > $OUT/bench.txt 2>&1 ||
  { echo "$B: the copy did not run (Smart App Control?)"; tail -3 $OUT/bench.txt; exit 1; }
measure_declare "after" >> $OUT/load.txt
tail -1 $OUT/load.txt
awk '/%$/ { s = $NF; sub("%", "", s); if (s + 0 > 10) { n++; print "  noisy (spread " $NF "): " $0 } }
     END { if (n) print "'$B': " n " lines above 10% spread: remeasure before concluding on them" }' \
    $OUT/bench.txt > $OUT/noisy.txt
cat $OUT/noisy.txt
echo "$B: done, $OUT/bench.txt"
}
main "$@"; exit
