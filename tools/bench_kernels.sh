#!/bin/sh
# bench_kernels.sh — the kernels' microbenchmark (tests/bench_kernels.c: every tier, dot_row and x4
# of every weight type, median and spread) under the rules of every native measurement
# (tools/measure_guard.lib): the machine's marker, a still machine, the load declared before and
# after. Native, from a copy of the binary one byte longer (Smart App Control refuses a freshly
# built one: LESSONS #12); if the copy is refused too, `CONTAINER=1` runs it in trochilus-dev.
#
#   sh tools/bench_kernels.sh            from the repo root, in Git Bash; about 3 minutes
#
# Results in build/bench_kernels/ (bench.txt, load.txt, commit.txt).
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
OUT=build/bench_kernels
mkdir -p $OUT
make WERROR=1 build/tests/bench_kernels.exe > /dev/null
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin bench_kernels
git rev-parse HEAD > $OUT/commit.txt
measure_machine bench_kernels
measure_still bench_kernels
measure_declare "before" > $OUT/load.txt
cat $OUT/load.txt
if [ -n "${CONTAINER:-}" ]; then
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local \
    sh -c 'make BUILD=build/linux-gcc CC=gcc build/linux-gcc/tests/bench_kernels > /dev/null && build/linux-gcc/tests/bench_kernels' \
    > $OUT/bench.txt 2>&1
else
  cp build/tests/bench_kernels.exe build/tests/bench_kernels_run.exe
  printf 'x' >> build/tests/bench_kernels_run.exe
  cleanup_run ./build/tests/bench_kernels_run.exe > $OUT/bench.txt 2>&1 ||
    { echo "bench_kernels: the copy did not run (Smart App Control?): CONTAINER=1"; tail -3 $OUT/bench.txt; exit 1; }
fi
measure_declare "after" >> $OUT/load.txt
tail -1 $OUT/load.txt
# a line whose spread is above 10% is not a number to conclude on: named here, not left to the eye
# (docs/LESSONS.md #150: a ratio of 0.74-0.80 drawn from lines at 19-56%, 0.84-0.86 remeasured)
awk '/%$/ { s = $NF; sub("%", "", s); if (s + 0 > 10) { n++; print "  noisy (spread " $NF "): " $0 } }
     END { if (n) print "bench_kernels: " n " lines above 10% spread: remeasure before concluding on them" }' \
    $OUT/bench.txt > $OUT/noisy.txt
cat $OUT/noisy.txt
echo "bench_kernels: done, $OUT/bench.txt"
}
main "$@"; exit
