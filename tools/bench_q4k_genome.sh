#!/bin/sh
# bench_q4k_genome.sh — the decode's Q4_K dot sequenced in L1 (tests/bench_q4k_genome.c: the cost
# of each instruction, the kernel with one piece removed at a time; docs/MEASUREMENTS.md question
# 66) under the rules of every native measurement (tools/measure_guard.lib): the machine's marker,
# a still machine, the load declared before and after. In the trochilus-dev container by default
# (one core in L1: the same silicon, and no wait for Smart App Control); `NATIVE=1` runs a copy
# of the Windows binary one byte longer (LESSONS #12).
#
#   sh tools/bench_q4k_genome.sh [runs] [args...]   from the repo root, in Git Bash; about a minute
#   (args go to the binary: --only <name part> times "tier" and the kernels whose name has it,
#   without the instruction lines; --smt adds the two-threads-on-one-core lines, native only)
#
# Results in build/bench_q4k_genome/ (bench.txt, load.txt, commit.txt).
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
R=${1:-11}
[ $# -gt 0 ] && shift
ARGS="$*"
OUT=build/bench_q4k_genome
mkdir -p $OUT
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
if [ -n "${NATIVE:-}" ]; then
  make WERROR=1 build/tests/bench_q4k_genome.exe > /dev/null
else
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd):/src" -w /src trochilus-dev:local \
    sh -c 'make -s BUILD=build/linux-gcc CC=gcc WERROR=1 build/linux-gcc/tests/bench_q4k_genome > /dev/null' \
    || { echo "bench_q4k_genome: build failed"; exit 1; }
fi
measure_begin bench_q4k_genome
git rev-parse HEAD > $OUT/commit.txt
measure_machine bench_q4k_genome
measure_still bench_q4k_genome
measure_declare "before" > $OUT/load.txt
cat $OUT/load.txt
if [ -n "${NATIVE:-}" ]; then
  cp build/tests/bench_q4k_genome.exe build/tests/bench_q4k_genome_run.exe
  printf 'x' >> build/tests/bench_q4k_genome_run.exe
  cleanup_run ./build/tests/bench_q4k_genome_run.exe --runs $R $ARGS > $OUT/bench.txt 2>&1 ||
    { echo "bench_q4k_genome: the copy did not run (Smart App Control?)"; tail -3 $OUT/bench.txt; exit 1; }
else
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm --privileged -v "$(pwd):/src" -w /src trochilus-dev:local \
    build/linux-gcc/tests/bench_q4k_genome --runs $R $ARGS > $OUT/bench.txt 2>&1 ||
    { echo "bench_q4k_genome: the run failed"; tail -5 $OUT/bench.txt; exit 1; }
fi
measure_declare "after" >> $OUT/load.txt
tail -1 $OUT/load.txt
cat $OUT/bench.txt
awk '/%$/ { s = $NF; sub("%", "", s); if (s + 0 > 10) { n++; print "  noisy (spread " $NF "): " $0 } }
     END { if (n) print "bench_q4k_genome: " n " lines above 10% spread: remeasure before concluding on them" }' \
    $OUT/bench.txt
echo "bench_q4k_genome: done, $OUT/bench.txt"
}
main "$@"; exit
