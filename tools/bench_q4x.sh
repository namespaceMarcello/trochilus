#!/bin/sh
# bench_q4x.sh — Q4_K's integer kernels timed the way a short verify pass runs them (tests/bench_q4x.c: two
# rows against T prepared rows by q4x_dot2 a token at a time, by q4x_dot_xt, by the W16 panel and a tile; in L1
# and from RAM) under the rules of every native measurement (tools/measure_guard.lib): the machine's marker, a
# still machine, the load declared before and after. In the trochilus-dev container by default (one core in
# L1: the same silicon, and no wait for Smart App Control); `NATIVE=1` runs a copy of the Windows binary one
# byte longer (docs/LESSONS.md #12), which the RAM lines want (the container's reads are slower).
#
#   sh tools/bench_q4x.sh [runs] [args...]   from the repo root, in Git Bash; about a minute
#   (args go to the binary: --ram P [--mib M] times the kernels from RAM on P threads)
#
# Results in build/bench_q4x/ (bench.txt, load.txt, commit.txt); BENCH_Q4X_OUT names another folder.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
R=${1:-11}
[ $# -gt 0 ] && shift
ARGS="$*"
OUT=${BENCH_Q4X_OUT:-build/bench_q4x}
mkdir -p $OUT
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
if [ -n "${NATIVE:-}" ]; then
  make WERROR=1 build/tests/bench_q4x.exe > /dev/null
else
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd):/src" -w /src trochilus-dev:local \
    sh -c 'make -s BUILD=build/linux-gcc CC=gcc WERROR=1 build/linux-gcc/tests/bench_q4x > /dev/null' \
    || { echo "bench_q4x: build failed"; exit 1; }
fi
measure_begin bench_q4x
git rev-parse HEAD > $OUT/commit.txt
measure_machine bench_q4x
measure_still bench_q4x
measure_declare "before" > $OUT/load.txt
cat $OUT/load.txt
if [ -n "${NATIVE:-}" ]; then
  cp build/tests/bench_q4x.exe build/tests/bench_q4x_run.exe
  printf 'x' >> build/tests/bench_q4x_run.exe
  cleanup_run ./build/tests/bench_q4x_run.exe --runs $R $ARGS > $OUT/bench.txt 2>&1 ||
    { echo "bench_q4x: the copy did not run (Smart App Control?)"; tail -3 $OUT/bench.txt; exit 1; }
else
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm --privileged -v "$(pwd):/src" -w /src trochilus-dev:local \
    build/linux-gcc/tests/bench_q4x --runs $R $ARGS > $OUT/bench.txt 2>&1 ||
    { echo "bench_q4x: the run failed"; tail -5 $OUT/bench.txt; exit 1; }
fi
measure_declare "after" >> $OUT/load.txt
tail -1 $OUT/load.txt
cat $OUT/bench.txt
awk '/%/ { for (i = 1; i <= NF; i++) if ($i ~ /%$/) { s = $i; sub("%", "", s); if (s + 0 > 10) { n++; print "  noisy (spread " $i "): " $0 } } }
     END { if (n) print "bench_q4x: " n " lines above 10% spread: remeasure before concluding on them" }' $OUT/bench.txt
echo "bench_q4x: done, $OUT/bench.txt"
}
main "$@"; exit
