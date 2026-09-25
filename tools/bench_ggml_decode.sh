#!/bin/sh
# bench_ggml_decode.sh — the decode's matmul, ggml's kernels against trochilus's in one binary
# (tools/bench_ggml_decode.c; docs/MEASUREMENTS.md question 65). Builds in the trochilus-dev
# container against ref/llama.cpp/build-trochilus's static ggml (commit b49650a) and trochilus's
# core built by make for linux-gcc, then runs one weight type per container run, under 60 s each.
#
#   sh tools/bench_ggml_decode.sh [rounds] [types]    from the repo root, in Git Bash; ~3 minutes
#                                                      types default: q8_0 q4_k q4_k-repack
#
# The rules of every native measurement (tools/measure_guard.lib): the machine's marker, one
# measurement at a time, a still machine before every run. Results in build/bench_ggml_decode/.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
R=${1:-5}
TYPES=${2:-q8_0 q4_k q4_k-repack}
B=ref/llama.cpp/build-trochilus
OUT=build/bench_ggml_decode
mkdir -p $OUT
[ -f $B/ggml/src/libggml-cpu.a ] || { echo "bench_ggml_decode: $B's ggml is missing"; exit 1; }
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM

echo "bench_ggml_decode: building"
cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd):/src" -w /src trochilus-dev:local sh -c '
  set -e
  L=build/linux-gcc
  make -s BUILD=$L CC=gcc WERROR=1 $L/tests/bench_mem > /dev/null
  mkdir -p $L/tools '$B'/bin
  gcc -O2 -std=c11 -Wall -Wextra -Wshadow -Wno-unused-parameter -Werror -Iref/llama.cpp/ggml/include \
      -c tools/bench_ggml_decode.c -o $L/tools/bench_ggml_decode.o 2>&1 || exit 1
  CORE=$(find $L/src -name "*.o" ! -path "*/app/*" | sort)
  g++ $L/tools/bench_ggml_decode.o $CORE -Wl,--start-group $(find '$B'/ggml/src -name "libggml*.a") \
      -Wl,--end-group -lpthread -lm -ldl -o '$B'/bin/bench_ggml_decode' > $OUT/build.txt 2>&1 \
  || { echo "bench_ggml_decode: build failed"; tail -20 $OUT/build.txt; exit 1; }

measure_begin bench_ggml_decode
git rev-parse HEAD > $OUT/commit.txt
measure_machine bench_ggml_decode
measure_still bench_ggml_decode
measure_declare "before the first run" > $OUT/load.txt
cat $OUT/load.txt
for T in $TYPES; do
  cleanup_run sh -c "$AB_GUARD" || { echo "bench_ggml_decode: the machine is not still, or the marker is not ours: stopping"; exit 3; }
  echo "bench_ggml_decode: $T ($(date +%H:%M))"
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm --privileged -v "$(pwd):/src" -w /src trochilus-dev:local \
    $B/bin/bench_ggml_decode $T --rounds $R > $OUT/$T.txt 2>&1 || { echo "bench_ggml_decode: $T failed"; tail -5 $OUT/$T.txt; exit 1; }
  cat $OUT/$T.txt
done
measure_declare "after the last run" >> $OUT/load.txt
tail -1 $OUT/load.txt
echo "bench_ggml_decode: done, results in $OUT"
}
main "$@"; exit
