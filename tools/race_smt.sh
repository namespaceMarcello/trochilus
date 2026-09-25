#!/bin/sh
# race_smt.sh — Trochilus and llama.cpp confined to the same logical processors (taskset, in the
# trochilus-dev container), at two thread counts, on the real OLMoE in Q4_K: prompt 512, 128
# generated, through tools/speed_compare.py (median of RUNS runs after one warm-up, the decode width
# forced to the thread count; Trochilus pins every thread to one logical processor, TR_POOL_PIN=1).
# The container's "siblings" are not a core's two threads: the VM's processors land wherever
# Windows puts them (the prefill at 8 threads on "4 cores" runs 1.8x, which SMT never gives:
# docs/LESSONS.md #184), so this races 4 against 8 virtual processors. SMT is measured natively
# (docs/MEASUREMENTS.md §SMT in the decode).
#
#   sh tools/race_smt.sh [runs]        from the repo root, in Git Bash; about 10 minutes
#   SMT_CPUS=0-7 SMT_THREADS=4,8       the defaults: the container's first 4 cores (lscpu -e) and
#                                      their siblings, 4 and 8 threads
#   SMT_ENV="TR_X=1"                   extra environment for Trochilus (A/B of an engine setting)
#
# Order Trochilus, llama.cpp, llama.cpp, Trochilus: two series per engine, their gap the A/A. The
# rules of every native measurement (tools/measure_guard.lib). Results in build/race_smt/.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
R=${1:-5}
M4=models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf
LB=ref/llama.cpp/build-trochilus/bin/llama-bench
CPUS=${SMT_CPUS:-0-7}
THREADS=${SMT_THREADS:-4,8}
OUT=build/race_smt
mkdir -p $OUT
[ -f build/linux-gcc/trochilus ] && [ -f $LB ] || { echo "race_smt: build/linux-gcc/trochilus or $LB is missing"; exit 1; }
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin race_smt
git rev-parse HEAD > $OUT/commit.txt
echo "cpus $CPUS threads $THREADS env ${SMT_ENV:-none}" > $OUT/setup.txt
measure_machine race_smt
measure_still race_smt
measure_declare "before the first series" > $OUT/load.txt
cat $OUT/load.txt

# $1: engine, $2: name of the series
series() {
  cleanup_run sh -c "$AB_GUARD" || { echo "race_smt: the machine is not still, or the marker is not ours: stopping"; exit 3; }
  echo "race_smt: $2 ($1, cpus $CPUS, threads $THREADS, $(date +%H:%M))"
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm --privileged -v "$(pwd):/src" \
    -v trochilus-models:/src/models -w /src trochilus-dev:local \
    env TR_POOL_PIN=1 ${SMT_ENV:-} taskset -c $CPUS /opt/venv/bin/python tools/speed_compare.py --model $M4 \
    --trochilus build/linux-gcc/trochilus --llama-bench $LB --engines $1 --threads $THREADS \
    --prompt 512 --gen 128 --runs $R --out $OUT/$2.json > $OUT/$2.txt 2>&1 \
    || { echo "race_smt: $2 failed"; tail -5 $OUT/$2.txt; exit 1; }
}
series trochilus trochilus-a
series llama.cpp llama-a
series llama.cpp llama-b
series trochilus trochilus-b
measure_declare "after the last series" >> $OUT/load.txt
tail -1 $OUT/load.txt
echo "race_smt: done, results in $OUT"
}
main "$@"; exit
