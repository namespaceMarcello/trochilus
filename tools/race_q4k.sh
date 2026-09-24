#!/bin/sh
# race_q4k.sh — the real OLMoE in Q4_K against the same model in Q8_0 (docs/MEASUREMENTS.md §Q4_K on
# the CPU), Trochilus in the trochilus-dev container, and llama.cpp on the same Q4_K file for
# reference: prompt 512, 128 generated, 16 and 8 threads, through tools/speed_compare.py (median
# of RUNS runs after one warm-up, the decode width forced to the thread count).
#
#   sh tools/race_q4k.sh [runs] [m]    from the repo root, in Git Bash; about 20 minutes
#
# Order Q4_K, Q8_0, Q8_0, Q4_K: two series per file, their gap the A/A. With `m`: Q4_K_M against
# Q4_K in the same order, llama.cpp on the Q4_K_M. The Q4_K and Q4_K_M files come from
# tools/quantize_q4k.sh. The rules of every native measurement (tools/measure_guard.lib).
# Results in build/race_q4k/ (build/race_q4km/ with `m`).
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
R=${1:-5}
M8=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
M4=models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf
LB=ref/llama.cpp/build-trochilus/bin/llama-bench
if [ "${2:-}" = m ]; then A=models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf NA=q4km B=$M4 NB=q4k OUT=build/race_q4km
else A=$M4 NA=q4k B=$M8 NB=q8 OUT=build/race_q4k; fi
mkdir -p $OUT
[ -f build/linux-gcc/trochilus ] && [ -f $LB ] || { echo "race_q4k: build/linux-gcc/trochilus or $LB is missing"; exit 1; }
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin race_q4k
git rev-parse HEAD > $OUT/commit.txt
measure_machine race_q4k
measure_still race_q4k
measure_declare "before the first series" > $OUT/load.txt
cat $OUT/load.txt

# $1: engine, $2: model, $3: name of the series
series() {
  sh -c "$AB_GUARD" || { echo "race_q4k: the machine is not still, or the marker is not ours: stopping"; exit 3; }
  echo "race_q4k: $3 ($1, $2, $(date +%H:%M))"
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm --privileged -v "$(pwd):/src" \
    -v trochilus-models:/src/models -w /src trochilus-dev:local \
    /opt/venv/bin/python tools/speed_compare.py --model $2 --trochilus build/linux-gcc/trochilus \
    --llama-bench $LB --engines $1 --threads 16,8 --prompt 512 --gen 128 --runs $R \
    --out $OUT/$3.json > $OUT/$3.txt 2>&1 || { echo "race_q4k: $3 failed"; tail -5 $OUT/$3.txt; exit 1; }
}
series trochilus $A $NA-a
series trochilus $B $NB-a
series trochilus $B $NB-b
series trochilus $A $NA-b
series llama.cpp $A llama-$NA
measure_declare "after the last series" >> $OUT/load.txt
tail -1 $OUT/load.txt
echo "race_q4k: done, results in $OUT"
}
main "$@"; exit
