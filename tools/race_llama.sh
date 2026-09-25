#!/bin/sh
# race_llama.sh — Trochilus against llama.cpp on the same GGUF (docs/MEASUREMENTS.md §Speed —
# Trochilus vs llama.cpp): OLMoE-1B-7B Q8_0 all in RAM, prompts of 512 and 2048 tokens, 128
# generated, 16 and 8 threads. Both engines in the trochilus-dev container (the llama.cpp build is
# a Linux one, ref/llama.cpp/build-trochilus), one engine at a time, through tools/speed_compare.py:
# the median of RUNS runs after one warm-up, Trochilus's decode width forced to its thread count.
#
#   sh tools/race_llama.sh [runs]        from the repo root, in Git Bash; about 30 minutes
#   RACE_MODEL=models/<f>.gguf RACE_PROMPTS="512 2048 4000" RACE_THREADS=8 RACE_OUT=build/<dir>
#     sh tools/race_llama.sh 3            another model, other contexts or widths (the decode's line
#                                          against the context: MEASUREMENTS §The attention against llama.cpp's)
#
# Order A B B A for each prompt: every engine is measured twice, the gap between its two series is
# its A/A (the noise a difference must beat), and a drift of the machine falls on both engines.
# The rules of every native measurement (tools/measure_guard.lib): the machine's marker, one
# measurement at a time, a still machine before every series. Results in build/race_llama/.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
R=${1:-5}
M=${RACE_MODEL:-models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf}
LB=ref/llama.cpp/build-trochilus/bin/llama-bench
OUT=${RACE_OUT:-build/race_llama}
mkdir -p $OUT
[ -f build/linux-gcc/trochilus ] && [ -f $LB ] || { echo "race_llama: build/linux-gcc/trochilus or $LB is missing"; exit 1; }
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin race_llama
git rev-parse HEAD > $OUT/commit.txt
measure_machine race_llama
measure_still race_llama
measure_declare "before the first series" > $OUT/load.txt
cat $OUT/load.txt

# $1: engine, $2: prompt tokens, $3: name of the series
series() {
  cleanup_run sh -c "$AB_GUARD" || { echo "race_llama: the machine is not still, or the marker is not ours: stopping"; exit 3; }
  echo "race_llama: $3 ($1, prompt $2, $(date +%H:%M))"
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm --privileged -v "$(pwd):/src" \
    -v trochilus-models:/src/models -w /src trochilus-dev:local \
    /opt/venv/bin/python tools/speed_compare.py --model $M --trochilus build/linux-gcc/trochilus \
    --llama-bench $LB --engines $1 --threads ${RACE_THREADS:-16,8} --prompt $2 --gen 128 --runs $R \
    --out $OUT/$3.json > $OUT/$3.txt 2>&1 || { echo "race_llama: $3 failed"; tail -5 $OUT/$3.txt; exit 1; }
}
for P in ${RACE_PROMPTS:-512 2048}; do
  series trochilus $P trochilus-$P-a
  series llama.cpp $P llama-$P-a
  series llama.cpp $P llama-$P-b
  series trochilus $P trochilus-$P-b
done
measure_declare "after the last series" >> $OUT/load.txt
tail -1 $OUT/load.txt
echo "race_llama: done, results in $OUT"
}
main "$@"; exit
