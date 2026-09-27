#!/bin/sh
# row_price.sh -- what one more verified row of a pass costs: the cost side of every speculative pass
# (docs/MEASUREMENTS.md §The post-it taken apart). Native, still machine, the attention on the CPU for every
# mode (TR_GPU=0: the GPU takes only one-row passes, which would price the rows against another pass), the
# modes alternated run by run, round 0 dropped. code-edit.txt with the draft fixed at 1, 2, 4, 8: the prompt
# lookup always proposes there, so a mode's passes have 2, 3, 5, 9 rows (rows a pass = the profile's tokens
# / passes). Q8_0 and Q4_K at 8 threads, Q8_0 at 16; ~25 minutes, most of it waiting for a still machine.
# Then tools/row_price_report.py build/rowprice (the tables, and build/rowprice/prices.json for
# tools/draft_gate_sim.py).
#
#   sh tools/row_price.sh [binary] [rounds] [binary-before]
#       binary: build/trochilus.exe (a copy with a byte appended when Smart App Control holds it,
#       docs/LESSONS.md #12). With binary-before the two alternate run by run (before and after a change,
#       the same machine minute by minute), the one that runs first swapped every round (an even number of
#       rounds balances it; docs/LESSONS.md #232): the runs go to <out>/after and <out>/before, and
#       tools/row_price_report.py <out>/after <rounds> <out>/before prints them side by side.
#       ROW_PRICE_AA=1 runs binary-before a second time beside them (<out>/before2): the A/A.
#       ROW_PRICE_EXTRA="<binary> ..." runs more variants in the same rotation, the k-th in <out>/x<k>
#       (tools/row_price_report.py <out>/x<k> <rounds> <out>/before): a series of variants against one before.
#       ROW_PRICE_FORCED=1 runs every pass on the config's threads (--decode-threads): a kernel's A/B that
#       the widths the sessions measure (the decode's, the verify passes' own) must not move.
#   ROW_PRICE_OUT (default build/rowprice), ROW_PRICE_CFGS (default "Q8_0 8,Q4_K 8,Q8_0 16": model and
#   threads), ROW_PRICE_MODES (default "0,1,2,4,8": the fixed drafts) narrow a run. A folder that already
#   holds runs is refused: a narrowed run would be reported beside an older one's files.
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
B=${1:-build/trochilus.exe}
R=${2:-3}
BB=${3:-}
TR_GPU=0
export TR_GPU
OUT=${ROW_PRICE_OUT:-build/rowprice}
CFGS=${ROW_PRICE_CFGS:-"Q8_0 8,Q4_K 8,Q8_0 16"}
MODES=${ROW_PRICE_MODES:-"0,1,2,4,8"}
mkdir -p $OUT
[ -f "$B" ] || { echo "row_price: $B is missing"; exit 1; }
[ -z "$BB" ] || [ -f "$BB" ] || { echo "row_price: $BB is missing"; exit 1; }
[ -z "${ROW_PRICE_EXTRA:-}" ] || [ -n "$BB" ] || { echo "row_price: ROW_PRICE_EXTRA needs a binary-before"; exit 1; }
for xb in ${ROW_PRICE_EXTRA:-}; do [ -f "$xb" ] || { echo "row_price: $xb is missing"; exit 1; }; done
if [ -n "$(find $OUT -name '*.json' 2> /dev/null | head -1)" ]; then
  echo "row_price: $OUT already holds runs: move them, or give another ROW_PRICE_OUT"
  exit 1
fi
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin rowprice
measure_machine rowprice
measure_still rowprice
measure_declare "before the first run"
# one binary: its runs in OUT; two: OUT/after and OUT/before, alternated
if [ -n "$BB" ]; then
  mkdir -p $OUT/after $OUT/before
  echo "$B" > $OUT/after/binary.txt
  echo "$BB" > $OUT/before/binary.txt
  PAIRS="$B=$OUT/after $BB=$OUT/before"
  if [ "${ROW_PRICE_AA:-0}" = 1 ]; then
    mkdir -p $OUT/before2
    echo "$BB (again: the A/A)" > $OUT/before2/binary.txt
    PAIRS="$PAIRS $BB=$OUT/before2"
  fi
  k=1
  for xb in ${ROW_PRICE_EXTRA:-}; do
    mkdir -p $OUT/x$k
    echo "$xb" > $OUT/x$k/binary.txt
    PAIRS="$PAIRS $xb=$OUT/x$k"
    k=$((k + 1))
  done
else
  echo "$B" > $OUT/binary.txt
  PAIRS="$B=$OUT"
fi
OLDIFS=$IFS
IFS=,
for cfg in $CFGS; do
  IFS=$OLDIFS
  set -- $cfg
  M=$1
  T=$2
  MF=models/OLMoE-1B-7B-0125-Instruct-$M.gguf
  IDS=$($B tokenize -m $MF -f bench/prompts/code-edit.txt)
  round=0
  while [ $round -le "$R" ]; do
    IFS=,
    for k in $MODES; do
      IFS=$OLDIFS
      if [ "$k" = 0 ]; then mode="0"; else mode="$k --spec-fixed"; fi
      tag=$(echo "$mode" | tr -d ' -')
      # the first to run swapped every round: an odd round runs the list backwards
      ORDER=$PAIRS
      if [ $((round % 2)) = 1 ]; then
        ORDER=""
        for pair in $PAIRS; do ORDER="$pair $ORDER"; done
      fi
      for pair in $ORDER; do
        BIN=${pair%%=*}
        DIR=${pair#*=}
        eval "$AB_GUARD" || { echo "row_price: machine busy or marker lost"; exit 3; }
        FORCE=""
        [ "${ROW_PRICE_FORCED:-0}" != 1 ] || FORCE="--decode-threads $T"
        cleanup_run $BIN generate -m $MF --tokens "$IDS" -n 200 -t $T --spec $mode $FORCE \
          --profile-json $DIR/$M-t$T-$tag-$round.json > $DIR/$M-t$T-$tag-$round.out 2>&1
      done
    done
    IFS=$OLDIFS
    echo "row_price: $M t$T round $round done"
    round=$((round + 1))
  done
  IFS=,
done
IFS=$OLDIFS
measure_declare "after the last run"
if [ -n "$BB" ]; then
  echo "done: $OUT (tools/.venv/Scripts/python.exe tools/row_price_report.py $OUT/after $R $OUT/before)"
else
  echo "done: $OUT (tools/.venv/Scripts/python.exe tools/row_price_report.py $OUT $R)"
fi
}
main "$@"; exit
