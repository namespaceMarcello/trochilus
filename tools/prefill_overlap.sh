#!/bin/sh
# prefill_overlap.sh — how much of the prefill is the engine WAITING for the disk, and what
# overlapping that wait with the compute could give (docs/MEASUREMENTS.md §M1 misurato, domanda 47).
#
# Under a partial budget the prompt reads the whole expert table once: the bytes are necessary
# (past ~40 tokens every expert is used). What is not necessary is waiting for them in line with
# the compute: today the read happens on the calling thread, inside the zone `weight_read`, while
# the pool has nothing to do. In the prefill the next layer's experts are known in full and in
# order -- there is nothing to predict -- so a reader could fetch layer L+1 while layer L runs.
#
# This script measures the two halves and prints the CEILING of that overlap:
#     ceiling = total / max(disk, compute)
# It is an upper bound of a change nobody has written yet, not a measurement of one: with a
# perfect reader and no contention the prefill cannot go below max(disk, compute).
#
#   sh tools/prefill_overlap.sh [rounds]     from the repo root, in Git Bash
#   SET=prefetch sh tools/prefill_overlap.sh [rounds]
#
# SET=prefetch measures the overlap once it exists (src/memory/experts.h tr_experts_prefetch: the
# next layer's units read by an I/O thread while this one computes): at half budget, prompts 512
# and 2048 with it (the default) and without (TR_PREFETCH=0), and the default at 2048 a second
# time as its A/A. The order rotates by round. With reading ahead the zone `weight_read` is the
# time the compute WAITED for the disk, no longer all of the disk's time.
#
# Four modes in one session, order rotating by round, round 0 dropped as warm-up: prompt 512 and
# 2048, each with the whole model resident and with half of it (--expert-budget). The resident
# runs are the control: there `weight_read` must be ~0, and the difference between the two budgets
# must come out as the disk time the zone reports. Results in build/prefill_overlap/.
#
# Same rules as the other native measurements: the machine's marker is held for the duration
# (tools/measure_guard.lib: the other windows pause, their containers stay up), a binary Smart
# App Control blocks is waited for and never rebuilt (TROCHILUS=<binary> to use a second copy,
# docs/LESSONS.md #12, #81), the machine must be still (AB_GUARD, LESSONS #73) and stays awake
# for the duration (LESSONS #82). About 20 minutes.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
R=${1:-5}
B=${TROCHILUS:-build/trochilus.exe}
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
PY=tools/.venv/Scripts/python.exe
[ -f "$PY" ] || PY=tools/.venv/bin/python
OUT=build/prefill_overlap
mkdir -p $OUT
[ -f "$B" ] && [ -f "$M" ] || { echo "prefill_overlap: $B or $M is missing"; exit 1; }

wait_runs() {
  TRY=0
  while :; do
    RC=0
    "$@" > /dev/null 2>&1 || RC=$?
    [ "$RC" = 126 ] || break
    TRY=$((TRY + 1))
    [ "$TRY" -le 60 ] || { echo "prefill_overlap: $1 still does not run after an hour"; exit 1; }
    echo "prefill_overlap: $1 does not run yet (Smart App Control?), try $TRY, waiting 60 s"
    sleep 60
  done
}
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
# after the lock, before the machine's marker: a binary Smart App Control holds (up to an hour)
# keeps no other window waiting (tools/measure_guard.lib)
measure_ready() {
  wait_runs $B cpu
}
measure_begin prefill_overlap
sha256sum $B > $OUT/binaries.sha256 2> /dev/null || true

# the other windows' containers stay up, their work paused by the marker; from here on a CPU
# busy with other work, or the marker lost, stops the measurement at the next run
# (tools/measure_guard.lib, tools/machine_still.sh, docs/LESSONS.md #73, #84)
measure_machine prefill_overlap
still() { sh -c "$AB_GUARD" || { echo "prefill_overlap: the machine is not still $1 (the marker lost, or a busy CPU), stopping"; exit 3; }; }

# the model takes 7 GiB and the memory guard wants 3 more free (docs/LESSONS.md #72)
TRY=0
while :; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "prefill_overlap: only ${AVAIL:-?} GiB available after 15 minutes"; exit 1; }
  echo "prefill_overlap: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done
measure_still prefill_overlap
measure_declare "before the first run"

# Half the expert table, the same budget the other M1 sessions use: 1024 units of 6.375 MiB for
# OLMoE-1B-7B Q8_0 is 6528 MiB (tools/experts_budget.sh).
FULL=6528
B50=$((FULL * 50 / 100))
echo "half of the expert table: $B50 MiB"

# $1: label, $2: prompt, $3: context, $4: budget flag, $5: "off" to run it with TR_PREFETCH=0
run_mode() {
  still "before $1 round $ROUND"
  if [ "$5" = off ]; then export TR_PREFETCH=0; else unset TR_PREFETCH; fi
  # -n 8: the prompt is what this session measures; the decode width is forced so it plays no part
  cleanup_run $B generate -m $M -p $2 -n 8 -c $3 -t 16 --decode-threads 8 $4 \
    --profile-json $OUT/$1-$ROUND.json > /dev/null 2>> $OUT/runs.log
  # the experts: line of the run (units and MiB read), for the report
  echo "$1-$ROUND $(grep '^experts:' $OUT/runs.log | tail -1)" >> $OUT/experts.log
  unset TR_PREFETCH
}

rm -f $OUT/*.json $OUT/runs.log $OUT/experts.log
ROUND=0
while [ $ROUND -le $R ]; do
  echo "round $ROUND of $R"
  if [ "$SET" = prefetch ]; then
    case $((ROUND % 5)) in
      0) ORDER="p512b50 p512b50off p2048b50 p2048b50off p2048b50aa" ;;
      1) ORDER="p2048b50off p2048b50aa p512b50 p512b50off p2048b50" ;;
      2) ORDER="p2048b50aa p512b50off p2048b50 p512b50 p2048b50off" ;;
      3) ORDER="p512b50off p2048b50 p2048b50aa p2048b50off p512b50" ;;
      *) ORDER="p2048b50 p512b50 p2048b50off p2048b50aa p512b50off" ;;
    esac
  else
    case $((ROUND % 5)) in
      0) ORDER="p512b100 p512b50 p2048b100 p2048b50 p2048b50one" ;;
      1) ORDER="p2048b50 p2048b50one p512b100 p512b50 p2048b100" ;;
      2) ORDER="p2048b100 p2048b50 p2048b50one p512b100 p512b50" ;;
      3) ORDER="p512b50 p2048b100 p2048b50 p2048b50one p512b100" ;;
      *) ORDER="p2048b50one p512b100 p512b50 p2048b100 p2048b50" ;;
    esac
  fi
  for MODE in $ORDER; do
    case $MODE in
      p512b100)  run_mode $MODE 512 600 "" ;;
      p512b50)   run_mode $MODE 512 600 "--expert-budget $B50" ;;
      p2048b100) run_mode $MODE 2048 2200 "" ;;
      p2048b50)  run_mode $MODE 2048 2200 "--expert-budget $B50" ;;
      # the same prompt in ONE forward pass instead of four of 512 (OLMOE_DEFAULT_BATCH): a pass
      # walks every layer, so under a partial budget each pass reads the expert table again
      p2048b50one) run_mode $MODE 2048 2200 "--expert-budget $B50 -b 2048" ;;
      # SET=prefetch: without reading ahead, and the default again (A/A)
      p512b50off)  run_mode $MODE 512 600 "--expert-budget $B50" off ;;
      p2048b50off) run_mode $MODE 2048 2200 "--expert-budget $B50" off ;;
      p2048b50aa)  run_mode $MODE 2048 2200 "--expert-budget $B50" ;;
    esac
  done
  ROUND=$((ROUND + 1))
done
still "after the last run"
measure_declare "after the last run"
echo
$PY tools/prefill_overlap_report.py $OUT > $OUT/report.txt
cat $OUT/report.txt
echo "done: $OUT"
}
main "$@"; exit
