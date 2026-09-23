#!/bin/sh
# threads_phase.sh — how many threads each phase wants (docs/MEASUREMENTS.md §Thread per fase): native
# Windows, still machine, rotating order, A/A control (tools/ab_modes.sh, docs/LESSONS.md #66).
# Run from the repo root in Git Bash. Results in build/threads_phase/.
#
#   sh tools/threads_phase.sh sweep [rounds]
#       one -t for both phases: 16, 8, 12, 4 and 16 again (the A/A), context 512 then 2048.
#       About 30 minutes. This is the measurement the width was chosen from.
#   sh tools/threads_phase.sh widths [rounds]
#       decode-512 alone (below): the decode forced on 16, 8, 4, 12, 6 threads and the measured
#       default, prompt on 16. About 10 minutes.
#   sh tools/threads_phase.sh change <binary-before> [rounds]
#       change-512, change-2048: before and after at -t 16, each with its A/A copy, and the new
#       binary with the decode forced on 8; "width" is the decode width each run of the new
#       binary measured for itself. About 25 minutes.
#   sh tools/threads_phase.sh after <binary-before> [rounds]
#       the same, then, about 25 minutes more:
#         decode-512               prompt on 16 threads, decode forced on 16, 8, 4, 12, 6, the
#                                  measured default, and 16 again
#         spec-code-edit, spec-code   --spec 8 where the draft is accepted and where it is not:
#                                  before, after, after with every verify pass narrow
#                                  (TR_DECODE_ROWS=16), before again
#
# The machine's marker (~/.claude/macchina-ferma) is held for the duration and given back at the
# end, also when the script fails or is interrupted: the other windows pause their work, their
# containers stay up (tools/measure_guard.lib). A binary Smart App Control still blocks
# (docs/LESSONS.md #12) is waited for, never rebuilt: a rebuild starts the wait again.
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
WHAT=$1
B=build/trochilus.exe
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
OUT=build/threads_phase
case "$WHAT" in
  sweep|widths) R=${2:-8} ;;
  change|after) BEFORE=$2; R=${3:-8}; [ -f "$BEFORE" ] || { echo "threads_phase: no binary '$BEFORE'"; exit 2; } ;;
  *) echo "usage: threads_phase.sh sweep [rounds] | widths [rounds] | change <binary-before> [rounds] | after <binary-before> [rounds]"; exit 2 ;;
esac
mkdir -p $OUT
[ -f $B ] && [ -f $M ] || { echo "threads_phase: $B or $M is missing"; exit 1; }

# up to an hour, one try a minute (a blocked exe exits 126 from Git Bash)
wait_runs() {
  TRY=0
  until "$1" cpu > /dev/null 2>&1; do
    TRY=$((TRY + 1))
    [ "$TRY" -le 60 ] || { echo "threads_phase: $1 still does not run after an hour"; exit 1; }
    echo "threads_phase: $1 does not run yet (Smart App Control?), try $TRY, waiting 60 s"
    sleep 60
  done
}
# one measurement at a time, and the machine stays awake while it lasts (docs/LESSONS.md #82)
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
# after the lock, before the machine's marker: a binary Smart App Control holds (up to an hour)
# keeps no other window waiting (tools/measure_guard.lib)
measure_ready() {
  wait_runs $B
  [ -z "$BEFORE" ] || wait_runs $BEFORE
}
measure_begin threads_phase

# the other windows' containers stay up, their work paused by the marker; from here on a CPU
# busy with other work, or the marker lost, stops the measurement at the next run
# (tools/measure_guard.lib, tools/machine_still.sh, docs/LESSONS.md #73, #84)
measure_machine threads_phase

# The model takes 7 GiB and the memory guard wants 3 more left free. After a `make check` Windows
# needs minutes to take back what the VM has released: a measurement started at once stops at its
# first run with "not enough memory" (docs/LESSONS.md #72). Up to 15 minutes, one look every 30 s.
TRY=0
while :; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "threads_phase: only ${AVAIL:-?} GiB available after 15 minutes, 12 wanted"; exit 1; }
  echo "threads_phase: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done
measure_still threads_phase
measure_declare "before the first run"

# $1: prompt length, $2: context
gen() { echo "generate -m $M -p $1 -n 48 -c $2 -t 16"; }
# $1: prompt file
spec() { echo "run -m $M -f bench/prompts/$1 -n 200 -t 16 --spec 8"; }

widths() {
  G=$(gen 512 600)
  echo "##### decode width forced, prompt 512"
  cleanup_run sh tools/ab_modes.sh $R "d16=$B $G --decode-threads 16" "d8=$B $G --decode-threads 8" \
     "d4=$B $G --decode-threads 4" "auto=$B $G" "d12=$B $G --decode-threads 12" \
     "d6=$B $G --decode-threads 6" "d16again=$B $G --decode-threads 16" > $OUT/decode-512.txt
  tail -22 $OUT/decode-512.txt
}

if [ "$WHAT" = widths ]; then
  widths
elif [ "$WHAT" = sweep ]; then
  for P in "512 600" "2048 2200"; do
    set -- $P
    G="generate -m $M -p $1 -n 48 -c $2"
    echo "##### sweep, prompt $1"
    cleanup_run sh tools/ab_modes.sh $R "t16=$B $G -t 16" "t8=$B $G -t 8" "t12=$B $G -t 12" "t4=$B $G -t 4" \
       "t16again=$B $G -t 16" > $OUT/sweep-$1.txt
    tail -11 $OUT/sweep-$1.txt
  done
else
  for P in "512 600" "2048 2200"; do
    set -- $P
    G=$(gen $1 $2)
    echo "##### change, prompt $1"
    cleanup_run sh tools/ab_modes.sh $R "before=$BEFORE $G" "after=$B $G" "forced8=$B $G --decode-threads 8" \
       "beforeagain=$BEFORE $G" "afteragain=$B $G" > $OUT/change-$1.txt
    tail -14 $OUT/change-$1.txt
  done
  [ "$WHAT" = after ] || { measure_declare "after the last run"; echo "done: $OUT"; exit 0; }
  widths
  for F in code-edit code; do
    S=$(spec $F.txt)
    echo "##### --spec 8, $F"
    cleanup_run sh tools/ab_modes.sh $R "before=$BEFORE $S" "after=$B $S" "rows16=TR_DECODE_ROWS=16 $B $S" \
       "beforeagain=$BEFORE $S" > $OUT/spec-$F.txt
    tail -11 $OUT/spec-$F.txt
  done
fi
measure_declare "after the last run"
echo "done: $OUT"
}
main "$@"; exit
