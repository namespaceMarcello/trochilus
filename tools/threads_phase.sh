#!/bin/sh
# threads_phase.sh — how many threads each phase wants (docs/MISURE.md §Thread per fase): native
# Windows, still machine, rotating order, A/A control (tools/ab_modes.sh, docs/LEZIONI.md #66).
# Run from the repo root in Git Bash. Results in build/threads_phase/.
#
#   sh tools/threads_phase.sh sweep [rounds]
#       one -t for both phases: 16, 8, 12, 4 and 16 again (the A/A), context 512 then 2048.
#       About 30 minutes. This is the measurement the width was chosen from.
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
# The containers of the other projects are stopped for the duration and started again at the end,
# also when the script fails or is interrupted. A binary Smart App Control still blocks
# (docs/LEZIONI.md #12) is waited for, never rebuilt: a rebuild starts the wait again.
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LEZIONI.md #69).
main() {
WHAT=$1
B=build/trochilus.exe
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
OUT=build/threads_phase
case "$WHAT" in
  sweep) R=${2:-8} ;;
  change|after) BEFORE=$2; R=${3:-8}; [ -f "$BEFORE" ] || { echo "threads_phase: no binary '$BEFORE'"; exit 2; } ;;
  *) echo "usage: threads_phase.sh sweep [rounds] | change <binary-before> [rounds] | after <binary-before> [rounds]"; exit 2 ;;
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
wait_runs $B
[ -z "$BEFORE" ] || wait_runs $BEFORE

RUNNING=$(docker ps -q 2>/dev/null || true)
restart() { if [ -n "$RUNNING" ]; then docker start $RUNNING > /dev/null 2>&1 || true; echo "containers started again"; fi; }
trap restart EXIT INT TERM
if [ -n "$RUNNING" ]; then
  # the VM's file cache goes back to Windows first (docs/LEZIONI.md #38), then everything stops
  MSYS_NO_PATHCONV=1 docker run --rm --privileged trochilus-dev:local sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" || true
  docker stop $RUNNING > /dev/null
  echo "containers stopped: $(echo $RUNNING | wc -w)"
fi
# from here on a running container means somebody else is using the machine: ab_modes.sh stops
# at the next run instead of going on (docs/LEZIONI.md #73)
AB_GUARD='[ -z "$(docker ps -q 2>/dev/null)" ]'
export AB_GUARD

# The model takes 7 GiB and the memory guard wants 3 more left free. After a `make check` Windows
# needs minutes to take back what the VM has released: a measurement started at once stops at its
# first run with "not enough memory" (docs/LEZIONI.md #72). Up to 15 minutes, one look every 30 s.
TRY=0
while :; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "threads_phase: only ${AVAIL:-?} GiB available after 15 minutes, 12 wanted"; exit 1; }
  echo "threads_phase: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done

# $1: prompt length, $2: context
gen() { echo "generate -m $M -p $1 -n 48 -c $2 -t 16"; }
# $1: prompt file
spec() { echo "run -m $M -f bench/prompts/$1 -n 200 -t 16 --spec 8"; }

if [ "$WHAT" = sweep ]; then
  for P in "512 600" "2048 2200"; do
    set -- $P
    G="generate -m $M -p $1 -n 48 -c $2"
    echo "##### sweep, prompt $1"
    sh tools/ab_modes.sh $R "t16=$B $G -t 16" "t8=$B $G -t 8" "t12=$B $G -t 12" "t4=$B $G -t 4" \
       "t16again=$B $G -t 16" > $OUT/sweep-$1.txt
    tail -11 $OUT/sweep-$1.txt
  done
else
  for P in "512 600" "2048 2200"; do
    set -- $P
    G=$(gen $1 $2)
    echo "##### change, prompt $1"
    sh tools/ab_modes.sh $R "before=$BEFORE $G" "after=$B $G" "forced8=$B $G --decode-threads 8" \
       "beforeagain=$BEFORE $G" "afteragain=$B $G" > $OUT/change-$1.txt
    tail -14 $OUT/change-$1.txt
  done
  [ "$WHAT" = after ] || { echo "done: $OUT"; exit 0; }
  G=$(gen 512 600)
  echo "##### decode width forced, prompt 512"
  sh tools/ab_modes.sh $R "d16=$B $G --decode-threads 16" "d8=$B $G --decode-threads 8" \
     "d4=$B $G --decode-threads 4" "auto=$B $G" "d12=$B $G --decode-threads 12" \
     "d6=$B $G --decode-threads 6" "d16again=$B $G --decode-threads 16" > $OUT/decode-512.txt
  tail -22 $OUT/decode-512.txt
  for F in code-edit code; do
    S=$(spec $F.txt)
    echo "##### --spec 8, $F"
    sh tools/ab_modes.sh $R "before=$BEFORE $S" "after=$B $S" "rows16=TR_DECODE_ROWS=16 $B $S" \
       "beforeagain=$BEFORE $S" > $OUT/spec-$F.txt
    tail -11 $OUT/spec-$F.txt
  done
fi
echo "done: $OUT"
}
main "$@"; exit
