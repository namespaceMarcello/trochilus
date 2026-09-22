#!/bin/sh
# experts_budget.sh — what M1 costs: the engine's speed with 25, 50, 75 and 100% of the experts in
# RAM, the rest read from the disk when a token asks for them (docs/ARCHITECTURE.md §Esecuzione
# Esperti M1, docs/MEASUREMENTS.md §M1). Native Windows, still machine, rotating order, A/A control
# (tools/ab_modes.sh, docs/LESSONS.md #66). Run from the repo root in Git Bash. Results in
# build/experts_budget/.
#
#   sh tools/experts_budget.sh measure [rounds]
#       speed: one ab_modes session over the four budgets, each with its A/A copy, plus the
#       store's own counters (hits, misses, MiB read) per run. About 30 minutes.
#   sh tools/experts_budget.sh misses [rounds]
#       how much a GENERATED token really costs, which is what the simulation of domanda 14
#       predicted (22.4 misses and 143 MiB per token with half the model in RAM). The `experts:`
#       counters cover the prompt too, and the prompt touches almost every expert, so one run
#       cannot tell the two apart: each budget is run twice, generating 8 and 72 tokens, and the
#       difference over the 64 tokens between them is the steady state. About 25 minutes.
#   sh tools/experts_budget.sh long [rounds]
#       the same difference over a LONG generation: 200 and 1000 tokens, so the 800 between them
#       are far from the prompt. The `misses` session measures the 64 tokens right after a 512
#       token prompt, when the LRU still holds what the prompt just read: that is the best case,
#       and the simulation of domanda 14 ran over a generation that drifts. This says which of
#       the two the engine lives in. The counters do not move from run to run (spread 0.0% in
#       every session so far), so two rounds are enough. About 30 minutes.
#       The resident budget is in the session on purpose (domanda 46): if it gains from 200 to
#       1000 tokens too, the fixed cost at the start of the decode is not about the disk.
#   sh tools/experts_budget.sh direct [rounds]
#       the same budget read without the system's cache and through it (TR_EXPERT_DIRECT=0): how
#       much the cache flatters a measurement, and what it costs when the model does not fit.
#
# The decode width is FORCED to 8 everywhere (--decode-threads 8): the estimator that would
# measure it is not validated yet (punto 0 di docs/STATUS.md), and a measurement must not rest on
# it. The containers of the other projects are stopped for the duration and started again at the
# end, also on failure; if somebody starts one meanwhile the session stops at the next run
# instead of going on with a busy machine (AB_GUARD, docs/LESSONS.md #73).
#
# Before any speed: the store must really be reading from the disk. A run whose `experts:` line
# says `buffered` would measure the RAM of the page cache (12-26 GB/s) and not the disk
# (~1.5 GB/s, docs/MEASUREMENTS.md §M1 point 4), so the script refuses to start.
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
WHAT=$1
B=build/trochilus.exe
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
OUT=build/experts_budget
case "$WHAT" in
  measure|misses|direct) R=${2:-6} ;;
  long) R=${2:-2} ;;
  *) echo "usage: experts_budget.sh measure | misses | long | direct [rounds]"; exit 2 ;;
esac
mkdir -p $OUT
[ -f $B ] && [ -f $M ] || { echo "experts_budget: $B or $M is missing"; exit 1; }

# The store's resident size is what the engine prints; the four budgets are fractions of it.
# 1024 units of 6.375 MiB for OLMoE-1B-7B Q8_0: 6528 MiB, and a budget above it is resident.
FULL=6528
B75=$((FULL * 75 / 100))
B50=$((FULL * 50 / 100))
B25=$((FULL * 25 / 100))

# up to an hour, one try a minute (a blocked exe exits 126 from Git Bash)
wait_runs() {
  TRY=0
  while :; do
    RC=0
    "$@" > /dev/null 2>&1 || RC=$?
    [ "$RC" = 126 ] || break
    TRY=$((TRY + 1))
    [ "$TRY" -le 60 ] || { echo "experts_budget: $1 still does not run after an hour"; exit 1; }
    echo "experts_budget: $1 does not run yet (Smart App Control?), try $TRY, waiting 60 s"
    sleep 60
  done
}
# one measurement at a time, and the machine stays awake while it lasts (docs/LESSONS.md #82)
. tools/measure_guard.lib
measure_begin experts_budget
trap measure_end EXIT
trap 'exit 130' INT TERM
wait_runs $B cpu

RUNNING=$(docker ps -q 2>/dev/null || true)
restart() { measure_end; if [ -n "$RUNNING" ]; then docker start $RUNNING > /dev/null 2>&1 || true; echo "containers started again"; fi; }
trap restart EXIT
if [ -n "$RUNNING" ]; then
  # the VM's file cache goes back to Windows first (docs/LESSONS.md #38), then everything stops
  MSYS_NO_PATHCONV=1 docker run --rm --privileged trochilus-dev:local sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" || true
  docker stop $RUNNING > /dev/null
  echo "containers stopped: $(echo $RUNNING | wc -w)"
fi
AB_GUARD=$MEASURE_AB_GUARD
export AB_GUARD
still() { sh -c "$AB_GUARD" || { echo "experts_budget: the machine is not still $1 (a container, or a busy CPU), stopping"; exit 3; }; }

# The model takes 7 GiB and the memory guard wants 3 more left free; Windows needs minutes to
# take back what the VM has released (docs/LESSONS.md #72). Up to 15 minutes, a look every 30 s.
TRY=0
while :; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "experts_budget: only ${AVAIL:-?} GiB available after 15 minutes, 12 wanted"; exit 1; }
  echo "experts_budget: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done
measure_still experts_budget
measure_declare "before the first run"

# $1: budget in MiB (0: no flag, the plan decides), $2: generated tokens, $3: further arguments
# CTX holds the context: it must take the 512 token prompt and every generated token.
CTX=${CTX:-600}
gen() {
  BUDGET=""
  [ "$1" = 0 ] || BUDGET="--expert-budget $1"
  echo "$B generate -m $M -p 512 -n $2 -c $CTX -t 16 --decode-threads 8 $BUDGET $3"
}

# The store must be reading the disk, not the page cache: one short run, and the engine's own
# line decides. A binary without the word (before lot 3) stops the script too.
preflight() {
  LINE=$(sh -c "$(gen $B50 4)" 2>&1 >/dev/null | grep '^experts:' || true)
  echo "preflight: $LINE"
  case "$LINE" in
    *direct*) ;;
    *) echo "experts_budget: the store is not reading direct ('$LINE'): the numbers would be about"
       echo "  the page cache, not the disk. TR_EXPERT_DIRECT=0 in the environment? lot 3 missing?"
       exit 4 ;;
  esac
}

# $1: name of the output, then label/budget/tokens triples already turned into modes
session() {
  NAME=$1
  shift
  echo "##### $NAME"
  still "before $NAME"
  cleanup_run sh tools/ab_modes.sh $R "$@" > $OUT/$NAME.txt
  still "after $NAME"
  sed -n '/^medians/,$p' $OUT/$NAME.txt
}

preflight
if [ "$WHAT" = measure ]; then
  session speed \
    "b100=$(gen 0 64)"   "b75=$(gen $B75 64)" \
    "b50=$(gen $B50 64)" "b25=$(gen $B25 64)" \
    "b100a=$(gen 0 64)"  "b75a=$(gen $B75 64)" \
    "b50a=$(gen $B50 64)" "b25a=$(gen $B25 64)"
elif [ "$WHAT" = misses ]; then
  session steady \
    "b75short=$(gen $B75 8)" "b75long=$(gen $B75 72)" \
    "b50short=$(gen $B50 8)" "b50long=$(gen $B50 72)" \
    "b25short=$(gen $B25 8)" "b25long=$(gen $B25 72)"
  echo
  echo "per generated token, from the difference between 72 and 8 tokens (64 tokens apart):"
  awk -f tools/experts_steady.awk $OUT/steady.txt
elif [ "$WHAT" = long ]; then
  CTX=1600
  session steady-long \
    "b100short=$(gen 0 200)" "b100long=$(gen 0 1000)" \
    "b50short=$(gen $B50 200)" "b50long=$(gen $B50 1000)" \
    "b25short=$(gen $B25 200)" "b25long=$(gen $B25 1000)"
  echo
  echo "per generated token, from the difference between 1000 and 200 tokens (800 apart),"
  echo "that is a token far from the prompt:"
  awk -v short_n=200 -v gap=800 -f tools/experts_steady.awk $OUT/steady-long.txt
else
  session direct \
    "d50=$(gen $B50 64)" "b50=TR_EXPERT_DIRECT=0 $(gen $B50 64)" \
    "d50a=$(gen $B50 64)" "b50a=TR_EXPERT_DIRECT=0 $(gen $B50 64)"
fi
measure_declare "after the last run"
echo "done: $OUT"
}
main "$@"; exit
