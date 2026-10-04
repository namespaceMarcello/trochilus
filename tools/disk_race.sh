#!/bin/sh
# disk_race.sh -- the expert store's reads at the disk's limit, raced (docs/MEASUREMENTS.md §The disk at its limit):
# a layer's consecutive experts one request a part (TR_EXPERT_RUNS) and the slots' pages faulted before the reads
# (TR_EXPERT_TOUCH), each arm a process of its own, rounds rotating (tools/ab_modes.sh, AB_WALL=1: the whole run in
# ms, the prompt's and the decode's tok/s, the store's read time and requests), under the native guards
# (tools/measure_guard.lib). Four arms: new (both on, the default), runs (touch off), old (both off, as before
# 10-02), aa (the default again: the A/A floor). Every run's generated tokens kept and compared: one text for all.
#
#   sh tools/disk_race.sh <binary> <rounds> [scenario ...]
#   scenarios (default all, in this order):
#     res       the whole table resident, Q8_0, code.txt (321 tokens), 8 tokens, 8 threads
#     half321   half the table (--expert-budget 3264), the same prompt: one pass, every read on demand
#     half2048  half the table, a 2048-token prompt (-c 2200): layer-major, the I/O thread's runs
#     weak321   the below-average machine emulated (docs/ARCHITECTURE.md §The roadmap): -t 4, TR_CPU_MAX=avx2,
#               half the table (~4.3 GiB in all with the dense weights and the KV: within 8 GB), 321 tokens
#     weak2048  the same, 2048 tokens
#   DISK_RACE_ARMS=inflight races the requests in flight instead: new (a run's three parts in flight together),
#   serial (TR_EXPERT_INFLIGHT=0: one call a part), aa (new again).
#   DISK_RACE_OUT (default build/disk_race) takes one folder per scenario; a folder that already holds runs is
#   refused. TR_GPU=0 everywhere (the 8 GB machine has none; the disk is what is raced). Round 0 is a warm-up.
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
B=${1:-}
R=${2:-}
[ -n "$B" ] && [ -n "$R" ] || { echo "usage: sh tools/disk_race.sh <binary> <rounds> [res half321 half2048 weak321 weak2048]"; exit 2; }
shift 2
SCEN=${*:-res half321 half2048 weak321 weak2048}
[ -f "$B" ] || { echo "disk_race: $B is missing"; exit 1; }
OUT=${DISK_RACE_OUT:-build/disk_race}
MF=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
[ -f "$MF" ] || { echo "disk_race: $MF is missing"; exit 1; }
for S in $SCEN; do
  case $S in res|half321|half2048|weak321|weak2048) ;; *) echo "disk_race: unknown scenario $S"; exit 2 ;; esac
  if [ -n "$(find "$OUT/$S" -name '*.out' 2> /dev/null | head -1)" ]; then
    echo "disk_race: $OUT/$S already holds runs: move them, or give another DISK_RACE_OUT"
    exit 1
  fi
done
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin disk_race
mkdir -p "$OUT"
sha256sum "$B" > "$OUT/binary.sha256" 2> /dev/null || true
measure_machine disk_race
# the resident table takes 7 GiB and the memory guard wants 3 more free (docs/LESSONS.md #72)
TRY=0
while :; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "disk_race: only ${AVAIL:-?} GiB available after 15 minutes"; exit 1; }
  echo "disk_race: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done
measure_still disk_race
measure_declare "before the first run" > "$OUT/load.txt"
cat "$OUT/load.txt"
IDS=$($B tokenize -m $MF -f bench/prompts/code.txt)
# the whole run in ms, process start and load included: what a user waits for (tools/ab_modes.sh)
export AB_WALL=1
for S in $SCEN; do
  D=$OUT/$S
  mkdir -p "$D"
  case $S in
    res)      RUN="generate -m $MF --tokens $IDS -n 8 -t 8 --decode-threads 8"; PRE="" ;;
    half321)  RUN="generate -m $MF --tokens $IDS -n 8 -t 8 --decode-threads 8 --expert-budget 3264"; PRE="" ;;
    half2048) RUN="generate -m $MF -p 2048 -c 2200 -n 8 -t 8 --decode-threads 8 --expert-budget 3264"; PRE="" ;;
    weak321)  RUN="generate -m $MF --tokens $IDS -n 8 -t 4 --decode-threads 4 --expert-budget 3264"; PRE="TR_CPU_MAX=avx2" ;;
    weak2048) RUN="generate -m $MF -p 2048 -c 2200 -n 8 -t 4 --decode-threads 4 --expert-budget 3264"; PRE="TR_CPU_MAX=avx2" ;;
  esac
  MODES=""
  case ${DISK_RACE_ARMS:-runs} in
    runs)     set -- "new=TR_EXPERT_RUNS=1 TR_EXPERT_TOUCH=1" "runs=TR_EXPERT_RUNS=1 TR_EXPERT_TOUCH=0" \
                     "old=TR_EXPERT_RUNS=0 TR_EXPERT_TOUCH=0" "aa=TR_EXPERT_RUNS=1 TR_EXPERT_TOUCH=1" ;;
    inflight) set -- "new=TR_EXPERT_INFLIGHT=1" "serial=TR_EXPERT_INFLIGHT=0" "aa=TR_EXPERT_INFLIGHT=1" ;;
    *)        echo "disk_race: DISK_RACE_ARMS is runs or inflight"; exit 2 ;;
  esac
  for MODE in "$@"; do
    LABEL=${MODE%%=*}
    VARS=${MODE#*=}
    # stdout (the generated tokens) to a file of its own, stderr (the speed lines) to ab_modes
    MODES="$MODES \"$LABEL=TR_GPU=0 $PRE $VARS $B $RUN > $D/$LABEL-\\\$(date +%s%N).out\""
  done
  echo "== $S: $B $RUN ($PRE)" | cut -c1-300 | tee "$D/what.txt"
  measure_still "disk_race $S"
  eval "cleanup_run sh tools/ab_modes.sh $R $MODES" > "$D/ab_modes.txt"
  grep 'median' "$D/ab_modes.txt" || true
  # one text for every arm and round: the reads change where the bytes land, never which bytes
  FIRST=$(ls "$D"/*.out | head -1)
  SAME=yes
  for F in "$D"/*.out; do cmp -s "$FIRST" "$F" || { SAME=no; echo "disk_race: $F differs from $FIRST"; }; done
  echo "$S: the same tokens in all $(ls "$D"/*.out | wc -l) runs: $SAME" | tee "$D/same.txt"
  [ "$SAME" = yes ] || exit 4
done
measure_declare "after the last run" >> "$OUT/load.txt"
tail -1 "$OUT/load.txt"
echo "done: $OUT"
}
main "$@"; exit
