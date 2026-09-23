#!/bin/sh
# serve_first_prompt.sh — question 49 (docs/MEASUREMENTS.md): the first prompt of a session when a
# `trochilus serve` kept the model and its expert store from the session before, against a new
# process that starts from an empty store. Native, still machine, rotating order, A/A control.
#
#   sh tools/serve_first_prompt.sh [rounds]        from the repo root, in Git Bash
#
# Two ab_modes sessions, each with a server of its own (a server keeps one model):
#   half   cold = a new process (TR_SERVER=0); warm = the same command through a server kept at the
#          same budget (half the expert table), each run finding the store the run before left;
#          coldagain = the A/A control. Prompt 2048, 8 tokens generated.
#   full   the whole table resident: the prefill cannot change there, and what the server takes
#          away is the load (wall_ms, AB_WALL=1 of tools/ab_modes.sh). Two resident copies do not
#          fit in this machine's RAM beside the server, so this session is not interleaved: cold
#          against coldagain with no server, then warm against warmagain through it, each pair its
#          own A/A. The load it measures is seconds, far above any spread.
# Written before measuring (docs/MEASUREMENTS.md question 49): at half budget the kept store should
# help little on a 2048-token prompt. The prompt sweeps the layers in order (layer-major prefill),
# LRU keeps the last layers it swept, and the next sweep starts from layer 0 and evicts them before
# it reaches them: misses warm ~ misses cold. At full budget wall_ms drops by the load.
#
# The rules of every native measurement (tools/measure_guard.lib): the machine's marker, one
# measurement at a time, a still machine before every run, a blocked binary waited for, never
# rebuilt (TROCHILUS=<binary> for a second copy, docs/LESSONS.md #12, #81). About 20 minutes.
# Results in build/serve_first_prompt/.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
R=${1:-5}
B=${TROCHILUS:-build/trochilus.exe}
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
OUT=build/serve_first_prompt
mkdir -p $OUT
[ -f "$B" ] && [ -f "$M" ] || { echo "serve_first_prompt: $B or $M is missing"; exit 1; }

wait_runs() {
  TRY=0
  while :; do
    RC=0
    "$@" > /dev/null 2>&1 || RC=$?
    [ "$RC" = 126 ] || break
    TRY=$((TRY + 1))
    [ "$TRY" -le 60 ] || { echo "serve_first_prompt: $1 still does not run after an hour"; exit 1; }
    echo "serve_first_prompt: $1 does not run yet (Smart App Control?), try $TRY, waiting 60 s"
    cleanup_run sleep 60
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
measure_begin serve_first_prompt
sha256sum $B > $OUT/binaries.sha256 2> /dev/null || true

# the other windows' containers stay up, their work paused by the marker; from here on a CPU
# busy with other work, or the marker lost, stops the measurement at the next run
# (tools/measure_guard.lib, tools/machine_still.sh, docs/LESSONS.md #73, #84)
measure_machine serve_first_prompt

# $1: GiB the engine must see available, waited for up to 15 minutes (docs/LESSONS.md #72)
wait_memory() {
  TRY=0
  while :; do
    AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
    [ "${AVAIL:-0}" -lt "$1" ] || break
    TRY=$((TRY + 1))
    [ "$TRY" -le 30 ] || { echo "serve_first_prompt: only ${AVAIL:-?} GiB available after 15 minutes, $1 wanted"; exit 1; }
    echo "serve_first_prompt: ${AVAIL:-?} GiB available, $1 wanted: waiting 30 s"
    cleanup_run sleep 30
  done
}
wait_memory 12
measure_still serve_first_prompt

# 1024 units of 6.375 MiB for OLMoE-1B-7B Q8_0 is 6528 MiB (tools/experts_budget.sh); a budget
# above the table is clamped to the table, the whole store resident
FULL=6528
HALF=$((FULL * 50 / 100))
G="generate -m $M -p 2048 -n 8 -c 2200 -t 16 --decode-threads 8"

# the store must read the disk, not the page cache the server and the run before warmed: one short
# run, and the engine's own line decides (the same check as tools/experts_budget.sh)
LINE=$(TR_SERVER=0 $B generate -m $M -p 64 -n 1 -t 16 --expert-budget $HALF 2>&1 > /dev/null | grep '^experts:' || true)
case "$LINE" in
  *direct*) ;;
  *) echo "serve_first_prompt: the store is not reading direct ('$LINE'): TR_EXPERT_DIRECT=0 in the environment?"
     exit 4 ;;
esac
measure_declare "before the first run"

# $1: label, $2: budget in MiB: a server keeping the model at that budget, up when this returns
server_up() {
  NAME=q49-$1-$$
  TR_SERVER=1 TR_SERVER_NAME=$NAME $B serve -m $M -t 16 --expert-budget $2 --idle 0 2> $OUT/server-$1.log &
  SERVER=$!
  UP=0
  for _ in $(seq 1 240); do
    if TR_SERVER=1 TR_SERVER_NAME=$NAME timeout 60 $B serve --status > /dev/null 2>&1; then UP=1; break; fi
    # a server that already left (its load refused, a bad argument) will not come up
    kill -0 $SERVER 2> /dev/null || break
    cleanup_run sleep 1
  done
  [ $UP = 1 ] || { echo "serve_first_prompt: the $1 server did not come up"; cat $OUT/server-$1.log; exit 1; }
}
# $1: label, $2: the warm runs it must have served. Its status goes to a file of its own: the
# server's log is open without O_APPEND, and its last lines overwrote a status appended to it.
# A client the server does not answer, or declines, runs the command itself with the same output,
# so warm would silently be cold: the server's own counters decide (one load, every warm run).
server_down() {
  ST=$(TR_SERVER=1 TR_SERVER_NAME=$NAME timeout 60 $B serve --status 2>&1 || true)
  echo "$ST" > $OUT/status-$1.txt
  TR_SERVER=1 TR_SERVER_NAME=$NAME timeout 60 $B serve --stop > /dev/null 2>&1 || true
  # only the server: a bare `wait` would wait for tools/stay_awake.ps1 too, which lasts as long as
  # the measurement; and not for ever, the marker holds the other windows meanwhile
  N=0
  while kill -0 $SERVER 2> /dev/null && [ $N -lt 60 ]; do cleanup_run sleep 1; N=$((N + 1)); done
  if kill -0 $SERVER 2> /dev/null; then
    echo "serve_first_prompt: the $1 server did not stop in 60 s, killed"
    kill $SERVER 2> /dev/null || true
  fi
  wait $SERVER || true
  GOT=$(echo "$ST" | sed -n 's/^server: \([0-9]*\) requests, \([0-9]*\) loads.*/\1 \2/p')
  [ "$GOT" = "$2 1" ] || { echo "serve_first_prompt: the $1 server says '$ST': $2 requests and 1 load wanted, warm runs ran elsewhere"; exit 1; }
}
# $1: label, then the ab_modes modes
modes() {
  L=$1
  shift
  AB_WALL=1 cleanup_run sh tools/ab_modes.sh $R "$@" > $OUT/$L.txt
  grep -E '^(prefill|wall_ms|misses|hits) +[^ ]+ +median' $OUT/$L.txt || tail -20 $OUT/$L.txt
}

echo "##### half: budget $HALF MiB, prompt 2048, a new process and the server's kept store alternated"
server_up half $HALF
# the cold runs load half a model beside the server's: its own copy and the guard's 3 GiB
wait_memory 8
modes half "cold=TR_SERVER=0 $B $G --expert-budget $HALF" \
  "warm=TR_SERVER=1 TR_SERVER_NAME=$NAME $B $G --expert-budget $HALF" \
  "coldagain=TR_SERVER=0 $B $G --expert-budget $HALF"
server_down half $((R + 1))

FULLB=$((FULL + 64))
echo "##### full: the whole table resident; new processes, then the server"
wait_memory 11
modes full-cold "cold=TR_SERVER=0 $B $G --expert-budget $FULLB" "coldagain=TR_SERVER=0 $B $G --expert-budget $FULLB"
server_up full $FULLB
modes full-warm "warm=TR_SERVER=1 TR_SERVER_NAME=$NAME $B $G --expert-budget $FULLB" \
  "warmagain=TR_SERVER=1 TR_SERVER_NAME=$NAME $B $G --expert-budget $FULLB"
server_down full $((2 * (R + 1)))
measure_declare "after the last run"
echo "done: $OUT"
}
main "$@"; exit
