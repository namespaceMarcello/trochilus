#!/bin/sh
# ab_env.sh -- a switch read from the environment when a session starts, raced process by process: what changes
# state across passes (the KV's pages touched ahead, a load-time format) cannot take generate --ab's per-pass arms.
# One binary, modes "label=VAR=value ...", round-robin with a rotating first mode (tools/ab_modes.sh: its tok/s
# lines and medians), each run's zones kept (--profile-json), then tools/ab_env.py: per mode the prompt's and the
# decode's time and every zone, and each round's ratio to the first mode. Give a mode twice for the A/A. Native,
# still machine (tools/measure_guard.lib), the attention on the CPU (TR_GPU=0), every pass on the given threads,
# code-edit.txt (411 tokens), 200 tokens.
#
#   sh tools/ab_env.sh <binary> <rounds> "label=VAR=value" "label=VAR=value" [...]
#   e.g. sh tools/ab_env.sh build/q83/kt.exe 12 "on=TR_KV_TOUCH=1" "off=TR_KV_TOUCH=0" "on2=TR_KV_TOUCH=1"
#   AB_ENV_OUT (default build/ab_env) takes the runs; a folder that already holds some is refused.
#   AB_ENV_MODEL (Q4_K), AB_ENV_THREADS (8).
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
B=${1:-}
R=${2:-}
[ -n "$B" ] && [ -n "$R" ] && [ $# -ge 4 ] || { echo "usage: sh tools/ab_env.sh <binary> <rounds> \"label=VAR=value\" \"label=VAR=value\" [...]"; exit 2; }
shift 2
[ -f "$B" ] || { echo "ab_env: $B is missing"; exit 1; }
OUT=${AB_ENV_OUT:-build/ab_env}
M=${AB_ENV_MODEL:-Q4_K}
T=${AB_ENV_THREADS:-8}
MF=models/OLMoE-1B-7B-0125-Instruct-$M.gguf
[ -f "$MF" ] || { echo "ab_env: $MF is missing"; exit 1; }
mkdir -p "$OUT"
if [ -n "$(find "$OUT" -name '*.json' 2> /dev/null | head -1)" ]; then
  echo "ab_env: $OUT already holds runs: move them, or give another AB_ENV_OUT"
  exit 1
fi
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin ab_env
measure_machine ab_env
measure_still ab_env
measure_declare "before the first run" > "$OUT/load.txt"
cat "$OUT/load.txt"
IDS=$($B tokenize -m $MF -f bench/prompts/code-edit.txt)
# each mode's command: its variables, then the run; the profile's name carries the label and the clock, so
# tools/ab_env.py finds each round's runs in the order they ran (the ids have no spaces: "1,2,3")
MODES=""
for MODE in "$@"; do
  LABEL=${MODE%%=*}
  VARS=${MODE#*=}
  MODES="$MODES \"$LABEL=TR_GPU=0 $VARS $B generate -m $MF --tokens $IDS -n 200 -t $T --decode-threads $T --profile-json $OUT/$LABEL-\\\$(date +%s%N).json\""
done
echo "$B, $M, $T threads, modes:$MODES" | cut -c1-400 > "$OUT/what.txt"
eval "cleanup_run sh tools/ab_modes.sh $R $MODES" > "$OUT/ab_modes.txt"
grep '^[a-z]* *[a-z]* *median' "$OUT/ab_modes.txt" || true
measure_declare "after the last run" >> "$OUT/load.txt"
tail -1 "$OUT/load.txt"
PYBIN=tools/.venv/Scripts/python.exe
[ -x "$PYBIN" ] || PYBIN=tools/.venv/bin/python
$PYBIN tools/ab_env.py "$OUT" > "$OUT/report.txt"
cat "$OUT/report.txt"
}
main "$@"; exit
