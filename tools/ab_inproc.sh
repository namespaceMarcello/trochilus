#!/bin/sh
# ab_inproc.sh -- an in-process A/B: one binary, its passes alternated A B B A between the two arms of a switch
# (generate --ab), each arm's zones timed apart, N runs; then tools/ab_inproc.py. Both arms run in one file, one
# memory and one set of threads: what two files of the same bytes differ by (1-2%, docs/LESSONS.md #243) and the
# slots of a run-by-run race cannot enter. Native, still machine (tools/measure_guard.lib), the attention on the
# CPU (TR_GPU=0), every pass on the given threads (--decode-threads). code-edit.txt, 200 tokens (AB_INPROC_N).
#
#   sh tools/ab_inproc.sh <binary> <switch> [runs] [model] [threads] [draft]
#       switch: none (the A/A: the tool's own noise), or one the binary knows (generate --ab);
#       runs 6; model Q4_K; threads 8; draft 0 (one row a pass), or k: --spec k --spec-fixed (k + 1 rows a pass).
#   AB_INPROC_OUT (default build/ab_inproc) takes the runs; a folder that already holds some is refused.
#   AB_INPROC_PROMPT=R adds --ab-prompt R: the prompt evaluated R times more a run (a multiple of 4), its arms
#   A B B A, for a switch that changes the prompt's road.
#   AB_INPROC_N (default 200): the tokens a run generates; fewer keep a slow machine's run within 60 s (the
#   8 GB machine's Q8_0: 80, after its 411-token prompt).
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
B=${1:-}
SW=${2:-}
R=${3:-6}
M=${4:-Q4_K}
T=${5:-8}
D=${6:-0}
[ -n "$B" ] && [ -n "$SW" ] || { echo "usage: sh tools/ab_inproc.sh <binary> <switch> [runs] [model] [threads] [draft]"; exit 2; }
[ -f "$B" ] || { echo "ab_inproc: $B is missing"; exit 1; }
OUT=${AB_INPROC_OUT:-build/ab_inproc}
MF=models/OLMoE-1B-7B-0125-Instruct-$M.gguf
[ -f "$MF" ] || { echo "ab_inproc: $MF is missing"; exit 1; }
mkdir -p "$OUT"
if [ -n "$(find "$OUT" -name '*.json' 2> /dev/null | head -1)" ]; then
  echo "ab_inproc: $OUT already holds runs: move them, or give another AB_INPROC_OUT"
  exit 1
fi
TR_GPU=0
export TR_GPU
. tools/measure_guard.lib
# a binary Smart App Control holds: a copy a byte longer, found before the marker is taken (LESSONS #322)
B=$(measure_runnable "$B") || { echo "ab_inproc: $B is held by Smart App Control, its longer copies too"; exit 1; }
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin ab_inproc
measure_machine ab_inproc
measure_still ab_inproc
measure_declare "before the first run"
SPEC=""
[ "$D" = 0 ] || SPEC="--spec $D --spec-fixed"
[ "${AB_INPROC_PROMPT:-0}" = 0 ] || SPEC="$SPEC --ab-prompt $AB_INPROC_PROMPT"
echo "$B --ab $SW, $M, $T threads, draft $D" > "$OUT/what.txt"
IDS=$($B tokenize -m $MF -f bench/prompts/code-edit.txt)
k=1
while [ $k -le "$R" ]; do
  eval "$AB_GUARD" || { echo "ab_inproc: machine busy or marker lost"; exit 3; }
  cleanup_run $B generate -m $MF --tokens "$IDS" -n "${AB_INPROC_N:-200}" -t $T --decode-threads $T $SPEC --ab $SW \
    --profile-json "$OUT/run-$k.json" > "$OUT/run-$k.out" 2>&1
  grep '^ab:' "$OUT/run-$k.out" || true
  k=$((k + 1))
done
measure_declare "after the last run"
PYBIN=tools/.venv/Scripts/python.exe
[ -x "$PYBIN" ] || PYBIN=tools/.venv/bin/python
$PYBIN tools/ab_inproc.py "$OUT"/run-*.json
}
main "$@"; exit
