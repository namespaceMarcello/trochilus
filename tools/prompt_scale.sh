#!/bin/sh
# prompt_scale.sh -- the prompt pass at two thread counts, traced, for tools/prompt_timeline.py:
# where 8 -> 16 threads lose against a perfect 2x (work per unit, tails, the slow cores).
#
#   sh tools/prompt_scale.sh <out dir> [threads...]              native, build/win-trace/trochilus.exe
#   PS_BIN=build/linux-trace/trochilus sh tools/prompt_scale.sh <out dir> 16 8   in the container:
#     run it inside trochilus-dev with -v trochilus-models:/src/models (the VM needs ~6 GB free)
#   TR_POOL_PIN=0 ...                                            the same without the pool's pin
#
# The prompt is 2048 tokens of src/models/olmoe.c repeated, as tools/speed_compare.py builds it
# (PS_TOKENS for another length, PS_MODEL for another model; Q4_K by default: 3.7 GB, the Q8_0 is 7).
# One run per thread count: the machine is shared, so read structure (ratios, tails, which chunk is
# slow), not times. Then:
#   tools/.venv/Scripts/python.exe tools/prompt_timeline.py <out>/trace_t8.txt <out>/trace_t16.txt --ccd 8
set -e
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
OUT=${1:?usage: prompt_scale.sh <out dir> [threads...]}
shift
[ $# -gt 0 ] || set -- 8 16
BIN=${PS_BIN:-build/win-trace/trochilus.exe}
M=${PS_MODEL:-models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf}
N=${PS_TOKENS:-2048}
mkdir -p "$OUT"
"$BIN" tokenize -m "$M" -f src/models/olmoe.c --no-add-special > "$OUT/source_ids.txt"
# the first N ids, the file repeated as often as needed, comma separated
IDS=$(tr -cs '0-9' ' ' < "$OUT/source_ids.txt" | awk -v n="$N" '{ for (i = 1; i <= NF; i++) a[c++] = $i }
  END { s = a[0]; for (i = 1; i < n; i++) s = s "," a[i % c]; print s }')
echo "prompt_scale: $N tokens, $BIN, pin ${TR_POOL_PIN:-2}, $(date +%H:%M:%S)"
for T in "$@"; do
  TR_POOL_TRACE_FILE="$OUT/trace_t$T.txt" "$BIN" generate -m "$M" --tokens "$IDS" -n 1 -t "$T" \
    -c $((N + 1)) --decode-threads "$T" > "$OUT/out_t$T.txt" 2> "$OUT/err_t$T.txt" ||
    { echo "prompt_scale: t=$T failed"; tail -2 "$OUT/err_t$T.txt"; exit 1; }
  echo "t=$T $(grep -h 'prompt:' "$OUT/err_t$T.txt")"
done
}
main "$@"; exit
