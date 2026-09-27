#!/bin/sh
# draft_table_gen.sh — the text for tools/draft_table_sim.py (docs/MEASUREMENTS.md §A draft without a
# model): the greedy continuation of every prompt in bench/prompts/new-code (256 tokens, Q4_K), with the
# prompt lookup fixed at 8 so the engine prints its own counters; then three of them stopped at 200
# tokens, whose counters the replay must match exactly. Tokens do not depend on --spec.
#
#   sh tools/draft_table_gen.sh [binary]    (default build/trochilus.exe; ~20 min at 8 threads)
# A binary Smart App Control blocks runs as a copy one byte longer (docs/LESSONS.md #12).
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
BIN=${1:-build/trochilus.exe}
M=${DRAFT_MODEL:-models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf}
OUT=build/draft_table
mkdir -p $OUT/gen $OUT/val
for p in bench/prompts/new-code/*.txt; do
  b=$(basename "$p" .txt)
  "$BIN" tokenize -m "$M" -f "$p" > $OUT/gen/$b.prompt || exit 1
  cleanup_run "$BIN" generate -m "$M" --tokens "$(cat $OUT/gen/$b.prompt)" -n 256 -t 8 --spec 8 \
    --spec-fixed > $OUT/gen/$b.out 2> $OUT/gen/$b.err || exit 1
  echo "$b $(grep -h '^speculation' $OUT/gen/$b.err)"
done
for b in c1_hashmap.c js2_express.js py2_cache.py; do
  cleanup_run "$BIN" generate -m "$M" --tokens "$(cat $OUT/gen/$b.prompt)" -n 200 -t 8 --spec 8 \
    --spec-fixed > $OUT/val/$b.out 2> $OUT/val/$b.err || exit 1
  echo "val $b $(grep -h '^speculation' $OUT/val/$b.err)"
done
}
main "$@"; exit
