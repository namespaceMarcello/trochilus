#!/bin/sh
# draft_ctx_gen.sh — the texts for `tools/draft_gate_sim.py --set build/draft_ctx` (docs/MEASUREMENTS.md
# §The post-it taken apart, question 81: the cheap drafts with a real context). The greedy continuation of
# every prompt in bench/prompts/real-context (a file of this repo cut at the start of a function's body
# mid-file, 1770-3798 tokens: sources.tsv, cut.py), 256 tokens, Q4_K, the prompt lookup fixed at 8 so the engine
# prints its own counters; two of them stopped at 200 tokens, whose counters the replay must match
# exactly; and this repo's code tokenized one file a line (build/draft_ctx/repo.ids, the paths in
# repo.files) for the repo table, built without the prompt's own file. Tokens do not depend on --spec.
#
#   sh tools/draft_ctx_gen.sh [binary]    (Git Bash or Linux, with Docker; ~15 min)
#   binary: a Linux engine under this directory (default: build/linux-gcc/trochilus, made here)
#   then:   tools/.venv/Scripts/python.exe tools/draft_gate_sim.py --set build/draft_ctx \
#             --repo build/draft_ctx/repo.ids --sources bench/prompts/real-context/sources.tsv --ctx real
#
# Correctness only, no timing: every engine call runs in the container (trochilus-dev:local, the models
# in the trochilus-models volume), one at a time, and none starts while the machine's marker
# (~/.claude/macchina-ferma, tools/measure_guard.lib) says a native measurement is on: it waits,
# looking again every minute.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap on_exit EXIT
trap 'exit 130' INT TERM
export MSYS_NO_PATHCONV=1
MACHINE_MARKER=${MACHINE_MARKER:-$HOME/.claude/macchina-ferma}
M=${DRAFT_MODEL:-models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf}
SET=bench/prompts/real-context
OUT=build/draft_ctx
VAL=${VAL:-"c05_olmoe py03_route_trace"}
RUNS=0
mkdir -p $OUT/gen $OUT/val
BIN=$1
if [ -z "$BIN" ]; then
  BIN=build/linux-gcc/trochilus
  engine make BUILD=build/linux-gcc CC=gcc $BIN > /dev/null || exit 1
fi

# this repo's code, one record a file (<byte length>\n<bytes>, tokenize --batch): C, headers, Python,
# shell and the Makefile of src/, tools/, tests/; not the generated tables, not the prompts (.txt)
git ls-files --cached --others --exclude-standard src tools tests Makefile |
  grep -E '(\.(c|h|py|sh|lib|awk)|^Makefile)$' | grep -v -E 'unicode_data\.h$|expf_table\.h$' |
  while read -r f; do [ -s "$f" ] && echo "$f"; done > $OUT/repo.files
while read -r f; do
  n=$(wc -c < "$f")
  echo $n
  cat "$f"
done < $OUT/repo.files > $OUT/repo.batch
engine $BIN tokenize -m $M --no-add-special --batch $OUT/repo.batch > $OUT/repo.ids || exit 1
echo "repo: $(wc -l < $OUT/repo.files) files, $(tr ',' '\n' < $OUT/repo.ids | wc -l) tokens"

for p in $SET/*.txt; do
  b=$(basename "$p" .txt)
  engine $BIN tokenize -m $M -f $p > $OUT/gen/$b.prompt || exit 1
  # the ids are read inside the container: a command line of Windows stops at 32 767 characters
  engine sh -c '"$1" generate -m "$2" --tokens "$(cat "$3")" -n 256 -t 8 --spec 8 --spec-fixed' sh \
    $BIN $M $OUT/gen/$b.prompt > $OUT/gen/$b.out 2> $OUT/gen/$b.err || exit 1
  echo "$b $(tr ',' '\n' < $OUT/gen/$b.prompt | wc -l) prompt tokens, $(grep -h '^speculation' $OUT/gen/$b.err)"
done
for b in $VAL; do
  engine sh -c '"$1" generate -m "$2" --tokens "$(cat "$3")" -n 200 -t 8 --spec 8 --spec-fixed' sh \
    $BIN $M $OUT/gen/$b.prompt > $OUT/val/$b.out 2> $OUT/val/$b.err || exit 1
  echo "val $b $(grep -h '^speculation' $OUT/val/$b.err)"
done
rm -f $OUT/repo.batch
}

# one command in the container, never while a native measurement holds the machine's marker
engine() {
  if [ -f "$MACHINE_MARKER" ]; then
    echo "waiting: $(head -1 "$MACHINE_MARKER" 2> /dev/null)" >&2
    while [ -f "$MACHINE_MARKER" ]; do cleanup_run sleep 60; done
    echo "the marker is gone, going on" >&2
  fi
  RUNS=$((RUNS + 1))
  cleanup_run docker run --rm --label draftctx=$$ --name draftctx-$$-$RUNS -v "$PWD:/src" \
    -v trochilus-models:/src/models -w /src trochilus-dev:local "$@"
}

# a container outlives the docker client that started it: it goes first, then this script's children
on_exit() {
  CONTAINERS=$(docker ps -aq --filter label=draftctx=$$ 2> /dev/null)
  [ -z "$CONTAINERS" ] || docker rm -f $CONTAINERS > /dev/null 2>&1
  cleanup_children
}

main "$@"; exit
