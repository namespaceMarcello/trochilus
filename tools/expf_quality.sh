#!/bin/sh
# expf_quality.sh — what tr_expf changed on Linux, measured on the real model (docs/MEASUREMENTS.md
# question 37). glibc's expf is not the correctly rounded value for 170 648 floats
# (tests/bench_expf.c), tr_expf is for every float: on Linux the engine's numbers moved, by one
# unit of the last place in one exponential out of 25 000. Three binaries:
#
#   before   the engine of the commit before tr_expf, on glibc's expf
#            (BEFORE, default build/linux-before/trochilus: built before touching the code with
#            make BUILD=build/linux-before CC=gcc build/linux-before/trochilus)
#   emul     that same commit (REV, default ef3cb99) with every expf turned into the double
#            precision exp rounded to float, the slow way to the correctly rounded value
#            (tools/cr_expf_emul.h): its sources come out of git, the tree no longer calls expf
#   after    this tree: tr_expf
#
# and two answers: how far "after" is from "before" (tokens of a greedy generation, and at every
# position the chosen token, the KL, the largest difference between two logits), and that
# "after" and "emul" are the SAME BYTES, logits and tokens: tr_expf in the engine is the
# correctly rounded exp and nothing else moved. The second is a check: exit 1 if it fails.
# In the trochilus-dev container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -v trochilus-models:/src/models -w /src \
#       trochilus-dev:local sh tools/expf_quality.sh [tokens to generate, default 1000]
#
# About 10 minutes. On Windows there is nothing to measure: MinGW's expf is the correctly rounded
# value everywhere, and the stage `exact` of tools/prefill_context.sh asks for the same bytes.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
N=${1:-1000}
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
REV=${REV:-ef3cb99}
BEFORE=${BEFORE:-build/linux-before/trochilus}
AFTER=build/linux-gcc/trochilus
OUT=build/expf_quality
TREE=$OUT/src-$REV
EMUL=$TREE/b/trochilus
PYBIN=${PY:-tools/.venv/bin/python}
[ -f $M ] || { echo "expf_quality: $M is missing (mount the trochilus-models volume)"; exit 1; }
[ -x $BEFORE ] || { echo "expf_quality: $BEFORE is missing: the engine of the commit before tr_expf"; exit 1; }
mkdir -p $OUT
make BUILD=build/linux-gcc CC=gcc $AFTER > /dev/null
# emul: the sources of REV, a normal build, then only the files that call expf compiled again
# with the emulation in front (in front of every file it would come before their _GNU_SOURCE)
rm -rf $TREE && mkdir -p $TREE
git -c safe.directory="$(pwd)" archive $REV | tar -x -C $TREE
( cd $TREE
  make BUILD=b CC=gcc b/trochilus > /dev/null
  CALLERS=$(grep -l "expf *(" src/*/*.c)
  [ -n "$CALLERS" ] || { echo "expf_quality: no source of $REV calls expf"; exit 1; }
  for SRC in $CALLERS; do
    make BUILD=b CC=gcc EXTRA_CFLAGS="-include tools/cr_expf_emul.h" -W $SRC b/${SRC%.c}.o > /dev/null
    echo "expf_quality: $REV $SRC compiled with every expf correctly rounded"
  done
  make BUILD=b CC=gcc b/trochilus > /dev/null )

PROMPT=$($AFTER tokenize -m $M -f bench/prompts/code-edit.txt)
[ -n "$PROMPT" ] || { echo "expf_quality: no tokens from the prompt"; exit 1; }
CTX=$(( $(echo "$PROMPT" | tr ',' '\n' | wc -l) + N + 8 ))
for W in before emul after; do
  case $W in before) BIN=$BEFORE ;; emul) BIN=$EMUL ;; after) BIN=$AFTER ;; esac
  $BIN generate -m $M --tokens "$PROMPT" -n $N -c $CTX -t 16 2> /dev/null | sed -n 's/^tokens: *//p' > $OUT/tokens-$W.txt
  [ -s $OUT/tokens-$W.txt ] || { echo "expf_quality: the generation of '$W' gave no tokens"; exit 1; }
done
# teacher forcing on the prompt and on what "before" generated: logits at every position
ALL="$PROMPT,$(cat $OUT/tokens-before.txt)"
for W in before emul after; do
  case $W in before) BIN=$BEFORE ;; emul) BIN=$EMUL ;; after) BIN=$AFTER ;; esac
  $BIN logits -m $M --tokens "$ALL" --out /tmp/logits-$W.bin -t 16 > /dev/null 2>&1
done
# to a file and then shown: through a pipe into tee a failed comparison would end with 0
# (docs/LESSONS.md #90)
{
  echo "##### after (tr_expf) against before (glibc's expf)"
  $PYBIN tools/expf_quality.py /tmp/logits-before.bin /tmp/logits-after.bin $OUT/tokens-before.txt $OUT/tokens-after.txt
  echo "##### after (tr_expf) against emul ($REV with every expf correctly rounded)"
  $PYBIN tools/expf_quality.py /tmp/logits-emul.bin /tmp/logits-after.bin $OUT/tokens-emul.txt $OUT/tokens-after.txt
} > $OUT/report.txt
cat $OUT/report.txt
SAME=0
cmp /tmp/logits-emul.bin /tmp/logits-after.bin && cmp $OUT/tokens-emul.txt $OUT/tokens-after.txt || SAME=1
rm -f /tmp/logits-before.bin /tmp/logits-emul.bin /tmp/logits-after.bin
[ $SAME = 0 ] || { echo "expf_quality: FAILED, after and emul are not the same bytes" | tee -a $OUT/report.txt; exit 1; }
echo "expf_quality: after and emul are the same bytes, logits and tokens" | tee -a $OUT/report.txt
}
main "$@"; exit
