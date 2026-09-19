#!/bin/sh
# expf_quality.sh — what an expf of our own would change on Linux, measured before writing it
# (docs/MISURE.md question 37). glibc's expf is not the correctly rounded value for 170 648 floats
# (tests/bench_expf.c); an expf of ours that rounds correctly gives, on Linux, what the engine
# gives when every expf is the double precision exp rounded to float: that binary is built here
# (tools/cr_expf_emul.h, a measurement build, slow) and compared with the normal one on the real
# model: the tokens of a greedy generation, and at every position the chosen token, the KL and
# the largest difference between logits. In the trochilus-dev container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -v trochilus-models:/src/models -w /src \
#       trochilus-dev:local sh tools/expf_quality.sh [tokens to generate, default 1000]
#
# About 6 minutes. On Windows there is nothing to measure: MinGW's expf is the correctly rounded
# value everywhere, so the two binaries would be the same numbers.
set -e
# The body is one function, called on the last line (docs/LEZIONI.md #69).
main() {
N=${1:-1000}
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
LIB=build/linux-gcc/trochilus
CR=build/linux-crexpf/trochilus
OUT=build/expf_quality
PYBIN=${PY:-tools/.venv/bin/python}
[ -f $M ] || { echo "expf_quality: $M is missing (mount the trochilus-models volume)"; exit 1; }
mkdir -p $OUT
make BUILD=build/linux-gcc CC=gcc $LIB > /dev/null
# the second binary: a normal build, then only the sources that call expf compiled again with the
# emulation in front (in front of every file it would come before their _GNU_SOURCE), and linked
make BUILD=build/linux-crexpf CC=gcc $CR > /dev/null
for SRC in $(grep -l "expf *(" src/*/*.c); do
  make BUILD=build/linux-crexpf CC=gcc EXTRA_CFLAGS="-include tools/cr_expf_emul.h" -W $SRC \
       build/linux-crexpf/${SRC%.c}.o > /dev/null
  echo "expf_quality: $SRC compiled with every expf correctly rounded"
done
make BUILD=build/linux-crexpf CC=gcc $CR > /dev/null

PROMPT=$($LIB tokenize -m $M -f bench/prompts/code-edit.txt)
[ -n "$PROMPT" ] || { echo "expf_quality: no tokens from the prompt"; exit 1; }
CTX=$(( $(echo "$PROMPT" | tr ',' '\n' | wc -l) + N + 8 ))
$LIB generate -m $M --tokens "$PROMPT" -n $N -c $CTX -t 16 2> /dev/null | sed -n 's/^tokens: *//p' > $OUT/tokens-lib.txt
$CR generate -m $M --tokens "$PROMPT" -n $N -c $CTX -t 16 2> /dev/null | sed -n 's/^tokens: *//p' > $OUT/tokens-cr.txt
[ -s $OUT/tokens-lib.txt ] && [ -s $OUT/tokens-cr.txt ] || { echo "expf_quality: a generation gave no tokens"; exit 1; }

# teacher forcing on the prompt and on what the normal binary generated: logits at every position
ALL="$PROMPT,$(cat $OUT/tokens-lib.txt)"
$LIB logits -m $M --tokens "$ALL" --out /tmp/logits-lib.bin -t 16 > /dev/null 2>&1
$CR logits -m $M --tokens "$ALL" --out /tmp/logits-cr.bin -t 16 > /dev/null 2>&1
$PYBIN tools/expf_quality.py /tmp/logits-lib.bin /tmp/logits-cr.bin $OUT/tokens-lib.txt $OUT/tokens-cr.txt | tee $OUT/report.txt
rm -f /tmp/logits-lib.bin /tmp/logits-cr.bin
}
main "$@"; exit
