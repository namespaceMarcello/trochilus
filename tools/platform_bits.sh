#!/bin/sh
# platform_bits.sh — does the engine give the same bytes on Windows and on Linux? With tr_expf
# the exponential is the same on both; what is left from a C library is the RoPE table (pow, cos
# and sin in double, rounded to float: tr_rope_table). Native Windows, Git Bash, from the repo
# root, with Docker running; results in build/platform_bits/.
#
#   sh tools/platform_bits.sh [tokens, default 1000]
#
#   logits   the same tokens through build/trochilus.exe and through build/linux-gcc/trochilus
#            in the container, one token a pass, one row of logits per position: the tiny
#            fixtures (f32, f16, q8_0; 120 positions, their whole context) and the real model
#            cut to 2 layers (fixtures/olmoe-2layer, skipped when it is not there). Same bytes
#            or not, and if not the first position that differs, the largest difference, the KL.
#   rope     the RoPE tables of the two platforms (tests/dump_rope.c) entry by entry, and the
#            first position where they differ: if the table is the only source, no row of logits
#            differs before that position (tools/rope_table_compare.py, tools/expf_quality.py).
#
# A measurement, not a gate: it changes nothing and always ends with 0 unless a step fails.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
N=${1:-1000}
OUT=build/platform_bits
WIN=build/trochilus.exe
LIN=build/linux-gcc/trochilus
PY=tools/.venv/Scripts/python.exe
REAL=fixtures/olmoe-2layer/model.gguf
# DUMP_ROPE=<binary>: a second copy of the tool built elsewhere (make BUILD=build/after2
# build/after2/tests/dump_rope.exe) when Smart App Control still blocks the first (docs/LESSONS.md #81)
DUMP=${DUMP_ROPE:-build/tests/dump_rope.exe}
mkdir -p $OUT
make $WIN build/tests/dump_rope.exe > /dev/null
in_container() {
  MSYS_NO_PATHCONV=1 docker run --rm --security-opt seccomp=unconfined -v "$(pwd -W):/src" -w /src trochilus-dev:local sh -c "$1"
}
in_container "make BUILD=build/linux-gcc CC=gcc $LIN build/linux-gcc/tests/dump_rope > /dev/null"

# $1: name, $2: model, $3: vocabulary, $4: tokens
one() {
  NAME=$1; MODEL=$2; VOCAB=$3; COUNT=$4
  TOK=$(awk -v n=$COUNT -v v=$VOCAB 'BEGIN { for (i = 0; i < n; i++) printf "%s%d", (i ? "," : ""), (i * 7919 + 13) % v }')
  echo "##### $NAME: $COUNT tokens, one a pass"
  $WIN logits -m $MODEL --tokens $TOK --out $OUT/win-$NAME.bin -t 8 > /dev/null 2>&1
  in_container "$LIN logits -m $MODEL --tokens $TOK --out $OUT/linux-$NAME.bin -t 8 > /dev/null 2>&1"
  [ -s $OUT/win-$NAME.bin ] && [ -s $OUT/linux-$NAME.bin ] || { echo "platform_bits: no logits for $NAME"; exit 1; }
  if cmp -s $OUT/win-$NAME.bin $OUT/linux-$NAME.bin; then
    echo "logits identical byte for byte: $(wc -c < $OUT/win-$NAME.bin) bytes"
  else
    echo "logits DIFFER"
    $PY tools/expf_quality.py $OUT/win-$NAME.bin $OUT/linux-$NAME.bin --vocab $VOCAB
  fi
  rm -f $OUT/win-$NAME.bin $OUT/linux-$NAME.bin
}
# $1: name, $2: positions, $3: head_dim
rope() {
  echo "##### the RoPE tables, $1: $2 positions, head_dim $3"
  $DUMP $2 $3 10000 $OUT/rope-win-$1.bin > /dev/null
  in_container "build/linux-gcc/tests/dump_rope $2 $3 10000 $OUT/rope-linux-$1.bin > /dev/null"
  $PY tools/rope_table_compare.py $OUT/rope-win-$1.bin $OUT/rope-linux-$1.bin $2 $3 10000
}
body() {
  echo "platform_bits, $(date '+%Y-%m-%d %H:%M'): A = Windows (MinGW-w64), B = Linux (glibc)"
  one tiny-f32 fixtures/tiny-olmoe/model-f32.gguf 128 120
  one tiny-f16 fixtures/tiny-olmoe/model-f16.gguf 128 120
  one tiny-q8_0 fixtures/tiny-olmoe/model-q8_0.gguf 128 120
  if [ -f $REAL ]; then one real-2layer $REAL 50304 $N; else echo "##### real-2layer: SKIPPED, $REAL not found"; fi
  # the tables last: a tool built a minute ago may be one Smart App Control still blocks (exit 126
  # from Git Bash, docs/LESSONS.md #12); the logits above do not need it
  RC=0
  $DUMP > /dev/null 2>&1 || RC=$?
  if [ "$RC" = 126 ]; then
    echo "##### the RoPE tables: SKIPPED, $DUMP does not run yet (Smart App Control): run this script again later"
  else
    rope tiny 120 16
    rope real 4096 128
  fi
}
# a failure inside must fail the script: through a pipe into tee it ended with 0
set +e
( set -e; body ) > $OUT/report.txt 2>&1
RC=$?
set -e
cat $OUT/report.txt
exit $RC
}
main "$@"; exit
