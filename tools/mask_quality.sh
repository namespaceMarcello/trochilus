#!/bin/sh
# mask_quality.sh — does the model's output on code need only the experts code uses most?
# (docs/MEASUREMENTS.md domanda 44, the functional test.) Experts are switched off with --expert-mask, a
# measurement-only mode: by usage in ONE trace of code (the most used 75, 50, 25% of the units
# stay), and, as a control, the same number drawn at random. Then, on texts the mask has never
# seen (other code, and prose), the logits of every position against the whole model: mean KL
# and how many positions choose another token (tools/expf_quality.py).
#
#   sh tools/mask_quality.sh            from the repo root, in Git Bash
#
# Needs build/route/code-1000.bin (trochilus run --route-trace on bench/prompts/code-1000.txt)
# and the Linux build of the engine (build/linux-gcc/trochilus); the engine runs in the container
# with the models volume (Smart App Control blocks a fresh Windows exe, docs/LESSONS.md #12), the
# masks and the comparison run here with the project's Python. Counts and KL, no stopwatch: the
# machine does not have to be still. About 25 minutes. Results in build/mask/quality.txt.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PY=${PY:-tools/.venv/Scripts/python.exe}
[ -f "$PY" ] || PY=tools/.venv/bin/python
TRACE=build/route/code-1000.bin
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
OUT=build/mask
TEXTS="code-1000 trace-c2 trace-py trace-sh trace-prose-en"
KEEPS="75 50 25"
[ -f $TRACE ] && [ -f build/linux-gcc/trochilus ] || { echo "mask_quality: $TRACE or build/linux-gcc/trochilus is missing"; exit 1; }
mkdir -p $OUT

for K in $KEEPS; do
  $PY tools/route_graph_report.py --mask-from $TRACE --keep 0.$K --out $OUT/usage-$K.txt
  $PY tools/route_graph_report.py --mask-from $TRACE --keep 0.$K --out $OUT/random-$K.txt --random 1
done

# What the container runs for one text ($1): the whole model, then every mask; a file of logits
# each, one row per position (~170 MB), and the engine's own line about the mask as the proof
# that the mask was really applied.
cat > $OUT/run_text.sh <<'EOF'
set -e
T=$1; M=$2; OUT=$3; KEEPS=$4
B=build/linux-gcc/trochilus
IDS=$($B tokenize -m $M -f bench/prompts/$T.txt)
$B logits -m $M --tokens $IDS --out $OUT/$T-full.bin -t 8 --decode-threads 8 2> /dev/null
for K in $KEEPS; do
  for HOW in usage random; do
    $B logits -m $M --tokens $IDS --out $OUT/$T-$HOW-$K.bin -t 8 --decode-threads 8 --expert-mask $OUT/$HOW-$K.txt 2> $OUT/$T-$HOW-$K.err
    grep -q '^expert mask:' $OUT/$T-$HOW-$K.err
  done
done
EOF

: > $OUT/quality.txt
for T in $TEXTS; do
  cleanup_run env MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -v trochilus-models:/src/models -w /src trochilus-dev:local sh $OUT/run_text.sh $T $M $OUT "$KEEPS"
  for K in $KEEPS; do
    for HOW in usage random; do
      OFF=$(grep '^expert mask:' $OUT/$T-$HOW-$K.err | cut -d' ' -f3-5)
      echo "== $T keep $K% by $HOW ($OFF units off)" >> $OUT/quality.txt
      $PY tools/expf_quality.py $OUT/$T-full.bin $OUT/$T-$HOW-$K.bin >> $OUT/quality.txt
    done
  done
  rm -f $OUT/$T-full.bin $OUT/$T-usage-*.bin $OUT/$T-random-*.bin
  echo "$T done"
done
rm -f $OUT/run_text.sh
cat $OUT/quality.txt
echo "done: $OUT/quality.txt"
}
main "$@"; exit
