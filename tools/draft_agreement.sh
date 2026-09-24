#!/bin/sh
# draft_agreement.sh — how often a quantized copy of the model picks the exact model's token
# (docs/MEASUREMENTS.md question 53: a draft on the GPU verified exactly by the engine).
#
# Three real texts (prose, code, Italian prose from this repo), the first 1024 tokens as the prompt,
# 512 tokens of the exact model's greedy continuation; then the logits of every position of the
# 1536, through `logits -b 1`, of the exact model and of each draft model.
# tools/draft_agreement_report.py reads them: top-1 agreement on the real text and on the greedy
# trajectory, KL, tokens a pass of a k-token draft would give.
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/src" -v trochilus-models:/src/models -w /src \
#     trochilus-dev:local sh tools/draft_agreement.sh [exact.gguf] [draft.gguf...]
#   (defaults: OLMoE Q8_0 exact; Q4_K and Q4_K_M drafts, from tools/quantize_q4k.sh)
#   ... trochilus-dev:local sh -c '/opt/venv/bin/python tools/draft_agreement_report.py models/draft q4_k'
#
# Output in models/draft/ (the volume: ~300 MB of logits per text and model). Correctness only, no
# timing: it runs in the container, beside anything.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
T=build/linux-gcc/trochilus
make BUILD=build/linux-gcc CC=gcc $T > /dev/null
EXACT=${1:-models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf}
[ $# -gt 0 ] && shift
DRAFTS=${*:-models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf}
O=models/draft
mkdir -p $O
sed -n 1633,1760p docs/MEASUREMENTS.md > $O/ital.txt
for name in prose code ital; do
  case $name in
    prose) f=docs/ARCHITECTURE.md ;;
    code) f=src/models/olmoe.c ;;
    ital) f=$O/ital.txt ;;
  esac
  $T tokenize -m $EXACT -f $f | tr ',' '\n' | head -n 1024 | paste -sd, > $O/$name.prompt
  $T generate -m $EXACT --tokens "$(cat $O/$name.prompt)" -n 512 2> $O/$name.gen.err | sed 's/^tokens: //' > $O/$name.gen
  echo "$(cat $O/$name.prompt),$(cat $O/$name.gen)" > $O/$name.all
  $T logits -m $EXACT --tokens "$(cat $O/$name.all)" --out $O/$name.exact.bin -b 1 2> $O/$name.exact.err
  for d in $DRAFTS; do
    tag=$(basename $d .gguf | sed 's/.*-//' | tr 'A-Z' 'a-z')
    $T logits -m $d --tokens "$(cat $O/$name.all)" --out $O/$name.$tag.bin -b 1 2> $O/$name.$tag.err
  done
  echo "$name: $(tr ',' '\n' < $O/$name.all | wc -l) tokens"
done
}
main "$@"; exit
