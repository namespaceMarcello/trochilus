#!/bin/sh
# gpu_exact.sh — the decode's attention on the GPU gives the CPU's bytes (src/backend/gpu_attn.h,
# docs/MEASUREMENTS.md question 51). Native; the same binary with TR_GPU=0 (the CPU) against its
# default (the GPU when there is one):
#
#   sh tools/gpu_exact.sh [full] [binary]     the real model, a few minutes:
#       1. logits of 2000 positions one token a pass: every position is a decode token, so the
#          GPU's path from context 1 to 2000, byte for byte (compared by SHA-256: 400 MB a run);
#       2. the tokens after a prompt of 4000 (passes of 512 written to VRAM by tr_gpu_attn_write,
#          then 48 decode tokens at contexts 4000-4048);
#       3. a speculative run on code (--spec 8: passes of up to 9 rows written to VRAM, rewinds,
#          decode tokens in between);
#   sh tools/gpu_exact.sh quick [binary]      the real model's 2-layer cut (models/olmoe-2layer,
#       the real attention's shape), for make check: logits of 1000 positions one token a pass,
#       and the tokens after a prompt of 3000; about 20 s.
#
# Every GPU run must say it ran: a gpu: line with more than 0 decode tokens (else a green
# comparison could be two CPU runs, docs/LESSONS.md #43). A machine with no GPU (no gpu: line at
# all) is skipped, not failed. Results in build/gpu_exact/.
set -e
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
# a kept server would run these commands with its own TR_GPU: never forward (src/app/serve.c)
export TR_SERVER=0
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
MODE=full
case "$1" in full|quick) MODE=$1; shift ;; esac
B=${1:-build/trochilus.exe}
OUT=build/gpu_exact
mkdir -p $OUT
if [ "$MODE" = quick ]; then
  M=models/olmoe-2layer/model.gguf; N_LOGITS=1000; PROMPT=3000; GEN=16
else
  M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf; N_LOGITS=2000; PROMPT=4000; GEN=48
fi
[ -f "$B" ] || { echo "gpu_exact: no binary '$B'"; exit 2; }
if [ ! -f "$M" ]; then
  [ "$MODE" = quick ] && { echo "gpu_exact quick: skipped (no $M: make oracle-real builds it)"; exit 0; }
  echo "gpu_exact: no model '$M'"; exit 2
fi

echo "== gpu_exact $MODE: logits of $N_LOGITS positions, one token a pass"
# a new binary Smart App Control still blocks is skipped with a line, like tools/native_tests.sh
rc=0; $B cpu > /dev/null 2>&1 || rc=$?
[ $rc -ne 126 ] || { echo "gpu_exact: SKIPPED, Smart App Control blocked $B (docs/LESSONS.md #12)"; exit 0; }
TOK=$(awk -v n=$N_LOGITS 'BEGIN { for (i = 0; i < n; i++) printf "%s%d", (i ? "," : ""), (i * 7919 + 13) % 50304 }')
TR_GPU=0 $B logits -m $M --tokens $TOK --out $OUT/logits-cpu.bin -t 8 > /dev/null 2> $OUT/logits-cpu.err
$B logits -m $M --tokens $TOK --out $OUT/logits-gpu.bin -t 8 > /dev/null 2> $OUT/logits-gpu.err
if ! grep -q '^gpu:' $OUT/logits-gpu.err; then
  rm -f $OUT/logits-cpu.bin $OUT/logits-gpu.bin
  # no gpu: line is a skip only where the driver sees no NVIDIA GPU either: on a machine that has
  # one, it means the engine failed to open it, and that is a failure, not a skip
  if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L 2> /dev/null | grep -q '^GPU '; then
    echo "gpu_exact: FAILED: nvidia-smi sees $(nvidia-smi -L | head -1 | cut -c1-60) but the engine opened no GPU"
    echo "  why: TR_GPU=1 $B logits ... prints 'gpu: none (<reason>)'"
    exit 3
  fi
  echo "gpu_exact: skipped (no NVIDIA GPU on this machine)"
  exit 0
fi
ran_on_gpu $OUT/logits-gpu.err
C=$(sha256sum < $OUT/logits-cpu.bin | cut -c1-64)
G=$(sha256sum < $OUT/logits-gpu.bin | cut -c1-64)
SIZE=$(wc -c < $OUT/logits-gpu.bin)
rm -f $OUT/logits-cpu.bin $OUT/logits-gpu.bin
[ "$C" = "$G" ] || { echo "gpu_exact: logits differ (cpu $C, gpu $G)"; exit 4; }
echo "  logits identical: $SIZE bytes, sha256 $C"

echo "== tokens after a prompt of $PROMPT"
TR_GPU=0 $B generate -m $M -p $PROMPT -n $GEN -c 4096 -t 8 > $OUT/tokens-cpu.txt 2> $OUT/tokens-cpu.err
$B generate -m $M -p $PROMPT -n $GEN -c 4096 -t 8 > $OUT/tokens-gpu.txt 2> $OUT/tokens-gpu.err
ran_on_gpu $OUT/tokens-gpu.err
cmp $OUT/tokens-cpu.txt $OUT/tokens-gpu.txt || { echo "gpu_exact: tokens differ after $PROMPT"; exit 4; }
echo "  tokens identical: $(cut -c1-60 $OUT/tokens-gpu.txt)..."

if [ "$MODE" = full ]; then
  echo "== speculation on code"
  head -c 11600 src/models/olmoe.c > $OUT/code.txt
  TR_GPU=0 $B run -m $M -f $OUT/code.txt -n 128 --spec 8 -t 8 -c 4600 > $OUT/spec-cpu.txt 2> $OUT/spec-cpu.err
  $B run -m $M -f $OUT/code.txt -n 128 --spec 8 -t 8 -c 4600 > $OUT/spec-gpu.txt 2> $OUT/spec-gpu.err
  ran_on_gpu $OUT/spec-gpu.err
  cmp $OUT/spec-cpu.txt $OUT/spec-gpu.txt || { echo "gpu_exact: speculative runs differ"; exit 4; }
  echo "  speculative text identical: $(grep '^speculation:' $OUT/spec-gpu.err | cut -c1-80)"
fi
echo "gpu_exact $MODE: the GPU's attention gave the CPU's bytes"
}

# ran_on_gpu <stderr file>: the run's gpu: line counts more than 0 decode tokens
ran_on_gpu() {
  n=$(sed -n 's/^gpu: .*attention of \([0-9]*\) decode tokens$/\1/p' "$1" | tail -1)
  [ -n "$n" ] && [ "$n" -gt 0 ] || { echo "gpu_exact: $1 did not run on the GPU (gpu: line: '${n:-none}')"; exit 3; }
  echo "  ran on the GPU: $n decode tokens"
}

main "$@"; exit
