#!/bin/sh
# quantize_q4k.sh — the real OLMoE in Q4_K, made from our Q8_0 by llama.cpp's llama-quantize: no
# download (M2, first step). --pure: every matrix llama-quantize quantizes becomes Q4_K, none Q6_K
# as Q4_K_M would mix them (the router and the norms stay F32, as llama-quantize keeps them);
# --allow-requantize: the source is already quantized, so these weights are Q8_0's rounded again,
# a model for the engine's speed and exactness, not for its quality. Runs in the trochilus-dev
# container, on the models volume:
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/src" -v trochilus-models:/src/models -w /src \
#       trochilus-dev:local sh tools/quantize_q4k.sh
# Output: models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf (about 3.9 GB; `make check` then runs its
# 2-layer cut against transformers, REAL_MODEL_Q4K in the Makefile).
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
SRC=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
DST=models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf
Q=ref/llama.cpp/build-trochilus/bin/llama-quantize
[ -f $SRC ] || { echo "quantize_q4k: $SRC is missing"; exit 1; }
[ -x $Q ] || sh tools/build_llamacpp.sh
FREE=$(df -Pk models | awk 'NR == 2 { print int($4 / 1048576) }')
[ "$FREE" -ge 8 ] || { echo "quantize_q4k: $FREE GB free on the models volume, 8 wanted"; exit 1; }
cleanup_run $Q --allow-requantize --pure $SRC $DST.tmp Q4_K "$(nproc)"
mv -f $DST.tmp $DST
ls -la $DST
}
main "$@"; exit
