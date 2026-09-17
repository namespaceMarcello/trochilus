#!/bin/sh
# Builds llama.cpp (ref/llama.cpp, commit in docs/ORIGINI.md) for the CPU and the reference
# program tools/llamacpp_logits.c against it. Runs in the trochilus-dev container:
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/src" -w /src trochilus-dev:local sh tools/build_llamacpp.sh
# Output: ref/llama.cpp/build-trochilus/bin/{llamacpp_logits,llama-tokenize,llama-bench}
set -e
command -v cmake >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq cmake >/dev/null; }
B=ref/llama.cpp/build-trochilus
cmake -S ref/llama.cpp -B "$B" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
    -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
cmake --build "$B" --target llama llama-tokenize llama-bench -j "$(nproc)" >/dev/null
mkdir -p "$B/bin"
g++ -O2 -x c tools/llamacpp_logits.c -x none -Iref/llama.cpp/include -Iref/llama.cpp/ggml/include \
    "$B/src/libllama.a" $(find "$B/ggml/src" -name 'libggml*.a' | sort -r) \
    -lpthread -lm -o "$B/bin/llamacpp_logits" 2>/dev/null || \
g++ -O2 -x c tools/llamacpp_logits.c -x none -Iref/llama.cpp/include -Iref/llama.cpp/ggml/include \
    -Wl,--start-group "$B/src/libllama.a" $(find "$B/ggml/src" -name 'libggml*.a') -Wl,--end-group \
    -lpthread -lm -o "$B/bin/llamacpp_logits"
ls -la "$B/bin/llamacpp_logits" "$B/bin/llama-tokenize" "$B/bin/llama-bench"
