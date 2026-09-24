#!/bin/sh
# bench_ggml.sh -- builds and runs tools/bench_ggml.c against llama.cpp's ggml (ref/llama.cpp,
# commit b49650a, already built in ref/llama.cpp/build-trochilus/: GGML_NATIVE, GGML_LLAMAFILE,
# GGML_CPU_REPACK all ON). Runs in the trochilus-dev container:
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/src" -w /src trochilus-dev:local sh tools/bench_ggml.sh
# Output: ref/llama.cpp/build-trochilus/bin/bench_ggml, run once at the end of this script.
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
B=ref/llama.cpp/build-trochilus
mkdir -p "$B/bin"
g++ -O2 -x c tools/bench_ggml.c -x none -Iref/llama.cpp/ggml/include \
    $(find "$B/ggml/src" -name 'libggml*.a' | sort -r) \
    -lpthread -lm -o "$B/bin/bench_ggml" 2>/dev/null || \
g++ -O2 -x c tools/bench_ggml.c -x none -Iref/llama.cpp/ggml/include \
    -Wl,--start-group $(find "$B/ggml/src" -name 'libggml*.a') -Wl,--end-group \
    -lpthread -lm -o "$B/bin/bench_ggml"
"$B/bin/bench_ggml"
}
main "$@"; exit
