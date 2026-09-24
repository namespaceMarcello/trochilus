#!/bin/sh
# bench_expf_refs.sh — builds and runs tools/bench_expf_refs.c in the container: llama.cpp's exp
# and the C library's against tr_expf on every float, then their cost a value (ORIGINS row 4).
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/src" -w /src trochilus-dev:local sh tools/bench_expf_refs.sh [threads]
# Output on stdout and in build/bench_expf_refs.txt.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
mkdir -p build
gcc -O2 -std=c11 -ffp-contract=off -Wall -Wextra -Werror -Wno-old-style-declaration tools/bench_expf_refs.c src/kernels/expf.c \
    -o build/bench_expf_refs -lm -lpthread || exit 1
cleanup_run build/bench_expf_refs "${1:-16}" > build/bench_expf_refs.txt
RC=$?
cat build/bench_expf_refs.txt
exit $RC
}
main "$@"; exit
