#!/bin/sh
# mutate_files.sh -- tools/mutate_auto.py on the files of the engine, each with the tests and the
# checks that guard it: the C tests, and for a file whose numbers only transformers can judge (a
# model, the kernels, the command line) the oracles too, because the C tests of a model check that
# every split of the work gives the same bits, not that the bits are right (docs/LESSONS.md #112).
# Survivors are read one by one and either killed or named in docs/MEASUREMENTS.md
# §Generated mutations.
#
# Jobs are sized to the Docker VM's memory, not to its processors (docs/LESSONS.md #115): the
# model tests load a 600 MB model (test_stream) under a memory guard that reads the real free RAM,
# so 6 jobs; the tokenizer oracle runs without its full Unicode sweep (394 MB and 6 s instead of
# 4 GB and 31 s: twelve of those filled the VM and its OOM killer stepped in), and unicode-sweep,
# the one run that needs the sweep, takes 2 jobs.
#
# In the container, from the repo root, after `make oracle` and `make oracle-tokenizer` have built
# the fixtures:
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local \
#       sh tools/mutate_files.sh [olmoe kernels kernels_x86 expf chat unicode unicode-sweep tokenizer main serve prof
#                                 gguf experts threads platform]
# No name: all but unicode-sweep, one after the other. One report per file in
# build/mutate/<name>.txt, its last line on stdout; how far it is, and how long is left, in the last
# line of build/mutate/<name>.progress. Nothing else may load the VM meanwhile.
# After a change: CHANGED=HEAD (or any git ref) mutates only the lines that differ from it, e.g.
#   ... trochilus-dev:local sh -c 'CHANGED=HEAD sh tools/mutate_files.sh main'
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
O='$PY /src/tools/oracle.py /src/fixtures/tiny-olmoe /src/fixtures/tiny-olmoe/model-f32.gguf --binary b/trochilus --expect exact'
OMIN="TR_EXPERT_BUDGET_MIB=min $O"
OV='$PY /src/tools/oracle.py /src/fixtures/tiny-olmoe-opts /src/fixtures/tiny-olmoe-opts/model-f32.gguf --binary b/trochilus --expect exact'
TOKALL='$PY /src/tools/tokenizer_oracle.py --gguf /src/fixtures/olmoe-1b-7b-0125-instruct-tokenizer/vocab.gguf --hf /src/fixtures/olmoe-1b-7b-0125-instruct-tokenizer --binary b/trochilus'
TOK="$TOKALL --no-sweep"
SCALAR='TR_CPU_MAX=scalar ./b/tests/test_tier_used'
MODEL_TESTS="test_prefill test_hot test_spec test_session test_route test_stream test_model_prof test_phase test_tier_used test_model_load"
mkdir -p build/mutate
# the pools of the tests neither pin nor spin: twelve jobs pinned their threads to the same cores
# and spun on them (12 test_cli at once: 8.2-8.9 s pinned, 6.0-7.4 s not). threads.c and
# platform.c, whose code is the pin and the spin, run without this and with fewer jobs.
export TR_POOL_PIN=0 TR_POOL_SPIN_US=0
run() {
    name=$1; jobs=$2; shift 2
    echo "== $name $(date +%T)"
    python3 tools/mutate_auto.py "$@" --jobs "$jobs" ${CHANGED:+--changed "$CHANGED"} \
        > "build/mutate/$name.txt" 2> "build/mutate/$name.progress"
    tail -1 "build/mutate/$name.txt"
}
[ $# -gt 0 ] || set -- olmoe kernels kernels_x86 expf chat unicode tokenizer main serve prof gguf experts threads platform
for f in "$@"; do
    case $f in
    olmoe) run olmoe 6 src/models/olmoe.c $MODEL_TESTS --asan --cmd "$O" --cmd "$OMIN" --cmd "$OV" ;;
    kernels) run kernels 12 src/kernels/kernels.c test_kernels test_expf test_tier_used test_prefill --asan \
                 --cmd "$SCALAR" --cmd "$O" --cmd "$OV" ;;
    kernels_x86) run kernels_x86 12 src/kernels/kernels_x86.c test_kernels test_tier_used --asan --cmd "$SCALAR" ;;
    expf) run expf 12 src/kernels/expf.c test_expf ;;
    chat) run chat 12 src/tokenizer/chat.c test_tokenizer test_cli --asan --cmd "$TOK" ;;
    unicode) run unicode 12 src/tokenizer/unicode.c test_unicode test_tokenizer --asan --cmd "$TOK" ;;
    unicode-sweep) run unicode-sweep 2 src/tokenizer/unicode.c test_unicode test_tokenizer --asan --cmd "$TOKALL" ;;
    tokenizer) run tokenizer 12 src/tokenizer/tokenizer.c test_tokenizer test_cli --asan --cmd "$TOK" ;;
    main) run main 12 src/app/main.c test_cli test_serve --asan --cmd "$O" --cmd "$TOK" ;;
    serve) run serve 12 src/app/serve.c test_serve test_cli --asan ;;
    prof) run prof 12 src/base/prof.c test_prof test_model_prof --asan ;;
    gguf) run gguf 12 src/format/gguf.c test_gguf test_model_load test_tokenizer --asan --cmd "$O" ;;
    experts) run experts 6 src/memory/experts.c test_experts test_stream test_prefetch --asan --cmd "$OMIN" ;;
    # the pool's own files, with the pin and the spin they exist for: fewer jobs, so twelve pools do
    # not pin their threads to the same cores (the export above)
    threads) (unset TR_POOL_PIN TR_POOL_SPIN_US; run threads 4 src/base/threads.c test_base test_hot test_phase --asan) ;;
    platform) (unset TR_POOL_PIN TR_POOL_SPIN_US; run platform 4 src/base/platform.c test_base test_stream test_experts \
                  test_model_load --asan) ;;
    *) echo "mutate_files: unknown file '$f'"; exit 2 ;;
    esac
done
echo "== done $(date +%T)"
}
main "$@"; exit
