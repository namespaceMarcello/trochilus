#!/bin/sh
# tier_check.sh <build dir> <tiny fixture dir> — every kernel tier end to end, not only the best
# one this CPU has. tests/test_kernels.c compares the kernels of each tier with scalar; this
# runs the ENGINE under each tier (TR_CPU_MAX) and compares what comes out:
#   1. the model tests (prefill, speculation, session, phase) pass under scalar and avx2 too, and
#      under each of them the engine really goes through the active tier (test_tier_used);
#   2. the logits of the tiny fixtures (f32, f16, q8_0) are the same bytes for every tier,
#      thread count and -b.
# A cap that did not take effect (a typo in TR_CPU_MAX is ignored by design) would turn all of
# this into the best tier tested three times: the first check is that the cap really capped
# (docs/LESSONS.md #64).
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
B=$1
FIX=$2
OUT=$B/tier-check
mkdir -p "$OUT"

for tier in scalar avx2; do
    kernels=$(TR_CPU_MAX=$tier "$B/tests/test_kernels")
    echo "$kernels" | grep -q "tier avx512 *not available" ||
        { echo "tier-check: TR_CPU_MAX=$tier did not cap the tier"; exit 1; }
    if [ "$tier" = scalar ]; then
        echo "$kernels" | grep -q "tier avx2 *not available" ||
            { echo "tier-check: TR_CPU_MAX=scalar did not cap the tier"; exit 1; }
    fi
    for t in test_prefill test_spec test_session test_phase test_tier_used; do
        TR_CPU_MAX=$tier "$B/tests/$t" > "$OUT/$t-$tier.log" 2>&1 ||
            { echo "tier-check: $t fails under TR_CPU_MAX=$tier"; tail -5 "$OUT/$t-$tier.log"; exit 1; }
    done
done

# 40 tokens below the fixture's vocabulary (128) and context (128)
TOKENS=$(i=0; s=""; while [ $i -lt 40 ]; do s="$s$(( (i * 37 + 11) % 128 )),"; i=$((i + 1)); done; echo "${s%,}")
for type in f32 f16 q8_0; do
    model=$FIX/model-$type.gguf
    [ -f "$model" ] || { echo "tier-check: $model is missing (make oracle builds it)"; exit 1; }
    for b in 1 7 64; do
        first=""
        for tier in scalar avx2 best; do
            # 5 threads measure their own decode width (5, 2, 1 in turn, then one of them); the
            # last mode forces it: the short passes must give the same bytes on any of them
            for t in "1" "5" "5 --decode-threads 2"; do
                f=$OUT/$type-b$b-$tier-t$(echo "$t" | tr -d ' -').bin
                if [ "$tier" = best ]; then
                    "$B/trochilus" logits -m "$model" --tokens "$TOKENS" --out "$f" -t $t -b $b > /dev/null 2>&1
                else
                    TR_CPU_MAX=$tier "$B/trochilus" logits -m "$model" --tokens "$TOKENS" --out "$f" -t $t -b $b > /dev/null 2>&1
                fi
                [ -s "$f" ] || { echo "tier-check: no logits from $model ($tier, -t $t, -b $b)"; exit 1; }
                if [ -z "$first" ]; then first=$f; else
                    cmp -s "$first" "$f" || { echo "tier-check: $f differs from $first"; exit 1; }
                fi
            done
        done
    done
    # across -b only the row of the last position is in every file: 128 floats
    for b in 7 64; do
        tail -c 512 "$OUT/$type-b1-scalar-t1.bin" > "$OUT/last-1.bin"
        tail -c 512 "$OUT/$type-b$b-scalar-t1.bin" > "$OUT/last-b.bin"
        cmp -s "$OUT/last-1.bin" "$OUT/last-b.bin" ||
            { echo "tier-check: $type, last row with -b $b differs from -b 1"; exit 1; }
    done
done
echo "== tier-check: model tests pass under scalar and avx2; logits identical across tiers, threads and -b (f32, f16, q8_0)"
}
main "$@"; exit
