#!/bin/sh
# proof.sh — the head's bound proved (docs/MEASUREMENTS.md §The head's bound proved; make proof): every CBMC harness in
# tests/proof/ on the real src/kernels/head_bound.c, then every case of the per-element inequalities (proof_enum.c).
# Needs cbmc: the image trochilus-proof:local (tools/docker/Dockerfile.proof), from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-proof:local make proof
#
# The lemmas two at a time, the argmax harnesses one at a time (several GB each), each stopped at 30 minutes (a lemma
# that needs more is split, not waited for); about 15 minutes in all. One line a harness: proved, or FAILED with the tail of CBMC's trace. Exit 1 if any is not proved. OUT=<dir> keeps the logs
# (default build/proof); ONLY=<text>: only the harnesses whose name holds it (the mutations, tools/mutate_head.sh).
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
  . tools/cleanup.lib
  trap cleanup_children EXIT
  trap 'exit 130' INT TERM
  OUT=${OUT:-build/proof}
  # file/harness/unwind[/unwindset]: the unwind covers every loop (--unwinding-assertions fails one it cuts short);
  # the argmax's loops run at most 7 times but the prep's over h's 256 and 32 elements, so those get their own
  AP="tr_hb_prep_build.0:257,tr_hb_prep_build.1:257,tr_hb_prep_build.2:257,tr_hb_prep_build.3:257"
  RUNS="proof_q8_0/planes_q8_0/70 proof_q8_0/bound_q8_0/40 proof_q4_k/planes_q4_k/300 proof_q4_k/bound_q4_k/300
    proof_prep/digits/4 proof_prep/next_float/4 proof_prep/up/4 proof_prep/margin/4
    proof_argmax/argmax_q8_0/9/$AP proof_argmax/argmax_q8_0_pool/9/$AP proof_argmax/argmax_q8_0_steal/9/$AP
    proof_argmax/argmax_q4_k_pool/9/$AP"
  if [ "$1" = one ]; then
    f=${2%%/*}; rest=${2#*/}; fn=${rest%%/*}; rest=${rest#*/}; unwind=${rest%%/*}; set=""
    [ "$rest" = "$unwind" ] || set="--unwindset ${rest#*/}"
    start=$(date +%s)
    # shellcheck disable=SC2086
    if timeout 1800 cbmc "tests/proof/$f.c" --function "$fn" --bounds-check --pointer-check --signed-overflow-check \
        --conversion-check --unwinding-assertions --object-bits 12 --unwind "$unwind" $set > "$OUT/$fn.log" 2>&1 &&
        grep -q "VERIFICATION SUCCESSFUL" "$OUT/$fn.log" && ! grep -q "no body for function" "$OUT/$fn.log"; then
      # (a function without a body returns anything: an assumption on it would be void, so it fails the proof)
      echo "proof: $fn proved ($(( $(date +%s) - start )) s)"
    else
      echo "proof: $fn FAILED ($(( $(date +%s) - start )) s)"
      grep -E "FAILURE|rror|no body" "$OUT/$fn.log" | head -8
      echo FAILED > "$OUT/$fn.failed"
    fi
    return 0
  fi
  command -v cbmc > /dev/null || { echo "proof.sh: no cbmc here (the trochilus-proof image, docs/COMMANDS.md)"; exit 1; }
  mkdir -p "$OUT"
  rm -f "$OUT"/*.failed
  start=$(date +%s)
  # the lemmas two at a time, then the argmax harnesses one at a time: each takes several GB, two at once ran the
  # Docker VM (15 GB) out of memory
  for r in $RUNS; do
    case "$r" in *argmax*) ;; *"${ONLY:-}"*) echo "$r" ;; esac
  done | xargs -P 2 -I{} sh tools/proof.sh one {}
  for r in $RUNS; do
    case "$r" in *argmax*"${ONLY:-}"* | *"${ONLY:-}"*argmax*) echo "$r" ;; esac
  done | xargs -P 1 -I{} sh tools/proof.sh one {}
  ok=1
  ls "$OUT"/*.failed > /dev/null 2>&1 && ok=0
  if [ -z "${ONLY:-}" ] || [ "$ONLY" = enum ]; then
    cc -O2 -std=c11 -o "$OUT/proof_enum" tests/proof/proof_enum.c && "$OUT/proof_enum" || ok=0
  fi
  echo "proof: $(( $(date +%s) - start )) s"
  [ $ok = 1 ] || { echo "proof: NOT PROVED"; exit 1; }
  # the whole proof passed: the bytes it covers, which tools/lint.py compares at every gate (#350)
  if [ -z "${ONLY:-}" ]; then
    sha256sum src/kernels/head_bound.c $(ls tests/proof/*.c tests/proof/*.h | LC_ALL=C sort) > tests/proof/proved.sha256
    echo "proof: proved; hashes in tests/proof/proved.sha256"
  fi
}
main "$@"; exit
