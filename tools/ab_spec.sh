#!/bin/sh
# ab_spec.sh — what speculation on the prompt is worth, runs alternated (docs/LESSONS.md #46).
#
# The same binary and the same text prompt, once with --spec 0 and once with --spec <draft>,
# round robin so that a machine warming up costs both the same. Round 0 is dropped as warm-up.
# The generated text must be identical (speculation is a speed knob, never a result): the
# script compares it and stops if it is not.
#
#   tools/ab_spec.sh <model.gguf> <binary> <prompt.txt> [draft] [threads] [gen] [rounds]
#
# In the container with the models in the trochilus-models volume:
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/src" -v trochilus-models:/src/models -w /src \
#       trochilus-dev:local sh tools/ab_spec.sh models/<file>.gguf build/linux-gcc/trochilus \
#       bench/prompts/code.txt 8 16 200 5
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
MODEL=$1
BIN=$2
PROMPT=$3
DRAFT=${4:-8}
THREADS=${5:-16}
GEN=${6:-200}
ROUNDS=${7:-5}
if [ -z "$MODEL" ] || [ -z "$BIN" ] || [ -z "$PROMPT" ]; then
  echo "usage: ab_spec.sh <model.gguf> <binary> <prompt.txt> [draft] [threads] [gen] [rounds]" >&2
  exit 2
fi
OUT=$(mktemp)
TXT0=$(mktemp)
TXT1=$(mktemp)
R=0
while [ $R -le "$ROUNDS" ]; do
  for K in 0 "$DRAFT"; do
    for T in $THREADS; do
      if [ "$K" = 0 ]; then TXT=$TXT0; else TXT=$TXT1; fi
      "$BIN" run -m "$MODEL" -f "$PROMPT" -n "$GEN" -t "$T" --spec "$K" 2>"$OUT.err" >"$TXT"
      LINE=$(sed -n "s/^generate: [0-9]* tokens in .* (\([0-9.]*\) tok.s)/spec$K t$T $R \1/p" "$OUT.err")
      # No tok/s line means the run never generated: stop instead of printing medians with a hole
      # in them (docs/LESSONS.md #56).
      if [ -z "$LINE" ]; then
        echo "ab_spec: --spec $K, t$T, round $R produced no tok/s line, stopping. The run said:" >&2
        tail -3 "$OUT.err" >&2
        rm -f "$OUT" "$OUT.err" "$TXT0" "$TXT1"
        exit 1
      fi
      printf '%s\n' "$LINE" | tee -a "$OUT"
      sed -n "s/^speculation: /    spec$K t$T $R accepted: /p" "$OUT.err"
      # identical text, every round: a faster run that changed a token is not a win
      if [ "$K" != 0 ] && ! cmp -s "$TXT0" "$TXT1"; then
        echo "ERROR: --spec $K produced different text from --spec 0" >&2
        rm -f "$OUT" "$OUT.err" "$TXT0" "$TXT1"
        exit 1
      fi
    done
  done
  R=$((R + 1))
done
echo
echo "medians (round 0 dropped as warm-up), tokens identical in every round"
awk '$3 > 0 { key = $1 " " $2; v[key] = v[key] " " $4 }
     END { for (k in v) { n = split(v[k], a, " "); for (i = 1; i < n; i++) for (j = i + 1; j <= n; j++)
             if (a[j] + 0 < a[i] + 0) { t = a[i]; a[i] = a[j]; a[j] = t }
           printf "%-14s median %8.2f  min %8.2f  max %8.2f  (n=%d)\n", k, a[int((n + 1) / 2)], a[1], a[n], n } }' "$OUT" | sort
rm -f "$OUT" "$OUT.err" "$TXT0" "$TXT1"
}
main "$@"; exit
