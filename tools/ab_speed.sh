#!/bin/sh
# ab_speed.sh — two binaries on the same prompt, runs alternated (docs/LEZIONI.md #46).
#
# A sweep that measures all of A and then all of B is not a comparison: this laptop loses
# 10-25% as it heats up, which is more than most changes are worth. Here the runs go round
# robin A, B, A, B ..., so a drift costs both the same, and round 0 is dropped as warm-up.
#
#   tools/ab_speed.sh <model.gguf> <binary A> <binary B> [threads] [prompt] [gen] [rounds]
#
# Prints one line per run (binary, threads, phase, round, tok/s) and the medians per cell.
# In the container with the models in the trochilus-models volume:
#   docker run --rm -v "$PWD:/src" -v trochilus-models:/src/models -w /src trochilus-dev:local \
#       sh tools/ab_speed.sh models/<file>.gguf build/base/b/trochilus build/linux-gcc/trochilus
set -e
MODEL=$1
A=$2
B=$3
THREADS=${4:-"16 8 4 1"}
PROMPT=${5:-512}
GEN=${6:-16}
ROUNDS=${7:-5}
if [ -z "$MODEL" ] || [ -z "$A" ] || [ -z "$B" ]; then
  echo "usage: ab_speed.sh <model.gguf> <binary A> <binary B> [threads] [prompt] [gen] [rounds]" >&2
  exit 2
fi
OUT=$(mktemp)
R=0
while [ $R -le "$ROUNDS" ]; do
  for WHO in A B; do
    if [ "$WHO" = A ]; then BIN=$A; else BIN=$B; fi
    for T in $THREADS; do
      $BIN generate -m "$MODEL" -p "$PROMPT" -n "$GEN" -c $((PROMPT + GEN + 64)) -t "$T" 2>&1 >/dev/null |
        sed -n "s/^prompt: $PROMPT tokens in .* (\([0-9.]*\) tok.s)/$WHO t$T prefill $R \1/p;
                s/^generate: .* evaluations in .* (\([0-9.]*\) tok.s)/$WHO t$T decode $R \1/p" | tee -a "$OUT"
    done
  done
  R=$((R + 1))
done
echo
echo "medians (round 0 dropped as warm-up)"
awk '$4 > 0 { key = $1 " " $2 " " $3; v[key] = v[key] " " $5 }
     END { for (k in v) { n = split(v[k], a, " "); for (i = 1; i < n; i++) for (j = i + 1; j <= n; j++)
             if (a[j] + 0 < a[i] + 0) { t = a[i]; a[i] = a[j]; a[j] = t }
           printf "%-16s median %8.2f  min %8.2f  max %8.2f  (n=%d)\n", k, a[int((n + 1) / 2)], a[1], a[n], n } }' "$OUT" | sort
rm -f "$OUT"
