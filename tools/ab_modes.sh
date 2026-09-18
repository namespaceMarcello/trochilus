#!/bin/sh
# ab_modes.sh — two or more ways of running the engine, runs alternated (docs/LEZIONI.md #46).
#
# tools/ab_speed.sh compares two binaries; this compares MODES of one: an environment variable
# (TR_POOL_PIN=1 against 2), a flag (-t 8 against -t 16, --spec 8 against --spec 0). Each mode is
# "label=command line", run through sh, and must print the engine's own speed lines on stderr
# (`generate` and `run` both do). Two things ab_speed.sh does not do:
#   - the first mode of a round rotates (round r starts from mode r mod M): with a fixed order
#     B always runs on the machine A has just warmed, and that does not cancel out;
#   - a mode given twice under two labels is an A/A control: the difference between two
#     identical modes is the floor under which no difference between two real ones means
#     anything (docs/LEZIONI.md #66).
#
#   tools/ab_modes.sh <rounds> "label=command" "label=command" [...]
#
#   sh tools/ab_modes.sh 8 \
#      "pin2=TR_POOL_PIN=2 build/trochilus.exe generate -m models/x.gguf -p 512 -n 24 -t 16" \
#      "pin1=TR_POOL_PIN=1 build/trochilus.exe generate -m models/x.gguf -p 512 -n 24 -t 16"
#
# Round 0 is dropped as warm-up. Prints every run, then per mode and phase: median, min, max,
# spread and the ratio to the first mode. Native runs on a still machine only (LEZIONI #47, #57).
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LEZIONI.md #69).
main() {
ROUNDS=$1
if [ -z "$ROUNDS" ] || [ $# -lt 3 ]; then
  echo "usage: ab_modes.sh <rounds> \"label=command\" \"label=command\" [...]" >&2
  exit 2
fi
shift
M=$#
OUT=$(mktemp)
R=0
while [ "$R" -le "$ROUNDS" ]; do
  I=0
  while [ "$I" -lt "$M" ]; do
    K=$(( (R + I) % M + 1 ))
    eval "MODE=\${$K}"
    LABEL=${MODE%%=*}
    CMD=${MODE#*=}
    ERR=$(sh -c "$CMD" 2>&1 >/dev/null) || true
    # "width": the decode threads the run settled on, where the binary says it (measured per
    # session, so min and max across the rounds say how stable that choice is)
    LINES=$(printf '%s\n' "$ERR" |
      sed -n "s/^prompt: [0-9]* tokens in .* (\([0-9.]*\) tok.s)/$LABEL prefill $R \1/p;
              s/^generate: .* in .* (\([0-9.]*\) tok.s)/$LABEL decode $R \1/p;
              s/^threads: [0-9]* prompt, \([0-9]*\) decode.*/$LABEL width $R \1/p")
    # a run that measured nothing must stop the comparison, not leave a hole in the table (#56)
    if [ -z "$LINES" ]; then
      echo "ab_modes: '$LABEL' round $R produced no tok/s line, stopping. The run said:" >&2
      printf '%s\n' "$ERR" | tail -3 >&2
      rm -f "$OUT"
      exit 1
    fi
    printf '%s\n' "$LINES" | tee -a "$OUT"
    I=$((I + 1))
  done
  R=$((R + 1))
done
echo
echo "medians (round 0 dropped as warm-up; ratio to the first mode, phase by phase)"
FIRST=${1%%=*}
awk -v first="$FIRST" '
  $3 > 0 { key = $2 " " $1; v[key] = v[key] " " $4; if (!(key in seen)) { seen[key] = 1; order[++n_keys] = key } }
  END {
    for (q = 1; q <= n_keys; q++) {
      k = order[q]; n = split(v[k], a, " ")
      for (i = 1; i < n; i++) for (j = i + 1; j <= n; j++) if (a[j] + 0 < a[i] + 0) { t = a[i]; a[i] = a[j]; a[j] = t }
      med[k] = (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2; lo[k] = a[1]; hi[k] = a[n]; cnt[k] = n
    }
    for (q = 1; q <= n_keys; q++) {
      k = order[q]; split(k, part, " "); base = part[1] " " first
      printf "%-8s %-14s median %8.2f  min %8.2f  max %8.2f  spread %5.1f%%  n=%d  vs %s %.3fx\n",
             part[1], part[2], med[k], lo[k], hi[k], (hi[k] - lo[k]) / med[k] * 100, cnt[k], first,
             (base in med) ? med[k] / med[base] : 0
    }
  }' "$OUT" | sort
rm -f "$OUT"
}
main "$@"; exit
