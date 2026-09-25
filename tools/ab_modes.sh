#!/bin/sh
# ab_modes.sh — two or more ways of running the engine, runs alternated (docs/LESSONS.md #46).
#
# tools/ab_speed.sh compares two binaries; this compares MODES of one: an environment variable
# (TR_POOL_PIN=1 against 2), a flag (-t 8 against -t 16, --spec 8 against --spec 0). Each mode is
# "label=command line", run through sh, and must print the engine's own speed lines on stderr
# (`generate` and `run` both do). Two things ab_speed.sh does not do:
#   - the first mode of a round rotates (round r starts from mode r mod M): with a fixed order
#     B always runs on the machine A has just warmed, and that does not cancel out;
#   - a mode given twice under two labels is an A/A control: the difference between two
#     identical modes is the floor under which no difference between two real ones means
#     anything (docs/LESSONS.md #66).
#
#   tools/ab_modes.sh <rounds> "label=command" "label=command" [...]
#
#   sh tools/ab_modes.sh 8 \
#      "pin2=TR_POOL_PIN=2 build/trochilus.exe generate -m models/x.gguf -p 512 -n 24 -t 16" \
#      "pin1=TR_POOL_PIN=1 build/trochilus.exe generate -m models/x.gguf -p 512 -n 24 -t 16"
#
# Round 0 is dropped as warm-up. Prints every run, then per mode and phase: median, min, max,
# spread and the ratio to the first mode. Native runs on a still machine only (LESSONS #47, #57).
#
# AB_GUARD, when set, is a shell command run before every run: if it fails the machine is no
# longer the one the comparison started on, and the comparison stops there instead of mixing
# runs of two machines (docs/LESSONS.md #73: another session started its containers again
# seven minutes into a measurement). Example: AB_GUARD='sh tools/machine_still.sh 3.5 600 2';
# the measuring scripts set it with measure_machine (tools/measure_guard.lib).
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
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
    if [ -n "$AB_GUARD" ] && ! cleanup_run sh -c "$AB_GUARD" > /dev/null 2>&1; then
      echo "ab_modes: the guard '$AB_GUARD' failed before '$LABEL' round $R: the machine changed, stopping." >&2
      rm -f "$OUT"
      exit 3
    fi
    # AB_WALL=1: the whole run in ms, process start and model load included (the first prompt of a
    # session is what a user waits for, docs/MEASUREMENTS.md question 49); lower is better. Only
    # then the clock is read (GNU date's %N), and the wall line is added after the check below: a
    # failed run lasts some ms too, and must still stop the comparison (docs/LESSONS.md #132)
    [ "${AB_WALL:-0}" != 1 ] || T0=$(date +%s%N)
    RC=0
    ERR=$(sh -c "$CMD" 2>&1 >/dev/null) || RC=$?
    WALL=
    [ "${AB_WALL:-0}" != 1 ] || WALL="$LABEL wall_ms $R $(( ($(date +%s%N) - T0) / 1000000 ))"
    # "width": the decode threads the run settled on, where the binary says it (measured per
    # session, so min and max across the rounds say how stable that choice is). "hits", "misses"
    # and "mib": the expert store's own counters, from the `experts:` line of a streaming run
    # (nothing matches when the store is resident, and those rows are simply absent).
    LINES=$(printf '%s\n' "$ERR" |
      sed -n "s/^prompt: [0-9]* tokens in .* (\([0-9.]*\) tok.s)/$LABEL prefill $R \1/p;
              s/^generate: .* in .* (\([0-9.]*\) tok.s)/$LABEL decode $R \1/p;
              s/^threads: [0-9]* prompt, \([0-9]*\) decode.*/$LABEL width $R \1/p")
    # the store's three counters come from one line, so awk (a second s/// would work on what the
    # first one already rewrote); "MiB)," of the resident form is not the field "MiB"
    EXPERTS=$(printf '%s\n' "$ERR" | awk -v l="$LABEL" -v r="$R" '
      /^experts: / { for (i = 2; i <= NF; i++) {
          if ($i == "hits,")   print l, "hits", r, $(i - 1)
          if ($i == "misses,") print l, "misses", r, $(i - 1)
          if ($i == "MiB")     print l, "mib", r, $(i - 1)
      } }')
    [ -z "$EXPERTS" ] || LINES="$LINES
$EXPERTS"
    # a run that measured nothing must stop the comparison, not leave a hole in the table (#56);
    # nor one that failed after its prompt line (a crash in decode, a server gone mid-command)
    if [ -z "$LINES" ] || [ "$RC" != 0 ]; then
      echo "ab_modes: '$LABEL' round $R exited $RC or produced no tok/s line, stopping. The run said:" >&2
      printf '%s\n' "$ERR" | tail -3 >&2
      rm -f "$OUT"
      exit 1
    fi
    [ -z "$WALL" ] || LINES="$LINES
$WALL"
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
