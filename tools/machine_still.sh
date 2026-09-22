#!/bin/sh
# machine_still.sh — is the machine still? A speed measured beside somebody else's work is a
# number about two programs (docs/LESSONS.md #84: four forgotten `yes` processes at 100% for 37
# hours under two days of "still machine" measurements; #73: the work of another window shows in
# no container). Counts the logical processors busy over a window (tools/cpu_busy.ps1 on
# Windows, /proc/stat elsewhere) and waits until they are at most the limit.
#
#   sh tools/machine_still.sh [limit, default 3.0] [seconds to wait, default 600] [window, default 10]
#
# Exit 0 when the machine is still; exit 1 after the wait, with the processes that are using the
# CPU. The measuring scripts call it before the first run (tools/measure_guard.lib,
# measure_still) and, through AB_GUARD of tools/ab_modes.sh, before every run: a machine that
# gets busy half way stops the comparison instead of ending up inside a median.
# The limit is this machine's: with the containers stopped it idles at 1.8 logical processors
# busy over 10 s (the kernel's System process alone holds 0.8, then browser, Defender, the
# terminals), between 0.8 and 3.3 over 2 s; a process running flat out is 1.0 more, a build 16.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
LIMIT=${1:-3.0}
WAIT=${2:-600}
WINDOW=${3:-10}
busy() {
  if command -v powershell.exe > /dev/null 2>&1; then
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/cpu_busy.ps1 $WINDOW $1 | tr -d '\r'
  elif [ -r /proc/stat ]; then
    A=$(sed -n 's/^cpu  *//p' /proc/stat); sleep $WINDOW; B=$(sed -n 's/^cpu  *//p' /proc/stat)
    echo "$A $B" | awk -v n="$(nproc)" '{ for (i = 1; i <= 10; i++) { t0 += $i; t1 += $(i + 10) }
      idle = ($14 + $15) - ($4 + $5); printf "%.2f\n", n * (1 - idle / (t1 - t0)) }'
  else
    echo 0
  fi
}
T=0
while :; do
  BUSY=$(busy | head -1)
  if awk -v b="$BUSY" -v l="$LIMIT" 'BEGIN { exit !(b != "" && b + 0 <= l + 0) }'; then exit 0; fi
  [ "$T" -lt "$WAIT" ] || break
  echo "machine_still: $BUSY logical processors busy (limit $LIMIT), waiting" >&2
  sleep 8
  T=$((T + WINDOW + 9))
done
echo "machine_still: $BUSY logical processors busy after $WAIT s (limit $LIMIT): the machine is not still" >&2
busy -Top | tail -n +2 >&2
exit 1
}
main "$@"; exit
