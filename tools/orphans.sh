#!/bin/sh
# orphans.sh — is anything this project started still running? (docs/LEZIONI.md #84)
#
# On 2026-09-17 a busy-machine test left four `yes` processes at 100%: they ran for 37 hours,
# under every native measurement of two days, and corrupted them all. Scripts now end what they
# start (tools/cleanup.lib); this is the other half: `make check` and every measurement refuse to
# start while something is left, and say what.
#
#   sh tools/orphans.sh       lists what is left, exit 1 if anything is
#
# What counts: any `yes`, any program running out of build/ (an engine, a benchmark, a test), a
# measuring script of tools/, a container of the image trochilus-dev; except the processes above
# the caller. A measurement that is really running is not an orphan, and is a reason not to
# start all the same: a build beside it spoils it, and two measurements spoil each other
# (docs/LEZIONI.md #57, #82).
# The body is one function, called on the last line (docs/LEZIONI.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
FOUND=""
if command -v powershell.exe > /dev/null 2>&1; then
  REPO=$(cygpath -w "$(pwd)" 2> /dev/null || pwd)
  # the callers of this script, as Windows pids: a measuring script that asks is not an orphan.
  # Windows cannot tell: an MSYS exec leaves the new program with a dead creator as its parent,
  # and the chain of parents breaks there. MSYS knows its own (/proc/<pid>/ppid and winpid).
  MINE=""
  P=$$
  while [ -n "$P" ] && [ "$P" != 1 ] && [ -r /proc/$P/winpid ]; do
    MINE="$MINE,$(cat /proc/$P/winpid)"
    P=$(cat /proc/$P/ppid 2> /dev/null || true)
  done
  FOUND=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/orphans.ps1 "$REPO" "${MINE#,}" | tr -d '\r')
else
  # everything outside this script's own session (the caller's tree shares it)
  SID=$(ps -o sid= -p $$ | tr -d ' ')
  FOUND=$(ps -eo pid=,sid=,lstart=,args= | awk -v sid="$SID" -v build="$(pwd)/build/" '
    $2 == sid { next }
    { args = ""; for (i = 8; i <= NF; i++) args = args " " $i
      what = ""
      if ($8 == "yes" || $8 ~ /\/yes$/) what = "a load generator"
      else if (index($8, build) == 1) what = "a program of build/"
      else if (args ~ /tools\/(prefill_context|decode_context|threads_phase|remeasure|ab_modes|ab_speed|ab_spec|platform_bits|busy_machine)\.sh|tools\/profile_suite\.py/) what = "a measuring script"
      if (what != "") print what " pid " $1 ":" args }')
fi
# a plain listing, filtered here: the same call the guard makes before every run. With a filter
# on the image the engine has to be asked, and on 2026-09-19 the Docker VM came up again three
# seconds after that call, bringing back by itself every container with a restart policy
# (docs/LEZIONI.md #87).
CONTAINERS=$(docker ps --format '{{.Image}} {{.Names}}, {{.Status}}' 2> /dev/null | sed -n 's/^trochilus-dev[^ ]* /a container of trochilus-dev: /p' || true)
[ -z "$CONTAINERS" ] || FOUND=$(printf '%s\n%s' "$FOUND" "$CONTAINERS" | sed '/^$/d')
[ -n "$FOUND" ] || exit 0
echo "orphans: something this project started is still running (docs/LEZIONI.md #84):" >&2
echo "$FOUND" | sed 's/^/  /' >&2
echo "orphans: end it (a measurement: kill the pid in build/.measuring.lock/pid), then start again" >&2
exit 1
}
main "$@"; exit
