#!/bin/sh
# test_cleanup.sh — does a script that is told to stop take its children with it?
# (tools/cleanup.lib, docs/LESSONS.md #84 and #82)
#
#   sh tools/test_cleanup.sh        from the repo root; part of `make check`, a few seconds
#
# Two victims, the same script with and without the trap of tools/cleanup.lib. Each starts a
# native program under a shell under itself, which is what an engine run under ab_modes.sh under
# a measuring script is, and is then sent TERM:
#   with the trap      nothing is left
#   without the trap   the program must STILL be running: that is the orphan of 2026-09-17, and
#                      if this half did not find it the first half would prove nothing. The test
#                      is seen red at every run, not once (then the leftover is removed by force).
# The native program is ping on Windows (always there, and Smart App Control knows it) and sleep
# elsewhere; the count on its command line is how each victim's child is told from anything else.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
if command -v powershell.exe > /dev/null 2>&1; then WINDOWS=1; else WINDOWS=""; fi

if [ "$1" = victim ]; then
  if [ "$2" != trap ]; then
    # the victim without the trap: what every script was before tools/cleanup.lib
    trap - EXIT
    trap - INT
    trap - TERM
  fi
  if [ -n "$WINDOWS" ]; then sh -c "ping -n $3 127.0.0.1 > /dev/null" & else sh -c "sleep $3; :" & fi
  wait $!
  exit 0
fi

# how many programs with this count on their command line are running
alive() {
  if [ -n "$WINDOWS" ]; then
    powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='PING.EXE'\" | Where-Object { \$_.CommandLine -match '-n $1 ' }).Count" | tr -d '\r'
  else
    ps -eo args= | grep -c "^sleep $1\$" || true
  fi
}
remove() {
  if [ -n "$WINDOWS" ]; then
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='PING.EXE'\" | Where-Object { \$_.CommandLine -match '-n $1 ' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null
  else
    pkill -f "^sleep $1\$" || true
  fi
}
# $1: trap | notrap, $2: the count that marks this victim's child
stop_victim() {
  sh tools/test_cleanup.sh victim $1 $2 &
  VICTIM=$!
  TRY=0
  while [ "$(alive $2)" = 0 ]; do
    TRY=$((TRY + 1)); [ "$TRY" -le 20 ] || { echo "test_cleanup: the victim ($1) started no child"; exit 1; }
    sleep 1
  done
  kill $VICTIM
  wait $VICTIM 2> /dev/null || true
  sleep 1
  alive $2
}
FAIL=0
LEFT=$(stop_victim trap 97)
[ "$LEFT" = 0 ] || { echo "test_cleanup: FAILED, with the trap $LEFT child is still running"; remove 97; FAIL=1; }
LEFT=$(stop_victim notrap 98)
[ "$LEFT" != 0 ] || { echo "test_cleanup: FAILED, without the trap nothing was left: this test cannot see an orphan"; FAIL=1; }
remove 98
[ "$(alive 98)" = 0 ] || { echo "test_cleanup: FAILED, could not remove the leftover of the second half"; FAIL=1; }
[ $FAIL = 0 ] || exit 1
echo "== cleanup: a stopped script takes its children with it; without the trap the child stays (seen)"
}
main "$@"; exit
