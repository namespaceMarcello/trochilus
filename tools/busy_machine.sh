#!/bin/sh
# busy_machine.sh — a command run beside N processes that each burn a core, and the N processes
# ended when it is over: the ONLY way this project loads the machine on purpose.
#
#   sh tools/busy_machine.sh <n> <command...>
#   sh tools/busy_machine.sh 4 sh tools/ab_modes.sh 3 "pin2=..." "pin1=..."
#
# On 2026-09-17 the same test was four `yes > /dev/null &` typed by hand: the shell that started
# them went away, they stayed, and ran at 100% for 37 hours under two days of measurements
# (docs/LESSONS.md #84). Here they are children of this script, killed from its EXIT trap
# whatever ends it (tools/cleanup.lib), and the script fails if one is left.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
N=$1
case "$N" in ''|*[!0-9]*) echo "usage: busy_machine.sh <n> <command...>" >&2; exit 2 ;; esac
[ $# -ge 2 ] && [ "$N" -ge 1 ] && [ "$N" -le 16 ] || { echo "usage: busy_machine.sh <n, 1..16> <command...>" >&2; exit 2; }
shift
ended() {
  cleanup_children
  if ! sh tools/orphans.sh; then echo "busy_machine: a load generator is still running" >&2; exit 1; fi
}
trap ended EXIT
trap 'exit 130' INT TERM
I=0
while [ "$I" -lt "$N" ]; do
  yes > /dev/null &
  I=$((I + 1))
done
echo "busy_machine: $N load generators running beside: $*" >&2
RC=0
cleanup_run "$@" || RC=$?
exit $RC
}
main "$@"; exit
