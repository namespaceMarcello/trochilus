#!/bin/sh
# host_memory.sh — wait until Windows has NEED_GB free, before work that grows the Docker VM.
#
#   sh tools/host_memory.sh NEED_GB [WAIT_MIN]     exit 0 when there is, 1 after WAIT_MIN minutes
#                                                  (default 20) without; 0 at once off Windows
#
# Inside the container the engine's own guard (2 GB or 10% left) sees the VM's 15 GB, not what
# Windows has left: the gate's real-model lanes grow the VM by up to ~10 GB, and with other
# windows' containers and programs using the rest, Windows ran out and Claude Code stopped the
# gate, twice (docs/LESSONS.md #183). So the gate asks Windows first, and waits rather than fail.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
NEED=${1:?usage: host_memory.sh NEED_GB [WAIT_MIN]}
WAIT=${2:-20}
[ "${OS:-}" = Windows_NT ] || exit 0
T0=$(date +%s)
while :; do
  F=$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)" | tr -d '\r')
  [ "$F" -ge $((NEED * 1024)) ] && exit 0
  if [ $(( $(date +%s) - T0 )) -ge $((WAIT * 60)) ]; then
    echo "host_memory: Windows has $F MB free, $NEED GB needed; waited $WAIT min: stopping" >&2
    exit 1
  fi
  echo "host_memory: Windows has $F MB free, $NEED GB needed: waiting ($(date +%H:%M))" >&2
  cleanup_run sleep 30
done
}
main "$@"; exit
