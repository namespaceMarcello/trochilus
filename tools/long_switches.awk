# long_switches.awk — the switches of every run of `decode_context.sh long`. A line of long.txt:
#   run 1 decode 31.20 tok/s choices 1006:4 1030:4(8) 2050:8
# position:width in effect after each measurement of the session, in brackets the pick the
# debounce held back. A switch is a width in effect that differs from the one before it.
{
  n = 0; m = 0; prev = ""
  for (i = 7; i <= NF; i++) {
    if (index($i, ":") == 0) continue # "...": more measurements than the engine keeps
    split($i, c, ":"); w = c[2]; sub(/\(.*/, "", w)
    if (prev != "" && w != prev) n++
    prev = w; m++
  }
  printf "run %s: %d measurements, %d switches, ends on %s\n", $2, m, n, prev
  tot += n
}
END { printf "switches in all: %d over %d runs\n", tot, NR }
