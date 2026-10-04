#!/bin/sh
# machines.sh -- R1's machines (docs/ARCHITECTURE.md §The roadmap): the same prompt and generation on the
# below-average machine (weak: 4 threads, AVX2, 8 GB), the average one (avg: 8 threads, 16 GB) and the average one
# with a small dedicated GPU (avgd: the same, the GPU on), emulated on this PC, and on this PC as it is; each one's
# decode set against its theoretical limit, its RAM's bandwidth over the bytes a token reads (plus, under a
# budget, the disk's bytes over the disk's speed). weak and avg have an integrated GPU the engine does not use yet
# (R1 block r1-igpu): emulated with TR_GPU=0, and its numbers stay estimates until a real PC (question 88).
#
#   sh tools/machines.sh <binary> <rounds> [q4k q8]
#   MACHINES_LIST="weak aa" MACHINES_N=16 MACHINES_OUT=build/machines_weak sh tools/machines.sh <binary> <rounds> q8
#
# The emulation: -t, TR_CPU_MAX=avx2, TR_GPU=0, the RAM the engine's own plan sees (TR_MEM_TOTAL_MIB and
# TR_MEM_AVAILABLE_MIB: what Windows 11 leaves free at rest, 4.5 GiB of 8 and 11 GiB of 16), so the engine picks
# its expert budget as it would there, and the machine's disk (TR_EXPERT_DISK_MBPS: the expert store's reads at
# that rate, one disk; an estimate until a real PC, docs/MEASUREMENTS.md question 88). Not emulated: a slower RAM
# (the limit takes its bandwidth, the runs give what the cores do with this PC's).
# MACHINES_LIST (default "weak avg avgd pc aa"): the arms, in order; aa is the machine before it run again (the
# A/A). MACHINES_N (default 64): the tokens generated, fewer where a slow disk would pass 60 s a run.
# MACHINES_ARGS: more options for every run (-c 4096: a chat's plan, the default context).
# MACHINES_BIN_<machine>=<binary>: that arm runs another binary (a race of before against after; the same
# tokens are still required of every run). A binary Smart App Control holds is replaced by a copy a byte longer
# (measure_runnable). MACHINES_STOP (default 3, 0: off): the race stops once every arm's times spread at most that
# many % (tools/ab_modes.sh AB_STOP). MACHINES_COUNTS=1: counts, not times: one profiled run an arm, the store's
# slots, hits, misses and bytes, no marker and no still machine (a choice those decide needs no race).
# Per model: one profiled run a machine (the bytes a token reads: weights, KV, disk), then tools/ab_modes.sh over
# the machines and an A/A (this PC twice), rounds rotating, AB_WALL=1, under the native guards
# (tools/measure_guard.lib); every run's tokens compared: one text on every machine (no kernel and no thread
# count changes a bit). Then the table, machines.txt. MACHINES_OUT (default build/machines) takes one folder a
# model; a folder that already holds runs is refused. Round 0 is a warm-up. ~15 min a model at 8 rounds.
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
if [ "${1:-}" = table ]; then
  # sh tools/machines.sh table <out> [q4k q8]: the table again, from the runs a folder holds
  shift
  OUT=${1:?usage: sh tools/machines.sh table <out> [q4k q8]}
  shift
  machines_rows "$OUT" "${*:-q4k q8}" | machines_table
  return 0
fi
B=${1:-}
R=${2:-}
[ -n "$B" ] && [ -n "$R" ] || { echo "usage: sh tools/machines.sh <binary> <rounds> [q4k q8]"; exit 2; }
shift 2
MODELS=${*:-q4k q8}
[ -f "$B" ] || { echo "machines: $B is missing"; exit 1; }
OUT=${MACHINES_OUT:-build/machines}
LIST=${MACHINES_LIST:-weak avg avgd pc aa}
N=${MACHINES_N:-64}
for L in $LIST; do
  [ "$L" = aa ] || printf '%s\n' "$MACHINES" | grep -q "^${L}|" || { echo "machines: unknown machine $L"; exit 2; }
done
case " $LIST " in " aa "*) echo "machines: aa repeats the machine before it, and none is"; exit 2 ;; esac
for M in $MODELS; do
  case $M in
    q4k) MF=models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf ;;
    q8)  MF=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf ;;
    *)   echo "machines: unknown model $M (q4k, q8)"; exit 2 ;;
  esac
  [ -f "$MF" ] || { echo "machines: $MF is missing"; exit 1; }
  if [ -n "$(find "$OUT/$M" -name '*.out' 2> /dev/null | head -1)" ]; then
    echo "machines: $OUT/$M already holds runs: move them, or give another MACHINES_OUT"
    exit 1
  fi
done
. tools/measure_guard.lib
# a binary Smart App Control holds: a copy a byte longer, which it judges apart (measure_runnable)
B=$(measure_runnable "$B") || { echo "machines: $B is held by Smart App Control, its longer copies too: build it in a new folder (FRESH=1)"; exit 1; }
for V in $(env | sed -n 's/^\(MACHINES_BIN_[A-Za-z0-9_]*\)=.*/\1/p'); do
  eval "VB=\$$V"
  VB=$(measure_runnable "$VB") || { echo "machines: $VB is held by Smart App Control"; exit 1; }
  eval "$V=\$VB"
done
if [ "${MACHINES_COUNTS:-0}" = 1 ]; then
  machines_counts
  return 0
fi
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin machines
mkdir -p "$OUT"
sha256sum "$B" > "$OUT/binary.sha256" 2> /dev/null || true
# the arms' own binaries (MACHINES_BIN_<machine>)
for V in $(env | sed -n 's/^\(MACHINES_BIN_[A-Za-z0-9_]*\)=.*/\1/p'); do
  eval "sha256sum \"\$$V\"" >> "$OUT/binary.sha256" 2> /dev/null || true
done
measure_machine machines
# this PC loads the Q8_0 whole (7 GiB) and the memory guard wants 3 more free (docs/LESSONS.md #72)
TRY=0
while :; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "machines: only ${AVAIL:-?} GiB available after 15 minutes"; exit 1; }
  echo "machines: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done
measure_still machines
measure_declare "before the first run" > "$OUT/load.txt"
cat "$OUT/load.txt"
export AB_WALL=1
# the race stops once every arm is steady (tools/ab_modes.sh AB_STOP): <rounds> is then the most
[ "${MACHINES_STOP:-3}" = 0 ] || export AB_STOP=${MACHINES_STOP:-3}
for M in $MODELS; do
  case $M in
    q4k) MF=models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf ;;
    q8)  MF=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf ;;
  esac
  D=$OUT/$M
  mkdir -p "$D"
  IDS=$($B tokenize -m $MF -f bench/prompts/code.txt)
  RUN="generate -m $MF --tokens $IDS -n $N${MACHINES_ARGS:+ $MACHINES_ARGS}"
  echo "== $M: $B generate -m $MF --tokens <bench/prompts/code.txt, 321> -n $N ($LIST)" | tee "$D/what.txt"
  # the bytes a token reads, one profiled run a machine (the profiler's timers stay out of the timed rounds)
  for LABEL in $LIST; do
    [ "$LABEL" != aa ] || continue
    LINE=$(printf '%s\n' "$MACHINES" | grep "^${LABEL}|")
    ENV=$(printf '%s' "$LINE" | cut -d'|' -f2)
    THR=$(printf '%s' "$LINE" | cut -d'|' -f3)
    eval "BL=\${MACHINES_BIN_$LABEL:-$B}"
    measure_still "machines $M $LABEL profile"
    eval "cleanup_run env TR_BAR=0 $ENV $BL $RUN $THR --profile" > "$D/$LABEL-profile.out" 2> "$D/$LABEL.prof"
  done
  MODES=""
  PREV=""
  for LABEL in $LIST; do
    # aa: the machine before it in the list, run again
    if [ "$LABEL" = aa ]; then
      LINE=$(printf '%s\n' "$MACHINES" | grep "^${PREV}|")
      eval "BL=\${MACHINES_BIN_$PREV:-$B}"
    else
      LINE=$(printf '%s\n' "$MACHINES" | grep "^${LABEL}|")
      PREV=$LABEL
      eval "BL=\${MACHINES_BIN_$LABEL:-$B}"
    fi
    ENV=$(printf '%s' "$LINE" | cut -d'|' -f2)
    THR=$(printf '%s' "$LINE" | cut -d'|' -f3)
    # stdout (the generated tokens) to a file of its own, stderr (the speed lines) to ab_modes
    MODES="$MODES \"$LABEL=$ENV $BL $RUN $THR > $D/$LABEL-\\\$(date +%s%N).out\""
  done
  measure_still "machines $M"
  eval "cleanup_run sh tools/ab_modes.sh $R $MODES" > "$D/ab_modes.txt"
  grep 'median' "$D/ab_modes.txt" || true
  FIRST=$(ls "$D"/*.out | head -1)
  SAME=yes
  for F in "$D"/*.out; do cmp -s "$FIRST" "$F" || { SAME=no; echo "machines: $F differs from $FIRST"; }; done
  echo "$M: the same tokens in all $(ls "$D"/*.out | wc -l) runs, on every machine: $SAME" | tee "$D/same.txt"
  [ "$SAME" = yes ] || exit 4
done
measure_declare "after the last run" >> "$OUT/load.txt"
tail -1 "$OUT/load.txt"
machines_rows "$OUT" "$MODELS" | machines_table > "$OUT/machines.txt"
cat "$OUT/machines.txt"
echo "done: $OUT"
}

# label, environment, threads, the RAM's nominal bandwidth in GB/s (docs/ARCHITECTURE.md §The roadmap), and its
# practical ceiling: measured on this PC (build/tests/bench_mem.exe ram, 2026-10-03: 56.3 GB/s sequential at 8
# threads, 68% of the nominal 83), an estimate elsewhere (the nominal at this PC's 68%, until a real PC). The disk
# each machine reads its experts from is in its environment (TR_EXPERT_DISK_MBPS, estimates: an 8 GB laptop's
# SATA-class SSD 500 MB/s, an average one's NVMe 2000); without it, this PC's (3.5 GB/s at 64 MiB requests,
# docs/MEASUREMENTS.md §The disk at its limit).
# weakr: the weak machine's cores with every expert in RAM (no TR_MEM_*): its compute alone, the deletion
# series' first piece (R1 phase 3, docs/MEASUREMENTS.md §The three machines). weakhot: weak with ds4's eviction
# (TR_EXPERT_EVICT=hot); weakold and weakrold: weak and weakr again, for a binary of before (MACHINES_BIN_weakold,
# MACHINES_BIN_weakrold).
MACHINES="weak|TR_GPU=0 TR_CPU_MAX=avx2 TR_MEM_TOTAL_MIB=8192 TR_MEM_AVAILABLE_MIB=4608 TR_EXPERT_DISK_MBPS=500|-t 4|21|14.3
weakr|TR_GPU=0 TR_CPU_MAX=avx2|-t 4|21|14.3
weakhot|TR_GPU=0 TR_CPU_MAX=avx2 TR_MEM_TOTAL_MIB=8192 TR_MEM_AVAILABLE_MIB=4608 TR_EXPERT_DISK_MBPS=500 TR_EXPERT_EVICT=hot|-t 4|21|14.3
weakold|TR_GPU=0 TR_CPU_MAX=avx2 TR_MEM_TOTAL_MIB=8192 TR_MEM_AVAILABLE_MIB=4608 TR_EXPERT_DISK_MBPS=500|-t 4|21|14.3
weakrold|TR_GPU=0 TR_CPU_MAX=avx2|-t 4|21|14.3
avg|TR_GPU=0 TR_MEM_TOTAL_MIB=16384 TR_MEM_AVAILABLE_MIB=11264 TR_EXPERT_DISK_MBPS=2000|-t 8|51|34.6
avgd|TR_MEM_TOTAL_MIB=16384 TR_MEM_AVAILABLE_MIB=11264 TR_EXPERT_DISK_MBPS=2000|-t 8|51|34.6
pc|||83|56.3"

# MACHINES_COUNTS=1: counts, not times. One profiled run an arm, no marker and no still machine: the expert
# store's slots, hits, misses and bytes are exact counts, the same on a busy machine (every race of 2026-10-03:
# 0.0% spread), so a choice they decide needs no race; the race comes after, to confirm the time.
machines_counts() {
  . tools/cleanup.lib
  trap cleanup_children EXIT
  trap 'exit 130' INT TERM
  mkdir -p "$OUT"
  for M in $MODELS; do
    case $M in
      q4k) MF=models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf ;;
      q8)  MF=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf ;;
    esac
    D=$OUT/$M
    mkdir -p "$D"
    IDS=$($B tokenize -m $MF -f bench/prompts/code.txt)
    RUN="generate -m $MF --tokens $IDS -n $N${MACHINES_ARGS:+ $MACHINES_ARGS}"
    echo "== $M counts: generate --tokens <bench/prompts/code.txt, 321> -n $N"
    printf '%-10s %-28s %8s %8s %8s %10s %10s\n' arm "experts in RAM" hits misses dropped "MiB read" "disk/tok"
    for LABEL in $LIST; do
      [ "$LABEL" != aa ] || continue
      LINE=$(printf '%s\n' "$MACHINES" | grep "^${LABEL}|")
      # the emulated disk kept: what a pass's read ahead drops unread depends on how far the disk got (#285)
      ENV=$(printf '%s' "$LINE" | cut -d'|' -f2)
      THR=$(printf '%s' "$LINE" | cut -d'|' -f3)
      eval "BL=\${MACHINES_BIN_$LABEL:-$B}"
      eval "cleanup_run env TR_BAR=0 $ENV $BL $RUN $THR --profile" > "$D/$LABEL-profile.out" 2> "$D/$LABEL.prof"
      awk -v l="$LABEL" '
        /^experts: / { ex = $2 "/" $4; for (i = 2; i <= NF; i++) {
            if ($i == "RAM") { u = $(i + 1); sub(/^[(]/, "", u); ex = ex " " u " MiB" }
            if ($i == "hits,") h = $(i - 1); if ($i == "misses,") m = $(i - 1); if ($i == "MiB" && $(i + 1) == "read") r = $(i - 1)
            if ($i == "dropped") d = $(i - 1) } }
        /^== decode:/ { dec = 1; tok = $3; next }
        /^== / { dec = 0 }
        dec && /read from disk:/ { disk = $4 }
        END { printf "%-10s %-28s %8s %8s %8s %10s %10.1f\n", l, ex, h, m, d + 0, r, (tok > 0 ? disk / tok : 0) }' "$D/$LABEL.prof"
    done
  done
}

# every machine's row of every model a folder holds
machines_rows() {
  for M in $2; do
    [ -f "$1/$M/ab_modes.txt" ] || continue
    printf '%s\n' "$MACHINES" | while IFS='|' read -r LABEL ENV THR BW CEIL; do
      # a machine a folder's runs did not have (avgd came after build/machines of 10-03): no row
      [ -f "$1/$M/$LABEL.prof" ] || continue
      machines_row "$1/$M" "$M" "$LABEL" "$BW" "$CEIL"
    done
  done
}

# one machine's row: its profile (the bytes of a decode token), its medians (ab_modes), its expert budget
machines_row() {
  awk -v m="$2" -v l="$3" -v bw="$4" -v ceil="$5" '
    FILENAME ~ /[.]prof$/ && /^== decode:/ { dec = 1; tok = $3; next }
    FILENAME ~ /[.]prof$/ && /^== / { dec = 0 }
    FILENAME ~ /[.]prof$/ && dec && /weights touched:/ { w = $3 }
    FILENAME ~ /[.]prof$/ && dec && /KV cache read:/ { kv = $4 }
    FILENAME ~ /[.]prof$/ && dec && /read from disk:/ { disk = $4 }
    # the disk the run read from: the one emulated (TR_EXPERT_DISK_MBPS), else the 3.5 GB/s of this PC
    FILENAME ~ /[.]prof$/ && /^experts: / && match($0, /disk emulated at [0-9]+ MB[/]s/) {
      dk = substr($0, RSTART + 17, RLENGTH - 22) / 1000 }
    FILENAME ~ /[.]prof$/ && /^experts: / {
      ex = "resident"
      for (i = 2; i <= NF; i++)
        if ($i == "units" && $(i + 1) == "in") { u = $(i + 3); sub(/^[(]/, "", u); ex = $(i - 3) "/" $(i - 1) " " u " MiB" } }
    FILENAME ~ /ab_modes/ && $1 == l && $3 ~ /^[0-9]+$/ && $3 > 0 { n[$2]++; a[$2, n[$2]] = $4 }
    END {
      # the medians from the runs themselves, round 0 out (the warm-up, as tools/ab_modes.sh)
      split("prefill decode wall_ms", f, " ")
      for (q = 1; q <= 3; q++) {
        k = f[q]; c = n[k]
        for (i = 1; i < c; i++) for (j = i + 1; j <= c; j++) if (a[k, j] + 0 < a[k, i] + 0) { t = a[k, i]; a[k, i] = a[k, j]; a[k, j] = t }
        v[k] = c == 0 ? 0 : (c % 2 ? a[k, (c + 1) / 2] : (a[k, c / 2] + a[k, c / 2 + 1]) / 2)
      }
      # the profiler counts the bytes read from disk as weights touched too (olmoe.c TR_PROF_WEIGHT_READ):
      # they land by DMA while the token waits on the disk, so the RAM term takes them out
      dpt = tok > 0 ? disk / tok : 0
      w -= dpt
      printf "%s|%s|%s|%s|%s|%s|%s|%s|%.1f|%s|%s|%s\n", m, l, ex, v["prefill"], v["decode"], v["wall_ms"], w, kv, dpt, bw,
             ceil, (dk > 0 ? dk : 3.5)
    }' "$1/$3.prof" "$1/ab_modes.txt"
}

# the table: a limit is the RAM's bandwidth over weights + KV a token, plus the disk's bytes a token at the
# run's disk speed ("disk", GB/s); "nominal" with the RAM's nominal bandwidth, "ceiling" with its practical one
# (measured on this PC, estimated elsewhere), and "of ceil" the decode against the latter
machines_table() {
  awk -F'|' '
    BEGIN { printf "%-4s %-5s %-24s %8s %8s %8s %9s %8s %6s %8s %8s %8s %8s\n", "", "", "experts in RAM", "prompt",
                   "decode", "wall ms", "MiB/tok", "disk/tok", "disk", "GB/s", "nominal", "ceiling", "of ceil" }
    {
      mib = $7 + $8
      dsk = $9 * 1048576 / ($12 * 1e9)
      ln = 1 / (mib * 1048576 / ($10 * 1e9) + dsk)
      lc = 1 / (mib * 1048576 / ($11 * 1e9) + dsk)
      printf "%-4s %-5s %-24s %8.2f %8.2f %8d %9.1f %8.1f %6.2f %8.2f %8.2f %8.2f %7.0f%%\n", $1, $2, $3, $4, $5, $6, mib,
             $9, $12, $5 * mib * 1048576 / 1e9, ln, lc, 100 * $5 / lc
    }'
}
main "$@"; exit
