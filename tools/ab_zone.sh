#!/bin/sh
# ab_zone.sh — one zone of the profiler, before and after a change, alternated run by run: the
# same prompt, the same minute of the machine, the order flipped every round, so a burst of load
# elsewhere falls on both binaries. On a shared machine a change of a few percent of the prefill
# is lost in the A/A of tokens a second (4-8% on 2026-09-26), while the zone's own seconds, paired
# run by run, read it (docs/LESSONS.md #191, docs/MEASUREMENTS.md §The prompt's attention in tiles).
#
#   sh tools/ab_zone.sh <binary-before> <binary-after> <zone> [rounds] [prompts]
#   e.g. sh tools/ab_zone.sh build/before/build/trochilus.exe build/trochilus.exe attention 8 "2048 4000"
#   AB_ZONE_PHASE=decode AB_ZONE_N=64 AB_ZONE_THREADS=8 sh tools/ab_zone.sh <before> <after> \
#     expert_gate_up+expert_down 8 "512 2048"    the decode's zones (a + sums several), 8 threads forced
#
# Native, the rules of every measurement (tools/measure_guard.lib): the machine's marker, a still
# machine before every run. Binaries Smart App Control already lets run (a fresh copy may be held:
# docs/LESSONS.md #12). Results in build/ab_zone/: runs.txt (binary, prompt, round, prefill seconds,
# zone seconds) and the medians of the per-round ratios, after / before.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
[ $# -ge 3 ] || { echo "usage: sh tools/ab_zone.sh <binary-before> <binary-after> <zone> [rounds] [prompts]"; exit 2; }
BB=$1
BA=$2
Z=$3
R=${4:-8}
PROMPTS=${5:-2048 4000}
M=${AB_ZONE_MODEL:-models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf}
PH=${AB_ZONE_PHASE:-prefill}
N=${AB_ZONE_N:-2}
T=${AB_ZONE_THREADS:-16}
XA=""
[ $PH = decode ] && XA="--decode-threads $T"
PY=tools/.venv/Scripts/python.exe
[ -x $PY ] || PY=tools/.venv/bin/python
OUT=build/ab_zone
mkdir -p $OUT
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin ab_zone
measure_machine ab_zone
measure_still ab_zone
measure_declare "before the first run" > $OUT/load.txt
cat $OUT/load.txt
: > $OUT/runs.txt
i=0
while [ $i -lt $R ]; do
  for P in $PROMPTS; do
    if [ $((i % 2)) -eq 0 ]; then ORDER="before after"; else ORDER="after before"; fi
    for W in $ORDER; do
      if [ $W = before ]; then B=$BB; else B=$BA; fi
      cleanup_run sh -c "$AB_GUARD" || { echo "ab_zone: the machine is not still: stopping"; exit 3; }
      cleanup_run $B generate -m $M -p $P -n $N -t $T $XA -c $((P + N + 96)) --profile-json $OUT/p.json > /dev/null 2>&1
      $PY -c "
import json; z = json.load(open('$OUT/p.json'))['engine']['phases']['$PH']
print('$W', $P, $i, round(z['seconds'], 4), round(sum(z['zones'][k]['seconds'] for k in '$Z'.split('+')), 4))" >> $OUT/runs.txt
    done
  done
  i=$((i + 1))
done
measure_declare "after the last run" >> $OUT/load.txt
tail -1 $OUT/load.txt
$PY -c "
import statistics, collections
d = collections.defaultdict(dict)
for w, p, i, pre, z in (l.split() for l in open('$OUT/runs.txt')):
    d[(int(p), i)][w] = (float(pre), float(z))
for p in sorted({k[0] for k in d}):
    pairs = [v for (pp, i), v in d.items() if pp == p and len(v) == 2]
    rz = sorted(v['after'][1] / v['before'][1] for v in pairs)
    rp = sorted(v['after'][0] / v['before'][0] for v in pairs)
    print('prompt %d, %d pairs: zone $Z after/before %.3f [%.3f-%.3f], $PH %.3f [%.3f-%.3f]' % (
        p, len(pairs), statistics.median(rz), rz[0], rz[-1], statistics.median(rp), rp[0], rp[-1]))
" > $OUT/summary.txt
cat $OUT/summary.txt
}
main "$@"; exit
