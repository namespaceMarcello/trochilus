#!/bin/sh
# remeasure.sh — the conclusions of docs/STATUS.md that sit inside their own spread, measured again
# (docs/LESSONS.md #66): native Windows, still machine, 8 rounds, rotating order, A/A control.
# Run from the repo root in Git Bash; about 15 minutes. Results in build/remeasure/.
#
# The machine's marker (~/.claude/macchina-ferma) is held for the duration and given back at the
# end, also when the script fails or is interrupted: the other windows pause their work, their
# containers stay up (tools/measure_guard.lib; docs/STATUS.md: measurements want a still machine).
#
#   sh tools/remeasure.sh [rounds]
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LESSONS.md #69).
main() {
R=${1:-8}
B=build/trochilus.exe
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
PY=tools/.venv/Scripts/python.exe
OUT=build/remeasure
mkdir -p $OUT
[ -f $B ] && [ -f $M ] || { echo "remeasure: $B or $M is missing"; exit 1; }

# one measurement at a time, and the machine stays awake while it lasts (docs/LESSONS.md #82)
. tools/measure_guard.lib
trap measure_end EXIT
trap 'exit 130' INT TERM
measure_begin remeasure
# the other windows' containers stay up, their work paused by the marker; from here on a CPU
# busy with other work, or the marker lost, stops the measurement at the next run
# (tools/measure_guard.lib, tools/machine_still.sh, docs/LESSONS.md #73, #84)
measure_machine remeasure

# The model takes 7 GiB and the memory guard wants 3 more left free; after a `make check` Windows
# needs minutes to take back what the VM has released (docs/LESSONS.md #72). Up to 15 minutes.
TRY=0
while :; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "remeasure: only ${AVAIL:-?} GiB available after 15 minutes, 12 wanted"; exit 1; }
  echo "remeasure: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done
measure_still remeasure
measure_declare "before the first run"

echo "##### 1. speculation, worst case (code.txt), with an A/A control"
cleanup_run sh tools/ab_modes.sh $R \
  "spec0=$B run -m $M -f bench/prompts/code.txt -n 200 -t 16 --spec 0" \
  "spec8=$B run -m $M -f bench/prompts/code.txt -n 200 -t 16 --spec 8" \
  "spec0again=$B run -m $M -f bench/prompts/code.txt -n 200 -t 16 --spec 0" > $OUT/1-spec-code.txt
tail -4 $OUT/1-spec-code.txt

echo "##### 2. speculation, good case (code-edit.txt)"
cleanup_run sh tools/ab_modes.sh $R \
  "spec0=$B run -m $M -f bench/prompts/code-edit.txt -n 200 -t 16 --spec 0" \
  "spec8=$B run -m $M -f bench/prompts/code-edit.txt -n 200 -t 16 --spec 8" > $OUT/2-spec-code-edit.txt
tail -3 $OUT/2-spec-code-edit.txt

echo "##### 3. pin modes at 16 threads, with an A/A control"
cleanup_run sh tools/ab_modes.sh $R \
  "pin2=TR_POOL_PIN=2 $B generate -m $M -p 512 -n 24 -c 600 -t 16" \
  "pin1=TR_POOL_PIN=1 $B generate -m $M -p 512 -n 24 -c 600 -t 16" \
  "pin0=TR_POOL_PIN=0 $B generate -m $M -p 512 -n 24 -c 600 -t 16" \
  "pin2again=TR_POOL_PIN=2 $B generate -m $M -p 512 -n 24 -c 600 -t 16" > $OUT/3-pin.txt
tail -9 $OUT/3-pin.txt

echo "##### 4. threads per phase: 8 against 16 (default pin), with an A/A control"
cleanup_run sh tools/ab_modes.sh $R \
  "t16=$B generate -m $M -p 512 -n 48 -c 600 -t 16" \
  "t8=$B generate -m $M -p 512 -n 48 -c 600 -t 8" \
  "t16again=$B generate -m $M -p 512 -n 48 -c 600 -t 16" > $OUT/4-threads.txt
tail -7 $OUT/4-threads.txt

echo "##### 5. question 12, the timed half: ms per pass by zone with 1, 2 and 9 rows (3 alternated rounds)"
IDS=$($B tokenize -m $M -f bench/prompts/code-edit.txt)
for round in 1 2 3; do
  for mode in "0" "1 --spec-fixed" "8 --spec-fixed"; do
    tag=$(echo "$mode" | tr -d ' -')
    $B generate -m $M --tokens "$IDS" -n 200 -t 16 --spec $mode --profile-json $OUT/5-zones-$tag-$round.json > /dev/null 2>&1
  done
done
# to a file and then shown: through a pipe into tee a failed report would end with 0 (docs/LESSONS.md #90)
$PY - $OUT > $OUT/5-zones.txt <<'EOF'
import json, statistics, sys
out = sys.argv[1]
def load(tag):
    rows = []
    for r in (1, 2, 3):
        d = json.load(open(f"{out}/5-zones-{tag}-{r}.json"))["engine"]["phases"]["decode"]
        p = d["zones"]["token"]["calls"]
        rows.append((d["tokens"] / p, {k: v["seconds"] / p * 1e3 for k, v in d["zones"].items()}))
    return rows[0][0], {k: statistics.median(z[k] for _, z in rows) for k in rows[0][1]}
_, z0 = load("0")
print(f"one row: {z0['token']:.1f} ms per pass")
for tag in ("1specfixed", "8specfixed"):
    rows, z = load(tag)
    extra = rows - 1.0
    dense = sum(z[k] - z0[k] for k in ("qkv_proj", "attn_out_proj", "lm_head")) / extra
    experts = sum(z[k] - z0[k] for k in ("expert_gate_up", "expert_down")) / extra
    total = (z["token"] - z0["token"]) / extra
    print(f"{tag}: {rows:.2f} rows/pass, {z['token']:.1f} ms/pass; one more row = {total:.1f} ms "
          f"(dense matmul {dense:.1f}, experts {experts:.1f}, rest {total - dense - experts:.1f})")
EOF
cat $OUT/5-zones.txt
measure_declare "after the last run"
echo "done: $OUT"
}
main "$@"; exit
