#!/bin/sh
# prefill_context.sh — the prefill as the prompt grows (docs/MISURE.md §Prefill su prompt lunghi):
# native Windows, still machine, rotating order, A/A control (tools/ab_modes.sh, docs/LEZIONI.md
# #66). Run from the repo root in Git Bash. Results in build/prefill_context/.
#
#   sh tools/prefill_context.sh bench
#       only bench_attn, below: it needs no engine binary. About 3 minutes.
#   sh tools/prefill_context.sh measure [rounds]
#       bench_attn   the attention of a whole prompt on one layer, taken apart (tests/bench_attn.c):
#                    one query at a time against several queries per block of keys, with and
#                    without the softmax, at 512, 2048 and 4000; 16 threads, and one thread with
#                    one head; then the sizes of group and block at 4000
#       speed        prefill (and the 48 tokens after it) at prompt 512, 2048 and 4000 in ONE
#                    ab_modes session, 16 threads, every length with its A/A copy. About 15 minutes.
#       profile      ms and bytes read per zone at every length (tools/profile_suite.py on
#                    bench/scenarios-prefill-context.json). About 10 minutes.
#   sh tools/prefill_context.sh change <binary-before> [rounds]
#       exact: the two binaries give the same logits byte for byte (600 positions one token a
#       pass, passes of 64 on 8 threads, and a prompt of 4000 in passes of 512 and of 100 on 16)
#       and the same tokens after a prompt of 4000, or the measurement does not start.
#       speed: before and after at every length, each with its A/A copy; then bench_attn and the
#       profile of the new binary. About 50 minutes. PROF_BEFORE=<binary> profiles that one too.
#
# The order of a session holds every length after every other one: a run on the machine a prompt
# of 4000 has just heated is not always the same length, and a mode and its A/A copy never follow
# the same length, so the A/A difference carries that effect too.
#
# The containers of the other projects are stopped for the duration and started again at the end,
# also when the script fails or is interrupted; if somebody starts one in the meantime the
# measurement stops at the next run (AB_GUARD, docs/LEZIONI.md #73). A binary Smart App Control
# still blocks (docs/LEZIONI.md #12) is waited for, never rebuilt: a rebuild starts the wait again.
# When the wait goes past half an hour, a second copy built somewhere else may run at once
# (make BUILD=build/after2 build/after2/trochilus.exe, then TROCHILUS=build/after2/trochilus.exe
# sh tools/prefill_context.sh ...; docs/LEZIONI.md #81). One measurement at a time, and the machine
# is kept awake while it lasts (tools/measure_guard.lib, docs/LEZIONI.md #82).
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LEZIONI.md #69).
main() {
WHAT=$1
# TROCHILUS=<binary>: the engine to measure, when it is not the one make builds
B=${TROCHILUS:-build/trochilus.exe}
ATTNB=build/tests/bench_attn.exe
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
PY=tools/.venv/Scripts/python.exe
SCEN=bench/scenarios-prefill-context.json
OUT=build/prefill_context
case "$WHAT" in
  bench) R=0 ;;
  measure) R=${2:-8} ;;
  change) BEFORE=$2; R=${3:-8}; [ -f "$BEFORE" ] || { echo "prefill_context: no binary '$BEFORE'"; exit 2; } ;;
  *) echo "usage: prefill_context.sh bench | measure [rounds] | change <binary-before> [rounds]"; exit 2 ;;
esac
mkdir -p $OUT
[ -f $ATTNB ] || { echo "prefill_context: $ATTNB is missing"; exit 1; }
[ "$WHAT" = bench ] || { [ -f $B ] && [ -f $M ]; } || { echo "prefill_context: $B or $M is missing"; exit 1; }
[ -z "$PROF_BEFORE" ] || [ -f "$PROF_BEFORE" ] || { echo "prefill_context: no binary '$PROF_BEFORE'"; exit 2; }

# up to an hour, one try a minute (a blocked exe exits 126 from Git Bash; bench_attn without
# arguments prints its usage and exits 2, which is a binary that runs)
wait_runs() {
  TRY=0
  while :; do
    RC=0
    "$@" > /dev/null 2>&1 || RC=$?
    [ "$RC" = 126 ] || break
    TRY=$((TRY + 1))
    [ "$TRY" -le 60 ] || { echo "prefill_context: $1 still does not run after an hour"; exit 1; }
    echo "prefill_context: $1 does not run yet (Smart App Control?), try $TRY, waiting 60 s"
    sleep 60
  done
}
# one measurement at a time, and the machine stays awake while it lasts (docs/LEZIONI.md #82)
. tools/measure_guard.lib
measure_begin prefill_context
trap measure_end EXIT
trap 'exit 130' INT TERM
[ "$WHAT" = bench ] || wait_runs $B cpu
wait_runs $ATTNB
[ -z "$BEFORE" ] || wait_runs $BEFORE cpu
[ -z "$PROF_BEFORE" ] || wait_runs $PROF_BEFORE cpu
# which binaries these numbers belong to: a name like trochilus-before.exe is reused from one
# session to the next, a hash is not (docs/LEZIONI.md #81, #86)
sha256sum $B $ATTNB $BEFORE $PROF_BEFORE > $OUT/binaries.sha256 2> /dev/null || true

RUNNING=$(docker ps -q 2>/dev/null || true)
restart() { measure_end; if [ -n "$RUNNING" ]; then docker start $RUNNING > /dev/null 2>&1 || true; echo "containers started again"; fi; }
trap restart EXIT
if [ -n "$RUNNING" ]; then
  # the VM's file cache goes back to Windows first (docs/LEZIONI.md #38), then everything stops
  MSYS_NO_PATHCONV=1 docker run --rm --privileged trochilus-dev:local sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" || true
  docker stop $RUNNING > /dev/null
  echo "containers stopped: $(echo $RUNNING | wc -w)"
fi
# from here on a running container, or a CPU busy with other work, means somebody else is using
# the machine (tools/measure_guard.lib, tools/machine_still.sh)
AB_GUARD=$MEASURE_AB_GUARD
export AB_GUARD
still() { sh -c "$AB_GUARD" || { echo "prefill_context: the machine is not still $1 (a container, or a busy CPU), stopping"; exit 3; }; }

# The model takes 7 GiB and the memory guard wants 3 more left free; Windows needs minutes to
# take back what the VM has released (docs/LEZIONI.md #72). Up to 15 minutes, a look every 30 s.
TRY=0
while [ "$WHAT" != bench ]; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "prefill_context: only ${AVAIL:-?} GiB available after 15 minutes, 12 wanted"; exit 1; }
  echo "prefill_context: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done
measure_still prefill_context
measure_declare "before the first run"

# $1: binary, $2: prompt length, $3: context. GEN_EXTRA: further arguments of every run, for
# instance "--decode-threads 8": the width of the decode is measured by each session for itself
# and changes from run to run (docs/MISURE.md question 31); forcing it on both binaries keeps
# that choice out of a before/after comparison of the 48 tokens after the prompt.
gen() { echo "$1 generate -m $M -p $2 -n 48 -c $3 -t 16 $GEN_EXTRA"; }

# $1: name of the output. Every call is one run, well under 60 s.
bench_attn() {
  echo "##### bench_attn"
  still "before bench_attn"
  {
    for N in 512 2048 4000; do
      $ATTNB $N
      $ATTNB $N --threads 1 --heads 1
    done
    # the sizes of the blocked variants, where the keys of a head no longer fit the caches
    for G in 4 16 64; do
      for K in 16 64 256; do
        [ "$G-$K" = 16-64 ] || $ATTNB 4000 --only blk --group $G --block $K
      done
    done
  } > $OUT/bench_attn-$1.txt
  still "after bench_attn"
  grep -c "^attn" $OUT/bench_attn-$1.txt
}

# $1: name of the output, $2: binary
profile() {
  echo "##### profile by zone, $2"
  still "before the profile"
  cleanup_run $PY tools/profile_suite.py --binary $2 --scenarios $SCEN > $OUT/profile-$1.txt
  still "after the profile"
  grep -c "^==" $OUT/profile-$1.txt
}

# Before any speed: the two binaries give the same numbers on the real model, or there is
# nothing to compare. `logits -b k` writes one row per pass of k tokens, the last token's.
# $1: name, $2: number of tokens, then the arguments of `logits`
same_logits() {
  NAME=$1; N=$2; shift 2
  TOK=$(awk -v n=$N 'BEGIN { for (i = 0; i < n; i++) printf "%s%d", (i ? "," : ""), (i * 7919 + 13) % 50304 }')
  $BEFORE logits -m $M --tokens $TOK --out $OUT/logits-before-$NAME.bin "$@" > /dev/null 2>&1
  $B logits -m $M --tokens $TOK --out $OUT/logits-after-$NAME.bin "$@" > /dev/null 2>&1
  [ -s $OUT/logits-before-$NAME.bin ] || return 4
  cmp $OUT/logits-before-$NAME.bin $OUT/logits-after-$NAME.bin || return 4
  echo "logits identical byte for byte, $NAME: $N tokens, $(wc -c < $OUT/logits-after-$NAME.bin) bytes ($*)"
}
exact() {
  echo "##### exact: logits and tokens, before against after"
  same_logits b1 600 -t 16 || return 4
  same_logits b64 600 -t 8 -b 64 || return 4
  ROW=$((50304 * 4))
  tail -c $ROW $OUT/logits-after-b1.bin > $OUT/row-b1.bin
  tail -c $ROW $OUT/logits-after-b64.bin > $OUT/row-b64.bin
  cmp $OUT/row-b1.bin $OUT/row-b64.bin || return 4
  echo "position 599 the same one token a pass and in passes of 64"
  same_logits p4000-b512 4000 -t 16 -b 512 || return 4
  same_logits p4000-b100 4000 -t 16 -b 100 || return 4
  tail -c $ROW $OUT/logits-after-p4000-b512.bin > $OUT/row-b512.bin
  tail -c $ROW $OUT/logits-after-p4000-b100.bin > $OUT/row-b100.bin
  cmp $OUT/row-b512.bin $OUT/row-b100.bin || return 4
  echo "position 3999 the same in passes of 512 and of 100"
  rm -f $OUT/logits-*.bin $OUT/row-*.bin
  $BEFORE generate -m $M -p 4000 -n 48 -c 4096 -t 16 2> /dev/null > $OUT/tokens-before.txt
  $B generate -m $M -p 4000 -n 48 -c 4096 -t 16 2> /dev/null > $OUT/tokens-after.txt
  cmp $OUT/tokens-before.txt $OUT/tokens-after.txt || return 4
  echo "tokens identical after a prompt of 4000: $(cut -c1-60 $OUT/tokens-after.txt)..."
}

echo "prefill_context $WHAT, $R rounds, $(date '+%Y-%m-%d %H:%M')"
if [ "$WHAT" = bench ]; then
  bench_attn bench
elif [ "$WHAT" = measure ]; then
  bench_attn measure
  echo "##### speed"
  # lengths 0 1 2 0 2 1: every length after each of the other two, the copies after different ones
  cleanup_run sh tools/ab_modes.sh $R \
     "p512=$(gen $B 512 600)" "p2048=$(gen $B 2048 2200)" "p4000=$(gen $B 4000 4096)" \
     "p512-again=$(gen $B 512 600)" "p4000-again=$(gen $B 4000 4096)" "p2048-again=$(gen $B 2048 2200)" \
     > $OUT/speed.txt
  sed -n '/^medians/,$p' $OUT/speed.txt
  profile measure $B
else
  # in a || list set -e is off inside the function: every comparison returns by itself
  exact > $OUT/exact.txt 2>&1 || { cat $OUT/exact.txt; echo "prefill_context: before and after differ, nothing to measure"; exit 4; }
  cat $OUT/exact.txt
  echo "##### speed"
  # lengths 0 0 1 0 2 1 1 2 2 0 1 2: every length after every length; before and after of a length
  # follow the same length, a mode and its A/A copy follow different ones
  cleanup_run sh tools/ab_modes.sh $R \
     "before-512=$(gen $BEFORE 512 600)" "before-512-again=$(gen $BEFORE 512 600)" \
     "before-2048=$(gen $BEFORE 2048 2200)" "after-512-again=$(gen $B 512 600)" \
     "before-4000-again=$(gen $BEFORE 4000 4096)" "before-2048-again=$(gen $BEFORE 2048 2200)" \
     "after-2048-again=$(gen $B 2048 2200)" "before-4000=$(gen $BEFORE 4000 4096)" \
     "after-4000-again=$(gen $B 4000 4096)" "after-512=$(gen $B 512 600)" \
     "after-2048=$(gen $B 2048 2200)" "after-4000=$(gen $B 4000 4096)" \
     > $OUT/speed-change.txt
  sed -n '/^medians/,$p' $OUT/speed-change.txt
  bench_attn change
  [ -z "$PROF_BEFORE" ] || profile before $PROF_BEFORE
  profile change $B
fi
measure_declare "after the last run"
echo "done: $OUT, $(date '+%H:%M')"
}
main "$@"; exit
