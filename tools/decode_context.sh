#!/bin/sh
# decode_context.sh — the decode as the context grows (docs/MISURE.md §Decode a contesto lungo):
# native Windows, still machine, rotating order, A/A control (tools/ab_modes.sh, docs/LEZIONI.md
# #66). Run from the repo root in Git Bash. Results in build/decode_context/.
#
#   sh tools/decode_context.sh measure [rounds]
#       bench_mem    what the RAM gives: plain reads in a row and scattered, the engine's matmul,
#                    one token's attention on the two KV layouts (tests/bench_mem.c)
#       speed        decode at context 32, 512, 2048 and 4000 in ONE ab_modes session: the default
#                    (the session measures its decode width) and --decode-threads 8, each with its
#                    A/A copy. About 35 minutes.
#       profile      ms and bytes read per zone at every context (tools/profile_suite.py on
#                    bench/scenarios-decode-context.json). About 15 minutes.
#   sh tools/decode_context.sh change <binary-before> [rounds]
#       exact: the two binaries give the same logits (600 positions, byte for byte) and the same
#       tokens after a prompt of 4000 on the real model, or the measurement does not start.
#       speed-auto, speed-d8: before and after at every context, each with its A/A copy, with the
#       default width and with the decode forced on 8; then bench_mem and the profile of the new
#       binary. About 90 minutes. PROF_BEFORE=<binary> profiles that binary too (profile-before):
#       a binary older than the bytes per zone has no such numbers, its twin built with them has.
#
# The order of the 16 modes of a session holds every context after every context once (a de
# Bruijn sequence over the 4 contexts): a run on the machine a 4000-token prompt has just heated
# is not always the same context, and a mode and its A/A copy never follow the same context, so
# the A/A difference carries that effect too.
#
# The containers of the other projects are stopped for the duration and started again at the end,
# also when the script fails or is interrupted; if somebody starts one in the meantime the
# measurement stops at the next run instead of going on with a busy machine (AB_GUARD,
# docs/LEZIONI.md #73). A binary Smart App Control still blocks (docs/LEZIONI.md #12) is waited
# for, never rebuilt: a rebuild starts the wait again.
set -e
# The body is one function, called on the last line: the shell parses all of it before it runs
# any, so editing this file while it runs cannot change a run under way (docs/LEZIONI.md #69).
main() {
WHAT=$1
B=build/trochilus.exe
MEMB=build/tests/bench_mem.exe
M=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
PY=tools/.venv/Scripts/python.exe
SCEN=bench/scenarios-decode-context.json
OUT=build/decode_context
case "$WHAT" in
  measure) R=${2:-8} ;;
  change) BEFORE=$2; R=${3:-8}; [ -f "$BEFORE" ] || { echo "decode_context: no binary '$BEFORE'"; exit 2; } ;;
  *) echo "usage: decode_context.sh measure [rounds] | change <binary-before> [rounds]"; exit 2 ;;
esac
mkdir -p $OUT
[ -f $B ] && [ -f $MEMB ] && [ -f $M ] || { echo "decode_context: $B, $MEMB or $M is missing"; exit 1; }
[ -z "$PROF_BEFORE" ] || [ -f "$PROF_BEFORE" ] || { echo "decode_context: no binary '$PROF_BEFORE'"; exit 2; }

# up to an hour, one try a minute (a blocked exe exits 126 from Git Bash; bench_mem without
# arguments prints its usage and exits 2, which is a binary that runs)
wait_runs() {
  TRY=0
  while :; do
    RC=0
    "$@" > /dev/null 2>&1 || RC=$?
    [ "$RC" = 126 ] || break
    TRY=$((TRY + 1))
    [ "$TRY" -le 60 ] || { echo "decode_context: $1 still does not run after an hour"; exit 1; }
    echo "decode_context: $1 does not run yet (Smart App Control?), try $TRY, waiting 60 s"
    sleep 60
  done
}
# one measurement at a time, and the machine stays awake while it lasts (docs/LEZIONI.md #82)
. tools/measure_guard.lib
measure_begin decode_context
trap measure_end EXIT INT TERM
wait_runs $B cpu
wait_runs $MEMB
[ -z "$BEFORE" ] || wait_runs $BEFORE cpu
[ -z "$PROF_BEFORE" ] || wait_runs $PROF_BEFORE cpu

RUNNING=$(docker ps -q 2>/dev/null || true)
restart() { measure_end; if [ -n "$RUNNING" ]; then docker start $RUNNING > /dev/null 2>&1 || true; echo "containers started again"; fi; }
trap restart EXIT INT TERM
if [ -n "$RUNNING" ]; then
  # the VM's file cache goes back to Windows first (docs/LEZIONI.md #38), then everything stops
  MSYS_NO_PATHCONV=1 docker run --rm --privileged trochilus-dev:local sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" || true
  docker stop $RUNNING > /dev/null
  echo "containers stopped: $(echo $RUNNING | wc -w)"
fi
# from here on a running container means somebody else is using the machine
AB_GUARD='[ -z "$(docker ps -q 2>/dev/null)" ]'
export AB_GUARD
still() { sh -c "$AB_GUARD" || { echo "decode_context: a container is running $1: the machine is not still, stopping"; exit 3; }; }

# The model takes 7 GiB and the memory guard wants 3 more left free; Windows needs minutes to
# take back what the VM has released (docs/LEZIONI.md #72). Up to 15 minutes, a look every 30 s.
TRY=0
while :; do
  AVAIL=$($B cpu 2>&1 | sed -n 's/^ram: .* total, \([0-9]*\)\.[0-9]* GiB available.*/\1/p')
  [ "${AVAIL:-0}" -lt 12 ] || break
  TRY=$((TRY + 1))
  [ "$TRY" -le 30 ] || { echo "decode_context: only ${AVAIL:-?} GiB available after 15 minutes, 12 wanted"; exit 1; }
  echo "decode_context: ${AVAIL:-?} GiB available, 12 wanted: waiting 30 s"
  sleep 30
done

# $1: binary, $2: prompt length, $3: context, $4: further arguments
gen() { echo "$1 generate -m $M -p $2 -n 48 -c $3 -t 16 $4"; }

# $1: name of the output
bench_mem() {
  echo "##### bench_mem"
  still "before bench_mem"
  for G in ram weights "kv 512" "kv 2048" "kv 4000"; do
    $MEMB $G --runs 7
  done > $OUT/bench_mem-$1.txt
  still "after bench_mem"
  grep -c "GB/s" $OUT/bench_mem-$1.txt
}

# $1: name of the output, $2: binary
profile() {
  echo "##### profile by zone, $2"
  still "before the profile"
  $PY tools/profile_suite.py --binary $2 --scenarios $SCEN > $OUT/profile-$1.txt
  still "after the profile"
  grep -c "^==" $OUT/profile-$1.txt
}

# One session over the four contexts. $1: output name; $2 $3 $4: label, binary and further
# arguments of the first mode; $5 $6 $7: of the second. Positions of a context in the de Bruijn
# order 0 0 1 0 2 0 3 1 1 2 1 3 2 2 3 3: the first two take the two modes, the last two their
# A/A copies.
session() {
  NAME=$1; LA=$2; BA=$3; AA=$4; LB=$5; BB=$6; AB=$7
  a() { echo "$LA-$1$3=$(gen $BA $1 $2 "$AA")"; }
  b() { echo "$LB-$1$3=$(gen $BB $1 $2 "$AB")"; }
  echo "##### $NAME"
  sh tools/ab_modes.sh $R \
     "$(a 32 128)" "$(b 32 128)" "$(a 512 600)" "$(a 32 128 -again)" \
     "$(a 2048 2200)" "$(b 32 128 -again)" "$(a 4000 4096)" "$(b 512 600)" \
     "$(a 512 600 -again)" "$(b 2048 2200)" "$(b 512 600 -again)" "$(b 4000 4096)" \
     "$(a 2048 2200 -again)" "$(b 2048 2200 -again)" "$(a 4000 4096 -again)" "$(b 4000 4096 -again)" \
     > $OUT/$NAME.txt
  sed -n '/^medians/,$p' $OUT/$NAME.txt
}

# Before any speed: the two binaries give the same numbers on the real model, or there is
# nothing to compare. `logits -b k` writes one row per pass of k tokens, the last token's:
# 600 positions one token a pass (the decode, context growing to 600), then passes of 64 on 8
# threads (the prompt's path), each before against after, byte for byte; the last row of the
# two shapes is the same position, so it must be the same bytes too. Then the tokens generated
# after the longest prompt.
exact() {
  echo "##### exact: logits and tokens, before against after"
  TOK=$(awk 'BEGIN { for (i = 0; i < 600; i++) printf "%s%d", (i ? "," : ""), (i * 7919 + 13) % 50304 }')
  $BEFORE logits -m $M --tokens $TOK --out $OUT/logits-before.bin -t 16 > /dev/null 2>&1
  $B logits -m $M --tokens $TOK --out $OUT/logits-after.bin -t 16 > /dev/null 2>&1
  $BEFORE logits -m $M --tokens $TOK --out $OUT/logits-before-b64.bin -t 8 -b 64 > /dev/null 2>&1
  $B logits -m $M --tokens $TOK --out $OUT/logits-after-b64.bin -t 8 -b 64 > /dev/null 2>&1
  cmp $OUT/logits-before.bin $OUT/logits-after.bin || return 4
  cmp $OUT/logits-before-b64.bin $OUT/logits-after-b64.bin || return 4
  ROW=$((50304 * 4))
  tail -c $ROW $OUT/logits-after.bin > $OUT/row-b1.bin
  tail -c $ROW $OUT/logits-after-b64.bin > $OUT/row-b64.bin
  cmp $OUT/row-b1.bin $OUT/row-b64.bin || return 4
  echo "logits identical byte for byte: $(wc -c < $OUT/logits-before.bin) bytes one token a pass on 16 threads," \
       "$(wc -c < $OUT/logits-before-b64.bin) bytes in passes of 64 on 8, and position 599 the same in the two shapes"
  rm -f $OUT/logits-before.bin $OUT/logits-after.bin $OUT/logits-before-b64.bin $OUT/logits-after-b64.bin \
        $OUT/row-b1.bin $OUT/row-b64.bin
  $BEFORE generate -m $M -p 4000 -n 48 -c 4096 -t 16 2> /dev/null > $OUT/tokens-before.txt
  $B generate -m $M -p 4000 -n 48 -c 4096 -t 16 2> /dev/null > $OUT/tokens-after.txt
  cmp $OUT/tokens-before.txt $OUT/tokens-after.txt || return 4
  echo "tokens identical after a prompt of 4000: $(cut -c1-60 $OUT/tokens-after.txt)..."
}

if [ "$WHAT" = measure ]; then
  bench_mem measure
  session speed auto $B "" d8 $B "--decode-threads 8"
  profile measure $B
else
  # in a || list set -e is off inside the function: every comparison returns by itself
  exact > $OUT/exact.txt 2>&1 || { cat $OUT/exact.txt; echo "decode_context: before and after differ, nothing to measure"; exit 4; }
  cat $OUT/exact.txt
  session speed-auto before $BEFORE "" after $B ""
  session speed-d8 before8 $BEFORE "--decode-threads 8" after8 $B "--decode-threads 8"
  bench_mem change
  [ -z "$PROF_BEFORE" ] || profile before $PROF_BEFORE
  profile change $B
fi
echo "done: $OUT"
}
main "$@"; exit
