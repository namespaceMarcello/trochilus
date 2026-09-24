#!/bin/sh
# mutate_gpu.sh — does tests/test_gpu_attn.c see a wrong GPU attention? A test never seen red proves
# nothing (docs/LESSONS.md #43): each mutation is applied to a copy of src/backend/gpu_attn.c in a copy
# of the tree, the test is built against it and run, and must fail. The kernels are PTX that the
# driver compiles, so this runs natively, on the machine with the GPU (the container has none: there
# the test is skipped, and a skipped test proves nothing here either). From the repo root:
#
#   sh tools/mutate_gpu.sh
#
# One line per mutation. "no mutation" must be green, every other line RED; the script exits 1
# otherwise. A binary Smart App Control blocks (exit 126) is run as copies with 1, 2, ... bytes
# appended, each a new hash with a verdict of its own (docs/LESSONS.md #12, #81). About a minute,
# most of it the first build of the copy; each blocked copy adds 10 s.
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) EXE=.exe; PYBIN=${PY:-tools/.venv/Scripts/python.exe} ;;
  *) EXE=; PYBIN=${PY:-tools/.venv/bin/python} ;;
esac
ROOT=$(pwd)
case "$PYBIN" in /*) ;; *) PYBIN=$ROOT/$PYBIN ;; esac
W=${TMPDIR:-/tmp}/tr-mutate-gpu
rm -rf "$W" && mkdir -p "$W" && cp -r Makefile src tests "$W/" || exit 1
cp src/backend/gpu_attn.c "$W/gpu_attn.orig.c"
cat > "$W/mut.py" <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit("mutation does not apply once: " + old)
open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
EOF
FAILED=0
# $1: name, $2: text of gpu_attn.c, $3: its replacement; $4 = green: the run that must pass
run() {
  cd "$W" || exit 1
  cp gpu_attn.orig.c src/backend/gpu_attn.c
  if ! "$PYBIN" mut.py src/backend/gpu_attn.c "$2" "$3"; then
    echo "$1: MUTATION DOES NOT APPLY (fix the script)"; FAILED=1; cd "$ROOT"; return
  fi
  if ! make BUILD=b WERROR=1 "b/tests/test_gpu_attn$EXE" > build.log 2>&1; then
    tail -20 build.log
    echo "$1: BUILD FAILED (not a mutation seen: fix the mutation)"; FAILED=1; cd "$ROOT"; return
  fi
  # a copy Smart App Control blocks is replaced by one with one more byte: another hash, another verdict
  # (docs/LESSONS.md #81); up to 8 copies, 10 s apart
  TRY=1
  while :; do
    cp "b/tests/test_gpu_attn$EXE" "b/run$TRY$EXE" && head -c $TRY /dev/zero >> "b/run$TRY$EXE"
    "./b/run$TRY$EXE" > run.log 2>&1
    RC=$?
    if [ $RC -ne 126 ] || [ $TRY -ge 8 ]; then break; fi
    TRY=$((TRY + 1))
    sleep 10
  done
  if [ $RC -eq 126 ]; then RES="BLOCKED by Smart App Control ($TRY copies)"
  elif grep -q "skipped" run.log; then RES="SKIPPED (no GPU: nothing proven)"
  elif [ $RC -eq 0 ]; then RES=green
  else RES=RED
  fi
  # what the test saw: the floats that differ from scalar's and from the active tier's, or its first failed check
  WHY=$(grep -o "[0-9]* and [0-9]* floats differ" run.log | head -1)
  [ -n "$WHY" ] || WHY=$(grep -m1 "check failed" run.log)
  echo "$1: $RES ($WHY)"
  if [ "$4" = green ]; then [ "$RES" = green ] || { tail -5 run.log; FAILED=1; }
  else [ "$RES" = RED ] || FAILED=1
  fi
  cd "$ROOT"
}
run "no mutation" "static void emit_v(sbuf *b) {" "static void emit_v(sbuf *b) {" green
run "values' sum without .rn: ptxas fuses it into an fma" \
  'mul.rn.f32 %%m, %%a, %%v;\nadd.rn.f32 %%o, %%o, %%m;\n' 'mul.f32 %%m, %%a, %%v;\nadd.f32 %%o, %%o, %%m;\n'
run "dot's lane tree: (s01+s45) where tr_lane_combine has (s01+s23)" 'w + 4, w + 0, w + 1);' 'w + 4, w + 0, w + 2);'
run "sum's lane tree in the other order (lane i+8 first)" '1 << k);' '8 >> k);'
run "values' sum starting at -0" 'mov.f32 %%o, 0f00000000;' 'mov.f32 %%o, 0f80000000;'
run "tr_expf without its exception table" 'i < TR_EXPF_N_EXCEPTIONS; i++)' 'i < 0; i++)'
run "kv_write: the transposed key read one dim off" 'mad.lo.u32 %%r20, %%r4, 33, %%r5;' \
  'mad.lo.u32 %%r20, %%r4, 33, %%r5;\nadd.u32 %%r20, %%r20, 1;'
run "decode appends its key and value one position late" 'cvt.u64.u32 %%rd9, %%r9;' 'cvt.u64.u32 %%rd9, %%r1;'
rm -rf "$W"
if [ $FAILED -ne 0 ]; then echo "mutate_gpu: FAILED (a mutation not seen, or the clean run not green)"; exit 1; fi
echo "mutate_gpu: every mutation seen"
}
main "$@"; exit
