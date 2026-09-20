#!/bin/sh
# mutate_expf.sh — do the checks see a wrong tr_expf? A test never seen red proves nothing
# (docs/LEZIONI.md #43): each mutation is applied to a copy of the tree, the copy is built and
# the checks that should notice are run. Linux container, from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_expf.sh
#
# One line per mutation, four checks: test_expf (the quick test), every-float (bench_expf
# --check, all the 2^32 floats), table (tools/gen_expf_table.py --check: the header is what its
# script writes) and lint (the hot zone). "no mutation" must be all green, every other line must
# have at least one RED. About 10 minutes.
set -e
# The body is one function, called on the last line (docs/LEZIONI.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PYBIN=${PY:-tools/.venv/bin/python}
cat > /tmp/mut.py <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
if s.count(old) < 1:
    sys.exit("mutation does not apply: " + old)
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
# $1: name, $2: file, $3: text, $4: replacement
run() {
  rm -rf /tmp/mut && mkdir -p /tmp/mut
  tar -c --exclude=tools/.venv --exclude=tools/docker Makefile CLAUDE.md src tests tools bench docs | tar -x -C /tmp/mut
  cd /tmp/mut
  $PYBIN /tmp/mut.py "$2" "$3" "$4"
  RES=""
  if make BUILD=b CC=gcc b/tests/test_expf b/tests/bench_expf > /tmp/mut-build.log 2>&1; then
    if ./b/tests/test_expf > /dev/null 2>&1; then RES="$RES test_expf=green"; else RES="$RES test_expf=RED"; fi
    if ./b/tests/bench_expf --check > /tmp/mut-bench.log 2>&1; then RES="$RES every-float=green"; else RES="$RES every-float=RED"; fi
    RES="$RES ($(sed -n 's/^bits, tr_expf: \([0-9]*\) results are not the library.s expf, \([0-9]*\) are not the reference/\2 wrong/p' /tmp/mut-bench.log), $(sed -n 's/^bits, tr_expf: \([0-9]*\) arguments left unproven.*entries, \([0-9]*\) that.*/\1 unproven, \2 dead entries/p' /tmp/mut-bench.log))"
  else
    RES="$RES build=RED"
  fi
  if $PYBIN tools/gen_expf_table.py --check > /dev/null 2>&1; then RES="$RES table=green"; else RES="$RES table=RED"; fi
  if $PYBIN tools/lint.py > /dev/null 2>&1; then RES="$RES lint=green"; else RES="$RES lint=RED"; fi
  echo "$1:$RES"
  cd /src
}
H=src/kernels/expf_table.h
run "no mutation" $H "TR_EXPF_TABLE_H" "TR_EXPF_TABLE_H"
run "table: 2^(1/64) one unit of the last place up" $H "0x1.02c9a3e778061p+0" "0x1.02c9a3e778062p+0"
run "table: 2^(4/64) holds the value of 2^(5/64)" $H "0x1.0b5586cf9890fp+0" "0x1.0e3ec32d3d1a2p+0"
run "polynomial: 1/120 doubled" $H "TR_EXPF_C5 0x1.1111111111111p-7" "TR_EXPF_C5 0x1.1111111111111p-6"
run "polynomial: 1/6 one unit of the last place down" $H "TR_EXPF_C3 0x1.5555555555555p-3" "TR_EXPF_C3 0x1.5555555555554p-3"
run "reduction: ln2/64 low piece one unit of the last place up" $H "0x1.a39ef35793c76p-39" "0x1.a39ef35793c77p-39"
run "reduction: the low piece of ln2/64 forgotten" src/kernels/expf.c "(xd - kd * TR_EXPF_LN2_64_HI) - kd * TR_EXPF_LN2_64_LO" "(xd - kd * TR_EXPF_LN2_64_HI)"
run "rounding test: margin too wide, 2^-40" $H "TR_EXPF_MARGIN 0x1p-50" "TR_EXPF_MARGIN 0x1p-40"
run "rounding test: margin too narrow, 2^-60" $H "TR_EXPF_MARGIN 0x1p-50" "TR_EXPF_MARGIN 0x1p-60"
run "rounding test: never fails" src/kernels/expf.c "if (a.u == b.u) return a.f;" "if (a.u == b.u || a.u != b.u) return a.f;"
run "exceptions: one value off by one" $H "0x3f800c6du" "0x3f800c6eu"
run "exceptions: one argument lost" $H "{0x40315b33u, 0x417fa47du}" "{0x40315b34u, 0x417fa47du}"
run "overflow: the threshold one float too low" $H "0x1.62e42e0000000p+6f" "0x1.62e42c0000000p+6f"
run "overflow: the threshold one float too high" $H "0x1.62e42e0000000p+6f" "0x1.62e4300000000p+6f"
run "underflow: the threshold at -103" $H "-0x1.a000000000000p+6f" "-0x1.9c00000000000p+6f"
run "scale: k / 64 rounded toward zero" src/kernels/expf.c "((k >> 6) + 1023)" "((k / 64) + 1023)"
run "softmax on the library's expf" src/kernels/kernels.c "x[i] = tr_expf(x[i] - m)" "x[i] = expf(x[i] - m)"
run "SiLU on the library's expf" src/kernels/kernels.c "(1.0f + tr_expf(-v))" "(1.0f + expf(-v))"
}
main "$@"; exit
