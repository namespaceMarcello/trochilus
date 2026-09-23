#!/bin/sh
# mutate_prefetch.sh — do the tests see a wrong read ahead (src/memory/experts.c tr_experts_prefetch
# and its I/O thread, src/models/olmoe.c's use of it in a layer-major prompt, src/base/threads.c
# tr_thread/tr_monitor)? A test never seen red proves nothing (docs/LESSONS.md #43). The tree is
# copied and built once per flavour (gcc; ASan+UBSan; TSan); each mutation is applied to a fresh
# copy of its flavour's build, which then recompiles only the mutated file. Linux container, from
# the repo root (seccomp unconfined: TSan needs setarch -R):
#
#   MSYS_NO_PATHCONV=1 docker run --rm --security-opt seccomp=unconfined -v "$(pwd -W):/src" -w /src \
#       trochilus-dev:local sh tools/mutate_prefetch.sh
#
# One line per mutation: each test's verdict (a crash or a hang past 60 s counts as RED) and the
# checks that failed, by line. "no mutation" must be green, every other line RED. ONLY=<word> runs
# "no mutation" and the mutations with that word in the name.
set -e
# The body is one function, called on the last line (docs/LESSONS.md #69).
main() {
. tools/cleanup.lib
trap cleanup_children EXIT
trap 'exit 130' INT TERM
PYBIN=${PY:-tools/.venv/bin/python}
[ -x "$PYBIN" ] || PYBIN=python3
cat > /tmp/mutp.py <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit("mutation does not apply exactly once: " + old)
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
TESTS="test_experts test_prefetch test_hot test_base"
for fl in gcc asan tsan; do
  rm -rf /tmp/mutp-$fl && mkdir -p /tmp/mutp-$fl && cp -r Makefile src tests /tmp/mutp-$fl/
  (cd /tmp/mutp-$fl && mk $fl -j4 $(for t in $TESTS; do echo b/tests/$t; done) > /tmp/mutp-$fl.log 2>&1) ||
    { echo "the unmutated tree does not build ($fl)"; tail -20 /tmp/mutp-$fl.log; exit 1; }
done
B=src/memory/experts.c
O=src/models/olmoe.c
T=src/base/threads.c
run "no mutation" gcc "$TESTS" $B "int tr_experts_prefetching(" "int tr_experts_prefetching("

# starting the I/O thread
run "start ignores the margin" gcc "test_experts test_prefetch" $B \
  "x->stats.n_slots < 2 * x->n_expert + x->n_used" "x->stats.n_slots < 2 * x->n_expert"
run "start on a resident store" gcc "test_experts" $B "if (tr_experts_resident(x) || x->stats.n_slots" "if (x->stats.n_slots"
run "a second start makes a second thread" asan "test_experts" $B "    if (x->io != NULL) return 0;
    if (tr_experts_resident(x)" "    if (tr_experts_resident(x)"

# what is readable, and when
run "a reserved unit is readable before it is taken in" gcc "test_experts" $B \
  "    if (x->pending != NULL && x->pending[slot]) return NULL; /* in flight: its bytes are not ours yet */
" ""
run "acquire does not wait for a unit in flight" gcc "test_experts test_prefetch" $B \
  "            if (x->pending != NULL && x->pending[slot] && prefetch_take(x, slot) != 0) return -1;
" ""
run "a failed read ahead asked for is not an error" gcc "test_experts" $B \
  "x->pending[slot] && prefetch_take(x, slot) != 0) return -1;" "x->pending[slot]) prefetch_take(x, slot);"
run "the wait takes nothing in" gcc "test_experts test_prefetch" $B \
  "        if (x->pending[s]) prefetch_take(x, (int32_t)s);" "        if (0) prefetch_take(x, (int32_t)s);"

# victims
run "the kept layer may be evicted" gcc "test_experts test_prefetch" $B \
  "int32_t v = coldest_victim(x, layer, keep);" "int32_t v = coldest_victim(x, layer, -1);"
run "a resident unit of the target layer may be evicted" gcc "test_experts" $B \
  "int32_t v = coldest_victim(x, layer, keep);" "int32_t v = coldest_victim(x, -1, keep);"
run "a slot in flight may be a victim" asan "test_experts test_prefetch" $B \
  "        if (x->pending != NULL && x->pending[s]) continue;
" ""
run "the read ahead does not stop when no victim is left" gcc "test_experts" $B \
  "        if (v == -1) break;" "        if (v == -1) continue;"

# counting and failures
run "read ahead not counted as prefetched" gcc "test_experts test_prefetch test_hot" $B \
  "    x->stats.prefetched++;
" ""
run "read ahead not counted as a miss" gcc "test_experts test_prefetch" $B "    x->stats.misses++;
    x->stats.prefetched++;" "    x->stats.prefetched++;"
run "the wait time is not measured" gcc "test_experts" $B "        x->stats.prefetch_wait_sec += tr_time_sec() - t0;
" ""
run "a failed read ahead is not reported" gcc "test_experts test_prefetch" $B "        x->prefetch_failed = 1;
" ""
run "a failure is reported forever" gcc "test_experts" $B "    x->prefetch_failed = 0;
    return failed ? -1 : 0;" "    return failed ? -1 : 0;"
run "a failed read ahead keeps its unit" gcc "test_experts test_prefetch" $B "        x->slot_of[x->slots[s].unit] = -1;
        x->slots[s].unit = -1;
        lru_unlink(x, s);" "        lru_unlink(x, s);"
run "a failed demand read stays hot" gcc "test_experts" $B "    if (rc != 0) {
        lru_unlink(x, victim);
        lru_push_back(x, victim);
        return -1;" "    if (rc != 0) {
        return -1;"

# the I/O thread
run "the I/O thread reads the next expert" gcc "test_experts test_prefetch" $B \
  "job.rc = read_parts(x, job.read, job.read_ctx, job.layer, job.id, base" \
  "job.rc = read_parts(x, job.read, job.read_ctx, job.layer, (job.id + 1) % x->n_expert, base"
run "the I/O thread never says done" gcc "test_experts" $B "        job.done = 1;" "        job.done = 0;"
run "the I/O thread publishes without the lock" tsan "test_experts test_prefetch" $B "        tr_monitor_lock(x->mon);
        job.done = 1;" "        job.done = 1;"
run "freed without joining the I/O thread" asan "test_experts" $B "        tr_thread_join(x->io);
" ""

# olmoe.c: when a prompt reads ahead
run "no read ahead in a prompt" gcc "test_prefetch test_hot" $O \
  "s->prefetch_layer = off == 0 && L + 1 < m->n_layers ? L + 1 : -1;" "s->prefetch_layer = -1;"
run "the rule ignored" gcc "test_prefetch" $O \
  "n_ids >= n_expert - n_expert / OLMOE_PREFETCH_SPARE_DIV" "n_ids >= 0"
run "the layer computing is not kept" gcc "test_prefetch" $O \
  "tr_experts_prefetch(m->experts, s->prefetch_layer, L);" "tr_experts_prefetch(m->experts, s->prefetch_layer, -1);"
run "the prompt does not wait for its reads ahead" gcc "test_prefetch" $O \
  "    if (olmoe_prefetch_drain(m, s) != 0) return -1;" "    olmoe_prefetch_drain(m, s);"
run "a failed layer leaves reads in flight" gcc "test_prefetch" $O "                s->prefetch_layer = -1;
                olmoe_prefetch_drain(m, s);
                return -1;" "                s->prefetch_layer = -1;
                return -1;"
run "TR_PREFETCH=0 ignored" gcc "test_prefetch" $O \
  "(want_prefetch == NULL || strcmp(want_prefetch, \"0\") != 0)" "(want_prefetch == NULL || 1)"

# threads.c: tr_thread and tr_monitor
run "join does not wait" gcc "test_base" $T "    WaitForSingleObject(t->handle, INFINITE);
    CloseHandle(t->handle);
#else
    pthread_join(t->handle, NULL);" "    WaitForSingleObject(t->handle, INFINITE);
    CloseHandle(t->handle);
#else
    pthread_detach(t->handle);"
run "broadcast wakes nobody" gcc "test_base" $T "    pthread_cond_broadcast(&m->cond);
#endif
}" "    (void)m;
#endif
}"
run "the thread never runs its function" gcc "test_base" $T "    tr_thread *t = (tr_thread *)arg;
    t->fn(t->arg);
    return NULL;" "    (void)arg;
    return NULL;"
}

# make, in the current directory, with a flavour's sanitizer flags ($1: gcc, asan or tsan)
mk() {
  fl=$1
  shift
  case $fl in
    asan) EXTRA_CFLAGS="-O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all" \
            EXTRA_LDFLAGS="-fsanitize=address,undefined" make BUILD=b CC=gcc "$@" ;;
    tsan) EXTRA_CFLAGS="-O1 -g -fsanitize=thread" EXTRA_LDFLAGS="-fsanitize=thread" make BUILD=b CC=gcc "$@" ;;
    *) make BUILD=b CC=gcc "$@" ;;
  esac
}

# $1: name, $2: flavour, $3: tests, $4: file, $5: text, $6: replacement
run() {
  [ -z "$ONLY" ] || case "$1" in "no mutation"|*"$ONLY"*) ;; *) return 0 ;; esac
  rm -rf /tmp/mutp && cp -a /tmp/mutp-$2 /tmp/mutp
  cd /tmp/mutp
  $PYBIN /tmp/mutp.py "$4" "$5" "$6"
  line="$1 ($2):"
  for t in $3; do
    if ! mk $2 b/tests/$t > /tmp/mutp.log 2>&1; then line="$line $t=does-not-build"; continue; fi
    rc=0
    if [ "$2" = tsan ]; then
      TSAN_OPTIONS=halt_on_error=1 timeout 60 setarch "$(uname -m)" -R ./b/tests/$t > /tmp/mutp.out 2>&1 || rc=$?
    else
      timeout 60 ./b/tests/$t > /tmp/mutp.out 2>&1 || rc=$?
    fi
    if [ $rc = 0 ]; then
      line="$line $t=green"
    else
      what=$(grep -o 'test_[a-z]*\.c:[0-9]*: check failed' /tmp/mutp.out | sed 's/: check failed//' | sort -u -t: -k2n | tr '\n' ' ')
      [ $rc = 124 ] && what="$what hang"
      grep -q "ThreadSanitizer" /tmp/mutp.out && what="$what tsan"
      grep -q "AddressSanitizer\|runtime error" /tmp/mutp.out && what="$what asan"
      line="$line $t=RED [$what]"
    fi
  done
  echo "$line"
  cd /src
}
main "$@"; exit
