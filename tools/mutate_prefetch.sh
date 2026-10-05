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
B=src/memory/experts.c
O=src/models/olmoe.c
T=src/base/threads.c
# every mutation is first applied to a copy without a build (seconds): a text the code no longer has stops the
# script here, not three builds and half an hour later at its own line (docs/LESSONS.md #283)
DRY=1
mutations
echo "every mutation applies once"
[ "${MUTATE_DRY:-0}" = 1 ] && return 0 # MUTATE_DRY=1 (docker run -e): only this pass
DRY=0
for fl in gcc asan tsan; do
  rm -rf /tmp/mutp-$fl && mkdir -p /tmp/mutp-$fl && cp -r Makefile src tests /tmp/mutp-$fl/
  (cd /tmp/mutp-$fl && mk $fl -j4 $(for t in $TESTS; do echo b/tests/$t; done) > /tmp/mutp-$fl.log 2>&1) ||
    { echo "the unmutated tree does not build ($fl)"; tail -20 /tmp/mutp-$fl.log; exit 1; }
done
mutations
}

mutations() {
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
run "a slot in flight may be a victim (the LRU)" asan "test_experts test_prefetch" $B \
  "        if (x->pending != NULL && x->pending[s]) continue;
        int32_t u = x->slots[s].unit;
        if (u != -1) {" "        int32_t u = x->slots[s].unit;
        if (u != -1) {"
run "a slot in flight may be a victim (ds4's)" asan "test_experts test_prefetch test_hot" $B \
  "        if (x->pending != NULL && x->pending[s]) continue;
        if (x->slots[s].pick == x->call_seq) continue;" "        if (x->slots[s].pick == x->call_seq) continue;"
# (not "the read ahead does not stop when no victim is left", break -> continue: equivalent, a victim's
# absence does not depend on the unit, so every later turn fails the same way and changes nothing; #283)

# counting and failures
run "read ahead not counted as prefetched" gcc "test_experts test_prefetch test_hot" $B \
  "    else x->stats.prefetched++;
" ""
run "read ahead not counted as a miss" gcc "test_experts test_prefetch" $B "    x->stats.misses++;
    if (job.demand) x->stats.arrived++;" "    if (job.demand) x->stats.arrived++;"
run "the wait time is not measured" gcc "test_experts" $B "        x->stats.prefetch_wait_sec += tr_time_sec() - t0;
" ""
run "a failed read ahead is not reported" gcc "test_experts test_prefetch" $B "        if (!job.demand) x->prefetch_failed = 1; /* a call's own unit fails that call (tr_experts_acquire_take) */
" ""
run "a failure is reported forever" gcc "test_experts" $B "    x->prefetch_failed = 0;
    return failed ? -1 : 0;" "    return failed ? -1 : 0;"
run "a failed read ahead keeps its unit" gcc "test_experts test_prefetch" $B "    if (job.rc != 0) {
        x->slot_of[x->slots[s].unit] = -1;
        x->slots[s].unit = -1;
        lru_unlink(x, s);" "    if (job.rc != 0) {
        lru_unlink(x, s);"
run "a failed demand read stays hot" gcc "test_experts" $B "    if (rc != 0) {
        lru_unlink(x, victim);
        lru_push_back(x, victim);
        return -1;" "    if (rc != 0) {
        return -1;"

# the I/O thread
run "the I/O thread reads the next expert" gcc "test_experts test_prefetch" $B \
  "job.rc = read_parts(x, job.read, job.read_ctx, job.layer, job.id, base" \
  "job.rc = read_parts(x, job.read, job.read_ctx, job.layer, (job.id + 1) % x->n_expert, base"
run "the I/O thread never says done" gcc "test_experts" $B "            j->done = 1;" "            j->done = 0;"
run "the I/O thread publishes without the lock" tsan "test_experts test_prefetch" $B "        tr_monitor_lock(x->mon);
        for (int64_t e = 0; e < k; e++) {" "        for (int64_t e = 0; e < k; e++) {"
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
# olmoe.c and experts.c: a pass's read ahead, in short requests, and what its next layer drops
run "a pass reads nothing ahead" gcc "test_prefetch" $O "        s->prefetch_layer = L + 1 < m->n_layers ? L + 1 : -1;
        s->prefetch_pass = 1;" "        s->prefetch_layer = -1;
        s->prefetch_pass = 1;"
run "a pass's gate ignored (a decode token reads ahead)" gcc "test_prefetch" $O \
  "if (s->prefetch_pass && 2 * n_ids > n_expert)" "if (s->prefetch_pass && n_ids > 0)"
run "a layer-major prompt takes the pass rule (short requests)" gcc "test_prefetch" $O \
  "            s->prefetch_pass = 0;" "            s->prefetch_pass = 1;"
run "a pass's read ahead never dropped" gcc "test_prefetch" $O \
  "if (s->cancel_layer == L) tr_experts_prefetch_cancel(" "if (0) tr_experts_prefetch_cancel("
run "a pass does not wait for its reads ahead" gcc "test_prefetch" $O \
  "    if (s->ahead_issued && olmoe_prefetch_drain(m, s) != 0) return -1;" "    (void)0;"
run "a failed pass leaves reads in flight" gcc "test_prefetch" $O "            if (s->ahead_issued) olmoe_prefetch_drain(m, s);
" ""
run "cancel drops nothing" gcc "test_prefetch" $B "        int asked = j->layer != layer;" "        int asked = 1;"
run "cancel drops the units asked for" gcc "test_prefetch" $B \
  "        for (int64_t k = 0; k < n && !asked; k++) asked = ids[k] == j->id;" ""
run "a dropped unit stays in the index" gcc "test_prefetch" $B "cold end, as a failed read leaves it (prefetch_take) */
        x->slot_of[x->slots[s].unit] = -1;
        x->slots[s].unit = -1;" "cold end, as a failed read leaves it (prefetch_take) */"
run "the read ahead returns before its first run is taken" gcc "test_experts test_prefetch" $B \
  "    while (x->q_len >= queued) tr_monitor_wait(x->mon);" ""
run "the short requests ignored" gcc "test_prefetch" $B \
  " && (job.cap == 0 || left < job.cap)) {" ") {"
run "TR_PREFETCH=0 ignored" gcc "test_prefetch" $O \
  "(want_prefetch == NULL || strcmp(want_prefetch, \"0\") != 0)" "(want_prefetch == NULL || 1)"
# the prompt's routings told the store's eviction (olmoe_refresh_experts, test_prefetch's heat)
run "the routings not passed" gcc "test_prefetch" $O \
  "tr_experts_acquire_counts(m->experts, L, s->acquire_ids, counts, n_tok, n_ids)" \
  "tr_experts_acquire_counts(m->experts, L, s->acquire_ids, NULL, n_tok, n_ids)"
run "a pass's tokens counted as one" gcc "test_prefetch" $O \
  "tr_experts_acquire_counts(m->experts, L, s->acquire_ids, counts, n_tok, n_ids)" \
  "tr_experts_acquire_counts(m->experts, L, s->acquire_ids, counts, 1, n_ids)"
run "every expert's rows counted one" gcc "test_prefetch" $O \
  "            counts[n_ids] = s->offsets[e + 1] - s->offsets[e];" "            counts[n_ids] = 1;"

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
  if [ "$DRY" = 1 ]; then
    cp "$4" /tmp/mutp-dry && $PYBIN /tmp/mutp.py /tmp/mutp-dry "$5" "$6" || { echo "($1)"; exit 1; }
    return 0
  fi
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
