#!/usr/bin/env python3
"""mutate_auto.py -- which mutations of a C file do its tests fail to see?

A test never seen red proves nothing (docs/LESSONS.md #43). The tools/mutate_*.sh scripts try
mutations written by hand for one area; this one generates them for any file: each relational,
equality and logical operator swapped for its neighbour, each `+ 1` / `- 1` made `+ 0` / `- 0`,
each `return 0;` / `return -1;` swapped, each `(!` negation dropped. Every mutant is built and
the given tests run on it. A mutant whose object file comes out byte-identical changes nothing
(a branch this platform does not compile, or code the compiler folds to the same) and is set aside
without running anything. A mutant that builds, differs and passes every test SURVIVED: either no
test looks at that line, or the mutation is equivalent in a way the compiler cannot see. Read each.
A check that fails is run again on the same mutant: only a second failure kills it; a failure that
does not repeat is printed as FLAKY and the next checks decide (a check that fails under load
killed mutants it could not see, docs/LESSONS.md #115). A check that runs out of time is run
again with three times the budget, and a mutant that times out twice is listed as TIMEOUT, to be
read like a survivor (under load a timeout hid three survivors, docs/LESSONS.md #117). A failure
that is the machine's memory (the engine's guard refusing a model, the OOM killer) is no verdict:
that mutant is judged again at the end, alone, and listed as PRESSURE if refused again (a memory
refusal that lasted killed a mutant twice, docs/LESSONS.md #119).

Linux container, from the repo root; the tree is copied to /tmp, the checkout is never touched:
  MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local \\
      python3 tools/mutate_auto.py src/format/gguf.c test_gguf [more tests...]
Options: --jobs N (default: half the processors), --lines A-B[,C-D...] (only those lines),
--changed REF (only the lines that differ from git REF, e.g. HEAD: after a change, its own lines
in minutes instead of the whole file), --list (print
the mutants, build nothing), --asan (build with AddressSanitizer and UBSan: a mutant that reads
out of bounds is then killed, not lucky), --cmd "SHELL COMMAND" (repeatable: run after the tests,
in the worker's tree, with b/trochilus built; a mutant that makes it exit non-zero is killed --
the C tests of a model check that every split of the work gives the same bits, not that the bits
are right, so a model file wants the oracle too:
  --cmd '$PY /src/tools/oracle.py /src/fixtures/tiny-olmoe /src/fixtures/tiny-olmoe/model-f32.gguf
         --binary b/trochilus --expect exact'
). Output: one line per survivor and per timeout (file:line, mutation, the line), then killed / survived / same
object / did not build / timed out. Exit status 0 whatever survives: this is a report.
Speed: the checks run fastest first, each with ten times its own time as budget, and with
TR_TEST_FAILFAST=1 (tests/test.h: a C test stops at its first failed check); stderr gets one line
per mutant judged, with how long is left. Three more, 2026-09-24:
- a coverage pass first (gcov, -O0, every check once, beside the builds): a mutant on a line of
  code no check runs cannot be killed, and it was the dearest one (every check to its end, twice
  under ASan's slowness); it is listed UNCOVERED, not built (--no-coverage: judge them all);
- with --asan, the plain build judges every mutant and the sanitizers only its survivors: a kill
  is a kill in either build, and the plain one runs its checks 2-4x faster;
- the trees are built once and copied (their mtimes kept, so make finds them up to date), not
  built once per job.
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

# (name, pattern on the line with comments and strings blanked, replacement)
OPERATORS = [
    ("<= to <", r"(?<![<>=!])<=(?!=)", "<"),
    ("< to <=", r"(?<![<\-])<(?![<=])", "<="),
    (">= to >", r"(?<![<>=!])>=(?!=)", ">"),
    ("> to >=", r"(?<![>\-])>(?![>=])", ">="),
    ("== to !=", r"==", "!="),
    ("!= to ==", r"!=", "=="),
    ("&& to ||", r"&&", "||"),
    ("|| to &&", r"\|\|", "&&"),
    ("+ 1 to + 0", r"\+ 1\b(?!\.)", "+ 0"),
    ("- 1 to - 0", r"(?<![=(,]) - 1\b(?!\.)", " - 0"),
    ("return 0 to -1", r"\breturn 0;", "return -1;"),
    ("return -1 to 0", r"\breturn -1;", "return 0;"),
    ("drop (!", r"\(!(?!=)", "("),
]


def masked_lines(text):
    """The file's lines with comments, string and character literals blanked (same length), and
    preprocessor lines blanked whole: mutations are looked for in code only."""
    out, in_block = [], False
    for line in text.split("\n"):
        chars, i, n = list(line), 0, len(line)
        if not in_block and line.lstrip().startswith("#"):
            out.append(" " * n)
            continue
        while i < n:
            if in_block:
                end = line.find("*/", i)
                stop = n if end < 0 else end + 2
                chars[i:stop] = " " * (stop - i)
                in_block, i = end < 0, stop
            elif line.startswith("/*", i):
                in_block = True
            elif line.startswith("//", i):
                chars[i:] = " " * (n - i)
                break
            elif line[i] in "\"'":
                q, j = line[i], i + 1
                while j < n and line[j] != q:
                    j += 2 if line[j] == "\\" else 1
                chars[i:j + 1] = " " * (min(j + 1, n) - i)
                i = j + 1
            else:
                i += 1
        out.append("".join(chars))
    return out


def mutants(path, lines_range):
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    for no, (line, masked) in enumerate(zip(lines, masked_lines(text)), 1):
        if lines_range and not any(lo <= no <= hi for lo, hi in lines_range):
            continue
        for name, pat, rep in OPERATORS:
            for m in re.finditer(pat, masked):
                new = line[:m.start()] + rep + line[m.end():]
                yield no, name, line, "\n".join(lines[:no - 1] + [new] + lines[no:])


def changed_lines(path, ref):
    """The line ranges of path, as it is now, that differ from git ref (git diff -U0): the lines
    a change touched, which a run after that change needs to mutate."""
    out = subprocess.run(["git", "-c", "safe.directory=*", "diff", "-U0", ref, "--", path],
                         capture_output=True, text=True, check=True).stdout
    rng = []
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", out, re.M):
        start, n = int(m.group(1)), int(m.group(2) or "1")
        if n > 0:
            rng.append((start, start + n - 1))
    return rng


def copy_tree(dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for d in ("src", "tests"):
        shutil.copytree(d, os.path.join(dst, d))
    shutil.copy("Makefile", dst)


# a failure that says the machine had no room is no verdict on the mutant: the engine's memory
# guard refusing a model, or the OOM killer (docs/LESSONS.md #115, #119)
PRESSURE = re.compile(rb"not enough memory")

# a C test stops at its first failed check (tests/test.h): most mutants are killed, and a kill
# then costs the time to the first failure, not the whole test
CHECK_ENV = dict(os.environ, TR_TEST_FAILFAST="1")


def run(cmd, cwd, timeout):
    """Exit status (None when out of time), and whether a failure was the machine's memory.
    The check runs in a session of its own, and out of time the whole session is killed: a test
    that runs the binary (test_cli, through system()) left its looping child running for hours,
    one per timed-out mutant, and they slowed every file after it (docs/LESSONS.md #122)."""
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                         env=CHECK_ENV)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)  # the leader is not reaped yet: its group id is still its own
        p.communicate()
        return None, False
    pressure = p.returncode != 0 and (p.returncode in (-9, 137) or PRESSURE.search(out[-65536:]) is not None)
    return p.returncode, pressure


def left_behind(dirs):
    """Processes still running with their working directory in one of the worker trees."""
    pids = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if any(cwd == d or cwd.startswith(d + "/") for d in dirs):
            pids.append(int(pid))
    return pids


def put(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


ROOT = "/tmp/mutate_auto"


def copy_built(src, dst):
    """A worker's tree as a copy of one already built: copy2 keeps every mtime, so make finds every
    object up to date, and the build of every target is paid once instead of once per job."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def phase(jobs, suffix, make, targets, checks, obj):
    """The trees of one build (plain, or with the sanitizers): tree 0 built, the others copies of
    it made before any check writes in it; the unmutated object; and the checks, fastest first,
    each with ten times its own time on the unmutated tree as budget (at least 20 s), not ten times
    all of them together: a timeout of a 1-minute test cost 53 minutes of a job when the budget was
    the sum of the test and two oracles; and the fastest check runs first, so a mutant that dies
    dies soon."""
    dirs = [f"{ROOT}/{w}{suffix}" for w in range(jobs)]
    copy_tree(dirs[0])
    b = subprocess.run(make + ["-j8"] + targets, cwd=dirs[0], capture_output=True)
    if b.returncode:
        sys.exit("mutate_auto: the unmutated tree does not build:\n" + b.stderr.decode()[-2000:])
    for d in dirs[1:]:
        copy_built(dirs[0], d)
    with open(os.path.join(dirs[0], obj), "rb") as f:
        base_obj = f.read()
    timed = []
    for c in checks:
        t = time.time()
        if run(c, dirs[0], None)[0] != 0:
            sys.exit(f"mutate_auto: {' '.join(c)} fails without any mutation")
        timed.append((time.time() - t, c))
    timed.sort(key=lambda tc: tc[0])
    return {"dirs": dirs, "make": make, "base_obj": base_obj, "checks": [c for _, c in timed],
            "budgets": [max(20.0, 10 * t) for t, _ in timed]}


# gcov's first field on a line: "-" no code, "#####" code never run, else a count
GCOV_LINE = re.compile(r"^\s*([^:]+):\s*(\d+):")


def uncovered_lines(path, targets, checks):
    """The lines of path that are code no check runs, from one pass of every check over a tree
    built with --coverage at -O0 (where gcov gives each line its own count). A line gcov calls no
    code ("-": a declaration, an initializer, the rest of a statement) is not in the set, so its
    mutants are judged. None when the pass cannot be trusted: a check that fails in that build, or
    no report from gcov. A line only a race or a timing reaches (the pool's spin) may be missed by
    one pass: its mutant is listed UNCOVERED, to be read like a survivor, never counted killed."""
    d = f"{ROOT}/cov"
    copy_tree(d)
    mk = ["make", "BUILD=b", "CC=gcc", "EXTRA_CFLAGS=-g0 -O0 --coverage", "EXTRA_LDFLAGS=--coverage", "-j8"]
    if subprocess.run(mk + targets, cwd=d, capture_output=True).returncode != 0:
        return None
    for c in checks:
        if run(c, d, None)[0] != 0:
            return None
    g = subprocess.run(["gcov", "-o", os.path.join("b", os.path.dirname(path)), path], cwd=d,
                       capture_output=True, text=True)
    report = os.path.join(d, os.path.basename(path) + ".gcov")
    if g.returncode != 0 or not os.path.exists(report):
        return None
    zero = set()
    with open(report, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = GCOV_LINE.match(line)
            if m and m.group(1).strip() == "#####":
                zero.add(int(m.group(2)))
    shutil.rmtree(d, ignore_errors=True)
    return zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("tests", nargs="+")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--lines")
    ap.add_argument("--changed", metavar="REF")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--asan", action="store_true")
    ap.add_argument("--cmd", action="append", default=[])
    ap.add_argument("--no-coverage", action="store_true")
    a = ap.parse_args()
    rng = [tuple(int(x) for x in r.split("-")) for r in a.lines.split(",")] if a.lines else None
    if a.changed:
        rng = changed_lines(a.file, a.changed)
        if not rng:
            print(f"mutate_auto {a.file}: no line differs from {a.changed}")
            return 0
    todo = list(mutants(a.file, rng))
    if a.list:
        for no, name, line, _ in todo:
            print(f"{a.file}:{no}: {name}: {line.strip()}")
        print(f"{len(todo)} mutants")
        return 0
    bins = [f"b/tests/{t}" for t in a.tests]
    targets = bins + (["b/trochilus"] if a.cmd else [])
    all_checks = [[f"./{b}"] for b in bins] + [["sh", "-c", c] for c in a.cmd]
    obj = "b/" + a.file[:-2] + ".o"
    original = open(a.file, encoding="utf-8").read()
    t0 = time.time()

    # the lines no check runs (gcov): their mutants cannot be killed, and a survivor was the dearest
    # mutant, every check run to its end; they are listed, not built. The pass runs beside the
    # builds of the trees below (its budgets only grow by it: ten times a check's time under load)
    cov = {}
    cov_thread = None
    if not a.no_coverage:
        cov_thread = threading.Thread(target=lambda: cov.update(zero=uncovered_lines(a.file, targets, all_checks)))
        cov_thread.start()

    # -g0: debug info records columns, and a mutated line would differ even where the code does not
    plain = phase(a.jobs, "", ["make", "BUILD=b", "CC=gcc", "EXTRA_CFLAGS=-g0"], targets, all_checks, obj)
    # with --asan the sanitizers judge only what the plain build lets live: a kill there is a kill
    # (a check failed twice), and the plain build runs its checks 2-4x faster
    asan = phase(a.jobs, "a", ["make", "BUILD=b", "CC=gcc",
                               "EXTRA_CFLAGS=-g0 -O1 -fno-omit-frame-pointer -fsanitize=address,undefined "
                               "-fno-sanitize-recover=all", "EXTRA_LDFLAGS=-fsanitize=address,undefined"],
                 targets, all_checks, obj) if a.asan else None

    uncovered = None
    if cov_thread is not None:
        cov_thread.join()
        uncovered = cov.get("zero")
        if uncovered is None:
            print("mutate_auto: the coverage pass could not be trusted (a check failed or gcov had no "
                  "report): every mutant is judged", file=sys.stderr, flush=True)
    results = {}
    run_k = []
    for k, (no, _, _, _) in enumerate(todo):
        if uncovered is not None and no in uncovered:
            results[k] = "uncovered"
        else:
            run_k.append(k)

    flaky, lock = [], threading.Lock()
    progress = {"done": 0, "t": time.time()}

    def tick(k, verdict, why):
        """One line on stderr per mutant judged: how many, how long is left, and the verdict with
        the check that gave it -- a kill can be audited afterwards (a mutant that passes every check
        alone had been reported killed, docs/LESSONS.md #125)."""
        progress["done"] += 1
        n, el = progress["done"], time.time() - progress["t"]
        left = el / n * (len(run_k) - n)
        print(f"{n}/{len(run_k)} judged, {el / 60:.1f} min, about {left / 60:.0f} min left | "
              f"{a.file}:{todo[k][0]}: {todo[k][1]}: {verdict}{' by ' + why if why else ''}", file=sys.stderr,
              flush=True)

    def judge(ph, d):
        """The verdict on the mutant written in tree d of phase ph, the checks that failed only
        once, and the check that decided it with its exit statuses."""
        if run(ph["make"] + targets, d, 300)[0] != 0:
            return "nobuild", [], ""
        if open(os.path.join(d, obj), "rb").read() == ph["base_obj"]:
            return "same", [], ""
        once = []
        for c, budget in zip(ph["checks"], ph["budgets"]):
            name = " ".join(c)[-60:]
            rc, pressure = run(c, d, budget)
            # a timeout too must happen twice, with room: under load a check that would pass
            # ran out of time, and a survivor hid behind it (docs/LESSONS.md #117)
            if rc is None:
                rc, pressure = run(c, d, 3 * budget)
            if rc is None:
                return "timeout", once, name
            if pressure:
                return "pressure", once, f"{name} ({rc})"
            # a kill must happen twice: a check that fails once under load and passes again
            # killed mutants it cannot see (docs/LESSONS.md #115); it is reported and the next
            # checks decide
            if rc != 0:
                rc2, pressure = run(c, d, budget)
                if pressure:
                    return "pressure", once, f"{name} ({rc}, {rc2})"
                if rc2 != 0:
                    return "killed", once, f"{name} ({rc}, {rc2})"
                once.append(" ".join(c))
        return "survived", once, ""

    def verdict_of(w, k):
        """Mutant k in worker w's trees: the plain build first, the sanitizers on what it lets live."""
        d = plain["dirs"][w]
        put(os.path.join(d, a.file), todo[k][3])
        verdict, once, why = judge(plain, d)
        if asan is not None and verdict == "survived":
            da = asan["dirs"][w]
            put(os.path.join(da, a.file), todo[k][3])
            v2, once2, why2 = judge(asan, da)
            once += once2
            # an object the sanitizers' build compiles the same is the plain build's survivor
            if v2 != "same":
                verdict, why = v2, (why2 + " under ASan" if why2 else "")
        return verdict, once, why

    def worker(w):
        for k in run_k[w::a.jobs]:
            verdict, once, why = verdict_of(w, k)
            with lock:
                results[k] = verdict
                flaky.extend((todo[k][0], todo[k][1], c) for c in once)
                tick(k, verdict, why)
        for ph in (plain, asan):
            if ph is not None:
                put(os.path.join(ph["dirs"][w], a.file), original)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(a.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # the mutants the machine had no room for are judged again one at a time, nothing else of
    # ours running; one still refused for memory is listed, to be read like a survivor. A mutant
    # can make the engine refuse by itself (it broke the memory plan, whose refusal reads like the
    # machine's, docs/LESSONS.md #123), but a refusal alone does not prove it: counting such a
    # refusal as a kill once the unmutated tree had passed alone gave kills that did not repeat
    # (docs/LESSONS.md #125)
    pending = sorted(k for k, v in results.items() if v == "pressure")
    t_alone = time.time()
    for i, k in enumerate(pending, 1):
        verdict, once, why = verdict_of(0, k)
        results[k] = verdict
        flaky.extend((todo[k][0], todo[k][1], c) for c in once)
        print(f"{i}/{len(pending)} judged again alone (refused for memory), {(time.time() - t_alone) / 60:.1f} min | "
              f"{a.file}:{todo[k][0]}: {todo[k][1]}: {results[k]}{' by ' + why if why else ''}", file=sys.stderr,
              flush=True)
    # nothing of ours may outlive the run (docs/LESSONS.md #122): counted in the last line, and
    # stopped
    orphans = left_behind([ROOT])
    for pid in orphans:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    shutil.rmtree(ROOT, ignore_errors=True)

    count = {v: 0 for v in ("killed", "survived", "same", "nobuild", "timeout", "pressure", "uncovered")}
    listed = {"survived": "SURVIVED", "timeout": "TIMEOUT", "pressure": "PRESSURE", "uncovered": "UNCOVERED"}
    for k in sorted(results, key=lambda k: todo[k][0]):
        no, name, line, _ = todo[k]
        count[results[k]] += 1
        if results[k] in listed:
            print(f"{listed[results[k]]} {a.file}:{no}: {name}: {line.strip()}")
    for no, name, check in sorted(flaky):
        print(f"FLAKY {check}: failed once, then passed, on {a.file}:{no}: {name}")
    print(f"mutate_auto {a.file} vs {' '.join(a.tests + [f'[{c}]' for c in a.cmd])}: {len(results)} mutants, {count['killed']} killed, "
          f"{count['survived']} survived, {count['uncovered']} on lines no check runs, {count['same']} same object, "
          f"{count['nobuild']} did not build, "
          f"{count['timeout']} timed out, {count['pressure']} refused for memory, {len(flaky)} flaky failures, "
          f"{len(pending)} judged again alone ({(time.time() - t_alone) / 60:.0f} min), "
          f"{len(orphans)} processes left behind "
          f"({time.time() - t0:.0f} s, {a.jobs} jobs{', ASan on the plain build survivors' if a.asan else ''}"
          f"{'' if uncovered is not None else ', no coverage pass'})")
    return 1 if orphans else 0


if __name__ == "__main__":
    sys.exit(main())
