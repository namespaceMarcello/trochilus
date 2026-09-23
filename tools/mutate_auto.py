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
killed mutants it could not see, docs/LESSONS.md #115).

Linux container, from the repo root; the tree is copied to /tmp, the checkout is never touched:
  MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local \\
      python3 tools/mutate_auto.py src/format/gguf.c test_gguf [more tests...]
Options: --jobs N (default: half the processors), --lines A-B (only those lines), --list (print
the mutants, build nothing), --asan (build with AddressSanitizer and UBSan: a mutant that reads
out of bounds is then killed, not lucky), --cmd "SHELL COMMAND" (repeatable: run after the tests,
in the worker's tree, with b/trochilus built; a mutant that makes it exit non-zero is killed --
the C tests of a model check that every split of the work gives the same bits, not that the bits
are right, so a model file wants the oracle too:
  --cmd '$PY /src/tools/oracle.py /src/fixtures/tiny-olmoe /src/fixtures/tiny-olmoe/model-f32.gguf
         --binary b/trochilus --expect exact'
). Output: one line per survivor (file:line, mutation, the line), then killed / survived / same
object / did not build / timed out. Exit status 0 whatever survives: this is a report.
"""
import argparse
import os
import re
import shutil
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
        if lines_range and not lines_range[0] <= no <= lines_range[1]:
            continue
        for name, pat, rep in OPERATORS:
            for m in re.finditer(pat, masked):
                new = line[:m.start()] + rep + line[m.end():]
                yield no, name, line, "\n".join(lines[:no - 1] + [new] + lines[no:])


def copy_tree(dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for d in ("src", "tests"):
        shutil.copytree(d, os.path.join(dst, d))
    shutil.copy("Makefile", dst)


def run(cmd, cwd, timeout):
    try:
        return subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("tests", nargs="+")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--lines")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--asan", action="store_true")
    ap.add_argument("--cmd", action="append", default=[])
    a = ap.parse_args()
    rng = tuple(int(x) for x in a.lines.split("-")) if a.lines else None
    todo = list(mutants(a.file, rng))
    if a.list:
        for no, name, line, _ in todo:
            print(f"{a.file}:{no}: {name}: {line.strip()}")
        print(f"{len(todo)} mutants")
        return 0
    bins = [f"b/tests/{t}" for t in a.tests]
    targets = bins + (["b/trochilus"] if a.cmd else [])
    checks = [[f"./{b}"] for b in bins] + [["sh", "-c", c] for c in a.cmd]
    # -g0: debug info records columns, and a mutated line would differ even where the code does not
    flags = "-g0" + (" -O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all"
                     if a.asan else "")
    mk = ["make", "BUILD=b", "CC=gcc", f"EXTRA_CFLAGS={flags}"]
    if a.asan:
        mk.append("EXTRA_LDFLAGS=-fsanitize=address,undefined")
    obj = "b/" + a.file[:-2] + ".o"
    original = open(a.file, encoding="utf-8").read()

    # every worker gets its own tree, built once: a mutant then recompiles one object and relinks
    dirs = [f"/tmp/mutate_auto/{w}" for w in range(a.jobs)]
    t0 = time.time()
    for d in dirs:
        copy_tree(d)
    base = [subprocess.run(mk + ["-j4"] + targets, cwd=d, capture_output=True) for d in dirs]
    if any(b.returncode for b in base):
        sys.exit("mutate_auto: the unmutated tree does not build:\n" + base[0].stderr.decode()[-2000:])
    with open(os.path.join(dirs[0], obj), "rb") as f:
        base_obj = f.read()
    t1 = time.time()
    for c in checks:
        if run(c, dirs[0], None) != 0:
            sys.exit(f"mutate_auto: {' '.join(c)} fails without any mutation")
    budget = max(20.0, 10 * (time.time() - t1))

    results, flaky, lock = [], [], threading.Lock()

    def worker(w):
        d = dirs[w]
        target = os.path.join(d, a.file)
        for k in range(w, len(todo), a.jobs):
            no, name, line, text = todo[k]
            with open(target, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            if run(mk + targets, d, 300) != 0:
                verdict = "nobuild"
            elif open(os.path.join(d, obj), "rb").read() == base_obj:
                verdict = "same"
            else:
                verdict = "survived"
                for c in checks:
                    rc = run(c, d, budget)
                    if rc is None:
                        verdict = "timeout"
                        break
                    # a kill must happen twice: a check that fails once under load and passes
                    # again killed mutants it cannot see (docs/LESSONS.md #115); it is reported
                    # and the next checks decide
                    if rc != 0 and run(c, d, budget) != 0:
                        verdict = "killed"
                        break
                    if rc != 0:
                        with lock:
                            flaky.append((no, name, " ".join(c)))
            with lock:
                results.append((no, name, line, verdict))
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(original)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(a.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    shutil.rmtree("/tmp/mutate_auto", ignore_errors=True)

    count = {v: 0 for v in ("killed", "survived", "same", "nobuild", "timeout")}
    for no, name, line, verdict in sorted(results):
        count[verdict] += 1
        if verdict == "survived":
            print(f"SURVIVED {a.file}:{no}: {name}: {line.strip()}")
    for no, name, check in sorted(flaky):
        print(f"FLAKY {check}: failed once, then passed, on {a.file}:{no}: {name}")
    print(f"mutate_auto {a.file} vs {' '.join(a.tests + [f'[{c}]' for c in a.cmd])}: {len(results)} mutants, {count['killed']} killed, "
          f"{count['survived']} survived, {count['same']} same object, {count['nobuild']} did not build, "
          f"{count['timeout']} timed out, {len(flaky)} flaky failures ({time.time() - t0:.0f} s, {a.jobs} jobs"
          f"{', ASan' if a.asan else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
