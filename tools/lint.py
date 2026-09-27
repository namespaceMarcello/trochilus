#!/usr/bin/env python3
"""Project lint: the lessons in docs/LESSONS.md that a script can check.

Each check names the lesson it enforces. Exit 1 on any failure.

  python tools/lint.py            run every check
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
failures = []


def fail(lesson, msg):
    failures.append(f"[LESSONS #{lesson}] {msg}")


def check_docs_control_chars():
    """#14: backslash sequences written through shell or Python become tabs or carriage returns."""
    files = [ROOT / "CLAUDE.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for f in files:
        data = f.read_bytes()
        for name, byte in (("tab", b"\t"), ("carriage return", b"\r")):
            if byte in data:
                line = data[: data.index(byte)].count(b"\n") + 1
                fail(14, f"{f.relative_to(ROOT)}:{line}: contains a {name}")


def check_line_endings():
    """#95: an agent rewrote four source files with Windows line endings; with core.autocrlf off
    git then shows the whole file as changed and every later text replacement on LF misses."""
    suffixes = {".c", ".h", ".py", ".sh", ".lib", ".awk", ".json", ".txt", ".ps1", ".md", ".cjs"}
    files = [ROOT / "Makefile", ROOT / "CLAUDE.md"]
    # docs too: #168, a Python edit turned three documents to CRLF and only the size of STATUS saw it
    for top in ("src", "tests", "tools", "bench", ".claude", "docs"):
        files += [f for f in sorted((ROOT / top).rglob("*")) if f.is_file() and f.suffix in suffixes
                  and ".venv" not in f.parts and "results" not in f.parts]  # bench/results: run output
    for f in files:
        data = f.read_bytes()
        if b"\r" in data:
            line = data[: data.index(b"\r")].count(b"\n") + 1
            fail(95, f"{f.relative_to(ROOT)}:{line}: carriage return (the file must end its lines with LF)")


def check_doc_limits():
    """struttura-repo: CLAUDE.md is a map (<= 200 lines), STATUS.md does not grow (<= 40 KB)."""
    claude = ROOT / "CLAUDE.md"
    n = claude.read_text(encoding="utf-8").count("\n")
    if n > 200:
        fail("structure", f"CLAUDE.md has {n} lines (limit 200)")
    stato = ROOT / "docs" / "STATUS.md"
    if stato.stat().st_size > 40_000:
        fail("structure", f"docs/STATUS.md is {stato.stat().st_size} bytes (limit 40 KB)")


def check_no_future_dates():
    """#216: a day's work was dated from the story ("the evening of the 26th, so tomorrow") instead of the
    machine's clock: every document of that day a day late. No date in the documents may be after today."""
    import datetime
    today = datetime.date.today().isoformat()
    files = [ROOT / "CLAUDE.md", ROOT / "docs" / "status.json", *sorted((ROOT / "docs").rglob("*.md"))]
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for d in re.findall(r"\b20\d\d-[01]\d-[0-3]\d\b", line):
                if d > today:
                    fail(216, f"{f.relative_to(ROOT)}:{n}: {d} is after today ({today})")


def check_lessons_table():
    """The lessons table stays machine-readable: 8 columns, consecutive numbers."""
    rows = [l for l in (ROOT / "docs" / "LESSONS.md").read_text(encoding="utf-8").splitlines()
            if re.match(r"^\| \d+ \|", l)]
    expected = 1
    for row in rows:
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]  # "\|" is a literal pipe
        if len(cells) != 8:
            fail("register", f"LESSONS row {cells[0]} has {len(cells)} columns, expected 8")
        if cells[0] != str(expected):
            fail("register", f"LESSONS row numbered {cells[0]}, expected {expected}")
        expected = int(cells[0]) + 1


def check_type_table():
    """#10: src/format/gguf.c type sizes must equal ggml's (gguf-py GGML_QUANT_SIZES)."""
    try:
        import gguf
    except ImportError:
        fail(10, "gguf-py not importable: run with tools/.venv python")
        return
    src = (ROOT / "src" / "format" / "gguf.c").read_text(encoding="utf-8")
    table = dict((m.group(1), (int(m.group(2)), int(m.group(3))))
                 for m in re.finditer(r'\[TR_TYPE_(\w+)\]\s*=\s*\{"\w+",\s*(\d+),\s*(\d+)\}', src))
    if not table:
        fail(10, "type table not found in src/format/gguf.c")
        return
    for name, (elems, nbytes) in table.items():
        qt = getattr(gguf.GGMLQuantizationType, name, None)
        if qt is None:
            fail(10, f"TR_TYPE_{name} has no ggml counterpart")
            continue
        ref = gguf.GGML_QUANT_SIZES.get(qt)
        if ref is None:
            continue
        if (elems, nbytes) != tuple(ref):
            fail(10, f"TR_TYPE_{name}: gguf.c says {elems}/{nbytes}, ggml says {ref[0]}/{ref[1]}")
    header = (ROOT / "src" / "format" / "gguf.h").read_text(encoding="utf-8")
    for m in re.finditer(r"TR_TYPE_(\w+) = (\d+)", header):
        qt = getattr(gguf.GGMLQuantizationType, m.group(1), None)
        if qt is not None and int(qt) != int(m.group(2)):
            fail(10, f"TR_TYPE_{m.group(1)} = {m.group(2)} in gguf.h, ggml numbers it {int(qt)}")


def check_tests_no_tmpfile():
    """#3: tmpfile() needs write access to the drive root on Windows."""
    for f in sorted((ROOT / "tests").glob("*.c")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\btmpfile\s*\(", line) and "needs write access" not in line:
                fail(3, f"{f.relative_to(ROOT)}:{i}: uses tmpfile(); write next to the test binary")


# Hot zone (docs/ARCHITECTURE.md §Hot path): code that runs for every token, between
# /* hot: begin */ and /* hot: end */. A line may allow a name with /* hot-ok: name -- reason */.
HOT_FILES = ["src/models/olmoe.c", "src/models/model.c", "src/kernels/kernels.c", "src/kernels/kernels_x86.c",
             "src/kernels/kernels_internal.h", "src/kernels/expf.c", "src/base/threads.c", "src/base/prof.h",
             "src/kv/kv.c", "src/kv/kv.h", "src/backend/gpu_attn.c", "src/backend/gpu_attn.h"]
HOT_RULES = [
    ("allocation", re.compile(r"\b(malloc|calloc|realloc|free|tr_alloc_aligned|tr_free_aligned|"
                               r"_aligned_malloc|_aligned_free|posix_memalign|aligned_alloc)\s*\(")),
    ("string or I/O", re.compile(r"\b(str[a-z]+|printf|fprintf|snprintf|sprintf|vsnprintf|puts|fputs|"
                                 r"fopen|fread|fwrite|getenv|tr_log|tr_file_\w+|tr_gguf_\w+)\s*\(")),
    ("string literal", re.compile(r'(")(?:[^"\\]|\\.)*"')),
    ("math to put in a table", re.compile(r"\b(pow|powf|cos|cosf|sin|sinf|tan|tanf|log|logf|log2|log10)\s*\(")),
    # the library's exponential: 30 ns a call with MinGW, and other bits with glibc (docs/MEASUREMENTS.md question 37)
    ("the C library's exponential (use tr_expf)", re.compile(r"\b(expf|exp|exp2f|exp2|expm1f|expm1)\s*\(")),
]
HOT_BEGIN, HOT_END = "/* hot: begin */", "/* hot: end */"


def hot_problems(text):
    """(line, message) for every hot-zone rule broken in a C source."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    code_lines = re.sub(r"/\*.*?\*/|//[^\n]*", blank, text, flags=re.S).split("\n")
    problems, inside, regions = [], False, 0
    for i, raw in enumerate(text.split("\n"), 1):
        if HOT_BEGIN in raw:
            if inside:
                problems.append((i, "hot zone opened twice"))
            inside, regions = True, regions + 1
            continue
        if HOT_END in raw:
            if not inside:
                problems.append((i, "end of a hot zone never opened"))
            inside = False
            continue
        if not inside:
            continue
        allowed, used = set(), set()
        w = re.search(r"hot-ok:\s*([^*]*?)\s+--\s+\S", raw)
        if w:
            allowed = set(w.group(1).split())
        code = code_lines[i - 1]
        if code.lstrip().startswith("#"):
            continue
        for label, rx in HOT_RULES:
            for m in rx.finditer(code):
                name = m.group(1) if m.group(1) != '"' else "string"
                if name in allowed:
                    used.add(name)
                else:
                    problems.append((i, f"{label}: {name}"))
        for name in sorted(allowed - used):
            problems.append((i, f"hot-ok for '{name}' but the line no longer uses it: remove the exception"))
    if inside:
        problems.append((len(text.split("\n")), "hot zone never closed"))
    if regions == 0:
        problems.append((1, "no hot zone marked"))
    return problems


def check_hot_zones():
    """Hot zone rules, after proving on samples that the checker sees each kind of problem."""
    b, e = HOT_BEGIN, HOT_END
    samples = [
        (f"{b}\nvoid f(void) {{ char *p = malloc(4); }}\n{e}\n", 1),
        (f"{b}\nvoid f(void) {{ strlen(s); }}\n{e}\n", 1),
        (f"{b}\nvoid f(void) {{ g(\"x\"); }}\n{e}\n", 1),
        (f"{b}\nvoid f(void) {{ y = pow(2, 3); }}\n{e}\n", 1),
        (f"{b}\nvoid f(void) {{ y = pow(2, 3); /* hot-ok: pow -- sample */ }}\n{e}\n", 0),
        (f"{b}\nvoid f(void) {{ y = 0; /* hot-ok: cos -- sample */ }}\n{e}\n", 1),
        (f"{b}\n/* malloc( in a comment */\n{e}\nvoid g(void) {{ malloc(1); }}\n", 0),
        (f"{b}\nvoid f(void) {{ y = expf(x); }}\n{e}\n", 1),
        (f"{b}\nvoid f(void) {{ y = exp((double)x); }}\n{e}\n", 1),
        (f"{b}\nvoid f(void) {{ y = tr_expf(x); }}\n{e}\n", 0),
        (f"{b}\nvoid f(void) {{ }}\n", 1),
        ("void f(void) { }\n", 1),
    ]
    for n, (text, want) in enumerate(samples, 1):
        got = len(hot_problems(text))
        if got != want:
            failures.append(f"[hot path] the check is broken: sample {n} gives {got} problems instead of {want}")
            return
    for rel in HOT_FILES:
        path = ROOT / rel
        for line, msg in hot_problems(path.read_text(encoding="utf-8")):
            failures.append(f"[hot path, docs/ARCHITECTURE.md] {rel}:{line}: {msg}")


# No global state per model (CLAUDE.md): a mutable static variable in src/ is shared by every
# model and thread in the process. Process-wide facts (CPU features, log level) are allowed on
# the line with /* global-ok: reason */.
def mutable_static(line):
    """True if the line declares a static variable whose outermost object can be written."""
    m = re.match(r"^\s*static\s+(.*)$", line)
    if not m or m.group(1).startswith("inline"):
        return False
    parts = re.split(r"[=;\[]", m.group(1), maxsplit=1)
    head = parts[0]
    if len(parts) == 1 or "(" in head:      # a function, or a declaration going on the next line
        return False
    if "*" in head:                         # the pointer itself must be const: T *const p
        return re.search(r"\*\s*const\b[^*]*$", head) is None
    return re.search(r"\bconst\b", head) is None


def global_problems(text):
    """(line, declaration) for every mutable static variable without a global-ok note."""
    stripped = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    stripped = re.sub(r"//[^\n]*", "", stripped)
    raw = text.split("\n")
    return [(n + 1, line.strip()) for n, line in enumerate(stripped.split("\n"))
            if mutable_static(line) and "global-ok:" not in raw[n]]


def check_global_state():
    """#28: a qsort comparison context kept in a static variable made loading two tokenizers at once a race."""
    samples = [
        ("static int x = 0;\n", 1),
        ("    static uint32_t counter;\n", 1),
        ("static tr_kernels g_a, g_b;\n", 1),
        ("static char buf[64];\n", 1),
        ("static const int x = 0;\n", 0),
        ("static const tr_kernels *g_active = NULL;\n", 1),
        ("static const char *names[] = {0};\n", 1),
        ("static const char *const names[] = {0};\n", 0),
        ("static int f(void) {\n", 0),
        ("static inline int g(int a) { return a; }\n", 0),
        ("static int x = 0; /* global-ok: log level -- sample */\n", 0),
        ("/* static int x; */\n", 0),
    ]
    for n, (text, want) in enumerate(samples, 1):
        got = len(global_problems(text))
        if got != want:
            fail(28, f"the global state check is broken: sample {n} gives {got} problems instead of {want}")
            return
    for path in sorted((ROOT / "src").rglob("*.[ch]")):
        rel = path.relative_to(ROOT).as_posix()
        for line, decl in global_problems(path.read_text(encoding="utf-8")):
            fail(28, f"{rel}:{line}: mutable static variable (global state): `{decl}`; "
                     "if it is a fact of the process, annotate it with /* global-ok: reason */")


def check_makefile_recipes_ascii():
    """#36: GNU make on Windows passes recipes to the shell in the local code page."""
    for n, line in enumerate((ROOT / "Makefile").read_text(encoding="utf-8").split("\n"), 1):
        if line.startswith("\t") and not line.isascii():
            fail(36, f"Makefile:{n}: non-ASCII character in a recipe (on Windows it arrives ruined); "
                     "put it in a script in tools/")


def check_shell_scripts_whole():
    """#69: a shell reads its script while it runs it, so a script edited during a long run (a
    measurement) goes on from shifted text. A body inside main(), called on the last line and
    followed by exit, is parsed whole before anything runs."""
    for f in sorted((ROOT / "tools").glob("*.sh")):
        lines = [l for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]
        if "main() {" not in lines or lines[-1] != 'main "$@"; exit':
            fail(69, f"tools/{f.name}: the body goes inside main() {{ ... }} and the last line is "
                     'main "$@"; exit (a script edited while it runs breaks)')


def cleanup_problems(text):
    """what a tools/*.sh lacks to end what it starts (tools/cleanup.lib)."""
    problems = []
    if not re.search(r"^\s*\. tools/(cleanup|measure_guard)\.lib\s*$", text, flags=re.M):
        problems.append("does not load tools/cleanup.lib (or tools/measure_guard.lib, which loads it)")
    if not re.search(r"^\s*trap \S+ EXIT\s*$", text, flags=re.M):
        problems.append("missing `trap <cleanup> EXIT`")
    if not re.search(r"^\s*trap 'exit 130' INT TERM\s*$", text, flags=re.M):
        problems.append("missing `trap 'exit 130' INT TERM`: a signal must end the script, and go through the EXIT trap")
    if re.search(r"^\s*trap [^\n]*\bEXIT\b[^\n]*\b(INT|TERM)\b", text, flags=re.M):
        problems.append("one trap for both EXIT and the signals: after the signal the script would go on")
    return problems


def check_scripts_clean_up():
    """#84: a busy-machine test left four `yes` processes that ran for 37 hours under two days of
    measurements. Every script ends what it started: it loads tools/cleanup.lib, kills its
    descendants from the EXIT trap, and turns INT and TERM into an exit."""
    good = ". tools/cleanup.lib\ntrap cleanup_children EXIT\ntrap 'exit 130' INT TERM\n"
    samples = [
        (good, 0),
        (". tools/measure_guard.lib\ntrap restart EXIT\ntrap 'exit 130' INT TERM\n", 0),
        ("trap cleanup_children EXIT\ntrap 'exit 130' INT TERM\n", 1),
        (". tools/cleanup.lib\ntrap 'exit 130' INT TERM\n", 1),
        (". tools/cleanup.lib\ntrap cleanup_children EXIT\n", 1),
        (". tools/cleanup.lib\ntrap cleanup_children EXIT INT TERM\n", 3),
    ]
    for n, (text, want) in enumerate(samples, 1):
        got = len(cleanup_problems(text))
        if got != want:
            fail(84, f"the scripts' cleanup check is broken: sample {n} gives {got} problems instead of {want}")
            return
    for f in sorted((ROOT / "tools").glob("*.sh")):
        for msg in cleanup_problems(f.read_text(encoding="utf-8")):
            fail(84, f"tools/{f.name}: {msg}")


def tee_problems(text):
    """lines of a shell script that pipe into tee something that can fail."""
    problems = []
    for i, line in enumerate(text.split("\n"), 1):
        code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
        if re.search(r"\|\s*tee\b", code) and not re.match(r"^\s*(\[.*\]\s*\|\|\s*\{\s*)?(printf|echo)\b", code):
            problems.append(i)
    return problems


def check_no_failure_into_tee():
    """#90: the exit status of `a | tee f` is tee's: tools/platform_bits.sh ended with 0 after a step
    inside its block had failed. What can fail goes to a file and the file is shown (cat); only an
    echo or a printf may be piped into tee."""
    samples = [
        ('echo "done" | tee -a $OUT/report.txt\n', 0),
        ("    printf '%s\\n' \"$LINES\" | tee -a \"$OUT\"\n", 0),
        ('[ $SAME = 0 ] || { echo "FAILED" | tee -a $OUT/report.txt; exit 1; }\n', 0),
        ("} | tee $OUT/report.txt\n", 1),
        ("$PY - $OUT <<'EOF' | tee $OUT/5-zones.txt\n", 1),
        ("# a comment about a | tee b\n", 0),
    ]
    for n, (text, want) in enumerate(samples, 1):
        got = len(tee_problems(text))
        if got != want:
            fail(90, f"the check on pipes to tee is broken: sample {n} gives {got} problems instead of {want}")
            return
    for f in sorted((ROOT / "tools").glob("*.sh")):
        for line in tee_problems(f.read_text(encoding="utf-8")):
            fail(90, f"tools/{f.name}:{line}: a pipe to tee swallows the failure of what is on its left; "
                     "write to a file and show the file with cat")


def struct_malloc_problems(text):
    """lines that allocate one object with malloc(sizeof *x): its fields start as whatever the heap held."""
    problems = []
    for i, line in enumerate(text.split("\n"), 1):
        if re.search(r"\bmalloc\s*\(\s*sizeof\s*\(?\s*\*", line) and "malloc-ok" not in line:
            problems.append(i)
    return problems


def check_struct_calloc():
    """#252: tr_session_create took its struct from malloc and set its fields one by one; the field the argmax
    switch added was never set, so a heap's leftover byte turned the parallel argmax into the serial scan (118
    against 6 us a token) outside generate --ab, with every result the same. An object comes from calloc: a field a
    later edit forgets starts at zero."""
    samples = [
        ("    tr_session *s = (tr_session *)malloc(sizeof *s);\n", 1),
        ("    reader *r = malloc(sizeof(*r));\n", 1),
        ("    p->workers = malloc(sizeof *p->workers * (size_t)(n - 1));\n", 1),
        ("    tr_session *s = (tr_session *)calloc(1, sizeof *s);\n", 0),
        ("    float *x = malloc((size_t)n * sizeof(float));\n", 0),
        ("    t *q = malloc(sizeof *q); /* malloc-ok: every field set below */\n", 0),
    ]
    for n, (text, want) in enumerate(samples, 1):
        got = len(struct_malloc_problems(text))
        if got != want:
            fail(252, f"the check on structs from malloc is broken: sample {n} gives {got} problems instead of {want}")
            return
    for path in sorted((ROOT / "src").rglob("*.c")):
        rel = path.relative_to(ROOT).as_posix()
        for line in struct_malloc_problems(path.read_text(encoding="utf-8")):
            fail(252, f"{rel}:{line}: malloc(sizeof *x) leaves the fields to whatever was on the heap: use calloc(1, sizeof *x)")


def check_expf_table():
    """#83: src/kernels/expf_table.h is generated (tools/gen_expf_table.py, mpmath at 200 bits). A
    constant edited by hand, or a script changed without running it, would leave the header and
    its generator telling two stories: the header on disk must be what the script writes."""
    try:
        import mpmath  # noqa: F401
    except ImportError:
        print("lint: mpmath is missing, src/kernels/expf_table.h not compared with its generator")
        return
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_expf_table.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail(83, (r.stdout + r.stderr).strip() or "tools/gen_expf_table.py --check failed")


def check_readme_numbers():
    """#218: the README's draft gave a GPU time no document held ("168 us a layer"). The README carries no
    number of its own: every decimal in it, and every integer followed by a unit of measure, appears as it
    is in docs/MEASUREMENTS.md (thin and no-break spaces read as spaces; code blocks and links skipped)."""
    def norm(s):
        return s.replace(" ", " ").replace(" ", " ").replace("\xa0", " ")
    readme = norm((ROOT / "README.md").read_text(encoding="utf-8"))
    readme = re.sub(r"```.*?```", "", readme, flags=re.S)
    readme = re.sub(r"\]\([^)]*\)|https?://\S+", "", readme)
    meas = norm((ROOT / "docs" / "MEASUREMENTS.md").read_text(encoding="utf-8"))
    units = r"(µs|ms|tok/s|GFLOP/s|GB/s|MB/s|MiB|GiB)"
    for m in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w.]*\d)", readme):
        if m.group(1) not in meas:
            fail(218, f"README.md: {m.group(1)} is in no line of docs/MEASUREMENTS.md")
    for m in re.finditer(r"(?<![\w.])(\d{1,3}(?: \d{3})+|\d+) ?" + units + r"(?!\w)", readme):
        num, unit = m.group(1), m.group(2)
        if not re.search(r"(?<![\w.])" + re.escape(num) + r" ?" + re.escape(unit), meas):
            fail(218, f"README.md: '{num} {unit}' is in no line of docs/MEASUREMENTS.md")


def english_problems(text):
    """Line numbers of the lines that read as Italian: two or more distinct words that exist only in
    Italian, outside inline code spans, on a line that does not carry `italian-ok`."""
    problems = []
    for i, line in enumerate(text.split("\n"), 1):
        if "italian-ok" in line:
            continue
        words = {w.lower() for w in re.findall(r"[A-Za-zàèéìòùÀÈÉÌÒÙ]+", re.sub(r"`[^`]*`", " ", line))}
        if len(words & ITALIAN_ONLY) >= 2:
            problems.append(i)
    return problems


ITALIAN_ONLY = set((  # italian-ok: the word list itself
    "della delle degli dello dalla dalle dagli nella nelle negli sulla sulle sugli alla alle agli stessa "  # italian-ok
    "stesso stessi contro riga righe respinto respinta tenuto tenuta resta restano sempre ancora dove quando "  # italian-ok
    "poi ogni dopo senza misurato misurata misurati passata passate questo questa questi quello quella perché "  # italian-ok
    "quindi anche già più così cioè invece mentre sono viene vengono essere stato stata tutto tutti tutte "  # italian-ok
    "niente nessun nessuna nessuno uguale uguali lettura letture scrittura calcolo calcoli esperto esperti "  # italian-ok
    "pesi prova prove misura misure che è non"  # italian-ok
).split())


def check_english():
    """#257: "everything in the repository is written in English" (2026-09-22) stayed a sentence for five
    days: two documents half translated, the status page, the lint's and the hooks' messages still Italian.
    Every text file of ours: no line that reads as Italian. Measurement inputs (bench/prompts), the
    tokenizer's test texts and the translation tools are Italian on purpose."""
    samples = [("Il prefill resta più lento della misura.", 1),  # italian-ok: the samples
               ("The prefill stays slower than the measure.", 0),
               ("the test input `città è perché` stays", 0), ("Il prefill resta più lento. # italian-ok", 0)]
    for n, (text, want) in enumerate(samples):
        got = len(english_problems(text))
        if got != want:
            fail(257, f"the English check is broken: sample {n} gives {got} problems instead of {want}")
            return
    import subprocess
    skip = ("bench/prompts/", "bench/results/", "tests/tokenizer_bank.json", "tools/italian_map.py",
            "tools/check_translation.py", "src/tokenizer/unicode_data.h")
    suffixes = {".c", ".h", ".py", ".sh", ".lib", ".awk", ".json", ".txt", ".ps1", ".md", ".cjs", ".cu", ".ptx"}
    # what the repository holds: the files git tracks (a local note under docs/ that .gitignore keeps out
    # is not in it), plus the new ones not yet added
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        fail(257, "git ls-files failed: the English check needs git")
        return
    for rel in r.stdout.splitlines():
        f = ROOT / rel
        if rel.startswith(skip) or not f.is_file() or (f.suffix not in suffixes and f.name not in ("Makefile", "Dockerfile")):
            continue
        for line in english_problems(f.read_text(encoding="utf-8", errors="replace")):
            fail(257, f"{rel}:{line}: reads as Italian (everything in the repository is written in English)")


def status_problems(data, gloss):
    """The status page's data problems: shapes, states the page knows, references that resolve."""
    bad = []
    states = set(data["states"])
    if not states <= PAGE_STATES:
        bad.append(f"states {sorted(states - PAGE_STATES)} have no label on the page ({sorted(PAGE_STATES)})")
    milestones = {m["id"] for m in data["milestones"]}
    blocks = {b["id"] for b in data["blocks"]}
    for m in data["milestones"]:
        if m["state"] not in states:
            bad.append(f"milestone {m['id']}: state {m['state']!r}, not one of {sorted(states)}")
    for b in data["blocks"]:
        where = f"block {b['id']}"
        if b["state"] not in states:
            bad.append(f"{where}: state {b['state']!r}, not one of {sorted(states)}")
        if b["milestone"] not in milestones:
            bad.append(f"{where}: milestone {b['milestone']!r} does not exist")
        bad += [f"{where}: depends on {d!r}, which is no block" for d in b["depends"] if d not in blocks]
        for q in b["questions"]:
            if not isinstance(q, dict) or q.get("state") not in ("open", "closed") or not q.get("text"):
                bad.append(f"{where}: a question is not {{n, state: open|closed, text}}: {str(q)[:60]}")
        for n in b["numbers"]:
            if not isinstance(n, dict) or not n.get("v") or not re.fullmatch(r"20\d\d-\d\d-\d\d", n.get("d", "")):
                bad.append(f"{where}: a number is not {{v, d: date}}: {str(n)[:60]}")
    ids = {e["id"] for e in gloss["entries"]}
    bad += [f"glossary {e['id']}: sees {s!r}, which is no entry" for e in gloss["entries"] for s in e.get("see", [])
            if s not in ids]
    return bad


PAGE_STATES = {"done", "in-progress", "next", "open", "no"}  # tools/status_html.py's LABEL


def check_status_json():
    """#258: three questions of docs/status.json were plain strings instead of {n, state, text}, and the
    status page showed them as "#undefined"; nothing read the data before the page did. Its shape, its
    states and its references (a block's milestone, `depends`, the glossary's `see`) are checked here."""
    import copy
    import json

    def sample(**change):
        d = {"states": {"done": "", "open": ""}, "milestones": [{"id": "M0", "state": "done"}],
             "blocks": [{"id": "a", "milestone": "M0", "state": "done", "depends": [],
                         "questions": [{"n": 1, "state": "open", "text": "q"}], "numbers": [{"v": "1", "d": "2026-09-27"}]},
                        {"id": "b", "milestone": "M0", "state": "open", "depends": ["a"], "questions": [], "numbers": []}]}
        g = {"entries": [{"id": "x", "see": ["y"]}, {"id": "y", "see": []}]}
        d, g = copy.deepcopy(d), copy.deepcopy(g)
        for key, value in change.items():
            if key == "question":
                d["blocks"][0]["questions"].append(value)
            elif key == "depends":
                d["blocks"][1]["depends"].append(value)
            elif key == "see":
                g["entries"][1]["see"].append(value)
            elif key == "state":
                d["blocks"][1]["state"] = value
        return d, g
    samples = [(sample(), 0), (sample(question="83: a question written as a string"), 1),
               (sample(question={"state": "open", "text": "no number: allowed"}), 0),
               (sample(depends="gone"), 1), (sample(see="gone"), 1), (sample(state="aperto"), 1)]
    for n, ((d, g), want) in enumerate(samples):
        got = len(status_problems(d, g))
        if got != want:
            fail(258, f"the status data check is broken: sample {n} gives {got} problems instead of {want}")
            return
    data = json.loads((ROOT / "docs" / "status.json").read_text(encoding="utf-8"))
    gloss = json.loads((ROOT / "docs" / "glossary.json").read_text(encoding="utf-8"))
    for msg in status_problems(data, gloss):
        fail(258, f"docs/status.json: {msg}")


def main():
    for check in (check_docs_control_chars, check_line_endings, check_doc_limits, check_no_future_dates, check_lessons_table,
                  check_type_table, check_tests_no_tmpfile, check_hot_zones, check_global_state,
                  check_makefile_recipes_ascii, check_shell_scripts_whole, check_scripts_clean_up,
                  check_no_failure_into_tee, check_struct_calloc, check_expf_table, check_readme_numbers,
                  check_english, check_status_json):
        check()
    for f in failures:
        print(f)
    print(f"lint: {len(failures)} problem(s)" if failures else "lint: ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
