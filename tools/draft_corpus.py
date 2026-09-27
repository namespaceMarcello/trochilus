"""tools/draft_corpus.py - collects public source code already on this machine into a batch file for `trochilus tokenize --batch`.

Records: "<decimal byte length>\n<bytes>", one per file. Caps per language, identical files once,
minified or generated files skipped (long lines, huge files). Then:
  build/trochilus.exe tokenize -m models/OLMoE-1B-7B-0125-Instruct-Q4_K.gguf --no-add-special \
      --batch build/draft_table/corpus.batch > build/draft_table/corpus.ids
The sources are whatever this machine holds (docs/MEASUREMENTS.md §A draft without a model).
"""
import hashlib
import os
import sys
from pathlib import Path

D = Path.home() / "Desktop"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "build/draft_table/corpus.batch")

# (language, roots, extensions, cap in bytes)
SOURCES = [
    ("c", [D / "trochilus/ref/llama.cpp", D / "trochilus/ref/ds4", D / "trochilus/ref/colibri"],
     {".c", ".h", ".cpp", ".hpp", ".cc"}, 24 << 20),
    ("py", [D / "trochilus/tools/.venv/Lib/site-packages"], {".py"}, 24 << 20),
    ("js", [D / "axe-core", D / "cornerstone3D", D / "birdnet-go/frontend", D / "ColdCuts", D / "auguri"],
     {".js", ".mjs", ".ts", ".tsx", ".jsx"}, 16 << 20),
    ("go", [D / "birdnet-go", Path.home() / "go/pkg/mod"], {".go"}, 10 << 20),
    ("rs", [Path.home() / ".cargo/registry/src", D], {".rs"}, 6 << 20),
    ("sh", [D], {".sh", ".bash"}, 2 << 20),
    ("sql", [D], {".sql"}, 3 << 20),
]
SKIP_DIRS = {".git", "__pycache__", "dist", "build", ".next", "coverage", "tmp"}
WIDE = {"rs", "sh", "sql"}  # walked from the whole Desktop: no dependency trees there
SKIP_WIDE = SKIP_DIRS | {"node_modules", ".venv", "venv", "site-packages", "vendor"}
MAX_FILE = 300 << 10


def files(roots, exts, skip):
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in skip)
            for f in sorted(filenames):
                if os.path.splitext(f)[1] in exts and not f.endswith(".min.js"):
                    yield Path(dirpath) / f


def main():
    seen = set()
    total = {}
    with OUT.open("wb") as out:
        for lang, roots, exts, cap in SOURCES:
            got = 0
            for p in files(roots, exts, SKIP_WIDE if lang in WIDE else SKIP_DIRS):
                if got >= cap:
                    break
                try:
                    b = p.read_bytes()
                except OSError:
                    continue
                if not b or len(b) > MAX_FILE:
                    continue
                lines = b.count(b"\n") + 1
                if len(b) / lines > 160:  # minified or generated
                    continue
                try:
                    b.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                h = hashlib.sha1(b).digest()
                if h in seen:
                    continue
                seen.add(h)
                out.write(b"%d\n" % len(b))
                out.write(b)
                got += len(b)
            total[lang] = got
    for k, v in total.items():
        print(f"{k}: {v / 2**20:.1f} MiB")
    print(f"files: {len(seen)}")


main()
