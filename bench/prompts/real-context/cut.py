"""cut.py - the prompts of this folder from sources.tsv (the rule is in its head): each file at the commit
up to the end of the line that opens the named function's body, its neighbours first. Then a leak check
for the repo table of tools/draft_gate_sim.py, which leaves out only the file under test: the body's
distinctive lines (20 characters or more) found verbatim in another file of src/, tools/, tests/.

  tools/.venv/Scripts/python.exe bench/prompts/real-context/cut.py
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def show(commit, path):
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout


def body_start(b, path, func):
    """the byte after the line that opens the k-th definition of the function (func = name or name#k)"""
    name, _, k = func.partition("#")
    n = re.escape(name.encode())
    if path.endswith((".c", ".h")):
        rx = re.compile(rb"^(?![ \t#])[^\n;{}]*\b" + n + rb"\s*\([^;{}]*\)\s*\{[ \t]*\n", re.M)
    elif path.endswith(".py"):
        rx = re.compile(rb"^def " + n + rb"\((?:[^()]|\([^()]*\))*\)(?:\s*->\s*[^:\n]+)?:[ \t]*\n", re.M)
    else:
        rx = re.compile(rb"^" + n + rb"\(\)\s*\{[ \t]*\n", re.M)
    return list(rx.finditer(b))[int(k or 1) - 1].end()


def body_of(b, start, path):
    out = []
    for line in b[start:].split(b"\n"):
        if (line and not line[:1].isspace()) if path.endswith(".py") else line.startswith(b"}"):
            break
        out.append(line)
    return out


def marker(path):
    return (b"// file: " if path.endswith((".c", ".h")) else b"# file: ") + path.encode() + b"\n"


def main():
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "src",
                        "tools", "tests"], capture_output=True, check=True, text=True)
    texts = {p: (ROOT / p).read_bytes() for p in r.stdout.split("\n") if p and (ROOT / p).is_file()}
    for line in (HERE / "sources.tsv").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, path, func, neigh, commit = line.split("\t")
        b = show(commit, path)
        s = body_start(b, path, func)
        parts = []
        if neigh != "-":
            for q in neigh.split(","):
                nb = show(commit, q)
                parts += [marker(q), nb if nb.endswith(b"\n") else nb + b"\n"]
            parts.append(marker(path))
        prompt = b"".join(parts) + b[:s]
        (HERE / f"{name}.txt").write_bytes(prompt)
        key = [x.strip() for x in body_of(b, s, path) if len(x.strip()) >= 20]
        dup = []
        for p, t in texts.items():
            if p != path:
                hit = sum(1 for x in key if x in set(y.strip() for y in t.split(b"\n")))
                if hit >= 3 or (hit >= 2 and hit >= 0.3 * len(key)):
                    dup.append(f"{p}:{hit}/{len(key)}")
        print(f"{name:24s} {len(prompt):6d} bytes, cut at {100 * s / len(b):3.0f}% of {path}; "
              f"the body's lines elsewhere: {' '.join(dup) or '-'}")


main()
