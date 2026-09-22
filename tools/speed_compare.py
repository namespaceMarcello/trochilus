#!/usr/bin/env python3
"""speed_compare.py — Trochilus against llama.cpp and colibri on the same model (docs/STATUS.md, M0 step 5).

For each thread count, each engine processes the same prompt length and generates the same number of
tokens; every figure is the median of --runs runs (one warm-up run discarded), with min and max.

  prefill tok/s = prompt tokens / prompt time
  decode tok/s  = single-token evaluations / their time (the first generated token comes from the
                  prompt's logits and is not an evaluation)

Engines:
  trochilus  `generate --tokens <ids>` on the GGUF; tokens must be identical in every run.
  llama.cpp  `llama-bench -p P -n N` on the same GGUF (its own random tokens, mmap, warm-up included).
  colibri    `olmoe 64 8 ref.json` on its int8 conversion (tools/convert_olmoe_merged.py): 64 cached
             experts per layer, i.e. all of them. It prints one time for prefill + decode, so prefill
             is a run with 1 new token and decode = N / (time(N + 1 new) - time(1 new)), on medians.

With --tok-file, also the tokenizer: bytes/s and tokens/s of Trochilus `tokenize` and llama.cpp
`llama-tokenize` (wall time minus the same command on a one-byte text: loading is not counted) and of
HF `tokenizers` in process.

Runs in the trochilus-dev container (Linux binaries), one engine at a time:
  tools/speed_compare.py --model models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf \
      --trochilus build/linux-gcc/trochilus --llama-bench ref/llama.cpp/build-trochilus/bin/llama-bench \
      --colibri build/colibri-src/olmoe --colibri-snap models/olmoe_colibri --threads 16,8,4,1
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFETY_TIMEOUT_S = 60       # docs/ARCHITECTURE.md: at most 60 s per benchmark run
MIN_AVAILABLE_GB = 9.0      # a 7 GB model plus the 2 GB margin of the machine-safety invariant


def run(cmd, env=None, timeout=SAFETY_TIMEOUT_S):
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        sys.exit("error: '%s' exceeded the %d s safety timeout" % (" ".join(map(str, cmd)), timeout))
    if r.returncode != 0:
        sys.exit("error: '%s' failed (%d)\n%s" % (" ".join(map(str, cmd)), r.returncode, r.stderr[-2000:]))
    return r


def check_memory():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f}
    avail = info["MemAvailable"] / (1024 * 1024)
    if avail < MIN_AVAILABLE_GB:
        sys.exit("error: %.1f GB available, %.1f GB needed (close other programs, drop the VM cache)" %
                 (avail, MIN_AVAILABLE_GB))


def drop_caches():
    """Gives the page cache of the previous engine back (docs/LESSONS.md #38); needs --privileged."""
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
    except OSError:
        pass


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "runs": values}


def prompt_ids(args):
    out = run([args.trochilus, "tokenize", "-m", args.model, "-f", args.prompt_file, "--no-add-special"]).stdout
    ids = [int(x) for x in re.findall(r"-?\d+", out)]
    if not ids:
        sys.exit("error: no tokens from %s" % args.prompt_file)
    return [ids[i % len(ids)] for i in range(args.prompt)]


def bench_trochilus(args, ids, threads):
    cmd = [args.trochilus, "generate", "-m", args.model, "--tokens", ",".join(map(str, ids)),
           "-n", args.gen, "-t", threads, "-c", args.prompt + args.gen]
    prefill, decode, tokens = [], [], None
    for i in range(args.runs + 1):
        r = run(cmd)
        p = re.search(r"prompt: (\d+) tokens in ([\d.]+)s", r.stderr)
        g = re.search(r"generate: \d+ tokens, (\d+) evaluations in ([\d.]+)s", r.stderr)
        line = [l for l in r.stdout.splitlines() if l.startswith("tokens:")][0]
        if tokens is None:
            tokens = line
        elif line != tokens:
            sys.exit("error: trochilus produced different tokens in run %d" % i)
        if i == 0:
            continue
        prefill.append(int(p.group(1)) / float(p.group(2)))
        decode.append(int(g.group(1)) / float(g.group(2)))
    return {"prefill": stats(prefill), "decode": stats(decode)}


def bench_llamacpp(args, threads):
    """Prefill from an empty context; decode after a context of --prompt tokens (-d), like the others."""
    res = {}
    for key, shape in (("prefill", ["-p", args.prompt, "-n", 0]), ("decode", ["-p", 0, "-n", args.gen, "-d", args.prompt])):
        r = run([args.llama_bench, "-m", args.model, *shape, "-t", threads, "-r", args.runs, "-o", "json"],
                timeout=SAFETY_TIMEOUT_S * (args.runs + 2))
        res[key] = stats(json.loads(r.stdout)[0]["samples_ts"])
    return res


def bench_colibri(args, ids, threads):
    env = dict(os.environ, SNAP=os.path.abspath(args.colibri_snap), OMP_NUM_THREADS=str(threads))
    times = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n_new in (1, args.gen + 1):
            ref = os.path.join(tmp, "ref.json")
            with open(ref, "w") as f:
                json.dump({"prompt_ids": ids, "full_ids": ids + [0] * n_new}, f)
            dts, tokens, hits = [], None, []
            for i in range(args.runs + 1):
                out = run([os.path.abspath(args.colibri), "64", "8", ref], env=env).stdout
                m = re.search(r"TUNE decode: \d+ tokens in ([\d.]+)s", out)
                line = [l for l in out.splitlines() if l.startswith("C engine")][0]
                if tokens is None:
                    tokens = line
                elif line != tokens:
                    sys.exit("error: colibri produced different tokens in run %d" % i)
                if i > 0:
                    dts.append(float(m.group(1)))
                    hits.append(float(re.search(r"Expert cache hit rate: ([\d.]+)%", out).group(1)))
            times[n_new] = dts
            hit_rate = statistics.median(hits)
    prefill = [len(ids) / t for t in times[1]]
    base = statistics.median(times[1])
    decode = [args.gen / (t - base) for t in times[args.gen + 1]]
    return {"prefill": stats(prefill), "decode": stats(decode), "expert_cache_hit_pct": hit_rate}


def bench_tokenizers(args):
    with open(args.tok_file, "rb") as f:
        data = f.read()
    res = {"bytes": len(data)}
    with tempfile.TemporaryDirectory() as tmp:
        tiny = os.path.join(tmp, "tiny.txt")
        with open(tiny, "w") as f:
            f.write("a")

        def wall(cmd):
            t = time.perf_counter()
            r = run(cmd)
            return time.perf_counter() - t, r.stdout

        engines = [("trochilus", lambda f: [args.trochilus, "tokenize", "-m", args.model, "-f", f])]
        if args.llama_tokenize:
            engines.append(("llama.cpp", lambda f: [args.llama_tokenize, "-m", args.model, "-f", f, "--ids",
                                                    "--log-disable"]))
        for name, cmd in engines:
            secs = []
            for i in range(args.runs + 1):
                base, _ = wall(cmd(tiny))
                full, out = wall(cmd(args.tok_file))
                if i > 0:
                    secs.append(full - base)
            n_tok = len(re.findall(r"-?\d+", out.strip().splitlines()[-1]))
            s = statistics.median(secs)
            res[name] = {"seconds": stats(secs), "tokens": n_tok, "MB/s": len(data) / s / 1e6,
                         "tok/s": n_tok / s}
        if args.hf_tokenizer:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(args.hf_tokenizer)
            text = data.decode("utf-8")
            secs = []
            for i in range(args.runs + 1):
                t = time.perf_counter()
                enc = tok.encode(text)
                if i > 0:
                    secs.append(time.perf_counter() - t)
            s = statistics.median(secs)
            res["hf-tokenizers"] = {"seconds": stats(secs), "tokens": len(enc.ids),
                                    "MB/s": len(data) / s / 1e6, "tok/s": len(enc.ids) / s}
    return res


def fmt(s):
    return "**%.2f** (%.2f–%.2f)" % (s["median"], s["min"], s["max"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--trochilus", required=True)
    ap.add_argument("--llama-bench")
    ap.add_argument("--llama-tokenize")
    ap.add_argument("--colibri")
    ap.add_argument("--colibri-snap")
    ap.add_argument("--threads", default="16,8,4,1")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--prompt-file", default=os.path.join(ROOT, "src", "models", "olmoe.c"))
    ap.add_argument("--tok-file")
    ap.add_argument("--hf-tokenizer")
    ap.add_argument("--engines", default="trochilus,llama.cpp,colibri")
    ap.add_argument("--out")
    args = ap.parse_args()

    threads = [int(t) for t in args.threads.split(",")]
    engines = args.engines.split(",")
    ids = prompt_ids(args)
    result = {"date": datetime.now().isoformat(timespec="seconds"), "model": args.model,
              "prompt": args.prompt, "gen": args.gen, "runs": args.runs, "speed": []}

    rows = []
    for name in engines:
        if name == "llama.cpp" and not args.llama_bench or name == "colibri" and not args.colibri:
            continue
        drop_caches()
        check_memory()
        for t in threads:
            if name == "trochilus":
                r = bench_trochilus(args, ids, t)
            elif name == "llama.cpp":
                r = bench_llamacpp(args, t)
            else:
                r = bench_colibri(args, ids, t)
            result["speed"].append({"engine": name, "threads": t, **r})
            hits = " (expert cache hits %.1f%%)" % r["expert_cache_hit_pct"] if "expert_cache_hit_pct" in r else ""
            rows.append("| %s | %d | %s | %s%s |" % (name, t, fmt(r["prefill"]), fmt(r["decode"]), hits))
            print(rows[-1], flush=True)
    drop_caches()

    print("\nprompt %d, gen %d, median of %d runs (min–max)\n" % (args.prompt, args.gen, args.runs))
    print("| Engine | Threads | Prefill tok/s | Decode tok/s |\n|---|---|---|---|")
    print("\n".join(rows))

    if args.tok_file:
        result["tokenizer"] = bench_tokenizers(args)
        print("\ntokenizer on %s (%d bytes)" % (args.tok_file, result["tokenizer"]["bytes"]))
        for k, v in result["tokenizer"].items():
            if k != "bytes":
                print("  %-14s %8.2f MB/s %10.0f tok/s  (%d tokens)" % (k, v["MB/s"], v["tok/s"], v["tokens"]))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
