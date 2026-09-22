#!/usr/bin/env python3
"""profile_suite.py — scenario profiling suite (docs/ARCHITECTURE.md §Profiling).

Runs `trochilus generate -p <prompt_tokens> -n <gen_tokens> --profile-json <tmp>`
for each scenario in bench/scenarios.json, several times (median + spread), and
prints a report: prefill/decode tokens/s, decode zones by share of token time,
weights touched per token, and a comparison with the most recent earlier run on
the same machine ("better" / "worse" / "noise").

Usage:
    profile_suite.py [--binary <path>] [--scenarios <file>] [--runs <n>]
                      [--no-save] [--smoke]

--smoke: 1 run per scenario, no warm-up, no results file saved; also checks that
    `generate` with and without --profile produces the same tokens (exit 1 if not).
"""
import argparse
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFETY_TIMEOUT_S = 60          # docs/ARCHITECTURE.md: at most 60 s per scenario run
SAC_RETRY_INTERVAL_S = 15
SAC_RETRY_BUDGET_S = 5 * 60    # Windows Smart App Control can block a fresh binary (LESSONS #12)


def default_binary():
    return "build/trochilus.exe" if platform.system() == "Windows" else "build/trochilus"


def parse_tokens_line(stdout):
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("tokens:"):
            rest = line[len("tokens:"):].strip()
            if not rest:
                return []
            return [int(x) for x in rest.split(",")]
    return None


def launch(binary, args):
    """Runs `binary args`, retrying on a permission error (LESSONS #12: Windows
    Smart App Control blocks a freshly built .exe for minutes) every 15 s for up
    to 5 minutes, then gives up. A run over SAFETY_TIMEOUT_S is a hard failure."""
    if not os.path.isfile(binary):
        print("error: binary not found: %s" % binary, file=sys.stderr)
        sys.exit(1)

    deadline = time.time() + SAC_RETRY_BUDGET_S
    attempt = 0
    while True:
        try:
            return subprocess.run([binary] + args, capture_output=True, text=True, timeout=SAFETY_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print("error: '%s %s' exceeded the %ds safety timeout" % (binary, " ".join(args), SAFETY_TIMEOUT_S),
                  file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            raise
        except OSError as e:
            attempt += 1
            if time.time() >= deadline:
                print("error: still blocked launching %s after 5 minutes (Windows Smart App Control?): %s" %
                      (binary, e), file=sys.stderr)
                sys.exit(1)
            print("permission error launching %s (attempt %d): %s; Windows Smart App Control can block a "
                  "freshly built binary for minutes, retrying in %ds..." %
                  (binary, attempt, e, SAC_RETRY_INTERVAL_S), file=sys.stderr)
            time.sleep(SAC_RETRY_INTERVAL_S)


def run_generate(binary, model, prompt_tokens, gen_tokens, threads, profile_json=None, also_profile_flag=False,
                 context=0, batch=0, decode_threads=0):
    args = ["generate", "-m", model, "-p", str(prompt_tokens), "-n", str(gen_tokens)]
    if batch:
        args += ["-b", str(batch)]
    if threads:
        args += ["-t", str(threads)]
    if decode_threads:
        args += ["--decode-threads", str(decode_threads)]
    if context:
        args += ["-c", str(context)]
    if profile_json:
        args += ["--profile-json", profile_json]
    if also_profile_flag:
        args += ["--profile"]
    return launch(binary, args)


def git_info():
    """(short commit, dirty) or (None, False) outside a git repo / without git."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
        commit = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except OSError:
        commit = None
    dirty = False
    try:
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT)
        dirty = d.returncode == 0 and bool(d.stdout.strip())
    except OSError:
        pass
    return commit, dirty


def med(xs):
    return statistics.median(xs) if xs else 0.0


def spread(xs):
    m = med(xs)
    return ((max(xs) - min(xs)) / m) if xs and m else 0.0


def aggregate(records):
    """Per-scenario aggregate over N raw --profile-json records: median tokens/s
    and spread per phase, decode zones (median seconds and share of token time),
    and decode weight traffic (MiB/token, GB/s)."""
    agg = {}
    for phase in ("prefill", "decode"):
        tps = [r["engine"]["phases"][phase]["tokens_per_sec"] for r in records]
        agg[phase] = {"tokens_per_sec_median": med(tps), "tokens_per_sec_spread": spread(tps),
                      "tokens_per_sec_min": min(tps), "tokens_per_sec_max": max(tps)}

    for phase in ("prefill", "decode"):
        phase_records = [r["engine"]["phases"][phase] for r in records]
        zone_names = set()
        for d in phase_records:
            zone_names.update(d["zones"].keys())
        zones = {}
        for z in zone_names:
            secs = [d["zones"].get(z, {"seconds": 0.0})["seconds"] for d in phase_records]
            token_secs = [d["seconds"] for d in phase_records]
            shares = [(s / t) if t > 0 else 0.0 for s, t in zip(secs, token_secs)]
            # bytes of weights and KV cache the zone read: per token, and over the zone's own time
            zbytes = [d["zones"].get(z, {}).get("bytes", 0) for d in phase_records]
            ztoks = [d["tokens"] for d in phase_records]
            zones[z] = {"seconds_median": med(secs), "share_median": med(shares),
                        "ms_per_token_median": med([(s / n * 1000.0) if n else 0.0 for s, n in zip(secs, ztoks)]),
                        "mib_per_token_median": med([(b / n / (1024.0 * 1024.0)) if n else 0.0
                                                     for b, n in zip(zbytes, ztoks)]),
                        "gb_per_sec_median": med([(b / s / 1e9) if s > 0 else 0.0 for b, s in zip(zbytes, secs)])}
        agg[phase]["zones"] = zones

        mibs, gbps, kv_mibs, all_gbps = [], [], [], []
        for d in phase_records:
            toks, wb, secs = d["tokens"], d["weight_bytes"], d["seconds"]
            kvb = d.get("kv_bytes", 0)
            mibs.append((wb / toks / (1024.0 * 1024.0)) if toks else 0.0)
            gbps.append((wb / secs / 1e9) if secs > 0 else 0.0)
            kv_mibs.append((kvb / toks / (1024.0 * 1024.0)) if toks else 0.0)
            all_gbps.append(((wb + kvb) / secs / 1e9) if secs > 0 else 0.0)
        agg[phase]["mib_per_token_median"] = med(mibs)
        agg[phase]["gb_per_sec_median"] = med(gbps)
        agg[phase]["kv_mib_per_token_median"] = med(kv_mibs)
        agg[phase]["weights_kv_gb_per_sec_median"] = med(all_gbps)
    return agg


def load_previous(hostname, exclude_prefix):
    out_dir = os.path.join(ROOT, "bench", "results")
    if not os.path.isdir(out_dir):
        return None, None
    suffix = "-%s.json" % hostname
    candidates = sorted(
        (fn for fn in os.listdir(out_dir) if fn.endswith(suffix) and not fn.startswith(exclude_prefix)),
        reverse=True,  # filenames start with YYYYMMDD-HHMMSS: lexicographic == chronological
    )
    if not candidates:
        return None, None
    with open(os.path.join(out_dir, candidates[0]), encoding="utf-8") as f:
        data = json.load(f)
    return candidates[0], {sc["name"]: sc for sc in data["scenarios"]}


def compare(prev_scenarios, name, phase, new_tps, new_spread):
    if not prev_scenarios or name not in prev_scenarios:
        return None
    old = prev_scenarios[name]["aggregate"][phase]
    old_tps = old["tokens_per_sec_median"]
    if old_tps <= 0:
        return None
    change = (new_tps - old_tps) / old_tps
    noise_bound = max(new_spread, old["tokens_per_sec_spread"])
    verdict = "noise" if abs(change) <= noise_bound else ("better" if change > 0 else "worse")
    return old_tps, change * 100.0, verdict


def print_report(results, prev_name, prev_scenarios):
    if prev_scenarios:
        print("comparing with %s\n" % prev_name)
    for res in results:
        agg = res["aggregate"]
        print("== %s (model=%s prompt=%d gen=%d threads=%s decode_threads=%s)" %
              (res["name"], res["model"], res["prompt_tokens"], res["gen_tokens"], res["threads"] or "default",
               res.get("decode_threads") or "measured"))
        for phase in ("prefill", "decode"):
            tps = agg[phase]["tokens_per_sec_median"]
            sp = agg[phase]["tokens_per_sec_spread"]
            line = "  %-8s %9.2f tok/s (median of %d, min %.2f, max %.2f, spread %.1f%%)" % (
                phase, tps, len(res["runs"]), agg[phase]["tokens_per_sec_min"], agg[phase]["tokens_per_sec_max"],
                sp * 100.0)
            cmp = compare(prev_scenarios, res["name"], phase, tps, sp)
            if cmp is not None:
                old_tps, pct, verdict = cmp
                line += "  vs %.2f tok/s: %+.1f%% (%s)" % (old_tps, pct, verdict)
            print(line)
        for phase in ("prefill", "decode"):
            zones = sorted(agg[phase]["zones"].items(), key=lambda kv: kv[1]["seconds_median"], reverse=True)
            if not zones:
                continue
            print("  %s zones by share (ms per token; MiB of weights and KV read per token, GB/s in the zone):" % phase)
            for zname, zinfo in zones:
                line = "    %-16s %8.3f ms  %5.1f%%  %8.4f ms/tok" % (
                    zname, zinfo["seconds_median"] * 1000.0, zinfo["share_median"] * 100.0,
                    zinfo["ms_per_token_median"])
                if zinfo["mib_per_token_median"] > 0:
                    line += "  %9.2f MiB/tok  %6.2f GB/s" % (zinfo["mib_per_token_median"], zinfo["gb_per_sec_median"])
                print(line)
            print("  %s weights: %.2f MiB/token, %.2f GB/s of memory traffic" %
                  (phase, agg[phase]["mib_per_token_median"], agg[phase]["gb_per_sec_median"]))
            print("  %s KV cache: %.2f MiB/token read; weights + KV: %.2f GB/s" %
                  (phase, agg[phase]["kv_mib_per_token_median"], agg[phase]["weights_kv_gb_per_sec_median"]))
        print("")


def save_results(ts, commit_part, hostname, results):
    out_dir = os.path.join(ROOT, "bench", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s-%s-%s.json" % (ts, commit_part, hostname))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"host": hostname, "timestamp": ts, "commit": commit_part, "scenarios": results}, f, indent=2)
    print("saved %s" % os.path.relpath(path, ROOT))


def resolve_model(model):
    return model if os.path.isabs(model) else os.path.join(ROOT, model)


def run_scenario(binary, sc, runs_override, smoke):
    name = sc["name"]
    model = resolve_model(sc["model"])
    prompt_tokens, gen_tokens, threads = sc["prompt_tokens"], sc["gen_tokens"], sc.get("threads", 0)
    context, batch = sc.get("context", 0), sc.get("batch", 0)
    decode_threads = sc.get("decode_threads", 0)  # 0: the session measures its own (the default)
    n_runs = 1 if smoke else (runs_override if runs_override is not None else sc.get("runs", 1))

    tmp_dir = tempfile.mkdtemp(prefix="trprof_")
    try:
        tmp_json = os.path.join(tmp_dir, "run.json")

        if not smoke:
            run_generate(binary, model, prompt_tokens, gen_tokens, threads, profile_json=tmp_json,
                         context=context, batch=batch, decode_threads=decode_threads)  # warm-up, discarded

        records, tokens = [], []
        smoke_ok = True
        for _ in range(n_runs):
            r = run_generate(binary, model, prompt_tokens, gen_tokens, threads, profile_json=tmp_json,
                             also_profile_flag=True, context=context, batch=batch, decode_threads=decode_threads)
            if r.returncode != 0:
                print("error: scenario '%s' failed (exit %d):\n%s" % (name, r.returncode, r.stderr), file=sys.stderr)
                sys.exit(1)
            with open(tmp_json, encoding="utf-8") as f:
                records.append(json.load(f))
            tokens.append(parse_tokens_line(r.stdout))

            if smoke:
                r2 = run_generate(binary, model, prompt_tokens, gen_tokens, threads, context=context, batch=batch,
                                  decode_threads=decode_threads)
                if r2.returncode != 0:
                    print("error: scenario '%s' (no profile) failed (exit %d):\n%s" %
                          (name, r2.returncode, r2.stderr), file=sys.stderr)
                    sys.exit(1)
                tok_a, tok_b = parse_tokens_line(r.stdout), parse_tokens_line(r2.stdout)
                if tok_a is None or tok_a != tok_b:
                    print("smoke: '%s' tokens differ with/without --profile" % name, file=sys.stderr)
                    print("  with --profile:    %s" % tok_a, file=sys.stderr)
                    print("  without --profile: %s" % tok_b, file=sys.stderr)
                    smoke_ok = False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = {"name": name, "model": sc["model"], "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
              "threads": threads, "decode_threads": decode_threads, "context": context, "batch": batch,
              "runs": records, "aggregate": aggregate(records),
              "tokens": tokens[0] if tokens else None, "tokens_stable": all(t == tokens[0] for t in tokens)}
    return result, smoke_ok


def tokens_problems(results):
    """Every run of a scenario, and every scenario with the same model, prompt and length
    (different threads), must generate the same tokens: a difference is a bug, not noise."""
    problems, seen = [], {}
    for res in results:
        if not res["tokens_stable"]:
            problems.append("'%s': tokens differ between its runs" % res["name"])
        key = (res["model"], res["prompt_tokens"], res["gen_tokens"])
        if key in seen and seen[key][1] != res["tokens"]:
            problems.append("'%s' and '%s': same prompt, different tokens" % (seen[key][0], res["name"]))
        seen.setdefault(key, (res["name"], res["tokens"]))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=None)
    ap.add_argument("--scenarios", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # os.path.abspath: on Windows, CreateProcess does not reliably resolve a
    # relative forward-slash path even from the right cwd (LESSONS #15).
    binary = os.path.abspath(args.binary or default_binary())
    scenarios_path = args.scenarios or os.path.join(ROOT, "bench", "scenarios.json")
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    hostname = socket.gethostname()
    results = []
    smoke_ok = True
    for sc in scenarios:
        if not os.path.exists(resolve_model(sc["model"])):
            print("skip '%s': %s not found" % (sc["name"], sc["model"]))
            continue
        result, ok = run_scenario(binary, sc, args.runs, args.smoke)
        results.append(result)
        smoke_ok = smoke_ok and ok

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    prev_name, prev_scenarios = load_previous(hostname, exclude_prefix=ts)
    print_report(results, prev_name, prev_scenarios)
    problems = tokens_problems(results)
    for p in problems:
        print("tokens: %s" % p, file=sys.stderr)
    if problems:
        return 1
    print("tokens: identical in every run and every thread count (%d scenarios)" % len(results))

    if args.smoke:
        if not smoke_ok:
            print("smoke: FAILED", file=sys.stderr)
            return 1
        print("smoke: ok")
        return 0

    if not args.no_save:
        commit, dirty = git_info()
        commit_part = (commit or "nocommit") + ("-dirty" if dirty else "")
        save_results(ts, commit_part, hostname, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
