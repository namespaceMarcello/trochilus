"""row_price_report.py -- the price of one more verified row, from tools/row_price.sh's runs.

Per configuration (model, threads): the one-row pass, then for passes of 2, 3, 5, 9 rows the extra row's
milliseconds, as a share of a pass, split by zone (experts, dense matmuls, attention, the rest), and the
MiB/ms its new experts were read at (against the one-row pass's). Medians of the rounds after round 0.
Writes <out>/prices.json: ms a pass by rows and the one-row pass's experts, for tools/draft_gate_sim.py.
With a second folder (row_price.sh's binary-before, the runs alternated with <out>'s): its table too, then
each pass before and after side by side, the ratio, and the row's share of a pass before and after.

  python tools/row_price_report.py [build/rowprice] [rounds] [<before-folder> [<a/a-folder>]]

With a fourth folder (row_price.sh's ROW_PRICE_AA=1: the before binary run a second time), the before
against its own second run: the A/A, the size of a difference that means nothing. The configurations and
modes are the ones the folder's files hold (a mode or a model the lists below do not name is added, not
dropped).
"""
import json
import os
import re
import statistics
import sys
import time

MODES = ["0", "1specfixed", "2specfixed", "4specfixed", "8specfixed"]
CONFIGS = ("Q8_0-t8", "Q4_K-t8", "Q8_0-t16")
RUN = re.compile(r"^(.+-t\d+)-(\d+(?:specfixed)?)-(\d+)\.json$")


def discover(out):
    """The folder's configurations and modes, the known ones in their order, then the others."""
    cfgs, modes = set(), set()
    for f in os.listdir(out):
        m = RUN.match(f)
        if m:
            cfgs.add(m.group(1))
            modes.add(m.group(2))
    cfg_list = [c for c in CONFIGS if c in cfgs] + sorted(cfgs - set(CONFIGS))
    mode_list = sorted(modes, key=lambda t: int(t.replace("specfixed", "")))
    return cfg_list, mode_list


def load(out, cfg, tag, rounds):
    rows = []
    for r in range(1, rounds + 1):
        f = f"{out}/{cfg}-{tag}-{r}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))["engine"]["phases"]["decode"]
        p = d["zones"]["token"]["calls"]
        z = {k: v["seconds"] / p * 1e3 for k, v in d["zones"].items()}
        zb = {k: v.get("bytes", 0) / p / 2**20 for k, v in d["zones"].items()}
        rows.append((d["tokens"] / p, z, zb))
    if not rows:
        return None
    return (statistics.median(x[0] for x in rows),
            {k: statistics.median(x[1][k] for x in rows) for k in rows[0][1]},
            (min(x[1]["token"] for x in rows), max(x[1]["token"] for x in rows)),
            {k: statistics.median(x[2][k] for x in rows) for k in rows[0][2]})


def table(out, rounds, label=""):
    """Prints out's tables; returns {cfg: {rows: (ms a pass, min, max)}} and the prices."""
    runs = [os.path.join(out, f) for f in os.listdir(out) if f.endswith(".json") and f != "prices.json"]
    last = max(os.path.getmtime(f) for f in runs)  # the last run's end, not the folder's last write
    prices = {"measured": time.strftime("%Y-%m-%d %H:%M", time.localtime(last)),
              "binary": open(f"{out}/binary.txt").read().strip() if os.path.exists(f"{out}/binary.txt") else "",
              "configs": {}}
    passes = {}
    cfgs, modes = discover(out)
    for cfg in cfgs:
        res = {tag: load(out, cfg, tag, rounds) for tag in modes}
        if res.get("0") is None:
            print(f"== {label}{cfg}: no one-row pass (mode 0): skipped")
            continue
        _, z0, mm0, zb0 = res["0"]
        ex0 = z0["expert_gate_up"] + z0["expert_down"]
        exb0 = zb0["expert_gate_up"] + zb0["expert_down"]
        print(f"== {label}{cfg}: one row {z0['token']:.2f} ms a pass ({mm0[0]:.2f}-{mm0[1]:.2f}), "
              f"experts {ex0:.2f} ms for {exb0:.0f} MiB ({exb0 / ex0:.1f} MiB/ms)")
        ms = {1: z0["token"]}
        passes[cfg] = {1: (z0["token"], mm0[0], mm0[1])}
        for tag in modes:
            if tag == "0" or res[tag] is None:
                continue
            rows, z, mm, zb = res[tag]
            extra = rows - 1.0
            ms[round(rows)] = z["token"]
            passes[cfg][round(rows)] = (z["token"], mm[0], mm[1])
            total = (z["token"] - z0["token"]) / extra
            experts = (z["expert_gate_up"] + z["expert_down"] - ex0) / extra
            newb = (zb["expert_gate_up"] + zb["expert_down"] - exb0) / extra
            dense = sum(z[k] - z0[k] for k in ("qkv_proj", "attn_out_proj", "lm_head")) / extra
            att = (z["attention"] - z0["attention"]) / extra
            print(f"  {rows:.2f} rows: {z['token']:.2f} ms ({mm[0]:.2f}-{mm[1]:.2f}); one more row {total:.2f} ms = "
                  f"{total / z0['token']:.2f} of a pass (experts {experts:.2f} ms for {newb:.0f} new MiB, "
                  f"{newb / experts:.1f} MiB/ms; dense {dense:.2f}; attention {att:.2f}; "
                  f"rest {total - experts - dense - att:.2f})")
        prices["configs"][cfg] = {"ms_by_rows": ms, "experts_ms_one_row": ex0, "experts_mib_one_row": exb0}
    with open(f"{out}/prices.json", "w") as f:
        json.dump(prices, f, indent=1)
    print(f"prices: {out}/prices.json")
    return passes


def side_by_side(title, after, prev):
    print(title)
    for cfg in after:
        if cfg not in prev:
            continue
        a1, b1 = after[cfg][1][0], prev[cfg][1][0]
        for rows in sorted(after[cfg]):
            if rows not in prev[cfg]:
                continue
            a, b = after[cfg][rows], prev[cfg][rows]
            share = ""
            if rows > 1:
                share = "; a row %.2f -> %.2f of a pass" % ((b[0] - b1) / (rows - 1) / b1, (a[0] - a1) / (rows - 1) / a1)
            print(f"  {cfg} {rows} rows: {b[0]:.2f} ({b[1]:.2f}-{b[2]:.2f}) -> {a[0]:.2f} ({a[1]:.2f}-{a[2]:.2f}), "
                  f"{a[0] / b[0]:.3f}{share}")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "build/rowprice"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    before = sys.argv[3] if len(sys.argv) > 3 else None
    again = sys.argv[4] if len(sys.argv) > 4 else None
    after = table(out, rounds, "after " if before else "")
    if before is None:
        return 0
    prev = table(before, rounds, "before ")
    side_by_side("== before -> after, ms a pass (min-max), after / before; one more row's share of a pass", after, prev)
    if again is not None:
        aa = table(again, rounds, "before, again ")
        side_by_side("== the A/A: before -> before again (a difference of this size means nothing)", aa, prev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
