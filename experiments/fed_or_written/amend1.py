"""E100 Amendment 1: does the sum of single-part reverts tell fed from written on seeds E100 never read? No training.
Plan: reports/plans/2026-11-05_plan_e100-amend1.md. Rule, bars and pass rule: results/predictions_amend1.json (committed
first; the run refuses otherwise; never edited) and `commute.fed_rules.sum_reading` / `sum_pass`. Per state it reads G
and the 28 single reverts (`locator.own`), nothing else; E100's run.py and its files are not touched.

  stage 0, alone: E62 copy seed 0's sum of single reverts, against E100's saved value; exit 1 on a miss.
  stage 1: E58's four open-pair maps at seeds 3-7 (open_pair_instruments, sha256 against its seeds.json), each on the
           site of its seeds 0-2 in E100 (E97's sites); results/amend1.json written as each state finishes, and a
           resubmit skips the states already there.
  table:   the outcome in results/amend1.json; results/table_amend1.json = E100's table rows with reverts_sum_reading
           added, then the fresh rows.
"STAGE <n> DONE" is printed when a stage is complete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import fed_rules as FR
from commute import locator as L
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment
from commute.keydep import log, sha256

JB = load_experiment("locator_table", "job")
sites = JB.sites
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions_amend1.json", RESULTS / "amend1.json"
TABLE_E100, TABLE = RESULTS / "table.json", RESULTS / "table_amend1.json"
FRESH = sites.EXP / "open_pair_instruments" / "results"
MAPS = {"T13 copy-24": "copy", "T11 cipher": "cipher", "T12 rank-1": "rank1", "T11 elicit": "elicit"}
SEEDS = range(3, 8)
CONTROL = ("T5 copy", 0)
dump, say, load, stop = JB.dump, JB.say, JB.load, JB.stop


def fresh(F: Any) -> dict[str, tuple[str, int, Path]]:
    """name -> (task, seed, file) for the 20 fresh states; each file's sha256 record from seeds.json goes into F."""
    rec = json.loads((FRESH / "seeds.json").read_text())["states"]
    out = {}
    for task, m in MAPS.items():
        for s in SEEDS:
            path = FRESH / "large" / f"{m}_s{s}.pt"
            F.rec[path] = rec[f"{m}_s{s}"]["ft"]["sha256"]
            out[f"{task} s{s}"] = (task, s, path)
    return out


def read_sum(J: Any, key: str, dW: ST.DW) -> dict[str, Any]:
    """G and the 28 single parts' own shares (unit lo) of one state on site `key`, and their sum."""
    t0 = time.monotonic()
    st = L.state(J.net, J.base(key), dW)
    own = {L.name(p): L.own(J.net, st, L.name(p))["lo"] for p in L.PARTS}
    return {"G": float(st.G[0]), "own": own, "reverts_sum": FR.reverts_sum({n: {"own": x} for n, x in own.items()}),
            "seconds": time.monotonic() - t0}  # fmt: skip


def reading(th: dict[str, Any], G: float, s: float) -> str:
    return FR.sum_reading(G, s, th["fed_bar"], th["written_bar"], th["min_gain"])


def stage0(J: Any, out: dict[str, Any]) -> None:
    task, seed = CONTROL
    T, th = sites.TASKS[task], J.th
    rec = read_sum(J, T.site, J.F.state(T.states[seed])) | {"task": task, "seed": seed, "site": T.site}
    rec["miss"] = not abs(rec["reverts_sum"] - th["stage0_ref"]) <= th["stage0_tol"]
    rec["status"] = "miss" if rec["miss"] else "pass"
    out["stage0"] = rec
    dump(OUT, out)
    say("stage0", rec)
    if rec["miss"]:
        stop(OUT, out, f"stage 0: {task} s{seed} sums to {rec['reverts_sum']}, not {th['stage0_ref']} within {th['stage0_tol']}")


def stage1(J: Any, out: dict[str, Any]) -> None:
    states = out.setdefault("states", {})
    for name, (task, s, path) in fresh(J.F).items():
        if name in states:
            continue
        site = sites.TASKS[task].site
        rec = read_sum(J, site, J.F.state(path)) | {"task": task, "seed": s, "site": site, "file": str(path.relative_to(sites.EXP))}
        rec["reading"] = reading(J.th, rec["G"], rec["reverts_sum"])
        states[name] = rec
        dump(OUT, out)
        say(f"state {name}", {k: v for k, v in rec.items() if k != "own"})
    out["files"] = J.F.seen


def outcome(P: dict[str, Any], out: dict[str, Any]) -> None:
    by = {t: [r["reading"] for r in out["states"].values() if r["task"] == t] for t in MAPS}
    leans = P["expected"]["leans"]
    out["tasks"] = {t: {"readings": by[t], "lean": leans[t], "seeds_at_lean": sum(r == leans[t] for r in by[t])} for t in MAPS}
    out["outcome"] = FR.sum_pass(by["T13 copy-24"], by["T11 cipher"], P["rule"]["pass_need"])
    dump(OUT, out)
    say("tasks", out["tasks"])
    say("outcome", out["outcome"])


def write_table(th: dict[str, Any], out: dict[str, Any]) -> None:
    """E100's rows, each with the rule's reading of its sum (the seeds the bars were set from), then the fresh rows."""
    old = [r | {"reverts_sum_reading": reading(th, r["G"], r["reverts_sum"]), "fresh": False} for r in load(TABLE_E100)["table"]]
    new = [{"task": r["task"], "seed": r["seed"], "G": r["G"], "reverts_sum": r["reverts_sum"], "reverts_sum_reading": r["reading"],
            "fresh": True, "stage": "amend1"} for r in out["states"].values()]  # fmt: skip
    dump(TABLE, {"table": old + new, "outcome": out["outcome"], "note": "bars set post hoc on the rows with fresh false"})
    for r in new:
        say("table", r)


def filed() -> dict[str, Any]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    return json.loads(PRED.read_text())


def timed(k: int, f: Any, *a: Any) -> None:
    t0 = time.monotonic()
    f(*a)
    log(f"[stage {k} done] {time.monotonic() - t0:.0f} s")
    print(f"STAGE {k} DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    P = filed()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E100's run.py
    net = ST.Net(args.device)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {head} predictions {sha256(PRED)}")
    J = JB.Job(net, sites.Files(), P, {})
    out = load(OUT)
    out |= {"git_head": head, "predictions_sha256": sha256(PRED)}
    timed(0, stage0, J, out)
    timed(1, stage1, J, out)
    outcome(P, out)
    write_table(J.th, out)
    hard_exit(0)


if __name__ == "__main__":
    main()
