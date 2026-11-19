"""E212 (P2.38 run 3) on the CPU: the card call from run3.py's job files. Per side, d in both forms on its own read rows;
per pair, the gap (reuse side minus new side); the Gaussian update's d on each side's read rows (the floor); the planted
control beside; the call per form by commute.realistic.crosscoder_call, d_authors filed, d_alt beside. A side whose job did
not finish is not read (its pair's gap is nan). Writes run3_summary.json next to the job files.

    cpu-run python experiments/realistic_tasks/score3.py [--rehearsal | --dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from commute import realistic as R
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
PRED = HERE / "results/predictions_run3.json"
SIDES, FORMS = ("L1", "L3", "T1", "T2", "T3", "T4"), ("d_authors", "d_alt")


def load(d: Path, job: str) -> dict[str, Any] | None:
    p = d / f"run3_{job}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def score(d: Path, P: dict[str, Any]) -> dict[str, Any]:
    S, th = P["settings"], P["thresholds"]["full"]
    pairs = {k: (v[0], v[1]) for k, v in S["pairs"].items()}
    jobs = {j: load(d, j) for j in ("planted", "gauss", *SIDES)}
    sides = {s: {f: r["side"][f] for f in FORMS} if r and r.get("done") else None for s, r in ((s, jobs[s]) for s in SIDES)}
    g = jobs["gauss"]
    g_reads = g["side"]["reads"] if g and g.get("done") else {}  # a Gaussian stopped by another check is not read
    gauss = {s: {f: g_reads[s][f] for f in FORMS} for s in SIDES if s in g_reads}
    stopped = bool(g and str(g.get("stopped", "")).startswith("stop rule"))
    out: dict[str, Any] = {"experiment": P["experiment"], "dir": str(d), "sides": sides, "gaussian": gauss,
                           "gaussian_stopped": stopped, "gaussian_done": bool(g and g.get("done"))}  # fmt: skip
    p = jobs["planted"]
    if p and "side" in p:
        r = p["side"]["reads"]["L1"]
        out["planted"] = {"pass": p["side"].get("planted_pass"), "part": r["specific"], "lead": r.get("lead"),
                          "cos_planted_best": r.get("cos_planted_best"), **{f: r[f] for f in FORMS}}  # fmt: skip
    for f in FORMS:
        if stopped:
            out[f] = {"call": "rival b (stop: the Gaussian read at or over the bar; the sides were not run)",
                      "gaussian_own": g["side"][f] if g and "side" in g else None}  # fmt: skip
            continue
        dd = {s: v[f] for s, v in sides.items() if v is not None}
        out[f] = R.crosscoder_call(dd, {s: v[f] for s, v in gauss.items()}, pairs, th["gap_min"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--dir", type=Path, default=None)
    a = ap.parse_args()
    d = a.dir or (HERE / "results" / "run3_rehearsal" if a.rehearsal else HERE / "results")
    out = jsonable(score(d, json.loads(PRED.read_text())))  # nan as "nan": JSON has none
    (d / "run3_summary.json").write_text(json.dumps(out, indent=1))
    for f in FORMS:
        print(f"{f}: {json.dumps(out[f])}")
    print(f"sides {json.dumps(out['sides'])}\ngaussian {json.dumps(out['gaussian'])}\nplanted {json.dumps(out.get('planted'))}")
    print(f"wrote {d / 'run3_summary.json'}")


if __name__ == "__main__":
    main()
