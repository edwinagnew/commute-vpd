"""E224's score (P2.41 runs 6-7), from run_newtasks.py's and read_newtasks.py's files only: no model. Per task and arm,
the share's readable seeds in the set contrast (gain at or over MIN_GAIN); per task (1 plural, 2 sequence), E222's filed
call between the card's rivals and the third outcome (score_stage3.pick_call / card_call, unchanged; rules in
results/predictions_E224.json "rules"); the scorecard baselines through the same rule beside (not filed); and each
state's training reads (held-out top-1, contrast, control top-1 at step 0 and the stop, KL, steps). Writes
results/stage3c_score.json and prints the calls.

    ~/commute-agents/fleet/cpu-run python experiments/rl_67m/score_newtasks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment

RESULTS = Path(__file__).resolve().parent / "results"
SS = load_experiment("rl_67m", "score_stage3")  # E222's rule: readable, pick_call, card_call, num, GET
PICKS, ANCHOR, UNIT = {"1": "plural", "2": "sequence"}, "addition", "contrast"
METHODS = [m for m in SS.GET if m != "k_half"]  # the set contrast reads one target token per row: no k_half


def seeds(states: dict[str, Any], task: str, arm: str, method: str) -> list[float]:
    return [SS.num(SS.GET[method](s, UNIT)) for s in states.values() if s["task"] == task and s["arm"] == arm]


def score(states: dict[str, Any], R: dict[str, Any]) -> dict[str, Any]:
    g, m = R["smallest_gap"], R["min_readable_seeds"]
    out: dict[str, Any] = {"values": {}, "calls": {}}
    for method in METHODS:
        out["values"][method] = {f"{t} {a}": seeds(states, t, a, method) for t in (*PICKS.values(), ANCHOR) for a in SS.ARMS
                                 if any(s["task"] == t and s["arm"] == a for s in states.values())}  # fmt: skip
    for method in SS.RATIOS:
        anchor = SS.readable(seeds(states, ANCHOR, "demo", method), m)
        calls = {t: SS.pick_call({a: SS.readable(seeds(states, t, a, method), m) for a in SS.ARMS}, anchor, g) for t in PICKS.values()}
        out["calls"][method] = {"picks": calls, "card": SS.card_call({k: v["call"] for k, v in calls.items()}), "filed": method == "share"}
    return out


def training(rec: dict[str, Any]) -> dict[str, Any]:
    r0, rs = rec["reads"]["0"], rec["reads"]["stop"]
    keep = ("top1", "contrast", "ctrl_top1")
    return {"steps": rec["steps"], "reached": rec["reached"], "size_saved": rec["size_saved"], "kl_saved": rec.get("kl_saved"),
            "collapsed": rec["collapsed"], **{f"{k} 0 -> stop": [r0[k], rs[k]] for k in keep}}  # fmt: skip


def main() -> None:
    P = json.loads((RESULTS / "predictions_E224.json").read_text())
    jobs = {j: json.loads((RESULTS / f"stage3c_{j}.json").read_text()) for j in ("ctl", *PICKS, "anchor")}
    assert all(j.get("done") for j in jobs.values()), {k: j.get("stopped") for k, j in jobs.items()}
    states = {f.stem: json.loads(f.read_text()) for f in sorted((RESULTS / "stage3c").glob("*.json"))}
    arms = {f.stem: json.loads(f.read_text()) for f in sorted(RESULTS.glob("stage2c_arm_*.json"))}
    out = score(states, P["rules"]) | {"gauss": {j: jobs[j]["gauss"]["gain"] for j in (*PICKS, "anchor")}}
    out["training"] = {f"{r['task']}_{r['arm']}_s{r['seed']}": training(r) for a in arms.values() for r in a.get("seeds", {}).values() if "reads" in r}
    (RESULTS / "stage3c_score.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for method, c in out["calls"].items():
        print(method, c["card"], json.dumps({k: v["call"] for k, v in c["picks"].items()}))
    print("share values", json.dumps(out["values"]["share"]))


if __name__ == "__main__":
    main()
