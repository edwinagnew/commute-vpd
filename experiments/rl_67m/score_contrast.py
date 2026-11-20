"""E223's score (P2.41 run 5), from run_contrast.py's per-state files and E222's anchor records only: no model. E222's
filed rule (score_stage3.pick_call, card_call) on the share in the per-row contrast for the picks (repeat, lower)
against the anchor's E222 share in log-odds (not re-read: no two-form pair is natural for French after English); the
baselines' seed values beside, each run through the same rule (beside, not filed). Writes results/contrast_score.json.

    ~/commute-agents/fleet/cpu-run python experiments/rl_67m/score_contrast.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment

RESULTS = Path(__file__).resolve().parent / "results"
S = load_experiment("rl_67m", "score_stage3")  # E222's rule: readable, pick_call, card_call, ARMS, RATIOS, GET
PICKS = ("1", "2")
METHODS = [m for m in S.GET if m != "k_half"]  # k_half reads one target per row: not read on a pair


def seeds(states: dict[str, Any], name: str, arm: str, method: str, unit: str) -> list[float]:
    return [S.num(S.GET[method](s, unit)) for s in states.values() if s["behaviour"] == name and s["arm"] == arm]


def score(states: dict[str, Any], anchor: dict[str, Any], names: dict[str, str], R: dict[str, Any]) -> dict[str, Any]:
    g, m = R["smallest_gap"], R["min_readable_seeds"]
    out: dict[str, Any] = {"values": {}, "calls": {}}
    for method in METHODS:
        out["values"][method] = {f"{names[p]} {a}": seeds(states, names[p], a, method, "contrast") for p in PICKS for a in S.ARMS}
        out["values"][method][f"{names['anchor']} demo (E222, lo)"] = seeds(anchor, names["anchor"], "demo", method, "lo")
    out["values"]["gain"] = {f"{names[p]} {a}": [s["gain"]["contrast"] for s in states.values() if s["behaviour"] == names[p] and s["arm"] == a]
                             for p in PICKS for a in S.ARMS}  # fmt: skip
    for method in S.RATIOS:
        anc = S.readable(seeds(anchor, names["anchor"], "demo", method, "lo"), m)
        calls = {names[p]: S.pick_call({a: S.readable(seeds(states, names[p], a, method, "contrast"), m) for a in S.ARMS}, anc, g) for p in PICKS}
        out["calls"][method] = {"picks": calls, "card": S.card_call({k: v["call"] for k, v in calls.items()}), "filed": method == "share"}
    return out


def main() -> None:
    P = json.loads((RESULTS / "predictions_E223.json").read_text())
    jobs = {j: json.loads((RESULTS / f"contrast_{j}.json").read_text()) for j in ("ctl", *PICKS)}
    assert all(j.get("done") for j in jobs.values()), {k: j.get("stopped") for k, j in jobs.items()}
    c = json.loads((RESULTS / "stage1.json").read_text())["calls"]
    names = {"1": c["picks"][0], "2": c["picks"][1], "anchor": c["anchor"]}
    states = {f.stem: json.loads(f.read_text()) for f in sorted((RESULTS / "contrast").glob("*.json"))}
    anchor = {f.stem: json.loads(f.read_text()) for f in sorted((RESULTS / "stage3").glob(f"{names['anchor']}_demo_s*.json"))}
    assert len(states) == 24 and len(anchor) == 3, (len(states), len(anchor))
    out = score(states, anchor, names, P["rules"]) | {"gauss": {names[j]: jobs[j]["gauss"]["gain"] for j in PICKS}}
    (RESULTS / "contrast_score.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for method, call in out["calls"].items():
        print(method, call["card"], json.dumps({k: v["call"] for k, v in call["picks"].items()}))
    print("share values", json.dumps(out["values"]["share"]))
    print("gain values", json.dumps(out["values"]["gain"]))


if __name__ == "__main__":
    main()
