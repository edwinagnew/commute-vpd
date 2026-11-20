"""E222's score (P2.41 stage 3), from run_stage3.py's per-state files only: no model. Per behaviour and arm, the share's
readable seeds (gain at or over MIN_GAIN); per pick, the filed call between the card's rivals and the third outcome
(results/predictions_E222.json "rules"); the scorecard baselines' seed values beside, each run through the same rule
(beside, not filed). Writes results/stage3_score.json and prints it.

    ~/commute-agents/fleet/cpu-run python experiments/rl_67m/score_stage3.py
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment

RESULTS = Path(__file__).resolve().parent / "results"
PICKS, ANCHOR = ("1", "2"), "anchor"
ARMS = ("rl", "shuffled", "own", "demo")
ROUTE = ("rl", "own", "demo")  # the arms rival a says read alike
RATIOS = ("share", "untrained", "untrained_rotated", "steering", "neuron", "neuron_kmatched")  # read through the rule
GET = load_experiment("realistic_tasks", "score2").GET  # E211's scorecard: every method's value in a state record


def num(x: Any) -> float:
    return float("nan") if x is None else float(x)  # jsonable wrote nan as "nan"


def readable(vals: list[float], min_seeds: int) -> list[float] | None:
    """The finite seed values, or None (unreadable) when fewer than min_seeds are finite."""
    ok = [v for v in vals if math.isfinite(v)]
    return ok if len(ok) >= min_seeds else None


def overlap(a: list[float], b: list[float]) -> bool:
    return max(min(a), min(b)) <= min(max(a), max(b))  # touching ranges overlap


def gap(a: list[float], b: list[float]) -> float:
    return abs(statistics.fmean(a) - statistics.fmean(b))


def alike(a: list[float], b: list[float], g: float) -> bool:
    return overlap(a, b) and gap(a, b) < g


def apart(a: list[float], b: list[float], g: float) -> bool:
    return not overlap(a, b) and gap(a, b) >= g


def above(a: list[float], b: list[float]) -> bool:
    return min(a) > max(b)


def pick_call(arms: dict[str, list[float] | None], anchor: list[float] | None, g: float) -> dict[str, Any]:
    """The filed rule on one pick: arms = readable seed values per arm (None unreadable), anchor = the taught anchor's."""
    got = {a: v for a, v in arms.items() if v is not None}
    unread = sorted(set(ARMS) - set(got)) + ([] if anchor is not None else ["anchor"])
    third = "shuffled" in got and "rl" in got and alike(got["shuffled"], got["rl"], g)
    route = [a for a in ROUTE if a in got]
    pairs = {f"{a}~{b}": alike(got[a], got[b], g) for a, b in itertools.combinations(route, 2)}
    refs = {"anchor": anchor, "shuffled": got.get("shuffled")}
    over = {f"{a}>{r}": above(got[a], v) for a in route for r, v in refs.items() if v is not None}
    rec = {"third (shuffled reads like RL)": third, "alike": pairs, "above": over, "not_read": unread}
    if len(got) < 2 or "rl" not in got or "demo" not in got:  # the rivals differ only on demo against RL
        return {"call": "no call: fewer than two arms, RL or demo under the floor"} | rec
    rival_a = all(pairs.values()) and all(over.values())
    rival_b = apart(got["demo"], got["rl"], g) and ("own" not in got or alike(got["own"], got["rl"], g))
    call = "rival a" if rival_a else "rival b" if rival_b else "neither"
    return {"call": call, "gap demo-rl": gap(got["demo"], got["rl"])} | rec


def card_call(calls: dict[str, str]) -> str:
    """Both picks' calls: the same call on every pick that has one; 'split' when they differ; 'no call' when none has."""
    had = {c for c in calls.values() if not c.startswith("no call")}
    return "no call" if not had else had.pop() if len(had) == 1 else "split"


def seeds(states: dict[str, Any], name: str, arm: str, method: str) -> list[float]:
    return [num(GET[method](s, "lo")) for s in states.values() if s["behaviour"] == name and s["arm"] == arm]


def score(states: dict[str, Any], names: dict[str, str], R: dict[str, Any]) -> dict[str, Any]:
    g, m = R["smallest_gap"], R["min_readable_seeds"]
    out: dict[str, Any] = {"values": {}, "calls": {}}
    for method in GET:
        out["values"][method] = {f"{names[p]} {a}": seeds(states, names[p], a, method) for p in (*PICKS, ANCHOR) for a in ARMS
                                 if any(s["behaviour"] == names[p] and s["arm"] == a for s in states.values())}  # fmt: skip
    for method in RATIOS:
        anchor = readable(seeds(states, names[ANCHOR], "demo", method), m)
        calls = {names[p]: pick_call({a: readable(seeds(states, names[p], a, method), m) for a in ARMS}, anchor, g) for p in PICKS}
        out["calls"][method] = {"picks": calls, "card": card_call({k: v["call"] for k, v in calls.items()}),
                                "filed": method == "share"}  # fmt: skip
    return out


def main() -> None:
    P = json.loads((RESULTS / "predictions_E222.json").read_text())
    jobs = {j: json.loads((RESULTS / f"stage3_{j}.json").read_text()) for j in ("ctl", *PICKS, ANCHOR)}
    assert all(j.get("done") for j in jobs.values()), {k: j.get("stopped") for k, j in jobs.items()}
    names = json.loads((RESULTS / "stage1.json").read_text())["calls"]
    names = {"1": names["picks"][0], "2": names["picks"][1], ANCHOR: names["anchor"]}
    states = {f.stem: json.loads(f.read_text()) for f in sorted((RESULTS / "stage3").glob("*.json"))}
    out = score(states, names, P["rules"]) | {"gauss": {names[j]: jobs[j]["gauss"]["gain"] for j in (*PICKS, ANCHOR)}}
    (RESULTS / "stage3_score.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for method, c in out["calls"].items():
        print(method, c["card"], json.dumps({k: v["call"] for k, v in c["picks"].items()}))
    print("share values", json.dumps(out["values"]["share"]))


if __name__ == "__main__":
    main()
