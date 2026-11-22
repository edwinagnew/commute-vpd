"""P2.56 scoring on the CPU from saved readings only, no model.

Run 1 (no argument): per task, RL against SFT on the re-gating share by E222's rule (results/predictions.json "rules"),
written to results/score.json. Run 2 (--rb PATH): every proxy against definition-lead's R_b on this run's states and on
E220's RL repeat states (E222's readings of them, experiments/rl_67m/results/stage3/repeat_rl_s*.json); Spearman per
proxy over the states both read, written to results/score_rb.json, and the figure figures/proxies_vs_rb.png.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/procedural_rl/score.py
    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/procedural_rl/score.py --rb PATH
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from commute import rl_tasks as RT
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
E222 = REPO / "experiments/rl_67m/results/stage3"
S3 = load_experiment("rl_67m", "score_stage3")  # E222's readable / alike / apart / above
AN = load_experiment("proxies_vs_r", "analyse")  # P2.51's spearman
PLOT = load_experiment("procedural_rl", "plot")
GET = {  # proxy: its value in a state record (E211's scorecard keys, experiments/realistic_tasks/score2.py, and the KL)
    "share": lambda s: s["share"]["share"]["lo"],
    "prior": lambda s: s["prior"]["lo"],
    "steering": lambda s: s["steering"]["share"]["lo"],
    "erank": lambda s: s["erank"],
    "neuron": lambda s: s["neuron"]["k64"]["share"]["lo"],
    "neuron_kmatched": lambda s: s["neuron"]["kmatched"]["share"]["lo"],
    "untrained": lambda s: s["untrained"]["share"]["lo"]["svd"],
    "kl": lambda s: s["kl_saved"] if "kl_saved" in s else s["e220"]["kl_saved"],
}


def num(x: Any) -> float:
    return float("nan") if x is None else float(x)  # jsonable wrote nan as "nan"


def states() -> dict[str, dict[str, Any]]:
    """This run's state readings and E222's of E220's RL repeat states, keyed by the state file they read."""
    ours = [json.loads(f.read_text()) for f in sorted((RESULTS / "read").glob("*.json"))]
    e222 = [json.loads(f.read_text()) | {"task": "repeat (E220)"} for f in sorted(E222.glob("repeat_rl_s*.json"))]
    return {s["file"]: s for s in ours + e222}


# ----------------------------------------------------------------------------- run 1


def arm_call(rl: list[float] | None, sft: list[float] | None, g: float) -> str:
    """E222's rule on one task: readable seed lists (None: fewer than min_readable_seeds over the floor)."""
    if rl is None or sft is None:
        return "no call"
    if S3.apart(rl, sft, g):
        return "rl above" if S3.above(rl, sft) else "sft above"
    return "alike" if S3.alike(rl, sft, g) else "neither"


def run1(st: dict[str, dict[str, Any]], R: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in RT.PROCEDURAL:
        mine = [s for s in st.values() if s["task"] == task]
        if not mine:
            out[task] = {"call": "not run"}
            continue
        vals = {a: [num(GET["share"](s)) for s in mine if s["arm"] == a] for a in ("rl", "sft")}  # nan under the floor (realistic_reads.ratio)
        read = {a: S3.readable(v, R["min_readable_seeds"]) for a, v in vals.items()}
        out[task] = {"call": arm_call(read["rl"], read["sft"], R["smallest_gap"]), "share": vals}
    return out


# ----------------------------------------------------------------------------- run 2


def load_rb(path: Path) -> dict[str, float]:
    """FILL IN once definition-lead's R_b format is known: {state file, as the readings record it: R_b}. The guess
    below reads {"R_b": {file: value}}; any other shape stops here, naming its keys."""
    d = json.loads(path.read_text())
    if not isinstance(d.get("R_b"), dict):
        raise SystemExit(f"load_rb: {path} has keys {sorted(d)}; write its loader here")
    return {k: float(v) for k, v in d["R_b"].items()}


def run2(st: dict[str, dict[str, Any]], rb: dict[str, float]) -> dict[str, Any]:
    """Per proxy: Spearman with R_b over the states both read (a ratio under the floor is not read), and every point."""
    pts = [{"file": f, "task": s["task"], "arm": s["arm"], "R_b": rb[f], **{p: num(g(s)) for p, g in GET.items()}} for f, s in st.items() if f in rb]
    out: dict[str, Any] = {"n_states": len(pts), "missing_R_b": sorted(set(st) - set(rb)), "proxies": {}}
    for p in GET:
        a = {q["file"]: q[p] for q in pts if math.isfinite(q[p])}
        out["proxies"][p] = AN.spearman(a, {q["file"]: q["R_b"] for q in pts})
    return out | {"points": pts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb", type=Path, help="definition-lead's R_b file (run 2)")
    args = ap.parse_args(argv)
    st, R = states(), json.loads((RESULTS / "predictions.json").read_text())["rules"]
    s1 = run1(st, R)
    (RESULTS / "score.json").write_text(json.dumps(s1, indent=1) + "\n")
    print(json.dumps({t: v["call"] for t, v in s1.items()}))
    if args.rb:
        s2 = run2(st, load_rb(args.rb))
        (RESULTS / "score_rb.json").write_text(json.dumps(s2, indent=1) + "\n")
        print(json.dumps({p: (v["rho"], v["n"]) for p, v in s2["proxies"].items()}))
        print(PLOT.draw(s2["points"], list(GET), HERE / "figures" / "proxies_vs_rb.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
