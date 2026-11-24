"""P2.76 (E292's difficulty figure), CPU, no model, descriptive: the gain of the old tasks and of E272's realistic panel
read at one KL, by E272's own rule (`panel.nearest` / `panel.at_rung`: the state whose KL is nearest the target in
ratio, within match_tol, with gain at least the floor).

  old at kl_hi      each old state (summary.json "difficulty": its scored state's KL and read gain) matched to E272's
                    wide kl_hi target; an old run has one state, the emoticon's E1 run four (steps 10-300). No old run
                    logged KL per step, so a state matches only if its scored state already sits in the window.
  realistic at old  E272's wide kl_hi tasks' saves (train/TASK.json: per save KL and held gain) matched to the old
                    states' median KL, the reverse match; beside it the same field (held gain) at kl_hi.

    ~/commute-agents/fleet/cpu-run --mem 6G -t 20 python3 experiments/old_tasks_read/matched_gain.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from commute import panel as PN
from commute.jobs import hard_exit, load_experiment
from commute.open_pair import jsonable

RUN: Any = load_experiment("convergence_panel", "run")  # E272: trained (task -> seed -> saves, floors), RESULTS
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WIDE = RUN.RESULTS / "wide"
OLD_FLOOR = 1.0  # plot.py's gain floor (under_gain_floor); every old state's gain is above it


def old_runs() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """task -> run -> states as `panel.nearest` reads them (kl, held_gain = the read gain, state)."""
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for s, d in json.loads((RESULTS / "summary.json").read_text())["difficulty"].items():
        run = "e1" if "_e1_" in s else s
        runs.setdefault(d["task"], {}).setdefault(run, []).append({"state": s, "kl": d["kl"], "held_gain": d["gain"]})
    return runs


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def main() -> int:
    tol = json.loads((RUN.RESULTS / "predictions.json").read_text())["settings"]["match_tol"]
    sel = json.loads((WIDE / "selection.json").read_text())["kl"]["hi"]
    kl_hi = sel["target"]
    old = old_runs()
    states = [x for by in old.values() for st in by.values() for x in st]
    old_at_hi = {t: {r: PN.nearest(st, "kl", kl_hi, tol, OLD_FLOOR) for r, st in by.items()} for t, by in old.items()}
    kl_old = median([x["kl"] for x in states])
    names = json.loads((WIDE / "dropped.json").read_text())["kept"]["kl_hi"]
    runs, floors, _ = RUN.trained(names)
    rev = PN.at_rung(runs, "kl", kl_old, tol, floors)
    held_hi = [s["held_gain"] for by in sel["tasks"].values() for s in by.values()]
    held_old = [s["held_gain"] for by in rev["tasks"].values() for s in by.values()]
    matched_old = [m["held_gain"] for by in old_at_hi.values() for m in by.values() if m]
    out = {
        "experiment": "E292", "cell": "P2.76", "label": "descriptive", "match_tol": tol, "kl_hi_target": kl_hi, "old_floor": OLD_FLOOR,
        "old_at_kl_hi": {t: {r: (m and {k: m[k] for k in ("state", "kl", "held_gain")}) for r, m in by.items()} for t, by in old_at_hi.items()},
        "old_states_matched_at_kl_hi": len(matched_old), "old_states": len(states),
        "old": {"median_kl": kl_old, "median_gain": median([x["held_gain"] for x in states]), "kl_range": [min(x["kl"] for x in states), max(x["kl"] for x in states)]},
        "realistic_at_old_kl": {"target": kl_old, "tasks_in": len(rev["tasks"]), "out": rev["out"], "median_held_gain": median(held_old),
                                "tasks": {t: {sd: {k: s[k] for k in ("step", "kl", "held_gain")} for sd, s in by.items()} for t, by in rev["tasks"].items()}},
        "realistic_at_kl_hi": {"states": len(held_hi), "median_held_gain": median(held_hi)},
    }  # fmt: skip
    out["ratio_old_over_realistic_at_old_kl"] = out["old"]["median_gain"] / out["realistic_at_old_kl"]["median_held_gain"]
    (RESULTS / "matched_gain.json").write_text(json.dumps(jsonable(out), indent=1))
    print(f"kl_hi {kl_hi:.4f} (tol {tol}): {len(matched_old)}/{len(states)} old states in the window, gains {matched_old}")
    print(f"old: median KL {kl_old:.4f} (range {out['old']['kl_range']}), median gain {out['old']['median_gain']:.3f}")
    print(f"realistic at KL {kl_old:.4f}: {len(rev['tasks'])} tasks in, out {sorted(rev['out'])}; median held gain {median(held_old):.3f}")
    print(f"realistic at kl_hi: median held gain {median(held_hi):.3f} over {len(held_hi)} states; ratio at old KL {out['ratio_old_over_realistic_at_old_kl']:.2f}")
    return 0


if __name__ == "__main__":
    hard_exit(main())
