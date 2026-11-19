"""E211's score (P2.38 run 2), from the six task files only: no model. Per pair, matched size and method, the scorecard's
call on the three seeds a side (`realistic.seed_call`), the call against the share's and against the filed direction
(results/predictions_run2.json), the card's rival from the share, and the controls' readings beside. The label pair is
scored in the two-label contrast with the whole-vocabulary log-odds beside; the LM pairs in log-odds.
Writes results/run2_score.json and prints it.

    ~/commute-agents/fleet/cpu-run python experiments/realistic_tasks/score2.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from commute import realistic as R

RESULTS = Path(__file__).resolve().parent / "results"
PAIRS = {"L1/L3": ("L1", "L3"), "T2/T3": ("T2", "T3"), "T1/T4": ("T1", "T4")}
LM_PAIRS = ("T2/T3", "T1/T4")
GET = {
    "share": lambda s, u: s["share"]["share"][u],
    "untrained": lambda s, u: s["untrained"]["share"][u]["svd"],
    "untrained_rotated": lambda s, u: s["untrained"]["share"][u]["rotated"],
    "steering": lambda s, u: s["steering"]["share"][u],
    "neuron": lambda s, u: s["neuron"]["k64"]["share"][u],
    "neuron_kmatched": lambda s, u: s["neuron"]["kmatched"]["share"][u],
    "k_half": lambda s, u: s["k_half"][u]["k_half"],
    "prior": lambda s, u: s["prior"][u],
    "erank": lambda s, u: s["erank"],
    "size": lambda s, u: s["size"],
}


def num(x: Any) -> float | None:
    return None if x is None else float(x)  # jsonable wrote nan as "nan"


def side(task: dict[str, Any], method: str, u: str, target: float) -> list[float | None]:
    """The method's value per seed at one size (the prior, a W0 reading equal across seeds, once)."""
    states = [s for s in task["states"].values() if s["target"] == target]
    vals = [num(GET[method](s, u)) for s in states]
    return vals[:1] if method == "prior" else vals


def pair_rows(tasks: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pid, (a, b) in PAIRS.items():
        units = ["contrast", "lo"] if pid == "L1/L3" else ["lo"]
        targets = sorted({s["target"] for s in tasks[a]["states"].values()})
        for i, t in enumerate(targets):
            for u in units:
                cells = {m: R.seed_call(m, side(tasks[a], m, u, t), side(tasks[b], m, u, t)) for m in GET}
                for m, c in cells.items():
                    c["against_share"] = None if m == "share" else R.against_share(c, cells["share"])
                    c["filed"] = R.filed(c, P["predictions"][pid][m]["direction"]) if u == units[0] else None
                out[f"{pid} size{i} {u}"] = {"pair": pid, "target": t, "unit": u, "cells": cells}
    return out


def controls(tasks: dict[str, Any]) -> dict[str, Any]:
    """Per task and size, the Gaussian draws' gain, numerators' range and erank; the states that missed their size; the
    label tasks' held-out gains (the coin's contrast should be near 0)."""
    out: dict[str, Any] = {}
    for name, t in tasks.items():
        rec: dict[str, Any] = {"not_reached": [k for k, s in t["states"].items() if not s["reached"]]}
        for k, g in t.get("gauss", {}).items():
            rec[k] = {"gain": g["gain"], "erank": g["erank"], "size": g["size"], "share": g["share"]["share"]}
        if name.startswith("L"):
            rec["held_gain"] = {k: s["held_gain"] for k, s in t["states"].items()}
        out[name] = rec
    return out


def main() -> None:
    P = json.loads((RESULTS / "predictions_run2.json").read_text())
    tasks = {n: json.loads((RESULTS / f"run2_{n}.json").read_text()) for p in PAIRS.values() for n in p}
    missing = [n for n, t in tasks.items() if not t.get("done")]
    assert not missing, f"task files not finished: {missing}"
    rows = pair_rows(tasks, P)
    share = {i: {pid: rows[f"{pid} size{i} {'contrast' if pid == 'L1/L3' else 'lo'}"]["cells"]["share"] for pid in PAIRS} for i in (0, 1)}
    by_size = {i: R.rival(share[i], LM_PAIRS) for i in (0, 1)}
    card = "rival a" if all(v == "rival a" for v in by_size.values()) else "rival b" if by_size[1] == "rival b" else "between"
    out = {"rows": rows, "rival_by_size": by_size, "card": card, "controls": controls(tasks)}
    (RESULTS / "run2_score.json").write_text(json.dumps(out, indent=1, default=lambda x: str(x)) + "\n")
    for k, r in rows.items():
        print(k, json.dumps({m: [c["call"], None if math.isnan(c["gap"]) else round(c["gap"], 4), c["filed"]] for m, c in r["cells"].items()}))
    print("card:", card, by_size)


if __name__ == "__main__":
    main()
