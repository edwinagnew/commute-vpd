"""E257 (cell P1.50) figures from results/summary.json (score.py).

figures/eb_pairs.png: E_b of every seed of each side of E187's nine pairs, the side built to reuse against the side
built to be new, beside the random update's E_b, the built reuse edits' range (P1.49) and the do-nothing's 1.
figures/eb_arms.png: E_b of every seed of each arm of the rl stage (E220, E218 repeat updates) and the manifest stage
(P2.56 procedural states), usable and degenerate seeds marked apart, beside the trained SFT pairs' range, the random
update and the do-nothing's 1. Each writes .png, .pdf and .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/rb_states/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from commute import figures as FG

HERE = Path(__file__).resolve().parent
PAIR_NAMES = {"e187_e62_copy_lookup": "copy / lookup", "e187_e46_copy_random": "identity / random map",
              "e187_e53_lowercase_cipher": "lowercase / cipher", "e187_e54_matched": "elicit / cipher",
              "e187_e54_own": "elicit / own cipher", "e187_open_pair_c3": "rank-1 / cipher (c3)",
              "e187_open_pair_c3_copy": "copy / cipher (c3)", "e187_e58_rank1_96": "rank-1 / cipher (96 keys)",
              "e187_e58_copy_prior_cipher": "copy / prior cipher"}  # fmt: skip
SIDES = {"reuse": ("regating", -0.15, "side built to reuse, each seed"), "new": ("neuron", 0.15, "side built to be new, each seed")}
ROUTES = {"rl": "RL", "shuffled": "RL, shuffled reward", "own": "SFT, own samples", "demo": "SFT, demonstrations",
          "sft": "SFT", "rlg": "RL, guarded", "random": "random", "randg": "random, guarded size"}  # fmt: skip
RUNS = {"E220": "repeat, size-matched", "E218": "repeat, KL-matched"}
ORDER = ["sft", "own", "demo", "rl", "rlg", "shuffled", "random", "randg"]  # row order within each run or task
MARKS = {"usable": ("regating", "trained seed"), "other": ("neuron", "RL seed that collapsed or failed its gate"),
         "random": ("random", "random update matched to that task's states")}  # fmt: skip
XLABEL = "E_b: share of the changed parts' write change that the base model's parts explain"


def pairs_figure(s: dict[str, Any]) -> None:
    fig, ax = FG.figure(FG.FULL_WIDTH)
    data = []
    for kind, (method, off, _) in SIDES.items():
        groups = {PAIR_NAMES[p]: [r["E_b"] for r in v[kind]["items"]] for p, v in s["pairs"].items()}
        FG.dots(ax, groups, method=method, xlabel=XLABEL, offset=off)
        data += [{"pair": PAIR_NAMES[p], "side": kind, "item": r["item"], "E_b": r["E_b"], "dw_rel_norm": r["dw_rel_norm"]}
                 for p, v in s["pairs"].items() for r in v[kind]["items"]]  # fmt: skip
    rnd = max(r["E_b"] for r in s["randoms"].values())
    b = s["built_reference_P1_49"]
    lo, hi = min(b[k]["E_b"] for k in ("x2", "x10", "rebuilt")), max(b[k]["E_b"] for k in ("x2", "x10", "rebuilt"))
    ax.axvspan(lo, hi, color="#0072B2", alpha=0.15, linewidth=0)
    FG.reference(ax, rnd, f"random update {rnd:.3f}")
    FG.reference(ax, 1.0, "do-nothing 1")
    ax.annotate(f"built reuse edits\n{lo:.2f} to {hi:.2f}", ((lo * hi) ** 0.5, 0.5), xycoords=("data", "axes fraction"),
                ha="center", fontsize=7, color="#0072B2")  # fmt: skip
    for p in ("reuse", "new"):
        data.append({"pair": "reference", "side": p, "item": "random update", "E_b": rnd, "dw_rel_norm": None})
    data += [{"pair": "reference", "side": "built", "item": k, "E_b": b[k]["E_b"], "dw_rel_norm": None} for k in ("x2", "x10", "rebuilt")]
    ax.set_xscale("log")
    ax.set_xlim(0.007, 1.3)
    handles: list[Artist] = [Line2D([], [], color=FG.METHODS[m].color, marker=FG.METHODS[m].marker, linestyle="", label=lab) for m, _, lab in SIDES.values()]
    handles.append(Patch(color="#0072B2", alpha=0.15, label="built gain and rebuilt reuse edits"))
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
    ax.set_title("Trained fine-tunes read far below the built reuse edits, and the new side is higher in six of nine pairs",
                 fontsize=8, pad=14)  # fmt: skip
    n = ("9 pairs, 2 to 8 seeds per side, the larger side scaled to the smaller's update size; E_b summed over every "
         "position of D, log scale; built edits from the 2026-11-21 rescore, random update from this job")  # fmt: skip
    FG.save(fig, HERE / "figures" / "eb_pairs.png", n=n, data=data)


def arm_rows(s: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Row label -> (arm key, arm record), the rl stage's arms first, then each task's."""
    key = lambda a: (a[0].split()[0], ORDER.index(a[0].split()[1]))
    rows = {f"{RUNS[a.split()[0]]}: {ROUTES[a.split()[1]]}": (a, v) for a, v in sorted(s["rl"].items(), key=key)}
    return rows | {f"{a.split()[0]}: {ROUTES[a.split()[1]]}": (a, v) for a, v in sorted(s["manifest"].items(), key=key)}


def mark(arm: str, status: str) -> str:
    return "random" if "rand" in arm else "usable" if status == "usable" else "other"


def arms_figure(s: dict[str, Any]) -> None:
    fig, ax = FG.figure((7.5, 6.5))
    rows = arm_rows(s)
    for m, (method, _) in MARKS.items():
        groups = {lab: [x["E_b"] for x in v["seeds"].values() if mark(a, x["status"]) == m] for lab, (a, v) in rows.items()}
        FG.dots(ax, groups, method=method, xlabel="E_b: share of write change the base model's parts explain")
    data = [{"row": lab, "arm": a, "seed": k, "status": x["status"], "E_b": x["E_b"], "untouched_share": x["untouched_share"]}
            for lab, (a, v) in rows.items() for k, x in v["seeds"].items()]  # fmt: skip
    it = s["items"]["trained"]
    rnd = max(r["E_b"] for r in s["items"]["randoms"].values())
    ax.axvspan(it["E_b_min"], it["E_b_max"], color="#CC79A7", alpha=0.15, linewidth=0)
    FG.reference(ax, rnd, f"random update {rnd:.3f}")
    FG.reference(ax, 1.0, "do-nothing 1")
    ax.axhline(len(s["rl"]) - 0.5, color="#bbbbbb", linewidth=0.6)
    data += [{"row": "reference", "arm": k, "seed": None, "status": None, "E_b": v, "untouched_share": None}
             for k, v in (("random update", rnd), ("SFT pairs min", it["E_b_min"]), ("SFT pairs max", it["E_b_max"]), ("do-nothing", 1.0))]  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(0.007, 1.3)
    handles: list[Artist] = [Line2D([], [], color=FG.METHODS[m].color, marker=FG.METHODS[m].marker, linestyle="", label=lab) for m, lab in MARKS.values()]
    handles.append(Patch(color="#CC79A7", alpha=0.15, label="range of the 94 trained SFT fine-tunes of the nine pairs"))
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
    ax.set_title("RL reads far above SFT on the repeat behaviour but close to SFT on the procedural tasks;\nevery trained arm reads above its random update",
                 fontsize=8, pad=14)  # fmt: skip
    n = ("3 seeds per arm; repeat rows: 200 read rows, last position; procedural rows: 256 held-out items per task, last "
         "position; E_b summed over D's positions, log scale; above the grey line the repeat updates, below the procedural tasks")  # fmt: skip
    FG.save(fig, HERE / "figures" / "eb_arms.png", n=n, data=data)


def main() -> None:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    pairs_figure(s["items"])
    arms_figure(s)


if __name__ == "__main__":
    main()
