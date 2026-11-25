"""P2.82: each task's fine-tune shares beside the shares of random changes scaled to a one log-odds gain, and each random
change's KL against its fine-tune's. $0, CPU, saved files only:

    ~/commute-agents/fleet/cpu-run python3 experiments/random_logodds_share/plot.py            # figures/shares.png
    ~/commute-agents/fleet/cpu-run python3 experiments/random_logodds_share/plot.py --sketch   # figures/sketch.png

Rows are Fig `known`'s 16 tasks (nine built pairs, two sides shared), in known3.json's order. A random draw that never
reached the gain band (status cap or no_match) has no share: it is drawn in the KL panel only, hollow, at the scale of
its largest gain. Values only: no verdict and no counts of comparisons.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from commute.figures import FULL_WIDTH, METHODS, figure, save

HERE = Path(__file__).resolve().parent
KNOWN3 = HERE.parents[0] / "known_3seeds" / "results" / "known3.json"
REUSE, NEW, RANDOM = METHODS["built_reuse"].color, METHODS["built_new"].color, METHODS["random"].color


def num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) else math.nan  # nan is written as the string "nan"


def tasks() -> dict[str, tuple[str, list[str]]]:
    """task -> (side, its three states), from known3.json's share cells."""
    out: dict[str, tuple[str, list[str]]] = {}
    for c in json.loads(KNOWN3.read_text())["cells"]:
        if c["method"] == "share":
            for side in ("reuse", "new"):
                task = c["states"][side][0].rsplit("_s", 1)[0]
                out.setdefault(task, (side, c["states"][side]))
    return out


def records(rows: dict[str, tuple[str, list[str]]]) -> list[dict[str, Any]]:
    """One row per plotted point: the fine-tunes' shares and KL, each random draw's share, KL and KL ratio."""
    out = []
    for task, (side, states) in rows.items():
        for s in states:
            path = HERE / "results" / "states" / f"{s.replace('/', '_')}.json"
            if not path.is_file():
                continue
            rec = json.loads(path.read_text())
            own = rec["own"]
            out.append({"task": task, "side": side, "state": s, "kind": "fine-tune", "seed": "", "status": "",
                        "share": num(own["r_route_split"]), "gain": num(own["gain"]), "kl": num(own["kl"]), "kl_ratio": 1.0})  # fmt: skip
            for seed, d in rec["draws"].items():
                out.append({"task": task, "side": side, "state": s, "kind": "random", "seed": seed, "status": d["status"],
                            "share": num(d["r"]), "gain": num(d["gain"]), "size_rel": num(d["size_rel"]), "kl": num(d["kl"]),
                            "kl_ratio": num(d["kl"]) / num(own["kl"])})  # fmt: skip
    return out


def draw(axes: Any, names: list[str], data: list[dict[str, Any]]) -> None:
    a, b = axes
    y = {t: i for i, t in enumerate(names)}
    for p in data:
        i = y[p["task"]]
        if p["kind"] == "fine-tune":
            a.scatter(p["share"], i - 0.15, s=20, color=REUSE if p["side"] == "reuse" else NEW, marker="o", linewidths=0)
        else:
            ok = p["status"] == "ok"
            if ok:
                a.scatter(p["share"], i + 0.15, s=22, color=RANDOM, marker="X", linewidths=0)
            b.scatter(p["kl_ratio"], i, s=22, marker="X" if ok else "o", linewidths=0.8,
                      color=RANDOM if ok else "none", edgecolors=RANDOM)  # fmt: skip
    for ax in (a, b):
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_ylim(len(names) - 0.5, -0.5)
    b.set_yticklabels([])
    a.set_xlabel("re-gating share, log-odds form")
    b.set_xscale("log")
    b.set_xlabel("random change's KL / its fine-tune's KL")
    for colour, label, marker in ((REUSE, "fine-tune, task built to reuse", "o"), (NEW, "fine-tune, task built new", "o"),
                                  (RANDOM, "random change, gain about 1", "X")):  # fmt: skip
        a.scatter([], [], s=20, color=colour, marker=marker, label=label)
    b.scatter([], [], s=22, marker="o", color="none", edgecolors=RANDOM, label="never reached gain 1 (largest gain)")
    a.legend(fontsize=6, loc="best", frameon=False)
    b.legend(fontsize=6, loc="best", frameon=False)


def main() -> None:
    rows = tasks()
    data = records(rows)
    fig, axes = figure(size=(FULL_WIDTH[0], 5.0), ncols=2)
    draw(axes, list(rows), data)
    axes[0].set_title("Share of each fine-tune and of random changes that gain 1 log-odds", fontsize=8)
    axes[1].set_title("Random changes are not matched to KL", fontsize=8)
    save(fig, HERE / "figures/shares.png", data=data,
         n="per task: 3 fine-tune seeds, 3 random directions each (E228's seeds), each scaled to a gain of 1.00-1.10 "
           "log-odds on the trained answer; KL to the original model on the same rows, NOT matched; 67M model")  # fmt: skip


def sketch() -> None:
    """The planned figure with made-up numbers, one row of panels per rival's outcome."""
    rng = random.Random(0)
    rows = {f"task {k}": ("reuse" if k < 3 else "new", []) for k in range(6)}
    fig, axes = figure(size=(FULL_WIDTH[0], 5.5), ncols=2, nrows=2)
    for r, (rival, centre) in enumerate((("a", 0.45), ("b", 0.0))):
        data: list[dict[str, Any]] = []
        for i, (task, (side, _)) in enumerate(rows.items()):
            own = 0.85 if side == "reuse" else 0.25
            for _ in range(3):
                data.append({"task": task, "side": side, "kind": "fine-tune", "share": own + rng.gauss(0, 0.03), "kl_ratio": 1.0})
                spread = 0.05 if rival == "a" else 0.4
                data.append({"task": task, "side": side, "kind": "random", "status": "ok" if i != 5 else "cap",
                             "share": centre + rng.gauss(0, spread), "kl_ratio": 10 ** rng.uniform(0.3, 1.5)})  # fmt: skip
        draw(axes[r], list(rows), data)
        axes[r][0].set_title(f"MADE-UP rival {rival}: randoms read {'near the new side or above' if rival == 'a' else 'near 0, erratic in sign'}", fontsize=8)
        axes[r][1].set_title("each random's KL / its fine-tune's (not matched)", fontsize=8)
    save(fig, HERE / "figures/sketch.png", n="MADE-UP numbers: 6 tasks, 3 seeds and 3 random draws each", data=[{"sketch": "made-up"}])


if __name__ == "__main__":
    if sys.argv[1:] == ["--sketch"]:
        sketch()
    else:
        main()
