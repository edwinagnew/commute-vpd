"""Merge sgd_ladder's shard files and draw the one figure the decision reads: per task, each state's adversarial Pile
ratio against the pass line and the shard's inconclusive band (left), and its held-out KL elsewhere (right). Prints the
Q1 decision. Writes results/sgd_ladder.json (merged) and figures/ladder.png.

Usage (repo root): uv run python experiments/sgd_ladder/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LINE = 1.5
COLOURS = {"do_nothing": "0.6", "adam_1e-05": "C1", "adam_3e-06": "C3", "adam_1e-06": "C4"}  # SGD: C0


def merge() -> dict[str, Any]:
    shards = sorted((HERE / "results").glob("sgd_ladder_shard*.json"))
    tasks: dict[str, Any] = {}
    floors: dict[str, float] = {}
    for p in shards:
        s = json.loads(p.read_text())
        for t, rec in s["tasks"].items():
            tasks[t], floors[t] = rec, s["W0"]["pgd_floor"]
    scored = [r for r in tasks.values() if "score" in r]
    failing = [r["task"] for r in scored if r["score"]["e11_failed"]]
    passes = [r["task"] for r in scored if r["score"]["e11_failed"] and r["score"]["any_new_arm_passes"]]
    unsure = [r["task"] for r in scored if r["score"]["e11_failed"] and r["score"]["any_new_arm_inconclusive"]]
    decision = ("3b follows" if passes else "rerun the inconclusive states first" if unsure
                else "the frozen-Gamma failure holds across optimisers at step 10")
    return {"shards": [p.name for p in shards], "tasks": tasks, "pgd_floor": floors,
            "Q1": {"failing_tasks_scored": failing, "passes_on": passes, "inconclusive_on": unsure, "decision": decision}}


def main() -> None:
    m = merge()
    (HERE / "results" / "sgd_ladder.json").write_text(json.dumps(m, indent=1, sort_keys=True, default=float))
    print(json.dumps(m["Q1"], indent=1))
    names = list(m["tasks"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4))
    for i, t in enumerate(names):
        rec, f = m["tasks"][t], m["pgd_floor"][t]
        arms = rec["arms"]
        ax.fill_between([i - 0.4, i + 0.4], LINE / (1 + f), LINE * (1 + f), color="0.9", lw=0)
        for j, (name, a) in enumerate(arms.items()):
            x = i - 0.3 + 0.6 * j / max(len(arms) - 1, 1)
            c = COLOURS.get(name, "C0")
            ax.scatter(x, a["pile"]["pgd_kl"], color=c, label=name if i == 0 else None, zorder=3)
            bx.scatter(x, a["heldout"]["kl_other"], color=c, zorder=3)
        bx.hlines(rec["limit"], i - 0.4, i + 0.4, color="k", lw=0.8)
    ax.axhline(LINE, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("adversarial Pile KL / W0's (grey: inconclusive band)")
    bx.set_ylabel("held-out KL elsewhere, nats (line: rate rule's limit)")
    for a in (ax, bx):
        a.set_xticks(range(len(names)), names, rotation=30, fontsize=8)
    ax.legend(fontsize=7)
    fig.suptitle("Component training at step 10: SGD (selected rate) and Adam at three rates, beside the do-nothing")
    fig.tight_layout()
    out = HERE / "figures" / "ladder.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
