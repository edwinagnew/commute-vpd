"""E326's figure and per-task table (no GPU).

    ~/commute-agents/fleet/cpu-run python3 experiments/fig2_all_tasks/plot.py

Reads results/summary.json (score.py). Writes results/per_task.json (per featurization, task and side: the three
seeds' E_b over K, their mean, min and max) and figures/fig2_all_tasks.png: for VPD subcomponents and SAE latents,
every seed's trained and matched-random value per task, one row per task. The top-neuron panel is held (its identity
control is off on two states), so neurons go into per_task.json only.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FEATS = {"VPD": "VPD subcomponents", "top": "top neurons", "SAE": "SAE latents"}
PLOTTED = ("VPD", "SAE")


def per_task(summary: dict) -> dict:
    """{feat: {task: {"side": reuse|new, "trained"|"random": {"seeds", "mean", "min", "max"}}}}, tasks in score.py's order."""
    out: dict = {}
    for feat in FEATS:
        out[feat] = {}
        for task, read in summary["reading"][feat]["per_task"].items():
            entry: dict = {"side": read["side"]}
            for side in ("trained", "random"):
                vals = [r["value"] for r in summary["rows"]
                        if r["feat"] == feat and r["side"] == side and r["item"].rsplit("_s", 1)[0] == task]  # fmt: skip
                assert len(vals) == 3, (feat, task, side, vals)
                assert abs(mean(vals) - read[side]) < 1e-12, (feat, task, side)
                entry[side] = {"seeds": vals, "mean": mean(vals), "min": min(vals), "max": max(vals)}
            out[feat][task] = entry
    return out


def ordered(tasks: dict) -> list[str]:
    """Tasks whose known answer is reuse first, then new; score.py's order within each."""
    return [t for t in tasks if tasks[t]["side"] == "reuse"] + [t for t in tasks if tasks[t]["side"] == "new"]


def main() -> None:
    table = per_task(json.loads((RESULTS / "summary.json").read_text()))
    (RESULTS / "per_task.json").write_text(json.dumps(table, indent=1))
    order = ordered(table["VPD"])
    n_reuse = sum(table["VPD"][t]["side"] == "reuse" for t in order)
    labels = [f"{t} ({table['VPD'][t]['side']})" for t in order]
    fig, axes = FG.figure((7.5, 5.2), ncols=2)
    data = []
    for ax, feat in zip(axes, PLOTTED, strict=True):
        for side, method, offset in (("trained", "eb", -0.15), ("random", "random", 0.15)):
            groups = {lab: table[feat][t][side]["seeds"] for lab, t in zip(labels, order, strict=True)}
            FG.dots(ax, groups, method=method, xlabel="E_b over K, leftovers in", offset=offset)
            data += [{"feat": feat, "task": t, "side": side, "seed": s, "value": v}
                     for t in order for s, v in enumerate(table[feat][t][side]["seeds"])]  # fmt: skip
        ax.axhline(n_reuse - 0.5, color="#bbbbbb", linewidth=0.6)
        ax.set_xlim(0, 0.07)
        ax.set_title(FEATS[feat], fontsize=9)
        if feat != PLOTTED[0]:
            ax.set_yticklabels([""] * len(order))
    handles = [axes[0].collections[0], axes[0].collections[len(order)]]
    fig.legend(handles, ["fine-tune, one point per seed", FG.METHODS["random"].label + ", one per seed"],
               fontsize=7, loc="outside upper center", ncols=2, frameon=False)  # fmt: skip
    fig.suptitle("How much of each task's fine-tune write the needed set carries, against a matched random", fontsize=9)
    FG.save(fig, HERE / "figures" / "fig2_all_tasks.png", data=data,
            n="16 known-answer tasks x seeds 0-2, each state beside one random update of the same size; "
              "512 windows of 8 tokens; top-neuron panel held pending its identity check")  # fmt: skip


if __name__ == "__main__":
    main()
