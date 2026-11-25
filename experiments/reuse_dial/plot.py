"""E300 (P2.63) figures, CPU.
  figures/rho.png   the claim's comparison: per task, each reading's rank correlation with the dial setting (left) and,
                    for the two cheap readings and the construction check, how far its size exceeds KL's (right; the
                    rule asks for at least 0.2 with p under 0.05; a cell that passes is boxed, marked with its sign)
  figures/dial.png  every state's reading against the dial setting, one column per task, the residual share on top;
                    E256's saved SFT and RL states to the right of the ordinary fine-tune

    ~/commute-agents/fleet/cpu-run python3 experiments/reuse_dial/plot.py [--states DIR] [--score FILE]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from commute import figures as FG
from commute.jobs import load_experiment

SC = load_experiment("reuse_dial", "score")
HERE = Path(__file__).resolve().parent
TASKS = ["sequence", "brackets", "addition", "lowercase"]
NAMES = {"steering": "steering share (cheap)", "erank": "effective rank (cheap)", "share": "re-gating share (check)",
         "kl": "KL from W0 (reference)", "lens": "lens D (ADL)", "neuron_k64": "neuron share, k=64",
         "neuron_kmatched": "neuron share, k matched", "E_b_K": "E_b over K (definition)"}  # fmt: skip
SCORED = ["steering", "erank", "share"]
ROWS = [("residual", "residual share (the dial)"), ("E_b_K", "E_b over K (definition)"), ("share", "re-gating share"),
        ("steering", "steering share"), ("erank", "effective rank"), ("kl", "KL from W0"), ("lens", "lens D (ADL)")]  # fmt: skip
SAVED = {"sft": ("#444444", "s", "E256 SFT"), "rl": ("#c0392b", "^", "E256 RL"), "rlg": ("#e67e22", "v", "E256 RL, guarded")}  # fmt: skip


def rho_figure(sc: dict[str, Any], out: Path) -> None:
    R = json.loads(SC.PRED.read_text())["rule"]
    tabs = [sc["tables"][t] for t in TASKS]
    rho = [[tb["rho"][k]["rho"] for tb in tabs] for k in NAMES]
    n = [[tb["rho"][k]["n"] for tb in tabs] for k in NAMES]
    kl = [abs(tb["rho"]["kl"]["rho"]) for tb in tabs]
    gap = [[abs(tb["rho"][k]["rho"]) - kl[j] for j, tb in enumerate(tabs)] for k in SCORED]
    fig, (a1, a2) = FG.figure((11.5, 4.8), ncols=2)
    FG.gap_heatmap(a1, rho, list(NAMES.values()), TASKS, label="rank correlation with the dial setting", vmax=1.0, counts=n)
    FG.gap_heatmap(a2, gap, [NAMES[k] for k in SCORED], TASKS, label="size of rank correlation minus KL's", vmax=1.0)
    for i, k in enumerate(SCORED):
        for j, tb in enumerate(tabs):
            if SC.beats_kl(tb, k, R):
                a2.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="black", linewidth=2))
                a2.text(j + 0.38, i - 0.32, "+" if tb["rho"][k]["rho"] > 0 else "−", ha="center", va="center", fontsize=12)
    a1.set_title("Rank correlation with the dial, per task", loc="left")
    a2.set_title("Size minus KL's size; boxed: passes (p<0.05, 0.2 or more), sign of rho", loc="left", fontsize=8)
    for ax in (a1, a2):
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Steering passes on sequence and lowercase only; effective rank passes with a sign that flips on lowercase")
    data = [{"reading": k, "task": t, "rho": tb["rho"][k]["rho"], "p": tb["rho"][k]["p"], "n": tb["rho"][k]["n"]}
            for k in NAMES for t, tb in zip(TASKS, tabs)]  # fmt: skip
    FG.save(fig, out, n="n = states with the reading per task: 15 (five dial settings x seeds 0-2); lowercase 10 (seed 1 not "
            "read)", data=data)  # fmt: skip


def value(s: dict[str, Any], key: str) -> float:
    return s["residual_share"] if key == "residual" else s["x"][key]


def dial_figure(sc: dict[str, Any], by: dict[str, list[dict[str, Any]]], out: Path) -> None:
    arms = sc["arms"]
    fig, axes = plt.subplots(len(ROWS), len(TASKS), figsize=(3.2 * len(TASKS), 2.2 * len(ROWS)), squeeze=False)
    for j, t in enumerate(TASKS):
        saved = SC.saved(t)
        for i, (key, label) in enumerate(ROWS):
            ax = axes[i][j]
            for s in by[t]:
                if s["arm"] in arms and math.isfinite(value(s, key)):
                    ax.plot(arms.index(s["arm"]) + 0.08 * (s["seed"] - 1), value(s, key), "o", ms=4, color="#1f77b4", alpha=0.8 if s["reached"] else 0.25)
            for name, x in saved.items():
                arm = name.split("_")[1]
                if key != "residual" and arm in SAVED and math.isfinite(x[key]):
                    ax.plot(len(arms) + 0.5, x[key], SAVED[arm][1], ms=4, color=SAVED[arm][0])
            r = sc["tables"][t]["rho"].get(key, {}).get("rho") if key != "residual" else sc["tables"][t]["dial_order"]
            ax.set_title(f"{t}: rho {r:.2f}" + (" (penalised arms)" if key == "residual" else "") if r is not None else f"{t}: rho undefined", fontsize=8)
            ax.set_xticks([*range(len(arms)), len(arms) + 0.5], [*arms, "saved"], fontsize=6)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=6)
    handles = [Line2D([], [], marker="o", ls="", color="#1f77b4", label="dial state (seeds 0-2)")]
    handles += [Line2D([], [], marker=m, ls="", color=c, label=lab) for c, m, lab in SAVED.values()]
    fig.legend(handles=handles, loc="upper right", fontsize=7, ncol=4)
    fig.suptitle("Readings against the reuse dial (left: gains only; then penalty 1e6, 1e5, 1e4; ft: the ordinary fine-tune)", fontsize=9, x=0.35)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=HERE / "results" / "states")
    ap.add_argument("--score", type=Path, default=HERE / "results" / "score.json")
    a = ap.parse_args()
    sc, by = json.loads(a.score.read_text()), SC.load_states(a.states)
    rho_figure(sc, HERE / "figures" / "rho.png")
    dial_figure(sc, by, HERE / "figures" / "dial.png")
    print("wrote figures/rho.png, figures/dial.png")


if __name__ == "__main__":
    main()
