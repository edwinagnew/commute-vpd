"""P2.56 figures. `draw` is the cell's figure, one panel per proxy with R_b on x, colour by task, marker by arm; score.py
calls it on the real points. Run as a script, it draws the sketches with made-up numbers (the card's): the proxy figure
and the RL-vs-SFT share per task under each of the card's two rivals.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/procedural_rl/plot.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from commute import figures as F

HERE = Path(__file__).resolve().parent
TASKS = {"sequence": "#0072B2", "brackets": "#D55E00", "addition": "#009E73", "repeat (E220)": "#777777"}
ARMS = {"rl": ("o", "RL"), "sft": ("s", "SFT on demonstrations"), "random": ("x", "matched random")}
LABEL = {"share": "re-gating share", "prior": "W0's log-odds of the answer", "steering": "steering share", "erank": "effective rank",
         "neuron": "neuron share, k 64", "neuron_kmatched": "neuron share, k matched", "untrained": "untrained SVD share",
         "kl": "KL from W0 (nats per token)"}  # fmt: skip


def dot(ax: Axes, x: float, y: float, task: str, arm: str) -> None:
    """Filled for RL, open for SFT, a cross for random."""
    c, mk = TASKS[task], ARMS[arm][0]
    ax.scatter(x, y, marker=mk, s=20, facecolors="none" if arm == "sft" else c, edgecolors=c if mk != "x" else None)


def draw(points: list[dict[str, Any]], proxies: list[str], path: Path, note: str = "") -> Path:
    fig, axes = F.figure((7.5, 2.0 * math.ceil(len(proxies) / 4)), ncols=4, nrows=math.ceil(len(proxies) / 4))
    for ax, p in zip(axes.flat, proxies, strict=False):
        for q in points:
            if math.isfinite(q[p]):
                dot(ax, q["R_b"], q[p], q["task"], q["arm"])
        ax.set_title(LABEL[p], fontsize=7)
        ax.set_xlabel("R_b", fontsize=7)
    for ax in list(axes.flat)[len(proxies) :]:
        ax.axis("off")
    handles = [Line2D([], [], color=c, marker="o", linestyle="", label=t) for t, c in TASKS.items()]
    handles += [Line2D([], [], color="black", marker=m, markerfacecolor="none" if a == "sft" else "black", linestyle="", label=lab)
                for a, (m, lab) in ARMS.items()]  # fmt: skip
    fig.legend(handles=handles, loc="outside right center", ncols=1, fontsize=7, frameon=False)
    fig.suptitle("Does each cheap proxy rank the procedural fine-tunes as R_b does?" + note, fontsize=9)
    return F.save(fig, path, data=points, n="per task (colour) and arm (marker), one point per seed: 3 seeds; E220's RL repeat states grey")


def made_up() -> list[dict[str, Any]]:
    rng, pts = random.Random(0), []
    for task, base in (("sequence", 0.5), ("brackets", 0.3), ("addition", 0.6)):
        for arm, shift in (("rl", 0.15), ("sft", -0.1), ("random", float("nan"))):
            for s in range(3):
                rb = base + (0 if math.isnan(shift) else shift) + rng.gauss(0, 0.05) - (0.4 if math.isnan(shift) else 0)
                share = float("nan") if arm == "random" else rb * 0.9 + rng.gauss(0, 0.05)
                pts.append({"state": f"{task}_{arm}_s{s}", "task": task, "arm": arm, "R_b": rb, "share": share, "prior": -2 + 2 * base + rng.gauss(0, 0.1),
                            "steering": float("nan") if arm == "random" else 0.5 + rng.gauss(0, 0.2), "erank": (700 if arm == "random" else 560) + rng.gauss(0, 10),
                            "neuron": float("nan") if arm == "random" else 0.02 + rng.gauss(0, 0.01),
                            "neuron_kmatched": float("nan") if arm == "random" else 0.01 + rng.gauss(0, 0.005),
                            "untrained": float("nan") if arm == "random" else 0.1 + rng.gauss(0, 0.05), "kl": (0.05 if arm == "random" else 0.4) + abs(rng.gauss(0, 0.1))})  # fmt: skip
    for s, rb in enumerate((0.7, 0.72, 0.78)):
        pts.append({"state": f"repeat_rl_s{s}", "task": "repeat (E220)", "arm": "rl", "R_b": rb, "share": rb, "prior": 3.6, "steering": 0.3, "erank": 534,
                    "neuron": 0.02, "neuron_kmatched": 0.005, "untrained": 0.1, "kl": 3.1})  # fmt: skip
    return pts


def rivals() -> Path:
    """The card's sketch: share per task, RL and SFT seeds, under rival a (RL above SFT by 0.1+) and rival b (within 0.1)."""
    fig, axes = F.figure((7.0, 2.4), ncols=2)
    rng, data = random.Random(1), []
    for ax, (name, gap) in zip(axes, (("rival a: RL above SFT by 0.1 or more", 0.25), ("rival b: RL and SFT within 0.1", 0.02)), strict=True):
        for i, task in enumerate(("sequence", "brackets", "addition")):
            for arm, dx, d in (("rl", -0.12, gap / 2), ("sft", 0.12, -gap / 2)):
                for s in range(3):
                    y = 0.5 + d + rng.gauss(0, 0.04)
                    dot(ax, i + dx, y, task, arm)
                    data.append({"panel": name, "task": task, "arm": arm, "seed": s, "share": y})
        ax.set_xticks(range(3), ["sequence", "brackets", "addition"], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("re-gating share", fontsize=7)
        ax.set_title(name, fontsize=8)
    fig.suptitle("Sketch, made-up numbers: RL (filled circles) against SFT (open squares), 3 seeds", fontsize=9)
    return F.save(fig, HERE / "figures" / "sketch_rivals.png", data=data, n="made up: 3 seeds per arm per task")


if __name__ == "__main__":
    print(draw(made_up(), list(LABEL), HERE / "figures" / "sketch_proxies_vs_rb.png", note="\n(sketch, made-up numbers)"))
    print(rivals())
