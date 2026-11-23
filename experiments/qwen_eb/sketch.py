"""The card's sketch for P3.14 run 1: made-up numbers, one panel per rival's outcome. Not a result. The random band
and the trained values are scaled from the 67M (E257: random 0.011, trained pairs ~0.04, gain x2 0.76).

    ~/commute-agents/fleet/cpu-run python experiments/qwen_eb/sketch.py
"""

from __future__ import annotations

from pathlib import Path

from commute import figures as F

HERE = Path(__file__).resolve().parent
ARMS = ("RL", "RL, shuffled reward", "SFT, own samples")


def panel(ax, trained: dict[str, list[float]], title: str) -> list[dict[str, object]]:
    rand = {a: [0.010, 0.012] for a in ARMS}
    F.column_dots(ax, trained, method="regating", ylabel="write share E_b",
                  bands={a: (min(r), max(r)) for a, r in rand.items()})
    ax.axhline(0.76, color="#777777", linewidth=0.8, linestyle="--")
    ax.annotate("gain x2 control", (0.0, 0.76), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
    ax.set_ylim(0, 0.85)
    ax.set_title(title, fontsize=8)
    return [{"panel": title, "arm": a, "E_b": v} for a, vs in trained.items() for v in vs]


def main() -> None:
    fig, axes = F.figure((F.FULL_WIDTH[0], 2.8), ncols=2)
    data = panel(axes[0], {"RL": [0.09, 0.11], "RL, shuffled reward": [0.07, 0.08], "SFT, own samples": [0.05, 0.06]},
                 "Rival A: the update leans on M's own writes")
    data += panel(axes[1], {"RL": [0.012, 0.013], "RL, shuffled reward": [0.011, 0.012], "SFT, own samples": [0.010, 0.011]},
                  "Rival B: no more than a random change")
    fig.suptitle("Sketch (made-up numbers): E_b of the step-75 Qwen updates against matched random changes")
    F.save(fig, HERE / "figures" / "sketch.png", data=data,
           n="made-up numbers; 2 seeds per arm; band = matched random changes of the same per-matrix norm")


if __name__ == "__main__":
    main()
