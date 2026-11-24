"""E### (P2.63) figure, CPU: each reading against the dial setting, one column per task, one row per reading; the dial
states by seed, E256's saved SFT and RL states to the right of the ordinary fine-tune; each panel titled with its rho
from score.json. Writes figures/dial.png.

    ~/commute-agents/fleet/cpu-run python3 experiments/reuse_dial/plot.py [--states DIR] [--score FILE] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commute.jobs import load_experiment

SC = load_experiment("reuse_dial", "score")
HERE = Path(__file__).resolve().parent
ROWS = [("E_b_K", "E_b over K (the definition)"), ("share", "re-gating share"), ("steering", "steering share"), ("erank", "effective rank"),
        ("kl", "KL from W0"), ("lens", "lens D (ADL)")]  # fmt: skip
SAVED = {"sft": ("#444444", "s"), "rl": ("#c0392b", "^"), "rlg": ("#e67e22", "v")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=HERE / "results" / "states")
    ap.add_argument("--score", type=Path, default=HERE / "results" / "score.json")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "dial.png")
    a = ap.parse_args()
    sc, by = json.loads(a.score.read_text()), SC.load_states(a.states)
    arms, tasks = sc["arms"], list(by)
    fig, axes = plt.subplots(len(ROWS), len(tasks), figsize=(3.2 * len(tasks), 2.2 * len(ROWS)), squeeze=False)
    for j, t in enumerate(tasks):
        saved = SC.saved(t)
        for i, (key, label) in enumerate(ROWS):
            ax = axes[i][j]
            for s in by[t]:
                if s["arm"] in arms and math.isfinite(s["x"][key]):
                    ax.plot(arms.index(s["arm"]) + 0.08 * (s["seed"] - 1), s["x"][key], "o", ms=4, color="#1f77b4", alpha=0.8 if s["reached"] else 0.25)
            for name, x in saved.items():
                arm = name.split("_")[1]
                if arm in SAVED and math.isfinite(x[key]):
                    ax.plot(len(arms) + 0.5, x[key], SAVED[arm][1], ms=4, color=SAVED[arm][0])
            r = sc["tables"][t]["rho"].get(key, {}).get("rho")
            ax.set_title(f"{t}: rho {r:.2f}" if r is not None else f"{t}: rho undefined", fontsize=8)
            ax.set_xticks([*range(len(arms)), len(arms) + 0.5], [*arms, "saved"], fontsize=6)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Readings against the reuse dial (left: gains only; right: the ordinary fine-tune; saved: E256 SFT square, RL triangles)", fontsize=9)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
