"""Figure for qwen_adversary (RunPod H100, 2026-10-27), read from results/run.log and results/adversary.json.
Plan: reports/plans/2026-10-27_plan_qwen-adversary-128.md; analysis: reports/analyses/2026-10-28_analysis_qwen-adversary-128.md.

  climb.png  the adversary's KL (nats, mean over every position) at each of its 20 steps, for the three starts on
             the 4 control rows and the three on all 128 rows, log scale. Steps 0-19 are the per-step lines of
             run.log (the KL before that step's update); step 20 is the final reading in adversary.json. Drawn on
             the same axis: the importance-masked KL on the 128 rows (m = g, Δ off), the filed "ceiling" KL(clean ‖ uniform),
             which the KL can pass when the broken output is confidently wrong, and half the ceiling (the filed "unusable" line).

Prints, unfiled, the growth of each start's KL over its last step (final / step 19).

Usage (repo root): ~/commute-agents/fleet/cpu-run env PYTHONPATH=$PWD/src python -P experiments/qwen_adversary/plot.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOG, RES, FIG = HERE / "results" / "run.log", HERE / "results" / "adversary.json", HERE / "figures"
STEP = re.compile(r"seed (\d) step (\d+): KL ([0-9.]+)")
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
ROWS4_C, ROWS128_C = "#eb6834", "#2a78d6"


def trajectories() -> dict[str, dict[int, list[float]]]:
    """Per-step KL by stage and seed; the 128-row stage starts after the references line."""
    out: dict[str, dict[int, list[float]]] = {"4": {}, "128": {}}
    stage = "4"
    for line in LOG.read_text().splitlines():
        if "references:" in line:
            stage = "128"
        m = STEP.search(line)
        if m:
            out[stage].setdefault(int(m[1]), []).append(float(m[3]))
    return out


def main() -> None:
    res = json.loads(RES.read_text())
    traj = trajectories()
    finals = {"4": res["control"]["readings"], "128": res["adv128"]}
    ref, ceiling = res["references"]["ci128"], res["references"]["ceiling128"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for stage, colour, ls, name in (("4", ROWS4_C, "--", "4 rows"), ("128", ROWS128_C, "-", "128 rows")):
        for seed, ys in sorted(traj[stage].items()):
            assert len(ys) == 20, (stage, seed, len(ys))
            ys = ys + [finals[stage][seed]]
            print(f"{name} seed {seed}: step 19 {ys[19]:.3f}, final {ys[20]:.3f}, growth {ys[20] / ys[19]:.2f}x")
            ax.plot(range(21), ys, ls, color=colour, lw=2, label=f"{name}, three random starts" if seed == 0 else None)
            ax.plot(20, ys[20], "o", ms=8, color=colour, mec="white", mew=2)
        mean = sum(finals[stage]) / 3
        ax.annotate(f"{name}: mean {mean:.1f} nats", (20, mean), xytext=(20.6, mean), va="center", color=INK, fontsize=9)
    for y, ls, x, text in ((ceiling, "-", 0.3, f"'ceiling': an even spread over every token,{ceiling:.1f}"),
                           (ceiling / 2, ":", 0.3, f"half the ceiling: the filed 'unusable' line, {ceiling / 2:.1f}"),
                           (ref, "-.", 8.5, f"importance-masked, no search, {ref:.2f}")):
        ax.axhline(y, color=MUTED, ls=ls, lw=1.2)
        ax.text(x, y * 1.12, text, color=MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlim(0, 25.5)
    ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel("adversary step (20 = the reading)")
    ax.set_ylabel("KL from the model's outputs, nats per token")
    ax.set_title("Does the adversary still break the unchanged decomposition\nwhen one setting must serve 128 rows?",
                 fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.33), frameon=False, fontsize=9)
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "climb.png", dpi=150)
    print("wrote", FIG / "climb.png")


if __name__ == "__main__":
    main()
