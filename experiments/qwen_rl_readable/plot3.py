"""E178 run 3 figure: the shuffled-reward null's training reward rate and mean answer length by step, for every null
run of the cell and of E69, so the collapse and its timing show against the nulls that held. Reads the pods' step
records only (no model); writes figures/run3.png.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_rl_readable/plot3.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import figure, save

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
E69 = HERE.parent / "qwen_rl" / "results"
LAST = 100  # every 1e-6 null ended by step 100; run 2's nulls ran to 400, flat near zero after step 35
RUNS = [  # (label, file, colour, line width, line style)
    ("E69, 1e-6, seed 0 (held)", E69 / "lr1e-6w10" / "N.json", "#56B4E9", 1.0, "-"),
    ("E69, 1e-6, seed 70 (held)", E69 / "lr1e-6w10_s70" / "N.json", "#009E73", 1.0, "-"),
    ("run 1, 1e-6, seed 0 (held)", RES / "N.json", "#0072B2", 1.0, "-"),
    ("run 2, 3e-6, seed 0 (collapsed)", RES / "run2" / "N_s0.json", "#E69F00", 1.0, "--"),
    ("run 2, 3e-6, seed 70 (collapsed)", RES / "run2" / "N_s70.json", "#F0C24B", 1.0, "--"),
    ("run 3, 1e-6, seed 0 (collapsed, stopped at 90)", RES / "run3" / "N_s0.json", "#D55E00", 2.0, "-"),
    ("run 3, 1e-6, seed 70 (stopped at 20)", RES / "run3" / "N_s70_stop1.json", "#CC79A7", 2.0, "-"),
]
PANELS = [("reward_rate", "share of training answers right", 0.5), ("mean_len", "mean answer length, tokens", 100)]


def rows() -> list[dict]:
    out = []
    for label, path, *_ in RUNS:
        for s in json.loads(path.read_text())["steps"]:
            if s["step"] <= LAST:
                out.append({"run": label, "step": s["step"], "reward_rate": s["reward_rate"], "mean_len": s["mean_len"]})
    return out


def main() -> None:
    data = rows()
    fig, axes = figure((10.0, 6.0), nrows=2)
    for ax, (key, ylabel, floor) in zip(axes, PANELS, strict=True):
        for label, _, colour, width, style in RUNS:
            pts = [r for r in data if r["run"] == label]
            ax.plot([r["step"] for r in pts], [r[key] for r in pts], color=colour, linewidth=width, linestyle=style,
                    label=label)
        ax.axhline(floor, color="#777777", linewidth=0.8, linestyle=":", label="filed collapse floor (5 steps in a row under)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, LAST)
    axes[0].set_ylim(-0.02, 1.0)
    axes[1].set_xlabel("training step")
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[0].set_title("Run 3's seed-0 null at 1e-6 ran like the nulls that held until about step 65, then collapsed by step 90",
                      fontsize=9)
    save(fig, HERE / "figures" / "run3.png", data=data,
         n="Shuffled-reward null, 64 problems x 8 samples per step; each line is one pod's step records. Run 1's line "
           "lies under run 3's seed 0 to step 30, where the two were equal. Run 2's nulls "
           "ran to step 400 and stayed near zero after step 35 (cut here at step 100). Run 3 seed 70 ended at the filed "
           "step-20 reward-change check; its restart was not run.")


if __name__ == "__main__":
    main()
