"""E167 run 2 figure: held-out GSM8K pass of RL, its shuffled-reward null and the own-samples fine-tune at each matched
state, beside the unchanged model and the earlier run's (E69, peak 1e-6, step 100) RL and null on the same two seeds.
Reads results/run2_reads/{controls,verdict}.json and qwen_rl's E69 results; writes figures/run2.png.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_rl_readable/plot2.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import figure, reference, save

HERE = Path(__file__).resolve().parent
READS = HERE / "results" / "run2_reads"
E69 = HERE.parent / "qwen_rl" / "results"
ARMS = {  # arm: (label, colour, marker)
    "R": ("RL (reward = correct)", "#0072B2", "o"),
    "N": ("shuffled-reward null", "#555555", "X"),
    "F": ("fine-tune on own correct answers", "#009E73", "D"),
}
ROWS = [  # (row label, seed, step) for the run-2 states the verdict read
    ("2x, seed 0, step 75", "0", 75),
    ("2x, seed 70, step 75", "70", 75),
    ("5x, seed 70, step 250\n(seed 0 never reached 5x)", "70", 250),
]
EARLIER = [("earlier run, peak 1e-6,\nstep 100, seed 0", "lr1e-6w10"), ("earlier run, peak 1e-6,\nstep 100, seed 70", "lr1e-6w10_s70")]


def run2_rows() -> list[dict]:
    states = json.loads((READS / "verdict.json").read_text())["states"]
    return [{"row": label, "arm": arm, "seed": seed, "step": step, "pass": states[f"{arm}_s{seed}@{step}"]["pass"],
             "multiple": states[f"{arm}_s{seed}@{step}"]["multiple"]}
            for label, seed, step in ROWS for arm in ARMS]


def earlier_rows() -> list[dict]:
    out = []
    for label, folder in EARLIER:
        for arm in ("R", "N"):
            acc = json.loads((E69 / folder / f"{arm}.json").read_text())["accuracy"]["gsm8k"]
            out.append({"row": label, "arm": arm, "seed": "0" if folder == "lr1e-6w10" else "70", "step": 100,
                        "pass": acc["strict_dropped"], "multiple": None})
    return out


def main() -> None:
    w0 = json.loads((READS / "controls.json").read_text())["controls"]["W0"]["pass"]
    data = run2_rows() + earlier_rows()
    labels = list(dict.fromkeys(r["row"] for r in data))
    fig, ax = figure((7.5, 4.6))
    for arm, (name, colour, marker) in ARMS.items():
        rows = [r for r in data if r["arm"] == arm]
        ax.scatter([r["pass"] for r in rows], [labels.index(r["row"]) for r in rows], color=colour, marker=marker,
                   s=46, zorder=3, label=name)
    ax.axhline(len(ROWS) - 0.5, color="#bbbbbb", linewidth=0.8)
    reference(ax, w0, f"unchanged model {w0:.3f}")
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("held-out GSM8K pass rate (answers that fit the window)")
    ax.set_title("RL sits far above its shuffled-reward null at 2x, but this run's null had stopped answering", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower left")
    save(fig, HERE / "figures" / "run2.png", data=data,
         n="1319 GSM8K test problems, greedy; run 2 read on one L40, the earlier run (E69) on an H100. "
           "Sizes are RL's output change on 128 maths rows as a multiple of the masked error, 0.0932 nats per token; "
           "the null and fine-tune are saved at RL's step, not at its size.")


if __name__ == "__main__":
    main()
