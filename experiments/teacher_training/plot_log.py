"""The research log's faithfulness figure for E34, redrawn from saved results in one panel (no model).

  log_faithful.png  per fine-tune (three tasks, SGD and Adam teachers): the adversarial ratio of the fine-tune left in
                    Δ (do-nothing) and of its two copies (task text, Pile text) at teacher-training step 1000 as bars from 1,
                    step 100 as dots, with the pass and fail lines.

Reads results/teacher_training_<task>.json.
Usage (repo root):  python experiments/teacher_training/plot_log.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from plot import COLOR, HERE, INK, LINE, R0, R1, EMO, TASKS, TEACHERS, floor, load, state, style

TASK = {R1: "' which' → ' was'", R0: "' it' → ' is'", EMO: "emoticon"}
TEACHER = {"sgd": "SGD", "adam": "Adam"}


def main() -> None:
    runs, _, _ = load(HERE / "results")
    f = floor(runs)
    pass_line, fail_line = LINE / (1 + f), LINE * (1 + f)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bars = (("nothing", "the fine-tune left in Δ (do-nothing)", "#b9b8b3"),
            ("A", "copy trained on the task's text", COLOR["A"]),
            ("B", "copy trained on ordinary Pile text", COLOR["B"]))
    labels = []
    for i, (task, opt) in enumerate((t, o) for t in TASKS for o in TEACHERS):
        labels.append(f"{TASK[task]}\n{TEACHER[opt]} fine-tune")
        for j, (arm, lab, col) in enumerate(bars):
            x = i + (j - 1) * 0.26
            if arm == "nothing":
                v = state(runs, task, opt, "nothing")["pgd"]
            else:
                v = state(runs, task, opt, arm, "1000")["pgd"]
                ax.scatter([x], [state(runs, task, opt, arm, "100")["pgd"]], s=16, color="k", zorder=4,
                           label="the same copy at step 100" if i == 0 and arm == "A" else None)
            ax.bar(x, v - 1, 0.24, bottom=1, color=col, label=lab if i == 0 else None, zorder=2)
            ax.text(x, v + (0.015 if v >= 1 else -0.015), f"{v:.2f}", ha="center", va="bottom" if v >= 1 else "top",
                    fontsize=7)
    ax.axhline(1, color=INK, lw=0.8, ls=":", zorder=1)
    ax.axhspan(pass_line, fail_line, color="#f0efec", zorder=0)
    ax.axhline(pass_line, color="g", ls="--", lw=0.9, label=f"pass line ({pass_line:.2f}); grey band: inconclusive")
    ax.axhline(fail_line, color="r", ls="--", lw=0.9, label=f"fail line ({fail_line:.2f})")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(0.8, 1.75)
    ax.set_ylabel("adversarial ratio\n(1 = unchanged decomposition)")
    ax.set_title("Every copy passes the check after 1,000 steps, including copies of the three fine-tunes that are "
                 "inconclusive themselves", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "log_faithful.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
