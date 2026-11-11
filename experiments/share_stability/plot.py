"""E161 (P2.6): the every-position share of the code-text fine-tune per 64-row batch, beside how far each update lowers
held-out log-probability, for the two fine-tune states and the Gaussian of the step-100 update's per-matrix norm.

Saved results only. Reads `results/run.json` and `results/predictions.json` (the readability floor); writes
`figures/stability.png`, `.pdf` and `.csv`.

    ~/commute-agents/fleet/cpu-run experiments/share_stability/plot.py
"""

import json
from pathlib import Path
from typing import Any

from commute.figures import FULL_WIDTH, dots, figure, reference, save

HERE = Path(__file__).resolve().parent
STATES = {"stage1_step10": "fine-tune, step 10", "stage1_step100": "fine-tune, step 100", "stage2_gaussian": "Gaussian, step-100 norm"}
GAUSS = "stage2_gaussian"


def main() -> None:
    run: dict[str, Any] = json.loads((HERE / "results/run.json").read_text())
    floor = json.loads((HERE / "results/predictions.json").read_text())["thresholds"]["readable_G"]
    fig, (share_ax, fall_ax) = figure(FULL_WIDTH, ncols=2)
    shares: dict[str, list[float]] = {name: [] if key == GAUSS else run[key]["r_b"] for key, name in STATES.items()}
    dots(share_ax, shares, method="regating", xlabel="share of the fall reproduced\nby Γ's values alone")
    share_ax.set_xlim(0, 1.15)
    share_ax.text(0.05, 2, "no readable share:\none batch's fall is under the floor", va="center", fontsize=7, color="#555555")
    reference(share_ax, 1.0, "all of the fall")
    share_ax.set_title("Share, one dot per batch", pad=14)

    falls = {name: [-x for x in run[key]["G_b"]] for key, name in STATES.items()}
    xlabel = "fall in held-out log-probability,\nnats per token, log scale"
    dots(fall_ax, {k: ([] if k == STATES[GAUSS] else v) for k, v in falls.items()}, method="regating", xlabel=xlabel)
    dots(fall_ax, {k: (v if k == STATES[GAUSS] else []) for k, v in falls.items()}, method="random", xlabel=xlabel)
    fall_ax.set_xscale("log")
    fall_ax.set_xlim(4e-4, 0.15)
    fall_ax.set_yticklabels([])
    reference(fall_ax, floor, "readability floor")
    fall_ax.set_title("How far the update lowers the text", pad=14)
    for ax in (share_ax, fall_ax):
        ax.set_ylim(2.5, -0.5)

    fig.suptitle("The share holds near 0.8 on every batch;\nthe random update barely moves the text")
    data = [
        {"state": name, "batch": b, "G_b": run[key]["G_b"][b], "g_b": run[key]["g_b"][b], "r_b": None if run[key]["r_b"] is None else run[key]["r_b"][b]}
        for key, name in STATES.items()
        for b in range(len(run[key]["G_b"]))
    ]
    n = "5 batches of 64 held-out code-text rows per state, the same rows for all three; one fine-tune seed, one Gaussian draw; 67M model, L40"
    save(fig, HERE / "figures/stability.png", n=n, data=data)


if __name__ == "__main__":
    main()
