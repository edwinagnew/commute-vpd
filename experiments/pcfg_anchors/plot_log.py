"""The research log's figures for E55 Stage A and Amendment 1, drawn by plot.py's functions from the saved results with the
log's titles (findings, no filing words). Writes figures/log_{labels,reproduction,controls,amend1_trajectory}.png.

  log_labels.png        every fine-tuned state on the two checks (behaviour against patch), with the bars
  log_reproduction.png  count-b accuracy at 10K against Jain et al.'s Table 6
  log_controls.png      the patch by block on the pretrained models: positive controls and nulls
  log_amend1_trajectory.png  Amendment 1: along the slow all-spurious fine-tune, the shares answering count(a) + 1 and the midpoint

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/pcfg_anchors/plot_log.py
"""

from __future__ import annotations

from plot import FIG, fig_controls, fig_labels, fig_table6
from plot_amend1 import fig_trajectory


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_labels("log_labels.png", "Fast fine-tunes never use the count of a; slow ones with many spurious rows use it early, not by the end")
    fig_table6("log_reproduction.png", "The rebuilt grammar reproduces seven of the paper's nine fine-tune results")
    fig_controls("log_controls.png", "Swapping the count in moves the answer on the models that count, never on the nulls")
    fig_trajectory("log_amend1_trajectory.png", "The all-spurious fine-tune stops answering count(a) + 1 between 1K and 5K steps")


if __name__ == "__main__":
    main()
