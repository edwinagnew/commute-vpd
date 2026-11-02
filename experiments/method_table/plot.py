"""The table's one figure, from results/table.json only: each method's reuse and new readings on one axis of shares.

Run from the clone root: ~/commute-agents/fleet/cpu-run python experiments/method_table/plot.py
Only methods whose reading is a share of a gain or a behaviour are drawn; reverse fine-tuning (steps), neuron sparsity
(counts) and the size labeller are not on this scale. Blue and vermillion are the Okabe-Ito pair; shape repeats them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE = Path("experiments/method_table/results/table.json")
FIG = Path("experiments/method_table/figures/shares.png")
REUSE, NEW, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#777777"
SHOWN = [
    ("counting_c1b", "steering"), ("counting_c1b", "pruning"), ("counting_c1b", "probe"), ("counting_c1b", "r"),
    ("counting_c1", "steering"), ("counting_c1", "pruning"), ("e62_copy_lookup", "r"),
    ("e46_copy_random", "r"), ("e46_copy_random", "steering"), ("e46_copy_random", "neuron_patch"),
    ("e53_lowercase_cipher", "r"), ("e53_lowercase_cipher", "steering"),
    ("e54_matched", "r"), ("e54_matched", "steering"), ("e54_matched", "neuron_patch"), ("e54_matched", "readout_64"),
    ("open_pair_c3", "r"), ("open_pair_c3", "steering"), ("open_pair_c3", "pruning"),
]


def side(ax: Any, values: list[float], y: float, color: str, marker: str, face: str) -> None:
    ax.plot([min(values), max(values)], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter([sum(values) / len(values)], [y], s=46, marker=marker, facecolor=face, edgecolor=color, linewidth=1.4, zorder=3)


def main() -> None:
    table = json.loads(TABLE.read_text())
    rows = {r["id"]: r for r in table["rows"]}
    fig, ax = plt.subplots(figsize=(11, 8.2))
    y, last, ticks, labels = 0.0, None, [], []
    for row_id, method in SHOWN:
        row, cell = rows[row_id], rows[row_id]["cells"][method]
        if row["truth"]["how"] != last:
            y += 0.6
            ax.text(-0.33, y, f"Truth set {row['truth']['how']}", fontsize=10, fontweight="bold", color=INK, va="center")
            last = row["truth"]["how"]
            y += 0.8
        side(ax, cell["reading"]["new"], y, NEW, "s", "white")
        side(ax, cell["reading"]["reuse"], y, REUSE, "o", REUSE)
        value = cell["positive_control"]["value"]
        if value is not None:
            ax.scatter([value], [y], s=52, marker="D", facecolor="white", edgecolor=INK, linewidth=1.1, zorder=4)
        star = "*" if table["counting_note"] in cell["notes"] else ""
        ax.text(1.27, y, cell["verdict"] + star, fontsize=9, color=INK, va="center")
        ticks.append(y)
        labels.append(f"{row['name']}: {table['methods'][method]['name']}")
        y += 1
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.text(0.005, 0.25, "0: do-nothing", fontsize=8, color=MUTED, va="center")
    ax.plot([0.95, 1.15], [0.25, 0.25], color=INK, lw=2)
    ax.text(1.05, 0.0, "minimum gap 0.2", fontsize=8, color=INK, ha="center", va="center")
    ax.scatter([], [], s=46, marker="o", color=REUSE, label="Labelled reuse: seed mean and range")
    ax.scatter([], [], s=46, marker="s", facecolor="white", edgecolor=NEW, label="Labelled new: seed mean and range")
    ax.scatter([], [], s=52, marker="D", facecolor="white", edgecolor=INK, label="Positive control's reading, where a share exists")
    ax.set_yticks(ticks, labels, fontsize=9)
    ax.set_ylim(y, -0.3)
    ax.set_xlim(-0.33, 1.25)
    ax.set_xlabel("Reading, as a share of the fine-tune's gain or of the old behaviour")
    ax.set_title("Does a method's reading tell the two sides apart, and against what kind of truth?", loc="left", fontsize=12)
    ax.grid(axis="x", color="#e6e6e6", lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.025), fontsize=8.5, frameon=False, ncol=3)
    fig.text(0.5, 0.008, "* counting set: labels equal the two tasks, so a full score is not discrimination", fontsize=8.5, color=INK, ha="center")
    fig.savefig(FIG, dpi=150)


if __name__ == "__main__":
    main()
