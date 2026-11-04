"""E80 Amendment 2's Qwen scorecard, laid out like the notes' lowercase scorecard (slide 5), with its thresholds and colours; from results/amend2.json, no model.

  figures/scorecard_qwen.png  Faithful, behaviour kept and share left in Δ for the four methods on the first update

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_d1_adversary/plot_scorecard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
GOOD, BETWEEN, BAD, INK, MUTED, REF = "#d3ead6", "#f6e6b4", "#f4c6c2", "#1a1a1a", "#555555", "#777777"
W, H, DPI = 2400, 1040, 200
ROWS = (("Do-nothing (reference)", "N", 1.0, 232), ("Least squares", "LS", None, 412), ("Adjoint split", "ADJ", None, 555), ("Teacher training", "A", 0.0, 699))
COLS = (("Faithful", "adversarial ratio, 1 = unchanged", 645), ("Share of log-prob gain", "behaviour kept, on GSM8K answers", 1193), ("Update left in Δ", "share of the update", 1742))
CELL_W, CELL_H = 447, 122


def grade(v: float, good: float, bad: float, higher: bool = False) -> str:
    """The slide's bands: ✓ at or past the good line, ✗ at or past the bad one (behaviour's bad line is strict), ~ between."""
    if higher:
        return "good" if v >= good else "bad" if v < bad else "between"
    return "good" if v <= good else "bad" if v >= bad else "between"


def main() -> None:
    d = json.loads((HERE / "results" / "amend2.json").read_text())
    score, left = d["score"], d["left_in_delta"]
    delta = {"N": 1.0, "LS": left["ls"]["plain_energy"], "ADJ": left["adjoint"]["plain_energy"], "A": 0.0}  # do-nothing and teacher training by construction
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for title, sub, x in COLS:
        cx = x + CELL_W / 2
        ax.text(cx, 42, title, ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(cx, 105, sub, ha="center", va="center", fontsize=10, color=MUTED)
        ax.text(cx, 202, "Qwen, GSM8K fine-tune", ha="center", va="center", fontsize=10.5, color=MUTED)
    colour, mark = {"good": GOOD, "between": BETWEEN, "bad": BAD}, {"good": "✓", "between": "~", "bad": "✗"}
    for label, key, _, y in ROWS:
        ax.text(610, y + CELL_H / 2, label, ha="right", va="center", fontsize=14, color=REF if key == "N" else INK)
        cells = ((f"{score['mean'][key]:.2f}", grade(score["mean"][key], 1.40, 1.61)),
                 (f"{score['kept'][key]:.2f}", grade(score["kept"][key], 0.9, 0.5, higher=True)),
                 (f"{delta[key]:.0%}", grade(delta[key], 0.10, 0.30)))
        for (text, band), (_, _, x) in zip(cells, COLS, strict=True):
            ax.add_patch(Rectangle((x, y), CELL_W, CELL_H, color=colour[band], lw=0))
            ax.text(x + CELL_W / 2, y + CELL_H / 2, f"{text} {mark[band]}", ha="center", va="center", fontsize=14, color=INK)
    for x, band, text in ((236, "good", "✓ good: ≤ 1.40; ≥ 0.9; ≤ 10%"), (816, "between", "~ in between (ratio 1.40 to 1.61 is inconclusive)"), (1660, "bad", "✗ bad: ≥ 1.61; < 0.5; ≥ 30%")):
        ax.add_patch(Rectangle((x, 1008), 58, 22, color=colour[band], lw=0))
        ax.text(x + 82, 1019, text, ha="left", va="center", fontsize=10, color=INK)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "scorecard_qwen.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
