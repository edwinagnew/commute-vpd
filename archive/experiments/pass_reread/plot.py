"""Figure for E48 stage 1: is the unchanged decomposition's masked model already poor at the cue of the cipher windows?
Per task's held-out windows, F = W0's KL to its own CI-masked model at the cue position, divided by the same KL per
position on the harness's 256 Pile rows; the stop line 1.5 and the "as on ordinary text" line 1.

Reads results/saved.json only. Usage (repo root): uv run python experiments/pass_reread/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
# (key in saved.json, label, group); groups: the stop rule's two ciphers, the identity control, the E34 references
ROWS = (("rand1", "cipher, 1 random pair", "stop"), ("rand32", "cipher, 32 random pairs", "stop"),
        ("ident32", "identity control, 32 copy pairs", "beside"), ("emoticon", "emoticon → o", "ref"),
        ("r2_t275_v2667", "rare target, ' in' → ' Fig'", "ref"), ("r1_t534_v369", "ladder rung 1", "ref"),
        ("r0_t352_v310", "ladder rung 0", "ref"))
COLOURS = {"stop": "#c0392b", "beside": "#eb6834", "ref": "#2a78d6"}
NAMES = {"stop": "cipher windows the stop rule reads", "beside": "identity control (same windows as 32 pairs)",
         "ref": "earlier tasks' cue windows, for reference"}
LINE = 1.5


def main() -> None:
    s1 = json.loads((RESULTS / "saved.json").read_text())["S1"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ys = range(len(ROWS))[::-1]
    for y, (key, _, group) in zip(ys, ROWS):
        ax.hlines(y, 0, s1["F"][key], color=COLOURS[group], lw=2, alpha=0.35)
        ax.scatter(s1["F"][key], y, s=70, color=COLOURS[group], edgecolor="white", linewidth=1.5, zorder=3,
                   label=NAMES[group])
        ax.annotate(f"{s1['F'][key]:.2f}", (s1["F"][key], y), xytext=(8, -3), textcoords="offset points", fontsize=8)
    ax.axvline(LINE, color="0.2", ls="--", lw=1.2, label="stop line 1.5: at or over it, the reading falls back")
    ax.axvline(1.0, color="0.55", ls=":", lw=1.2, label="1: as close to W0 as on ordinary text")
    ax.set_yticks(list(ys), [label for _, label, _ in ROWS], fontsize=9)
    ax.set_xlim(0, 2.6)
    ax.set_xlabel("W0's KL to its own masked model at the cue, ÷ the same on ordinary text", fontsize=9)
    ax.set_title("Is the unchanged decomposition already poor at the cipher's cue?\n"
                 "Yes: both random ciphers sit over the stop line", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    ax.legend(keep.values(), keep.keys(), fontsize=7.5, loc="lower right", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "e48_s1_fidelity.png", dpi=150)
    print(FIGURES / "e48_s1_fidelity.png")


if __name__ == "__main__":
    main()
