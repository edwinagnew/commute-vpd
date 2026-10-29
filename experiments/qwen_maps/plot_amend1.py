"""Figure for E44 Amendment 1 (the layer-cut check) and 1b (consistent inputs): m(l), m_H(l) and u(l) on the same
axes against the cut layer l.

Reads results/amend1/amend1_score.json and results/amend1b/amend1b_score.json only (no model).
Writes figures/amend1_layer_cut.png.
Usage (repo root): python experiments/qwen_maps/plot_amend1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SCORE = HERE / "results" / "amend1" / "amend1_score.json"
SCORE_B = HERE / "results" / "amend1b" / "amend1b_score.json"
OUT = HERE / "figures" / "amend1_layer_cut.png"
U_COLOUR, M_COLOUR, MH_COLOUR = "#2a78d6", "#c0392b", "#1e8449"


def main() -> None:
    s = json.loads(SCORE.read_text())["score"]
    b = json.loads(SCORE_B.read_text())["score"]
    cuts = sorted(int(k) for k in s["m"])
    m = [s["m"][str(c)] for c in cuts]
    u = [s["u"][str(c)] for c in cuts]
    # m_H(0) = 0 and m_H(28) = 1 by definition: H(0) = W0, H(28) = FT.
    h_cuts = [0, *sorted(int(k) for k in b["m_H"]), 28]
    m_h = [0.0, *(b["m_H"][str(c)] for c in h_cuts[1:-1]), 1.0]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.axhline(0, color="0.6", lw=1)
    ax.axhline(1, color="0.6", lw=1, ls=":")
    ax.text(
        0.3,
        1.02,
        "the fine-tuned model's whole effect",
        color="0.4",
        fontsize=8,
        va="bottom",
    )
    ax.plot(
        cuts,
        u,
        "o-",
        color=U_COLOUR,
        lw=2,
        ms=8,
        label="u: share of the update's output effect made by\nthe fine-tuned weights of layers before the cut",
    )
    ax.plot(
        cuts,
        m,
        "s-",
        color=M_COLOUR,
        lw=2,
        ms=8,
        label="m, spliced inputs (Amendment 1): Γ sees fine-tuned\nactivations before the cut, original ones after",
    )
    ax.plot(
        h_cuts,
        m_h,
        "^--",
        color=MH_COLOUR,
        lw=2,
        ms=8,
        label="m_H, consistent inputs (Amendment 1b): Γ sees the\nmodel with fine-tuned weights before the cut",
    )
    l_star = s["l_star"]
    ax.axvline(l_star, color="0.5", lw=1, ls="--")
    ax.text(
        l_star - 0.3,
        0.3,
        f"scored cut, layer {l_star}:\nreproduction needs m ≥ half of u",
        fontsize=8,
        ha="right",
        color="0.3",
    )
    ax.plot(
        [l_star], [0.5 * s["u"][str(l_star)]], marker="_", ms=22, mew=2, color="0.3"
    )
    ax.plot([28.6], [s["s_taps"]], "D", color="0.15", ms=7)
    ax.plot([28.6], [s["s_reads"]], "D", mfc="white", mec="0.15", ms=7)
    ax.text(
        28.0,
        s["s_taps"] - 0.07,
        "residual taps only",
        fontsize=8,
        ha="right",
        color="0.15",
    )
    ax.text(
        28.0,
        s["s_reads"] - 0.07,
        "per-site readouts only",
        fontsize=8,
        ha="right",
        color="0.15",
    )
    ax.set_xticks(cuts)
    ax.set_xlim(-1, 30)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel(
        "cut layer: fine-tuned before it, original after (28 = all fine-tuned)"
    )
    ax.set_ylabel("share, 0 = original model, 1 = fine-tuned model")
    ax.set_title("Do Γ's masks reproduce the update where the update is made?")
    ax.legend(
        loc="upper center",
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.13),
        ncol=1,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    OUT.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
