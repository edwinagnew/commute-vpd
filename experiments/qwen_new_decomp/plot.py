"""E131's figures: the old (p-e9fcffd5) and new (p-9707d6ff) decompositions on the same rows, one figure per number.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_new_decomp/plot.py
        -> figures/adversary.png, figures/ci_kl.png, figures/active.png

Each figure has three row sets (T web text, G maths thinking off, G-think maths thinking on); per set, every row's
value for each decomposition as a dot, the mean over rows as a large marker with its row-bootstrap 95% interval, and
the new/old ratio with its interval as score.json holds it. The mean's interval uses score.py's resampling (B row
resamples, numpy seed SEED); the ratio's is read from score.json, not recomputed. Reference lines are the filed
references of results/predictions.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
score = load_experiment("qwen_new_decomp", "score")
ROW_SETS = {"T": "web text\n(FineWeb, 512 tokens)", "G": "maths, thinking off\n(GSM8K + Qwen's answer)",
            "G-think": "maths, thinking on\n(104 of 128 cut at 512)"}
DECOMPS = {"old": ("old decomposition (p-e9fcffd5)", "#2a78d6", -0.18),
           "new": ("new decomposition (p-9707d6ff)", "#eb6834", 0.18)}
INK, MUTED = "#2b2b2b", "#8a8a85"


def per_row(decomp: str, rows: str, number: str) -> np.ndarray:
    return np.asarray(json.loads((RESULTS / f"measure_{decomp}_{rows}.json").read_text())[number], dtype=np.float64)


def mean_interval(x: np.ndarray) -> tuple[float, float, float]:
    """Mean over rows and the 2.5th/97.5th percentiles of its row-bootstrap resamples (score.py's B and SEED)."""
    idx = np.random.default_rng(score.SEED).integers(0, len(x), size=(score.B, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def panel(number: str, ylabel: str, title: str, refs: list[tuple[float, str, float | None]], log: bool) -> None:
    """One figure: dots per row, mean with interval, ratio label. A ref is (value, label, x or None for a full line)."""
    ratios = json.loads((RESULTS / "score.json").read_text())["ratios"]
    rng = np.random.default_rng(1)  # jitter only
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, rows in enumerate(ROW_SETS):
        tops = []
        for decomp, (label, colour, dx) in DECOMPS.items():
            x = per_row(decomp, rows, number)
            ax.scatter(i + dx + rng.uniform(-0.07, 0.07, len(x)), x, s=9, color=colour, alpha=0.35, lw=0)
            m, lo, hi = mean_interval(x)
            ax.errorbar(i + dx, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=9, color=colour, mec=INK, mew=1,
                        ecolor=INK, capsize=4, label=label if i == 0 else "_nolegend_", zorder=3)
            ax.text(i + dx + 0.11, m, f"{m:#.3g}" if m < 100 else f"{m:,.0f}", fontsize=8, color=INK, va="center")
            tops.append(x.max())
        r = ratios[f"{rows}/{number}"]
        y = max(tops) * (1.35 if log else 1.06)
        ax.text(i, y, f"new / old {r['ratio']:.3f}\n95% [{r['lo95']:.3f}, {r['hi95']:.3f}]", ha="center",
                fontsize=8, color=INK)
    for value, text, x in refs:
        if x is None:
            ax.axhline(value, color=MUTED, lw=1, ls="--")
            ax.text(2.45, value, text, fontsize=7, color=INK, ha="right", va="bottom")
        else:
            ax.hlines(value, x - 0.12, x + 0.12, color=INK, lw=2)
            ax.text(x - 0.14, value, text, fontsize=7, color=INK, ha="right", va="center")
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(ROW_SETS)), list(ROW_SETS.values()), fontsize=9)
    ax.set_xlim(-0.95, 2.6)
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y * (1.8 if log else 1.08))
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{number if number != 'adv_kl' else 'adversary'}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    panel("adv_kl", "KL under the adversary's masks (nats, log scale)",
          "Is the new decomposition harder for the adversary to break? On maths text, about 7 times",
          [(1.70, "old, stored\nseed 0, H100: 1.70", -0.18)], log=True)
    panel("ci_kl", "KL with its own CI as masks (nats)",
          "Is the new decomposition more faithful with its own masks? On maths about the same; worse on web text",
          [(0.1417, "old, stored: 0.142", -0.18),
           (0.17, "new, bundle README: held-out chat text, 2,048 tokens", None)], log=False)
    panel("active", "subcomponents with CI > 0.1 per token, summed over sites",
          "Does the new decomposition use fewer subcomponents per token? Yes, 26 to 30% fewer on every row set",
          [(6200, "new, bundle README: chat text", None)], log=False)


if __name__ == "__main__":
    main()
