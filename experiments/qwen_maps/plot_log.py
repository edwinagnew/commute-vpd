"""The research log's three figures for E44, redrawn from saved results with plain labels (no model).

  log_masks.png   KL from the fine-tuned model to the masked model of the same, unchanged subcomponents, with Γ's
                  masks computed from the original model's activations (K(D), D.json) or from the fine-tuned model's
                  (do-nothing, D.json), against the original model's own floor (C.json), steps 1 and 10.
  log_kept.png    share of the update kept (r2, D.json and K.json) for do-nothing, both halves and the random change,
                  steps 1 and 10, against the 0.8 the pass test asks for.
  log_ratios.png  the faithfulness ratios (masked KL and replayed adversary, maths and web rows) at step 10 for the same
                  four states (D.json, K.json), against the 1.5 pass line.

Reads results/C.json, D.json, K.json.
Usage (repo root):  python experiments/qwen_maps/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from plot import LINE, STATES, load, save

KEYS = ["do_nothing", "lower", "upper", "random"]
NAMES = {
    "do_nothing": "do-nothing\n(whole update in Δ)",
    "lower": "split onto the\nrarely-on half",
    "upper": "split onto the\noften-on half",
    "random": "random change\nof the same size",
}
KEPT_LINE = 0.8  # the pass test's line for the share of the update kept


def style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def records(D: dict[str, Any], K: dict[str, Any], t: str) -> dict[str, dict[str, Any]]:
    return {"do_nothing": D[t]["do_nothing"]} | {k: K[t][k] for k in KEYS[1:]}


def fig_masks(C: dict[str, Any], D: dict[str, Any]) -> None:
    bars = (
        ("masks from the original model's activations\n(the update dropped)", "#8c8c8c",
         lambda t: D[t]["KD"]),
        ("masks from the fine-tuned model's activations\n(do-nothing)", STATES["do_nothing"][1],
         lambda t: D[t]["do_nothing"]["maths"]["K"]),
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    w = 0.36
    for i, t in enumerate(("1", "10")):
        for b, (lab, col, f) in enumerate(bars):
            v = f(t)
            ax.bar(i + (b - 0.5) * w, v, w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(i + (b - 0.5) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(C["ci"]["maths"], color="0.3", ls=":", lw=1.2,
               label=f"floor: the original model against its own masked model ({C['ci']['maths']:.2f})")
    ax.set_xticks([0, 1], ["after 1 fine-tuning step", "after 10 fine-tuning steps"])
    ax.set_ylabel("distance from the fine-tuned model\nto the masked model, maths rows (nats)")
    ax.set_ylim(0, 1.75)
    ax.set_title("Same subcomponents, update left out of them: on maths rows, Γ's masks alone\n"
                 "bring the masked model almost to the fine-tuned one", fontsize=10.5)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_masks")


def fig_kept(D: dict[str, Any], K: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.3))
    w = 0.36
    for j, t in enumerate(("1", "10")):
        recs = records(D, K, t)
        for i, k in enumerate(KEYS):
            v = recs[k]["r2"]["r2"]
            x = i + (j - 0.5) * w
            ax.bar(x, v, w * 0.92, color=STATES[k][1], alpha=0.45 if t == "1" else 1.0)
            ax.text(x, max(v, 0) + 0.02, f"{abs(v) if abs(v) < 0.005 else v:.2f}", ha="center", fontsize=8)
            ax.text(x, -0.07, f"step {t}", ha="center", fontsize=7, color="0.35")
    ax.axhline(KEPT_LINE, color="#c0392b", ls="--", lw=1.2, label="the pass test asks for at least 0.8")
    ax.plot([-0.4, 0.4], [0, 0], color="k", lw=3, zorder=5, label="expected for do-nothing: about 0")
    ax.axhline(0, color="0.55", lw=0.8)
    ax.set_xticks(range(len(KEYS)), [NAMES[k] for k in KEYS], fontsize=8.5)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylabel("share of the update kept by the masked model\n(0 = none, 1 = all)")
    ax.set_ylim(-0.12, 1.15)
    ax.set_title("All three placements that hold the fine-tuned weights keep about 97% of the update;\n"
                 "only the random change, which does not hold them, keeps none", fontsize=10.5)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    save(fig, "log_kept")


def fig_ratios(D: dict[str, Any], K: dict[str, Any]) -> None:
    reads = {"r1m": "masked model,\nmaths rows", "r1w": "masked model,\nweb rows",
             "r3m": "saved adversary,\nmaths rows", "r3w": "saved adversary,\nweb rows"}
    recs = records(D, K, "10")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.2
    for i, k in enumerate(KEYS):
        xs = [j + (i - 1.5) * w for j in range(len(reads))]
        vals = [recs[k][r] for r in reads]
        ax.bar(xs, vals, w * 0.92, color=STATES[k][1], label=NAMES[k].replace("\n", " "))
        ax.text(xs[0], vals[0] + 0.03, f"{vals[0]:.2f}", ha="center", fontsize=7.5)
    ax.axhline(LINE, color="#c0392b", ls="--", lw=1.2, label="pass line, 1.5")
    ax.axhline(1, color="0.55", lw=0.8)
    ax.text(3.55, 1.02, "unchanged\ndecomposition = 1", fontsize=7, color="0.4", va="bottom", ha="right")
    ax.set_xticks(range(len(reads)), list(reads.values()), fontsize=8.5)
    ax.set_ylabel("faithfulness ratio after 10 steps\n(this state's distance ÷ the unchanged decomposition's)")
    ax.set_ylim(0, 2.0)
    ax.set_title("The faithfulness ratios stay near 1: three states cross the line once, just,\n"
                 "and the random change reads 1.0 on all four", fontsize=10.5)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=3)
    save(fig, "log_ratios")


if __name__ == "__main__":
    C, D, K = load("C"), load("D"), load("K")
    fig_masks(C, D)
    fig_kept(D, K)
    fig_ratios(D, K)
