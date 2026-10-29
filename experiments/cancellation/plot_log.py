"""The research log's figures for E17, redrawn from saved results with plain labels (no model).

  log_half_revert.png      do the changes cancel in the output? Per state, the cost of reverting a random half of
                           the changes as a share of reverting them all (filled, with the range of three halves),
                           beside what it would read if the changes' effects added the way their weights do
                           (hollow, (1 + 1/coherence) / 4). Lines at 0.25 (all one way), 0.5 (unrelated), 1 (cancel).
  log_adversary_share.png  how much of the adversary's extra damage is removing the changes? Per state, the share of
                           the adversary's excess over the original model carried by removing only its fraction of
                           each change (bar), the same with the fractions shuffled (tick), and the unmeasured rest.

The card also uses figures/half_revert_toy.png from plot.py, unchanged.
Reads results/cancellation.json.
Usage (repo root):  python experiments/cancellation/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot import COLOR, FIG, GROUP, load, save

# top to bottom
ROWS = (
    ("ct emoticon", "component training, emoticon"),
    ("ct r1_t588_v452", "component training, ' will' then ' have'"),
    ("ct r2_t359_v816", "component training, ' we' then ' just'"),
    ("ct r0_t352_v310", "component training, ' it' then ' is'"),
    ("ct r1_t534_v369", "component training, ' which' then ' was'"),
    ("ct r2_t275_v2667", "component training, ' in' then ' Fig'"),
    ("adjoint emoticon", "adjoint split of the fine-tune, emoticon"),
    ("ls emoticon", "least-squares split of the fine-tune, emoticon"),
    ("c1 emoticon", "control: component training, emoticon,\nits changes mixed to cancel in the weights"),
)
INK = "#333333"


def ylabels(ax: Any) -> None:
    ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS], fontsize=8.5)
    ax.set_ylim(len(ROWS) - 0.4, -0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_half_revert(out: dict[str, Any]) -> None:
    st = out["states"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axvspan(1, 10, color="#f4a582", alpha=0.3, zorder=0)
    for x, txt in ((0.25, "0.25\nall one way"), (0.5, "0.5\nunrelated"), (1, "1\ncancelling")):
        ax.axvline(x, color="#888888", ls=":", lw=1, zorder=1)
        ax.text(x, -0.75, txt, ha="center", va="bottom", fontsize=8, color="#444444")
    for i, (key, _) in enumerate(ROWS):
        r = st[key]
        halves = np.array(r["F_half"]) / r["F_full"]
        kappa = r["coherence"]["pooled"]
        expect = (1 + 1 / kappa) / 4
        c = COLOR[GROUP[key]]
        ax.plot([r["h"], expect], [i, i], color="#bbbbbb", lw=1, zorder=2)
        ax.plot(expect, i, "o", ms=8, mfc="white", mec=c, mew=1.5, zorder=3)
        ax.hlines(i, halves.min(), halves.max(), color="black", lw=1, zorder=4)
        ax.plot(r["h"], i, "o", ms=8, color=c, mec="black", mew=0.5, zorder=5)
        ax.text(9.5, i, f"{kappa:.2g}", ha="right", va="center", fontsize=8.5, color=INK)
    ax.text(9.5, -0.75, "coherence in weights\n(1 unrelated, <1 cancel)", ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.18, 10)
    ax.set_xticks([0.25, 0.5, 1, 2, 5], ["0.25", "0.5", "1", "2", "5"])
    ylabels(ax)
    ax.set_xlabel("cost of reverting a random half of the changes ÷ cost of reverting them all (log scale)")
    ax.set_title("The half revert reads below 1 on every state, but also on the control built to cancel,\nso this test is not validated",
                 fontsize=10.5, loc="left", pad=34)
    handles = [Line2D([], [], marker="o", color=COLOR["passes"], mec="black", mew=0.5, ls="",
                      label="measured, state passes the adversarial check"),
               Line2D([], [], marker="o", color=COLOR["fails"], mec="black", mew=0.5, ls="", label="measured, state fails it"),
               Line2D([], [], marker="o", mfc="white", mec=INK, ls="", label="expected from coherence: (1 + 1/coherence) ÷ 4"),
               Line2D([], [], color="black", lw=1, label="range of three random halves")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right", bbox_to_anchor=(0.92, 0.99), framealpha=0.95)
    save(fig, "log_half_revert.png")


def fig_adversary_share(out: dict[str, Any]) -> None:
    st = out["states"]
    w0 = out["W0"]["pgd_kl"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, (key, _) in enumerate(ROWS):
        r = st[key]
        x = r["pgd_kl"] - w0
        part = r["F_adv"] / x
        perm = float(np.mean(r["F_perm"])) / x
        ax.barh(i, part, 0.62, color=COLOR[GROUP[key]], zorder=3)
        ax.barh(i, 1 - part, 0.62, left=part, color="white", edgecolor="#b8b8b8", hatch="///", zorder=2)
        ax.plot([perm, perm], [i - 0.36, i + 0.36], color="black", lw=2, zorder=4)
        ax.text(part + 0.012, i, f"{part:.0%}", va="center", fontsize=8, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        ax.text(1.03, i, f"{r['pgd_kl'] / w0:.2f}", va="center", fontsize=8.5, color=INK)
        ax.text(1.18, i, f"{x:.2g}", va="center", fontsize=8.5, color=INK)
    ax.text(1.03, -0.75, "ratio\n(fail > 1.5)", va="bottom", fontsize=8, color=INK)
    ax.text(1.18, -0.75, "extra damage,\nnats", va="bottom", fontsize=8, color=INK)
    ax.axvline(0.5, color="black", ls="--", lw=1, zorder=6)
    ax.text(0.5, -0.75, "expected if the changes\ncarried the damage: half or more", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, 1.34)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ylabels(ax)
    ax.set_xlabel("share of the adversary's extra damage over the original model")
    ax.set_title("On every failing state, removing the adversary's share of the changes\n"
                 "carries under a tenth of its extra damage, no more than a shuffled pattern does", fontsize=10.5, loc="left", pad=30)
    handles = [Patch(color=COLOR["passes"], label="removing its fraction of each change only (state passes)"),
               Patch(color=COLOR["fails"], label="the same, state fails the adversarial check"),
               Line2D([], [], color="black", lw=2, label="the same fractions shuffled among subcomponents"),
               Patch(facecolor="white", edgecolor="#b8b8b8", hatch="///",
                     label="the rest, not measured here (E20: the mask on the\ndead subcomponents and Δ, with the alive ones)")]
    fig.legend(handles=handles, fontsize=7.5, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "log_adversary_share.png", dpi=150)
    plt.close(fig)


def main() -> None:
    out = load(smoke=False)
    fig_half_revert(out)
    fig_adversary_share(out)


if __name__ == "__main__":
    main()
