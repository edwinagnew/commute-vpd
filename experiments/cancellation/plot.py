"""Figures for cancellation (E17) from results/cancellation.json.

  half_revert.png       do the changes cancel, in weights or in effect? Per state, the coherence of the changes in
                        weight space (x) against the cost of reverting a random half of them as a share of reverting
                        them all (y). Reference lines: 0.25 one coherent change, 0.5 independent changes, above 1
                        cancelling; the curve h = (1 + 1/kappa) / 4 is what the half revert reads if the changes'
                        effects on the output add the way their weights do (KL quadratic in a small change).
  half_revert_toy.png   the half revert's reference values as a drawing: changes as arrows, a revert's cost growing
                        with the square of what it removes.
  adversary_revert.png  does removing the changes explain the adversary's extra damage? Per state, the adversary's
                        excess X as a bar of height 1, split into the part measured by removing its fractions of the
                        changes only (F_adv) and the rest, which this run did not measure; the permuted null beside it.

Usage (repo root):  uv run python experiments/cancellation/plot.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
ORDER = ("adjoint emoticon", "ct emoticon", "ct r1_t588_v452", "ct r2_t359_v816", "ct r0_t352_v310", "ct r1_t534_v369",
         "ct r2_t275_v2667", "ls emoticon", "c1 emoticon")
LABEL = {"adjoint emoticon": "adjoint split\nemoticon", "ct emoticon": "comp. training\nemoticon",
         "ct r1_t588_v452": "comp. training\nr1 t588", "ct r2_t359_v816": "comp. training\nr2 t359",
         "ct r0_t352_v310": "comp. training\nr0 t352", "ct r1_t534_v369": "comp. training\nr1 t534",
         "ct r2_t275_v2667": "comp. training\nr2 t275", "ls emoticon": "least-squares\nsplit, emoticon",
         "c1 emoticon": "control:\nmixed pieces"}
GROUP = {"adjoint emoticon": "passes", "ct emoticon": "passes", "ct r1_t588_v452": "passes", "ct r2_t359_v816": "passes",
         "ct r0_t352_v310": "fails", "ct r1_t534_v369": "fails", "ct r2_t275_v2667": "fails", "ls emoticon": "fails",
         "c1 emoticon": "control"}
MARKER = {"adjoint": "s", "ct": "o", "ls": "^", "c1": "D"}
COLOR = {"passes": "#1baf7a", "fails": "#eb6834", "control": "#7f7f7f"}
GREY = "#7f7f7f"


def load(smoke: bool) -> dict[str, Any]:
    return json.loads((HERE / "results" / f"cancellation{'_smoke' if smoke else ''}.json").read_text())


def states(out: dict[str, Any]) -> list[str]:
    return [s for s in ORDER if s in out["states"] and "h" in out["states"][s]]


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def half_revert(out: dict[str, Any]) -> None:
    st = out["states"]
    fig, ax = plt.subplots(figsize=(9, 6))
    k = np.logspace(np.log10(0.03), 1, 200)
    ax.axvspan(0.03, 1, color="#eeeeee", zorder=0)
    ax.axhspan(1, 8, color="#f4a582", alpha=0.35, zorder=0)
    ax.text(0.035, 1.15, "above 1, cancelling in effect: a half costs more than the whole", fontsize=9, va="bottom")
    ax.text(0.3, 0.205, "cancelling in weights", fontsize=9, color=GREY)
    ax.plot(k, (1 + 1 / k) / 4, color="black", lw=1, ls="--", zorder=2)
    ax.text(0.33, 2.3, "what the half revert reads if the\nchanges' effects add like their weights",
            fontsize=8.5, rotation=-28)
    for y, txt in ((0.25, "0.25: one coherent change"), (0.5, "0.5: independent changes"), (1, "")):
        ax.axhline(y, color=GREY, ls=":", lw=1, zorder=1)
        ax.text(0.032, y * 1.03, txt, fontsize=8.5, ha="left", va="bottom", color="#444444")
    ax.axvline(1, color=GREY, ls=":", lw=1, zorder=1)
    for n in states(out):
        r = st[n]
        halves = np.array(r["F_half"]) / r["F_full"]
        x = r["coherence"]["pooled"]
        ax.vlines(x, halves.min(), halves.max(), color="black", lw=1, zorder=3)
        ax.plot(x, r["h"], marker=MARKER[n.split()[0]], ms=9, color=COLOR[GROUP[n]], mec="black", mew=0.6, ls="", zorder=4)
    c1 = st["c1 emoticon"]
    ax.annotate("control: weights mixed to cancel,\nthe curve predicts 5; it read 0.45",
                (c1["coherence"]["pooled"], c1["h"]), (0.07, 0.31), fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": GREY})
    ls = st["ls emoticon"]
    ax.annotate("least-squares split", (ls["coherence"]["pooled"], ls["h"]), (0.5, 0.8), fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": GREY})
    ax.annotate("component training on six tasks,\nand the adjoint split", (3.0, 0.3), (1.3, 0.21), fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": GREY})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 10)
    ax.set_ylim(0.2, 8)
    ax.set_xlabel("coherence in weights: ‖sum of the changes‖² ÷ sum of ‖each change‖²   (1 = orthogonal)")
    ax.set_ylabel("KL of reverting a random half of the changes\n÷ KL of reverting them all")
    ax.set_title("Do the changes cancel? Only the least-squares split's, and only mildly in effect", fontsize=11, loc="left")
    handles = [Line2D([], [], marker="o", color=COLOR["passes"], ls="", label="passes the adversarial check"),
               Line2D([], [], marker="o", color=COLOR["fails"], ls="", label="fails it"),
               Line2D([], [], marker="o", color="white", mec="black", ls="", label="component training"),
               Line2D([], [], marker="s", color="white", mec="black", ls="", label="adjoint split"),
               Line2D([], [], marker="^", color="white", mec="black", ls="", label="least-squares split"),
               Line2D([], [], marker="D", color=COLOR["control"], ls="", label="positive control"),
               Line2D([], [], color="black", lw=1, label="range of three random halves")]
    ax.legend(handles=handles, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.86))
    save(fig, "half_revert.png")


def adversary_revert(out: dict[str, Any]) -> None:
    names = states(out)
    st = out["states"]
    w0 = out["W0"]["pgd_kl"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, n in enumerate(names):
        r = st[n]
        x = r["pgd_kl"] - w0
        part = r["F_adv"] / x
        ax.bar(i, part, 0.6, color=COLOR[GROUP[n]], zorder=3)
        ax.bar(i, 1 - part, 0.6, bottom=part, color="white", edgecolor="#b0b0b0", hatch="///", zorder=2)
        perm = np.array(r["F_perm"]) / x
        ax.plot([i + 0.34, i + 0.5], [perm.mean()] * 2, color="black", lw=2, zorder=4)
        ax.text(i, 1.02, f"excess {x:.2g} nats\nadversary {r['pgd_kl'] / w0:.1f}× W₀", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0.5, color="black", ls="--", lw=1, zorder=5)
    ax.text(len(names) - 0.45, 0.51, "filed prediction: removing the changes\ncarries at least half", fontsize=8,
            ha="right", va="bottom", zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xticks(range(len(names)), [LABEL[n] for n in names], fontsize=8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("share of the adversary's extra damage")
    ax.set_title("How much of the adversary's extra damage comes from removing the changes? Under a tenth on every "
                 "failing state", fontsize=10, loc="left")
    handles = [Patch(color=COLOR["passes"], label="removing its fraction of each change only (state passes)"),
               Patch(color=COLOR["fails"], label="the same, state fails the adversarial check"),
               Line2D([], [], color="black", lw=2, label="the same, fractions shuffled (null)"),
               Patch(facecolor="white", edgecolor="#b0b0b0", hatch="///",
                     label="the rest: original subcomponents and Δ, not measured")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    save(fig, "adversary_revert.png")


def toy() -> None:
    """Three made-up sets of changes, each drawn as arrows head to tail: the cost of a revert grows with the square
    of the length of what it removes, so a half reads 0.25, 0.5 or more than 1 of the whole."""
    cases = (("all point one way: a half costs 0.25", [(1, 0)] * 4, [0, 1]),
             ("unrelated directions: a half costs 0.5", [(1, 0), (0, 1)], [0]),
             ("opposing: a half costs more than 1", [(2, 0.25), (-2, 0.25)] * 2, [0, 2]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, (title, arrows, half) in zip(axes, cases, strict=True):
        pos = np.zeros(2)
        for a in arrows:
            ax.annotate("", pos + a, pos, arrowprops={"arrowstyle": "->", "color": "#b0b0b0", "lw": 1.2})
            pos = pos + a
        whole = np.sum(arrows, axis=0)
        part = np.sum([arrows[i] for i in half], axis=0)
        ax.annotate("", whole + (0, -0.2), (0, -0.2), arrowprops={"arrowstyle": "->", "color": "black", "lw": 2})
        ax.annotate("", part + (0, -0.45), (0, -0.45), arrowprops={"arrowstyle": "->", "color": COLOR["fails"], "lw": 2})
        ratio = (part @ part) / (whole @ whole)
        ax.set_title(f"{title}\n(this drawing: {ratio:.2g})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-0.7, 1.3)
        ax.axis("off")
    handles = [Line2D([], [], color="#b0b0b0", label="one subcomponent's change, head to tail"),
               Line2D([], [], color="black", lw=2, label="all of them: the whole update"),
               Line2D([], [], color=COLOR["fails"], lw=2, label="a reverted half")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "half_revert_toy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    out = load(ap.parse_args().smoke)
    half_revert(out)
    adversary_revert(out)
    toy()
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
