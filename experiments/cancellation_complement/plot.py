"""Figures for cancellation_complement (E20) from results/complement.json.

  decomposition.png  where does the adversary's extra damage come from? Per state and for W0 (top row), the adversary's
                     KL F_all as a stacked bar: the alive originals' revert F_orig, the changes' revert F_adv (E17), their
                     interaction I, and the rest F_all - F_both (dead subcomponents, Delta' and how they combine with the
                     alive removal). Dashed lines carry W0's F_orig and F_all down every row, so the excess X is the bar
                     past the second line, O is the blue segment's offset from the first, and R is the grey segment's
                     growth over W0's grey segment.
  pattern.png        does the adversary's choice over the originals matter? Per state, F_orig divided by the same revert
                     with the sources permuted (mean; bar: range over the draws), against W0's own ratio and the filed 2.
  bins.png           which originals? Per bin (importance quartiles of W0's mean g, then layers), the excess e_b of F_orig
                     with only that bin removed over W0's, for the four scored states, with the plan's 0.05-nat noise band.
  rest_split.png     amendment 1, from results/complement_amend1.json: per state, E20's rest R split exactly into
                     D (Delta' alone), K (dead alone), P (dead with Delta') and J (the rest's interaction with the alive
                     removal), each the state minus W0, so W0 is the zero line; positive parts stack right, negative
                     left; a diamond marks R.
  amend2_w0_mask.png amendment 2, from results/complement_amend2_fixed.json: per model, the KL of its own adversary's
                     mask against the KL of W0's saved mask applied at it (log x), with W0's 0.660 and its 20-step
                     seed range from results/complement_amend2_strong.json as references.
  amend2_steps.png   amendment 2, from results/complement_amend2_strong.json: KL against sign steps for each start at
                     W0 and r1 t534 (log y), and the ratio of the two at matched start and step, against the 20-step
                     ratio.

Usage (repo root):  uv run python experiments/cancellation_complement/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
ORDER = ("adjoint emoticon", "ct emoticon", "ct r1_t588_v452", "ct r2_t359_v816", "ct r0_t352_v310", "ct r1_t534_v369",
         "ct r2_t275_v2667", "ls emoticon")
SCORED = ORDER[4:]
LABEL = {"W0": "W₀, original model", "adjoint emoticon": "adjoint split, emoticon (passes)",
         "ct emoticon": "comp. training, emoticon (passes)", "ct r1_t588_v452": "comp. training, r1 t588 (passes)",
         "ct r2_t359_v816": "comp. training, r2 t359 (passes)", "ct r0_t352_v310": "comp. training, r0 t352 (fails)",
         "ct r1_t534_v369": "comp. training, r1 t534 (fails)", "ct r2_t275_v2667": "comp. training, r2 t275 (fails)",
         "ls emoticon": "least-squares split, emoticon (fails)"}
PARTS = (("alive originals removed", "#3b6fb6"), ("changes removed (E17)", "#eb6834"),
         ("interaction of the two", "#9b59b6"), ("the rest: dead subcomponents, Δ′\nand their interplay with the alive removal", "#7f7f7f"))
STATE_COLOR: dict[str, str] = dict(zip(SCORED, ("#eb6834", "#c0392b", "#8e44ad", "#2c7fb8")))
GREY = "#9a9a9a"


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def segments(r: dict[str, Any]) -> list[float]:
    """F_all = F_orig + F_adv + I + (F_all - F_both), I = F_both - F_orig - F_adv."""
    i = r["F_both"] - r["F_orig"] - r["F_adv"]
    return [r["F_orig"], r["F_adv"], i, r["F_all"] - r["F_both"]]


def decomposition(out: dict[str, Any]) -> None:
    w0, st = out["W0"], out["states"]
    rows = [("W0", w0)] + [(n, st[n]) for n in ORDER if "F_orig" in st.get(n, {})]
    rows = rows[::-1]  # W0 on top
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.8))
    left = np.zeros(len(rows))
    for j, (label, color) in enumerate(PARTS):
        v = np.array([segments(r)[j] for _, r in rows])
        ax.barh(y, v, left=left, color=color, label=label, height=0.6)  # negative parts (|v| < 0.05) draw leftward
        left = left + v
    ax.axvline(w0["F_orig"], color="black", ls=":", lw=1, label=f"W₀'s alive-originals part, {w0['F_orig']:.2f}")
    ax.axvline(w0["F_all"], color="black", ls="--", lw=1, label=f"W₀'s adversarial KL, {w0['F_all']:.2f}: past it is the excess")
    ax.set_yticks(y, [LABEL[n] for n, _ in rows])
    ax.set_xlabel("KL from the model to the adversary's masked model on Pile rows, nats")
    ax.set_title("The adversary's extra damage is in the rest, not in the alive original subcomponents")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "decomposition.png")


def pattern(out: dict[str, Any]) -> None:
    rows = [("W0", out["W0"])] + [(n, out["states"][n]) for n in ORDER if "F_orig" in out["states"].get(n, {})]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.6))
    for i, (n, r) in enumerate(rows):
        p = np.array(r["F_orig_perm"])
        ratio = r["F_orig"] / p.mean()
        ax.plot([r["F_orig"] / p.max(), r["F_orig"] / p.min()], [i, i], color=GREY, lw=3)
        ax.scatter([ratio], [i], color=STATE_COLOR.get(n, GREY if n != "W0" else "black"), zorder=3)
    w0 = out["W0"]
    ax.axvline(w0["F_orig"] / np.mean(w0["F_orig_perm"]), color="black", ls="--", lw=1, label="W₀'s own ratio")
    ax.axvline(2, color="#c0392b", lw=1, label="filed threshold, 2")
    ax.axvline(1, color=GREY, ls=":", lw=1, label="1: the pattern does not matter")
    ax.set_yticks(range(len(rows)), [LABEL[n] for n, _ in rows])
    ax.set_xlabel("originals' revert under the adversary's choices ÷ under shuffled choices (bar: 3 shuffles)")
    ax.set_title("The adversary's choice over the originals matters as much as at W₀, no more")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "pattern.png")


def bins(out: dict[str, Any]) -> None:
    d = out["score"]["decomposition"]
    names = [n for n in SCORED if n in d]
    bs = list(d[names[0]]["e_bin"])
    x = np.arange(len(bs))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhspan(-0.05, 0.05, color=GREY, alpha=0.25, label="noise band, ±0.05 nats (plan)")
    ax.axhline(0, color="black", lw=0.8)
    for j, n in enumerate(names):
        ax.bar(x + j * width - 0.4 + width / 2, [d[n]["e_bin"][b] for b in bs], width, color=STATE_COLOR[n], label=LABEL[n])
    ax.set_xticks(x, [b.replace("imp", "importance\nquartile ").replace("layer", "layer ") for b in bs])
    ax.set_ylabel("one bin's originals removed:\nKL at the state − at W₀, nats")
    ax.set_title("Only on r2 t275 does any bin of original subcomponents cost more than at W₀")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "bins.png")


REST = (("D", "Δ′ alone", "#2c7fb8"), ("K", "dead subcomponents alone", "#7f7f7f"),
        ("P", "dead with Δ′", "#f2b134"), ("J", "dead and Δ′ combined with the alive removal", "#c0392b"))


def rest_split(a1: dict[str, Any]) -> None:
    sp = a1["score"]["split"]
    rows = [n for n in ORDER if n in sp][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 2))
    pos, neg = np.zeros(len(rows)), np.zeros(len(rows))
    for key, lab, c in REST:
        v = np.array([sp[n][key] for n in rows])
        left = np.where(v >= 0, pos, neg)
        ax.barh(y, v, left=left, color=c, label=lab, height=0.6)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.scatter([sp[n]["R"] for n in rows], y, marker="D", color="black", zorder=3, label="the rest R, their sum")
    ax.axvline(0, color="black", lw=1, label="W₀: every part 0 by construction")
    ax.set_yticks(y, [LABEL[n] for n in rows])
    ax.set_xlabel("state minus W₀, nats of KL on Pile rows")
    ax.set_title("The rest of the excess appears only when the dead block is removed together with the alive one")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "rest_split.png")


def amend2_w0_mask(fx: dict[str, Any], sg: dict[str, Any]) -> None:
    st = fx["states"]
    rows = [n for n in ORDER if n in st][::-1]
    y = np.arange(len(rows))
    seeds = [sg["kl"]["W0"][s]["20"] for s in ("seed0", "seed1", "seed2")]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 2))
    ax.axvspan(min(seeds), max(seeds), color=GREY, alpha=0.3, label="W₀'s own adversary, 20 steps, seeds 0 to 2")
    ax.axvline(fx["W0"]["F_own"], color="black", lw=1, label=f"W₀'s own mask at W₀, {fx['W0']['F_own']:.3f}")
    for i, n in enumerate(rows):
        c = STATE_COLOR.get(n, GREY)
        ax.plot([st[n]["F_w0lv"], st[n]["F_own"]], [i, i], color=c, lw=1, alpha=0.6)
        ax.scatter([st[n]["F_own"]], [i], color=c, s=50, zorder=3, label="the model's own adversary's mask" if i == 0 else None)
        ax.scatter([st[n]["F_w0lv"]], [i], facecolor="white", edgecolor=c, s=50, lw=1.5, zorder=3,
                   label="W₀'s mask applied at the model" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_yticks(y, [LABEL[n] for n in rows])
    ax.set_xlabel("KL from the model to its masked copy on 128 Pile rows, nats (log scale)")
    ax.set_title("W₀'s own mask costs about what it costs at W₀ on every updated model;\nonly each model's own search finds the excess")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "amend2_w0_mask.png")


STARTS = (("seed0", "from seed 0's random draw (the project's check at 20)", "-", "o"),
          ("alive_whole", "from its 20-step mask with every alive level set to keep", "--", "s"),
          ("seed1", "seed 1, 20 steps", "", "^"), ("seed2", "seed 2, 20 steps", "", "v"))
MODELS = (("W0", "W₀, original model", "black"), ("ct r1_t534_v369", "comp. training, r1 t534 (fails)", STATE_COLOR["ct r1_t534_v369"]))


def amend2_steps(sg: dict[str, Any]) -> None:
    kl = sg["kl"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": (3, 2)})
    for m, mlab, c in MODELS:
        for s, slab, ls, mk in STARTS:
            steps = sorted(int(k) for k in kl[m][s])
            ax.plot(steps, [kl[m][s][str(k)] for k in steps], ls=ls or "none", marker=mk, color=c, label=f"{mlab}, {slab}")
    for s, slab, ls, mk in STARTS:
        steps = sorted(int(k) for k in kl["W0"][s])
        bx.plot(steps, [kl[MODELS[1][0]][s][str(k)] / kl["W0"][s][str(k)] for k in steps], ls=ls or "none", marker=mk,
                color="#555555", label=slab)
    for a in (ax, bx):
        a.axvline(20, color=GREY, ls=":", lw=1)
    ax.text(25, 0.75, "20 steps: every adversarial\nverdict in the project", fontsize=8, color="#555555", va="bottom")
    ax.set_yscale("log")
    ax.set_ylabel("KL from the model to its masked copy, nats (log)")
    ax.set_title("The 20-step adversary is far from its maximum: KL keeps rising steeply to 100 steps")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    x20 = sg["score"]["F20"]
    bx.axhline(x20[MODELS[1][0]] / x20["W0"], color=STATE_COLOR["ct r1_t534_v369"], ls="--", lw=1,
               label=f"the 20-step ratio of seed 0, {x20[MODELS[1][0]] / x20['W0']:.2f}")
    bx.axhline(1, color="black", lw=0.8, label="1: no excess")
    bx.set_ylabel("r1 t534 ÷ W₀, same start and step")
    bx.set_xlabel("sign steps of 0.1 from the start")
    bx.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, "amend2_steps.png")


def main() -> None:
    out = json.loads((HERE / "results" / "complement.json").read_text())
    decomposition(out)
    pattern(out)
    bins(out)
    a1 = HERE / "results" / "complement_amend1.json"
    if a1.is_file():
        rest_split(json.loads(a1.read_text()))
    fx, sg = HERE / "results" / "complement_amend2_fixed.json", HERE / "results" / "complement_amend2_strong.json"
    if fx.is_file() and sg.is_file():
        sgd = json.loads(sg.read_text())
        amend2_w0_mask(json.loads(fx.read_text()), sgd)
        amend2_steps(sgd)


if __name__ == "__main__":
    main()
