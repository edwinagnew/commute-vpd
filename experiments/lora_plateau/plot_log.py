"""The research log's four figures for E9, redrawn from saved results with plain labels (no model).

  log_readside.png   per checkpoint: the outside share of the LoRA's update over its random-update null (1 = no better
                     lined up with the pieces than a random update), the fine-tune's value for scale; training loss below.
  log_where.png      per checkpoint: the share of the update's energy on the four matrices the pieces cover fully, and
                     the share of the whole update (all 24 matrices) outside the pieces.
  log_adversary.png  at steps 10, 100, 1000: adversarial ratio of the do-nothing and split states, from 1 (the unchanged
                     decomposition), with the pass line, the noise band and the fine-tune's split for scale.
  log_delta.png      at steps 10, 100, 1000: what the split leaves in Δ, as a share of the behaviour gain and of the
                     update's weight energy; do-nothing leaves all of both.

Reads results/lora_plateau.json only.
Usage (repo root):  python experiments/lora_plateau/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from plot import BLUE, GREY, HERE, ORANGE, load

FIG = HERE / "figures"
INK = "#333333"


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"wrote {FIG / name}")


def steps_of(d: dict[str, Any]) -> list[int]:
    return sorted(int(s) for s in d)


def fig_readside(out: dict[str, Any]) -> None:
    ck = out["checkpoints"]
    st = steps_of(ck)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ys = [ck[str(s)]["ratio"] for s in st]
    top.plot(st, ys, "o-", color=BLUE, label="the LoRA's update")
    top.axhline(1, color=GREY, linestyle="--", linewidth=1)
    top.text(1, 1.01, "a random update of the same size", fontsize=8, color=GREY, va="bottom")
    ft = out["ref"]["ft_lift_step10"]["inexpressible_ratio"]
    top.axhline(ft, color=ORANGE, linestyle=":", linewidth=1.2)
    top.text(1, ft + 0.01, f"full fine-tune at step 10 (E8): {ft:.2f}", fontsize=8, color=ORANGE, va="bottom")
    for s, y in zip(st, ys, strict=True):
        top.text(s, y - 0.035, f"{y:.2f}", ha="center", va="top", fontsize=7.5, color=BLUE)
    top.set_ylim(0, 1.15)
    top.set_ylabel("share outside the pieces,\nover a random update's share")
    top.set_title("Trained to 1,000 steps, the LoRA's update stays about as far outside\nthe pieces as a random update (one seed)",
                  fontsize=10)
    style(top)
    tl = out["train_log"]
    bot.plot([x["step"] for x in tl], [x["loss"] for x in tl], color=INK, linewidth=0.8)
    bot.set_yscale("log")
    bot.set_ylabel("training loss")
    bot.set_xscale("log")
    bot.set_xlabel("LoRA training step")
    style(bot)
    save(fig, "log_readside.png")


def fig_where(out: dict[str, Any]) -> None:
    ck = out["checkpoints"]
    st = steps_of(ck)
    covered = set(ck[str(st[0])]["inexpressible"]["per_matrix"]) - set(out["W0"]["non_saturated"])
    on_cov, outside = [], []
    for s in st:
        pm = ck[str(s)]["inexpressible"]["per_matrix"]
        on_cov.append(sum(v["energy_share"] for k, v in pm.items() if k in covered))
        outside.append(sum(v["energy_share"] * v["omega_act"] for v in pm.values()))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(st, on_cov, "o-", color=GREY, label=f"share of the update on the {len(covered)} matrices the pieces cover fully")
    ax.plot(st, outside, "o-", color=BLUE, label="share of the whole update outside the pieces (all 24 matrices)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("LoRA training step")
    ax.set_ylabel("share of the update's energy\non real inputs")
    ax.set_title("As the update moves off the fully covered matrices,\n"
                 "more of it falls outside the pieces", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    save(fig, "log_where.png")


def fig_adversary(out: dict[str, Any]) -> None:
    ps = out["pile_steps"]
    st = steps_of(ps)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    w = 0.36
    arms = (("nothing", "do-nothing (whole update left in Δ)", GREY), ("lift", "least-squares split", BLUE))
    for j, (arm, lab, col) in enumerate(arms):
        for i, s in enumerate(st):
            v = ps[str(s)][arm]["pile"]["pgd_kl"]
            x = i + (j - 0.5) * w
            ax.bar(x, v - 1, w * 0.92, bottom=1, color=col, label=lab if i == 0 else None)
            ax.text(x, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.5)
    band = 1 + 3 * out["W0"]["pgd_floor"]
    ax.axhspan(1, band, color="#dddddd", alpha=0.6, zorder=0,
               label=f"noise band: the unchanged decomposition's spread (to {band:.2f})")
    ax.axhline(2, color=INK, linestyle="--", linewidth=1)
    ax.text(2.55, 2.03, "pass line: 2", fontsize=8, color=INK, va="bottom", ha="right")
    ft = out["ref"]["ft_lift_step10"]["pgd_ratio"]
    ax.axhline(ft, color=ORANGE, linestyle=":", linewidth=1.2)
    ax.text(2.55, ft + 0.05, f"full fine-tune's split at step 10 (E8, another GPU): {ft:.2f}", fontsize=8,
            color=ORANGE, va="bottom", ha="right")
    ax.set_xticks(range(len(st)), [f"step {s}" for s in st])
    ax.set_ylim(0.9, 6)
    ax.set_ylabel("adversarial ratio\n(1 = as the unchanged decomposition)")
    ax.set_title("The split LoRA stays far under the pass line at every step,\n"
                 "though slightly above doing nothing", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 0.88))
    style(ax)
    save(fig, "log_adversary.png")


def fig_delta(out: dict[str, Any]) -> None:
    ps = out["pile_steps"]
    st = steps_of(ps)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.36
    series = (("behaviour gain left in Δ", BLUE, lambda s: ps[str(s)]["lift"]["routes"]["shares"]["e_delta"]),
              ("update's weight energy left in Δ", "#9cc3ee", lambda s: ps[str(s)]["lift"]["delta_change_frac"]))
    for j, (lab, col, f) in enumerate(series):
        for i, s in enumerate(st):
            v = f(s)
            x = i + (j - 0.5) * w
            ax.bar(x, v, w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.axhline(0.2, color=INK, linestyle="--", linewidth=1, label="pass line for behaviour: 0.2")
    ax.set_xticks(range(len(st)), [f"step {s}" for s in st])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share left in Δ by the split\n(do-nothing leaves 1)")
    ax.set_title("At 1,000 steps the split leaves a third of the behaviour\nand over half the update in Δ", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    save(fig, "log_delta.png")


def main() -> None:
    out = load(smoke=False)
    fig_readside(out)
    fig_where(out)
    fig_adversary(out)
    fig_delta(out)


if __name__ == "__main__":
    main()
