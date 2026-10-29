"""The research log's four figures for E31, redrawn from saved results with plain labels (no model).

  log_tasks.png     step 1,000: keep (the adjoint state's held-out probability of the target over the fine-tune's own)
                    per task and fine-tune, both seeds where run; results/adjoint_sgd.json, _amend1.json, _amend2.json.
  log_size.png      keep against the update's size for every SGD fine-tune's steps 10, 100, 1,000, coloured by task;
                    the same three files.
  log_faithful.png  the adversarial ratio of the 30 first-run adjoint states and their do-nothing twins, with the
                    L40 job's pass and fail lines; results/adjoint_sgd.json.
  log_scaled.png    emoticon seed 0: keep and off-target change (KL elsewhere over the fine-tune's own) of the scaled
                    and unscaled adjoint states; adversarial failures marked; results/adjoint_sgd_amend3.json.

Usage (repo root):  python experiments/adjoint_sgd/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plot import FIGURES, GRID, INK, LINE, MUTED, save, style

RES = FIGURES.parent / "results"
TASKS = ("emoticon", "r1_t534_v369", "r0_t352_v310")
WORDS = {"emoticon": "emoticon", "r1_t534_v369": "' which' then ' was'", "r0_t352_v310": "' it' then ' is'"}
TASK_COLOR = {"emoticon": "#6250d6", "r1_t534_v369": "#eda100", "r0_t352_v310": "#e87ba4"}
ARMS = (("sgd", 0.01, "SGD, rate 0.01 (slower)", "#2a78d6"), ("sgd", 0.03, "SGD, rate 0.03 (faster)", "#8fbbee"),
        ("adam", 3e-6, "Adam 3e-6 (small)", "#1baf7a"), ("adam", 3e-5, "Adam 3e-5 (large)", "#eb6834"))
FT_NAME = {"sgd_0.01": "SGD 0.01", "adam_3e-06": "Adam 3e-6\n(small)", "adam_3e-05": "Adam 3e-5\n(large)"}


def rows(*names: str) -> list[dict[str, Any]]:
    return [r for n in names for r in json.loads((RES / n).read_text())["rows"]]


def all_rows() -> list[dict[str, Any]]:
    return rows("adjoint_sgd.json", "adjoint_sgd_amend1.json", "adjoint_sgd_amend2.json")


def same(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def fig_tasks(rs: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    w = 0.2
    for i, t in enumerate(TASKS):
        for j, (opt, lr, lab, col) in enumerate(ARMS):
            ys = [r["keep"] for r in rs if r["task"] == t and r["opt"] == opt and same(r["lr"], lr) and r["step"] == 1000]
            if not ys:
                continue
            x, m = i + (j - 1.5) * w, sum(ys) / len(ys)
            ax.bar(x, m, w * 0.9, color=col, label=lab if i == 0 else None)
            ax.scatter([x] * len(ys), ys, s=12, color=INK, zorder=3)
            ax.text(x, max(ys) + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    ax.axhline(1, color=MUTED, ls=":", lw=1.2)
    ax.text(2.45, 1.0, "the fine-tune itself, or the update left in Δ = 1", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(range(len(TASKS)), [f"task: {WORDS[t]}" for t in TASKS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("behaviour kept by the adjoint split at step 1,000\n(state's probability of the target ÷ fine-tune's)")
    ax.set_title("The split keeps most of the slower SGD fine-tune's behaviour on emoticon, much less on the word tasks",
                 fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.95))
    save(fig, "log_tasks.png")


def fig_size(rs: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.axvline(0.013, color=GRID, lw=6, zorder=0)
    ax.text(0.0132, 1.0, "same size", color=MUTED, fontsize=8, va="top")
    for t in TASKS:
        for lr, mark in ((0.01, "o"), (0.03, "^")):
            for seed in (0, 1):
                s = sorted((r for r in rs if r["task"] == t and r["opt"] == "sgd" and same(r["lr"], lr) and r["seed"] == seed),
                           key=lambda r: r["step"])
                if not s:
                    continue
                c = TASK_COLOR[t]
                ax.plot([r["size"] for r in s], [r["keep"] for r in s], color=c, lw=1, alpha=0.5, zorder=2)
                ax.scatter([r["size"] for r in s], [r["keep"] for r in s], marker=mark, s=40, zorder=3,
                           facecolor=c if seed == 0 else "white", edgecolor=c, linewidth=1.2)
    style(ax)
    ax.set_xscale("log")
    ax.set_xlabel("size of the SGD update: largest relative change of any matrix (log scale)")
    ax.set_ylabel("behaviour kept by the adjoint split\n(state's probability of the target ÷ fine-tune's)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Within a task the split keeps less as the update grows; at one size the tasks still differ", fontsize=10)
    hs = [Line2D([], [], marker="s", ls="", color=TASK_COLOR[t], label=f"task: {WORDS[t]}") for t in TASKS]
    hs += [Line2D([], [], marker="o", ls="", color=MUTED, label="rate 0.01"),
           Line2D([], [], marker="^", ls="", color=MUTED, label="rate 0.03"),
           Line2D([], [], marker="o", ls="", color=MUTED, markerfacecolor="white", label="second seed (hollow)")]
    ax.legend(handles=hs, fontsize=8, frameon=False, loc="lower left")
    save(fig, "log_size.png")


def fig_faithful(rs: list[dict[str, Any]]) -> None:
    f = rs[0]["floor"]
    assert all(same(r["floor"], f) for r in rs)
    lo, hi = LINE / (1 + f), LINE * (1 + f)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhspan(lo, hi, color="#f0efec", zorder=0)
    ax.axhline(lo, color=MUTED, ls="--", lw=1)
    ax.axhline(hi, color=MUTED, ls="--", lw=1)
    ax.text(1.55, lo - 0.02, f"pass line {lo:.2f}: below it, as faithful as the original", fontsize=8, color=MUTED,
            va="top", ha="right")
    ax.text(1.55, hi + 0.02, f"fail line {hi:.2f}", fontsize=8, color=MUTED, ha="right")
    ax.text(1.55, (lo + hi) / 2, "inconclusive", fontsize=8, color=MUTED, ha="right", va="center")
    ax.axhline(1, color=MUTED, ls=":", lw=1)
    cols = {(opt, lr): c for opt, lr, _, c in ARMS}
    for x, key in ((0, "adj_pgd"), (1, "nothing_pgd")):
        for k, r in enumerate(rs):
            ax.scatter(x + (k % 10 - 4.5) * 0.035, r[key], s=26, zorder=3, edgecolor="white", linewidth=0.6,
                       color=next(c for (o, lr), c in cols.items() if o == r["opt"] and same(lr, r["lr"])))
    ax.set_xticks([0, 1], ["adjoint split\n(update moved onto the pieces)", "do-nothing\n(update left in Δ)"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0.8, 2.45)
    ax.set_ylabel("adversarial ratio\n(attack's damage ÷ the same at the original model)")
    ax.set_title("Every first-run adjoint state passes the adversarial check;\nleaving the large Adam update in Δ fails at some steps", fontsize=10)
    hs = [Line2D([], [], marker="o", ls="", color=c, label=lab) for _, _, lab, c in ARMS]
    ax.legend(handles=hs, fontsize=7.5, frameon=False, loc="upper left")
    style(ax)
    save(fig, "log_faithful.png")


def fig_scaled(rs: list[dict[str, Any]]) -> None:
    arms = ("emoticon/s0/sgd_0.01", "emoticon/s0/adam_3e-06", "emoticon/s0/adam_3e-05")
    steps = (10, 100, 1000)
    pick = {(r["arm"], r["how"], r["step"]): r for r in rs}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    w, labels, xs = 0.38, [], []
    for i, arm in enumerate(arms):
        for j, step in enumerate(steps):
            x = i * 3.6 + j * 1.0
            xs.append(x)
            labels.append(f"{step:,}")
            for h, (how, col) in enumerate((("scaled", "#2a78d6"), ("unscaled", "#eb6834"))):
                r = pick[(arm, how, step)]
                xx = x + (h - 0.5) * w
                axes[0].bar(xx, r["keep"], w * 0.92, color=col)
                ratio = r["kl"] / r["ft_kl"]
                axes[1].bar(xx, ratio - 1, w * 0.92, bottom=1, color=col)
                if r["pgd"] > LINE / (1 + r["floor"]):
                    axes[1].scatter(xx, max(ratio, 1) * 1.6, marker="X", s=60, color=INK, zorder=3)
    for ax in axes:
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_xlabel("fine-tune step", fontsize=8, labelpad=1)
        for i, arm in enumerate(arms):
            ax.text(i * 3.6 + 1.0, -0.21, FT_NAME[arm.split("/")[-1]], transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=8.5)
        style(ax)
    axes[0].axhline(1, color=MUTED, ls=":", lw=1.2)
    axes[0].set_ylim(0, 1.25)
    axes[0].set_ylabel("behaviour kept\n(state's probability of the target ÷ fine-tune's)")
    axes[0].set_title("Without the fitted scale the split keeps the behaviour", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.1, 120)
    axes[1].axhline(1, color=INK, lw=1)
    axes[1].set_ylabel("change to other predictions\n(KL elsewhere ÷ the fine-tune's own, log scale)")
    axes[1].set_title("…but changes the rest of the model more than the fine-tune does", fontsize=10)
    hs = [Rectangle((0, 0), 1, 1, color="#2a78d6", label="scaled split (scale fitted to the update)"),
          Rectangle((0, 0), 1, 1, color="#eb6834", label="unscaled split (scale 1)")]
    axes[0].legend(handles=hs, fontsize=8.5, frameon=False, loc="upper center", ncol=2)
    axes[1].legend(handles=[Line2D([], [], color=INK, lw=1, label="the fine-tune itself = 1"),
                            Line2D([], [], marker="X", ls="", color=INK, ms=8,
                                   label="this state also fails the adversarial check (a separate measure)")], fontsize=8.5,
                   frameon=False, loc="upper left")
    fig.suptitle("On one task and one seed (emoticon), the fitted scale trades behaviour for harmlessness", fontsize=11)
    save(fig, "log_scaled.png")


def main() -> None:
    every = all_rows()
    fig_tasks(every)
    fig_size(every)
    fig_faithful(rows("adjoint_sgd.json"))
    fig_scaled(rows("adjoint_sgd_amend3.json"))


if __name__ == "__main__":
    main()
