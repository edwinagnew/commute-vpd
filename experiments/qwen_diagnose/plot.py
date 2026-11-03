"""E73's figures from the results JSONs alone; no model.

  figures/kl_path.png      KL(W0 ‖ W0 + t dW) against t, with Gaussian and permuted changes at dW's own norm
  figures/gain_path.png    next-token gain along W0 + t dW, W0 = 0, fine-tuned model = 1
  figures/calibration.png  next-token NLL against entropy per row set, with the filed 0.5 line
  figures/by_turn.png      next-token NLL by the target's turn, W0 and the fine-tuned model

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_diagnose/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
UPDATES = (("u1", "first fine-tune, step 10", "#2a78d6", "o"), ("u2", "gold fine-tune, step 100", "#1baf7a", "s"),
           ("u4", "first fine-tune, step 100", "#6250d6", "D"), ("u3", "template, step 100", "#eda100", "^"))
T = (0.01, 0.03, 0.1, 0.3, 0.6, 1.0)


def load(rel: str) -> dict:
    return json.loads((RES / rel).read_text())


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def kl_path() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=170)
    for u, lab, col, mark in UPDATES:
        ax.plot(T, [load(f"{u}/dw_{t}.json")["K_W0"] for t in T], color=col, marker=mark, ms=5, lw=2, label=lab)
    nulls = ((0.1, 0.3, 1.0), "gauss", "Gaussian at the update's norm, all four"), ((0.1, 0.3, 1.0), "perm", "shuffled update, all four")
    for ts, kind, lab in nulls:
        for i, (u, *_rest) in enumerate(UPDATES):
            ax.plot(ts, [load(f"{u}/{kind}_{t}.json")["K_W0"] for t in ts], color="#8a8984", ls=":" if kind == "perm" else "--", lw=1,
                    marker="x" if kind == "perm" else "+", ms=5, label=lab if i == 0 else None)
    ax.plot([0.01, 1], [6e-5, 0.6], color=INK, lw=0.8, ls="-.", label="slope of a quadratic path, t²")
    ctrl = load("W0_rel_noise.json")["K_W0"]
    ax.axhline(ctrl, color=MUTED, lw=0.9)
    ax.text(0.0105, ctrl * 1.15, "0.2% relative noise on every weight", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t, the fraction of the update added")
    ax.set_ylabel("KL(W0 ‖ W0 + t·change), nats per position")
    ax.set_title("Is the clean model sensitive to any change, or only along the update?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    style(ax)
    finish(fig, "kl_path.png")


def gain_path() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
    for u, lab, col, mark in UPDATES:
        w0 = load("W0_on_a1.json" if u == "u3" else "W0_on_maths.json")["nll"]
        ft = load(f"{u}/dw_1.0.json")["nll"]
        ax.plot((0, *T), [0, *((w0 - load(f"{u}/dw_{t}.json")["nll"]) / (w0 - ft) for t in T)], color=col, marker=mark, ms=5, lw=2, label=lab)
    ax.axhline(1, color=INK, lw=1.2)
    ax.text(0.8, 0.94, "fine-tuned model = 1", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("t, the fraction of the update added (W0 + t·δW, clean model)")
    ax.set_ylabel("gain (share of the fine-tune's next-token NLL drop)")
    ax.set_title("Is a fraction of the update better on next-token NLL than the whole of it?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    style(ax)
    finish(fig, "gain_path.png")


def calibration() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=170)
    lim = (0, 3.2)
    ax.plot(lim, lim, color=INK, lw=1, label="calibrated: NLL = entropy")
    ax.plot(lim, [lim[0] + 0.5, lim[1] + 0.5], color=MUTED, lw=1, ls="--", label="filed line: NLL − entropy = 0.5")
    for f, lab, dy in (("W0_on_maths", "W0, maths rows", -0.14), ("W0_on_web", "W0, FineWeb rows", 0.08), ("W0_on_a1", "W0, template rows", 0.08)):
        r = load(f"{f}.json")
        ax.scatter(r["H"], r["nll"], color="#8a8984", s=60, zorder=3, label="W0" if f == "W0_on_maths" else None)
        ax.annotate(lab, (r["H"], r["nll"]), xytext=(r["H"] + 0.05, r["nll"] + dy), fontsize=8, color=INK)
    for u, lab, col, mark in UPDATES:
        r = load(f"{u}/dw_1.0.json")
        ax.scatter(r["H"], r["nll"], color=col, marker=mark, s=55, zorder=3, label=f"fine-tuned: {lab}")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("mean entropy of the prediction, nats")
    ax.set_ylabel("next-token NLL, nats per position (T − 1 positions)")
    ax.set_title("How much more sure is each model than it has a right to be?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    finish(fig, "calibration.png")


def by_turn() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=170)
    roles = ("user", "assistant", "other")
    for j, role in enumerate(roles):
        for i, (u, lab, col, mark) in enumerate(UPDATES):
            r = load(f"{u}/detail.json")["by_role"][role]
            x = j + (i - 1.5) * 0.17
            ax.annotate("", xy=(x, r["nll_FT"]), xytext=(x, r["nll_W0"]), arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.8})
            ax.scatter(x, r["nll_W0"], color="#8a8984", s=26, zorder=3, label="W0" if (i, j) == (0, 0) else None)
            ax.scatter(x, r["nll_FT"], color=col, marker=mark, s=36, zorder=3, label=f"fine-tuned: {lab}" if j == 0 else None)
    ax.set_xticks(range(3), ["target in a user turn", "target in an assistant turn", "any other position"])
    ax.set_ylabel("next-token NLL, nats per position")
    ax.set_ylim(0.8, 4)
    ax.set_title("Where on the rows does the next-token NLL fall, and where does it rise?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    finish(fig, "by_turn.png")


if __name__ == "__main__":
    kl_path()
    gain_path()
    calibration()
    by_turn()
