"""E72's figures from the results JSONs alone; no model.

  figures/t1_gain.png    gain kept under masks made on the fine-tuned model, per state and update
  figures/t2_kl.png      KL from the fine-tuned model of W0 plus one part of the update, clean model
  figures/t2_gain.png    next-token gain of the same clean states
  figures/t0_random.png  share of its own KL gap a random change and the whole update close under W0's masks

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_fixed_masks/plot.py
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
UPDATES = (("u1", "first fine-tune\nstep 10"), ("u2", "gold fine-tune\nstep 100"), ("u3", "template\nstep 100"))
T1_ARMS = (("do_nothing", "do-nothing", "#8a8984", "o"), ("lower", "rarely-on half", "#2a78d6", "v"),
           ("random", "random change", "#eb6834", "X"), ("adjoint", "adjoint split", "#1baf7a", "s"),
           ("ls", "least-squares split", "#6250d6", "D"), ("upper", "often-on half", "#eda100", "^"),
           ("ceiling", "whole update, unmasked", "#e34948", "*"))
T2_ARMS = (("W0", "W0, no change", "#8a8984", "o"), ("random", "random change", "#eb6834", "X"),
           ("remainder", "remainder, sent to Δ", "#c2409a", "P"), ("lower", "rarely-on half's part", "#2a78d6", "v"),
           ("upper", "often-on half's part", "#eda100", "^"), ("inspan", "in-span part", "#6250d6", "D"))
SCORE = json.loads((RES / "score.json").read_text())


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


def ft_on_t1_scale(u: str) -> float:
    """The clean fine-tuned model on T1's gain, whose zero is masked W0 rather than clean W0 (E68's formula)."""
    s = json.loads((HERE.parent / "qwen_pass_test" / "results" / u / "W0.json").read_text())["sums"]
    return (sum(s["nll"]) - sum(s["ft"])) / (sum(s["clean"]) - sum(s["ft"]))


def scatter(ax: plt.Axes, arms: tuple, table: str, key: str, label_values: tuple[str, ...]) -> None:
    n = len(arms)
    for x, (u, _) in enumerate(UPDATES):
        for i, (arm, label, col, mark) in enumerate(arms):
            v = SCORE[u][table]["states"][arm][key]
            dx = (i - (n - 1) / 2) * 0.115
            ax.scatter(x + dx, v, color=col, marker=mark, s=110 if mark == "*" else 46, edgecolor="white", linewidth=0.6,
                       zorder=3, label=label if x == 0 else None)
            if arm in label_values:
                ax.annotate(f"{v:.3g}", (x + dx, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(UPDATES)), [lab for _, lab in UPDATES])


def t1_gain() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=170)
    for x, (u, _) in enumerate(UPDATES):
        top = ft_on_t1_scale(u)
        print(u, "FT on T1's scale", top)
        ax.plot([x - 0.45, x + 0.45], [top, top], color=INK, lw=1.5, label="clean fine-tuned model" if x == 0 else None)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.text(2.48, 0.012, "W0 under W0's masks", ha="right", fontsize=8, color=MUTED)
    scatter(ax, T1_ARMS, "T1", "gain", ("do_nothing", "ceiling"))
    ax.set_ylabel("gain kept (share of the fine-tune's next-token NLL drop)")
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("Under masks made on the fine-tuned model, does adding the update move the gain?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    style(ax)
    finish(fig, "t1_gain.png")


def t2_kl() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=170)
    scatter(ax, T2_ARMS, "T2", "K", ("W0", "random", "remainder", "lower", "upper", "inspan"))
    ax.set_yscale("log")
    ax.set_ylim(0.0015, 4)
    ax.set_ylabel("KL(fine-tuned model ‖ state), nats per position, log scale")
    ax.set_title("In the clean model, which parts of the update land near the fine-tuned model?\nThe fine-tuned model itself reads 0, off this axis", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    style(ax)
    finish(fig, "t2_kl.png")


def t2_gain() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=170)
    ax.axhline(1, color=INK, lw=1.3)
    ax.text(1.5, 1.06, "fine-tuned model = 1", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.text(1.5, 0.06, "clean W0 = 0", ha="center", fontsize=8, color=MUTED)
    scatter(ax, T2_ARMS[1:], "T2", "gain", ("random", "remainder", "lower", "upper", "inspan"))
    ax.set_ylabel("gain (share of the fine-tune's next-token NLL drop)")
    ax.set_ylim(-4.2, 1.8)
    ax.set_title("In the clean model, does each part keep the next-token gain?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    style(ax)
    finish(fig, "t2_gain.png")


def t0_random() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=170)
    for x, (u, _) in enumerate(UPDATES[:2]):
        t = SCORE[u]["T0"]["specific"]
        ax.fill_between([x - 0.3, x + 0.3], max(t["r2_C"] - 0.2, 0), t["r2_C"] + 0.2, color=GRID, zorder=1,
                        label="filed 'any change' band: whole update ± 0.2, a stated guess" if x == 0 else None)
        ax.scatter(x - 0.08, t["r2_C"], color="#e34948", marker="*", s=130, edgecolor="white", zorder=3, label="whole update, unmasked" if x == 0 else None)
        ax.scatter(x + 0.08, t["r2_R"], color="#eb6834", marker="X", s=60, edgecolor="white", zorder=3, label="random change, unmasked" if x == 0 else None)
        ax.annotate(f"{t['r2_C']:.3f}", (x - 0.08, t["r2_C"]), xytext=(-30, 0), textcoords="offset points", fontsize=8, va="center", color=INK)
        ax.annotate(f"{t['r2_R']:.3f}", (x + 0.08, t["r2_R"]), xytext=(7, 0), textcoords="offset points", fontsize=8, va="center", color=INK)
    ax.axhline(1, color=INK, lw=1.3)
    ax.text(1.45, 0.96, "as close as W0's own masked forward = 1", ha="right", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(2), [lab for _, lab in UPDATES[:2]])
    ax.set_ylim(-0.04, 1.05)
    ax.set_ylabel("share of its own KL gap closed under W0's masks")
    ax.set_title("Under W0's masks, does a random change do any better than the real update?", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.62))
    style(ax)
    finish(fig, "t0_random.png")


if __name__ == "__main__":
    t1_gain()
    t2_kl()
    t2_gain()
    t0_random()
