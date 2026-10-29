"""The research log's four figures for E40, drawn from saved results with plain labels (no model).

  log_control.png    the unchanged decomposition's adversary reading on web text (the earlier run's and this run's)
                     and on the model's own maths answers.            Reads results/G.json, results/predictions.json.
  log_firststep.png  the fine-tune's change at each saved step: KL to the starting model on maths text and the fall
                     in the gold solutions' log-loss.                  Reads results/F.json.
  log_accuracy.png   greedy accuracy before the fine-tune and at step 100, GSM8K test and GSM1k (answers over 512
                     tokens dropped).                                  Reads results/A0.json, results/F.json.
  log_donothing.png  do-nothing's web-text adversary reading over the unchanged decomposition's, steps 1 and 10, one
                     dot per adversary start, with the 1.5 pass line.  Reads results/D.json.

Usage (repo root):  python experiments/qwen_sft/plot_log.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c0392b"
PASS_LINE = 1.5  # predictions.json, P2: a ratio at or under 1.5 passes


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def label_bars(ax: Any, xs: list[float], ys: list[float], fmt: str = "{:.2f}") -> None:
    for x, y in zip(xs, ys, strict=True):
        ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=9)


def fig_control(g: dict[str, Any], pred: dict[str, Any]) -> None:
    earlier = pred["rules"]["G_control"]["adv"]["ref"]
    names = ["web text,\nearlier run", "web text,\nthis run", "the model's own\nmaths answers, this run"]
    ys = [earlier, g["fineweb"]["adv"], g["maths"]["adv"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(range(3), ys, 0.6, color=[MUTED, BLUE, ORANGE])
    label_bars(ax, [0, 1, 2], ys)
    ax.set_xticks(range(3), names, fontsize=9)
    ax.set_ylabel("damage the adversary does to the unchanged\ndecomposition (KL, nats per token)")
    ax.set_title("The unchanged decomposition reproduces its web-text reading,\n"
                 "and the adversary does 2.5 times as much damage on maths text", fontsize=10)
    style(ax)
    save(fig, "log_control.png")


def fig_firststep(f: dict[str, Any]) -> None:
    steps = sorted(int(s) for s in f["saved"])
    kl = [f["saved"][str(s)]["kl_to_start"] for s in steps]
    gold = [f["saved"][str(s)]["gold_drop"] for s in steps]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(steps, gold, "o-", color=BLUE, label="fall in the gold solutions' log-loss (held-out problems)")
    ax.plot(steps, kl, "s-", color=ORANGE, label="change in the model's predictions on maths text (KL to the start)")
    for ys in (gold, kl):
        ax.annotate(f"{ys[0]:.2f}", (steps[0], ys[0]), textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.annotate(f"{ys[-1]:.2f}", (steps[-1], ys[-1]), textcoords="offset points", xytext=(-4, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.set_xlabel("fine-tuning step (constant rate, no warm-up)")
    ax.set_ylabel("nats per token")
    ax.set_ylim(0, 2.2)
    ax.set_title("Most of the fine-tune's gain on the gold solutions comes at its first step", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style(ax)
    save(fig, "log_firststep.png")


def fig_accuracy(a0: dict[str, Any], f: dict[str, Any]) -> None:
    sets = (("test", "GSM8K test"), ("gsm1k", "GSM1k (fresh problems)"))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    w = 0.36
    for i, (key, _) in enumerate(sets):
        before, after = a0["accuracy"][key]["strict_dropped"], f["accuracy"][key]["strict_dropped"]
        ax.bar(i - w / 2, before, w * 0.92, color=MUTED, label="before the fine-tune" if i == 0 else None)
        ax.bar(i + w / 2, after, w * 0.92, color=BLUE, label="after 100 steps" if i == 0 else None)
        label_bars(ax, [i - w / 2, i + w / 2], [before, after], "{:.3f}")
    ax.set_xticks(range(len(sets)), [n for _, n in sets], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of problems answered correctly\n(greedy; answers over 512 tokens dropped)")
    ax.set_title("Fine-tuning on GSM8K's solutions lowers accuracy on both test sets", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    save(fig, "log_accuracy.png")


def fig_donothing(d: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    steps = sorted(d, key=int)
    for i, s in enumerate(steps):
        r = d[s]["fineweb"]
        ax.bar(i, r["ratio"], 0.55, color=BLUE, alpha=0.85)
        dots = list(r["ratios"].values())
        ax.scatter([i + 0.08 * (j - (len(dots) - 1) / 2) for j in range(len(dots))], dots, s=16, color=INK, zorder=3)
        ax.text(i + 0.33, r["ratio"], f"{r['ratio']:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.axhline(1.0, color=MUTED, lw=0.9)
    ax.text(1.45, 1.0, "unchanged\ndecomposition", va="center", fontsize=8, color=MUTED)
    ax.axhline(PASS_LINE, color=RED, lw=1.0, ls="--")
    ax.text(1.45, PASS_LINE + 0.02, "pass line (1.5)", va="bottom", fontsize=8, color=RED)
    ax.set_xticks(range(len(steps)), [f"step {s}" for s in steps], fontsize=9)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0, 1.8)
    ax.set_ylabel("adversarial ratio on web text\n(damage over the unchanged decomposition's)")
    ax.set_title("Leaving the whole update in Δ passes the web-text test\nat steps 1 and 10", fontsize=10)
    style(ax)
    save(fig, "log_donothing.png")


def main() -> None:
    fig_control(load("G.json"), load("predictions.json"))
    fig_firststep(load("F.json"))
    fig_accuracy(load("A0.json"), load("F.json"))
    fig_donothing(load("D.json"))


if __name__ == "__main__":
    main()
