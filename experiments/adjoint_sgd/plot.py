"""Figures for adjoint_sgd (E31), read from results/adjoint_sgd.json only:
  keep.png    held-out P(target) of every adjoint state against the reproduced share R, one point per state; colour =
              optimiser and rate, marker = task and seed, the fine-tune's own P as the reference line
  steps.png   adjoint P(target) along steps 10/100/1000 per task and seed, the fine-tunes' own P dashed
  share.png   R along steps per task and seed, E14's Adam 3e-5 emoticon series beside it
  floor.png   adversarial ratio of the adjoint and do-nothing states per task and seed, with the same-job pass band
  budget.png  budget-sweep ratio to W0 against the adversary's step count, W0's own seed-1 / seed-0 as the noise

Usage (repo root): uv run python experiments/adjoint_sgd/plot.py [results json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
LINE, BUDGET_PASS = 1.5, 1.241
E14_R = (0.5763, 0.4222, 0.3064)
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COLOR = {"sgd": "#2a78d6", "adam_3e-05": "#eb6834", "adam_3e-06": "#1baf7a"}
NAME = {"sgd": "SGD", "adam_3e-05": "Adam 3e-5 (large)", "adam_3e-06": "Adam 3e-6 (small)"}
PANELS = {("emoticon", 0): "emoticon, seed 0", ("emoticon", 1): "emoticon, seed 1",
          ("r1_t534_v369", 0): "r1_t534_v369", ("r0_t352_v310", 0): "r0_t352_v310"}
MARK = {("emoticon", 0): "o", ("emoticon", 1): "D", ("r1_t534_v369", 0): "s", ("r0_t352_v310", 0): "^"}


def kind(r: dict) -> str:
    return "sgd" if r["opt"] == "sgd" else f"adam_{r['lr']:.0e}"


def style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def by_arm(rows: list[dict]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in arms.items()}


def fig_keep(out: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(1, color=MUTED, ls=":", lw=1.2)
    ax.text(0.01, 0.985, "fine-tune itself ≈ 1", color=MUTED, fontsize=8, va="top", transform=ax.get_yaxis_transform())
    ax.axhline(0.004, color=MUTED, ls="--", lw=1)
    ax.text(0.01, 0.012, "unchanged model ≈ 0.004", color=MUTED, fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
    for r in out["rows"]:
        ax.scatter(r["R"], r["adj_p"], s={10: 22, 100: 48, 1000: 95}[r["step"]], marker=MARK[(r["task"], r["seed"])], color=COLOR[kind(r)],
                   edgecolor="white", linewidth=1.2, zorder=3)
    style(ax)
    ax.set_xlabel("reproduced share R (share of the update the adjoint split's change reproduces)")
    ax.set_ylabel("adjoint state's held-out P(target)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Behaviour kept by the adjoint split against how much of the update it reproduces\n(marker size: step 10, 100, 1000)",
                 fontsize=10, loc="left")
    hs = [Line2D([], [], marker="o", ls="", color=c, label=NAME[k]) for k, c in COLOR.items()]
    hs += [Line2D([], [], marker=m, ls="", color=MUTED, label=PANELS[p]) for p, m in MARK.items()]
    ax.legend(handles=hs, fontsize=7.5, frameon=False, loc="lower right")
    save(fig, "keep.png")


def panels(out: dict, draw, ylabel: str, name: str, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    arms = by_arm(out["rows"])
    for ax, (p, ptitle) in zip(axes, PANELS.items(), strict=True):
        for rs in arms.values():
            if (rs[0]["task"], rs[0]["seed"]) == p:
                draw(ax, rs)
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000], ["10", "100", "1000"])
        ax.set_title(ptitle, fontsize=9, loc="left")
        ax.set_xlabel("fine-tuning step")
    axes[0].set_ylabel(ylabel)
    hs = [Line2D([], [], marker="o", color=c, label=NAME[k]) for k, c in COLOR.items()]
    fig.legend(handles=hs, fontsize=8, frameon=False, loc="upper right", ncol=3)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    save(fig, name)


def fig_steps(out: dict) -> None:
    def draw(ax: Axes, rs: list[dict]) -> None:
        c, x = COLOR[kind(rs[0])], [r["step"] for r in rs]
        ax.plot(x, [r["adj_p"] for r in rs], marker="o", color=c, lw=2, ms=6)
        ax.plot(x, [r["ft_p"] for r in rs], color=c, lw=1, ls="--", alpha=0.7)
        if rs[0]["opt"] == "sgd":
            ax.annotate(f"lr {rs[0]['lr']:g}", (x[-1], rs[-1]["adj_p"]), xytext=(-8, 6), textcoords="offset points",
                        fontsize=7, color=MUTED, ha="right")

    panels(out, draw, "held-out P(target)", "steps.png",
           "Adjoint state's P(target) along training (solid) against the fine-tune's own (dashed)")


def fig_share(out: dict) -> None:
    def draw(ax: Axes, rs: list[dict]) -> None:
        ax.plot([r["step"] for r in rs], [r["R"] for r in rs], marker="o", color=COLOR[kind(rs[0])], lw=2, ms=6)
        if (rs[0]["task"], rs[0]["seed"], kind(rs[0])) == ("emoticon", 0, "adam_3e-05"):
            ax.plot([10, 100, 1000], E14_R, color=MUTED, ls=":", marker="x", lw=1, label="E14 Adam 3e-5")
            ax.legend(fontsize=7, frameon=False, loc="lower left")

    panels(out, draw, "reproduced share R", "share.png", "Reproduced share R of each update along training")


def fig_floor(out: dict) -> None:
    def draw(ax: Axes, rs: list[dict]) -> None:  # the band is the row's own job's W0 floor (shards differ slightly)
        c, x, f = COLOR[kind(rs[0])], [r["step"] for r in rs], rs[0]["floor"]
        ax.axhspan(LINE / (1 + f), LINE * (1 + f), color="#f0efec")
        ax.axhline(1, color=MUTED, ls=":", lw=1)
        ax.plot(x, [r["adj_pgd"] for r in rs], marker="o", color=c, lw=2, ms=6)
        ax.plot(x, [r["nothing_pgd"] for r in rs], marker="s", color=c, lw=1, ls="--", ms=4, alpha=0.7)

    panels(out, draw, "adversarial damage ÷ W₀'s", "floor.png",
           "Adversarial ratio of the adjoint state (solid) and do-nothing (dashed); grey band: inconclusive, below it: pass")


def fig_budget(out: dict) -> None:
    kl = out.get("budget", {})
    if "W0" not in kl or "0" not in kl["W0"]:
        return
    w0 = kl["W0"]["0"]
    ns = sorted(int(n) for n in w0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(BUDGET_PASS, color=INK, ls="--", lw=1)
    ax.text(ns[0], BUDGET_PASS + 0.01, "pass line 1.24", fontsize=8, va="bottom")
    ax.axhline(1, color=MUTED, ls=":", lw=1)
    if "1" in kl["W0"]:
        ax.plot(ns, [kl["W0"]["1"][str(n)] / w0[str(n)] for n in ns], color=MUTED, marker="x", lw=1, label="W₀ seed 1 ÷ seed 0")
    for m, v in kl.items():
        if m == "W0" or "0" not in v:
            continue
        k = "sgd" if "/sgd_" in m else "adam_3e-06"
        got = sorted(int(n) for n in v["0"])
        ax.plot(got, [v["0"][str(n)] / w0[str(n)] for n in got], marker="o", color=COLOR[k], lw=2,
                label=f"{NAME[k]} adjoint, step 1000")
    style(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("adversary's sign steps (step size 2 / n)")
    ax.set_ylabel("masked KL ÷ W₀'s at the same budget")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Does the adjoint state's pass hold under a longer attack?", fontsize=10, loc="left")
    save(fig, "budget.png")


def save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES / name}")


def main() -> None:
    out = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "adjoint_sgd.json").read_text())
    for f in (fig_keep, fig_steps, fig_share, fig_floor, fig_budget):
        f(out)


if __name__ == "__main__":
    main()
