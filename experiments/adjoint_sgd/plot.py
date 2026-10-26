"""Figures for adjoint_sgd (E31), read from results/adjoint_sgd.json only:
  keep.png    held-out P(target) of every adjoint state against the reproduced share R, one point per state; colour =
              optimiser and rate, marker = task and seed, the fine-tune's own P as the reference line
  steps.png   adjoint P(target) along steps 10/100/1000 per task and seed, the fine-tunes' own P dashed
  share.png   R along steps per task and seed, E14's Adam 3e-5 emoticon series beside it
  floor.png   adversarial ratio of the adjoint and do-nothing states per task and seed, with the same-job pass band
  budget.png  budget-sweep ratio to W0 against the adversary's step count, W0's own seed-1 / seed-0 as the noise
  size.png    amendments 1-2: keep (adjoint P / fine-tune P) against fine-tune size for every saved SGD step of E31,
              results/adjoint_sgd_amend1.json and _amend2.json; colour = task, marker = SGD rate, seed 1 hollow,
              Adam states as faint references
  scaled.png  amendment 3: keep, KL elsewhere and adversarial ratio of the scaled (solid) and unscaled (dashed) adjoint
              states along steps, emoticon seed 0, from results/adjoint_sgd_amend3.json; the fine-tune's own KL dotted

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
TASK_COLOR = {"emoticon": "#6250d6", "r1_t534_v369": "#eda100", "r0_t352_v310": "#e87ba4"}
RATE_MARK = {0.01: "o", 0.03: "^"}
AMEND1 = HERE / "results" / "adjoint_sgd_amend1.json"
AMEND2 = HERE / "results" / "adjoint_sgd_amend2.json"
AMEND3, STEPS3 = HERE / "results" / "adjoint_sgd_amend3.json", (10, 100, 1000)
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


def fig_size(out: dict) -> None:
    if not (AMEND1.is_file() and AMEND2.is_file()):
        return
    arms = by_arm(out["rows"] + json.loads(AMEND1.read_text())["rows"] + json.loads(AMEND2.read_text())["rows"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(1, color=MUTED, ls=":", lw=1.2)
    ax.text(0.01, 0.985, "do-nothing (Δ carries the update) ≈ 1", color=MUTED, fontsize=8, va="top",
            transform=ax.get_yaxis_transform())
    for rs in arms.values():
        x, y, c = [r["size"] for r in rs], [r["keep"] for r in rs], TASK_COLOR[rs[0]["task"]]
        if rs[0]["opt"] != "sgd":
            ax.scatter(x, y, s=28, marker="x", color=c, alpha=0.45, linewidth=1.2, zorder=2)
            continue
        seed1 = rs[0]["seed"] == 1
        ax.plot(x, y, color=c, lw=1, alpha=0.6, ls="--" if seed1 else "-", zorder=2)
        ax.scatter(x, y, s=[{10: 30, 100: 55, 1000: 95}[r["step"]] for r in rs], marker=RATE_MARK[rs[0]["lr"]],
                   facecolor="white" if seed1 else c, edgecolor=c if seed1 else "white", linewidth=1.2, zorder=3)
    style(ax)
    ax.set_xscale("log")
    ax.set_xlabel("fine-tune size: largest relative change of a matrix, max ||δW|| / ||W||")
    ax.set_ylabel("keep: adjoint state's P(target) ÷ the fine-tune's own")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does the SGD update's size set how much behaviour the adjoint split keeps?\n"
                 "(lines join one fine-tune's steps 10, 100, 1000; marker size grows with step; two seeds)", fontsize=10, loc="left")
    hs = [Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in TASK_COLOR.items()]
    hs += [Line2D([], [], marker=m, ls="", color=MUTED, label=f"SGD {lr:g}") for lr, m in RATE_MARK.items()]
    hs += [Line2D([], [], marker="o", ls="-", color=MUTED, label="seed 0 (filled, solid)"),
           Line2D([], [], marker="o", ls="--", color=MUTED, markerfacecolor="white", label="seed 1 (hollow, dashed)")]
    hs += [Line2D([], [], marker="x", ls="", color=MUTED, label="Adam 3e-6 / 3e-5 (reference)")]
    ax.legend(handles=hs, fontsize=7.5, frameon=False, loc="lower left")
    save(fig, "size.png")


def fig_scaled(_out: dict) -> None:
    if not AMEND3.is_file():
        return
    rows = json.loads(AMEND3.read_text())["rows"]
    kinds = {r["arm"]: "sgd" if "/sgd_" in r["arm"] else r["arm"].split("/")[-1] for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    panels = (("keep", "keep: state's P(target) ÷ the fine-tune's own"),
              ("kl", "KL elsewhere from W₀ (held-out, non-trigger)"), ("pgd", "adversarial ratio (Pile, ÷ W₀)"))
    for ax, (q, ylabel) in zip(axes, panels, strict=True):
        for arm, k in kinds.items():
            for how, ls in (("scaled", "-"), ("unscaled", "--")):
                rs = sorted((r for r in rows if r["arm"] == arm and r["how"] == how), key=lambda r: r["step"])
                ax.plot([r["step"] for r in rs], [r[q] for r in rs], ls=ls, marker="o", color=COLOR[k], lw=2)
            if q == "kl":  # the fine-tune itself, the reference on the same axes
                rs = sorted((r for r in rows if r["arm"] == arm and r["how"] == "scaled"), key=lambda r: r["step"])
                ax.plot([r["step"] for r in rs], [r["ft_kl"] for r in rs], ls=":", color=COLOR[k], lw=1.2)
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks(STEPS3, [str(s) for s in STEPS3])
        ax.set_xlabel("fine-tune step")
        ax.set_ylabel(ylabel)
    axes[0].axhline(1, color=MUTED, ls=":", lw=1.2)
    axes[0].text(1000, 0.97, "do-nothing ≈ 1 ", color=MUTED, fontsize=8, va="top", ha="right")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    f = max(r["floor"] for r in rows)
    axes[2].axhline(LINE / (1 + f), color=MUTED, ls=":", lw=1.2)
    axes[2].text(10, LINE / (1 + f), f" pass line {LINE / (1 + f):.2f}", color=MUTED, fontsize=8, va="bottom")
    hs = [Line2D([], [], color=COLOR[k], lw=2, label=NAME[k]) for k in dict.fromkeys(kinds.values())]
    hs += [Line2D([], [], color=MUTED, ls="-", label="scaled adjoint (c fitted)"),
           Line2D([], [], color=MUTED, ls="--", label="unscaled adjoint (c = 1)"),
           Line2D([], [], color=MUTED, ls=":", label="the fine-tune itself (KL panel)")]
    axes[0].legend(handles=hs, fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle("Does fitting the scale c matter? The unscaled adjoint split beside the scaled one, emoticon seed 0",
                 fontsize=10, x=0.01, ha="left")
    save(fig, "scaled.png")


def save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES / name}")


def main() -> None:
    out = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "adjoint_sgd.json").read_text())
    for f in (fig_keep, fig_steps, fig_share, fig_floor, fig_budget, fig_size, fig_scaled):
        f(out)


if __name__ == "__main__":
    main()
