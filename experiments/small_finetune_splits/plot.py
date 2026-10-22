"""Figures for small_finetune_splits (E22), read from results/small_finetune_splits.json only:
  placements.png  adversarial ratio of the do-nothing, least-squares and adjoint states at steps 10/100/1000, with W0 and
                  the same-job pass and fail lines
  adjoint.png     held-out P(target) of the adjoint state against the fine-tune's own, and the least-squares state's
                  share of the change left in Delta, along steps
  outside.png     outside-both share / its random-update null along the nine checkpoints, E14's large series beside it
  rescaling.png   rescaling share / its random-update null along the nine checkpoints, E14's large series beside it

Usage (repo root): uv run python experiments/small_finetune_splits/plot.py [results json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
LINE = 1.5
NAMES = {"sgd": "SGD 0.01 (small)", "adam_small": "Adam 3e-6 (small)", "adam_large": "Adam 3e-5 (large)"}
COLOR = {"sgd": "#2a78d6", "adam_small": "#eb6834", "adam_large": "#1baf7a"}
MARK = {"sgd": "o", "adam_small": "s", "adam_large": "^"}
OPS = {"nothing": "do-nothing", "least_squares": "least-squares split", "adjoint": "adjoint split"}
E14_OUT_OUT = (0.184, 0.175, 0.185, 0.199, 0.219, 0.274, 0.331, 0.352, 0.410)
E14_RESCALING = (2.03, 1.99, 1.91, 1.91, 2.08, 2.65, 4.26, 5.70, 11.43)
INK, MUTED = "#0b0b0b", "#8a8984"


def style(ax: Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def series(rows: list[dict], ft: str, op: str, key: str) -> tuple[list[int], list[float]]:
    pts = sorted((r["step"], r[key]) for r in rows if r["ft"] == ft and r["op"] == op)
    return [p[0] for p in pts], [p[1] for p in pts]


def fig_placements(out: dict) -> None:
    f = out["W0"]["pgd_floor"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (op, title) in zip(axes, OPS.items(), strict=True):
        ax.axhspan(LINE / (1 + f), LINE * (1 + f), color="#f0efec", label="inconclusive band (same-job W₀ spread)")
        ax.axhline(1, color=MUTED, ls=":", lw=1.2, label="W₀ (unchanged model)")
        ax.axhline(LINE, color=INK, ls="--", lw=1, label="pass line 1.5")
        for ft, label in NAMES.items():
            x, y = series(out["rows"], ft, op, "pgd")
            ax.plot(x, y, marker=MARK[ft], color=COLOR[ft], lw=2, ms=7, label=label)
        ax.set(xscale="log", yscale="log", xticks=[10, 100, 1000], xticklabels=["10", "100", "1000"], title=title,
               xlabel="fine-tune step")
        style(ax)
        ax.set_yticks([0.9, 1, 1.5, 2, 3, 5, 7])
    axes[0].set_ylabel("adversarial KL ratio to W₀ (Pile, 20-step attack)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Does each placement of a fine-tune stay as faithful as the unchanged model?")
    fig.tight_layout()
    fig.savefig(FIGURES / "placements.png", dpi=130)


def fig_adjoint(out: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ft, label in NAMES.items():
        x, y = series(out["rows"], ft, "adjoint", "p_target")
        a.plot(x, y, marker=MARK[ft], color=COLOR[ft], lw=2, ms=7, label=f"{label}: adjoint split")
        x, y = series(out["rows"], ft, "nothing", "p_target")
        a.plot(x, y, ls="--", color=COLOR[ft], lw=1, alpha=0.7)
        x, y = series(out["rows"], ft, "least_squares", "delta_share")
        b.plot(x, y, marker=MARK[ft], color=COLOR[ft], lw=2, ms=7, label=f"{label}: least-squares split")
        x, y = series(out["rows"], ft, "adjoint", "delta_share")
        b.plot(x, y, ls=":", color=COLOR[ft], lw=1.2)
    a.plot([], [], ls="--", color=MUTED, label="the fine-tune itself (do-nothing)")
    a.axhline(0.004, color=MUTED, ls=":", lw=1.2, label="W₀ (0.004)")
    a.set(ylabel="held-out P(target) at the trigger", ylim=(0, 1.05), title="behaviour the adjoint split keeps")
    b.plot([], [], ls=":", color=MUTED, label="adjoint split (0 by construction)")
    b.set(ylabel="share of ‖δW‖² left in Δ", ylim=(-0.02, 0.6), title="change the least-squares split leaves in Δ")
    for ax in (a, b):
        ax.set(xscale="log", xticks=[10, 100, 1000], xticklabels=["10", "100", "1000"], xlabel="fine-tune step")
        ax.legend(fontsize=7.5)
        style(ax)
    fig.suptitle("What each faithful placement gives up, along training")
    fig.tight_layout()
    fig.savefig(FIGURES / "adjoint.png", dpi=130)


def fig_series(out: dict, block: str, key: str, ref: tuple[float, ...], name: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(1, color=MUTED, ls=":", lw=1.2, label="random update of the same size (null = 1)")
    steps = None
    for ft, label in NAMES.items():
        by = out["series"].get(ft, {})
        steps = sorted(by, key=int) or steps
        ax.plot([int(s) for s in sorted(by, key=int)], [by[s][block]["pooled_ns"][key] for s in sorted(by, key=int)],
                marker=MARK[ft], color=COLOR[ft], lw=2, ms=7, label=label)
    if steps:
        ax.plot([int(s) for s in steps], ref[: len(steps)], ls="none", marker="x", color=INK, ms=7,
                label="large fine-tune in E14 (reference)")
    ax.set(xscale="log", yscale="log", xlabel="fine-tune step", ylabel=ylabel, title=title)
    ax.legend(fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=130)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "small_finetune_splits.json"
    out = json.loads(path.read_text())
    FIGURES.mkdir(exist_ok=True)
    fig_placements(out)
    fig_adjoint(out)
    fig_series(out, "blocks", "out_out_ratio", E14_OUT_OUT, "outside.png", "outside-both share ÷ its null",
               "How much of each fine-tune lies where no subcomponent reaches?")
    fig_series(out, "rescaling", "rescaling_ratio", E14_RESCALING, "rescaling.png", "rescaling share ÷ its null",
               "How much of each fine-tune is a rescaling of existing subcomponents?")
    print(f"wrote {sorted(p.name for p in FIGURES.iterdir())}")


if __name__ == "__main__":
    main()
