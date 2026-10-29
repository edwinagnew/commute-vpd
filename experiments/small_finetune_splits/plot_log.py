"""The research log's four figures for E22, redrawn from saved results with plain labels (no model, no new runs).

  log_faithful.png  adversarial ratio of each map (do-nothing, least-squares split, adjoint split) of the three
                    fine-tunes at steps 10, 100 and 1000, with the pass / inconclusive / fail bands.
  log_size.png      the same ratio for do-nothing and the least-squares split against the update's size (the largest
                    relative change of any of the 24 matrices), each point labelled with its step.
  log_cost.png      what each faithful map gives up: the target's held-out probability after the adjoint
                    split, and the share of the change the least-squares split leaves in the residual.
  log_outside.png   the share of each update that no first-order change of the subcomponents reaches, over a random
                    update's, along nine checkpoints.

Reads results/small_finetune_splits.json only.
Usage (repo root):  python experiments/small_finetune_splits/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from plot import COLOR, FIGURES, HERE, INK, MARK, MUTED, series, style

LABEL = {"sgd": "small SGD fine-tune (rate 0.01)", "adam_small": "small Adam fine-tune (rate 3e-6)",
         "adam_large": "large Adam fine-tune (rate 3e-5)"}
OPS = {"nothing": "do-nothing (whole change left in the residual)",
       "least_squares": "least-squares split", "adjoint": "adjoint split"}
STEPS = [10, 100, 1000]
BASE_PROB = 0.004  # the unchanged model's held-out probability of the target (report, "What we were trying to test")


def bands(ax: Axes, floor: float, labels: bool) -> None:
    """Pass at or below 1.5/(1+f), fail at or above 1.5*(1+f): f is the unchanged model's spread over two attack seeds."""
    lo, hi = 1.5 / (1 + floor), 1.5 * (1 + floor)
    ax.axhspan(0.85, lo, color="#e3f1e8", zorder=0, label=f"pass (≤ {lo:.2f})" if labels else None)
    ax.axhspan(lo, hi, color="#eeeeea", zorder=0, label="inconclusive" if labels else None)
    ax.axhspan(hi, 9, color="#f8e3df", zorder=0, label=f"fail (≥ {hi:.2f})" if labels else None)
    ax.axhline(1, color=MUTED, ls=":", lw=1.2, label="unchanged model (1)" if labels else None)


def log_axes(ax: Axes) -> None:
    ax.set(yscale="log", ylim=(0.85, 9))
    style(ax)
    ax.set_yticks([0.9, 1, 1.5, 2, 3, 5, 7])


def fig_faithful(out: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, (op, title) in zip(axes, OPS.items(), strict=True):
        bands(ax, out["W0"]["pgd_floor"], ax is axes[0])
        for ft, label in LABEL.items():
            x, y = series(out["rows"], ft, op, "pgd")
            ax.plot(x, y, marker=MARK[ft], color=COLOR[ft], lw=2, ms=7, label=label if ax is axes[0] else None)
            if ft == "adam_large" and op != "adjoint":
                for xi, yi in zip(x, y, strict=True):
                    ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(8, 4), fontsize=8.5)
        ax.set(xscale="log", xticks=STEPS, xticklabels=[str(s) for s in STEPS], xlim=(7, 1500),
               xlabel="fine-tune step", title=title)
        log_axes(ax)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("adversarial ratio\n(unchanged model = 1)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Both small fine-tunes read like the unchanged model under every map at every step;\n"
                 "only the large Adam fine-tune fails (do-nothing at step 10, least squares at steps 10 and 1,000)", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / "log_faithful.png", dpi=130)
    plt.close(fig)


def fig_size(out: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, op in zip(axes, ("nothing", "least_squares"), strict=True):
        bands(ax, out["W0"]["pgd_floor"], ax is axes[0])
        for ft, label in LABEL.items():
            rows = sorted((r for r in out["rows"] if r["ft"] == ft and r["op"] == op), key=lambda r: r["step"])
            xs, ys = [r["rel_norm_max"] * 100 for r in rows], [r["pgd"] for r in rows]
            ax.plot(xs, ys, marker=MARK[ft], color=COLOR[ft], lw=1, ls="--", ms=8,
                    label=label if ax is axes[0] else None)
            for r, xi, yi in zip(rows, xs, ys, strict=True):
                ax.annotate(f"step {r['step']}", (xi, yi), textcoords="offset points", xytext=(6, -12 if
                            ft == "sgd" else 6), fontsize=7.5, color=COLOR[ft])
        ax.set(xscale="log", xlim=(0.15, 4.5), xlabel="size of the update: largest change of any matrix (% of its norm)",
               title=OPS[op])
        log_axes(ax)
        ax.set_xticks([0.2, 0.5, 1, 2, 3], ["0.2", "0.5", "1", "2", "3"])
        ax.set_title(OPS[op], fontsize=10)
    axes[0].set_ylabel("adversarial ratio (unchanged model = 1)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Size alone does not decide: the SGD fine-tune at step 1,000 is larger than the large Adam one\n"
                 "at step 10, and passes where that one fails", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / "log_size.png", dpi=130)
    plt.close(fig)


def fig_cost(out: dict[str, Any]) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    a.axhline(1, color=INK, ls="--", lw=1, label="the fine-tunes themselves (0.97 to 1.00)")
    a.axhline(BASE_PROB, color=MUTED, ls=":", lw=1.2, label="unchanged model (0.004)")
    for ft, label in LABEL.items():
        x, y = series(out["rows"], ft, "adjoint", "p_target")
        a.plot(x, y, marker=MARK[ft], color=COLOR[ft], lw=2, ms=7, label=label)
        a.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points", xytext=(8, -3), fontsize=8.5)
        x, y = series(out["rows"], ft, "least_squares", "delta_share")
        b.plot(x, y, marker=MARK[ft], color=COLOR[ft], lw=2, ms=7)
    a.set(ylim=(0, 1.08), ylabel="held-out probability of the target token",
          title="adjoint split: keeps the residual fixed, loses behaviour")
    b.set(ylim=(0, 1.05), ylabel="share of the change left in the residual",
          title="least-squares split: keeps behaviour, leaves change in the residual")
    b.axhline(1, color=MUTED, ls=":", lw=1.2)
    b.text(12, 0.95, "do-nothing leaves all of it (1)", fontsize=8, color=MUTED, va="top")
    for ax in (a, b):
        ax.set(xscale="log", xticks=STEPS, xticklabels=[str(s) for s in STEPS], xlim=(7, 1500),
               xlabel="fine-tune step")
        style(ax)
        ax.title.set_fontsize(10)
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("Every faithful map gives something up; of the adjoint splits, the SGD fine-tune's keeps the most by step 1,000",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    fig.savefig(FIGURES / "log_cost.png", dpi=130)
    plt.close(fig)


def fig_outside(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhline(1, color=MUTED, ls=":", lw=1.2, label="a random update of the same size (1)")
    for ft, label in LABEL.items():
        by = out["series"][ft]
        steps = sorted(by, key=int)
        ys = [by[s]["blocks"]["pooled_ns"]["out_out_ratio"] for s in steps]
        ax.plot([int(s) for s in steps], ys, marker=MARK[ft], color=COLOR[ft], lw=2, ms=6, label=label)
        ax.annotate(f"{ys[-1]:.2f}", (1000, ys[-1]), textcoords="offset points", xytext=(8, -3 if ft != "adam_small"
                    else -10), fontsize=8.5)
    ax.set(xscale="log", yscale="log", xlim=(0.8, 2000), ylim=(0.02, 1.3), xlabel="fine-tune step",
           ylabel="share out of the subcomponents' reach\n(a random update = 1)")
    style(ax)
    ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.4, 1])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("Both Adam fine-tunes end with about 0.4 of a random update's share out of reach;\n"
                 "the SGD one stays at 0.03 to 0.06", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "log_outside.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = json.loads((HERE / "results" / "small_finetune_splits.json").read_text())
    fig_faithful(out)
    fig_size(out)
    fig_cost(out)
    fig_outside(out)
    print(f"wrote {sorted(p.name for p in FIGURES.glob('log_*.png'))}")


if __name__ == "__main__":
    main()
