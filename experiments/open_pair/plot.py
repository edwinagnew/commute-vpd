"""E58's figures, one per claim, from results/run.json (no model). E54's 32-key states, read from the step-2 table
(experiments/regating_truth_table/results/table.json), are the references for r, the stand-in ratio and S.

  figures/label.png         the held-out label x per state, elicit and cipher, with the 0.5 cut, each state's 3 s mark,
                            and the raw (pool-shift included) x beside
  figures/agreement.png     r, the stand-in ratio and S against the label x for all six states, cuts at 0.5 on both axes,
                            E54's 32-key values as reference lines
  figures/heldout_gain.png  at the held-out keys' windows: the log-odds gain of the window's own value, of the other 7
                            held-out values, and of the 24 trained values (inferred from H, H_spec and H_other)

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/open_pair/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
COLOR = {"elicit": "#1f9e6e", "cipher": "#eb6834"}
VALUE_COLOR = {"own": "#2a2a2a", "held": "#9a9893", "trained": "#2a78d6"}
INK, GRID, SHADE = "#52514e", "#e4e3df", "#f6efe2"
SEEDS = (0, 1, 2)
N_POOL, N_HELD = 32, 8


def load() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, float]]]:
    run = json.loads((HERE / "results" / "run.json").read_text())
    table = json.loads((HERE.parent / "regating_truth_table" / "results" / "table.json").read_text())["runs"]
    e54 = {k.split("/")[1]: {m: v[m]["value"] for m in ("r", "ratio", "S")} for k, v in table.items() if k.startswith("e54/")}
    return run["states"], run["score"], e54


def trained_values_gain(s: dict[str, Any]) -> float:
    """Mean gain of the 24 trained values at the held-out windows. H_spec = H - mean gain of the other 31 pool values
    (32 distinct values), and H_other is the mean over the 7 other held-out values, so the 24 trained values' mean is
    (31 (H - H_spec) - 7 H_other) / 24. Inferred, not read directly."""
    n_other, n_sib = N_POOL - 1, N_HELD - 1
    return (n_other * (s["H"] - s["H_spec"]) - n_sib * s["H_other"]) / (n_other - n_sib)


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def label_fig(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, task in enumerate(("elicit", "cipher")):
        for j, s in enumerate(SEEDS):
            d = st[f"{task}_s{s}"]
            x0 = i + 0.12 * (j - 1)
            ax.scatter([x0], [d["x"]], s=40, color=COLOR[task], zorder=3, label="scored: value-specific x" if i == j == 0 else None)
            ax.scatter([x0 + 0.04], [d["x_raw"]], s=30, color="white", edgecolors=COLOR[task], lw=1.3, zorder=3,
                       label="raw x, pool shift included (not scored;\nno bootstrap of its own, so no 3 s mark)" if i == j == 0 else None)
            ax.plot([x0 - 0.04, x0 + 0.04], [3 * d["s"]] * 2, color=INK, lw=1.4, zorder=2,
                    label="each state's 3 s mark for the scored x:\n'new' at or below it" if i == j == 0 else None)
    ax.axhline(1, color=INK, lw=0.7, ls="--")
    ax.axhline(0.5, color=INK, lw=1.0, ls=":")
    ax.axhline(0, color=INK, lw=0.7)
    for y, text in ((1, "full carry-over: held-out keys gain as much as trained keys"), (0.5, "'reuses' cut, 0.5"),
                    (0, "no carry-over")):
        ax.annotate(text, (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.set_xticks([0, 1], ["elicit map\n(values W0 half-knows), 3 seeds", "cipher\n(arbitrary values), 3 seeds"], fontsize=8)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.5, 1.12)
    ax.legend(fontsize=7, loc="center right", frameon=False)
    style(ax, "Did the fine-tune on 24 keys carry over to the 8 keys it never saw?",
          "x: held-out-key gain over trained-key gain\n(own value's log-odds, minus the other values' mean)")
    fig.tight_layout()
    fig.savefig(FIG / "label.png", dpi=150)
    plt.close(fig)


def agreement_fig(st: dict[str, Any], e54: dict[str, dict[str, float]]) -> None:
    metrics = (("r", "re-gating share r"), ("ratio", "stand-in ratio"), ("S", "steering share S"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True, sharex=True)
    for ax, (m, name) in zip(axes, metrics, strict=True):
        ax.axvspan(-0.6, 0.5, ymin=0, ymax=1, color=SHADE, zorder=0)
        ax.fill_between([-0.6, 0.5], 0.5, 1.15, color="#f2ddd0", zorder=0)
        ax.text(-0.55, 1.08, "behaviour says new,\ninstrument says reuses", fontsize=7, color=INK, va="top")
        ax.axvline(0.5, color=INK, lw=1.0, ls=":")
        ax.axhline(0.5, color=INK, lw=1.0, ls=":")
        for task in ("elicit", "cipher"):
            xs = [st[f"{task}_s{s}"]["x"] for s in SEEDS]
            ys = [st[f"{task}_s{s}"][m] for s in SEEDS]
            ax.scatter(xs, ys, s=40, color=COLOR[task], zorder=3, label=f"{task}, 24 keys trained, 3 seeds")
            ref = [e54[f"{task}_s{s}"][m] for s in (0, 1)]
            ax.hlines(ref, 0.62, 1.0, color=COLOR[task], lw=1.2, ls="--", zorder=2,
                      label=f"{task}, E54: all 32 keys trained, 2 seeds")
        ax.text(0.81, 0.62, "E54: no held-out\nkeys, so no x", fontsize=6.5, color=INK, ha="center")
        ax.text(0.52, -0.03, "'reuses' if x ≥ 0.5", fontsize=7, color=INK)
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("held-out label x (value-specific)", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("reading: 'reuses' if ≥ 0.5", color=INK)
    axes[0].set_xlim(-0.6, 1.05)
    axes[0].set_ylim(-0.08, 1.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Do r, the stand-in ratio and steering agree with the held-out label on the six fine-tunes?",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "agreement.png", dpi=150)
    plt.close(fig)


def heldout_fig(st: dict[str, Any]) -> None:
    groups = (("own", "the window's own value\n(read: H)"), ("held", "the other 7 held-out\nvalues (read)"),
              ("trained", "the 24 trained values\n(inferred)"))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, task in enumerate(("elicit", "cipher")):
        for j, (g, name) in enumerate(groups):
            vals = []
            for s in SEEDS:
                d = st[f"{task}_s{s}"]
                vals.append({"own": d["H"], "held": d["H_other"], "trained": trained_values_gain(d)}[g])
            x0 = i * 4 + j
            ax.bar([x0], [sum(vals) / len(vals)], width=0.7, color=VALUE_COLOR[g], alpha=0.35,
                   edgecolor=VALUE_COLOR[g], label=name if i == 0 else None)
            ax.scatter([x0 + 0.15 * (k - 1) for k in range(3)], vals, s=18, color=VALUE_COLOR[g], zorder=3)
    ax.axhline(0, color=INK, lw=0.8)
    ax.annotate("W0: no change", (0.01, 0), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                fontsize=7, color=INK)
    ax.set_xticks([1, 5], ["elicit map: own value is one of W0's\nrank 2-20 tokens at that key",
                           "cipher: own value is arbitrary"], fontsize=8)
    ax.text(3.0, 3.6, "for scale: at the\ntrained keys the\nown value gained\n11.0-11.3 (elicit),\n12.5-12.9 (cipher)",
            ha="center", va="top", fontsize=7, color=INK)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    style(ax, "At the 8 keys it never saw, which values did each fine-tune push up?",
          "log-odds gain at the held-out keys' windows\n(fine-tuned minus W0; bars = mean of 3 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "heldout_gain.png", dpi=150)
    plt.close(fig)


def main() -> None:
    st, _, e54 = load()
    FIG.mkdir(exist_ok=True)
    label_fig(st)
    agreement_fig(st, e54)
    heldout_fig(st)
    for task in ("elicit", "cipher"):
        print(task, "trained values' gain at held-out windows:",
              [round(trained_values_gain(st[f"{task}_s{s}"]), 3) for s in SEEDS])


if __name__ == "__main__":
    main()
