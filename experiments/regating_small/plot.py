"""Figures for regating_small (E24) from results/regating_small.json (JSON only; no model).

  r.png          P1: the share of the fine-tune's gain that comes from Gamma's switching (r), per state, the large
                 Adam 3e-5 state first as the reference, with E18's reading for it.
  shared.png     P2: one fixed change added to W0's values (shared), as a share of what each window's own updated
                 values recover at the trigger (R_shared), per state, with the filed 'fixed pattern' (>= 0.7) and
                 'per-window' (<= 0.3) regions; grey dots: the random null on the same scale.
  recovery.png   every pattern's recovery per state on one axis (own values, shared change, mean values and their W0
                 reference, the ' http' change, the random null): the detail behind both figures and the control C.

Usage (repo root):  uv run python experiments/regating_small/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
NAME = {"adam_3e-05": "Adam 3e-5\n(large, size 0.011)", "sgd_0.01": "SGD 0.01\n(small, 0.0024)",
        "adam_3e-06": "Adam 3e-6\n(small, 0.0022)"}
BARS = {"trig": ("own updated values", BLUE), "shared": ("W0's values + one fixed change", AQUA),
        "mean": ("updated mean values", YELLOW), "mean_W0": ("W0's mean values (reference)", GRID),
        "cross": ("a ' http' fine-tune's change", ORANGE), "random": ("change moved to random subcomponents", MUTED)}


def states(out: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(label, state) in run order: emoticon's large, SGD, Adam 3e-6; then r1_t534's large and SGD if run."""
    rows = list(out["states"].values()) + list(out.get("extra", {}).get("states", {}).values())
    return [(label(v), v) for v in rows]


def label(v: dict[str, Any]) -> str:
    opt = "Adam" if v["optimizer"] == "adam" else "SGD"
    kind = "large" if is_large(v) else "small"
    return f"{v['task'].replace('_v369', '')}\n{opt} {v['lr']:g}\n({kind}, size {v['heldout']['rel_norm_max']:.4f})"


def is_large(v: dict[str, Any]) -> bool:
    return v["optimizer"] == "adam" and v["lr"] == 3e-5


def colors(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [BLUE if is_large(v) else AQUA for _, v in rows]


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_r(out: dict[str, Any]) -> None:
    rows = states(out)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(rows))
    ax.bar(x, [v["r"] for _, v in rows], 0.55, color=colors(rows))
    for i, (_, v) in enumerate(rows):
        ax.text(i, v["r"] + 0.02, f"{v['r']:.2f}", ha="center", color=INK, fontsize=9)
    for i, (_, v) in enumerate(rows):
        if is_large(v):
            ref = out["refs"]["big"][v["task"]]["r"]
            ax.plot([i - 0.3, i + 0.3], [ref, ref], color=INK, lw=2, label="E18's reading of the large state" if i == 0 else None)
    ax.set_xticks(x, [n for n, _ in rows], fontsize=8)
    ax.set_ylabel("r: share of the gain from switching")
    ax.set_ylim(0, 1)
    ax.set_title("Share of the gain from switching: small updates against the large one (blue)", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "r.png")


def fig_shared(out: dict[str, Any]) -> None:
    rows = states(out)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(rows))
    ax.axhspan(0.7, 1.2, color=AQUA, alpha=0.08)
    ax.axhspan(-0.2, 0.3, color=ORANGE, alpha=0.08)
    ax.text(len(rows) - 0.5, 1.15, "fixed pattern (>= 0.7)", ha="right", va="top", color=MUTED, fontsize=8)
    ax.text(len(rows) - 0.5, -0.15, "needs per-window values (<= 0.3)", ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.bar(x, [v["R_shared"] for _, v in rows], 0.55, color=colors(rows))
    ax.scatter(x, [v["random_mean"] / v["rho"]["trig"] for _, v in rows], color=MUTED, s=30, zorder=3,
               label="random null, same scale")
    for i, (_, v) in enumerate(rows):
        ax.text(i, v["R_shared"] + 0.03, f"{v['R_shared']:.2f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, [n for n, _ in rows], fontsize=8)
    ax.set_ylabel("fixed change's recovery / own values' recovery")
    ax.set_ylim(-0.2, 1.2)
    ax.set_title("How much one fixed change gives back: small updates against the large one (blue)", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "shared.png")


def fig_recovery(out: dict[str, Any]) -> None:
    rows = states(out)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.13
    for j, (b, (label, color)) in enumerate(BARS.items()):
        vals = [v["random_mean"] if b == "random" else v["rho"][b] for _, v in rows]
        ax.bar(np.arange(len(rows)) + (j - 2.5) * w, vals, w * 0.88, color=color, label=label,
               edgecolor=MUTED if b == "mean_W0" else None, linewidth=0.6)
    ax.axhline(0.1, color=ORANGE, lw=0.8, ls="--")
    ax.text(len(rows) - 0.45, 0.12, "control limit 0.1", color=MUTED, fontsize=8, ha="right")
    ax.set_xticks(np.arange(len(rows)), [n for n, _ in rows], fontsize=8)
    ax.set_ylabel("recovery of the switching effect")
    ax.set_title("Each pattern at the trigger, W0's values elsewhere (0 = W0's values, 1 = updated values everywhere)",
                 fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "recovery.png")


def main() -> None:
    out = json.loads((HERE / "results" / "regating_small.json").read_text())
    fig_r(out)
    fig_shared(out)
    fig_recovery(out)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
