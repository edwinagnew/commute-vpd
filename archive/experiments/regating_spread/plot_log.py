"""The research log's figures for E36, redrawn from saved results with plain labels (no model, no runs).

Every recovery is a share of Γ's switching recovered at every held-out position of the lowercase-text fine-tune:
0 = the original model's importance values everywhere, 1 = Γ's values when it reads the fine-tuned model.
  log_switching.png  per step, the switching share for the lowercase fine-tune (learned) and the code-like fine-tune
                     (learned nothing), with each one's gain in nats per token
                     (regating_spread_amend1.json, regating_spread.json)
  log_sampling.png   the nearest-values rule against the number of fitted positions, one line per step
                     (regating_spread_amend2.json for 16,384 and 65,536, regating_spread_amend3.json for 262,144)
  log_space.png      which notion of "alike" predicts the change: nearest by original values, by the attention inputs
                     Γ reads, by the last 4 tokens of text, by a bag of the whole preceding text; the current token; the
                     position in the row; one change everywhere (regating_spread_amend4/5/6.json, amend1 for the last)
  log_room.png       the nearest-values rule against rules that treat each subcomponent's own value alone: 16 bins of
                     it, and a straight line in it (regating_spread_amend7.json)

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/regating_spread/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import GRID, HERE, MUTED

RES, FIG = HERE / "results", HERE / "figures"
STEPS = ("10", "100", "300")
SHADES = ("#9ec5f0", "#2a78d6", "#123f7a")  # step 10, 100, 300: light to dark
GREEN, GREY = "#1baf7a", "#a3a29d"


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())["states"]


def style(ax: Any, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"wrote {FIG / name}")


def fig_switching(low: dict[str, Any], code: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.36
    for j, (st, lab, c) in enumerate(((low, "lowercase text (learned)", GREEN),
                                      (code, "code-like text (learned nothing)", GREY))):
        for i, s in enumerate(STEPS):
            if s not in st:
                continue
            x, r = i + (j - 0.5) * w, float(st[s]["r"])
            ax.bar(x, r, w * 0.92, color=c, label=lab if i == 0 else None)
            ax.text(x, r + 0.02, f"{r:.2f}\ngain {float(st[s]['gain']):+.2f} nats", ha="center", fontsize=7.5)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(STEPS)), [f"step {s}" for s in STEPS])
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("switching share of the gain\n(1 = switching alone gives the whole gain)")
    ax.set_title("The switching share is high on both runs; for the run that learned nothing\n"
                 "it is a share of a gain below zero, so it reads nothing", fontsize=10)
    ax.text(2 + 0.5 * w, 0.03, "code-like:\nnot run", ha="center", fontsize=7.5, color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "log_switching.png")


def fig_sampling(a2: dict[str, Any], a3: dict[str, Any]) -> None:
    ns = (16384, 65536, 262144)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for s, c, dy in zip(STEPS, SHADES, (0, 0.012, -0.012), strict=True):
        ys = [a2[s]["R_near16"], a2[s]["R_near65"], a3[s]["R_near262"]]
        ax.plot(ns, ys, "o-", color=c, label=f"step {s}")
        ax.text(ns[-1] * 1.12, ys[-1] + dy, f"{ys[-1]:.2f}", va="center", fontsize=8, color=c)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, ["16,384", "65,536", "262,144"])
    ax.set_xlim(12000, 400000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("training positions the rule draws its neighbours from")
    ax.set_ylabel("share of the switching recovered\non held-out rows")
    ax.set_title("The nearest-values rule improves with more fitted positions at every step", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "log_sampling.png")


def grouped_barh(ax: Any, rows: list[tuple[str, list[float]]]) -> None:
    """One group of three bars (steps 10, 100, 300) per rule, the first rule at the top."""
    h = 0.26
    for i, (_, vals) in enumerate(rows):
        for j, (v, c) in enumerate(zip(vals, SHADES, strict=True)):
            y = i + (j - 1) * h
            ax.barh(y, v, h * 0.9, color=c, label=f"step {STEPS[j]}" if i == 0 else None)
            ax.text(max(v, 0) + 0.01, y, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-0.06, 1)
    ax.set_xlabel("share of the switching recovered on held-out rows (1 = Γ's own values)")
    style(ax, "x")
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def fig_space(a1: dict[str, Any], a4: dict[str, Any], a5: dict[str, Any], a6: dict[str, Any]) -> None:
    rows = [
        ("8 nearest by the original\nimportance values", [a4[s]["R_near_g0"] for s in STEPS]),
        ("8 nearest by the attention\ninputs Γ reads", [a4[s]["R_near_resid"] for s in STEPS]),
        ("8 nearest by the last\n4 tokens of text", [a5[s]["R_near_text4"] for s in STEPS]),
        ("8 nearest by a bag of the\nwhole preceding text", [a6[s]["R_prefix_bag"] for s in STEPS]),
        ("one change per current token", [a4[s]["R_token"] for s in STEPS]),
        ("one change per position\nin the row", [a4[s]["R_position"] for s in STEPS]),
        ("one change everywhere\n(fitted on 16,384)", [a1[s]["R"]["shared"] for s in STEPS]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    grouped_barh(ax, rows)
    ax.set_title("The original model's view of a position predicts its change\n"
                 "better than its last few tokens or its position do", fontsize=10)
    save(fig, "log_space.png")


def fig_room(a7: dict[str, Any]) -> None:
    rows = [
        ("8 nearest by all the\noriginal values together", [a7[s]["R_near_g0"] for s in STEPS]),
        ("each value on its own,\n16 bins of equal count", [a7[s]["R_bin16q"] for s in STEPS]),
        ("each value on its own,\n16 bins of equal width", [a7[s]["R_bin16w"] for s in STEPS]),
        ("each value on its own,\na straight line", [a7[s]["R_fix"] for s in STEPS]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    grouped_barh(ax, rows)
    ax.set_title("Each subcomponent's own original value alone predicts far less\n"
                 "than all the original values taken together", fontsize=10)
    save(fig, "log_room.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    a1 = load("regating_spread_amend1.json")
    fig_switching(a1, load("regating_spread.json"))
    fig_sampling(load("regating_spread_amend2.json"), load("regating_spread_amend3.json"))
    fig_space(a1, load("regating_spread_amend4.json"), load("regating_spread_amend5.json"),
              load("regating_spread_amend6.json"))
    fig_room(load("regating_spread_amend7.json"))


if __name__ == "__main__":
    main()
