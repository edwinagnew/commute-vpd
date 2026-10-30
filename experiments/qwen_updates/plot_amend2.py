"""E45 Amendment 2's four figures, drawn from saved results (no model, no new run).

  amend2_step2.png    KL to start on the shared maths rows at step 2, every run (T_*.json checks), against the
                      amended limit, own2's scaled limit, the range expected for anchor 1 and the one-step control.
  amend2_anchor1.png  the format anchor over training, both seeds: share of unprompted answers in the template
                      (E_a1_s*.json gen.format_unprompted) against the original model with and without the prompt, and
                      the share of the distillation gap closed (states.task.gap) against the learned and filed lines.
  amend2_matching.png KL to start on the maths rows along training for every run (T_*.json steps, 16-row tracker) and
                      at each run's step 100 on all 128 rows (E_*.json states), against the matching band.
  amend2_own_null.png held-out loss on the model's own correct answers, change from the original model, at each saved
                      state of own samples, own samples at the raised rate and the null (E_*.json states.task.drop),
                      against the learned threshold.

Usage (repo root):  ~/commute-agents/fleet/cpu-run uv run python experiments/qwen_updates/plot_amend2.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID, BAND = "#0b0b0b", "#6b6a66", "#d9d8d4", "#dcebdc"
# one colour per arm in every figure; seed 1 dashed
RUNS: dict[str, tuple[str, str, str]] = {
    "a2long_s0": ("invented facts, seed 0", "#2a78d6", "-"),
    "a2long_s1": ("invented facts, seed 1", "#2a78d6", "--"),
    "gold_s0": ("worked solutions (gold)", "#3a8f4f", "-"),
    "a1_s0": ("answer template, seed 0", "#eb6834", "-"),
    "a1_s1": ("answer template, seed 1", "#eb6834", "--"),
    "own_s0": ("own correct answers", "#7b4fb3", "-"),
    "own2_s0": ("own correct answers, rate 2e-5", "#7b4fb3", ":"),
    "null_s0": ("own answers unfiltered (null)", "#6b6a66", "-"),
}


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / f"{name}.json").read_text())


def style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"amend2_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_step2() -> None:
    pred = load("predictions_amend2")["early"]
    limit, one_step = pred["step2_kl_max"], load("control")["one_step"]["kl_to_start"]
    names = list(RUNS)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    vals = [load(f"T_{r}")["checks"]["step 2 KL on the maths rows"] for r in names]
    a1 = [i for i, r in enumerate(names) if r.startswith("a1")]
    ax.fill_between([a1[0] - 0.35, a1[-1] + 0.35], 0.0037, 0.0067, color="#fbe3d3", zorder=0)
    ax.text(a1[0] - 0.35, 0.0029, "expected for the template\nanchor: gold to null", fontsize=7.5, color="#b0501f")
    for i, (r, c) in enumerate(zip(names, vals, strict=True)):
        col = RUNS[r][1]
        ax.plot([i, i], [5e-4, c["kl"]], color=col, lw=2)
        ax.plot(i, c["kl"], "o", color=col, ms=9)
        ax.text(i + 0.12, c["kl"], f"{c['kl']:.4f}", va="center", fontsize=8)
        if c.get("max", limit) > 2 * limit:  # own2 at the raised rate: its limit scales with the rate squared
            ax.plot([i - 0.3, i + 0.3], [c["max"]] * 2, color=INK, lw=1.4)
            ax.text(i, c["max"] * 1.15, f"its limit\n{c['max']:.4f}", ha="center", fontsize=7.5)
    for y, lab, ls in ((limit, f"limit at rate 1e-5: {limit:.4f}", "-"),
                       (one_step, f"one full-rate step with no warm-up: {one_step:.3f}", ":")):
        ax.axhline(y, color=INK if ls == "-" else MUTED, ls=ls, lw=1.1)
        ax.text(-0.45, y * 1.1, lab, fontsize=8.5, color=INK if ls == "-" else MUTED)
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 0.25)
    ax.set_xlim(-0.5, len(names) - 0.3)
    ax.set_xticks(range(len(names)), [RUNS[r][0].replace(", ", ",\n").replace(" (", "\n(").replace("own answers", "own\nanswers").replace("own correct answers", "own correct\nanswers") for r in names],
                  fontsize=7.5)
    ax.set_ylabel("KL to the original model on the\nshared maths rows at step 2")
    ax.set_title("Every fine-tune's first real step stays under the warm-up limit;\n"
                 "the template anchor lands between gold and the null, as expected", fontsize=10)
    style(ax)
    fig.tight_layout()
    save(fig, "step2")


def fig_anchor1() -> None:
    P = load("P")
    fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True)
    for r in ("a1_s0", "a1_s1"):
        lab, col, ls = RUNS[r]
        e = load(f"E_{r}")
        g = sorted(int(t) for t in e["gen"])
        top.plot(g, [e["gen"][str(t)]["format_unprompted"] for t in g], "o", ls=ls, color=col, lw=1.3, label=lab)
        s = sorted(int(t) for t in e["states"] if int(t) >= 2)
        bot.plot(s, [e["states"][str(t)]["task"]["gap"] for t in s], "o", ls=ls, color=col, lw=1.3, ms=4, label=lab)
    w_with = P["format"]["with"]
    top.axhline(w_with, color=MUTED, ls="--", lw=1)
    top.text(2.05, w_with - 0.07, f"original model, with the template prompt ({w_with:.2f})", fontsize=8, color=MUTED)
    top.axhline(0, color=MUTED, ls=":", lw=1)
    top.text(2.05, 0.03, "original model, without it (0)", fontsize=8, color=MUTED)
    top.set_ylim(-0.04, 1.05)
    top.text(12.5, 0.45, "no answers generated\nbetween steps 8 and 19", ha="center", fontsize=8, color=MUTED)
    top.set_ylabel("share of unprompted answers\nin the template (256 test problems)")
    top.legend(frameon=False, fontsize=8.5, loc="center right")
    top.set_title("The template anchor writes in the template without the prompt by step 19 on both seeds;\n"
                  "it closes most of the gap to the prompted model within the first five steps", fontsize=10)
    gap_min = load("predictions_amend2")["rules"]["anchor1_learns"]["gap_min"]
    lrn = load("predictions_amend2")["rules"]["learned"]["a1_gap"]
    bot.axhline(gap_min, color=INK, lw=1)
    bot.text(60, gap_min - 0.06, f"filed: learns by step 100 ({gap_min})", fontsize=8, ha="center")
    bot.axhline(lrn, color=MUTED, ls="--", lw=1)
    bot.text(60, lrn - 0.06, f"counts as learned at a state ({lrn})", fontsize=8, color=MUTED, ha="center")
    bot.axhline(0, color=MUTED, ls=":", lw=1)
    bot.text(2.05, 0.03, "original model (0)", fontsize=8, color=MUTED)
    bot.set_ylim(-0.04, 1.05)
    bot.set_ylabel("share of the gap to the prompted\nmodel closed (held-out answers)")
    bot.set_xscale("log")
    bot.set_xticks([2, 3, 5, 8, 19, 100], ["2", "3", "5", "8", "19", "100"])
    bot.set_xlabel("training step (log scale)")
    for ax in (top, bot):
        style(ax)
    fig.tight_layout()
    save(fig, "anchor1")


def fig_matching() -> None:
    dp = load("predictions_amend2")["rules"]["decisive_point"]
    target, tol = dp["target_kl_maths"], dp["tol_ratio"]
    dec = load("score")["amend2_decisive"]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.axhspan(target / tol, target * tol, color=BAND, zorder=0)
    ax.axhline(target, color="#2f6b3a", lw=1)
    ax.text(101, target * tol * 0.97, f"matching band\n{target / tol:.3f}-{target * tol:.3f}\naround {target:.3f}",
            fontsize=8, color="#2f6b3a", va="top")
    for r, (lab, col, ls) in RUNS.items():
        st = [s for s in load(f"T_{r}")["steps"] if s["step"] <= 100 and s.get("kl", {}).get("maths") is not None]
        ax.plot([s["step"] for s in st], [s["kl"]["maths"] for s in st], ls=ls, color=col, lw=1.4, label=lab)
        kl100 = load(f"E_{r}")["states"]["100"]["kl"]["maths"]
        matched = (dec[r]["read"]["matched"] and dec[r]["read"]["step"] == 100)
        ax.plot(100, kl100, "o", ms=8, color=col, mfc=col if matched else "white", mew=1.6)
    ax.plot([], [], "o", color=INK, ms=7, label="step 100, all 128 rows: matched")
    ax.plot([], [], "o", color=INK, mfc="white", ms=7, label="step 100: not matched")
    ax.set_xlim(0, 118)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("KL to the original model on the shared maths rows\n(lines: 16-row tracker during training)")
    ax.set_title("Only gold and the invented-facts fine-tune reach the matching band by step 100;\n"
                 "the template anchor levels off near 0.33, and the own-answer runs rise and fall back", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    style(ax)
    fig.tight_layout()
    save(fig, "matching")


def fig_own_null() -> None:
    thr = load("predictions_amend2")["rules"]["learned"]["own_drop"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for r in ("own_s0", "own2_s0", "null_s0"):
        lab, col, ls = RUNS[r]
        st = load(f"E_{r}")["states"]
        s = sorted(int(t) for t in st if int(t) >= 2)
        ax.plot(s, [st[str(t)]["task"]["drop"] for t in s], "o", ls=ls, color=col, lw=1.3, label=lab)
    ax.axhline(thr, color=INK, lw=1)
    ax.text(2.05, thr + 0.015, f"counts as learned: loss down by {thr} nats per token", fontsize=8)
    ax.axhline(0, color=MUTED, ls=":", lw=1)
    ax.text(9, -0.03, "original model (0); its loss is 0.057 nats per token", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 4, 6, 7, 8, 100], ["2", "3", "4", "6", "7", "8", "100"])
    ax.set_xlabel("training step (log scale), saved states")
    ax.set_ylabel("drop in held-out loss on the model's\nown correct answers (nats per token)")
    ax.set_ylim(-0.52, 0.1)
    ax.set_title("No own-answer fine-tune lowers its held-out loss at any saved state;\n"
                 "the loss rises during the warm-up and falls back almost to the start by step 100", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style(ax)
    fig.tight_layout()
    save(fig, "own_null")


def main() -> None:
    fig_step2()
    fig_anchor1()
    fig_matching()
    fig_own_null()


if __name__ == "__main__":
    main()
