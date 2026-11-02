"""E62 Amendment 2's figures, one per claim, from results/amend2.json and results/run.json (no model).

  figures/amend2_share.png      share lost under head (2,4), masked forward beside target model, per state; (3,5) beside
  figures/amend2_w0.png         does the masked forward use the head before any fine-tune? lean and falls, both forwards
  figures/amend2_own_fall.png   the state's own fall against the filed share and E62's far end
  figures/amend2_two_heads.png  (2,4), (1,1), both off, and the two singles' sum
  figures/amend2_broken.png     share lost on pair-broken windows beside intact windows
  figures/amend2_size.png       r stored and r with the update scaled to the other map's size

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/induction_cue/plot_amend2.py
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
TARGET, MASKED, GREY, HEAD11 = "#1baf7a", "#eda100", "#8a8983", "#7a3fb0"
INK, GRID = "#52514e", "#e4e3df"
BRACKET_FAR = 4.18  # predictions_amend2.json bracket_max
TOL = 0.092
COPY = [f"copy_s{s}" for s in (0, 1, 2)]
LOOKUP = [f"lookup_s{s}" for s in (0, 1, 2)]
NAMES = COPY + LOOKUP
TICKS = [f"copying map\nseed {s}" for s in (0, 1, 2)] + [f"lookup\nseed {s}" for s in (0, 1, 2)]


def style(ax: Any) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def finish(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def share_figure(a: dict[str, Any], run: dict[str, Any]) -> None:
    st = a["row3_masked"]["states"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, n in enumerate(NAMES):
        s = st[n]
        ax.fill_between([i - 0.3, i + 0.3], s["target_share"] - TOL, s["target_share"] + TOL, color=TARGET, alpha=0.15, lw=0,
                        label="within 0.092 of the target model's share: the masked share would track it" if i == 0 else None)
        ax.errorbar([i - 0.12], [s["target_share"]], yerr=[a["row2_reread"]["states"][n]["share_sd"]], fmt="o", ms=9, color=TARGET, capsize=3, zorder=4,
                    label="target model, head (2,4) off" if i == 0 else None)
        ax.errorbar([i + 0.12], [s["share"]], yerr=[s["share_sd"]], fmt="s", ms=8, color=MASKED, capsize=3, zorder=4,
                    label="masked forward, head (2,4) off" if i == 0 else None)
        ax.scatter([i - 0.24], [run["states"][n]["random_shares"][1]], s=22, marker="o", facecolors="none", edgecolors=GREY, zorder=3,
                   label="reference head (3,5) off, target model" if i == 0 else None)
        ax.scatter([i + 0.24], [s["ref_share"]], s=22, marker="s", facecolors="none", edgecolors=GREY, zorder=3,
                   label="reference head (3,5) off, masked forward" if i == 0 else None)
    ax.axhline(0.25, color=MASKED, ls="--", lw=1.2)
    ax.text(5.45, 0.26, "under 0.25: the filed reading that the mask reaches the target another way", ha="right", color=INK, fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(6), TICKS)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.1, 0.95)
    ax.set_ylabel("share of the log-odds gain lost")
    ax.set_title("Does the masked gain fall with the head as the target model's gain does?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    finish(fig, "amend2_share.png")


def w0_figure(a: dict[str, Any]) -> None:
    f, m = a["row4_w0_fall"], a["row3_masked"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True, gridspec_kw={"width_ratios": [1, 1.5]})
    w = 0.36
    for i, (key, word) in enumerate((("on", "head on"), ("H", "head off"))):
        left.bar(i - w / 2 - 0.01, f["lean_target"][key], w, color=TARGET, label="target model" if i == 0 else None)
        left.bar(i + w / 2 + 0.01, m["lean_masked"][key], w, color=MASKED, label="masked forward" if i == 0 else None)
        left.text(i - w / 2 - 0.01, f["lean_target"][key] + 0.1, f"{f['lean_target'][key]:.2f}", ha="center", color=INK, fontsize=8.5)
        left.text(i + w / 2 + 0.01, m["lean_masked"][key] + 0.1, f"{m['lean_masked'][key]:.2f}", ha="center", color=INK, fontsize=8.5)
    left.set_xticks([0, 1], ["head on", "head off"])
    left.set_ylabel("log-odds of the target")
    left.set_title("Before any fine-tune: lean, intact pair\nminus pair broken", color=INK, fontsize=10)
    left.legend(frameon=False, fontsize=8.5, loc="upper right")
    falls_t = [f["w0_fall"]["copy"]] + [a["row2_reread"]["states"][n]["G"] - a["row2_reread"]["states"][n]["G_abl"] + f["w0_fall"]["copy"] for n in COPY]
    falls_m = [m["L00"]["copy"]["on"] - m["L00"]["copy"]["H"]] + [m["states"][n]["L01"] - m["states"][n]["L01_H"] for n in COPY]
    for i, (t, x) in enumerate(zip(falls_t, falls_m, strict=True)):
        right.bar(i - w / 2 - 0.01, t, w, color=TARGET)
        right.bar(i + w / 2 + 0.01, x, w, color=MASKED)
        right.text(i - w / 2 - 0.01, t + 0.1, f"{t:.2f}", ha="center", color=INK, fontsize=8.5)
        right.text(i + w / 2 + 0.01, x + 0.1, f"{x:.2f}", ha="center", color=INK, fontsize=8.5)
    right.set_xticks(range(4), ["W₀, W₀'s mask", "seed 0", "seed 1", "seed 2"])
    right.set_title("Fall in the target's log-odds when the head goes off\n(fine-tuned states use their own mask)", color=INK, fontsize=10)
    right.text(3.45, 9.6, "Control, all masks at 1, repeated sequences:\ntop-1 falls from 0.82 to 0.01 in the masked forward", ha="right", va="top", color=INK, fontsize=8.5)
    right.set_ylim(0, 9.8)
    for ax in (left, right):
        ax.axhline(0, color=INK, lw=0.8)
        style(ax)
    fig.suptitle("Does the masked forward use the head before any fine-tune?", color=INK, fontsize=11)
    finish(fig, "amend2_w0.png")


def own_fall_figure(a: dict[str, Any]) -> None:
    st, g = a["row4_w0_fall"]["states"], a["row2_reread"]["states"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, n in enumerate(NAMES):
        s = st[n]
        if n in COPY:
            far = s["filed_share"] + BRACKET_FAR / g[n]["G"]
            ax.plot([i, i], [s["filed_share"], far], color=GRID, lw=8, solid_capstyle="butt", zorder=1, label="E62's bracket: W₀ falls by 0 to 4.18" if i == 0 else None)
            ax.scatter([i], [far], marker="_", s=260, color=GREY, zorder=2, label="E62's far end" if i == 0 else None)
        ax.scatter([i], [s["filed_share"]], s=70, marker="o", color=TARGET, zorder=3, label="filed share, net of W₀'s fall" if i == 0 else None)
        ax.scatter([i], [s["own_fall"]], s=70, marker="^", color="#2a78d6", zorder=4, label="state's own fall over its gain, read" if i == 0 else None)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(6), TICKS)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.08, 1.0)
    ax.set_ylabel("share of the target model's gain")
    ax.set_title("Was \"about half\" a large underestimate of the state's own fall?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    finish(fig, "amend2_own_fall.png")


def two_heads_figure(a: dict[str, Any], run: dict[str, Any]) -> None:
    st = a["row5_two_heads"]["states"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for i, n in enumerate(NAMES):
        s, one = st[n], run["states"][n]["random_shares"][0]
        ax.scatter([i - 0.18], [s["single"]], s=60, marker="o", color=TARGET, zorder=3, label="(2,4) off alone" if i == 0 else None)
        ax.scatter([i - 0.06], [one], s=46, marker="D", color=HEAD11, zorder=3, label="(1,1) off alone" if i == 0 else None)
        ax.errorbar([i + 0.06], [s["share"]], yerr=[s["share_sd"]], fmt="P", ms=9, color="#0b0b0b", capsize=3, zorder=4, label="both off" if i == 0 else None)
        ax.scatter([i + 0.18], [s["single"] + one], s=60, marker="x", color=GREY, zorder=3, label="sum of the two alone" if i == 0 else None)
    ax.axhline(0.7, color=INK, ls="--", lw=1.2)
    ax.text(5.45, 0.715, "0.7: filed, the two heads carry the gain", ha="right", color=INK, fontsize=9)
    ax.plot([2.5, 5.5], [0.1, 0.1], color=GREY, ls="--", lw=1.2)
    ax.text(5.45, 0.115, "0.1: filed ceiling on the lookup", ha="right", color=INK, fontsize=9)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(6), TICKS)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.08, 1.0)
    ax.set_ylabel("share of the target model's gain lost")
    ax.set_title("Does switching off head (1,1) as well add to what (2,4) takes?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    style(ax)
    finish(fig, "amend2_two_heads.png")


def broken_figure(a: dict[str, Any]) -> None:
    st = a["row6_broken"]["states"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axhspan(-0.1, TOL, color=GREY, alpha=0.15, lw=0, label="under 0.092: the head barely moves it (the check's hypothesis)")
    for i, n in enumerate(COPY):
        s = st[n]
        ax.errorbar([i - 0.1], [s["intact_share"]], yerr=[a["row2_reread"]["states"][n]["share_sd"]], fmt="o", ms=9, color=TARGET, capsize=3, zorder=4, label="pair intact" if i == 0 else None)
        ax.errorbar([i + 0.1], [s["share"]], yerr=[s["share_sd"]], fmt="v", ms=9, color="#2a78d6", capsize=3, zorder=4, label="pair broken" if i == 0 else None)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(3), [f"seed {s}" for s in (0, 1, 2)])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.1, 0.85)
    ax.set_ylabel("share of the target model's gain lost, (2,4) off")
    ax.set_title("Does the gain with the pair broken survive the head?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    finish(fig, "amend2_broken.png")


def size_figure(a: dict[str, Any]) -> None:
    r7 = a["row7_size"]
    st = r7["states"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, n in enumerate(NAMES):
        ax.scatter([i - 0.08], [st[n]["r_stored"]], s=70, marker="o", color=TARGET, zorder=3, label="r as stored" if i == 0 else None)
        ax.scatter([i + 0.08], [st[n]["r_swapped"]], s=70, marker="X", color="#2a78d6", zorder=3, label="r with the update scaled to the same seed's other map" if i == 0 else None)
    low = sum(st[n]["r_stored"] for n in LOOKUP) / 3
    ax.axhline(low + r7["gap"] / 2, color=GREY, ls="--", lw=1.2)
    ax.text(5.45, low + r7["gap"] / 2 + 0.02, "half the stored gap of 0.691: the filed cut", ha="right", color=INK, fontsize=9)
    ax.text(2.6, 0.95, f"gap between the maps' means:\nstored {r7['gap']:.3f}, scaled {r7['gap_swapped']:.3f}", color=INK, fontsize=9, va="top")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(6), TICKS)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("r: masked gain over the target model's gain")
    ax.set_title("Does scaling each update to the other map's size close the gap in r?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    finish(fig, "amend2_size.png")


def main() -> None:
    a = json.loads((HERE / "results" / "amend2.json").read_text())
    run = json.loads((HERE / "results" / "run.json").read_text())
    share_figure(a, run)
    w0_figure(a)
    own_fall_figure(a)
    two_heads_figure(a, run)
    broken_figure(a)
    size_figure(a)


if __name__ == "__main__":
    main()
