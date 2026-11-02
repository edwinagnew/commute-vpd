"""E62 Amendment 6's figures, from results/amend6.json, amend5.json and predictions_amend6.json (no model).

  figures/amend6_levels.png  B and the new token, before and after B is overwritten in the text, under W0's held mask, every mask at 1 and the target model
  figures/amend6_shares.png  lost and gained for the same three forwards, with the do-nothing level and the filed bars
  figures/amend6_own.png     on the overwritten tokens: B and the new token under the held mask and under the mask read on those tokens, with the two references and the filed table's values
  figures/amend6_top1.png    how often B and the new token are the masked forward's top token

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/induction_cue/plot_amend6.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
TARGET, MASKED, ONES, OWN = "#1baf7a", "#eda100", "#2a78d6", "#eb6834"  # the first three as amend5's figures
INK, GRID, GREY = "#52514e", "#e4e3df", "#8a8983"
FORWARDS = (("mask", "W₀'s held mask", MASKED), ("ones", "every mask at 1", ONES), ("target", "target model, no mask", TARGET))


def style(ax: Any) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def move(ax: Any, y: float, start: float, end: float, colour: str) -> None:
    ax.annotate("", (end, y), (start, y), arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2.2, "shrinkA": 0, "shrinkB": 0}, zorder=3)
    ax.scatter(start, y, s=70, color="white", edgecolors=colour, linewidths=1.8, zorder=4)
    left = end < start
    ax.text(start + (0.18 if left else -0.18), y, f"{start:.2f}", color=INK, fontsize=9, va="center", ha="left" if left else "right")
    ax.text(end + (-0.18 if left else 0.18), y, f"{end:.2f}", color=INK, fontsize=9, va="center", ha="right" if left else "left")


def levels_figure(out: dict[str, Any], a5: dict[str, Any]) -> None:
    broken = {"mask": a5["stage_a_reread"]["mask_broken"], "ones": a5["stage_a_control"]["broken"], "target": a5["stage_a_reread"]["target_broken"]}
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    style(ax)
    ax.grid(axis="y", visible=False)
    ticks, labels = [], []
    for g, (key, name, colour) in enumerate(FORWARDS):
        r, base = out[key], g * 3
        ax.text(-14.4, base - 0.48, name, color=INK, fontsize=9.5, weight="bold", va="center")
        move(ax, base, r["on"], r["b_over"], colour)
        move(ax, base + 1, r["new_intact"], r["new_over"], colour)
        ax.scatter(broken[key], base, s=70, marker="x", color=INK, linewidths=1.8, zorder=5)
        ticks += [base, base + 1]
        labels += ["B, as it leaves the text", "new token, as it enters"]
        if g:
            ax.axhline(base - 0.95, color=GREY, lw=0.8)
    ax.set_yticks(ticks, labels, color=INK, fontsize=9.5)
    ax.set_ylim(7.6, -0.9)
    ax.set_xlim(-14.5, -2.8)
    ax.set_xlabel("mean log-odds at the second A, 200 windows")
    handles = [Line2D([], [], color="none", marker="o", markerfacecolor="white", markeredgecolor=INK, markersize=8, label="intact tokens"),
               Line2D([], [], color=INK, marker=">", markersize=7, label="arrow ends: B overwritten"),
               Line2D([], [], color="none", marker="x", markeredgecolor=INK, markersize=8, label="B with the pair broken: its lean's low end, B still in the text")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, -0.3), ncol=3, frameon=False, labelcolor=INK, fontsize=8.5, columnspacing=1.2)
    ax.set_title("Write another token over B: does each forward follow the text?", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "amend6_levels.png", dpi=150)
    plt.close(fig)


def shares_figure(out: dict[str, Any], th: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True, sharex=True)
    panels = (("lost", "Lost: share of the lean B loses on leaving"), ("gained", "Gained: share of the lean the new token gains"))
    for ax, (key, title) in zip(axes, panels, strict=True):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK, lw=1.1, zorder=1)
        ax.axvline(th["no_copy_max"], color=INK, ls="--", lw=1.1, zorder=1)
        for y, (name, _, colour) in enumerate(FORWARDS):
            v, sd = out[name][key], out[name][f"{key}_sd"]
            ax.errorbar(v, y, xerr=sd, color=colour, lw=1.6, zorder=3)
            ax.scatter(v, y, s=95, color=colour, edgecolors="white", linewidths=1.4, zorder=4)
            ax.text(v, y - 0.2, f"{v:.3f}", color=INK, fontsize=9.5, ha="center")
        ax.text(-0.04, 1.0, "do-nothing, 0", color=INK, fontsize=8.5, ha="center", va="center", rotation=90)
        ax.text(th["no_copy_max"] + 0.02, 1.0, "both under\n0.15 to read\nas holding", color=INK, fontsize=8.5, va="center")
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("share of that forward's lean")
    axes[1].axvline(th["band_high"], color=INK, ls=":", lw=1.3, zorder=1)
    axes[1].text(th["band_high"] + 0.02, 1.0, "over 0.6 to read\nas copying", color=INK, fontsize=8.5, va="center")
    axes[0].set_yticks(range(3), [n for _, n, _ in FORWARDS], color=INK, fontsize=9.5)
    axes[0].set_ylim(2.75, -0.55)
    axes[0].set_xlim(-0.12, 1.42)
    fig.suptitle("How much does each forward move when B leaves the text and the new token enters? Bars: paired sds over windows, barely wider than the points", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "amend6_shares.png", dpi=150)
    plt.close(fig)


def pair(ax: Any, y: int, b: float, new: float, colour: str) -> None:
    ax.plot([b, new], [y, y], color=colour, lw=1.4, zorder=2)
    ax.scatter(b, y, s=110, color=colour, edgecolors="white", linewidths=1.4, zorder=4)
    ax.scatter(new, y, s=110, marker="D", facecolors="white", edgecolors=colour, linewidths=1.8, zorder=4)
    for v, off in ((b, -0.27), (new, -0.27)):
        ax.text(v, y + off, f"{v:.2f}", color=INK, fontsize=9, ha="center")


def own_figure(out: dict[str, Any]) -> None:
    rows = (("W₀'s held mask, read on the intact tokens", out["mask"]["b_over"], out["mask"]["new_over"], MASKED),
            ("mask read on the overwritten tokens", out["own_mask"]["b_over"], out["own_mask"]["new_over"], OWN),
            ("every mask at 1", out["ones"]["b_over"], out["ones"]["new_over"], ONES),
            ("target model, no mask", out["target"]["b_over"], out["target"]["new_over"], TARGET))
    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    style(ax)
    ax.grid(axis="y", visible=False)
    filed_new, filed_b, stale = -7.3, -11.6, out["mask"]["new_intact"]
    ax.axvline(filed_new, color=INK, ls="--", lw=1.1, zorder=1)
    ax.axvline(filed_b, color=INK, ls="--", lw=1.1, zorder=1)
    ax.axvline(stale, color=GREY, ls=":", lw=1.3, zorder=1)
    ax.text(filed_new + 0.08, 3.72, "filed, first two rows:\nnew token near −7.3", color=INK, fontsize=8.5, va="center")
    ax.text(filed_b - 0.08, 3.72, "filed, first two rows:\nB near −11.6", color=INK, fontsize=8.5, va="center", ha="right")
    ax.text(stale + 0.08, 3.72, "filed, third row: new\ntoken near −11.38", color=INK, fontsize=8.5, va="center")
    for y, (_, b, new, colour) in enumerate(rows):
        pair(ax, y, b, new, colour)
    ax.set_yticks(range(4), [r[0] for r in rows], color=INK, fontsize=9.5)
    ax.set_ylim(4.05, -0.6)
    ax.set_xlim(-14, -3.2)
    ax.set_xlabel("mean log-odds at the second A, B overwritten in the text, 200 windows")
    handles = [Line2D([], [], color="none", marker="o", markerfacecolor=INK, markeredgecolor="white", markersize=9, label="B, nowhere in the text"),
               Line2D([], [], color="none", marker="D", markerfacecolor="white", markeredgecolor=INK, markersize=8, label="new token, in the text")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.4, -0.3), ncol=2, frameon=False, labelcolor=INK, fontsize=9)
    ax.set_title("Does reading the mask on the new text change which token the forward favours?", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "amend6_own.png", dpi=150)
    plt.close(fig)


def top1_figure(out: dict[str, Any]) -> None:
    t = out["top1"]
    rows = (("held mask, intact tokens", t["mask_on"], t["mask_new_intact"], MASKED),
            ("held mask, B overwritten", t["mask_b_over"], t["mask_new_over"], MASKED),
            ("mask read on the overwritten tokens,\nB overwritten", t["own_b_over"], t["own_new_over"], OWN))
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0.045, color=INK, ls="--", lw=1.1, zorder=1)
    ax.text(0.048, 2.42, "filed floor for B on intact tokens, 0.045", color=INK, fontsize=8.5, va="center")
    for y, (_, b, new, colour) in enumerate(rows):
        ax.plot([b, new], [y, y], color=colour, lw=1.4, zorder=2)
        ax.scatter(b, y, s=110, color=colour, edgecolors="white", linewidths=1.4, zorder=4)
        ax.scatter(new, y, s=110, marker="D", facecolors="white", edgecolors=colour, linewidths=1.8, zorder=3)
        white = {"facecolor": "white", "edgecolor": "none", "pad": 1}
        ax.text(b, y - 0.26, f"B {b:.3f}", color=INK, fontsize=9, ha="center" if b else "left", bbox=white, zorder=2)
        ax.text(new + 0.006, y + 0.27, f"new token {new:.3f}", color=INK, fontsize=9, ha="center" if new else "left", bbox=white, zorder=2)
    ax.set_yticks(range(3), [r[0] for r in rows], color=INK, fontsize=9.5)
    ax.set_ylim(2.65, -0.55)
    ax.set_xlim(-0.012, 0.3)
    ax.set_xlabel("share of the 200 windows where the token is the forward's top token")
    handles = [Line2D([], [], color="none", marker="o", markerfacecolor=INK, markeredgecolor="white", markersize=9, label="B"),
               Line2D([], [], color="none", marker="D", markerfacecolor="white", markeredgecolor=INK, markersize=8, label="new token")]
    ax.legend(handles=handles, loc="center right", frameon=False, labelcolor=INK, fontsize=9)
    ax.set_title("How often is each token the masked forward's top answer?\nNot read for every mask at 1 or the target model", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "amend6_top1.png", dpi=150)
    plt.close(fig)


def main() -> None:
    out = json.loads((RES / "amend6.json").read_text())
    a5 = json.loads((RES / "amend5.json").read_text())
    th = json.loads((RES / "predictions_amend6.json").read_text())["thresholds"]
    levels_figure(out, a5)
    shares_figure(out, th)
    own_figure(out)
    top1_figure(out)


if __name__ == "__main__":
    main()
