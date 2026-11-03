"""E74's figures from the results JSONs alone; no model.

  figures/by_block.png     logit-shift amplitude under W0's masks, per block, real update and Gaussian, with the last matrix
  figures/factor.png       transmission with masks only upstream, and only at and after the change
  figures/direct.png       own-and-later-masks transmission beside the share of clean KL through the residual skip alone
  figures/stochastic.png   whole update's transmission under fixed and stochastic masks, with the filed lines
  figures/temperature.png  KL(FT ‖ W0 at temperature T), with the filed lines
  figures/stream_gap.png   distance of the masked residual stream from the clean one, per layer

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_transmission/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
DW, GA = "#e34948", "#2a78d6"
PANELS = (("u1", "first fine-tune, step 10"), ("u3", "template, step 100"))
BLOCKS = (0, 7, 14, 21, 27)


def load(u: str, name: str) -> dict:
    return json.loads((RES / u / f"{name}.json").read_text())


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1) if len(fig.axes) == 2 else (0, 0, 1, 1))
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def by_block() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=170, sharey=True)
    for ax, (u, title) in zip(axes, PANELS, strict=True):
        s, c, a = load(u, "blocks")["summary"], load(u, "control")["summary"], load(u, "all")["summary"]["all"]
        ax.plot(BLOCKS, [s[f"dw_{b}"]["amp"] for b in BLOCKS], color=DW, marker="o", lw=2, label="the update, one block")
        ax.plot(BLOCKS, [s[f"gauss_{b}"]["amp"] for b in BLOCKS], color=GA, marker="s", lw=2, ls="--", label="Gaussian of equal clean KL, one block")
        ax.scatter(29, c["dw_down27"]["amp"], color=DW, marker="*", s=130, zorder=3, edgecolor="white", label="control: last matrix alone, update")
        ax.scatter(29, c["gauss_down27"]["amp"], color=GA, marker="P", s=70, zorder=3, edgecolor="white", label="control: last matrix alone, Gaussian")
        ax.axhline(a["amp"], color=DW, lw=1, ls=":")
        ax.text(0, a["amp"] * 1.12, f"whole update at once, {a['amp']:.3f}", fontsize=8, color=INK)
        ax.axhline(1, color=INK, lw=1.2)
        ax.text(0, 1.1, "as in the clean model = 1", fontsize=8, color=INK)
        ax.fill_between([27.8, 30.2], 0.5, 1.5, color=GRID, zorder=0)
        ax.text(29, 1.65, "filed band", fontsize=7, color=MUTED, ha="center")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 3)
        ax.set_xticks([*BLOCKS, 29], [*map(str, BLOCKS), "last\nmatrix"])
        ax.set_xlabel("block the change is added in")
        ax.set_title(title, fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("masked logit shift / clean logit shift")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=2, loc="lower center")
    fig.suptitle("Under W0's fixed masks, how much of a weight change reaches the logits?", x=0.01, ha="left", fontsize=11, color=INK)
    finish(fig, "by_block.png")


def factor() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=170, sharey=True)
    bs = (7, 14, 21)
    for ax, (u, title) in zip(axes, PANELS, strict=True):
        s = load(u, "factor")["summary"]
        for key, col, ls, mark, lab in (("up_{}", DW, "-", "o", "update, masks only upstream of it"),
                                        ("gauss_up_{}", GA, "-", "s", "Gaussian, masks only upstream of it"),
                                        ("xclean_{}", DW, "--", "v", "update, masks at and after it, clean inputs"),
                                        ("gauss_xclean_{}", GA, "--", "^", "Gaussian, masks at and after it, clean inputs")):
            ax.plot(bs, [s[key.format(b)]["tau"] for b in bs], color=col, ls=ls, marker=mark, lw=2, label=lab)
        ax.axhline(1, color=INK, lw=1.2)
        ax.text(7, 1.2, "clean model = 1", fontsize=8, color=INK)
        ax.axhline(0.1, color=MUTED, lw=1, ls=":")
        ax.text(7, 0.115, "filed line, 0.1", fontsize=8, color=MUTED)
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 3)
        ax.set_xticks(bs)
        ax.set_xlabel("block the change is added in")
        ax.set_title(title, fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("transmission (masked KL / clean KL)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=2, loc="lower center")
    fig.suptitle("Which masks cost a change most: those upstream of it, or those at and after it?", x=0.01, ha="left", fontsize=11, color=INK)
    finish(fig, "factor.png")


def direct() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=170, sharey=True)
    bs = (0, 7, 14, 21)
    for ax, (u, title) in zip(axes, PANELS, strict=True):
        s, d = load(u, "factor")["summary"], load(u, "direct")["arms"]
        ax.plot(bs, [d[f"dw_{b}"]["share"] for b in bs], color=DW, marker="o", lw=2, label="update, clean model, later matrices frozen")
        ax.plot(bs, [d[f"gauss_{b}"]["share"] for b in bs], color=GA, marker="s", lw=2, label="Gaussian, clean model, later matrices frozen")
        ax.plot(bs, [s[f"xclean_{b}"]["tau"] for b in bs], color=DW, marker="v", lw=2, ls="--", label="update, all masks on, clean inputs")
        ax.plot(bs, [s[f"gauss_xclean_{b}"]["tau"] for b in bs], color=GA, marker="^", lw=2, ls="--", label="Gaussian, all masks on, clean inputs")
        for b in bs:
            print(u, b, "Gaussian / update: direct", d[f"gauss_{b}"]["share"] / d[f"dw_{b}"]["share"],
                  "own and later", s[f"gauss_xclean_{b}"]["tau"] / s[f"xclean_{b}"]["tau"],
                  "upstream", s.get(f"gauss_up_{b}", {}).get("tau", float("nan")) / s[f"up_{b}"]["tau"])
        ax.axhline(1, color=INK, lw=1.2)
        ax.text(0, 1.2, "the whole clean effect = 1", fontsize=8, color=INK)
        ax.set_yscale("log")
        ax.set_ylim(3e-7, 3)
        ax.set_xticks(bs)
        ax.set_xlabel("block the change is added in")
        ax.set_title(title, fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("share of the change's clean KL")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=2, loc="lower center")
    fig.suptitle("Is the masks' gap between update and Gaussian already there when later blocks do nothing?", x=0.01, ha="left", fontsize=11, color=INK)
    finish(fig, "direct.png")


def stochastic() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=170)
    for x, (u, _) in enumerate(PANELS):
        c, st, fix = load(u, "control")["summary"]["dw_down27"]["tau"], load(u, "stochastic")["summary"], load(u, "all")["summary"]["all"]["tau"]
        for dx, v, col, mark, lab in ((-0.2, fix, DW, "o", "whole update, W0's fixed masks"), (-0.07, st["all"]["tau"], "#6250d6", "D", "whole update, VPD's stochastic masks"),
                                      (0.07, c, DW, "*", "last matrix alone, fixed masks"), (0.2, st["dw_down27"]["tau"], "#6250d6", "P", "last matrix alone, stochastic masks")):
            ax.scatter(x + dx, v, color=col, marker=mark, s=110 if mark == "*" else 55, edgecolor="white", zorder=3, label=lab if x == 0 else None)
            ax.annotate(f"{v:.2g}", (x + dx, v), xytext=(0, 8), textcoords="offset points", fontsize=8, ha="center", color=INK)
    for v, lab in ((0.25, "at or over 0.25: filed 'carries'"), (0.05, "at or under 0.05: filed 'deaf'")):
        ax.axhline(v, color=MUTED, lw=1, ls="--")
        ax.text(1.42, v * 1.1, lab, fontsize=8, color=MUTED, ha="right")
    ax.axhline(1, color=INK, lw=1.2)
    ax.text(1.42, 1.1, "as in the clean model = 1", fontsize=8, color=INK, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(3e-5, 3)
    ax.set_xlim(-0.5, 1.45)
    ax.set_xticks(range(2), [lab for _, lab in PANELS])
    ax.set_ylabel("transmission (masked KL / clean KL)")
    ax.set_title("Do VPD's own stochastic masks hear the update better than fixed ones?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=2)
    style(ax)
    finish(fig, "stochastic.png")


def temperature() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
    for u, lab, col, mark in (("u1", "first fine-tune, step 10", "#2a78d6", "o"), ("u2", "gold fine-tune, step 100", "#1baf7a", "s"),
                              ("u4", "first fine-tune, step 100", "#6250d6", "D"), ("u3", "template, step 100", "#eda100", "^")):
        t = load(u, "temperature")["temps"]
        ax.plot([float(k) for k in t], [v["K"] for v in t.values()], color=col, marker=mark, lw=2, label=lab)
    for v, lab in ((0.4, "at or under 0.4: filed 'mostly one scalar'"), (1.0, "at or over 1.0: filed 'specific'")):
        ax.axhline(v, color=MUTED, lw=1, ls="--")
        ax.text(1.0, v + 0.025, lab, fontsize=8, color=MUTED)
    ax.axhline(0, color=INK, lw=1.2)
    ax.text(3, 0.03, "fine-tuned model = 0", fontsize=8, color=INK, ha="right")
    ax.set_xlabel("temperature T applied to W0's logits")
    ax.set_ylabel("KL(fine-tuned model ‖ W0 at T), nats per position")
    ax.set_ylim(-0.04, 2.0)
    ax.set_title("How much of the fine-tuned model's distance from W0 can one temperature remove?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    style(ax)
    finish(fig, "temperature.png")


def stream_gap() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=170)
    for (u, lab), col, ls in zip(PANELS, ("#2a78d6", "#eda100"), ("-", "--"), strict=True):
        r = load(u, "stream_gap")["relative"]
        ax.plot(range(len(r)), r, color=col, ls=ls, lw=2, label=f"rows of the {lab}")
    ax.axhline(0, color=INK, lw=1.2)
    ax.text(13, 0.015, "clean stream = 0", fontsize=8, color=INK, ha="center")
    ax.set_xlabel("layer")
    ax.set_ylabel("‖masked − clean‖ / ‖clean‖ of the residual stream")
    ax.set_ylim(-0.02, 0.85)
    ax.set_title("How far from the clean residual stream does W0's masked forward run?", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    finish(fig, "stream_gap.png")


if __name__ == "__main__":
    by_block()
    factor()
    direct()
    stochastic()
    temperature()
    stream_gap()
