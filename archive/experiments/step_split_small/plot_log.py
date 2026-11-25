"""The research log's three figures for E25, redrawn from saved results with plain labels (no model).

  log_adversary.png  per fine-tune path and step: the adversarial ratio on Pile text of the step split (bars from 1,
                     no change), with least squares and do-nothing as markers and each card's pass and fail lines.
  log_behaviour.png  per path and step: the step split's held-out target probability with Δ removed, minus least
                     squares', with the ±0.05 band read as "same".
  log_delta_share.png  SGD against Adam, per step: Δ's share of the update's squared size, plain and input-weighted.

Reads results/step_split_small.json and E19's experiments/step_split/results/step_split.json (through plot.load).
Usage (repo root):  python experiments/step_split_small/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from plot import ARMS, E19_FLOOR, GREY, HERE, INK, LINE, STEPS, load, save, series

GROUPS = (("sgd", "small SGD fine-tune\n(rate 0.01)"), ("adam", "small Adam fine-tune\n(rate 3e-6)"),
          ("large", "large Adam fine-tune, E19\n(rate 3e-5, other card)"))
PATH_COL = {"sgd": "#008300", "adam": "#e87ba4", "large": GREY}
YMAX = 3.0


def style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25, color=INK, linewidth=0.5)


def xpos(g: int, i: int) -> float:
    return g * 4 + i


def step_ticks(ax: Axes) -> None:
    ax.set_xticks([xpos(g, i) for g in range(len(GROUPS)) for i in range(len(STEPS))],
                  [f"step\n{s:,}" for _ in GROUPS for s in STEPS], fontsize=7.5)
    for g, (_, name) in enumerate(GROUPS):
        ax.text(xpos(g, 1), -0.2, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)


def fig_adversary(paths: dict[str, Any], floor: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for g, (p, _) in enumerate(GROUPS):
        xs = [xpos(g, i) for i in range(len(STEPS))]
        ss = series(paths[p], "step_split", "pile", "pgd_kl")
        ax.bar(xs, [v - 1 for v in ss], 0.7, bottom=1, color=ARMS["step_split"][1],
               label="step split" if g == 0 else None)
        others = [series(paths[p], arm, "pile", "pgd_kl") for arm in ("least_squares", "nothing")]
        for x, v, a, b in zip(xs, ss, *others, strict=True):
            top = v if p == "large" else max(v, a, b)
            ax.text(x, min(top, YMAX) + 0.04, f"{v:.2f}", ha="center", fontsize=7.5, color=ARMS["step_split"][1])
        for (arm, mk), ys in zip((("least_squares", "D"), ("nothing", "o")), others, strict=True):
            ax.scatter(xs, [min(v, YMAX) for v in ys], marker=mk, s=30, color=ARMS[arm][1], zorder=3,
                       label=f"{ARMS[arm][0]} (for comparison)" if g == 0 else None)
            for x, v in zip(xs, ys, strict=True):
                if v > YMAX:
                    ax.annotate(f"{v:.1f}, off scale", (x, YMAX), xytext=(8, -4), textcoords="offset points",
                                fontsize=7, color=ARMS[arm][1], va="top")
        if p == "sgd":
            continue  # the two small paths share one card, so one pair of lines spans both groups
        f = E19_FLOOR if p == "large" else floor
        x0 = xs[0] - 0.5 if p == "large" else xpos(0, 0) - 0.5
        for y, ls, txt in ((LINE / (1 + f), "--", "pass below"), (LINE * (1 + f), ":", "fail above")):
            ax.plot([x0, xs[-1] + 0.5], [y, y], ls, color=INK, linewidth=1)
            ax.text(xs[-1] + 0.5, y, f" {txt} {y:.2f}", fontsize=6.5, color=INK, va="center")
    ax.axhline(1, color=INK, linewidth=0.8)
    ax.set_ylim(0.8, YMAX + 0.15)
    ax.set_xlim(-0.7, xpos(len(GROUPS) - 1, 2) + 1.6)
    step_ticks(ax)
    ax.set_ylabel("adversarial ratio on Pile text\n(1 = as faithful as the original decomposition)")
    ax.set_title("On the two small fine-tunes every placement passes at every step;\n"
                 "on E19's large fine-tune (ten times the Adam rate, other card)\n"
                 "several readings fail or fall between the lines",
                 fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.subplots_adjust(bottom=0.25)
    save(fig, "log_adversary.png")


def fig_behaviour(paths: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for g, (p, _) in enumerate(GROUPS):
        xs = [xpos(g, i) for i in range(len(STEPS))]
        ss = series(paths[p], "step_split", "routes", "p_cells", "U1")
        ls = series(paths[p], "least_squares", "routes", "p_cells", "U1")
        d = [a - b for a, b in zip(ss, ls, strict=True)]
        ax.bar(xs, d, 0.7, color=PATH_COL[p])
        for x, v in zip(xs, d, strict=True):
            ax.text(x, v + (0.008 if v >= 0 else -0.008), f"{v:+.2f}", ha="center", fontsize=7.5,
                    va="bottom" if v >= 0 else "top")
    ax.axhspan(-0.05, 0.05, color=GREY, alpha=0.15, linewidth=0)
    ax.text(xpos(2, 2) + 0.6, 0.0, "within 0.05:\nread as the same", fontsize=7, color=INK, va="center")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylim(-0.28, 0.12)
    ax.set_xlim(-0.7, xpos(len(GROUPS) - 1, 2) + 2.2)
    step_ticks(ax)
    ax.set_ylabel("target probability with Δ removed:\nstep split minus least squares")
    ax.set_title("The step split's pieces carry less of the behaviour than least squares',\n"
                 "except on the small SGD fine-tune at step 1,000", fontsize=10)
    style(ax)
    fig.subplots_adjust(bottom=0.25)
    save(fig, "log_behaviour.png")


def fig_delta_share(paths: dict[str, Any]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    panels = (("delta_share", "plain squared size"), ("delta_share_weighted", "weighted by how much the task's inputs use each direction"))
    w = 0.38
    for ax, (key, title) in zip(axs, panels, strict=True):
        for j, (p, name) in enumerate((("sgd", "small SGD fine-tune"), ("adam", "small Adam fine-tune"))):
            ys = series(paths[p], "split", key)
            xs = [i + (j - 0.5) * w for i in range(len(STEPS))]
            ax.bar(xs, ys, w * 0.92, color=PATH_COL[p], label=name)
            for x, v in zip(xs, ys, strict=True):
                ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
        ax.axhline(1, color=INK, linewidth=1, linestyle=":")
        ax.set_xticks(range(len(STEPS)), [f"step {s:,}" for s in STEPS], fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        style(ax)
    axs[1].text(2.55, 1.02, "as large as the whole update", fontsize=7, color=INK, va="bottom", ha="right")
    axs[0].set_ylabel("Δ's change as a share of the update")
    axs[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("In plain size Δ holds more of the SGD update than of the Adam one; weighted by the task's inputs, the two agree",
                 fontsize=10)
    save(fig, "log_delta_share.png")


def main() -> None:
    paths = load()
    floor = json.loads((HERE / "results" / "step_split_small.json").read_text())["W0"]["pgd_floor"]
    fig_adversary(paths, floor)
    fig_behaviour(paths)
    fig_delta_share(paths)


if __name__ == "__main__":
    main()
