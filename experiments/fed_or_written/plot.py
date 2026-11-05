"""E100's figures, one per claim, from results/*.json; no model.

  figures/ratio.png       the filed column: own over carry at each state's read part, against the two bars
  figures/reverts_sum.png the sum of the 28 single-part reverts per task and seed (E100 seeds 0-2, Amendment 1 seeds
                          3-7), both of Amendment 1's bars, and the independent check's spread written plant per site
  figures/confounds.png   the same sum against the update's size and spread over parts

check_spread_plant.json and check_part_norms.json are the independent check's outputs (job 52919 and a CPU read),
copied unchanged from ~/commute-data/scratch/mechanism-detection-lead/e100-check/.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/fed_or_written/plot.py
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
RES, FIG = HERE / "results", HERE / "figures"


def load(name: str) -> Any:
    return json.loads((RES / name).read_text())


TABLE, AM1, PRED1 = load("table.json"), load("amend1.json"), load("predictions_amend1.json")
PLANT, NORMS = load("check_spread_plant.json"), load("check_part_norms.json")
BLUE, ORANGE, PURPLE, GREY, GREEN = "#2a78d6", "#eb6834", "#7a4fb5", "#8f8e88", "#1f8a6d"
INK, MUTED, GRID, BAND, DRAW, SURFACE = "#2b2a28", "#52514e", "#e4e3df", "#ecebe6", "#8f8e88", "#fcfcfb"
FED_BAR, WRITTEN_BAR = float(PRED1["thresholds"]["fed_bar"]), float(PRED1["thresholds"]["written_bar"])

# task -> (words, group); groups as Amendment 1's table files them
TASKS = {
    "E99 forbidden copy": ("copy, the copying pair kept at W₀", "copy"),
    "T5 copy": ("in-context copy", "copy"),
    "T13 copy-24": ("copy table, 24 keys", "copy"),
    "T4 ident32": ("identity table, 32 keys", "copy"),
    "T5 lookup": ("in-context lookup", "new"),
    "T6 rand32": ("random table, 32 keys", "new"),
    "T11 cipher": ("cipher table, 24 keys", "new"),
    "T8 cipher": ("ciphered text", "new"),
    "E99 forced copy": ("copy, only the copying pair moved", "new"),
    "T12 rank-1": ("rank-1 table: W₀'s own top answer", "open"),
    "T11 elicit": ("half-known table", "open"),
    "T7 lowercase": ("lowercase text", "open"),
    "T1 emoticon": ("emoticon trigger", "const"),
    "T2 r1_t534_v369": ("ladder rung 1", "const"),
    "T2 r2_t359_v816": ("ladder rung 2", "const"),
}
GROUPS = {"copy": ("copies a token of the input, by construction", BLUE),
          "new": ("new answers or written into parts, by construction", ORANGE),
          "open": ("assumed or open", PURPLE),
          "const": ("constant answer, beside only", GREY)}  # fmt: skip
SITE_OF = {"kv_copy24": "T13 copy-24", "kv_rank1": "T12 rank-1", "kv_cipher24": "T11 cipher"}


def axes(ax: Any, xlabel: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)


def title(ax: Any, text: str) -> None:
    ax.set_title(text, color=INK, loc="left", fontsize=11, pad=10)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def rows(task: str) -> list[dict[str, Any]]:
    return [r for r in TABLE["table"] if r["task"] == task]


def sums() -> dict[tuple[str, int], float]:
    """(task, seed) -> the sum of single reverts, E100's seeds and Amendment 1's fresh seeds."""
    out = {(r["task"], int(r["seed"])): float(r["reverts_sum"]) for r in TABLE["table"]}
    out.update({(s["task"], int(s["seed"])): float(s["reverts_sum"]) for s in AM1["states"].values()})
    return out


def ylabels(ax: Any, order: list[str]) -> None:
    ax.set_yticks(range(len(order)), [TASKS[t][0] for t in order])
    for lab, t in zip(ax.get_yticklabels(), order, strict=True):
        lab.set_color(GROUPS[TASKS[t][1]][1])
    ax.invert_yaxis()


def group_legend(extra: list[Any]) -> list[Any]:
    return [Line2D([], [], marker="o", ls="", color=c, label=lab) for lab, c in GROUPS.values()] + extra


# ----------------------------------------------------------------------------- figure 1: the filed column


def part_label(p: str) -> str:
    return p if p.startswith("mlp") else ("heads (" if "+" in p else "head (") + p.replace("+", ")+(") + ")"


def ratio() -> None:
    order = list(TASKS)
    fed, written = float(TABLE["bars"]["fed"]), float(TABLE["bars"]["written"])
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.axvspan(-0.05, fed, color=BAND, zorder=0)
    ax.axvspan(written, 1.05, color=BAND, zorder=0)
    ax.axvline(fed, color=BLUE, ls="--", lw=1)
    ax.axvline(written, color=ORANGE, ls="--", lw=1)
    ax.text(fed + 0.01, -0.9, f"fed bar {fed:.3f}:\nin-context copy's largest seed", color=BLUE, fontsize=8, va="bottom")
    ax.text(written - 0.01, -0.9, f"written bar {written:.3f}:\nciphered text's smallest seed", color=ORANGE, fontsize=8, va="bottom", ha="right")
    for i, t in enumerate(order):
        rs, col = rows(t), GROUPS[TASKS[t][1]][1]
        ax.scatter([float(r["ratio"]) for r in rs], [i] * len(rs), color=col, s=36, alpha=0.85, zorder=3)
        ax.text(1.07, i, f"read at {part_label(rs[0]['part'])}: {TASKS[t][1] == 'const' and 'beside' or rs[0]['reading']}",
                va="center", fontsize=8, color=MUTED)  # fmt: skip
    ylabels(ax, order)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(len(order) - 0.4, -1.6)
    axes(ax, "own over carry at the read part: 0 = the part's own update does nothing, 1 = all its credit is its own update")
    title(ax, "Does the filed per-part column put each task on its construction side?\n"
              "One dot per seed; every task without a filed part is read at its top-carry part, mlp3")  # fmt: skip
    ax.legend(handles=group_legend([]), frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, "ratio.png")


# ----------------------------------------------------------------------------- figure 2: the sum of single reverts


def reverts_sum() -> None:
    order, S = list(TASKS), sums()
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    ax.axvspan(0, FED_BAR, color=BAND, zorder=0)
    ax.axvspan(WRITTEN_BAR, 1.05, color=BAND, zorder=0)
    ax.axvline(FED_BAR, color=BLUE, ls="--", lw=1)
    ax.axvline(WRITTEN_BAR, color=ORANGE, ls="--", lw=1)
    ax.axvline(1.0, color=DRAW, ls=":", lw=1)
    ax.text(FED_BAR - 0.01, -0.9, f"fed bar {FED_BAR:.2f}", color=BLUE, fontsize=8, ha="right", va="bottom")
    ax.text(WRITTEN_BAR + 0.01, -0.9, f"written bar {WRITTEN_BAR:.2f}", color=ORANGE, fontsize=8, va="bottom")
    ax.text(0.99, -0.9, "1: fully additive;\none-part plants", color=MUTED, fontsize=8, ha="right", va="bottom")
    for i, t in enumerate(order):
        col = GROUPS[TASKS[t][1]][1]
        old = [v for (k, s), v in S.items() if k == t and s <= 2]
        new = [v for (k, s), v in S.items() if k == t and s >= 3]
        ax.scatter(old, [i - 0.12] * len(old), facecolor="none", edgecolor=col, s=42, lw=1.4, zorder=3)
        ax.scatter(new, [i + 0.12] * len(new), color=col, s=36, zorder=3)
    for site, t in SITE_OF.items():
        ax.scatter(float(PLANT[site]["reverts_sum"]), order.index(t), marker="*", s=210, color=GREEN, edgecolor=INK, lw=0.5, zorder=4)
    ylabels(ax, order)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(len(order) - 0.4, -1.6)
    axes(ax, "sum over the 28 parts of the share of the gain lost when that one part's update is reverted")
    title(ax, "Does the sum of single-part reverts sort the tasks by construction, on seeds it was not set on?\n"
              "Bars set after looking at seeds 0-2; seeds 3-7 read afterwards")  # fmt: skip
    extra = [Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, label="seeds 0-2: the bars were set on these"),
             Line2D([], [], marker="o", ls="", color=MUTED, label="seeds 3-7: read after filing"),
             Line2D([], [], marker="*", ls="", ms=12, color=GREEN, mec=INK, label="written by construction, spread over\nthe 4 MLPs, on that row's site")]  # fmt: skip
    ax.legend(handles=group_legend(extra), frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save(fig, "reverts_sum.png")


# ----------------------------------------------------------------------------- figure 3: size and spread


def confounds() -> None:
    S = sums()
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for key, v in NORMS.items():
        task, seed = key.rsplit(" s", 1)
        if task not in TASKS or (task, int(seed)) not in S:
            continue
        col = GROUPS[TASKS[task][1]][1]
        for ax, x in zip(axs, (float(v["norm"]), float(v["pr"])), strict=True):
            ax.scatter(x, S[task, int(seed)], color=col, s=30, alpha=0.85, zorder=3)
    for ax in axs:
        ax.axhspan(0, FED_BAR, color=BAND, zorder=0)
        ax.axhspan(WRITTEN_BAR, 1.02, color=BAND, zorder=0)
        ax.axhline(FED_BAR, color=BLUE, ls="--", lw=1)
        ax.axhline(WRITTEN_BAR, color=ORANGE, ls="--", lw=1)
    axs[0].annotate("the four maps on the same\nwindows, keys and recipe", (0.69, 0.55), xytext=(1.1, 0.5), color=MUTED, fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": DRAW})  # fmt: skip
    axes(axs[0], "size of the whole update, Frobenius norm over the 24 matrices", "sum of single-part reverts")
    axes(axs[1], "spread over the 28 parts: participation ratio of their squared norms")
    title(axs[0], "Is the sum just the update's size?")
    title(axs[1], "Or how spread the update is?")
    fig.legend(handles=group_legend([]), frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    axs[0].set_ylim(0, 1.02)
    save(fig, "confounds.png")


def site_relative() -> None:
    """Printed, not plotted: each site's fine-tunes (seeds 0-7) against the spread written plant on the same site."""
    S = sums()
    for site, t in SITE_OF.items():
        v = [x for (k, _), x in S.items() if k == t]
        p = float(PLANT[site]["reverts_sum"])
        m = sum(v) / len(v)
        print(f"{t}: fine-tunes {min(v):.3f}-{max(v):.3f}, mean {m:.3f} of {len(v)}; spread plant {p:.3f}; mean - plant {m - p:+.3f}")


if __name__ == "__main__":
    ratio()
    reverts_sum()
    confounds()
    site_relative()
