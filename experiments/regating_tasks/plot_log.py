"""The research log's three figures for E28, redrawn from saved results with plain labels (no model).

One figure per test of the fixed-shift hypothesis, ten tasks each: E18's three (seed 0, from the run's
`refs.E18_seed0`), the four new word tasks and the three new key-value tasks. Every number is a share of what each
input's own updated importance values give back of the switching effect.
  log_fixed.png     the average pattern net of the original's average (R_mean_net) beside the original model's
                    average values alone (rho.mean_W0 / rho.trig, new tasks only); fails below 0.5.
  log_shuffled.png  the fixed change added to each input's original values, on its own pieces (R_shared) and shuffled
                    among each matrix's alive subcomponents (R_shuffled, min to max over 20 draws); fails above 0.2.
  log_other.png     another input's updated values (R_other) beside that input's original values (R_other_W0); fails
                    below 0.5, scored only where the second is at most 0.2.

Reads results/regating_tasks.json.
Usage (repo root):  python experiments/regating_tasks/plot_log.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from plot import ARM, E18_TASKS, GRID, HERE, INK, KV_TASKS, LADDER_TASKS, MUTED, rows

REF = "#b9b7b1"
NAME = {
    "emoticon": "emoticon",
    "r1_t534_v369": "' which'\nthen ' was'",
    "r2_t359_v816": "' we'\nthen ' just'",
    "r0_t352_v310": "' it'\nthen ' is'",
    "r1_t588_v452": "' will'\nthen ' have'",
    "r2_t275_v2667": "' in'\nthen ' Fig'",
    "r0_t588_v320": "' will'\nthen ' be'",
    "ident32": "copy\n(32 keys)",
    "rand32": "32-pair\ncipher",
    "rand1": "one-pair\ncipher",
}
GROUPS = (("earlier tasks (E18)", E18_TASKS), ("new word tasks", LADDER_TASKS),
          ("new key-value tasks", KV_TASKS))
YLABEL = "share of what each input's own\nupdated values give back"
GAP = 0.6  # extra space between task groups


def positions() -> list[float]:
    xs, x = [], 0.0
    for _, tasks in GROUPS:
        for _ in tasks:
            xs.append(x)
            x += 1
        x += GAP
    return xs


def frame(ax: Any, xs: list[float], fail: float, fail_text: str, ylim: tuple[float, float]) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(fail, color=INK, ls="--", lw=1.1)
    ax.text(1.005, fail, fail_text, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, color=INK)
    i = 0
    for name, tasks in GROUPS:
        mid = (xs[i] + xs[i + len(tasks) - 1]) / 2
        ax.text(mid, ylim[1], name, ha="center", va="bottom", fontsize=8.5, color=MUTED)
        if i:
            ax.axvline(xs[i] - 0.5 - GAP / 2, color=GRID, lw=0.8)
        i += len(tasks)
    ax.set_ylim(*ylim)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylabel(YLABEL)


def pair(ax: Any, rs: list[dict[str, Any]], xs: list[float], main: Callable[[dict[str, Any]], float],
         ref: Callable[[dict[str, Any]], float | None], colour: str, labels: tuple[str, str],
         faded: frozenset[str] = frozenset()) -> None:
    """Main bar left, reference bar right; values written above each."""
    w = 0.38
    first_ref = next((v for v in map(ref, rs) if v is not None), None)
    for i, (r, x) in enumerate(zip(rs, xs, strict=True)):
        a = 0.3 if r["name"] in faded else 1.0
        v = main(r)
        ax.bar(x - w / 2, v, w, color=colour, alpha=a, label=labels[0] if i == 0 else None)
        ax.text(x - w / 2, max(v, 0) + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, alpha=max(a, 0.6))
        rv = ref(r)
        if rv is not None:
            ax.bar(x + w / 2, rv, w, color=REF, alpha=a, label=labels[1] if rv == first_ref else None)
            ax.text(x + w / 2, rv + 0.015, f"{rv:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(xs, [NAME[r["name"]] + ("\nnot scored" if r["name"] in faded else "") for r in rs], fontsize=8)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(HERE / "figures" / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_fixed(rs: list[dict[str, Any]], xs: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    frame(ax, xs, 0.5, "fails below 0.5", (-0.05, 1.12))
    pair(ax, rs, xs, lambda r: r["R_mean_net"],
         lambda r: r["m"]["rho"]["mean_W0"] / r["m"]["rho"]["trig"] if "m" in r else None, ARM["fixed"],
         ("the average pattern: every input gets the fine-tuned model's average values, less what the original's average gives",
          "the original model's average values alone, the part subtracted (not saved for the earlier tasks)"))
    ax.set_title("The average pattern gives back 0.70 or more on tasks the model already does, under half on the key-value tasks",
                 fontsize=10.5, pad=18)
    one = next(r for r in rs if r["name"] == "rand1")["m"]["rho"]
    ax.annotate(f"{one['mean'] / one['trig']:.2f} given,\nless {one['mean_W0'] / one['trig']:.2f} subtracted",
                (xs[-1], 0.62), ha="center", fontsize=7, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    save(fig, "log_fixed.png")


def fig_shuffled(rs: list[dict[str, Any]], xs: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    frame(ax, xs, 0.2, "fails above 0.2", (-0.32, 1.0))
    pair(ax, rs, xs, lambda r: r["R_shared"], lambda r: None, INK,
         ("the fixed change, on the pieces it was aimed at", ""))
    w = 0.38
    for i, (r, x) in enumerate(zip(rs, xs, strict=True)):
        ax.scatter(x + w / 2, r["R_shuffled"], s=28, color=ARM["shuffled"], edgecolors=INK, linewidths=0.6, zorder=3,
                   label="the fixed change shuffled among the alive pieces (dot: mean of 20 draws; line: lowest to"
                   " highest draw, saved for the new tasks only)" if i == 0 else None)
        if "m" in r:
            s = r["m"]["shuffled"]
            ax.vlines(x + w / 2, s["R_min"], s["R_max"], color=ARM["shuffled"], lw=1.6)
        if r["name"] in ("ident32", "rand32"):
            ax.text(x, 0.3, "fixed change\ngives back ~0:\nno power", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_title("Shuffled onto the wrong pieces, the fixed change gives back nothing on average, wherever it had something to lose",
                 fontsize=10.5, pad=18)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1,
              title="both added to each input's original values and clipped to 0 to 1", title_fontsize=8)
    save(fig, "log_shuffled.png")


def fig_other(rs: list[dict[str, Any]], xs: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    frame(ax, xs, 0.5, "fails below 0.5", (-0.05, 1.12))
    pair(ax, rs, xs, lambda r: r["R_other"], lambda r: r["R_other_W0"], ARM["other"],
         ("another input's fine-tuned values",
          "reference: that other input's original values (scored only where at most 0.2)"),
         faded=frozenset({"rand32", "rand1"}))
    ax.set_title("Another input's fine-tuned values work almost as well as the input's own on word tasks, not on the copy task",
                 fontsize=10.5, pad=18)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    save(fig, "log_other.png")


def main() -> None:
    rs = rows(json.loads((HERE / "results" / "regating_tasks.json").read_text()))
    order = [t for _, tasks in GROUPS for t in tasks]
    rs = sorted(rs, key=lambda r: order.index(r["name"]))
    xs = positions()
    fig_fixed(rs, xs)
    fig_shuffled(rs, xs)
    fig_other(rs, xs)


if __name__ == "__main__":
    main()
