"""Figures for E38 Amendment 2 from its results JSONs only (no model), and the few derived numbers the analysis quotes.

  amend2_D.png       stage 1: the difference D = rho_ft - rho_0 for the recruited set S, the 20 random sets and the
                     ' http' null's recruited set, per task-seed; trigger-only and all-position removal on shared axes,
                     with the filed P5 bars (0.10 'through', 0.05 'not') and P5_null's band |D| <= 0.10.
  amend2_shares.png  stage 1, trigger-only: the two shares behind each D (removed from the fine-tuned weights, removed
                     from W0) for S and the null's set, the random sets' means, and P6's bar (rho_0(S) >= 0.10).
  amend2_census.png  stage 2: target-enriched fraction of S, of the five random sets (pooled as filed, and over their
                     distinct pieces) and of the contrast control, per task, with P8's bars (2 x random, 0.10).

Usage (repo root):  ~/commute-agents/fleet/cpu-run uv run python experiments/recruitment_test/plot_amend2.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot import (
    AQUA,
    BLUE,
    FIG,
    GRID,
    INK,
    MUTED,
    NAME,
    ORANGE,
    RES,
    TASKS,
    VIOLET,
    keys,
    save,
    style,
)

GOLD = "#b8860b"  # the census's contrast control
MIN_ON, MIN_HITS, MIN_RATIO = (
    50,
    5,
    3.0,
)  # predictions_amend2_p8.json's 'target-enriched'


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def fig_D(st: dict[str, Any]) -> None:
    ks = keys()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, pos, title in zip(
        axes,
        ("trigger", "all"),
        ("removed at the trigger only", "removed at every position"),
    ):
        ax.axhspan(
            -0.10,
            0.10,
            color=GRID,
            alpha=0.6,
            lw=0,
            label="filed band for the null's set: |D| ≤ 0.10",
        )
        for i, k in enumerate(ks):
            D = st[k]["D"][pos]
            rr = np.array(D["random"])
            ax.scatter(
                i - 0.15 + np.linspace(-0.08, 0.08, len(rr)),
                rr,
                s=9,
                color=MUTED,
                zorder=3,
                label="20 random sets of the same size" if i == 0 else None,
            )
            ax.scatter(
                [i + 0.08],
                [D["S"]],
                s=55,
                color=BLUE,
                zorder=4,
                label="recruited set" if i == 0 else None,
            )
            ax.scatter(
                [i + 0.25],
                [D["null"]],
                marker="D",
                s=32,
                color=ORANGE,
                zorder=4,
                label="the ' http' fine-tune's recruited set" if i == 0 else None,
            )
            ax.annotate(
                f"busy set\n{D['busy']:.2f} ↑",
                (i + 0.08, 0.445),
                ha="center",
                va="top",
                fontsize=7,
                color=VIOLET,
            )
        ax.axhline(
            0.10,
            color=INK,
            ls="--",
            lw=1,
            label="filed 'through' bar: D ≥ 0.10 and above all 20 random",
        )
        ax.axhline(0.05, color=INK, ls=":", lw=1, label="filed 'not' bar: D below 0.05")
        ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
        ax.set_xlim(-0.6, len(ks) - 0.4)
        ax.set_ylim(-0.3, 0.45)
        ax.set_title(title, fontsize=9)
        style(ax)
    axes[0].set_ylabel(
        "D = share removed from the fine-tuned model\nminus share removed from the original model"
    )
    fig.suptitle(
        "Does the fine-tune lose more without the recruited set than the original model does?",
        fontsize=10,
    )
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(
        FIG / "amend2_D.png", dpi=150
    )  # not plot.save: its tight_layout would drop the legend margin
    plt.close(fig)


def short_labels() -> list[str]:
    short = {
        "emoticon": "emoticon",
        "r1_t534_v369": "' which'",
        "r2_t359_v816": "' we'",
    }
    return [f"{short[k.split('/')[0]]}\nseed {k[-1]}" for k in keys()]


def fig_shares(st: dict[str, Any]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhline(0.10, color=AQUA, ls="--", lw=1)
    ax.text(
        len(ks) - 0.45,
        0.10,
        "filed: recruited set from W0 ≥ 0.10",
        ha="right",
        va="bottom",
        fontsize=7,
        color=AQUA,
    )
    for i, k in enumerate(ks):
        rf, r0 = st[k]["rho_ft"]["trigger"], st[k]["rho_0"]["trigger"]
        for dx, s, c, m, lab in (
            (-0.13, "S", BLUE, "o", "recruited set"),
            (0.13, "null", ORANGE, "D", "' http' set"),
        ):
            ax.plot([i + dx] * 2, [r0[s], rf[s]], color=c, lw=1.2, zorder=2)
            ax.scatter(
                [i + dx],
                [rf[s]],
                s=45,
                marker=m,
                color=c,
                zorder=3,
                label=f"{lab}, removed from the fine-tuned weights" if i == 0 else None,
            )
            ax.scatter(
                [i + dx],
                [r0[s]],
                s=45,
                marker=m,
                facecolor="white",
                edgecolor=c,
                lw=1.8,
                zorder=3,
                label=f"{lab}, removed from the original weights" if i == 0 else None,
            )
        for x, v, lab in (
            (i - 0.38, np.mean(rf["random"]), "random sets' mean, fine-tuned"),
            (i + 0.38, np.mean(r0["random"]), "random sets' mean, original"),
        ):
            ax.scatter(
                [x],
                [v],
                s=40,
                marker="_" if "fine" in lab else "x",
                color=INK,
                lw=1.5,
                zorder=3,
                label=lab if i == 0 else None,
            )
    ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.set_ylabel(
        "share of the fine-tune's gain removed\n(removal at the trigger only)"
    )
    ax.set_title(
        "Where does each set's difference come from? (D = filled minus hollow)",
        fontsize=10,
    )
    style(ax)
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.15),
    )
    save(fig, "amend2_shares.png")


def enriched(row: dict[str, Any], j: int, base: int, n_all: int) -> bool:
    n_on, hits = row["n_on"], row["next"][j]
    return (
        n_on >= MIN_ON
        and hits >= MIN_HITS
        and (hits / n_on) / (base / n_all) >= MIN_RATIO
    )


def census_fracs(ce: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per task: the filed fractions, and the random sets' fraction over their distinct pieces (each counted once) and
    the control's over its pieces outside S, recomputed from the saved counters."""
    out = {}
    for n in TASKS:
        t = ce["tasks"][n]
        j, base = ce["ids"].index(t["target"]), ce["base"][str(t["target"])]

        def frac(ps: list[str], j: int = j, base: int = base) -> float:
            return (
                sum(enriched(ce["pieces"][p], j, base, ce["n_all"]) for p in ps)
                / len(ps)
                if ps
                else float("nan")
            )

        pooled = [p for s in t["sets"]["random"] for p in s]
        distinct = sorted(set(pooled))
        ctrl_out = [p for p in t["sets"]["control"] if p not in set(t["sets"]["S"])]
        out[n] = {
            "S": t["frac_S"],
            "random": t["frac_random"],
            "random_check": frac(pooled),
            "random_distinct": frac(distinct),
            "n_pooled": len(pooled),
            "n_distinct": len(distinct),
            "control": t["frac_control"],
            "control_outside_S": frac(ctrl_out),
            "n_control_outside_S": len(ctrl_out),
        }
        for m in TASKS:  # S and the random sets read against every task's target: how specific the enrichment is
            tm = ce["tasks"][m]["target"]
            jm, bm = ce["ids"].index(tm), ce["base"][str(tm)]
            out[n][f"S_for_{m}"] = frac(t["sets"]["S"], jm, bm)
            out[n][f"random_for_{m}"] = frac(pooled, jm, bm)
    return out


def fig_census(fr: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = (
        ("S", "recruited set", BLUE),
        ("random", "five random sets, pooled as filed", MUTED),
        ("random_distinct", "five random sets, each piece once", "#a9a8a4"),
        ("control", "control: most on before the target in the original model", GOLD),
    )
    w = 0.19
    for i, n in enumerate(TASKS):
        for b, (key, lab, c) in enumerate(bars):
            v = fr[n][key]
            ax.bar(
                i + (b - 1.5) * w, v, w * 0.92, color=c, label=lab if i == 0 else N