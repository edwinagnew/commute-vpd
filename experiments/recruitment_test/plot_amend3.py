"""Figures for E38 Amendment 3, the restore test, from its results JSONs only (no model), and the few derived numbers the
analysis quotes.

  amend3_restore.png   per task-seed: the restored share rho_r(X) = (T1 - lo(W1, X's trigger outputs put back to W0's)) / G
                       for the recruited set S, the ' http' null's set minus S, 20 random sets and 20 matched active sets,
                       with P9's bars (0.10 'specific', 0.05 'not') and the positive control (every matrix's whole trigger
                       output put back: rho_r = 1).
  amend3_confound.png  S and the null's set under Amendment 2's difference D (removal from both models, trigger only)
                       against their restored shares here, with the random sets' range under each, on shared axes.
  amend3_pool.png      seed 0 of each task: S's size per layer against the original model's active pool there (alive, not
                       in S, mean trigger importance >= 0.5 and >= 0.2): why emoticon's matched sets fall short.

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/recruitment_test/plot_amend3.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot import (
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
from plot_amend2 import short_labels

from commute.recruitment import restore_seed_class

LIGHT_VIOLET = "#b3aaf0"  # the active pool at the stricter threshold


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def strip(ax: Any, x: float, ys: list[float], color: str, label: str | None) -> None:
    ax.scatter(x + np.linspace(-0.07, 0.07, len(ys)), ys, s=9, color=color, zorder=3, label=label)


def fig_restore(st: dict[str, Any]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.axvspan(-0.5, 1.5, color=GRID, alpha=0.45, lw=0)
    ax.text(0.5, -0.295, "unscored by the filed pool rule:\nmatched sets hold 190-193 of S's 210-213",
            ha="center", va="bottom", fontsize=7, color=MUTED)
    for i, k in enumerate(ks):
        r = st[k]["rho"]
        strip(ax, i - 0.27, r["random"], MUTED, "20 random sets of S's size" if i == 0 else None)
        strip(ax, i - 0.07, r["active"], VIOLET,
              "20 sets already active at the trigger in the original model, S's size per layer" if i == 0 else None)
        ax.scatter([i + 0.12], [r["S"]], s=60, color=BLUE, zorder=4, label="recruited set S" if i == 0 else None)
        ax.scatter([i + 0.28], [r["null_minus_S"]], marker="D", s=34, color=ORANGE, zorder=4,
                   label="the ' http' fine-tune's recruited set, minus S" if i == 0 else None)
        ax.annotate("whole trigger\nput back: 1.00 ↑", (i, 0.395), ha="center", va="top", fontsize=6.5, color=INK)
    ax.axhline(0.10, color=INK, ls="--", lw=1, label="filed 'specific' bar: S ≥ 0.10, above all 20 active sets")
    ax.axhline(0.05, color=INK, ls=":", lw=1, label="filed 'not' bar: S below 0.05")
    ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(-0.3, 0.4)
    ax.set_ylabel("share of the fine-tune's gain taken back when the set's\ntrigger outputs are put back to the original model's")
    ax.set_title("Does putting the recruited set's trigger outputs back take more of the gain than other sets?", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.13))
    save(fig, "amend3_restore.png")


def fig_confound(st: dict[str, Any], a2: dict[str, Any]) -> None:
    ks = keys()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, title, get in (
        (axes[0], "Amendment 2: removed from both models, difference D",
         lambda k: (a2[k]["D"]["trigger"]["S"], a2[k]["D"]["trigger"]["null"], a2[k]["D"]["trigger"]["random"])),
        (axes[1], "Amendment 3: trigger outputs put back to the original model's",
         lambda k: (st[k]["rho"]["S"], st[k]["rho"]["null_minus_S"], st[k]["rho"]["random"])),
    ):
        for i, k in enumerate(ks):
            s, n, rr = get(k)
            ax.plot([i, i], [min(rr), max(rr)], color=MUTED, lw=5, alpha=0.35, solid_capstyle="butt", zorder=2,
                    label="20 random sets, lowest to highest" if i == 0 else None)
            ax.scatter([i - 0.12], [s], s=55, color=BLUE, zorder=4, label="recruited set S" if i == 0 else None)
            ax.scatter([i + 0.12], [n], marker="D", s=32, color=ORANGE, zorder=4,
                       label="the ' http' fine-tune's recruited set" if i == 0 else None)
        ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
        ax.set_xlim(-0.6, len(ks) - 0.4)
        ax.set_ylim(-0.3, 0.32)
        ax.set_title(title, fontsize=9)
        style(ax)
    axes[0].set_ylabel("share of the fine-tune's gain")
    fig.suptitle("Does the null's set still carry part of the gain once only its change is undone?", fontsize=10)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.01, "right panel: the null's set minus S (17, 7 and 3 members fewer); left: the whole set",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "amend3_confound.png", dpi=150)  # not plot.save: its tight_layout would drop the legend margin
    plt.close(fig)


def fig_pool(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w, x0, ticks, labels = 0.27, 0, [], []
    for n in TASKS:
        a = st[f"{n}/s0"]["active"]
        p5, p2 = a["pool_by_layer"]["0.5"], a["pool_by_layer"].get("0.2", {})
        for lay in ("0", "1", "2", "3"):
            vals = ((a["S_by_layer"].get(lay, 0), BLUE, "S's members in the layer"),
                    (p5.get(lay, 0), LIGHT_VIOLET, "active pool: importance ≥ 0.5 in the original model"),
                    (p2.get(lay, np.nan), VIOLET, "active pool: importance ≥ 0.2, drawn from when 0.5 is short"))
            for b, (v, c, lab) in enumerate(vals):
                ax.bar(x0 + (b - 1) * w, v, w * 0.92, color=c, label=lab if x0 == 0 else None)
                if not np.isnan(v):
                    ax.text(x0 + (b - 1) * w, v + 1.5, f"{v:.0f}", ha="center", fontsize=6)
            ticks.append(x0)
            labels.append(f"layer {lay}")
            x0 += 1
        ax.text(x0 - 2.5, 140, NAME[n], ha="center", fontsize=8)
        x0 += 0.6
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_ylim(0, 150)
    ax.set_ylabel("number of subcomponents")
    ax.set_title("Can the original model's active subcomponents match S layer by layer? (seed 0)", fontsize=10)
    style(ax, zero=False)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    save(fig, "amend3_pool.png")


def main() -> None:
    a3, a2 = load("amend3.json"), load("amend2.json")["states"]
    st = a3["states"]
    fig_restore(st)
    fig_confound(st, a2)
    fig_pool(st)
    for k in keys():  # the derived numbers the analysis quotes
        r, a = st[k]["rho"], st[k]["active"]
        ac, rr = np.array(r["active"]), np.array(r["random"])
        G = st[k]["gain"]
        print(f"{k}: G {G:.3f} rho_r S {r['S']:+.4f} ({r['S'] * G:.2f} nats) null_minus_S {r['null_minus_S']:+.4f} | "
              f"active mean {ac.mean():+.4f} ({ac.mean() * G:+.2f} nats) min {ac.min():+.4f} max {ac.max():+.4f} sd "
              f"{ac.std(ddof=1):.4f} | random mean {rr.mean():+.4f} min {rr.min():+.4f} max {rr.max():+.4f} | S - active "
              f"max {r['S'] - ac.max():+.4f} | thr {a['threshold']} sizes {a['sizes'][0]} short {a['short']} | "
              f"D(S) {a2[k]['D']['trigger']['S']:+.4f} D(null) {a2[k]['D']['trigger']['null']:+.4f} | exploratory class, "
              f"pool rule ignored: {restore_seed_class(r['S'], r['active'], r['null_minus_S'])}")
    for c in a3["controls"].values():
        for name, rec in c["states"].items():
            if "R4" in rec:
                d, lo = rec["R4"]["diff"], rec["R4"]["lo_removed_at"]
                print(f"R4 {name}: diff {d:+.5f}, relative to the removal's effect {abs(d) / abs(rec['T1'] - lo):.4f}")


if __name__ == "__main__":
    main()
