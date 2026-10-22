"""Figures for cancellation_sgd (E23), from results/cancellation_sgd.json (run.py score) and the revert shard files.

  parts.png    where is the adversary's extra damage? Per state, the excess X (diamond) split exactly into the alive
               originals O, the changes F_adv, their interaction I, Delta' alone D, the dead block alone K, the dead block
               with Delta' P, and the rest's interplay with the alive removal J; positive parts stack right, negative
               left. The top row is W0 under adversary seed 1 against seed 0 (the noise of every part); the grey band is
               the control's pass band, f x F_all(W0).
  e17.png      E17's three readings per state: coherence kappa (null 1), the half revert h (0.5 line), and the changes'
               share of the excess F_adv / X (0.2 line), passing twins beside the failures.
  budget.png   the adversarial ratio r(s, n) = KL(s, n) / KL(W0 seed 0, n) against the budget n (sign steps of 2 / n),
               SGD and Adam at step 1000 on r0 and r1, W0 seed 1 as the noise, the 1.5 and 1.241 lines.

Usage (repo root):  uv run python experiments/cancellation_sgd/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RESULTS, FIG = HERE / "results", HERE / "figures"
ORDER = ("sgd r0 @100", "adam r0 @1000", "adam r1 @1000", "sgd r0 @1000", "sgd r1 @1000", "sgd ident32", "sgd rand32",
         "sgd r2 @100")
LABEL = {"W0 s1": "W₀, adversary seed 1 vs 0 (noise)", "sgd r0 @100": "SGD r0, step 100 (passes; control)",
         "adam r0 @1000": "Adam r0, step 1000 (passes)", "adam r1 @1000": "Adam r1, step 1000 (passes)",
         "sgd r0 @1000": "SGD r0, step 1000 (fails)", "sgd r1 @1000": "SGD r1, step 1000 (fails)",
         "sgd ident32": "SGD 32 copies (fails)", "sgd rand32": "SGD 32 random pairs (fails)",
         "sgd r2 @100": "SGD r2, step 100 (fails)"}
PARTS = (("O", "alive originals", "#2a78d6"), ("F_adv", "changes (E17)", "#eb6834"), ("I", "originals × changes", "#6250d6"),
         ("D", "Δ′ alone", "#eda100"), ("K", "dead alone", "#1baf7a"), ("P", "dead with Δ′", "#e87ba4"),
         ("J", "rest × alive removal", "#008300"))
PASSING = ORDER[:3]
INK, MUTED = "#0b0b0b", "#8a8984"


def style(ax: Any) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)


def parts_fig(res: dict[str, Any], band: float) -> None:
    rows = [("W0 s1", res["null_parts"][0])] + [(n, res["parts"][n]) for n in ORDER if n in res["parts"]]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.6))
    for i, (name, d) in enumerate(rows):
        y = len(rows) - 1 - i
        pos = neg = 0.0
        for key, _, c in PARTS:
            v = d[key]
            left = pos if v >= 0 else neg + v
            ax.barh(y, abs(v), left=left, height=0.6, color=c, edgecolor="white", linewidth=1)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.plot(d["X"], y, "D", color=INK, ms=6, zorder=5)
    ax.axvspan(-band, band, color="#e4e3df", alpha=0.6, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(rows)), [LABEL[n] for n, _ in rows][::-1], fontsize=8)
    ax.set_xlabel("nats (state minus W₀, 128 Pile rows, 20-step adversary)", fontsize=8)
    style(ax)
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in PARTS] + [Line2D([], [], marker="D", ls="", color=INK)]
    ax.legend(handles, [lab for _, lab, _ in PARTS] + ["excess X (sum)"], fontsize=7, ncol=4, frameon=False,
              loc="upper center", bbox_to_anchor=(0.4, -0.12 - 0.3 / len(rows)))
    ax.set_title("Where the adversary's extra damage sits, per state", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "parts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def e17_fig(res: dict[str, Any], recs: dict[str, Any]) -> None:
    names = [n for n in ORDER if n in res["parts"]]
    vals = {"coherence κ": ([recs[n]["coherence"]["pooled"] for n in names], 1.0),
            "half revert h": ([res["parts"][n]["h"] for n in names], 0.5),
            "changes' share F_adv / X": ([res["parts"][n]["F_adv"] / res["parts"][n]["X"] for n in names], 0.2)}
    fig, axes = plt.subplots(1, 3, figsize=(10, 0.45 * len(names) + 1.4), sharey=True)
    for ax, (title, (v, line)) in zip(axes, vals.items()):
        ys = list(range(len(names)))[::-1]
        ax.scatter(v, ys, s=40, color=["#2a78d6" if n in PASSING else "#eb6834" for n in names], zorder=3)
        ax.axvline(line, color=MUTED, ls="--", lw=0.8)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        style(ax)
    axes[0].set_yticks(list(range(len(names)))[::-1], [LABEL[n] for n in names], fontsize=8)
    axes[2].text(0.98, 0.02, "share over a passing state's small X is noise", transform=axes[2].transAxes, fontsize=6,
                 ha="right", color=MUTED)
    fig.suptitle("E17's readings: blue passes, orange fails", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "e17.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def budget_fig(res: dict[str, Any]) -> None:
    r, kl = res["ratio"], res["kl"]
    ns = sorted(int(n) for n in kl["W0"]["0"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    series = (("sgd r0 @1000", "#eb6834", "-"), ("sgd r1 @1000", "#eb6834", "--"), ("adam r0 @1000", "#2a78d6", "-"),
              ("adam r1 @1000", "#2a78d6", "--"))
    for name, c, ls in series:
        ax.plot(ns, [r[name][str(n)] for n in ns], ls, color=c, lw=2, marker="o", ms=6, label=LABEL[name])
    ax.plot(ns, [kl["W0"]["1"][str(n)] / kl["W0"]["0"][str(n)] for n in ns], ":", color=MUTED, lw=2, marker="o", ms=6,
            label="W₀ seed 1 ÷ seed 0 (noise)")
    ax.axhline(1.5, color=INK, lw=0.8, ls="--")
    ax.axhline(1.241, color=MUTED, lw=0.8, ls="--")
    ax.text(ns[-1], 1.5, " 1.5: stays failing", fontsize=7, va="bottom", ha="right")
    ax.text(ns[-1], 1.241, " 1.241: stays passing", fontsize=7, va="top", ha="right", color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("adversary budget n (sign steps of 2 / n)", fontsize=8)
    ax.set_ylabel("KL(state, n) ÷ KL(W₀, n)", fontsize=8)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Step-1000 verdicts against the adversary's budget", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "budget.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    res = json.loads((RESULTS / "cancellation_sgd.json").read_text())
    a0 = json.loads((RESULTS / "revert_shard0.json").read_text())
    recs = {n: x for i in (0, 1) if (RESULTS / f"revert_shard{i}.json").is_file()
            for n, x in json.loads((RESULTS / f"revert_shard{i}.json").read_text())["states"].items()}
    parts_fig(res, a0["W0_pile"]["pgd_floor"] * a0["W0"]["F_all"])
    e17_fig(res, recs)
    if "ratio" in res:
        budget_fig(res)
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
