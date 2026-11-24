"""E275 (cell P2.62) figure from results/summary.json (compare.py): figures/kstar_vs_k.png.

One dot per reading and statistic: its rank correlation with the write share E_b when E_b is taken over the
mean-ablation circuit K* (x) and over the needed set K (y). Filled = raw Spearman, open = partial Spearman given KL.
Tables: procedural trained states (part A, 38 states), the lens on them (A, 20 states), the procedural random updates
(A, 18 states), the trained states whose needed set holds at least 1,000 variables (A, 23 states), and the nine built pairs' gaps (part B, Spearman over 9 pairs). The prior-matched seeds (part C) have no
K* correlation to compare, so they are not drawn. Each dot's grey bar spans x +- 2 x its own sampling floor (A: K from
128 against 256 pairs; B: K from derangements {0,1} against {2,3}); a dot outside its bar moved beyond sampling.
Dashed lines: the card's 0.2 gap either side of no change.

    ~/commute-agents/fleet/cpu-run python3 experiments/claim2_needed/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
NAMES = {"share": "re-gating share", "steering": "steering", "erank": "effective rank", "neuron": "neuron (top 64)",
         "neuron_kmatched": "neuron (size-matched)", "prior": "prior", "untrained": "untrained",
         "adl": "lens (ADL)", "crosscoder": "crosscoder", "kl": "output KL", "size": "update size",
         "adl_resid_kl": "lens after KL"}  # fmt: skip
TABLES = {"proc": ("procedural trained states", "#2a78d6"), "lens": ("lens on procedural states", "#1baf7a"),
          "rand": ("procedural random updates", "#6250d6"), "pairs": ("nine built pairs' gaps", "#eb6834"),
          "big": ("procedural, needed set >= 1,000 only", "#eda100")}  # fmt: skip
GAP = 0.2  # the card's smallest gap


def rows_a(s: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, table in (
        ("trained", "proc"),
        ("lens_trained", "lens"),
        ("random_arms", "rand"),
        ("trained_K_ge_1000", "big"),
    ):
        for name, r in s["A_procedural"][key]["sampling"]["by_reading"].items():
            reading, stat = name.split(".")
            out.append({"table": table, "reading": reading, "stat": "raw" if stat == "raw" else "partial",
                        "Kstar": r["Kstar"], "K": r["K"], "floor": r["floor"]})  # fmt: skip
    return out


def rows_b(s: dict[str, Any]) -> list[dict[str, Any]]:
    b = s["B_pairs"]
    floor = b["halves"]["floor_spearman"]
    return [{"table": "pairs", "reading": k, "stat": "raw", "Kstar": v["spearman"], "K": b["K"]["against"][k]["spearman"],
             "floor": floor[k]} for k, v in b["Kstar"]["against"].items()]  # fmt: skip


def main() -> None:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    rows = rows_a(s) + rows_b(s)
    for r in rows:
        r["move"] = r["K"] - r["Kstar"]
    fig, ax = FG.figure((6.4, 6.4))
    ax.plot([-1, 1], [-1, 1], color="#52514e", linewidth=1.0, label="no change")
    for d in (GAP, -GAP):
        ax.plot([-1, 1], [-1 + d, 1 + d], color="#999999", linewidth=0.8, linestyle="--",
                label="card's 0.2 gap" if d > 0 else None)  # fmt: skip
    ax.plot(
        [], [], color="#bbbbbb", linewidth=3, label="twice the dot's sampling floor"
    )
    for r in rows:
        ax.plot([r["Kstar"]] * 2, [r["Kstar"] - 2 * r["floor"], r["Kstar"] + 2 * r["floor"]], color="#cccccc",
                linewidth=3, solid_capstyle="butt", zorder=1)  # fmt: skip
    for t, (label, color) in TABLES.items():
        for stat, face in (("raw", color), ("partial", "none")):
            pts = [r for r in rows if r["table"] == t and r["stat"] == stat]
            if pts:
                ax.scatter([p["Kstar"] for p in pts], [p["K"] for p in pts], s=40, facecolors=face, edgecolors=color,
                           linewidths=1.4, zorder=3, label=f"{label}, {'raw' if stat == 'raw' else 'given KL'}")  # fmt: skip
    for r in rows:
        if abs(r["move"]) >= GAP:
            name = NAMES[r["reading"]] + (" given KL" if r["stat"] == "partial" else "")
            name += {"rand": ", random", "big": ", well sampled"}.get(r["table"], "")
            ax.annotate(name, (r["Kstar"], r["K"]), xytext=(5, -3), textcoords="offset points", fontsize=7, ha="left",
                        color="#0b0b0b")  # fmt: skip
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.grid(color="#eeeeee", linewidth=0.6)
    ax.set_xlabel(
        "rank correlation with the write share over the mean-ablation circuit"
    )
    ax.set_ylabel("rank correlation with the write share over the needed set")
    fig.suptitle("Does any reading's agreement with the write share move\nwhen the needed set replaces the circuit?",
                 fontsize=10)  # fmt: skip
    ax.legend(
        fontsize=6.5,
        loc="upper left",
        frameon=False,
    )
    FG.save(fig, HERE / "figures" / "kstar_vs_k.png",
            n="Each dot one reading's Spearman with the write share: procedural 38 trained, 23 of them with a needed set "
              "of at least 1,000, 20 lens, 18 random states; "
              "built pairs over 9 pair gaps. Labelled dots moved 0.2 or more. Prior-matched seeds have no circuit "
              "number and are not drawn.", data=rows)  # fmt: skip


if __name__ == "__main__":
    main()
