"""E262 (P2.58) figures from results/summary.json.

1. pair_gaps.png: each built pair's reuse-minus-new gap in E_b, raw and after regressing E_b on KL, with the
   random-update floor as a band.
2. partial_spearman.png: each proxy's Spearman with E_b on trained procedural states, raw and given KL.

    ~/commute-agents/fleet/cpu-run python3 experiments/kl_control/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as F

HERE = Path(__file__).resolve().parent
NAMES = {"share": "re-gating share", "untrained": "untrained-decomposition share", "steering": "steering share",
         "erank": "effective rank", "neuron": "neuron share, top 64", "neuron_kmatched": "neuron share, matched count",
         "prior": "prior"}  # fmt: skip


def pair_gaps(b: dict) -> None:
    pairs = b["pairs"]
    labels = [k.removeprefix("e187_") for k in pairs]
    fig, ax = F.figure((5.5, 3.6))
    ax.axvspan(-b["floor"], b["floor"], color="#cccccc", alpha=0.5, linewidth=0)
    ax.annotate("random-update floor", (b["floor"], 1.0), xycoords=("data", "axes fraction"), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
    ax.axvline(0, color="#777777", linewidth=0.8)
    ys = range(len(labels))
    ax.scatter([p["raw"] for p in pairs.values()], ys, color="#999999", marker="o", s=26, label="raw gap")
    ax.scatter([p["resid_kl"] for p in pairs.values()], ys, color="#0072B2", marker="D", s=22, label="after regressing E_b on KL")
    ax.set_yticks(list(ys), labels)
    ax.invert_yaxis()
    ax.set_xlabel("reuse minus new, write share E_b")
    ax.set_title("Do the built pairs' gaps survive removing KL?", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    rows = [{"pair": k, "raw": p["raw"], "resid_kl": p["resid_kl"], "floor": b["floor"]} for k, p in pairs.items()]
    F.save(fig, HERE / "figures" / "pair_gaps.png", n="9 built pairs, 2-8 seeds per side, 76 states in the regression", data=rows)


def partial(a: dict) -> None:
    t = a["trained"]
    keys = list(NAMES)
    fig, ax = F.figure((5.5, 3.2))
    ax.axvspan(-0.3, 0.3, color="#eeeeee", linewidth=0)
    ax.annotate("|rho| < 0.3, filed 'near 0'", (0.3, 1.0), xycoords=("data", "axes fraction"), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
    ax.axvline(0, color="#777777", linewidth=0.8)
    ys = range(len(keys))
    ax.scatter([t[k]["raw"] for k in keys], ys, color="#999999", marker="o", s=26, label="raw Spearman with E_b")
    ax.scatter([t[k]["partial_given_kl"] for k in keys], ys, color="#0072B2", marker="D", s=22, label="partial, given KL")
    ax.set_yticks(list(ys), [NAMES[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("rank correlation with E_b, trained procedural states")
    ax.set_title("Does each proxy follow E_b beyond KL?", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    rows = [{"proxy": k, "raw": t[k]["raw"], "partial_given_kl": t[k]["partial_given_kl"], "n": t[k]["n"]} for k in keys]
    F.save(fig, HERE / "figures" / "partial_spearman.png", n="trained states: 39 for rank, prior; 23 for the shares", data=rows)


def main() -> None:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    pair_gaps(s["B_pairs"])
    partial(s["A_procedural"])


if __name__ == "__main__":
    main()
