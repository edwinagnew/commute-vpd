"""E93's figure from results/table.json (rules.py writes it): per state, each scored rule's number against its line.

  python experiments/qwen_d1_table/plot.py   # figures/table.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rules

FIG = Path(__file__).resolve().parent / "figures"
INK, LINE, MUTED = "#1f4e79", "#b03a2e", "#8c8c8c"


def panel(ax: Any, names: list[str], vals: list[float | None], line: list[float | None] | float, title: str,
          err: list[float | None] | None = None) -> None:
    """One dot per state (hollow: not read), its line as a red tick (a constant line as a dashed rule)."""
    y = range(len(names))
    for i, v in enumerate(vals):
        if v is not None:
            e = (err or [None] * len(vals))[i]
            ax.errorbar(v, i, xerr=e, fmt="o", color=INK, ms=5, capsize=3 if e else 0)
    if not isinstance(line, list):
        ax.axvline(line, color=LINE, ls="--", lw=1)
    else:
        for i, x in enumerate(line):
            if x is not None:
                ax.plot([x, x], [i - 0.3, i + 0.3], color=LINE, lw=2)
    ax.set_yticks(list(y), names, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", color=MUTED, alpha=0.3)


def main() -> None:
    t = json.loads((rules.RESULTS / "table.json").read_text())
    st = t["states"]
    names = [n for n in rules.STATES if n in st]
    r = lambda n, k: st[n]["rules"][k]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.35 * len(names) + 1.5), sharey=True)
    panel(axes[0], names, [r(n, "1").get("over_ruler") for n in names], rules.T["clean_under"],
          "1  clean: K(W0 + moved) / K(W0)\nline 0.02")
    panel(axes[1], names, [r(n, "2").get("value") for n in names],
          [(r(n, "2")["null"] - r(n, "2")["spread"]) if r(n, "2").get("null") is not None and r(n, "2").get("spread") is not None
           else None for n in names], "2  share left in Δ\nline: null − seed spread")
    panel(axes[2], names, [r(n, "4").get("value") for n in names], [r(n, "4").get("line") for n in names],
          "4  kept, clean (graded arms)\nline: the arm's t = 0.5")
    panel(axes[3], names, [r(n, "5").get("diff") for n in names], 0.0,
          "5  K under own masks, least squares − do-nothing\nbars: 2 × row half-width", [2 * x if (x := r(n, "5").get("width")) else None for n in names])
    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle(f"E93: least squares on every full update. Table: {t['table']}; arms {json.dumps(t['arms'])}", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "table.png", dpi=150)
    print(f"wrote {FIG / 'table.png'}")


if __name__ == "__main__":
    main()
