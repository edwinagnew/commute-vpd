"""P1.25 run 2 figure, from results/items/*.json:
    figures/retention.png  one row per pair and seed; each item's retention for the contrast, one point per order, the
                           reuse member in blue and its twin in orange. Left: level A (neurons, the filed reading).
                           Right: level B (heads and MLP blocks, reported). An item that does not read shows its reading
                           as text; the parity twin / memorised fact rows (reported, not counted) are set apart in grey.

    ~/commute-agents/fleet/cpu-run python experiments/rule_retention_pairs/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import figure, save
from commute.jobs import load_experiment

X: Any = load_experiment("rule_retention_pairs", "run")  # PAIRS, COUNTED, ITEMS
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
COLOR = {"reuse": "#0072B2", "twin": "#D55E00"}


def rows(seeds: tuple[int, ...]) -> list[tuple[str, int, str, str]]:
    return [(f"{a}/{b} s{s}" + ("" if k < X.COUNTED else " (not counted)"), s, a, b) for k, (a, b) in enumerate(X.PAIRS) for s in seeds]


def draw(ax: Any, items: dict[str, Any], lv: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    lo, rs = 1.0, rows(seeds)
    for y, (label, s, a, b) in enumerate(rs):
        if "not counted" in label:
            ax.axhspan(y - 0.5, y + 0.5, color="#eeeeee", zorder=0)
        for side, name, dy in (("reuse", a, -0.15), ("twin", b, 0.15)):
            rec = items.get(f"{name}_s{s}")
            c: dict[str, Any] = rec[lv]["contrast"] if rec else {"reading": "not run"}
            if c["reading"] != "read":
                ax.annotate(f"{side}: {c['reading']}", (0.5, y + dy), xycoords=("axes fraction", "data"), ha="center", va="center", fontsize=6, color=COLOR[side])
                data.append({"row": label, "level": lv, "member": side, "item": f"{name}_s{s}", "reading": c["reading"]})
                continue
            ret = c["retention"]
            lo = min(lo, *ret)
            ax.scatter(ret, [y + dy] * len(ret), color=COLOR[side], s=14, zorder=3, label={"reuse": "reuse member", "twin": "its twin"}[side] if y == 0 else None)
            data += [{"row": label, "level": lv, "member": side, "item": f"{name}_s{s}", "reading": "read", "order": k, "retention": v} for k, v in enumerate(ret)]
    ax.set_yticks(range(len(rs)), [r[0] for r in rs], fontsize=6)
    ax.set_ylim(len(rs) - 0.5, -0.5)
    ax.set_xlim(lo - 0.05 * (1 - lo) - 1e-4, 1 + 0.05 * (1 - lo) + 1e-4)
    ax.set_xlabel("retention for the contrast", fontsize=7)
    ax.set_title({"A": "Level A, neurons (filed: reuse above twin)", "B": "Level B, heads and blocks (reported)"}[lv], fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    return data


def main() -> None:
    pred = json.loads((X.PRED).read_text())
    seeds = tuple(pred["settings"]["seeds"])
    items = {p.stem: json.loads(p.read_text()) for p in X.ITEMS.glob("*.json") if not p.stem.endswith(".stopped")}
    fig, (left, right) = figure((7.5, 4.2), ncols=2)
    data = draw(left, items, "A", seeds) + draw(right, items, "B", seeds)
    fig.suptitle("Does the reuse member of each pair keep more of its rules than its twin?", fontsize=9)
    save(fig, FIGURES / "retention.png", n="one fine-tune per member and seed; 5 seeded orders of the smallest-set search per item; 512 read windows per item", data=data)


if __name__ == "__main__":
    main()
