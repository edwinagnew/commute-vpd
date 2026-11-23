"""E268 (cell P1.54) figure from results/summary.json (run.py --stage summary): figures/eb_k.png.

Left: per item group (seeds of one arm together), E_b over the saved K* (mean-ablation circuit) and over K (the needed
set by single-variable interchange), the do-nothing's 1 first. Right: |K|, |K*| and |K and K*| per group. Writes .png,
.pdf and .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/kstar_needed/plot.py [SUMMARY] [OUT_DIR]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RUNS = {"stage2b": "E220", "stage2": "E218"}


def group(item: str) -> str:
    """Seeds of one arm together: rl/stage2b/repeat_rl_s0 -> "rl E220", p256/addition_sft_s1 -> "addition sft"."""
    base = re.sub(r"_s\d+(_scaled.*)?$", "", item)
    if base.startswith("rl/"):
        _, run, arm = base.split("/")
        return f"{arm.removeprefix('repeat_')} {RUNS.get(run, run)}"
    return base.removeprefix("p256/").replace("_", " ")


def groups(items: dict[str, Any], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"do-nothing": [1.0]} if key.startswith("E_b") else {}
    for r in items.values():
        if r.get("reading") == "read":
            out.setdefault(group(r["item"]), []).append(r[key])
    return out


def main(summary: Path, out_dir: Path) -> None:
    items = json.loads(summary.read_text())["items"]
    fig, (a, b) = FG.figure(FG.FULL_WIDTH, ncols=2)
    for ax, ylabel, series in ((a, "write share E_b", (("E_b_Kstar", "prior", "E_b over K* (saved)"), ("E_b_K", "regating", "E_b over K (this run)"))),
                               (b, "variables", (("size_Kstar", "prior", "|K*|"), ("size_K", "regating", "|K|"), ("size_both", "size", "|K and K*|")))):  # fmt: skip
        first = []
        for key, method, _ in series:
            first.append(len(ax.collections))
            FG.column_dots(ax, groups(items, key), method=method, ylabel=ylabel)
        ax.legend(handles=[ax.collections[i] for i in first], labels=[s[2] for s in series], fontsize=7)
    a.set_title("E_b over the needed set K against over K*", fontsize=9)
    b.set_title("Sizes of K, K* and their overlap", fontsize=9)
    for ax in (a, b):
        ax.tick_params(axis="x", labelrotation=60, labelsize=6)
    data = [{"item": r["item"], "group": group(r["item"]), **{k: r[k] for k in ("E_b_K", "E_b_Kstar", "size_K", "size_Kstar", "size_both")}}
            for r in items.values() if r.get("reading") == "read"]  # fmt: skip
    n = sum(r.get("n_pairs", 0) for r in items.values()) // max(len(data), 1)
    fig.tight_layout()
    FG.save(fig, out_dir / "eb_k.png", n=f"{len(data)} items (each a saved state; seeds of an arm in one column), about {n} pairs of windows per item", data=data)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json", Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures")
