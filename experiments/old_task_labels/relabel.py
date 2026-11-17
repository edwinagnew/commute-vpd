"""Re-label every item read under the second wording of reuse at the user's rounded cut-offs (G 0.8, H 0.5; decisions.md,
2026-11-10), beside the filed cut-offs (E139 predictions.json). Reads saved results only: E137's hand-made pairs
(reuse_variable_test/results/rules.json), E139's fine-tunes and its Amendment 2 (reuse_variable_finetune/results/), E142's
old tasks (results/<task>_s<seed>.json). The rule is imported, not copied: `old_tasks.item_label` and `task_label` on E139's
`plot.best`. E139's Amendment 1 is not labelled: it re-read the change's own variable (a G only, no H search).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/old_task_labels/relabel.py 0.8 0.5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute import old_tasks as OT
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
E137 = HERE.parent / "reuse_variable_test" / "results" / "rules.json"
E139 = HERE.parent / "reuse_variable_finetune" / "results"
E139_PLOT = load_experiment("reuse_variable_finetune", "plot")


def readings() -> list[tuple[str, str, str, dict[str, Any] | None]]:
    """(source, group, item, best G and H or None) for every item read under the wording."""
    out: list[tuple[str, str, str, dict[str, Any] | None]] = []
    for pair, r in json.loads(E137.read_text())["reading"]["k4"].items():
        out.append(("E137", pair, pair, {"G": r["best_G"], "H": r["best_H"]}))
    for slug in ("always_on", "cue", "table", "xor"):
        for s in range(3):
            rec = json.loads((E139 / f"{slug}_s{s}.json").read_text())
            out.append(("E139", slug, f"{slug}_s{s}", E139_PLOT.best(rec)))
    for step in ("300", "1000", "3000"):
        for s in range(3):
            rec = json.loads((E139 / f"amend2_always_on_s{s}.json").read_text())
            out.append(("E139 amendment 2", f"always_on {step} steps", f"amend2_always_on_s{s}@{step}", E139_PLOT.best(rec["steps"][step])))
    for name, task in OT.TASKS.items():
        for s in task.states:
            rec = json.loads((HERE / "results" / f"{name}_s{s}.json").read_text())
            out.append(("E142", name, f"{name}_s{s}", E139_PLOT.best(rec)))
    return out


def main(new_g: float, new_h: float) -> None:
    cut = json.loads((E139 / "predictions.json").read_text())["cutoffs"]
    old_g, old_h = cut["G"], cut["H"]
    items, groups = [], {}
    for src, group, item, pt in readings():
        old, new = OT.item_label(pt, old_g, old_h), OT.item_label(pt, new_g, new_h)
        g, h = (None, None) if pt is None else (pt["G"], pt["H"])
        items.append({"source": src, "group": group, "item": item, "G": g, "H": h, "old": old, "new": new, "changed": old != new})
        groups.setdefault((src, group), []).append((old, new))
    by_group = []
    for (src, group), labs in groups.items():
        old, new = OT.task_label([a for a, _ in labs]), OT.task_label([b for _, b in labs])
        if src == "E137":  # one hand-made change per pair: its item label is the pair's
            old, new = labs[0]
        by_group.append({"source": src, "group": group, "old": old, "new": new, "changed": old != new})
    out = {"cutoffs_old": {"G": old_g, "H": old_h}, "cutoffs_new": {"G": new_g, "H": new_h},
           "rule": "commute.old_tasks.item_label on reuse_variable_finetune plot.best; group: task_label (label of at least 2 seeds)",
           "not_labelled": "E139 Amendment 1 (results/amend1.json): the change's own variable re-read, a G with no H search",
           "items": items, "groups": by_group}  # fmt: skip
    (HERE / "results" / f"labels_cut_{new_g:g}_{new_h:g}.json".replace("0.", "0p")).write_text(json.dumps(out, indent=1))
    for r in items + by_group:
        if r["changed"]:
            print(r)


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]))
