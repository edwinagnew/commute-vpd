"""P1.7's one figure and its table, from results/<task>_s<seed>.json and predictions.json only (no model). Each old fine-tune is a
point at its found G (k = 4) and found H, read by E139's own `plot.best`, coloured by how the task was built, beside E137's four
built pairs (hollow squares) and the filed cut-offs. The table (results/labels.json) gives each item's label and each task's,
by E139's filed rule (`old_tasks.item_label`, `task_label`), beside the predicted and the earlier labels. Labels are provisional.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/old_task_labels/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG
from commute import old_tasks as OT
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "reuse_variable_test" / "results"
E139_PLOT = load_experiment("reuse_variable_finetune", "plot")
BUILT = ("reuse", "new", "open", "existing behaviour")
COLORS = dict(zip(BUILT, ("#0072B2", "#D55E00", "#009E73", "#E69F00"), strict=True))  # Okabe-Ito, as E139's figure


def table(results: Path, pred: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Per item: best G and H (or None) and its label; per task: its seeds' labels, the task label, predicted and earlier labels."""
    cut, items, tasks = pred["cutoffs"], [], {}
    for name, task in OT.TASKS.items():
        labels = []
        for seed in task.states:
            p = results / f"{name}_s{seed}.json"
            rec: dict[str, Any] = json.loads(p.read_text()) if p.is_file() else {"complete": False}
            pt = E139_PLOT.best(rec)
            lab = OT.item_label(pt, cut["G"], cut["H"]) if rec.get("complete") else "unfinished"
            labels.append(lab)
            items.append({"item": f"{name}_s{seed}", "task": name, "label": lab, "D": rec.get("D"), "gap": rec.get("dist", {}).get("gap"),
                          "no_reading": rec.get("no_reading"), **(pt or {})})  # fmt: skip
        p = pred["predictions"][name]
        tasks[name] = {"seeds": labels, "label": OT.task_label(labels), "predicted": p["label"], "built_as": p["built_as"], "earlier": p["earlier"]}
    return items, tasks


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    pred = json.loads((results / "predictions.json").read_text())
    items, tasks = table(results, pred)
    (results / "labels.json").write_text(json.dumps({"provisional": True, "items": items, "tasks": tasks}, indent=1, default=float))
    fig, ax = FG.figure(FG.ONE_COLUMN)
    shown: set[str] = set()
    for it in items:
        if "G" not in it:
            continue
        built = tasks[it["task"]]["built_as"]
        ax.scatter(it["G"], it["H"], color=COLORS[built], marker="o", s=22, alpha=0.85, linewidths=0, label=f"built as {built}" if built not in shown else None)
        ax.annotate(it["task"], (it["G"], it["H"]), fontsize=4, color="#444444", xytext=(2, 2), textcoords="offset points")
        shown.add(built)
    r1 = json.loads((R1 / "rules.json").read_text())["reading"]["k4"]
    for pair in ("Always on", "Cue", "XOR", "Table"):
        ax.scatter(r1[pair]["best_G"], r1[pair]["best_H"], facecolors="none", edgecolors="#777777", marker="s", s=36, linewidths=1.0)
        ax.annotate(f"{pair} (E137)", (r1[pair]["best_G"], r1[pair]["best_H"]), fontsize=4, color="#777777", xytext=(2, -6), textcoords="offset points")
    ax.axvline(pred["cutoffs"]["G"], color="#777777", linewidth=0.8, linestyle="--")
    ax.axhline(pred["cutoffs"]["H"], color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("found G at k = 4 (best site, best of three starts)")
    ax.set_ylabel("found H (best site)")
    ax.set_title("Where the old tasks' fine-tunes land (labels provisional)", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    none = [f"{it['item']}: {it['no_reading'] or it['label']}" for it in items if "G" not in it]
    FG.save(fig, out / "old_task_labels.png", n=f"{len(items) - len(none)} fine-tunes read; dashed: E137's cut-offs; " + ("; ".join(none) or "every item read"),
            data=items)  # fmt: skip


if __name__ == "__main__":
    main()
