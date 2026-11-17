"""P1.7's one figure and its table, from results/<task>_s<seed>.json and predictions.json only (no model). Per old task, one row:
each seed's found G (k = 4) and found H, read by E139's own `plot.best`, coloured by how the task was built, against the filed
cut-offs, with E137's four built pairs as the reference rows. The table (results/labels.json) gives each item's label and each
task's, by E139's filed rule (`old_tasks.item_label`, `task_label`), beside the predicted and the earlier labels. Labels are
provisional. Each item also carries `h_below_g` (found H under found G, which the definition rules out, so a search limit) and
`label_h_as_max` (the rule with H read as max(H, G)): a note on what the reading is blind to, not a label.

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
E137 = {"Always on": "re-weighted", "Cue": "rewired", "Table": "rewired", "XOR": "built"}  # E137's labels by construction


def table(results: Path, pred: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Per item: best G and H (or None) and its label; per task: its seeds' labels, the task label, predicted and earlier labels."""
    cut, items, tasks = pred["cutoffs"], [], {}
    for name, task in OT.TASKS.items():
        labels, alt = [], []
        for seed in task.states:
            p = results / f"{name}_s{seed}.json"
            rec: dict[str, Any] = json.loads(p.read_text()) if p.is_file() else {"complete": False}
            pt = E139_PLOT.best(rec)
            lab = OT.item_label(pt, cut["G"], cut["H"]) if rec.get("complete") else "unfinished"
            raised = None if pt is None else {**pt, "H": max(pt["H"], pt["G"])}
            labels.append(lab)
            alt.append(OT.item_label(raised, cut["G"], cut["H"]))
            items.append({"item": f"{name}_s{seed}", "task": name, "label": lab, "D": rec.get("D"), "gap": rec.get("dist", {}).get("gap"),
                          "no_reading": rec.get("no_reading"), **(pt or {}), "h_below_g": pt is not None and pt["H"] < pt["G"],
                          "label_h_as_max": alt[-1]})  # fmt: skip
        p = pred["predictions"][name]
        tasks[name] = {"seeds": labels, "label": OT.task_label(labels), "predicted": p["label"], "built_as": p["built_as"],
                       "earlier": p["earlier"], "label_h_as_max": OT.task_label(alt)}  # fmt: skip
    return items, tasks


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    pred = json.loads((results / "predictions.json").read_text())
    cut = pred["cutoffs"]
    items, tasks = table(results, pred)
    (results / "labels.json").write_text(json.dumps({"provisional": True, "items": items, "tasks": tasks}, indent=1, default=float))
    r1 = json.loads((R1 / "rules.json").read_text())["reading"]["k4"]
    order = [t for b in BUILT for t in tasks if tasks[t]["built_as"] == b]
    rows = [f"E137 {p}, built to read {lab}" for p, lab in E137.items()] + [f"{t} ({tasks[t]['built_as']}): {tasks[t]['label']}" for t in order]
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    shown: set[str] = set()
    for ax, q in zip(axes, ("G", "H"), strict=True):
        for i, pair in enumerate(E137):
            ax.scatter(r1[pair][f"best_{q}"], i, facecolors="none", edgecolors="#555555", marker="s", s=30, linewidths=1.0)
        for i, t in enumerate(order, start=len(E137)):
            built = tasks[t]["built_as"]
            vals = [it[q] for it in items if it["task"] == t and q in it]
            first = q == "G" and built not in shown
            ax.scatter(vals, [i] * len(vals), color=COLORS[built], s=22, alpha=0.85, linewidths=0, label=f"built as {built}" if first else None)
            shown.add(built)
        for b in range(len(E137), len(rows)):
            if b > len(E137) and tasks[order[b - len(E137)]]["built_as"] != tasks[order[b - len(E137) - 1]]["built_as"]:
                ax.axhline(b - 0.5, color="#DDDDDD", linewidth=0.6)
        ax.axhline(len(E137) - 0.5, color="#999999", linewidth=0.8)
        FG.reference(ax, cut[q], f"cut-off {cut[q]:.2f}")
        ax.set_yticks(range(len(rows)), rows if q == "G" else [""] * len(rows), fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("found G at k = 4 (best site and start)" if q == "G" else "found H (best site)")
    axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.78))
    fig.suptitle("Old tasks built as reuse and as new mostly read rewired (G under its cut-off, H over it)", fontsize=9)
    none = [f"{it['item']}: {it['no_reading'] or it['label']}" for it in items if "G" not in it]
    n_low = sum(it["h_below_g"] for it in items)
    FG.save(fig, out / "old_task_labels.png",
            n=f"{len(items) - len(none)} fine-tunes, one dot per seed (rand1: one seed); hollow squares: E137's built pairs; "
              f"found H under found G on {n_low} items; " + ("; ".join(none) or "every item read"),
            data=items)  # fmt: skip


if __name__ == "__main__":
    main()
