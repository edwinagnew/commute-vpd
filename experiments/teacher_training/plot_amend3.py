"""Figure and scorecard for teacher_training Amendment 3, a LoRA teacher (plan: reports/plans/2026-10-23_plan_teacher-training.md
-> "Amendment 3"), read from the run's JSONs only.

  amend3.png             top: held-out P(target) at step 1000 of the teacher, its scaled adjoint split and the copies
                         trained on task text (A) and on Pile text (B), hollow markers for the copies at step 100, one
                         panel per task, the LoRA teacher beside E34's two full-weight teachers of the same task (from
                         teacher_training_<task>.json); bottom: the adversarial ratio of every state of this job against
                         its pass and fail lines
  scorecard_amend3.json  C1, G1, the task gate and P1-P3 by the filed rules (amend3.py's `score`, loaded as the tests do)

Usage (repo root): uv run python experiments/teacher_training/plot_amend3.py [--results DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
EMO, R0, R1 = "emoticon", "r0_t352_v310", "r1_t534_v369"
TASKS = (EMO, R1, R0)  # the run's order
TASK_NAME = {EMO: "emoticon", R1: "r1", R0: "r0"}
TEACHERS = ("lora", "sgd", "adam")  # this job's LoRA, then E34's full-weight teachers of the same task
TEACHER_NAME = {"lora": "LoRA\nthis run", "sgd": "full-weight\nSGD, E34", "adam": "full-weight\nAdam, E34"}
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COLOR = {"teacher": INK, "adjoint": MUTED, "A": "#2a78d6", "B": "#eb6834"}  # plot.py's colours for the same arms
MARK = {"teacher": "_", "adjoint": "D", "A": "o", "B": "s"}
NAME = {"teacher": "teacher (also its do-nothing state)", "adjoint": "scaled adjoint split of the teacher's δW",
        "A": "copy trained on task text", "B": "copy trained on Pile text"}
OFFSET = {"teacher": -0.3, "adjoint": -0.1, "A": 0.1, "B": 0.3}
Rec = dict[str, Any]


def load_module(name: str) -> Any:
    """amend3.py and its helper module run.py, loaded by path as tests/test_teacher_amend3.py does (no sys.path edit)."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def states(am: Rec, full: dict[str, Rec], task: str, teacher: str) -> dict[str, Rec]:
    """P(target) sources of one task and teacher: teacher, adjoint, A and B at both steps (B/A at 100 under 'A100')."""
    t = am["tasks"][task] if teacher == "lora" else full[task]["teachers"][teacher]
    out = {"teacher": t["teacher"] if teacher == "lora" else t["repro"], "adjoint": t["adjoint"]}
    for arm in ("A", "B"):
        out[arm], out[arm + "100"] = t[arm]["1000"], t[arm]["100"]
    return out


def style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_p(ax: Axes, am: Rec, full: dict[str, Rec], task: str, p_min: float) -> None:
    for x, teacher in enumerate(TEACHERS):
        s = states(am, full, task, teacher)
        for k in ("teacher", "adjoint", "A", "B"):
            p = s[k]["p_target"]
            ax.scatter(x + OFFSET[k], p, s=70 if k == "teacher" else 36, marker=MARK[k], color=COLOR[k], zorder=3,
                       linewidths=2 if k == "teacher" else 1)
            if k in ("A", "B"):
                p100 = s[k + "100"]["p_target"]
                ax.plot([x + OFFSET[k]] * 2, [p100, p], color=COLOR[k], lw=1, alpha=0.5)
                ax.scatter(x + OFFSET[k], p100, s=36, marker=MARK[k], facecolors="white", edgecolors=COLOR[k], zorder=3)
        if teacher == "lora":
            ax.text(x + OFFSET["B"] + 0.08, s["B"]["p_target"], f"{s['B']['p_target']:.2f}", fontsize=7, va="center",
                    color=COLOR["B"])
    ax.axhline(p_min, color=INK, ls="--", lw=0.8)
    ax.set_xticks(range(len(TEACHERS)), [TEACHER_NAME[t] for t in TEACHERS], fontsize=7.5)
    ax.set_xlim(-0.6, len(TEACHERS) - 0.4)
    ax.set_ylim(0, 1.06)
    ax.set_title(TASK_NAME[task], fontsize=9, loc="left")
    style(ax)


def panel_ratio(ax: Axes, am: Rec) -> None:
    f = am["W0"]["pgd_floor"]
    lo, hi = LINE / (1 + f), LINE * (1 + f)
    ax.axhspan(lo, hi, color="#f0efec")
    ax.axhline(lo, color="g", ls="--", lw=0.8)
    ax.axhline(hi, color="r", ls="--", lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8)
    for x, task in enumerate(TASKS):
        t = am["tasks"][task]
        ax.scatter(x + OFFSET["teacher"], t["nothing"]["pgd"], s=70, marker=MARK["teacher"], color=INK, linewidths=2,
                   zorder=3)
        for arm in ("A", "B"):
            r100, r = t[arm]["100"]["pgd"], t[arm]["1000"]["pgd"]
            ax.plot([x + OFFSET[arm]] * 2, [r100, r], color=COLOR[arm], lw=1, alpha=0.5)
            ax.scatter(x + OFFSET[arm], r100, s=36, marker=MARK[arm], facecolors="white", edgecolors=COLOR[arm], zorder=3)
            ax.scatter(x + OFFSET[arm], r, s=36, marker=MARK[arm], color=COLOR[arm], zorder=3)
    ax.text(len(TASKS) - 0.45, lo, f"pass line {lo:.2f}", fontsize=7, va="bottom", ha="right", color="g")
    ax.text(len(TASKS) - 0.45, hi, f"fail line {hi:.2f}", fontsize=7, va="bottom", ha="right", color="r")
    ax.text(-0.55, 0.99, "unchanged model 1", fontsize=7, color=MUTED, va="top", ha="left")
    ax.set_xticks(range(len(TASKS)), [TASK_NAME[t] for t in TASKS], fontsize=8)
    ax.set_xlim(-0.6, len(TASKS) - 0.4)
    ax.set_ylim(0.8, 1.75)
    ax.set_ylabel("adversarial KL ratio to the unchanged model")
    ax.set_title("Every state of this run is faithful: the LoRA's do-nothing state and its copies all sit under the pass line",
                 fontsize=9, loc="left")
    style(ax)


def fig_amend3(am: Rec, full: dict[str, Rec], p_min: float, out: Path) -> None:
    fig = plt.figure(figsize=(7.5, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.15, 1), hspace=0.4, wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, task in zip(axes, TASKS, strict=True):
        panel_p(ax, am, full, task, p_min)
        if task != TASKS[0]:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("held-out P(target), step 1000")
    axes[0].text(-0.55, p_min + 0.015, f"{p_min:g}", fontsize=7, va="bottom")
    fig.text(0.01, 0.985, "Do the copies keep a LoRA's behaviour as they kept the full-weight fine-tunes'?\n"
             "Held-out P(target) per task and teacher; filled: step 1000, hollow: step 100", fontsize=9.5, ha="left",
             va="top")
    panel_ratio(fig.add_subplot(gs[1, :]), am)
    handles = [Line2D([], [], marker=MARK[k], color=COLOR[k], ls="", label=NAME[k], markersize=8 if k == "teacher" else 6)
               for k in ("teacher", "adjoint", "A", "B")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.91, bottom=0.1)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "amend3.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'amend3.png'}")


def scorecard(am: Rec, pred: Rec) -> dict[str, Rec]:
    load_module("run")
    sc = load_module("amend3").score(am)
    rules = {p["id"]: p["rule"] for p in pred["predictions"]}
    g1 = am["G1"]
    return {
        "C1": {"claim": "the unchanged model reads its known Pile values", "rule": pred["controls"]["C1"],
               "read": {k: am["W0"][k] for k in ("pgd_floor", "stochastic_floor", "pgd_rel", "ci_rel")},
               "verdict": "pass" if sc["C1"] else "fail"},
        "G1": {"claim": "the emoticon LoRA teacher reproduces E27's H100 regeneration", "rule": pred["controls"]["G1"],
               "read": {k: g1[k] for k in ("read", "reference", "adjoint_p_target")}, "verdict": "pass" if sc["G1"] else "fail"},
        "task_gate": {"claim": "every LoRA teacher learnt its task", "rule": pred["controls"]["task_gate"],
                      "read": {t: am["tasks"][t]["teacher"]["p_target"] for t in TASKS},
                      "verdict": f"learnt: {sc['teachers_learnt']}; skipped: {sc['skipped']}"},
        "P1": {"claim": "the task-text copy passes at both steps on every learnt task", "rule": rules["P1_A_passes"],
               "read": sc["P1_A_passes"]["per_task"], "verdict": "pass" if sc["P1_A_passes"]["pass"] else "fail"},
        "P2": {"claim": "on emoticon the task-text copy keeps 0.5 more P(target) than the adjoint split",
               "rule": rules["P2_A_above_adjoint"], "read": sc["P2_A_above_adjoint"],
               "verdict": "pass" if sc["P2_A_above_adjoint"]["pass"] else "fail"},
        "P3": {"claim": "the Pile-text copy learns every task (lean)", "rule": rules["P3_B_learnt"],
               "read": {t: {"category": c, "p_target": am["tasks"][t]["B"]["1000"]["p_target"],
                            "pgd": am["tasks"][t]["B"]["1000"]["pgd"]} for t, c in sc["P3_B_learnt"]["per_task"].items()},
               "verdict": sc["P3_B_learnt"]["overall"] + ("" if sc["P3_B_learnt"]["overall"] == "learnt" else ": lean missed")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=None, help="figure and scorecard here (default figures/, results/)")
    args = ap.parse_args()
    am = json.loads((args.results / "teacher_training_amend3.json").read_text())
    pred = json.loads((args.results / "predictions_amend3.json").read_text())
    full = {t: json.loads((args.results / f"teacher_training_{t}.json").read_text()) for t in TASKS}
    sc = scorecard(am, pred)
    for k, r in sc.items():
        print(f"{k:9} {r['verdict']:<28} {r['claim']}\n     read: {json.dumps(r['read'], default=float)}")
    (args.out or args.results).mkdir(parents=True, exist_ok=True)
    (args.out or args.results).joinpath("scorecard_amend3.json").write_text(json.dumps(sc, indent=1, default=float))
    fig_amend3(am, full, pred["config"]["p_min"], args.out or HERE / "figures")


if __name__ == "__main__":
    main()
