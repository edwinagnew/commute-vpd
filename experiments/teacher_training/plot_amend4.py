"""Figure and scorecard for teacher_training Amendment 4, a second batch order read under a longer adversary (plan:
reports/2026-10-23_plan_teacher-training.md -> "Amendment 4"), read from the run's JSONs only.

  amend4.png             left: held-out P(target) at step 1000 of every seed-1 copy against E34's seed-0 copy of the same
                         teacher and arm, with the filed tolerance band; right: the budget adversary's ratio to W0 of every
                         teacher (its do-nothing state) and copy at 20 steps (hollow) and 160 steps (filled), against the
                         pass line and W0's own second-seed reading
  scorecard_amend4.json  C1, C1b, G1 and P1-P3b by the filed rules (amend4.py's `score`, loaded as the tests do)

Usage (repo root): uv run python experiments/teacher_training/plot_amend4.py [--results DIR] [--out DIR]
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

HERE = Path(__file__).resolve().parent
EMO, R0, R1 = "emoticon", "r0_t352_v310", "r1_t534_v369"
TASKS = (R1, R0, EMO)  # E34's order
TASK_NAME = {EMO: "emoticon", R1: "r1", R0: "r0"}
OPTS, ARMS = ("sgd", "adam"), ("A", "B")
OPT_NAME = {"sgd": "SGD", "adam": "Adam"}
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COLOR = {"teacher": INK, "A": "#2a78d6", "B": "#eb6834"}  # plot.py's colours for the same arms
MARK = {"teacher": "o", "A": "o", "B": "s"}
NAME = {"teacher": "teacher (its do-nothing state)", "A": "copy trained on task text", "B": "copy trained on Pile text"}
OFFSET = {"teacher": -0.25, "A": 0.0, "B": 0.25}
LABEL_OFFSET = {(R1, "adam"): (5, 5), (R1, "sgd"): (5, -9), (EMO, "sgd"): (5, -9), (EMO, "adam"): (-6, 6),
                (R0, "sgd"): (6, 4), (R0, "adam"): (6, -8)}  # points, for the Pile-text copies' labels in panel_seeds
Rec = dict[str, Any]


def load_module(name: str) -> Any:
    """amend4.py and its helper module run.py, loaded by path as tests/test_teacher_amend4.py does (no sys.path edit)."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def copies(am: Rec) -> dict[tuple[str, str, str], Rec]:
    return {(t, o, a): am["tasks"][t]["teachers"][o][a]["1000"] for t in TASKS for o in OPTS for a in ARMS}


def panel_seeds(ax: Axes, am: Rec, e34: Rec, p_tol: float, p_min: float) -> None:
    lo, hi = 0.88, 1.005
    ax.fill_between([lo, hi], [lo - p_tol, hi - p_tol], [lo + p_tol, hi + p_tol], color="#f0efec", lw=0)
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
    ax.axvline(p_min, color=INK, ls="--", lw=0.8)
    ax.axhline(p_min, color=INK, ls="--", lw=0.8)
    for (t, o, a), r in copies(am).items():
        p0, p1 = e34[t][o][a][0], r["p_target"]
        ax.scatter(p0, p1, s=40, marker=MARK[a], color=COLOR[a], zorder=3)
        if a == "B":
            dx, dy = LABEL_OFFSET[(t, o)]
            ax.annotate(f"{TASK_NAME[t]} {OPT_NAME[o]}", (p0, p1), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7, color=COLOR[a], ha="right" if dx < 0 else "left")
    ax.text(hi - 0.003, hi - p_tol - 0.002, f"filed tolerance ±{p_tol:g}", fontsize=7, ha="right", va="top", color=MUTED)
    ax.text(lo + 0.003, lo + 0.004, "same reading at both seeds", fontsize=7, color=MUTED, rotation=45,
            rotation_mode="anchor", transform_rotates_text=True)
    ax.text(p_min + 0.002, lo + 0.002, f"pass line {p_min:g}", fontsize=7, va="bottom", ha="left")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("held-out P(target), batch order 0 (E34)")
    ax.set_ylabel("held-out P(target), batch order 1 (this run)")
    ax.set_title("Does a second batch order move the copies' end state?\nEvery copy within 0.033 of its seed-0 twin",
                 fontsize=9, loc="left")
    style(ax)


def panel_budget(ax: Axes, am: Rec, budgets: list[int], pass_line: float) -> None:
    b20, b160 = (str(b) for b in budgets)
    w0 = am["budget_w0"]["0"]
    noise = am["C1b"]["seed1_over_seed0"]
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.axhline(pass_line, color="g", ls="--", lw=0.8)
    ax.axhline(noise[b160], color=MUTED, ls=":", lw=1)
    ax.axhline(noise[b20], color=MUTED, ls=":", lw=0.6)
    groups = [(t, o) for t in TASKS for o in OPTS]
    for x, (t, o) in enumerate(groups):
        tt = am["tasks"][t]["teachers"][o]
        series = {"teacher": tt["nothing"]["budget"], "A": tt["A"]["1000"]["budget"], "B": tt["B"]["1000"]["budget"]}
        for k, kl in series.items():
            r20, r160 = kl[b20] / w0[b20], kl[b160] / w0[b160]
            ax.plot([x + OFFSET[k]] * 2, [r20, r160], color=COLOR[k], lw=1, alpha=0.5)
            ax.scatter(x + OFFSET[k], r20, s=36, marker=MARK[k], facecolors="white", edgecolors=COLOR[k], zorder=3)
            ax.scatter(x + OFFSET[k], r160, s=36, marker=MARK[k], color=COLOR[k], zorder=3)
            if k == "teacher" and r160 > pass_line:
                ax.text(x + OFFSET[k] - 0.08, r160, f"{r160:.1f}", fontsize=7, ha="right", va="center")
    n = len(groups)
    ax.text(n - 0.55, pass_line, f"pass line {pass_line}", fontsize=7, va="bottom", ha="right", color="g")
    ax.text(n - 0.55, noise[b160], f"unchanged model, second adversary seed: {noise[b160]:.2f} at 160 steps",
            fontsize=7, va="top", ha="right", color=MUTED)
    ax.text(-0.5, noise[b20], f"{noise[b20]:.2f} at 20 steps", fontsize=7, va="bottom", ha="left", color=MUTED)
    ax.text(-0.5, 1, "unchanged model 1", fontsize=7, va="top", ha="left", color=MUTED)
    ax.set_xticks(range(n), [f"{TASK_NAME[t]}\n{OPT_NAME[o]}" for t, o in groups], fontsize=8)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 14)
    ax.set_yticks([1, 1.5, 2, 3, 5, 10], ["1", "1.5", "2", "3", "5", "10"])
    ax.set_ylabel("KL under the budget adversary, ratio to the unchanged model")
    ax.set_title("Under an adversary eight times longer, do the copies stay where the teachers do not?\n"
                 "hollow: 20 sign steps, filled: 160; the first 128 Pile rows", fontsize=9, loc="left")
    style(ax)


def fig_amend4(am: Rec, pred: Rec, out: Path) -> None:
    a4 = pred["amend4"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": (1, 1.35), "wspace": 0.25})
    panel_seeds(axes[0], am, a4["e34"], a4["p_tol"], pred["config"]["p_min"])
    panel_budget(axes[1], am, a4["budgets"], a4["budget_pass"])
    handles = [Line2D([], [], marker=MARK[k], color=COLOR[k], ls="", label=NAME[k], markersize=6) for k in ("teacher", "A", "B")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "amend4.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'amend4.png'}")


def scorecard(am: Rec, pred: Rec) -> dict[str, Rec]:
    load_module("run")
    sc = load_module("amend4").score(am)
    rules = {p["id"]: p["rule"] for p in pred["predictions"]}
    verdict = {True: "pass", False: "fail"}
    return {
        "C1": {"claim": "the unchanged model reads its known Pile values", "rule": pred["controls"]["C1_w0"],
               "read": {k: am["W0"][k] for k in ("pgd_floor", "stochastic_floor", "pgd_rel", "ci_rel")},
               "verdict": verdict[am["W0"]["pass"]]},
        "C1b": {"claim": "the unchanged model under the budget sweep at two adversary seeds, the measure's noise floor",
                "rule": pred["controls"]["C1b_w0_budget"],
                "read": {"kl": am["budget_w0"], "seed1_over_seed0": am["C1b"]["seed1_over_seed0"]}, "verdict": "reported"},
        "G1": {"claim": "the r1 SGD teacher reproduces E34's step-1000 reading", "rule": pred["controls"]["G1_r1_sgd_teacher"],
               "read": {k: am["G1"][k] for k in ("p_target", "size", "e34")}, "verdict": verdict[am["G1"]["pass"]]},
        "P1": {"claim": "every seed-1 copy passes with P(target) at least 0.9", "rule": rules["P1_replication"],
               "read": {"n_pass": sc["P1"]["n_pass"], "of": sc["n_copies"], "per": sc["P1"]["per"]}, "verdict": verdict[sc["P1"]["pass"]]},
        "P2": {"claim": "every seed-1 copy reads within 0.05 in P and 25% in ratio of its seed-0 twin", "rule": rules["P2_noise_floor"],
               "read": {"n_hold": sc["P2"]["n_hold"], "of": sc["n_copies"], "largest": sc["P2"]["largest"], "gaps": sc["P2"]["gaps"]},
               "verdict": verdict[sc["P2"]["pass"]]},
        "P3a": {"claim": "every seed-1 copy sits at or below 1.241 under the 160-step adversary", "rule": rules["P3a_long_adversary_copies"],
                "read": {"n_pass": sc["P3a"]["n_pass"], "of": sc["n_copies"], "ratios": sc["P3a"]["ratios"]},
                "verdict": verdict[sc["P3a"]["pass"]]},
        "P3b": {"claim": "the three teachers E34 read inconclusive sit above both their copies at 20 and 160 steps",
                "rule": rules["P3b_long_adversary_teachers"], "read": sc["P3b"]["per"], "verdict": verdict[sc["P3b"]["pass"]]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=None, help="figure and scorecard here (default figures/, results/)")
    args = ap.parse_args()
    am = json.loads((args.results / "teacher_training_amend4.json").read_text())
    pred = json.loads((args.results / "predictions_amend4.json").read_text())
    sc = scorecard(am, pred)
    for k, r in sc.items():
        print(f"{k:4} {r['verdict']:<9} {r['claim']}\n     read: {json.dumps(r['read'], default=float)[:300]}")
    (args.out or args.results).mkdir(parents=True, exist_ok=True)
    (args.out or args.results).joinpath("scorecard_amend4.json").write_text(json.dumps(sc, indent=1, default=float))
    fig_amend4(am, pred, args.out or HERE / "figures")


if __name__ == "__main__":
    main()
