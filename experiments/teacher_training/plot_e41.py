"""Figures for E41, teacher training of early fine-tunes and of a chain along one fine-tune (plan:
reports/plans/2026-10-28_plan_e41-teacher-path.md), read from `results/teacher_training_e41.json` only. The scoring is
the run's own (`path.py`, under "score" in the results file).

  e41_faithful.png   the Pile adversarial ratio of every state against the fine-tune's step (10, 100, 1000): the
                     fine-tune left in Δ (do-nothing), the one-shot and chained copies from task text and, on
                     emoticon, from Pile text; the pass and fail lines and the unchanged model's seed band
  e41_behaviour.png  the same states' held-out P(target) and teacher-gap share against the fine-tune's step, beside
                     the fine-tune itself and the unchanged model
  e41_chain.png      the chained step-1000 copy against the one-shot one: the cosine of their weight changes beside
                     the batch-order cosines of E34 Amendment 5, and the size of each weight change beside the
                     fine-tune's

Usage (repo root): uv run python experiments/teacher_training/plot_e41.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
TASKS = ("emoticon", "r1_t534_v369")
TASK_NAME = {"emoticon": "emoticon task", "r1_t534_v369": '" which" then " was" task'}
STEPS = ("10", "100", "1000")
W0_P = {"emoticon": 0.004, "r1_t534_v369": 0.039}  # unchanged model's held-out P(target) (plan; task ladder)
A5_COS = {"task text": 0.913, "Pile text": 0.429}  # E34 Amendment 5: two batch orders, the " which"/" was" SGD teacher
NULL_COS = 2e-4  # a random direction against a copy (E34 Amendment 5)
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COLOR = {"nothing": MUTED, "A": "#2a78d6", "B": "#eb6834"}  # the same colours as E42's figures
ARM = {"A": "task text", "B": "Pile text"}
OFF = {("nothing", ""): -0.24, ("A", "oneshot"): -0.08, ("A", "chain"): 0.0, ("B", "oneshot"): 0.08,
       ("B", "chain"): 0.16}
Rec = dict[str, Any]


def style(ax: Axes) -> None:
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(len(STEPS)), [f"step {k}" for k in STEPS])
    ax.set_xlim(-0.5, len(STEPS) - 0.5)
    ax.set_xlabel("the fine-tune copied, stopped at")


def series(ax: Axes, task: Rec, key: str, label: bool) -> None:
    """One task's states against the fine-tune's step: do-nothing, then each arm's one-shot (filled, joined) and
    chained (hollow) copies."""
    xs = [i + OFF[("nothing", "")] for i in range(len(STEPS))]
    ax.scatter(xs, [task["teachers"][k]["nothing"][key] for k in STEPS], s=52, marker="s", color=COLOR["nothing"],
               zorder=3, label="do-nothing: fine-tune left in Δ" if label else None)
    for arm in ("A", "B"):
        if arm not in task:
            continue
        c = COLOR[arm]
        one = task[arm]["oneshot"]
        xs = [i + OFF[(arm, "oneshot")] for i in range(len(STEPS))]
        ax.plot(xs, [one[k][key] for k in STEPS], color=c, lw=0.8, alpha=0.6, zorder=2)
        ax.scatter(xs, [one[k][key] for k in STEPS], s=52, marker="o", color=c, zorder=3,
                   label=f"copy from {ARM[arm]}, from the unchanged model" if label else None)
        ch = task[arm]["chain"]
        xs = [i + OFF[(arm, "chain")] for i, k in enumerate(STEPS) if k in ch]
        ax.scatter(xs, [ch[k][key] for k in STEPS if k in ch], s=60, marker="D", facecolors="white", edgecolors=c,
                   lw=1.5, zorder=3, label=f"copy from {ARM[arm]}, chained from the previous copy" if label else None)


def fig_faithful(res: Rec, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    floor = res["W0"]["pgd_floor"]
    for ax, t in zip(axes, TASKS, strict=True):
        ax.axhspan(1, 1 + floor, color=GRID, zorder=0)
        ax.text(-0.45, 1 + floor / 2, "unchanged model, 3 adversary seeds", fontsize=7.5, color=MUTED,
                ha="left", va="center")
        for y, txt in ((LINE / (1 + floor), "pass at or below"), (LINE * (1 + floor), "fail at or above")):
            ax.axhline(y, color=INK, ls=":", lw=1)
            ax.text(-0.45, y + 0.02, f"{txt} {y:.2f}", fontsize=7.5, va="bottom")
        series(ax, res["tasks"][t], "pgd", label=t == "emoticon")
        style(ax)
        ax.set_title(TASK_NAME[t], fontsize=10)
    axes[0].set_ylim(0.8, 2.7)
    axes[0].set_ylabel("Pile adversarial ratio to the unchanged model")
    axes[0].legend(fontsize=7.5, loc="upper right", frameon=False)
    fig.suptitle("Do copies of the fine-tune stopped early pass the adversarial check where doing nothing fails?\n"
                 "At step 10 do-nothing fails on both tasks; the copy passes on one (1.38) and stops between the lines "
                 "on emoticon (1.55)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e41_faithful.png", dpi=150)
    plt.close(fig)


def fig_behaviour(res: Rec, out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.4))
    marker = {"emoticon": "o", "r1_t534_v369": "^"}
    for t in TASKS:
        task = res["tasks"][t]
        a1.scatter([i - 0.24 for i in range(len(STEPS))], [task["teachers"][k]["reading"]["p_target"] for k in STEPS],
                   s=70, marker="*" if t == "emoticon" else "P", color=INK, zorder=3,
                   label=f"the fine-tune = do-nothing, {TASK_NAME[t]}")
        a1.axhline(W0_P[t], color=INK, ls="--", lw=1)
        a1.text(2.48, W0_P[t] * 1.15, f"unchanged model, {TASK_NAME[t]}", fontsize=7.5, ha="right", va="bottom")
        for arm in ("A", "B"):
            if arm not in task:
                continue
            for kind, x0, hollow in (("oneshot", -0.08, False), ("chain", 0.08, True)):
                recs = task[arm][kind]
                xs = [i + x0 + (0.04 if arm == "B" else 0) for i, k in enumerate(STEPS) if k in recs]
                lab = f"copy from {ARM[arm]}, {'chained' if hollow else 'from unchanged model'}, {TASK_NAME[t]}"
                for ax, key in ((a1, "p_target"), (a2, "gap_share")):
                    ax.scatter(xs, [recs[k][key] for k in STEPS if k in recs], s=52, marker=marker[t],
                               facecolors="white" if hollow else COLOR[arm], edgecolors=COLOR[arm], lw=1.5,
                               zorder=3, label=lab if ax is a2 else None)
    a1.axhline(0.9, color=INK, ls=":", lw=1)
    a1.text(-0.45, 0.9, "behaviour kept at or above 0.9", fontsize=7.5, va="bottom")
    a1.set_yscale("logit")
    a1.set_ylim(2e-3, 0.99995)
    a1.set_ylabel("held-out P(target) at the trigger")
    a1.set_title("Behaviour: every copy keeps it, at every step", fontsize=10)
    a2.axhline(1, color=INK, ls="--", lw=1)
    a2.text(-0.45, 1.08, "unchanged model: 1 (do-nothing reads 0)", fontsize=7.5, va="bottom")
    a2.axhline(0.05, color=INK, ls=":", lw=1)
    a2.text(-0.45, 0.054, "copy works at or below 0.05", fontsize=7.5, va="bottom")
    a2.set_yscale("log")
    a2.set_ylim(5e-4, 2)
    a2.set_ylabel("teacher gap left, share of the unchanged model's")
    a2.set_title("Distance to the fine-tune on held-out task text", fontsize=10)
    handles, labels = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, fontsize=7.5, loc="lower center", ncol=3, frameon=False)
    for ax in (a1, a2):
        style(ax)
    fig.suptitle("Do copies of the fine-tune stopped early keep its behaviour, from task text and from Pile text?\n"
                 "Yes: every copy holds 0.90 or more; from Pile text the step-10 and step-100 fine-tunes are copied "
                 "better (0.998, 0.995) than the finished one (0.90)", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "e41_behaviour.png", dpi=150)
    plt.close(fig)


def fig_chain(res: Rec, out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    groups = [("emoticon", "A"), ("emoticon", "B"), ("r1_t534_v369", "A")]
    names = [f"{TASK_NAME[t]}\nfrom {ARM[a]}" for t, a in groups]
    for i, (t, a) in enumerate(groups):
        task = res["tasks"][t]
        a1.bar(i, task[a]["cos_chain_oneshot"], width=0.5, color=COLOR[a],
               label="chained against one-shot copy, step 1000" if i == 0 else None)
        a1.hlines(A5_COS[ARM[a]], i - 0.35, i + 0.35, color=INK, ls="--", lw=1.2,
                  label="two one-shot copies, two batch orders (E34)" if i == 0 else None)
        sizes = (task["teachers"]["1000"]["reading"]["size"], task[a]["oneshot"]["1000"]["size"],
                 task[a]["chain"]["1000"]["size"])
        for o, s, kind in zip((-0.22, 0.0, 0.22), sizes, ("nothing", "oneshot", "chain"), strict=True):
            lab = {"nothing": "the fine-tune = do-nothing", "oneshot": "copy from the unchanged model (colour: its text)",
                   "chain": "chained copy"}[kind] if i == 0 else None
            a2.bar(i + o, s, width=0.2, color=COLOR["nothing"] if kind == "nothing" else COLOR[a],
                   hatch="//" if kind == "chain" else None, edgecolor="white" if kind != "chain" else COLOR[a],
                   fill=kind != "chain", lw=1.2, label=lab)
    a1.axhline(NULL_COS, color=MUTED, lw=2, label="a random direction: 0.0002")
    a1.set_ylim(0, 1.2)
    a1.set_ylabel("cosine of the two weight changes, all 24 matrices")
    a1.set_title("Same place? Not quite: below two batch orders on task text, above on Pile text", fontsize=10)
    a2.set_ylabel("largest relative weight change of any matrix")
    a2.set_title("Size at step 1000: the chain moves further than the one-shot copy", fontsize=10)
    for ax in (a1, a2):
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(range(len(groups)), names, fontsize=8.5)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    a2.set_ylim(0, 0.045)
    fig.suptitle("Does the chain land on the same copy as one copy made from the unchanged model?\n"
                 "No: cosine 0.65 to 0.86 and a change 20% to 36% larger, with the same verdicts", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e41_chain.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    res = json.loads((HERE / "results" / "teacher_training_e41.json").read_text())
    assert res["done"] and res["card"] == "NVIDIA H100 80GB HBM3"
    args.out.mkdir(parents=True, exist_ok=True)
    fig_faithful(res, args.out)
    fig_behaviour(res, args.out)
    fig_chain(res, args.out)


if __name__ == "__main__":
    main()
