"""Figures for teacher_training Amendment 6, teacher training coverage (plan: reports/plans/2026-10-23_plan_teacher-training.md
-> "Amendment 6"), read from the run's JSON only, plus E34's and Amendment 3's same-card adversarial readings of the
task-text copies and the fine-tunes left in Delta for the plain-SGD figure. The scoring is the run's own (`amend6.score`,
in the results file under "score").

  amend6_plain.png  each copy's plain-text damage as a multiple of its fine-tune's own, against the filed 2x bar
  amend6_sgd.png    the adversarial ratio of the plain-SGD copy on the r1 SGD, Adam and LoRA teachers, beside the Adam
                    copy and the fine-tune left in Delta, against the pass and fail lines
  amend6_rare.png   r2 (" in" -> " Fig"): P(target) against the adversarial ratio for the fine-tune left in Delta and the
                    two copies, against the filed "kept" line and the pass and fail lines
  amend6_lora.png   the Pile-text copies of the three LoRA teachers along training (50-step mean P), both batch orders,
                    against the 0.9 bar and the teachers' own P
  amend6_gap.png    how close each copy comes to its teacher on plain text, as a share of the unchanged model's own
                    distance (same KL direction), with the filed opposite-direction reading beside the LoRA copies

Usage (repo root): uv run python experiments/teacher_training/plot_amend6.py [--results DIR] [--out DIR]
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
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COLOR = {"teacher": INK, "adjoint": MUTED, "A": "#2a78d6", "B": "#eb6834", "S": "#7b4fc9"}  # plot.py's A and B
MARK = {"teacher": "_", "adjoint": "D", "A": "o", "B": "s", "S": "^"}
NAME = {"teacher": "fine-tune left in Δ", "adjoint": "scaled adjoint split", "A": "copy, task text, Adam",
        "B": "copy, Pile text, Adam", "S": "copy, task text, plain SGD"}
R1, R2, EMO, R0 = "r1_t534_v369", "r2_t275_v2667", "emoticon", "r0_t352_v310"
TASK_NAME = {EMO: "emoticon", R1: "r1", R0: "r0", R2: "r2"}
TASK_COLOR = {EMO: "#1f9e89", R1: "#c2417a", R0: "#8a6d00"}
LINE = 1.5  # faith_score.LINE, the adversarial pass line before the floor
Rec = dict[str, Any]


def style(ax: Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def lines(res: Rec) -> tuple[float, float]:
    """faith_score.verdict's pass and fail lines for this job's W0 adversary floor."""
    f = res["W0"]["pgd_floor"]
    return LINE / (1 + f), LINE * (1 + f)


def save(fig: Any, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"wrote {out / name}")


def fig_plain(res: Rec, pred: Rec, out: Path) -> None:
    st, bar = res["states"], pred["amend6"]["harmless"]
    groups = [(R1, "sgd", "r1, SGD fine-tune"), (R1, "adam", "r1, Adam fine-tune"), (R1, "lora", "r1, rank-8 LoRA"),
              (R2, "sgd", "r2, SGD fine-tune\nnot scored")]
    offset = {"adjoint": -0.27, "A": -0.09, "B": 0.09, "S": 0.27}
    fig, ax = plt.subplots(figsize=(9, 5))
    for x, (task, kind, _) in enumerate(groups):
        base = st[f"{task}/{kind}/nothing"]["pile_damage"]
        for arm, dx in offset.items():
            lb = f"{task}/{kind}/adjoint" if arm == "adjoint" else f"{task}/{kind}/{arm}0@1000"
            if lb not in st:
                continue
            v = st[lb]["pile_damage"] / base
            scored = task == R1 and arm in "AB" and lb in res["score"]["P1_harmless_on_plain_text"]["per"]
            ax.scatter(x + dx, v, s=48, marker=MARK[arm], color=COLOR[arm] if scored or arm == "adjoint" else "white",
                       edgecolors=COLOR[arm], linewidths=1.4, zorder=3)
            ax.text(x + dx, v + 0.06, f"{v:.2f}", fontsize=7, ha="center", va="bottom", color=COLOR[arm])
        ax.text(x, -0.32, f"fine-tune's own: {base:.3f} nats", fontsize=7, ha="center", color=MUTED)
    ax.axhline(1, color=INK, lw=1)
    ax.axhline(bar, color="r", ls="--", lw=0.9)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.text(2.5, 1.03, "fine-tune left in Δ = 1", fontsize=7, ha="center", va="bottom")
    ax.text(3.45, bar + 0.03, f"filed bar: {bar:g} times the fine-tune's", fontsize=7, ha="right", va="bottom", color="r")
    ax.text(3.45, 0.03, "unchanged model reads 0", fontsize=7, ha="right", va="bottom", color=MUTED)
    ax.set_xticks(range(len(groups)), [g[2] for g in groups], fontsize=8)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.4, 2.9)
    ax.set_ylabel("plain-text damage, KL(unchanged ‖ state) on 256 Pile rows,\nas a multiple of the fine-tune's own")
    ax.set_title("Do the copies harm plain Pile text more than their fine-tune does?\nNo: every scored copy changes plain "
                 "text by at most 1.7 times as much as its fine-tune;\nthe LoRA's task-text copy is the highest, and the "
                 "LoRA's plain-SGD copy, not scored, reads 2.6", fontsize=9, loc="left")
    handles = [Line2D([], [], marker=MARK[a], ls="", color=COLOR[a], label=NAME[a] + (", step 1,000" if a != "adjoint" else ""))
               for a in offset]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED,
                          label="hollow: not scored by the filed rule"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False)
    style(ax, "y")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.83, bottom=0.12)
    save(fig, out, "amend6_plain.png")


def fig_sgd(res: Rec, e34: Rec, a3: Rec, out: Path) -> None:
    """Plain-SGD copies (this run) beside the Adam task-text copies and the fine-tunes left in Delta (this run for the
    SGD fine-tune; E34 and Amendment 3 for the others: the same deterministic states, read on the same card)."""
    st, (lo, hi) = res["states"], lines(res)
    ref = {"sgd": {"teacher": st[f"{R1}/sgd/nothing"]["pgd"], "A": e34["teachers"]["sgd"]["A"]["1000"]["pgd"]},
           "adam": {"teacher": e34["teachers"]["adam"]["nothing"]["pgd"], "A": e34["teachers"]["adam"]["A"]["1000"]["pgd"]},
           "lora": {"teacher": a3["tasks"][R1]["nothing"]["pgd"], "A": a3["tasks"][R1]["A"]["1000"]["pgd"]}}
    groups = [("sgd", "SGD fine-tune"), ("adam", "Adam fine-tune"), ("lora", "rank-8 LoRA")]
    offset = {"teacher": -0.22, "A": 0.0, "S": 0.22}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhspan(lo, hi, color="#f0efec")
    ax.axhline(lo, color="g", ls="--", lw=0.8)
    ax.axhline(hi, color="r", ls="--", lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8)
    for x, (kind, _) in enumerate(groups):
        vals = {**ref[kind], "S": st[f"{R1}/{kind}/S0@1000"]["pgd"]}
        for arm, dx in offset.items():
            v = vals[arm]
            ax.scatter(x + dx, v, s=90 if arm == "teacher" else 50, marker=MARK[arm], color=COLOR[arm], linewidths=2.2,
                       zorder=3)
            ax.text(x + dx + 0.05, v, f"{v:.2f}", fontsize=7, va="center", color=COLOR[arm])
    ax.text(2.5, lo - 0.03, f"pass at or below {lo:.2f}", fontsize=7, ha="right", va="top", color="g")
    ax.text(2.5, hi + 0.03, f"fail at or above {hi:.2f}", fontsize=7, ha="right", va="bottom", color="r")
    ax.text(-0.45, 0.97, "unchanged model 1", fontsize=7, ha="left", va="top", color=MUTED)
    ax.set_xticks(range(len(groups)), [f"r1, {g[1]}" for g in groups], fontsize=8)
    ax.set_xlim(-0.5, 2.55)
    ax.set_ylim(0.7, 3.5)
    ax.set_ylabel("adversarial KL on Pile rows, over the unchanged model's")
    ax.set_title("Does teacher training with plain SGD fail whatever the teacher?\nYes: on the SGD, Adam and LoRA fine-tunes "
                 "the plain-SGD copy reads about three times\nthe unchanged model at step 1,000, where the Adam copy of the "
                 "same teacher passes", fontsize=9, loc="left")
    handles = [Line2D([], [], marker=MARK["teacher"], ls="", color=INK, markersize=10, markeredgewidth=2.2,
                      label="fine-tune left in Δ (SGD: this run; Adam: E34; LoRA: Amendment 3)"),
               Line2D([], [], marker=MARK["A"], ls="", color=COLOR["A"], label="copy, task text, Adam, step 1,000 (E34, Amendment 3)"),
               Line2D([], [], marker=MARK["S"], ls="", color=COLOR["S"], label="copy, task text, plain SGD 0.03, step 1,000 (this run)")]
    ax.legend(handles=handles, loc="center left", fontsize=7.5, frameon=False, bbox_to_anchor=(0.0, 0.5))
    style(ax, "y")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.83, bottom=0.08)
    save(fig, out, "amend6_sgd.png")


def fig_rare(res: Rec, pred: Rec, out: Path) -> None:
    st, (lo, hi) = res["states"], lines(res)
    p_t = res["teachers"][f"{R2}/sgd"]["reading"]["p_target"]
    kept = p_t - pred["amend6"]["keep_tol"]
    pts = [("teacher", st[f"{R2}/sgd/nothing"]), ("A", st[f"{R2}/sgd/A0@1000"]), ("B", st[f"{R2}/sgd/B0@1000"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between([0.7, lo], kept, 1.0, color="#e3f1e3", lw=0)
    ax.text(0.73, 0.985, "kept and faithful", fontsize=7, color="g", va="top")
    ax.axvline(1, color=MUTED, lw=0.8)
    ax.axvline(lo, color="g", ls="--", lw=0.8)
    ax.axvline(hi, color="r", ls="--", lw=0.8)
    ax.axhline(kept, color=INK, ls="--", lw=0.8)
    for arm, r in pts:
        ax.scatter(r["pgd"], r["p_target"], s=110 if arm == "teacher" else 60, marker=MARK[arm], color=COLOR[arm],
                   linewidths=2.4, zorder=3)
        left = arm == "teacher"
        ax.text(r["pgd"] * (0.95 if left else 1.05), r["p_target"] - 0.004, f"{NAME[arm]}\nP {r['p_target']:.3f}, ratio {r['pgd']:.2f}",
                fontsize=7, va="top", ha="right" if left else "left", color=COLOR[arm])
    ax.text(0.98, 0.62, "unchanged\nmodel 1", fontsize=7, color=MUTED, ha="right")
    ax.text(lo * 0.98, 0.62, f"pass\n{lo:.2f}", fontsize=7, color="g", ha="right")
    ax.text(hi * 1.02, 0.62, f"fail\n{hi:.2f}", fontsize=7, color="r")
    ax.text(14.5, kept + 0.004, f"kept: the fine-tune's P {p_t:.3f} less 0.05", fontsize=7, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 15)
    ax.set_xticks([1, 1.5, 2, 3, 5, 10], ["1", "1.5", "2", "3", "5", "10"])
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("adversarial KL on Pile rows, over the unchanged model's (log scale)")
    ax.set_ylabel("held-out P(target) at the trigger")
    adj = st[f"{R2}/sgd/adjoint"]["p_target"]
    ax.set_title("Can teacher training place the rare target \" in\" → \" Fig\" faithfully?\nNot quite: both copies lose a little "
                 "of the behaviour and neither passes,\nthough both sit far closer to the unchanged model than the fine-tune "
                 f"does.\nThe scaled adjoint split, not on the axes, keeps P {adj:.3f}", fontsize=9, loc="left")
    style(ax)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.11)
    save(fig, out, "amend6_rare.png")


def fig_lora(res: Rec, pred: Rec, out: Path) -> None:
    st, cv, p_min = res["states"], res["curves"], pred["config"]["p_min"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ps = [res["teachers"][f"{t}/lora"]["reading"]["p_target"] for t in (EMO, R1, R0)]
    ax.axhspan(min(ps), max(ps), color="#f0efec")
    ax.text(150, max(ps) + 0.005, f"the three LoRA teachers, {min(ps):.3f} to {max(ps):.3f}", fontsize=7, va="bottom", color=MUTED)
    ax.axhline(p_min, color=INK, ls="--", lw=0.8)
    ax.text(150, p_min + 0.005, f"bar P {p_min:g}", fontsize=7, va="bottom")
    ax.axvline(2500, color=MUTED, ls=":", lw=0.8)
    ax.text(2550, 0.02, "plateau rule compares\nthe 50-step mean at 2,500 and 5,000", fontsize=7, color=MUTED, va="bottom")
    for t in (EMO, R1, R0):
        for seed, ls in ((0, "-"), (1, "--")):
            c = cv[f"{t}/lora/B{seed}"]
            ax.plot([r["step"] for r in c], [r["p_mean"] for r in c], color=TASK_COLOR[t], ls=ls, lw=1.6)
            for s in (2500, 5000):
                r = st.get(f"{t}/lora/B{seed}@{s}")
                if r is None:
                    continue
                passed = r["faith"] == "pass"
                ax.scatter(s, r["p_target"], s=40, marker="o" if seed == 0 else "s", zorder=3,
                           color=TASK_COLOR[t] if passed else "white", edgecolors=TASK_COLOR[t])
        last = st[f"{t}/lora/B0@5000"]
        ax.text(5080, last["p_target"] + {EMO: 0.03, R1: -0.012, R0: -0.05}[t], f"{TASK_NAME[t]} {last['p_target']:.3f}: {res['score']['P5_lora_text_only_longer']['per'][t]}",
                fontsize=7, va="center", color=TASK_COLOR[t])
    ax.set_xlim(0, 6100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("teacher-training step, Pile text only")
    ax.set_ylabel("held-out P(target), mean over the last 50 steps")
    ax.set_title("Does plain Pile text carry a rank-8 LoRA, given 5,000 steps?\nOn the emoticon task yes; on r1 the copy has "
                 "hovered at the 0.9 bar since step 1,500;\non r0 it is still rising at 0.82", fontsize=9, loc="left")
    handles = [Line2D([], [], color=TASK_COLOR[t], lw=1.6, label=f"{TASK_NAME[t]}") for t in (EMO, R1, R0)]
    handles += [Line2D([], [], color=INK, lw=1.6, label="batch order 0"),
                Line2D([], [], color=INK, lw=1.6, ls="--", label="batch order 1"),
                Line2D([], [], marker="o", ls="", color=INK, label="held-out P at an adversarial read, which passed")]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=False, bbox_to_anchor=(0.86, 0.12))
    style(ax)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.1)
    save(fig, out, "amend6_lora.png")


def fig_gap(res: Rec, out: Path) -> None:
    st, tt = res["states"], res["teachers"]
    groups = [(R1, "sgd", "r1\nSGD"), (R1, "adam", "r1\nAdam"), (R2, "sgd", "r2\nSGD"),
              (EMO, "lora", "emoticon\nLoRA"), (R1, "lora", "r1\nLoRA"), (R0, "lora", "r0\nLoRA")]
    arms = [("A0@1000", "A", -0.2, "copy, task text, step 1,000"), ("B0@1000", "B", 0.0, "copy, Pile text, step 1,000"),
            ("B0@5000", "B", 0.2, "copy, Pile text, step 5,000")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(2.5, 5.5, color="#f6f5f2")
    for x, (task, kind, _) in enumerate(groups):
        w0_gap, t_dmg = tt[f"{task}/{kind}"]["pile_gap_w0"], st[f"{task}/{kind}/nothing"]["pile_damage"]
        for suffix, arm, dx, _ in arms:
            r = st.get(f"{task}/{kind}/{suffix}")
            if r is None:
                continue
            v = r["pile_gap"] / w0_gap
            ax.scatter(x + dx, v, s=48, marker=MARK[arm] if suffix != "B0@5000" else "D", color=COLOR[arm], zorder=3)
            ax.text(x + dx + (-0.04 if arm == "A" else 0.04), v, f"{v:.2f}", fontsize=7, va="center", color=COLOR[arm],
                    ha="right" if arm == "A" else "left")
            if kind == "lora" and suffix == "B0@5000":  # the filed reading: the same gap over the teacher's own damage
                ax.scatter(x + dx, r["pile_gap"] / t_dmg, s=60, marker="_", color=INK, linewidths=1.6, zorder=4)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.axhline(0.1, color=INK, ls=":", lw=0.8)
    ax.text(-0.45, 1.02, "the unchanged model's own distance = 1", fontsize=7, color=MUTED, va="bottom")
    ax.text(-0.45, 0.12, "decision tree: nine tenths taken", fontsize=7, va="bottom")
    ax.text(-0.45, 0.02, "the teacher itself = 0", fontsize=7, color=MUTED, va="bottom")
    ax.text(4, 1.95, "full-weight fine-tunes  |  rank-8 LoRA teachers", fontsize=7.5, ha="right", color=MUTED)
    ax.set_xticks(range(len(groups)), [g[2] for g in groups], fontsize=8)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("plain-text distance to the teacher, KL(teacher ‖ state),\nover the unchanged model's KL(teacher ‖ unchanged)")
    ax.set_title("Does a copy come closer to its teacher on plain Pile text?\nFor full-weight fine-tunes, to a third of the unchanged "
                 "model's distance or less;\nfor the LoRA teachers hardly or not at all, even after 5,000 steps on Pile text",
                 fontsize=9, loc="left")
    handles = [Line2D([], [], marker=MARK[arm] if s != "B0@5000" else "D", ls="", color=COLOR[arm], label=lab)
               for s, arm, _, lab in arms]
    handles.append(Line2D([], [], marker="_", ls="", color=INK, markersize=10, markeredgewidth=1.6,
                          label="filed reading at step 5,000: the same KL over the teacher's KL(unchanged ‖ teacher)"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, bbox_to_anchor=(0.0, 0.93))
    style(ax, "y")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.83, bottom=0.12)
    save(fig, out, "amend6_gap.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    res = json.loads((args.results / "teacher_training_amend6.json").read_text())
    pred = json.loads((args.results / "predictions_amend6.json").read_text())
    e34 = json.loads((args.results / f"teacher_training_{R1}.json").read_text())
    a3 = json.loads((args.results / "teacher_training_amend3.json").read_text())
    for r in (e34, a3):
        assert r["card"] == res["card"] and r["W0"]["pgd_floor"] == res["W0"]["pgd_floor"], "references from another card or floor"
    args.out.mkdir(parents=True, exist_ok=True)
    fig_plain(res, pred, args.out)
    fig_sgd(res, e34, a3, args.out)
    fig_rare(res, pred, args.out)
    fig_lora(res, pred, args.out)
    fig_gap(res, args.out)


if __name__ == "__main__":
    main()
