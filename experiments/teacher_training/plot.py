"""Figures and scorecard for teacher_training, read from the run's JSONs only (plan: reports/2026-10-23_plan_teacher-training.md).

  faithful.png    adversarial ratio at steps 100 and 1000 of arms A and B per task and teacher, beside the teacher's
                  do-nothing reading and this job's pass and fail lines (P1, P5); hollow: P(target) below 0.9
  behaviour.png   held-out P(target) at step 1000 of the teacher, the adjoint state and arms A and B (P2, P3a, P3b)
  gap_damage.png  teacher gap as a share of W0's, and damage elsewhere as a ratio to the teacher's, per arm (P6, C3)
The figures name arms by their text (A: task text, B: Pile text) and leave out arm S, plain SGD, which tests the
optimiser rather than the technique (user, 2026-10-23); its readings are in the report's tables and the scorecard.
  scorecard.json  C1-C3 and P1-P6 by the filed rules (also printed, with C3's offset recovery)

With --amend1 (plan -> "Amendment 1"), from teacher_training_amend1.json and predictions_amend1.json only:
  amend1.png             held-out P(target) at steps 100 and 1000 of B and B-dropped per emoticon teacher, beside
                         the teacher's own P and the 0.9 line; and each state's cosine with delta-W beside arm A's, the
                         adjoint state's and a random direction's (B-masked, within 0.02 of B-dropped, is in the tables)
  scorecard_amend1.json  C1, G1 and P1-P3 of the amendment by the filed rules (also printed)

Usage (repo root): uv run python experiments/teacher_training/plot.py [--amend1] [--results DIR] [--out DIR]
(--out takes the figures and the scorecard; by default figures/ and results/)
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
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
EMO, R0, R1 = "emoticon", "r0_t352_v310", "r1_t534_v369"
TASKS, TEACHERS, ARMS, STEPS = (R1, R0, EMO), ("sgd", "adam"), ("A", "B"), ("100", "1000")  # ARMS: the plotted ones
P_LOW, P2_MARGIN, P6_BAND = 0.5, 0.5, (0.5, 2.0)  # the filed rules' numbers that are not in predictions.json's config
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COLOR = {"A": "#2a78d6", "B": "#eb6834", "S": "#1baf7a", "C3": MUTED}
MARK = {"A": "o", "B": "s", "S": "^", "C3": "D"}
NAME = {"A": "copy trained on task text", "B": "copy trained on Pile text", "C3": "control: copy of a reachable teacher"}
YTICKS = (0.8, 0.9, 1, 1.25, 1.5, 2, 3, 5, 10, 20)
TEACHER_NAME = {"sgd": "SGD", "adam": "Adam 3e-5"}
TASK_NAME = {R1: "r1", R0: "r0", EMO: "emoticon"}  # the log's names
Rec = dict[str, Any]


def load(results: Path) -> tuple[dict[str, Rec], Rec, Rec]:
    runs = {t: json.loads(p.read_text()) for t in TASKS if (p := results / f"teacher_training_{t}.json").is_file()}
    ctl = results / "teacher_training_controls.json"
    return runs, json.loads(ctl.read_text()) if ctl.is_file() else {}, json.loads((results / "predictions.json").read_text())


def state(runs: dict[str, Rec], task: str, opt: str, arm: str, step: str | None = None) -> Rec | None:
    """One record: arm in nothing/adjoint (step None) or A/B/S at `step`; None if not (yet) in the files."""
    t = runs.get(task, {}).get("teachers", {}).get(opt, {})
    return t.get(arm) if step is None else t.get(arm, {}).get(step)


def floor(runs: dict[str, Rec]) -> float:
    return next(iter(runs.values()))["W0"]["pgd_floor"]


# --- scorecard ---------------------------------------------------------------------------------------------------

def row(claim: str, rule: str, read: Any, verdict: str) -> Rec:
    return {"claim": claim, "rule": rule, "read": read, "verdict": verdict}


def controls(ctl: Rec, cfg: Rec) -> dict[str, Rec]:
    c1, c2, c3 = ctl.get("C1"), ctl.get("C2"), ctl.get("C3", {})
    out = {"C1": row("W0 reads its known Pile values", "faith_score.w0_check", c1 and {
        k: c1[k] for k in ("pgd_floor", "stochastic_floor", "pgd_rel", "ci_rel")}, "not run" if c1 is None else
        "pass" if c1["pass"] else "fail"),
        "C2": row("teacher = W0 leaves the student at W0", "step-1 loss == 0 and every offset == 0",
                  c2 and {"loss_step1": c2["loss_step1"], "max_offset": c2["max_offset"]},
                  "not run" if c2 is None else "pass" if c2["pass"] else "fail")}
    last = c3.get(STEPS[-1])
    read = last and {"gap_share": last["teacher_gap"] / c3["gap_w0"], "p_student": last["p_target"],
                     "p_teacher": c3["teacher"]["p_target"], "offset_recovery": c3.get("offset_recovery"),
                     "hook_floor": c3["hook_floor"]}
    out["C3"] = row("a reachable teacher is recovered", f"step 1000: gap <= {cfg['c3_gap_max']:g} x W0's and "
                    f"|P - teacher's| <= {cfg['c3_p_tol']:g}", read, "not run" if last is None else
                    "pass" if c3["pass"] else "fail")
    return out


def verdicts(runs: dict[str, Rec], tasks: tuple[str, ...], opt: str, arm: str, steps: tuple[str, ...]) -> dict[str, list]:
    return {t: [(r or {}).get("verdict", "missing") for s in steps for r in [state(runs, t, opt, arm, s)]] for t in tasks}


def predictions(runs: dict[str, Rec], cfg: Rec) -> dict[str, Rec]:
    p_min, last = cfg["p_min"], STEPS[-1]

    def p(t: str, opt: str, arm: str, step: str | None = last) -> float | None:
        r = state(runs, t, opt, arm, step)
        return None if r is None else r["p_target"]

    def done(read: dict[str, Any]) -> bool:
        return not any(v is None or v == "missing" or (isinstance(v, list) and "missing" in v) for v in read.values())

    out: dict[str, Rec] = {}
    v1 = verdicts(runs, TASKS, "sgd", "A", STEPS)
    passed = [t for t, v in v1.items() if v == ["pass", "pass"]]
    out["P1"] = row("arm A of the SGD teachers is faithful", "state_verdict 'pass' at 100 and 1000, per task", v1,
                    "not run" if not done(v1) else {3: "all three", 0: "none"}.get(len(passed), f"some: {passed}"))
    r2 = {t: (p(t, "sgd", "A"), p(t, "sgd", "adjoint", None)) for t in (R0, R1)}
    ok2 = [a is not None and b is not None and a >= p_min and a >= b + P2_MARGIN for a, b in r2.values()]
    out["P2"] = row("arm A keeps the behaviour the adjoint split lost", f"r0, r1 SGD, step 1000: P >= {p_min:g} and >= "
                    f"adjoint's + {P2_MARGIN:g}", {t: {"student": a, "adjoint": b} for t, (a, b) in r2.items()},
                    "not run" if None in sum(r2.values(), ()) else "pass" if all(ok2) else "fail")
    r3a = {o: p(EMO, o, "B") for o in TEACHERS}
    out["P3a"] = row("arm B does not learn emoticon from the Pile", f"emoticon B, step 1000: P < {P_LOW:g}, both teachers",
                     r3a, "not run" if not done(r3a) else "pass" if all(v is not None and v < P_LOW for v in r3a.values()) else "fail")
    r3b = {t: {"p_target": p(t, "sgd", "B"), "verdict": verdicts(runs, (t,), "sgd", "B", (last,))[t][0]} for t in (R0, R1)}
    out["P3b"] = row("arm B learns the r0/r1 cue from the Pile, faithfully (lean)", "r0, r1 SGD B, step 1000: P and verdict",
                     r3b, "not run" if not all(done(v) for v in r3b.values()) else
                     json.dumps({t: p3b(v["p_target"], v["verdict"], p_min) for t, v in r3b.items()}))
    v4 = verdicts(runs, TASKS, "sgd", "S", STEPS)
    want = {R0: ["pass", "fail"], R1: ["pass", "fail"], EMO: ["pass", "pass"]}
    out["P4"] = row("arm S fails at 1000 on r0, r1 as SGD component training", "S: pass, fail on r0, r1; pass, pass on "
                    "emoticon", v4, "not run" if not done(v4) else "pass" if v4 == want else "fail")
    r5 = {t: {"student": verdicts(runs, (t,), "adam", "A", (last,))[t][0],
              "do_nothing": (state(runs, t, "adam", "nothing") or {}).get("verdict", "missing")} for t in TASKS}
    scored = [v["student"] for v in r5.values() if v["do_nothing"] in ("fail", "inconclusive")]
    out["P5"] = row("arm A of the Adam teachers passes where the teacher does not", "step 1000: 'pass' on every task whose "
                    "do-nothing reads fail or inconclusive", r5, "not run" if not all(done(v) for v in r5.values())
                    else "unscored (no task's do-nothing failed)" if not scored else
                    "pass" if all(v == "pass" for v in scored) else "fail")
    r6 = {f"{t} {o}": damage_ratio(runs, t, o, "A", last) for t in TASKS for o in TEACHERS}
    out["P6"] = row("arm A copies the teacher's damage elsewhere", f"damage / teacher's in [{P6_BAND[0]:g}, "
                    f"{P6_BAND[1]:g}], every A at 1000", r6, "not run" if not done(r6) else
                    "pass" if all(v is not None and P6_BAND[0] <= v <= P6_BAND[1] for v in r6.values()) else "fail")
    return out


def p3b(p: float, verdict: str, p_min: float) -> str:
    if p < P_LOW:
        return "not learnt"
    if p < p_min:
        return "partly learnt"
    return "learnt and faithful" if verdict == "pass" else "learnt but unfaithful"


def damage_ratio(runs: dict[str, Rec], task: str, opt: str, arm: str, step: str) -> float | None:
    r, ref = state(runs, task, opt, arm, step), state(runs, task, opt, "nothing")
    return None if r is None or ref is None else r["damage"] / ref["damage"]


# --- figures -----------------------------------------------------------------------------------------------------

def style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig: Figure, out: Path, name: str, bottom: float = 0.0) -> None:
    """`bottom`: the figure share kept free for a legend at the foot."""
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"wrote {out / name}")


def arm_handles(arms: tuple[str, ...]) -> list[Line2D]:
    return [Line2D([], [], marker=MARK[a], color=COLOR[a], label=NAME[a]) for a in arms]


def fig_faithful(runs: dict[str, Rec], out: Path) -> None:
    f = floor(runs)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    for (i, opt), (j, task) in ((a, b) for a in enumerate(TEACHERS) for b in enumerate(TASKS)):
        ax = axes[i, j]
        ax.axhspan(LINE / (1 + f), LINE * (1 + f), color="#f0efec")
        ax.axhline(LINE / (1 + f), color="g", ls="--", lw=0.8)
        ax.axhline(LINE * (1 + f), color="r", ls="--", lw=0.8)
        ax.axhline(1, color=MUTED, ls=":", lw=1)
        if (nothing := state(runs, task, opt, "nothing")) is not None:
            ax.axhline(nothing["pgd"], color=INK, lw=1.2)
            ax.text(1.01, nothing["pgd"], "teacher", fontsize=7, va="center", transform=ax.get_yaxis_transform())
        for k, arm in enumerate(ARMS):
            pts = [(int(s) * (0.88 + 0.12 * k), r) for s in STEPS if (r := state(runs, task, opt, arm, s)) is not None]
            ax.plot([x for x, _ in pts], [r["pgd"] for _, r in pts], color=COLOR[arm], lw=1, alpha=0.6)
            for x, r in pts:
                ax.scatter(x, r["pgd"], s=45, marker=MARK[arm], zorder=3, edgecolors=COLOR[arm],
                           color=COLOR[arm] if r["p_target"] >= 0.9 else "none")
        style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([100, 1000], ["100", "1000"])
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in YTICKS if lo <= t <= hi], [f"{t:g}" for t in YTICKS if lo <= t <= hi])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{TASK_NAME[task]}, {TEACHER_NAME[opt]} teacher", fontsize=9, loc="left")
        ax.set_xlabel("teacher-training step") if i == 1 else None
        ax.set_ylabel("adversarial ratio to W₀") if j == 0 else None
    refs = [Line2D([], [], color=INK, label="teacher (do-nothing)"), Line2D([], [], color=MUTED, ls=":", label="W₀"),
            Line2D([], [], color="g", ls="--", label=f"pass line ({LINE / (1 + f):.2f})"),
            Line2D([], [], color="r", ls="--", label=f"fail line ({LINE * (1 + f):.2f}); grey: inconclusive")]
    fig.legend(handles=arm_handles(ARMS) + refs, fontsize=7.5, frameon=False, loc="lower center", ncol=4)
    fig.suptitle("Is the teacher-trained copy faithful? (hollow: held-out P(target) below 0.9)",
                 fontsize=10, x=0.01, ha="left")
    save(fig, out, "faithful.png", 0.1)


def fig_behaviour(runs: dict[str, Rec], out: Path) -> None:
    cols = (("nothing", "teacher"), ("adjoint", "adjoint split"), ("A", "copy,\ntask text"), ("B", "copy,\nPile text"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, task in zip(axes, TASKS, strict=True):
        ax.axhline(0.9, color=INK, ls="--", lw=0.8)
        ax.axhline(P_LOW, color=MUTED, ls="--", lw=0.8)
        for k, opt in enumerate(TEACHERS):
            for x, (arm, _) in enumerate(cols):
                r = state(runs, task, opt, arm, None if arm in ("nothing", "adjoint") else STEPS[-1])
                if r is not None:
                    ax.scatter(x - 0.12 + 0.24 * k, r["p_target"], s=50, marker="o" if opt == "sgd" else "D",
                               color=COLOR.get(arm, INK if arm == "nothing" else MUTED), zorder=3)
        style(ax)
        ax.set_xticks(range(len(cols)), [c[1] for c in cols])
        ax.set_ylim(0, 1.05)
        ax.set_title(TASK_NAME[task], fontsize=9, loc="left")
    axes[0].set_ylabel("held-out P(target), step 1000")
    axes[0].text(-0.4, 0.91, "0.9", fontsize=7, va="bottom")
    axes[0].text(-0.4, P_LOW + 0.01, "0.5", fontsize=7, va="bottom", color=MUTED)
    hs = [Line2D([], [], marker="o", ls="", color="none", markeredgecolor=INK, label="SGD teacher (left of each pair)"),
          Line2D([], [], marker="D", ls="", color="none", markeredgecolor=INK, label="Adam 3e-5 teacher (right)")]
    fig.legend(handles=hs, fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.suptitle("Does the copy keep the teacher's behaviour where the adjoint split loses it?", fontsize=10,
                 x=0.01, ha="left")
    save(fig, out, "behaviour.png")


def fig_gap_damage(runs: dict[str, Rec], ctl: Rec, cfg: Rec, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for j, task in enumerate(TASKS):
        for opt in TEACHERS:
            g0 = runs.get(task, {}).get("teachers", {}).get(opt, {}).get("gap_w0")
            for arm in ARMS:
                rs = [(int(s), r, damage_ratio(runs, task, opt, arm, s)) for s in STEPS
                      if (r := state(runs, task, opt, arm, s)) is not None]
                kw = {"color": COLOR[arm], "marker": MARK[arm], "ls": "-" if opt == "sgd" else "--", "ms": 6}
                if g0 and rs:
                    axes[0, j].plot([s for s, _, _ in rs], [r["teacher_gap"] / g0 for _, r, _ in rs], **kw)
                    axes[1, j].plot([s for s, _, _ in rs], [d for _, _, d in rs], **kw)
        c3 = ctl.get("C3", {})
        if task == R0 and all(s in c3 for s in STEPS):
            axes[0, j].plot([int(s) for s in STEPS], [c3[s]["teacher_gap"] / c3["gap_w0"] for s in STEPS],
                            color=COLOR["C3"], marker=MARK["C3"], lw=1)
        axes[0, j].axhline(1, color=MUTED, ls=":", lw=1)
        axes[0, j].axhline(cfg["c3_gap_max"], color=INK, ls="--", lw=0.8)
        axes[0, j].set_yscale("log")  # the teacher's own gap, 0, is below this axis
        axes[0, j].set_ylim(1e-3, 1.5)
        axes[1, j].axhspan(*P6_BAND, color="#f0efec")
        axes[1, j].axhline(1, color=INK, lw=1.2)
        axes[0, j].set_title(TASK_NAME[task], fontsize=9, loc="left")
        axes[1, j].set_xlabel("teacher-training step")
        for ax in axes[:, j]:
            style(ax)
            ax.set_xscale("log")
            ax.set_xticks([100, 1000], ["100", "1000"])
    axes[0, 0].set_ylabel("teacher gap ÷ W₀'s gap (log)")
    axes[1, 0].set_ylabel("damage elsewhere ÷ teacher's")
    refs = [Line2D([], [], color=INK, label="teacher (do-nothing): damage ratio 1, gap 0 (off the log axis)"),
            Line2D([], [], color=MUTED, ls=":", label="W₀: gap 1"),
            Line2D([], [], color=INK, ls="--", label=f"control's bar ({cfg['c3_gap_max']:g})"),
            Line2D([], [], color=MUTED, ls="-", label="SGD teacher"), Line2D([], [], color=MUTED, ls="--", label="Adam teacher")]
    fig.legend(handles=arm_handles((*ARMS, "C3")) + refs, fontsize=7.5, frameon=False, loc="lower center", ncol=5)
    fig.suptitle("How much of the teacher does the copy reach, and does it copy the teacher's side effects? "
                 "(grey band: the predicted 0.5-2)", fontsize=10, x=0.01, ha="left")
    save(fig, out, "gap_damage.png", 0.1)


# --- Amendment 1 -------------------------------------------------------------------------------------------------

VARIANTS = ("B", "B-masked", "B-dropped")  # scored
PLOTTED = ("B", "B-dropped")
VAR_NAME = {"B": "Pile text", "B-dropped": "Pile text,\ntrigger rows\nremoved"}
BOX = {"facecolor": "white", "edgecolor": "none", "pad": 0.5}
W0_P_EMO = 0.004  # the unchanged model's held-out P(target) on emoticon (plan's measurement table; not in the JSONs)


def category(r: Rec, p_min: float) -> str:
    """P1's per-teacher category: learnt (P >= p_min and 'pass'), not learnt (P < 0.5), or between."""
    return "learnt" if r["p_target"] >= p_min and r["verdict"] == "pass" else "not learnt" if r["p_target"] < P_LOW \
        else "between"


def scorecard_amend1(am: Rec, cfg: Rec) -> dict[str, Rec]:
    t, p_min = am["teachers"], cfg["p_min"]
    w0, g1 = am["W0"], am["G1"]
    out = {"C1": row("W0 reads its known Pile values", "faith_score.w0_check", {k: w0[k] for k in (
        "pgd_floor", "stochastic_floor", "pgd_rel", "ci_rel")}, "pass" if w0["pass"] else "fail"),
        "G1": row("the SGD teacher's arm B reproduces E34", "|ratio / E34's - 1| <= 0.25 and |P - E34's| <= 0.05",
                  {"pgd": g1["pgd"], "p_target": g1["p_target"], "e34": g1["e34"]}, "pass" if g1["pass"] else "fail")}
    cat = {o: {v: category(t[o][v], p_min) for v in VARIANTS[1:]} for o in TEACHERS}
    r1 = {o: {"p_target": t[o]["B-dropped"]["p_target"], "verdict": t[o]["B-dropped"]["verdict"],
              "category": cat[o]["B-dropped"]} for o in TEACHERS}
    both = [cat[o]["B-dropped"] for o in TEACHERS]
    out["P1"] = row("B-dropped learns the trigger behaviour without the trigger", f"step 1000, each teacher: P >= {p_min:g}"
                    " and 'pass' ('both'), P < 0.5 ('neither'), else 'between'", r1,
                    "both" if both == ["learnt"] * 2 else "neither" if both == ["not learnt"] * 2 else "between")
    out["P2"] = row("B-masked falls in B-dropped's category", "per teacher, the P1 category of each", cat,
                    "pass" if all(c["B-masked"] == c["B-dropped"] for c in cat.values()) else "fail")
    r3 = {o: {"dropped": t[o]["B-dropped"]["cos"]["all"], "half_A": 0.5 * t[o]["cos_ref"]["A"]["all"],
              "adjoint": t[o]["cos_ref"]["adjoint"]["all"], "null": t[o]["cos_ref"]["null"]["all"]} for o in TEACHERS}
    ok = {o: v["dropped"] >= v["half_A"] and v["dropped"] > 0.2 for o, v in r3.items()}
    out["P3"] = row("B-dropped's weight change aligns with delta-W", "cosine >= 0.5 x arm A's and > 0.2, both teachers",
                    {o: {**v, "pass": ok[o]} for o, v in r3.items()}, "pass" if all(ok.values()) else
                    "fail" if not any(ok.values()) else f"fail: one of two ({next(o for o in ok if ok[o])} passes)")
    return out


def fig_amend1(am: Rec, out: Path) -> None:
    t = am["teachers"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": (1, 1.3)})
    for g, opt in enumerate(TEACHERS):
        x0 = 3 * g
        p_t = t[opt]["repro"]["p_target"]
        ax1.plot([x0 - 0.4, x0 + 1.4], [p_t] * 2, color=INK, lw=1.4)
        ax1.text(x0 + 1.45, p_t, f"teacher\n{p_t:.3f}", fontsize=7, va="center")
        for k, v in enumerate(PLOTTED):
            r = t[opt][v]
            ax1.plot([x0 + k] * 2, [r["p_target_100"], r["p_target"]], color=COLOR["B"], lw=1, alpha=0.6)
            ax1.scatter(x0 + k, r["p_target_100"], s=45, marker="o", color="none", edgecolors=COLOR["B"], zorder=3)
            ax1.scatter(x0 + k, r["p_target"], s=45, marker="o", color=COLOR["B"], zorder=3)
            ax1.text(x0 + k + 0.12, r["p_target"] - 0.02, f"{r['p_target']:.3f}", fontsize=7, va="top", bbox=BOX)
        ax1.text(x0 + 0.5, -0.24, f"{TEACHER_NAME[opt]} teacher", ha="center", fontsize=8.5,
                 transform=ax1.get_xaxis_transform())
    ax1.axhline(0.9, color=INK, ls="--", lw=0.8)
    ax1.axhline(W0_P_EMO, color=MUTED, ls=":", lw=1)
    ax1.text(5.35, 0.9, "0.9", fontsize=7, va="center", ha="right", bbox=BOX)
    ax1.text(-0.45, W0_P_EMO + 0.01, f"unchanged model {W0_P_EMO:g}", fontsize=7, color=MUTED, va="bottom")
    ax1.set_xticks([3 * g + k for g in range(2) for k in range(len(PLOTTED))],
                   [VAR_NAME[v] for _ in TEACHERS for v in PLOTTED], fontsize=7)
    ax1.set_xlim(-0.6, 5.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("held-out P(target)")
    ax1.set_title("Is the behaviour learnt without the trigger? (hollow: step 100, filled: step 1000)", fontsize=9,
                  loc="left")
    style(ax1)
    cos_cols = (*PLOTTED, "A", "adjoint", "null")
    cos_name = {**VAR_NAME, "A": "task text", "adjoint": "adjoint\nsplit", "null": "random\ndirection"}
    cos_color = {"A": COLOR["A"], "adjoint": MUTED, "null": INK}
    for g, opt in enumerate(TEACHERS):
        x0 = 6 * g
        for k, c in enumerate(cos_cols):
            v = (t[opt][c]["cos"] if c in PLOTTED else t[opt]["cos_ref"][c])["all"]
            ax2.bar(x0 + k, v, width=0.7, color=cos_color.get(c, COLOR["B"]))
            ax2.text(x0 + k, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5)
        ax2.text(x0 + 2, -0.24, f"{TEACHER_NAME[opt]} teacher", ha="center", fontsize=8.5,
                 transform=ax2.get_xaxis_transform())
    ax2.set_xticks([6 * g + k for g in range(2) for k in range(len(cos_cols))],
                   [cos_name[c] for _ in TEACHERS for c in cos_cols], fontsize=7)
    ax2.set_ylabel("cosine of the weight change with δW")
    ax2.set_title("Does the copy's weight change point along the fine-tune's δW? (orange, blue: copies)", fontsize=9,
                  loc="left")
    style(ax2)
    ax2.grid(False, axis="x")
    fig.suptitle("Emoticon: copies trained on Pile text, with and without the rows that hold the trigger",
                 fontsize=10, x=0.01, ha="left")
    save(fig, out, "amend1.png", 0.04)


def main_amend1(results: Path, out: Path | None) -> None:
    am = json.loads((results / "teacher_training_amend1.json").read_text())
    sc = scorecard_amend1(am, json.loads((results / "predictions_amend1.json").read_text())["config"])
    for k, r in sc.items():
        print(f"{k:4} {r['verdict']:<28} {r['claim']}\n     rule: {r['rule']}\n     read: {json.dumps(r['read'], default=float)}")
    (out or results).mkdir(parents=True, exist_ok=True)
    (out or results).joinpath("scorecard_amend1.json").write_text(json.dumps(sc, indent=1, default=float))
    fig_amend1(am, out or HERE / "figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=None, help="figures and scorecard here (default figures/, results/)")
    ap.add_argument("--amend1", action="store_true", help="Amendment 1's figure and scorecard only")
    args = ap.parse_args()
    if args.amend1:
        return main_amend1(args.results, args.out)
    runs, ctl, pred = load(args.results)
    cfg = pred["config"]
    sc = {**controls(ctl, cfg), **(predictions(runs, cfg) if runs else {})}
    for k, r in sc.items():
        print(f"{k:4} {r['verdict']:<28} {r['claim']}\n     rule: {r['rule']}\n     read: {json.dumps(r['read'], default=float)}")
    if (rec := sc["C3"]["read"]) is not None:
        print(f"C3 offset recovery at step 1000: {rec['offset_recovery']:.4f}")
    (args.out or args.results).mkdir(parents=True, exist_ok=True)
    (args.out or args.results).joinpath("scorecard.json").write_text(json.dumps(sc, indent=1, default=float))
    if runs:
        figs = args.out or HERE / "figures"
        fig_faithful(runs, figs)
        fig_behaviour(runs, figs)
        fig_gap_damage(runs, ctl, cfg, figs)


if __name__ == "__main__":
    main()
