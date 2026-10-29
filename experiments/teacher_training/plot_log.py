"""The research log's figures for E34, E43 and E41, redrawn from saved results with plain labels (no model).

E34
  log_faithful.png        per fine-tune (three tasks, SGD and Adam teachers): the adversarial ratio of the fine-tune left
                          in Δ (do-nothing) and of its two copies (task text, Pile text) at teacher-training step 1000 as
                          bars from 1, step 100 as dots, with the pass and fail lines.
                          Reads results/teacher_training_<task>.json.
E43 (reads results/teacher_training_e43.json)
  log_e43_check.png       the adversarial ratio at fine-tune steps 10 and 1000 of do-nothing, the least-squares split,
                          teacher training (Δ untouched) and teacher training plus the remainder in Δ, bars from 1.
  log_e43_removal.png     KL from the fine-tune to the state with the remainder taken out of Δ, as a share of the
                          unchanged model's KL, on task text and on Pile text (b_task, b_pile).
  log_e43_size.png        squared size of what was added to Δ over the fine-tune's squared size (delta_share), with the
                          cosine of the remainder and teacher training's own weight change (cos_R_copy).
E41 (reads results/teacher_training_e41.json, _e41_amend1.json, _e41_amend2.json)
  log_e41_step10.png      the adversarial ratio at fine-tune step 10, the one step where do-nothing fails: do-nothing,
                          teacher training from task text (emoticon at three batch orders), from Pile text, and the
                          short chain's step-10 link; the " which" task beside it.
  log_e41_chain.png       the short chain (seven checkpoints, 100 student steps per link) against do-nothing at the same
                          checkpoints, with the checkpoints where do-nothing passes shaded.
  log_e41_behaviour.png   held-out P(target) of every teacher-trained state against the checkpoint it copies.
  log_e41_pileup.png      size and damage at step 1000 of the two chains, as multiples of a single 1,000-step copy's.

Usage (repo root):  python experiments/teacher_training/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from plot import (
    COLOR,
    EMO,
    HERE,
    INK,
    LINE,
    MUTED,
    R0,
    R1,
    TASKS,
    TEACHERS,
    floor,
    load,
    state,
    style,
)

TASK = {R1: "' which' → ' was'", R0: "' it' → ' is'", EMO: "emoticon"}
TEACHER = {"sgd": "SGD", "adam": "Adam"}
FIG, RES = HERE / "figures", HERE / "results"
TT, GREEN, PURPLE, ORANGE = COLOR["A"], "#1a9850", "#7b52ab", "#eb6834"
Rec = dict[str, Any]


def read(name: str) -> Rec:
    return json.loads((RES / f"teacher_training_{name}.json").read_text())


def lines(ax: Axes, f: float) -> None:
    """The pass and fail lines of a job whose unchanged model has adversary-seed floor f, labelled in the legend."""
    for y, txt, c in ((LINE * (1 + f), "fails at or above", "r"), (LINE / (1 + f), "passes at or below", "g")):
        ax.axhline(y, color=c, ls="--", lw=0.9, zorder=1, label=f"check {txt} {y:.2f}")
    ax.axhline(1, color=INK, lw=0.8, ls=":", zorder=1)


def bar(ax: Axes, x: float, v: float, w: float, c: str, lab: str | None, top: float = 99.0) -> None:
    """A bar from 1 to v, clipped at `top` with its value written on it."""
    ax.bar(x, min(v, top) - 1, w, bottom=1, color=c, label=lab, zorder=2)
    y = min(v, top) + (0.03 if v >= 1 else -0.03)
    ax.text(x, y, f"{v:.2f}" + (" (off scale)" if v > top else ""), ha="center", va="bottom" if v >= 1 else "top",
            fontsize=7.5)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / name}")


# --- E34 ---------------------------------------------------------------------------------------------------------

def pgd(runs: dict[str, Rec], task: str, opt: str, arm: str, step: str | None = None) -> float:
    s = state(runs, task, opt, arm, step)
    assert s is not None, (task, opt, arm, step)
    return s["pgd"]


def e34_faithful() -> None:
    runs, _, _ = load(RES)
    f = floor(runs)
    pass_line, fail_line = LINE / (1 + f), LINE * (1 + f)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bars = (("nothing", "the fine-tune left in Δ (do-nothing)", "#b9b8b3"),
            ("A", "copy trained on the task's text", COLOR["A"]),
            ("B", "copy trained on ordinary Pile text", COLOR["B"]))
    labels = []
    for i, (task, opt) in enumerate((t, o) for t in TASKS for o in TEACHERS):
        labels.append(f"{TASK[task]}\n{TEACHER[opt]} fine-tune")
        for j, (arm, lab, col) in enumerate(bars):
            x = i + (j - 1) * 0.26
            if arm == "nothing":
                v = pgd(runs, task, opt, "nothing")
            else:
                v = pgd(runs, task, opt, arm, "1000")
                ax.scatter([x], [pgd(runs, task, opt, arm, "100")], s=16, color="k", zorder=4,
                           label="the same copy at step 100" if i == 0 and arm == "A" else None)
            ax.bar(x, v - 1, 0.24, bottom=1, color=col, label=lab if i == 0 else None, zorder=2)
            ax.text(x, v + (0.015 if v >= 1 else -0.015), f"{v:.2f}", ha="center", va="bottom" if v >= 1 else "top",
                    fontsize=7)
    ax.axhline(1, color=INK, lw=0.8, ls=":", zorder=1)
    ax.axhspan(pass_line, fail_line, color="#f0efec", zorder=0)
    ax.axhline(pass_line, color="g", ls="--", lw=0.9, label=f"pass line ({pass_line:.2f}); grey band: inconclusive")
    ax.axhline(fail_line, color="r", ls="--", lw=0.9, label=f"fail line ({fail_line:.2f})")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(0.8, 1.75)
    ax.set_ylabel("adversarial ratio\n(1 = unchanged decomposition)")
    ax.set_title("Every copy passes the check after 1,000 steps, including copies of the three fine-tunes that are "
                 "inconclusive themselves", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "log_faithful.png", dpi=150, bbox_inches="tight")


# --- E43 ---------------------------------------------------------------------------------------------------------

E43_STATES = (
    ("nothing", "doing nothing: the whole fine-tune in Δ", MUTED),
    ("least_squares", "least-squares split", PURPLE),
    ("teacher_training", "teacher training, Δ untouched", TT),
    ("plus_remainder", "teacher training plus the remainder in Δ", ORANGE),
)
E43_STEPS = ("10", "1000")


def e43_check(res: Rec) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w, top = 0.19, 4.4
    for i, k in enumerate(E43_STEPS):
        for j, (name, lab, c) in enumerate(E43_STATES):
            bar(ax, i + (j - 1.5) * w, res["steps"][k][name]["pgd"], w * 0.92, c, lab if i == 0 else None, top)
    lines(ax, res["W0"]["pgd_floor"])
    ax.set_xticks(range(len(E43_STEPS)), [f"fine-tune stopped at step {k}" for k in E43_STEPS])
    ax.set_ylim(0.8, top + 0.35)
    ax.set_ylabel("adversarial ratio\n(1 = unchanged decomposition)")
    ax.set_title("Adding the remainder to Δ fails the check at both steps, on one adversary start", fontsize=10.5)
    style(ax)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "log_e43_check.png")


def e43_removal(res: Rec) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    w = 0.36
    groups = [(t, k) for t in ("b_task", "b_pile") for k in E43_STEPS]
    text = {"b_task": "task text", "b_pile": "Pile text"}
    for i, (key, k) in enumerate(groups):
        for j, (name, lab) in enumerate((("plus_remainder", "teacher training's remainder"),
                                         ("least_squares", "least-squares split's part in Δ"))):
            v = res["steps"][k][name][key]
            c = ORANGE if name == "plus_remainder" else PURPLE
            ax.bar(i + (j - 0.5) * w, v, w * 0.92, color=c, label=lab if i == 0 else None, zorder=2)
            ax.text(i + (j - 0.5) * w, v + 0.008, f"{v:.3f}" if v < 0.01 else f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5)
    ax.axvline(1.5, color=MUTED, lw=0.6)
    ax.set_xticks(range(len(groups)), [f"{text[t]}\nstep {k}" for t, k in groups])
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("share of the fine-tune's effect lost\nwhen that part is taken out of Δ")
    ax.set_title("Taking the remainder out loses almost nothing on task text, a quarter to two-fifths on Pile text",
                 fontsize=10.5)
    style(ax)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_e43_removal.png")


def e43_size(res: Rec) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    w = 0.32
    for i, k in enumerate(E43_STEPS):
        for j, name in enumerate(("least_squares", "plus_remainder")):
            s = res["steps"][k][name]
            c = PURPLE if name == "least_squares" else ORANGE
            lab = {"least_squares": "least-squares split's part in Δ", "plus_remainder": "the remainder"}[name]
            ax.bar(i + (j - 0.5) * w, s["delta_share"], w * 0.92, color=c, label=lab if i == 0 else None, zorder=2)
            txt = f"{s['delta_share']:.2f}"
            if name == "plus_remainder":
                txt += f"\ncosine with teacher\ntraining's change {s['cos_R_copy']:.2f}"
            ax.text(i + (j - 0.5) * w, s["delta_share"] + 0.08, txt, ha="center", va="bottom", fontsize=7.5)
    ax.axhline(1, color=INK, ls=":", lw=1)
    ax.text(-0.55, 1.06, "the fine-tune's own update (doing nothing puts all of it in Δ)", fontsize=7.5, va="bottom")
    ax.set_xticks(range(len(E43_STEPS)), [f"fine-tune stopped at step {k}" for k in E43_STEPS])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 7.2)
    ax.set_ylabel("squared size of what goes into Δ,\nover the fine-tune's squared size")
    ax.set_title("At step 10 the remainder is five times the fine-tune's squared size and mostly undoes teacher "
                 "training's change",
                 fontsize=10.5)
    style(ax)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "log_e43_size.png")


# --- E41 ---------------------------------------------------------------------------------------------------------

LINKS = ("1", "3", "10", "30", "100", "300", "1000")
WHICH = "r1_t534_v369"


def e41_step10(e41: Rec, a1: Rec, a2: Rec) -> None:
    emo, wh = e41["tasks"]["emoticon"], e41["tasks"][WHICH]
    rows = (
        ("emoticon", "doing nothing", emo["teachers"]["10"]["nothing"]["pgd"], MUTED),
        ("emoticon", "teacher training,\ntask text,\nbatch order 0", emo["A"]["oneshot"]["10"]["pgd"], TT),
        ("emoticon", "same,\nbatch order 1", a1["copies"]["1"]["pgd"], TT),
        ("emoticon", "same,\nbatch order 2", a1["copies"]["2"]["pgd"], TT),
        ("emoticon", "teacher training,\nPile text", emo["B"]["oneshot"]["10"]["pgd"], ORANGE),
        ("emoticon", "short chain's\nstep-10 link", a2["links"]["10"]["pgd"], GREEN),
        ("which", "doing nothing", wh["teachers"]["10"]["nothing"]["pgd"], MUTED),
        ("which", "teacher training,\ntask text", wh["A"]["oneshot"]["10"]["pgd"], TT),
    )
    xs = [i + (0.6 if t == "which" else 0) for i, (t, *_) in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for x, (_, _, v, c) in zip(xs, rows, strict=True):
        bar(ax, x, v, 0.7, c, None)
    lines(ax, e41["W0"]["pgd_floor"])
    ax.set_xticks(xs, [r[1] for r in rows], fontsize=7.5)
    ax.axvline(5.8, color=MUTED, lw=0.6)
    ax.text(2.5, 2.75, "emoticon task", ha="center", fontsize=9)
    ax.text(7.1, 2.75, "' which' → ' was' task", ha="center", fontsize=9)
    ax.set_xlim(-0.6, 8.5)
    ax.set_ylim(0.8, 2.9)
    ax.set_ylabel("adversarial ratio\n(1 = unchanged decomposition)")
    ax.set_title("At step 10, where doing nothing fails, teacher training sits around the pass line: passes and inconclusives",
                 fontsize=10.5)
    style(ax)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.2, 0.62))
    save(fig, "log_e41_step10.png")


def e41_chain(a2: Rec) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    sees = a2["score"]["check_sees_the_update"]
    for k in LINKS:
        if not sees[k]:
            ax.axvspan(float(k) / 1.4, float(k) * 1.4, color="#f0efec", zorder=0)
    xs = [float(k) for k in LINKS]
    ax.plot(xs, [a2["teachers"][k]["nothing"]["pgd"] for k in LINKS], "s-", color=MUTED, ms=6, lw=1,
            label="doing nothing: the fine-tune at that step left in Δ")
    ax.plot(xs, [a2["links"][k]["pgd"] for k in LINKS], "o-", color=GREEN, ms=7, lw=1.6, zorder=4,
            label="short chain: teacher training, 100 steps per checkpoint, each from the last")
    lines(ax, a2["W0"]["pgd_floor"])
    ax.text(170, 2.0, "shaded: doing nothing passes,\nso the check cannot tell", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(xs, LINKS)
    ax.minorticks_off()
    ax.set_xlim(0.7, 1500)
    ax.set_ylim(0.8, 2.75)
    ax.set_xlabel("emoticon fine-tune stopped at step")
    ax.set_ylabel("adversarial ratio\n(1 = unchanged decomposition)")
    ax.set_title("One emoticon run: the short chain passes at steps 10 and 1,000, the two checkpoints the check can tell",
                 fontsize=10.5)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "log_e41_chain.png")


def e41_behaviour(e41: Rec, a2: Rec) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    emo, wh = e41["tasks"]["emoticon"], e41["tasks"][WHICH]
    steps = ("10", "100", "1000")
    series = (
        ("emoticon, task text", [float(k) * 0.9 for k in steps], [emo["A"]["oneshot"][k]["p_target"] for k in steps],
         TT, "o"),
        ("emoticon, Pile text", [float(k) for k in steps], [emo["B"]["oneshot"][k]["p_target"] for k in steps],
         ORANGE, "o"),
        ("' which' → ' was', task text", [float(k) * 1.1 for k in steps],
         [wh["A"]["oneshot"][k]["p_target"] for k in steps], TT, "^"),
        ("emoticon, short chain", [float(k) for k in LINKS], [a2["links"][k]["p_target"] for k in LINKS], GREEN, "D"),
    )
    for lab, xs, ys, c, m in series:
        ax.plot(xs, ys, marker=m, color=c, ms=6, lw=0.8, label=lab, mfc=c if m != "^" else "white")
    ax.axhline(0.04, color=MUTED, ls="--", lw=0.9)
    ax.text(1.5, 0.06, "unchanged model: 0.004 on emoticon, 0.04 on ' which'", fontsize=7.5, color=MUTED)
    ax.axhline(0.9, color=INK, ls=":", lw=0.9)
    ax.text(1.5, 0.84, "0.90", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([float(k) for k in LINKS], LINKS)
    ax.minorticks_off()
    ax.set_xlim(0.7, 1500)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("fine-tune stopped at step")
    ax.set_ylabel("probability of the target\nat the trigger, held-out text")
    ax.set_title("Every teacher-trained state keeps the target probability at 0.90 or more", fontsize=10.5)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="center right", title="teacher training of", title_fontsize=8)
    save(fig, "log_e41_behaviour.png")


def e41_pileup(e41: Rec, a2: Rec) -> None:
    a = e41["tasks"]["emoticon"]["A"]
    one, long_, short = a["oneshot"]["1000"], a["chain"]["1000"], a2["links"]["1000"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.34
    for i, key in enumerate(("size", "damage")):
        for j, (s, lab, c) in enumerate(((long_, "chain of three 1,000-step links", TT),
                                         (short, "chain of seven 100-step links", GREEN))):
            bar(ax, i + (j - 0.5) * w, s[key] / one[key], w * 0.92, c, lab if i == 0 else None)
    ax.axhline(1, color=INK, ls=":", lw=1)
    ax.axhline(1.25, color="r", ls="--", lw=0.9)
    ax.text(1.55, 1.26, "limit set\nbefore the\nshort chain", color="r", fontsize=7.5, va="bottom", ha="right")
    ax.set_xticks([0, 1], ["largest weight change", "damage elsewhere\n(KL from the unchanged model\noff the trigger)"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0.6, 1.65)
    ax.set_ylabel("at step 1,000, multiple of a single\n1,000-step teacher training's")
    ax.set_title("Seven 100-step links end inside the limit; three 1,000-step links had grown to 1.4 and 1.5 times", fontsize=10.5)
    style(ax)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_e41_pileup.png")


def main() -> None:
    e34_faithful()
    e43 = read("e43")
    assert e43["done"]
    e43_check(e43)
    e43_removal(e43)
    e43_size(e43)
    e41, a1, a2 = read("e41"), read("e41_amend1"), read("e41_amend2")
    assert e41["card"] == a1["card"] == a2["card"]
    e41_step10(e41, a1, a2)
    e41_chain(a2)
    e41_behaviour(e41, a2)
    e41_pileup(e41, a2)


if __name__ == "__main__":
    main()
