"""Figures for E42, teacher training on a behaviour the model lacks (plan:
reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md), read from the three results JSONs and, for the numbers the
JSONs do not keep (the trigger's P(target) under the unmasked and CI-masked models, and the Pile CI-masked ratio), from
the run logs' route-split lines. The scoring is the run's own (`newskill.py`, in each results file under "score").

  e42_copy.png     held-out P(target) and the Pile adversarial ratio of the copy trained on task text (arm A), beside
                   the unchanged model, the fine-tune left in Delta (do-nothing) and its least-squares split
  e42_trigger.png  the CI-masked model on Pile text (ratio to the unchanged model) and at the trigger (P(target), with
                   each state's own full-model P(target) above it), for the same three states
  e42_routes.png   the masked-off share of the copy's gain at the trigger at steps 100 and 1000, beside E10's
                   label-trained component training, the least-squares split and do-nothing, against the filed lines
  e42_pile.png     the copy trained on Pile text only (arm B): its held-out P(target) at 100 and 1000 against the
                   filed 0.5 and 0.9 lines; and the teacher's P(value) right after a key word in plain Pile text

Usage (repo root): uv run python experiments/teacher_training/plot_e42.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = ("rand32", "rand1", "ident32")  # the two cipher tasks, then the identity control
TASK_NAME = {"rand32": "32 random pairs", "rand1": "1 random pair", "ident32": "identity control\n(32 keys copied)"}
JOB = {"rand32": 40166, "ident32": 40194, "rand1": 40195}  # cluster/logs/commute-teacher-e42-<job>.out
W0_P = {"rand32": 1.5e-5, "ident32": 1.7e-4, "rand1": 1.7e-4}  # unchanged model's held-out P(target), E10's route split
E10_LABEL = {"rand32": 0.75, "ident32": 0.21, "rand1": 0.86}  # E10's label-trained masked-off share (plan, Stage 2)
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COLOR = {"nothing": MUTED, "least_squares": "#b8326b", "A": "#2a78d6", "B": "#eb6834", "label": "#1baf7a"}
MARK = {"nothing": "s", "least_squares": "D", "A": "o", "B": "s", "label": "^"}
NAME = {"nothing": "do-nothing: fine-tune left in Δ", "least_squares": "least-squares split (control)",
        "A": "copy trained on task text", "B": "copy trained on Pile text", "label": "E10: trained on the task's labels"}
LOGNAME = {"nothing": "do-nothing", "least_squares": "least-squares", "A": "A@1000", "B": "B@1000"}
ROUTE = re.compile(r"\[(\w+) ([\w@-]+)\] gain .*\| P\(o\) target ([\d.]+) unmasked ([\d.]+) ci ([\d.]+) \| "
                   r"Delta change [\d.]+(?: \| pile (\{.*\}))?$")
Rec = dict[str, Any]


def load() -> tuple[dict[str, Rec], dict[str, dict[str, Rec]]]:
    """Results per task, and per task and state the log's trigger cells (target, unmasked, ci) and Pile ratios."""
    res = {t: json.loads((HERE / "results" / f"teacher_training_e42_{t}.json").read_text()) for t in TASKS}
    logs: dict[str, dict[str, Rec]] = {t: {} for t in TASKS}
    for t, job in JOB.items():
        for line in (ROOT / "cluster" / "logs" / f"commute-teacher-e42-{job}.out").read_text().splitlines():
            if m := ROUTE.search(line):
                assert m[1] == t, (m[1], t)
                logs[t][m[2]] = {"target": float(m[3]), "unmasked": float(m[4]), "ci": float(m[5]),
                                 "pile": json.loads(m[6]) if m[6] else None}
    return res, logs


def style(ax: Axes, x_labels: bool = True) -> None:
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if x_labels:
        ax.set_xticks(range(len(TASKS)), [TASK_NAME[t] for t in TASKS], fontsize=9)
        ax.set_xlim(-0.5, len(TASKS) - 0.5)


def dot(ax: Axes, x: float, y: float, key: str, hollow: bool = False, label: str | None = None) -> None:
    c = COLOR[key]
    ax.scatter(x, y, s=52, marker=MARK[key], facecolors="white" if hollow else c, edgecolors=c, lw=1.5, zorder=3,
               label=label)


def w0_dashes(ax: Axes, label: str) -> None:
    for i, t in enumerate(TASKS):
        ax.hlines(W0_P[t], i - 0.4, i + 0.4, color=INK, ls="--", lw=1, label=label if i == 0 else None)


def fig_copy(res: dict[str, Rec], out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    states, off = ("nothing", "least_squares", "A"), (-0.2, 0.0, 0.2)
    for i, t in enumerate(TASKS):
        r = res[t]
        for s, o in zip(states, off, strict=True):
            rec = r[s] if s != "A" else r["A"]["1000"]
            dot(a1, i + o, rec["p_target"], s, label=NAME[s] if i == 0 else None)
            dot(a2, i + o, rec["pgd"], s, label=NAME[s] if i == 0 else None)
    w0_dashes(a1, "unchanged model")
    a1.set_yscale("log")
    a1.set_ylim(5e-6, 2)
    a1.set_ylabel("held-out P(target) at the trigger")
    a1.set_title("Behaviour: every state keeps it", fontsize=10)
    floor = res["rand32"]["W0"]["pgd_floor"]
    assert all(res[t]["W0"]["pgd_floor"] == floor for t in TASKS)
    a2.axhspan(1, 1 + floor, color=GRID, zorder=0)
    a2.text(2.45, 1 + floor / 2, "unchanged model\nacross 3 adversary seeds", fontsize=7.5, color=MUTED, ha="right",
            va="center")
    for y, txt in ((LINE / (1 + floor), "pass at or below"), (LINE * (1 + floor), "fail at or above")):
        a2.axhline(y, color=INK, ls=":", lw=1)
        a2.text(-0.45, y + 0.015, f"{txt} {y:.2f}", fontsize=7.5, va="bottom")
    a2.set_ylim(0.8, 2.15)
    a2.set_ylabel("Pile adversarial ratio to the unchanged model")
    a2.set_title("Faithfulness: only the copy passes", fontsize=10)
    for ax in (a1, a2):
        style(ax)
    a1.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.suptitle("Does teacher training copy a behaviour the model lacked, faithfully and with Δ untouched?\n"
                 "Yes on all three tasks: the copy keeps 0.96 to 1.00 and passes; Δ takes under 1e-8 of its change",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e42_copy.png", dpi=150)
    plt.close(fig)


def fig_trigger(logs: dict[str, dict[str, Rec]], out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    states, off = ("nothing", "least_squares", "A"), (-0.2, 0.0, 0.2)
    for i, t in enumerate(TASKS):
        for s, o in zip(states, off, strict=True):
            c = logs[t][LOGNAME[s]]
            dot(a1, i + o, c["pile"]["ci_masked_kl"], s, label=NAME[s] if i == 0 else None)
            a2.vlines(i + o, c["ci"], c["target"], color=COLOR[s], lw=1, alpha=0.5)
            dot(a2, i + o, c["target"], s, hollow=True, label="full model (with Δ)" if (i, s) == (0, "A") else None)
            dot(a2, i + o, c["ci"], s, label="CI-masked subcomponents" if (i, s) == (0, "A") else None)
    a1.axhline(1, color=INK, ls="--", lw=1)
    a1.text(-0.45, 1.005, "unchanged model", fontsize=7.5, va="bottom")
    a1.set_ylim(0.8, 1.2)
    a1.set_ylabel("CI-masked KL on Pile text, ratio to the unchanged model")
    a1.set_title("On Pile text: every state as good as the unchanged model", fontsize=10)
    w0_dashes(a2, "unchanged model, full")
    a2.set_yscale("log")
    a2.set_ylim(5e-6, 2)
    a2.set_ylabel("P(target) at the trigger, held-out windows")
    a2.set_title("At the trigger: the CI-masked model keeps the behaviour\non the identity control only", fontsize=10)
    for ax in (a1, a2):
        style(ax)
    a1.legend(fontsize=7.5, loc="upper left", frameon=False)
    a2.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.suptitle("Does the CI-masked model carry the new behaviour? On ordinary text yes, at the trigger no\n"
                 "Copy on 32 random pairs: 0.96 with Δ, 0.007 CI-masked; do-nothing and the split read the same",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e42_trigger.png", dpi=150)
    plt.close(fig)


def fig_routes(res: dict[str, Rec], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, t in enumerate(TASKS):
        r = res[t]
        dot(ax, i - 0.24, r["nothing"]["share_e_gate"], "nothing", label=NAME["nothing"] if i == 0 else None)
        dot(ax, i - 0.12, r["least_squares"]["share_e_gate"], "least_squares",
            label=NAME["least_squares"] if i == 0 else None)
        dot(ax, i, E10_LABEL[t], "label", label=NAME["label"] if i == 0 else None)
        dot(ax, i + 0.12, r["A"]["100"]["share_e_gate"], "A", hollow=True, label="copy, step 100" if i == 0 else None)
        dot(ax, i + 0.24, r["A"]["1000"]["share_e_gate"], "A", label="copy, step 1000 (scored)" if i == 0 else None)
    for y, txt in ((0.5, "masked-off at or above 0.5"), (0.25, "not masked-off at or below 0.25")):
        ax.axhline(y, color=INK, ls=":", lw=1)
        ax.text(2.5, y + 0.015, txt, fontsize=7.5, ha="right", va="bottom")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-0.75, 1.05)
    ax.set_ylabel("masked-off share of the gain at the trigger")
    style(ax)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.set_title("Does the copy's gain run through subcomponents Γ keeps off at the trigger?\n"
                 "Yes on both cipher tasks, 0.84 and 0.92, as with label training; 0.31 on the identity control",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "e42_routes.png", dpi=150)
    plt.close(fig)


def fig_pile(res: dict[str, Rec], out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6))
    for i, t in enumerate(TASKS):
        r = res[t]
        a1.scatter(i - 0.2, r["teacher"]["p_target_1000"], s=52, marker="*", color=INK, zorder=3,
                   label="the fine-tune it copies" if i == 0 else None)
        dot(a1, i, r["A"]["1000"]["p_target"], "A", label="copy on task text, step 1000" if i == 0 else None)
        dot(a1, i + 0.2, r["B"]["100"]["p_target"], "B", hollow=True, label="copy on Pile text, step 100" if i == 0 else None)
        dot(a1, i + 0.2, r["B"]["1000"]["p_target"], "B", label="copy on Pile text, step 1000" if i == 0 else None)
        pr = r["pile_reading"]
        for o, key, c in ((-0.12, "p_value_w0", MUTED), (0.12, "p_value_teacher", COLOR["B"])):
            a2.bar(i + o, pr[key], width=0.22, color=c, label=("unchanged model" if key.endswith("w0") else
                                                               "the fine-tune") if i == 0 else None)
        a2.text(i, 1.6, f"{pr['key_positions']:,} key\npositions", fontsize=7.5, ha="center", va="bottom")
        a2.text(i + 0.12, pr["p_value_teacher"] * 1.3, f"×{pr['p_value_teacher'] / pr['p_value_w0']:,.0f}",
                fontsize=8, ha="center", va="bottom")
    w0_dashes(a1, "unchanged model")
    for y, txt in ((0.9, "learnt needs ≥ 0.9 and a pass"), (0.5, "neither below 0.5")):
        a1.axhline(y, color=INK, ls=":", lw=1)
        a1.text(2.48, y, txt, fontsize=7.5, va="bottom", ha="right")
    a1.set_yscale("logit")  # spreads both ends: the unchanged model near 1e-5 and the 0.5 and 0.9 lines
    a1.set_ylim(5e-6, 1 - 5e-6)
    ticks = (1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999)
    a1.set_yticks(ticks, [f"{v:g}" for v in ticks])
    a1.minorticks_off()
    a1.set_ylabel("held-out P(target) at the trigger")
    a1.set_title("The Pile-text copy learns the one-pair cipher only", fontsize=10)
    a2.set_yscale("log")
    a2.set_ylim(1e-5, 10)
    a2.set_ylabel("mean P(the key's value) right after a key word\nin plain Pile text, no marker")
    a2.set_title("Plain text shows the fine-tune's one-pair map", fontsize=10)
    for ax in (a1, a2):
        style(ax)
    a1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    a2.legend(fontsize=7.5, loc="center right", frameon=False)
    fig.suptitle("Can ordinary Pile text alone teach the copy a behaviour the model lacked?\n"
                 "For one pair, yes: 0.91 and a pass, filed as 'neither'; the fine-tune shows the pair in plain text",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e42_pile.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res, logs = load()
    for t in TASKS:  # the log's numbers and the JSON's agree where both exist
        assert abs(logs[t]["A@1000"]["target"] - res[t]["A"]["1000"]["p_target"]) < 1e-3, t
        assert abs(logs[t]["A@1000"]["pile"]["pgd_kl"] - res[t]["A"]["1000"]["pgd"]) < 1e-3, t
    fig_copy(res, args.out)
    fig_trigger(logs, args.out)
    fig_routes(res, args.out)
    fig_pile(res, args.out)
    for t in TASKS:
        c = logs[t]["A@1000"]
        print(f"{t:8s} copy at the trigger: full {c['target']:.3f} unmasked {c['unmasked']:.3f} CI-masked {c['ci']:.3f}"
              f" | Pile CI-masked ratio {c['pile']['ci_masked_kl']:.3f}")


if __name__ == "__main__":
    main()
