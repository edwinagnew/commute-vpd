"""Figures for E42 Amendment 2, each masked term alone with nothing to copy (plan:
reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md -> "Amendment 2"), read from the results JSONs only. The
scoring is the run's own (`newskill_amend2.py`, under "score" in its results file).

  e42_amend2_controls.png  the adversarial and stochastic Pile ratios of the three Amendment 2 controls (teacher = W0),
                           beside the unchanged model's three adversary seeds, E42's copy with no masked terms (passes)
                           and Amendment 1's control with both terms, against the pass and fail lines
  e42_amend2_curves.png    each control's masked terms over training in 50-step blocks, as a share of their first
                           block, with Amendment 1's last block beside them; and the unmasked term, the target model's
                           KL from the teacher, against its value at step 1

Usage (repo root): uv run python experiments/teacher_training/plot_e42_amend2.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
CONTROLS = ("M0_ci", "M0_st", "M0_trig")
COLOR = {"W0": MUTED, "copy": "#2a78d6", "A1": "#7a52c7", "M0_ci": "#eb6834", "M0_st": "#1baf7a",
         "M0_trig": "#b8326b"}
NAME = {"W0": "unchanged\nmodel, three\nadversary seeds", "copy": "E42's copy,\nno masked\nterms",
        "A1": "both terms,\nAmendment 1", "M0_ci": "CI-masked\nterm alone", "M0_st": "stochastic\nterm alone",
        "M0_trig": "both terms,\ntrigger\nonly"}
STEP1_UNMASKED = 0.00163  # every control's step-1 unmasked term (job 40497 log; = job 40338's)
Rec = dict[str, Any]


def read(name: str) -> Rec:
    return json.loads((RES / name).read_text())


def load() -> tuple[Rec, Rec, Rec]:
    return (read("teacher_training_e42_amend2_rand32.json"), read("teacher_training_e42_amend1_rand32.json"),
            read("teacher_training_e42_rand32.json"))


def fig_controls(a2: Rec, a1: Rec, e42: Rec, out: Path) -> None:
    """Adversarial (log) and stochastic Pile ratios per state, the W0 seeds as three dots."""
    w0 = a2["W0"]["pile"]
    f = a2["W0"]["pgd_floor"]
    keys = ("W0", "copy", "A1", *CONTROLS)
    pgd = {"copy": e42["A"]["1000"]["pgd"], "A1": a1["M0"]["pgd"], **{k: a2[k]["pgd"] for k in CONTROLS}}
    sto = {"copy": e42["A"]["1000"]["stochastic"], "A1": a1["M0"]["stochastic"],
           **{k: a2[k]["stochastic"] for k in CONTROLS}}
    w0_pgd = [w0[s]["pgd_kl"] / w0["0"]["pgd_kl"] for s in ("0", "1", "2")]
    w0_sto = [w0[s]["stochastic_kl"] / w0["0"]["stochastic_kl"] for s in ("0", "1", "2")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, vals, w0v in ((ax1, pgd, w0_pgd), (ax2, sto, w0_sto)):
        ax.scatter([0] * 3, w0v, s=40, color=COLOR["W0"], zorder=3)
        for i, k in enumerate(keys[1:], start=1):
            ax.scatter(i, vals[k], s=70, color=COLOR[k], zorder=3, marker="o" if k != "copy" else "D")
            ax.annotate(f"{vals[k]:.2f}", (i, vals[k]), textcoords="offset points", xytext=(9, -3), fontsize=8)
        ax.set_xticks(range(len(keys)), [NAME[k] for k in keys], fontsize=8)
        ax.set_xlim(-0.5, len(keys) - 0.4)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 100)
    for y, txt in ((LINE / (1 + f), f"pass at or below {LINE / (1 + f):.2f}"),
                   (LINE * (1 + f), f"fail at or above {LINE * (1 + f):.2f}")):
        ax1.axhline(y, color=INK, ls=":", lw=1)
        ax1.text(len(keys) - 0.45, y * (0.86 if "pass" in txt else 1.08), txt, fontsize=7.5, ha="right")
    ax1.set_ylabel("adversarial Pile ratio to the unchanged model (log)")
    ax1.set_title("Worst-case masks: every masked control fails", fontsize=10)
    sf = a2["W0"]["stochastic_floor"]
    ax2.axhline(LINE / (1 + sf), color=INK, ls=":", lw=1)
    ax2.text(len(keys) - 0.45, LINE / (1 + sf) * 0.97, f"pass at or below {LINE / (1 + sf):.3f}", fontsize=7.5,
             ha="right", va="top")
    ax2.set_ylim(0.8, 1.6)
    ax2.set_ylabel("stochastic Pile ratio to the unchanged model")
    ax2.set_title("Random masks: every state stays level", fontsize=10)
    fig.suptitle("With nothing to copy, does any masked recipe keep the Pile checks? (rand32, L40)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e42_amend2_controls.png", dpi=150)
    plt.close(fig)


def fig_curves(a2: Rec, a1: Rec, out: Path) -> None:
    """Left: masked terms per 50-step block over their first block; right: the unmasked term in nats."""
    # (control, logged term, trained?) ; the trigger-only control's logged terms are its trigger KLs
    lines = (("M0_ci", "ci_masked", True), ("M0_ci", "stochastic", False), ("M0_st", "stochastic", True),
             ("M0_st", "ci_masked", False), ("M0_trig", "ci_masked_trigger", True),
             ("M0_trig", "stochastic_trigger", True))
    term = {"ci_masked": "CI-masked", "stochastic": "stochastic", "ci_masked_trigger": "CI-masked at trigger",
            "stochastic_trigger": "stochastic at trigger"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.0), sharex=True)
    for k, t, trained in lines:
        blocks = a2[k]["train"]["blocks50"]
        x = [50 * i + 25 for i in range(len(blocks))]
        y = [b[t] / blocks[0][t] for b in blocks]
        ls = "-" if trained else "--"
        if k == "M0_trig" and t == "stochastic_trigger":
            ls = ":"
        tag = "trained" if trained else "not trained"
        ax1.plot(x, y, ls, color=COLOR[k], lw=1.5, label=f"{NAME[k].replace(chr(10), ' ')}: {term[t]} ({tag})")
    for t, mk, xa in (("ci_masked", "o", 955), ("stochastic", "s", 995)):
        r = a1["M0"]["train"]["last50"][t] / a1["M0"]["train"]["first50"][t]
        ax1.scatter(xa, r, color=COLOR["A1"], marker=mk, s=45, zorder=4,
                    label=f"both terms (Amendment 1), last 50 steps: {term[t]}")
    ax1.axhline(1, color=MUTED, ls="--", lw=1)
    ax1.text(20, 1.01, "first 50 steps (no fit)", fontsize=7.5, color=MUTED, va="bottom")
    ax1.set_ylim(0.4, 1.25)
    ax1.set_ylabel("masked term / its first-50-step mean")
    ax1.set_xlabel("training step (50-step blocks)")
    ax1.set_title("The masked students move about a third toward the teacher", fontsize=10)
    ax1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    for k in CONTROLS:
        blocks = a2[k]["train"]["blocks50"]
        ax2.plot([50 * i + 25 for i in range(len(blocks))], [b["unmasked"] for b in blocks], color=COLOR[k], lw=1.5,
                 label=NAME[k].replace("\n", " "))
    m = a1["M0"]["train"]
    ax2.scatter([25, 975], [m["first50"]["unmasked"], m["last50"]["unmasked"]], color=COLOR["A1"], s=45, zorder=4,
                label="both terms (Amendment 1), first and last 50 steps")
    ax2.axhline(STEP1_UNMASKED, color=MUTED, ls="--", lw=1)
    ax2.text(1000, STEP1_UNMASKED * 1.15, "step 1: the trainer's path at the unchanged model", fontsize=7.5,
             color=MUTED, ha="right", va="bottom")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 0.3)
    ax2.set_ylabel("unmasked term, KL from the teacher (nats)")
    ax2.set_xlabel("training step (50-step blocks)")
    ax2.set_title("The target model drifts alike under every recipe", fontsize=10)
    ax2.legend(fontsize=7, loc="upper right", frameon=False)
    for ax in (ax1, ax2):
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("What did each control train, and what did the target model give up? (teacher = unchanged model)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e42_amend2_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a2, a1, e42 = load()
    fig_controls(a2, a1, e42, args.out)
    fig_curves(a2, a1, args.out)


if __name__ == "__main__":
    main()
