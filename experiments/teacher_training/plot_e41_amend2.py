"""Figures for E41 Amendment 2, the short-links chain: seven 100-step copies along the emoticon fine-tune, each link
starting where the last ended (plan: reports/plans/2026-10-28_plan_e41-teacher-path.md -> "Amendment 2"), read from the
result files only. The scoring is the run's own (`path_amend2.py`, under "score" in the results file).

  e41_amend2_faithful.png   the Pile adversarial ratio of every link against the checkpoint it copies, beside the
                            checkpoint left in Δ (do-nothing), the one-shot copies of steps 10, 100 and 1000 (step 10
                            at batch orders 0-2, E41 and Amendment 1) and E41's chain of 1,000-step links; the pass and
                            fail lines, the unchanged model's adversary-seed band, and the checkpoints where doing
                            nothing passes, so the check cannot tell a copy from it
  e41_amend2_behaviour.png  the same states' held-out P(target) and gap share against the checkpoint
  e41_amend2_pileup.png     the same states' size and damage against the checkpoint, beside the fine-tune's own, with
                            P3's limit of 1.25 x the one-shot step-1000 copy's at step 1000

Usage (repo root): uv run python experiments/teacher_training/plot_e41_amend2.py [--out DIR]
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
LINKS = ("1", "3", "10", "30", "100", "300", "1000")
ANCHORS = ("10", "100", "1000")
INK, MUTED, GRID, BLIND = "#0b0b0b", "#8a8984", "#e6e5e0", "#f4f3ef"
SHORT, ONESHOT = "#1a9850", "#2a78d6"  # the short-links chain; plot_e41.py's task-text blue for one-shot and long chain
W0_P = 0.004  # the unchanged model's held-out P(target) on emoticon (plan)
Rec = dict[str, Any]


def load() -> tuple[Rec, Rec, Rec]:
    res = HERE / "results"
    a2, e41, a1 = (json.loads((res / f"teacher_training_{n}.json").read_text()) for n in ("e41_amend2", "e41", "e41_amend1"))
    assert a2["done"] and a2["card"] == e41["card"] == a1["card"] == "NVIDIA H100 80GB HBM3"
    return a2, e41, a1


def x(k: str, shift: float = 1.0) -> float:
    return float(k) * shift


def style(ax: Axes, a2: Rec, blind: bool = False) -> None:
    """Log x over the checkpoints; with `blind`, a pale column where doing nothing passes, so the check cannot see the
    update."""
    ax.set_xscale("log")
    ax.set_xticks([float(k) for k in LINKS], LINKS)
    ax.minorticks_off()
    ax.set_xlim(0.7, 1500)
    for k in LINKS:
        if blind and not a2["score"]["check_sees_the_update"][k]:
            ax.axvspan(float(k) / 1.35, float(k) * 1.35, color=BLIND, zorder=0)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("checkpoint copied: the fine-tune stopped at step")


def series(ax: Axes, a2: Rec, e41: Rec, a1: Rec, key: str, teacher: bool = True) -> None:
    """Do-nothing (the teacher's own reading), the short chain, the one-shot copies and E41's long chain, one key."""
    one, chain = e41["tasks"]["emoticon"]["A"]["oneshot"], e41["tasks"]["emoticon"]["A"]["chain"]
    if teacher:
        ax.plot([x(k, 0.86) for k in LINKS], [a2["teachers"][k]["nothing"][key] for k in LINKS], "s-", color=MUTED,
                ms=6, lw=1, label="doing nothing: the checkpoint left in Δ")
    ax.plot([x(k) for k in LINKS], [a2["links"][k][key] for k in LINKS], "o-", color=SHORT, ms=7, lw=1.6, zorder=4,
            label="short chain: 100 steps per link, each from the last")
    ax.scatter([x(k, 1.16) for k in ANCHORS], [one[k][key] for k in ANCHORS], marker="D", s=40, color=ONESHOT, zorder=3,
               label="one-shot copy: 1,000 steps from the unchanged model")
    ax.scatter([x("10", 1.16)] * 2, [a1["copies"][s][key] for s in ("1", "2")], marker="D", s=40, facecolors="white",
               edgecolors=ONESHOT, lw=1.2, zorder=3, label="one-shot copy of step 10, batch orders 1 and 2")
    ax.plot([x(k, 1.16) for k in ANCHORS], [one["10"][key], chain["100"][key], chain["1000"][key]], "^--",
            color=ONESHOT, mfc="white", ms=7, lw=0.9, zorder=2, markevery=[1, 2],
            label="E41's chain: 1,000 steps per link, from the step-10 copy")


def fig_faithful(a2: Rec, e41: Rec, a1: Rec, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    floor = a2["W0"]["pgd_floor"]
    ax.axhspan(1, 1 + floor, color=GRID, zorder=1)
    ax.text(0.75, 0.93, "grey band: unchanged model,\n3 adversary seeds", fontsize=7, color=MUTED, va="center")
    for y, txt in ((LINE / (1 + floor), "pass at or below"), (LINE * (1 + floor), "fail at or above")):
        ax.axhline(y, color=INK, ls=":", lw=1)
        ax.text(1450, y + 0.02, f"{txt} {y:.2f}", fontsize=7.5, va="bottom", ha="right")
    series(ax, a2, e41, a1, "pgd")
    for k in LINKS:
        r = a2["links"][k]["pgd"]
        ax.annotate(f"{r:.2f}", (x(k), r), xytext=(0, -14), textcoords="offset points", fontsize=8, ha="center",
                    color=SHORT)
    ax.text(2.0, 2.45, "pale columns: doing nothing passes,\nso the check cannot tell a copy from it", fontsize=7.5,
            color=MUTED, ha="center")
    style(ax, a2, blind=True)
    ax.set_ylim(0.8, 2.7)
    ax.set_ylabel("Pile adversarial ratio to the unchanged model")
    ax.legend(fontsize=7.5, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.92))
    fig.suptitle("Does every link of the short chain pass the adversarial check?\n"
                 "All seven pass. Only steps 10 and 1000 are tested: there doing nothing fails or sits between the lines.\n"
                 "At step 10 the link reads 1.32, inside the one-shot copies' spread over batch orders, 1.28 to 1.55",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "e41_amend2_faithful.png", dpi=150)
    plt.close(fig)


def fig_behaviour(a2: Rec, e41: Rec, a1: Rec, out: Path) -> None:
    fig, (ap, ag) = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    series(ap, a2, e41, a1, "p_target")
    ap.axhline(0.9, color=INK, ls=":", lw=1)
    ap.text(1450, 0.901, "behaviour lost below 0.9", fontsize=7.5, ha="right", va="bottom")
    ap.text(0.75, 0.885, f"unchanged model: {W0_P} (off the axis)", fontsize=7.5, color=MUTED)
    ap.set_ylim(0.88, 1.005)
    ap.set_ylabel("held-out P(target) at the trigger")
    series(ag, a2, e41, a1, "gap_share", teacher=False)
    ag.axhline(0.05, color=INK, ls=":", lw=1)
    ag.text(1450, 0.0505, "behaviour lost above 0.05", fontsize=7.5, ha="right", va="bottom")
    ag.axhline(0, color=MUTED, lw=1)
    ag.text(0.75, -0.0008, "doing nothing: 0", fontsize=7.5, color=MUTED, va="top")
    ag.text(0.75, 0.054, "unchanged model: 1 (off the axis)", fontsize=7.5, color=MUTED)
    ag.set_ylim(-0.004, 0.06)
    ag.set_ylabel("gap share: KL(teacher ‖ copy) / KL(teacher ‖ unchanged)")
    for ax in (ap, ag):
        style(ax, a2)
    ag.legend(*ap.get_legend_handles_labels(), fontsize=7.5, loc="center left", frameon=False,
              bbox_to_anchor=(0.0, 0.55))
    fig.suptitle("Does every link keep the behaviour?\n"
                 "Yes: every link holds P(target) of 0.91 or more and stays within 0.006 of its teacher's output, "
                 "as the one-shot copies do", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "e41_amend2_behaviour.png", dpi=150)
    plt.close(fig)


def fig_pileup(a2: Rec, e41: Rec, a1: Rec, out: Path) -> None:
    fig, (az, ad) = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    one = e41["tasks"]["emoticon"]["A"]["oneshot"]["1000"]
    tol = 1 + json.loads((HERE / "results" / "predictions_e41_amend2.json").read_text())["e41_amend2"]["pileup"]
    for ax, key, name in ((az, "size", "size: largest relative weight change of one matrix"),
                          (ad, "damage", "damage: KL from the unchanged model, non-target positions")):
        series(ax, a2, e41, a1, key)
        lim = tol * one[key]
        ax.plot([1000 / 1.5, 1000 * 1.4], [lim, lim], color=INK, ls=":", lw=1.4,
                label=f"filed limit at step 1000: {tol:.2f} × the one-shot copy")
        ax.set_ylim(0, None)
        ax.set_ylabel(name)
        style(ax, a2)
    ad.legend(fontsize=7, loc="upper right", frameon=False)
    s, d = a2["score"]["P3_no_pileup"]["size_over_oneshot"], a2["score"]["P3_no_pileup"]["damage_over_oneshot"]
    fig.suptitle("Does the short chain pile up size or damage?\n"
                 f"No, narrowly on damage: at step 1000 its size is {s:.2f}× and its damage {d:.2f}× the one-shot copy's, "
                 "under the 1.25× limit.\nE41's chain of 1,000-step links read 1.36× and 1.49×", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "e41_amend2_pileup.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    a2, e41, a1 = load()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_faithful(a2, e41, a1, args.out)
    fig_behaviour(a2, e41, a1, args.out)
    fig_pileup(a2, e41, a1, args.out)


if __name__ == "__main__":
    main()
