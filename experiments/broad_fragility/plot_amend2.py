"""Figures for E39 Amendment 2, read from results/amend2.json, with E39's first run (results/broad_fragility.json),
Amendment 1 (results/amend1.json) and E37 (experiments/teacher_spread/results/teacher_spread.json) as references on
the same card, rows and unchanged-model readings.
Plan: reports/plans/2026-10-27_plan_broad-fragility.md -> "Amendment 2 (2026-10-28)";
analysis: reports/analyses/2026-10-28_analysis_e39-amend2.md.

  amend2_labels.png  (A) the check ratio and the damage on plain Pile text of component training on the lowercase
                     task's labels, Adam and SGD, at the gain-matched state, beside W0, the Adam fine-tune's
                     do-nothing and teacher training on the Adam fine-tune
  amend2_lora.png    (B, C) the check ratio after teacher training on the check's rows and on a second set of Pile
                     rows, for the Adam, plain SGD, SGD-held-to-rank-8 and two LoRA fine-tunes, with the Adam
                     do-nothing and W0
  amend2_size.png    (D and every state) the check ratio against damage on plain Pile text for every state read on
                     the check's rows in E37, E39 and its amendments, the damage-matched random moves among them

Prints the excess ratios quoted in the analysis. Per-block check readings are in results/large/amend2.log only.

Usage (repo root): ~/commute-agents/fleet/cpu-run uv run python experiments/broad_fragility/plot_amend2.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import (
    ADAM_C,
    CHECK_X,
    E37,
    FIG,
    GRID,
    HERE,
    INK,
    LORA_C,
    MOVE_C,
    MUTED,
    SGD_C,
    check_lines,
    dot_rows,
    style,
)

R8_C, LAB_ADAM_C, LAB_SGD_C = "#6b4fbb", "#1b9e9e", "#b5651d"  # SGD held to rank 8; labels with Adam; with SGD
DAMAGE_X = "damage on plain Pile text:\nKL(W₀ ‖ state), nats per token"


def load() -> dict[str, Any]:
    res = HERE / "results"
    a2 = json.loads((res / "amend2.json").read_text())
    assert a2["status"] == "done", a2["status"]
    a2["config"] = json.loads((res / "predictions.json").read_text())["config"]  # pass and fail lines
    return {"a2": a2, "e39": json.loads((res / "broad_fragility.json").read_text()),
            "a1": json.loads((res / "amend1.json").read_text()), "e37": json.loads(E37.read_text())}


def fig_labels(d: dict[str, Any]) -> None:
    rd, lab = d["a2"]["reads"], d["a2"]["label"]
    rows = [("unchanged model, W₀", 1.0, 0.0, "white", "o"),
            ("Adam fine-tune, do-nothing:\nthe update left in Δ", rd["nothing_adam"]["check"],
             rd["nothing_adam"]["damage"], "white", "o"),
            ("teacher training on the Adam fine-tune,\n1,000 steps", rd["trained"]["check"], rd["trained"]["damage"],
             ADAM_C, "o"),
            (f"labels, Adam 1e-5, step 1,000:\ngain {lab['adam']['gain']:.3f}", rd["label_adam"]["check"],
             rd["label_adam"]["damage"], LAB_ADAM_C, "s"),
            (f"labels, SGD 0.03, step 100:\ngain {lab['sgd']['gain']:.3f}", rd["label_sgd"]["check"],
             rd["label_sgd"]["damage"], LAB_SGD_C, "s")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, gridspec_kw={"width_ratios": (1.6, 1)})
    check_lines(a1, d["a2"], True)
    dot_rows(a1, [(n, c, col, m) for n, c, _, col, m in rows], "{:.2f}")
    a1.set_xlim(0.8, 8.2)
    a1.set_xlabel(CHECK_X)
    a1.legend(fontsize=7, loc="lower right", frameon=True, facecolor="white", edgecolor=GRID)
    a1.set_title("Can the subcomponents be trained on the task's own labels and pass the check?\n"
                 "With SGD at step 100, yes; with Adam at step 1,000, no: it fails worse than teacher training",
                 fontsize=9, loc="left")
    style(a1)
    a2.axvline(rd["nothing_adam"]["damage"], color=INK, ls=":", lw=0.9,
               label=f"the Adam fine-tune: {rd['nothing_adam']['damage']:.3f}")
    dot_rows(a2, [(n, dm, col, m) for n, _, dm, col, m in rows], "{:.3f}")
    a2.set_xlim(-0.01, 0.33)
    a2.set_xlabel(DAMAGE_X)
    a2.legend(fontsize=7, loc="lower right", frameon=True, facecolor="white", edgecolor=GRID)
    a2.set_title("The Adam labelled state changes plain text the most;\nthe SGD one about half as much",
                 fontsize=9, loc="left")
    style(a2)
    fig.text(0.01, 0.005, "Hollow: do-nothing and W₀. Gain: held-out lowercase log-probability gain, nats per token; "
             f"the Adam fine-tune's is {d['a2']['stage0']['gain']:.3f}.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "amend2_labels.png", dpi=150)
    plt.close(fig)


def fig_lora(d: dict[str, Any]) -> None:
    rd, sh = d["a2"]["reads"], d["a2"]["shape"]["sgd_r8"]
    tail, tail_cap = sh["plain_tail"]["pooled"], sh["tail"]["max"]
    rows = [("unchanged model, W₀", 1.0, 1.0, "white"),
            ("Adam fine-tune, do-nothing", rd["nothing_adam"]["check"], None, "white"),
            ("teacher training on the Adam fine-tune", rd["trained"]["check"], rd["trained_rows2"]["check"], ADAM_C),
            (f"... on the plain SGD fine-tune, Amendment 1:\n{tail:.0%} of its update lies beyond rank 8",
             d["a1"]["reads"]["tt_sgd"]["check"], None, SGD_C),
            (f"... on SGD held to rank 8:\nat most {tail_cap:.0e} beyond rank 8", rd["tt_sgd_r8"]["check"], None, R8_C),
            ("... on the LoRA, seed 0, batch order 0", rd["tt_lora_s0"]["check"], rd["tt_lora_s0_rows2"]["check"],
             LORA_C),
            ("... on the LoRA, seed 1, batch order 1", rd["tt_lora_s1"]["check"], rd["tt_lora_s1_rows2"]["check"],
             LORA_C)]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    check_lines(ax, d["a2"], True)
    dot_rows(ax, [(n, v1, c, "o") for n, v1, _, c in rows], "{:.2f}")
    for i, (_, _, v2, c) in enumerate(rows):
        if v2 is not None and i > 0:
            ax.scatter(v2, -i - 0.28, s=55, marker="D", color="white", edgecolors=INK if c == "white" else c, lw=1.4,
                       zorder=5)
            ax.annotate(f"{v2:.2f}", (v2, -i - 0.28), xytext=(7, -3), textcoords="offset points", fontsize=7)
    ax.scatter([], [], s=60, color=MUTED, label="the check's rows, Pile val 0-255")
    ax.scatter([], [], s=55, marker="D", color="white", edgecolors=MUTED, lw=1.4,
               label=f"second rows, Pile val 256-511: W₀'s spread {d['a2']['rows2']['spread']:.3f}")
    ax.set_xlim(0.8, 5.3)
    ax.set_ylim(-len(rows) + 0.3, 0.6)
    ax.set_xlabel(CHECK_X)
    ax.legend(fontsize=7, loc="lower right", frameon=True, facecolor="white", edgecolor=GRID)
    ax.set_title("After teacher training, does the LoRA's pass repeat, and does a rank-8 cap make SGD pass?\n"
                 "The LoRA passes at a second seed and on new rows; SGD held to rank 8 still fails",
                 fontsize=9, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "amend2_lora.png", dpi=150)
    plt.close(fig)


def size_points(d: dict[str, Any]) -> list[tuple[str, list[tuple[float, float]], str, str, bool, float]]:
    """(legend, [(damage, check)], colour, marker, hollow, size) for every state read on the check's rows."""
    a2, e39, a1, e37 = d["a2"]["reads"], d["e39"]["reads"], d["a1"]["reads"], d["e37"]["states"]

    def pts(src: dict[str, Any], *keys: str) -> list[tuple[float, float]]:
        return [(src[k]["damage"], src[k]["check"]) for k in keys]

    reverts = [k for k in e39 if k.startswith("revert_") and k != "revert_all"]
    adam_s1 = (e37["lower_s1"]["damage"], e37["lower_s1"]["pile"]["pgd_kl"])
    adjoint_adam = (e37["adjoint"]["damage"], e37["adjoint"]["pile"]["pgd_kl"])
    return [
        ("E39's reverts of the Adam teacher-trained decomposition, partial", pts(e39, *reverts), "#bdbcb6", "o", False, 28),
        ("scaled adjoint split: Adam, SGD, LoRA", [adjoint_adam, *pts(e39, "adjoint_lora"), *pts(a1, "adjoint_sgd")],
         "#bdbcb6", "X", False, 40),
        ("do-nothing, the update left in Δ: Adam, SGD, LoRA",
         pts(a2, "nothing_adam") + pts(a1, "nothing_sgd") + pts(e39, "nothing_lora"), MUTED, "o", True, 55),
        ("random moves matched in weight size, E39", pts(e39, "move_s0_x0.5", "move_s0_x1", "move_s1_x1"), MOVE_C,
         "D", True, 50),
        ("random moves matched in damage, seeds 0 and 1", pts(a2, "move_s0", "move_s1"), MOVE_C, "D", False, 60),
        ("teacher training: Adam fine-tune, batch orders 0 and 1", [*pts(a2, "trained"), adam_s1], ADAM_C, "o", False,
         75),
        ("teacher training: plain SGD fine-tune", pts(a1, "tt_sgd"), SGD_C, "o", False, 75),
        ("teacher training: SGD held to rank 8", pts(a2, "tt_sgd_r8"), R8_C, "o", False, 75),
        ("teacher training: LoRA, seeds 0 and 1", pts(a2, "tt_lora_s0", "tt_lora_s1"), LORA_C, "o", False, 75),
        ("labels, Adam, step 1,000", pts(a2, "label_adam"), LAB_ADAM_C, "s", False, 70),
        ("labels, SGD, step 100", pts(a2, "label_sgd"), LAB_SGD_C, "s", False, 70),
        ("unchanged model, W₀", [(0.0, 1.0)], "white", "o", True, 60),
    ]


def fig_size(d: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    check_lines(ax, d["a2"], False)
    ax.axvspan(0.088, 0.136, color="#f1eef8", lw=0, zorder=0, label="damage 0.09-0.13: checks from 1.27 to 2.70")
    for name, xy, c, m, hollow, s in size_points(d):
        xs, ys = zip(*xy, strict=True)
        edge = INK if c == "white" else c
        ax.scatter(xs, ys, s=s, marker=m, color="white" if hollow else c, edgecolors=edge, lw=1.3, zorder=5,
                   label=name)
    notes = {"tt_lora_s1": ("LoRA, teacher-trained 1.27", (8, -12)), "label_sgd": ("labels, SGD 1.28", (-78, -12)),
             "tt_sgd_r8": ("SGD rank 8, teacher-trained 2.12", (7, 4)), "label_adam": ("labels, Adam 7.24", (-82, 6)),
             "move_s0": ("random 16.5", (7, 4)), "move_s1": ("random 27.2", (7, 4)),
             "trained": ("Adam, teacher-trained 2.20", (7, 4))}
    a2, a1 = d["a2"]["reads"], d["a1"]["reads"]
    for k, (text, off) in notes.items():
        ax.annotate(text, (a2[k]["damage"], a2[k]["check"]), xytext=off, textcoords="offset points", fontsize=7)
    ax.annotate("plain SGD, teacher-trained 2.07", (a1["tt_sgd"]["damage"], a1["tt_sgd"]["check"]), xytext=(-128, 6),
                textcoords="offset points", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(0.85, 35)
    ax.set_yticks([1, 1.4, 1.61, 2, 3, 5, 10, 20, 30], ["1", "1.40", "1.61", "2", "3", "5", "10", "20", "30"])
    ax.set_xlim(-0.01, 0.29)
    ax.set_xlabel(DAMAGE_X)
    ax.set_ylabel(CHECK_X + " (log scale)")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title("Does the size of the output change decide the verdict? Only at the ends: every state under 0.06 "
                 "passes and every\nstate over 0.2 fails, but in between the check runs from 1.27 to 2.70, and at "
                 "0.24 random directions fail far worse", fontsize=9, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "amend2_size.png", dpi=150)
    plt.close(fig)


def report(d: dict[str, Any]) -> None:
    """The excess of each state over the trained decomposition's excess, as the analysis quotes it."""
    a2 = d["a2"]
    rt = a2["reads"]["trained"]["check"]
    for k in ("move_s0", "move_s1", "label_adam"):
        print(f"{k:12} check {a2['reads'][k]['check']:.3f} excess / trained excess "
              f"{(a2['reads'][k]['check'] - 1) / (rt - 1):.1f}")
    print(f"P14 mean {a2['score']['P14_mean']:.3f}")


def main() -> None:
    d = load()
    report(d)
    fig_labels(d)
    fig_lora(d)
    fig_size(d)
    print(f"wrote {sorted(p.name for p in FIG.glob('amend2_*.png'))}")


if __name__ == "__main__":
    main()
