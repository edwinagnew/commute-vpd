"""Figures for broad_fragility (E39), read from results/broad_fragility.json and results/amend1.json, with the Adam
fine-tune's do-nothing and scaled adjoint split from E37 (experiments/teacher_spread/results/teacher_spread.json; same card, same W0 readings).
Plan: reports/plans/2026-10-27_plan_broad-fragility.md; analysis: reports/analyses/2026-10-27_analysis_broad-fragility.md.

  moves.png    test 1: the check ratio and the damage on plain Pile text of the random moves (x0.5, x1 at two noise
               seeds) beside the trained decomposition, revert-all and W0, with W0's seed band and the pass and fail lines
  reverts.png  test 2: the check ratio and behaviour kept of the trained decomposition with each subset of its
               subcomponents put back at the released values, with the share of the excess each revert removes
  shape.png    test 3: the check ratio of do-nothing, the scaled adjoint split and teacher training for the Adam, the
               SGD (Amendment 1, results/amend1.json) and the rank-8 LoRA fine-tunes, stable rank and outside share
               in the legend
  amend1_rank.png  the check after teacher training for the three fine-tunes against the fine-tune's stable rank and
               against the trained decomposition's damage on plain text, with the pass and fail lines

Prints the shares of the excess removed, recomputed from the check ratios, to check the run's own `score`.

Usage (repo root): ~/commute-agents/fleet/cpu-run uv run python experiments/broad_fragility/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
E37 = HERE.parent / "teacher_spread" / "results" / "teacher_spread.json"
INK, MUTED, GRID, BAND, INCONCL = "#0b0b0b", "#8a8984", "#e6e5e0", "#f0efec", "#fbeee0"
PASS_C, FAIL_C = "#2e8b57", "#c0392b"
ADAM_C, LORA_C, SGD_C, MOVE_C = "#2a78d6", "#d17a00", "#a0306e", "#7f7f7f"  # Adam keeps E37's colour
GROUP_C = {"layer": "#6b4fbb", "type": "#1b9e9e", "half": "#b5651d"}


def style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def check_lines(ax: Axes, r: dict[str, Any], vertical: bool) -> None:
    """W0's band over adversary seeds 0-2, the inconclusive zone, and the filed pass and fail lines, labelled for the
    legend."""
    pass_l, fail_l = r["config"]["pass_line"], r["config"]["fail_line"]
    top = 1 + r["stage0"]["spread"]  # seed 0 is the lowest of the three, so the band runs from 1 up
    span, line = (ax.axvspan, ax.axvline) if vertical else (ax.axhspan, ax.axhline)
    span(1, top, color=BAND, lw=0, label=f"unchanged model over 3 adversary seeds: 1 to {top:.3f}")
    span(pass_l, fail_l, color=INCONCL, lw=0, label="between the lines: inconclusive")
    line(pass_l, color=PASS_C, ls="--", lw=0.9, label=f"pass line, {pass_l}")
    line(fail_l, color=FAIL_C, ls="--", lw=0.9, label=f"fail line, {fail_l}")


def dot_rows(ax: Axes, rows: list[tuple[str, float, str, str]], fmt: str) -> None:
    """One labelled dot per row, top to bottom: (label, value, colour, marker)."""
    for i, (_, v, c, m) in enumerate(rows):
        ax.scatter(v, -i, s=60, color=c, marker=m, zorder=5, edgecolors=INK if c == "white" else c)
        ax.annotate(fmt.format(v), (v, -i), xytext=(7, -3), textcoords="offset points", fontsize=7)
    ax.set_yticks([-i for i in range(len(rows))], [r[0] for r in rows], fontsize=8)
    ax.set_ylim(-len(rows) + 0.4, 0.6)


CHECK_X = "check ratio: the adversary's KL\nover the unchanged decomposition's"


def fig_moves(r: dict[str, Any], e37: dict[str, Any]) -> None:
    rd = r["reads"]
    states = [("unchanged model, W₀", None, "white", "o"),
              ("revert-all: released U, V restored", "revert_all", "white", "s"),
              ("random move, half size, noise seed 0", "move_s0_x0.5", MOVE_C, "D"),
              ("random move, full size, noise seed 0", "move_s0_x1", MOVE_C, "D"),
              ("random move, full size, noise seed 1", "move_s1_x1", MOVE_C, "D"),
              ("the trained decomposition, step 1,000", "trained", ADAM_C, "o")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, gridspec_kw={"width_ratios": (1.5, 1)})
    check_lines(a1, r, True)
    dot_rows(a1, [(n, 1.0 if k is None else rd[k]["check"], c, m) for n, k, c, m in states], "{:.2f}")
    a1.set_xlim(0.85, 2.45)
    a1.set_xlabel(CHECK_X)
    a1.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.85), frameon=True, facecolor="white",
              edgecolor=GRID)
    a1.set_title("Does a random move of the same size per matrix fail the check?\n"
                 "No: both full-size moves sit inside the unchanged model's band", fontsize=9, loc="left")
    style(a1)
    ft = e37["states"]["do-nothing"]["damage"]
    a2.axvline(ft, color=INK, ls=":", lw=0.9, label=f"the fine-tuned model, E37: {ft:.3f}")
    dot_rows(a2, [(n, 0.0 if k is None else rd[k]["damage"], c, m) for n, k, c, m in states], "{:.3f}")
    a2.set_xlim(-0.01, 0.3)
    a2.set_xlabel("damage on plain Pile text:\nKL(W₀ ‖ state), nats per token")
    a2.legend(fontsize=7, loc="upper center", frameon=False)
    a2.set_title("But they barely change the model's outputs:\nabout a twenty-fifth of the trained one's damage",
                 fontsize=9, loc="left")
    style(a2)
    fig.tight_layout()
    fig.savefig(FIG / "moves.png", dpi=150)
    plt.close(fig)


REVERTS = [("nothing reverted: the trained decomposition", "trained", ADAM_C),
           ("layer 0 reverted", "revert_layer0", GROUP_C["layer"]), ("layer 1", "revert_layer1", GROUP_C["layer"]),
           ("layer 2", "revert_layer2", GROUP_C["layer"]), ("layer 3", "revert_layer3", GROUP_C["layer"]),
           ("all attention matrices reverted", "revert_attn", GROUP_C["type"]),
           ("all MLP matrices reverted", "revert_mlp", GROUP_C["type"]),
           ("rarely important half reverted", "revert_rare", GROUP_C["half"]),
           ("random half reverted", "revert_random", GROUP_C["half"]),
           ("often important half reverted", "revert_common", GROUP_C["half"]),
           ("everything reverted: revert-all", "revert_all", "white")]


def fig_reverts(r: dict[str, Any]) -> None:
    rd, share = r["reads"], r["score"]["share_removed"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.0), sharey=True, gridspec_kw={"width_ratios": (1.7, 1)})
    check_lines(a1, r, True)
    rt = rd["trained"]["check"]
    a1.axvline(rt, color=ADAM_C, ls=":", lw=0.9, label=f"nothing reverted, {rt:.2f}")
    dot_rows(a1, [(n, rd[k]["check"], c, "o") for n, k, c in REVERTS], "{:.2f}")
    a1.text(3.55, 0, "share of the\nexcess removed", fontsize=7, color=MUTED, ha="right", va="center")
    for i, (_, k, _) in enumerate(REVERTS):
        if k.startswith("revert_"):
            a1.text(3.55, -i, f"{share[k.removeprefix('revert_')]:+.2f}", fontsize=8, color=MUTED, ha="right",
                    va="center")
    a1.set_xlim(0.85, 3.6)
    a1.set_xlabel(CHECK_X + ";\nshare removed = (trained − reverted) / (trained − 1)")
    a1.legend(fontsize=7, loc="lower right", bbox_to_anchor=(0.86, 0.0), frameon=True, facecolor="white",
              edgecolor=GRID)
    a1.set_title("Which subcomponents carry the excess? Reverting the MLP matrices, or almost any half of\n"
                 "each matrix, removes it; the rarely important half removes less than a random half",
                 fontsize=9, loc="left")
    style(a1)
    a2.axvline(0, color=MUTED, ls=":", lw=0.9, label="unchanged model, 0")
    a2.axvline(1, color=INK, ls=":", lw=0.9, label="fine-tuned model, 1")
    dot_rows(a2, [(n, rd[k]["s"], c, "o") for n, k, c in REVERTS], "{:.2f}")
    a2.set_xlim(-0.08, 1.25)
    a2.set_xlabel("behaviour kept: share of the fine-tune's\nheld-out log-probability gain")
    a2.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.05, 0.12), frameon=True, facecolor="white",
              edgecolor=GRID)
    a2.set_title("What does each revert cost in behaviour?", fontsize=9, loc="left")
    style(a2)
    fig.tight_layout()
    fig.savefig(FIG / "reverts.png", dpi=150)
    plt.close(fig)


ARMS = {"adam": ("Adam fine-tune, rate 3e-5", ADAM_C), "sgd": ("SGD fine-tune, rate 0.01", SGD_C),
        "lora": ("rank-8 LoRA fine-tune, rate 1e-4", LORA_C)}


def shape_points(r: dict[str, Any], a1: dict[str, Any], e37: dict[str, Any]) -> dict[str, list[tuple[float, float, bool]]]:
    """Per fine-tune: (check, behaviour kept, read in E37) for do-nothing, the scaled adjoint split, teacher training."""
    rd, ra, es = r["reads"], a1["reads"], e37["states"]
    return {"adam": [(es["do-nothing"]["pile"]["pgd_kl"], 1.0, True), (es["adjoint"]["pile"]["pgd_kl"], 0.804, True),
                     (rd["trained"]["check"], rd["trained"]["s"], False)],
            "sgd": [(ra[k]["check"], ra[k]["s"], False) for k in ("nothing_sgd", "adjoint_sgd", "tt_sgd")],
            "lora": [(rd[k]["check"], rd[k]["s"], False) for k in ("nothing_lora", "adjoint_lora", "tt_lora")]}


def fig_shape(r: dict[str, Any], a1: dict[str, Any], e37: dict[str, Any]) -> None:
    sh = r["shape"] | {"sgd": a1["shape"]["sgd"]}
    places = ["do-nothing:\nthe update left in Δ", "scaled adjoint split", "teacher training,\n1,000 steps"]
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    check_lines(ax, r, False)
    for off, (arm, pts) in zip((-0.22, 0.0, 0.22), shape_points(r, a1, e37).items(), strict=True):
        s, (name, c) = sh[arm], ARMS[arm]
        ax.scatter([], [], color=c, s=60, label=f"{name}: held-out gain {s['gain']:.3f}, stable rank "
                   f"{s['stable_rank']['mean']:.1f}")
        for i, (v, kept, hollow) in enumerate(pts):
            ax.scatter(i + off, v, s=70, zorder=5, color="white" if hollow else c, edgecolors=c, lw=1.6)
            ax.annotate(f"{v:.2f}\nkept {kept:.2f}", (i + off, v), xytext=(7, 0), textcoords="offset points",
                        fontsize=6.5, ha="left", va="center")
    ax.scatter([], [], s=70, color="white", edgecolors=ADAM_C, lw=1.6, label="hollow: read in an earlier run (E37), same setup")
    ax.plot([], [], " ", label="kept: the share of the fine-tune's held-out gain the state keeps")
    ax.set_xticks(range(3), places, fontsize=8)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0.8, 5.2)
    ax.set_ylabel(CHECK_X)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.13), frameon=False)
    ax.set_title("Does a less spread-out update give a less fragile trained decomposition?\n"
                 "On these Pile rows the LoRA's copy passes and the SGD one fails like Adam's; on new Pile rows the SGD one passes", fontsize=9, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "shape.png", dpi=150)
    plt.close(fig)


def fig_rank(r: dict[str, Any], a1: dict[str, Any]) -> None:
    """Amendment 1: the trained decomposition's check against the fine-tune's stable rank and against the trained
    decomposition's damage on plain text, for the three fine-tunes."""
    sh = r["shape"] | {"sgd": a1["shape"]["sgd"]}
    tt = {"adam": r["reads"]["trained"], "sgd": a1["reads"]["tt_sgd"], "lora": r["reads"]["tt_lora"]}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, key, xlabel in ((left, "rank", "stable rank of the fine-tune's δW, mean over 24 matrices (log scale)"),
                            (right, "damage", ("damage of the trained decomposition on plain Pile text:\n"
                                              "KL(W₀ ‖ state), nats per token"))):
        check_lines(ax, r, False)
        for arm, (name, c) in ARMS.items():
            x = sh[arm]["stable_rank"]["mean"] if key == "rank" else tt[arm]["damage"]
            ax.scatter(x, tt[arm]["check"], s=80, color=c, zorder=5, label=name.split(",")[0])
            note = f"{tt[arm]['check']:.2f}; " + (f"rank {x:.1f}" if key == "rank" else f"damage {x:.3f}")
            ax.annotate(note + f"\nsize {tt[arm]['size']:.4f}", (x, tt[arm]["check"]), xytext=(8, -4),
                        textcoords="offset points", fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        style(ax)
    left.set_xscale("log")
    left.set_xlim(1, 100)
    right.set_xlim(0, 0.3)
    left.set_ylim(0.8, 2.6)
    left.set_ylabel(CHECK_X + "\nafter teacher training, 1,000 steps")
    left.legend(fontsize=6.5, loc="lower right", frameon=True, facecolor="white", edgecolor=GRID)
    left.set_title("Does a low stable rank predict a passing trained decomposition?\n"
                   "No: SGD's rank is near the LoRA's, yet it fails like Adam's", fontsize=9, loc="left")
    right.set_title("Does less damage on plain text predict a pass?\n"
                    "No: SGD's trained decomposition does the least and still fails", fontsize=9, loc="left")
    fig.text(0.01, 0.005, "size: the largest relative change of any matrix in the trained decomposition", fontsize=7,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "amend1_rank.png", dpi=150)
    plt.close(fig)


def check_shares(r: dict[str, Any]) -> None:
    rd, rt = r["reads"], r["reads"]["trained"]["check"]
    for k in sorted(n for n in rd if n.startswith("revert_")):
        mine, run = (rt - rd[k]["check"]) / (rt - 1), r["score"]["share_removed"][k.removeprefix("revert_")]
        assert abs(mine - run) < 1e-12, (k, mine, run)
        print(f"{k:16} check {rd[k]['check']:.4f} share removed {mine:+.4f} s {rd[k]['s']:.4f} damage {rd[k]['damage']:.4f}")
    m = (rd["move_s0_x1"]["check"] + rd["move_s1_x1"]["check"]) / 2
    print(f"move x1 mean {m:.4f}; rare - random {r['score']['share_removed']['rare'] - r['score']['share_removed']['random']:+.4f}")


def main() -> None:
    r = json.loads((HERE / "results" / "broad_fragility.json").read_text())
    r["config"] = json.loads((HERE / "results" / "predictions.json").read_text())["config"]
    assert r["status"] == "done", r["status"]
    a1 = json.loads((HERE / "results" / "amend1.json").read_text())
    assert a1["status"] == "done", a1["status"]
    e37 = json.loads(E37.read_text())
    FIG.mkdir(exist_ok=True)
    check_shares(r)
    fig_moves(r, e37)
    fig_reverts(r)
    fig_shape(r, a1, e37)
    fig_rank(r, a1)
    print(f"wrote {sorted(p.name for p in FIG.glob('*.png'))}")


if __name__ == "__main__":
    main()
