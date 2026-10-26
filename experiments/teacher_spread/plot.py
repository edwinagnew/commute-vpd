"""Figures for teacher_spread (E37), read from results/teacher_spread.json only. Plan:
reports/plans/2026-10-26_plan_teacher-spread.md; analysis: reports/analyses/2026-10-26_analysis_teacher-spread.md.

  behaviour_faithful.png  behaviour kept s against the harness's PGD ratio to W0 for every state, with the behaviour
                          bar, the harness's pass and fail lines, W0's adversary-seed spread and the do-nothing level
  adversaries.png         the harness's PGD ratio beside the budget adversary's ratio per state, with each one's line
                          and E34's copies of trigger fine-tunes as the reference band
  placement.png           cosines between weight changes: the two batch orders, the lowercase and plain copies, each
                          state with the fine-tune's delta-W, and the random direction, against P7's lines and E34's
                          copies' range

Usage (repo root): fleet/cpu-run uv run python experiments/teacher_spread/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
INK, MUTED, GRID, BAND = "#0b0b0b", "#8a8984", "#e6e5e0", "#f0efec"
PASS_C, FAIL_C = "#2e8b57", "#c0392b"
# E34's colours: teacher ink, task-text copy blue, Pile-text copy orange; the adjoint split grey as in E34's behaviour.png
STYLE: dict[str, dict[str, Any]] = {
    "W0": {"name": "unchanged model", "c": INK, "m": "o", "fill": False},
    "do-nothing": {"name": "do-nothing: fine-tune left in Δ", "c": INK, "m": "o", "fill": True},
    "adjoint": {"name": "adjoint split", "c": MUTED, "m": "s", "fill": True},
    "plain_s0": {"name": "copy, plain Pile text", "c": "#eb6834", "m": "s", "fill": True},
    "lower_s0": {"name": "copy, lowercase text", "c": "#2a78d6", "m": "o", "fill": True},
    "lower_s1": {"name": "copy, lowercase text, 2nd batch order", "c": "#2a78d6", "m": "o", "fill": False},
}
S_MIN, COS_SAME, COS_MANY = 0.9, 0.8, 0.5  # the filed P1/P6 and P7 lines (predictions.json)
BUDGET_LINE, BUDGET_NOISE = 1.241, 1.070  # E20 A3's budget pass line; W0 seed 1 over seed 0 at 20 steps (E34 A4)
E34_COPIES_RATIO = (0.895, 1.37)  # E34 A4 table: the twelve Adam copies at two batch orders, harness and budget
E34_COPIES_COS = (0.075, 0.285)  # E34 A1/A2's copies' cosine with delta-W (emoticon teachers)


def style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def dot(ax: Axes, name: str, x: float, y: float, size: float = 60) -> None:
    st = STYLE[name]
    ax.scatter(x, y, s=size, marker=st["m"], facecolors=st["c"] if st["fill"] else "white", edgecolors=st["c"],
               linewidths=1.5, zorder=4, label=st["name"])


def lines(r: dict[str, Any]) -> tuple[float, float]:
    """The harness's pass and fail lines on this job's W0 floor (faith_score.verdict)."""
    f = r["C1"]["pgd_floor"]
    return LINE / (1 + f), LINE * (1 + f)


def behaviour_faithful(r: dict[str, Any]) -> None:
    st, d = r["states"], r["score"]["derived"]
    pass_l, fail_l = lines(r)
    w0_spread = max(p["pgd_kl"] for p in r["W0_pile"].values()) / r["W0_pile"]["0"]["pgd_kl"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axhspan(1, w0_spread, color=BAND, lw=0)
    ax.axhspan(pass_l, fail_l, color="#fbeee0", lw=0)
    ax.axhline(pass_l, color=PASS_C, ls="--", lw=0.9)
    ax.axhline(fail_l, color=FAIL_C, ls="--", lw=0.9)
    dn = st["do-nothing"]["pile"]["pgd_kl"]
    ax.axhline(dn, color=INK, ls=":", lw=0.8)
    ax.axvline(S_MIN, color=MUTED, ls=":", lw=0.9)
    for n in STYLE:
        y = 1.0 if n == "W0" else st[n]["pile"]["pgd_kl"]
        dot(ax, n, d[n]["s"], y)
    ax.text(0.02, w0_spread * 1.01, f"unchanged model across adversary seeds, 1 to {w0_spread:.2f}", fontsize=7,
            color=MUTED, va="bottom")
    ax.text(0.02, pass_l * 0.99, f"pass line {pass_l:.2f}", fontsize=7, color=PASS_C, va="top")
    ax.text(0.02, fail_l * 1.01, f"fail line {fail_l:.2f}; between the lines: inconclusive", fontsize=7, color=FAIL_C,
            va="bottom")
    ax.text(0.02, dn * 0.98, f"do-nothing's level {dn:.2f}: below it passes if judged against do-nothing", fontsize=7,
            color=INK, va="top")
    ax.text(S_MIN - 0.01, 0.86, f"behaviour bar {S_MIN}", fontsize=7, color=MUTED, ha="right")
    for n, (dx, dy) in {"lower_s0": (-8, 4), "lower_s1": (8, -3), "plain_s0": (-8, 2), "adjoint": (-8, 4)}.items():
        ax.annotate(f"{st[n]['pile']['pgd_kl']:.2f}", (d[n]["s"], st[n]["pile"]["pgd_kl"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, ha="right" if dx < 0 else "left", color=INK)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1, 1.5, 2, 3, 4, 5], ["1", "1.5", "2", "3", "4", "5"])
    ax.set_ylim(0.82, 6.0)
    ax.set_xlim(-0.04, 1.06)
    ax.set_xlabel("behaviour kept: share of the fine-tune's held-out log-probability gain on lowercase text")
    ax.set_ylabel("adversarial ratio: PGD KL over the unchanged model's")
    ax.set_title("Do the copies keep the lowercase behaviour and stay faithful?\n"
                 "They keep all of it but sit above the fail line; the adjoint split keeps 0.8 and passes",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "behaviour_faithful.png", dpi=150)
    plt.close(fig)


def adversaries(r: dict[str, Any]) -> None:
    st, d = r["states"], r["score"]["derived"]
    pass_l, fail_l = lines(r)
    order = ["W0", "adjoint", "plain_s0", "lower_s0", "lower_s1", "do-nothing"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(*E34_COPIES_RATIO, color=BAND, lw=0)
    ax.axhline(pass_l, color=PASS_C, ls="--", lw=0.9)
    ax.axhline(fail_l, color=FAIL_C, ls="--", lw=0.9)
    ax.axhline(BUDGET_LINE, color=PASS_C, ls=":", lw=1.1)
    for x, n in enumerate(order):
        c = STYLE[n]["c"]
        h = 1.0 if n == "W0" else st[n]["pile"]["pgd_kl"]
        ax.scatter(x - 0.12, h, s=55, marker="o", color=c, zorder=4)
        ax.text(x - 0.2, h, f"{h:.2f}", fontsize=7, ha="right", va="center")
        if "budget_ratio" in d[n]:
            b = d[n]["budget_ratio"]
            ax.scatter(x + 0.12, b, s=55, marker="D", facecolors="white", edgecolors=c, linewidths=1.5, zorder=4)
            ax.text(x + 0.2, b, f"{b:.2f}", fontsize=7, ha="left", va="center")
    n = len(order)
    ax.text(n - 0.45, E34_COPIES_RATIO[0] * 1.01, "E34's copies of trigger fine-tunes, either adversary",
            fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.text(n - 0.45, pass_l * 1.01, f"harness pass line {pass_l:.2f}", fontsize=7, color=PASS_C, ha="right",
            va="bottom")
    ax.text(n - 0.45, fail_l * 1.01, f"harness fail line {fail_l:.2f}", fontsize=7, color=FAIL_C, ha="right",
            va="bottom")
    ax.text(-0.45, BUDGET_LINE * 1.01, f"budget pass line {BUDGET_LINE}", fontsize=7, color=PASS_C, va="bottom")
    ax.scatter([], [], marker="o", color=INK, label="harness: 20 steps, 256 rows, mean PGD KL")
    ax.scatter([], [], marker="D", facecolors="white", edgecolors=INK,
               label="budget adversary: 20 steps, 128 rows, KL under the masks found")
    ax.set_xticks(range(n), ["unchanged\nmodel", "adjoint\nsplit", "copy,\nplain text", "copy,\nlowercase",
                             "copy, lowercase,\n2nd order", "do-nothing"], fontsize=8)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1, 1.5, 2, 3, 4, 5], ["1", "1.5", "2", "3", "4", "5"])
    ax.set_ylim(0.85, 6.0)
    ax.set_ylabel("ratio to the unchanged model under the same adversary")
    ax.set_title("Is the copies' failure the harness's alone?\n"
                 "No: the budget adversary also puts them above its line, a little lower than the harness",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "adversaries.png", dpi=150)
    plt.close(fig)


def placement(r: dict[str, Any]) -> None:
    st, u = r["states"], r["uniqueness"]
    rows = [("lowercase copy: batch order 0 against order 1", u["cos"], STYLE["lower_s0"]["c"]),
            ("lowercase copy against plain-text copy", u["cos_lower_plain"], STYLE["plain_s0"]["c"]),
            ("lowercase copy, order 0, against δW", st["lower_s0"]["cos_dW"], STYLE["lower_s0"]["c"]),
            ("lowercase copy, order 1, against δW", st["lower_s1"]["cos_dW"], STYLE["lower_s0"]["c"]),
            ("plain-text copy against δW", st["plain_s0"]["cos_dW"], STYLE["plain_s0"]["c"]),
            ("adjoint split against δW", st["adjoint"]["cos_dW"], STYLE["adjoint"]["c"]),
            ("random direction against δW", u["null"], MUTED)]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.axvspan(*E34_COPIES_COS, color=BAND, lw=0)
    ax.axvline(COS_SAME, color=PASS_C, ls="--", lw=0.9)
    ax.axvline(COS_MANY, color=FAIL_C, ls="--", lw=0.9)
    for i, (label, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([0, v], [y, y], color=c, lw=2)
        ax.scatter(v, y, s=50, color=c, zorder=4)
        ax.text(v + 0.015, y, f"{abs(v) if abs(v) < 0.005 else v:.2f}", fontsize=7, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8)
    ax.text(COS_SAME + 0.01, -0.85, f"one placement from {COS_SAME}", fontsize=7, color=PASS_C)
    ax.text(COS_MANY - 0.01, -0.85, f"one of many below {COS_MANY}", fontsize=7, color=FAIL_C, ha="right")
    ax.text(sum(E34_COPIES_COS) / 2, len(rows) - 0.45, "E34's copies\nagainst their δW", fontsize=7, color=MUTED,
            ha="center", va="bottom")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.1, len(rows) + 0.2)
    ax.set_xlabel("cosine between weight changes, all 24 matrices at once")
    ax.set_title("Do two copies land on the same weight change?\n"
                 "Yes, 0.93 across batch orders; each copy points along δW more than E34's",
                 fontsize=9, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "placement.png", dpi=150)
    plt.close(fig)


def main() -> None:
    r = json.loads((HERE / "results" / "teacher_spread.json").read_text())
    assert r["status"] == "done", r["status"]
    FIG.mkdir(exist_ok=True)
    behaviour_faithful(r)
    adversaries(r)
    placement(r)
    print(f"wrote {sorted(p.name for p in FIG.glob('*.png'))}")


if __name__ == "__main__":
    main()
