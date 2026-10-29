"""The research log's four figures for E37, redrawn from saved results with plain labels (no model, no new runs).

  log_faithful.png  per state: the adversarial ratio as a bar from 1 (the unchanged decomposition), with the pass and
                    fail lines and each state's share of the fine-tune's gain kept (results/teacher_spread.json)
  log_placement.png cosines between weight changes: two batch orders, lowercase against plain-text copy, each copy and
                    the adjoint split against the fine-tune's change, a random direction (results/teacher_spread.json)
  log_steps.png     the adversarial ratio and the gap to the fine-tuned model against training steps, for the longer
                    run and the run started from the adjoint split (results/amend1.json)
  log_masked.png    the adversarial ratio of VPD's masked losses run with nothing to copy, at two loss weights, beside
                    the unchanged decomposition and teacher training at 1,000 steps (results/amend1.json)

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/teacher_spread/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from plot import FAIL_C, FIG, HERE, INK, MUTED, PASS_C, lines, style
from plot_amend1 import series

BLUE, PURPLE, ORANGE, GREY, BROWN = "#2a78d6", "#6b4fbb", "#eb6834", "#8a8984", "#b5651d"


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / "results" / name).read_text())


def verdict_lines(ax: Axes, r: dict[str, Any], vertical: bool) -> None:
    pass_l, fail_l = lines(r)
    span, line = (ax.axvspan, ax.axvline) if vertical else (ax.axhspan, ax.axhline)
    span(0, pass_l, color="#e3f1e8", lw=0, zorder=0)
    span(fail_l, 10, color="#f8e1de", lw=0, zorder=0)
    line(pass_l, color=PASS_C, ls="--", lw=0.9)
    line(fail_l, color=FAIL_C, ls="--", lw=0.9)


def ratio_bars(ax: Axes, r: dict[str, Any], rows: list[tuple[str, float, str, str]]) -> None:
    """Horizontal bars from 1 (the unchanged decomposition) to each ratio, top to bottom, with a note per bar."""
    verdict_lines(ax, r, vertical=True)
    for i, (label, v, c, note) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v - 1, left=1, height=0.6, color=c, zorder=3)
        ax.text(max(v, 1) + 0.04, y, f"{v:.2f}{note}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows][::-1], fontsize=8.5)
    pass_l, fail_l = lines(r)
    ax.text(pass_l - 0.03, len(rows) - 0.45, f"passes\n≤ {pass_l:.2f}", fontsize=7.5, color=PASS_C, ha="right")
    ax.text(fail_l + 0.03, len(rows) - 0.45, f"fails\n≥ {fail_l:.2f}", fontsize=7.5, color=FAIL_C, ha="left")
    ax.set_ylim(-0.6, len(rows) + 0.3)
    ax.set_xlabel("adversarial ratio: damage the mask adversary finds, over what it finds\n"
                  "at the unchanged decomposition (1 = no more fragile)")
    style(ax)


def fig_faithful(r: dict[str, Any]) -> None:
    st, d = r["states"], r["score"]["derived"]

    def kept(n: str) -> str:
        return f"   keeps {d[n]['s']:.2f} of the gain"

    rows = [("unchanged decomposition", 1.0, INK, "   keeps 0 of the gain (by definition)"),
            ("adjoint split of the fine-tune", st["adjoint"]["pile"]["pgd_kl"], GREY, kept("adjoint")),
            ("trained on the same texts, capitals kept", st["plain_s0"]["pile"]["pgd_kl"], ORANGE, kept("plain_s0")),
            ("trained on lowercase text", st["lower_s0"]["pile"]["pgd_kl"], BLUE, kept("lower_s0")),
            ("trained on lowercase text, other batch order", st["lower_s1"]["pile"]["pgd_kl"], BLUE, kept("lower_s1"))]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ratio_bars(ax, r, rows)
    ax.set_xlim(0.9, 3.4)
    ax.set_title("Teacher training keeps all of the lowercasing but fails the adversarial check;\n"
                 "the adjoint split keeps most of it and passes", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "log_faithful.png", dpi=150)
    plt.close(fig)


def fig_placement(r: dict[str, Any]) -> None:
    st, u = r["states"], r["uniqueness"]
    rows = [("lowercase: batch order 1 vs order 2", u["cos"], BLUE),
            ("lowercase vs capitals-kept training", u["cos_lower_plain"], ORANGE),
            ("lowercase training vs the fine-tune", st["lower_s0"]["cos_dW"], BLUE),
            ("adjoint split vs the fine-tune", st["adjoint"]["cos_dW"], GREY),
            ("random direction vs the fine-tune", u["null"], MUTED)]
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    for i, (label, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.6, color=c, zorder=3)
        ax.text(max(v, 0) + 0.015, y, f"{0.0 if abs(v) < 0.005 else v:.2f}", fontsize=8, va="center")
    ax.set_yticks(range(len(rows)), [row[0] for row in rows][::-1], fontsize=8.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("cosine between two weight changes over all 24 matrices (1 = same direction)")
    ax.set_title("Two runs in different batch orders land on nearly the same weight change", fontsize=10,
                 loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "log_placement.png", dpi=150)
    plt.close(fig)


RUNS = {"arm1": ("from the released decomposition", BLUE, "o", "-"),
        "arm3": ("from the adjoint split", PURPLE, "s", "--")}


def panel_ratio(ax: Axes, a: dict[str, Any]) -> None:
    verdict_lines(ax, a, vertical=False)
    pass_l, fail_l = lines(a)
    for arm, (lab, c, m, ls) in RUNS.items():
        x, y = series(a, arm, "check")
        ax.plot(x, y, color=c, marker=m, ls=ls, lw=1.6, ms=7, label=lab, zorder=4)
        for xi, yi in zip(x, y, strict=True):
            if arm == "arm1" and xi == 0:
                ax.annotate("1.00 unchanged", (xi, yi), xytext=(6, -12), textcoords="offset points", fontsize=7.5)
                continue
            ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(6, 6 if arm == "arm1" or xi == 0 else -12), textcoords="offset points",
                        fontsize=7.5)
    ax.text(4150, pass_l - 0.03, f"passes ≤ {pass_l:.2f}", fontsize=7.5, color=PASS_C, ha="right", va="top")
    ax.text(4150, fail_l + 0.03, f"fails ≥ {fail_l:.2f}", fontsize=7.5, color=FAIL_C, ha="right", va="bottom")
    ax.set_ylim(0.85, 2.6)
    ax.set_ylabel("adversarial ratio\n(1 = unchanged decomposition)")
    ax.set_title("The adversarial ratio stays above the fail line", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="lower right", frameon=False, title="teacher training started", title_fontsize=8)


def panel_gap(ax: Axes, a: dict[str, Any]) -> None:
    for arm, (_, c, m, ls) in RUNS.items():
        x, y = series(a, arm, "gap")
        ax.plot(x, y, color=c, marker=m, ls=ls, lw=1.6, ms=7, zorder=4)
        for xi, yi in zip(x[1:], y[1:], strict=True):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(6, 6 if arm == "arm1" else -12), textcoords="offset points",
                        fontsize=7.5)
    ax.annotate(f"unchanged {a['states']['W0']['gap']:.2f}", (0, a["states"]["W0"]["gap"]), xytext=(8, 0),
                textcoords="offset points", fontsize=7.5, va="center")
    ax.annotate(f"adjoint split {a['states']['arm3_0']['gap']:.2f}", (0, a["states"]["arm3_0"]["gap"]), xytext=(8, -2),
                textcoords="offset points", fontsize=7.5, va="top")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 0.5)
    ax.set_ylabel("gap to the fine-tuned model\n(nats per token, 0 = identical)")
    ax.set_title("while the match to the fine-tuned model keeps improving", fontsize=9.5, loc="left")


def fig_steps(a: dict[str, Any]) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    panel_ratio(left, a)
    panel_gap(right, a)
    for ax in (left, right):
        ax.set_xlim(-200, 4300)
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_xlabel("teacher-training steps")
        style(ax)
    fig.suptitle("Training four times longer, or from a start that passes, does not bring the ratio under the line",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "log_steps.png", dpi=150)
    plt.close(fig)


def fig_masked(a: dict[str, Any]) -> None:
    d = a["score"]["derived"]

    def dmg(n: str) -> str:
        return f"   shift on Pile text {d[n]['damage']:.3f}"

    rows = [("unchanged decomposition", 1.0, INK, ""),
            ("teacher training, 1,000 steps", d["arm1_1000"]["check"], BLUE, dmg("arm1_1000")),
            ("masked losses, nothing to copy", d["M0_1000"]["check"], BROWN, dmg("M0_1000")),
            ("the same at a tenth of their weight", d["M0b_1000"]["check"], BROWN, dmg("M0b_1000"))]
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ratio_bars(ax, a, rows)
    ax.set_xlim(0.9, 5.8)
    ax.set_title("With nothing to copy, VPD's masked losses alone push the ratio\n"
                 "higher than copying does, so they could not be tried as a fix", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "log_masked.png", dpi=150)
    plt.close(fig)


def main() -> None:
    r, a = load("teacher_spread.json"), load("amend1.json")
    fig_faithful(r)
    fig_placement(r)
    fig_steps(a)
    fig_masked(a)
    print(f"wrote {sorted(p.name for p in FIG.glob('log_*.png'))}")


if __name__ == "__main__":
    main()
