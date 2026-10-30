"""Figures for E44 Amendment 2 (reading B: every state's CI masks from the original model's activations), step 10.

amend2_kept.png: share of the update's behaviour kept, r2_B, per placement, with the filed expected ranges and the
pass line. amend2_rows.png: kept share per unit of the update carried, rho, per maths row, often-on half against
rarely-on half, with the no-separation diagonal and the filed margin.

Reads results/amend2/{score,controls,t10}.json only (no model). Writes figures/amend2_kept.png, amend2_rows.png.
Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/qwen_maps/plot_amend2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "amend2"
FIG = HERE / "figures"
LOWER, UPPER, RANDOM, DN = "#7f8c8d", "#c0392b", "#2a78d6", "#1e1e1e"
MARGIN, RANDOM_MAX = 0.2, 0.2  # filed teeth_i margin; filed teeth_ii rule, random r2_B <= 0.2
PASS_R2 = 0.977  # r2_B the r1 line asks at step 10 on the first 16 rows (pass.note); about 0.973 on all 128


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def per_row_rho(c: dict, t: dict, carried: float, half: str) -> list[float]:
    """rho per row: [K(D) - K_B(half)] / [K(D) - K0] on that row, divided by the half's carried share."""
    kd, k0, kb = c["KD_10"]["sums"]["K"], c["W0_maths"]["sums"]["ci"], t[half]["sums"]["K"]
    return [(d - b) / (d - z) / carried for d, z, b in zip(kd, k0, kb, strict=True)]


def kept_figure(sc: dict) -> None:
    ti, tii = sc["teeth_10"]["teeth_i"], sc["teeth_10"]["teeth_ii"]
    rows = [  # label, r2_B, colour, filed expected range (random: expected <= 0, drawn from the axis edge)
        ("do-nothing: the whole update\nleft in Δ", sc["control_do_nothing"]["10"]["r2"], DN, (0.0, 0.0)),
        (f"random change to the alive (U, V),\nKL-matched (×{tii['a']:.1f} E44's size)", tii["r2"], RANDOM,
         (-0.08, 0.0)),
        ("least-squares split onto the half\nΓ rarely switches on", ti["r2_lower"], LOWER, (-0.05, 0.10)),
        ("least-squares split onto the half\nΓ often switches on", ti["r2_upper"], UPPER, (0.10, 0.60)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (_, v, col, (lo, hi)) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.5)
        ax.add_patch(Rectangle((lo, i - 0.32), max(hi - lo, 0.004), 0.64, fc="none", ec="0.35", ls="--", lw=1))
        ax.text(max(v, hi) + 0.015, i, f"{v:.3f}", va="center", fontsize=9)
    ax.plot([RANDOM_MAX] * 2, [1 - 0.32, 1 + 0.32], color=RANDOM, lw=2)
    ax.text(RANDOM_MAX + 0.01, 1 + 0.2, "filed rule: ≤ 0.2", fontsize=8, color=RANDOM)
    ax.axvline(0, color="0.6", lw=1)
    ax.axvline(1, color="0.6", lw=1, ls=":")
    ax.axvline(PASS_R2, color="0.3", lw=1.2, ls="-.")
    ax.text(PASS_R2 - 0.01, 2.55, "kept share a state needs to pass\nat step 10: 0.977 on the first 16 rows,\nabout 0.973 on all 128",
            ha="right", fontsize=8,
            color="0.3", va="top")
    ax.text(1.01, 3.55, "1 = the\nfine-tuned\nmodel", ha="left", fontsize=8, color="0.4", va="top")
    ax.plot([], [], ls="--", color="0.35", label="filed expected range (box); random: expected ≤ 0")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.7, 3.7)
    ax.set_xlim(-0.08, 1.12)
    ax.set_xlabel("share of the update's behaviour kept, r2_B\n0 = original decomposition at W₀'s masks, "
                  "1 = the fine-tuned model")
    ax.set_title("With masks from the original model, how much of the update\ndoes each placement keep? (step 10)",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "amend2_kept.png", dpi=150)
    plt.close(fig)


def rows_figure(sc: dict, c: dict, t: dict) -> None:
    ti = sc["teeth_10"]["teeth_i"]
    lo = per_row_rho(c, t, ti["carried_lower"], "lower")
    up = per_row_rho(c, t, ti["carried_upper"], "upper")
    above = sum(u > v for u, v in zip(up, lo, strict=True))
    print(f"per-row rho: lower [{min(lo):.4f}, {max(lo):.4f}], upper [{min(up):.4f}, {max(up):.4f}], "
          f"upper above lower on {above}/{len(lo)} rows, upper - lower >= {MARGIN} on "
          f"{sum(u - v >= MARGIN for u, v in zip(up, lo, strict=True))} rows")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.add_patch(Rectangle((-0.17, 0.21), 0.52, 1.05, fc="#f3e1df", ec="none", zorder=0,
                               label="filed expected box (often-on half to 1.26)"))
    xs = [-0.2, 0.4]
    ax.plot(xs, xs, color="0.5", lw=1, label="no separation: both halves keep the same per unit")
    ax.plot(xs, [x + MARGIN for x in xs], color="0.2", lw=1.2, ls="--",
            label=f"filed margin: often-on must keep {MARGIN} more per unit")
    ax.scatter(lo, up, s=14, color=UPPER, alpha=0.55, label=f"one maths row ({len(lo)} rows)")
    ax.scatter([ti["rho_lower"]], [ti["rho_upper"]], s=110, marker="D", color="black", zorder=3,
               label=f"all rows together: {ti['rho_lower']:.3f} and {ti['rho_upper']:.3f}")
    ax.set_xlim(-0.2, 0.4)
    ax.set_ylim(-0.05, 0.6)
    ax.set_xlabel("kept share per unit carried, half Γ rarely switches on")
    ax.set_ylabel("kept share per unit carried, half Γ often switches on")
    ax.set_title("Does the update keep more of its behaviour on subcomponents\nΓ switches on at W₀? (step 10)",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "amend2_rows.png", dpi=150)
    plt.close(fig)


def main() -> None:
    sc, c, t = load("score.json"), load("controls.json"), load("t10.json")
    FIG.mkdir(exist_ok=True)
    kept_figure(sc)
    rows_figure(sc, c, t)
    print(f"wrote {FIG / 'amend2_kept.png'}, {FIG / 'amend2_rows.png'}")


if __name__ == "__main__":
    main()
