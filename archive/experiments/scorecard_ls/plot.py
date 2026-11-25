"""E71's figures from results/reads.json (job 50287), results/predictions.json and the block KLs in results/run.log; no model.

  figures/check.png       the check ratio per state for both fine-tunes, with the pass and fail lines and E39's H100 controls
  figures/keep_delta.png  share of the gain kept against share of the update left in Delta, pass or fail marked

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/scorecard_ls/plot.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
ADAM, LORA, INK, GRID, PASS, FAIL = "#2f6db5", "#d9822b", "#52514e", "#e4e3df", "#2e8b57", "#b0413e"
FT = (("adam", "full-matrix Adam", ADAM, -0.15), ("lora", "rank-8 LoRA", LORA, 0.15))
STATES = (("nothing", "do-nothing\n(all in Δ)"), ("ls", "least-squares\nsplit"),
          ("expr", "expressible part\n(Δ moves by the\nsecond-order term)"),
          ("expr_fixed", "expressible part\n(Δ exactly fixed)"), ("adjoint", "scaled adjoint\nsplit"))
L40_PASS = 1.241  # E47's pass line for these rows on the L40 (reports/analyses/2026-10-29_analysis_e47-claim-tests.md)


def block_ratios(w0: float) -> dict[str, list[float]]:
    """Each state's two block KLs over W0's mean KL. The check is the mean of the two."""
    out: dict[str, list[float]] = {}
    pending: list[float] = []
    for line in (RES / "run.log").read_text().splitlines():
        if m := re.search(r"\[faith\].*pgd_kl ([\d.]+)", line):
            pending.append(float(m[1]))
        elif (m := re.search(r"\[(\w+)\] check ", line)):
            out[m[1]] = [k / w0 for k in pending]
            pending = []
        elif "[pile W0" in line:
            pending = []
    return out


def check_figure(R: dict, pred: dict, w0: float) -> None:
    blocks = block_ratios(w0)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=170)
    for i, (st, _) in enumerate(STATES):
        for ft, label, col, dx in FT:
            v = R[f"{st}_{ft}"]["check"]
            ax.plot([i + dx, i + dx], [1, v], color=col, lw=5, alpha=0.45, solid_capstyle="butt", zorder=2)
            ax.scatter(i + dx, v, color=col, s=46, zorder=4, label=label if i == 0 else None)
            ax.scatter([i + dx] * 2, blocks[f"{st}_{ft}"], color="white", edgecolor=col, s=20, zorder=3,
                       label="each block of 128 rows" if i == 0 and ft == "lora" else None)
            ax.annotate(f"{v:.2f}", (i + dx, v), xytext=(-7, 0) if ft == "adam" else (17, 0), textcoords="offset points", ha="right" if ft == "adam" else "left", va="center", fontsize=8, color=col)
            h = pred["controls"]["h100"][ft].get(st)
            if h:
                ax.scatter(i + dx + 0.09, h, marker="D", color="#8a8984", s=20, zorder=5,
                           label="same state, earlier H100 read" if i == 0 and ft == "adam" else None)
    lines = pred["lines"]
    ax.axhline(1, color=INK, lw=1)
    ax.axhline(lines["pass"], color=PASS, lw=1, ls="--")
    ax.axhline(lines["fail"], color=FAIL, lw=1, ls="--")
    ax.axhline(L40_PASS, color=PASS, lw=0.8, ls=":")
    tx = len(STATES) - 0.45
    ax.text(tx, 1.0, "unchanged model 1", fontsize=7.5, color=INK, va="top")
    ax.text(tx, L40_PASS, f"pass, L40's own floor {L40_PASS}", fontsize=7.5, color=PASS, va="center", bbox={"fc": "white", "ec": "none", "pad": 1})
    ax.text(tx, lines["pass"] * 1.015, f"pass {lines['pass']:.2f}", fontsize=7.5, color=PASS, va="bottom")
    ax.text(tx, lines["fail"], f"fail {lines['fail']:.2f}", fontsize=7.5, color=FAIL, va="bottom")
    ax.set_yscale("log")
    ax.set_yticks([0.8, 1, 1.4, 2, 3, 5, 8, 12, 16], labels=["0.8", "1", "1.4", "2", "3", "5", "8", "12", "16"])
    ax.minorticks_off()
    ax.set_ylim(0.75, 19)
    ax.set_xlim(-0.5, len(STATES) + 0.75)
    ax.set_xticks(range(len(STATES)), [s[1] for s in STATES], fontsize=8.5)
    ax.set_ylabel("adversarial ratio, one attacker start (log scale)", fontsize=9, color=INK)
    ax.set_title("Least squares fails the check on Adam and passes on the rank-8 LoRA, at one attacker start", fontsize=10.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncols=2)
    fig.tight_layout()
    fig.savefig(FIG / "check.png")
    plt.close(fig)


def keep_figure(R: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=170)
    ax.fill_between([-0.03, 0.1], 0.9, 1.28, color="#d3ead6", alpha=0.6, zorder=0)
    ax.text(0.11, 1.22, "shaded: keeps at least 0.9 of the gain, leaves at most 0.1 in Δ", fontsize=7.5, color=PASS, ha="left", va="center")
    ax.axhline(1, color=INK, lw=0.8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.text(0.97, 1.0, "the fine-tune's gain", fontsize=7.5, color=INK, ha="right", va="bottom")
    ax.text(1.1, 0.01, "the unchanged model", fontsize=7.5, color=INK, ha="right", va="bottom")
    for st, name in (("nothing", "do-nothing"), ("ls", "least squares"), ("expr_fixed", "expressible part, Δ fixed"), ("adjoint", "scaled adjoint")):
        for ft, _, col, _ in FT:
            r = R[f"{st}_{ft}"]
            face = col if r["verdict"] == "pass" else "white"
            y = r["s"] + (0 if st not in ("nothing", "ls") else (0.03 if ft == "adam" else -0.03))
            ax.scatter(r["delta_share"], y, s=70, facecolor=face, edgecolor=col, lw=1.8, zorder=3)
            ax.annotate(f"{name}, {r['check']:.2f}", (r["delta_share"], y), xytext=(8, 0), textcoords="offset points", fontsize=8, color=col, va="center")
    ax.scatter([], [], s=70, facecolor=INK, edgecolor=INK, label="passes the check")
    ax.scatter([], [], s=70, facecolor="white", edgecolor=INK, lw=1.8, label="fails the check")
    ax.scatter([], [], s=40, color=ADAM, label="full-matrix Adam")
    ax.scatter([], [], s=40, color=LORA, label="rank-8 LoRA")
    ax.set_xlim(-0.03, 1.2)
    ax.set_ylim(-0.08, 1.28)
    ax.text(1.19, -0.06, "at 1.0 the two fine-tunes are drawn just apart; both are exactly 1.0", fontsize=7, color=INK, ha="right")
    ax.set_xlabel("share of the update left in Δ (0 = none, 1 = all)", fontsize=9, color=INK)
    ax.set_ylabel("share of the fine-tune's held-out gain kept", fontsize=9, color=INK)
    ax.set_title("Only the LoRA's expressible part keeps the gain, leaves Δ alone and passes, at one start", fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center", bbox_to_anchor=(0.6, 0.42), title="number beside each point: the adversarial ratio", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "keep_delta.png")
    plt.close(fig)


def main() -> None:
    reads, pred = json.loads((RES / "reads.json").read_text()), json.loads((RES / "predictions.json").read_text())
    FIG.mkdir(exist_ok=True)
    check_figure(reads["reads"], pred, reads["w0"]["pgd_kl"])
    keep_figure(reads["reads"])


if __name__ == "__main__":
    main()
