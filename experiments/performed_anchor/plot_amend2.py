"""Figures for E53 Amendment 2 (kind or breadth, settled with saved states), read from results/read_amend2.json,
results/read_amend1b.json (stand-in ratios of E53's and the broad cipher's step-300 states, E53 cipher seed 1's
breadth) and E54's stage2.json (its stand-in ratios, through read_amend2.json's score); nothing is recomputed from the
model. Plan: reports/plans/2026-10-29_plan_performed-anchor.md, section "Amendment 2"; analysis:
reports/analyses/2026-10-29_analysis_e53-performed-anchor.md, section "Amendment 2".

  a2_breadth.png        P1: lowercase's breadth before and after the correction, against the broad cipher's, with
                        90% intervals and the bar at the broad cipher's higher seed
  a2_ratio_breadth.png  P2 and the reading: stand-in ratio against breadth for both pairs (E53 lowercase-cipher, E54
                        elicit-cipher), each seed's pair joined; breadth compares within a pair only

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/performed_anchor/plot_amend2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
A2 = json.loads((HERE / "results" / "read_amend2.json").read_text())
A1B = json.loads((HERE / "results" / "read_amend1b.json").read_text())["rows"]
ROWS, S = A2["rows"], A2["score"]
FIG = HERE / "figures"
LOW_C, CIP_C, BROAD_C, ELI_C, E54C_C = "#2a78d6", "#eb6834", "#7a4fc9", "#1a9e77", "#b8336a"
INK, MUTED = "#0b0b0b", "#8a8984"
MARK = {0: "o", 1: "s"}


def ratio_1b(key: str) -> float:
    r = A1B[key]
    return r["masked_gain"] / r["piece_gain"]


def e53_cipher_breadth(s: int) -> float:
    """Seed 0 re-read here (identity row: corrected = uncorrected); seed 1 from 1b's true KL, equal by the identity."""
    r = ROWS["identity/cipher"] if s == 0 else A1B["e53/cipher/s1/step300"]
    return r["G"] / (r["kl_off_changed"] if s == 0 else r["kl_off"])


def fig_breadth() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    rows = [("E53 lowercase, old count", LOW_C, "none"), ("E53 lowercase, corrected", LOW_C, None),
            ("broad cipher, step 300", BROAD_C, None)]
    for s in (0, 1):
        old = S["breadth_uncorrected_reported"][f"lowercase/s{s}/step300"]
        ax.plot(old, 0, MARK[s], mfc="none", color=LOW_C, ms=8)
        for i, key in ((1, f"lowercase/s{s}/step300"), (2, f"broad_cipher/s{s}/step300")):
            b, (lo, hi) = S["breadth"][key], S["breadth_ci90_reported"][key]
            y = i + (0.12 if s else -0.12)
            ax.plot([lo, hi], [y, y], color=rows[i][1], lw=1.5)
            ax.plot(b, y, MARK[s], color=rows[i][1], ms=8, label=f"seed {s}" if i == 1 else None)
        ax.annotate("", xy=(S["breadth"][f"lowercase/s{s}/step300"], 1 + (0.12 if s else -0.12)), xytext=(old, 0),
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})
    bar = S["P1"]["max_broad_cipher"]
    ax.axvline(bar, color=BROAD_C, ls=":", lw=1.2)
    ax.text(bar + 0.08, -0.45, f"filed bar {bar:.2f}: lowercase's lower seed\nmust reach the broad cipher's higher one",
            fontsize=8, color=BROAD_C, va="bottom")
    ax.set_yticks(range(3), [r[0] for r in rows])
    ax.set_ylim(-0.6, 2.5)
    ax.set_xlim(4, 10)
    ax.set_xlabel("breadth: gain per nat of off-target KL, Pile windows (lower = broader); bars: 90% interval")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Corrected, is E53's lowercase as targeted as the broad cipher?", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "a2_breadth.png", dpi=130)
    plt.close(fig)


def fig_ratio_breadth() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for y, lab in ((0.7, "E53's lean for a performed task: at least 0.7"), (0.5, "E53's lean for a new task: at most 0.5")):
        ax.axhline(y, color=INK, ls="--", lw=0.8)
        ax.text(4.3, y + 0.015, lab, fontsize=8)
    for s in (0, 1):
        lo_b, lo_r = S["breadth"][f"lowercase/s{s}/step300"], ratio_1b(f"e53/lowercase/s{s}/step300")
        ci_b, ci_r = e53_cipher_breadth(s), ratio_1b(f"e53/cipher/s{s}/step300")
        ax.plot([lo_b, ci_b], [lo_r, ci_r], "-", color=MUTED, lw=1, zorder=1)
        ax.plot(lo_b, lo_r, MARK[s], color=LOW_C, ms=9, label=f"E53 lowercase, performed, seed {s}")
        ax.plot(ci_b, ci_r, MARK[s], color=CIP_C, ms=9, label=f"E53 cipher, new, seed {s}")
        bb, br = S["breadth"][f"broad_cipher/s{s}/step300"], ratio_1b(f"broad_cipher/s{s}/step300")
        ax.plot(bb, br, MARK[s], color=BROAD_C, ms=8, label=f"E53 broad cipher, new, seed {s}")
        el_b, el_r = S["breadth"][f"e54/elicit_s{s}"], S["e54_ratio_reported"][f"elicit_s{s}"]
        c4_b, c4_r = S["breadth"][f"e54/cipher_s{s}"], S["e54_ratio_reported"][f"cipher_s{s}"]
        ax.plot([el_b, c4_b], [el_r, c4_r], "-", color=MUTED, lw=1, zorder=1)
        ax.plot(el_b, el_r, MARK[s], color=ELI_C, ms=9, label=f"E54 elicit map, performed, seed {s}")
        ax.plot(c4_b, c4_r, MARK[s], color=E54C_C, ms=9, label=f"E54 cipher, new, seed {s}")
    ax.text(5.6, 1.0, "E53 pair: 127-token Pile windows\nperformed member is the broader", fontsize=8, color=MUTED)
    ax.text(18.5, 1.0, "E54 pair: 41-token marker windows\nperformed member is the more targeted", fontsize=8,
            color=MUTED)
    ax.text(4.3, 0.03, "grey lines join each seed's two members of one pair; breadth compares within a pair only, "
            "not across pairs", fontsize=7.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax.set_xlim(4, 70)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("breadth: full-model gain per nat of off-target KL, log scale (lower = broader)")
    ax.set_ylabel("stand-in ratio: masked gain / subcomponent gain")
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("Does the high-ratio member of each pair have to be the broader one?", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "a2_ratio_breadth.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_breadth()
    fig_ratio_breadth()


if __name__ == "__main__":
    main()
