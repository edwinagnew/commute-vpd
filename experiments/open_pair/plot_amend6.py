"""E58 Amendment 6's figures, from results/amend6.json (job 46489) only. No model.

  figures/a6_prior.png        r at step 30 against W0's prior of the trained value (geometric-mean probability after
                              the marker and key, log scale) for all seven maps, three seeds each and the seed mean;
                              the prior-matched cipher marked; the filed prior band [2e-5, 2e-4], the 0.5 cut and the
                              0.6 upper band; the line through the shuffle's and copy's seed means, one smooth
                              prior-only trend fitted after the data (the code review's power check); the first-letter
                              rule's prior as a tick on the x axis only, with no r, since it did not carry
  figures/a6_first_letter.png the first-letter rule's held-out carry x, its null x_null and x - x_null per seed, at
                              each run's first step over the gate and at step 30, with the filed entry bar 0.3

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/open_pair/plot_amend6.py
"""

from __future__ import annotations

import json
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import COLOR, FIG, GRID, HERE, INK, SHADE, style

COLOR = {
    **COLOR,
    "rank1": "#7b5ab8",
    "copy": "#2a78d6",
    "shuffle": "#6b6a66",
    "prior_cipher": "#c2307a",
    "first_letter": "#8a6d3b",
}
NAME = {
    "cipher": "cipher (no rule)",
    "shuffle": "shuffled elicit",
    "prior_cipher": "prior-matched cipher (new)",
    "copy": "copy (value = key)",
    "elicit": "elicit (half-known)",
    "rank1": "rank-1 (W0's top word)",
    "first_letter": "first-letter rule (new)",
}
SEEDS = (0, 1, 2)
R_LABEL = "r at step 30: share of the update's gain\non W0's own subcomponents"


def load() -> dict[str, Any]:
    return json.loads((HERE / "results" / "amend6.json").read_text())


def mean(v: list[float]) -> float:
    return sum(v) / len(v)


def prior_only_line(pb: dict[str, Any]) -> tuple[float, float]:
    """Slope and intercept of seed-mean r on log10 prior through the shuffle and copy, the two maps either side of the
    prior-matched cipher's prior."""
    (p0, r0), (p1, r1) = (
        (math.log10(pb[m]["w0_geo_p_trained"]), mean(pb[m]["r30"]))
        for m in ("shuffle", "copy")
    )
    slope = (r1 - r0) / (p1 - p0)
    return slope, r0 - slope * p0


def prior_fig(a6: dict[str, Any]) -> None:
    pb = a6["prior_beside_r"]
    lo, hi = json.loads((HERE / "results" / "predictions_amend6.json").read_text())[
        "settings"
    ]["band"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axvspan(lo, hi, color=SHADE, zorder=0)
    ax.text(
        math.sqrt(lo * hi),
        0.995,
        "filed prior band\nfor the new cipher",
        fontsize=6.5,
        color=INK,
        ha="center",
        va="top",
    )
    ax.axhline(0.5, color=INK, lw=0.8, ls="--", zorder=1)
    ax.text(
        2.2e-3,
        0.505,
        "0.5 cut: under it on every seed\n= cipher band",
        fontsize=6.5,
        color=INK,
        va="bottom",
    )
    ax.axhline(0.6, color=INK, lw=0.8, ls=":", zorder=1)
    ax.text(
        2.2e-3,
        0.605,
        "0.6: at or over it on every seed\n= own-guess band",
        fontsize=6.5,
        color=INK,
        va="bottom",
    )
    slope, icpt = prior_only_line(pb)
    xs = [5e-6, 2e-4]
    ax.plot(
        xs,
        [icpt + slope * math.log10(x) for x in xs],
        color=INK,
        lw=1.0,
        ls="-.",
        zorder=1,
        label="one smooth prior-only trend, fitted after the data:\nline through shuffle and copy means",
    )
    for m in ("cipher", "shuffle", "prior_cipher", "copy", "elicit", "rank1"):
        p, rs = pb[m]["w0_geo_p_trained"], pb[m]["r30"]
        big = m == "prior_cipher"
        ax.scatter([p] * len(rs), rs, s=14, color=COLOR[m], alpha=0.5, zorder=3)
        ax.scatter(
            [p],
            [mean(rs)],
            s=150 if big else 50,
            marker="*" if big else "o",
            color=COLOR[m],
            edgecolors=INK if big else "none",
            lw=0.6,
            zorder=4,
            label=f"{NAME[m]}: seeds and mean",
        )
    pc = pb["prior_cipher"]
    pred = icpt + slope * math.log10(pc["w0_geo_p_trained"])
    ax.annotate(
        f"prior-matched cipher: prior {pc['w0_geo_p_trained']:.1e}, r {mean(pc['r30']):.2f};\n"
        f"prior-only line reads {pred:.2f} here; copy, prior {pb['copy']['w0_geo_p_trained']:.1e}, r "
        f"{mean(pb['copy']['r30']):.2f}",
        (pc["w0_geo_p_trained"], mean(pc["r30"])),
        xytext=(2.4e-4, 0.30),
        fontsize=6.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6},
    )
    # The first-letter rule has no r (it did not carry): its prior is marked on the x axis only, never at an r value.
    fl = pb["first_letter"]["w0_geo_p_trained"]
    ax.axvline(fl, ymin=0, ymax=0.06, color=COLOR["first_letter"], lw=2.0, zorder=4)
    ax.text(
        fl * 1.15,
        0.165,
        "first-letter rule's prior (x axis only):\nno r read, it did not carry",
        fontsize=6.5,
        color=COLOR["first_letter"],
        va="bottom",
    )
    ax.text(
        2.4e-4,
        0.22,
        "a cut on the prior anywhere between the star and copy\nstill sorts all six maps into the two bands",
        fontsize=6.5,
        color=INK,
    )
    ax.set_xscale("log")
    ax.set_xlim(7e-7, 3e-2)
    ax.set_ylim(0.15, 1.0)
    ax.set_xlabel(
        "W0's prior of the trained value after its key (geometric-mean probability, log scale)",
        color=INK,
    )
    style(
        ax,
        "At a prior near copy's, does r read like copy or like the cipher? (96 trained keys, step 30)",
        R_LABEL,
    )
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "a6_prior.png", dpi=150)
    plt.close(fig)
    print(
        f"prior-only line: slope {slope:.4f} per decade; at the prior-matched cipher's prior {pred:.4f}"
    )
    for m, d in pb.items():
        r30 = d["r30"]
        print(
            m,
            f"prior {d['w0_geo_p_trained']:.3e}",
            "r30",
            [round(r, 4) for r in r30],
            f"mean {mean(r30):.4f}" if r30 else "not read",
        )
    print(
        "copy / prior-matched prior ratio",
        round(pb["copy"]["w0_geo_p_trained"] / pc["w0_geo_p_trained"], 3),
        "| Spearman",
        a6["rho_r_logprior"],
    )


def first_letter_fig(a6: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    entry = json.loads((HERE / "results" / "predictions_amend6.json").read_text())[
        "thresholds"
    ]["x_entry"]
    ax.axhline(entry, color=INK, lw=0.9, ls="--", zorder=1)
    ax.text(
        5.45,
        entry + 0.008,
        f"entry bar: x - x_null over {entry}",
        fontsize=6.5,
        color=INK,
        ha="right",
    )
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    series = (
        ("x", "held-out carry x", "o", "white"),
        ("x_null", "x_null: frequency-only null", "s", "white"),
        ("x_minus_null", "x - x_null (scored)", "o", COLOR["first_letter"]),
    )
    for j, (label, step_of) in enumerate(
        (("first step over the gate", "gate"), ("step 30 (scored)", "30"))
    ):
        for s in SEEDS:
            step = a6["gate_step"][f"first_letter_s{s}"] if step_of == "gate" else 30
            d = a6["reads"][f"first_letter_s{s}@{step}"]
            x0 = j * 3 + s
            for k, (key, name, mk, face) in enumerate(series):
                ax.scatter(
                    [x0 + 0.2 * (k - 1)],
                    [d[key]],
                    marker=mk,
                    s=36,
                    color=face,
                    edgecolors=COLOR["first_letter"],
                    lw=1.2,
                    zorder=3,
                    label=name if (j, s) == (0, 0) else None,
                )
            ax.text(
                x0,
                -0.08,
                f"seed {s}\nstep {step}",
                fontsize=6.5,
                color=INK,
                ha="center",
                va="top",
            )
        ax.text(j * 3 + 1, 0.47, label, fontsize=7, color=INK, ha="center")
    ax.set_xticks([])
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(-0.17, 0.5)
    style(
        ax,
        "Did the first-letter rule carry to unseen keys beyond what letter frequency gives?",
        "share of the trained-key gain\nreached at the 32 unseen keys",
    )
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.legend(
        fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.92), frameon=False
    )
    fig.tight_layout()
    fig.savefig(FIG / "a6_first_letter.png", dpi=150)
    plt.close(fig)
    for key in ("x", "x_null", "x_minus_null"):
        print(
            "FL",
            key,
            "step 30",
            [round(v, 4) for v in a6["score"]["FL"][key]],
            "| gate",
            [round(v, 4) for v in a6["at_gate_step"]["FL"][key]],
        )


def main() -> None:
    a6 = load()
    prior_fig(a6)
    first_letter_fig(a6)
    print("PM", a6["score"]["PM"], "| at gate", a6["at_gate_step"]["PM"]["r"])
    print(
        "FL class",
        a6["score"]["FL"]["class"],
        "| harness",
        a6["harness_control"],
        "| W0",
        a6["w0_control"],
    )


if __name__ == "__main__":
    main()
