"""E58 Amendment 7's figure, from results/amend7.json (job 46673), with the step-30 carry of the earlier maps from
amend3.json (seed 0 of elicit, rank-1, copy), amend4.json, amend5.json and amend6.json. No model.

  figures/a7_r.png  two panels sharing the r axis, every map at step 30, three seeds and the seed mean:
                    left, r against W0's prior of the trained value (log scale); right, r against the held-out carry,
                    x - x_null for the class map and its twin (the cross-class null), x for the maps that have no class
                    null. The class map (filled star) and its shuffled-class twin (hollow diamond) share a colour.
                    Marked: the 0.5 cut, the 0.6 upper band, the entry bar 0.3 and the twin bar 0.15 on the carry axis,
                    and the filed D bound (twin mean +/- 0.0904) on the r axis. The elicit-prior cipher was skipped and
                    has no point.

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/open_pair/plot_amend7.py
"""

from __future__ import annotations

import json
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import COLOR, FIG, GRID, HERE, INK, style

COLOR = {
    **COLOR,
    "rank1": "#7b5ab8",
    "copy": "#2a78d6",
    "shuffle": "#6b6a66",
    "prior_cipher": "#c2307a",
    "class_map": "#b8860b",
    "class_shuffled": "#b8860b",
}
NAME = {
    "cipher": "cipher (no rule)",
    "shuffle": "shuffled elicit",
    "prior_cipher": "prior-matched cipher",
    "copy": "copy (value = key)",
    "elicit": "elicit (half-known)",
    "rank1": "rank-1 (W0's top word)",
    "class_map": "class map: new 4-word rule (new)",
    "class_shuffled": "its shuffled-class twin (new)",
}
ORDER = (
    "cipher",
    "shuffle",
    "prior_cipher",
    "copy",
    "elicit",
    "rank1",
    "class_shuffled",
    "class_map",
)
SEEDS = (0, 1, 2)


def read(name: str) -> dict[str, Any]:
    return json.loads((HERE / "results" / f"{name}.json").read_text())


def mean(v: list[float]) -> float:
    return sum(v) / len(v)


def carry30(a7: dict[str, Any]) -> dict[str, list[float]]:
    """Step-30 held-out carry per map and seed: x - x_null for the class map and twin, x for the rest."""
    a3, a4, a5, a6 = (
        read(n)["reads"] for n in ("amend3", "amend4", "amend5", "amend6")
    )
    out = {
        m: [a7["reads"][f"{m}_s{s}@30"]["x_minus_null"] for s in SEEDS]
        for m in ("class_map", "class_shuffled")
    }
    out["cipher"] = [a4[f"cipher_s{s}@30"]["x"] for s in SEEDS]
    for m in ("copy", "elicit", "rank1"):
        out[m] = [a3[f"{m}@30"]["x"]] + [a4[f"{m}_s{s}@30"]["x"] for s in (1, 2)]
    out["shuffle"] = [a4["shuffle_s0@30"]["x"]] + [
        a5[f"shuffle_s{s}@30"]["x"] for s in (1, 2)
    ]
    out["prior_cipher"] = [a6[f"prior_cipher_s{s}@30"]["x"] for s in SEEDS]
    return out


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v: list[float]) -> list[int]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0] * len(v)
        for r, i in enumerate(order):
            rk[i] = r
        return rk

    ra, rb, n = ranks(a), ranks(b), len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def points(ax: Any, m: str, xs: list[float], rs: list[float], label: bool) -> None:
    star, twin = m == "class_map", m == "class_shuffled"
    face = "white" if twin else COLOR[m]
    ax.scatter(
        xs, rs, s=14, color=face, edgecolors=COLOR[m], lw=0.8, alpha=0.6, zorder=3
    )
    ax.scatter(
        [mean(xs)],
        [mean(rs)],
        s=180 if star else 70 if twin else 50,
        marker="*" if star else "D" if twin else "o",
        color=face,
        edgecolors=INK if star else COLOR[m],
        lw=1.2 if twin else 0.6,
        zorder=4,
        label=f"{NAME[m]}: seeds and mean" if label else None,
    )


def cuts(ax: Any, x_text: float) -> None:
    ax.axhline(0.5, color=INK, lw=0.8, ls="--", zorder=1)
    ax.text(
        x_text,
        0.505,
        "0.5 cut: under it on every seed = cipher band",
        fontsize=6.5,
        color=INK,
        va="bottom",
    )
    ax.axhline(0.6, color=INK, lw=0.8, ls=":", zorder=1)
    ax.text(
        x_text,
        0.605,
        "0.6: over it on every seed = own-guess band",
        fontsize=6.5,
        color=INK,
        va="bottom",
    )


def main() -> None:
    a7, th = read("amend7"), read("predictions_amend7")["thresholds"]
    pb, cx = a7["prior_beside_r"], carry30(a7)
    fig, (axp, axc) = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)
    twin_mean = mean(pb["class_shuffled"]["r30"])
    for ax in (axp, axc):
        ax.axhspan(
            twin_mean - th["r_contrast"],
            twin_mean + th["r_contrast"],
            color="#f3e6c4",
            zorder=0,
        )
    for m in ORDER:
        p, rs = pb[m]["w0_geo_p_trained"], pb[m]["r30"]
        points(axp, m, [p] * len(rs), rs, label=True)
        points(axc, m, cx[m], rs, label=False)
    cuts(axp, 1.2e-7)
    cuts(axc, -0.18)
    axc.text(
        0.35,
        twin_mean + th["r_contrast"] - 0.012,
        f"shaded: twin's mean r +/- {th['r_contrast']};\nclass map inside it = r reads the kind, not carry",
        fontsize=6.5,
        color=INK,
        va="top",
    )
    axc.annotate(
        "class map",
        (mean(cx["class_map"]), mean(pb["class_map"]["r30"])),
        xytext=(0.66, 0.205),
        fontsize=7,
        color=COLOR["class_map"],
    )
    axc.annotate(
        "twin",
        (mean(cx["class_shuffled"]), mean(pb["class_shuffled"]["r30"])),
        xytext=(0.03, 0.245),
        fontsize=7,
        color=COLOR["class_map"],
    )
    axp.text(
        1e-3,
        0.17,
        "elicit-prior cipher: skipped,\n46 keys had no admissible value; no point",
        fontsize=6.5,
        color=INK,
    )
    axp.set_xscale("log")
    axp.set_xlim(8e-8, 5e-2)
    axp.set_xlabel(
        "W0's prior of the trained value after its key (geometric-mean probability, log scale)",
        color=INK,
    )
    style(
        axp,
        "Does r follow W0's prior of the target?",
        "r at step 30: share of the update's gain\non W0's own subcomponents",
    )
    axp.legend(
        fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False
    )
    for bar, text, dx, ha in (
        (th["x_entry"], "entry bar 0.3:\nclass map carries\nif over it", 0.01, "left"),
        (th["twin_max"], "twin bar 0.15:\ntwin clean\nif under it", -0.01, "right"),
    ):
        axc.axvline(bar, color=INK, lw=0.8, ls="-.", zorder=1)
        axc.text(bar + dx, 0.17, text, fontsize=6.5, color=INK, va="bottom", ha=ha)
    axc.set_xlim(-0.2, 1.0)
    axc.set_xlabel(
        "held-out carry at step 30: x - x_null for the class map and twin,\n"
        "x for the maps with no class null",
        color=INK,
    )
    style(axc, "Does r follow how far the map carries to unseen keys?", "")
    for ax in (axp, axc):
        ax.grid(color=GRID, lw=0.6, zorder=0)
    axp.set_ylim(0.15, 1.0)
    fig.suptitle(
        "A new rule that carries, at a target W0 does not guess: does r read it as new? "
        "(96 trained keys, step 30)",
        fontsize=10.5,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "a7_r.png", dpi=150)
    plt.close(fig)
    for m in ORDER:
        rs = pb[m]["r30"]
        print(
            f"{m:15s} prior {pb[m]['w0_geo_p_trained']:.2e}  r30 {[round(r, 4) for r in rs]} mean {mean(rs):.4f}"
            f"  carry30 {[round(v, 3) for v in cx[m]]} mean {mean(cx[m]):.3f}"
        )
    print(
        "D",
        a7["score"]["D"],
        "| twin mean r",
        round(twin_mean, 4),
        "| class mean r",
        round(mean(pb["class_map"]["r30"]), 4),
    )
    lp = [math.log10(pb[m]["w0_geo_p_trained"]) for m in ORDER]
    print(
        "Spearman of seed-mean r on log prior, eight maps (reported, post hoc):",
        round(spearman(lp, [mean(pb[m]["r30"]) for m in ORDER]), 4),
    )
    print(
        "Spearman of seed-mean r on step-30 carry, eight maps (reported, post hoc):",
        round(
            spearman([mean(cx[m]) for m in ORDER], [mean(pb[m]["r30"]) for m in ORDER]),
            4,
        ),
    )
    print("score", json.dumps(a7["score"]))


if __name__ == "__main__":
    main()
