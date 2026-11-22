"""P2.54's figure (no GPU): the prior-by-mechanism 2x2, one panel per reading, every seed a dot.

Columns: the two reused corners (copy, low-prior copy) left of the separator, the two new corners (prior-matched cipher,
cipher) right of it; each labelled by its prior. Reads E250's table (results/stage2/table.json), the head switch-off
files (stage1/ for the saved states, stage2/head/ for the low-prior copy) and, for the random-update band on effective
rank, E237's summary (experiments/key_grid/results/summary.json) and E250's random updates.

`cued_main` draws P2.55's version (experiments/cued_copy/figures/prior_mech_2x2_cued.png): prior on the x axis, mechanism
as the two lines, the cued copy (E253, experiments/cued_copy/results/table.json) in the low-prior copy-head corner and
the row-selected low copy faint beside it, the top-3-heads share from E252/E253's k-head files.

    ~/commute-agents/fleet/cpu-run python3 experiments/prior_mech_2x2/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from commute.figures import METHODS, column_dots, figure, save

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TABLE = json.loads((RES / "stage2" / "table.json").read_text())
E237 = json.loads((HERE.parent / "key_grid" / "results" / "summary.json").read_text())
CORNERS = {  # corner: (tick label, head switch-off files)
    "copy": ("high\ncopy", "stage1/e58-copy_s{}.json"),
    "lowcopy": ("low\nlow copy", "stage2/head/e58-lowcopy_s{}.json"),
    "prior_cipher": ("high\nmatched\ncipher", "stage1/e58-prior_cipher_s{}.json"),
    "cipher": ("low\ncipher", "stage1/e58-cipher_s{}.json"),
}
READINGS = {  # proxy key: (panel title, y label, figures.METHODS style)
    "head_share": ("Copy head (2,4) off", "share of gain lost", "head"),
    "r": ("Re-gating share r", "r", "regating"),
    "log10_prior": ("Base model's prior", "log10 P(answer)", "prior"),
    "steering_pooled": ("Pooled steering share", "steering share", "steering"),
    "erank": ("Update's effective rank", "effective rank", "spectrum"),
    "neuron_k64": ("Neuron share, top 64", "neuron share", "neuron"),
}
SEEDS = (0, 1, 2)


def head_files(corner: str) -> list[dict[str, Any]]:
    return [json.loads((RES / CORNERS[corner][1].format(s)).read_text()) for s in SEEDS]


def erank_band(corner: str) -> tuple[float, float]:
    if corner == "lowcopy":
        vals = [TABLE["random"][f"e58/lowcopy_s{s}"]["erank"] for s in SEEDS]
        return min(vals), max(vals)
    lo, hi = E237["table"]["erank"][corner]["band"]
    return lo, hi


def head_panel(ax: Any) -> None:
    """(2,4)'s share, with single random heads beside it: (1,1), the first draw, as crosses; the other four as a band."""
    ax.scatter([], [], color=METHODS["head"].color, marker=METHODS["head"].marker, s=22, label="copy head (2,4) off")
    for i, corner in enumerate(CORNERS):
        recs = head_files(corner)
        others = [x for r in recs for x in r["random_shares"][1:]]
        ax.fill_between([i - 0.3, i + 0.3], min(others), max(others), color="#555555", alpha=0.25, linewidth=0,
                        label="four other single heads off" if i == 0 else None)  # fmt: skip
        ax.scatter([i + 0.18] * 3, [r["random_shares"][0] for r in recs], color="#555555", marker="x", s=20,
                   label="head (1,1) off" if i == 0 else None)  # fmt: skip
    ax.axhline(0.2, color="#777777", linewidth=0.8, linestyle="--", label="reused bar")


def rank_panel(ax: Any) -> None:
    """The random update of the same size: its lowest and highest effective rank per corner, as grey crosses."""
    for i, corner in enumerate(CORNERS):
        ax.scatter([i, i], erank_band(corner), color="#555555", marker="X", s=20, label="random update, same size" if i == 0 else None)


def panel(ax: Any, key: str) -> list[dict[str, Any]]:
    title, ylabel, method = READINGS[key]
    seeds = {CORNERS[c][0]: [row[key] for row in TABLE["rows"][c]] for c in CORNERS}
    column_dots(ax, seeds, method=method, ylabel=ylabel)
    if key == "head_share":
        head_panel(ax)
    if key == "erank":
        rank_panel(ax)
    if key == "head_share":
        ax.legend(fontsize=5, frameon=False, loc="upper right", bbox_to_anchor=(1.12, 1.0))
    if key == "erank":
        ax.legend(fontsize=5.5, frameon=False, loc="center right")
    if key in ("r", "steering_pooled", "neuron_k64"):
        ax.set_ylim(bottom=0)  # a share of 0 is "nothing": show how far above it each corner sits
    ax.axvline(1.5, color="#333333", linewidth=0.8)
    ax.text(0.5, 1.0, "reused", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, style="italic")
    ax.text(2.5, 1.0, "new", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, style="italic")
    p = TABLE["proxies"][key]
    ax.set_title(f"{title}: follows {p['follows'].replace('_', ' ')}\nmechanism {p['mechanism_effect']:.2g}, "
                 f"prior {p['prior_effect']:.2g}, band {p['band']:.2g}", fontsize=6.5, pad=11)  # fmt: skip
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    return [{"reading": key, "corner": c, "seed": s, "value": row[key]} for c in CORNERS for s, row in zip(SEEDS, TABLE["rows"][c])]


def main() -> None:
    fig, axes = figure((7.5, 7.0), ncols=3, nrows=2)
    data = [rec for ax, key in zip(axes.flat, READINGS) for rec in panel(ax, key)]
    for ax in axes[1]:
        ax.set_xlabel("base model's prior, and corner", fontsize=7)
    for c in CORNERS:
        for s, r in zip(SEEDS, head_files(c)):
            data.append({"reading": "head_share_(1,1)", "corner": c, "seed": s, "value": r["random_shares"][0]})
        data.append({"reading": "erank_random_band", "corner": c, "seed": "min-max", "value": "/".join(f"{v:.1f}" for v in erank_band(c))})
    fig.suptitle("Every reading of the update splits the 2x2 by mechanism; only the prior splits it by prior", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, HERE / "figures" / "prior_mech_2x2.png", data=data,
         n="3 seeds per corner (dots), every update at one common size. Low copy = the copy task's lowest-prior rows. "
           "Effects: mechanism = mean(reused) - mean(new), prior = mean(high) - mean(low), band = largest seed range. "
           "Random updates have no gain, so r, steering and neuron shares have no random reference; "
           "rank band: E237's random updates (copy, ciphers) and E250's (low copy).")  # fmt: skip


# P2.55's redraw: the cued copy (E253) in the low-prior copy-head corner, the row-selected low copy faint beside it.
CUED_DIR = HERE.parent / "cued_copy"
CUED = json.loads((CUED_DIR / "results" / "table.json").read_text())
SERIES = {  # mechanism: (legend label, colour, (high-prior corner, low-prior corner))
    "copy_heads": ("runs through the copy heads", "#0072B2", ("copy", "cuedcopy")),
    "new_lookup": ("needs a new lookup", "#D55E00", ("prior_cipher", "cipher")),
}
CUED_READINGS = {  # table key: (panel title, y label)
    "r": ("Re-gating share r", "r"),
    "k3_share": ("Top 3 copy heads off", "share of gain lost"),
    "steering_pooled": ("Pooled steering share", "steering share"),
    "erank": ("Update's effective rank", "effective rank"),
    "neuron_k64": ("Neuron share, top 64", "neuron share"),
    "log10_prior": ("Base model's prior", "log10 P(answer)"),
}
KHEAD_FILES = {  # corner: the k-head file pattern of its states (E252 for the key corners, E253 for the cued copy)
    **{c: RES / "khead" / f"e58-{c}_s{{}}.json" for c in ("copy", "lowcopy", "prior_cipher", "cipher")},
    "cuedcopy": CUED_DIR / "results" / "items" / "khead" / "e58-cuedcopy_s{}.json",
}
X = {"copy": -0.06, "prior_cipher": 0.06, "cuedcopy": 0.94, "cipher": 1.06, "lowcopy": 1.6}  # offset so bars don't overlap


def k3_null_band(corner: str) -> tuple[float, float]:
    """Lowest and highest, over seeds, of the median share lost with 3 random heads off (clean draws, as E252 reads)."""
    meds = []
    for s in SEEDS:
        draws = json.loads(Path(str(KHEAD_FILES[corner]).format(s)).read_text())["by_k"]["3"]["null"]
        meds.append(float(np.median([d["share"] for d in draws if not d["flags"]])))
    return min(meds), max(meds)


def random_band(corner: str, key: str) -> tuple[float, float] | None:
    if key == "k3_share":
        return k3_null_band(corner)
    if key != "erank":
        return None  # a random update has no gain, so r, steering and neuron shares cannot be read on it
    if corner == "cuedcopy":
        return min(CUED["random"]["erank"]), max(CUED["random"]["erank"])
    return erank_band(corner)


def seed_bar(ax: Any, corner: str, key: str, color: str, alpha: float = 1.0) -> dict[str, Any]:
    vals = [row[key] for row in CUED["rows"][corner]]
    mean, x = float(np.mean(vals)), X[corner]
    ax.errorbar([x], [mean], yerr=[[max(0.0, mean - min(vals))], [max(0.0, max(vals) - mean)]], fmt="o", color=color, alpha=alpha,
                markersize=4, capsize=2, linewidth=1)  # fmt: skip
    band = random_band(corner, key)
    if band is not None:
        ax.fill_between([x - 0.05, x + 0.05], band[0], max(band[1], band[0] + 0.004 * abs(band[0]) + 0.002),
                        color="#555555", alpha=0.4 * alpha, linewidth=0)  # fmt: skip
    return {"reading": key, "corner": corner, "mean": mean, "seed_min": min(vals), "seed_max": max(vals),
            "random_min": band[0] if band else "", "random_max": band[1] if band else ""}  # fmt: skip


def cued_panel(ax: Any, key: str) -> list[dict[str, Any]]:
    title, ylabel = CUED_READINGS[key]
    data = []
    for label, color, (hi, lo) in SERIES.values():
        ax.plot([X[hi], X[lo]], [np.mean([r[key] for r in CUED["rows"][c]]) for c in (hi, lo)], color=color, linewidth=1, label=label)
        data += [seed_bar(ax, hi, key, color), seed_bar(ax, lo, key, color)]
    data.append(seed_bar(ax, "lowcopy", key, SERIES["copy_heads"][1], alpha=0.3))
    if key == "k3_share":
        ax.axhline(0.2, color="#777777", linewidth=0.8, linestyle="--")
        ax.text(0.5, 0.2, "copy label bar 0.2", fontsize=5.5, color="#555555", ha="center", va="bottom")
    p = CUED["proxies"][key]
    ax.set_title(f"{title}\nmechanism {p['mechanism_effect']:.2g}, prior {p['prior_effect']:.2g}, band {p['band']:.2g}", fontsize=6.5)
    if key == "erank":
        ax.text(0.5, 690, "random updates, about 700", fontsize=5.5, color="#555555", ha="center", va="top")
    ax.set_xticks([0, 1, 1.6], ["high prior", "low prior", "old low\ncopy"], fontsize=6)
    ax.set_xlim(-0.3, 1.85)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    return data


def cued_main() -> None:
    fig, axes = figure((7.5, 6.2), ncols=3, nrows=2)
    data = [rec for ax, key in zip(axes.flat, CUED_READINGS) for rec in cued_panel(ax, key)]
    axes.flat[0].plot([], [], color="#555555", alpha=0.4, linewidth=6, label="random reference, same size or same k")
    axes.flat[0].errorbar([], [], fmt="o", color=SERIES["copy_heads"][1], alpha=0.3, label="row-selected low copy (old corner)")
    axes.flat[0].legend(fontsize=5, frameon=False, loc="center left")
    fig.suptitle("With the cued copy in the low-prior corner, every reading but the prior splits by mechanism", fontsize=9)
    save(fig, CUED_DIR / "figures" / "prior_mech_2x2_cued.png", data=data,
         n="Points: mean of 3 seeds per corner; bars: lowest to highest seed. Every update at one common size. Lines join "
           "the two corners of one mechanism. Grey: on effective rank, a random update of the same size; on the top 3 "
           "heads, the median share lost with 3 random heads off, lowest to highest seed. Random updates have no gain, "
           "so r, steering and neuron shares have no random reference. Effects as in P2.54: mechanism = mean of the "
           "copy-head corners minus mean of the new-lookup corners, prior = high minus low, band = largest seed range.")  # fmt: skip


if __name__ == "__main__":
    main()
    cued_main()
