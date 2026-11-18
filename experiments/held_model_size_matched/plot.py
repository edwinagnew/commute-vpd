"""P1.27 figures, from results/summary.json (no GPU, no model):
    figures/gaps.png    two panels, reading 3 (the per-site map) and reading 1 (one gain per unit), never pooled: per pair,
                        the reuse side's mean S minus the new side's at the saved sizes (open circle, bar = its seed spread
                        se_original either side) and with the larger side matched in size (filled). 0 = no difference.
    figures/shifts.png  the same two readings: per pair, each larger-side state's S at the matched size minus its S at
                        its own size, one point per state; the grey bar is the saved gap's seed spread either side of 0,
                        the band rule R3 reads; the vertical line marks 0, "size made no difference".

    ~/commute-agents/fleet/cpu-run python experiments/held_model_size_matched/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.ticker import MaxNLocator

from commute.figures import FULL_WIDTH, METHODS, dots, figure, save

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
READINGS = {"3": "reading 3, per-site map", "1": "reading 1, one gain per unit"}
N_TEXT = "2 to 8 seeds per side (E187's saved states); 512 read windows per item, last position; fits on 16,384 fit windows"
NAMES = {  # experiments/method_table/build.py's names for E187's built pairs
    "e62_copy_lookup": "copy at a new marker vs lookup",
    "e46_copy_random": "copy maps vs random maps",
    "e53_lowercase_cipher": "lowercase vs ciphers",
    "e54_matched": "half-known vs cipher, equal size",
    "e54_own": "half-known vs cipher, own steps",
    "open_pair_c3": "rank-1 vs cipher, step 10",
    "open_pair_c3_copy": "copy vs cipher, step 10",
    "e58_rank1_96": "rank-1 vs cipher, 96 keys",
    "e58_copy_prior_cipher": "copy vs prior-matched cipher, 96 keys",
}
MATCHED = METHODS["size"].color


def label(pair: str, p: dict[str, Any]) -> str:
    return f"{NAMES[pair]} (c {p['c']:.2f}, {p['larger']} side shrunk)"


def gaps(summary: dict[str, Any]) -> None:
    fig, axs = figure(FULL_WIDTH, ncols=2)
    rows = []
    for k, (ax, (r, title)) in enumerate(zip(axs, READINGS.items(), strict=True)):
        for i, (pair, p) in enumerate(summary["pairs"].items()):
            g = p[f"reading {r}"]
            ax.errorbar(g["gap_original"], i - 0.12, xerr=g["se_original"], fmt="o", mfc="white", color="#333333", ms=4,
                        capsize=2, label="saved sizes, bar = seed spread" if i == 0 else None)  # fmt: skip
            ax.plot(g["gap_matched"], i + 0.12, "P", color=MATCHED, ms=5, label="larger side shrunk to match" if i == 0 else None)
            rows.append({"reading": r, "pair": pair, "gap_saved": g["gap_original"], "se_saved": g["se_original"],
                         "gap_matched": g["gap_matched"]})  # fmt: skip
        labels = [label(pair, p) for pair, p in summary["pairs"].items()]
        ax.set_yticks(range(len(labels)), labels if k == 0 else [""] * len(labels), fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_xlabel("gap in S, reuse side minus new side", fontsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylim(len(labels) + 0.6, -0.5)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("Does matching update size shrink the reuse-minus-new gap in survival?", fontsize=9)
    save(fig, FIGURES / "gaps.png", n=N_TEXT, data=rows)


def shifts(summary: dict[str, Any]) -> None:
    fig, axs = figure(FULL_WIDTH, ncols=2)
    rows = []
    for k, (ax, (r, title)) in enumerate(zip(axs, READINGS.items(), strict=True)):
        groups = {label(pair, p): p[f"reading {r}"]["shift_by_state"] for pair, p in summary["pairs"].items()}
        for i, p in enumerate(summary["pairs"].values()):
            se = p[f"reading {r}"]["se_original"]
            ax.fill_betweenx([i - 0.3, i + 0.3], -se, se, color="#cccccc", linewidth=0,
                             label="seed spread of the saved gap" if i == 0 else None)  # fmt: skip
        dots(ax, groups, method="size", xlabel="shift in S, shrunk minus saved, same state")
        ax.xaxis.label.set_fontsize(7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylim(len(groups) + 0.6, -0.5)
        ax.set_yticks(range(len(groups)), list(groups) if k == 0 else [""] * len(groups), fontsize=6)
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
        rows += [{"reading": r, "pair": key, "shift": v} for key, vs in groups.items() for v in vs]
    fig.suptitle("Does shrinking a state's update move its own survival by more than the seed spread?", fontsize=9)
    save(fig, FIGURES / "shifts.png", n=N_TEXT, data=rows)


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    gaps(summary)
    shifts(summary)


if __name__ == "__main__":
    main()
