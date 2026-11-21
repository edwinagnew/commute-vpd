"""P1.38's figures. No GPU.

r_gap_matched.png: per E187 pair, the R gap (reuse minus new, read grade) at saved size (E221, black bar, seeds' se as
a line), at matched size (this run, blue dot with its se) and as the c-squared law predicts it (open diamond), one
panel per level; 0 is no gap.
r_gap_shift.png: per pair, the change in R gap from saved to matched size against the update-size gap that matching
removed, one panel per level.

    python experiments/kstar_size_matched/plot.py      (results/summary.json -> both figures)
    python experiments/kstar_size_matched/plot.py --input figures/sketch_input.json --out figures/sketch.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from commute import figures as fg

HERE = Path(__file__).resolve().parent
E205_SUMMARY = HERE.parent / "held_model_size_matched" / "results" / "summary.json"
LEVELS = {"head_mlp": "heads and MLP layers", "mlp_attn": "MLP and attention layers", "block": "blocks"}
BLACK, BLUE, ORANGE = "#000000", "#0072B2", "#E69F00"
PAIR_NAMES = {  # pair key -> row name of the paper's Table tab:pairs
    "e62_copy_lookup": "Copy at a new marker vs lookup",
    "e46_copy_random": "Copy maps vs random maps",
    "e53_lowercase_cipher": "Lowercase vs ciphers",
    "e54_matched": "Mid vs cipher, equal size",
    "e54_own": "Mid vs cipher, own steps",
    "open_pair_c3": "Top vs cipher, matched step",
    "open_pair_c3_copy": "Copy vs cipher, matched step",
    "e58_rank1_96": "Top vs cipher, 96 keys",
    "e58_copy_prior_cipher": "Copy vs prior-matched cipher, 96 keys",
}


def panels_from_summary(d: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The saved "plot" panels, each point given its matched se and c-squared-law gap from the level's read grade."""
    out = {}
    for level, pts in d["plot"]["panels"].items():
        read = d["levels"][level]["read"]
        out[level] = [p | {"se_matched": read["se_matched"]["e187_" + p["pair"]], "c2_law": read["gap_c2_law"]["e187_" + p["pair"]]} for p in pts]
    return out


def pair_labels(pairs: list[str]) -> list[str]:
    """'Mid vs cipher, own steps, new x0.52': which side was shrunk and by what factor c (E205's summary, E187's c)."""
    if not E205_SUMMARY.exists():
        return [PAIR_NAMES[p] for p in pairs]
    e205 = json.loads(E205_SUMMARY.read_text())["pairs"]
    return [f"{PAIR_NAMES[p]}, {e205[p]['larger']} x{e205[p]['c']:.2f}" if p in e205 else PAIR_NAMES[p] for p in pairs]


def gaps_figure(panels: dict[str, list[dict[str, Any]]], n: str, out: Path) -> None:
    fig, axs = fg.figure(fg.FULL_WIDTH, ncols=len(panels))
    rows = []
    for k, (ax, (title, pts)) in enumerate(zip(axs, panels.items(), strict=True)):
        for i, p in enumerate(pts):
            first = i == 0
            ax.plot([p["unscaled"] - p["se"], p["unscaled"] + p["se"]], [i, i], color=BLACK, linewidth=0.6)
            ax.scatter([p["unscaled"]], [i], color=BLACK, marker="|", s=50, label="saved size (E221)" if first else None)
            if p.get("matched") is not None:
                if "se_matched" in p:
                    ax.plot([p["matched"] - p["se_matched"], p["matched"] + p["se_matched"]], [i + 0.18] * 2, color=BLUE, linewidth=0.6)
                ax.scatter([p["matched"]], [i + 0.18], color=BLUE, marker="o", s=14, label="matched size (this run)" if first else None)
            if p.get("c2_law") is not None:
                ax.scatter([p["c2_law"]], [i - 0.18], facecolors="none", edgecolors=ORANGE, marker="D", s=14,
                           label="matched size if 1 - R scaled as c squared" if first else None)
            rows.append({"panel": title} | p)
        fg.reference(ax, 0, "no gap")
        labels = pair_labels([p["pair"] for p in pts])
        ax.set_yticks(range(len(pts)), labels if k == 0 else [""] * len(pts), fontsize=6)
        ax.set_ylim(len(pts) + 1.2, -0.6)  # inverted, with an empty band at the bottom for the legend
        ax.set_title(LEVELS.get(title, title), fontsize=8)
        ax.set_xlabel("R gap, reuse minus new", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
    axs[0].legend(fontsize=5.5, loc="lower left")
    fig.suptitle("Does the reuse side keep its higher R once both sides moved the weights equally far?", fontsize=9)
    fg.save(fig, out, n=n, data=rows)


def shift_figure(d: dict[str, Any], out: Path) -> None:
    """y: matched minus saved R gap; x: the size gap matching removed, saved minus matched (reuse minus new)."""
    size = d["size_gap"]
    fig, axs = fg.figure(fg.FULL_WIDTH, ncols=len(LEVELS))
    rows = []
    for ax, (level, name) in zip(axs, LEVELS.items(), strict=True):
        read = d["levels"][level]["read"]
        for p in read["gap_matched"]:
            x = size["unscaled"][p] - size["matched"][p]
            y = read["gap_matched"][p] - read["gap_unscaled"][p]
            ax.scatter([x * 1e3], [y], color=BLUE, s=14)
            ax.annotate(PAIR_NAMES[p.removeprefix("e187_")], (x * 1e3, y), fontsize=4.5, xytext=(2, 2), textcoords="offset points")
            rows.append({"level": level, "pair": p, "size_gap_removed": x, "r_gap_change": y})
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.axvline(0, color="#777777", linewidth=0.8)
        rho = d["levels"][level]["within_pair_rho"]["rho"]
        ax.set_title(f"{name}\nrank correlation {rho:.2f}", fontsize=8)
        ax.set_xlabel("update-size gap removed, reuse minus new (x1e-3)", fontsize=6)
        ax.tick_params(labelsize=6)
    axs[0].set_ylabel("change in R gap, matched minus saved", fontsize=7)
    fig.suptitle("Does a pair's R gap fall by more when matching shrank the new side by more?", fontsize=9)
    fg.save(fig, out, n="9 pairs, 2 to 8 seeds per side; read grade; 0 on both axes is no change", data=rows)


def main(src: Path, out: Path) -> None:
    d = json.loads(src.read_text())
    if "levels" in d:
        gaps_figure(panels_from_summary(d), d["plot"]["n"], out)
        shift_figure(d, out.with_name("r_gap_shift.png"))
    else:
        gaps_figure(d["panels"], d["n"], out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=HERE / "results" / "summary.json")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "r_gap_matched.png")
    a = ap.parse_args()
    main(a.input if a.input.is_absolute() else Path.cwd() / a.input, a.out if a.out.is_absolute() else Path.cwd() / a.out)
