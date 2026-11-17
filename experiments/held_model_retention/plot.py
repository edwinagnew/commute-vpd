"""E197 figures, from results/items/*.json and results/stage0.json (no GPU, no model):
    figures/gaps.png      reading 3, the reuse member's survival minus its twin's, one point per seed (0 to 3), one row
                          per pair; for the parity twin / fact pair also the side number's gap (F's embedding,
                          unembedding and final norm in H). The four seeds are the spread; 0 is "no difference".
    figures/survival.png  three panels, one per reading, never pooled: S per item, one point per seed, with the code
                          checks and E195's rebuilt item on top. 0 = H no closer to F than M is; 1 = H reproduces F.
                          A value left of a panel's range is written as text at its edge.

    ~/commute-agents/fleet/cpu-run python experiments/held_model_retention/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import dots, figure, reference, save

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
PAIRS: list[list[str]] = json.loads((RESULTS / "predictions.json").read_text())["pairs"]["order"]
SEEDS = (0, 1, 2, 3)
NAMES = {
    "copy24": "copy", "cipher24": "cipher", "ident32": "identity map", "rand32": "random map", "case_real": "case, real",
    "case_twin": "case, twin", "repeat_real": "repeat, real", "repeat_twin": "repeat, twin", "twin": "parity twin",
    "fact": "memorised fact",
}
CHECKS = {"gains_unit": "check: gains per unit", "gains_block": "check: gains per block", "circuit_off": "check: circuit off"}
N_TEXT = "seeds 0 to 3 per fine-tune; 512 read windows per item, last position; fits on 16,384 fit windows"


def load() -> tuple[dict[str, Any], dict[str, Any]]:
    items = {p.stem: json.loads(p.read_text()) for p in (RESULTS / "items").glob("*.json") if not p.stem.endswith(".stopped")}
    return items, json.loads((RESULTS / "stage0.json").read_text())["items"]


def s(item: dict[str, Any], r: str, side: bool = False) -> float:
    rd = item["readings"][r]
    return rd["side_F_ends"]["S"] if side else rd["S"]


def gaps(items: dict[str, Any]) -> None:
    groups: dict[str, list[float]] = {}
    data: list[dict[str, Any]] = []
    for a, b in PAIRS:
        for side in (False, True) if a == "twin" else (False,):
            label = f"{NAMES[a]} − {NAMES[b]}" + ("\nwith F's ends in H" if side else "")
            groups[label] = [s(items[f"{a}_s{k}"], "3", side) - s(items[f"{b}_s{k}"], "3", side) for k in SEEDS]
            data += [{"pair": label.replace("\n", " "), "seed": k, "gap": v} for k, v in zip(SEEDS, groups[label], strict=True)]
    fig, ax = figure((7.5, 3.4))
    dots(ax, groups, method="regating", xlabel="reading 3 survival, reuse member minus its twin\n(symmetric log scale, linear within ±0.001)")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(-0.01, 0.6)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.annotate("no difference", (0, 1.0), xycoords=("data", "axes fraction"), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
    ax.axhspan(len(PAIRS) - 1.5, len(groups) - 0.5, color="#eeeeee", zorder=0)
    ax.set_title("Reading 3: the reuse member keeps a little more of its change on three of four counted pairs.\n"
                 "The parity/fact gap shrinks about sevenfold once F's embeddings are in H.", fontsize=8, pad=12)
    save(fig, FIGURES / "gaps.png", n=N_TEXT + ". Grey rows: the fact pair, reported, not counted. Each dot is one seed.", data=data)


def rows(items: dict[str, Any], stage0: dict[str, Any], r: str) -> dict[str, list[float]]:
    out = {label: [stage0[k]["readings"][r]["S"]] for k, label in CHECKS.items()}
    out["rebuilt item, E195"] = [items["rebuilt"]["readings"][r]["S"]]
    for a, b in PAIRS:
        for name in (a, b):
            out[NAMES[name]] = [s(items[f"{name}_s{k}"], r) for k in SEEDS]
    for name in ("twin", "fact"):
        out[NAMES[name] + ", F's ends in H"] = [s(items[f"{name}_s{k}"], r, side=True) for k in SEEDS]
    return out


def survival(items: dict[str, Any], stage0: dict[str, Any]) -> None:
    fig, axes = figure((7.5, 4.6), ncols=3)
    data: list[dict[str, Any]] = []
    titles = {"1": "Reading 1: neurons and heads,\none gain each", "2": "Reading 2: heads and MLP blocks,\none gain each",
              "3": "Reading 3: per-site least-squares\nmap of M's unit activations"}
    for ax, r in zip(axes, ("1", "2", "3"), strict=True):
        groups = rows(items, stage0, r)
        lo = -0.2
        shown = {k: [max(v, lo) for v in vs] for k, vs in groups.items()}
        dots(ax, shown, method="regating", xlabel="survival S")
        for i, vs in enumerate(groups.values()):
            for v in vs:
                if v < lo:
                    ax.annotate(f"{v:.1f}", (lo, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6, color="#D55E00")
        ax.set_xlim(lo - 0.05, 1.08)
        ax.axvline(0, color="#222222", linewidth=0.8)
        reference(ax, 1.0, "H = F")
        ax.axhspan(-0.5, len(CHECKS) - 0.5, color="#eeeeee", zorder=0)
        ax.set_title(titles[r], fontsize=7, pad=12)
        if r != "1":
            ax.set_yticklabels([])
        data += [{"reading": r, "row": k, "point": j, "S": v} for k, vs in groups.items() for j, v in enumerate(vs)]
    fig.suptitle("Survival of each fine-tune's change in the held model, by reading (three targets, never pooled)", fontsize=9)
    save(fig, FIGURES / "survival.png", n=N_TEXT + ". Grey band: code checks on copy's windows. 0 = H no closer to F than M; "
         "1 = H reproduces F. Values below -0.2 are drawn at the edge with their value as text.", data=data)


def main() -> None:
    items, stage0 = load()
    gaps(items)
    survival(items, stage0)


if __name__ == "__main__":
    main()
