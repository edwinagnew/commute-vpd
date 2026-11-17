"""E192 (cell P1.23) figures: the filed gap (reuse side's closure at 16 names minus the new side's) per pair and seed against
the bar, and the closure curves of seeds 2 and 3 beside E184's seeds 0 and 1 (E191's curves.json, list all).

    ~/commute-agents/fleet/cpu-run python experiments/curve_test/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import figure, reference, save
from commute.jobs import load_experiment

RU: Any = load_experiment("curve_test", "rule")
HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
E191 = HERE.parent / "e184_curves" / "results" / "curves.json"
LABELS = ("copy (reuse) / cipher (new)", "identity (reuse) / random map (new)", "case (reuse) / its twin (new)",
          "repeat (reuse) / its twin (new)", "parity twin (reuse) / memorised fact (new)")  # fmt: skip
REUSE, NEW, OLD = "#0072B2", "#D55E00", "#777777"
TITLES = {"full": "full next-token distribution", "answer": "answer positions only (chosen after E184)"}


def e184_gaps() -> dict[str, list[float]]:
    """E184 seeds 0 and 1, full reading, at the last name each search reached (case and repeat stopped near 0.95)."""
    items = json.loads(E191.read_text())["items"]
    return {lab: [items[f"{a}_s{s}"]["lists"]["all"]["best"][RU.CAP] - items[f"{b}_s{s}"]["lists"]["all"]["best"][RU.CAP] for s in (0, 1)]
            for (a, b), lab in zip(RU.PAIRS, LABELS, strict=True)}  # fmt: skip


def draw_gaps(panels: list[tuple[str, dict[str, Any], bool]], bar: float, old: dict[str, list[float]], path: Path, n_text: str) -> None:
    """One panel per (title, pair-seed rows of rule.score, whether to draw E184's seeds 0, 1 beside)."""
    fig, axes = figure((7.5, 3.6), ncols=len(panels))
    data = []
    for col, (ax, (title, rows, show_old)) in enumerate(zip(axes, panels, strict=True)):
        for i, ((a, b), lab) in enumerate(zip(RU.PAIRS, LABELS, strict=True)):
            new = [v["gap"] for k, v in rows.items() if k.startswith(f"{a}/{b} ") and v["read"]]
            ax.scatter(new, [i] * len(new), color=REUSE, s=24, label="seeds 2, 3" if i == 0 else None)
            data += [{"panel": title, "pair": lab, "seeds": "2-3", "gap": g} for g in new]
            if show_old:
                ax.scatter(old[lab], [i + 0.2] * 2, facecolors="none", edgecolors=OLD, s=24, label="E184 seeds 0, 1" if i == 0 else None)
                data += [{"panel": title, "pair": lab, "seeds": "0-1", "gap": g} for g in old[lab]]
        reference(ax, bar, f"bar {bar:.3f}")
        ax.axvline(0, color="#bbbbbb", linewidth=0.6)  # no difference between the sides
        ax.set_yticks(range(len(LABELS)), LABELS if col == 0 else [""] * len(LABELS), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("closure at 16 names, reuse minus new")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Does the reuse side close more than the new side by more than the bar, on every pair and seed?")
    save(fig, path, n=n_text, data=data)


SHORT = ("copy / cipher", "identity / random map", "case / its twin", "repeat / its twin", "parity twin / fact")
CONTROLS = (("l1_neuron", "one shifted neuron", "#009E73"), ("l2_triggered", "one triggered neuron", "#CC79A7"), ("l4_bit", "random bit per window", "#555555"))


def draw_curves(new: dict[str, Any], path: Path) -> None:
    old = json.loads(E191.read_text())["items"]
    fig, axes = figure((7.5, 10.0), ncols=4, nrows=len(RU.PAIRS))
    ns, data = list(range(RU.CAP + 1)), []
    for r, (a, b) in enumerate(RU.PAIRS):
        for col, s in enumerate((0, 1, 2, 3)):
            ax = axes[r][col]
            for side, color in ((a, REUSE), (b, NEW)):
                name = f"{side}_s{s}"
                curves = {"full": old[name]["lists"]["all"]["best"]} if s < 2 else {k: RU.best_by_n(new[name], key) for k, (key, _) in RU.READINGS.items()}
                for reading, cur in curves.items():
                    ax.plot(ns, cur, color=color, linewidth=1.6 if reading == "full" else 1.0, linestyle="-" if reading == "full" else "--")
                    data += [{"item": name, "reading": reading, "n": n, "closure": v} for n, v in zip(ns, cur, strict=True)]
            ax.set_ylim(0, 1.02)
            if r == 0:
                ax.set_title(f"seed {s}" + (", E184" if s < 2 else ", fresh"), fontsize=8)
            ax.set_xlabel("number of names", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{SHORT[r]}\nclosure", fontsize=8)
    h = [axes[0][0].plot([], [], color=c, linestyle=ls)[0] for c, ls in ((REUSE, "-"), (NEW, "-"), ("#555555", "--"))]
    axes[2][3].legend(h, ["built to reuse", "built to be new", "same names,\nanswer positions only"], fontsize=6, frameon=False, loc="lower right")
    fig.suptitle("Closure against names, list all: E184's seeds 0, 1 beside the fresh seeds 2, 3")
    save(fig, path, n="one fine-tune per side and seed; closure on 16,384 held-out read windows. E184's case and repeat searches stopped near 0.95 before 16 names, so their curves are flat after the stop", data=data)


def draw_controls(path: Path) -> None:
    fig, ax = figure()
    ns, data = list(range(RU.CAP + 1)), []
    for item, label, color in CONTROLS:
        cur = RU.best_by_n(json.loads((RESULTS / f"{item}.json").read_text()), "closure")
        ax.plot(ns, cur, color=color, linewidth=1.6, label=label, marker="o", markersize=3)
        data += [{"item": item, "n": n, "closure": v} for n, v in zip(ns, cur, strict=True)]
    ax.axhline(0.95, color="#777777", linewidth=0.8, linestyle="--", label="planted neurons must pass 0.95")
    ax.axhline(0.8, color="#777777", linewidth=0.8, linestyle=":", label="random bit must stay under 0.8")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("number of names")
    ax.set_ylabel("closure, full distribution")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.3, 0.74))
    ax.set_title("Does the search find what was planted, and nothing in noise?", fontsize=9)
    save(fig, path, n="three built changes to the original model, searched as every fine-tune; 16,384 read windows each", data=data)


def main() -> None:
    names = [f"{side}_s{s}" for pr in RU.PAIRS for side in pr for s in (2, 3)]
    new = {n: json.loads((RESULTS / f"{n}.json").read_text()) for n in names}
    score = json.loads((RESULTS / "score.json").read_text())
    panels = [(TITLES[r], score[r]["pairs"], r == "full") for r in RU.READINGS]
    draw_gaps(panels, score["bar"], e184_gaps(), FIGURES / "gaps.png", "2 fresh seeds per pair (filled); E184's 2 seeds (open), full reading only")
    draw_curves(new, FIGURES / "curves.png")
    draw_controls(FIGURES / "controls.png")
    print(f"wrote {FIGURES}/gaps.png, curves.png and controls.png")


if __name__ == "__main__":
    main()
