"""E193 (cell P1.24) and amendment figures, from <results>/stage0.json, stage1.json and stage2.json (results/ for E193,
results/<amend>/ with `--amend <amend>`, figures then in figures/<amend>/):
    figures/profile.png   the retention profile: r of every unit of F not at r = 1, per item and level, with the count at 1
    figures/contrast.png  the retention for the contrast per item and level against the 0.3 and 0.7 bars; do-nothing, which
                          has no contrast, is drawn as the reference at r = 1
With `--report`, from both runs (stage 2 stopped in each, so no replaced item has readings):
    figures/calibration.png    the retention for the changed behaviour of do-nothing and rescaled, per level and run
    figures/replaced_fits.png  every try at the replaced item: fit ratio against max |corr|, with both bars

    ~/commute-agents/fleet/cpu-run python experiments/rule_retention/plot.py [--amend amend1 | --report]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute.figures import figure, reference, save

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
LEVELS = {"A": "level A: MLP neurons", "B": "level B: heads and MLP blocks"}
COLOR = {"do-nothing": "#777777", "rescaled": "#0072B2", "replaced": "#D55E00"}


def items(s0: dict[str, Any], s1: dict[str, Any], s2: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Item label -> readings by level, in the order drawn."""
    out = {"do-nothing": s0["readings"], "rescaled": s1["readings"]}
    return out | {f"replaced s{s}": v["readings"] for s, v in sorted(s2["seeds"].items(), key=lambda kv: int(kv[0]))}


def kind(label: str) -> str:
    return label.split(" ")[0]


def draw_profile(its: dict[str, dict[str, Any]], path: Path, note: str) -> None:
    fig, axes = figure((7.5, 2.0 + 1.1 * len(its)), ncols=2, nrows=len(its))
    data = []
    for i, (lab, rd) in enumerate(its.items()):
        for j, lv in enumerate("AB"):
            ax, v = axes[i][j], rd[lv]
            names, r = list(v["changed"]), list(v["changed"].values())
            ax.vlines(range(len(r)), 0, r, color=COLOR[kind(lab)], linewidth=1)
            ax.scatter(range(len(r)), r, color=COLOR[kind(lab)], s=12)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-1, max(len(r), 8))
            ax.set_xticks(range(len(r)), names, rotation=90, fontsize=5)
            ax.set_title(f"{lab}, {LEVELS[lv]}: {v['n_units'] - len(r)} of {v['n_units']} units at r = 1", fontsize=7)
            ax.set_ylabel("r", fontsize=7)
            data += [{"item": lab, "level": lv, "unit": n, "r": x} for n, x in zip(names, r, strict=True)]
    fig.suptitle(f"Retention r of each changed unit (dots); every other unit reads r = 1{note}")
    save(fig, path, n="one reading per unit on copy24's 75 held-out windows; replaced: one per fit seed", data=data)


def draw_contrast(panels: list[tuple[str, dict[str, dict[str, Any]]]], th: dict[str, Any], path: Path, note: str) -> None:
    """One panel per (title, items); per level a row of items, each the range over orders of the smallest sets' mean r."""
    fig, axes = figure((7.5, 3.4), ncols=len(panels))
    axes = axes if len(panels) > 1 else [axes]
    data = []
    for ax, (title, its) in zip(axes, panels, strict=True):
        labels = list(its)
        for j, lv in enumerate("AB"):
            for i, lab in enumerate(labels):
                y, c = i + 0.18 * (2 * j - 1), its[lab][lv]["contrast"]
                if c["reading"] == "no change":
                    ax.scatter([1.0], [y], facecolors="none", edgecolors=COLOR["do-nothing"], marker="o" if lv == "A" else "s", s=26)
                    continue
                if c["reading"] != "read":
                    ax.annotate(f"{lv}: {c['reading']}", (0.5, y), fontsize=6, ha="center", va="center")
                    continue
                lo, hi = min(c["retention"]), max(c["retention"])
                ax.plot([lo, hi], [y, y], color=COLOR[kind(lab)], linewidth=2)
                ax.scatter([sum(c["retention"]) / len(c["retention"])], [y], color=COLOR[kind(lab)], marker="o" if lv == "A" else "s", s=26)
                data += [{"panel": title, "item": lab, "level": lv, "order": k, "retention": x} for k, x in enumerate(c["retention"])]
        for x, t in ((th["low"], f"low bar {th['low']}"), (th["high"], f"high bar {th['high']}")):
            ax.axvline(x, color="#777777", linewidth=0.8, linestyle="--")
            ax.annotate(t, (x, 0.0), xycoords=("data", "axes fraction"), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
        ax.axvspan(th["low"], th["high"], color="#eeeeee", zorder=0)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.05)
        ax.set_xlabel("retention for the contrast (mean r over a smallest set)", fontsize=8)
        ax.set_title(title, fontsize=8)
    fig.suptitle(f"Rescaled must read at or above 0.7, replaced at or below 0.3; grey band: no call{note}\n"
                 "circles: level A (MLP neurons); squares: level B (heads and MLP blocks)", fontsize=9)  # fmt: skip
    save(fig, path, n="one smallest set per random order (5); replaced: every fit seed; open markers: do-nothing, every r = 1", data=data)


RUNS = {"E193": ("", "16 new neurons, 4,000 steps", "#E69F00", "o"), "E194": ("amend1", "32 new neurons, 12,000 steps", "#CC79A7", "s")}


def draw_calibration(path: Path) -> None:
    """The retention for the changed behaviour of the two items that were built, per level, from both runs' stages 0-1,
    against the 0.3 and 0.7 bars and the analytic value of a doubled neuron (0.80); the replaced item is marked not built."""
    fig, ax = figure((6.5, 3.2))
    rows, data = [], []
    for run, (sub, _, color, marker) in RUNS.items():
        s0, s1 = (json.loads((RESULTS / sub / f"stage{n}.json").read_text()) for n in (0, 1))
        for i, (item, rd) in enumerate((("do-nothing", s0["readings"]), ("rescaled", s1["readings"]))):
            for j, lv in enumerate("AB"):
                v, y = rd[lv], 2 * i + j
                vals = [1.0] if v["contrast"]["reading"] == "no change" else v["contrast"]["retention"]
                face = "none" if item == "do-nothing" else color
                ax.scatter(vals, [y + (0.12 if run == "E194" else -0.12)] * len(vals), facecolors=face, edgecolors=color, marker=marker, s=30)
                data += [{"run": run, "item": item, "level": lv, "retention": x, "changed_units": len(v["changed"]), "units": v["n_units"]} for x in vals]
                if run == "E193":
                    rows.append(f"{item}, {LEVELS[lv].split(': ')[1]}\n{len(v['changed'])} of {v['n_units']:,} units changed")
    for j, lv in enumerate("AB"):
        ax.annotate("replaced: never built (no fit passed both checks)", (0.5, 4 + j), ha="center", va="center", fontsize=7, color=COLOR["replaced"])
        rows.append(f"replaced, {LEVELS[lv].split(': ')[1]}")
    ax.axvspan(0.3, 0.7, color="#eeeeee", zorder=0)
    reference(ax, 0.3, "low bar 0.3")
    reference(ax, 0.7, "high bar 0.7")
    ax.axvline(0.8, color="#0072B2", linewidth=0.8, linestyle=":")
    ax.annotate("0.80: a neuron with its output doubled", (0.8, 0.5), xytext=(-3, 0), textcoords="offset points", ha="right", fontsize=6, color="#0072B2")
    ax.set_yticks(range(len(rows)), rows, fontsize=6.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("retention for the changed behaviour (mean r over a smallest set)", fontsize=8)
    ax.scatter([], [], facecolors="none", edgecolors="#555555", marker="o", label="open: do-nothing, no change, every r = 1")
    for run, (_, lab, color, marker) in RUNS.items():
        ax.scatter([], [], color=color, marker=marker, label=f"{run}: {lab}")
    ax.legend(fontsize=6, loc="center left")
    ax.set_title("Do-nothing reads 1, the doubled circuit reads high at both levels,\nand the replaced item was never built", fontsize=8, pad=14)
    save(fig, path, n="copy24's 75 held-out windows; 5 random orders of the smallest-set search per item, all equal; two runs, identical stages 0-1", data=data)


def draw_fits(path: Path) -> None:
    """Each try at building the replaced item: worse held-out fit ratio against max |corr| with a circuit neuron."""
    fig, ax = figure((5.5, 3.6))
    data = []
    for run, (sub, lab, color, marker) in RUNS.items():
        s2 = json.loads((RESULTS / sub / "stage2.json").read_text())
        for k, t in enumerate(s2["seeds"]["0"]["tries"]):
            x, y = max(t["fit_ratio_task"], t["fit_ratio_pile"]), t["max_abs_corr"]
            ax.scatter([x], [y], color=color, marker=marker, s=34, label=f"{run}: {lab}" if not data or data[-1]["run"] != run else None)
            off = (0, -12 - 9 * k) if run == "E194" else (4, 3)
            ax.annotate(f"λ {t['lambda']:g}", (x, y), xytext=off, textcoords="offset points", fontsize=6, color=color, ha="center" if run == "E194" else "left")
            data.append({"run": run, "seed": 0, "lambda": t["lambda"], "fit_ratio": x, "max_abs_corr": y})
    ax.set_xscale("log")
    ax.set_xlim(0.004, 1.2)
    ax.set_ylim(0, 1.08)
    ax.axvspan(0.004, 0.1, ymin=0, ymax=0.9 / 1.08, color="#d9f0d9", zorder=0)
    ax.annotate("passing corner:\nfits and is not a copy", (0.005, 0.45), fontsize=7, color="#2b7a2b")
    ax.axvline(0.1, color="#777777", linewidth=0.8, linestyle="--")
    ax.annotate("fit bar 0.1", (0.1, 0.02), xytext=(3, 0), textcoords="offset points", fontsize=7, color="#555555")
    ax.axhline(0.9, color="#777777", linewidth=0.8, linestyle="--")
    ax.annotate("copy bar 0.9", (0.25, 0.9), xytext=(0, -9), textcoords="offset points", fontsize=7, color="#555555")
    ax.set_xlabel("fit ratio: KL(replaced ‖ rescaled) / KL(rescaled ‖ M), worse of the two held-out sets", fontsize=7)
    ax.set_ylabel("largest |correlation| of a new neuron\nwith a circuit neuron", fontsize=7)
    ax.legend(fontsize=6, loc="center right")
    ax.set_title("No try lands in the passing corner: the fits that work copy a circuit neuron", fontsize=8)
    save(fig, path, n="one point per penalty weight λ tried, fit seed 0 of each run; held-out windows: copy24's 75 and 1,024 Pile", data=data)


def main(amend: str) -> None:
    res, figs = (RESULTS / amend, FIGURES / amend) if amend else (RESULTS, FIGURES)
    s = {n: json.loads((res / f"stage{n}.json").read_text()) for n in range(3)}
    pred = json.loads((res / "predictions.json").read_text())
    its = items(s[0], s[1], s[2])
    figs.mkdir(parents=True, exist_ok=True)
    draw_profile(its, figs / "profile.png", "")
    draw_contrast([(f"{pred['experiment']}, the three built items", its)], pred["thresholds"], figs / "contrast.png", "")


if __name__ == "__main__":
    if "--report" in sys.argv:
        draw_calibration(FIGURES / "calibration.png")
        draw_fits(FIGURES / "replaced_fits.png")
    else:
        main(sys.argv[sys.argv.index("--amend") + 1] if "--amend" in sys.argv else "")
