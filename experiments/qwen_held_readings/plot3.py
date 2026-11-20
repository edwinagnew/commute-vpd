"""Cell P3.13 run 3: one figure from results/run3/summary.json (summary3.py) and run 2's results/run2/summary.json.
Top: per word pair, the gap S(reuse side) - S(new side) per seed, one line per save and kind (saved size; size-matched,
the larger side times E187's c), with the rule's bar at +-smallest_gap; gold / shuffled gold from run 2 (read cleanly
there) on the same axis for reference. Bottom: stage 0, the do-nothing KL(F||H) per fit and row set (log), with the
largest floor a fit may have to be chosen; a floor that reads <= 0 (rounding) is drawn at BOTTOM as an open triangle.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_held_readings/plot3.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from commute import figures as F

HERE = Path(__file__).resolve().parent
SAVED, MATCHED, ROWSET = "#0072B2", "#D55E00", ["#009E73", "#CC79A7"]
BOTTOM = 3e-11  # where a do-nothing floor that reads <= 0 is drawn on the log axis
TITLE = {"copy_cipher": "copy / cipher", "facts": "known / invented", "gold": "gold / shuffled (run 2)"}


def gaps(ax: Axes, name: str, pair: dict[str, Any], bar: float, label_lines: bool) -> list[dict[str, Any]]:
    """Each seed's gap as a dot, one line per save and kind; a missing gap (an S 'no change' or not read) is not drawn."""
    data, labels = [], []
    for save, c in pair["saves"].items():
        for kind, color in (("saved", SAVED), ("matched", MATCHED)):
            gs = c[kind]["gaps"]
            xs = [g for g in gs if g is not None]
            ax.scatter(xs, [len(labels)] * len(xs), s=22, color=color, linewidths=0)
            labels.append(f"{save}, {'saved size' if kind == 'saved' else 'size-matched'}: {c[kind]['rule']}")
            data += [{"pair": name, "save": save, "kind": kind, "seed_index": i, "gap": g} for i, g in enumerate(gs)]
    ax.set_yticks(range(len(labels)), labels if label_lines else [], fontsize=6.5)
    ax.invert_yaxis()
    bar_label = f"the rule's bar, gap = ±{bar:g}"
    for x in (bar, -bar):
        ax.axvline(x, color="#777777", linewidth=0.8, linestyle="--", label=bar_label)
    if label_lines:
        handles = [Line2D([], [], color="#777777", linewidth=0.8, linestyle="--", label=bar_label),
                   *(Line2D([], [], color=c, marker="o", ms=4, linestyle="", label=t)
                     for c, t in ((SAVED, "saved size"), (MATCHED, "larger side shrunk to the smaller's size")))]
        ax.legend(handles=handles, fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(0, 0.5))
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("gap per seed", fontsize=7)
    ax.set_title(f"{TITLE.get(name, name)}:\n{pair['outcome']}", fontsize=7)
    return data


def gap_axes(axes: Any, s3: dict[str, Any], s2: dict[str, Any]) -> list[dict[str, Any]]:
    """The two word pairs and run 2's gold pair, on one x range spanning every gap and the bar."""
    data = []
    for i, (name, pair) in enumerate([*s3["pairs"].items(), ("gold", s2["pairs"]["gold"])]):
        data += gaps(axes[i], name, pair, s3["smallest_gap"], i == 0)
    xs = [abs(d["gap"]) for d in data if d["gap"] is not None] + [2 * s3["smallest_gap"]]
    for ax in axes:
        ax.set_xlim(-1.1 * max(xs), 1.1 * max(xs))
    return data


def floors(ax: Axes, summary: dict[str, Any], floor_max: float) -> list[dict[str, Any]]:
    """Stage 0: per pod, per row set, the do-nothing KL(F||H) of each fit (x: the fits in filed order)."""
    data, fits = [], []
    for name, pair in summary["pairs"].items():
        for j, (rs, by_fit) in enumerate(pair["stage0"]["kl_fh"].items()):
            fits = list(by_fit)
            pts = [(i, v if v > 0 else BOTTOM) for i, v in enumerate(by_fit.values()) if v is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, lw=0.8, color=ROWSET[j],
                    linestyle="-" if name == "copy_cipher" else ":",
                    label=f"{rs} (pair's fit: {pair['stage0']['choice'] or 'none cleared'})")
            low = [p for p in pts if p[1] == BOTTOM]
            ax.scatter([p[0] for p in low], [p[1] for p in low], marker="v", s=40, facecolors="white",
                       edgecolors=ROWSET[j], zorder=3,
                       label=f"{rs}: floor reads at or below 0 (value beside), drawn at {BOTTOM:g}" if low else None)
            for i, _ in low:
                last = i == len(by_fit) - 1
                ax.annotate(f"{list(by_fit.values())[i]:.1e}", (i, BOTTOM), xytext=(-8 if last else 8, 0),
                            textcoords="offset points", fontsize=6, ha="right" if last else "left", va="center",
                            color="#555555")
            data += [{"pair": name, "rowset": rs, "fit": f, "kl_fh": v} for f, v in by_fit.items()]
    ax.set_yscale("log")
    ax.set_ylim(bottom=BOTTOM / 3)
    ax.axhline(floor_max, color="#777777", linewidth=0.8, linestyle="--")
    ax.annotate(f"{floor_max:g}, largest floor chosen", (0, floor_max), xytext=(2, 2), textcoords="offset points",
                fontsize=6.5, color="#555555")
    ax.set_xticks(range(len(fits)), fits, fontsize=6.5)
    ax.set_xlabel("fit rows (own rows 0-63 first: run 2's fit)", fontsize=7)
    ax.set_ylabel("do-nothing KL(F||H) per position", fontsize=7)
    ax.set_title("stage 0: does the held model reproduce M on the read rows?", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False)
    return data


def main() -> None:
    s3 = json.loads((HERE / "results" / "run3" / "summary.json").read_text())
    s2 = json.loads((HERE / "results" / "run2" / "summary.json").read_text())
    fig, axes = F.figure((F.FULL_WIDTH[0], 6.5), ncols=3, nrows=2)
    data = gap_axes(axes[0], s3, s2)
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1]:
        ax.remove()
    data += floors(fig.add_subplot(gs[1, :]), s3, s3["floor_max"])
    fig.suptitle(f"Qwen word pairs, S gap per seed with a fit that reproduces M: {s3['outcome']}", fontsize=9)
    F.save(fig, HERE / "figures" / "run3_S.png", data=data,
           n="gap = S(reuse side) - S(new side), 3 seeds per pair (dots), paired by seed; blue saved size, orange the larger side times E187's c; S read on "
             "rows 64-127 of each arm's own rows with the fit chosen in stage 0 (bottom); gold from run 2, own rows 0-63")


if __name__ == "__main__":
    main()
