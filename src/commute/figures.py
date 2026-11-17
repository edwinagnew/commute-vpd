"""Shared figure code: one look for every plot, and the checks a plot must pass before it is saved.

Draw the comparison the claim rests on, not each arm's raw reading. A quantity indexed by two things (task pair by
method, block by matrix type) goes through `gap_heatmap`, as the difference, with the number in each cell and x for
"not run". Up to five points per group go through `dots`, which shows every point and refuses a mean or error bar.
`save` refuses a figure with no title, no axis label, or a dashed reference line (axhline/axvline) that is neither in
a legend nor labelled beside it (`reference` does the latter), writes the PNG (200 dpi), the PDF and the plotted numbers as a
CSV, and puts n and its unit under the figure.

    fig, ax = figure()
    gap_heatmap(ax, values, rows, cols, label="share, side built to reuse minus side built to be new")
    ax.set_title("The share separates the built sides on 9 of 10 pairs; steering on 5")
    save(fig, "experiments/x/figures/gaps.png", n="3 seeds per side, 256 rows", data=records)
"""

from __future__ import annotations

import csv
import math
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.text import Annotation


@dataclass(frozen=True)
class Style:
    label: str
    color: str
    marker: str


# The same method keeps its colour and marker in every figure (Okabe-Ito; the marker repeats the colour for greyscale).
METHODS: dict[str, Style] = {
    "regating": Style("re-gating share", "#0072B2", "o"),
    "neuron": Style("neuron version", "#D55E00", "s"),
    "neuron_half": Style("neuron version, half", "#009E73", "^"),
    "steering": Style("steering share", "#E69F00", "D"),
    "prior": Style("original model's prior", "#CC79A7", "v"),
    "size": Style("update size", "#56B4E9", "P"),
    "random": Style("random update, matched size", "#555555", "X"),
    "spectrum": Style("update's effective rank", "#000000", "h"),
}
ONE_COLUMN = (5.0, 3.2)
FULL_WIDTH = (7.5, 4.5)
MAX_DOTS_ONLY = 5


def figure(size: tuple[float, float] = ONE_COLUMN, ncols: int = 1, nrows: int = 1) -> tuple[Figure, Any]:
    """A figure at its final printed size, so its text is not shrunk later."""
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})
    return plt.subplots(nrows, ncols, figsize=size, layout="constrained")


def _missing(v: float | None) -> bool:
    return v is None or math.isnan(v)


def gap_heatmap(
    ax: Axes,
    values: Sequence[Sequence[float | None]],
    rows: Sequence[str],
    cols: Sequence[str],
    *,
    label: str,
    vmax: float | None = None,
    fmt: str = ".2f",
    counts: Sequence[Sequence[int | None]] | None = None,
    colorbar: bool = True,
) -> AxesImage:
    """Signed values on a diverging map centred at 0, each cell numbered; None or NaN is drawn x, meaning "not run".

    `label` names the colour in words. Pass one `vmax` to every panel of a grid so their colours can be compared.
    `counts`, the same shape as `values`, prints "n=k" under each number (seeds or rows behind the cell; None for none).
    `colorbar=False` leaves the bar off, for a panel that shares its `vmax` with one that draws it.
    A value that rounds to zero prints without a minus sign (friction 641).
    """
    if not label:
        raise ValueError("gap_heatmap: the colour needs a label")
    grid = [[math.nan if _missing(v) else float(v) for v in row] for row in values]  # type: ignore[arg-type]
    if len(grid) != len(rows) or any(len(r) != len(cols) for r in grid):
        raise ValueError(f"gap_heatmap: values do not match {len(rows)} rows by {len(cols)} columns")
    if counts is not None and (len(counts) != len(rows) or any(len(r) != len(cols) for r in counts)):
        raise ValueError(f"gap_heatmap: counts do not match {len(rows)} rows by {len(cols)} columns")
    seen = [abs(v) for row in grid for v in row if not math.isnan(v)]
    if vmax is None:
        vmax = max(seen, default=1.0) or 1.0
    image = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if math.isnan(v):
                ax.text(j, i, "×", ha="center", va="center", color="#777777")
                continue
            color = "white" if abs(v) > 0.6 * vmax else "#222222"
            count = None if counts is None else counts[i][j]
            ax.text(j, i if count is None else i - 0.12, _signed(format(v, fmt)), ha="center", va="center", color=color)
            if count is not None:
                ax.text(j, i + 0.22, f"n={count}", ha="center", va="center", color=color, fontsize=6)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, shrink=0.85).set_label(label)
    return image


def _signed(text: str) -> str:
    """'-0.00' -> '0.00': a minus on a number shown as zero says nothing (friction 641)."""
    return text[1:] if text.startswith("-") and not any(c in "123456789" for c in text.split("e")[0]) else text


def dots(ax: Axes, groups: Mapping[str, Sequence[float]], *, method: str, xlabel: str, offset: float = 0.0) -> None:
    """Every value as a point, one row per group. A group of up to five points gets no summary: its spread is the points."""
    style = METHODS[method]
    for i, values in enumerate(groups.values()):
        ax.scatter(values, [i + offset] * len(values), color=style.color, marker=style.marker, s=22, alpha=0.85, linewidths=0)
    ax.set_yticks(range(len(groups)), list(groups))
    ax.set_xlabel(xlabel)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def reference(ax: Axes, x: float, label: str) -> None:
    """A reference line labelled where it is drawn, not in a legend."""
    ax.axvline(x, color="#777777", linewidth=0.8, linestyle="--")
    ax.annotate(label, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")


def _unnamed_lines(ax: Axes, shown: set[str]) -> list[str]:
    """Dashed or dotted axhline/axvline in ax that a reader cannot name (friction 683: a cut-off and a "reached" line
    shipped unlabelled): named means its label is in a legend (`shown`), or a text or annotation in ax sits at its
    value, as `reference` draws it. Solid lines are separators and a line at 0 is the no-difference baseline: both pass."""
    spans = {"h": ax.get_yaxis_transform(which="grid"), "v": ax.get_xaxis_transform(which="grid")}
    places = [t.xy if isinstance(t, Annotation) else t.get_position() for t in ax.texts]
    out = []
    for line in ax.get_lines():
        kind = next((k for k, t in spans.items() if line.get_transform() is t), None)
        if kind is None or line.get_linestyle() in ("-", "None", " ", ""):
            continue
        value = float(line.get_ydata()[0] if kind == "h" else line.get_xdata()[0])  # type: ignore[index]
        at = [p[1] if kind == "h" else p[0] for p in places]
        if value == 0 or line.get_label() in shown or any(math.isclose(a, value, rel_tol=1e-6, abs_tol=1e-12) for a in at):
            continue
        out.append(f"ax{kind}line({value:g})")
    return out


def _check(fig: Figure) -> None:
    titled = bool(fig.get_suptitle()) or any(ax.get_title() for ax in fig.axes)
    if not titled:
        raise ValueError("save: the figure needs a title saying what it shows")
    legends = [leg for leg in [*fig.legends, *(ax.get_legend() for ax in fig.axes)] if leg is not None]
    shown = {t.get_text() for leg in legends for t in leg.get_texts()}
    for ax in fig.axes:
        if unnamed := _unnamed_lines(ax, shown):
            raise ValueError(f"save: {', '.join(unnamed)} has no legend entry and no label beside it; give it label= "
                             "and a legend, or draw it with figures.reference (friction 683)")
        if ax.images or ax.get_label() == "<colorbar>":
            continue
        if ax.has_data() and not (ax.get_xlabel() or ax.get_ylabel()):
            raise ValueError("save: a panel with data has no axis label")


def save(fig: Figure, path: str | Path, *, n: str, data: Sequence[Mapping[str, Any]]) -> Path:
    """Write path (.png, 200 dpi), the same name as .pdf, and the plotted numbers as .csv. `n` states n and its unit."""
    out = Path(path)
    if out.suffix != ".png":
        raise ValueError("save: give the .png path; the .pdf and .csv are written beside it")
    if not n.strip():
        raise ValueError("save: say what n counts, e.g. '3 seeds per side, 256 rows'")
    if not data:
        raise ValueError("save: pass the plotted numbers, so the figure can be redrawn without a rerun")
    _check(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Wrapped to the figure's width: one centred line clipped long footers without warning (friction 641).
    width = max(20, int(fig.get_figwidth() * 72 / (7 * 0.55)))
    fig.supxlabel("\n".join(textwrap.fill(part, width) for part in n.splitlines()), fontsize=7, color="#555555")
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    names = list(dict.fromkeys(k for row in data for k in row))
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(data)
    plt.close(fig)
    return out
