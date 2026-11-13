"""P3.10's one figure, from results/summary.json (run.py --summary): per row set, the re-gating share against the
change's size K00 on p-015e0710 (MLP matrices only). Random changes (three draws at three sizes) give the curve and its
band; the built changes (neurons the model uses switched off, three choices of set) and the library's fine-tunes are
placed on the same axes. No reuse or new label: a description. `--sketch` draws sketch.png, the same panel with made-up
numbers, one panel per rival of the card.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_share_reference/plot.py [--sketch]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from commute import figures as F

HERE = Path(__file__).resolve().parent
CHOICE = {"top": ("most used", "#0072B2", "o"), "random": ("random among used", "#009E73", "^"),
          "specific": ("used most on this text", "#D55E00", "s")}
FT = {"5x": ("fine-tune, matched save", "#CC79A7", "v"), "last": ("fine-tune, late save", "#882255", "v")}
RND = F.METHODS["random"]
XLABEL, YLABEL = "size: K00, nats per position (log)", "re-gating share"


def panel(ax: Axes, rs: dict[str, Any], built: list[dict[str, Any]], fts: list[dict[str, Any]], title: str
          ) -> list[dict[str, Any]]:
    """One row set: random draws (X), the curve through their size means and its band, built changes, fine-tunes."""
    data = []
    cv, bd = rs["curve"], rs["band"]
    if len(cv) >= 2:
        xs, ys = [p[0] for p in cv], [p[1] for p in cv]
        ax.plot(xs, ys, color=RND.color, lw=1)
        if bd is not None:
            ax.fill_between(xs, [y - bd for y in ys], [y + bd for y in ys], color="#dddddd", lw=0, zorder=0)
    for d in rs["draws"]:
        ax.scatter([d["k00"]], [d["share"]], color=RND.color, marker=RND.marker, s=22, alpha=1 if d["usable"] else 0.3)
        data.append({"panel": title, "kind": "random", "label": f"d{d['draw']} {d['level']}", "k00": d["k00"],
                     "share": d["share"], "class": "" if d["usable"] else "unusable"})
    for f in fts:
        _, color, marker = FT[f["save"]]
        ax.scatter([f["k00"]], [f["share"]], color=color, marker=marker, s=14)
        data.append({"panel": title, "kind": "fine-tune", "label": f"{f['arm']} s{f['seed']} {f['save']}",
                     "k00": f["k00"], "share": f["share"], "class": f["class"]})
    for b in built:
        if b["k00"] is None:
            continue
        _, color, marker = CHOICE[b["choice"]]
        ax.scatter([b["k00"]], [b["share"]], color=color, marker=marker, s=40, edgecolor="black", lw=0.5, zorder=3)
        data.append({"panel": title, "kind": "built", "label": f"{b['choice']} k={b['k']}", "k00": b["k00"],
                     "share": b["share"], "class": b["class"]})
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    return data


def legend(ax: Axes) -> None:
    ax.axis("off")
    handles = [Line2D([], [], color=RND.color, marker=RND.marker, ls="-", label="random change; line: mean per size"),
               Patch(color="#dddddd", label="band: largest random draw's distance from the line")]
    handles += [Line2D([], [], color=c, marker=m, ls="", label=f"built: {n}") for n, c, m in CHOICE.values()]
    handles += [Line2D([], [], color=c, marker=m, ls="", label=n) for n, c, m in FT.values()]
    ax.legend(handles=handles, loc="center", fontsize=7, frameon=False)


def main() -> None:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    fig, axes = F.figure((F.FULL_WIDTH[0], 5.6), ncols=3, nrows=2)
    data: list[dict[str, Any]] = []
    for ax, (r, rs) in zip(axes.flat, s["rowsets"].items(), strict=False):
        data += panel(ax, rs, [b for b in s["built"] if b["rowset"] == r],
                      [f for f in s["finetunes"] if f["rowset"] == r], f"{r} rows")
    legend(axes.flat[-1])
    fig.suptitle(f"Re-gating share against output size on the frozen-neuron decomposition: built changes "
                 f"{s['outcome']['built']}, fine-tunes {s['outcome']['finetunes']}", fontsize=9)
    F.save(fig, HERE / "figures" / "share_reference.png", data=data,
           n="Per row set: 3 random draws at each of 3 sizes (the row set's fine-tunes' lowest, middle and highest "
             "K00), 3 built changes at the middle size, the library's 6 fine-tunes (12 on gold rows: gold and "
             "shuffled gold), 3 seeds x 2 saves; each read on the row set's 128 rows of 256 tokens; MLP matrices only")


def sketch() -> None:
    """The planned figure with made-up numbers, one panel per rival."""
    draws = [{"draw": d, "level": lv, "k00": k * (1 + 0.03 * (d - 1)), "share": s + 0.008 * (d - 1), "usable": True}
             for lv, k, s in [("low", 0.4, 0.97), ("mid", 0.8, 0.92), ("high", 1.6, 0.85)] for d in range(3)]
    rs = {"curve": [(0.4, 0.97), (0.8, 0.92), (1.6, 0.85)], "band": 0.009, "draws": draws}
    fts = [{"arm": "copy", "seed": i, "save": v, "k00": k, "share": sh, "class": ""}
           for i, (v, k, sh) in enumerate([("5x", 0.42, 0.985), ("5x", 0.45, 0.982), ("last", 1.5, 0.93),
                                           ("last", 1.6, 0.94), ("last", 1.4, 0.925), ("5x", 0.41, 0.99)])]
    rivals = {"rival A: Γ switches zeroed neurons\noff; built above the band": (0.995, 0.99, 0.985),
              "rival B: the share tracks size\nonly; built inside the band": (0.925, 0.915, 0.93)}
    fig, axes = F.figure((F.FULL_WIDTH[0], 3.4), ncols=3)
    data: list[dict[str, Any]] = []
    for ax, (title, shares) in zip(axes, rivals.items(), strict=False):
        built = [{"choice": c, "k": 0, "k00": 0.8 * (1 + 0.02 * i), "share": sh, "class": ""}
                 for i, (c, sh) in enumerate(zip(CHOICE, shares, strict=True))]
        data += panel(ax, rs, built, fts, title)
    legend(axes[-1])
    fig.suptitle("SKETCH, made-up numbers: the share against size on one row set, under each rival", fontsize=9)
    F.save(fig, HERE / "sketch.png", data=data, n="Made-up numbers drawn before the run (card sketch); one row set")


if __name__ == "__main__":
    sketch() if "--sketch" in sys.argv else main()
