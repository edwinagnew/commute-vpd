"""Step 1's figure: where the least-squares split puts a planted change, beside where it puts a random one (left);
the re-gating share of the nine states by kind and size (right). New decomposition p-9707d6ff.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_instrument_check/plot_step1.py            (after step 1)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_instrument_check/plot_step1.py --sketch   (the card's sketch)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
KIND = {"often": FG.Style("planted on often-on", "#0072B2", "o"), "rarely": FG.Style("planted on rarely-on", "#D55E00", "s"),
        "random": FG.METHODS["random"]}
SIZES = ("rl", "mid", "sft")
SIZE_LABEL = {"rl": "RL size\n0.064 nats", "mid": "midpoint\n0.32 nats", "sft": "fine-tune size\n1.64 nats"}
ROWS = (("often", "planted"), ("rarely", "planted"), ("often", "on"), ("rarely", "on"))
LABEL = {"planted": "on the planted set", "on": "on subcomponents already on"}
# Made-up numbers for the card: (kind, reading, planted, random) per rival.
SKETCH = {
    "If the split recovers a placement": [("often", "planted", 0.88, 0.01), ("rarely", "planted", 0.55, 0.002),
                                          ("often", "on", 0.93, 0.35), ("rarely", "on", 0.06, 0.35)],
    "If placement is free": [("often", "planted", 0.02, 0.01), ("rarely", "planted", 0.004, 0.002),
                             ("often", "on", 0.38, 0.35), ("rarely", "on", 0.33, 0.35)],
}


def draw(ax: Axes, recs: list[dict[str, Any]], title: str) -> None:
    """One row per (kind, reading): the planted change's share (kind colour) and the random change's at the same
    per-matrix energy (grey X), joined."""
    rnd = FG.METHODS["random"]
    seen: set[str] = set()
    for i, r in enumerate(recs):
        st = KIND[r["kind"]]
        ax.plot([r["planted"], r["random"]], [i, i], color="#bbbbbb", linewidth=0.8, zorder=1)
        ax.scatter([r["random"]], [i], color=rnd.color, marker=rnd.marker, s=26, zorder=2,
                   label="random change, same energy per matrix" if i == 0 else None)
        ax.scatter([r["planted"]], [i], color=st.color, marker=st.marker, s=26, zorder=3,
                   label=st.label if r["kind"] not in seen else None)
        seen.add(r["kind"])
    ax.set_yticks(range(len(recs)), [f"{r['kind']}-on change,\n{LABEL[r['reading']]}" for r in recs], fontsize=7)
    ax.set_ylim(len(recs) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the split's energy")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6.5, loc="center right", frameon=False)


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, rows) in zip(axes, SKETCH.items(), strict=True):
        recs = [{"kind": k, "reading": g, "planted": p, "random": r} for k, g, p, r in rows]
        draw(ax, recs, title)
        data += [{"panel": title, **r} for r in recs]
    axes[1].set_yticklabels([])  # the same rows as the left panel
    fig.suptitle("Planned figure (made-up numbers): does the split put a planted change where it was planted?", fontsize=9)
    FG.save(fig, out / "sketch_step1.png", n="made-up numbers; 648 planted subcomponents per kind over 28 down_proj matrices",
            data=data)


def regating(ax: Axes, s: dict[str, Any]) -> list[dict[str, Any]]:
    """The re-gating share of the nine states, one column per size, each with a bar of E132's seed spread centred on
    it (two bars that do not overlap differ by more than the spread). An unmatched state is drawn hollow."""
    half = s["seed_spread"] / 2
    data = []
    for j, (kind, st) in enumerate(KIND.items()):
        for i, z in enumerate(SIZES):
            p = s["states"][f"{kind}_{z}"]
            x, y = i + (j - 1) * 0.18, p["regating"]
            face = st.color if p["matched"] else "white"
            ax.errorbar([x], [y], yerr=[half], fmt="none", ecolor=st.color, elinewidth=1.0, capsize=2)
            ax.scatter([x], [y], marker=st.marker, s=30, facecolors=face, edgecolors=st.color, linewidths=1.0, zorder=3,
                       label=st.label if i == 0 else None)
            if not p["matched"]:
                ax.annotate(f"unmatched: KL {p['kl']:.2f},\ntarget {p['target']:.2f}", (x, y), xytext=(6, -16),
                            textcoords="offset points", fontsize=6, color=st.color)
            data.append({"kind": kind, "size": z, "x": x, **p})
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.annotate("do-nothing reads 1", (0.25, 1.0), xytext=(0, -9), textcoords="offset points", fontsize=6.5,
                color="#555555", ha="left")
    ax.set_xticks(range(len(SIZES)), [SIZE_LABEL[z] for z in SIZES], fontsize=7)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel("size, matched in output change on maths rows")
    ax.set_ylabel("re-gating share")
    ax.set_title(f"Re-gating share; bars = E132's seed spread {s['seed_spread']:.4f}", fontsize=8)
    ax.set_ylim(0.983, 1.0095)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    return data


def main(results: Path = HERE / "results" / "step1", out: Path = HERE / "figures") -> None:
    s = json.loads((results / "summary.json").read_text())
    for n, st in s["states"].items():
        st |= {k: json.loads((results / f"S_{n}.json").read_text())["search"][k] for k in ("kl", "target")}
    k = s["kinds"]
    recs = [{"kind": kind, "reading": g, "planted": k[kind][g], "random": k[kind]["chance" if g == "planted" else "on_null"]}
            for kind, g in ROWS]
    fig, (left, right) = FG.figure(FG.FULL_WIDTH, ncols=2)
    draw(left, recs, f"Split's placement (filed call: {s['calls']['split']['call']})")
    data = [{"panel": "split", **r} for r in recs] + [{"panel": "regating", **r} for r in regating(right, s)]
    fig.suptitle("The split returns the often-on set, only partly the rarely-on set;\n"
                 "the re-gating share stays between 0.985 and 1.006 for all nine states", fontsize=8.5)
    FG.save(fig, out / "step1.png", n="Left: 648 subcomponents per planted set over 28 down_proj matrices, one split per "
            "kind. Right: 9 states, 128 maths rows of 256; random at the fine-tune size is unmatched in size. "
            "New decomposition p-9707d6ff.", data=data)


if __name__ == "__main__":
    sketch() if "--sketch" in sys.argv else main()
