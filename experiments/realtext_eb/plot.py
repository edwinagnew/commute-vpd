"""E### (cell P2.64) figure from results/summary.json (run.py --stage summary): figures/eb_k.png.

Left: E_b over K per task side, the side built to need what M holds above its twin, one dot per seed, the fine-tunes
beside their matched random updates (E246's draws at the same per-matrix norms). A state whose K has no changed member
reads 1 by convention and is drawn as an open circle. Right: M's prior on each task (E246's geometric-mean probability of
the scored token at W0), a confound shown beside, not a reading. `--sketch` draws the card's sketch (figures/sketch.png)
from made-up E_b values, one panel per rival, with E246's saved priors. Writes .png, .pdf and .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/realtext_eb/plot.py [SUMMARY] [OUT_DIR]
    ~/commute-agents/fleet/cpu-run python3 experiments/realtext_eb/plot.py --sketch
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
E246 = HERE.parent / "realtext_proxies" / "results"
SIDES = {"L1": "topic label (L1)", "L3": "coin label (L3)", "T1": "lowercase (T1)", "T4": "reversed order (T4)",
         "T2": "French (T2)", "T3": "Yoruba (T3)"}  # fmt: skip
OWN, RND = "regating", "random"  # figures.METHODS styles: fine-tune, matched random


def eb_panel(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
    for rnd, method, off in ((False, OWN, -0.15), (True, RND, 0.15)):
        pts = [r for r in rows if r["random"] == rnd and not r["degenerate"]]
        FG.dots(ax, {SIDES[t]: [r["E_b_K"] for r in pts if r["task"] == t] for t in SIDES}, method=method,
                xlabel="E_b over K (write share over the needed set)", offset=off)  # fmt: skip
        st = FG.METHODS[method]
        ax.scatter([], [], color=st.color, marker=st.marker, s=22, label="random update, same per-matrix norms" if rnd else "fine-tune")
    degen = [r for r in rows if r["degenerate"]]
    for r in degen:
        ax.scatter([1.0], [list(SIDES).index(r["task"]) + (0.15 if r["random"] else -0.15)], facecolors="none", edgecolors="#D55E00", s=50, linewidths=1.1)
    if degen:
        ax.scatter([], [], facecolors="none", edgecolors="#D55E00", s=50, label="K has no changed member: 1 by convention")
    ax.set_xscale("log")
    ax.set_xlim(0.004, 1.3)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    ax.set_title(title, fontsize=9)


def prior_panel(ax: Any, prior: dict[str, float]) -> list[dict[str, Any]]:
    FG.dots(ax, {SIDES[t]: [math.log10(prior[t])] for t in SIDES}, method="prior", xlabel="log10 of M's prior on the scored token")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Base prior (a confound, not a reading)", fontsize=9)
    return [{"task": t, "panel": "prior", "log10_prior": math.log10(prior[t])} for t in SIDES]


def e246_prior() -> dict[str, float]:
    out = {}
    for i in (0, 1):
        out |= {t: v["prior"] for t, v in json.loads((E246 / f"pod{i}.json").read_text())["prior"].items() if t in SIDES}
    return out


def main(summary: Path, out_dir: Path) -> None:
    s = json.loads(summary.read_text())
    rows = [{k: r[k] for k in ("item", "task", "seed", "random", "E_b_K", "degenerate", "size_K", "n_changed_K", "size")} | {"panel": "E_b"}
            for r in s["items"].values()]  # fmt: skip
    fig, (a, b) = FG.figure((10.0, 4.2), ncols=2)
    eb_panel(a, rows, "E_b over K on the real-text fine-tunes and their matched randoms")
    data = rows + prior_panel(b, s["prior"])
    FG.save(fig, out_dir / "eb_k.png", data=data,
            n=f"{sum(not r['random'] for r in rows)} fine-tunes (3 seeds per side, scored size) and their matched random updates; "
              "K from 256 pairs of D's windows per update (4 derangements of 64), read at one position.")  # fmt: skip


def sketch(out_dir: Path) -> None:
    """Made-up E_b values: rival a (first sides above their twins, all above the randoms), rival b (all at the randoms)."""
    made = {"a": {"L1": 0.08, "L3": 0.03, "T1": 0.12, "T4": 0.04, "T2": 0.06, "T3": 0.03}, "b": dict.fromkeys(SIDES, 0.013)}
    fig, axes = FG.figure((13.0, 4.0), ncols=3)
    data: list[dict[str, Any]] = []
    for ax, (rival, eb) in zip(axes[:2], made.items(), strict=True):
        rows = [{"task": t, "seed": s, "random": rnd, "E_b_K": (0.011 if rnd else eb[t]) * (1 + 0.08 * (s - 1)), "degenerate": False, "rival": rival}
                for t in SIDES for s in (0, 1, 2) for rnd in (False, True)]  # fmt: skip
        eb_panel(ax, rows, f"SKETCH, made-up numbers: rival {rival}")
        data += rows
    data += prior_panel(axes[2], e246_prior())
    FG.save(fig, out_dir / "sketch.png", data=data, n="SKETCH with made-up E_b values (priors: E246's saved values); 3 seeds per side")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sketch"]:
        sketch(HERE / "figures")
    else:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json", Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures")
