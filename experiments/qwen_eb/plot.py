"""E270 (cell P3.14, qwen_eb): the card's figure from results/stage1.json and stage0.json. Per arm, the two seeds' E_b
as dots, their matched random changes as a band, the gain x2 control as a labelled reference line.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_eb/plot.py [--results DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as F

HERE = Path(__file__).resolve().parent
ARM = {"R": "RL", "N": "RL, shuffled reward", "F": "SFT, own samples"}


def draw(ax: Axes, trained: dict[str, list[float]], random: dict[str, list[float]], gain: float, title: str
         ) -> list[dict[str, object]]:
    """One panel: dots per arm, the band min..max of its random changes, the gain control as a dashed line."""
    F.column_dots(ax, trained, method="regating", ylabel="write share E_b",
                  bands={a: (min(r), max(r)) for a, r in random.items()})
    ax.axhline(gain, color="#777777", linewidth=0.8, linestyle="--")
    ax.annotate("gain x2 control", (0.0, gain), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
    ax.set_ylim(0, max(0.85, gain + 0.08))
    ax.set_title(title, fontsize=8)
    return ([{"panel": title, "arm": a, "kind": "update", "E_b": v} for a, vs in trained.items() for v in vs]
            + [{"panel": title, "arm": a, "kind": "random", "E_b": v} for a, vs in random.items() for v in vs]
            + [{"panel": title, "arm": "", "kind": "gain x2", "E_b": gain}])


def by_arm(values: dict[str, float]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name in sorted(values, key=lambda n: ("RNF".index(n[0]), n)):
        out.setdefault(ARM[name.split("_s")[0]], []).append(values[name])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "figures" / "eb.png"))
    a = ap.parse_args()
    s1: dict[str, Any] = json.loads((Path(a.results) / "stage1.json").read_text())
    fig, ax = F.figure()
    data = draw(ax, by_arm(s1["E_b"]), by_arm(s1["E_b_random"]), s1["gain_x2_E_b"],
                f"Step-75 Qwen updates on {s1['decomp']} (bf16 floor {s1['floor']:.4f})")
    F.save(fig, a.out, data=data, n="2 seeds per arm; band = the two matched random changes (same per-matrix norm)")


if __name__ == "__main__":
    main()
