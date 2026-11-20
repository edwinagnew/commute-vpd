"""E228's one figure: per fine-tune (bar: its own share, the band as an error bar) the KL-matched random draws' shares,
one panel per match site; a draw not scored (gain under 1 or match missed) is a cross below the axis, with its gain
written beside it. From results/ctl.json, draws_*.json and score.json (no GPU):

    ~/commute-agents/fleet/cpu-run python3 experiments/random_kl_share/plot.py   -> figures/random_share.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results"


def num(x: object) -> float:
    return float(x) if isinstance(x, int | float) else math.nan


def main() -> None:
    s, ctl = json.loads((R / "score.json").read_text()), json.loads((R / "ctl.json").read_text())["states"]
    states = list(ctl)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (rule, rec) in zip(axes, s["rules"].items(), strict=True):
        own = [num(ctl[x]["share"]) for x in states]
        ax.bar(range(len(states)), own, color="#bbbbbb", width=0.6, yerr=s["band"], capsize=3, label="fine-tune's own share, band")
        for i, st in enumerate(states):
            p = R / f"draws_{st.replace('/', '_')}.json"
            ds = [d for d in json.loads(p.read_text())["draws"].values() if d["site"] == rec["site"]] if p.is_file() else []
            for k, d in enumerate(ds):
                v, x = num(d["share"]), i + (k - 2) * 0.08
                if d["match"] == "ok" and math.isfinite(v):
                    ax.scatter(x, v, color="#7a3db8", s=22)
                else:
                    ax.scatter(x, -0.1, color="#7a3db8", marker="x", s=22)
                    ax.annotate(f"{num(d['gain']):.2f}", (x, -0.1), fontsize=5, rotation=90, va="top")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(states)), states, fontsize=7)
        ax.set_title(f"{rule}, KL matched at site '{rec['site']}': {rec['overall']}", fontsize=9)
    axes[0].set_ylabel("re-gating share r")
    axes[0].scatter([], [], color="#7a3db8", label="random draw, scored")
    axes[0].scatter([], [], color="#7a3db8", marker="x", label="not scored (gain under 1, printed)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{s['experiment']}: the share on random changes of each fine-tune's output size", fontsize=9)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "random_share.png", dpi=130)


if __name__ == "__main__":
    main()
