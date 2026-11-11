"""E145's figure: the original model's prior for the trained answer on E53's three tasks, each window a point, the
geometric mean with its 95% interval over windows, and E133's 24 maps' priors on the same axis as the reference.

    ~/commute-agents/fleet/cpu-run python experiments/prior_lowercase/plot.py [STAGE1_JSON]
"""

import json
import sys
from pathlib import Path

from commute.figures import FULL_WIDTH, METHODS, figure, save

HERE = Path(__file__).resolve().parent
E133 = HERE.parents[0] / "prior_vs_share" / "results" / "stage2.json"
LABEL = {"lowercase": "lowercase\n(built: performed)", "cipher": "cipher\n(built: new)", "broad_cipher": "broad cipher\n(built: new)"}


def main(path: Path) -> None:
    rec, ref = json.loads(path.read_text()), [p["log10_prior"] for p in json.loads(E133.read_text())["points"] if "log10_prior" in p]
    color = METHODS["prior"].color
    fig, ax = figure(FULL_WIDTH)
    data = []
    for i, (n, t) in enumerate(rec["tasks"].items()):
        ax.scatter(t["log10_p_windows"], [i] * len(t["log10_p_windows"]), color="#999999", s=6, alpha=0.25, linewidths=0)
        lo, hi = t["ci95"]
        ax.errorbar([t["log10_prior"]], [i], xerr=[[t["log10_prior"] - lo], [hi - t["log10_prior"]]], color=color, marker=METHODS["prior"].marker, ms=7, capsize=3)
        ax.annotate(f"{t['log10_prior']:.2f}", (t["log10_prior"], i), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=color)
        data += [{"row": n, "kind": "window", "log10_p": v} for v in t["log10_p_windows"]] + [{"row": n, "kind": "geometric mean", "log10_p": t["log10_prior"], "lo": lo, "hi": hi}]
    k = len(rec["tasks"])
    ax.scatter(ref, [k] * len(ref), color=color, marker="|", s=80, alpha=0.8)
    data += [{"row": "E133 maps", "kind": "map prior", "log10_p": v} for v in ref]
    ax.set_yticks(range(k + 1), [LABEL.get(n, n) for n in rec["tasks"]] + ["E133's 24 maps\n(reference)"])
    ax.invert_yaxis()
    ax.set_xlabel("log10 P(trained answer) under the original model")
    g = rec["gap"]
    ax.set_title(f"Original model's prior: lowercase minus the ciphers' mean, {g['gap']:+.1f} decades (95% interval {g['ci95'][0]:+.1f} to {g['ci95'][1]:+.1f})", fontsize=8)
    save(fig, HERE / "figures" / "prior.png", n="500 scored windows per task (grey, one per window); marker = geometric mean with its 95% bootstrap interval over windows; one value per task, the same for every seed",
         data=data)  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "stage1.json")
