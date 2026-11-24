"""E279's one figure: each task's mean untrained-SVD share (3 seeds) in a row per consensus tier, with the span of the
panel Gaussians' shares (each over its paired state's gain) behind them. Reads results/summary.json (run.py score).

    ~/commute-agents/fleet/cpu-run python3 experiments/untrained_panel/plot.py [--results DIR]
-> figures/untrained_by_tier.png (.pdf, .csv); with --results DIR (a stub) the figure goes to DIR/untrained_by_tier.png.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
SC: Any = load_experiment("convergence_panel", "score")  # raters
RUN: Any = load_experiment("convergence_panel", "run")  # PRED (E272's raters)


def main() -> None:
    stub = Path(sys.argv[sys.argv.index("--results") + 1]) if "--results" in sys.argv else None
    S = json.loads(((stub or HERE / "results") / "summary.json").read_text())
    tier = SC.raters(json.loads(RUN.PRED.read_text()), "raters_wide")["consensus"]
    means, gs = S["means"], {t: v for t, v in S["gauss"]["share_paired"].items() if isinstance(v, (int, float)) and math.isfinite(v)}
    levels = sorted({tier[t] for t in means}, reverse=True)
    groups = {f"tier {lv:g} ({', '.join(sorted(t for t in means if tier[t] == lv))})": [means[t] for t in sorted(means) if tier[t] == lv] for lv in levels}
    fig, ax = FG.figure(FG.FULL_WIDTH)
    if gs:
        ax.axvspan(min(gs.values()), max(gs.values()), color=FG.METHODS["random"].color, alpha=0.2, linewidth=0,
                   label=f"Gaussian updates, matched size ({len(gs)} tasks)")  # fmt: skip
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    FG.dots(ax, groups, method="regating", xlabel="untrained-SVD share (numerator over the fine-tune's gain)")
    tau = S["1_tiers"]
    ax.set_title(f"Untrained-SVD share by consensus tier: tau-b {tau['tau']:.2f} (p {tau['p']:.3f}, {tau['n']} tasks)")
    rows = [{"task": t, "tier": tier[t], "mean_share": means[t], "seeds": S["seeds"][t]} for t in sorted(means)]
    rows += [{"task": t, "tier": tier.get(t), "gaussian_share_paired": v} for t, v in sorted(gs.items())]
    missing = sorted(set(S["seeds"]) - set(means))
    FG.save(fig, (stub or HERE / "figures") / "untrained_by_tier.png", data=rows,
            n=f"each dot one task, the mean of 3 seeds at E272's KL hi; {len(means)} tasks read, missing (a seed's gain under 1 log-odds): {', '.join(missing) or 'none'}")  # fmt: skip


if __name__ == "__main__":
    main()
