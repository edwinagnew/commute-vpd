"""P2.76 (regating-lead, 2026-11-24; $0, descriptive): Fig 5's effective rank on the same tasks as the shares.

The wide KL 'hi' set reads effective rank on 12 tasks but the re-gating and steering shares on 8 (four tasks gained
under one log-odds). This scores effective rank and the base prior with the filed Hope 1 statistics (score.hope1,
unchanged: Kendall tau-b with the tier, partial tau given the prior, task-permutation p) on the shares' 8 tasks, beside
the 12-task numbers in summary.json, for each rater set.

CPU, no model: ~/commute-agents/fleet/cpu-run --mem 2G -t 10 python3 experiments/convergence_panel/equal_n.py
-> results/wide/equal_n.json
"""

from __future__ import annotations

import json
from typing import Any

from commute.jobs import load_experiment

RUN: Any = load_experiment("convergence_panel", "run")
SC: Any = load_experiment("convergence_panel", "score")
OUT = RUN.RESULTS / "wide" / "equal_n.json"
KEEP = ("tasks", "tau", "p_perm", "tau_prior", "tau_partial_prior", "p_partial", "call")


def main() -> None:
    P = json.loads((RUN.RESULTS / "predictions.json").read_text())
    R, s = P["rules"], json.loads((RUN.RESULTS / "summary.json").read_text())["wide"]["kl_hi"]
    tiers, means = SC.raters(P, "raters_wide"), s["means"]
    shared = sorted(set(means["share"]) & set(means["steering"]))
    out: dict[str, Any] = {"shared_tasks": shared, "all_tasks": sorted(means["erank"])}
    for m in ("erank", "share", "steering"):
        oriented = {t: R["orientation"][m] * v for t, v in means[m].items()}
        out[m] = {r: {n: {k: v for k, v in SC.hope1({t: tier[t] for t in ts if t in tier}, oriented, means["prior"], R).items() if k in KEEP}
                      for n, ts in (("n12", out["all_tasks"]), ("n8", shared))}
                  for r, tier in tiers.items()}  # fmt: skip
    OUT.write_text(json.dumps(out, indent=1))
    for r in tiers:
        for n in ("n12", "n8"):
            e = out["erank"][r][n]
            print(f"{r:16s} {n}: erank tau {e['tau']:.3f} p {e['p_perm']:.3f}; prior tau {e['tau_prior']:.3f}; "
                  f"partial {e['tau_partial_prior']:.3f} p {e['p_partial']:.3f}; {e['call']}")  # fmt: skip


if __name__ == "__main__":
    main()
