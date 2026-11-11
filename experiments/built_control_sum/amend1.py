"""E154 (cell P1.12, Amendment 1 of E150): run.py unchanged except for the plain-model capacity bar, 0.9 instead of 0.55,
filed in results/predictions_amend1.json (the run refuses unless it is committed unchanged). Plan:
reports/plans/2026-11-11_plan_built-control-sum.md, "Amendment 1, E154"; card: card_p1.12.txt.

Same jobs, stages, arms, stops and exit codes as run.py (its docstring). Writes results/amend1_{stage0,stage1,
taught_original}.json, results/amend1_{arm}_s{seed}.json and results/large/amend1_*.pt, so E150's stage0.json is kept.

    python experiments/built_control_sum/amend1.py job1|job2|job3     (cluster/built_control_sum_amend1.sbatch)
"""

from __future__ import annotations

from typing import Any

from commute.jobs import load_experiment

M: Any = load_experiment("built_control_sum", "run")
M.EXPERIMENT = "E154"
M.PRED = M.HERE / "results" / "predictions_amend1.json"
M.TAG = "amend1_"

if __name__ == "__main__":
    M.cli()
