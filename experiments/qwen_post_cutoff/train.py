"""E128 (qwen_post_cutoff) stage 3, one arm per job, through `commute.pool_sft`: SFT of Qwen3-1.7B on one post-cutoff or
known facts pool (stages, files, checks and exit codes in pool_sft's docstring). Plan:
reports/plans/2026-11-06_plan_e128-post-cutoff-facts.md, stage 3. The run refuses to start without
results/predictions.json. Answers are capped at 64 new tokens, the sampling cap of stages 0-1 (Fable (c): the facts'
cap is measured in stage 0; the appendix's 1,024 is E120's maths cap and would only make answers slower here).
cluster/e128_train.sbatch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from commute.pool_sft import POOL_NAMES, Arm, Setup, run

RESULTS = Path(__file__).resolve().parent / "results"
ANSWER_CAP = 64


def main() -> None:
    p = argparse.ArgumentParser(description="E128 stage 3: SFT on one pool; parts in commute.pool_sft's docstring")
    p.add_argument("--pool", required=True, choices=POOL_NAMES)
    p.add_argument("--seed", required=True, type=int, choices=(0, 1))
    p.add_argument("--t-chunk", type=int, help="write a checkpoint and exit 3 after this many steps (default: none)")
    p.add_argument("--budget-min", type=float, default=170.0, help="stop before a unit that would pass this (exit 3)")
    p.add_argument("--pools", type=Path, default=RESULTS / "pools.json")
    a = p.parse_args()
    run(Setup(RESULTS, a.pools, RESULTS / "predictions.json", RESULTS / "large", ANSWER_CAP), Arm(a.pool, a.seed),
        a.budget_min * 60, a.t_chunk)


if __name__ == "__main__":
    main()
