"""E120 (qwen_pass_pools) stage 3, 2026-11-06: SFT of Qwen3-1.7B on one pass-rate pool's gold solutions, one arm per
job, through `commute.pool_sft` (stages, files, checks and exit codes in its docstring). Plan:
reports/plans/2026-11-06_plan_e120-pass-rate-pools.md, stage 3. Files go to results/ and results/large/; the pools
come from `--pools` (default results/pools.json; Amendment 2, 2026-11-06: results/pools_amend2.json); answers are
capped at 1,024 new tokens. The run refuses to start without results/predictions.json. cluster/qwen_pass_pools.sbatch.

Time per arm on an L40. Training: E61's gold arm took 1,732 s (train_gold_1e-5.json → seconds) with 24 held-out reads
this run does not make; its plain steps took ~11 s on GSM8K gold (run.log, steps 21-29), so 100 steps ~19 min at that
length and more for longer MATH solutions, ~20-30 min, plus ~6 fp32 saves at ~30 s each (E78's run.log). Files check
~3 min. Reads: GSM8K ~3 min (E61's fp32 W0 reading, 165 s); answers at cap 1,024 in batches of 64 sequences, about
1-2 min a batch (unmeasured; each batch's time is logged): 2 x n_held / 8 + n_train / 64 batches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from commute.pool_sft import POOL_NAMES, Arm, Setup, run

RESULTS = Path(__file__).resolve().parent / "results"
ANSWER_CAP = 1024


def main() -> None:
    p = argparse.ArgumentParser(description="E120 stage 3: SFT on one pool; parts in commute.pool_sft's docstring")
    p.add_argument("--pool", required=True, choices=POOL_NAMES)
    p.add_argument("--seed", required=True, type=int, choices=(0, 1))
    p.add_argument("--t-chunk", type=int, help="write a checkpoint and exit 3 after this many steps (default: none)")
    p.add_argument("--budget-min", type=float, default=170.0, help="stop before a unit that would pass this (exit 3)")
    p.add_argument("--pools", type=Path, default=RESULTS / "pools.json",
                   help="the pools file (Amendment 2: results/pools_amend2.json)")
    a = p.parse_args()
    arm = Arm(a.pool, a.seed)
    setup = Setup(RESULTS, a.pools, RESULTS / "predictions.json", RESULTS / "large", ANSWER_CAP)
    run(setup, arm, a.budget_min * 60, a.t_chunk)


if __name__ == "__main__":
    main()
