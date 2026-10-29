# Run brief: E45 Amendment 1 — anchor 2 at 300 steps ($0, MATS cluster `debug`, one L40 per job)

Note `reports/plans/2026-10-28_plan_qwen-q2-amend1.md` (plan §8's branch "anchor 2 not learned at step 100"). Rules and first-minute values: `results/predictions_amend1.json` (an `a2long` run refuses to start without it). Arm `a2long` = anchor 2's data and recipe, 300 steps, cosine over 300, warm-up 10. Runs `a2long_s0`, `a2long_s1`; `a2_s0` stays as it ran, and `a2_s1` (RUN_BRIEF row 8) is not run.

**Before the first job**, as in `RUN_BRIEF.md`: `git pull` in the lead's clone, `ls -l experiments/qwen_updates/results/large` is the link into `~/commute-data/results-large/qwen_updates`, and `results/control.json` and `results/T_a2_s0.json` are there (T reads control's `gold_mean_tokens`; the reproduction check's values come from `T_a2_s0`). Submit from the clone root, one job at a time. Record job IDs.

**Exit codes.** 0 done. **2 = STOP**: a first-minute check missed (`[check] ... MISSED`): report to the lead before any further job. **3 = resubmit the same command**: T stops after step 150 with a checkpoint (`BUDGET: checkpoint after step 150 (...)`); E stops before its 25-min budget. Any other exit (an `AssertionError` such as "the checkpoint's weights were not put back") = STOP, report.

**T takes two jobs.** 300 steps would take ~27 min on an L40 (a2_s0: ~10 s per step to step 30, ~4.3 s after), too close to the 30-min cap. T writes a checkpoint after step 150 (fp32 weights and Adam moments, ~17 GB, `large/ckpt/a2long_s<seed>.pt`) and exits 3; the resubmitted job goes on from step 151 exactly as an unbroken run would (`tests/test_qwen_updates.py::test_checkpoint_resume_is_exact`) and deletes the checkpoint at the end. A T started without a checkpoint is a fresh start: it deletes that run's `E_<run>.json`, `large/E/<run>/` and `large/dW/<run>/` first. So never resubmit a finished T.

| # | command (`sbatch` + args) | L40 time | first values to see in `cluster/logs/commute-qwen_updates-<id>.out` | stop rule (code) | output |
|---|---|---|---|---|---|
| 1 | `--mem=40G cluster/qwen_updates.sbatch --stage T --arm a2long --seed 0` | ~17 min, exit 3 | step 1: lr 0, \|dW\| 0, KL 0; step 2: lr 1.00e-06, band ok (a2_s0: 0.73-0.91), KL ≤ 0.005 (a2_s0: 0.00087); **`step 2/3/5/10 tracker KL against a2_s0: ok`** (a2_s0 maths 0.00063, 0.0043, 0.075, 0.26; own 0.071, 0.25, 1.03, 2.33); saves at 1-6, 11, 12 (as a2_s0) and 100; last: `step 150`, `BUDGET: checkpoint after step 150` | step-1, step-2 checks; tracker KL off a2_s0 by > 5% at 2, 3, 5 or 10 | `results/T_a2long_s0.json`, `large/dW/a2long_s0/`, `large/ckpt/a2long_s0.pt` |
| 2 | the same command again | ~13 min, exit 0 | `[T a2long_s0] resuming after step 150`; step 151 lr ~5.27e-06, loss near step 150's, \|dW\| just above step 150's; step 300 lr ~3e-10; `[save] ... step_300` | resume assertion | `T_a2long_s0.json` `done` true; checkpoint gone |
| 3 | `--mem=32G cluster/qwen_updates.sbatch --stage E --arm a2long --seed 0`, resubmit on exit 3 | ~23 min (+1 resubmit likely) | one `state` line per saved state (~55 s each, 11 states), then `generation at` each chosen step, 100 and 300 (~95 s each) | — | `results/E_a2long_s0.json`, `score.json` |
| — | **read `results/score.json` → `amend1_anchor2_learns_s0`** | | `holds` true → rows 4-5; false → **stop, report to the lead** (plan §8: card to the user) | | |
| 4 | rows 1-2 with `--seed 1` | ~17 + 13 min | as row 1, without the reproduction check (no a2_s1); `data` the same `rows_per_step` (27) | step-1, step-2 checks | `T_a2long_s1.json` |
| 5 | row 3 with `--seed 1` | ~23 min (+1) | as row 3 | — | `E_a2long_s1.json`, `score.json` `amend1_anchor2_learns_s1` |

**What decides the next job.** Row 1's reproduction check is the control: a miss means a2long is not anchor 2's recipe, and nothing else runs. `amend1_anchor2_learns_s0` gates seed 1 and, with seed 1, the rest of Job 1 (`RUN_BRIEF.md` rows 5-8, whose `--arm a2 --seed 1` is replaced by rows 4-5 here). Either seed false → the lead (card to the user).

**Reporting.** After each job: job ID, exit code, wall time, peak GPU/host (last log line), and: T, KL at steps 2, 5, 10, 100, 150 and 300, the reproduction checks, saves, loss at 150/151; E, `chosen` with matched KLs, `learned` flags, facts (invented, known) and GSM8K strict accuracy at 100 and 300; score, `amend1_anchor2_learns_s<seed>` whole. Commit `results/*.json` and `run.log` as they appear (never `large/`).

**Disk.** Per run 10 δW states × 2.82 GB = 28.2 GB (steps 1, 2, 3, 4, 5, 6, 11, 12, 100, 300 expected), plus the 17 GB checkpoint between the two T jobs. Both seeds: ~56 GB on `commute-data`.

Expected totals: 8 `debug` jobs (T 2 + E ~2 per seed), ~2.5 h, $0.
