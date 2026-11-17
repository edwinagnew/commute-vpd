# qwen-rl-lead overnight digest, 2026-11-12

**P3.4, due 11-16, blocked on ask 15.** Run 1 of 3 is spent, and run 2 waits for your choice of recipe.

## Cells
- P3.4, open, 1 of 3 runs used: reinforcement-learning updates on Qwen large enough to read.

## What ran
- **Run 1 (E163), provisional, $50.82 on five H100/H200 pods.** No reinforcement-learning recipe reached five times the
  decomposition's error with held-out accuracy and answer length both held. At peak rate 3e-6 the update grew to 2.65
  times by step 150, at a pass rate of 0.891. At peak rate 1e-5 it reached 5.23 times in 25 steps, but only 0.829 of
  its answers fit the window. Write-up: `reports/analyses/2026-11-12_analysis_e163-qwen-rl-readable-run1.md`.
- **Check on the measure (cluster job 59301, $0, 2 min).** I applied run 1's own code to the earlier run's saved
  update. It read 0.064 again, the same as before, so the measure is right. The repeat in run 1 is a different update
  of the same norm: it moved the output only 0.048. Runs drift apart by up to about two times at a fixed step, so run 2
  matches its saved states to the reinforcement-learning arm of the same seed.

## Waiting on you
- **Ask 15** (card `reports/decisions/2026-11-12_decision_qwen-rl-run2-recipe.md`, published in the log as v326):
  which recipe run 2 trains.
  - I recommend (a): peak rate 3e-6 to step ~400, three arms on two seeds, six pods, about 8 h and $170. With run 1,
    that takes the cell to about $220, past the $200 allowance.
  - Other options: (b) peak 5e-6, about $115; (c) peak 1e-5 with a longer window, about $55 plus a code change;
    (d) save the 2x states only and report, $75-95.
  - The remote-kernels plugin still caps the session at $100, with about $49 left. Any option needs a `/mcp`
    reconnect to pick up the $300 setting.

## Spend
$50.82 this session, all on run 1. The check and the write-ups cost $0.

## Friction noted
732 (pods have no `uv` on PATH), 735 (job-gate rejects a colon in a pod label), 739 (runner polling on a detached
pod job), 740 (`cpu-run` looks for `.venv` relative to the current folder).

## Not run
Run 2 and run 3, both waiting on ask 15.
