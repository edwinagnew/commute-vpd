# Plan: SGD and Adam at a matched size of change (emoticon, component training)

Design check: item 2 of the SGD-finding review; approved class 2 (a control the review names). Held fixed: emoticon task, steps, card, loss. Varied: optimiser and momentum at matched largest relative change. Not launched until the user has seen it.

**Request, restated.** `reports/2026-10-21_review_sgd-finding-influence.md`, Re-runs row 2. The sgd_grid follow-up
found plain SGD component training at lr 0.01 ten times less diffuse than Adam at 1e-5 (top-10 revert removes 0.156 of
the gain against 0.016) with less damage elsewhere (KL 0.0065 against 0.009). But SGD's update is smaller: its largest
relative change per matrix at step 30 is 0.0026 against Adam's 0.0066 (`rel_norm_max`, logged per step by
`component_training.train`; the sgd_grid analysis's table). This run asks whether both differences survive at equal
size of change.

**Run.** `experiments/sgd_matched/run.py run`, `cluster/sgd_matched.sbatch` (cluster `debug`, 30-min limit). Component
training on all alive (U, V), Γ and Δ frozen, arm A's loss, batch 32, clip 1, seed 0, 30 steps: sgd_grid's recipe.

1. **Control, first and alone.** Plain SGD lr 0.01 must reproduce `experiments/sgd_grid/results/sgd_grid.json` within
   5% relative on step-30 P(o) 0.991, KL elsewhere 0.0065, size 0.00264, and top-10 0.156 and top-100 0.476. If not,
   the run stops before the grids. It is also the reference arm (b). What differs from sgd_grid's run: a new job, and
   no lr 0.3 run before it (the trainer seeds its own batch generator and offsets, so the order should not matter).
2. **Adam grid** (a): lr 2e-6, 3e-6, 3.5e-6, 4e-6, 5e-6, 7e-6, 1e-5 (the last re-reads E4's 0.0066).
3. **Momentum grid** (c): SGD momentum 0.9, lr 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3.
4. **Match rule, filed:** per grid, the lr whose step-30 size is closest to the control's in |log ratio|, called
   matched if within 15%. An unmatched arm leaves its predictions unscorable.
5. At each matched rate: the revert-one concentration (top 1, 10, 100) and the log-odds gain.

**Expected values.** W₀: P(o) about 0.005, KL elsewhere 0, size 0, log-odds −5.02. Control: the sgd_grid numbers above.
Adam matched near 4e-6 (its size is roughly linear in lr): top-10 about 0.02, top-100 about 0.1, P(o) somewhere in 0.3
to 0.99. Momentum matched near 1e-3 (about a tenfold rate after ten steps): top-10 and KL near plain SGD's.

**Predictions** (`experiments/sgd_matched/results/predictions.json`; every measure is the arm's number over plain
SGD's, relative):
- P1, Adam's diffuseness is the optimiser: top-10 ratio < 0.33 (fail ≥ 0.67). Predicted pass.
- P2, the same at top 100: ratio < 0.5 (fail ≥ 0.75). Predicted pass.
- P3, Adam buys less log-odds gain per size of change: ratio < 0.75 (fail ≥ 1). Predicted pass.
- P4, SGD's "less damage" was size: Adam's KL-elsewhere ratio < 1.5. Predicted pass.
- P5, momentum concentrates like SGD: top-10 ratio in [0.5, 2] (fail < 0.33). Predicted pass.
- P6, momentum's KL elsewhere is like SGD's: ratio in [0.5, 2] (fail outside [0.33, 3]). Predicted pass.

**Blind to.** Concentration: redundancy beyond the greedy top k, positions outside the trigger. KL elsewhere: the
Pile and the trigger position. Faithfulness is not measured here (that is row 1). One seed: there is no seed noise
floor, so ratios inside [0.67, 1.5] read as "same", and the control's 5% only bounds run-to-run noise on one seed.
Positive control for the measure: E4's Adam top-10 arm reads 0.997.

**Runtime and cost.** sgd_grid job 28106: about 3 s per 30-step run and 2.5 min per full revert-one on an L40. Here 14
runs and 3 revert-ones, about 10 min with model load; $0 on the cluster.

**Decision tree.** P1 pass: Adam's diffuseness is the optimiser, and the review's wording "ten times less diffuse"
stands with "at matched size". P1 fail: the diffuseness gap was size, and the log says so. P4 fail: SGD does less damage
at equal size, and "less damage" stays. P5 fail: momentum alone makes the update diffuse, which points at the step
history rather than Adam's per-coordinate scaling.

**Status.** Code lint-clean; `make test` passes; the smoke has not run (no local model loads; run it first on the
cluster: `sbatch cluster/sgd_matched.sbatch run --smoke`). The smoke's control reads `sgd_grid_smoke.json` at the same
sizes (3 steps, 16 pieces per matrix).

## Not run

Everything above. Seeds beyond 0; Pile ratios and trigger-local checks on the matched states; matching on a different
size measure (total ‖ΔW‖ or the parameter distance).
