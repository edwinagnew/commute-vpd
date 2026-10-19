# Plan: the task ladder. Does the decomposition predict which post-training updates it can follow?

*2026-10-15, coding agent. Round 3, confirmed by the user in chat after the design check ("go ahead with the ladder, but
fix the harness first"). The harness is fixed in this session (`src/commute/local_faith.py`, tests in
`tests/test_local_faith.py`); the run goes to another session. Nothing has been run. Predictions are filed by
`experiments/task_ladder/run.py --predictions` (the text is in `PREDICTIONS` there and summarised below); the run refuses
to start without the file. Experiment directory: `experiments/task_ladder/`.*

**Amended 2026-10-16 before running:** `reports/2026-10-16_prediction_ladder-routes-amendment.md` replaces G1 and G1b and adds the route split (A1, A2). Read it first.

## The claim, the null, and what separates them

**Claim.** Whether post-training *inside* the decomposition (the pull-back: train the alive U, V with Δ and Γ fixed)
keeps the new behaviour explained by Γ-gated components is predicted, before training, by how much of the behaviour's
gradient Γ already covers at the trigger. Rarity of the behaviour does not matter on its own.

**Null (rarity).** What matters is how rare the behaviour was: the rarer the target, the less of it ends up in Γ-gated
components, whatever the coverage. A second null (nothing matters): inside training is faithful at the trigger
everywhere, or nowhere.

**Separation.** Tasks at matched rarity but different coverage (P1), and tasks at matched coverage but different rarity
(P2). The emoticon task is the reference rung.

## Design

- **Tasks.** Trigger → target single tokens mined from Pile train rows 0..32767. Triggers: the 40 most frequent word-like
  tokens (leading space, letters only) with ≥ 400 occurrences at positions 20..491. Baseline P(v | t) is the mean, over
  the trigger's occurrences, of the model's next-token probability of v with the full-row context. **Rung** r is the
  decade of the baseline: 0: [0.1, 1), 1: [0.01, 0.1), 2: [0.001, 0.01), 3: [1e-4, 0.001), 4: < 1e-4. "Lower" rungs are
  behaviours the model already has to a degree; rung 4 is the new-skill end. **First batch: rungs 0, 1, 2**, two tasks
  each, so 6 tasks plus the emoticon reference, which is where the earlier "about ten" came from. Per (trigger, rung) the
  word-like target closest to the rung's log-centre is the candidate; per rung the candidates with the highest and the
  lowest coverage are the two tasks.
- **Windows.** 41 tokens, trigger at index 20, up to 500 per task, split by source row (every 5th row held out), round
  1's format and loss (CE at the trigger + 1.0 × KL to W₀ elsewhere in the window).
- **Coverage** (the x-axis; fixed before training, `src/commute/ladder.py`). With G_k the gradient of the target's
  cross-entropy at the trigger w.r.t. W_k over the training windows, the pull-back moves alive component c by
  (G v_c, Gᵀ u_c), energy e_c = ‖G v_c‖² + ‖Gᵀ u_c‖². Coverage = Σ_c ḡ_c e_c / Σ_c e_c pooled over the 24 matrices, with
  ḡ_c the mean of Γ's g_c at the trigger over the training windows: the share of the first pull-back step's energy that
  sits in components Γ gates on at the trigger. Reported only: L0 at the trigger (alive c with ḡ_c > 0.5) and max ḡ_c.
- **Arms per task.** (a) W fine-tune (round 1's recipe, lr 3e-5, 30 steps, checkpoints 1, 3, 10, 30), for the
  do-nothing reference. (b) Pull-back, Adam lr 1e-5 on all alive (U, V), 30 steps, states at 10 and 30 written to
  `results/large/<task>/pullback/`. (c) **Gated-off pull-back**, the positive control: the same with only the alive
  components whose ḡ_c ≤ 0.01 at the trigger trainable, i.e. the update is placed where Γ is not looking.
- **Scored step:** 10 (round 2's), on the held-out windows. Step 30 reported.
- **Dropped from scope.** A per-component ledger (which components move, does Γ gate them): the identity test showed the
  Adam pull-back spreads its update over thousands of components, so the functional CI-masked measure at the trigger
  *is* the Γ comparison. SGD and sparse pull-backs are the concentration test's (`reports/2026-10-15_plan_concentration-test.md`).

## Measurement

**Task-local harness** (`local_faith.local_faithfulness`): on the held-out windows, for the variants target, CI-masked
(Δ out), unmasked (Δ out), stochastic (the authors' sampling, 4 draws), PGD with the authors' objective (mean KL over the
window) and **PGD with the trigger as its objective** (new), each: KL to the target at the trigger, mean KL at the other
40 positions, and log-odds of the target token at the trigger.

**Scored quantity.** *retained_v* = (lo_v(state) − lo_v(W₀)) / (lo_target(state) − lo_target(W₀)) at the trigger: the
share of the target model's log-odds gain that variant v keeps, each variant measured against its own value at W₀ (a
masked model already differs from the target at W₀; CI-masked KL is 0.35 nats there). This replaces the "ratio to W₀"
language for task-local metrics; ratios stay for the Pile side-effect check.

**Blind to.**
- *retained* (trigger-local): damage elsewhere in the window and outside it. Cover: kl_other in the same harness, and
  the Pile ratios (round 2's harness, 2 blocks of 128 val rows) for the pull-back at the high-coverage tasks of rungs 0
  and 2.
- *PGD*: per-window attacks; one adversary is shared across all held-out windows.
- *CI-masked and unmasked*: deterministic, so no seed spread. Their discriminating controls are G1 and G1b, not
  do-nothing: with Δ excluded, do-nothing's unmasked variant is *identically* the W₀ masked model (retained exactly 0 by
  construction; `tests/test_local_faith.py` shows this), and its CI-masked variant differs only through Γ reading the
  fine-tuned model's activations, so it is near zero. Do-nothing is still reported, and is informative for the
  Δ-including variants. **This applies to any trigger-local criterion that uses do-nothing as its positive control.**
- *coverage*: a first-step quantity. Whether the trained update stays where the gradient started is what P1 tests.

**Gates (before anything is scored).**
- **G0, noise.** Pile ratios on W₀ for harness seeds 0, 1, 2 (spread reported). Local retained on the emoticon pull-back
  (optimiser seed 0, step 10) for harness seeds 0, 1, 2 (spread per variant), and a second optimiser seed:
  d_opt = |retained_ci(seed 0) − retained_ci(seed 1)|. The P1 margin is m = max(0.15, 3 d_opt). If m > 0.5 the design
  has no power and P1 is unscorable.
- **G1, positive control (low end).** For every task, the gated-off pull-back at step 10: if it learns (held-out
  P(target) ≥ 0.5), then retained_ci_masked ≤ 0.2 **and** retained_unmasked ≥ 0.7 are required. A failure means the
  harness does not separate placement, and P1/P2 are not scored. If it does not learn, that task is inconclusive and
  excluded.
- **G1b, positive control (high end).** In the noise stage, on the emoticon task: a **gated-on** pull-back (trainable set
  = alive components with ḡ_c ≥ 0.5 at the trigger) at step 10. If it learns, retained_ci_masked ≥ 0.7 × retained_unmasked
  is required: the CI-masked metric must be able to read high when the update is placed where Γ looks. A failure blocks
  P1/P2. If it does not learn, G1b is inconclusive and P1/P2 carry that caveat. Without G1b, the likely outcome "every
  task's CI-masked retained is low, emoticon included" (the Adam pull-back spreads over thousands of mostly gated-off
  components) could not be told from a metric that cannot read high.
- **G2, design power.** A rung whose two tasks differ in coverage by < 0.15 is excluded from P1.

## Predictions (thresholds fixed; filed by `run.py --predictions`)

- **P1, coverage effect.** Within each scorable rung, retained_ci_masked(pull-back, step 10) of the high-coverage task
  exceeds the low-coverage task's by ≥ m. Pass: ≥ 2 of 3 rungs. Null: ≤ 1 rung, or the sign reversed in 2 or more.
  **Predicted: pass.**
- **P2, rarity does not matter where covered.** Every scorable rung's high-coverage task retains ≥ 0.8 × the emoticon
  task's retained_ci_masked. Null: at least one falls below; the first rung that does is reported as the limit.
  **Predicted: pass for rungs 0 and 1, uncertain for 2 (filed as pass).**
- **Reported, not scored.** Curves (P(target), kl_other, relative norm) per arm; do-nothing under all variants; step
  30; stochastic and both PGD retained for every state; Pile ratios at rungs 0 and 2; the window-context baseline
  beside the mined one; L0 and max ḡ at the trigger.

## What would change next

- P1 passes, P2 passes: Γ's coverage tells you in advance which updates inside training can follow. Rungs 3 and 4
  (rare and never) go next, with coverage as the design variable, and Γ updating becomes the lever where coverage is low.
- P1 passes, P2 fails at rung r: coverage matters but so does rarity; rung r is the limit of frozen-Γ inside training.
- P1 null (coverage does not separate): the x-axis is wrong or the gradient does not stay where it started; the
  concentration test's constrained pull-backs become the way to place updates on purpose.
- G1 fails: the CI-masked metric does not read placement; the harness needs rework before any rung is interpreted.
- Every task's CI-masked retained is low, emoticon included, while G1b reads high: frozen Γ never gates the diffuse
  Adam update in, anywhere. Γ updating (or a constrained pull-back, the concentration test's arms) is round 4.
- Everything retains ≈ 1: inside training is stronger than the question assumes, and the strong version is next.

## Compute and how to run (for the running session)

About 1.75 h on one L40 in total, as three chained `debug` jobs (one stage each, so an overrun never loses another
stage); no cash. Steps:
1. `cluster/sync.sh up`, then `uv run python experiments/task_ladder/run.py --predictions` on the cluster copy (or locally
   and sync); the file must exist before any stage. *(2026-10-16: both `predictions.json` and `predictions_amend1.json`
   are filed locally; the amendment adds a G1b arm and a route split per state, a few minutes on top of the estimate.)*
2. Smoke first on a GPU node: `run.py all --smoke` (64 rows, 2 steps, writes `*_smoke.json`). It must finish without
   error and `noise_smoke.json`, `tasks_smoke.json`, `ladder_smoke.json` must exist.
3. Chain the stages (the sbatch header has the exact lines): `J1=$(sbatch --parsable cluster/task_ladder.sbatch noise)`,
   then `mine` with `--dependency=afterok:$J1`, then `ladder` with `--dependency=afterok:$J2`. Record all three job IDs
   (`cluster/logs/jobs.log` gets them automatically) in the analysis report. Expected: noise ~30 min, mine ~15 min,
   ladder ~60 min.
4. `cluster/sync.sh down`; `uv run python experiments/task_ladder/plot.py`; write
   `reports/2026-10-15_analysis_task-ladder.md` scoring P1 and P2 against `results/predictions.json`, with a **Not run**
   list at the end (at least: rungs 3 and 4, LoRA through the pull-back, SGD/sparse pull-back, Γ update); update
   `experiments/TRIED.md`, `README.md`, `backlog.md`.
Inputs assumed on the cluster: `experiments/transport_test/results/large/{alive.pt, windows.pt}`, `results/windows.json`,
and the cached Pile rows (train 0..32767, val 0..255).

## Deferred (in `backlog.md`)

Rungs 3 and 4; LoRA through the pull-back, definitions (i) same matrix and (ii) same rank; Γ updating; the strong
version; a per-window PGD adversary.

## Note for the concentration test (running in parallel)

Its plan uses do-nothing as the positive control for the trigger-local CI-masked criterion ("CI-masked P(o) < 0.5 ×
target"). With Δ excluded that control fails by construction whenever the target has learned the behaviour (see "Blind
to" above and `tests/test_local_faith.py::test_retained_do_nothing_is_near_zero_and_exact_lift_is_one`), so it does not
show that the criterion can separate placement. The placement controls here (G1 gated-off, G1b gated-on) are the
replacement; `local_faith.local_faithfulness` and `pullback.train` are importable for it.
