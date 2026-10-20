# Analysis: reading-A coverage round (the Adam pull-back under VPD's checks on every rung-0–2 task)

*2026-10-19, coding agent. Plan `reports/2026-10-19_plan_reading-a-coverage.md`; predictions filed before the run in
`experiments/reading_a_coverage/results/predictions.json`; results `results/coverage.json` (sha256 24c8fb13…), states (2.3 GB, sha256-checked) in
`/mnt/nw/home/e.agnew/commute/experiments/reading_a_coverage/results/large/` on the cluster NFS (`cluster/sync.sh large` brings them back; the laptop copy was deleted 2026-10-19), log `results/large/logs/scored.log`, code as ran in `results/large/code_as_ran/`. Figures
`experiments/reading_a_coverage/figures/`, notebook `notebooks/reading_a_coverage.ipynb`. RunPod L40S, 38 min, $0.70.*

We regenerated the ten-step pull-back and the plain fine-tune on the five ladder tasks that had no Pile numbers,
checked that they are the ladder's own states, and scored each on the four Pile reconstruction checks and on VPD's
sparsity loss, every number as a ratio to the unchanged model with a noise floor measured on the same card in the same
job. The pull-back fails the adversarial check on three of the five tasks, two of them common-word tasks, the rare-word
task's earlier excess is real but sits at the line, and on four of five tasks leaving the update outside the
decomposition is more robust under the adversary than pulling it in, while the sparsity loss barely moves anywhere and
cannot tell the two placements apart. For the goals file this means that "passes VPD's own checks about as well" is
not a general property of the Adam pull-back even on tasks the model already does: the two earlier passes were the
favourable cases, and what the pull-back buys is reconstruction without the residual, paid for in adversarial fragility.

## What was run

Per task (held-out windows saved by the ladder): the arm-A fine-tune (lr 3e-5, 10 steps, batch 32, δW placed in Δ,
released components and Γ unchanged: the **do-nothing** reference) and the **Adam pull-back** (lr 1e-5, 10 steps,
free zero-initialised offsets on every alive U, V, Δ and Γ frozen; seed = batch order). Tasks `r0_t352_v310`,
`r1_t534_v369`, `r1_t588_v452`, `r2_t275_v2667` at seed 0; `r2_t359_v816` at seeds 0, 1, 2. Each state scored on
round 2's Pile harness (val rows 0..255 in two 128-row blocks, one shared PGD adversary per block, 20 steps of 0.1,
seed 0; stochastic masks, 2 draws; CI-masked and unmasked KL) and on VPD's importance-minimality loss at the released
run's end-of-training setting (p 0.4, β 0.5) on the same rows, plus the same loss on the upper-leaky CI values at the
trigger position of the held-out windows. Everything as a ratio to W₀ on the same rows.

Floors at W₀ (same job, same card): seed floor over adversary seeds 0–2 for the stochastic and PGD ratios, block floor
(block 0 vs block 1) for the deterministic ones, half-split floor at the trigger. "Within the floor" = ratio ≤ 1 + 3f.

| | stochastic | PGD | CI-masked | unmasked | sparsity |
|---|---|---|---|---|---|
| W₀ mean (nats; sparsity: loss) | 0.248 | 0.701 | 0.348 | 0.0120 | 1196 |
| f_seed | 0.0022 | 0.046 | 0 | 0 | 0 |
| f_block | 0.023 | 0.058 | 0.025 | 0.038 | 0.028 |
| line 1 + 3f (filed floor) | 1.006 | 1.138 | 1.074 | 1.114 | 1.085 |

**Gate G1 (are these the ladder's states?) passed.** Every seed-0 pull-back's held-out P(target) and kl_other at step 10
are within 2e-4 and 6e-4 of the ladder's curve; r2_t359 seed 0's Pile PGD ratio reads 1.553 against the ladder's L40
1.532 (1.4%, tolerance 25%); W₀'s PGD KL 0.701 against the L40's 0.691–0.716. The L40S reproduces the L40's numbers.

## Results

Ratios to W₀. "pb" = pull-back, "dn" = do-nothing. kl_other is the held-out off-target KL of that state (nats).

| task | arm | stochastic | PGD | CI-masked | unmasked | sparsity (Pile) | sparsity (trigger) | kl_other |
|---|---|---|---|---|---|---|---|---|
| r0_t352_v310 | pb | 0.975 | **2.73** | 1.010 | 0.958 | 0.997 | 0.863 | 0.201 |
| | dn | 1.001 | 1.58 | 1.023 | 5.44 | 0.992 | 0.835 | 0.105 |
| r1_t534_v369 | pb | 1.001 | **2.24** | 1.050 | 0.974 | 0.996 | 1.083 | 0.232 |
| | dn | 1.010 | 2.17 | 1.011 | 8.42 | 0.988 | 1.150 | 0.122 |
| r1_t588_v452 | pb | 1.037 | 1.157 | 1.043 | 1.026 | 0.996 | 0.976 | 0.146 |
| | dn | 0.980 | 1.224 | 1.000 | 4.11 | 0.993 | 0.930 | 0.083 |
| r2_t275_v2667 | pb | 1.746 | **6.74** | 2.31 | 1.013 | 0.994 | 1.433 | 0.508 |
| | dn | 1.019 | 2.26 | 1.097 | 18.0 | 0.993 | 1.364 | 0.203 |
| r2_t359_v816 s0 | pb | 1.033 | **1.553** | 1.114 | 0.990 | 0.991 | 1.055 | 0.188 |
| | dn | 1.016 | 1.238 | 1.034 | 5.61 | 0.996 | 0.990 | 0.091 |
| r2_t359_v816 s1 | pb | 1.028 | 1.377 | 1.100 | 0.981 | 0.991 | 1.050 | 0.167 |
| r2_t359_v816 s2 | pb | 1.037 | 1.536 | 1.116 | 0.990 | 0.990 | 1.067 | 0.200 |

Trigger half-split floors f_trig: 0.031, 0.034, 0.007, 0.014, 0.0035 (lines 1.093, 1.10, 1.02, 1.043, 1.011).

## Predictions, scored as filed

- **G1 reproduces: pass** (above).
- **P1 rung-0/1 pass (filed pass): FAIL.** Rule: all four KL ratios within the floor and PGD ≤ 1.5 on all three
  common-word tasks. PGD reads 2.73 (r0_t352) and 2.24 (r1_t534), both over the 1.5 line; r1_t588 reads 1.157, over
  the floor line 1.138 by 0.02, and its stochastic ratio 1.037 is over the seed-floor line 1.006. CI-masked and
  unmasked are within the floor on all three. Bears on reading A for tasks the model already does, where the only
  earlier numbers (emoticon; r0_t588 at 1.09) passed.
- **P2 rung-2 over the floor (filed pass): pass.** r2_t275 PGD 6.74 (stochastic 1.75 and CI-masked 2.31 also far over
  their floors). The largest excess of the round, on the task with the largest kl_other.
- **P3 r2_t359's excess is real (filed pass): pass.** All three seeds' PGD ratios (1.553, 1.377, 1.536) are over the
  floor line 1.138. The decision's three-way rule on the 1.5 line reads **split** (two over, one under; mean 1.49;
  state-level seed floor 0.128): the excess over W₀ is real, whether it crosses 1.5 is seed noise. No verdict on the
  line, as the rule says.
- **P4 Pile sparsity within the floor (filed pass): pass.** All twelve states read 0.990–0.997 against a floor band of
  1 ± 0.085. Read with C1.
- **C1 trigger sparsity has power (filed pass): FAIL as a whole.** 5 of 12 states clear their line: r2_t275 (1.43 and
  1.36 against 1.043) and all four r2_t359 states (1.05–1.07 against 1.011). The six rung-0/1 states read 0.86–1.15,
  three below 1. So P4's pass reads "blind, not broken" on the rung-2 tasks only; on rung 0/1 the loss shows no power
  at the trigger either, and P4 says nothing there.
- **P5 Δ absorbing δW costs more under the adversary (filed pass): FAIL.** Rule: do-nothing PGD ratio > pull-back's on
  every task. True on one task of five (r1_t588: 1.224 vs 1.157). On the other four the pull-back is worse, by 1.03×
  (r1_t534) to 3.0× (r2_t275). Round 2's emoticon numbers (do-nothing 2.18 vs pull-back 1.03) do not generalise.

Three of five predictions and the control missed. The misses are in one direction: the pull-back's components are
less robust under the mask adversary than filed, on common-word tasks and against the do-nothing reference alike.

## What this means for GOALS and reading A

Reading A (`decisions.md` 2026-10-19): D′ passes VPD's own checks on W + δW about as well as D on W, with Δ not
absorbing δW. Now checked on seven tasks for the Adam pull-back at step 10:

| task | PGD ratio | verdict on the adversarial check |
|---|---|---|
| emoticon (round 2: 1.03; rerun on this harness 2026-10-20: 1.031) | 1.03 | holds |
| r0_t588 (ladder) | 1.09 | holds |
| r1_t588 | 1.16 | holds on the 1.5 line, 0.02 over the floor |
| r2_t359 | 1.38–1.55 (3 seeds) | at the line |
| r1_t534 | 2.24 | fails |
| r0_t352 | 2.73 | fails |
| r2_t275 | 6.74 | fails |

The non-adversarial checks (stochastic, CI-masked, unmasked) hold within the floor everywhere except r2_t275; as in
every earlier round, they are blind to what the adversary finds. "Δ not absorbing δW" is what the pull-back does buy:
its unmasked ratio is 0.96–1.03 on every task where the do-nothing reads 4–18, because δW sits in Δ there. But the
components that reconstruct W + δW without Δ are the more fragile ones on four of five tasks. So the weak version's
"about as well" does not hold for the Adam pull-back in general, and the rung does not sort the tasks: a rung-0 task
fails harder than a rung-2 one. The success-criterion decision's "holds on emoticon on every check run, checked once
on a common-word task" is now "holds on three of seven, at the line on one, fails on three".

## Patterns (reported, not causes)

1. **The PGD excess orders with the pull-back's off-target KL.** kl_other → PGD ratio: 0.146 → 1.16, 0.167 → 1.38,
   0.188 → 1.55, 0.200 → 1.54, 0.201 → 2.73, 0.232 → 2.24, 0.508 → 6.74. Monotone except for r0_t352, which is the
   outlier in both directions. The ladder matched tasks by steps, not by collateral change.
2. **The pull-back's kl_other is 1.6–2.5× the fine-tune's at the same step on every task** (0.20 vs 0.10, 0.23 vs
   0.12, 0.15 vs 0.08, 0.51 vs 0.20, 0.19 vs 0.09): it reaches the same P(target) with about twice the off-target
   change. This was visible in the ladder's curves and never scored.
3. **The fine-tune alone breaks the line on two tasks.** The do-nothing PGD ratio is 2.17 (r1_t534) and 2.26 (r2_t275)
   with δW only in Δ, at ‖δW‖/‖W₀‖ ≈ 0.012–0.013 per matrix. On those tasks an update of this size is adversarially
   fragile under either placement; the pull-back adds to it.
4. **Block spread at the pull-back states is far above W₀'s.** r0_t352's two blocks read 1.33 and 2.50 nats (W₀:
   0.68, 0.72); r2_t359 seed 0 reads 0.92 and 1.26. The W₀ floor understates the uncertainty of a state's ratio; the
   one state-level floor measured (r2_t359, seeds) is 0.128. A ratio of 2.7 is unambiguous, its second digit is not.

Candidate separations for a later round, through `/design`: matched kl_other instead of matched steps (pattern 1 and
2); the pull-back at the step where its kl_other equals the fine-tune's step-10 value; whether the excess sits in the
subcomponents Γ switches on at the trigger (the ladder's e_gate) or elsewhere. None of these is asserted here.

## Caveats

- The stochastic ratio's filed floor is the seed floor, 0.0022 (two draws agree to 0.2%); the block floor is 0.023.
  Under the block floor r1_t588's 1.037 would pass. The filed rule stands; P1 fails on PGD either way.
- The sparsity loss, with Γ frozen, follows the weights W′ only (the key-value round found do-nothing and lift
  identical; here the two arms differ only because their W′ differ). Its Pile ratio moves by under 1% for every state
  and at the trigger both arms move together. It is reported-only until a round updates Γ.
- The smoke (2 steps, 16 rows, 2 PGD steps) failed G1 as expected and was not scored; its step-2 state files were
  deleted from `results/large/`.
- Card: L40S against the ladder's L40; the W₀ PGD and the r2_t359 reproduction agree to 1.4%, so no card correction.

## Not run

Task-local harness and route split on these states (the ladder's numbers stand, tied by G1); r0_t588 (the ladder's
1.09 was not rerun); emoticon; rungs 3–4; the lift; any Γ update; seed repeats on the four single-seed tasks;
state-level block or seed floors other than r2_t359's; the matched-kl_other comparison; PGD with more rows or more
adversary steps.

## Hand-off

Start a new session for whatever follows. Read this report, `reports/2026-10-19_decision_success-criterion.md`, and
`reports/2026-10-19_analysis_key-value-task.md` (whose rand32 PGD 2.81 is the same kind of miss). The open design
question is which of the three candidate separations above is worth a round; it needs `/design` and the user.

## Addendum 2026-10-20: the emoticon task on the same harness

*Added at the user's request: the five tasks above were new, so the reference task is rerun on the same harness in one
cluster job, with its own same-job W₀ floors. Code: `run.py --task emoticon` (loads round 1's emoticon windows; the
measurement code is unchanged). Written before the run.*

**Expected, from round 2 (other Pile rows, another card) and the ladder:** gate G1 holds against the ladder's emoticon
curve at step 10; pull-back PGD ratio near 1.03 and do-nothing near 2.18; pull-back unmasked near 1 and do-nothing far
above 1; stochastic and CI-masked within the floor for both; Pile sparsity within 1 ± 3 f_block; trigger sparsity above
its half-split line (the concentration test counted 150–200 subcomponents switched on at the emoticon trigger). The
filed P5 rule (do-nothing PGD > pull-back PGD) is expected to hold here, as it did in round 2.

**Run 2026-10-20.** MATS cluster L40, job 25863 (smoke job 25724), 12.5 min, free. Results
`results/coverage_emoticon.json` (sha256 45e7a747…), log `cluster/logs/commute-reading-a-coverage-25863.out`, states
on the cluster NFS under `results/large/emoticon/`. Gate G1: pull-back held-out P(target) and kl_other at step 10
within 2e-5 and 2e-4 of the ladder's emoticon curve; W₀ PGD KL 0.692 against the five-task job's 0.701.

| | stochastic | PGD | CI-masked | unmasked | sparsity (Pile) | sparsity (trigger) | kl_other |
|---|---|---|---|---|---|---|---|
| W₀ mean (nats; sparsity: loss) | 0.248 | 0.692 | 0.348 | 0.0120 | 1196 | 892 | |
| this job's line 1 + 3f | 1.006 | **1.626** | 1.073 | 1.118 | 1.085 | 1.205 | |
| emoticon pull-back | 0.999 | 1.031 | 1.002 | 1.006 | 0.998 | 1.446 | 0.016 |
| emoticon do-nothing | 1.001 | 2.183 | 0.997 | 2.84 | 0.998 | 1.390 | 0.055 |

Every expected reading held: the pull-back is within every floor and the do-nothing fails the adversary at 2.18
(PGD blocks 0.773 and 2.247 nats) and the unmasked check at 2.84; the P5 rule holds; the trigger-sparsity control
fires for both placements, 1.45 and 1.39 against 1.205 (half-split floor 0.068, 36 windows per half). The round-2
numbers (1.03, 2.18) are reproduced to the second digit on different Pile rows and a different card.

**One new observation, reported not scored.** This job's W₀ PGD seed floor is 0.209 (line 1.626), against 0.046 in
the five-task job: adversary seed 1 found 1.001 nats on block 1 where the other five W₀ block-runs read 0.667–0.725
(five-task job, same seed and block: 0.798). The unchanged model's own adversary therefore has a tail that reaches a
ratio of 1.45 on a single block, and round 2's 1.5 line sits inside this job's floor band. It does not change any
verdict here (emoticon's pull-back 1.03 is inside every band, the do-nothing 2.18 outside every band), but a ratio
near 1.5 on three adversary seeds (r2_t359's 1.38–1.55) is less decided than the five-task floor suggested. The
seed floor should be measured with more seeds before another round reads a verdict off the 1.5 line.

Notebook updated (`notebooks/reading_a_coverage.ipynb`): emoticon joins every figure as the reference task, and the
first figure shows the stochastic check beside the adversarial one (user request 2026-10-20).
