# Plan: SGD and Adam component training on the ladder tasks at steps 100 and 1000 (sgd_ladder_long)

Design check: item 3b of the SGD-finding review; a branch of the user-approved 3a plan's decision tree (approved class
3); RunPod down, so the cluster's normal QoS. Held fixed: tasks, loss, Pile harness, L40, the rates 3a selected.
Varied: optimiser and step count (10, 100, 1000).

**Why.** 3a (`experiments/sgd_ladder`, `reports/2026-10-21_plan_sgd-ladder.md`) found that at step 10 SGD component
training passes the adversarial Pile check on r0_t352 (0.95; Adam 1e-5 2.70) and r1_t534 (1.10; Adam 2.24), and is
inconclusive on r2_t275 (1.50; Adam 6.83). Adam at 1e-6 and 3e-6 did not learn the ladder tasks in 10 steps. The
decision's first move-back bullet (`reports/2026-10-21_decision_regating-main-line.md`) asks for a map at the floor
with the behaviour at steps 100 and 1000; 3a's tree sends that to this run.

**Request, restated in the Terms' words.** On emoticon (the control) and r0_t352_v310, r1_t534_v369, r2_t275_v2667,
run component training once per arm to step 1000, keeping states at steps 10, 100 and 1000: plain SGD at the rate 3a
selected for the task (0.01, 0.03, 0.03, 0.1), Adam at 1e-5 and Adam at 3e-6. Beside them, the do-nothing placement of
the full fine-tune (Adam 3e-5) at the same steps. Score every state at steps 100 and 1000 on the Pile harness against
the same job's W₀; re-score step 10 of the SGD arm only, as a reproduction of 3a.

## Run

`experiments/sgd_ladder_long/run.py --task T` through `cluster/sgd_ladder_long.sbatch`: cluster `normal` QoS, account
c11, one L40, 8 CPUs, 32 GB, `--time 01:05:00`. One job per task. Emoticon goes first and alone; the three ladder jobs
are submitted only once its G1, C1 and C2 have passed. Each job writes `results/sgd_ladder_long_<task>.json`, dumped
after every stage and every scored state, and saves all nine component-training states as bf16 offsets to
`results/large/<task>/<arm>_step<n>.pt` (`commute.arms.save_offsets`). Stages, cheapest first:

1. **W₀** under attacker seeds 0, 1, 2 (`faith_score.w0_check`): stop unless the adversarial KL is within 10% of 0.69,
   the CI-masked KL within 2% of 0.348, and the adversarial floor at most 0.25.
2. **Training.** The fine-tune to step 1000, then the three arms to step 1000 (`component_training.train`, E11's
   recipe: arm A's loss, batch 32, clip 1, batch-order seed 0, Γ and Δ frozen). No schedule, so step 10 of each run
   is 3a's step 10.
3. **G1.** Every step-10 held-out P(target) and KL elsewhere (do-nothing, SGD, Adam 1e-5, Adam 3e-6) within 0.01 of
   3a's, or stop. Adam 1e-5 is compared with E11's saved state as 3a re-read it.
4. **C1.** The SGD state at step 10 through the Pile harness; its adversarial and stochastic ratios within 25% of 3a's,
   or stop before any scored state.
5. **Scored.** Step 100, then step 1000: SGD, Adam 3e-6, Adam 1e-5, do-nothing through `harness.measure`.

What differs from 3a's end state: the arms are trained afresh rather than loaded (G1 tests it), Adam 1e-5 is trained
here rather than read from E11 (G1 again), and the runs continue past step 10 (the thing varied). Optimiser state and
schedule position at step 10 are identical by construction.

New shared code: `harness.with_windows` (a task's windows on the shared context). The same step is `task_context` in
`experiments/sgd_ladder/run.py` and `experiments/sgd_kv/run.py`; it is not a pure move (theirs read files by task
name), so those stay as they ran.

## Measurements and expected values

| Number | W₀ | Do-nothing | Adam 1e-5 | SGD | Adam 3e-6 | Blind to |
|---|---|---|---|---|---|---|
| Adversarial Pile ratio | 1 | emoticon 1.14 / 1.50 at 100 / 1000 (E16, H100); ladder reported | emoticon pass (E16 1.03, 0.99); ladder fail expected (step 10: 2.70, 2.24, 6.83) | emoticon pass; ladder Q1 | emoticon pass; ladder Q2 | the trigger |
| Stochastic Pile ratio | 1 | near 1 | near 1; 1.75 on r2_t275 at step 10 | near 1 | near 1 | the trigger |
| Held-out P(target) | baseline (emoticon 0.005) | ≥ 0.97 | ≥ 0.96 | ≥ 0.9 at step 10 (3a) | open | off the trigger |
| KL elsewhere (nats), reported against 3a's limit 0.110, 0.209, 0.243, 0.407 | 0 | step 10: 0.055, 0.104, 0.122, 0.203 | step 10: 0.017, 0.20, 0.23, 0.51 | step 10: 0.011, 0.087, 0.115, 0.332 | open | Pile text |
| rel_norm_max (reported) | 0 | step 10 about 0.012 | emoticon 0.008 / 0.016 (E16) | grows from 0.002 to 0.011 at step 10 | open | direction |

**Verdict per state.** Held-out P(target) below 0.9 reads "below P 0.9". Otherwise each Pile reading is scored against
its own same-job floor (`faith_score.verdict`; stochastic against the stochastic floor, adversarial against the
adversarial floor): pass at or below 1.5/(1+f), fail at or above 1.5(1+f), inconclusive between. Either fail → fail;
either inconclusive → inconclusive; else pass. KL elsewhere against the task limit is recorded beside the verdict.

**Noise floor.** On the L40 the adversarial f is about 0.21, so its band runs from about 1.24 to 1.81; the stochastic
f is about 0.002. One batch-order seed per state; E11's state-level adversarial spread on r2_t359 was 0.128.

**Positive control.** The do-nothing at steps 100 and 1000 and Adam 1e-5 on the ladder tasks are expected to read fail;
3a showed on the same card that the harness reads fail on the Adam 1e-5 states (2.70, 2.24, 6.83). Emoticon's Adam
1e-5 at 100 and 1000 is the known pass (C2).

## Predictions (filed in `experiments/sgd_ladder_long/results/predictions.json` before any run)

- **Q1, two-sided, no direction filed.** On r0_t352 and r1_t534, does the SGD state read "pass" at step 100 and at
  step 1000 (four states)? All four pass → the move-back bullet is met and Q1 returns as the first sub-question, taken
  to the user. Any fail or "below P 0.9" → the step-10 pass is a small-update effect; re-gating stays the main line.
  Only inconclusive among the non-passes → rerun those states with more attacker seeds or on an H100 first.
- **Q2, two-sided, no direction filed.** On the three ladder tasks, at the first scored step where Adam 3e-6 reaches
  P(target) ≥ 0.9, what is its verdict? Learns and passes where Adam 1e-5 fails → Adam's ladder failure is a
  step-size effect; learns and fails where SGD passes → the optimiser; does not learn by step 1000 → unanswered at
  this one rate.
- Reported, not scored: r2_t275's states, KL elsewhere against the limit, rel_norm_max, route-split shares and
  trigger ratios, Adam 1e-5 on the ladder tasks.

## Runtime and cost

Per job on the L40, from 3a's log (job 28199: one Pile pass 137 s, load 1.5 min, W₀ under three seeds 6.8 min, ten
component-training steps under 1 s with the per-step held-out eval) and E16 (1000 component-training steps in 78 s on
an H100, taken as about 3.3 min on the L40):

- load and W₀: 8.3 min
- fine-tune to 1000 steps: about 1 min
- three arms to 1000 steps: about 10 min
- nine Pile passes (C1, then four states at each of two steps): 20.6 min
- route splits and saves: about 1 min

About **41 min per job**; `--time 01:05:00` is about 1.5 times that. Four jobs, at $0. With emoticon first and the
three ladder jobs in parallel, about 1.5 h of wall time if the queue is free.

## Not run

- Batch-order seeds other than 0, and more than three attacker seeds.
- r1_t588 and r2_t359 (3a: both optimisers already pass or are split there at step 10).
- SGD at rates other than 3a's selection, SGD with momentum, a matched size of change.
- Step 30 and 300 (states not kept), and E17's cancellation statistics on these states (they are saved for it).
- No local run: the model has never been loaded by this code; the emoticon job is its first run.
