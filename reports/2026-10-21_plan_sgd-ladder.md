# Plan: SGD and low-rate Adam component training on the ladder tasks at step 10 (sgd_ladder)

Design check: item 3a of the SGD-finding review; approved by the user 2026-10-21 to run after item 1's control passes;
falls under the GOALS priority subgoal (SGD vs Adam component training). Held fixed: the five ladder tasks + emoticon,
step 10, loss, Pile harness, card (L40). Varied: optimiser and rate. Decision tree: any failing task passes → 3b (steps
100, 1000, H100 when available); none passes → the frozen-Γ failure holds across optimisers, re-gating stays main line.

**Request, restated in the Terms' words.** Item 3a of `reports/2026-10-21_review_sgd-finding-influence.md`. Run
component training on emoticon and on the five ladder tasks E11 scored, to step 10, using plain SGD. Pick the rate per
task by E4's rule from 0.003, 0.01, 0.03 and 0.1. Put Adam at 1e-6 and 3e-6 beside it, so a change of optimiser is not
mistaken for a change of step size. Score every state on the Pile harness against the same job's W₀. Read E11's saved
Adam 1e-5 states and the do-nothing fine-tune again in the same job.

## Run

`experiments/sgd_ladder/run.py --shard N` through `cluster/sgd_ladder.sbatch`: cluster `debug`, 30-min limit, one L40.
RunPod is down. There are six shards, one task each (0 emoticon, 1 r0_t352, 2 r1_t534, 3 r1_t588, 4 r2_t275,
5 r2_t359). Each shard writes `results/sgd_ladder_shardN.json`, and `plot.py` merges them. Each shard works in this order,
dumping after every stage:

1. **W₀ first.** The Pile harness under attacker seeds 0, 1 and 2. The shard stops unless all of these hold
   (`commute.faith_score.w0_check`): W₀'s adversarial KL is within 10% of 0.69, its CI-masked KL within 2% of 0.348,
   and the adversarial floor at most 0.25. Past L40 readings were 0.692, 0.348 and a floor of 0.209 (E11 job 25863).
2. **The references, held out.** The do-nothing fine-tune is regenerated (lr 3e-5, 10 steps). E11's Adam 1e-5 state
   is loaded from `reading_a_coverage/results/large/<task>/pullback_s0/state_step10.pt`. Both must match E11's held-out
   P(target) and KL elsewhere to 0.01, or the shard stops.
3. **Training, the cheap stage.** SGD runs at each rate. E4's rule keeps the rates whose held-out P(target) is at least
   0.9 and whose KL elsewhere is at most twice the do-nothing's on that task (0.110 on emoticon, as in E4). Among those
   it picks the rate with the least KL elsewhere, not the smallest rate. Then Adam runs at 1e-6 and at 3e-6. The kept
   states are saved as bf16 offsets (dU, dV) to `results/large/<task>/` for E17's cancellation statistics.
4. **The control's Pile reading.** The do-nothing and the Adam 1e-5 state go through `harness.measure`, which gives the
   Pile ratios and the route split. Adam's adversarial ratio must match E11's to 25%, or the shard stops before any new
   arm is measured. On the three tasks where E11 found a failure, this makes it the harness's positive control: it
   must read "fail" again.
5. **The new arms.** The kept SGD state and the two Adam states go through `harness.measure`.

The scored run is its own smoke (rule of 2026-10-21): the stages run cheapest first, and each is written to disk as it
finishes, so a code error stops the job early. Shard 0 (emoticon) goes first and alone.

What differs from E11's end state. The do-nothing fine-tune is regenerated rather than loaded; E11's own gate showed
that regeneration reproduces it to 4e-4. The card is an L40, where the five-task E11 job ran on an L40S. G1 tests both.

## Measurements and expected values

| Number | W₀ | Do-nothing (E11) | Adam 1e-5 (E11) | New arms | Blind to |
|---|---|---|---|---|---|
| Adversarial Pile ratio | 1 | 2.18, 1.58, 2.17, 1.22, 2.26, 1.24 | 1.03, **2.73, 2.24**, 1.16, **6.74**, 1.55 | open (Q1) | the trigger |
| Stochastic Pile ratio | 1 | 0.98 to 1.02 | 0.98 to 1.04, 1.75 on r2_t275 | near 1 | the trigger |
| KL elsewhere (nats) | 0 | 0.055, 0.105, 0.122, 0.083, 0.203, 0.091 | 0.016, 0.200, 0.232, 0.146, 0.508, 0.188 | SGD below Adam 1e-5 (P3) | Pile text |
| Route shares d, r (reported) | undefined | e_delta near 1 | ladder's step-10 values | reported | off the trigger |

Tasks are in shard order: emoticon, r0_t352, r1_t534, r1_t588, r2_t275, r2_t359. For emoticon at step 10, SGD at 0.01
should be selected (sgd_grid: P 0.980, KL elsewhere 0.011). Adam at 1e-6 should not learn by step 10.

**Noise floor.** The same-shard W₀ adversarial floor is f. An adversarial ratio scores "pass" at or below 1.5/(1+f) and
"fail" at or above 1.5(1+f); between the two it is "inconclusive" (`faith_score.verdict`). On the L40, f is about 0.21,
so the band runs from about 1.24 to 1.81, and a verdict within about 0.3 of 1.5 decides nothing on this card. Each state
has one batch-order seed. E11's state-level spread on r2_t359 was 0.128.

## Predictions (filed in `experiments/sgd_ladder/results/predictions.json`)

- **Q1, two-sided, no direction filed.** Does some new arm (the selected SGD, Adam 1e-6 or Adam 3e-6) learn and read
  "pass" on r0_t352, r1_t534 or r2_t275? Yes → 3b. Only "inconclusive" → rerun those states with more attacker seeds or
  on an H100 before deciding 3b. No → the decision tree's second branch.
- **P2, predicted pass.** The rate rule finds an SGD rate on every task. A failure is reported with the grid's range,
  never as "no rate works".
- **P3, predicted pass.** Wherever SGD learned, its KL elsewhere is below the re-read Adam 1e-5 state's. The two are not
  matched for size of change.

## Runtime and cost

Per shard on the L40, from one Pile pass taking 2.4 min (E11 job 25863: 70 s per block) and sgd_grid's training speed
(job 28106: six 30-step runs in about 3 min):

- load and bases: 1.5 min
- W₀ under three seeds: 7.3 min
- the control's fine-tune and its two Pile passes: 5 min
- six 10-step trainings: 1 min
- three Pile passes: 7.2 min
- route splits and saves: 0.5 min

That is **about 22 min per shard**, under the 25-min target and the 30-min limit, at $0. Two tasks per shard would take
about 35 min, which is why each shard holds one task. If the six debug jobs run one after another, the round takes about
2.2 h of wall time.

## Not run

- Steps 100 and 1000 (item 3b, which follows only on the decision tree's first branch).
- Seeds other than batch order 0; more attacker seeds than three.
- SGD with momentum, and matched size of change (item 2).
- The trigger-local harness and the revert-one concentration measure (`attribution.concentration`).
- The key-value tasks (item 4).
- No smoke and no local run. Local runs that load the model have been barred since the laptop OOM of 2026-10-21, so the
  code that loads the model has never run. Shard 0 is the first run of it.
