# Plan: SGD and low-rate Adam component training on the key-to-value tasks (sgd_kv)

Design check: item 4 of the SGD-finding review, approved by the user 2026-10-21 to be written ahead; falls under the
GOALS optimiser subgoal. Held fixed: E10's three key-to-value tasks, steps, loss, Pile harness, L40. Varied: optimiser
and rate. Decision tree: some new arm learns rand32 and passes → E10's "new capability" failure was an Adam/rate
artefact; only inconclusive readings → rerun those states with more attacker seeds or on an H100; every learned new arm
fails → the failure holds across optimisers and rates on this task.

**Request, restated in the Terms' words.** Item 4 of `reports/reviews/2026-10-21_review_sgd-finding-influence.md`. E10
(`experiments/kv_task`, `reports/analyses/2026-10-19_analysis_key-value-task.md`) ran component training with Adam at 1e-5 on
three key-to-value tasks: rand1, ident32 and rand32. On rand32, a behaviour the model lacks, the trained state failed
the adversarial Pile check at 2.81. That is the project's only evidence that maps with Γ frozen fail on a new
capability. Run component training on the same three tasks with plain SGD, rate chosen per task by E4's rule from 0.003,
0.01, 0.03 and 0.1. Put Adam at E10's rate divided by 3 and by 10 beside it, so the optimiser is not confused with step
size. Score every state on the Pile harness against the same job's W₀. Read E10's saved Adam states again in the same
job.

## Run

`experiments/sgd_kv/run.py --shard N` through `cluster/sgd_kv.sbatch`: cluster `debug`, 30-min limit, one L40. Three
shards, one task each: 0 rand32, 1 ident32, 2 rand1. Each writes `results/sgd_kv_shardN.json`, and `plot.py` merges
them into `results/sgd_kv.json` and `figures/kv.png`. Each shard works in this order and dumps after every stage:

1. **W₀ first.** The Pile harness under attacker seeds 0, 1 and 2. The shard stops unless
   `commute.faith_score.w0_check` passes: adversarial KL within 10% of 0.69, CI-masked KL within 2% of 0.348,
   adversarial floor at most 0.25. E10 read 0.692 and 0.348 on the same card.
2. **The references, held out.** The W fine-tune is regenerated (lr 3e-5, 300 steps). It sets the rate rule's limit.
   E10's Adam 1e-5 state at its scored step is loaded from
   `/mnt/nw/home/e.agnew/commute/experiments/kv_task/results/large/<task>/pullback/state_step<n>.pt`: step 30 on
   rand32 and ident32, step 10 on rand1 (present on NFS, checked 2026-10-21). If a file is absent, the arm is
   regenerated in-job with E10's recipe (36 s). The fine-tune must reach its scored step at E10's step. Both must
   match E10's held-out P(target) and KL elsewhere to 0.01, or the shard stops.
3. **Training, the cheap stage.** SGD at each of the four rates, then Adam at 3.33e-6 and 1e-6, each to 300 steps.
   Every arm is scored at its first checkpoint (10, 30, 100, 300) where held-out P(target) reaches 0.5, which is E10's
   rule. E4's rule keeps the SGD rates that reach it with KL elsewhere at most twice the fine-tune's, and picks the one
   with the least KL elsewhere. The kept states are saved as bf16 offsets to `results/large/<task>/`.
4. **The control's Pile reading.** E10's Adam 1e-5 state goes through `harness.measure`. Its adversarial ratio must
   match E10's to 25%, or the shard stops before any new arm is measured. On rand32 this makes it the harness's
   positive control: it must read "fail" again.
5. **The new arms.** The kept SGD state and the two Adam states go through `harness.measure`.

The scored run is its own smoke (rule of 2026-10-21). Shard 0 goes first and alone, because its control is the one
that must read "fail". The other two follow once it is past stage 4.

**What differs from E10's end state.** The fine-tune is regenerated rather than loaded; G1 tests that. The Pile
harness is `harness.measure` (`pile_faithfulness`, 20 attack steps) where E10 called `faithfulness` per block; both read
W₀ at 0.69, and C1 tests the match on a real state. E10's least-squares split control, task-local harness and sparsity loss are not
rerun.

New shared code: `src/commute/arms.py` holds E10's scored-step rule, E4's rate rule, the held-out reading of a saved
state and the bf16 offset save. Before this they lived only inside `kv_task`, `sgd_grid` and `sgd_ladder`'s `run.py`,
which stay as written.

## Measurements and expected values

| Number | W₀ | Fine-tune (E10) | Adam 1e-5 (E10) | New arms | Blind to |
|---|---|---|---|---|---|
| Adversarial Pile ratio | 1 | not read | rand32 **2.81**, ident32 1.19, rand1 1.20 | open on rand32 (Q1) | the trained behaviour |
| Stochastic Pile ratio | 1 | not read | 1.24, 1.01, 1.03 | near 1 | the trained behaviour |
| KL elsewhere (nats) | 0 | 0.696, 0.189, 0.040 | 0.355, 0.069, 0.069 | SGD below Adam 1e-5 (P3) | Pile text |
| Held-out P(target) | below 0.002 | 0.585, 0.960, 0.998 at step 10 | 0.847, 0.971, 1.000 at steps 30, 30, 10 | at least 0.5 | everything off the key |
| Route shares (reported) | undefined | not read | E10's values | reported | everything off the key |

Tasks are in the order rand32, ident32, rand1. The rate rule's limits are therefore about 1.39, 0.38 and 0.08 nats.
Adam at 1e-6 may not reach 0.5 by step 300 on the 32-key tasks.

**Noise floor.** Each reading gets its own same-shard W₀ floor f: the stochastic ratio uses the stochastic floor, the
adversarial ratio the PGD floor (`faith_score.verdict`). A ratio is "pass" at or below 1.5/(1+f), "fail" at or above
1.5(1+f), and "inconclusive" between. On the L40 the PGD floor is about 0.21, which puts the band at about 1.24 to 1.81.
E10's passing states (1.19, 1.20) sit just under its lower edge. Each state has one batch-order seed.

## Predictions (filed in `experiments/sgd_kv/results/predictions.json`)

- **Q1, two-sided, no direction filed.** On rand32, does some new arm (the selected SGD rate, Adam 3.33e-6 or Adam
  1e-6) learn and read "pass"? Yes → the "new capability" evidence was an Adam/rate artefact. Only "inconclusive" →
  rerun first. No → it holds across optimisers and rates.
- **P2, predicted pass.** The rate rule finds an SGD rate on every task. A failure is reported with the grid's range.
- **P3, predicted pass.** Wherever SGD learned, its KL elsewhere is below E10's Adam 1e-5 state's. The two are not
  matched for size of change, and they may be read at different checkpoints.

## Runtime and cost

Per shard on the L40, from E10 job 22374 (one Pile pass about 2.1 min, 300 component-training steps 36 s, a 300-step
fine-tune 11 s) and E11 job 25863 (a Pile pass up to 2.4 min):

- load, bases and route cells: 1.5 min
- W₀ under three seeds: 6.3 to 7.2 min
- fine-tune and loading the Adam state: 0.5 min
- six 300-step trainings: 3.6 min (plus 0.6 min if the Adam state has to be regenerated)
- four Pile passes (the control and three new arms): 8.4 to 9.6 min
- route splits and saves: 0.5 min

That is **about 21 to 23 min per shard**, under the 30-min limit, at $0. Two tasks per shard would take about 36 min.

## Not run

- E10's least-squares split control, task-local harness and sparsity loss; the fine-tune's own Pile reading (no prediction reads it).
- The revert-one concentration measure (`attribution.concentration`). It adds about 2.5 min per state, which would
  break the 30-min limit, and `revert_one` takes a single target token, not the per-window table these tasks use.
- Seeds other than batch order 0; more attacker seeds than three.
- SGD with momentum; a size-of-change-matched comparison (item 2).
- No smoke and no local run. Local runs that load the model have been barred since the laptop OOM of 2026-10-21, so
  the code that loads the model has never run. Shard 0 is the first run of it. `make lint` is clean, and the pure
  rule functions in `commute.arms` were checked by hand on toy inputs.
