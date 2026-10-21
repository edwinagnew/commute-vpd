# Plan: a Γ refit protocol whose control stays at the floor (design check, awaiting the user's confirmation)

*2026-10-21, coding agent. Status: `/design` check confirmed by the user ("ok for it", 2026-10-21); built and launched the same day (§ Built). Follows the E15 control
(`reports/2026-10-20_analysis_gamma-refit.md`): the refit as filed takes the unchanged model off the released
decomposition's own adversarial floor, so no refit arm can be read until the protocol is fixed. Vocabulary: `GOALS.md` →
Terms.*

## The problem, in the Terms' words

E15's refit holds U, V and Δ of a state fixed and trains every parameter of Γ on the released objective minus its
faithfulness term, warm-started from the released Γ. Two things in that recipe are not the released run's end state.

- **The optimiser.** AdamW restarts at the released *starting* rate, 5e-5, with empty moment estimates; the released
  cosine schedule ended at 5e-6. Adam's first update is a sign step of size lr on every parameter, which is the
  twenty-fold spike of the importance term on step 2 in both runs.
- **The adversary.** The persistent adversary restarts from random sources with two warm-up steps; the released run's
  sources had 400k steps to strengthen. Its loss term rises for the whole refit on both states (0.42 → 0.63 at W₀,
  0.42 → 1.10 on the split), the curve of an adversary still catching up. While it is weak, the importance term is the
  main force on Γ and Γ gets sparser (the control's sparsity loss 0.94 at step 100), which the harness's fresh 20-step
  attacker then exploits (ratio 2.6 at step 100).

The control's numbers: adversarial 2.57 / 3.53 / 6.27 / 2.63 at steps 100 / 300 / 1000 / 3000, stochastic and
CI-masked within 8% of 1, sparsity 0.94 to 1.09. Both of the above are hypotheses about the cause; the test below
decides whether removing them is enough.

## The proposal

Change the refit's protocol in two places, keep everything else, and test it on the unchanged model alone before any
state arm.

1. **Adversary warm-up before Γ moves.** The first K steps of every refit run the full step with Γ's learning rate at
   zero: the persistent adversary takes its steps and AdamW accumulates its moment estimates, but Γ does not move.
   K = 300, fixed (the control's persistent term started rising at about step 300). The persistent term's curve over
   the warm-up is logged; if it is still rising by more than 5% per 100 steps at K, the test is read with that caveat and
   K is raised once, to 1000, in the next run.
2. **The released end-of-schedule rate.** Γ then trains at 5e-6, constant, with the warmed moments. No learning-rate
   ramp: with the moments warm, the first step is lr times a signal-to-noise ratio, not a sign step.

Steps, checkpoints (100 / 300 / 1000 / 3000, counted from the first Γ step), batch, micro-batches, loss stack, seeds and
the harness are unchanged. Dropped: "fewer steps" (the control is off the floor at step 100, so fewer steps is no
fix); a learning-rate ramp (the warmed moments do the same); initialising the adversary from the harness's attacker
(different objects: one source per slot and position against one mask per row and matrix).

## The design check

**Which goal.** E15's: the weak version with Φ allowed to include Γ's response to the new model, and GOALS's flagged
unknown "updating Γ may be essential". This check serves it indirectly: it makes the E15 arms readable. Without it the
Γ-updating line has no result at all.

**Nearest tried.** E15 (`TRIED.md`): the split refit at the released start rate, and its control, which failed. What
this learns that E15 could not: whether a Γ refit can be run as a like-for-like warm start at all. If the control still
drifts with a strong adversary and the end rate, the released Γ is not at rest on the objective as we have reconstructed
it (dropping the faithfulness term is exact for Γ; the annealed p and the coefficients are read from the released
config), and every refit result would be the recipe's.

**Patch or step.** A patch on E15's protocol. It is the patch that decides whether E15 is a step.

**Decisions.** Relies on E15's departure from "Γ frozen" (decisions.md 2026-10-14; for this test only). Inside GOALS's
"not re-deriving VPD's training recipe": both changes restore the released run's own end state rather than tune a new
recipe, which is why they go in together and are not ablated separately. Budget: the test alone fits the $10 session
cap; the split's rerun after it does not, and needs the user's confirmation.

**What result would change what we do next.**

- The control stays at the floor at every checkpoint: the protocol is validated. The split state reruns under it with
  new predictions filed beside the old ones, then do-nothing and component training as the refit's own controls, the
  adjoint split and the task-mixed arm; E15's branches apply as filed.
- The control drifts less, or only late: the protocol is read against its own drift curve (E15's filed null for the
  control) and K is raised once. A second failure shelves Γ updating.
- The control drifts the same: the released Γ is not at rest on this objective. Γ updating is shelved, the cancellation
  tests on the saved split state are next (no GPU), and the log records that "let Γ see the new model" needs
  re-decomposition-scale training, which is the postponed strong version.

**Numbers.** Reported at W₀ only in this run; one card; the floor from 3 adversary seeds as before.

| number | expected if the protocol works | the filed null | control |
|---|---|---|---|
| persistent-adversary loss term over the warm-up (K steps, Γ frozen) | rises from 0.36 (two warm-up steps) and flattens; the harness's 20-step attacker reaches 0.70 nats at W₀, so a plateau at or above about 0.6 | keeps rising past K (adversary not converged; K raised once) | its step-1 value matches E15's 0.36 |
| Pile adversarial ratio to W₀ at 100 / 300 / 1000 / 3000 | ≤ 1 + 3f (1.22 on this card) at every checkpoint | 2.6 / 3.5 / 6.3 / 2.6, E15's control | step 0 reads 1.000 (harness identity, as E15's gate) |
| sparsity loss ÷ W₀'s at the same checkpoints | in [0.99, 1.01] | 0.94 / 1.02 / 1.01 / 1.09 | 1.000 at step 0 |
| importance term on the first Γ step ÷ its warm-up mean | within 5%, no spike | 20× (E15's step 2) | — |
| stochastic and CI-masked ratios | within 0.02 of 1 (reported, not scored: they held in E15 too) | — | — |

Cut: the trigger's switched-on set (W₀ has none), the route split (undefined at W₀), any state arm.

**Cost and route.** The refit step was 2.29 s at the full batch on an H100 (E15's benchmark); the warm-up step is the
same step with lr 0, so K = 300 adds about 12 min. One W₀ refit to 3000 steps with four harness evaluations is about
2.2 h plus 15 min of set-up and prep: about $9.5 at $3.99/h. Early stop: the runner reads the step-100 and step-300
checkpoints from the log and kills the run if the control's adversarial ratio is above 1.22 at both, which caps a
failing protocol at about $3. Over 30 min, so RunPod (H100) via `/pod-runner`, per the compute rule. The cluster is
possible overnight (micro-batch 4 for an L40's 46 GB, about 6 h, `normal` queue days out on 2026-10-19) and not
recommended. The split's rerun under a passing protocol is another $9.5 and a separate confirmation.

## Build (after confirmation; not started)

- `gamma.RefitConfig` gains `warmup: int` (steps at lr 0); `gamma.refit` sets the optimizer's lr to 0 for those steps and
  to `cfg.lr` after; checkpoints count from the first Γ step. About ten lines; no new code path (the warm-up is the same
  `Refit.step`).
- `experiments/gamma_refit/run.py`: a second frozen `Sizes` instance (lr 5e-6, warmup 300, suffix `_v2`) selected by a
  flag; `prep` under the new suffix reuses the states on disk (about 6 min) and benchmarks ten steps of each phase at
  the full batch before the cap is confirmed; `--only w0_P`. Results in `results/gamma_refit_v2.json`; E15's file
  untouched.
- Predictions in `results/predictions_v2.json` from the table above, filed before the run.
- Smoke on the pod, Codex adversarial review on the diff, then the scored run; the runner's early-stop rule in the brief.

## Built (2026-10-21, after the user's confirmation)

- `gamma.RefitConfig.warmup` (default 0) and the loop in `gamma.refit`: loop step i over warmup + steps, Γ's learning
  rate set to 0 for the first `warmup` steps and to `cfg.lr` after, the curve's `step` ≤ 0 during the warm-up,
  checkpoints counted from the first Γ step, log lines `[warmup] step n` / `[refit] step n`. The warm-up is the same
  `Refit.step` (the persistent adversary steps, AdamW's moments accumulate; lr-scaled weight decay is 0 too).
- `run.py`: `Sizes.warmup`; `FULL` (warmup 0, E15 unchanged), `FULL_V2 = replace(FULL, lr=5e-6, warmup=300, suffix="_v2")`,
  `SMOKE` (warmup 1, so the smoke runs the new path); `--v2` selects the instance and `results/predictions_v2.json`
  (filed 2026-10-21: P1_w0_control_v2, W1_adversary_warmed, W2_no_spike, gates G0, G2); the benchmark runs ten steps at
  lr 0 and ten training steps and reports hours per refit. Invocation: `run.py all --v2 --only w0_P`.
- Codex adversarial review (2026-10-21), two findings. (1) The loop never calls the adversary's `update_lr`: moot in
  effect, since the released adversary schedule is constant 0.01 after a 2.5% warm-up, so a fresh PGD optimizer already
  holds the end-of-schedule value; made explicit by an assert in `Refit.__init__`. (2) The benchmark projected six
  refits: now hours per refit. Noted, not changed: W1 and W2 are scored by hand from the JSON, as E15's P1 was (`score()`
  runs only with all six arms).
- Run: RunPod H100 via `/pod-runner` (smoke, then `all --v2 --only w0_P` on the same pod), cap $12, the runner's early
  stop at step 300 if the control is off the floor at both step 100 and step 300. Outputs under the `_v2` suffix;
  E15's files untouched.

## Run (2026-10-21, 10:41 to 11:30 BST, RunPod H100, $3.35)

Smoke clean (105 s). Prep 374 s; benchmark 2.31 s/step, 53.7 GB. The W₀ refit: the persistent-adversary term over the
warm-up 0.36 → 0.44 (step 50) → 0.41 to 0.42 (steps 100 to 300), flat (W1 pass); the first Γ steps' importance term 855
and 819 against about 850 (W2 pass, no spike). Harness at Γ steps 0 / 100 / 300: adversarial 1.00 / 2.06 / 7.76,
stochastic 1.00 / 0.99 / 0.99, CI-masked 1.00 / 0.95 / 0.94, sparsity 1.00 / 0.97 / 0.97. **P1 failed at both
checkpoints**, so the runner's early-stop rule killed the run after step 300, 6 min before the RunPod credits ran out.
The harness attacker's KL on the two 128-row blocks: 0.68 / 0.72 at step 0, 1.43 / 1.45 at 100, 1.49 / 9.35 at 300;
the refit's own attacker 0.42 to 0.43 throughout. Branch taken: "the control drifts the same": refitting Γ alone is
shelved (`decisions.md` 2026-10-21); the split's rerun and the other arms with it. Analysis folded into
`reports/2026-10-20_analysis_gamma-refit.md` § 5 and `notebooks/E15_gamma_refit.ipynb`. Results
`experiments/gamma_refit/results/gamma_refit_v2.json` (`51bb108fe8c0ebd8…`), logs `results/large/full_v2.log`,
`scored_stdout_v2.log`.

## Not run

Steps 1000 and 3000 of the W₀ refit (early-stopped); the split's rerun and the other arms (shelved).
