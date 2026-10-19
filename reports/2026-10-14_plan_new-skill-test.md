# Plan: the new-skill test. Does training inside the decomposition still work for a behaviour the model lacks?

> **Shelved 2026-10-15 (user).** Not ready: the falsifier below (CI-masked ratio on Pile) has no power for a behaviour
> that fires at ~3 of 131k evaluated positions, the pull-back has not been run on a LoRA, and the pull-back's update has
> not been compared with Γ. See `reports/2026-10-15_review_faithfulness-scope.md` and the 2026-10-15 entries in
> `backlog.md`. Needs a new `/design` pass once those are done.

*2026-10-14, coding agent. Round 3. Design check below; awaiting the user's confirmation. Nothing has been filed or
run. Round 2 (`reports/2026-10-14_analysis_lift-test.md`, notebook `notebooks/lift_test_round2.ipynb`) found that
post-training **inside** the decomposition (the pull-back: train the alive U, V with Δ and Γ fixed) stays faithful,
while translating a finished update from **outside** does not. That was shown only for a behaviour the model already
has, over 30 steps. Experiment directory: `experiments/new_skill/`.*

## The claim, the null, and what separates them

**Claim.** Pull-back post-training stays faithful (stochastic, PGD and CI-masked ratios ≤ 1.5) for a behaviour the
model lacks, as it did for one it has.

**Null (new capabilities).** For a new behaviour the pull-back either cannot learn it, or learns it only by breaking
faithfulness. The expected way to break is the **CI-masked** KL: Γ is frozen and never switches on the components that
come to carry the new behaviour, so masking by Γ removes it (theory note §8).

**Predictions** (thresholds proposed here, fixed in `predictions.json` after confirmation):

1. *New skill, inside (pass / fail on the claim).* The pull-back reaches held-out P(target) ≥ 0.9 within 3× the steps
   a normal W fine-tune needs. At that step and at step 30, the stochastic, PGD and CI-masked ratios are ≤ 1.5.
2. *Strengthen, inside, long (the limit).* Continued pretraining on Pile (arm B) via the pull-back for 300 steps stays
   ≤ 1.5 on all three ratios at steps 10, 30, 100 and 300. Otherwise the first step that crosses is reported as the
   limit. Arm B is used because the emoticon task saturates by step 5, while arm B's update keeps growing.

**Falsifier.** Prediction 1 fails on the CI-masked ratio while the strengthen arm passes. That is the
new-capabilities branch, seen directly.

**One control, reported, not scored (placement versus solution).** Apply round 2's Euclidean lift to the pull-back's
*own* δW, from the emoticon run at steps 10 and 30.
- If that is faithful, the outside failure was about *which solution* a normal fine-tune finds (your "many solutions"
  reading).
- If it is PGD-fragile like round 2's lift, the failure is about *how the lift places* the update, whatever the
  solution.

Expected: faithful.

## Set-up

- **New skill.** A trigger → target association inserted into Pile windows, in the same format and with the same loss
  as the emoticon arm:
  - windows of 41 tokens, trigger at index 20;
  - cross-entropy for the target at index 20, plus KL to W₀ on the other 40 positions.

  "Lacks" is fixed before training: the trigger and the target are single tokens, and baseline held-out P(target |
  trigger) < 0.01 at W₀. Reported but not a gate: the largest mean causal importance of any alive subcomponent at
  trigger positions. The choice of tokens is low-level; the rule is fixed.
- **Routes compared.** Inside (the pull-back), and the normal W fine-tune for reference, both with round 2's recipe.
  The pull-back takes learning rates {1e-5, 2e-5, 3e-5}, chosen by amendment 1's rule (P(target) ≥ 0.9 at step 30,
  then the lowest off-target KL).
- **Matching.** Inside and outside are compared at matched ‖δW‖/‖W₀‖, not matched steps: the pull-back moves the
  weights about half as far per step. At matched norm, round 2's gap holds (pull-back PGD 1.07 at 0.0066 against the
  lift's 2.03 at 0.0065).
- **Interpretability, reported, no forward passes.** For each pull-back run:
  - the subcomponents with the largest update;
  - the cosine of each changed u_c and v_c to its original;
  - for the emoticon run, where `2359` ranks (links to round 1's prediction 1b).

## Design check

- **Goal served.** `GOALS.md`: the weak version, via the pull-back map (simple, user 2026-10-14). Also the "if not"
  branch directly: new capabilities against brittleness. The strengthen arm gives the "within some limit".
- **Nearest thing tried.** `TRIED.md` row 2: the pull-back passed on an existing behaviour over 30 steps. Arm C (new
  skill) has been in `backlog.md` since round 1 and was never run. Round 2 cannot say whether its success depended on
  the behaviour already being in the decomposition.
- **Patch or step.** A step. It tests the open branch of the question, not a flaw of round 2. The placement control is
  the one patch-like item, and it is report-only.
- **Decisions.** Relies on: weak version, Γ frozen, pull-back counts as simple, fine-tunes we run ourselves on the 24
  matrices, cluster first.
  - **New:** a synthetic task (trigger → target). That is a new dataset, so it needs your confirmation.
  - No departures.
- **What would change next.**
  - Prediction 1 passes: inside training covers new skills too, in the weak sense. Next is the strong version, or RL as
    a different kind of post-training.
  - Prediction 1 fails on CI-masked while 2 passes: the new-capabilities branch. Next is updating Γ (backlog, round 4).
  - Prediction 2 crosses at step N: that is the limit. Next is whether an occasional Γ refresh extends it.
  - Control fragile: drop "many solutions" as the explanation, and the locality lift becomes the outside-route
    question.

## Deferred (to `backlog.md`)

- **Locality lift:** subsumed by the placement control for now.
- **Γ update:** round 4 if prediction 1 fails as the null says.
- **RL arm:** the LoRA case is not settled, because round 2's LoRA read side was still at its random init.
- **Strong version:** waits on the weak version.

## Cost and compute

- Training is minutes.
- Faithfulness evaluation dominates at about 4.5 min each on the L40. About 12 evaluations:
  - calibration;
  - the new skill: three pull-back checkpoints and one W reference;
  - strengthen: four checkpoints;
  - control: two.

  That is about 55 min, plus about 10 min of training. It runs as two `debug` jobs (under 2 h each) on the MATS
  cluster (account `c11`) at no cash cost, driven by a forked subagent.
