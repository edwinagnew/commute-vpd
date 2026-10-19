# Matched-matrix LoRA: the site was the confound; a lifted LoRA passes the faithfulness half of the weak version at the step it learns, and the fine-tune's lift is fragile on the L40 too

We trained a rank-8 LoRA on all 24 decomposed matrices, the placement LoRAs normally get, and lifted it onto the
released subcomponents beside the full-weight fine-tune, so that for the first time both updates sit on the same
matrices and go through the same map. The LoRA's update is as unreadable to the subcomponents as a random one on the
matrices that are not already fully covered, yet it lifts with less of the behaviour left over than the one-matrix LoRA
did, and its lifted decomposition passes the Pile adversary as well as the released decomposition does, where the
fine-tune's lifted decomposition fails it fivefold. For the goal this says round 4's LoRA failure was the matrix and
not the LoRA, and that a lifted LoRA passes the faithfulness half of the weak version at the step it learns while
still leaving two fifths of the update's energy in the residual, and the comparison with the fine-tune is not yet
like for like, because the LoRA learned at step 2 and the fine-tune's lift was also faithful at its own first steps.

*2026-10-19, coding agent. Design: `reports/2026-10-19_review_lora-failure-and-fixes.md` section 3.2 (confirmed by the
user). Predictions `experiments/lora_matched/results/predictions.json` (filed before the run, unchanged); results
`results/lora_matched.json`; code `experiments/lora_matched/run.py` on the shared `src/commute/harness.py`; figures
`experiments/lora_matched/figures/`; notebook `notebooks/lora_matched.ipynb` (figures and every number, executed). MATS cluster, one L40: smoke job 22368 (85 s), scored job 22371 (18 min), $0.
Codex adversarial review of the new code before the run: approved, no findings.*

**Terms.** *Lift*: the finished weight change written as the minimum-norm change (dU, dV) to each matrix's alive
subcomponents; what no first-order change can express, O(δW) − dU dVᵀ, goes to Δ. *Do-nothing*: δW placed in Δ,
subcomponents untouched. *Inexpressible share*: the share of the update's input-weighted output energy that no
first-order change of the alive subcomponents can express; its *isotropic null* is that share for a random update of
the same matrix; a *saturated* matrix is one where the alive subcomponents span every direction, so the share is 0
whatever the update (four of the 24: `h.0.mlp.c_fc`, `h.3.mlp.c_fc`, `h.0.mlp.down_proj`, `h.3.mlp.down_proj`).
*Pooled* = energy-weighted over the 20 non-saturated matrices (round 4's rule). *Route split*: the held-out log-odds
gain of `o` at the trigger split into d (changed subcomponents at W₀'s CI values), r (changed CI values on the
unchanged subcomponents), i, e_gate (subcomponents Γ masks off) and e_Δ (not in the subcomponents at all). *Pile
ratios*: round 2's harness (2 × 128 Pile val rows; stochastic, CI-masked, PGD KL), each ÷ W₀'s under the same
adversary seed. *Scored step*: the first checkpoint with held-out P(o) ≥ 0.9 (checkpoints 1, 2, 3, 5, 10, 30, 100).

## Per arm (held fixed: windows, Γ, the released decomposition, the lift; varied: the update)

| arm | step | P(o) | kl_other | ‖δW‖/‖W₀‖ max | gain (nats) | inexpressible, pooled (null) | e_Δ | ‖Δ′−Δ₀‖²/‖δW‖² | d | r | Pile PGD ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ft nothing | 10 | 0.999 | 0.055 | 0.011 | 13.4 | 0.268 (0.531) | 1 | 1 | 0 | 0.60 | 2.18 |
| **ft lift** (control) | 10 | 0.999 | 0.055 | 0.011 | 13.4 | 0.268 (0.531) | 0.090 | 0.36 | 0.052 | 0.60 | **5.41** |
| LoRA nothing | 2 | 0.961 | 0.059 | 0.021 | 9.7 | 0.478 (0.553) | 1 | 1 | 0 | 0.61 | 1.17 |
| **LoRA lift** | 2 | 0.961 | 0.059 | 0.021 | 9.7 | 0.478 (0.553) | 0.218 | 0.40 | 0.023 | 0.61 | **1.13** |
| LoRA lift, seed 1 | 2 | 0.952 | 0.052 | 0.020 | 9.6 | 0.409 (0.528) | 0.212 | 0.40 | 0.024 | 0.62 | not run |
| LoRA lift | 100 | 1.000 | 0.012 | 0.057 | 14.2 | 0.550 (0.674) | 0.107 | 0.46 | 0.013 | 0.56 | not run |
| LoRA lift, seed 1 | 100 | 0.983 | 0.008 | 0.060 | 9.6 | 0.662 (0.595) | 0.167 | 0.46 | 0.029 | 0.62 | not run |

Stochastic and CI-masked Pile ratios are 0.996–1.007 for every arm. The LoRA's read factor A has |cos| 1.000 with its
initialisation at step 2 on every matrix (0.973 at step 100). Where the energy sits: 60% of the LoRA's step-2 update
energy lands on the four saturated matrices (45% on `h.3.mlp.c_fc` alone), against 38% of the fine-tune's; pooled over
all 24 matrices the inexpressible share is 0.19 (LoRA) and 0.17 (ft). By step 100 the LoRA's energy has moved onto
the non-saturated matrices (all-matrix share 0.44), and yet its e_Δ has fallen to 0.11.

## Gates

- **G0.** Seed spread of d and r (LoRA lifts, seeds 0 vs 1): 0.010, so m = 0.1. **PGD floor f = 0.209** (W₀ PGD KL
  0.692 / 0.836 / 0.705 under seeds 0, 1, 2), so "within 3 floors" is ≤ 1.63. Round 4 on the 4090 read f = 0.035 for
  the same three seeds; the 20-step adversary's seed-to-seed spread is itself GPU-dependent.
- **G1.** Every arm learned. The LoRA passes 0.9 at step 2 (P(o) 0.961 and 0.952), with kl_other 0.059 and 0.052,
  the same off-target cost as the fine-tune at step 10 (0.055); its relative update norm is twice the fine-tune's.
- **G2.** Holds for every lift (P(o) difference ≤ 6e-4, shares sum to 1 to 1e-16).
- **G4 closed.** The fine-tune's do-nothing and lift Pile PGD ratios on the L40 read **2.183 and 5.408** against
  round 2's 2.18 and 5.41. The code that ran is round 4's measurement path moved into `harness.py` unchanged, so the
  only thing that differed from round 4's 2.49 and 7.31 is the GPU: round 4's 35% miss was the 4090's adversarial
  search, not the code. Every PGD number from the 4090 run (round 4's analysis) carries that caveat; every number
  here is on the L40, like rounds 2 and 3.

## Scores as filed, with the reading each bears on

| id | filed | result | bears on | numbers |
|---|---|---|---|---|
| M1 read side stays random (pooled ω / null ≥ 0.8) | pass | **pass** | A (inexpressible) | 0.478 / 0.553 = 0.86; A's \|cos\| with init 1.000 |
| M2 lift leaves it in Δ (e_Δ ≥ 0.3; null ≤ 0.2) | pass | **neither** | A (e_Δ) | 0.218 (seed 1: 0.212); step 100: 0.107 / 0.167 |
| C2m LoRA lift fragile like ft (PGD ≥ 5; null ≤ 2 and e_Δ ≤ 0.2) | pass | **neither** (null missed by e_Δ 0.018) | A (faithfulness) | LoRA lift 1.13, LoRA do-nothing 1.17, ft lift 5.41, ft do-nothing 2.18; W₀ + 3 floors 1.63 |
| C1m direct share flat (\|Δd\| < m) | pass | **pass** | mechanism | d 0.023 vs 0.052, m 0.1 |

**Reversal to flag.** C2m was filed as "pass: the LoRA lift reads ≥ 5, fragile like the fine-tune's", on the reasoning
that round 4's fragility grew with how much was lifted. It read 1.13, within one W₀ floor, with the fine-tune's lift at
5.41 in the same job. The filed null ("≤ 2 and e_Δ ≤ 0.2") is missed only because e_Δ is 0.218, so the prediction is
"neither" by the letter and reversed in direction. It is not yet the clean reopening of round 4's C2 that the design
named, for the reason in the next section.

## What this says and does not say

- **Says (round 4's confound).** M1 passed as predicted: on the 20 non-saturated matrices the LoRA's update is as
  inexpressible as an isotropic one, because A has not moved from its random initialisation at the step the LoRA
  learns. Round 4's finding about the read side holds on every matrix. What changed is where the energy lands: a LoRA
  on all 24 matrices puts 60% of its energy on the four saturated matrices, where every update is expressible, and its
  lift leaves 0.22 of the behaviour in Δ instead of the one-site LoRA's 0.33–0.43. The site, not the LoRA, was round
  4's obstacle, as section 2 of the review argued from the fine-tune's per-matrix table.
- **Says (reading A, at the step the LoRA learns).** The lifted LoRA decomposition passes VPD's Pile checks as well
  as W₀ does (every ratio within one floor): the faithfulness half of the weak version. Its Δ takes 0.40 of ‖δW‖² and
  0.22 of the trigger gain. GOALS fixes no number for "Δ not absorbing δW", and round 4 read the fine-tune's lift
  (0.36 of ‖δW‖², 0.09 of the gain) as leaving a twelfth of the behaviour outside; this lift leaves more than twice
  that, so whether it meets the second half is the open reading, not a pass. The fine-tune's lift, with a similar
  share of ‖δW‖² in Δ, fails the Pile adversary fivefold: Δ's share of the update does not predict the fragility.
- **Does not say that a LoRA's lift survives where the fine-tune's does not.** The LoRA was scored at step 2, where its
  do-nothing placement already sits at the floor (1.17; the fine-tune's do-nothing at step 10 is 2.18), and round 2
  found the fine-tune's lift at the floor at its own step 1 (1.14) before rising to 5.41 by step 10. The two lifts
  were compared at different points of training, and the LoRA's lift at the fine-tune's matched step (10) or at its
  finished state (100, where e_Δ is 0.11 and the energy has moved onto the badly-read matrices) carries no Pile
  measurement. That measurement is the one that decides C2; it is a one-stage addition to this script (two Pile
  evaluations, 5 minutes).
- **Says (mechanism).** d is 0.013–0.05 and r 0.56–0.62 for every arm, as in rounds 3 and 4.
- **Says (noise).** The W₀ PGD floor on the L40 is 21% over three adversary seeds, six times round 4's 4090 figure.
  Under this floor round 4's one-site LoRA lifts (1.35–1.48) and do-nothing (1.39) all sit within 3 floors, and the
  1.5 "breaks" line of rounds 2–4 is inside the noise band. Only the fine-tune's lift (5.4) and the one-site
  pull-backs (2.1–4.1) stand clear of it.

## Not run

- Pile ratios of the LoRA lift and do-nothing at steps 10 and 100 (the step-matched comparison with the fine-tune's
  lift), and of the fine-tune's lift at a step where its do-nothing placement reads ≈ 1.2 (steps 1–3): the test that
  turns C2m's direction into a result.
- More than three adversary seeds for the W₀ floor; Pile measurements on the seed-1 LoRA.
- The task-local harness, moved counts, switched-on sets and Jaccards of round 4 (dropped from the harness as
  reported-only numbers without power).
- The per-step lift with updated bases (review section 3.3), still waiting on its tol rule.
- VPD's sparsity loss on any lifted state; any task other than emoticon; Γ updating.

**Hand-off.** A new session should start from this report and `reports/2026-10-19_review_lora-failure-and-fixes.md`.
The next measurement is the step-matched Pile comparison above: add a `pile_steps` size to `experiments/lora_matched`
(LoRA at 10 and 100, ft at 1 and 3), file it as an amendment prediction, smoke, and send it through `/pod-runner`
(cluster, debug QoS, about 10 minutes). After that, section 3.3 of the review needs its tol rule decided through
`/design` before it is built.
