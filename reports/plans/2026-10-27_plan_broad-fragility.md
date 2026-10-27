# Plan: why does teacher training's decomposition of a broad fine-tune fail the adversarial check? (E39)

*2026-10-27, commute-lead. Experiment `experiments/broad_fragility`. Follows E37 Amendment 1
(`reports/analyses/2026-10-26_analysis_teacher-spread.md`) and the card `decision-broad-fragile`.*

## The user's request, restated in the Terms' words

User, 2026-10-27 (task file): "accept that updates for broader tasks are messier, so I don't recommend training
further. Prioritise trying to find exactly why things break down; if there are quick and cheap tests for exploring
this then go for it, but don't get too lost. By the way, does the same apply for SGD or LoRA where the updates are less
diffuse?"

Restated: card `decision-broad-fragile` is decided against option (a), training longer. The commute line runs a few
cheap diagnostic tests of *where and why* the decomposition after teacher training on E36's lowercase fine-tune
fails the paper's adversarial check, and runs the same teacher training on an SGD fine-tune and a LoRA fine-tune of the
same lowercase task, to say whether an update less spread over weight space gives a less fragile decomposition. No fix
is attempted here; a fix would be a new plan.

## /design check

1. **Goal it serves.** `GOALS.md`'s second question: when the commute fails, is it the update (broad post-training
   teaches what the decomposition cannot hold) or the decomposition (too brittle)? E37 found the failure; this asks
   which of the two it is.
2. **Already tried.** E37 and Amendment 1 (longer, adjoint start, masked losses whose control failed). E34: trigger
   fine-tunes by Adam, SGD and LoRA all copied and passed. No random-move control, no localisation, no non-Adam broad
   update has been run.
3. **Approved class.** The user asked for these tests in their words (task file 2026-10-27). Tests 1-2 are also
   class 1-2 (saved states; the random-move control card option (d) names).
4. **Simplest version.** One job, four tests, about two dozen check reads and two 1,000-step teacher trainings; no
   new trainer (E37's teacher training and the harness's check are called as they are).
5. **What would change our mind.** If a random move of the same size fails as badly, the check punishes any move of
   the subcomponents and the fragility is the decomposition's (the brittleness answer). If it passes, the fragility
   is specific to where teacher training puts the update, and tests 2-3 say where.
6. **Cost.** About 40 minutes on one RunPod H100, about $2.5; under 10% of the session budget.

## One-sentence claims under test, the null, and the separating predictions

- **Claim A (generic brittleness).** Any move of the alive subcomponents of the trained decomposition's size breaks
  the check. Null: a random move of matched per-matrix size passes (≤ 1.40).
- **Claim B (frozen-Γ mismatch).** Teacher training writes part of the broad change into subcomponents Γ rates as
  rarely important on Pile text; the adversary may switch those off at little cost to the original but at a large
  cost once they carry the change. Null: reverting the rarely-active half removes no more of the excess than reverting
  a random half of the same size.
- **Claim C (update shape).** A less diffuse update of the same behaviour (SGD, or a rank-8 LoRA) gives a less fragile
  decomposition after teacher training. Null: it fails as the Adam one does.

## The job (`experiments/broad_fragility/run.py`), stages cheapest first, each written as it finishes

Shared settings are E37 Amendment 1's: the released 67M decomposition, Γ and Δ frozen, the check = 20-step mask
adversary from one random start on 256 Pile rows (adversary seed 0), ratio over the unchanged decomposition in the
same job; behaviour kept, gap, damage as in E37. Pass ≤ 1.40, fail ≥ 1.61. Saved state read: E37's lowercase copy at
seed 0, step 1,000 (`copy_lower_seed0_step1000.pt`, check 2.20), called *the trained state* below.

**Stage 0, controls, alone, with an early stop.** (i) The unchanged model through the check at adversary seeds 0, 1,
2: spread ≤ 0.10 and readings within 1% of E37's (0.679 and 0.718 per block). (ii) The trained state loaded and read:
check within 3% of 2.201 and behaviour kept within 0.01 of 0.998. (iii) Revert-all (released U, V on every
subcomponent of the trained state): check within the floor band of 1, behaviour kept within 0.01 of 0. Any miss
stops the job.

**Test 1, random move (claim A).** For each decomposed matrix, add Gaussian noise to the alive U and V, scaled so the
matrix's relative change ‖ΔW‖/‖W₀‖ equals the trained state's for that matrix. Two noise seeds at ×1, one at ×0.5.
Read check, damage, stochastic ratio.

**Test 2, where is it (claims A, B).** On the trained state, revert to released U, V:
- one layer at a time (each of the model's layers), and all attention matrices vs all MLP matrices;
- by Γ's importance: per matrix, rank the alive subcomponents by mean causal importance on the check's Pile rows at
  W₀; revert the bottom half (*rarely active*), then separately the top half; plus one random half of matched count
  per matrix (the null).
Read check, behaviour kept, damage. Report each as the share of the excess removed,
(r_trained − r_revert)/(r_trained − 1).

**Test 3, update shape (claim C).** Fine-tune the lowercase task (E36's data, rows, 300 steps, batch 8, each row once)
two more ways: plain SGD on all 24 matrices, and a rank-8 LoRA on all 24 matrices trained with Adam. For each, pick the
rate from a three-point grid (SGD {0.3, 1, 3}; LoRA {1e-4, 3e-4, 1e-3}) whose held-out gain is closest to the Adam
fine-tune's 0.130 nats per token, not diverged. For each fine-tune record: held-out gain, stable rank of δW per matrix
(‖δW‖_F²/‖δW‖₂², mean over matrices), the outside share (`harness.outside_share`), and the check of do-nothing, the
scaled adjoint split, and teacher training at E37's recipe for 1,000 steps (E37's rows and batch order). The Adam
fine-tune's own stable rank and outside share are read too, as the reference.

Every scored number is printed to the log as well as the JSON. States saved: the two teacher-trained states at step
1,000 (bf16 U, V only).

## Predictions (filed in `results/predictions.json` before the run)

| id | rule | expected | confidence |
|---|---|---|---|
| P1 | random move ×1, mean of 2 seeds: check ≥ 1.61 | fails, 1.8-3 | medium-low |
| P2 | random move ×0.5: check below the ×1 mean | yes | medium |
| P3 | no single layer's revert removes more than 0.6 of the excess | spread over layers | low |
| P4 | reverting the rarely-active half removes ≥ 0.2 more of the excess than the random half | claim B holds | low-medium |
| P5 | SGD and LoRA δW stable rank below Adam's (mean over matrices) | yes; the "less diffuse" premise, positive control | high |
| P6 | SGD and LoRA teacher-trained check ≥ 1.61, each | fail like Adam | medium-low |
| P7 | do-nothing check ≥ 1.61 for SGD and LoRA fine-tunes | fail (update in Δ) | high |

**What each number is blind to.** The check is one 20-step adversary from one start, a lower bound on fragility.
Share of excess removed is a difference of two noisy ratios (each ±~0.07 on the floor, ±14% between batch orders);
shares under 0.2 apart are not separated. Test 2 reverts are not additive: layer shares need not sum to 1. Stable rank
is blind to where the update acts in input space. Test 3's gain matching is by held-out log-probability only.

**Positive controls.** Revert-all reads as W₀ (stage 0); do-nothing fails (P7); the unchanged model's seed spread is
the floor.

**Readings.** P1 holds and random reads at or above the trained state: the check punishes any move of this size, and
broad updates are messier because a broad behaviour needs a large move (brittleness). P1 fails: the fragility is the
trained change's own; then P4 says whether it is the frozen Γ. P6 fails for SGD or LoRA: update shape matters, an
answer to the user's question with the stable ranks beside it.

## Not in scope

Fixes (masked losses, refit Γ, constrained teacher training), longer training (the user ruled it out), a second broad
task, the larger model.
