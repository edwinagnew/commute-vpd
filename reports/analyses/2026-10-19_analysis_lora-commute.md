# LoRA through the decomposition: a LoRA update is not easier to commute than a full-weight fine-tune; by the weak version it is harder

**Takeaway.** On the emoticon task, a LoRA on the paper's own matrix (`h.2.mlp.down_proj`) put through the lift, the
pull-back and the do-nothing placement is no better off than the full-weight fine-tune, and on the two numbers that say
whether Δ absorbs the update it is worse: its inexpressible share sits at the isotropic null at every rank (0.38–0.43
against 0.42; the full fine-tune 0.27 against 0.53) and its lift leaves 0.33–0.43 of the behaviour outside the
subcomponents (the full-weight lift 0.085). Under the Pile adversary every map applied to the LoRA sits above W₀ + 3
floors (1.11): lifts 1.35–2.0, do-nothing 1.39, pull-backs confined to the LoRA's matrix 2.1–4.1. By the filed 1.5
"breaks" line only lift_8 (2.02) and the one-matrix pull-backs break; lifts at ranks 1, 4 and 32 read 1.35–1.48, below
1.5 at a harness whose state-level noise is unmeasured (the reproduction miss on the full-weight lift was 35%), so
whether they pass or fail that line is unresolved. The LoRA lift is less fragile than the full-weight lift (2.0 against
7.3). The all-matrix pull-back stays at the floor (1.10), as in round 2. On the mechanism reading nothing distinguishes the LoRA from the fine-tune: the changed
subcomponents carry 0.05–0.11 of the gain for every arm and re-gating carries 0.5–0.6 for every arm. Of eight filed
predictions four are null (P1, P2, P5, C2), four neither (P3, P4, C1, C3), none pass. Figures and every number:
`notebooks/E07_lora_commute.ipynb`.

*2026-10-19, coding agent. Plan `reports/plans/2026-10-19_plan_lora-lift-and-pullback.md` with its addendum of the same day;
predictions `experiments/lora_commute/results/predictions.json` (filed before the run, unchanged); results
`results/lora_commute.json`; code `experiments/lora_commute/run.py`. RunPod RTX 4090 (`commute-lora_commute-1`), 37 min,
$1.51 for the session including the smoke run. Scoring frame set by the user mid-run (`reports/decisions/2026-10-19_decision_success-criterion.md`): reading A, the weak version in
`GOALS.md` as written (the updated decomposition is faithful to W + δW under VPD's own checks, Δ not absorbing the
update), read first; the direct share d is a mechanism reading, not a pass criterion.*

**Terms.** *Lift*: the finished weight change M written as the minimum-norm change (dU, dV) to its matrix's alive
subcomponents with dU Vᵀ + U dVᵀ = M − O(M); O(M) − dU dVᵀ goes to Δ. *Pull-back*: training the subcomponents' (U, V) on
the fine-tune loss with Γ and Δ fixed; here confined to the LoRA's matrix with LoRA-shaped rank-r offsets (`pb_r`), free
on that matrix (`pb_full`), or free on all 24 matrices (`pb_all`, the ladder's arm). *Inexpressible share*: the share of the
update's output energy (inputs weighted by E[a aᵀ] over the training windows) that no first-order change of the alive
subcomponents can express; the *null* is that share for an isotropic update. *Route split*: the held-out log-odds gain of
`o` at the trigger split into d (changed subcomponents at W₀'s CI values), r (changed CI values on the unchanged
subcomponents), i, e_gate (subcomponents Γ masks off) and e_Δ (not in the subcomponents at all). *Pile ratios*: round 2's
harness (2 × 128 Pile val rows; stochastic, CI-masked, PGD KL), each ÷ W₀'s under the same adversary seed.

## Per arm (held fixed: windows, Γ, the released decomposition; varied: the update and the map)

Scored steps: LoRA rank 1 at step 30 (P(o) 0.22 at step 10), ranks 4–32 at step 10; ft at step 10; pull-backs at 30.

| arm | P(o) | kl_other | inexpressible (null) | e_Δ | ‖Δ′−Δ₀‖²/‖δW‖² | d | r | Pile PGD ratio |
|---|---|---|---|---|---|---|---|---|
| ft nothing | 0.999 | 0.053 | 0.27 (0.53) | 1 | 1 | 0 | 0.60 | 2.49 |
| **ft lift** (24 matrices) | 0.999 | 0.053 | 0.27 (0.53) | 0.085 | 0.36 | 0.053 | 0.60 | **7.31** |
| LoRA_8 nothing | 0.999 | 0.154 | 0.41 (0.42) | 1 | 1 | 0 | 0.54 | 1.39 |
| **lift_1** (step 30) | 0.994 | 0.038 | 0.40 (0.42) | 0.41 | 0.49 | 0.077 | 0.59 | 1.46 |
| lift_4 | 0.974 | 0.047 | 0.43 (0.42) | 0.43 | 0.48 | 0.067 | 0.59 | 1.35 |
| **lift_8** | 0.999 | 0.154 | 0.41 (0.42) | 0.40 | 0.49 | 0.072 | 0.54 | 2.02 |
| lift_32 | 0.997 | 0.038 | 0.38 (0.42) | 0.33 | 0.50 | 0.098 | 0.59 | 1.48 |
| pb_r1 (lr 1e-3) | 0.998 | 0.108 | 0 | 0.011 | 0 | 0.071 | 0.57 | 2.13 |
| pb_r4 (lr 3e-4) | 0.929 | 0.059 | 0 | 0.007 | 0 | 0.069 | 0.63 | 2.19 |
| **pb_r8** (lr 3e-4) | 0.995 | 0.091 | 0 | 0.006 | 0 | 0.106 | 0.61 | **4.06** |
| pb_r32 (lr 1e-4) | 0.981 | 0.071 | 0 | 0.007 | 0 | 0.095 | 0.62 | 2.09 |
| pb_full (lr 3e-5) | 0.995 | 0.029 | 0 | 0.004 | 0 | 0.057 | 0.61 | 1.39 |
| pb_all (lr 1e-5) | 0.996 | 0.009 | 0 | 0.003 | 0 | 0.031 | 0.51 | 1.10 |

Stochastic and CI-masked Pile ratios are 0.99–1.04 for every arm. W₀ floor: 3-seed PGD spread 3.5%, so "within 3
floors" is ≤ 1.11. The step-100 lifts read the same (LoRA e_Δ 0.32–0.40, d 0.10–0.11; inexpressible 0.30–0.34; ft e_Δ
0.14). The LoRA's read factor keeps |cos| 0.92–0.98 with its random initialisation at the scored step (0.67–0.96 at step
100), which is why its inexpressible share sits at the null (round 2's finding, reproduced).

## Gates

- **G0.** Seed spread of d and r (pb_8, lift_8, seeds 0 vs 1): 0.038, so m = 0.115. PGD floor 0.035.
- **G1.** Every LoRA learned; every pb_r found an lr (rank 1 only at 1e-3, with kl_other 0.108 against the 0.110 limit).
- **G2.** Holds for every lift: the lifted state's target model has the fine-tune's P(o) to 6e-7 to 2e-5; shares sum to 1.
  Components alone (unmasked, Δ out) give P(o) 0.74–0.95 for the LoRA lifts, 0.998 for the full-weight lift.
- **G4 reproduction: one miss.** The ladder's pb_all shares reproduce exactly (d 0.025, r 0.53), the recruit test's LoRA_8
  do-nothing r exactly (0.54), round 2's do-nothing Pile PGD ratio within 25% (2.49 vs 2.18). **The full-weight lift's
  Pile PGD ratio reads 7.31 against round 2's 5.41 (+35%).** Everything else about that state reproduces: the fine-tune
  within 4% on kl_other (0.0531 vs 0.0550) and 1% on its norm (0.01126 vs 0.01119), with P(o) 0.9993 in both; W₀'s four
  Pile numbers and the lift's Δ-change fraction (0.3595 vs 0.3596) to four digits; the unmasked ratio 1.205 vs 1.20. The
  difference is in the 20-step adversarial search on a different GPU (RTX 4090 here, L40 in round 2), whose chaotic
  behaviour the PGD review documented. **Cause not established** (the pod is gone; a rerun on an L40 is the test). The
  ordering lift ≫ do-nothing > floor is the same in both runs; every PGD number here carries that ±35%.

## Scores as filed, with the reading each bears on

| id | filed | result | bears on | numbers |
|---|---|---|---|---|
| P1 smallest LoRA commutes (d ≥ 0.5, e_Δ ≤ 0.2 at rank 1) | pass | **null** | A (e_Δ), mechanism (d) | d 0.077, e_Δ 0.41 |
| P2 stops with rank | pass | **null** | A (inexpressible), mechanism (d) | d 0.077/0.067/0.072/0.098 (flat within m); inexpressible 0.40/0.43/0.41/0.38 |
| P3 lift breaks adversarially at r ≥ 8, floor at r = 1 | pass | neither | A | PGD 1.46/1.35/2.02/1.48; floor 1.11 (rank 32 under 1.5, rank 1 over the floor) |
| P4 confined pull-back d ≥ 0.3 | pass | neither | mechanism | d 0.07–0.11 |
| P5 lift and pull-back move the same subcomponents | null | **null** | mechanism | 1 of the top 20 shared |
| C1 LoRA lift's d beats the ft lift's by m | pass | neither | mechanism | 0.072 vs 0.053 |
| C2 LoRA lift faithful where the ft lift is not | pass | **null** | A | 2.02 vs 7.31, both > 1.5; at reference rank 4 or 32 (1.35, 1.48) the rule would have read pass, so this result rests on the rank-8 arm and the 1.5 line |
| C3 one-matrix pull-back's d beats the all-matrix one's by m | pass | neither | mechanism | 0.057 vs 0.031 |

**Reversals to flag.** P1, P2, C1, C2 and C3 were filed as the user's instinct (a LoRA is easier to decompose) and all
came out null or neither; none of the five reversed in the other direction either, since every d is within m of every
other d. P5 null as predicted.

## What this says and does not say

- **Says (reading A):** a LoRA on the paper's site is no more expressible by the released subcomponents than an isotropic
  update, because its read side is near its random initialisation; its lift therefore puts half of it in Δ and leaves a
  third to two-fifths of the behaviour outside the subcomponents, where the full-weight lift leaves a twelfth. The
  pull-back keeps Δ fixed by construction but, confined to one matrix, becomes adversarially fragile (2.1–4.1), and more so
  with a rank constraint; spread over all 24 matrices it stays at the floor (1.10, as round 2's 1.03–1.07).
- **Says (mechanism):** the direct share is 0.05–0.11 whatever the update, the map, the matrix count or the rank; the
  behaviour runs through re-gating (r 0.5–0.6) in every arm, as the ladder and the recruit test found. The LoRA and the
  pull-back move nearly every alive subcomponent of the matrix (321–358 of 358 by more than 1%), and not the same ones.
- **Says about "the limit":** rank is not the axis. Rank 1 behaves like rank 32 on every number. The one rank-ordered
  number, the inexpressible share falling from 0.40 to 0.30 by step 100, tracks the read factor's departure from its
  initialisation, not the rank.
- **Does not say** whether a LoRA whose read side is trained first (or initialised from the subcomponents' read vectors)
  would lift better; that is the obvious next arm. Nor anything about a behaviour the model lacks, other matrices, or a
  Γ that moves.

## Provenance

The pod was terminated after the run finished but before the final `lora_commute.json` and `w0_local_seeds.json` were
downloaded (the fork was told to download after each arm; the last JSON it got was after `pb_r8`). The missing records
(`pb_r32`, `pb_full`, `pb_all`, the seed arms, the score, the W₀ task-local seeds) were rebuilt from the run's own log by
`experiments/lora_commute/recover.py` and are flagged `recovered_from_log`; their per-step curves, task-local harness
numbers and top-20 lists are lost. `results/lora_commute_partial_download.json` is the file as downloaded. The fork made
no code edits (its kernel cells are set-up and polling only; the smoke log has no traceback); the code that ran is the
code in the repo. No state-level Pile PGD floor exists: the seed-1 arms carried no Pile measurement, and the W₀ 3-seed
spread (3.5%) is the floor used, against a 35% reproduction miss on one state.

## Not run

- A rerun of the full-weight lift's Pile PGD on an L40 (the G4 miss); more than one seed repeat; harness seeds on the
  Pile adversary for any arm other than W₀.
- A LoRA with its read side trained first or initialised from the matrix's alive read vectors; LoRAs on other matrices or
  on all 24; a LoRA at the pull-back's step size.
- A per-window or budgeted task-local adversary (the window adversary reads 41–57 nats at W₀ across seeds; its ratios
  of 1.5–2.4 on every arm are reported in the notebook and not scored).
- The Jaccard overlaps of the switched-on sets (computed on the pod, lost with the final JSON).
- Any task other than emoticon; Γ updating.

**Hand-off.** A new session should start from this report and `notebooks/E07_lora_commute.ipynb`. The one experiment this
run points at is the read-side arm above (a LoRA whose A is not random), through `/design`; the G4 miss should be
closed by a 5-minute L40 rerun of round 2's lift at step 10 before any PGD number from a 4090 is compared with one from
the cluster again.
