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
seed 0, step 1,000 (check 2.20; retrained in the job by the same recipe, which reproduced it exactly in Amendment 1), called *the trained state* below.

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

## Amendment 2 (2026-10-28): why does the broad failure happen?

*Approved by the user, 2026-10-28 (`decisions.md`, card `decision-lora-passes` closed with option (b) plus labelled
component training: "it sounds like its cheap amendment tests so you can go ahead and run them"). Follows the
analysis's Amendment 1 and its "What next", and gaps 2 and 5 of `reports/reviews/2026-10-28_review_technique-comparison.md`.
Code `experiments/broad_fragility/amend2.py` (imports `run.py`; new shared code in `src/commute/rank_cap.py`);
predictions `results/predictions_amend2.json`; run brief `RUN_BRIEF_amend2.md`.*

### The request, restated in the Terms' words

Four tests, one job, on E39's lowercase task, harness and card:

- **(A)** Component training on the lowercase task's own labels (next-token cross-entropy), Γ and Δ frozen, with Adam
  and with plain SGD, read by the check at the state whose held-out gain matches the Adam fine-tune's. It tells "a
  frozen-Γ decomposition cannot hold this broad behaviour" from "teacher training's copying fails".
- **(B)** Teacher training, as E39 ran it, on a plain SGD fine-tune held to rank 8 per matrix. It tells whether the
  LoRA's hard rank cap explains its pass, or something else about the LoRA does.
- **(C)** Teacher training on the rank-8 LoRA fine-tune repeated at a second seed, and both LoRA decompositions read
  on a second set of Pile rows. E39's pass was one seed and one set of rows, and the state was lost.
- **(D)** A random move of the alive subcomponents matched to the trained decomposition in output change (damage on
  plain text), not in weight size.

Two readings are mine and are flagged for the lead. "A second seed" changes the LoRA's initialisation seed and teacher
training's batch order together. "A second set of rows" is the check's Pile rows (val 256-511), not the fine-tune's
training rows, as E39's "What next" item 3 meant.

### /design check

1. **Goal.** `GOALS.md`'s branch question: if no simple map works, is it the update or the decomposition? (A) asks it
   directly for the broad task. (B) and (C) bear on LoRA as the canonical update and on the optimiser subgoal.
2. **Nearest tried.** E39 and Amendment 1 (TRIED row E39), E37, E21 and E26 (SGD component training fails late on
   narrow tasks), and E27 (LoRA with plain SGD on emoticon). Component training with labels has never been read on a
   broad task. No rank-capped full-matrix update, no LoRA repeat and no damage-matched random move has been run.
3. **Patch or step.** (B) to (D) are follow-ups to E39's surprises, so patches. (A) is a step on the branch question.
4. **Decisions.** Γ and Δ stay frozen. The check keeps the paper's setting (decision 2026-10-27), so its readings
   remain lower bounds. The user ruled out training teacher training longer on the broad task. Nothing here does that.
   (A) trains with labels, and the user approved it explicitly.
5. **What changes next.** See the decision tree below. Each test has a branch, and (A) with (C) together decides
   whether the broad line goes to the user as "the decomposition's limit" or as "teacher training's fault".
6. **Control first.** Stage 0 runs alone with an early stop: W₀ (spread, per-block readings), the Adam fine-tune and
   its teacher-trained decomposition reproduced (2.2011), and the Adam fine-tune's do-nothing (E37: 4.8678) as the
   positive control. The LoRA at seed 0 and batch order 0 reproduces E39 first (control G_C). The Adam teacher-trained
   decomposition must fail on the second rows (control G_rows2) before the LoRA's readings there count.
7. **Recipe differences from the reference.** (A) differs from teacher training only in its loss and in the state
   picked: labels instead of KL to the teacher, and the gain-matched state among steps 100, 300 and 1000. Rows,
   batch, batch order and clip are the same. (B) differs from Amendment 1's SGD fine-tune only in the cut to rank 8
   after each step; the rate is re-picked by E39's rule. (C) seed 0 is E39's recipe exactly; seed 1 changes the LoRA
   seed and the batch order. (D) is E39's random move scaled by a search instead of fixed at ×1.

### Stages, cheapest first, each written as it finishes (about 40 min on the H100)

| stage | what | reads |
|---|---|---|
| 0 | W₀ on adversary seeds 0-2; Adam fine-tune, its teacher-trained decomposition, its do-nothing | 2 check reads |
| move (D) | per noise seed 0 and 1, E39's random move scaled by x until damage is within 5% of the trained decomposition's (double x from 4, then bisect in log x; at most 12 damage reads) | 2 check reads |
| label_adam (A) | component training with labels, Adam at 1e-5 and 3e-5, 1,000 steps, states at 100, 300, 1000; the state not diverged whose gain is nearest the Adam fine-tune's | 6 gain reads, 1 check read |
| label_sgd (A) | the same with plain SGD at 0.01, 0.03 and 0.1 | 9 gain reads, 1 check read |
| lora (C) | LoRA seed 0 / batch order 0, LoRA seed 1 / batch order 1, each teacher-trained; then W₀ on val rows 256-511 (seeds 0-2) and the Adam, LoRA-0 and LoRA-1 teacher-trained decompositions read there | 5 check reads |
| cap (B) | plain SGD with each matrix's change cut to its best rank-8 approximation after every step, rates 0.01, 0.03, 0.1, E39's rate rule; plain SGD at 0.01 for its share beyond rank 8; teacher training on the capped fine-tune | 1 check read |

States saved as bf16 offsets: both labelled states, both LoRA teacher-trained states (E39's was lost), and the capped
SGD teacher-trained state.

### The numbers, and what each should read

| number | W₀ | do-nothing (Adam) | control | expected |
|---|---|---|---|---|
| check, Adam teacher-trained | 1 | 4.868 | 2.201 (stage 0) | 2.201 |
| check, matched random move (P14) | 1 | — | E39's size-matched moves 1.03, 1.07 | 2-5, x 4-7 |
| check, labelled Adam / SGD (P9, P10) | 1 | 4.868 | 2.201 | 1.6-2.4 / 1.6-3 |
| gain of the chosen labelled state | 0 | 0.130 | 0.130 | 0.10-0.16 |
| check, LoRA seed 0 teacher-trained (G_C) | 1 | 2.653 (E39) | E39 1.2726 | 1.27 |
| check, LoRA seed 1 teacher-trained (P12) | 1 | — | G_C | 1.15-1.45 |
| W₀ spread, second rows | 0 | — | first rows 0.0748 | 0.03-0.10 |
| check on the second rows: Adam / LoRA 0 / LoRA 1 (G_rows2, P13) | 1 | — | Adam must fail | 1.8-2.6 / 1.1-1.5 / 1.1-1.5 |
| share of the capped update beyond rank 8, largest matrix | 0 | — | must be ≤ 1e-4 | ~1e-10 (rounding) |
| share of the plain SGD update beyond rank 8, pooled | 0 | — | — | 0.1-0.4 |
| check, capped SGD teacher-trained (P11) | 1 | — | Amendment 1's SGD 2.07, LoRA 1.27 | 1.8-2.2 |

Behaviour kept, gap, damage, the stochastic ratio and size are read at every state as in E39. They enter no rule.

### Predictions (filed in `results/predictions_amend2.json` before the run)

| id | rule | expected | confidence |
|---|---|---|---|
| P9 | labelled Adam, gain-matched state: check ≥ 1.61 | fails, 1.6-2.4 | low-medium |
| P10 | labelled SGD, gain-matched state: check ≥ 1.61 | fails, 1.6-3 | medium-low |
| P11 | teacher training on SGD held to rank 8: check ≥ 1.61 | fails like plain SGD | medium-low |
| P12 | teacher training on the LoRA at seed 1, batch order 1: check ≤ 1.40 | passes | medium |
| P13 | on the second rows, both LoRA teacher-trained states ≤ 1.40 | both pass | medium-low |
| P14 | damage-matched random moves, mean over matched seeds: check ≥ 1.61 | fails, 2-5 | medium |

Gates, each scored as "unscored: why" rather than as a miss. A state's gain must be within 30% of the Adam
fine-tune's (P9-P12). For P11 the cap must hold, with no matrix's share beyond rank 8 over 1e-4. For P13, G_rows2
must hold. For P14, a seed counts only if its damage came within 5% of the target. A non-finite check is unscored.
Both lines are inclusive. The rules were run on 314,928 combinations of outcomes and gates, including
missing stages, ties at the lines, NaN, no matched seed, every rate diverged and a failed control. Every
combination scored, and every reading below was reached.

**What each number is blind to.** The check is one 20-step adversary from one start, a lower bound. (A)'s state is
picked by gain alone, from six (Adam) or nine (SGD) states. Its rows are drawn with replacement, unlike the
fine-tune's. Γ is at W₀ throughout. (B) caps the rank one way, by cutting after each step, not as a factored update.
It has one rate and one seed. (C) changes the seed and the batch order together and adds one seed only. The second
rows are one more set of 256. (D) uses one direction family per noise seed, isotropic per side, which moves every
alive subcomponent including those Γ rarely turns on. It is matched in mean damage on 64 rows, not in where the change
falls or in behaviour.

**Noise floor.** W₀'s spread over adversary seeds, 0.0748 on this card; the pass and fail lines bracket 1.5 by it.
The second rows' spread is read in the job. E37's batch-order spread of the check was 14% (2.20 and 1.88). Gains
reproduce to 0.0001 across jobs.

### Decision tree

- **Stage 0 fails:** stop and report; nothing else is read.
- **G_C misses** (the LoRA at seed 0 does not read 1.27 again): not a stop, but tell the user at once. C is read with it.
- **A, Adam (P9).** *Pass*: the frozen-Γ decomposition can hold this broad behaviour, so the fail belongs to teacher
  training's copying. Next, class 1: compare the labelled and teacher-trained states on the saved offsets. A fix to
  teacher training needs a new plan. *Fail*: training the subcomponents on this behaviour fails by labels as well as by
  copying, so the fail is not teacher training's alone. *Between or unscored*: not separated; report it.
- **A, SGD (P10)** is read beside Adam's. If Adam passes and SGD fails, it is E21/E26's late SGD fail again, an
  optimiser effect.
- **B (P11).** *Pass*: the rank cap is enough. If C stands, the cap explains the LoRA's pass; next is a new plan, a
  LoRA at higher rank. *Fail*: the cap does not explain it. If C stands, this plan's next branch is a rank-8 LoRA
  trained with plain SGD on the lowercase task, teacher-trained and read, to tell the factorisation from the Adam
  optimiser: about 15 min, about $1.5, RunPod H100, approved with this plan.
- **C (P12, P13).** *Stands* (seed 1 passes, both pass on the second rows): the LoRA's pass is a result. *Seed 1
  passes, second rows unscored*: it replicates on one set of rows only. *Fails* (seed 1 or a second-rows reading
  fails): E39's 1.27 was chance, B becomes a fact about SGD only, and "why does the LoRA pass" closes. *Between*:
  not robust enough to explain; a third seed is the user's call.
- **D (P14).** *Fail*: any move of the subcomponents with this much output change fails, so the size of the change can
  explain the broad fail. *Pass*: the fragility is specific to the directions training chooses.
- **A with C.** Labels pass and C stands: the decomposition holds broad behaviour, and teacher training's copy of
  full-matrix fine-tunes is what fails. Labels fail and C stands: the form of the update matters, and B says whether
  it is the rank. Labels fail and C fails: no broad update is held reliably by any map tried. That goes to the user as
  a card: the "brittle" branch for broad updates, with a Γ refit or a weak version scoped to narrow updates as the
  options. Labels pass and C fails: teacher training is unreliable on broad updates generally.

The tree's wording is filed with the predictions (`readings`). `amend2.py` writes the reading it lands on under
`score.tree`.

### Machine and cost

About 40 min of run: stage 0 about 7 min including the model load, D about 3, A about 8, C about 10, B about 6-8.
The cap's cut after every step is the uncertain part. That is over 30 min, so it goes to RunPod on E39's card, the
H100 80GB HBM3: stage 0's per-block readings are that card's, and an L40 would stop there. With about 10 min of
set-up and downloads, that is about 50-55 min of pod at about $4.4/h: about $4, capped at $6. That is 10-15% of the
session's $40.

### Not run

- The LoRA trained with plain SGD (B's next branch), a LoRA at higher rank, and SGD's teacher-trained state on the
  second rows.
- Per-block check readings on the second rows.
- A second batch order for (A), and Γ's importance read at the labelled states.
- Reverts on the labelled or LoRA states; random moves in directions the inputs use.

## Amendment 3 (2026-10-29): a rank-8 LoRA trained by plain SGD

*Approved by the user, 2026-10-29 (`decisions.md`, "Card triage, continued", item (6); triage table
`reports/decisions/2026-10-29_decision_card-triage-draft.md` row 7; card `decision-lora-sgd`). It is Amendment 2's
decision-tree branch for B failing with C standing, approved with that plan, plus the two cheap readings in the
analysis's "What next" (`reports/analyses/2026-10-28_analysis_e39-amend2.md`, items 2 and 3). Code
`experiments/broad_fragility/amend3.py` (imports `run.py`; new shared step `harness.on_rows`); predictions
`results/predictions_amend3.json`; draft run brief `RUN_BRIEF_amend3.md`.*

### The request, restated in the Terms' words

One job on E39's lowercase task, harness and card, with three readings in this order:

1. **The SGD teacher-trained decompositions on the second rows.** Teacher training on Amendment 1's plain SGD
   fine-tune and on Amendment 2's SGD fine-tune held to rank 8, read by the check on Pile val rows 256-511. Both
   failed on the first rows, on one block of 128 rows. If both pass here, the job stops. The contrast with the LoRA
   would then rest on that one block, and the SGD LoRA would have little to separate.
2. **The labelled SGD state at step 1,000.** Component training with labels by plain SGD, at the rate of Amendment 2's
   passing state (0.03), read at step 1,000 instead of step 100. It says whether that pass was only an early state,
   as SGD's passes were on the narrow tasks (E21, E26).
3. **A rank-8 LoRA trained by plain SGD.** E39's LoRA (all 24 matrices) trained with plain SGD on the lowercase task,
   with the rate picked from E27's grid by E39's rule, then teacher-trained and read by the check. A fail reads as
   "the LoRA form passes only with Adam". A pass reads as "the LoRA form, not the Adam optimiser, goes with the pass".

Two readings are mine and are flagged for the lead. First, "the labelled SGD state at step 1,000" is taken at rate
0.03, the rate of the passing state, so it is the same training run read later. Rate 0.01 also reaches the gain at
step 1,000 but is not read. Second, the stop rule ends the whole job, as the card words it. So if both SGD
decompositions pass on the second rows, reading 2 is not made either. Moving the stop after reading 2 would cost about
3 min more ($0.13).

### /design check

1. **Goal.** `GOALS.md`'s branch question: when the map fails, is it the update or the decomposition? E39 and its
   amendments found that teacher training holds one form of broad update (a rank-8 LoRA trained by Adam) and not
   full-matrix fine-tunes. This run tells the form from the optimiser.
2. **Nearest tried.** E39 Amendment 2: the Adam LoRA passes at two seeds and on the second rows; SGD held to rank 8
   fails; labelled SGD passes at step 100 and labelled Adam fails. E27 trained this LoRA by plain SGD on the emoticon
   task (rate 0.3 by E4's rule, learnt by step 10). No run has teacher-trained a LoRA made by SGD, read the SGD
   teacher-trained decompositions on new rows, or read labelled SGD at step 1,000 on the broad task.
3. **Patch or step.** A patch: it tests the last open difference between the passing LoRA and the failing fine-tunes.
4. **Decisions.** Γ and Δ stay frozen. The check keeps the paper's setting (decision 2026-10-27), so its readings are
   lower bounds. Teacher training is not run longer on the broad task (user, 2026-10-27). Nothing here departs from these.
5. **What changes next.** A pass moves the broad line's explanation to the update's form. The next question is then
   what in the form matters (E27 found SGD leaves the read factor near its random start). A fail closes "why does the
   LoRA pass" with "only with Adam", and the line goes to the user as a card. Readings 1 and 2 qualify Amendment 2's
   two contrasts.
6. **Control first.** Stage 0 runs alone with an early stop: W₀ on the first rows (spread and per-block readings),
   and the Adam fine-tune with its teacher-trained decomposition reproduced (2.2011), the failing reference. Each
   later stage opens with its own control, and a miss stops the job. On the second rows the control is W₀'s spread
   and the Adam teacher-trained decomposition failing as in Amendment 2 (1.8826). Before the SGD LoRA it is E39's Adam
   LoRA and its teacher-trained decomposition reproduced (gain 0.1479, check 1.2726), the passing reference. The SGD
   LoRA's do-nothing must fail (positive control G_nothing), or its teacher-trained reading is unscored.
7. **Recipe differences from the reference** (Amendment 2's LoRA at seed 0 and batch order 0, check 1.2726):
   - the fine-tune's optimiser, Adam (betas 0.9, 0.999) replaced by plain SGD (momentum 0, no weight decay): **the
     thing tested**;
   - the fine-tune's rate, 1e-4 replaced by the rate of E27's grid that E39's rule picks: tested with the optimiser,
     since a rate does not carry across optimisers;
   - the gain reached, 0.148 for the Adam LoRA, replaced by whatever SGD reaches within 30% of the Adam fine-tune's
     0.130: **not removed**; it is gated and named in P17's "blind to";
   - everything else is removed. Rank 8 on all 24 matrices, LoRA seed 0 (B = 0, A random), 300 steps of 8 rows each
     once, clip 1.0, the held-out rows, teacher training (Adam 1e-5, batch 8, 1,000 steps, batch order 0) and the check
     (first rows, adversary seed 0) are the same code and settings.

   The SGD states of reading 1 are regenerated in the job by their filed recipes (Amendment 1's rate 0.01, Amendment
   2's capped rate 0.03), not loaded from the saved files. Amendment 2 reproduced every regenerated state on this card
   to the 4th decimal (2.2011, 1.2726) and its gains to 0.0001. Each regenerated gain is checked against the filed one
   (within 0.005, E39's stage 0 tolerance), and a miss stops the job. The saved files total 241 MB, about 50 min at
   the ~80 KB/s uplink (CLAUDE.md), against about 6 min of training. Reading from memory also avoids the bf16 rounding
   of the saved offsets, which moved the check by up to 4.4% in E23's C1.

### Every setting and its source

| setting | value | source |
|---|---|---|
| SGD LoRA rate grid | 0.03, 0.1, 0.3, 1, 3 | E27's grid (0.003-3), its five middle rates, centred on E27's pick 0.3 |
| SGD LoRA rate rule | gain nearest the Adam fine-tune's among rates not diverged; scored if within 30% | E39 (`run.pick_rate`); A1's P8 and A2's gate |
| SGD LoRA optimiser | plain SGD, momentum 0, no weight decay | E27 (`LoraConfig.optimizer = "sgd"`) |
| LoRA rank, matrices, init, seed, steps, batch, clip | 8, all 24, B = 0 and A ~ N(0, 1/d_in), 0, 300, 8, 1.0 | E39 (`run.lora_ft`) |
| teacher training | Adam 1e-5, batch 8, 1,000 steps, batch order 0 | E37, as in E39 and A2 |
| the check and its lines | 20-step adversary, one start, seed 0, 256 rows; pass ≤ 1.40, fail ≥ 1.61 | E37, E39 |
| second rows | Pile val 256-511 | A2 |
| plain SGD fine-tune, SGD held to rank 8 | rates 0.01 and 0.03 | A1's and A2's chosen rates |
| labelled SGD | rate 0.03, 1,000 steps, batch 8, batch order 0, states at 100 and 1,000 | A2's chosen rate and recipe |
| reproduction tolerances | gain 0.005, check 3% | E39 stage 0 (ii) |

### Stages, each written as it finishes (about 28 min of run on the H100)

| stage | what | check reads | time |
|---|---|---|---|
| load | model, W₀ on the first rows at adversary seeds 0-2, data | — | ~4 min |
| 0 | W₀ spread and per-block readings; the Adam fine-tune and its teacher-trained decomposition | 1 | ~4 min |
| rows2 | W₀ on the second rows at seeds 0-2; the Adam teacher-trained decomposition there (control); plain and capped SGD fine-tunes regenerated, each teacher-trained and read there; the stop rule | 3 | ~11 min |
| label | labelled SGD at 0.03 to step 1,000; gains at step 100 (G_label) and 1,000; step 1,000 read and saved | 1 | ~3 min |
| lora_sgd | the Adam LoRA and its teacher-trained decomposition (control); the SGD LoRA grid; the chosen rate's do-nothing and teacher-trained decomposition, saved | 3 | ~7 min |

### The numbers, and what each should read

| number | W₀ | do-nothing | control | expected |
|---|---|---|---|---|
| W₀ spread, second rows | 0 | — | A2 0.0621 | 0.06 |
| check, Adam teacher-trained, second rows (G_rows2) | 1 | — | must fail; A2 1.8826 | 1.883 |
| check, plain SGD teacher-trained, second rows (P15) | 1 | 2.623 (A1, first rows) | first rows 2.067 | 1.4-2.2 |
| check, SGD held to rank 8 teacher-trained, second rows (P15) | 1 | — | first rows 2.122 | 1.4-2.2 |
| gain, labelled SGD at step 100 (G_label) | 0 | — | A2 0.1253 | 0.1253 |
| check, labelled SGD at step 1,000 (P16) | 1 | — | step 100: 1.275 | 1.6-2.6 |
| check, Adam LoRA teacher-trained (lora_ref) | 1 | 2.653 (E39) | A2 1.2726 | 1.273 |
| gains of the SGD LoRA grid | 0 | — | Adam fine-tune 0.130 | rising to 0.10-0.15 at 0.3 or 1 |
| check, SGD LoRA do-nothing (G_nothing) | 1 | itself | Adam LoRA 2.653 | 2-4 |
| check, SGD LoRA teacher-trained (P17) | 1 | G_nothing | Adam LoRA 1.273 | 1.15-1.5 |

Behaviour kept, gap, damage, the stochastic ratio and size are read at every state as in E39. They enter no rule.

### Predictions (filed in `results/predictions_amend3.json` before the run)

| id | rule | expected | confidence |
|---|---|---|---|
| P15 | on the second rows, both SGD teacher-trained decompositions: check ≥ 1.61 each (holds only if both do) | both fail, 1.6-2.2 | low-medium |
| P16 | labelled SGD at 0.03, step 1,000: check ≥ 1.61 | fails, 1.6-2.6 | medium-low |
| P17 | teacher training on the SGD LoRA: check ≤ 1.40 | passes, 1.15-1.5 | medium-low |

Gates, each scored as "unscored: why" rather than as a miss. P15 needs the second rows' control to hold (a miss also
stops the job). P16 needs G_label to hold, the run not to diverge, and the step-1,000 gain within 30% of the Adam
fine-tune's (A2 read 0.154, ratio 1.18). P17 needs a rate of the grid that did not diverge, the chosen rate's gain
within 30%, and the SGD LoRA's do-nothing failing (G_nothing). If no rate qualifies, nothing is scored, teacher
training is not run, and the grid's range of gains is reported. Both lines are inclusive, and a non-finite check is
unscored. The rules were run on every combination of outcomes and gates: checks at 1.40, just above it, 1.5, just
below 1.61, 1.61, 2.5 and NaN; a missing or zero Adam gain; each stage missing; every gate failing. Every combination
scored, and every reading below was reached.

**Why these expectations.** P15: on the second rows the Adam decomposition's check fell from 2.20 to 1.88. Both SGD
fails sat on the first rows' second block (2.93, 2.95; first block 1.16, 1.25). So a drop toward the lines is likely.
I lean to fail, with low confidence. P16: E21 and E26 saw SGD component training pass at step 100 and fail at step
1,000, and the step-1,000 damage (0.229) is nearly twice step 100's. P17: E27 found that plain SGD barely moves the
LoRA's read factor (|cos| with its start 0.997, against Adam's 0.83). The update then stays in a fixed random row
space, as most of the Adam LoRA's does, and labelled SGD passed. But the full SGD fine-tunes failed, so medium-low.

**What each number is blind to.** One adversary start, so each check is a lower bound. P15 adds one set of 256 rows
only. P16 is one rate and one batch order, with Γ at W₀. P17 is one LoRA seed, one rate and one batch order. The SGD
LoRA's gain differs from the Adam LoRA's, and a fail does not say which part of Adam matters (its per-parameter
scaling or its momentum).

**Noise floor.** W₀'s spread over adversary seeds is 0.0748 on the first rows and 0.0621 on the second. E37's
batch-order spread of the check is 14%. Regenerations on this card reproduced gains to 0.0001 and checks to the 4th
decimal.

### Stops and readings

- **Stage 0, the second rows' control, a regenerated gain, or the Adam LoRA control misses**: stop and report.
- **Both SGD teacher-trained decompositions pass on the second rows** (the stop rule): stop. The SGD fail then rests
  on one block of 128 rows, and little is left for the SGD LoRA to separate. *Both fail* (P15 holds): the contrast
  with the LoRA stands on new rows. *Mixed*: it holds in part.
- **P16.** *Fail*: E21/E26's pattern holds on the broad task, and Amendment 2's labelled pass was only early. *Pass*:
  the labelled SGD pass is not only early here.
- **P17.** *Pass*: the LoRA form, not the Adam optimiser, goes with the pass; the next step is to ask what in the form
  matters (a new plan). *Fail*: "the LoRA form passes only with Adam"; the broad line goes to the user as a card.
  *Between or unscored*: reported, with no branch.

The wording is filed with the predictions (`readings`). `amend3.py` writes the reading it lands on under `score.tree`.

### Machine and cost

About 28 min of run (above), plus about 10 min of set-up and 3 min of downloads: about 40 min of a RunPod H100 80GB
HBM3 (E39's card; stage 0's per-block readings are that card's) at about $2.7/h. That is about $1.8, capped at $3. It
is over 30 min, so not the cluster's `debug` QoS. No saved state goes to the pod.

### Not run

- The labelled SGD state at rate 0.01, step 1,000, and the labelled Adam state at step 100 (the analysis's item 2
  names it).
- The labelled states on the second rows.
- A second LoRA seed or rate for the SGD LoRA; SGD with momentum; the Adam LoRA at a matched gain.
- Stable rank, outside share or the read factor's movement for the SGD LoRA.
