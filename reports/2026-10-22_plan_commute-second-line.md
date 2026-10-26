# Plan: the commute line's four items (2026-10-22)

*Lead session (commute line), from `reports/2026-10-22_decision_working-hypothesis-and-two-lines.md` → "Two lines →
Commute (second)". Inputs: `reports/2026-10-22_analysis_sgd-vs-adam.md` (E21) and Amendment 3 of
`reports/2026-10-21_analysis_cancellation-complement.md` (the adversary's verdicts hold from 20 to 160 steps under the
VPD paper's step rule). Vocabulary: `GOALS.md` → Terms. The re-gating session runs E22, E23 and E25 and the E18/E20
amendments at the same time; this plan touches none of their directories.*

**In plain words.** The decision report gave the commute question four follow-ups. Two are runs we can start now: a
second training seed for the component-training states that decide whether a map holds at steps 100 and 1000, and a
LoRA trained with plain SGD, which has never been run although every LoRA verdict so far is Adam's. One waits for the
other session's E23, which measures the SGD step-1000 failures directly. The last, a map that shifts Γ's values by one
fixed change, is a new map and goes to the user as a design check before anything is built.

## What Amendment 3 changes for E21

E21 read every verdict under a caution: the 20-step adversary was far from converged (Amendment 2), so the step-1000
SGD fails (2.82, 3.00) against Adam's passes (1.07, 0.95) might reorder under a longer search. Amendment 3 removes most
of that caution: with the paper's rule (steps × step size ≈ 2) the search finds 7% more damage at 160 steps than at 20,
and every ratio moves by at most a tenth. The states Amendment 3 ran are E20's (Adam step-10 fails, the emoticon
splits), not E21's; E23's stage B runs the same sweep on the step-1000 SGD and Adam states. Until it lands, E21's
verdicts are read as budget-stable, with one caveat Amendment 3 itself raises: W₀'s two adversary seeds drift apart
with budget, and r0's ratio would sit near the line if divided by seed 1.

So the move-back condition of the 2026-10-21 decision (a map at the floor that keeps the behaviour at steps 100 and
1000) is met at one seed by Adam component training on emoticon, r0 and r1, and not by SGD at step 1000, nor by
anything on r2. The user kept re-gating as the main line and gave the commute line these items.

## Item A: a second training seed (E26, `experiments/ladder_second_seed`)

**Design check.** *Goal:* Q1, the weak version, and the move-back condition: is "Adam component training passes at
steps 100 and 1000 on r0 and r1, SGD fails at 1000" a property of the recipe or of one batch order. *Nearest tried:*
E21's `sgd_ladder_long` (seed 0 only; `ArmAConfig.seed` is the batch-order seed, free offsets start at zero so it is
the only training randomness). What it settled: the one-seed verdicts. What this adds: whether they replicate.
*Patch or step:* replication, approved class 2. *Decisions:* none departed from; component training stays
optimiser-agnostic, Γ and Δ frozen. *What changes next:* if seed 1 reproduces the pattern on r0 and r1, the move-back
condition is met at two seeds and the user decides whether Q1 returns to first place; if SGD at 1000 passes at seed 1,
its step-1000 fail is seed noise and item B is closed; if Adam fails at seed 1, the one-seed pass was noise and the
condition is not met. On r2, where nothing passes cleanly at seed 0 (Adam loses the target by step 1000: P 0.88 → 0.78;
SGD fails), seed 1 says whether that is the task or the seed. *Control first:* W₀ through the harness (three adversary
seeds, `faith_score.w0_check`, stop on fail), then the seed-0 Adam 1e-5 and SGD arms regenerated in-job and scored at
step 1000 as the reproduction gate (Adam pass, SGD fail on r0 and r1, within 25% of E21's ratios; r2 reported), before
any seed-1 state is scored. Regenerating differs from E21 only in the card (H100 instead of L40; the same-job floor
absorbs it) and is cheaper than moving 60 MB states off the cluster. *Numbers:* adversarial Pile ratio (W₀ 1; seed-0
gate as E21; seed 1 is the question), stochastic ratio (near 1), held-out P(target) (≥ 0.9 or "below P 0.9"), KL
elsewhere (against E21's task limits), rel_norm_max (reported; SGD 0.017, Adam 0.018 at 1000 on r0/r1 at seed 0).

**Recipe.** `sgd_ladder_long`'s arms and config with the batch-order seed as a run argument: per task in (r0_t352_v310,
r1_t534_v369, r2_t275_v2667), SGD at E21's rate (0.03, 0.03, 0.1) and Adam 1e-5, 1000 steps, states at 100 and 1000.
Adam 3e-6 and the do-nothing arm are dropped (no prediction reads them). Stages, cheapest first, results dumped after
each: W₀ → seed-0 arms, scored at 1000 (gate) → seed-1 arms, scored at 100 and 1000. One job, the three tasks in order
r0, r1, r2; emoticon is not run (every arm passed at every step at seed 0).

**Predictions** (to file in `results/predictions.json` before the run): P1, two-sided, named outcomes: on r0 and r1
Adam 1e-5 at seed 1 passes at both steps (replicates) / fails at either (one-seed pass). P2, two-sided: SGD at seed 1
fails at 1000 on both (replicates; item B stays open) / passes on either (seed noise). P3, r2: does any seed-1 arm hold
P ≥ 0.9 at 1000 and pass; two-sided, expected not. Noise: W₀'s same-job adversary spread (0.075 on the H100 in E16,
0.21 on the L40).

**Where and cost.** Per task on the L40, E21 took 33 min with four arms; here two arms × two seeds and six harness
readings, so about 25 min per task on the L40 and less on an H100. Three tasks in one job: over 30 min, so RunPod,
H100 (runs over ~1 h), cap $8.

## Item B: why SGD fails at step 1000 where Adam passes at the same size

**Held until E23.** The other session's `experiments/cancellation_sgd` (cluster job 29344, pending at 12:59) runs the
excess decomposition (E20's parts), E17's revert tests and Amendment 3's budget sweep on exactly these states: SGD 0.03
and Adam 1e-5 at step 1000 on r0 and r1, beside the passing SGD step-100 control. Designing a separating measurement
before those numbers exist would duplicate it. Item A adds the cheapest check: whether the fail is seed noise.

**Hypothesis on record, not a filed prediction** (E23's predictions are the other session's). E21 §5 read SGD's change
as about seven times more concentrated than Adam's at a matched size. If the adversary's extra damage comes from
masking a few subcomponents that carry a lot of the SGD change, E23's adversary-revert test should remove most of the
SGD state's excess with few subcomponents, and the same test on the Adam state should need many. If instead the excess
sits in the alive–dead interaction with no small carrier set (E20 Amendment 1's reading of the Adam step-10 fails), the
concentration story is wrong and the next measurement is which subcomponents the two states change in common. Either
way the design goes through `/design` after E23's analysis report.

## Item C: a LoRA trained with SGD (E27, `experiments/lora_sgd`)

**Design check.** *Goal:* GOALS's optimiser subgoal ("both as the component-training optimiser and as the fine-tune
optimiser whose update the splits consume") on the canonical update, the LoRA: every LoRA so far (E7, E8, E9, E14) was
trained with Adam at 1e-3, so no LoRA verdict counts yet as a property of the update. *Nearest tried:* E9/E14 (the
rank-8 Adam LoRA on all 24 matrices: outside-both ÷ null 0.54–1.07 at every step, do-nothing 1.17/1.09/1.04 and the
least-squares split 1.26/1.22/1.07 at steps 10/100/1000, Δ's share rising to 0.54 at 1000) and E21's SGD fine-tune
(outside share 0.034 against Adam's 0.218 at a matched size). What this adds: whether the LoRA's "outside like noise"
reading is the LoRA's parametrisation (a random read factor A that barely trains) or Adam's. *Patch or step:* a named
item of the user's decision (class 3). *Decisions:* LoRA is a canonical update; Γ, Δ frozen; no departure.
*What changes next:* outside-both ÷ null near 1 for the SGD LoRA too → the LoRA's outside share is the parametrisation,
and the fine-tune/LoRA contrast in E14 is not an optimiser artefact; well below 1 (like the SGD fine-tune) → E14's LoRA
rows are Adam's and need re-reading. A fail of the SGD LoRA's do-nothing or least-squares state where Adam's passes
would be the first LoRA fail at the floor. *Control first:* W₀ (three adversary seeds, stop on fail) → the Adam LoRA
regenerated in-job with E9's recipe and gated against E9's step-10 and step-1000 readings (do-nothing and least-squares
ratios within the floor's spread; outside-both ÷ null within 0.1) → the SGD rate grid (cheap) → the SGD LoRA's series
and states. *Numbers:* the four blocks ÷ null along nine checkpoints (W₀ none; random draws at 1; Adam LoRA 0.5–1.1);
rel_norm_max; held-out P(target) (≥ 0.9 by E4's rule); KL elsewhere; at steps 10, 100, 1000 the do-nothing,
least-squares and scaled-adjoint states' adversarial and stochastic ratios (W₀ 1; Adam LoRA as E9), Δ's share of
‖δW‖² under least squares (adjoint 0, the positive control) and the adjoint state's P(target).

**Recipe.** `finetune.finetune_lora_arm_a` gains an optimiser choice without changing its Adam behaviour (a field on
`LoraConfig` defaulting to "adam", the same branch as `component_training.train`). Rank 8 on all 24 matrices, seed 0,
E9's checkpoints (1, 2, 3, 5, 10, 30, 100, 300, 1000). SGD rate by E4's rule on a grid spanning four decades at 1000
steps (the rate that reaches P(target) ≥ 0.9 with the least KL elsewhere, under the fine-tune's limit); B starts at
zero so A's first gradient is zero under either optimiser. Series and states as `experiments/lora_split` (E14), through
`split.block_table`, `split.rescaling_share`, `harness.nothing_and_split` and `split.adjoint_state`.

**Predictions** (filed before the run): P1 the SGD LoRA's outside-both ÷ null is at least 0.7 at every checkpoint from
step 10 (parametrisation, not optimiser; the SGD fine-tune's 0.034 is the contrast); P2 its read factor's |cos| with its
init stays ≥ 0.97 at 1000 (Adam: 0.90); P3 do-nothing and least-squares states pass at steps 10, 100 and 1000 (as
Adam's); P4 two-sided: Δ's share of ‖δW‖² under least squares at 1000, SGD against Adam's 0.54.

**Where and cost.** E14's part 2 ran in about 35 min on an A5000; with the grid and the Adam gate this is about an hour
on a 4090 or less on an H100. RunPod, cap $8.

## Item D: a map suggested by H′ — subcomponents frozen, Γ's values shifted by one fixed change

**Needs the user's confirmation before anything is built** (a new map, and a departure from `decisions.md`; not in an
approved class). Design agent, 2026-10-22. No code or predictions until the user has picked a reading.

**The request as an operation.** s is E18's fixed change: per alive subcomponent, the mean of Γ's values at the
trigger over the task's training windows under W₀ + δW, minus the same under W₀ (`regating.shift_of`). Open in the
user's words: where s is applied, and which model's activations Γ reads.

- **(i), recommended.** U, V, Δ as at W₀; Γ′(x) = clip(Γ(x) + s) at every position, with Γ reading the decomposition's
  own weights, Σ components + Δ = W₀. This is E18's pattern applied everywhere. D′ decomposes W₀ and is scored against
  W + δW's outputs. Δ absorbs nothing, which satisfies "Δ may not absorb δW" vacuously; the components carry the new
  behaviour only through which of them are on. The unmasked KL (all masks 1, Δ out) ignores Γ, so it equals do-nothing's
  by construction. Only the CI-masked, stochastic and adversarial KLs and the sparsity loss can move. Raising a value
  narrows the adversary's range on that subcomponent, so a positive s lowers the adversarial KL by construction; the
  sparsity loss is what charges for that.
- **(ii)** As (i), shifted only where a trigger-like condition holds. This is not a map of (D, δW) unless the condition
  is read off δW. Not recommended.
- **(iii)** Do-nothing plus the shift: δW in Δ, so the weights are W + δW; Γ reads the updated model and s is added.
  This counts the switching twice, because frozen Γ already switches on the updated activations (E24's own values,
  0.85–0.98). It also fails "Δ may not absorb δW" by construction. Useful only as a mechanism contrast.
- **(iv)** Per-subcomponent constants that replace Γ's values (E18's mean pattern). Applied everywhere this removes Γ.
  Not a map.

Every reading needs the task's trigger and training windows to compute s, as component training needs the loss and the
data. The departure from "Γ stays frozen in Φ" is a different one from E15's: no parameter is trained and nothing is
refit. Γ's output gets one closed-form offset vector computed from (D, δW, the task's windows).

**Design check.** *Goal:* Q1, the weak version: the first map built from what re-gating showed, and the commute line's
test of H′. *Nearest tried:* E18 and E24 (`TRIED.md` rows E18, E24) score the fixed change at the trigger only, on
held-out windows, never on the Pile against the fine-tuned model. E15 refits Γ's parameters and failed its W₀ control.
*Patch or step:* a step, a new map. *Decisions:* departs from "Γ stays
frozen in Φ" (above); relies on reading A and on E15's row staying separate. *What changes next:* H′ predicts the
shifted D′ reproduces the fine-tuned model at the trigger. Applied everywhere, it should harm the Pile, because E18
found the change lowers the target away from the trigger on the word tasks (Pile ÷ trigger −0.07 to −0.18). Pile harm
→ the change must be conditioned on the input, which points back at Γ updates (E15) or reading (ii) built from δW. No
Pile harm with the trigger reproduced → a frozen-subcomponent map that passes, and Q1's move-back case. *Control first:*
W₀ with s = 0 must reproduce W₀'s own harness numbers exactly (same adversary seed); stop on any difference. Then the
reference s = 0 against the fine-tuned model: nothing changes, Γ blind to the update. *Wrong-target control:* the
` http` null fine-tune's s. *Numbers:*

| number | W₀ (s = 0) | s = 0 vs fine-tuned | do-nothing | ` http` s | shifted D′ (H′) |
|---|---|---|---|---|---|
| trigger recovery, share of own values | 0 | 0 | ≈1 (own values) | ≤ 0 (E24: −0.17 to −0.30) | ≈ 0.74 emoticon, ≈ 0.40 r1 |
| Pile CI-masked ÷ W₀ | 1 exactly | ≈ 1 | as E21 | > 1 | > the s = 0 reference |
| Pile adversarial ÷ W₀ floor | 1 | ≈ 1 | as E21 | reported | two-sided, pass line f |
| Pile stochastic ÷ W₀ | 1 | ≈ 1 | ≈ 1 | reported | ≈ 1 |
| sparsity ÷ W₀ | 1 | 1 | as E21 | reported | two-sided, cheating line 1.05 |

**Plan sketch.** *States:* E24's emoticon SGD 0.01 and Adam 3e-5 and r1_t534 SGD 0.03, each with its ` http` null,
regenerated in-job as E24 did. *Stages, cheapest first:* W₀ with s = 0 through the harness (three adversary seeds,
stop on mismatch) → fine-tunes and s (E24's gate: recoveries within 0.002) → trigger readouts of every reading-(i) state
(seconds) → Pile harness for s = 0 vs fine-tuned, shifted, ` http` s and s shuffled among alive subcomponents. *Predictions:* P1 (H′, directional): at the trigger, the shifted D′ recovers at least 0.6
of own values on emoticon and within 0.1 of E24's trigger-only figure on every state. P2 (directional, from E18):
CI-masked Pile KL against the fine-tuned model is higher than the s = 0 reference by more than the floor. P3
(two-sided): adversarial pass or fail at the floor line; sparsity above or below 1.05. P4: the ` http` s recovers at
most 0.1, and its Pile change is of similar size to the task's. *Noise:* W₀'s same-job adversary spread, f = 0.075 on
the H100 and 0.21 on the L40. Trigger readings are deterministic; seeds agree within 0.02. *Blind to:* accumulation
along training (one-shot, step 10); input-conditioned gating; one seed, one 67M model. *GPU:* the fine-tunes and s take about 2.5 min (E24, 143 s). The Pile harness
takes about 2 min per state on the L40: 3 W₀ seeds plus 4 readings for each of 3 updates is ~15 readings, ~35 min in
all. Over 30 min, so RunPod 4090 via `/pod-runner`, cap $3. *The user confirms:* the reading ((i) recommended); Γ
reading W₀'s activations rather than the updated model's; s measured at the trigger on training windows (the map then
uses the task's data); the departure from "Γ stays frozen in Φ"; and that the weak version's "about as well as D on W"
is read against W + δW's outputs although D′'s weights sum to W₀.

## Rules for the agents

- One Opus agent per experiment, end to end, as `reports/2026-10-22_plan_small-update-reruns.md` → "Rules for the
  agents": restate the check for its experiment under "Agent notes" below, write `run.py` and `plot.py`, file
  `results/predictions.json` before running, `make lint` clean, Codex adversarial review on the diff before the paid
  run, hand the GPU run to a `pod-runner` agent per `.claude/skills/pod-runner/SKILL.md`, then the analysis report
  `reports/2026-10-22_analysis_<topic>.md` per `/writeup` with `check.py`, the `TRIED.md` row with its E-code, one line
  in `docs/research_log/PENDING.md`, and the `README.md` codes-table row.
- Shared files (`experiments/TRIED.md`, `docs/research_log/PENDING.md`, `README.md`, this file) are edited by several
  agents and by another session: re-read right before each edit, append only, never rewrite others' lines. Do not
  flush the research log; the other session owns the queue today.
- Do not edit `experiments/sgd_ladder_long/`, `sgd_kv/`, `lora_split/`, `lora_plateau/`, `regating_leak/`,
  `cancellation_complement/`, `cancellation_sgd/`, `small_finetune_splits/`, `step_split_small/` or their results.
  Shared steps go into `src/commute/` as new functions or new optional fields with the old default, never by changing an
  existing function's behaviour.
- Budget: this session's `REMOTE_KERNELS_BUDGET` is $40; caps E26 $8, E27 $8. Terminate pods as soon as the results
  are downloaded and verified (sha256 both sides). Never load the model on the laptop. The scored run is its own smoke;
  W₀ first.

## Not run (by this plan)

- More adversary seeds per updated state, or an H100 re-read of E21's near-line verdicts.
- A second seed for the emoticon arms, Adam 3e-6, or the do-nothing arm.
- Item B's separating measurement (after E23) and item D (after the user's confirmation).

## Agent notes

### E26 ladder_second_seed (agent, 2026-10-22)

Design check as Item A; restated for E26. **Goal**: Q1 weak version and the move-back condition: is "Adam 1e-5 passes
at steps 100 and 1000 on r0 and r1, SGD fails at 1000" a property of the recipe or of batch-order seed 0. **Nearest
tried**: E21 `sgd_ladder_long` (seed 0 only). **Class** 2 (replication), no sign-off. **Control first**: W₀ (adversary
seeds 0, 1, 2, `faith_score.w0_check`, stop on fail) → per task the seed-0 SGD and Adam 1e-5 arms regenerated and
scored at step 1000 as the gate G1 → the seed-1 arms at steps 100 and 1000. **Numbers and expected readings**:
adversarial ratio (W₀ 1; seed-0 gate E21's 2.82 / 1.07 on r0, 3.00 / 0.95 on r1; seed 1 is the question), stochastic
ratio (≈ 1), held-out P(target) (≥ 0.9 or "below P 0.9"), KL elsewhere (E21's task limits 0.209 / 0.243 / 0.407),
rel_norm_max (≈ 0.017 SGD, 0.018 Adam at 1000).

What I changed and why:
- One process can run the three tasks and computes W₀ once (the Pile readings at W₀ do not depend on the task); each
  task's JSON carries the same W₀ record. Saves ~14 min of H100 time. A task whose JSON is complete is skipped on a
  rerun (resume).
- G1 = the seed-0 PGD ratio at 1000 within 25% of E21's and the same `faith_score.verdict` (E21's verdict against the
  L40 floor, ours against this job's floor). A failed G1 marks that task's predictions unscored; the run continues. On
  r2 G1 is reported, not gated. The seed-0 held-out P and KL at 1000 are recorded beside it, not gated.
- P1 and P2 keep the plan's two named outcomes and add a third, "unresolved", when the only non-passes (P1) or
  non-fails (P2) are "inconclusive", as E21's Q1 did; "below P 0.9" counts as not passing.
- SGD rates hard-coded (3a's selection, 0.03 / 0.03 / 0.1) and asserted against E21's recorded `ref_3a.sgd_lr`.
- New shared function `faith_score.state_verdict` (E21's per-state rule) with a CPU test; seed-0 states are not saved
  (only seed 1's, as bf16 offsets).
- Codex adversarial review (focus "gates, seed plumbing, every number has an expected value") on run.py, plot.py,
  predictions.json and `faith_score.state_verdict`: approve, no material findings; nothing skipped. GPU run handed to
  one pod-runner (RunPod H100, cap $8).

### E27 lora_sgd (agent, 2026-10-22)

Design check as Item C; restated for E27. **Goal**: the optimiser subgoal on the canonical update: is the LoRA's
"outside like noise" reading (E9, E14) the LoRA's parametrisation or Adam's? **Nearest tried**: E9/E14 (Adam LoRA, rank
8, all 24 matrices) and E21's SGD fine-tune (outside-both ÷ null 0.034 vs Adam's 0.218). **Class** 3 (named item of the
user's decision), no sign-off. **Control first**: W₀ (adversary seeds 0, 1, 2, `faith_score.w0_check`, stop on fail) →
the Adam LoRA regenerated with E9's recipe, gate G1 (stop on fail) → the SGD rate grid → the selected SGD LoRA's series
and states. **Numbers and expected readings**: outside-both ÷ null at nine checkpoints (random draws 1, the blocks
code's control C2; Adam LoRA 0.54–1.07, E14; SGD fine-tune 0.03–0.06); |cos| of A with its init (1 at step 1; Adam
0.90 at 1000); adversarial ratio of do-nothing / least squares / adjoint at 10, 100, 1000 (W₀ 1; Adam 1.17–1.26 at 10,
1.04–1.07 at 1000); Δ's share of ‖δW‖² (least squares: Adam 0.41 → 0.54; adjoint 0, the positive control); held-out
P(target) and KL elsewhere (Adam 0.9999 and 0.0048 at 1000).

What I changed and why:
- Rate grid 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3 (seven, half-decade steps; the brief's example range, which is three
  decades, not four). Reason: B starts at zero and A's rows have norm about 1 and are nearly orthogonal, so A Aᵀ ≈ I and
  a B step of size lr moves δW = BA by lr times the gradient restricted to eight random input directions; with clip 1
  the step's norm is at most lr. Adam at 1e-3 moves every B entry by about 1e-3, a step norm of order 0.1–1, so the
  matched SGD rate should sit near 0.1–1, inside the grid with a decade to spare on each side.
- E4's rule at step 1000 (the scored step): P(target) ≥ 0.9 and KL elsewhere ≤ 2 × E9's 0.00476 = 0.0095; among those
  the least KL elsewhere. Fallbacks, flagged in the results: if some rate reaches 0.9 but none is under the KL limit,
  the least KL among those that reach 0.9; if none reaches 0.9, the highest P(target). Non-finite runs are excluded.
- G1 references are E9's own readings (do-nothing 1.17 / 1.04, least squares 1.26 / 1.07 at 10 / 1000), tolerance
  relative max(f, 0.209) as E22; outside-both ÷ null at 10 within 0.1 of E14's 0.979; P(target) ≥ 0.99 at 1000. The
  Adam LoRA's states are measured at 10 and 1000 only (G1 and P4 read nothing else; step 100 comes from E14).
- P1 is filed as the plan states it (≥ 0.7 at every checkpoint from 10), with named outcomes: all ≥ 0.7
  "parametrisation"; any < 0.3 "optimiser" (the SGD fine-tune's side); else "mixed". Note: Adam's own LoRA read 0.54 at
  step 100 (E14), so the rule is stricter than the Adam series it is compared with; the in-job Adam series is reported
  beside it.
- P3 by `faith_score.verdict` on the adversarial ratio against the same-job floor. P4 compares SGD's least-squares Δ
  share at 1000 to the in-job Adam LoRA's (E14's 0.54 reported): "higher" / "same" / "lower" at ±0.1 absolute.
- Positive control C3: the adjoint states' Δ change fraction ≤ 0.01 (E14: 6e-6 to 3e-4). C2: the random draws'
  outside-both at its analytic null (0.85–1.15), from `split.rescaling_table` as E22.
- Code: `LoraConfig` gained `optimizer` ("adam" default) and `momentum`; `finetune_lora_arm_a` builds plain SGD when
  asked, Adam path unchanged (CPU test against numbers recorded from the old code). New module
  `src/commute/lora_readouts.py`: the read factor's initial value and |cos| with it (E9's, moved), and E4's rate rule
  with the fallbacks above. Everything else is existing shared code (`split.block_table`, `split.rescaling_table`,
  `split.adjoint_state`, `harness.nothing_and_split`, `harness.measure`).
- Codex adversarial review (2026-10-22, focus: SGD branch leaves the Adam path unchanged; rate rule; every number has an
  expected value): verdict approve, no material findings; nothing skipped. Added before the run: a CPU test of
  `run.py`'s `rows`/`score` on a synthetic results dict (`tests/test_lora_sgd_score.py`), since `score` runs inside every
  save and a bug there would have lost the last state. Predictions filed 13:15 (sha256 4a2175b4…). GPU run handed to one
  pod-runner (RunPod H100, cap $8).

### E30 gamma_shift_map (agent, 2026-10-22)

Design check as Item D, reading (i), confirmed by the user 2026-10-22 ("run it"); restated for E30. **Goal**: Q1 weak
version, the commute line's test of H′: does a decomposition with U, V, Δ frozen at W₀ and Γ′(x) = clip(Γ(x) + s) at
every position (Γ reading W₀'s activations, s = E18's fixed change) reproduce the fine-tuned model as well as D does W₀?
**Nearest tried**: E18/E24 (s at the trigger only, held-out windows, never the Pile); E15 (Γ refit, failed its W₀
control). **Step**, a new map; departs from "Γ stays frozen in Φ" by one closed-form offset (user-confirmed). **Control
first**: W₀ through the ordinary harness (seeds 0, 1, 2, `faith_score.w0_check`) and the new path at s = 0 against a copy
of W₀ must match to 1e-4 (stop otherwise) → the fine-tunes regenerated as E24, gate G1 (r and the shared pattern's
recovery within 0.05 of E24; stop otherwise) → the readout of s (cheap) → per update (emoticon SGD 0.01 first, then Adam
3e-5, then r1_t534 SGD 0.03) the Pile readings of (a) s = 0, (b) s, (c) the ` http` null's s, (d) s shuffled, then
do-nothing. **Numbers and expected readings**: as the plan's table; trigger recovery with Γ′ everywhere (a 0 exactly, b ≈
E24's 0.74 / 0.74 / 0.40, c ≤ 0, d ≈ 0); Pile ratios to W₀ (a ≈ 1; unmasked equal across a–d by construction); P2–P3
read (b) against (a).

What I changed and why:
- The new harness path is `src/commute/shifted_harness.py`, not `gamma.pgd_sources`/`cancel.masked_kl`: those differ
  from the authors' PGD in init order (sorted keys vs `target_module_paths`) and in the KL code, and sign steps amplify
  any difference, so the s = 0 control could not match to 1e-4. The module instead calls the authors' own pieces
  (`stochastic_recon_subset_loss`, `_forward_with_adv_sources`, `get_pgd_init_tensor`, `make_mask_infos`) with the gate
  dict and the reference logits swapped; the CPU test shows s = 0 reproduces `faithfulness.faithfulness` and
  `sparsity_loss` bit for bit on the toy, with and without a twin model.
- The fine-tuned logits come from a second copy of the model (the "twin") holding W₀ + δW, called per batch; caching
  the Pile logits (≈13 GB) is avoided. The main model stays at W₀ throughout, so stochastic-routing positions that skip
  the components use W₀'s weights, as D′ sums to W₀.
- Sparsity under Γ′: the upper-leaky values plus the change the shift makes to the gates, upper + (clip(lower + s) −
  lower); s = 0 is exact, and a subcomponent the clip pins at 0 or 1 is charged nothing extra. Read on the same 256
  rows in 4 blocks of 64 (kv_task's layout); ratio = mean of the block ratios.
- G1 gates the absolute recovery ρ_shared (E24's own gate quantity), not R_shared. P1 compares R (share of own values)
  with E24's R_shared, as the plan's table does.
- P2's δ = max(W₀'s CI-masked spread over seeds, 0.02): the CI-masked KL is deterministic, so the same-job spread is
  near 0; 0.02 is `faith_score.W0_CI_TOL`, the cross-job tolerance already used for it. P4 drops the plan's "Pile
  change of similar size" from the rule (reported as (c)/(a) beside P2).
- The readout's recoveries use E18's trigger-only pattern (as G1 and P5 need); its helpers are a new module
  `src/commute/shift_readout.py` (CPU-tested). Per update it runs on s, on the null's s and on s = 0 (the zero
  reference). Write energy = |u_cᵀ δW v_c| / (|u_c||v_c|).
- A CPU test runs `score()` and `plot.py` on a synthetic results dict (`tests/test_gamma_shift_map_score.py`). Each Pile
  reading is dumped as it finishes and a rerun resumes from the results file.
- Codex adversarial review (2026-10-22, focus: Γ reads W₀'s activations, KLs against the fine-tuned model, s = 0
  reproduces the harness, sparsity loss reported): verdict approve, no material findings; nothing skipped. Predictions
  filed 13:37 (sha256 f8e38622…). GPU run handed to one pod-runner (RunPod 4090 or faster, cap $4).

### E27 lora_sgd, run and provenance (agent, 2026-10-22; continues the E27 note above)

- Run done (2026-10-22): RunPod H100, machine 11a4986f, 1439 s of run; the pod-runner stopped on a usage limit before
  reporting, the files arrived nested in `results/downloaded_results/` and were moved into place (predictions sha256
  identical); no commute pod alive at 15:05. Cost not recorded (about $3-4). Rate 0.3 by the rule; P1 miss ("mixed"),
  P2, P3 pass, P4 "lower". Analysis `reports/2026-10-22_analysis_lora-sgd.md`. `plot.py` also draws E22's SGD fine-tune
  series on `outside.png` (P1's contrast).

## Hand-off (lead, 2026-10-22 16:10)

- **Done:** E26 (`reports/2026-10-22_analysis_ladder-second-seed.md`): E21's verdicts replicate at a second batch order;
  the move-back condition is met at two seeds on emoticon, r0, r1; the user's decision on the main line is pending.
  E27 (`reports/2026-10-22_analysis_lora-sgd.md`): the SGD LoRA sits between noise and a full fine-tune (outside-both
  ÷ null 0.33–0.87), its read factor stays at its init, every placement passes; P1 missed (filed ≥ 0.7 at every step).
- **E30 not run** (`experiments/gamma_shift_map`): code, tests, Codex review and predictions (13:37) are ready; three
  pod attempts failed in set-up (`uv run` re-synced the venv; a pod with no SSH endpoint; the user stopped the third
  during its first `sync()` and asked for a retry in a new session). For the retry: export `GIT_PAT` in the shell
  (the clone step was skipped without it), launch with `UV_NO_SYNC=1 .venv/bin/python`, never `uv run`, and check
  whether the plugin applies `workdir = "/root/work"` (the last pod's kernel ran in `/workspace`). Inputs and their
  sha256 are in the last runner's report; expected run ~35–40 min on a 4090, cap $4. Then the analysis per the E30
  agent note, the TRIED row, PENDING line and README row.
- **Item B** waits for E23's analysis (commute-97 will message when it lands). **PENDING.md** holds the E26 and E27
  lines; commute-97 flushes the log once E22–E25 are done.

### E30 run and analysis (lead, 2026-10-22)

- Run done on the third attempt (RunPod 4090, 2347 s, exit 0; no commute pod alive afterwards). Attempt 1 (5090) crashed
  in C1 on a shape bug, fixed in `run.py`; attempt 2 (4090) resumed the 5090's W₀ readings and missed C1's 1e-4 match;
  attempt 3 from scratch matched exactly. Analysis `reports/2026-10-22_analysis_gamma-shift-map.md`; TRIED row, README
  row and plan line, PENDING line written. `plot.py` gained `trigger.png` and `adversarial.png`.
- Result: P1, P4 pass; P2 fail (r1 reads "less", −0.03); P3 "cheating" on all three with adversarial fails 4.7–26× the
  s = 0 reading; P5 many small on emoticon, a few larger on r1. Reading (i) fails as a map. Next for this item: the
  raised and lowered halves of s applied separately on the Pile (~8 min), then reading (ii); both need `/design`.

## Amendment 1 to E30: the raised and the lowered half of s, each applied everywhere (agent, 2026-10-22)

**Design check (approved class 4: the commute line's named next step, E30's "What next" 1; user-approved run).**
- *Goal*: Q1 weak version. Scores the next candidate map, "raise Γ's values everywhere, lower them only at the
  trigger": on the Pile, where trigger positions are rare, that map is the raised half applied everywhere.
- *Nearest tried*: E30 (whole s everywhere: adversarial 8.0× s = 0, sparsity 3.10×; s shuffled: 1.11×, 3.13×). New:
  each half's adversarial, CI-masked and sparsity readings on its own.
- *Patch or step*: a patch on E30 that decides between two next maps.
- *Decisions*: as E30 (Γ frozen except the closed-form offset, user-confirmed for Item D). No departure.
- *What changes next*: see the routes below. **Honest note**: the adversarial outcome is largely implied already. The
  masks range over [Γ′, 1], so the lowered half's range contains the whole s's (which failed 8×) and the raised half's
  is contained in s = 0's; only the 20-step adversary's approximation can break that. The readings that are not implied
  are the raised half's sparsity (the decisive one for the raise-everywhere map; E30's shuffle suggests ~3×, i.e.
  cheating), the CI-masked KL per half, and each half's trigger recovery applied everywhere.
- *Control first*: C1 alone (W₀ ordinary harness, adversary seeds 0, 1, 2, `w0_check`; the shifted path at s = 0
  against a copy of W₀ equal to 1e-4). **Early stop**: any C1 failure stops the job (the script exits with status
  "stopped at C1"). Then G1: the regenerated fine-tune's r, ρ_trig, ρ_shared within 0.05 of E30's (stop otherwise).
  Positive control for the adversary: the whole s, read last in the same job, must fail again.

**Arms** (emoticon SGD 0.01, 10 steps, seed 0): s = 0, s₊ = max(s, 0), s₋ = min(s, 0), s; the Pile in that order.
Plus one cheap readout with no adversary: how far s₋ lowers Γ's values on the Pile, Σ_c min(Γ_c(x), |s_c|) averaged
over positions, against the lowered half of E30's shuffled s (E30's candidate says the real lowered entries sit on
subcomponents that are on at many Pile positions).

**Numbers and expected values** (ratios are to the s = 0 arm in the same job):

| number | expected |
|---|---|
| C1 rel. difference (4 KLs, sparsity) | 0.0; W₀ floor f 0.035 on a 4090, ~0.21 on an L40 |
| G1 differences (r 0.616, ρ_trig 0.977, ρ_shared 0.724) | 0.0 same card type, ≤ 0.01 across |
| trigger recovery everywhere, R | s = 0: 0; s: 0.748; s₊ ≈ 0.39, s₋ ≈ 0.24 (E30's trigger-only halves) |
| adversarial ratio | s ≈ 8; s₋ ≥ s's (≈ 8 or more); s₊ ≤ ~1 |
| CI-masked ratio | s 1.12; s₊, s₋ each 0.95–1.15, no direction |
| stochastic ratio (not scored) | s 1.09; halves 0.95–1.10 |
| sparsity ratio | s 3.10; s₊ ≈ 3.1; s₋ < 1 |
| opened(s₋) ÷ opened(shuffled s₋) | ≥ 3 |

**Predictions** (`experiments/gamma_shift_map/results/predictions_amend1.json`, written by `halves.py --predictions`):
P1 (central): verdicts of s₋ ÷ 0 and s₊ ÷ 0 against 1.5/(1+f) and 1.5(1+f), in-job f; lean "lowered half" (s₋ fails,
s₊ passes), high confidence; "no power" if s does not fail. P2: each half's trigger recovery everywhere within 0.1 of
E30's trigger-only figure; lean pass. P3: s₊'s sparsity ratio > 1.05 reads "cheating"; lean cheating (~3.1). P4: the
opened ratio ≥ 3; lean "premise holds", medium.

**Routes**: lowered half and s₊ not cheating → the raise-everywhere, lower-at-trigger map; lowered half and s₊ cheating
→ no fixed raise-everywhere map, next reading (ii) (s only where the trigger is); both halves fail → reading (ii);
raised half fails → candidate wrong and containment broken, check the adversary first; neither alone → reading (ii).

**Run**: `experiments/gamma_shift_map/halves.py` (shared bench moved from E30's `run.py` into
`src/commute/shift_map.py`, E30's behaviour unchanged; `shift_readout.opened`). 8 Pile readings at E30's ~115–120 s
each, plus ~1.5 min of loading, fine-tune and trigger readings: about 18 min on a 4090. Cluster `debug` first
(`cluster/gamma_shift_map_amend1.sbatch`), RunPod 4090 if not started within 10 min. Output
`results/gamma_shift_map_amend1.json`, log `results/large/full_amend1.log`.

Predictions filed 2026-10-22 19:17:37, sha256 a9db0c2d609cbdebf4dd173f53845748a1a1752f5cd53576d198d12f3ffad6c6. The
Codex adversarial review could not run (Codex usage limit); the code was re-read by its author instead, and CPU tests cover
`score()` (partial and complete results) and `shift_readout.opened`. `make lint` clean on the touched files; `make test`
192 passed.
