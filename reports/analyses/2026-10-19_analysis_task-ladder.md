# Task ladder: the round's hypothesis failed. Coverage does not predict how much of a new behaviour the CI-masked model keeps, and rarity does

## Takeaway

**The hypothesis of this round was refuted on both counts.** It said: whether training *inside* the decomposition keeps
the new behaviour explained by the subcomponents Γ marks as causally important is predicted by *coverage*, and rarity
of the target does not matter on its own (plan, "The claim"). Both filed predictions failed:
- **P1 (coverage predicts the CI-masked model's retained gain): null.** Within rung 0 the high-coverage task retained
  0.14 more than the low-coverage one (margin 0.15); within rung 1, 0.08. Rung 2 could not be scored.
- **P2 (rarity does not matter): null.** The high-coverage rung-2 task retained 0.34 of the gain against ≥ 0.45 required.

What the data show instead, on 7 tasks (two per rung plus emoticon):
1. **Changing the subcomponents contributes ≤ 8% of the gain.** If the trained subcomponents are kept but every
   subcomponent is masked with its *original* (W₀) CI values, the CI-masked model's log-odds of the target rises by only
   0.03–0.08 of the target model's gain.
2. **Most of what the CI-masked model keeps comes from changed CI values on the unchanged subcomponents.** Take the
   *released, untrained* subcomponents, and mask them with the CI values Γ computes from the trained model's
   activations. That alone reproduces 0.28–1.07 of the gain. It does so to within 0.16 whichever of three training
   methods produced the trained model: (a) full-weight fine-tune, (b) Adam on all subcomponents, (c) Adam on only the
   subcomponents whose CI at the trigger is ≤ 0.01. So for these three methods, this quantity is set by the task.
3. **That quantity is larger for common targets**: 0.75–0.92 of the gain at rungs 0–1, 0.28–0.62 at rung 2 and on
   emoticon. Coverage does not order it. With 7 tasks this is an ordering by rung, not a trend: the two rung-2 tasks
   have matched baselines but differ by 0.32.
4. **Where it is small, the gap is behaviour the unmasked model has but the CI-masked model loses.** At rung 2 that is
   0.34–0.66 of the gain.

Not established: whether point 2 is a mechanism the decomposition explains, or the trained model's computation passing
into the CI-masked model through Γ's input (Γ reads the trained model's activations at every layer and position, and at
the trigger those already encode the target). This run cannot tell the two apart. See "What changes next".

*2026-10-19, coding agent. Plan `reports/plans/2026-10-15_plan_task-ladder.md`, amended by
`reports/predictions/2026-10-16_prediction_ladder-routes-amendment.md`. Predictions `experiments/task_ladder/results/predictions.json`
(filed 2026-10-15) and `predictions_amend1.json` (filed 2026-10-16), both before the run. Results `results/{noise,tasks,
ladder}.json`. Ran 2026-10-19 on a RunPod pod via remote-kernels (name prefix `commute-task_ladder`; not on the cluster
as the plan assumed). The previous session's remote-kernels tally reads $3.69, which may include other pods. Code as it
ran: `results/large/code_as_ran/`. Figures: `figures/route_split.png`, `figures/regating_vs_coverage_and_rarity.png`.*

## Terms

- **Subcomponent** c: a rank-1 piece u_c v_cᵀ of one of the 24 decomposed matrices. **CI value** g_c: Γ's causal
  importance for c at one position, in [0, 1], computed by Γ from the model's activations (Γ is a transformer over
  all 24 matrices' inputs at all positions). **CI-masked model**: the forward pass in which each subcomponent is
  multiplied by a **mask**; here the mask is set equal to a CI value. Δ (the part of W not in the subcomponents) is left
  out. **Unmasked model**: every mask 1, Δ out. **Target model**: the trained W itself.
- **Task**: a trigger token followed by a target token, mined from Pile train rows (e.g. `r0_t588_v320` is " will" → " be").
  **Rung** = the decade of the baseline P(target | trigger) with full-row context: 0 is [0.1, 1), 1 is [0.01, 0.1),
  2 is [0.001, 0.01). Emoticon is the reference task.
- **Coverage**: the share of the first pull-back step's squared update that falls on subcomponents with high mean CI
  at the trigger (CI-weighted). Computed before training; not computed for emoticon.
- **Arms** (all trained on the same windows, same loss, 10 steps scored):
  - *Do-nothing*: a full-weight fine-tune; its δW goes in Δ, so the subcomponents are unchanged.
  - *Pull-back*: Adam trains all alive subcomponents (U, V); Γ and Δ fixed.
  - *Gated-off*: the same, but only subcomponents with mean CI ≤ 0.01 at the trigger are trainable.
- **Route split** (`local_faith.route_split`), at the trigger, mean over held-out windows. L(C, M) = log-odds of the
  target under the CI-masked model with subcomponents C and masks M. C₀ = released subcomponents, C₁ = trained ones;
  M₀ = CI values Γ computes from W₀'s activations, M₁ = from the trained model's. Gain = the target model's log-odds
  gain. Each share is ÷ gain; the five sum to 1 exactly:
  - d, *direct* = L(C₁,M₀) − L(C₀,M₀): changed subcomponents, original masks;
  - r, *re-masking* (written "re-gating" in the amendment and code) = L(C₀,M₁) − L(C₀,M₀): original subcomponents,
    new masks;
  - i, *interaction* = the remainder of L(C₁,M₁) − L(C₀,M₀);
  - e_gate, *masked-off*: what the unmasked model gains beyond the CI-masked model (carried by subcomponents with low CI);
  - e_Δ, *residual*: what the target model gains beyond the unmasked model.
- **retained_ci_masked** = d + r + i: the share of the gain the CI-masked model keeps. P1 and P2 were scored on it.

## The two failed predictions: what they mean

- **What P1 assumed.** That the CI-masked model keeps the behaviour through the subcomponents the update changes, so
  an update that starts on high-CI subcomponents (high coverage) is kept better. The route split shows the changed
  subcomponents carry ≤ 8% of the gain; the retained share is mostly r, which is the same for the gated-off arm, whose
  update starts on *zero*-coverage subcomponents. So P1 tested a channel that carries almost nothing.
- **Was this foreseeable? Yes, partly.** The concentration test (2026-10-15) had already shown on emoticon that r
  dominates and that do-nothing gets r ≈ 0.5. The 2026-10-16 amendment said so and predicted A1 (coverage predicts d)
  null, but P1 and P2 were left as filed on `retained_ci_masked`, which mixes d and r. Filed predictions are not edited,
  and the amendment did not withdraw them. That was the mistake: P1 was kept knowing its metric was dominated by a
  term coverage could not move.
- **What P2's failure means.** The round's claim "rarity does not matter" is refuted at rung 2 on the measure as
  filed. The CI-masked model keeps less of a rarer behaviour, and what it loses shows up as masked-off (the unmasked
  model still has it). For the project's question ("does the decomposition stay valid along training"), this is the
  first evidence that the answer depends on how new the behaviour is. It rests on two rung-2 tasks and emoticon.
- **What survives.** The amendment's A2 (r + i exceeds d by ≥ 0.15 on every task) passed. A1 (coverage does not predict
  d) came out null, as predicted. Neither rescues the round's hypothesis.

## Gates

| Gate | Result | Reading |
|---|---|---|
| G0 noise | d_opt(d) 0.0007, d_opt(r+i) 0.012, d_opt(retained_ci) 0.013. So m = m_d = m_r = 0.15 (the floor). Pile W₀ ratios: CI-masked and unmasked spread 1.0, stochastic 1.004, PGD 1.035 | The split is stable across optimiser seeds. The amendment's "brittle" branch (shares swing by more than m) is ruled out on emoticon |
| G1 (replaced) gated-off | Every task learned (held-out P 0.94–1.00). d ≤ 0.003 and \|e_Δ\| ≤ 0.01 on all 7: **pass** | The harness puts zero direct share on an update placed where Γ does not look. The old rule (retained_ci ≤ 0.2) would have failed on all 7 (0.33–0.95), from re-gating alone |
| G1b (replaced) gated-on, `h.3.mlp.down_proj` only | Did not learn: P = 0.018, gain 0.43 nat, below the 1-nat rule. Reported only: d = 0.79, r = i = 0, max\|dg\| = 0 | **Inconclusive.** A1 carries the caveat. The one reading available suggests d can read high, but it is below the noise rule |
| G1b all-matrix gated-on (reported) | P = 0.21, gain 3.3: d 0.18, r 0.39, e_gate 0.42 | Did not learn either |
| G2 design power | Coverage gaps: rung 0 0.21, rung 1 0.16, rung 2 **0.137** | Rung 2 is excluded from P1 and A1. P1 becomes "2 of 2" |
| G3 identities | Hold for every split. GPU repeat noise of the log-odds and of g at W₀ is exactly 0 (two separate forwards of W₀, `run.py` noise stage: the forward is deterministic) | — |
| G4 reproduces concentration test | L(C₀,G₁) P 0.533 (ref 0.535), L(C₁,G₁) P 0.635 (ref 0.635): **pass** | The ladder's emoticon pull-back is the concentration test's Adam-all arm |

## Scores (as filed)

Pull-back at step 10, held-out windows.

| Task | rung | coverage | base P | ret_ci | d | r | i | e_gate | e_Δ | do-nothing r | gated-off r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| emoticon | E | — | — | 0.56 | 0.025 | 0.53 | 0.011 | 0.43 | 0.003 | 0.60 | 0.52 |
| r0_t352_v310 (" it"→" is") | 0 low | 0.32 | 0.15 | 0.82 | 0.052 | 0.79 | −0.027 | 0.18 | 0.004 | 0.86 | 0.85 |
| r0_t588_v320 (" will"→" be") | 0 high | 0.53 | 0.25 | 0.96 | 0.081 | 0.92 | −0.039 | 0.04 | −0.001 | 1.07 | 0.95 |
| r1_t534_v369 (" which"→" was") | 1 low | 0.31 | 0.039 | 0.79 | 0.053 | 0.75 | −0.012 | 0.21 | 0.000 | 0.75 | 0.81 |
| r1_t588_v452 (" will"→" have") | 1 high | 0.47 | 0.030 | 0.87 | 0.051 | 0.84 | −0.014 | 0.13 | 0.000 | 0.90 | 0.89 |
| r2_t275_v2667 (" in"→" Fig") | 2 low | 0.30 | 0.0034 | 0.66 | 0.045 | 0.62 | −0.006 | 0.34 | −0.002 | 0.53 | 0.64 |
| r2_t359_v816 (" we"→" just") | 2 high | 0.44 | 0.0032 | 0.34 | 0.049 | 0.30 | −0.008 | 0.66 | 0.000 | 0.28 | 0.33 |

- Every task passes the 1-nat gain rule (gains 6.4–16.9 nats). Every pull-back learned (P ≥ 0.986).
- **P1, coverage effect: NULL. Filed: pass. ⚠ Reversal.**
  - Rung 0: high − low = 0.143 against m = 0.15. That is a miss by 0.007; the sign is as predicted.
  - Rung 1: 0.079.
  - Rung 2: excluded by G2.
  - Result: 0 of 2 scorable rungs.
- **P2, rarity does not matter where covered: NULL. Filed: pass for rungs 0 and 1, uncertain for rung 2, filed as
  pass. ⚠ Reversal at rung 2.**
  - Rung 0: 0.96, and rung 1: 0.87. Both are ≥ 0.8 × emoticon's 0.565 = 0.45, so both pass.
  - Rung 2's high-coverage task reads 0.34 < 0.45. **Rung 2 is the limit reported by the rule.**
  - Under the filed rules, G2 excludes rung 2 from P1 only, so P2 scores it.
- **A1, coverage predicts d: NULL, as predicted.** Rung 0: high − low = 0.029. Rung 1: −0.002. Rung 2 is excluded. A1
  carries the G1b caveat.
- **A2, the inside route runs through re-gating: PASS, as predicted.** r + i ≥ d + 0.15 on 6 of 6 ladder tasks and on
  emoticon.
  - The weakest is r2_t359: r + i = 0.29 against d + m = 0.20.
  - **Read with the next section.** The rule compares r + i with d only. On the rung-2 tasks and on emoticon, the
    masked-off share e_gate is as large as r or larger. "Re-gating dominates" holds at rungs 0 and 1 only.

## What the numbers say

1. **The pull-back's split is the gated-off control's split.**
   - On every task, the pull-back and the gated-off arm agree within 0.06 on r and on e_gate.
   - Their direct shares differ by 0.02–0.08: the pull-back's d is 0.025–0.081, the gated-off arm's is ≤ 0.003. That
     difference is 30–100× the optimiser-seed noise of d (0.0007). So the direct share is small but real, and it is the
     one place where the pull-back is not the gated-off arm.
   - **Held fixed:** task, windows, Γ, Δ, optimiser, steps. **Varied:** which components may move.
   - This bears on P1 and A1. Coverage measures where the *first gradient step* puts energy. By step 10 the Adam
     update's effect at the trigger does not depend on where it was put. Coverage can only predict a route the update
     takes, and the direct route carries ≤ 8%.
   - Two candidate explanations, neither tested: Adam spreads the update so evenly (identity test) that the 51–95
     components Γ gates on at the trigger carry little of it; or the moved components act through the activations Γ
     reads, not through their own gated output.
2. **r is a property of the task, not of the update.**
   - On every task, do-nothing's r (W fine-tune, components unchanged) is within 0.16 of the pull-back's.
   - Across tasks, r runs from 1.07 to 0.28 and orders with the baseline (figure 2, right panel):
     - rung 0: 0.79–0.92;
     - rung 1: 0.75–0.84;
     - rung 2 and emoticon: 0.30–0.62.
   - It does not order with coverage. Rung 2's higher-coverage task has the *lowest* r.
   - Rungs 0 and 1 overlap (the rung-1 high task out-reads the rung-0 low task). The two rung-2 tasks have matched
     baselines (0.0034, 0.0032) but r differs by 0.32. So rarity separates rungs 0–1 from rung 2. Within rung 2 neither
     rarity nor coverage explains the spread; cause not examined.
   - The masked model at W₀ also gives the target more probability where the target was common. P under L(C₀,G₀) is
     0.19–0.26 at rung 0, 0.03–0.04 at rung 1, and ≈ 0 at rung 2 and emoticon.
   - This is the plan's **rarity null**. Together with the recruit test's (d) (r is about 0.55–0.60 for `o` and for the
     different-target nulls), r reads as "how much of this behaviour the existing gate patterns can already produce".
     Γ's own coverage of the gradient does not come into it.
   - 7 tasks with two per rung: an ordering, not a fitted trend.
3. **Where r is small, e_gate takes over, not d.**
   - At rung 2 and on emoticon the masked-off share is 0.34–0.66 at step 10, and it grows by step 30 (r2_t359: 0.75;
     r2_t275: 0.58).
   - So for the rarer targets the frozen-Γ masked model loses the behaviour. The update sits in components Γ masks off
     at the trigger.
   - In the amendment's branch table this is the "frozen Γ is the bottleneck" branch, with rarity as its trigger.
4. **The split is stable.**
   - Across optimiser seeds: d_opt ≤ 0.013 per share.
   - Across steps 10 → 30: shares move ≤ 0.25, in the direction of more e_gate on the rare tasks.
   - So the brittleness reading is out, on this evidence.

## Reported, not scored

- **Step 30** (pull-back), retained_ci in task order: emoticon 0.55, r0_t352 0.79, r0_t588 1.02, r1_t534 0.73,
  r1_t588 0.89, r2_t275 0.43, r2_t359 0.24. The ordering is the same as at step 10.
- **Side effects.**
  - kl_other is the mean KL at the 40 non-trigger positions, CI-masked. At step 10 it is 0.41–0.46 on most tasks,
    against about 0.40–0.49 for the do-nothing masked model. The exceptions are r2_t275, at 0.82 (0.75 for gated-off),
    and r2_t359, at 0.51. At step 30 both are back to 0.50 and 0.41. Cause not examined.
  - Pile ratios (pull-back step 10 ÷ W₀, 2 × 128 val rows):
    - r0_t588: CI-masked 1.02, stochastic 1.00, PGD 1.09, unmasked 0.98;
    - r2_t359: CI-masked 1.11, stochastic 1.04, **PGD 1.53**, unmasked 0.99.
- **L0 at the trigger** (alive c with ḡ > 0.5): 51, 78, 63, 78, 58, 95 in ladder-task order. max ḡ is ≥ 0.994 on all
  of them. Neither orders with r.
- **Window-context baseline beside the mined one.** The rung is set by the mined full-row baseline; the windows have
  41-token context.
  - r0_t352: 0.026 (mined 0.15);
  - r0_t588: 0.195 (0.25);
  - r1_t534: 0.0039 (0.039);
  - r1_t588: 0.022 (0.030);
  - r2_t275: 0.00001 (0.0034);
  - r2_t359: 0.0010 (0.0032);
  - emoticon: 0.0066 (window only).
  - Under window context, r1_t534 and r0_t352 sit a rung lower and r2_t275 two rungs lower. Ordering r by the window
    baseline does not change the reading.
- **W fine-tune** at step 10: P(target) 0.97–0.999, rel_norm_max 0.011–0.013, kl_other 0.05–0.20 (r2_t275 the highest).
- **PGD** (both objectives). retained_pgd_all and retained_pgd_trigger are −9.5 to −15 on every ladder state, do-nothing
  included. The trigger log-odds under PGD are −50 to −85, and the harness spread on emoticon is 0.28 and 0.78. These
  are numbers only, with no interpretation here. **Hand-off for the PGD check (next session):** at W₀, why do both
  shared adversaries drive the trigger log-odds to −50 to −85 nats, while the Pile PGD KL is 0.69? And does
  retained_pgd mean anything at a trigger where the adversary already saturates? Files: `src/commute/local_faith.py` (the
  PGD variants), `results/large/code_as_ran/`, `ladder.json` → `tasks.<t>.base.pgd_*`, and the existing
  `experiments/transport_test/check_pgd.py`.
- The switched-on set at the trigger and its Jaccard across tasks were not computed by `run.py` (see Not run).

## Layout note

Cleaned 2026-10-19 (user request). The nested download copy `results/results/` (byte-identical JSONs) and the smoke
outputs (`*_smoke.json`, step-2 states) were deleted. So were the 96 candidate-task windows not used by the run, and
`pullback_seed0/state_step10.pt` (byte-identical to `pullback/state_step10.pt`). The trained states exist only for
emoticon: the six ladder tasks' state directories were empty, i.e. never saved or never downloaded. All of
`results/large/` (emoticon states, code as ran, logs, the six tasks' windows) is on the cluster at
`/mnt/nw/home/e.agnew/commute/experiments/task_ladder/results/large/`. Locally only the small files are kept;
`cluster/sync.sh large` fetches the rest.

## What changes next

No next experiment is chosen. Any proposal goes through `/design` first. What this run implies:
- **Coverage is dropped as a predictor.** It cannot separate tasks while changed subcomponents carry ≤ 8% of the gain.
- **The first open question is what r is.** Every CI-masked number in this project since the concentration test is
  dominated by r, and two readings fit this run equally well:
  - *mechanism*: the trained model's activations make Γ assign high CI to existing subcomponents that produce the
    target, so the decomposition explains the new behaviour as existing subcomponents used in a new context;
  - *leak*: Γ reads the trained model's activations, which at the trigger already encode the target, and any mask
    computed from them pushes the CI-masked model toward the target, whether or not those subcomponents mean anything
    for it.
  - The recruit test found the high-CI sets target-specific (Jaccard 0.04 between targets), which leans to
    "mechanism", but it did not test the leak directly. Until this is settled, a high r is not evidence that the
    decomposition stays valid, and a low r at rung 2 is not evidence that it fails.
- **Only after that**: why rare targets end up masked-off. Two directions were named in the amendment:
  - letting Γ train along with the subcomponents, so the CI values themselves can change;
  - restricting the pull-back to the subcomponents with high CI at the trigger, to see whether changed subcomponents
    can carry a rare target at all. This arm's control (G1b) did not learn in 10 steps here.
- Rungs 3 and 4 (rarer targets) would only extend an ordering whose meaning is open.
- The backlog's "define recruit/create in VPD terms" now has a constraint: r is about the same for three different
  training methods, so "recruit" cannot be defined as "r is high".

## Not run

- Rungs 3 and 4 (rare, never).
- LoRA through the pull-back (definitions (i) same matrix, (ii) same rank).
- SGD or sparse pull-backs on the ladder tasks (the concentration test ran them on emoticon only).
- Γ updating; the Γ-parameter route.
- Unfrozen-Δ pull-back (the arm that makes e_Δ informative).
- A gated-on positive control that learns (G1b, both variants, did not).
- The PGD check above, and a per-window PGD adversary.
- Switched-on set (dg > 0.3) and its cross-task Jaccard: in the amendment's reported list, not implemented in `run.py`.
- Optimiser-seed spread on any task other than emoticon.
- Coverage for the emoticon task.
