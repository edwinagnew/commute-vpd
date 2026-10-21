# Appendix to `2026-10-21_review_sgd-finding-influence.md`: the four audits as returned

*Four read-only Opus subagents, 2026-10-21, each on one slice. Working material; the review report is the summary.*

---

# Slice: reports

# Audit: prose files that depend on "no SGD learning rate works" (slice: reports ≥ 2026-10-15, decisions, backlog, GOALS, README, theory, TRIED)

Read-only audit, 2026-10-21. Sources read in full: `reports/2026-10-15_analysis_concentration-test.md` (E4),
`reports/2026-10-21_analysis_sgd-grid.md` and its plan. Grep over the slice for sgd / adam / optimi[sz] / momentum /
per-coordinate / diffuse / spread / rate search / sign-like / matched size, every hit read in context.

**One caveat that changes how every fix should be worded.** The sgd-grid comparison is not matched for size of change:
SGD at 0.01 has a largest relative change per matrix of 0.0026, Adam at 1e-5 has 0.0066 (sgd-grid report line 69-70).
So "SGD does less damage than Adam" and "SGD is ten times less diffuse" are both at a smaller SGD update. A fix should
say "at a smaller size of change" and should not assert that Adam *causes* the diffuseness. The sgd-grid report's own
"Adam's per-coordinate scaling explains part of the diffuseness" (line 62) already goes a step past its evidence.

**Second caveat.** E2's SGD *fine-tunes* (W, not U V) ran at 0.1–1 and reached off-target KL 0.27–6.6
(`reports/2026-10-14_analysis_lift-test.md`:89). That is the same "grid started too high" pattern as E4, and nobody has
corrected it. It matters for two items below (TRIED E2 row, backlog 2026-10-21 check (1)).

**House rules applied.** Dated reports "stand as written" (`README.md`:99-102), concerns about existing claims go in a
review report, not inline (`.claude/CLAUDE.md` Rules); filed predictions are never edited. So for dated reports the
action is "list in a review report" (suggested: `reports/2026-10-21_review_sgd-correction.md`) with the replacement
sentence given for that review to quote. There is one precedent for a dated inline correction (E4 report line 71,
"corrected 2026-10-16"), so a one-line pointer in the E4 header is the user's call. `GOALS.md` is user-owned: proposals
only. Living files (`backlog.md`, `TRIED.md`, `README.md`, `theory/`, `decisions.md`) can take text fixes.

---

## WRONG (contradicted by sgd-grid)

| # | file:line | Quoted | Asserts | Action |
|---|---|---|---|---|
| W1 | `backlog.md`:232-233 | "(1) plain-SGD component training at the fine-tune's matched size vs the adjoint state of the SGD fine-tune, same card (E4 found no SGD rate that learned without large off-target damage, so this needs a rate search first);" | Component-training SGD needs a rate search before use | Text fix: "(1) plain-SGD component training (rate 0.01 works on emoticon, E4 follow-up) at the SGD fine-tune's size of change, against the adjoint state of that fine-tune, same card. The SGD fine-tune itself still needs a rate check: E2's ran at 0.1 to 1, with off-target KL of 0.27 or more." Needs a re-run (R5). |
| W2 | `backlog.md`:229-231 | "the optimiser (Adam's per-coordinate scaling breaks the K structure), the per-step loss re-reading the changed pieces, or the scale c. That points at the optimiser building cancelling changes." | Attributes the gap to Adam with SGD unavailable to test it | Now testable, not yet tested. Text fix to the last sentence: "Plain SGD now learns emoticon within the limit (E4 follow-up), so the optimiser branch can be tested directly; until then nothing points at it more than at the other two." Re-run R5. (The chain of argument in the decision report it came from is another agent's.) |
| W3 | `theory/commuting_updates.md`:221-223 | "Every run of this operation used Adam for 10 to 300 steps, so the SGD step analysed below describes none of them exactly." | No SGD runs of component training exist | Wrong twice: E4's SGD arm (lr 0.3–30) and the E4 follow-up (lr 0.001–0.1, 30 steps) are plain SGD, and E16 ran Adam to 1000 steps. Text fix: "Most runs of this operation used Adam, for 10 to 1000 steps, and the SGD step below describes them only loosely. E4's SGD arm and its 2026-10-21 follow-up ran plain SGD for 30 steps, which is this step repeated." |
| W4 | `reports/2026-10-19_decision_success-criterion.md`:53-55 | "**Every run used Adam on (U, V)**, not SGD (`pullback.train`): the concentration test found no SGD lr inside the off-target limit." | The reason for Adam everywhere was that SGD did not work | Reason now wrong (and E4 had an SGD arm). Dated: list in the review report. Replacement: "Every scored run used Adam on (U, V). E4's SGD grid started too high; at 0.003 to 0.1 SGD learns within the limit (E4 follow-up, 2026-10-21), so Adam was a choice, not a necessity." |
| W5 | `reports/2026-10-19_review_update-map-variants.md`:68-69 | "The SGD version is practically blocked: the concentration test found no SGD lr for the pull-back within the off-target limit." | Variant (b) with SGD cannot be run | Wrong reason. Variant (b) follows the fine-tune's trajectory, so what it needs is a working SGD *fine-tune* rate, not a component-training one; that has never been found either (E2: KL 0.27–6.6). Dated: review report. Replacement: "The SGD version needs an SGD fine-tune at a rate that learns cleanly; E2's grid (0.1 to 1) had none, and smaller rates were never tried." |
| W6 | `reports/2026-10-15_analysis_concentration-test.md`:11-12 | "Plain SGD found no learning rate that learns without large off-target damage, so the "Adam causes the diffuseness" prediction is unscorable." | No SGD rate works | Wrong as a general claim; true for the grid 0.3–30. Dated report, already answered by a dated report (sgd-grid). Action: list in review report; optional one-line pointer under the takeaway, e.g. "*2026-10-21: the grid started too high; at 0.003 to 0.1 SGD learns within the limit and P1 scores fail, as predicted (`reports/2026-10-21_analysis_sgd-grid.md`).*" |
| W7 | same, :43-45 | "**P1 (SGD concentrates): unscorable.** No SGD learning rate met the rule. ... Higher learning rates damage the model or diverge." | Same | Same action as W6 (P1 is now scored: fail, 0.156, as predicted). |
| W8 | same, :83 | "Whether Adam causes the diffuseness. SGD did not learn within the rule." | Same | Same as W6. The question is now partly answered, and only at unmatched size of change. |
| W9 | `experiments/TRIED.md`:24 (E4 row) | "SGD found no lr within the off-target limit (P1 unscorable)." | Same | Text fix (living file): "SGD found no lr within the off-target limit in 0.3–30; corrected by the E4 follow-up row (0.003–0.1 learn; P1 fail, as predicted)." |
| W10 | `reports/2026-10-20_review_terminology.md`:109-110 | "Theory section 5.3 characterises the pull-back as one SGD step, −η K(∇ℓ); every run used Adam for 10 to 300 steps." | No SGD component-training runs | Wrong at the time (E4's SGD arm). Dated: review report. This is the source of W3. Row A at line 21 ("Adam, many steps", E2–E6) has the same omission. |

## WEAKENED (argument relied on SGD being unavailable, or generalises an Adam result to "component training")

| # | file:line | Quoted | Assumes | Action |
|---|---|---|---|---|
| K1 | `GOALS.md`:50 (Terms, Component training) | "train the alive (U, V); Adam, many steps." | The operation is Adam by definition | **Propose to the user** (user-owned): "train the alive (U, V) on the loss; Adam in most runs, plain SGD in E4 and its follow-up (rate 0.01 works on emoticon)." Whether the optimiser is part of the definition is the user's decision; it matters because "brittle" in GOALS is "unstable across small changes to Φ" (line 99). |
| K2 | `README.md`:31-34 | "Component training ... learns every task tried ... but its decomposition passes VPD's adversarial check on 3 of 7 next-word tasks, sits at the line on 1 and fails on 3 (E11), and fails for 32 new associations where 1 passes (E10)." | Adam results stated as properties of component training | Text fix: "Component training with Adam ... (E11) ... (E10). Plain SGD also learns emoticon, with less off-target damage at a smaller size of change (E4 follow-up); its faithfulness has not been measured." Re-run R1/R2 decide whether the failures are Adam's. |
| K3 | `README.md`:70-86 (codes table) and "State of play (2026-10-20)" | The table ends at E15; no E4 follow-up, E16 is there but E17–E19 and the sgd-grid are not | — | Text fix: add a row "E4 follow-up \| Plain-SGD component training at small rates \| `experiments/sgd_grid` \| `reports/2026-10-21_analysis_sgd-grid.md`". Low. |
| K4 | `backlog.md`:101-103 | "the identity test showed Adam component training spreads its update over thousands of components, so a per-component comparison is moot" | Every component-training update is diffuse | For SGD the top 100 remove 0.48 of the gain, so a per-subcomponent comparison against Γ's values is no longer moot. Text fix, append: "For plain SGD (E4 follow-up) the top 100 carry about half the gain, so the per-subcomponent version is worth doing on that state." |
| K5 | `experiments/TRIED.md`:23 (E3 row, "Why we moved on") | "open whether that is Adam or the task" | — | Now partly answered. Text fix, append: "Partly answered by the E4 follow-up: SGD's update is ten times less diffuse at a smaller size of change, still not concentrated." |
| K6 | `experiments/TRIED.md`:22 (E2 row) | "The outside the subcomponents part of a fine-tune is ~30% input-weighted ..., Adam or SGD alike" | E2's SGD fine-tunes were a fair SGD reference | They ran at 0.1–1 with off-target KL 0.27–6.6 (E2 report :89), the same too-high grid as E4. Text fix, append: "(SGD at 0.1 to 1 only, all with large off-target damage; a cleaner SGD rate is untried.)" UNKNOWN whether a clean SGD fine-tune agrees; re-run R6. |
| K7 | `reports/2026-10-19_analysis_task-ladder.md`:140-142 | "Two candidate explanations, neither tested: Adam spreads the update so evenly (identity test) that the 51–95 components Γ gates on at the trigger carry little of it; or the moved components act through the activations Γ reads" | — | Dated; nothing to edit. The first explanation is now testable on the SGD state: if d stays ≤ 0.08 with a ten-times less diffuse update, it loses. Re-run R3. |
| K8 | `reports/2026-10-16_prediction_ladder-routes-amendment.md`:97 | "**Predicted: null.** The emoticon pull-back's d is ≈ 0.03: Adam spreads the update, so d is small everywhere." | — | Filed prediction: never edited. Its rationale is testable by R3. Nothing. |
| K9 | `reports/2026-10-15_analysis_concentration-test.md`:79, :132 | "The diffuse Adam solution is the poor one at the trigger." / "Under either gating, the diffuse Adam update is the worst." | Diffuse ↔ poor at the trigger, with no less-diffuse all-subcomponent arm that learned cleanly | Still true of Adam. Note that amendment 1's SGD-all row (lr 0.3, damaging) read 0.19 under W₀ gates, best after the siblings. Whether the clean SGD state does better at the trigger is UNKNOWN. Re-run R2. Nothing to edit. |
| K10 | `reports/2026-10-20_analysis_adversarial-vs-stochastic.md`:59 | "bad: the update is not "a small change to a few subcomponents" but a spread of opposing changes that only reconstructs in full" | "The update" = component training's update in general | Measured on Adam states only. Dated; nothing to edit. Whether SGD's less diffuse update is also adversarially fragile on the failing tasks is R1. |
| K11 | `reports/2026-10-21_analysis_split-comparison.md`:35-36 | "free component training of all alive subcomponents, lr 1e-5. Paired with the LoRA: rank-8 component training, lr 1e-4." | Optimiser not named (Adam, per E2's recipe) | Now ambiguous since SGD is a working arm. Dated but fresh; review report or a two-word fix "(Adam)". Low. Its "component training is the only operation with both" (E16) holds for Adam only. |
| K12 | `reports/2026-10-21_decision_regating-main-line.md`:80-81 and `reports/2026-10-21_plan_cancellation.md`:97-99 | "many Adam steps of component training fail on half the tasks" / "Adam's step is sign-like and a ten-step gradient on a narrow task is low-rank ..." | — | In my slice by date but assigned to another agent (chain of argument). Noted only: both describe Adam states accurately; E17 has no SGD state among its arms, which is the cheap contrast (R4). |
| K13 | `reports/2026-10-15_plan_task-ladder.md`:46-48, :87, :111-112 | "the identity test showed the Adam pull-back spreads its update over thousands of components, so the functional CI-masked measure at the trigger *is* the Γ comparison. SGD and sparse pull-backs are the concentration test's" | Scoping that leaned on SGD being E4's job | Dated plan, correct about what it ran. Nothing; R3 is the follow-up it deferred. |

## STILL FINE (SGD-agnostic, accurate about Adam runs, or already updated)

| file:line | Quoted (short) | Why fine |
|---|---|---|
| `experiments/TRIED.md`:38 (E4 follow-up row) | "SGD learns within the off-target limit at 0.003–0.1 ... not matched to Adam's size of change" | Accurate, and carries the size caveat. |
| `reports/2026-10-15_plan_concentration-test.md`:7-9, :20, :39 | "Two readings: Adam spreads it (an optimiser artefact) ..." | Plan as filed; predictions now scored by the follow-up. |
| `reports/2026-10-15_analysis_identity-test.md`:51-52, :58 | "It does not say whether the diffuseness comes from Adam ..." | Correctly says "not known"; partly answered later. |
| `reports/2026-10-15_analysis_concentration-test.md`:92 | "SGD with a finer lr grid (0.03–0.3) or momentum, to score P1." | The Not-run item the follow-up executed (at 0.001–0.1); momentum still open. |
| `reports/2026-10-15_prediction_concentration-amendment.md`:19, `..._gating-amendment.md`:10 | "A3, Adam on all subcomponents ..." | Adam arms, accurate. |
| `reports/2026-10-19_decision_success-criterion.md`:44-46, :89 | "One SGD step moves the weights by K(G) ..." / "one Adam step at lr 1e-3 already gives P(o) 0.93" | Algebra and an Adam fact. |
| `backlog.md`:57-64 | "The plain SGD step of component training is δW = −η K(G) ..." / "avoids the Adam bookkeeping problem" | Algebra; optimiser-agnostic. |
| `backlog.md`:71-72, :89, :188 | "a free optimiser puts about 30% ... there"; "LoRA (or the optimiser) is inherently noisy"; "E2 Adam and SGD" | Factual or an open question; see K6 for the SGD caveat behind :188. |
| `theory/commuting_updates.md`:225-250 (Prop. 2, consequences) | "weight-faithful by construction, since Δ is fixed" | True for any optimiser (Δ fixed); the K structure is SGD-only, as stated. |
| `theory/reveal_vs_create_literature.md`:140 | "Breadth is a side effect of the optimiser" | Literature summary, unrelated. |
| `reports/2026-10-20_plan_new-vs-existing-signatures.md`:41 | "N5. SGD is free in which directions it uses" | "SGD" in the generic sense. |
| `reports/2026-10-20_analysis_gamma-refit.md`:84-85, :173, :202, :272-274, :285; `reports/2026-10-21_plan_gamma-refit-protocol.md`:13-37, :96-117; `reports/2026-10-20_plan_gamma-refit.md`:128-134 | "Adam's first update is then a sign step ..." | About Γ's AdamW, not component training. |
| `reports/2026-10-21_plan_step-split.md`:15-16 | "take the actual weight step S_t ... (the Adam step, not −ηG)" | The user's specification; not chosen because SGD was unavailable. |
| `reports/2026-10-19_review_update-map-variants.md`:41, :43, :56-62, :95, :109-112 | "pull-back (Adam, Δ and Γ fixed)"; "measures Adam's appetite" | Accurate about Adam runs; only :68-69 is wrong (W5). |
| `reports/2026-10-19_analysis_reading-a-coverage.md`:1, :15, :21, :94, :110 | "the Adam pull-back" | Names the optimiser correctly. |
| `reports/2026-10-20_review_research-log-readability.md`:50 | "The apparatus gives "Adam at 1e-5" as if fixed." | About rates per variant, still a valid fix request. |
| `reports/2026-10-19_plan_lora-lift-and-pullback.md`, `..._plan_key-value-task.md`:31, `..._plan_split-comparison.md`:16, :27, `..._analysis_lora-split.md`:16, `..._analysis_task-ladder.md`:19, :52, :88, :136, :166, :240, :246, `..._analysis_recruit-test.md`, `..._plan_recruit-test.md`, `..._review_obstruction-claim.md`:12, `..._review_faithfulness-scope.md`:41, :66, `..._analysis_key-value-task.md`:130, `..._analysis_lora-plateau.md` | recipes ("Adam lr 1e-5"), optimiser seeds | Recipe descriptions or seed noise; no SGD dependence. |
| `decisions.md` (all rows) | — | No row fixes the optimiser or relies on SGD failing. Optional, Low: add a row "Component training uses Adam lr 1e-5 by default; plain SGD at 0.01 is a working alternative on emoticon (E4 follow-up)" so the choice is on record. |

## UNKNOWN (needs a run to tell)

| Claim, where | What is unknown |
|---|---|
| Component training fails the adversarial check on 3 of 7 tasks (E11; README :31-34, TRIED E11) | Whether SGD component training, with lower off-target KL, also fails. E11 found the PGD excess "orders with component training's kl_other" (TRIED :31). R1. |
| "The diffuse Adam update is the worst at the trigger" (E4 :79, :132) | Whether the less diffuse SGD state is carried by its components under W₀'s gates. R2. |
| d ≤ 0.08 everywhere because "Adam spreads the update" (ladder :140, amendment :97) | Whether d rises for SGD. R3. |
| Cancellation as the failure mode of component training (adversarial-vs-stochastic :59, E17) | Whether SGD's state cancels. R4. |
| The "unused contrast" (backlog :226-235) | Whether SGD component training stays at the floor like the adjoint state. R5. |
| "Adam or SGD alike" outside share (TRIED E2) | Whether an SGD fine-tune at a clean rate gives the same ~30%. R6. |

---

## Re-runs proposed

Cost assumptions: MATS cluster `debug` ≤ 30 min ≈ $0; RunPod 4090 ≈ $0.5/h; H100 ≈ $2.5/h. The SGD state of the
follow-up is not on local disk (`experiments/sgd_grid/results/large/` is empty here); it may be on the cluster NFS, else
it regenerates in about 3 min (job 28106). Every SGD arm on a new task needs E4's rate rule per task (5 rates × 30
steps, about 3 min on emoticon, job 28106).

| # | Run | Based on | Runtime | Cost | Importance |
|---|---|---|---|---|---|
| R1 | SGD component training (rate 0.01, plus E4's rate rule per task) through E11's harness (Pile stochastic, adversarial, CI-masked ratios, sparsity, same-job W₀ floors) on emoticon, then the five ladder tasks | E11 emoticon rerun, cluster job 25863: 12.5 min; E11 five tasks: RunPod L40S 38 min, $0.70 | emoticon ~15 min; five tasks ~45–55 min (adds the per-task rate grid) | emoticon $0 (debug); five tasks ~$0.45 on a 4090, or $0 as two ≤ 30 min debug jobs | **High.** Decides whether "component training fails the weak version on 3 of 7 tasks" is a property of the operation or of Adam; that is the main evidence against component training as Φ (README state of play, reading A). |
| R2 | E4 amendments 1 and 2 (W₀-gated CI-masked P(o) at the trigger, switched-on set) on the SGD state | E4 amend1 RunPod 6 min $0.10; amend2 ~1 min | ~10 min | $0 (debug) | Medium. Tests whether a less diffuse all-subcomponent update is carried more by its own components (strengthen vs reveal, GOALS mechanism hypotheses). |
| R3 | Route split (d, r, i, e_gate, e_Δ) on the SGD state, emoticon (then ladder tasks with R1) | Ladder route-split stage; emoticon cost unrecorded in the slice, est. ≤ 10 min from E4 amend timings | ~10 min emoticon | $0 | Medium. Tests the ladder's untested "Adam spreads the update so d is small" explanation (K7, K8). |
| R4 | E17's cancellation statistics (κ, half revert, F_adv) on the SGD state | E17 plan: ~40 min for nine states on an L40 (~4–5 min per state) | ~5–10 min | $0 | Medium. A ready contrast for E17's Adam states; bears on "cancellation is how component training fails" only if R1 shows SGD failing somewhere. |
| R5 | Backlog check (1), the unused contrast: SGD fine-tune rate check (0.001–0.1), then SGD component training at its size of change vs the adjoint state of that fine-tune, Pile harness, emoticon | sgd-grid job 28106 (~3 min) + E11 harness (12.5 min) | ~20–25 min | $0 | Medium–High for the re-gating line's decision (the decision report leans on this contrast; another agent audits that chain). |
| R6 | E2's outside share (input-weighted) on SGD fine-tunes at 0.001–0.03 | E2 job 20281 (runtime not recorded in the slice); sgd-grid ~3 min for 5 rates | est. ~10 min | $0 | Low. Only the TRIED E2 "Adam or SGD alike" line depends on it. |
| R7 | Matched size of change: SGD vs Adam at equal largest relative change (or equal ‖δW‖), top-10/100 revert and KL elsewhere; plus SGD with momentum 0.9 | sgd-grid job 28106, ~3 min | ~5–10 min | $0 | Medium. Without it, "Adam causes part of the diffuseness" (sgd-grid :62) and "SGD does less damage" are confounded with update size; every text fix above is hedged because of this. |
| R8 | SGD component training at 1000 steps through E16's harness (steps 10/100/1000) | E16 scored run 2118 s on H100 for all operations; component training is one arm | est. 15–25 min on an L40 | $0 (debug), or ~$0.5–1 on an H100 | Low–Medium. Extends E16's "component training keeps both" to SGD; E16's claim is Adam-only. |

Suggested order: R7 and R1-emoticon first (both cheap, on the cluster), since R7 sets how the comparison is worded and
R1 is the one that can change the main verdict; then R1 on the ladder tasks, then R2/R3, then R5.

---

# Slice: experiments

# Audit: which experiments depend on the optimiser (experiments and code slice)

2026-10-21. Read-only audit of `experiments/*/run.py`, the predictions and result headers, `src/commute/*.py`,
`cluster/*`. Prompted by the SGD-grid correction (`reports/2026-10-21_analysis_sgd-grid.md`).

## Facts that every row below depends on

- **Component training is Adam everywhere except E4's `sgd_all` arm and the SGD grid (E4 follow-up).**
  - The shared trainer defaults to Adam: `src/commute/component_training.py:110`, `optimizer: str = "adam"`.
  - Only `experiments/sgd_grid/run.py:107` passes `optimizer="sgd"`.
  - E2, E3 and E4 have their own Adam loops: `lift_test/run.py:601`, `identity_test/run.py:155` and
    `concentration_test/run.py:312-313`. E4's loop also has an SGD branch.
  - No `run.py` has a command-line flag for the optimiser (grep of `add_argument`).
- **Every fine-tune is Adam at 3e-5, except E2's SGD control.**
  - The Adam fine-tunes use `ArmAConfig` defaults (`finetune.py:58,64`).
  - E2's control fine-tunes with SGD at 0.1, 0.3 and 1 (`lift_test/run.py:357-371`, `optimizer="sgd"`).
  - Every LoRA is Adam, and that is hard-coded: `finetune.py:269`. `LoraConfig` has no optimiser field.
- **Adam's component-training rate of 1e-5 was chosen on emoticon (E2) and then reused with no rate search.**
  - Reused in E5, E10, E11, E15 and E16: `task_ladder/run.py:46`, `kv_task/run.py:44`,
    `reading_a_coverage/run.py:42`, `gamma_refit/run.py:49` and `split_comparison/run.py:42`.
  - On E11's ladder tasks this rate overshoots. The step-10 off-target KL ("kl_other") of component training is 0.15
    to 0.51. On emoticon it is 0.016, the fine-tune's is 0.08 to 0.20 on the same tasks, and E4's limit is 0.110
    (`reading_a_coverage/results/coverage.json`, `heldout.kl_other`).
  - The adversarial (PGD) ratio follows that KL. For component training the pairs are:

    | Task | PGD ratio | kl_other |
    |---|---|---|
    | r2_t275 | 6.74 | 0.51 |
    | r0_t352 | 2.73 | 0.20 |
    | r1_t534 | 2.24 | 0.23 |
    | r2_t359 | 1.55 | 0.19 |
    | r1_t588 | 1.16 | 0.15 |
    | emoticon | 1.03 | 0.016 |

  - **So an SGD win on these tasks would mix the optimiser with step size.** Every SGD arm on them needs an Adam arm
    at lower rates (for example 1e-6, 3e-6 and 1e-5) beside it. Otherwise the cause is not separable.
- **"Plain SGD" here is normalised gradient descent.**
  - Clipping is at 1.0 (`component_training.py:144`). The gradient norm is 1.8 to 203 at every step of every SGD run
    (`sgd_grid/results/sgd_grid.json`, curves).
  - So each SGD step moves (dU, dV) by exactly the learning rate in parameter norm. The direction is the gradient's
    projection K(G), but the step length does not depend on the task.
  - Consequence 1: a rate carries over between tasks as a step length. Rarer tasks need a longer path, so they still
    need a rate search. A grid of {0.003, 0.01, 0.03, 0.1} is suggested.
  - Consequence 2: the description "plain SGD, ΔW is a sum of clipped gradients" (`finetune.py:64`) is accurate, but
    "plain SGD" in prose hides the normalisation.
- **Card floor.**
  - On the L40 the W₀ adversarial seed floor is about 0.21:
    - E8: "W₀ PGD floor 21% on the L40";
    - E11's emoticon job 25863: 0.21;
    - `cluster/logs/commute-step_split-28108.out`: "W0 PGD floor 0.2088 (stop above 0.15)". E19 stopped there.
  - With a 0.21 floor the "within 3 floors" line is about 1.63. The H100 read f = 0.075 (E16).
  - So on the L40 an SGD-against-Adam comparison where both sit near 1 (emoticon, E16) cannot be told apart.
    Failing states at 2.2 to 6.7 (E10, E11) can be.
- **Measured costs used in the projections.**
  - Component training costs about 0.1 s per step on an L40, held-out evaluation included (`commute-sgd-grid-28106.out`:
    150 steps over 5 rates in 17 s).
  - One Pile-harness state takes about 2 min on an L40 or L40S:
    - job 25863: about 70 s per 128-row block, two blocks;
    - E11: 241 s per task for 2 states (`coverage.json`, `seconds`).
  - The W₀ floor over 3 seeds took 435 s (job 25863).
  - Assumed prices: cluster debug $0; RunPod 4090 about $0.5/h; H100 $2.5 to $4/h (E15's secure H100 was $3.99/h,
    `reports/2026-10-20_analysis_gamma-refit.md:383`).

## Table

"Fine-tune" means the reference update on the full weights. "CT" means component training (the map).

| Exp | Optimiser(s) used | Exposed? | Why (claim and number) | Proposed SGD arm (everything else fixed) | Original runtime (source) | Projected SGD-arm runtime and cost | Importance |
|---|---|---|---|---|---|---|---|
| E2 lift_test | Fine-tune: Adam 3e-5 (`lift_test/run.py:669`), plus an SGD control at 0.1/0.3/1 (`:357-371`). LoRA: Adam. CT: own Adam loop, lrs 1e-4–1e-3 (`:681`), then 1e-5–3e-5 (`pullback2`, `:720`) | **Not exposed.** Only the fine-tune side was tested | "Inside works": Adam CT reads PGD 1.03/1.07 at steps 10/30. A pass needs no rescue. The obstruction (P1 reversed) was measured for Adam and SGD fine-tunes alike, so the optimiser was already tested there | None. The SGD CT state's Pile check belongs to the E4 row | Cluster jobs 20266, 20268, 20281, 20288 (7.5 min), 20289 (31.5 min), from log timestamps | n/a | Low |
| E3 identity_test | CT: Adam 1e-5, own loop (`identity_test/run.py:43,155`). Fine-tune: Adam (`:186`) | **Exposed, mostly answered** | "Top 10 remove 1.6% of the gain; open whether that is Adam or the task" (TRIED E3). The SGD grid answered it on emoticon: SGD's top 10 remove 0.156. The identity measure itself has no power (changes are 0.2–1.3% of u vᵀ, and the null scores the same) | None. The identity measure cannot read an SGD state any better | RunPod 4090, $0.35, about 9 min (report header) | n/a | Low |
| E4 concentration_test | CT: Adam all (1e-5); SGD all at 0.3–30 (`concentration_test/run.py:52,55`); Adam top-1/10/100/siblings at 1e-5–1e-3 (`:51,56-59`) | **Exposed** | P1 (`:109`, "Adam is why…") was scored by the SGD grid: it fails as predicted. Still unmeasured for SGD at a working rate: Pile ratios; trigger-local CI-masked P(o) (Adam-all 0.635); P(o) under W₀ gates (amendment 1: Adam-all 0.006, SGD at the failing 0.3 rate 0.19); the switched-on set (amendment 2). Those are what support "under either gating the diffuse Adam update is the worst" | Use the SGD grid's saved state at 0.01, step 30. Measure it with the Pile harness (3 W₀ seeds), E4's trigger-local CI-masked P(o) under state gates and under W₀ gates, and the switched-on set and recovery. Code: new stage or experiment, because `concentration_test/run.py` must stay as it ran. No rate search needed | 23 min, $0.88, 4090. Amendment 1: 6 min, $0.10. Amendment 2: about 1 min (report header) | About 11 min on cluster debug, $0 (7 min floor + 2 min state + about 1 min trigger-local). The L40 floor of about 0.21 blurs Pile readings near 1. Reading the trigger-local numbers is fine | **Medium.** It is the control arm for every SGD rerun, and it re-scores "the diffuse Adam update is the worst" |
| E5 task_ladder | Fine-tune: Adam 3e-5. CT: Adam 1e-5 on all alive and gated-off (`task_ladder/run.py:46-47,211`). **The coverage axis is defined from an SGD step** (`src/commute/ladder.py:9-13`) | **Exposed** | (a) The x-axis is the share of an *SGD* first step's energy in gated-on subcomponents, but the arm trained with Adam: a definition mismatch. (b) A1 was predicted null *because* "Adam spreads the update" (`predictions_amend1.json`, A1). (c) "Changed subcomponents carry 0.03–0.08 of the gain" (d) is a property of a diffuse Adam update. SGD at 0.01 is ten times less diffuse on emoticon. P1 and A1 are both null, so d and coverage could move under SGD | SGD CT at step 10 on the same 6 tasks plus emoticon. Rate grid 0.003–0.1, chosen by "P ≥ 0.5 learned, least kl_other". Report d, r and e_gate with the route split, and coverage against d. Same job as the E11 arm below: `harness.measure` gives routes and Pile together | UNKNOWN: RunPod, "tally $3.69, may include other pods" (report header) | Shared with E11 (below) | **Medium.** Bears on Result 3 ("the moved subcomponents never carry the behaviour") and on whether coverage predicts d |
| E6 recruit_test | Fine-tune: Adam 3e-5. LoRA: Adam 1e-3. CT reference: Adam 1e-5, 30 steps (`recruit_test/run.py:41,413-414`) | **Barely exposed** | The main claims are about do-nothing placements of finished Adam updates. The only CT number is J(ft, CT) = 0.64 at the trigger, against a seed floor of 0.93 | Optional: J(ft_s0, SGD CT) through E6's set code. Do it only if the ladder SGD arm shows SGD changes r or the switched-on set | RunPod 4090, $0.26, 106 s (report header) | About 3 min, cluster, $0 | Low |
| E7 lora_commute | LoRA: Adam 1e-3 (`lora_commute/run.py:53`). Fine-tune: Adam 3e-5. CT: Adam, rank-r and free on one matrix plus all 24 (`:90-91` pb_lrs 1e-5–1e-3; `:424`) | **Exposed** (the one-matrix CT arms) | "One-matrix component-training runs 2.1–4.1, pb_all 1.10". E13 left the cause unseparated: size 3.4% vs 1%, kl_other 0.091 vs 0.017, lr 3e-4 vs 1e-4. SGD's smaller, less diffuse steps are a direct handle on that | pb_8 on `h.2.mlp.down_proj` with SGD (rank 8 works with SGD: `Offsets` is optimiser-agnostic), rates 0.003–0.1, 30 steps, E7's rule, one Pile state | RunPod 4090, 37 min, $1.51 for the session (report header). `seconds` 2211 | About 11 min, cluster, $0 (7 min floor + 1 state + training) | Medium–Low |
| E8 lora_matched | LoRA: Adam 1e-3. Fine-tune: Adam 3e-5 (`lora_matched/run.py:47`). No CT | **Not exposed** | Splits and do-nothing of finished updates. A LoRA uses the standard Adam recipe | None | Cluster L40, job 22371, 18 min, $0 (report header) | n/a | — |
| E9 lora_plateau | LoRA: Adam 1e-3 (`lora_plateau/run.py:42`). No CT | **Not exposed** | "A moved 10%" is the standard Adam LoRA's behaviour. That is the recipe the question is about, not the map | None | RunPod L40S, 1872 s, $0.70 (report header) | n/a | — |
| E10 kv_task | Fine-tune: Adam 3e-5. CT: Adam 1e-5 on all alive, 300 steps, scored at the first checkpoint with P ≥ 0.5 (`kv_task/run.py:44,344`) | **Exposed** | CT's Pile PGD ratio is **2.81 on rand32** ("filed pass, reversed"), against 1.20 and 1.19 on rand1 and ident32. This is the only "create" task where CT breaks, and the report did not separate the causes (update size, off-target damage, K, step). Local CI-masked kl_other of the CT state is 0.42 to 0.60 | SGD CT on rand1, ident32 and rand32. Rates 0.01–0.1 with checkpoints 10/30/100/300, same "first checkpoint with P ≥ 0.5" rule. rand32 probably needs ≥ 0.03 or more steps (Adam needed 30; at 0.01 SGD's max relative change reaches only 0.0026 by step 30 on emoticon, against rand32's 0.015). Plus an Adam lower-rate arm. Measure Pile plus the local harness | Cluster L40, job 22374, 26 min, $0 (report header). About 456 s per task (`kv_task.json`, `seconds`) | About 20 min, cluster debug, $0. Training about 2700 steps × 0.1 s ≈ 5 min; 3 states × about 3 min; floor 2–7 min | **High.** Decides whether "the weak version breaks on Pile for 32" is a property of component training or of Adam at 1e-5 |
| E11 reading_a_coverage | Fine-tune: Adam 3e-5. CT: Adam 1e-5, step 10 (`reading_a_coverage/run.py:42-43,263`) | **Exposed: the most consequential** | Component training fails the adversary on 3 of 6 tasks: PGD 2.73, 2.24 and 6.74, with r2_t359 at 1.55 on the line. P1 and P5 are filed as passes and failed. The report's own pattern is "PGD excess orders with CT's kl_other, 1.6–2.5× the fine-tune's". On emoticon SGD has *lower* kl_other than Adam (0.0065 against 0.009). This is the research log's Result 1 ("faithful on about half the tasks") | SGD CT at step 10 on the 5 ladder tasks and emoticon (emoticon is the control and runs first: it must reproduce the SGD grid's step-10 P and KL and read at the floor). Rates {0.003, 0.01, 0.03, 0.1}, chosen by E11's learned rule plus least kl_other. **Plus Adam at {1e-6, 3e-6} on the same tasks** to separate optimiser from step size. Report E11's four Pile ratios, sparsity and kl_other. Code: a new experiment dir calling `component_training.train(optimizer="sgd")` and E11's measures. E11's `run.py` takes no flag and stays as it ran | RunPod L40S, 38 min, $0.70 (report header). Emoticon add-on: job 25863, about 11 min | SGD alone: about 21 min, cluster debug, $0 (floor 7 min + 6 states × 2 min + load). With the Adam step-size arm add about 12 min, so 2 debug jobs, or one RunPod 4090 run of about 40 min at about $0.35 | **High.** Can flip Result 1 and E11's P1 and P5, and the "fails on 3 of 7" count |
| E12 new_vs_existing | Fine-tune only: Adam 3e-5 (`new_vs_existing/run.py:43`) | **Not exposed** | Signatures of finished fine-tunes and their least-squares split. No CT | None | RunPod A40, 8 min 49 s, $0.24 (report) | n/a | — |
| E13 (deleted) | CT rank-8 on 24 matrices: Adam 1e-4 | Exposed, but the code is gone | Rank-8 CT: Pile PGD 1.14 and 1.28. One-matrix 4.52 | Covered by E16's rank-8 pair and E7's one-matrix arm | TRIED row only | n/a | Low |
| E14 lora_split | Fine-tune: Adam 3e-5 (`lora_split/run.py:47`). LoRA: E9's. No CT | **Not exposed** | Outside share and one-shot splits of finished updates | None | RunPod A5000, $0.73 (TRIED). `seconds` 5076 | n/a | — |
| E15 gamma_refit | Fine-tune: Adam 3e-5. CT state: Adam 1e-5, 10 steps (`gamma_refit/run.py:49,280`). Refit: AdamW on Γ (`gamma.py:87`, released optimiser, not this question) | **Exposed, but only in pending arms** | The CT state appears in stage 0 (`gamma_refit.json` → `stage0.ct`), and its refit ("ct","P") is among the four arms not run (`:52`). No CT claim has been made yet | Stage 0 only (no refit) on an SGD CT state, beside the Adam one, once the W₀ control passes. Not a refit | H100: $10.07 + $9.10, v2 $3.35 (TRIED). Prep 350–374 s; 2.12 h per refit (`gamma_refit_v2.json` benchmark) | Stage 0 about 6 min on an H100, about $0.4. A refit of the SGD state would take 2.1 h, about $5–8.5: not advised | Low (until E15's control passes) |
| E16 split_comparison | Fine-tune: Adam 3e-5. LoRA: E9's. CT: Adam free 1e-5 and rank-8 1e-4, 1000 steps (`split_comparison/run.py:42,206`) | **Exposed** | "Component training is the only operation with both": PGD 0.97–1.12, P ≥ 0.994 from step 100. One optimiser, one seed. SGD has never run 1000 steps (the SGD grid's "Not run"). A constant normalised step of 0.01 for 1000 steps may drift | SGD free CT at 0.003 and 0.01 (plus rank-8 at 0.01 optionally), 1000 steps, harness at steps 10/100/1000. **Re-run Adam CT in the same job**, because E16's G1 showed card dependence. On the H100: CT readings sit at 0.97–1.12, inside the L40's 1.63 band | RunPod H100 80 GB, scored run 2118 s, $3.68 (report header) | About 30 min on an H100 (3 floor + 6 SGD + 3 Adam states × about 65 s, about 3000 training steps × 0.1 s, about 10 min pod set-up), about $1.3–2 | **Medium–High.** Says whether Result 2's "inside the decomposition" route holds for both optimisers at E16's horizon |
| E17 cancellation | Reads E11's saved Adam CT states, with the path hard-coded (`cancellation/run.py:187-188`) | **Exposed** | The P1 rationale is "Adam's sign-like steps on a low-rank gradient give every column of dU the same direction" (`:109`). Under SGD each column of dU is G v_c, also coherent, so P1 has an optimiser-free reading. Only SGD states can tell the two apart | After E17's analysis: κ, h, F_adv and F_perm on the SGD CT states from the E11 arm (same three failing tasks plus emoticon). Code: the states need a path argument; `ct_state` hard-codes `E11/task/pullback_s0` | Scored job 28110 submitted 13:51. The local log holds only its 4 header lines and `cancellation.json` is not on local disk: UNKNOWN. Smoke on CPU took 370 s for 5 states at 8 rows (`commute-cancellation-28107.out`) | UNKNOWN until 28110's runtime is known. Same order as the E17 run itself, for 4 states | Medium. Directly tests a filed causal attribution to Adam (Q6, the cancellation reading) |
| E18 regating_leak | Planned: do-nothing of Adam fine-tunes. Not built (`reports/2026-10-21_plan_regating-leak.md`) | **Not exposed** | One-shot placement of finished updates. The tasks were chosen from E11's CT r (Adam) only as a selection reason | None | Not run | n/a | — |
| E19 step_split | Fine-tune: Adam 3e-5. The step split takes the *Adam* step S_t by the user's definition ("the Adam step, not −ηG": `reports/2026-10-21_plan_step-split.md:16`; `step_split/run.py:3,41,153`) | **Not exposed as filed** | The claims are about this fine-tune's own path. Note: with an SGD fine-tune the first-order part of the step split is exactly an SGD CT step K(S_t) taken along the fine-tune's path, a natural bridge to the SGD arm. The run **stopped at C0** on the L40: floor 0.2088 > 0.15 (`commute-step_split-28108.out`) | Optional, later: the same step split on an SGD fine-tune. Re-run on an H100 anyway for C0 | Stopped after 440 s of the W₀ control (log) | n/a | Low |

## Code-side adjustments (no runs)

1. **`src/commute/component_training.py:131-132`: any string other than "adam" silently becomes SGD.**
   `finetune._run` asserts the value (`finetune.py:140`). Make it `Literal["adam", "sgd"]` and assert, as `_run`
   does.
2. **`component_training.py:105`: the default `lr = 1e-5` is Adam's rate.**
   - With `optimizer="sgd"` and the default rate, each step is a normalised step of 1e-5, about 1000 times below the
     working range (0.003 to 0.1).
   - Fix: give the optimiser and the rate together, with no rate default, or put the default rate next to the
     optimiser.
3. **`component_training.py:116` and the docstrings in the run scripts say "Adam (or plain SGD)".**
   - Prose and comments say "plain SGD" for what is normalised gradient descent: clip 1.0 is active on every step
     (gradient norms 1.8 to 203).
   - One line in the `train` docstring and in the SGD grid report's terms would stop a reader taking it for raw
     SGD.
4. **`src/commute/ladder.py:9-13` defines coverage from an SGD step.** Every ladder arm trained with Adam. Note in
   the docstring that coverage describes an SGD first step, not the Adam step the ladder ran, so E5's P1 and A1
   compare a predictor and an outcome from different optimisers.
5. **Predictions whose rule or rationale assumes Adam.** Never edit these; a dated report is the fix.
   - `task_ladder/results/predictions_amend1.json` A1 ("Adam spreads the update");
   - `concentration_test/run.py:109` P1 (already scored by the SGD grid);
   - `identity_test/run.py:107` ("Adam spreads the update…");
   - `cancellation/run.py:109` P1 ("Adam's sign-like steps"). This one is live: its analysis should say the
     rationale is untested until SGD states are read.
6. **No `run.py` has an optimiser flag, and the rule says finished `run.py` stay as they ran.** So every SGD arm
   above is a new experiment dir calling `component_training.train(..., optimizer="sgd")` and `harness.measure`.
   Only `sgd_grid/run.py:107` sets SGD, and that is hard-coded.
7. **`src/commute/harness.py:50` (`load`) ties one windows file to the 7-minute W₀ Pile floor.**
   - A multi-task SGD arm (E5/E11, E10) would re-measure the floor per task, or need a refactor: model, blocks and
     floor loaded once, windows and route base per task.
   - `Context.token` is typed `int` (`harness.py:42`), but key-to-value tasks use a lookup-table `Tensor`
     (`route_cells` accepts `int | Tensor`, `local_faith.py:198`).
   - UNKNOWN whether the ladder's `windows_<task>.pt` files have the `{"train", "heldout"}` layout that `load`
     expects.
8. **`sgd_grid/run.py:174-175` rebuilds the alive (U, V) inline** instead of calling `component_training.alive_uv`
   (the no-copying rule).
9. **`sgd_grid/run.py:194-195` saves only the selected rate's step-30 state, on the cluster NFS.** The local
   `results/large/` is empty. E11 and E10 score at step 10 and at the first learned checkpoint, so their SGD arms
   retrain (seconds) rather than reuse it.
10. **`finetune.py:269`: the LoRA optimiser is hard-coded to Adam.** That is fine while LoRAs are the standard recipe
    (E8, E9, E14 and E16 are not exposed). Note it, so nobody assumes `ArmAConfig.optimizer` applies to LoRAs (it is
    ignored there).
11. **Cluster scripts.**
    - `cluster/sgd_grid.sbatch` says "~15 min on an L40". The scored job took about 3 min (13:40:20 submit,
      13:43:23 done).
    - Several debug scripts request walltimes far above the project's ≤ 30-minute cluster policy (`cluster/README.md`):
      - `cancellation.sbatch` 01:30:00;
      - `kv_run.sbatch` 02:00:00;
      - `new_vs_existing.sbatch` 01:30:00;
      - `task_ladder.sbatch` 01:30:00;
      - `lift_*` 01:00:00 to 01:30:00.
    - Any new SGD sbatch should ask for ≤ 00:30:00.
12. **Card choice for SGD-against-Adam readings near 1.** The L40's adversarial floor is about 0.21 (E8; job 25863;
    `commute-step_split-28108.out`). Put the E4 and E16 SGD arms on an H100 (f = 0.075), or average more attacker
    seeds. The E10 and E11 failing states (2.2 to 6.7) read fine on the L40.

---

# Slice: log_notebooks

# Audit: research log and notebooks after the SGD-grid correction (2026-10-21)

Slice: `docs/research_log/index.html` (local file, v20) and `notebooks/*.ipynb` (markdown cells, plus rendered outputs where they state a verdict). Read-only; nothing was edited.

Sources read in full: `reports/2026-10-15_analysis_concentration-test.md`, `reports/2026-10-21_analysis_sgd-grid.md`, `reports/2026-10-21_plan_sgd-grid.md` (head), `docs/research_log/README.md`, `PENDING.md`, GOALS Terms, the E2/E4/E4-follow-up rows of `experiments/TRIED.md`, both `predictions.json` files, `experiments/sgd_grid/figures/*.png`.

**Live artifact = local file.** `Artifact read` of https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu returned v20. Apart from the publish wrapper it is byte-identical to `docs/research_log/index.html` (diff: only the trailing `</body></html>`). So the line numbers below hold for both.

## A. What the PENDING line misses

The queued line names four sections: the E4 card's second Found point, the component-training technique card, "the unused-contrast line on the re-gating card", and Mistakes. Problems with it:

1. **Wrong location.** In `index.html` the unused contrast is not on the re-gating card (L834–838). It is in Backlog → "Maps not yet run" (L965). It sits on the re-gating card only in `reports/2026-10-21_decision_regating-main-line.md` (L80) and in the plan's wording.
2. **Sections it does not name** (details in section B):
   - the apparatus (L373);
   - Overview Result 1 and its figure caption (L321, L327);
   - the Q5, Q1 and Q2 statuses (L343–345);
   - the E3 card (L559, L568);
   - E4's Run field and a new Follow-up field, with figures (L576, L584) and the figures' entries in the `files` map in `docs/research_log/README.md`;
   - claim rows L872, L873 and L877, plus two new rows;
   - the timeline (L915);
   - two gap cards (L929, L941) and one new gap card;
   - two more backlog items (L984, plus a new measurement item);
   - the glossary (L1068);
   - the version and changelog (L165, L1135).
3. **Over-reach in the queued wording.** "SGD's update is ten times less diffuse than Adam's" is a post-hoc reading. By the filed rule, E4's P1 ("Adam is why the update is diffuse; with plain SGD it concentrates") is **refuted**: 0.156 falls in the null band (< 0.2). The "ten times" compares changes of different size (see D).

## B. Log edits, by line number (old → new)

House style: ≤ 2 numbers per sentence, current Terms, no "pull-back", "lift" or "projection". Priority: **H** = states the wrong finding or the Adam-only premise; **M** = should change; **L** = optional tidy.

1. **L165 (nav subtitle), H.** `Research log · 2026-10-21, v20` → `Research log · 2026-10-21, v21` (or the next free version at flush time).

2. **L321 (Overview, Result 1), M.** The result is Adam's, and the reader is not told.
   - Old: `Training the subcomponents themselves, with Γ and Δ frozen, learns every task tried`
   - New: `Training the subcomponents themselves with Adam, Γ and Δ frozen, learns every task tried`
   - Also append to the end of the card's `<p>`, after `…rather than mapping a finished one.`: ` Every run here used Adam. Plain SGD also learns emoticon within the off-target limit (E4's card, follow-up), but its faithfulness has not been checked.`

3. **L327 (Result 1 figure caption), L.** `Component training on the nine tasks it was scored on (E10, E11)` → `Adam component training on the nine tasks it was scored on (E10, E11)`

4. **L334 (current-direction box). No change.** It makes no optimiser claim; "looks like cancellation" is not tied to Adam there.

5. **L343 (Q5 status), M.** The main line's "moved pieces carry a tenth or less" was never measured for the SGD state. Insert before `Open first:`: `Every one of these states came from Adam or a finished update; plain SGD's update, less spread over the pieces (E4's card, follow-up), has no route split yet.`

6. **L344 (Q1 status), L.** `Component training keeps the decomposition faithful on about half the tasks` → `Component training with Adam keeps the decomposition faithful on about half the tasks`

7. **L345 (Q2 status), L.** `Component training, on the emoticon task:` → `Component training (Adam), on the emoticon task:`. Append: ` Plain SGD learns the task too, but its state has not been through these checks.` The chip stays.

8. **L349 (Q7 status), L, optional.** Append to the list: `; the first plain-SGD rate grid started above every working rate, and its failure was read as "no rate works" (E4)`.

9. **L373 (apparatus, Component training), H.** Old start: `<dd>Adam on the alive pieces' vectors, Γ and Δ frozen. Learning rate 1e-5 whenever…` → New start: `<dd>Adam on the alive pieces' vectors, Γ and Δ frozen, in every run since E2. Plain SGD was tried again on emoticon after E4: it learns within the off-target limit at rates from 0.003 to 0.1 (E4's card, follow-up), and no later run uses it. Adam's learning rate is 1e-5 whenever…`. Keep the rest of the field.

10. **L426 (technique card, Component training), H** (in the PENDING line).
    - Old: `In practice Adam on the vectors, as chosen in E2; plain SGD, tried in E4, learned at no rate within the off-target limit (E4's card, second Found point).`
    - New: `In practice Adam on the vectors, as chosen in E2 and used in every run since. Plain SGD works too: E4's grid started too high, and its follow-up found rates from 0.003 to 0.1 that learn emoticon within the off-target limit (E4's card, follow-up). Its update is less spread than Adam's but still diffuse, and its faithfulness is not measured.`
    - In **Status**: `The live candidate, and the only operation that kept…` → `The live candidate (with Adam), and the only operation that kept…`
    - In **Variants**: after `only pieces Γ keeps off (the gated-off control).` add ` Plain SGD on all alive pieces (emoticon, 30 steps only).`

11. **L559 (E3 one-liner), M.** `Component training spreads its update over thousands of subcomponents` → `Component training with Adam spreads its update over thousands of subcomponents`

12. **L568 (E3 Moved on), H.** The question it hands to E4 is now answered, but not as the card implies.
    - Old: `<dd>Adam or the task? E4 tried constrained component training to find out.</dd>`
    - New: `<dd>Adam or the task? E4 tried constrained component training and plain SGD. SGD's grid was too coarse to answer until the 2026-10-21 follow-up (E4's card): SGD's update is less spread than Adam's, but still diffuse.</dd>`

13. **L574 (E4 card, before the `<dl>`), H. New figures** for the follow-up, so the corrected claim has its figure:
    ```html
    <figure>
    <img src="fig/e4_sgd_lr_grid.png" alt="Two panels against the plain-SGD learning rate, log axis, 0.001 to 30. Left: held-out probability of the target at step 30 is above the 0.9 line from 0.003 to 3 and collapses at 10 and 30; 0.001 reaches 0.64. Right: off-target KL at step 30 is under the 0.110 limit from 0.001 to 0.1 (lowest, 0.0065, at 0.01, under Adam's 0.009 line) and between 0.48 and 7.7 on E4's grid of 0.3 to 30.">
    <figcaption>Follow-up, 2026-10-21. Plain SGD on the pieces' vectors learns within the off-target limit at every rate from 0.003 to 0.1 (circles). E4's grid (squares) began at 0.3, already over the limit, which is why it read as "no rate works". The orange line is Adam at 1e-5.</figcaption>
    </figure>
    <figure>
    <img src="fig/e4_sgd_concentration.png" alt="Share of the learned log-odds removed when the 1, 10 or 100 pieces with the largest single effects are reverted. Plain SGD at 0.01: 0.026, 0.156, 0.476. Adam with all pieces free: 0.002, 0.016, 0.083. Adam with only ten pieces free: 0.38, 1.0, 1.0. Dotted line at 0.5 (filed concentrated), dashed at 0.2 (filed diffuse).">
    <figcaption>Follow-up. Reverting the ten pieces with the largest effects removes 0.16 of SGD's gain (blue), inside the filed diffuse band, against 0.016 for Adam (orange). The green line, Adam with ten pieces free, shows the measure reads 1 when an update really is concentrated. SGD's change is smaller than Adam's here, so the gap is not a matched comparison.</figcaption>
    </figure>
    ```
    `files` map additions (`docs/research_log/README.md` and the publish call):
    - `"fig/e4_sgd_lr_grid.png": "experiments/sgd_grid/figures/lr_grid.png"`
    - `"fig/e4_sgd_concentration.png": "experiments/sgd_grid/figures/concentration.png"`

    If only one figure is wanted, keep `lr_grid` (the corrected claim) and leave the concentration numbers in the text.

14. **L576 (E4 Run), M.** `all (Adam, SGD)` → `all (Adam; plain SGD at rates 0.3 to 30)`

15. **L579 (E4 Found, second point), H** (in the PENDING line).
    - Old: `<li>Plain SGD on the vectors: no learning rate learned the behaviour without large off-target damage, so the runs after this use Adam; whether Adam causes the diffuseness is unscorable.</li>`
    - New: `<li>Plain SGD on the vectors: no rate in the grid, 0.3 to 30, learned without large off-target damage, so the runs after this used Adam. Corrected on 2026-10-21: the grid started above the working range (Follow-up, below).</li>`

16. **L584 (E4, after Moved on), H. New field.** Insert after the `Moved on` `<dd>…</dd>`:
    ```html
    <dt>Follow-up</dt><dd>2026-10-21 · MATS cluster L40 · <code>experiments/sgd_grid</code> · <code>reports/2026-10-21_analysis_sgd-grid.md</code>. E4's plain-SGD arm rerun at rates from 0.001 to 0.1, all else fixed; the control, rate 0.3 for two steps, reproduced E4 within 0.1%.<ul>
    <li>SGD learns within the off-target limit at every rate from 0.003 to 0.1 (first figure). At the selected rate, 0.01, its off-target KL is 0.0065 against Adam's 0.009: about the same, from one seed.</li>
    <li>Reverting the top ten pieces removes 0.16 of SGD's gain, against 0.016 for Adam (second figure). Less spread, but inside the filed diffuse band, so E4's P1 scores fail, as filed.</li>
    <li>Not matched: SGD's largest relative change is 0.0026, Adam's 0.0066. Not run: Pile and trigger checks on the SGD state, other tasks, 1,000 steps, momentum, more seeds.</li></ul></dd>
    ```
    The card's chip stays `surprise`: the amendments still carry it. Optionally add to `Moved on`: ` Plain SGD was dropped on the strength of a coarse grid; see Follow-up.`

17. **L709 (E11 one-liner), L.** `Component training's decomposition fails…` → `Adam component training's decomposition fails…`. L723 already says "Adam component training"; no change there.

18. **L835 (main-line re-gating card), L.** No optimiser claim. The "at most 11% … in every run" covers only Adam and finished updates. Item 5 (Q5) covers this once; don't repeat it here unless the lead wants it.

19. **L846 (What's next card 2, cancellation), M.**
    - `The lead explanation is cancellation: component training spreads opposing changes` → `The lead explanation is cancellation: Adam component training spreads opposing changes`
    - Optional, in the `small` footer (L849): ` The plain-SGD state from E4's follow-up, less spread than Adam's, is a cheap extra case for the coherence statistic.` Check against E17's plan first, since those tests now run as E17 (PENDING).

20. **L872 (claim row: component training keeps the decomposition faithful on Pile), H** (wording, not chip).
    - Claim: `Component training keeps the decomposition faithful on Pile` → `Adam component training keeps the decomposition faithful on Pile`
    - Evidence: append `; plain SGD's state not checked (E4 follow-up)`. Chip `refuted as general` stays.

21. **L873 (claim row: less off-target damage than the fine-tune), L.** Evidence: append `; plain SGD on emoticon 0.0065, about Adam's (E4 follow-up)`.

22. **L877 (claim row: attributable to a few pieces), H** (evidence and From). Chip `refuted` stays: SGD too stays under the filed 0.5.
    - Old evidence: `top 10 carry 1.6%, top 1000 36%`
    - New: `Adam: top 10 carry 1.6%, top 1000 36% (E3); plain SGD: top 10 carry 16%, top 100 48% (E4 follow-up)`
    - From: `E3` → `E3, E4`

23. **New claim row after L877, H.**
    `<tr><td>Plain SGD cannot do component training within the off-target limit</td><td><span class="chip refuted">refuted</span></td><td>E4's grid, 0.3 to 30, started above the working range; at 0.003 to 0.1 SGD learns emoticon within the limit (one seed, 30 steps)</td><td>E4</td></tr>`

24. **New claim row, H.** This is E4's P1, now scored.
    `<tr><td>Adam is why component training's update is diffuse; plain SGD concentrates it</td><td><span class="chip refuted">refuted as filed; SGD less spread</span></td><td>SGD's top 10 remove 0.16 of the gain, inside the filed diffuse band (under 0.2). That is ten times Adam's 0.016, but at a smaller size of change and from one seed</td><td>E4</td></tr>`
    Do not chip this `partial` or "holds in part": the filed claim failed, and the size of change is not matched.

25. **L915 (timeline, 2026-10-21), H.** Append: ` E4's plain-SGD arm is rerun at rates 0.001 to 0.1 on the cluster: SGD learns within the off-target limit, so the log's "no SGD rate works" is withdrawn. Its update is less spread than Adam's but still diffuse.`

26. **L929 (gap card "Component training is not 'the same fine-tune'"), M.** Append: ` Plain SGD now learns emoticon within the limit (E4's follow-up), so the optimiser can be varied, but no faithfulness check has been run on it.`

27. **L941 (gap card "Sparse component training never tried off emoticon"), M.**
    - Title → `Sparse or SGD component training never tried off emoticon`
    - Append: ` Nor has plain SGD, whose update is less spread than Adam's (E4's follow-up).`

28. **New gap card, H.** Put it in "The criterion and the comparisons", after L929:
    `<div class="card"><h4>One optimiser behind every component-training result</h4><p>After E4, every component-training run used Adam, because E4's coarse grid seemed to rule plain SGD out. On emoticon SGD learns within the off-target limit (E4's follow-up), but it has not been through the Pile checks, the trigger checks, the ladder tasks or a long run. Until it has, the component-training results are Adam's results, not the operation's.</p></div>`

29. **L965 (Backlog → Maps not yet run, "The unused contrast"), H** (named in PENDING, under the wrong section).
    - Old: `<li><strong>The unused contrast (2026-10-21).</strong> One SGD step of component training is the adjoint split of that step's gradient, yet the one-shot adjoint state stays at the floor (E14) while many Adam steps of component training fail on three of five ladder tasks (E11). Points at the optimiser building cancelling changes. Checks: plain SGD against the SGD fine-tune's adjoint state; coherence of both; component training with VPD's reconstruction terms in its loss, if that still counts as simple.</li>`
    - New: `<li><strong>The unused contrast (2026-10-21).</strong> One SGD step of component training is the adjoint split of that step's gradient. Yet the one-shot adjoint state stays at the floor (E14), while Adam component training fails on three of five ladder tasks (E11). One reading is that Adam builds cancelling changes; that is a hypothesis, not a finding. Plain SGD component training now learns emoticon within the limit (E4's follow-up), so the contrast can be run directly. Checks: Pile checks on the SGD state beside the SGD fine-tune's adjoint state, then the ladder tasks; coherence of both; component training with VPD's reconstruction terms in its loss, if that still counts as simple.</li>`

30. **L984 (Backlog → Tasks and arms), M.** `<li><strong>Sparse or sibling component training</strong> on the ladder tasks the diffuse one fails.</li>` → `<li><strong>Sparse, sibling or plain-SGD component training</strong> on the ladder tasks Adam's fails.</li>`

31. **New backlog item in "Measurements that settle an open question" (after L981), H.**
    `<li><strong>Plain SGD's state through the checks (from E4's follow-up).</strong> Pile faithfulness and the trigger-local checks on the saved SGD state; SGD and Adam matched for size of change, with more seeds and a noise floor for the top-ten measure; then 1,000 steps and momentum. Through the design check.</li>`
    Where the saved state is: UNKNOWN. `experiments/sgd_grid/results/large/` is empty locally. `run.py` writes the selected state there, so it is presumably on the cluster NFS; check before promising a no-GPU check.

32. **New Mistakes row (Premises that were wrong), after L1022, H** (in the PENDING line).
    `<li><strong>Plain SGD's learning-rate grid</strong> (E4): its rates ran from 0.3 to 30, and the smallest already overshot on the first step. Read as "no SGD rate works", it kept every component-training run on Adam for six days, until a grid from 0.001 to 0.1 found SGD learning within the limit. The second too-high grid after E2's. Rule: a grid whose lowest rate overshoots is extended down until a rate is too slow before "no rate works" is written.</li>`

33. **L1068 (glossary, Component training), H.**
    - Old: `Train the alive pieces' vectors on the fine-tune loss with Γ and Δ frozen; Adam, many steps.`
    - New: `Train the alive pieces' vectors on the fine-tune loss with Γ and Δ frozen, for many steps; Adam in every run so far except E4's plain-SGD arms, which learn emoticon within the off-target limit at rates from 0.003 to 0.1.`
    - **Flag:** GOALS.md → Terms has the same "Adam, many steps" in the Component training row. GOALS is user-owned. Get the user's acknowledgement before editing Terms; the glossary wording above does not contradict Terms, so it can go in first.

34. **L1135 (changelog), H. New top row:**
    `<tr><td class="c">21</td><td class="c">2026-10-21</td><td>The plain-SGD correction (<code>reports/2026-10-21_analysis_sgd-grid.md</code>): E4's "no SGD rate learns within the off-target limit" withdrawn. E4's card gains a Follow-up field and two figures (<code>sgd_grid/figures/lr_grid.png</code>, <code>concentration.png</code>). Component training marked as Adam in the apparatus, technique card, glossary, Result 1, Q1, Q2, Q5, E3's card and three claim rows. Two claims added: SGD cannot learn within the limit (refuted); Adam is why the update is diffuse (refuted as filed). Unused-contrast backlog item reworded as a hypothesis with SGD now runnable. A gap card on one optimiser behind every result, two backlog items, a Mistakes row (a coarse grid's failure read as "no rate works"), the timeline.</td></tr>`
    Merge this with whatever else the flush applies (E17–E19 lines).

35. **L176 (header "E1 to E16") and L514 (round-number note). No change** if the follow-up stays on E4's card (recommended, see C). E17–E19 from the other PENDING lines will change L176 anyway.

## C. New E-code for `experiments/sgd_grid`?

**Recommendation: no new E-code.** Put it on E4's card as a Follow-up field with both figures (items 13 and 16). Reasons:
- `TRIED.md` already files it as "E4 follow-up".
- E4's two amendments already live on its card the same way.
- E17, E18 and E19 are taken by the pending plans.
- The completeness check (`README.md` → Batched updates) greps `^| E[0-9]*` in TRIED. "E4 follow-up" matches `e4`, which has a card, so the check will not flag a missing card. The report itself is covered by the PENDING line.

If a separate card were wanted instead, it would need: a code beyond E19, a `held` chip (both filed predictions scored as filed), the two figures `fig/eN_lr_grid.png` and `fig/eN_concentration.png`, Bears on Q2, Q5 and Q7, and edits to L176, L514 and TRIED.

## D. Over-reach: where the log should not swing to "SGD is the answer"

The evidence is one task (emoticon), one seed, 30 steps, no faithfulness checks, and no matched size of change. Claims to avoid:

1. **"SGD does less damage than Adam"** (the report's answer paragraph; the PENDING line's "0.0065 vs Adam 0.009").
   - The difference is about 30%, from one seed, with no seed spread.
   - SGD's change is also smaller (largest relative change 0.0026 against 0.0066).
   - Write "about the same", not "less".
2. **"Ten times less diffuse"** and **"Adam causes the diffuseness in part".**
   - This is not size-matched.
   - The top-ten measure has no noise floor and is blind to redundancy.
   - It moves with the SGD rate: E4's SGD at 0.3 read 0.076, the follow-up at 0.01 reads 0.156.
   - By the filed rule the claim is refuted (null band). The "in part" reading is post hoc and could be a size effect. Chip it `refuted as filed` (item 24), not `partial`.
3. **SGD as "a working arm" or the new default** (the report: "plain SGD component training is now a working arm on emoticon").
   - It is an arm that learns. It is not shown to be faithful.
   - E4's own SGD state at 0.3 read a Pile adversarial ratio of 24.9 and CI-masked 0.999 at the trigger under state gates, but 0.19 under W₀ gates. SGD states can fail the checks.
   - Do not move SGD into Result 1, Q2's answer, or the technique card's "live candidate" status.
4. **"SGD works as well as Adam on this task"** (the report's "What we learned"). Only learning and off-target KL at 30 steps were compared. Don't copy this into the log.
5. **Cancellation.** Do not write that Adam "builds cancelling changes" or that SGD avoids them. Coherence was not measured for either state. Item 29 keeps it as a hypothesis.
6. **Re-gating.** The direct share and route split of the SGD state are unmeasured. A less spread update does not imply a larger direct share: the top-k revert is measured on the full model, not at the old masks. So nothing here moves Q5 or "what would move us back".

## E. Notebooks

Only `E02-E04_components_and_gating.ipynb` states the wrong E4 finding. The others mention Adam only as the recipe used, or concern fine-tunes or Γ's optimiser.

| Notebook | Cell | Sentence | Fix? |
|---|---|---|---|
| E02-E04_components_and_gating | 4 (code output, predictions table) | P1 row: `UNSCORABLE (no SGD lr learns within the KL limit)` | **Dated note.** This is the verdict as scored on 2026-10-15. Add a markdown line under cell 5's table: "2026-10-21: E4's SGD grid (0.3–30) started above the working range. At 0.003–0.1 SGD learns within the limit, and P1 scores fail, as filed (top 10 remove 0.16; `reports/2026-10-21_analysis_sgd-grid.md`)." |
| E02-E04 | 17 | "Each arm's learning rate comes from a grid, by a filed rule: the smallest lr that reaches held-out P(o) ≥ 0.9 with off-target KL ≤ 2× the W fine-tune's." | **Fix (factual).** The filed rule (`concentration_test/results/predictions.json` → `lr_rule`; `select_lr` in run.py L342–346) takes the **lowest KL elsewhere** among passing rates, not the smallest rate. Under the notebook's wording the SGD follow-up would select 0.003, not 0.01. Rewrite: "…among rates reaching held-out P(o) ≥ 0.9 with off-target KL ≤ 0.110 (2× the W fine-tune's), the one with the least off-target KL." |
| E02-E04 | 19 | "**SGD learns only at lr 0.3, with KL 0.48**, which breaks the rule. So 'Adam is what spreads the change' could not be scored." | **Dated note.** True of the 0.3–30 grid as run. Append: "(2026-10-21: the grid started too high. At 0.003–0.1 SGD learns within the rule; see the sgd-grid report.)" |
| E02-E04 | 23 | "every arm except SGD stays within 1.08 … SGD's PGD ratio is 24.9." | Optional clarifier: "SGD at lr 0.3". Otherwise a correct record. |
| E02-E04 | 43 (Not run) | "**Whether Adam is what spreads the change.** SGD needs a finer lr grid (0.03–0.3) or momentum to be scorable." | **Dated note.** Run on 2026-10-21 at 0.001–0.1: scorable, fail as filed (0.16). Momentum is still not run. |
| E02-E04 | 0, 1, 10, 40 | "all trained (SGD) / all 9 881, plain SGD"; "The all-subcomponent pull-back spreads its change thinly"; "Faithfulness on Pile survives every Adam arm" | No fix: correct records of the Adam arm. The notebook still uses "pull-back"/"lift" throughout (pre-rename record). Outside this task; `check.py` would flag it if rerun. |
| E02_lift_test | 14, 17 | "Not the optimiser. SGD and Adam leave about the same input-weighted share."; "the optimiser picks one that fits the decomposition poorly" | No fix: these are about W fine-tunes, not component training. |
| E11_reading_a_coverage | 2, 17 | "The runs here used Adam on (U, V) for ten steps"; "does not hold for the Adam pull-back as a general rule" | No fix. Already Adam-qualified, which is now the honest scope. |
| E12_new_vs_existing | 13 | "a diffuse weight change shifts the activations enough…" | No fix: fine-tunes. |
| E15_gamma_refit | 3, 14, 21, 23, 24 | "The identity test's diffuse update…"; Adam restart spike; cancellation reading | No fix. Γ's optimiser and the split state; no claim about component training's optimiser. |
| E16_split_comparison | 2 | "free component training of every alive subcomponent at learning rate 1e-5" | Optional: add "Adam". The headline "only operation that keeps both" is Adam's. |
| E01, E06, E07, E08, E09, E14 | — | Adam mentioned only as the fine-tune, LoRA or fit optimiser | No fix. |

No notebook says "needs a rate search first". That phrase is not in the log either; it must sit in reports, plans or skills (other slices).

---

# Slice: argument

# Audit: which links of the project's argument rested on "SGD is unavailable / Adam is the only working optimiser"

*2026-10-21, read-only audit (no repo file edited). Scope: the inferential chain behind the current position, after
`reports/2026-10-21_analysis_sgd-grid.md` showed that E4's "no SGD rate works" came from a grid that started too high.*

## 0. Where the premise entered the chain

The premise was never recorded as a decision. It shows up as a stated reason in three places:

- `reports/2026-10-19_decision_success-criterion.md`, clarification 3: "**Every run used Adam on (U, V)**, not SGD
  (`pullback.train`): the concentration test found no SGD lr inside the off-target limit."
- Research log, component-training technique card: "In practice Adam on the vectors, as chosen in E2; plain SGD, tried
  in E4, learned at no rate within the off-target limit". E4 card: "no learning rate learned the behaviour without
  large off-target damage, so the runs after this use Adam".
- `backlog.md` 2026-10-21, the unused contrast, check (1): "E4 found no SGD rate that learned without large off-target
  damage, so this needs a rate search first".

It is also built into definitions. `GOALS.md` → Terms (user-owned) defines component training as "train the alive
(U, V); Adam, many steps", and the log's glossary repeats it. `decisions.md` has no row for the optimiser, so Adam was
never a revisable choice on file. It drifted in from E2.

Every component-training state scored for faithfulness after E4 is Adam's: E5, E10, E11, E13, E16 and E15's
component-training arm. The fine-tunes whose δW feed the splits and do-nothing are also Adam (3e-5) in every scored
case. E2's SGD fine-tunes (lr 0.1 and 0.3) were used only for the outside share, at off-target KL 0.27 to 6.6, so they
had the same too-high-rate problem.

## 1. The one-sentence claim, part by part

The claim (`reports/2026-10-21_decision_regating-main-line.md`): "narrow post-training updates are mostly carried by
re-gating ..., no map with Γ frozen keeps the decomposition adversarially faithful, and the failure is cancellation".

### Part 1: "mostly carried by re-gating" is not optimiser-dependent where it counts

- r is measured on the update itself, under do-nothing placement (Adam fine-tune δW). The task ladder found it "within
  0.16" across "(a) full-weight fine-tune, (b) Adam on all subcomponents, (c) Adam on only the subcomponents whose CI
  at the trigger is ≤ 0.01" (`2026-10-19_analysis_task-ladder.md`). The update that re-gating describes is the
  fine-tune's, not component training's.
- E4's one SGD data point agrees. The sgd_all arm at lr 0.3 reads CI-masked P(o) 0.999 under the state's gates and
  0.19 under W₀'s gates (`concentration-test` amendment 1), so even the overshooting SGD state runs mostly through
  re-gating.
- What it does rest on: every update in the evidence was made by Adam (fine-tune 3e-5, LoRA 1e-3, component training
  1e-5 or 1e-3). Whether an SGD fine-tune re-gates the same way has not been measured. Low risk, but untested.

### Part 2: "no map with Γ frozen keeps the decomposition adversarially faithful". Its component-training half is Adam-only.

- **The splits.** The least-squares split fails and the adjoint and rescaling splits lose the behaviour (E2, E8, E14,
  E16). None of that depends on component training's optimiser. It does depend on the fine-tune's optimiser: every
  split was of an Adam δW. An Adam step is applied coordinate by coordinate, which breaks the low-rank,
  data-shaped structure of G that the read side relies on (E14: "The read side of a gradient update is the data").
- **Component training.** Every failure is Adam's: E11's 2.73, 2.24 and 6.74 (failing) and 1.38 to 1.55 (at the line),
  and E10's rand32 at 2.81. E11's own pattern ties the failures to off-target damage, which is the thing SGD changed:
  "The PGD excess orders with the pull-back's off-target KL" and "The pull-back's kl_other is 1.6–2.5× the
  fine-tune's at the same step on every task". On emoticon SGD at 0.01 reads KL elsewhere 0.0065 against Adam's 0.009.
- **The sentence already overstates the record for Adam.** E16 has Adam component training "the only operation with
  both" behaviour and floor to step 1000 on emoticon. The log's Result 1 says it "passes the adversarial check on
  four" of nine tasks. "No map" really means "no map that works on every task tried, and component training is
  discounted as an analogue". The breadth (the ladder and the cipher) is carried entirely by Adam states at one
  learning rate (1e-5, chosen on emoticon in E2 and never re-tuned per task).

### Part 3: "the failure is cancellation". Its evidence is Adam states plus one split of an Adam δW.

- The adversarial-vs-stochastic analysis (selective masks do "50–300×" the damage of the uniform shrink) reads only
  E11's Adam component-training states.
- E15 stage 0 reads the least-squares split of the Adam fine-tune. Stage 0 also has a signature of Adam: "the split and
  component training put their largest relative changes on subcomponents that do almost nothing on Pile at either
  model" (those subcomponents are 20× less important one at a time than random ones). Adam's per-coordinate
  normalisation gives a tiny-gradient (inert) subcomponent a step about as large as an active one's. SGD scales each
  step by the gradient and does not.
- **An internal contradiction to resolve.** The backlog's unused contrast says Adam "breaks the K structure" and "points
  at the optimiser building cancelling changes". E17's filed P1 predicts the opposite in weight space: "*Predicted:
  fail.* Adam's step is sign-like ... so the columns of dU share one direction and the changes add coherently". Both
  can hold together (coherent in Frobenius norm, cancelling in function, which is E17's own "P1 fails, P2 holds"
  branch). But the project has not said which of the two Adam stories it holds.

### Is SGD component training a direct test? Yes for part 2, partly for part 3, and its anchor is weaker than the backlog says.

- It is a map in GOALS's sense ("component training (the same training step applied to U and V instead of W)"). So a
  pass on the tasks where Adam fails directly falsifies part 2 as worded.
- **A caveat on the unused contrast's prediction ("SGD ≈ adjoint split ⇒ stays at the floor").** SGD component training
  equals the adjoint split only per step and to first order. It differs in three ways. Its gradients are taken on its
  own trajectory, not the fine-tune's. It has no scale c. It trains until the behaviour is learned. The scaled adjoint
  state, by contrast, reproduces only 58% of the energy and P(target) 0.93 at step 10, falling to 0.10 at 1000. So the
  adjoint's floor may come from its being *partial*, not from its structure.
- E14's explanation for the floor ("the adjoint moves the subcomponents only along directions they already read and
  write") does not separate the adjoint from Adam either. Any (dU, dV) moves W by dU Vᵀ + U dVᵀ, the same tangent space
  for Adam and SGD alike. The real differences are the preconditioner, the size, and the path. SGD at full behaviour
  removes the preconditioner and keeps the size, which makes it the sharper test.

**Outcomes of an SGD arm on the E11 ladder card** (emoticon plus the five ladder tasks, step 10, then 100 and 1000).
Each is read against the same task's Adam state and do-nothing on the same card in the same job:

- **(a) SGD keeps the behaviour and stays at the floor where Adam fails** (r0_t352, r1_t534, r2_t275):
  - Part 2 is false as worded. A frozen-Γ map stays adversarially faithful on tasks the model already does, and Adam's
    preconditioning (or its larger off-target change) was the failure.
  - Cancellation becomes a property of Adam's encoding, not of the decomposition. That weakens GOALS's "too brittle"
    branch.
  - If it also holds at steps 100 and 1000, it is the decision's first move-back bullet (section 2).
  - It also supplies the main line's control 2 ("re-gating is read only on states that pass") with passing states on
    the ladder tasks.
  - It must be checked against a size-matched Adam arm (or Adam at a lower rate). Otherwise "SGD passes" could be
    "smaller change passes".
- **(b) SGD fails like Adam** (same tasks, similar ratios, similar ordering with kl_other):
  - The optimiser is cleared. Part 2 is strengthened for component training.
  - The unused contrast resolves to size or path: the adjoint state is at the floor because it is partial. That
    points at the update's size, or at real brittleness.
  - E17's statistics on the SGD states then decide whether the cancellation is intrinsic.
  - The re-gating main line stands on firmer ground.
- **(c) SGD learns but sits between the floor and Adam** (or passes some failing tasks, not all):
  - The optimiser is one contributor. The likely shared cause is off-target damage, E11's pattern 1.
  - The next step is the matched-kl_other comparison already in the backlog, now with two optimisers as two points on
    the curve.
  - Part 2 then reads "fragility grows with off-target change, and Adam makes more of it", a size statement rather
    than a frozen-Γ impossibility.
  - A sub-case: SGD does not learn a rare task within 10 steps at any rate inside the limit. Then report the rule's
    grid and step as the scope ("no rate in [x, y] by step N"), not "SGD fails" (section 6).

## 2. Does the re-gating-main-line decision survive a working SGD map?

- The decision's stated reason (`decisions.md`): "Every experiment agrees that re-gating carries the behaviour while no
  frozen-Γ map passes". The first half does not depend on the optimiser (section 1, part 1). The second half is the
  Adam-only part.
- **Bullet hit: "E16 or the Γ refit (after a passing control) finds a map that stays at the floor and keeps the
  behaviour with Δ fixed at steps 100 and 1000, not only at step 10: the commute line returns to first place."** The
  bullet names only E16 and the Γ refit. An SGD component-training arm is a third route to the same condition, and the
  bullet should be read as "any map".
- **E16 already met the bullet's letter, with Adam, on emoticon** (0.97 to 1.12 with P ≥ 0.994 from step 100, Δ ≤ 2e-8).
  The project did not move back. It read component training as "trained in place of the update rather than mapping a
  finished one" (log Result 1) and pointed to the ladder failures. So a working SGD map moves the decision only if it
  meets two conditions:
  - **Breadth.** It passes on the ladder tasks where Adam failed, at steps 100 and 1000.
  - **The "analogue" objection is answered.** SGD helps here. With an SGD fine-tune, each SGD component-training step
    is exactly the adjoint split of the fine-tune's step taken at the component-training point. That is as close to a
    map of the update as a per-step operation gets. E19 run on an SGD fine-tune measures the remaining gap (path
    only).
- Verdict: the direction survives as science, because the re-gating findings stand. Its *priority* rests on Q1 being
  stuck, and that rests on Adam. If (a) holds at 100 and 1000 on the ladder, the decision's own rule sends Q1 back to
  first place. The user should be told before more re-gating runs are queued ahead of the SGD arm.

## 3. E17 (cancellation) and E19 (step split): do they assume Adam?

**E17 (`reports/2026-10-21_plan_cancellation.md`):**

- **States.** All component-training states are E11's Adam states. The splits are of the Adam fine-tune.
- **Optimiser-free parts.** The passing/failing classification comes from Adam ratios, but the statistics themselves (κ,
  h, F_adv, F_perm) and the controls are optimiser-free. G1, G2 and C1 (gauge-mixing, A = I + εE) all carry over.
- **Adam-dependent prediction.** P1's rationale is explicitly Adam's: "Adam's step is sign-like and a ten-step gradient
  on a narrow task is low-rank, so the columns of dU share one direction".
- **Missing from the decision tree.** The branch "Cancellation confirmed: ... the maps to try next constrain placement
  (the adjoint's coherent write, sparse or sibling component training)" leaves out the cheapest constraint, a different
  optimiser.
- **Without an SGD state, E17 cannot tell** "Adam builds cancelling changes" from "any training that learns these tasks
  does".
- **Recommendation: after, as an amendment, not instead.**
  - Run E17 as filed now (filed, smoke done, cheap, approved class 1).
  - Feed the SGD states in once the SGD ladder arm has Pile ratios, since E17 needs the passing/failing label first.
  - Add a dated amendment with its own predictions. Filed predictions are never edited.
  - The hypothesis to file: SGD's κ and h sit nearer the adjoint state's than Adam's do.
  - The SGD states must be saved to NFS. sgd_grid's `results/large/` is empty locally; check that the cluster copy
    exists.

**E19 (`reports/2026-10-21_plan_step-split.md`):**

- **Built on the Adam step.** "take the actual weight step S_t = W_t − W_{t−1} (the Adam step, not −ηG)".
- **Its numbers are Adam's.** The expected Δ share (1 − R = 0.42/0.58/0.69) and every reference are anchored to the
  one-shot adjoint of the Adam fine-tune.
- **R is likely low because of Adam.** An Adam step is G rescaled coordinate by coordinate, so it loses G's data-shaped
  read side, and R (the share K reproduces) is likely low for that reason. P3's prediction that e_Δ rises to 0.5 or
  more by step 1000 inherits this.
- **With an SGD fine-tune, S_t = −ηG_t**, so the step split is "SGD component training's update, taken along the
  fine-tune's path, with the remainder booked to Δ". That is the cleanest version of the unused contrast: it holds the
  optimiser and the per-step operator fixed and varies only the path.
- **Recommendation: an SGD fine-tune variant before E19's next scored attempt.** That attempt is a rerun anyway: E19
  stopped at its W₀ control on the L40, f = 0.209 > 0.15 (`PENDING.md`).
  - It needs an SGD fine-tune rate search first, about 3 min, the sgd_grid recipe on `finetune`.
  - It needs a dated amendment with its own predictions.
  - It needs a card whose W₀ floor passes (the E11 L40S job read f_seed 0.046, E16's H100 0.075), or more attacker
    seeds.
  - The Adam arm stays as filed.

## 4. The re-gating reading of E4–E6, E10 and E12: optimiser-dependent?

Mostly not, with one real exception.

- **Not optimiser-dependent:**
  - r is a property of the update and task under do-nothing placement (E5, E6, E10, E12 all read r on fine-tune δW).
  - It agrees across component-training variants that are *concentrated by construction* (top 1, top 10, siblings).
  - The switched-on set recurs across those variants (Jaccard 0.59 to 0.88, E4 amendment 2).
  - E4's SGD arm also runs through re-gating (0.999 vs 0.19).
  - "Pieces switch on" therefore does not rest on Adam's diffuseness.
- **The exception: "pieces don't change / never carry the behaviour" (≤ 0.08 to 0.11, log Result 3 and its claim
  row).**
  - For diffuse component training this rests on Adam. The ladder report names it as a candidate cause: "Adam spreads
    the update so evenly ... that the 51–95 components Γ gates on at the trigger carry little of it".
  - E4's own amendment 1, converted to the route split's log-odds share, gives the following. This is an indicative
    conversion from mean P (logit of the mean, against the E4 W₀ values P 0.0045 masked and 0.0115 target), not the
    route split itself:

    | arm | indicative d |
    |---|---|
    | Adam-all | ≈ 0.04 (consistent with the ladder's emoticon d 0.025) |
    | SGD-all, lr 0.3 | ≈ 0.21 |
    | top 10 | ≈ 0.31 |
    | top 100 | ≈ 0.16 |
    | siblings | ≈ 0.42 |
    | top 1 | ≈ 0.01 |

  - The claim row "at most 0.11 ... 0.08 or less for every other operation" cites E4 among its sources, yet E4's
    concentrated arms and its one SGD arm read several times that. The E4 report read them as P ratios ("siblings 37%,
    the rest ≤ 19%"), which hid the gap.
  - The sgd_grid SGD state at 0.01 (10× less diffuse than Adam's) was never route-split.
  - So the size of d is optimiser- and concentration-dependent. "Re-gating carries most of it" survives; "the moved
    pieces carry a tenth or less under every operation" does not, as written.
  - This needs a review report (the project's rule: concerns about existing claims go in a review). A route split on
    the saved sgd_grid state and E4's saved states settles it, as class 1 analysis.

## 5. Candidate re-runs, ranked by how much they move the position under GOALS

Assumptions:

- **Costs.** Cluster `debug` ≤ 30 min ≈ $0. RunPod 4090 ≈ $0.5/h, H100 ≈ $2.5/h, plus about 5 min of set-up.
- **Per-state Pile scoring.** About 2 min. E11's five-task job: 12 states plus W₀ floors in 38 min on an L40S. E19's
  plan says about 115 s per state on an L40.
- **Training speed.** About 1 s per step or less. sgd_grid ran a control and 5 rates × 30 steps in about 3 min on an
  L40.
- **Card.** The cluster L40's W₀ PGD floor has twice read about 0.21 (E11's emoticon job, E19's control), which fails
  a 0.15 early-stop. Ladder verdicts near 1.5 need a quieter card or 6 or more attacker seeds.

| rank | run | moves the position because | runtime (nearest record) | cost |
|---|---|---|---|---|
| **High** (run 2nd, the main test) | **A. SGD component training on the E11 ladder card** (emoticon plus 5 tasks), per-task rate by E4's rule, scored at step 10 against E11's Adam and do-nothing in the same job; then steps 100 and 1000 for both optimisers | Directly tests part 2 of the claim and the move-back bullet. Adam has no ladder numbers past step 10 either, so stage 2 needs both optimisers | Stage 1 ≈ E11: 38 min on an L40S, plus a 5-rate search per task (≈ 3 min each, sgd_grid) ≈ 50–60 min. Stage 2: 2 optimisers × 6 tasks × 1000 steps plus 24 Pile states ≈ 1.5–2 h (cf. E16: 2 component-training runs to 1000 plus 30 states in 2118 s on an H100) | Stage 1: RunPod 4090 ≈ $0.5, or two cluster debug jobs ≈ $0 (floor risk). Stage 2: H100 ≈ $4–5 |
| **Medium** (run 3rd) | **D. SGD on the key → value tasks** (rand1, ident32, rand32) | GOALS's "new capabilities" branch. Adam's rand32 failure (2.81) is the only cipher evidence that frozen-Γ maps fail on a behaviour the model lacks. An SGD pass there removes it | E10: 26 min on a cluster L40 for 3 tasks × 300 steps with fine-tune, splits and Pile. SGD component training only, plus a rate search ≈ 25–30 min | Cluster debug ≈ $0, or split by task |
| **Medium** (with A or after it) | **B. SGD component training vs the SGD fine-tune's adjoint state** (emoticon; or E19's SGD variant) | Settles *why* (path vs scale vs preconditioner), and answers the "analogue, not a map" objection that kept E16 from moving the decision. Changes the explanation more than the verdict | Fine-tune rate search ≈ 3 min, 3 to 4 Pile states plus W₀ seeds ≈ 15–20 min (E14's split stage was "~15 min" on the cluster) | Cluster debug ≈ $0 |
| **Medium as a control, Low alone** | **C. SGD matched to Adam's size of change** (max relative change 0.0066, or matched kl_other) | Without it, outcome (a) cannot separate "SGD" from "a smaller change" (sgd_grid: 0.0026 vs 0.0066). Belongs inside A as one reported arm per failing task, plus Adam at a lower rate | Emoticon ≈ 5 min; inside A, about 2 extra states per task ≈ +15 min | $0, or inside A's pod |
| **Low for the position, run 1st** | **E. Pile faithfulness (and the trigger-local masked check, route split) on the saved sgd_grid emoticon state** | Emoticon already passes with Adam (1.03, and to step 1000 in E16), so a pass changes little. It is the cheapest end-to-end check of the SGD path on the harness, and gives the direct share d for an SGD state (section 4) | E11's emoticon rerun: 12.5 min on a cluster L40 for 2 states plus W₀ floors | Cluster debug ≈ $0 |

- **Order.** E, then A stage 1, then A stage 2 (if any failing task passes at step 10) and D in parallel, then B
  (folded into E19's rerun).
- **The control.**
  - Run first and alone: W₀ under 3 to 6 attacker seeds, early stop if f > 0.15.
  - Gate: reproduce E11's Adam ratios on the same card by re-reading the saved E11 states from NFS (r0_t352 2.73,
    emoticon 1.031, do-nothing 2.183). Then the SGD verdicts are read against same-job references.
  - E (the sgd_grid state) is the positive sanity arm for the SGD code path (expected at the floor).
- **Total.** About $5–6 of the $40 session budget. Today's spend so far includes E16 at $3.68 and E15 v2 at $3.35.
  Check the live tally before the H100 stage.
- **Design check.** A is the ladder's "Not run" item ("SGD or sparse pull-backs on the ladder tasks"), so class 2 at a
  stretch. Because its outcome can change the main line, put the check to the user anyway: "a change of main line"
  needs confirmation.
- **Missing from the list.** Adam at a lower learning rate (or matched kl_other) on the failing ladder tasks. Adam's
  1e-5 was chosen on emoticon and never re-tuned, so "Adam fails" is itself a single-rate result.

## 6. Not runs or text fixes

1. **A rule for reporting a grid's null.** A failure over a hyperparameter grid is reported with its scope ("no rate in
   [0.3, 30] by step 30"), never as "no rate works". If the best value sits at an edge of the grid and fails in the
   direction of that edge (here the smallest rate overshot on step 1), the grid is extended before the null is
   written. Put this in `/writeup` or `.claude/CLAUDE.md` → Writing code, and in `/design`'s numbers question
   ("expected value at the grid's edges").
2. **A rule for unscorable predictions.** When a filed prediction is unscorable because its precondition failed (E4's
   P1: "no SGD lr met the rule"), the report names the precondition's failure as the result, without generalising it,
   and adds the fix to "Not run". The lead decides explicitly whether a later design may rely on the failure. Here a
   precondition failure became a premise (clarification 3, the technique card) with nobody deciding it.
3. **Record the optimiser as a decision.** Add a `decisions.md` row: "Component training uses Adam (since E2, lr 1e-5
   all-alive); revisable; plain SGD works on emoticon since 2026-10-21". Propose to the user (GOALS is user-owned)
   that the Terms row read "train the alive (U, V) with the run's optimiser" rather than "Adam, many steps", since
   GOALS's map definition ("the same training step applied to U and V") is optimiser-neutral.
4. **Dependencies on claims.** Give each claim row in the log a "rests on" field (other claim rows, the optimiser,
   the card, the task). When a claim is refuted, list its dependants and re-read them, as this audit did by hand. The
   "SGD unavailable" claim had three dependants nobody listed.
5. **Rescope the one-sentence claim** in the decision report's next revision and the log:
   - "Adam component training" in place of "no map";
   - note that E16 meets the move-back bullet's letter on emoticon;
   - rewrite the first bullet as "any map, including SGD component training".
6. **A review report on the direct share** (section 4). Re-express E4 amendment 1 in log-odds shares, then route-split
   the saved SGD state, before "the moved pieces carry a tenth or less under every operation" is repeated.
7. **Reconcile the two Adam stories** (the backlog's "optimiser builds cancelling changes" against E17 P1's "Adam's
   changes add coherently") in E17's analysis, through its own P1-fail/P2-pass branch. No prediction is edited.
8. **`/design`'s "held fixed / varied" line.** Make the optimiser an explicit entry in every comparison of operations.
   E16's "analogue" pairing (Adam component training at 1e-5 against an Adam fine-tune at 3e-5) varied the rate and
   the space together.
9. **Queue `PENDING.md` lines** for the items above when their files are written. The sgd-grid line already queues the
   E4-card and technique-card corrections. It does not queue the claim sentence, the Result 3 claim row, or the
   success-criterion decision's clarification 3.
