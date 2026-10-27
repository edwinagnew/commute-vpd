# Review: E37 Amendment 1 analysis (teacher spread: train longer, adjoint start, masked losses)

*2026-10-27, independent check by a fresh agent that did not write the code, run or analysis. Read-only on everything
except this file. Scope: `reports/analyses/2026-10-26_analysis_teacher-spread.md` → "Amendment 1", the Amendment 1 cells
of `notebooks/E37_teacher_spread.ipynb`, `experiments/teacher_spread/plot_amend1.py` and its two figures, the
`experiments/TRIED.md` row (line 75), and `reports/decisions/2026-10-27_decision_e37-amend1-card-draft.md`, against
`results/amend1_predictions.json`, `results/amend1.json`, `results/large/amend1_run.log`, `amend1.py`,
`src/commute/masked_teacher.py`.*

**Verdict: scores and numbers are correct; the branch taken is a judgement the write-ups label as the filed tree's,
and the card leaves out the extend option.** Every prediction is scored exactly by its filed rule, and every number
checked matches the JSON or log. The filed tree has no branch for what arm 1 did: the training KL kept falling while the
check did not. The code put that case under "flat" by default, and the "What next" section and the card present the
last branch as what the filed tree calls for. Two smaller problems: a few sentences link the check to damage on
ordinary text, which M0b undercuts, and the card's option (a) claims more for the zero-weight control than it can
show.

## Findings

1. **The filed tree has no branch for this outcome; the write-ups say it reached its last branch.** The filed text,
   `amend1_predictions.json` → `decision_tree` (and the plan, `reports/plans/2026-10-26_plan_teacher-spread.md:240-246`),
   has an extend branch that needs the train KL falling ≥ 10% from 2,000 to 4,000 *and* the ratio falling. It has a
   last branch that needs arm 1 "flat", and the plan's numbers table (`:220`) defines flat as the train KL falling
   "below 10% per doubling". Arm 1's train KL fell 43.7% (0.01137 → 0.00641). Single steps in the log keep falling:
   0.0189, 0.0085, 0.0057, 0.0043, 0.0038 at 2,000 to 4,000 (`amend1_run.log:73-77`). The check rose 18%
   (1.894 → 2.227). So neither branch's condition holds. `amend1.py:299-305` (`branch`) returns "arm 1 flat" for
   anything that is not the extend case, so the label is a default in the code, not something the filed text says.
   The analysis says this under Surprises (analysis line 558-561), then contradicts it: "The filed tree reaches its
   last branch" (line 567), the scorecard's "branch reached | the filed tree" (line 602), the notebook's "The filed
   tree's last branch" (cell 8), and the card's header, "calls for in its last branch" (card draft, line 3).
   The check reading is also weak evidence that training has stopped helping. Its 2,000 read, 1.89, is the lowest of
   the three. From 1,000 to 4,000 the check is flat, +1%. Three reads with a 14% batch-order spread cannot tell a flat
   check from a slow downward trend. **Fix:** in all four places, say the outcome falls between the filed branches.
   Say that taking the last branch is the analyst's judgement, because the extend branch existed to catch a falling
   check, and the check did not fall over 4×. Add "extend arm 1 to 16,000 steps, about $1.5" to the card as an option,
   or give the reason it is left out. Optionally relabel the code's fall-through as "not covered by the filed tree".

2. **Several sentences tie the check to damage on ordinary text, but M0b fails the check at almost no damage.** The
   analysis says "The adjoint start passes only while it does a fifth of the fine-tuned model's damage … Both trained
   decompositions failed once they reached the fine-tuned model's damage" (lines 533-534). The notebook says "its
   check rises as its damage reaches the fine-tuned model's" (cell 4). The card says "In both runs the check rose once
   the decomposition changed ordinary text as much as the fine-tuned model does". But M0b reads a check of 3.14 at
   damage 0.007, and M0 reads 3.84 at 0.021. A state can fail the check by more than any trained state while barely
   changing ordinary text, so damage cannot be the whole story. Arm 3 also has only one read before 1,000 steps, so
   "passes only while" rests on a single point. **Fix:** wherever the damage link appears, add M0b as evidence
   against damage being the driver. Card: "…whether that is the cause is untested, and the masked control failed the
   check at a thirtieth of that damage."

3. **Card option (a) claims more for the zero-weight control than it can show.** The card says the zero-weight control
   "says whether our trainer or the losses move the check, and whether a refit can be read at all". With both masked
   weights at 0, `masked_teacher.train` is plain teacher training against W₀ through the component forward. It
   separates the loop without the masked terms from the masked terms as set up here. It does not separate VPD's
   losses from our set-up of them. That set-up is batch 8 rather than 64, Γ frozen, the PPGD state from
   `gamma.loss_setup`, and lr 1e-5. It also says nothing about E15's Γ refit, which trains different parameters
   (Γ, with AdamW). The card's "E15's attempt … against the same losses … failed the same way" overstates as well:
   the analysis itself says "two of the same losses" (line 506). **Fix:** "says whether the masked terms or the rest
   of the loop move U, V off the floor". Drop "whether a refit can be read at all". Say "with our set-up of VPD's
   losses" in place of "against VPD's losses" in the card, analysis lines 507 and 572 and the TRIED row.

4. **"Did not come from a mismatch at the start" goes slightly past the data.** Analysis lines 525-526, notebook
   cell 6. G2 shows the step-1 KLs agree to 7e-5. But the two forward paths differ by KL 0.00065 at W₀ (`path_kl`).
   Under Adam, which rescales every step to about the learning rate, a small systematic gradient from that difference
   could drive steady drift. That drift is the trainer hypothesis the zero-weight control tests. The training-loss
   evidence also points at the loop: M0's random-mask loss rose 23% (0.254 → 0.312) while it was being minimised with
   no adversary in it, and the total loss rose 0.355 → 0.631 (M0) and 0.036 → 0.069 (M0b) (`amend1_run.log:119,130,
   141,152`). **Fix:** "M0's rise did not come from a mismatch in the step-1 loss; whether the 0.00065 path
   difference drives drift under Adam is what the zero-weight control tests."

5. **Small wording overstatements.**
   - Analysis line 357 and notebook cell 1: arm 3 "climbs to the same place within a thousand steps". At 1,000 steps
     it reads 1.76 against arm 1's 2.20, 20% lower. It is level only at 2,000 (−3%). Analysis line 422, "within 3% of
     the longer run at the same step", reads as if it held at both steps. **Fix:** "and by step 2,000 is within 3%
     of the longer run".
   - Card title and question: "however long it trains" and "wherever it starts". Only 4,000 steps and two starts were
     tested. **Fix:** "trained four times longer or started from the adjoint split".
   - M0 figure title (`plot_amend1.py:172`): "it barely changes the model". M0's weight change, size 0.067, is
     larger than the fine-tune's 0.049. **Fix:** "barely changes the model's predictions".

6. **Code naming (no prose error).** `amend1.py:282` stores `arm3_0_over_e37.s` as a *difference* (0.80411 − 0.804 =
   1.1e-4), while `check` beside it is a ratio. The prose correctly reports s = 0.8041. **Fix:** rename the field
   (`s_minus_e37`) or store the ratio.

7. **No figure for the training-KL claim** that the branch argument rests on (analysis lines 475-479; the 2,000 to 4,000 single steps are in the log only). The step figure
   shows the check and the held-out gap, not the train KL. **Fix (optional):** add the train KL as a third panel of
   `amend1_steps.png`, or annotate the bottom panel.

**Not raised in the write-ups, suggested for the card or Not-run list:** a control that moves U, V at random by M0b's
size (0.018) and reads the check. M0b's 3.14 at almost no damage, against the adjoint split's 1.05 at size 0.040,
asks whether the check punishes any move of U, V away from the released solution that is not adjoint-like. That
would bear on reading every trained state's roughly 2×, not only M0's.

## Checked and found fine

- **Scoring.** P1 holds: 2.227 ≥ 1.5 × 1.0748 = 1.612. P2 is not scored, by its filed rule, because arm 2 was not
  run. P3 holds: s 0.990 ≥ 0.9, and |1.836 / 1.894 − 1| = 3.1% ≤ 25%. The pass and fail lines use f = pgd_floor
  0.0748 (`faith_score.FLOOR_KEY`). M0 misses on damage (0.0214 > 0.02) and on the check (3.84 > 1.40). M0b misses on
  the check (3.14), so arm 2 is skipped as filed. G2 passes: |0.29012 − 0.29019| = 7e-5 ≤ 1e-3. C1 passes: floor
  0.07482, pgd_rel 1.3%, ci_rel 0.05%. C2 passes: gain 0.12995. C3 passes exactly. R1 is reproduced: s 0.9978,
  check 1.0005 × 2.20. The arm 3 start is reproduced: s 0.8041, check 1.049.
- **Every number in the analysis tables and prose** matches `amend1.json`: s, gap, damage, check, stochastic,
  CI-masked, unmasked, size, cosine and train KL for all nine states. It also checks against the log: the per-block
  KLs and ratios (1.69/2.68, 1.69/2.09, 2.02/2.42, 1.09/1.01, 1.66/1.85, 1.75/1.91, 3.91/3.77, 2.84/3.43 over W₀'s
  blocks 0.6794 and 0.7182) and the masked-term means. And against `teacher_spread.json`: do-nothing 4.868 and
  1.106, lower_s1 1.884, and W₀'s Pile readings at all three seeds identical to this job's. Derived figures check:
  "0.36 to 0.46 of do-nothing", "11 to 35 times the damage", "at least 9% above the fail line", "span 18%", the gap
  falling 17%, 26% and 21% per doubling, and the train KL −44%.
- **Provenance.** The sha256 prefixes of the JSON, predictions, both logs and the three `.pt` files match the files
  on disk. The predictions were committed (`e70ee46`, 11:12:26) after the review fixes (`5d9b87f`) and before the
  results (`87cf87d`, 12:49:59), and have not been edited since.
- **Notebook and TRIED row** carry the same numbers as the analysis, apart from the points in findings 1-3.
- **Figures** show what the prose says. The pass and fail lines, W₀'s seed band, E37's second batch order and the
  do-nothing reference sit on the same check axis. The gap panel carries W₀'s 0.272. The M0 figure shows W₀, the
  filed M0 box, every trained read, the adjoint start and do-nothing on the same damage and check axes.
- **Cause claims about M0 itself** are hedged correctly in the analysis ("This run does not say why … None of these
  was tested"; "Whether the losses, the set-up or our use of it is responsible is untested"). The problem is only the
  card's framing, finding 3.
- **Card options (c) and (d)** are stated fairly, including that the do-nothing reference is inflated by
  construction.
