# Advice: a weight-side route split that sees the gates (Fable, 2026-11-02)

Proposal under review: per matrix, on real inputs a, compare E‖(g₁C₁ − g₀C₀)a‖² with E‖δW a‖², with direct and
re-switching versions (g₀C₁ − g₀C₀) and (g₁C₀ − g₀C₀). Symbols: W₀ the original weights, δW the update, C₀ and C₁ the
subcomponents before and after (C = Σ_c u_c v_cᵀ, Δ the remainder), g₀ and g₁ Γ's importance values on W₀'s and
W₁'s activations of the same input, applied per subcomponent and per position. The scouts' notes are
`~/commute-data/scratch/regating-lead/wr-records/records.md` and `.../wr-states/states.md`; file:line references
below were re-checked.

## Verdict

**Drop it as proposed.** The norm-against-norm ratio is two known quantities and one new one: its direct term is the
input-weighted inside share with W₀'s gates multiplied in (algebra below), and its re-switching term does not contain
δW at all; it is Γ's response to the update, in output units, divided by the update's size. Neither separates a
fine-tune that learned nothing from one that learned, for the same reason log-odds r does not: in the do-nothing state
the stand-in has no δW in it, so the only thing it can report is how Γ reacts to perturbed activations, and Γ reacts to
any perturbation. Changing the unit from a token's log-odds to a vector norm removes the target dependence and the
trigger-only restriction, which are real gains, but not that.

**What would change it.** If the user wants a weight-side number, the honest form is not a norm ratio but a projection
onto the update's own output change, S_X = E⟨X a, δW a⟩ / E‖δW a‖², X ∈ {direct, re-switching, interaction,
residual}. It sums to 1 exactly by linearity (the telescoping the route split has in log-odds), it is signed, so an
orthogonal change reads 0 rather than 1, and it is the same matrix in the same units on both sides, which removes the
"numerator in the stand-in, denominator in the real model" objection as far as units go. One $0 run on saved states
settles whether it earns a place: the re-switching projection on E46's learned-nothing code fine-tune
(`regating_table/code_step10.pt`) against E12's copy fine-tunes. My expectation, below, is that it fails that null for the
reason above, in which case the whole idea goes to `TRIED.md` as dropped at design. If it passes (code ≤ 0.1, copy ≥ 0.4,
guesses), the re-gating line gains the first instrument that passes the null r failed, and E62/E58's readings get a
weight-side column.

## The algebra

Per matrix, write the masked output change as D + R + I with

    D = g₀(C₁ − C₀) a        direct
    R = (g₁ − g₀) C₀ a       re-switching
    I = (g₁ − g₀)(C₁ − C₀) a interaction

so ‖D + R + I‖² = ‖D‖² + ‖R‖² + ‖I‖² + 2⟨D,R⟩ + 2⟨D,I⟩ + 2⟨R,I⟩. The three cross terms are why the proposal cannot
sum to anything; nothing about the inputs or the pooling removes them.

**Direct term.** For a least-squares split, Σ_c(C₁ − C₀)_c = δW − O(δW) + dU dVᵀ, with O(δW) = (I − P_U) δW (I − P_V)
the remainder sent to Δ (`GOALS.md` Terms, `split.py:129-142`). With every gate on (g₀ ≡ 1), E‖D‖² / E‖δW a‖² is one
minus the algebraic outside share plus the cross term with O(δW) and the second-order dU dVᵀ: the input-weighted
inside share, already reported. With real gates it is the inside share with each subcomponent's part weighted by g₀ at
that position: the part of the update's effect that lands on pieces W₀ already uses on that input. That is the task
ladder's coverage (`reports/plans/2026-10-15_plan_task-ladder.md:36-39`) on the finished δW instead of the first
gradient, with per-position gates instead of mean gates. Coverage's prediction was null (P1, E9/E12). For the adjoint
split, Σ(C₁ − C₀) = c K(δW) and the ungated ratio is the reproduced share R. For a teacher-trained student,
Σ(C₁ − C₀) = δW_student, which is a different update, and the g ≡ 1 ratio is the student/teacher size ratio E12
Amendment 1 already tabulates (0.87 to 1.35, `reports/analyses/2026-11-02_analysis_e12-teacher-route-split.md:105-110`).
So the direct term restates the outside share or R, with gates multiplied in.

**Re-switching term.** R = (g₁ − g₀) C₀ a has no δW in it. δW enters only through g₁, Γ run on W₁'s activations (and
through a, if a is taken from W₁). Γ's forward is clamp(x, 0, 1) (`states.md` § 1), so gates at 0 or 1 do not move and
gates in between move by Γ's Jacobian times the activation change. To first order ‖R‖ is proportional to ‖δW‖, so the
ratio has a finite limit as δW → 0, but the limit is a property of Γ and C₀ along the update's direction, not of what
the update did. A random δW of the same input-weighted size has a ratio of the same kind. Dividing by E‖δW a‖² is a
statement about relative size only: the ratio can read 1 for a re-gating change orthogonal to everything the update
changed. In units, R is the change in this matrix's masked output; it is large wherever a gate near threshold sits on
a subcomponent with a large output on that input, and zero for a gate that moves on a subcomponent whose v_cᵀa is zero,
which is the one useful thing in it (question 1).

**What it is blind to.** Behaviour, by construction: no token, no loss. A large R on the target path and a large R
elsewhere read alike. Since the direct term is a δW statistic (E51: no δW statistic told copy from cipher,
`experiments/TRIED.md:107`) and the re-switching term is a Γ-response statistic that does not see what was learned,
I expect neither to separate E36's learned-nothing fine-tune. E36's code fine-tune did move the activations (held-out
loss 0.05 nats worse), which is all Γ needs.

**Γ pairs with W₁.** With a split, g₁ is W₁'s and identical to do-nothing's, so the re-switching term is identical to
do-nothing's for every split of the same δW (exact). With teacher training g₁ is the student's; E12 Amendment 1 saw r
fall on rand32b's students from that alone (0.187 → 0.132), and the norm version would inherit it.

## Answers to the four questions

1. **No, beyond gates-times-inside-share it restates existing measures; and it measures the user's definition only
   through a weighting.** Direct = inside share (least squares) or R (adjoint) with g₀ multiplied in; re-switching =
   Γ's response in output units, a new number but not about the update. The user's 2026-10-27 definition, "the
   difference in gating before and after fine-tuning", is g₁ − g₀ itself; the records show no all-position measurement
   of it (`records.md` § 4). E|g₁ − g₀| per subcomponent and position, pooled over Pile rows, gives the definition
   plainly. ‖(g₁ − g₀) C₀ a‖ adds the weighting by what the subcomponent does on that input, which says whether a gate
   change could matter; the projection ⟨(g₁ − g₀) C₀ a, δW a⟩ adds whether it matters in the update's direction. If one
   number is to be read, it is the projection, not the norm.
2. **Do-nothing: direct 0, interaction 0, re-switching unknown (Γ's response); learned nothing: the same kind of
   numbers; random δW: the same kind; known control: 1 by construction.** The table is below. The measure cannot
   separate the first three by anything derivable, and I guess it does not separate them when measured. It also has no
   W₀ reading (0/0), so the design check's "value at W₀" cannot be given; only the numerators in absolute units read 0
   there.
3. **Report the terms as sizes, never as shares; or switch to the projection, which sums to 1 exactly.** For the
   norm form: one a per matrix (W₀'s activation of that input, so all four terms see the same vector), all positions of
   512-token Pile rows (Γ's trained length; 41-token windows are off its distribution,
   `reports/reviews/2026-10-27_review_gamma-bidirectional-windows.md`), pooled by summing numerators and denominators
   over positions, and reported per matrix as well as pooled, since the pooled sum is dominated by down_proj and o.
   None of this removes the cross terms. The projection S_X with the residual S_res = 1 − S_D − S_R − S_I needs no
   pooling choice beyond the same.
4. **Yes, 67M only, about $0 and 20 minutes; Qwen no.** Everything needed is on disk (`ls -lL`, 2026-11-02): fine-tune
   δW fp32 113 MB at `~/commute-data/results-large/regating_table/{ident32,ident32b,rand32,rand32b}_s{0,1}.pt`,
   `code_step10.pt`, `code_step100.pt`, windows beside them; teacher-trained C₁ at
   `~/commute-data/results-large/new_vs_existing/amend1/{T}_s{0,1}.pt` (60 MB offsets, bf16; rebuilt C₁ carries that
   rounding); `teacher_training/amend8/amend8c/uv0.pt` for W₀'s alive (U, V) without the model. Least-squares and
   adjoint C₁ come from `split.split_state` / `split.adjoint_state` on CPU. Γ is the released
   `artifacts/vpd/s-55ea3f9b/model_400000.pth`. New code is needed: a `compute_ci`-like loop that keeps `out.cache` (a)
   and g at every position under W₀ and under W₁ (`vpd.py:167-188` discards the cache), then two factored products per
   term, U₁ᵀ(g₁ ⊙ V₁ᵀa) − U₀ᵀ(g₀ ⊙ V₀ᵀa), no dense matrix, in fp32 (δW is 1 to 2 % of W; bf16 rounding is 0.4 % per
   element, so differences of bf16 masked outputs are not readable). Cost per state on 128 Pile rows × 512 positions:
   two Γ passes and ~65 TFLOP of products, order a minute on an L40; g is 156 KB per position per state, so hold one
   chunk of 16 rows at a time. Ten states fit one `debug` job. Qwen: no scored C₁ state exists, two g's at 4,096
   positions (11 GB each) do not fit beside the 45 GB resident on an L40, and a split of δW is ~45 min of GPU. Not for Qwen.

## Expected values (norm ratio as proposed; the projection S_R in the last column)

"Algebra" = follows from the definitions; "guess" = my expectation, unmeasured. Direct and interaction use the
least-squares C₁ unless stated. Pooled over all positions of Pile rows.

| state | direct | re-switching | interaction | sum of norms | S_R (projection) |
|---|---|---|---|---|---|
| W₀ (δW = 0) | 0/0 (algebra) | 0/0 | 0/0 | undefined | undefined; numerators 0 |
| do-nothing, copy fine-tune (ident32) | 0 (algebra: C₁ = C₀) | unknown, Γ's response; guess 0.1 to 3 | 0 (algebra) | = re-switching | guess 0.2 to 0.6 |
| least-squares split, same δW | gated inside share, ≤ 1 − outside share (algebra); guess 0.3 to 0.6 | identical to do-nothing's (algebra: same g₁, same C₀) | nonzero, small; guess < 0.1 | sum plus cross terms, not 1 | S_D + S_R + S_I + S_res = 1 (algebra) |
| learned-nothing fine-tune (code step 10, E36 r 0.82) | a δW statistic; guess the same range as copy (E51) | guess the same order as copy at matched E‖δW a‖² | guess < 0.1 | not separable from copy (guess) | guess 0.2 to 0.6, not separable; the one number that could prove me wrong |
| random δW of matched E‖δW a‖² | the gated null, below the ungated null (1 − r_U/d_out)·read_out (algebra for the ungated) | guess smaller than trained δW, Γ reacting less to directions off the data | ~0 | not 1 | guess ~0 to 0.2 |
| known control: C₁ = (1 + α)C₀ on pieces with g₀ = 1 at every position, δW = Σ α_c C₀_c | 1 (algebra) | 0 to first order (activations move only by δW) | 0 | 1 | S_D = 1 (algebra) |
| teacher-trained student, g ≡ 1 | student/teacher size ratio, 0.87 to 1.35 (algebra, measured E12 A1) | 0 | 0 | the same | ⟨δW_s a, δW a⟩/‖δW a‖², a cosine times that ratio |

The control that runs first, alone: the last two rows are code checks. With g set to 1 everywhere the least-squares
state's direct term must reproduce 1 − `weighted_fractions`' `omega_act` (`split.py:129`) within the cross term,
and the student state's full term must reproduce the E12 Amendment 1 size ratios; stop if either misses.

## /design check

- **Goal.** Q5 of the re-gating line, what re-gating is and whether it can be trusted; it would give the line a
  weight-side column beside r. Indirect: no success criterion reads it.
- **Nearest tried.** Coverage (task ladder, 2026-10-15; P1 null), the outside share's four blocks (E2-E9), E51's
  weight statistics (no separation), E12 Amendment 1's size ratios; and r itself, whose do-nothing numerator is the
  same object as the proposed re-switching term in another unit. New: an all-position, target-free reading; not new:
  what it is blind to.
- **Patch or step.** A patch on r's unit; the step would be an instrument that passes the learned-nothing null, which
  only the projection form can even attempt.
- **Decisions.** Relies on the do-nothing reference and the split vocabulary; departs from none. It does not touch the
  log-odds choice for the route split, for which no decision record exists (`records.md` § 1); if the user wants that
  choice recorded, `decisions.md` should get a row either way.
- **Result that changes what we do.** Only the projection's learned-nothing null: S_R(code) well below S_R(copy)
  would make it the line's instrument; the same reading drops the idea. The norm ratio has no such outcome.
- **Apparatus arm first.** The two code checks above (g ≡ 1 identities), then do-nothing on the copy fine-tune, then
  the learned-nothing fine-tune, then random δW.
- **The numbers.** Three per state (S_D, S_R, S_res pooled; per-matrix kept as reference), the W₀ row is undefined
  by construction and the numerators are reported in absolute units instead; expected values as in the table.
- **Class.** Analysis of saved states on the 67M, $0 (approved class 1 for the run); the measure itself is a new
  method, so the user decides whether to build it.

## Smallest $0 version, if the user wants it

One `debug` job, L40, about 20 min, states named in question 4: the two code checks; then S_D, S_R, S_I, S_res on 128
Pile rows for do-nothing and least-squares states of ident32 s0/s1 and rand32 s0/s1, code step 10 and step 100, and a
Gaussian δW matched per matrix on E‖δW a‖² to ident32 s0. Filed before the run: S_R(code) ≤ 0.1 and S_R(ident32) ≥ 0.4
passes; S_R(code) within 0.15 of S_R(ident32) drops the idea. Not before E12 Amendment 2's debug job and not on Qwen.

## Departures from the literature

None cited; the quantities are this project's.

## Not run

Nothing; no GPU, no script. The algebra was done by hand (expansions and identities only).
