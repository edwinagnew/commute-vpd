# Review: E319 (P2.87) subspace-share page

*2026-11-27, redteam-lead. Page `reports/analyses/2026-11-26_analysis_p2-87-subspace-share.md` at 44146a60; results
`experiments/subspace_share/results/updates/*.json`, `lrgrid.json`, `control.json`.*

I recomputed both tables of the page with jq from the 24 update JSONs. Every number matches: the at-one steps, gains,
size and KL ratios, end states, and the per-task means and seed ranges. The filed rule reads "rival a" on the at-one
means, as the page says. The one MUST is in the prose. The page sorts the at-one shares by the new-side range, not
against each task's own fine-tune. Measured against its own fine-tune, the at-one drop is about 0.4 on all four copying
tasks, e62/copy included, and not on three.

## Recomputed
- At-one shares against each task's own fine-tune (fine-tune mean → at-one mean, drop):
  - e46/ident32: 0.78 → 0.31 (−0.47)
  - e62/copy: 1.09 → 0.68 (−0.41)
  - c3/copy: 0.84 → 0.46 (−0.39)
  - e58/copy: 0.77 → 0.38 (−0.39)
  - e58/rank1: 0.86 → 0.74 (−0.12)
  - c3/rank1: 0.84 → 0.70 (−0.14)
  - e54/elicit: 0.77 → 0.79 (+0.02)
  - e53/lowercase: 0.76 → 0.91 (+0.15)
- End against the fine-tune: the largest gaps are e53/lowercase +0.10 and e54/elicit −0.09, about three times the
  0.031 batch spread. The end share ranges from 0.674 to 1.077, and the fine-tune share from 0.752 to 1.099.
- Rule margin: the five tasks at or above 0.65 include e62/copy at 0.683, 0.033 over the bar, which is about one batch
  spread. With e62/copy under the bar the count is 4 of 8, which is not more than half, so the reading would be
  "between".
- LR grid (`lrgrid.json`, e62/copy s0): all three multipliers, ×10, ×30 and ×100, reach gain 1 at step 1. The grid
  chose its lowest point, so it does not bound the rate from below. At-one is step 1 on e62/copy (all seeds) and c3/copy
  s0 and s2. The step-1 norm is about lr·√442,136 ≈ 0.20, against about 0.16 for a full fine-tune's first Adam step. The
  page's "about as far as the fine-tune's first step" holds.

## Findings
1. **MUST: the copying-task pattern is described wrong.** The opening says at-one "sits lower on three copying tasks and
   near or above the fine-tune's elsewhere"; e62/copy reads 0.68 against its fine-tune's 1.09, which is not near. The
   body then calls e62/copy's 0.68 a counter-example to the copying pattern. Against each task's own fine-tune, all four
   copying tasks drop 0.39 to 0.47, the two rank tasks drop 0.12 to 0.14, and e53 and e54 hold or rise. Rewrite the
   opening, the share table's "Reading it" and the copying paragraph on that basis. Keep the new-side range only as a
   second reference.
2. **SHOULD: the "rival a" label rests on one task at one batch spread above the bar** (e62/copy, 0.683 against 0.65).
   Say so where the rule is read and in the opening sentence about it.
3. **SHOULD: the at-one state is a single step on three updates of two tasks**, so its value depends on a step size
   that the grid did not bound: every multiplier reached the gain at step 1. Add this to "Blind to".
4. **SHOULD: "about as many trainable numbers as a small LoRA" invites "low rank".** The update's rank can reach 111 on
   attention matrices and 175 on MLP ones, against the LoRA's 8. Say this once in "The update", and in the opening.
5. **NIT:** "close to the fine-tune's own on every task" at the end: give the largest gaps (+0.10 e53, −0.09 e54), which
   are about three batch spreads.
6. **NIT:** "The share's value is closer to a property of the task" is a reading of the method that this run does not
   test (no new-side reference). Mark it as a hypothesis, or cut it.

## Not run
- No independent recomputation from the saved updates; this is a transcription check of the JSONs.
- The figure, which is not drawn yet.
