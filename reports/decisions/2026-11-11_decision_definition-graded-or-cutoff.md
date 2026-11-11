# Decision: keep the rewiring share as a graded measure, test it on a difficulty ladder, or settle it at the gate? (rung L1.4, gate 11-13)

**Question.** What should part 1 do now that the built controls keep failing? Options: keep going with this task,
change the task, change the definition, or look to other work (the crosscoder papers) for a known built case.

**What we have.** The rewiring share is the part of a fine-tune's output change that one affine map, written into the
plain model's activations at one site, can reproduce.
- **It ranks well within pairs.** In all five matched pairs, every reuse seed reads higher than every seed of its new
  twin (gate page). Today's run (P1.14) read the sum-parity fine-tune at 0.06 and 0.07 against the last-number-odd
  control's 0.47 and 0.46. A planted edit the map can reach exactly read 0.985, and one it cannot reach read about 0.
- **Its absolute scale does not hold up.** A clean reuse control reads under the 0.5 cut-off, and the search reads only
  0.84 on an exact target edit (P1.13). So the cut-off sits above four of the five reuse sides.
- **Why no built control works (advice from Fable, checked against the gate page).** The wording counts a quantity as
  "computed" when an affine map can read it. So any built case is linearly unreadable by definition, and rung L1.4
  asks for a built case whose label does not follow from that. The premise check on today's run said the same.
- **The crosscoder papers have no built case to recreate.** They find directions in one model's activations that the
  other lacks, not new computations. Reuse can show up as "new". No one has validated them on a fine-tune with a known
  label, and their own authors read the clearest "new" latents (refusal, misalignment) as reuse
  (`lit_review/crosscoders_and_novelty.md`).

**Options.**
- **(a) A difficulty ladder, with the share read as graded (recommended).** One new cell, $0, cluster debug, about
  2 h. Fine-tunes on the same layout, in order of difficulty: the last number is odd, then the parity of a sum of two
  numbers, then of three. All are read in one job between the planted floor and ceiling, at one site and at two.
  Beside each, a linear and a small non-linear probe of the plain model for the target. Filed prediction: the
  rescaled share falls step by step and follows the non-linear probe, not only the linear one.
  - If it holds, the definition becomes "how much of the change is reachable from the base model's own computations,
    by a map of stated size". It is graded, reported within pairs and along ladders, and has no absolute cut-off;
    "built" is the low end where the non-linear probe also fails.
  - If the share is flat, out of order, or only mirrors the linear probe, it adds nothing over a probe, and that is
    the case for dropping it at the gate.
  - Either way, the 0.5 cut-off is set aside, which only you decide.
- **(b) Settle it at the 11-13 gate with what we have.** No run. The gate page already gives the five-pair ranking,
  the scale problem and today's reading. You choose there between graded, cut-off or drop.
- **(c) Drop the wording now** and look for another definition before the gate. That throws away the one result that
  holds: the within-pair ranking.
- **(d) Recreate the crosscoder findings at scale.** Not recommended as a source of truth, since there is no label to
  check against. A crosscoder scored on our labelled pairs is a fair part 2 baseline, for compare-lead.

**Recommendation: (a)**, with the cut-off question put at the gate. It is the cheapest run that can tell whether the
share measures something a probe does not. Nothing runs until you choose.

**Evidence.** `reports/analyses/2026-11-11_analysis_sum-parity-beside.md` (today's run),
`reports/analyses/2026-11-11_analysis_h-diagnosis.md` (the scale), `reports/analyses/2026-11-10_analysis_gate-page-working-definition.md`
(the five pairs and the wording), `lit_review/crosscoders_and_novelty.md`.
