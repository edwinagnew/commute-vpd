# Decision: four repairs the definition needs before the paper states it formally

*2026-11-18, definition-lead. From the review `reports/reviews/2026-11-18_review_definition-causal-abstraction.md`.
Each repair changes the definition's content, so each is yours to accept or refuse; nothing has been changed.*

**Measured so far.** Every reading so far uses an exact unit partition: neurons, heads, MLP layers or blocks, with no
remainder part. So no reading changes under any of the four repairs. They matter when the paper states the definition
formally, and for the first reading with an SAE, a VPD split or a crosscoder.

**1. Should a decomposition read on F be allowed to use F, if the rule for doing so is fixed before any reading?**
(review's D1)
- *Problem.* The definition says the decomposition is declared or fitted on M only. To read VPD components on F we
  must split F's weight change over the components, and every useful split is computed from that change, so it is
  fitted on F. Only the do-nothing split is not, and it reads every change as new. A crosscoder, trained on both
  models' activations at once, breaks the same rule.
- *Review's repair:* "F̃ is produced from M̃ and F by a rule declared before any reading (the same encoder; a named
  split of δW), never chosen to change the reading."
- *Changes:* content of the guard, not the sentence. No reading; future VPD and crosscoder readings.
- *Recommendation:* accept.

**2. Should everything that is not a part, the output above all, run M's mechanism in every hybrid?** (review's D2)
- *Problem.* The soft intervention from the original model gives M's rule only to parts. The output layer and any glue between parts keep F's. Example:
  a fine-tune that changed only the unembedding keeps its behaviour with every part on M's rule, so it reads as
  needing no parts, and "the soft intervention on every part gives M" is false. For an SAE at one site, the whole network after that
  site sits in the output.
- *Review's repair:* "Every variable of N other than the input lies in some part's cell; or, the output (and any
  non-part variable) runs M's mechanism in every F_K."
- *Changes:* content, a missing clause that makes a stated property true. The 67M readings already set the output
  and embeddings to M's, per an earlier decision row; the review did not re-check the code. No reading is expected
  to change.
- *Recommendation:* accept, in its second form, which matches what the readings did.

**3. Should every reading report how much of the parts F needs is remainder?** (review's D3)
- *Problem.* With a remainder part the merge back is exact for any encoder, so the error ε is always 0. A weak SAE
  that encodes little pushes F's change into the remainder; the remainder becomes the needed part, the reading says
  "new", and ε = 0 hides why.
- *Review's repair:* "report, beside ε, either the share of K that is remainder (the level note's Part B proposal) or
  ε of the same decomposition with its remainder removed."
- *Changes:* what is reported, not the definition. No reading, since none used a remainder.
- *Recommendation:* accept, with the remainder's share of the needed parts.

**4. Should M's and F's refined models be required to share one order of computation?** (review's D10)
- *Problem.* A part's inputs are whatever its mechanism reads, so the two refined models could in principle differ.
  If part a reads part b in F's model and b reads a in M's, a hybrid with the soft intervention on b alone contains a loop and cannot
  run. Every current case follows the network's layer and position order, so this never happens yet.
- *Review's repair:* "one clause requiring a shared causal order."
- *Changes:* content, a stated condition. No reading.
- *Recommendation:* accept.

**Lead's recommendation.** Accept all four. Urgency: soon, before the paper's formal statement; nothing runs on it.

**Where it sits.** `theory/2026-11-18_definition_consolidated.md` (Open, item 5); `GOALS.md`, working definition
(open list). Log: section "Pending decisions", card `decision-definition-repairs`.

**Not run.** No reading under any repair; the review's unchecked point, whether the 67M code puts the output and
embeddings at M's mechanism in every hybrid, was not checked here either.

*Names in the reports: D1, D2, D3 and D10 are the review's defect numbers; F̃ and M̃ are the refined models; F_K is
F with M's rule on every part outside K.*
