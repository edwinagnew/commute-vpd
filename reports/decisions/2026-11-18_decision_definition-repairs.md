# Decision: four repairs the definition needs before the paper states it formally

*2026-11-18, evening: the log card was rewritten shorter at the user's request (log v401); this file keeps the fuller text.*

*2026-11-18, definition-lead. From the review `reports/reviews/2026-11-18_review_definition-causal-abstraction.md`.
Each repair changes the definition's content, so each is yours to accept or refuse; nothing has been changed.*

*2026-11-18, later: vocabulary moved to Geiger et al. 2025's (`decisions.md`, "Featurized model replaces
refinement"); content unchanged. A decomposition or refinement is now a featurized model, a part a variable, a rule a
mechanism, a hybrid F_K the intervened model F̃ under the soft intervention I_K. The review's repairs are quoted in its
older words.*

**Measured so far.** Every reading so far groups the network's own units into variables: neurons, heads, MLP layers or
blocks, with no remainder variable. So no reading changes under any of the four repairs. They matter when the paper
states the definition formally, and for the first reading with an SAE, a VPD split or a crosscoder.

*2026-11-18, later: the user chose one fixed featurizer (`decisions.md`, "One fixed featurizer"). **Repair 1 is set
aside** (F's features are not fitted; the featurizer is fixed from M alone) and **repair 4 is automatic** (one
featurizer and one architecture give one causal order). Repair 3 was then adopted by the exact-featurization decision
(2026-11-18, `decisions.md`). Only repair 2 stays open.*

**1. [Set aside by the user's choice of one fixed featurizer, 2026-11-18.] Should the featurized model of F be allowed
to use F, if the rule for doing so is fixed before any reading?** (review's D1)
- *Problem.* The definition says the featurizer is fixed from M alone. To read VPD components on F we must split F's
  weight change over the components, and every useful split is computed from that change, so it is fitted on F. Only
  the do-nothing split is not, and it reads every change as new. A crosscoder, trained on both models' activations at
  once, breaks the same rule.
- *Review's repair:* "F̃ is produced from M̃ and F by a rule declared before any reading (the same encoder; a named
  split of δW), never chosen to change the reading."
- *Changes:* content of the guard, not the sentence. No reading; future VPD and crosscoder readings.
- *Recommendation:* accept.

**2. Should every variable outside U, the output above all, take M's mechanism in every intervened model?** (review's D2)
- *Problem.* The soft intervention from the base model gives M's mechanism only to the variables in U. The output
  layer and any glue between them keep F's. Example: a fine-tune that changed only the unembedding preserves its
  behavior with every variable in U on M's mechanism, so it reads as needing no variables, and "the soft intervention
  on all of U gives M" is false. For an SAE at one site, the whole network after that site sits in the output.
- *Review's repair:* "Every variable of N other than the input lies in some part's cell; or, the output (and any
  non-part variable) runs M's mechanism in every F_K."
- *Changes:* content, a missing clause that makes a stated property true. The 67M readings already set the output
  and embeddings to M's, per an earlier decision row; the review did not re-check the code. No reading is expected
  to change.
- *Recommendation:* accept, in its second form, which matches what the readings did.

**3. [Adopted 2026-11-18 by the user's decision to make the featurization exact.] Should every reading report how much
of the circuit F needs is remainder?** (review's D3)
- *Problem.* With a remainder variable the featurized model is exact for any encoder. A weak SAE that encodes little
  pushes F's change into the remainder; the remainder becomes the needed variable, and the reading says "new" without
  saying why.
- *Review's repair:* report the share of K that is remainder (the fit error ε it was to be reported beside is gone).
- *Changes:* what is reported, not the definition. No reading, since none used a remainder.
- *Adopted:* the remainder's share of the circuit F needs is stated beside every reading (`decisions.md`, "Exact
  featurization with a mandatory remainder", 2026-11-18).

**4. [Automatic with one featurizer and one architecture, 2026-11-18.] Should M̃ and F̃ be required to share one causal
order?** (review's D10)
- *Problem.* A variable's parents are whatever its mechanism depends on (Geiger's Remark 6), so the two featurized
  models could in principle differ. If variable a has parent b in F̃ and b has parent a in M̃, an intervened model with
  the soft intervention on b alone contains a cycle and cannot be run. Every current case follows the network's layer
  and position order, so this never happens yet.
- *Review's repair:* "one clause requiring a shared causal order."
- *Changes:* content, a stated condition. No reading.
- *Recommendation:* accept.

**Lead's recommendation.** Accept repair 2, the only one still open; 1 and 4 are closed and 3 is adopted (above). Urgency: soon, before the paper's formal statement; nothing runs on it.

**Where it sits.** `theory/2026-11-18_definition_consolidated.md` (Open, item 5); `GOALS.md`, working definition
(open list). Log: section "Pending decisions", card `decision-definition-repairs`.

**Not run.** No reading under any repair; the review's unchecked point, whether the 67M code puts the output and
embeddings at M's mechanism in every intervened model, was not checked here either.

*Names in the reports: D1, D2, D3 and D10 are the review's defect numbers; F̃ and M̃ are the featurized models; F_K,
in the review's words, is the intervened model, F with M's mechanism on every variable outside K.*
