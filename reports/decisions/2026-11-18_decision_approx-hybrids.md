*Closed 2026-11-18: featurization made exact; question no longer arises.*

# Decision card: when a featurized model fits the fine-tune only approximately, where do we compare?

*2026-11-18, later: vocabulary moved to Geiger et al. 2025's (`decisions.md`, "Featurized model replaces
refinement"); content unchanged. A refinement is now a featurized model, a part a variable, a hybrid an intervened
model. The file name keeps the older word "hybrids".*

**Measured so far.** Nothing bears on this: every reading so far is exact (every variable is a group of the network's
own units, or a split with a remainder), so ε, the error of Geiger's Def. 41 measured as mean KL on outputs between a
network and its featurized model, is 0 and the two choices below give the same number.

**Plain words.** Today you decided that a featurized model may be approximate: each network described by the variables
a method reads, with one featurizer fixed from the base model M and applied to both networks, and ε reported beside
every reading (`decisions.md`, "A refinement ... may be approximate", and "Featurized model replaces refinement").
When ε is above zero on the fine-tune F, F's featurized model F̃ does not reproduce F exactly. So should the test
"does this circuit preserve the behavior?" be run on intervened models of F̃ rather than of F, with the comparison
made between the two featurized models F̃ and M̃? That keeps the definition giving an answer in every case. In the
exact case F̃ and F coincide, so no reading changes.

**Options.**
- (a) Compare inside the featurized models: run the preservation test on intervened models of F̃ and take differences
  between F̃ and M̃. The definition is total. Raised and recommended by the writeup-lead (its small ask 1).
- (b) Compare against F itself and report ε_F beside the reading. The test uses the real fine-tune, but where the
  approximation is poor the reading may be undefined, and ε_F is the only warning.

**What the literature does (added 2026-11-18, from the review
`reports/reviews/2026-11-18_review_definition-causal-abstraction.md`, §4 Q3 and defect D4).** The standard in
replacement-model circuit work is to test a circuit inside the replacement model with its error nodes kept, so the
model is exact (Marks et al. 2024; Ameisen et al. 2025), or to normalise against the replacement model. Both amount
to option (a). The review's defect D4 adds that the definition is not total without this choice: when the featurized
model of F does not preserve the behavior, no set of variables can, so no reading exists under (b).

**Recommendation.** (a).

**Owner.** Raised by the writeup-lead; it is part 1's definition, so the definition-lead owns it. Urgency: no rush,
nothing runs on it.

**Where it sits.** `theory/2026-11-18_level-as-decomposition.md`, the "Open" paragraph in Part A;
`theory/2026-11-18_definition_consolidated.md`; `GOALS.md`, working definition (open list). Log: section "Pending
decisions", card `decision-approx-hybrids`.

**Not run.** No reading under either option, since all readings are exact; no case with ε above zero has been
measured.
