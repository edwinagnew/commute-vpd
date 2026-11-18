# Decision card: when a decomposition fits the fine-tune only approximately, where do we compare?

**Measured so far.** Nothing bears on this: every reading so far is exact (every part is a unit of the network itself, or a split with a remainder), so ε, the mean KL on outputs between a network and its refined model, is 0 and the two choices below give the same number.

**Plain words.** Today you decided that a refinement (what the log calls a level: a declared finer model of each network, whose parts are what a method reads) may be approximate, with ε reported beside every reading (`decisions.md`, "A refinement ... may be approximate"). When ε is above zero on the fine-tune F, F's refined model F̃ does not reproduce F exactly. So should the test "do these parts suffice for the behaviour?" be run against F̃ rather than F, with the comparison made between the two refined models F̃ and M̃ (M is the original model)? That keeps the definition giving an answer in every case. In the exact case F̃ and F coincide, so no reading changes.

**Options.**
- (a) Compare inside the refined models: run the keep-test against F̃ and take differences between F̃ and M̃. The definition is total. Raised and recommended by the writeup-lead (its small ask 1).
- (b) Compare against F itself and report ε_F beside the reading. The test uses the real fine-tune, but where the approximation is poor the reading may be undefined, and ε_F is the only warning.

**Recommendation.** (a).

**Owner.** Raised by the writeup-lead; it is part 1's definition, so the definition-lead owns it. Urgency: no rush, nothing runs on it.

**Where it sits.** `theory/2026-11-18_level-as-decomposition.md`, the "Open" paragraph in Part A; `theory/2026-11-18_definition_consolidated.md`; `GOALS.md`, working definition (open list). Log: section "Pending decisions", card `decision-approx-hybrids`.

**Not run.** No reading under either option, since all readings are exact; no case with ε above zero has been measured.
