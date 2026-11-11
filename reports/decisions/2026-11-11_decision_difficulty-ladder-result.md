# Decision: what should the gate page do with the graded reading, now that the difficulty ladder did not show the rewiring share falling? (gate 11-13)

**Blocking.** This goes against a filed expectation, and the gate page (due 11-13) cannot state the graded reading until it is settled.

**Question.** The difficulty ladder did not show the rewiring share falling as the target gets harder for the plain model. What should the gate page do with the graded reading?

**What we have.** The rewiring share is the part of a fine-tune's output change that one affine map, written into the plain model's activations at one site, can reproduce. The ladder has three fine-tunes in order of difficulty (the last number is odd, the parity of a sum of two numbers, the parity of a sum of three), two seeds each.
- **The share does not fall down the ladder.** Rescaled between a planted ceiling (1) and a planted floor (0) of the same gap, it read 0.47, 0.07, 0.47 at one site, and 0.70, 0.59, 0.63 at two sites.
- **It follows neither probe.** The plain model's small non-linear (MLP) probe for the target read 1.0, 0.98, 0.72 down the ladder, and its linear probe read 1.0, 0.58, 0.51.
- **It goes against the filed expectation.** At one site the filed expectation was "fall, then level". "Out of order" was read.
- **Caveats.** The sum-of-three fine-tunes passed the learned check (KL ratio 0.57) but end as far from their target in total variation (0.25) as the plain model is (0.23), so the hardest rung was barely learned. The first two rungs are saved readings on two-number windows; the third is new, on three-number windows, against its own ceiling and floor. The premise check said "doubtful" for this reason.

**Options.**
- **(a) One more run, $0, cluster debug, about 2 h (needs a new cell).** Retrain the sum-of-three rung until it closes on its target, and re-read all three rungs on the same three-number windows. It tells whether the rise came from partial learning or from the change of windows. If the share still does not fall, (b) follows. **Recommended.**
- **(b) Take the ladder as read.** The gate page says the share is not graded by difficulty on this ladder, and reports it within matched pairs only.
- **(c) Drop the graded reading.** The gate settles on the cut-off or on dropping the wording.

**Recommendation: (a).** It is the cheapest way to tell a real null from an artefact of the third rung, and nothing else on the ladder can be read until the third rung is learned. Nothing runs until you choose.

**Evidence.** `reports/analyses/2026-11-11_analysis_difficulty-ladder.md`, `experiments/difficulty_ladder/results/readings.json`. Earlier card: `reports/decisions/2026-11-11_decision_definition-graded-or-cutoff.md`.
