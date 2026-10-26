# Plan: the concentration test. Is the pull-back's diffuse update the optimiser, and can a few subcomponents carry it?

*2026-10-15, coding agent. Confirmed by the user ("go for it") after the design check in chat. Predictions:
`experiments/concentration_test/results/predictions.json` (filed before training). Code: `experiments/concentration_test/run.py`.
RunPod 4090, under $1.*

**Why.** The identity test found the pull-back's emoticon update spread over thousands of subcomponents, none of them
the paper's `2359`. Two readings: Adam spreads it (an optimiser artefact), or the behaviour can't be learned by a few
subcomponents inside this decomposition.

**Arms** (pull-back, arm A recipe, 30 steps, Γ frozen, Δ fixed):
- Adam on all alive subcomponents (round 2);
- plain SGD on all alive;
- Adam on only the top 1 / 10 / 100 subcomponents by the gauge-free first-order importance |u_cᵀ G_W v_c|;
- Adam on only `2359` and its five siblings.

Learning rates are picked by amendment 1's rule.

**Scored.**
- P1: SGD concentrates (top 10 remove ≥ 50% of the gain). Predicted: fail.
- P2: top 10 learns faithfully. Predicted: pass.
- P3: siblings learn faithfully. Predicted: pass.

"Faithfully" needs all three of:
- the lr rule finds a learning rate (P(o) ≥ 0.9, KL elsewhere ≤ 0.110);
- Pile ratios ≤ 1.5;
- **at the trigger**, the CI-masked component model gives P(o) ≥ 0.8 × target.

**Measurement scope** (per `reports/reviews/2026-10-15_review_faithfulness-scope.md`):
- Pile metrics are blind to the trigger, so trigger-local metrics are added on the held-out windows.
- Their positive control is do-nothing, which must fail (CI-masked P(o) < 0.5 × target). If it doesn't fail, the
  trigger criterion is unscorable.
- PGD noise floor: W₀ with adversary seeds 0, 1, 2.
- The identity measure is gated on a positive control: swapping v or randomising u must fail it, with rotations of
  20° and 45° reported.
- Γ against the update (reported): what share of the update sits in subcomponents with g > 0.5 at the trigger.

**What would change next.**
- P2/P3 pass and P1 fails: Adam's spreading is avoidable, and the decomposition's labels are usable when the update
  is constrained.
- P3 fails while P2 passes: the labelled component is not where the behaviour can be learned.
- Both fail at the trigger: a frozen Γ cannot gate in the new behaviour, which points to round 4 (Γ updates).
