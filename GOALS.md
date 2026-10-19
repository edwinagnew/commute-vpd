# Goals

The aim of the project, in one place. Owned by the user: agents propose edits, the user approves them. Read this
before proposing any new experiment, task, architecture or theory direction. Working choices that could be revised
live in `decisions.md`, and what has been tried lives in `experiments/TRIED.md`.

*Status: 2026-10-14. The question is the user's (verbatim below); its precise form was given by the user after round 1.*

## The question (user, 2026-10-14, verbatim)

> Given a (VPD) parameter decomposition, can one commute small gradient updates from post-training past the
> parameter components. If so, this gives "continuous interpretability" (within some limit). If not, is it because
> post-training teaches new capabilities (vs strengthens existing ones), or maybe just parameter decompositions are
> too brittle.

## The question made precise (user, 2026-10-14)

> Is there a simple operation that maps a post-training update (say, a LoRA) to an update on the parameter
> components, such that the resulting decomposition is faithful with respect to the new model? Or, a stronger
> version: such that the resulting decomposition is pretty much what you would get if you re-decomposed the updated
> model? Weight updates can be assumed small.

- **The map.** A simple operation Φ takes the decomposition D of W (subcomponents {U_c, V_c}, causal-importance
  function Γ, residual Δ) and an update δW, and returns D' = Φ(D, δW) for W + δW. "Simple" excludes re-fitting the
  decomposition. Examples: projecting δW onto the subcomponents; pulling the update back through W = Σ_c U_c V_cᵀ,
  i.e. applying the same training step to U and V instead of W.
- **Weak version (current target).** D' is a faithful decomposition of W + δW. The *components* must carry the new
  model: Δ may not absorb δW. Otherwise the trivial map "leave everything, put δW in Δ" passes. In VPD terms, D'
  scores on W + δW about as well as D scores on W, under the same losses (weight reconstruction by the components,
  masked-output reconstruction, sparse causal importance).
- **Strong version (postponed until the weak version is settled).** D' ≈ re-decomposition of W + δW, measured
  against a δW = 0 warm-start control and the cross-seed baseline.
- **If no simple Φ works.** Is that because post-training teaches new capabilities rather than strengthening
  existing ones, or because parameter decompositions are too brittle?
- **Canonical update:** LoRA, plus full-weight fine-tunes.
- **Γ:** frozen for now (see `decisions.md`), although updating it may turn out to be essential.

## Terms (the definitions both of us hold; 2026-10-19, after a chat found three of them had drifted)

- **Φ** maps (D, δW) to D', for one update step. Along training, Φ is applied at every step and the question is whether
  the error accumulates; the run is the limit, not the object. Experiments to 2026-10-19 scored finished 10-step runs
  one-shot; see `reports/2026-10-19_decision_success-criterion.md`.
- **Pull-back:** the same fine-tune run on the alive (U, V) with Γ and Δ frozen. One SGD step moves W by
  K(G) = UUᵀG + GVVᵀ, the part of the gradient the pieces can express. It consumes the training step, so it needs the
  loss and data. Runs so far used Adam on (U, V), not SGD.
- **Lift:** the weight change δW written as the smallest first-order change (dU, dV) of the alive pieces, with the
  obstruction O(δW) left in Δ. It consumes the weight difference, so it is the only Φ that applies to a LoRA handed
  over as BA.
- **Faithful (weak version):** D' passes VPD's own checks on W + δW about as well as D passes them on W (stochastic-mask,
  adversarial-mask, CI-masked and unmasked reconstruction KL, and the sparsity loss), with Δ not absorbing δW. The
  route split (direct, re-gating, interaction, masked-off, residual) describes mechanism and does not decide this.
- **Γ, the gate:** one network reading every matrix's input at every position. An edit in one layer moves CI values in
  others.
- **Small:** measured by a step sweep, never assumed.

## Where an update goes (mechanism hypotheses, user-approved 2026-10-16; they do not decide the weak version, 2026-10-19)

Under a map Φ, the change in the new model's behaviour splits exactly into five routes: (a) **direct**, the changed
subcomponents at the old gates; (b) **re-gating**, the frozen Γ gating differently on the new activations; (c) their
**interaction**; (d) **masked off**, carried by subcomponents Γ does not gate on; (e) **residual**, not carried by the
subcomponents at all. A sixth route, Γ's own parameters, is zero while Γ is frozen. The weak version holds when (d) and
(e) are small. Measurement: `reports/2026-10-16_prediction_ladder-routes-amendment.md`.
The weak version itself is decided by VPD's checks on D' (`reports/2026-10-19_decision_success-criterion.md`); a small
direct share (a) is a finding about mechanism, not a failure of Φ.

- **Strengthen:** (a) dominates. Post-training sharpens subcomponents that already do the job.
- **Reveal:** (b) + (c) dominate. The update is a small change that Γ amplifies by switching on subcomponents that
  already exist. The concentration test points this way on the emoticon task.
- **Create:** (e) dominates when Δ may move. When it may not, (d) dominates or inside training fails to learn.
- **Brittle:** the split is unstable across seeds or across small changes to Φ.

## Not the goal

- Improving VPD itself, or re-deriving its training recipe.
- Re-running the Goodfire decomposition from scratch unless the projection experiments demand it.
