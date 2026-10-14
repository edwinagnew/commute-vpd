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

## Not the goal

- Improving VPD itself, or re-deriving its training recipe.
- Re-running the Goodfire decomposition from scratch unless the projection experiments demand it.
