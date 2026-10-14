# Goals

The aim of the project, in one place. Owned by the user: agents propose edits, the user approves them. Read this
before proposing any new experiment, task, architecture or theory direction. Working choices that could be revised
live in `decisions.md`, and what has been tried lives in `experiments/TRIED.md`.

*Status: draft by the coding agent, 2026-10-14, from the user's opening message. Awaiting the user's edits.*

## The question (user, 2026-10-14, verbatim)

> Given a (VPD) parameter decomposition, can one commute small gradient updates from post-training past the
> parameter components. If so, this gives "continuous interpretability" (within some limit). If not, is it because
> post-training teaches new capabilities (vs strengthens existing ones), or maybe just parameter decompositions are
> too brittle.

## Working reading (agent, to be confirmed)

- **Commute.** A decomposition D maps weights W to subcomponents {U_c, V_c}, a causal-importance function Γ and a
  residual Δ. A gradient update T: W → W + δW *commutes past* D if there is a transport T' acting on the
  subcomponents (reweighting or slightly modifying a few of them, Γ and Δ fixed) such that D(T W) ≈ T' D(W): the
  transported decomposition is parameter-faithful and mechanistically faithful for the updated model, and each
  update reads as "these mechanisms strengthened / weakened / changed".
- **Continuous interpretability.** If commuting holds for every small step, the decomposition is a continuous
  function of the weights along training and need only be re-fitted occasionally. The "limit" is the size or kind of
  update at which it stops holding.
- **Three branches if it fails.** (a) The update is spread across the decomposition, no better than across random
  subcomponents: new capability, or the subcomponents are not the right objects. (b) The frozen decomposition stops
  being faithful at W + δW even for tiny δW: brittleness. (c) The update needs new subcomponents outside the span:
  new mechanism.

## Not the goal

- Improving VPD itself, or re-deriving its training recipe.
- Re-running the Goodfire decomposition from scratch unless the projection experiments demand it.
