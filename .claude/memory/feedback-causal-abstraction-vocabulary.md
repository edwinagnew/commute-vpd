---
name: feedback-causal-abstraction-vocabulary
description: State the reuse definition in Geiger et al.'s causal-abstraction terms AND notation wherever they exist; featurized model, not refinement; coin only where Geiger has nothing
metadata:
  type: feedback
---

Use Geiger et al. 2025's vocabulary and notation for the definition of reuse as much as possible (user, repeated many times; 2026-11-18: "update your memory to use geiger's notation as much as possible").
- Terms: featurized model Ñ_i (featurizer fixed from N_0, shared; N_i an approximate transformation of it, Def. 41; exact transformation Def. 25 when ε = 0) — NOT "refinement"/"level"/"decomposition" (decisions row 2026-11-18 "Featurized model replaces refinement"); variables / modular features (§3.6), not "parts"; mechanism F_X (Def. 5), not "rule"; parents Pa(X) (Remark 6); soft intervention (Def. 10), not "swap"; intervened model M_I, not "hybrid"; behavior B and minimal sub-circuit (§3.5.2) for the needed set; interchange intervention (Def. 44) only for value patching; Solve(M) for a model's solution; total settings, Proj_X, signature Σ, alignment ⟨Π, π⟩ (Def. 31), maps (τ, ω).
- Notation: Geiger's symbols (F_X, Pa, M_I, Solve, Proj, τ, ω, Σ, Val) over home-made ones; JMLR definition numbers.
- Coin a word only where Geiger has none (remainder variable; the shared featurizer across two networks) and say it is ours.

**Why:** the user wants the paper inside the causal-abstraction literature; home-made terms (rule, part, swap, hybrid, refinement) kept creeping back and confused readers.

**How to apply:** every brief, card, log edit, theory note and paper text about the definition; grep new text for rule/part/swap/hybrid/refinement/level. Local copy of Geiger: ~/commute-data/lit_review_pdfs/survey/2301.04709/paper.txt. Related: [[project-working-definition-revised]], [[project-paper-term-refinement]].
