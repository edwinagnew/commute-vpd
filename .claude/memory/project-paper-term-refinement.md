---
name: project-paper-term-refinement
description: "Paper term for what the log called a level: \"featurized model\" (user 2026-11-18, replaces \"refinement\"); Geiger vocabulary throughout"
metadata:
  type: project
---

User 2026-11-18 (via definition-lead, decisions.md row "Featurized model replaces refinement", commit 213f15c8): the object first called a "level", then in the paper a "refinement", is now a **featurized model** Ñ_i of each network N_i. N_i is an approximate transformation of Ñ_i (Geiger Def. 41, JMLR; exact transformation Def. 25 when ε = 0); Def. 33 no longer used. One featurizer φ, fixed from the base model alone, applied to both networks; a remainder variable is one way to get ε = 0, not part of the concept. Vocabulary: mechanism (not "rule"), variables / modular features (not "parts"), soft intervention from the base model (not "swap"), intervened model (not "hybrid"), behavior B and minimal sub-circuit (§3.5.2) for "the parts F needs".

**Why:** user adopted Geiger's own vocabulary ("yes, adopt it and do the full pass"); "decomposition" had been rejected as VPD-sounding, "refinement" was ours.
**How to apply:** paper text uses these terms; main.tex §4 lines 84/87 still say "refinement" — fix in the next main.tex batch with the user's yes. Related: [[project-working-definition-revised]], [[project-paper-repo-push-freeze]].
