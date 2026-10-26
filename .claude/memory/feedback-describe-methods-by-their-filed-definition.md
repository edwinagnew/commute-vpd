---
name: feedback-describe-methods-by-their-filed-definition
description: When explaining a method (pull-back, lift, projection) to the user, use the exact definition from GOALS or the published artifact; a loose paraphrase reads as a "curveball" and makes the user think we are talking past each other
metadata:
  type: feedback
---

On 2026-10-19 I described the pull-back as "finds whatever change to the pieces lowers the task loss". The user: "wtf
since when is that how pullback worked? ... every time i think were on the same page youll throw some crazy curveball".
The artifact ("Pull-back vs Lift", 2026-10-15) and GOALS line 25 define it as "run the same fine-tune on U_alive,
V_alive with Δ and Γ frozen", which is the same thing, but the paraphrase sounded like a search over arbitrary edits.
In the same chat I said "the pull-back is not a map of the update", which contradicts GOALS, where it is listed as an
example Φ.

**Why:** the user holds the definitions from GOALS and the artifact. A paraphrase that drifts from them costs trust
even when it is equivalent, and a claim that contradicts GOALS is simply wrong.

**How to apply:** (2026-10-19: GOALS now has a "Terms" section with the shared definitions of Φ, step vs run, pull-back, lift, faithful, Γ. The user declined a CLAUDE.md rule to quote it as overkill; reusing it is my habit, not a repo rule.)
- Before characterising a method in chat, re-read its definition in `GOALS.md`, `decisions.md` or the artifact and
  reuse that wording ("the same fine-tune run on the pieces", "K(G) = UUᵀG + GVVᵀ per step").
- Say what is new only as a delta from that definition (e.g. "all runs used Adam on (U, V), not SGD").
- The one real distinction worth stating: the pull-back consumes the training step (needs loss and data); the lift
  consumes the finished weight difference. For a handed-over LoRA only the lift applies.
Related: [[feedback-define-terms-when-reporting]], [[feedback-check-metric-power-before-claiming]],
[[project-success-criterion-reading-a]].
