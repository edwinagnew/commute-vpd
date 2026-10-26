---
name: feedback-define-terms-when-reporting
description: When reporting results, define any recently introduced term (measure, transport, null) precisely, and say which claim a result bears on
metadata:
  type: feedback
---

On 2026-10-14 the user asked "What is frobenius fraction? Please be more precise with technical terms, especially if
its been recently introduced", after a results message used "Frobenius fraction", "capture", "null band" and "alive
dictionary" without definitions. They also could not tell which reversed predictions contradicted the VPD paper and
which contradicted their own hopes.

**Why:** the user reads the chat message, not the plan or the code where the terms were defined, and terms coined by
the agent mid-project are not shared vocabulary.

**How to apply:** the first time a term the agent coined appears in a message, define it in one line (formula if it is
a measure, with what 0 and 1 mean). For each result, say which claim it bears on: the paper's findings, the paper's
untested proposals, or the user's hypothesis, and whether a negative result is a real negative or a test with no
power. Related: [[feedback-flag-qualitative-surprises-loudly]], [[feedback-verify-before-announcing-a-cause]].

**2026-10-16 follow-up.** Explaining a notebook claim, the agent used "freezing gates", "trained piece" and "gates"
instead of VPD terms; the user read "frozen" as "Γ retrained" and asked for VPD terminology (subcomponent, causal
importance, Γ, CI-masked, Δ, target model) or precise new terms. They also asked the agent to "practice communicating
what things are measuring more clearly". **How to apply:** for each measurement say, in this order, the quantity read
out (e.g. P(o) of the CI-masked model at the trigger), what is held fixed, and what one thing is varied between the
compared conditions. No metaphors ("freeze", "piece", "route") in place of VPD terms.

**2026-10-19 follow-up (task ladder results).** The user called the reply "terrible": "gate" was used for CI values,
masks and Γ's output interchangeably; "splits its gain the same way as..." and "re-gating depends on the task, not on
the update" read as important but were not checkable sentences (no quantity, no scope); "round 4" was offered as a
decision without saying what the options were or what evidence bore on them. **How to apply:** never write "gate"; say
"CI value" (Γ's output) or "mask" (what multiplies a subcomponent in the CI-masked forward). Every claim sentence names
the number, the conditions compared, and how far it generalises (what was not varied). Never hand the user a choice
without each option's one-line description and the evidence for and against it.
