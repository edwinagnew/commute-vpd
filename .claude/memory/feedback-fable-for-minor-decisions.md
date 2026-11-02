---
name: feedback-fable-for-minor-decisions
description: "Ask the user only the most important questions; minor calls (result readings, rule edge cases, next small step) go to a Fable agent and are acted on, FYI only"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0b5d59dd-0c8c-46be-84f0-8b63c7de6585
  modified: 2026-10-30T14:46:54.386Z
---

Only bring the user the most important decisions. For minor ones, spawn a Fable agent (`model: fable`) for advice and act on its recommendation, telling the user as an FYI.

**Why:** On 2026-10-30 the user said "youre asking too many questions. only come to me for the most important. use fable advice for minor things". They said it after an ask about how to read E56's k50 neuron-count difference, which came right after an ask to approve E56.

**How to apply:** "Most important" means things like a new experiment or line of work, spend above the budget or a gated run, or a result that reverses the main claim. Reading a result, edge cases in a rule, $0 amendments and choosing between branches of the plan go to Fable, and I act on its advice. Record its advice in the report or card, marked as Fable's. This loosens [[feedback-raise-decisions-often]] and the lead brief's rule that "Fable advises; it never decides" for minor calls. See also [[feedback-plain-approval-requests]].
