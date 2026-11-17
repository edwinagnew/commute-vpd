---
name: feedback-causal-tracing-definition-rejected
description: Why the user does not like the causal-tracing (base-to-fine-tune swap) definition of reuse; do not re-propose it without answering these objections
metadata:
  node_type: memory
  type: feedback
  originSessionId: 54466039-1baa-4f31-b9ee-66b74e23c531
  modified: 2026-11-09T15:25:10.493Z
---

The user does not like the tentative definition in `reports/plans/2026-11-09_plan_same-place-definition.md` (patch the original model's values for a named algorithm's steps into the fine-tune; passes if it answers as those values dictate). Not formally dropped, but they are seeking a better one in another session (2026-11-09).

The user's objections, their words:
1. "if patching from base to fine-tuned carries the capability, doesnt that imply the model didnt learn anything?"
2. "what does 'step-by-step account of that behaviour' even mean? if we had a good such account, couldnt we just interpret that before/after directly?"
3. "im not sure causal tracing is the right alternative to the probes we tried so far"
Earlier the same day: the swap is activation patching (causal tracing, ROME paper) and must be credited, with its known limitations addressed.

**Why (my analysis, which I told them and they did not dispute; reasoning only, nothing run):** the patch overwrites the value on the behaviour's inputs, so it cannot see how the fine-tune produced that value, which is usually what was learned. A fine-tune that memorises a table and writes the answer into an old "about to say w" direction would pass as reused. Requiring the value only on ordinary inputs opened this hole. Needing a hand-written algorithm presupposes the interpretation we lack, limits it to simple tasks, and makes "new" unprovable; on those tasks labels by construction already exist. It keeps the probes' need for a named variable.

**How to apply:** a replacement must look at how the fine-tune computes the answer on the new inputs, and must not need a hand-written algorithm. Option (c) (swap calibration) is not to run: ask 17 for its cell was withdrawn. `decisions.md` still has a row choosing (c), made before these objections. The user also hopes to build on an existing definition and show a technique measures it; they did not recognise "why existing definitions fail" as a plan. See [[project-working-definition-criteria]].

Open when the user left the discussion: my tentative idea (smallest part of the fine-tune that, transplanted into the original model, produces the behaviour; reuse if it is small and what it relies on was active on ordinary text; weaknesses are needing a way to divide the model and a bar for "small"), and my question whether what is learned in reuse is mainly how to get an old computation to run on new inputs, or could be strengthening one that already ran there. The user did not answer either.
