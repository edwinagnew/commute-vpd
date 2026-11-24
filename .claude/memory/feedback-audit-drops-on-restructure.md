---
name: feedback-audit-drops-on-restructure
description: "after any paper restructure or definition change, audit what silently dropped (featurizations, readings) and re-read kept items under the current definition"
metadata:
  node_type: memory
  type: feedback
  originSessionId: db10179f-7398-49f1-b4a8-db985dc150db
  modified: 2026-11-23T18:39:57.816Z
---

On 2026-11-23 the user found that the top-activating-neuron featurization (E217/E229, P2.45) had vanished from paper v2 (restructure 282d570, 11-22). No reason was recorded, and it had only ever been read under the old squared-cosine R. I hadn't noticed. The user's words: "clearly you failed to notice wed stopped running top activating neurons as a featurization even though wed talked about it quite extensively in the past."

**Why:** a restructure plus a definition change (the frozen definition, 11-22) silently orphaned results that were measured under the old reading. The user expects whip-lead to catch that, not the user.

**How to apply:**
- When the paper is restructured or the definition or reading changes, diff the old body and appendices against the new ones the same day.
- List every featurization, reading or result that disappeared or was demoted without a recorded decisions.md reason.
- For each kept item measured only under the old reading, open a re-read cell under the current definition.
- Relates to [[feedback-scope-change-means-restructure]].
