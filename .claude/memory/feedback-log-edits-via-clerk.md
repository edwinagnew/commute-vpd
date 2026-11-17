---
name: feedback-log-edits-via-clerk
description: "Research-log edits (decision cards, flushes, publishes) go to a clerk subagent, never done inline by the lead"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 7bb3f73d-0788-42bb-a253-18ba68159040
  modified: 2026-11-10T10:06:31.770Z
---

Research-log work (a decision card, a queue flush, a republish) is handed to a background clerk subagent with a full brief; the lead does not edit `docs/research_log/parts/` itself.

**Why:** user, 2026-11-10: "use a subagent. this should be standard procedure" when the lead started writing an ask's decision card by hand. The same message asked for a single place to review overnight results, because an ask sent only through fleet/ask was not understood ("i dont know what you're asking me").

**How to apply:** every ask to the user also becomes a cold-reader decision card in the log, written by a clerk. Point the user at one page per night that summarises results, which is the morning report linked from the card. See [[feedback-decisions-as-log-cards]], [[feedback-decision-cards-cold-reader]].
