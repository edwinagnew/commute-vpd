---
name: feedback-user-reads-log-and-chat-only
description: "Assume the user has read only the research log and the chat, never a report; every request for input is self-contained with a state-of-play line (2026-10-22)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 168db59e-8f64-4108-8ca0-ece8ecff2d5a
  modified: 2026-10-22T12:29:24.270Z
---

The user does not read reports in detail ("theres too much information and the writing quality is so poor that i
dont get much from it anyway"; a previous session told them to hand over a report and they did, unread). Never assume
a report was read, even one the user handed over. Every request for the user's input: one line of state of play,
then three plain sentences (observation, technique with every symbol named, the exact question), then detail. When
a session hands the user a report to pass to the next session, the receiving session owes the user the summary, not
a pointer. Reports stay as the record for sessions; the research log and the chat are what the user reads.

Two limits the user set on the rule (2026-10-22): it is not an excuse to make reports less readable or more waffly
(the user still spot-checks reports and reads some for detail, so /writeup standards hold in full), and pointing the
user to a report for more information is fine only when the pointer comes with a clear description of what is there.

**Why:** 2026-10-22 the user asked whether they should read every report or whether the agent should stop assuming
so; the recommendation (agreed as the working rule) was the latter, with the state-of-play line to keep them informed.

**How to apply:** see [[feedback-plain-approval-requests]], [[feedback-short-chat-replies]],
[[feedback-update-research-log-each-experiment]]. Decision reports are written as memos to the user.
