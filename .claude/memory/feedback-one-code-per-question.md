---
name: feedback-one-code-per-question
description: "One experiment code and one log card per question; follow-up runs fold into it, not new E-codes"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3b27ae7a-7763-4c63-8417-533b7f29e69b
  modified: 2026-11-10T18:03:56.495Z
---

The user (2026-11-10) noticed a drift toward new experiment codes for every follow-up run, which makes results hard to take in. The baselines comparison had been spread over E133, E135, E136, E140, E141, E143 and a P2.11 card, and the user could not find the result in the log. All of it was folded into one card, E133.

**Why:** the user reads the log, not reports ([[feedback-user-reads-log-and-chat-only]]). Many small cards scatter one answer.

**How to apply:**
- Follow-up jobs on the same question (filling holes, another baseline, a redraw) go under the existing code as further runs, listed in the card's Runs field.
- Open a new code only for a new question.
- A result must land in a numbered experiment card, never in a card id that sorts to the bottom of the page.
- Keep figures free of long footer text; colour cells by the call rather than by raw magnitude.
