---
name: feedback-blind-reader-per-section
description: each paper section writeup-lead updates goes to a blind subagent to see if it can infer what we mean; appendix cut to what the body needs
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3879d95e-1bea-42df-ada8-1c2dda7b644a
  modified: 2026-11-24T05:51:48.614Z
---

When writeup-lead updates a paper section, whip-lead gives it to a blind subagent (no project context, body files only) and checks whether it can infer what the paper is trying to communicate. Findings go back to writeup-lead. Use other methods if blind reading stops finding problems (user: "do this creatively").

**Why:** user, 2026-11-23 night: wants a paper they can skim and be happy with; the appendix was "30 pages of pure waffle", to be trimmed "down to what it actually needs".

**How to apply:** after each section commit, run a blind reader. Compare what it says the claims are against the user's claims (see [[feedback-paper-condense-claims]]). Keep appendix sections only if the body cites them for a check a reader needs.
