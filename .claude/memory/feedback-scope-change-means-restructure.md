---
name: feedback-scope-change-means-restructure
description: a change to the paper's claims or scope must become a same-day writeup-lead task to restructure the whole paper, not section edits
metadata:
  type: feedback
---

The 2026-11-20 decision to narrow the paper to two claims ([[feedback-paper-condense-claims]]) was relayed only as section edits (rebuild 5.3, cuts); writeup-lead waited for the user's own Methods/Results rewrite, and the restructure started only when the user asked on 2026-11-22: "i would have thought collectively deciding to narrow the paper down to two specific claims would have lead to rewriting the paper with that in mind automatically?" (friction 1003).

**Why:** the draft is due 11-25 and final 11-29 ([[project-draft-deadline-2026-11-29]]); structure lost to waiting cannot be recovered.

**How to apply:** when claims or scope change, the same day give writeup-lead a task to bring the whole outline in line (on a branch, committed, shown to the user), and at each full check compare the paper's section outline against the current claims. Don't wait for the user to rewrite sections themselves unless they said so for that change.

Also (user 2026-11-22, after asking definition-lead themselves to put the adopted definition wording in the research log): "thats the sort of thing you also need to be on top of." After any user decision (definition wording, claim, scope), check the same day that the log's definition box/cards and the paper both reflect it, and task the owner if not.

**Missed 2026-11-22 15:31:** the user froze the definition in a separate session (`~/commute-data/scratch/definition-session/2026-11-22_frozen-definition.md`, decisions.md row "Definition of reuse frozen"). I didn't see it until the user asked "are you aware of this?". At every iteration, not just full checks, run `git log --since` on decisions.md and look for new files in the scratch definition-session folder.
