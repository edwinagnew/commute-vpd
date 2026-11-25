---
name: feedback-cite-placeholders-are-tasks
description: a "[cite]" (or similar) placeholder in the paper is my task: find and verify a source from the lit index, without being asked
metadata:
  type: feedback
---
Any "[cite]", "[citation]" or "[one more point?]"-style gap in the paper is a job for writeup-lead: grep for them on every cleanup pass, search lit_review/digest/index.md first, then the web, and propose a verified source. 2026-11-25 the user found main.tex:59 "[cite]" (post-training compute share) unfilled after a full cleanup pass: "strange that you didnt notice [cite] as sign to find a citation".

**Why:** the user is short of time before the 11-29 deadline and expects gaps to be filled, not left.
**How to apply:** include `grep -n '\[cite\]\|\[citation\]\|?\]' *.tex` in every cleanup brief and handoff; ask litreview-lead (if live) for anything the index lacks. Related: [[feedback-literature-methods-spec-first]].
