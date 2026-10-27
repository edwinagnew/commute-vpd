---
name: feedback-no-ambiguous-shorthand
description: Never use shorthand like "the copy" to the user; name the object by what it is (the decomposition after teacher training)
metadata:
  type: feedback
---

Don't call results by coined shorthand such as "the copy" (meaning the decomposition after teacher training) in chat,
cards or digests. The user read "the copy" as the fine-tuned model itself, and had to ask (2026-10-27).

**Why:** the user reads only the log and chat ([[feedback-user-reads-log-and-chat-only]]); a shorthand that is clear in
reports reads as a different object to them, and a decision taken on the wrong object is wasted.

**How to apply:** say "the decomposition after teacher training" (or "the teacher-trained decomposition") and "the
fine-tuned model" every time; in cards, define any short name on first use. Related:
[[feedback-define-terms-when-reporting]], [[feedback-user-definitions-override-files]].
