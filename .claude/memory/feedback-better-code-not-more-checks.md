---
name: feedback-better-code-not-more-checks
description: "User wants simpler, clearer code that makes bugs visible, not more layers of tests and sanity checks"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 59bf13c3-728c-47ad-bc0a-a7081c36e606
  modified: 2026-10-19T09:13:19.845Z
---

On 2026-10-19 the user said the repo's code is so sloppy that you can't tell whether a strange result is a bug. They worry the agent copies the sloppy patterns as it extends the code. Their words: "i dont want to add loads more layers of unnecessary tests and sanity checks. I kind of just want you to write better code."

**Why:** The smoke-only KeyError and the unexplained ~80-nat PGD numbers in the task ladder (2026-10-19) came from code that is large (run.py files of 500–870 lines), full of `2 if args.smoke else X` branches, and copied from one experiment to the next.

**How to apply:**
- Fix bugs by simplifying the structure that allowed them, not by adding another test or another rule.
- Keep each experiment small: few variants, with one clear expected value for each number.
- Don't copy the existing patterns forward just because they're already there.

Related: [[feedback-verify-before-announcing-a-cause]], [[feedback-resolve-dont-complicate]].
