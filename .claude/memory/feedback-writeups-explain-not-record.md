---
name: feedback-writeups-explain-not-record
description: "Notebooks and reports were \"unreadable, no insight into what happened\"; the /writeup skill (2026-10-19) sets the structure, sentence limits and a checker"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b1178c59-87f8-486f-a976-c2f5cf16a0ae
  modified: 2026-10-19T18:46:44.339Z
---

On 2026-10-19 the user said the result notebooks were "basically unreadable and provide no insight into what
happened" and asked for a skill. The diagnosis, from reading `notebooks/lora_plateau.ipynb` and the 2026-10-19
analysis reports: sentences with four to fourteen numbers, parentheses in every sentence, 36-word titles, a
glossary block before the first result, codes (P1, G3, e_Δ) and history pointers ("round 5's C2") instead of
words, verdicts in place of what happened, 1×4 panel figures and 10-column tables. Earlier rules ("could it do it?",
"one question per figure") had been followed as box-ticking without fixing this.

The user's own definition of a good write-up (2026-10-19, recruit test): "abundantly clear what you were trying to
test, what you observed, what you learned, how compelling the evidence is and what we might look at next"; and
"legible even once I've forgotten all the details of our approach" (feature_wars, 2026-10-09).

**Why:** the write-ups were written as a record for a reader who already knew everything. The record exists in
`results/`, `predictions.json` and `TRIED.md`; the notebook's only job is to explain.

**How to apply:** run `/writeup` (`.claude/skills/writeup/SKILL.md`) for any notebook, analysis report, result
message or notebook-writing brief. Answer first in 80 plain words, the experiment as a story, one figure per claim
with its reference on the same axes, around 25 words and two numbers per sentence as guidelines, no parentheses or codes outside
the scorecard, and `uv run python .claude/skills/writeup/check.py <file>` before calling it done. The lead writes
the answer paragraph and "Reading it" paragraphs itself. Related: [[feedback-define-terms-when-reporting]],
[[feedback-avoid-overusing-gating]], [[feedback-short-chat-replies]], [[feedback-flag-qualitative-surprises-loudly]].
