---
name: feedback-log-high-level
description: Research-log sections above the cards are for skimming: what was learned or decided, in plain words; run details only in cards/reports (user, repeated 2026-10-22 and 2026-10-26); rewrite summary boxes, never append
metadata:
  type: feedback
---

Every log section other than the experiment cards is written high level: what was learned or decided, in plain
words. No arm names, gate results, step counts or recipe details there; those go in the cards and reports. The
timeline is one headline plus two to four short sentences per day, so the user can skim it to answer "what have you
been up to" a week later.

**Why:** 2026-10-22 the user, on the timeline: "I dont see how sentences like 'Rule added: the control runs first.
E16 ... is filed and run on an H100 ...' will remind me of anything in a weeks time. So please keep it more high level
(i feel like im having to say this for almost every section...)". Same day: overview box "way too long", E28 chip too
long.

2026-10-26, again: the Teacher training technique card, the sub-question boxes and the Overview's Result cards had
grown unreadable ("this keeps happening - its very frustrating i have to continually remind you"). Cause: each
experiment appended a sentence to Q statuses and Result cards, and headlines were never revisited, so they went stale
(Result 1 still said "about half the tasks" after E21; Result 2 ignored E34).

**How to apply:** default to the short version on every log edit, and when a subagent writes log text, put this rule
in its brief. After any experiment that touches a Q status, an Overview Result card or a line box, re-read the whole box and rewrite it (answer first, 3-5 plain sentences, headline included); before ending a log flush, skim sections 1-2 for stale or overlong boxes unprompted. Recipe lines in `docs/research_log/README.md` ("High level by default", and "What to update" item 3). Related:
[[feedback-research-log-style]], [[feedback-writeups-explain-not-record]], [[feedback-short-chat-replies]].
