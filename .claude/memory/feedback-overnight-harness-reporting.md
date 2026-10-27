---
name: feedback-overnight-harness-reporting
description: "Overnight harness feedback 2026-10-26 — more subagents/parallelism, log each experiment as it finishes, pending cards carry a run line, temporary 20k-word main-page cap"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b86dbd36-20e9-43cb-b02a-8053362095a6
  modified: 2026-10-26T17:40:49.774Z
---

User feedback on the first overnight fleet run (2026-10-26): the lead should use more subagents and work in parallel, and must write each finished experiment into the research log before moving on (E35 sat in PENDING.md while E36 ran).

**Why:** with more autonomy the user risks being flooded with information or losing track; the log is their view of the work ([[feedback-user-reads-log-and-chat-only]]).

**How to apply:** rules are in `~/commute-agents/fleet/roles/lead.md` and `docs/research_log/README.md`. The user rejected a "Tonight" status strip; instead each pending-decision card has a "Running / Due to run" line. Experiment cards: Found at most three plain bullets, rewritten per amendment, never appended. The morning digest is on trial. The main-page cap (20,000 words; changelog, glossary, mistakes and experiment cards exempt) is temporary because the log is too long now; the user expects it to grow as the work grows. Interrupts go out by PushNotification and also as a Blocking card. Whether that notification reaches the phone from the tmux session is untested.

Later the same day the user found E35/E36 with no notebook and `reports/` "a giant mess": reports now live in `reports/<type>/` (plans, analyses, reviews, decisions, predictions, digests), every experiment from E35 on gets a notebook with its report, and `~/commute-agents/fleet/iteration-start` flags stray report files and missing notebooks for the lead.
