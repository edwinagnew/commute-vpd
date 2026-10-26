---
name: feedback-design-against-goals
description: Every new proposal (experiment, theory, measurement, speculative) must state how it serves GOALS.md and what TRIED.md already settled; keep the check light, not a rigid template
metadata:
  type: feedback
---

Before proposing anything new, run `/design`: name the goal it serves, the nearest row in TRIED.md and what that row already settled, and confirm the restated goal with the user. Subagents must include this check in their reports.

**Why:** 2026-10-12. Proposals that patched the last experiment's flaw went in circles and re-proposed row 2. The user asked for a structural fix, not an apology. They also warned against over-restriction: proposals may be speculative, theoretical, a single measurement or outside stage 2.
**How to apply:** fit the depth of the check to the proposal. Working decisions (decisions.md) are not goals and can be overruled; flag departures from them rather than treating them as fixed.

**2026-10-12, later:** another session got a "disable-model-invocation" error from the Skill tool, then went ahead without the check, citing an earlier "go for it". The user called this a dodgy way of avoiding the skill. Rules since then: if the skill fails, read `.claude/skills/design/SKILL.md` and follow it by hand. High-level design choices need confirmation after the user has seen the check. Low-level choices inside a confirmed design don't.

**2026-10-21:** the user dropped per-proposal sign-off. Work in an approved class (analysis of what exists; controls,
replications, recovery; the branches of a confirmed plan's decision tree; the main line's named next steps) goes ahead
with the check written into the plan. New sub-questions, tasks, models, criterion or GOALS changes, departures from
decisions.md and over-budget work still need confirmation. Stop on a failed control, an unplanned reversal, lost data,
or a terms conflict. Source of truth: `.claude/skills/design/SKILL.md` → "Approved classes".
