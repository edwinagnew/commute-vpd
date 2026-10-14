---
name: design
description: Use before proposing or starting any new direction in commute - an experiment, task or dataset, architecture, theory note, measurement method, or speculative idea - and when a subagent is asked to design one. Checks the proposal against the project's goals and what has already been tried, before any theory is written or predictions filed.
---

# Design check

The point is to stop the project going in circles: each proposal fixing the last experiment's flaw until it ends up
back at something already tried, or drifting away from what the project is for. Fit the depth to the proposal. A
small measurement may need three sentences, a new task family a page. Proposals can be speculative, theoretical,
a single measurement, or outside the current stage. That is fine as long as the link to the goals is stated.

## Before writing anything substantive

1. Read `GOALS.md`, `experiments/TRIED.md` and `decisions.md`.
2. Write a short answer to each of these, in prose:
   - **Which goal does this serve, and how?** Name the part of `GOALS.md` it moves forward. If the link is
     indirect or speculative, say so and say what it would unlock.
   - **What is the nearest thing already tried or proposed?** Name the row in `TRIED.md`. Say what it settled, and
     what this proposal would learn that it could not. If the honest answer is "nothing new", stop here.
   - **Is this a patch on the last experiment, or a step towards the goal?** Both can be fine, but say which one.
   - **Which working decisions does it rely on or depart from?** Check `decisions.md`. Flag any departure.
   - **What result would change what we do next?** If no outcome would change anything, the proposal is not worth
     running.
3. Restate the project goal in one or two sentences and how the proposal serves it. Put that and the answers above
   to the user, then stop and wait for their confirmation before writing theory notes, code or predictions. A
   subagent cannot wait for approval, so it returns this check as the first part of its report and does not start
   building.

## What counts as confirmation

- High-level design choices need confirmation **after** the user has seen this check. These are: a new direction,
  task or dataset, architecture, theory note, or a departure from `decisions.md`. An earlier "go for it" or "sounds
  good", given before the check was shown, does not count.
- Low-level choices inside a confirmed design need no extra confirmation: hyperparameters, plotting, code structure,
  which seed to inspect first.
- If unsure which kind a choice is, treat it as high-level and ask.

## If this skill cannot be invoked

If the Skill tool refuses or fails, read this file
(`.claude/skills/design/SKILL.md`) and follow it by hand. A failure to load the skill is never a reason to skip the
check or to proceed without confirmation.

## After

When the proposal is trained, dropped or shelved, add it to `experiments/TRIED.md`: what it found, and why we
moved on. If a working decision changed, update `decisions.md`. Never edit `GOALS.md` without the user's approval.
