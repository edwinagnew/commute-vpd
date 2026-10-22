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
   - **Which arm checks the apparatus itself, and does it run first?** Name the control that must pass for any other
     arm to be readable (W₀, identity, do-nothing) and put it first in the run order, alone, with an early-stop rule.
     If the proposal reproduces a reference procedure (a warm start from a released checkpoint, a refit, a replay),
     list every way the recipe differs from the reference's end state: optimizer state and schedule position,
     adversary or sampler state, data, coefficients. Remove each difference or name it as what the control tests.
     (E15 ran its split arm before its W₀ control; the control then failed and the arm was unreadable.)
   - **What are the numbers, and what should each read?** List every number the run will report (keep it to the few
     the predictions use, plus one positive control). Give each one's expected value at W0, for the do-nothing
     reference and for the control. Cut any number that no prediction or decision uses, and any without an expected
     value. If this list is long, the experiment is too big: split it or drop arms.
3. Restate the project goal in one or two sentences and how the proposal serves it. Write that and the answers above
   at the top of the plan report. If the proposal is in an approved class (below), go ahead and tell the user in one
   line which class it falls in. Otherwise put the check to the user and wait for their confirmation before writing
   theory notes, code or predictions; a subagent then returns the check as the first part of its report and does not
   start building.

## Approved classes (user, 2026-10-21)

Work in these classes needs no sign-off. The check is still written, so the record shows why the work was done.

1. **Analysis of what exists.** Re-reading saved states and result files, new figures, CPU tests on saved states
   (for example the cancellation tests, E10's matched-size check), and GPU passes of the existing harness on saved
   states.
2. **Controls, replications and recovery.** More seeds of a run that already happened; reruns of a lost stage;
   the same code on another card to close a cross-card question; the control a plan already names.
3. **The steps of an approved plan's decision tree.** A plan the user has confirmed may list branches ("if G1
   fails, do X; if P2 holds, run Y"). Each branch is approved with the plan, including its predictions and its cost.
4. **Either line's named next steps.** The items on the research log's "What's next" cards for the two lines run in
   parallel since 2026-10-22 (re-gating, Q5: mechanism or leak, faithfulness under Γ, what the switched-on set did;
   commute, Q1: the weak version under reading A), on the existing model, tasks and harness.

Still needs the user's confirmation after they have seen the check: a new sub-question or a new line of work; a
new task family, dataset or model; a change to `GOALS.md` or to the success criterion; a departure from
`decisions.md`; dropping a line of work; anything over the session budget. Filed predictions are never edited.

**Stop and tell the user**, even inside an approved class or plan, when: a control fails; a filed prediction reverses
in a way no branch of the plan covers; data or a machine is lost; the user's words and a file's definition disagree;
the next step would exceed the budget. Report what happened and the options, and wait.

## What counts as confirmation (outside the approved classes)

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
Then queue the research-log update as a line in `docs/research_log/PENDING.md` (`docs/research_log/README.md` →
"Batched updates"): a dropped proposal goes in Mistakes → "Dropped at design", a changed decision in the timeline and
the sub-question it bears on, and a new experiment gets its E-code and card when it is run. The queue is flushed
before the session ends.
