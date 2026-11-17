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
   - **Does it read under an adopted definition, unchanged?** The reuse definition is under revision in cell P1.1
     (2026-11-09): until the user adopts a wording by a dated `decisions.md` row, no verdict is given under either
     wording. If the proposal adds a rule, clause, band or score to get a verdict, it changes the definition: stop and
     ask the user. A case that cannot be read gets "unreadable" or "does not apply" (E99, 2026-11-06).
   - **What result would change what we do next?** If no outcome would change anything, the proposal is not worth
     running.
   - **Which arm checks the apparatus itself, and does it run first?** Name the control that must pass for any other
     arm to be readable (W₀, identity, do-nothing) and put it first in the run order, alone, with an early-stop rule.
     Say what it can fail on, in its name too: E34's C2 read 0 by construction, so it tested nothing, and two other
     experiments' different controls were also called C2.
     If the proposal reproduces a reference procedure (a warm start from a released checkpoint, a refit, a replay),
     list every way the recipe differs from the reference's end state: optimizer state and schedule position,
     adversary or sampler state, data, coefficients. Remove each difference or name it as what the control tests.
     (E15 ran its split arm before its W₀ control; the control then failed and the arm was unreadable.)
   - **What are the numbers, and what should each read?** List every number the run will report (keep it to the few
     the predictions use, plus one positive control). Give each one's expected value at W0, for the do-nothing
     reference and for the control. Cut any number that no prediction or decision uses, and any without an expected
     value. If this list is long, the experiment is too big: split it or drop arms.
   - **What do the early guards read?** Each guard logs its reading and its projection even when it passes (E59's silent
     step-20 guard hid a 4-6 min estimate that took 15.6). Its bar comes from a batch sd, not one reading (E59 stopped
     at 5.0088 against 5.0). After a recipe change, a first-minute value tests what the change is for, not a line
     extrapolated from the old run (E61 arm 0). A fine-tune of a model new to the project also gets a rate grid, and
     someone reads generated answers against its "learned" rule (E45's rules passed at step 2-3). A rehearsal's resume
     check prints step-matched loss diffs and a verdict against a measured floor, and a score stage scores whatever
     stages exist and marks the rest unscored.
   - **What must be saved for a later audit?** Whatever is small and would otherwise need a GPU to rebuild: a per-state
     summary JSON, the full vector behind a top-k (E48's 9,881 drops, ~40 KB), a keyed task's selection pool, both units
     of a reading, labelled by window (E54), and the scored states in a small dtype, in `results/large/`. These are files
     to re-read, not numbers the run reports, so the cut above does not apply to them.
3. Restate the project goal in one or two sentences and how the proposal serves it. Write that and the answers above
   at the top of the plan report. If the proposal fills an open cell of yours (below), go ahead and tell the user in one
   line which cell it fills. Otherwise put the check to the user and wait for their confirmation before writing
   theory notes, code or predictions; a subagent then returns the check as the first part of its report and does not
   start building.

## Approved work (user, 2026-10-21; narrowed 2026-11-09)

One kind of work needs no sign-off; the check is still written, so the record shows why the work was done. Classes 1,
2 and 4 of 2026-10-21 (analysis of what exists; controls, replications and recovery; the Qwen deliverables' next
steps) were removed by `reports/plans/2026-11-09_plan_paper-roadmap.md`: with no cell, they do not run.

- **A run that fills a named open cell of yours, within its run count, or a branch that cell's "done when" names**
  (`~/commute-agents/fleet/cell show <id>`). An amendment is a run.

Anything else is proposed to the user (in a zero-run cell's page or one ask) and not run. In particular, these
always need the user's confirmation after they have seen the check: a new cell; a new sub-question or a new line of work; a
new task family, dataset or model; a change to `GOALS.md` or to the success criterion; a departure from
`decisions.md`; dropping a line of work; anything over the session budget. Filed predictions are never edited.

**Stop and tell the user**, even inside an open cell, when: a control fails; a filed prediction reverses
in a way no branch of the plan covers; data or a machine is lost; the user's words and a file's definition disagree;
the next step would exceed the budget. Report what happened and the options, and wait.

## What counts as confirmation (outside an open cell)

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
the sub-question it bears on, and a run's result goes into its cell (`fleet/cell fill`). The queue is flushed once a
day from the table (since 2026-11-09).
