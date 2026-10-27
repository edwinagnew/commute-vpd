# Plan template

Copy to `reports/plans/YYYY-MM-DD_plan_<topic>.md` and fill each heading; delete this note and the guide lines.
Each guide line says what goes there and the rule it comes from. Model: `2026-10-27_plan_regating-recruitment.md`.

# Plan: <the question, as a question> (E<NN>)

## The question, in plain words
What we want to know and why, in a paragraph a colleague can follow; the user's request restated in `GOALS.md` →
Terms' words, which is what the user confirms (`.claude/CLAUDE.md` → Terms; `/writeup` → sentence rules).

## Design check (`/design`)
Goal served; nearest `TRIED.md` row and what this learns that it could not; patch or step; `decisions.md` relied
on or departed from; which result changes what we do next; approved class, or "needs the user's confirmation"
(`.claude/skills/design/SKILL.md` → Before writing anything, Approved classes).

## States
Every model state read: how it is made or which saved file it loads (`experiments/SAVED.md`), and for a replayed
reference recipe, each way it differs from the reference's end state (`/design` → apparatus question; `.claude/CLAUDE.md`
→ The control runs first).

## Stage 0: the control (runs first, alone; early stop)
One table: check | reads | must read | else ("stop, fix"). W₀, identity or do-nothing; the job stops here if it
fails (`/design`; `.claude/CLAUDE.md` → The control runs first).

## Stage <k>: <name>
One table per stage: number | how measured | at W₀ | do-nothing | control | expected if <reading A> | if <reading B>.
Only numbers a prediction or decision uses, plus one positive control; each with an expected value
(`/design` → What are the numbers; `.claude/CLAUDE.md` → Fewer measurements).

## Predictions
P1…: fixed measurement rule, threshold, relative or absolute, and a *Lean: high / medium / low* with one reason;
what each metric is blind to and its noise floor. The agent files them in `results/predictions.json` before the run
(`.claude/CLAUDE.md` → Rules).

## Decision tree
Pass / partial / fail per prediction, each with the next step; a branch listed here is approved with the plan.
Name what is a Blocking card and PushNotification (`/design` → Approved classes 3, Stop and tell the user;
`roles/lead.md` → Interrupts).

## Cost and routing
Time per stage, card, cluster `debug` or RunPod, dollars against the session budget (`.claude/CLAUDE.md` → Rules:
compute, budget).

## Order of work for the fleet agent
Numbered: files to read; `run.py` to write with shared steps in `src/commute/` (`src/commute/README.md`); file
predictions; fresh-Opus review of the measurement code; `/pod-runner`, stage 0 alone first; report and notebook by
`/writeup`, independent check, `TRIED.md` row, log card and PENDING flush; handoff, commit, push
(`roles/lead.md` → Each iteration; `.claude/CLAUDE.md` → One review before a paid run).

## Not in this plan
What was considered and left out, so the agent does not add it (`/writeup` → "Not run"; `roles/lead.md` → only
approved work).
