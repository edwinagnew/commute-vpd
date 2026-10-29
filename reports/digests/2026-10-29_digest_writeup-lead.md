# Writeup-lead digest, 2026-10-29

*Update 10:05: you answered asks 4-7 at 10:00 (E44 rerun running; E45 Amendment 2 takes Fable's three changes); the skeleton now says so.*

The skeleton of the preliminary-results write-up is done and current to the log's v97, so this lead has finished:
what is left on the Qwen line waits on four decisions only you can make (asks 4-7, listed in
`reports/digests/2026-10-29_digest_overnight-qwen.md`).

**What it says.** On the small model, teacher training copies narrow fine-tunes into the decomposition and keeps
behaviour and faithfulness, with the red team's caveat that the margins sit inside adversary noise until E47 runs.
The re-gating line offers a measuring instrument, not a mechanism. On Qwen nothing about an update is scored yet.
File: `reports/analyses/2026-10-29_analysis_prelim-results-skeleton.md`; §6 ranks the gaps by how cheaply each closes.

**Overnight edits.** Statuses refreshed three times as results landed: E44's precision fix, rehearsed clean, waits on
your go-ahead; E45's anchor 2 is learned on both seeds but off the filed grid, and its maths drop is mostly answer
format (0.70-0.77 with a few-shot prompt, against 0.75 unchanged); anchor 1 has no qualifying prompt; the own-answers
and null fine-tunes stopped at a step-2 limit set from the gold arm.

**Spend.** $0: no pods, no Slurm jobs, no subagents this session.

**Blocked agents.** `redteam-lead` has waited at a question since 2026-10-28 21:35.

## Not run
- No fresh check of the rows edited overnight; the last independent check of the skeleton predates them.
