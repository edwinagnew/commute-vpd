# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).

- 2026-11-20 · reports/plans/2026-11-20_plan_rl-67m-task-rechoice.md · P2.41 card/timeline · user chose new RL tasks from task#2191 (plural pronoun agreement, day/month sequences; single-digit addition as the taught anchor); E224 trains the arms; E223 re-reads the old states as a contrast
- 2026-11-20 · parts/sections/decisions.html · decisions · remove the card decision-p3-13-held-fit-floor (ask 10 answered: user chose option 1, one more run; P3.13 raised to 3 runs)
- 2026-11-20 · reports/plans/2026-11-20_plan_p3-13-run3-fit-floor.md · P3.13 card/timeline · E225 planned: fit-row sweep (own rows, plus gold rows, plus repacked rows) until the held model reproduces Qwen on the word rows (do-nothing under 1e-7), then reread copy/cipher and known/invented; RunPod, ~$8
