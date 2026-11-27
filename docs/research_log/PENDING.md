# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-26 · decisions.md (definition clauses renamed) · main page sections that state the definition, pending decisions, glossary · the clauses are now "necessary" (was needed) and "retained" (was kept); rho = retained share; sentence "the variables necessary for F's behavior are still computed by M's mechanisms"; experiment cards keep their old wording
- 2026-11-26 · fleet/paper/cells.tsv (cell P2.85) · next · New cell P2.85 for sae-lead, 1 run, due 2026-11-27: two cheap SAE proxies on the 67M at the base-only SAEs (error-term share; reconstruction gap as a Kissane et al. 2024 baseline), user 11-26 via writeup-lead.
- 2026-11-26 · fleet/paper/cells.tsv (cell P2.86) · next · New cell P2.86 for regating-lead, 1 run, due 2026-11-27: the two blind raters rate Fig 3s 16 known-answer tasks, blind, as a calibration of the realistic panel tiers; user 11-26 via writeup-lead.
- 2026-11-26 · reports/analyses/2026-11-26_analysis_rater-calibration.md (cell P2.86, E315) · experiments; main page where Fig 4's rater tiers are described · Rater calibration on Fig 3: the two blind raters (fresh Opus, GPT) of the realistic panel, same prompt and tier scale, rated Fig 3's 16 known-answer tasks from blind descriptions. Seven of eight tasks built new were rated new by both; ciphered text (built new) was rated middle. Lowercase and both top-guess tasks were rated reuse by both, runner-up guess and copy at a new marker reuse by one and middle by the other, the three copy-key tasks middle. The tiers' ends track construction; the middle tier mixes both kinds. Descriptive; redteam pending.
