# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).

2026-11-16 compare-lead: P2.30 filled (E190, job 60144, $0): Goodfire-style neuron baselines on the nine 67M built pairs. Into E133's baselines card as a Goodfire block with figure experiments/neuron_jobs_k/figures/jobs.png and source reports/analyses/2026-11-16_analysis_e190-neuron-jobs.md (copy numbers from there). Retention: reuse side keeps more of its moved neurons' top Pile windows on 7 of 9 pairs; gain kept under top-K unreadable (mask breaks the 67M). Filed lean P4 reversed (no call on 2 of 9, filed >=6): say so in the card. Timeline line for 11-16; next-section rows P2.29 closed, P2.30 filled.
