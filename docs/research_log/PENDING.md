# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · reports/analyses/2026-11-21_analysis_r-under-a-held-items.md · new E248 card (P1.48, figure experiments/kstar_vpd_wording_a/figures/changed_only_pairs.png); claims-table row on wording A; decision card decision-wording-a-at-vpd's "Running: the held E248 readings" line becomes the result · R under wording A at VPD, descriptive: reuse side above new side in all nine trained pairs (gaps 0.11 to 0.32; old R 4 of 9), copy 0.98 over lookup 0.90, keys copy > elicit ~ rank-1 > prior cipher > cipher; trained circuits are at most 0.0001 untouched, so the order is unchanged among changed parts only and E247's untouched-parts leniency does not reach trained fine-tunes (small-change leniency untested; new-side searches capped)
