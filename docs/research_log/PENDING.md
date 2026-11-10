# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · reports/analyses/2026-11-10_analysis_e145-prior-lowercase.md · E133 card (baselines comparison), its figure · the E133 card gains the lowercase-against-ciphers prior cell (E145): the original model rates lowercase's answer 1.63 decades above the ciphers' (interval 1.46 to 1.80 over windows), one value per task; final.png redrawn
- 2026-11-10 · reports/analyses/2026-11-10_analysis_e146-steering-reversal.md · E133 card (baselines comparison) · E133 card: why steering reverses (E146): at strength 1, the fine-tune's mean change added as it is, the reversal is gone on both scored pairs; the reuse sides stop gaining at strength 2 while the ciphers keep gaining, and the strength search picks the larger multiples. Layer, random baseline and denominator did not remove it. E133 cells unchanged
