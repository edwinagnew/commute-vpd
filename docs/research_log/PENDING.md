# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-20 · decisions.md · definition box, decisions, timeline · Definition wording A adopted (user ~19:50): kept = M mechanism, up to an output gain, can stand in for the variable and the behaviour survives (soft intervention); a memorised fact reads new where stored however small the change; R redone under A on the six built edits and nine pairs at matched size only, other R readings become future work.
- 2026-11-21 · reports/analyses/2026-11-21_analysis_e237-key-grid.md · E237 card, P2.50 status, claims table, timeline · Five key tasks at one update size beside a matched random update: the log-odds share puts the prior-matched cipher with the cipher, 0.40 below copy at a 1.6x prior gap (every kind gap over 0.17, predicted); the KL share stays inside or at the edge of its random band (P4 missed; random draws barely move the KL); every R sits below its random band and moves between tasks only by small steps. Figure experiments/key_grid/figures/columns.png; steering, S and the crosscoder not read on these tasks.
- 2026-11-20 · experiments/TRIED.md (E240) · experiments, timeline · P1.44 closed no call: E240's c = 1 reproduction failed from card-to-card float noise (R 1.2e-4 vs tol 1e-5); VPD R moves ~1e-4 between L40 and H100, within 3x of the smallest known-item gap; superseded by P1.46 (R under wording A).
