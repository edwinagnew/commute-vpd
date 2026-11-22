# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-22 P1.52 filled (sae-lead, E263): E_b at the SAE featurization on the saved E245/E251 states. Gain edits 0.97/0.95, rebuilt 0.135 (VPD 0.63), new and random <= 0.020; filed 0.2 bar missed (gap 0.115) though the order holds; the SAE remainders carry 0.83 of rebuilt's change, latents-only rebuilt 0.79 vs random <= 0.076. Page reports/analyses/2026-11-22_analysis_eb-at-sae.md, figure experiments/sae_eb/figures/eb_sae_vs_vpd.png.
