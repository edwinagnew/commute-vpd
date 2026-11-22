# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · reports/analyses/2026-11-21_analysis_p1-41-gain-only-r.md · new E236 card (P1.41, figure experiments/gain_only_r/figures/r_gap_gain_vs_full.png); claims-table row on R at VPD · R and the held model on E231's 81 strength-only states (one gain per neuron and head), each beside the full fine-tune of the same task and seed: the held model contains all 81 (reads >= 0.999998); with E226's squared-cosine grade unchanged, R at VPD reads every strength-only state below its full fine-tune (0 of 49 above, median -0.016) though each moved the output less, while heads-and-MLP-layers reads every one above (57 of 57) and the two coarser levels split 37 of 57; the stop rule needed every level, so not triggered; R at VPD reads turned weights, not kept mechanisms, for a pure strengthening; no verdict.
