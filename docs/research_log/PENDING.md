# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/analyses/2026-11-06_analysis_e113-case-patch.md (notebook notebooks/E113_case_patch.ipynb) · experiments page: new E113 card (figures experiments/case_patch/figures/readout.png, flips.png); timeline; E112 card "Moved on" → E113 · E113 ran (job 54194, $0): patching block 1's case readout to the other case flips block 2's case reading on 0.714 of windows (bar 0.70, random ≤ 0.134) but the fine-tuned label on only 0.002/0.014 (real) and 0.002/0.008 (twin) against 0.50; real "not through C" under rows 81 and 87, a miss of the filed "reuses C", the third causal reading of the pair to miss; twin as filed; cause untested; next step the user's (ask 6).
