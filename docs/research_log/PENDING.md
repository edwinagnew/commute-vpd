# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-24 compare-lead: P2.75 filled (E296, diff-SAE with latent scaling as published, Dumas/Minder et al. 2025). On the 67M at site A a fine-tune-size grafted head is not picked out: in the code's scoring it reads under the bar like its matched random updates (below the method's reach, which cannot be told from selecting nothing at site A), and in the post's wording the random updates read higher than the head (does not separate). The planted real-text control stopped on a diverging loss, so there is no positive control at site A; the larger run on the built pairs was not opened. Page reports/analyses/2026-11-24_analysis_p2-75-diff-sae.md.
