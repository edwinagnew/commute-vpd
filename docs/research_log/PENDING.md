# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-24 P1.63 (regating-lead): Fig 2's built edits on bigram and induction circuits + three more layer-0 cue draws; strengthening/rebuilt edits read above size-matched randoms at VPD, top neurons and SAE (induction at SAE only via SAE error terms); Fig 2's own site no call at top neurons; four randoms over their filed 0.03 cap. Page reports/analyses/2026-11-24_analysis_p1-63-more-circuits.md
- 2026-11-24 P2.66 (regating-lead): realistic panel run 4 (E307) at n=10 (arxiv, german never reach gain 1 by step 1200; user "n=10 is fine"): no reading orders the fine-tunes like the raters; the effective rank's earlier pass rested on arxiv and german; the kept share at VPD runs opposite to the raters, about half of that reproduced by random updates; readings agree less than baselines. Page reports/analyses/2026-11-24_analysis_p2-66-run4-n10.md, redteamed. Cell P2.66 filled.
