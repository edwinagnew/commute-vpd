# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-27 · reports/decisions/2026-11-26_decision_cluster-worker-stuck.md (ask whip-lead 1) · main, section 3 card decision-cluster-worker-stuck · Add the recommendation back: B first (you run the three Figure 7 scripts locally; seconds, and it unblocks the paper push at once), and A as well (an admin resumes the node, which every lead's CPU checks and the log builds need). C (wait) is not recommended.
- 2026-11-27 · reports/analyses/2026-11-27_analysis_e320-qwen-neuron-draws.md (P3.15, E320, redteam PASS reports/reviews/2026-11-27_review_e320-qwen-neuron-draws.md) · experiments page, new card e320 (Qwen part) + timeline 11-27 · Headline: "The Qwen neuron control's size-matched count depends on the random draw (figure: size-matched gaps, three draws per side)". Two more KL-matched random draws per side on the Qwen frozen-neuron control. Unmatched, every fine-tune still reads above all three random changes. Size-matched, which pair stands apart changes with the draw: real against shuffled maths at the last save clears the band on draws 0 and 1 but not 2; both copy-against-cipher saves clear it on draw 2 only; known against invented facts clears it on no draw. Descriptive; cost $32.
