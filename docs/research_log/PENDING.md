# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e77-open-points.md · experiments card E77, claims table, sub-question status, timeline · new card from `notebooks/E77_open_points.ipynb`, figures `experiments/qwen_open_points/figures/`, row in TRIED.md: with all 196 matrices masked, small changes barely reach the logits at early blocks, reversing the lean on the own-output change; masks keep the output within about 0.5 nats of their own row, and dependence on tokens is not measured; all-masked is the extreme case; the lead's chat reading "masks carry the output" narrowed by the check.
- E69 (qwen-rl-lead, 2026-11-03): filed null prediction missed. Shuffled rewards gained 5.6 points (se 1.2) on GSM8K test under the drop rule against a filed |gain| < 1.5, and the filed early-signal ratio missed at step 100 (8.69 against 5.10, needed 3x). On capped answers the null reads +1.7 and -1.9 against RL +9.0 and +6.4; whole answers +4.9 and +3.0 against +9.4 and +7.8. Cause not established. Card when the arm comparison is read; report and notebook due.
- E69 (qwen-rl-lead, 2026-11-03): all three arms scored. RL drop-rule gain 9.0 (se 1.1) GSM8K test, 7.1 (1.2) GSM1K; own-samples SFT 2.6 (0.7) and 1.2 (0.9), KL 0.0017 against RL 0.194, filed P7 first part missed. P0, P1 learns, P2 same, P6 hold; P3, P4 at step 100, P7 miss. Total pod $33. Report and notebook due.
