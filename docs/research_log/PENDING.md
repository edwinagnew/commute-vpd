# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 qwen-decomp-lead · experiments/qwen_recipe/results/A.json (job 48367) · E45, E57 cards, timeline · confirmed: bf16 autocast rounds the fp32 master weights, so training barely sees its first 5-10 steps. Own-answer state 4 reads held-out loss 0.441 on full-precision weights and 0.055 under training's forward; template step 8 answers 0.13 correct at full precision, 0.70 rounded. At step 100 the readers agree. Report and notebook follow with the rate grid.
