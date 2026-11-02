# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 qwen-decomp-lead · reports/reviews/2026-11-02_review_qwen-finetune-recipes.md · E45 card, E57 card · audit of the Qwen fine-tune recipes: the template and invented-facts runs learned at their final states, no loss, data or held-out bug; own-answer runs look over-rated; no rate sweep ever ran; filed "learned" rules pass too early; leading explanation, unproven, that bf16 autocast rounds the weights so training barely sees early updates; check job E61 stage A next.
