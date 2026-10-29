# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 · decisions.md (E39 LoRA-SGD run dropped) · E39 card "What next", next section commute line · the rank-8 LoRA trained by plain SGD is dropped (user): E49 shows the check has no power on SGD-trained updates on these rows.
- 2026-10-29 · decisions.md (E47 claim cards 2-4) · pending decisions (remove cards decision-e47-plain-sgd, decision-e47-lora-rank, decision-e47-remainder), claims, Line 2 box, E47/E39/E43 cards · rescope "teacher training needs Adam" to the Adam full-matrix fine-tune (fails on 8 blocks; LoRA passes; SGD unresolved, no power, E49); count the rank-8 Adam-trained LoRA's pass, scoped to rank 8 (rank sweep to backlog); reword the remainder claim: E43 reproduces the fine-tuned weights at a faithfulness cost (2.5x, 1.9x), a live candidate under the Δ criterion, not "worse than leaving the update in Δ" as a verdict.
