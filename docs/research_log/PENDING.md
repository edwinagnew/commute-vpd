# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · reports/analyses/2026-11-02_analysis_e34-amend9-student-move.md · E34 card, teacher-training claims, size card (`decision-chain-size`), timeline · E34 Amendment 9, $0: student's cosine to the part of its fine-tune's update the subcomponents could write is 0.35 on the real task, 0.46 on the null fine-tune (labels from the unchanged model's own predictions, no task taught, predictions still moved 0.53 nats; inside the map tables' range, so the cosine cannot be read as how much was taught), 0.37-0.51 on map tables, none "mostly along"; the cosine tells which task's fine-tune a student followed, only weakly which seed's; inside share 0.62-0.67, about a random update's (four saturated matrices give over half of it); reach a third on the real task against 0.86-1.10 elsewhere, mostly the real fine-tune's length, not "two different things"; the filed "can express" holds for any student by construction, so P1 could only say "not copies, counted by weight" (filing flaw, no new verdict); open oddity h.3.attn.k_proj, no cause; Frobenius only, so the headline wording does not change until an input-weighted read.
