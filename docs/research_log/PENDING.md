# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 · decisions.md · pending decisions (remove card decision-e44-pass-test), E44 card Running/Due line, timeline · user chose option 1: masks from the original model's activations (reading B) as E44 Amendment 2
- 2026-10-30 · decisions.md (last E53 row) · E53 card (Next line), next section, timeline · E53 Amendment 1 approved: breadth-swapped pair (narrow lowercase of 50 word types, ~500-word cipher) to tell kind from breadth.
