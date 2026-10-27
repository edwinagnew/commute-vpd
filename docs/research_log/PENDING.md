# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · docs/research_log/parts/sections/decisions.html · decisions (card decision-regating-reading) · Opening reworded: re-gating is measured with the subcomponents held at their original values to separate it from changes to them, not because they were never changed; E5 and E19 changed them and found little effect (user, fix 2).
