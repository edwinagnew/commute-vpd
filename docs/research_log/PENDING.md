# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · docs/research_log/parts/sections/techniques.html, parts/style.css · techniques ("The techniques compared": a small subheading above each of its three tables, new h4.tblhead style; already in parts/, just rebuild and publish) · none
