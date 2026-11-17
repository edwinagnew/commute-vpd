# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · decisions.md (held-model row), GOALS.md, parts/sections/overview.html (box #existing-new rewritten by definition-lead, already in parts) · overview box, glossary (mark #def-reuse-variable superseded 2026-11-18 and point to #existing-new), decisions section (remove any card on adopting a definition), changelog · User adopted the held-model wording of causal-rule retention as the working definition of reuse (2026-11-18); second wording superseded; box rewritten as a cross-model interchange with caveats (asked: user 2026-11-18 chat, add class asked to the box)
