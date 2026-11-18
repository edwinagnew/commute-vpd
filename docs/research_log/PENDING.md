# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 definition-lead: remove pending-decision card decision-known-answer-levels (ask 47 answered: user approved proposal 30, cell P1.31 added, L1.4, 2 runs, due 11-20); changelog line.
- 2026-11-18 definition-lead: overview box existing-new: remove the "Measuring it" section (r_u, d, tau test, declared-settings list as "the standard reading"); replace with a short "Readings so far": rho and the held-model S were tested readings, numbers stand (E195/E196 rho 0.996+, E208 0.999+ saturating at neuron level; S separated only items new by construction, E197/E201/E205/E206); the neuron level is too low to see new; P1.31 (known-answer test, interchange at neuron, head, MLP-layer, block levels) under way. Definition = the sentence alone (user, chat 2026-11-18).
