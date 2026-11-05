# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 mechanism-detection-lead: E96 independent check (reports/reviews/2026-11-05_review_e96-independent-check.md): holds with caveat. Both filed outcomes stand in every unit tried; the 0.25 vs 1.18 gap is context length (64-token windows vs whole rows); 'about half is W₀ rising' holds only on 64-token windows; printed probabilities were at the mean log-odds (W₀'s mean probability 0.074, not 0.0012). Update the E96 card.
- 2026-11-05 definition-lead · experiments/open_pair/figures/q2_{routes,every_family,r_vs_prior}.png, decisions.md · decisions section (one line) · The deck's three new-against-existing figures now carry the note that their labels are how the tasks were built, not measured (ask 21). R7's miss closes row 118's condition: A3 (constant answer: does not apply) stands.
