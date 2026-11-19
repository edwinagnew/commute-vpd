# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · parts/sections/decisions.html · decisions, table of work · remove the card "Question for you · writeup-lead" (proposal 35): approved as cell P2.39 and filled the same day (paper repo eaeaaf4)
- 2026-11-18 · reports/drafts/2026-11-18_draft_sec4-bullets-ask48b.md · definition box (main page) · approved wording (user ~19:00): keep the box as the sentence alone plus "Readings so far" (user 10:35 note); change only the touched clauses: a remainder "the rest of the network" covers what the features do not (coverage); a graft variable is given to the base model with a mechanism that outputs zero; K* = every variable the behavior runs through, under mean ablation; reuse R = share of K* whose mechanisms are still the base model's, compared by direction only (squared cosine, output scaling kept), graft s = 0; rho is a mean over E208's circuit of the change K, not K*; re-gating share = "a share of the log-odds gain in VPD's masked model"; "Only the definition read at VPD needs a declared split of delta W; r does not"
