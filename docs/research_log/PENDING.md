# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 · reports/reviews/2026-10-29_review_stand-in-ratio-description.md · E46 card (Found bullet 1, one-liner), overview Line 1 box (Caveats, Now), claims row "Reveal and create show up as different routes", glossary (Re-gating's list of measures; new entries "Gate gain, g" and "Stand-in ratio") · what v107-v109 left: the stand-in ratio described as nearly the gate gain over the split's subcomponent gain (the split's own changes add 0.01-0.05), never as "keep ... of what the updated subcomponents do"; "a hand-made write that uses no existing subcomponent / uses none of the pieces" becomes "built by no training" (38% of it lies in the subcomponents' span); claims row's "stage 0 asks whether it survives the nulls" replaced by stage 0 and Amendment 3's results; E12's reading stands (built on g via r), no claim changes in substance; proposed wording in the review's table
