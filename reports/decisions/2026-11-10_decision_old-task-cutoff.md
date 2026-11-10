# The old tasks' labels do not follow how they were built, but their ordering does. What should the gate page carry?

*2026-11-10, definition-lead, cell P1.7 (filled) feeding the gate page P1.1 (rung L1.5). Results:
`reports/analyses/2026-11-10_analysis_old-task-labels.md`, `experiments/old_task_labels/results/labels.json`.*

**What was done.** Each of the 17 old fine-tunes of the 67M model was read with the second wording's two shares: the
re-weighted share (how much of the fine-tune's change one fixed simple map of a few of the original's own directions
reproduces) and the rewired share (the same with a map from everything the original computed). The labels used the
cut-offs filed on four hand-made changes the day before: re-weighted at 0.794 or above, rewired at 0.476 or above.

**What was found.**
- 11 of the 16 labelled tasks read rewired, whether they were built to reuse or built to be new. Only the real case
  label reads re-weighted. The two long-text ciphers, lowercase and the emoticon read "no map found" (inputs were cut to
  8 tokens, so long-context tasks show little). The one-pair map has a single seed, so it gets no reading.
- The re-weighted share orders every matched pair by construction. Each reuse side reads about 0.7 to 0.83 and each new side
  0.26 to 0.57: in-context copy against lookup, the real case label against its shuffled twin, the real repeat label
  against its twin, identity against random map.
- The 0.794 cut-off sits above every reuse side except the case label, so the ordering does not become a label.
- In 10 of 33 readings the search found less rewired share than re-weighted share, which the definition rules out, so
  those rewired shares are search limits. Taking the larger of the two would change only the emoticon.

**The question.** The labels as filed say little; the ordering says a lot. Which should the gate page carry?

- **Option a: the labels as read, cut-offs unchanged.** Nothing runs. The page says the wording, with cut-offs from
  hand-made changes, calls most ordinary fine-tunes rewired whatever their history. Next: the gate page is written on
  that.
- **Option b: reset the re-weighted cut-off on ordinary fine-tunes.** This is a definition change only you can make.
  A rule for the new cut-off is filed first, then checked on fine-tunes it was not set on (for example the Table
  imitators of E139, or new matched pairs). It would tell you whether a single cut-off can label ordinary fine-tunes.
  It needs a new run cell.
- **Option c: report the share relative to a matched twin, not as an absolute label.** Nothing runs. The page says the
  wording's re-weighted share separates reuse from new within every matched pair we have, and that an absolute cut-off
  is not yet supported. Next: the gate page carries the ordering figure, with the cut-off question listed as open.

**Recommendation: option c**, with option b as a later cell if you want absolute labels. A cut-off reset on these same
16 tasks would be fitted to the answer; the within-pair ordering is what the data support today.
