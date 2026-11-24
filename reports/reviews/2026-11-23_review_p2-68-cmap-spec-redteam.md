# P2.68 CMAP spec: redteam verdict (2026-11-23)

redteam-lead reviewed `reports/plans/2026-11-23_plan_p2-68-cmap-spec.md` at 725325a4, against the paper, the authors'
code at 6d6c098 and our readers. **Verdict: OK after fixes, no blocker.** Each step matches the paper and code as
quoted, and the label "whole-group head CMAP" is right. The full note, with 3 majors, 8 minors, spot-checks of
Appendix A and answers to the open questions (a)-(h), is
`~/commute-data/agents/regating-lead/note-2026-11-23-redteam-p268-cmap-verdict.md`.

The majors:
1. The gate was measured on the wrong rows for most items.
2. The spec did not name the one scalar compared with the definition.
3. Seven departures were missing from §5.

The lead decided major 2: the compared scalar is the whole-circuit CMAP log-odds share, computed as
`neuron_r.shares` computes it and compared within a format. The A+B share is also reported for every item, as a
common column.

Revision 4 of the spec applies all three majors, minors 4-11, the answers to (a)-(h), and the paper and neuron labels.
Its Appendix H lists the changes.
