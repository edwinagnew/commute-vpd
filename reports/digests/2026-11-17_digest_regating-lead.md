# regating-lead digest, 2026-11-17

**P2.33, due 11-19: filled, 2 of 2 runs.** On fine-tunes whose kind of change was built in, the re-gating share does not
separate re-weighting from rewiring. A re-weighting fine-tune, a rewiring fine-tune and one that learned nothing all
read a little above one; only the Table fine-tune, whose change is about ten times larger, reads low. Whether the share
follows the size of the change was not tested here. Provisional, no independent check.

## Cells
- P2.33 (rung L2.2): filled, 2/2 runs, page `reports/analyses/2026-11-17_analysis_share-built-kind.md`. No other cell of
  mine is open, so I am finishing.

## What ran
- E137's hand-made edits could not be read: they are forward hooks, not weight changes. The user chose (dialog) to read
  E139's fine-tunes and to fine-tune Cue again with the step-10 check fixed.
- E200 (job 60937, debug L40, $0): the identity and positive controls passed (scorecard state read 0.802, equal to its
  saved share). Always on and Cue moved their target token too little for the share to be scored (under 1 log-odds).
- E202 (job 60967, $0), on windows where the hand-made edit itself moves its token by at least 1 log-odds (the user's
  choice): Table 0.26, Always on 1.11 to 1.13, Cue 1.21 to 1.23, XOR (learned nothing) 1.10 to 1.12. P1 (Always on
  above Table) held; P2 (Always on above Cue) came out reversed.
- Premise check: doubtful. The share may track the size of a change rather than its kind; what it adds over update size
  is compare-lead's rung L2.1/L2.3 question.

## Asks waiting on you
None. Ask 26 (the 67M story) was dropped as superseded by the 2026-11-09 roadmap. A possible next cell, not filed: a
large re-weighting fine-tune matched to Table's size, which would tell size from kind.

## Money
$0 (cluster debug only). No pods.

## Friction noted
- 830: the 67M share had no named entry point for full-weight fine-tunes; the writer spent most of its time finding it.
- 834: the plan gave expected shares but not the expected gain against the share's floor; a $0 gain scan first would
  have saved run 1.

Log republished (changelog v366) with the E200 card.
