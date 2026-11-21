# The thin new edit reads as reused because untouched parts count as reused: what should the paper do?

*2026-11-21, definition-lead. Pending decision for the user. Only the user changes the definition's wording.*

We measured why a new edit written thinly into one layer reads as reused under wording A. It did not come out as
expected: we expected the output gains to be the cause, and they are not.

- **E242**, R under A at the VPD level on six built edits. Expected every reuse edit above both new edits. Got the thin
  new edit at 0.98, above the rebuilt reuse edit at 0.86. [summary](../../experiments/kstar_vpd_wording_a/results/summary.json)
- **E245**, the same at the SAE level, by sae-lead. Expected the same order. Got it: thin 0.57, rebuilt 0.69.
  [summary](../../experiments/sae_r_a/results/summary_built.json)
- **E247**, this diagnosis, $0. Expected the gains to absorb the new computation or the change to sit in few parts.
  Got neither. Gains at one change nothing. Instead 0.89 of the thin edit's circuit is untouched and counts as reused
  for free. At SAE the order holds for the same reason: there the rebuilt edit has more untouched parts.
  [result page](../analyses/2026-11-21_analysis_thin-edit-diagnosis.md)

**Question.** Wording A counts every untouched part as reused. How should the paper treat that?

- **(a)** Make the SAE reading primary and call the VPD failure a limit of that featurization. Weak: SAE's order rests on
  the same count.
- **(b)** Weigh each part by the change it carries. Inverts the order: thin 0.54, rebuilt 0.04.
- **(c) Recommended.** Keep A, report both levels with the untouched share beside R, and state thin-edit leniency as
  a limit. The held E248 readings on trained pairs, copy against lookup and key tasks are running now as descriptions.
  They show whether real fine-tunes hit it.

An independent premise check doubts the other half: an edit that leaves 89% of its circuit untouched may really be
mostly reuse, so "built new" may be a construction label rather than the truth, and rewording A to call it new would
fit the definition to the label. It asks what, apart from these readings, makes the thin edit new
([check](/mnt/nw/home/e.agnew/commute-data/premise/P1.47.md)).
