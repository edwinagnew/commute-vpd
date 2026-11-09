# P1.1 appendix: the /design check for the definition page

*definition-lead, 2026-11-09. Companion to `2026-11-09_plan_definition-page.md`. The page proposes cells R1 to R4; none is
in the table, so nothing here runs.*

- **Goal.** `GOALS.md` asks whether post-training teaches new capabilities or strengthens existing ones. This gives that
  question a label the other parts can be scored against.
- **Nearest tried.** `experiments/TRIED.md` E62 settled that the copying head carries about half the copy fine-tune's gain
  and none of the lookup's. E112, E113 and E114 read the case pair and found no causal part. E125 did the same for
  the repeat label. E116 found that no old task could be read under the readout definition. What is new: a part-based
  possession test applied to all of them, a real shuffled twin on copying, and a planted control.
- **Patch or step.** A step: the plan's part 1. R1 alone answers a question nothing has asked, whether the switch-off
  test can read a reuse we built in.
- **Decisions relied on or departed from.** Departs from `decisions.md` rows 81, 87 and 164 (the frozen definition) by
  dropping the linear readout and the readout-direction removal; row 87's held-out carrying stays. Rests on the
  2026-11-09 plan row. All of this needs the user's dated row.
- **Reads under an adopted definition?** No. Until the user adopts a wording, no verdict is given under either; P1.2's
  readings are labelled provisional for that reason.
- **What result changes what we do.** R1 under 0.9 stops the line. A twin reading reuse stops the line. Copy passing and
  the others not passing changes the paper's part 1, by the plan's gate.
- **Control first.** R1, alone, before any pair. It can fail by the planted model not learning, in which case it is
  unread, or by the share reading low, in which case the test is blind.

## Numbers for each run

| Number | At M itself | Do-nothing fine-tune | Planted control |
|---|---|---|---|
| Carrying on unseen values | near its prior (0.25 on E62's windows) | same as M, under 0.90, so never read | at least 0.90 |
| Share with P off, real | no gain to divide: not read | not read | at least 0.9 |
| Share with P off, twin | not read | not read | at most 0.2 (filed, reference from R1) |
| Median random part's share | not read | not read | under 0.15 |

**Edge cases.** A zero or near-zero gain leaves the share undefined; clause 1 of the definition screens it, since a
fine-tune that has not learned fails carrying. A negative share (the part helped remove something) reads as under the
bar. A rule with no unseen values reads "unreadable". A part of one head with a random-part draw that hits the same
chain, as (1,1) did in E62, is expected: the bar is on the median.

**Saved for audit.** Per-state JSON with the share, its row sd and the random draws; the located part and its rule's
scores; W₀'s level with the part off, which E62 did not keep.
