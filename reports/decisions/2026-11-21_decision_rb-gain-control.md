# Counting only what changed: no version yet reads a pure strengthening as reused

*2026-11-21, definition-lead. Pending decision for the user. Only the user changes the definition's wording.*

You chose on 2026-11-21 to count reuse only over the parts the fine-tune changed, each weighted by its change. We
measured that choice, $0, on the saved built edits (cell P1.49, E247's edits). We expected the two pure-gain reuse
edits to read near 1, the do-nothing at 1, and the new and random edits low. It did not come out as expected: no
version we built reads a pure strengthening near 1.

| Edit | Wording A (R) | Untouched share | Search-based version | Order-free version |
|---|---|---|---|---|
| do-nothing | 1 | 1 | 1 | 1 |
| gain x2 (reuse) | 1.00 | 0.91 | 0.08 | 0.76 |
| gain x10 (reuse) | 0.98 | 0.91 | 0.02 | 0.76 |
| rebuilt (reuse) | 0.86 | 0.63 | 0.04 | 0.63 |
| few parts (new) | 0.89 | 0.73 | 0.37 | 0.012 |
| thin (new) | 0.98 | 0.89 | 0.54 | 0.019 |
| random, matched size | 0.29 | 0.00 | 0.15 | 0.011 |

- **Search-based version** (the earlier card's 0.54 and 0.04). Which parts count as kept comes from a search that keeps
  the fewest parts needed to get under a bar. Those are the largest writers, whether or not they are strengthened. It
  moves with the bar: random reads 0.08, 0.15 and 0.28 at bars 0.05, 0.1 and 0.2. Not usable.
- **Order-free version.** Of the changed parts' change, the share that the base model's own write explains at each
  part's best strength. No search, no bar. The order is right: reuse 0.63 to 0.76, new and random about 0.01 to 0.02.
  But the pure strengthenings read 0.76, not near 1. Doubling 8 neurons is not a strength change of VPD's parts, so each
  part's write is only about three quarters a rescale. Two further differences: it tests a part's write, not the
  model's behaviour (wording A's stand-in tests behaviour), and so far it can be computed only on the built edits.
  The trained pairs and the SAE readings need a short GPU pass per model to save the per-part changes.
- **What waits on it.** Cell P1.50 reads this version on compare-lead's new RL and SFT fine-tunes and the saved RL
  repeat updates, on pods, $80 approved by the whip-lead. It starts only once the strength control passes.

**Question.** Does the definition test each changed part by behaviour (does the model's output stay put when that part
alone is replaced by the base model's own part at its best strength), with the order-free write share as its cheap
reading, or by write (the order-free write share itself)?

The write share is computed tonight. The behavioural test starts once you approve it (last line). Your answer chooses
the reading. No recommendation: this is a premise question for you.

- **(a) By behaviour.** About 3.5 h of one GPU for the built edits (plus about 10 h for the random ones), needs your
  approval below. A per-part bar is then a definition choice. The write share becomes a method checked against it.
- **(b) By write.** The write share is the definition. Cheap, minutes per model. The pure strength edits read 0.76
  under it, and the paper states that VPD reads a neuron rescale as about three quarters a strength change.

For you: the behavioural test of each part needs a run cell you approve (proposal 52). Run
`~/commute-agents/fleet/cell approve 52`, which approves the about 3.5 h one-GPU behavioural per-part test on the
built edits, cap $40.

Evidence: `experiments/thin_edit_diagnosis/results/summary.json`; [the rescore page](../analyses/2026-11-21_analysis_option-b-rescore.md), `experiments/option_b_rescore/results/summary.json`;
`reports/analyses/2026-11-21_analysis_thin-edit-diagnosis.md`.

## Measured since (2026-11-21 night)

The behavioural test has now run on the built edits and on three trained pairs. It swaps one changed part at a time
for the base model's part at its best strength and asks how much of that part's effect on the output goes away,
weighted by change as the write share is. It reads the pure strength edits at or below a random update, so it fails
the strength control that the write share passes. Most parts barely move the output alone, and the few that do decide
the result: on the gain edit by ten one part pushes the output further than removing its change, which drives the
reading below zero.

| Edit | By behaviour, one part at a time | By write (order-free version) |
|---|---|---|
| gain x2 (reuse) | 0.011 | 0.76 |
| gain x10 (reuse) | -0.50 | 0.76 |
| rebuilt (reuse) | -0.41 | 0.63 |
| few parts (new) | 0.011 | 0.012 |
| thin (new) | 0.023 | 0.019 |
| random, matched size | 0.011 | 0.011 |
| random, matched KL | 0.008 | 0.011 |

No part passed the per-part bar on any edit.

- **Trained pairs** (copy/lookup, copy/prior cipher, lowercase/ciphers): behaviour reads 0.004 to 0.10, within 0.06 of
  the write share on every fine-tune, and neither reading puts the reuse side above the new side by more than 0.04.
- **RL against SFT on the same behaviour** (the saved repeat updates, write share only): the RL updates read 0.10 to
  0.22, the supervised ones on the same task 0.015 to 0.023, and a shuffled-reward control 0.03 to 0.05.

Page: [behaviour against write](../analyses/2026-11-21_analysis_behaviour-vs-write.md).

## Update 2026-11-22

Both ways are measured. The question is now: does a changed part count as kept by what it writes (the order-free
write share is the definition), or by what it does to the output (replace one changed part at a time)?

**Recommended: (a), by write.** It is the only version that passes the known-answer check (order right; pure
strengthenings read 0.76 because doubling neurons is only about three quarters a strength change of VPD's parts, which
the paper states). Behaviour per part has no power at this size: it cannot tell strengthening from random (E258) and
shows no separation on three trained pairs (E259).

**Caveat.** On the trained fine-tunes the write share follows how far the output moved (rank correlation with KL 0.73,
the highest of any reading), so part of it may be update size rather than reuse.

What waits on it: definition-lead writes the paper's one-page justification of the definition once the user answers.
No runs wait on it. The earlier "Nothing runs until this is decided" and "$80 approved" lines are superseded.
