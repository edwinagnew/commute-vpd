# Decision: approve a stronger known-answer test at coarser levels (proposal 30, 2 runs over the rung's budget)?

*2026-11-18, definition-lead. Question for the user, filed as an ask.*

**Measured so far, and whether it was expected.** The definition keeps its sentence: a fine-tune reuses the original
model M to the extent that the parts it needs compute what M's parts compute. You dropped the neuron-level score
(mean rule retention of the parts needed, "rho") from it this morning, saying "the neuron level is not a high-enough
level abstraction to detect new". The last run (E208, cell P1.30) read that score on 20 saved fine-tunes.
- Controls came out as expected: do-nothing reads "no change", a circuit turned up two-fold and ten-fold reads 1, new
  units placed in M's own slots read 0.002, appended grafts 0.
- Reuse against new did not separate: every fine-tune keeps 0.999 or more, and the reuse-minus-twin gaps follow how far
  the twin moved its weights (about 5e-4 for case, 9e-5 for repeat, 2e-5 for key tables). That was not expected.
- Heads alone did show a gap (0.0036 and 0.0039 against 0.0007 and -0.0024 for repeat), reported only.

**What the proposed run measures that nothing yet has.** A test where the right answer is known by construction, read
at more than one level. Four hand-built edits of M, each labelled when it is built:
1. reuse with a small update: an existing circuit re-triggered on a new cue;
2. reuse with a large update: the same, with its gain turned up to match the new items' update size;
3. new, spread thinly over many existing neurons;
4. new, written into a few units.

Each is read at the neuron level and at coarser declared levels (head, MLP layer, block). A part of several units keeps
M's rule by interchange: M's output for that part is put into F on the same input, and it is kept to the extent F's
behaviour holds. The rebuilt circuit is read and reported but is not an anchor. Predictions are filed first. Pass: some
declared level recovers all four labels, with the neuron level predicted to miss item 3 (thinly spread new). Then the
trained pairs are read once at the level that passed.

**The question.** The table refused to add this as a cell: rung L1.4 has a run budget of 30 and its cells already allow
33, so `fleet/cell add` queued it as proposal 30. Approve proposal 30, which puts the rung 2 runs over its budget?

**Numbers.** 2 runs, due 2026-11-20; rung L1.4's run budget is 30 and its cells already allow 33. Your own task
file of this morning ordered this cell.

**Options.**
- **(a) Approve.** The cell opens; the test runs and shows whether any declared level separates new from reuse before
  the trained pairs are read again.
- **(b) Decline.** No coarser-level test; the definition stays without a score, and the neuron-level reading stays a
  saturated reading (0.999 or more).

**Recommendation: (a).** It is your own order, and it is the one test that would show whether any declared level can
tell new from reuse.

Command for you: `~/commute-agents/fleet/cell approve 30`

**Links.** `reports/analyses/2026-11-18_analysis_rho-saved-items.md`, `experiments/rho_saved_items/results/summary.json`,
earlier card `reports/decisions/2026-11-18_decision_rho-rule-test.md` (dropped).
