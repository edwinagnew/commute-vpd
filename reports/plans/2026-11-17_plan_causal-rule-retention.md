# Plan: calibrating causal-rule retention on built items (67M)

*2026-11-17, definition-lead. The cell is proposal 28 until the user approves it. The definition is
`~/commute-data/scratch/definition-session/2026-11-17_causal-rule-retention.md`. It is tested here, not adopted: it
replaces nothing in `decisions.md` and gives no verdict on any fine-tune.*

**The user's request in the Terms' words.** Describe the original 67M model M and an edited copy F at two declared
levels. Pair each unit of F with the unit at the same place in M. For each unit, score how far F's rule is from M's. Run 1
reads three edits whose machinery is known by construction, and only on them. Run 2 reads the five trained matched pairs
at the levels that pass, and it runs only if run 1 passes.

## Levels and how a unit's rule is read

- **Level A, MLP neurons.** A unit is one neuron, and its output is what it writes: activation times its output
  row. Attention, embeddings and the unembedding are part of the equations between units.
- **Level B, heads plus MLP blocks.** A unit is one attention head, read after its output projection, or one whole MLP
  block.
- **The rule of a unit in network N.** The rule is N's unit applied to **M's** input at that layer, on every evaluation
  window and position. Both networks are fed M's values, so the comparison is between rules and is not affected by
  upstream changes.
- **Distance.** d(u) = min(1, ‖f_F − f_M‖² ⁄ (‖f_F‖² + ‖f_M‖²)), with sums over windows, positions and residual
  dimensions. Retention is r(u) = 1 − d(u).
  - An unchanged unit has r = 1 exactly.
  - A unit whose output is multiplied by α has r = 1 − (α−1)²⁄(α²+1). At α = 2 that is 0.80.
  - A unit that writes zero, or a unit of F with no pair in M, has r = 0. So does a rule unrelated to the old one at
    the same size.
- **Decomposition components** are not in run 1. They do not add up exactly to the weights, and their gates make "the
  rule" a fitted object, so the level is not cheap. They go in the "Not run" list.
- **Faithfulness gate, per level, on M and on each F.** Rebuild the residual stream from the units' outputs and check
  it reproduces the network's logits. The threshold is a maximum relative logit error of 1e-3 in fp32. For both
  architecture levels this holds by construction: the description is the network's own computation. The gate can only
  bite on a fitted level, which is not run here. This is stated, not hidden.
- **Retention for the changed behaviour.** The contrast is F against M on the task windows. A set S of F's units
  preserves the contrast when F, with every unit outside S put back to M's rule, closes at least 0.9 of KL(F‖M) on
  those windows. To find the smallest such sets, run greedy backward elimination over the changed units (r < 1 −
  1e-6) in 5 random orders. The retention is the mean r over each set found, and the range across the orders is
  reported. Do-nothing has no contrast, so it reads "no change".

## The three built items

The circuit C is the k = 8 neurons of one MLP layer with the largest attribution (activation × gradient of the answer
logit) on the task windows (copy24). The layer is the one with the most attribution mass.

1. **Do-nothing.** F = M. Every r is 1; the contrast reads "no change".
2. **Rescaled.** C's output rows are multiplied by α = 2. Expected:
   - Level A: r = 0.80 on each unit of C and 1 elsewhere; contrast retention 0.80.
   - Level B: that MLP block's r is high. Its value follows from C's share of the block's output, which is computed and
     reported.
3. **Replaced** (proposed construction):
   - Zero C's output rows. Append 16 fresh neurons, randomly initialised, to the same MLP layer.
   - Train only those new neurons so the network's next-token distribution matches the rescaled item's on Pile and task
     windows.
   - Fit check (held out): KL(replaced‖rescaled) ≤ 0.1 · KL(rescaled‖M).
   - Not-a-copy check: no new neuron's activation correlates above 0.9 in absolute value with any neuron of C.
   - The result computes the same values as the rescaled item through different neurons with different input rules.
   - Expected, Level A: C's units r = 0, new units r = 0 (they have no pair); contrast retention ≈ 0.
   - Expected, Level B: **blind.** The replacement sits inside one MLP block, so that block's output matches the
     rescaled item's, and B reads it like the rescaled one. This is the definition's own stated limit ("blind inside a
     unit"). So I expect Level B to fail calibration on this item.

**Filed rule (user's words; thresholds filed before any reading).** At each level that passes the gate:
- do-nothing: r = 1 on every unit, within 1e-6;
- rescaled: contrast retention ≥ 0.7 ("high");
- replaced: contrast retention ≤ 0.3 ("low").

A level that meets all three is frozen for run 2. Between 0.3 and 0.7 is no call. Expected: Level A passes; Level B
fails on replaced.

**Order and cost.** Stage 0, do-nothing and the gate, runs first and alone. It stops the job if any r differs from 1 or
the gate misses. Then the rescaled item, then the replaced fit and its checks, then the readings. Each stage writes its
JSON as it finishes. The run is small (one fit of 16 neurons, a few thousand forwards). It goes to cluster `debug` if
that starts at once, otherwise to a RunPod pod; under $3.

**Run 2** (only if a level passes): the five trained matched pairs, seeds 0 and 1, at the frozen levels. It reports
the profile for every unit and the contrast retention, as defined above. Its card is filed after run 1.

**Not run.** Decomposition components; a replacement spread across two units of Level B; built items with other
values of α.
