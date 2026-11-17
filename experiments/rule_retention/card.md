cell: P1.24
rival_a: The definition's level B (attention heads plus whole MLP blocks) sees a circuit replaced by new neurons as a changed rule, because the new neurons' outputs differ from the old block's, which predicts level B reads the replaced item's contrast retention at 0.3 or below at every fit seed, so level B passes calibration with level A
rival_b: Level B is blind inside a unit (the definition's own stated limit): the replaced block computes what the rescaled block computes, so its rule hardly moves, which predicts level B reads the replaced item at about the rescaled item's block retention, 0.7 or above at every fit seed, so level B fails calibration on the replaced item while level A passes
rung: strategy
smallest_gap: 0.5
spread: 0.5
n: 4
design: paired
unit: fit seeds of the replaced item (its 16 appended neurons trained from 4 random starts), each read against the fixed low bar of 0.3
spread_source: experiments/rule_retention/results/predictions.json
stop: Level A fails calibration (do-nothing off r = 1, rescaled at or below 0.3, or replaced at or above 0.7): no architecture level of causal-rule retention goes to run 2 and the cell closes with that reading
sketch: experiments/rule_retention/figures/sketch.png

## What runs (run 1 of cell P1.24, user-approved 2026-11-17)

Plan: `reports/plans/2026-11-17_plan_causal-rule-retention.md`. Definition (tested, not adopted, no change to
`decisions.md`): `~/commute-data/scratch/definition-session/2026-11-17_causal-rule-retention.md`. The user chose "Same-block
only" (2026-11-17 dialog): three built items, the replacement inside one MLP layer. No trained fine-tune is read.

The 67M (`artifacts/vpd/t-9d2b8f02/model_config.yaml`, LlamaSimpleMLP): 4 blocks, d 768, pre-RMSNorm, each block
x + attn(rms_1(x)) then x + mlp(rms_2(x)); attention 6 heads of 128 with rotary positions and no bias; MLP 768 -> 3072 -> 768
with NewGELU and no bias; embedding tied to the unembedding; vocabulary 50,277; context 512. So level A has 4 x 3072 = 12,288
neuron units (plus 16 unpaired ones in the replaced item) and level B has 4 x 6 heads + 4 MLP blocks = 28 units.

- **Do-nothing** (F a copy of M): every r = 1 and the contrast reads "no change", at both levels. Runs first and alone with
  the faithfulness gate (1e-3 relative logit error, fp32), which holds by construction at both levels.
- **Rescaled**: C, the 8 neurons of one layer with the largest attribution on copy24's answer logit, output times 2. Level A
  reads each of C at r = 0.80 exactly and a contrast retention of 0.80. Level B reads C's block at the r its share of the
  block gives (computed and reported; expected above 0.9).
- **Replaced**, 4 fit seeds: C's output zeroed and 16 fresh neurons appended to the same layer, trained alone by KL to the
  rescaled item on Pile and copy24 windows. Kept only if, on held-out windows, KL(replaced || rescaled) is at most 0.1 of
  KL(rescaled || M) and no new neuron correlates with a neuron of C above 0.9 in absolute value; otherwise the job stops.
  Level A reads 0 by construction (C's neurons write zero, the new ones have no pair).

Filed rule (`results/predictions.json`, `rule.py`): a level passes when do-nothing reads r = 1 everywhere, rescaled reads
at least 0.7 and replaced at most 0.3; between them is no call.

## The rivals, including level A failing

Rival A and rival B differ only on level B's replaced reading, which is what the run decides. A third outcome is live:
**level A fails**. Its readings are analytic (0.80, 0 and 1), so a level A failure would mean the apparatus is wrong, not
that the level is: the closure search keeping an unchanged unit, the hybrid forward not reverting what it names, or the
rescaled item having no contrast on copy24 (stage 1 stops then). The do-nothing control and the rescaled positive control
run first so that such a failure shows before the replaced item is read.

## Kind of change, gap and spread

Rung: strategy, a new reading (rule retention at a declared level) calibrated on items whose machinery is known.

The smallest decisive gap is 0.5: a level that reads the replaced item like the rescaled one sits at 0.80 or above (level
A's analytic rescaled value; level B's block value is expected higher), 0.5 above the low bar of 0.3. The seed-to-seed sd
of level B's replaced reading cannot be measured before the run. The card uses the largest sd any reading bounded in [0, 1]
can have, 0.5 (Popoviciu's inequality), an upper bound on any measured value, so the check is conservative. The comparison
is one sample against a fixed bar, SE = sd / sqrt(n), the check's "paired" formula: 0.5 / sqrt(4) = 0.25, and the gap is
2.0 SE. That is why the run fits 4 seeds rather than 2: with 2 the bound gives 1.4 SE, under the bar.

## Known risk

The not-a-copy check may stop stage 2. A KL fit from random starts can pull the new neurons onto C's own input
directions (each new neuron then correlates near 1 with one of C), since copying C is the simplest way to reproduce the
rescaled item. The plan's recipe has no term against it; the run stops and reports the correlation if it happens.
