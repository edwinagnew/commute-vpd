cell: P1.24
rival_a: The definition's level B (attention heads plus whole MLP blocks) sees a circuit replaced by new neurons as a changed rule, because the new neurons' outputs differ from the old block's, which predicts level B reads the rebuilt replaced item's contrast retention at 0.3 or below at every fit seed, so level B passes calibration with level A
rival_b: Level B is blind inside a unit (the definition's own stated limit): the replaced block computes what the rescaled block computes, so its rule hardly moves, which predicts level B reads the rebuilt replaced item at about the rescaled item's block retention (E193 read 0.906), 0.7 or above at every fit seed, so level B fails calibration on the replaced item while level A passes
rung: parameter
smallest_gap: 0.5
spread: 0.5
n: 4
design: paired
unit: fit seeds of the replaced item (its 32 appended neurons trained from 4 random starts), each read against the fixed low bar of 0.3
spread_source: experiments/rule_retention/results/amend1/predictions.json
stop: No lambda of 0.2, 0.3, 0.5 builds a passing replaced item on some seed, or level A fails calibration: the cell closes with E193's and this run's readings as they stand (its two runs spent; the five trained pairs are not read in this cell)
sketch: experiments/rule_retention/figures/sketch.png

## What runs (run 2 of cell P1.24, amendment 1 of E193; the user's choice "Rebuild with run 2", dialog 2026-11-17)

E193 (`card.md`, `results/predictions.json`, results committed in 1d6cd5da) stopped in stage 2 at seed 0: no lambda built a
replaced item that both fits and is not a copy. At lambda 0.1 the fit passed (fit ratio 0.013 on copy24, 0.019 on Pile)
but the new neurons copied C (max |corr| 0.998). At lambda 1 they did not copy (0.29) but the fit missed (0.44, 0.45).
Stages 0 and 1 were clean: do-nothing read r = 1 everywhere, and the rescaled item read 0.80 at level A and 0.906 at level B.

This run rebuilds the replaced item. It uses 32 appended neurons instead of 16, 12,000 fit steps per try instead of 4,000,
and the lambda grid 0.2, 0.3, 0.5, which sits between E193's copying fit and its misfit. The fit bar (0.1), the copy bar
(0.9), the rule and every other setting are E193's. Filed predictions: `results/amend1/predictions.json`, whose
"changes_against_E193" lists every change. Outputs go to `results/amend1/` and `results/large/amend1/`. Stages 0 and 1 are
rerun rather than reused, so the run is self-contained and the control again runs first. The five trained pairs are not
read in this cell.

The replaced item is built to be a non-copy by construction plus the check. Each fit seed trains with KL plus lambda times
the mean squared correlation between the new neurons' and C's activations on the batch. Lambda goes through 0.2, 0.3 and
0.5 in order, each try from the same start, and the run keeps the first fit that passes both held-out checks. Those checks
are: KL(replaced || rescaled) at most 0.1 of KL(rescaled || M) on copy24's and Pile's held-out windows, and no new neuron
correlating above 0.9 with a neuron of C. After a pass the run goes on to the next seed. It stops only if no lambda passes on
a seed. The lambda used and both check values for every lambda tried are recorded.

The 67M (LlamaSimpleMLP) has 4 blocks, d 768, 6 heads of 128 and an MLP of 3072 NewGELU neurons. Level A has 12,288 neuron
units (12,320 in the replaced item, 32 of them unpaired) and level B has 28: 24 heads and 4 MLP blocks.

## The rivals, including level A failing

The rivals differ only on level B's reading of the replaced item. Level A reads the replaced item at 0 by construction: C's
8 neurons write zero, and the 32 new ones have no pair. A level A failure would therefore point at the apparatus, as in E193's
card. The analytic level A readings repeated in E193's stages 0 and 1.

## Kind of change, gap and spread

Rung: parameter. This is the same reading with a new setting of the replaced item's build.

The gap and spread are E193's. The smallest decisive gap is 0.5: a reading like the rescaled one (0.80 or above) sits 0.5
above the low bar of 0.3. For the spread the card uses 0.5, the largest sd of a reading bounded in [0, 1] (Popoviciu's
inequality). This is a bound, not a measurement, since the seed-to-seed sd of level B's replaced reading is still
unmeasured: E193 read no replaced item. With 4 fit seeds read against a fixed bar, SE = 0.5 / sqrt(4) = 0.25, so the gap is
2.0 SE.

The sketch is E193's (`figures/sketch.png`): the planned figure and the two outcomes are unchanged.
