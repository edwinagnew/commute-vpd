cell: P1.26
rival_a: Held survival tracks reuse: the reuse member keeps its units as gains of M's units while its twin writes rules M's units do not, which predicts reuse-minus-twin survival above zero by more than its seed spread in each of the four counted pairs in at least one reading, with gaps not ordered by the twins' update-size ratios
rival_b: Held survival tracks how far the weights moved, and per-site linear maps absorb almost any change, which predicts reading 3's survival near 1 on every item including the rebuilt one (its gaps within seed spread), and readings 1 and 2's gaps ordered by MLP update size: largest in the case pair, smallest in the two key-table pairs
rung: strategy
smallest_gap: 0.02
spread: 0.00014
n: 4
design: paired
unit: seeds 0 to 3 per pair, reuse member against its twin on each member's own read windows; spread is the pooled sd over seeds of E196's level A reuse-minus-twin gap (a different reading of the same pairs, the only measured seed spread)
spread_source: experiments/rule_retention_pairs/results/summary.json
stop: A code check misses (do-nothing, gains-only per unit or per block, circuit-off not reproduced by H within the 1e-3 logit tolerance where arithmetic says it must, or the planted gains and shuffle not recovered): the job stops before any other item and the cell waits on the fix
sketch: experiments/held_model_retention/figures/sketch.png
descriptive: user, 2026-11-17 chat: run 1 is the agreed calibration, with gaps and their spread, no bars

## What runs (run 1 of cell P1.26)

The held model of `reports/plans/2026-11-18_plan_held-model-retention.md` ("Frozen specification" and "Settled"). H is M
with only its output weights replaced, fitted per site on M's inputs on 16,384 fit-half windows: reading 1 one gain per
neuron and per head, reading 2 one gain per head and per MLP block, reading 3 the least-squares map from M's unit
activations. Survival S = 1 - KL(F || H) / KL(F || M) at the last position of 512 read windows. Stage 0, alone: the four
code checks (do-nothing, gains-only per unit with shuffled units, gains-only per block, circuit-off), each with its
arithmetic expected value; any miss stops the job. Stage 1: E195's rebuilt item, unproven, no label. Stages 2 to 6: the
five E196 pairs at seeds 0 to 3, 40 fine-tunes. Reported per item and reading: S, KL(F || M), KL(F || H), each site's
local fit residual; per pair and seed the reuse-minus-twin S per reading, and its spread over seeds. The three readings
are three targets, never combined. No bars, no verdicts.

## Cost

No smallest-set search: per item one pass of second moments over about 131,000 positions, 8 pseudo-inverses and 3
forwards of H on 512 windows. Estimate 1 to 2 min per item, about 45 items: under an hour on one H100, or two cluster
debug jobs.
