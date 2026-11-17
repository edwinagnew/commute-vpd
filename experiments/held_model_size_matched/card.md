cell: P1.27
rival_a: Held survival reads how far the weights moved, so the larger (mostly new) side keeps less of its change because its update is longer, which predicts that shrinking the larger side to its partner's size raises that side's reading 3 survival by about the pair's original gap, so each matched gap moves toward zero by more than its seed spread
rival_b: Held survival reads the update's direction and not its length (to first order KL(F||M) and KL(F||H) both scale as the square of the update), which predicts each larger-side state's reading 3 survival at the matched size within its pair's seed spread of its survival at the saved size, and matched gaps with the original gaps' signs
rung: parameter
smallest_gap: 0.002
spread: 0.0008
n: 2
design: paired
unit: saved states (seeds) per side of a pair, 2 to 8; each larger-side state is read at its saved size and at E187's matched size, so the shift is paired within a state; spread is the smallest reading 3 seed sd of a reuse-minus-twin gap over E197's counted pairs (identity against random table)
spread_source: experiments/held_model_retention/results/summary.json
stop: A code check misses in stage 0 (do-nothing on the scaling path not reproduced within the 1e-3 logit tolerance, the scaled gains-per-block build not reproduced by H, an unscaled item not repeating E197's S within 1e-4, or a state's size not E187's): the job stops before any pair and the cell waits on the fix
sketch: experiments/held_model_size_matched/figures/sketch.png
descriptive: user 2026-11-18, P1.27 done_when "no cut-offs or labels", reports S with spread

## What runs (the one run of cell P1.27)

The held model of the working definition (decisions row 2026-11-18; reading 3, the per-site linear map, main; reading
1, one gain per unit, beside it) on E187's nine built pairs at every saved seed, on the 67M. Each state is read at its
saved size and, for the larger side, times E187's c (0.52 to 0.997), the number that matched the two sides' mean update
size. Stage 0, alone: do-nothing on the scaling path, a scaled gains-only build, and four items that must repeat E197.
Reported per item: S for each reading, KL(F || M), update size (whole and MLP). Per pair: the reuse-minus-new gap at
saved and matched sizes with its seed spread, and each state's shift. Rules R1 to R3 are comparisons, no bars, no
labels. Premise doubt: a scaled-down update is not a model that learned less.

## Cost

Cluster debug, one L40 per job, $0. 108 items at about 30 s each: one stage-0 job (~5 min), then four parallel jobs of
11 to 18 min. Plan `reports/plans/2026-11-18_plan_held-model-size-matched.md`.
