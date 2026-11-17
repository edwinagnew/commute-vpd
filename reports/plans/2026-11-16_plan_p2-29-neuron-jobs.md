# P2.29, due 11-18: E189 (neuron_jobs), Goodfire-style neuron baselines on the nine 67M built pairs

Card `reports/plans/2026-11-16_plan_p2-29-card.md` (descriptive, user 2026-11-16 dialog). Filed rules
`experiments/neuron_jobs/results/predictions.json`. One job, MATS debug L40, ~5 min, $0. No reuse/new verdict; sides
named as built.

## Design check
- **Goal.** Part 2 compares re-gating with the neuron reading and simple baselines. Goodfire's two articles make the
  neuron reading's strongest case: a neuron has a job (its top-firing text), and a model run on each layer's top ~8% of
  neurons per token keeps its behaviour. This asks both of a fine-tune: do the neurons it moves keep their job, and does
  its gain survive top-K. It adds the two cleanest Goodfire-style columns to the baseline scorecard.
- **Nearest tried.** E155 (P2.18) overlap of the top-240 *set* at the read position: real fine-tunes 0.34-0.69 vs random
  0.84-0.95, no call on 5 of 7 pairs, and its positive control failed (0.962 vs 0.965), so a set overlap never showed it
  rewards "same neurons, harder". E101 read W0 jobs of E98's carrying neurons, not jobs after the fine-tune. E143's
  top-set swap masks W0, not the fine-tuned model. New here: per-neuron job on Pile (not the task rows), a two-sided
  control (same job harder vs jobs swapped), and the gain under top-K on the fine-tuned model itself.
- **Patch or step.** A step: two named baselines the user asked for, replacing a control that failed.
- **Decisions.** None departed from. No verdict (P1.1 under revision).
- **What changes what we do.** A pair whose sides separate beyond seed range on either reading enters the gate's
  "cheap readings" column; neither separating closes the Goodfire line for part 2 (card's stop).
- **Controls first, alone (stage 0, stop on a miss).** On the reference state `e46/ident32_s0`:
  (a) zero update: retention exactly 1.0 and the top-K numerator exactly 0;
  (b) retention two-sided: "same job, harder" (each block's c_fc rows and bias of the reference's top-changed neurons
  times 1.5) vs "jobs swapped" (those rows and biases cyclically permuted within the block); stop unless harder >
  swapped + 0.5;
  (c) top-K: the reference's c_proj update restricted to the 240 neurons per block most often in W0's top set on the read
  rows, kept-share under the own top-K mask vs under a random-K mask; stop unless own > random + 0.4. The same restricted
  to the 240 least often is recorded, not a stop.

## Readings (the only numbers called)
- **Retention.** Per state, per block, the 32 neurons with the largest mean |firing change| (GELU output) on the task's
  read rows at the read position. Per neuron, its top 20 of 2,000 Pile windows (val rows 2048-4048, E101's settings) by
  peak firing over positions, at W0 and at the state; retention = shared / 20; mean over the 128 neurons.
  W0 1.0 exactly; harder ~1; swapped near chance (~0.01-0.1); size-matched random update: high, ~0.8-1 (E155's random
  read 0.84-0.95 on the set overlap).
- **Gain kept under top-K.** Per state, each block keeps its 240 neurons of largest |firing| per token and position
  (7.8% of 3,072; Goodfire keeps 320 of 4,096, |activation| x router weight, zeroes the rest), applied block by block in
  the forward (sequential, as Goodfire at inference). kept = mean(M1 - M0) / mean(T1 - T0) in E143's task units (log-odds
  of the target), T unmasked, M masked, 0 = W0, 1 = state. Random-K floor: the same with 240 random neurons per block
  per token (fixed generator). W0: numerator 0. nan when mean(T1 - T0) <= 0.
- **Pair call.** E155's `rules.call` unchanged (imported): gap of side means vs the larger within-side seed range,
  measured in this job. Floor per side: E155's `rules.floor` against a Frobenius-matched random update of the side's
  first seed.

## Stages
0 controls (above), stop on a miss. 1 every state of the nine pairs (E187's `size_matched.pairs()`: E155's seven plus
E147's `e58_rank1_96`, `e58_copy_prior_cipher`), each written as it lands; projected-time stop at 9 min after four
states. 2 one random update per side, calls, figure. Saved: per state the per-neuron retention and top-changed ids, the
W0 Pile top-20 table (`results/large/`, ~1 MB).

## Not run
Goodfire's light fine-tune under the mask; percentile bands; a K sweep (Goodfire gives none); router weighting (dense
model, no router).
