cell: P1.59
rival_a: Trained updates rotate the writes of the variables their fine-tuned model needs most, as a random update does, so the write share is low at its core and not only on average, which predicts a median kept share of the top 1% most-needed variables at or under 0.15 in every trained class (toy, procedural, RL, realistic) at both featurizations, against at least 0.8 for x2 and x10
rival_b: Trained updates rescale a small core of most-needed variables and the write share is diluted by a near-total needed set and by the remainders, which predicts a median kept share of the top 1% at or above 0.5 in the trained classes while their write share over the needed set stays near 0.03
rung: premise
smallest_gap: 0.3
spread: 0.001
n: 2
design: unpaired
unit: items (one fine-tune, built edit or random update read on its own D); the built contrast side holds x2 and x10 per featurization, the trained classes 21 to 114 items
spread_source: experiments/claim2_needed/results/items
spread_of: proxy: E_b over K across E58 copy seeds 0-2 at VPD (0.0339, 0.0324, 0.0320; sd 0.001), the dd-weighted mean of the same per-variable kept share over the needed set, on [0, 1]
floor: none (descriptive); an item whose pool (needed, changed, nonzero write change) is empty reads "no reading", and |K| under 100 is flagged too_small; both rivals expect every trained item's pool in the thousands
stop: If x2 and x10 read a top-1% kept share under 0.7 at either featurization, or the layout checks fail (variables outside the 24 matrices with nonzero dd, x2's changed set not one down_proj block, x10's kept shares not x2's within 1e-6), reading 1 fails its control there and no claim about trained items is made from it at that featurization
sketch: experiments/why_fail/figures/sketch.png
rehearsal: cluster only: $0 CPU on saved arrays, no pod
descriptive: user 2026-11-23 ~17:20 via whip-lead, for the Discussion; a synthesis reported as a description

## What runs (cell P1.59; user via whip-lead 2026-11-23)

The question, in the user's words: the Discussion "must say why the methods failed on realistic tasks, rule out bad task
choice and poor implementation, and give future work a starting point". Every needed-set reading so far (the write
share over the needed set, its flip-weighted version, the two reliance readings) read every trained fine-tune about the
same small amount. This cell reads, from the per-variable totals already saved, where that small amount comes from.

Four readings on every saved item of E268, E275, E277, E278, E281 and E283 (`inventory.tsv`, 335 rows, 314 unique
reads), at VPD and at neurons and heads, all by `run.py` on CPU:

1. **Rotation or rescale at the core.** For the 1% of changed needed variables with the most flips, the kept share of
   each one's write change: 1 for a pure rescale of the base model's write, near 0 for a rotation. Built x2 and x10
   (pure strengthening) are the contrast.
2. **How much of the model is "needed", and the margins.** |K| over the number of changed variables and over all
   variables, beside each item's smallest and median logit margin.
3. **Where the change sits.** The remainders' share of the write change over the needed set, and each layer's share.
4. **Toy against realistic.** Readings 1-3 on the toy fine-tunes against the realistic panel (neurons), and against
   the procedural and RL states (VPD), so a failure seen on both is not the task choice.

Rules, edge cases and filed expectations: `results/predictions.json`. Layout checks run before any reading and stop it.
