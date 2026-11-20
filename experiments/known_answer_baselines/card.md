cell: P2.43
rival_a: a baseline that tracks reuse reads the built labels as R is meant to (the share: a reuse item's gain comes through the original subcomponents re-gated; neurons for half the gain: fewer neurons carry a strengthened circuit; effective rank: E141's filed direction, the reuse side higher) predicts at least one method orders all four reuse items above both new items by more than its split-half band (rule P1 of results/predictions.json), or, moved left out, the three others (P1b)
rival_b: each baseline reads something other than reuse (the share and neurons-for-half need a gain of 1 log-odds and a weight change of M's 24 matrices, which moved, rebuilt and possibly x2 do not give; effective rank follows how the edit was written, about 8 for x2, x10 and both new edits) predicts no method passes P1 or P1b: an item unreadable, or a new item at or above a reuse item
rung: strategy
smallest_gap: 0.07
spread: 0.0314
n: 2
design: unpaired
unit: known-answer items per side, each one built change read once (the new side has two: new_few, new_thin; the reuse side four: x2, x10, moved, rebuilt; n is the smaller side)
spread_source: experiments/share_stability/results/run.json
spread_of: proxy: E161's batch sd of the re-gating share over five 64-row batches of one fine-tune (key .stage1_step10.sd_r = 0.0314), the same share in the same units (log-odds share) on another row set; this run measures its own band per method from split halves of each item's rows and reads the rules against that
floor: MIN_GAIN 1 log-odds (commute.local_faith): the share and k_half are not scored below it; rival a expects the gain over 1 on every reuse and new item, rival b expects it under 1 (or the change not a weight change) on at least moved and rebuilt
stop: no method passes P1b: the paper states that none of the three cheap baselines recovers the known-answer labels, and no further baseline is tried on these items without a new cell
sketch: experiments/known_answer_baselines/figures/sketch.png
rehearsal: cluster only: one debug L40 job of forward passes (about 15 min, no pod, nothing trained), the control stage first in the same job with an exit-1 stop
