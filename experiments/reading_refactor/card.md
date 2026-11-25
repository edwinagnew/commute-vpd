cell: P2.79
rival_a: the moved function computes the same quantity as the old experiment code predicts every reading on the saved states equal to the old code's value within the old code's own run-to-run spread
rival_b: a hidden variant (batch, rows, unit, floor or dtype) survives the move predicts at least one reading differs from the old code by more than that spread
rung: parameter
smallest_gap: 0.001
spread: 0.0013
n: 12
design: paired
unit: saved states read by both old and new code
spread_source: reports/plans/2026-11-25_plan_reading-refactor.md
spread_of: proxy: re-gating share of e53/lowercase_s0 read by E136, E211 and E228 (0.75188, 0.75376, 0.75129), the same reading under batch and card differences
floor: none
stop: a reading whose new function cannot be brought within the old code's spread without changing its definition: it stays on the old code and goes to writeup-lead as a finding
sketch: experiments/reading_refactor/figures/card_sketch.png
rehearsal: cluster only: reproduction jobs are $0 cluster debug QoS batches on saved states, no pod
descriptive: user 2026-11-24 ~22:00 via whip-lead asked for the code refactor overnight (scope "Measurements + archive"); these jobs only check that moved code reproduces saved numbers, no claim
