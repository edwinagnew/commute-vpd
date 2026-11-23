cell: P2.64
rival_a: Text unfamiliarity to M drives the write share over K on these fine-tunes, which predicts M's all-position loss per byte on these read rows higher on reversed order than on lowercase and higher on Yoruba than on French, the same order as E_b over K on both readable pairs
rival_b: The E273 reversal is not about how unfamiliar the text is to M, which predicts M's all-position loss per byte on these rows not higher on reversed order than lowercase, or not higher on Yoruba than French (by under 0.1 nats per byte)
rung: premise
smallest_gap: 0.1
spread: 0.49
n: 2
design: unpaired
unit: built pairs read (lowercase vs reversed order, French vs Yoruba), one deterministic read of M per side on its 500-512 saved rows
spread_source: experiments/convergence_panel/results/wide/base_loss.json
spread_of: proxy: sd of M's all-position nats per byte across E272's six text tasks (0.78-2.15), the same reading on other rows of comparable text, so the units match
floor: none (descriptive); the control (scored-token mean log P against E246's saved prior on the same rows) is reported, expected 0 to float noise
stop: reversed order not above lowercase or Yoruba not above French in M's loss: unfamiliarity is dropped as the candidate on the E273 page, and no further unfamiliarity read is proposed
sketch: experiments/realtext_eb/figures/eb_loss_sketch.png
rehearsal: cluster only: one debug L40 job, a forward pass of the 67M on 3,036 rows, no pod in any run of this card
descriptive: whip-lead 2026-11-23 04:35 (gate 6 for in-plan runs is whip-lead's, user 2026-11-20), inside P2.64 which the user opened as descriptive (via writeup-lead 2026-11-22, aim a); completes the E273 page's unfamiliarity check that redteam-lead asked for
