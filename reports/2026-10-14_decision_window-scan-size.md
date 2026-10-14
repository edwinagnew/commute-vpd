# Decision: scan 32 768 Pile train rows for emoticon windows, not 4 096

*2026-10-14, coding agent. A data-size deviation from `results/predictions.json` (`setup.emoticon_windows`: "rows
0..4095"), made after the windows stage and before any arm-A fine-tune, transport or faithfulness measurement.*

## What happened

Scanning Pile train rows 0..4095 for positions where `h.2.mlp.down_proj:2359` has g > 0.5 found 37 windows; the
declared fallback to g > 0.1 gave 53 (43 training, 10 held-out). The positions are the right ones (` :`, ` ;`, ` :-`,
` =` followed by `/`, `D`, `P`, `-)`), but ten held-out windows cannot resolve a capture threshold of 0.5 with a 0.3
margin over the random null, and the paper's LoRA baseline used 947 examples. The subcomponent fires about once per
40 000 tokens.

## Decision

Scan rows 0..32767 (eight times as many; about 11 minutes on an L40S). The threshold rule is unchanged: g > 0.5,
falling to 0.1 only if fewer than 200 windows result. The held-out rule (source row index divisible by 5) is
unchanged. Nothing in the fine-tune recipe, the transports, the measures or the thresholds changes.

## Why this is not a post-hoc choice

The count of windows is visible before any measurement that the predictions score. More data makes the test
stricter for the claim, not looser: with 43 training windows a full-weight fine-tune would mostly memorise, and the
capture numbers on ten held-out windows would be noise in either direction. The 4 096-row figure in the filed set-up
was a guess at the subcomponent's firing rate, and it was wrong by an order of magnitude.
