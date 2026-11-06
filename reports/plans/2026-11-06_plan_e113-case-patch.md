# E113: does patching the case readout flip the case-label fine-tune's answer?

*definition-lead, 2026-11-06. User's choice after E112 closed (ask 5, "patching test"). Fast lane, $0 cluster debug.*

## Request, in the Terms' words

E112's real case-label fine-tune passes possession (block 1 readout 0.99 against a 0.52 control), held-out carrying
(0.998) and the threshold-on-readout check (0.99). But removing the readout's direction took only 0.08-0.21 of its
gain, and full erasure was too damaging to read (E112 Amendment 2). Row 81's causal check reads "ablating **or
patching** the readout direction breaks the fine-tuned behaviour". E113 runs the patching form on E112's saved states:
set the case readout's coordinate to the *other* case's value, and see whether the fine-tuned label flips.

## Definition (rows 81, 87; unchanged)

Verdict per state, by E112's filed rule with the causal check read by patching: possession passes (E112 stage 0),
P1 and P2 meet the real bars (E112 stage 2), and the patch check below meets its bar → "reuses C"; any miss →
"not through C". Nothing else is added. Using patching rather than removal is a choice row 81 already names; the
user picked it (ask 5).

## The edit

At block l* = 1's output, on each task window: at the trigger position, along E112 stage 0's previous-token case
direction u_p (unit, from `results/large/probes.pt`), set h·u_p to the mean of h·u_p over Pile windows of the
*opposite* class (fit windows, computed once): h ← h + (μ_opp − h·u_p) u_p. At the word position, the same with
E112 Amendment 1's own-token direction u_o. Both edits together are the filed patch. No other position or block.

## Numbers (filed in `experiments/case_patch/results/predictions.json` before the run)

| # | Number | W0 | Real | Twin | Control |
|---|---|---|---|---|---|
| Q0 | Apparatus: after the patch, E112's block-3 previous-token probe at the trigger reads the patched case (accuracy against the flipped labels) | ≥ 0.80 (stop if missed: the patch does not reach later blocks) | - | - | random-direction patch of the same size: ≈ the unpatched accuracy |
| Q1 | Flip rate: share of train-word windows (where both arms perform, as E112's P3) whose label (logit A vs B) changes under the patch | - | ≥ 0.50 | ≤ 0.20 | random unit directions, same shift size: ≤ 0.10 on every state (stop if missed) |
| Q2 | W0's next-token loss change at the trigger under the patch, printed | small (< 0.1 nats expected) | - | - | - |

Verdict uses Q1 as the causal check. Filed outcome: real "reuses C", twin "not through C", both seeds.

Beside, no rule: the flip rate on held-out-word windows; the flip rate with each edit alone (trigger only, word only), and at shift 2× (μ_opp + (μ_opp − μ_own)).

Edge cases filed: a state whose unpatched label accuracy is < 0.6 (the twin on held-out words) reads Q1 against its own
unpatched labels, as written; a Q1 at exactly 0.50 passes (inclusive, as E112's bars); Q0 missed → stop, no reading.

## Not in this plan

Subject number. The route-split readings at this trigger. Any change to the definition.
