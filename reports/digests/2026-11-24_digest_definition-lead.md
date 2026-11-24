# definition-lead digest, 2026-11-24 morning

**P1.4, due 11-10: the trained reuse control (E297) could not be read, because the fine-tunes built to imitate the reuse pairs never got close to them.** Its filed test is reported as "no call", and its neuron and VPD numbers are written up as descriptive. redteam-lead agreed, and writeup-lead added one Limitations sentence to paper v2. I am finishing because all my cells are filled, no asks are open, and no pods or jobs of mine are running.

## Cells
Filled since the last digest: P1.56 (the reading at the top-activating-neuron featurization), P1.57 (four runs: the copy-against-lookup pair and the dropped five-task results), P1.59 (why readings fail on trained fine-tunes, $0), P1.60 (a rotation-tolerant kept share, two runs), P1.61 (the per-neuron gain fine-tune control), P2.69 (circuit overlap after Wang et al.), P2.73 (WeightWatch) and P1.4 run 2 (E297). Open: none. The lowest open rung is L1.2, where P1.4 has one of three runs left; I am not proposing it.

## What the last runs found
- P2.73 (WeightWatch): its score separates the strengthening edit from the new edits by far more than the random spread. But it cannot tell the sign of a change: a deletion also reads high. Report: reports/analyses/2026-11-24_analysis_p2-73-weightwatch.md.
- P2.69 (circuit overlap): the built controls pass. On most trained updates it gives no reading, because their circuits fall below the faithfulness bar.
- P1.60: rotating a neuron toward a needed neuron reads as rotation, and rotating it toward an unneeded one does not, so the control passes. The literal own-space share cannot see rotation at all.
- P1.4 (E297): the x2 imitators were left 0.300 away from their target, against a closeness bar of 0.192.

## Waiting on you
Nothing from definition-lead.

## Spend
This session has spent $0 on pods; no machines are on record. Tonight's earlier pods for E294 and E297 stayed inside their approved caps of $15 and $30.

## Friction noted
1086, 1095 and 1099. One run's code called git on pods, which have no git history. One brief had no pod fallback when the cluster was full. One pod-runner edited `remote-kernels.toml` while its session ran.

Five days remain until the 2026-11-29 draft.

## Not run
- P1.4 run 3, a closer imitation fine-tune: not proposed.
- Backlog 11: checking the sha of every predictions file a run loads.
- Restoring the P1.61 log caveat, "so low VPD readings are not evidence that mechanisms changed", which the clerk cut for length.
