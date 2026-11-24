cell: P2.73
rival_a: S reads where an update's top output directions lie in M's own output variance, so it separates built reuse from built new wiring predicts x2's S above both new edits' S (new_few, new_thin) by more than the Haar null's spread, with the trigger flagged and dW = 0 empty: S earns a column
rival_b: S reads only how busy the written direction is, and the new edits write along busy unembedding directions predicts new_few and new_thin's S within the Haar spread of x2's or above it: S is a related-work negative
rung: strategy
smallest_gap: 0.11
spread: 0.11
n: 3
design: unpaired
unit: built states per side of the column rule's contrast (x2 against new_few and new_thin), one S each; the 50 states and their 48 shams are reported beside, read within their task sets
spread_source: experiments/weightwatch/results/predictions.json
spread_of: proxy: the sd of S under the same-spectrum Haar null for 8 equal-weight vectors at an assumed participation ratio of 20 for C (predictions.json card_spread_proxy); the same reading in the same units, but on an assumed spectrum: stage 0 measures the null's spread itself, and the column rule reads the measured one
floor: the column rule's own: x2's S must exceed the larger new edit's by more than the measured Haar spread (rival a expects a gap above it, rival b a gap at or under it); the detector's rates are read against each state's M floor and sham on the same rows
stop: stage 0's or the trigger's filed stops (the staged core or the known-range test differing, x2 not keeping exactly 8, the lambda constructions off their fp64 floor, parity differing, no trigger gain up to 2^20, under 0.95 of the T windows flagged): the detector is not reproduced on our model and nothing after it is read; else the column rule decides between a column and a related-work negative, and no further WeightWatch work is proposed
sketch: experiments/weightwatch/figures/sketch.png
rehearsal: pod (whip-lead 2026-11-23, cluster full): a short pod commute-E294-rehearse-stage0 runs moments + stage 0 at full size, its results reused by the main pod
descriptive: user 2026-11-23 via whip-lead ("open delta-crosscoder and weight-diff svd cells"), a published baseline reported as a description

What runs (spec `reports/plans/2026-11-23_plan_p2-73-weightwatch.md`, revision 3, and its appendix, with redteam-lead's two
revision-3 conditions): WeightWatch as published, Zhong and Raghunathan's `LatentStats` copied verbatim from their pinned
code (8d3874fb) with one storage-only NumPy patch, on our built edits, E62's copy and lookup fine-tunes and the realistic
panel (50 states, 48 orientation shams, 3 M passes); beside it the score after Zhong, S, against its same-spectrum Haar
null and the gradient-geometry diagnostic S_G. Stage 0 and the trigger stage hold every stop.

Interpretations of the spec, written into the code and predictions.json: the 200 trigger build windows are the first 128
tokens of 200 rows drawn from rows 12,500-13,749, and the other 1,050 rows' windows are the Pile held-out windows for the
slot choice and the detector row; the orientation sham rotates only the 8 monitored matrices' changes and keeps every other
change of F (the new edits' detector rows, a fine-tune's q, k, v and c_fc); the two matched randoms take the saved path's
tau; x2's and the trigger's F and sham passes run in stage 0 and the trigger stage (the timed passes), not again later.

Run (debug chains, $0):

    ~/commute-agents/fleet/resubmit --max 10 --comment='E___ weightwatch' cluster/weightwatch.sbatch
