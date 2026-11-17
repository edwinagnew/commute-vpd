cell: P1.28
rival_a: Each graft's computation is absent from M's units at its own site (no M head reads its offset; no M neuron computes a random derangement of the current token), so a per-site linear map of M's units cannot rebuild it, predicts reading 3's survival far under the lowest old-task fine-tune (0.984) for all six grafts, the MLP graft no higher than the last-layer head graft at k 16
rival_b: A per-site linear map absorbs whatever M's units at that site already carry, and M's last-layer neurons carry the current token, predicts the MLP-neuron graft's reading 3 survival near the old-task fine-tunes (0.98 or more) while the head grafts stay near E201's 0.134, rising with the offset's largest M-head share
rung: strategy
smallest_gap: 0.1
spread: 0.0025
n: 6
design: unpaired
unit: grafts, one build each, read on the same 512 read windows; each graft's reading 3 survival is compared with the lowest of E197's 32 old-task fine-tunes (8 items x 4 seeds) and with the last-layer k 16 rebuild; spread is the largest seed sd of reading 3 survival over E197's eight old-task items (case_twin); the rebuild against E201's 0.134 gives one build-to-build difference
spread_source: experiments/held_model_retention/results/summary.json
stop: Stage 0 misses a control (do-nothing, zero-write grafts read as H = M, attention-pattern check, M's head shares off E201's on the same windows): the job stops before any build and the cell's run is spent with no graft read
sketch: experiments/graft_sites/figures/sketch.png

## What runs (cell P1.28, its one run)

Plan `reports/plans/2026-11-18_plan_graft-sites.md`. M, the 67M, frozen. Six grafts, each trained alone with E201's
recipe (the 67M's pretraining AdamW, 4,000 steps of 64 windows, one attempt) and read with E197's held model through
`commute.held_model`: a head on the last layer at E201's offset (a rebuild, new seed); 256 MLP neurons on the last layer
trained to output a fixed random derangement of the current token over the 256 most frequent tokens; a head on layer 1
at the same offset; heads on the last layer at the offsets ranked 5, 9 and 13 by largest M-head share (12, 8 and 4 on
E201's windows). Stage 0 runs the controls alone and stops on a miss. Each graft then gets build checks (held-out
accuracy at least 0.9 against M's; its own mechanism at least 0.8; graft zeroed equals M); a graft that misses is
reported and not read. Reported: S per graft under the per-site map (reading 3) and one gain per unit (reading 1), with
reading 2 beside them, the graft units' pairing, and the share of the graft's own write that reading 3's map leaves
unfitted. No cut-offs and no labels; the filed rules compare survivals only.

## Cost

Cluster debug, one L40: stage 0 about 3 min, each graft about 2-3 min (E201's whole job took 3.4 min for one graft).
About 20 min in all, one job, $0.
