cell: P1.26
rival_a: The grafted head's route is new to M, as no head of M reads offset k and no chain of M's heads brings that token to what M's last-layer heads output at the same position, predicts reading 3's survival near 0 on the built model, with the share of the graft's own write that the attention site's map leaves unfitted near 1
rival_b: M's last-layer heads already carry the token k back at each position through a chain of earlier heads, so a per-site linear map absorbs the graft, predicts reading 3's survival near 1, with the share of the graft's write left unfitted near 0
rung: strategy
smallest_gap: 0.1
spread: 0.00014
n: 2
design: paired
unit: the two sides read: reading 3's S on the one built graft against S = 0, the arithmetic value of readings 1 and 2 when the graft is left unpaired, on the same 512 read windows; one build attempt, so no seed spread exists; spread is the pooled sd over seeds of E196's level A reuse-minus-twin gap, a different reading, as E197's card used
spread_source: experiments/rule_retention_pairs/results/summary.json
stop: The build misses a build check (held-out accuracy under 0.9, the graft's attention at offset k from the last position under 0.8, an M head's share at offset k off its stage-0 value, or the model with the graft's output zeroed off M beyond the 1e-3 logit tolerance): the job stops before the readings, run 2 of the cell is spent, and the negative case is reported unbuilt
sketch: experiments/graft_negative/figures/sketch.png
descriptive: user, 2026-11-17 dialog: build the graft negative case as run 2; reported as a description, no bars

## What runs (run 2 of cell P1.26)

The negative case of `reports/plans/2026-11-18_plan_held-model-retention.md` ("The negative case", "Settled" answer 6,
appendix recipe). M, the 67M, fully frozen. Stage 0: E197's do-nothing check on this item's windows, and on M the share
of attention each of its 24 heads puts at each offset 3 to 16; k is the offset whose largest head share is lowest, s that
share. Stage 1: one fresh head (q, k, v, o, width 128, independent seed) appended to the last layer's attention, reading
its `rms_1` output with rotary positions, trained alone to output at the last position the token k positions back; at
most 4,000 steps, one attempt. Stage 2: build checks that do not use the held model (held-out accuracy and M's on the same
windows, the graft's attention at k, M's head shares at k, the model with the graft's output zeroed against M); any miss
stops. Stage 3: E197's three readings through `commute.held_model`, unchanged. Reported: S per reading, the build checks,
each site's local fit residual, the graft's pairing in readings 1 and 2, and at the graft's site the share of the graft's
own write that reading 3's map leaves unfitted. No bars and no verdicts.

## How the held model composes the grafted site

F's last-layer attention has seven heads where M's has six. The site's unit activations are the seven heads' outputs
before the output matrix, concatenated (896 wide), and F's output matrix is M's 768 x 768 with the graft's 768 x 128 beside
it, so F's site write is M's attention write plus the graft's write. Readings 1 and 2 pair seven F heads with six M
heads; the graft is left out, and H is M exactly, only when the assignment does not give it an M head's slot (always so
when the graft writes no more than M's weakest head there; otherwise it depends on the overlap). Reading 3's map becomes
M's output matrix plus the graft's output matrix times the least-squares regression of the graft's head output on M's six
head outputs at the same position. The run reports which case held.

## Cost

Cluster debug, one L40 or larger: data and model ~1 min, stage 0 ~2 min, stage 1 (4,000 steps of 64 windows of 32
tokens through a frozen 4-layer model) ~3-6 min, stages 2-3 ~3 min. Under 15 min, $0.
