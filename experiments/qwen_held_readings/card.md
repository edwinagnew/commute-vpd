cell: P3.13
rival_a: RL's step-75 change is mostly a re-weighting of what Qwen's own MLP neurons and attention heads already write, which predicts the held-model reading S of both RL states at least 0.02 above S of a random change at the same per-matrix norms, both seeds on the same side
rival_b: At 1.7B with 16,384 fit positions for a 6,144-wide site, the least-squares map from M's units fits any change about equally well (or equally badly), which predicts S of each RL state within 0.02 of S of its matched random change, or on opposite sides at the two seeds
rung: parameter
smallest_gap: 0.02
spread: 0.0031
n: 2
design: unpaired
unit: seeds (0 and 70) of the E167 step-75 RL run, each read for S beside a Gaussian change at its own per-matrix norms; S in its own units (1 = F's output fully kept by H)
spread_source: experiments/held_model_retention/results/summary.json
spread_of: proxy: S (reading 3) on the 67M, E197; the largest seed sd of a reading-3 gap there (twin/fact, 0.0031, n 4). No Qwen spread of S exists before this run; run 1 reads the two seeds and reports their spread beside it
floor: S is reported 'no change' when KL(F||M) on the read rows is at or under 100 times the do-nothing's KL(M||H) measured in stage 0; both rivals expect every state above it (RL's KL to W0 on maths rows 0.195-0.285, own-samples SFT 0.0043-0.0049)
stop: A stage-0 control misses (the share's W0 or do-nothing; S's do-nothing H = M or the output-matrices-only change not read as S within 1e-3 of 1; a planted rank-1 or rank-10 change not read as 1 or 10): the job stops before any state and the port is fixed first. If both RL states read S within 0.02 of their random changes, S does not tell a trained change from a random one on Qwen, and run 2's library S is put to the user before it runs
sketch: experiments/qwen_held_readings/figures/sketch.png

## What runs (run 1 of cell P3.13, rung L3.2)

Plan `reports/plans/2026-11-19_plan_l3-2-qwen-readings.md` (and `_appendix.md`). On the six E167 step-75 states (RL,
the shuffled-reward run that collapsed, own-samples SFT; seeds 0 and 70), three readings, each beside its random
reference: (a) the re-gating share on the frozen-neuron decomposition p-015e0710 (MLP matrices only) against a random
curve of 3 sizes x 3 draws on the 128 maths rows; (b) the held-model reading S (reading 3: each site's output matrix
replaced by the least-squares map from M's unit activations to F's write, fitted on 64 maths rows, KL read on 64 others)
beside a Gaussian change at the state's per-matrix norms; (c) effective rank of the update beside a Gaussian floor.
Stage 0, alone and first, holds every control. Own-samples SFT's share is expected under the status rule's floor and is
reported "does not apply"; RL's share is reported unreadable if the rule marks it unscored. The collapsed shuffled-reward
run is read as a state, not as a null. The share on p-9707d6ff is not read for these states. No reuse or new label.
