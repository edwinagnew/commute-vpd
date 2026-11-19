cell: P2.41
rival_a: RL on the 67M only re-weights what the model already samples, and the route by which a behaviour is reached does not matter, which predicts that at matched KL the RL, own-sample and demonstration arms read alike on the re-gating share (within their seed ranges) and above the taught anchor and the shuffled-reward arm
rival_b: The route matters, so outside demonstrations build the behaviour differently from the model's own samples, which predicts that at matched KL the demonstration arm reads apart from RL (a gap of at least 0.1 in the share, seed ranges not overlapping) while own samples read like RL
rung: strategy
smallest_gap: 0.1
spread: 0.0174
n: 3
design: unpaired
unit: seeds per arm (fine-tunes of one behaviour by one arm, stopped at the KL target)
spread_source: experiments/realistic_tasks/results/run2_score.json (E211, rows "L1/L3 size0 contrast" cells.share.reuse: 0.3965, 0.3796, 0.3619 over seeds 0-2, sd 0.0174)
spread_of: this reading
floor: the share's MIN_GAIN (1 log-odds, the scorecard's) on the behaviour's read rows; an arm under it is unreadable on the share. Both rivals expect RL and the supervised arms over it at the KL target; the shuffled arm may sit under it
stop: no rate gives a stable shuffled-reward null on both behaviours, or RL does not raise the held-out reward by the KL target: the cell stops at stage 2 and reports. A third result, the shuffled arm reading like RL, says the readings see drift or update size, not the reward
sketch: reports/plans/2026-11-19_plan_rl-67m-stage2-card_sketch.png

Stage 2 (E218) of cell P2.41: the arms, read in stage 3. Plan `reports/plans/2026-11-19_plan_rl-67m.md`, section
"Stage 2"; brief `~/commute-data/scratch/top-session/2026-11-19_plan_67m-rl.md`, "What the result would say". The
rivals are read in stage 3 on the states this run saves; stage 2 itself reports KL, size and held-out reward per arm.
