cell: P2.41
rival_a: RL on the 67M only re-weights what the model already samples, and the route by which a behaviour is reached does not matter, which predicts that on the E220 size-matched states, scored as the two-form contrast the reward separates, the RL, own-sample and demonstration arms read alike on the re-gating share (within their seed ranges) and above the taught anchor and the shuffled-reward arm
rival_b: The route matters, so outside demonstrations build the behaviour differently from the model's own samples, which predicts that on the E220 size-matched states, scored as the two-form contrast, the demonstration arm reads apart from RL (a gap of at least 0.1 in the share, seed ranges not overlapping) while own samples read like RL
rung: parameter
smallest_gap: 0.1
spread: 0.0174
n: 3
design: unpaired
unit: seeds per arm (fine-tunes of one behaviour by one arm, stopped at update size 0.006874 in E220)
spread_source: experiments/realistic_tasks/results/run2_score.json (E211, rows "L1/L3 size0 contrast" cells.share.reuse: 0.3965, 0.3796, 0.3619 over seeds 0-2, sd 0.0174; a two-form contrast share, as here)
spread_of: this reading
floor: the share's MIN_GAIN (1 log-odds, the scorecard's) on the contrast score: lowercase = log p(lowercase form) - log p(capitalised form) of the text's next word; repeat = log p(token a copy writes) - log p(text's token) on non-copying rows. An arm under it is unreadable on the share, and both rivals then say nothing about it
stop: the control (a zero update reads gain 0 and share undefined; E53's lowercase state re-read on its own case-pair contrast to its saved share) misses, or fewer than two arms per behaviour clear the floor: the cell closes with the readings as they stand. The shuffled arm reading like RL says the readings see drift or update size, not the reward
sketch: reports/plans/2026-11-19_plan_rl-67m-stage2-card_sketch.png
rehearsal: the readers are E222's (experiments/rl_67m/run_stage3.py), run at full read sizes on cluster L40s in E222 (~18 s per state); the contrast driver runs its control item first and alone as `E### rehearse:` on a cluster card before the readings

Run 5 of cell P2.41 (user 2026-11-20, ask 29 option A): E222's re-gating share and scorecard baselines re-read on
E220's 27 saved states with the per-row two-form contrast in place of next-token log-odds. No retraining. Decision
card reports/decisions/2026-11-19_decision_p2-41-read-unit.md.
