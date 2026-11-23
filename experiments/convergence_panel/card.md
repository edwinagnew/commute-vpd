cell: P2.66
rival_a: the readings measure reuse as intuition sorts it, so they agree with each other predicts each readable reading's task means rank the nine tasks with the raters' consensus tier (Kendall tau > 0, task-permutation p <= 0.05) and Kendall W across the readings above W across the baselines (KL, ADL, update norm, prior) at matched KL
rival_b: the readings track output size and the base prior only predicts each reading ranks tasks as the baselines do, not as the raters do: tau with the tier near 0 or below, and W across the readings no higher than W across the baselines
rung: premise
smallest_gap: 0.05
spread: 0.018
n: 9
design: paired
unit: tasks (each the mean of 3 seeds at one matched KL); paired because every reading and baseline is read on the same task states
spread_source: experiments/realistic_tasks/results/run2_T3.json
spread_of: proxy: the re-gating share's seed sd within one task (E211 Yoruba at size 0.0175: 0.497, 0.461, 0.482, sd 0.018; lowercase 0.770, 0.790, 0.764, sd 0.014, run2_T1.json), the same reader on the same kind of state as this panel's share
floor: MIN_GAIN 1 log-odds on the read rows (share, steering and ADL are nan under it: a task under it is missing for that reading, both rivals); a reading is read only when its between-task sd exceeds its seed sd (both rivals expect the share to pass, E211's spread of task means 0.2-0.8 against seed sd 0.018); random updates read within about 0.02 of 0 (E211's Gaussians), E_b's random about 0.011
stop: every reading unreadable (between-task spread not above seed spread) at both matched KL sizes, or the stage-0 controls fail twice: the readings then say nothing about realistic reuse on the 67M, and claim 2 is written as a negative
sketch: experiments/convergence_panel/figures/card_sketch.png
rehearsal: cluster only: every stage of 0-2 runs on the MATS cluster debug QoS at $0 (each job under 30 min); stage 3 (E_b, pods, gated) is rehearsed as 'E_TBD rehearse: eb' on a cluster L40 with stage 0 and one panel state at the full 200 windows, measuring seconds per state and peak memory before any pod
