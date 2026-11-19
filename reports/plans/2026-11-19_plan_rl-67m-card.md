cell: P2.41
rival_a: The 67M base model already samples several programmed behaviours some of the time, which predicts at least two behaviours of different families whose reward rate on held-out prompts lies in 0.05 to 0.5, so RL has signal and room to rise
rival_b: A 16-token sample at temperature 1 from a 67M base model shows a programmed behaviour either almost always or almost never, which predicts rates bunched under 0.05 or over 0.5, with fewer than two usable behaviours of different families, and the cell stops at stage 1
rung: premise
smallest_gap: 0.1
spread: 0.38
n: 256
design: unpaired
unit: held-out prompts per behaviour (each read as the mean reward of its 8 samples)
spread_source: experiments/induction_cue/results/run.json (E62 stage_a.top1_base 0.8247: one 0/1 copy hit per window at W0, sd sqrt(p(1-p)) = 0.38)
spread_of: proxy: the per-window sd of a 0/1 hit at W0 (E62), the same unit as one sampled reward, and an upper bound on the sd of a prompt's mean over 8 samples
floor: none
stop: fewer than two usable behaviours of different families: the cell stops at stage 1 and reports, before any training
sketch: reports/plans/2026-11-19_plan_rl-67m-card_sketch.png
descriptive: user, 2026-11-19, task file: benchmark what the 67M can do before the RL arms

Stage 1 of cell P2.41: a benchmark of what the 67M base model samples, reported as a description. Plan
`reports/plans/2026-11-19_plan_rl-67m.md`; brief `~/commute-data/scratch/top-session/2026-11-19_plan_67m-rl.md`.
smallest_gap 0.1 is the distance between two candidates' rates that would change which one the rule picks near 0.2.
