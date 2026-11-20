cell: P2.41
rival_a: RL on the 67M only re-weights what the model already samples, and the route does not matter, which predicts that on two behaviours already in the 67M's top 5 (plural pronoun agreement, day/month sequences; task#2191 B187, C14), trained to the E220 matched update size and scored as a per-row contrast of the target set against the competing tokens at the cue position, the RL, own-sample and demonstration arms read alike on the re-gating share (within their seed ranges) and above the taught anchor (single-digit addition, task#2191 C51) and the shuffled-reward arm
rival_b: The route matters, so outside demonstrations build the behaviour differently from the model's own samples, which predicts that on the same states the demonstration arm reads apart from RL (a gap of at least 0.1 in the share, seed ranges not overlapping) while own samples read like RL
rung: parameter
smallest_gap: 0.1
spread: 0.0174
n: 3
design: unpaired
unit: seeds per arm (fine-tunes of one behaviour by one arm, stopped at update size 0.006874 as in E220)
spread_source: experiments/realistic_tasks/results/run2_score.json (E211, rows "L1/L3 size0 contrast" cells.share.reuse: 0.3965, 0.3796, 0.3619 over seeds 0-2, sd 0.0174)
spread_of: this reading
floor: the share's MIN_GAIN (1 log-odds) on the contrast at the cue position on held-out items; an arm under it is unreadable and both rivals say nothing about it. Each RL arm must also raise held-out top-1 on its task by 2 standard errors, and the cue-removed control prompts must not rise by more than 2 standard errors (the target is not learned everywhere)
stop: the control (zero update gives KL 0 and contrast gain 0; E53's lowercase state re-reads its saved share) misses, or RL seed 0 fails the learned gate on either behaviour, or no shuffled-reward rate stays put: the cell closes with the readings as they stand
sketch: reports/plans/2026-11-19_plan_rl-67m-stage2-card_sketch.png
rehearsal: E220's arm loop (run_stage2.py, src/commute/rl_arms.py) ran at these sizes on cluster L40s; the new task loaders run first in the ctl job as `E### rehearse:` on a cluster card, measuring items per task and seconds per step, before the arms

Runs 6 and 7 of cell P2.41 (user 2026-11-20 dialog: "Yes, these three"): E220's four arms and E222's readings on new
tasks from task#2191 where reward and reading are the same next token. Plan
reports/plans/2026-11-20_plan_rl-67m-task-rechoice.md. This code covers both runs (train, then read).
