cell: P2.56
rival_a: (guarded retrain: E256 RL collapsed on 4 of 9 states, so RL is retrained with an entropy bonus, a zero-advantage skip and stop, and a log-odds gate on every seed) RL on a task the base model already partly does mostly re-weights pieces the model has, more than supervised training on demonstrations does, predicts RL's re-gating share above SFT's by at least 0.1 on each task that survives, seed ranges apart, at the same update size
rival_b: the task, not the route, sets how the featurized model carries a procedural update, predicts RL's and SFT's re-gating shares within 0.1 of each other with overlapping seed ranges on every task that survives
rung: strategy
smallest_gap: 0.1
spread: 0.0555
n: 3
design: unpaired
unit: seeds of one arm on one task (RL against SFT)
spread_source: experiments/rl_67m/results/stage3/repeat_rl_s0.json
spread_of: this reading: the re-gating share in log-odds of the scored token, E222's three RL repeat seeds 0.676, 0.715, 0.786 (stage3/repeat_rl_s{0,1,2}.json .share.share.lo), sample sd 0.0555; SE of a 3-vs-3 gap 0.045, 2 SE 0.091
floor: MIN_GAIN 1 log-odds per state (realistic_reads.ratio); both rivals need RL and SFT readable on at least 2 seeds per task; E224's RL raised top-1 from 0.26 to 0.56-0.66 (sequence) and 0.045 to 0.68 (bare addition), so RL is expected over the floor (probability 0.8 per task) and SFT, whose loss is on the one answer token, at probability 0.7; a task under it reads 'no call'
stop: if the guard stops 2 or more of 3 seeds on a task as collapsed, RL on that task reads "no call" and the RL arm is not retrained again; if no task gets 2 readable guarded RL seeds, the RL-vs-SFT comparison on procedural tasks ends at E256's sequence reading and the cell fills with that
sketch: experiments/procedural_rl/figures/sketch_rivals.png
rehearsal: cluster only: every job is under 20 min on a cluster L40 in the debug QoS (train_guarded ~2 min per task, at most ~14 min; read_guarded ~3.5 min, measured from E256 jobs 63152-63157), so no pod is used
