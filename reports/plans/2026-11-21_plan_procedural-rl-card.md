cell: P2.56
rival_a: RL on a task the base model already partly does mostly re-weights pieces the model has, more than supervised training on demonstrations does, predicts RL's re-gating share above SFT's by at least 0.1 on each task that survives, seed ranges apart, at the same update size
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
stop: no task inside the band 0.05-0.5 at W0, or RL seed 0 fails the learned gate on every task: the line of procedural RL on the 67M ends there and the cell closes with the benchmark; if run 2 shows no proxy agreeing in rank with R_b on these states, claim 2 is not carried to realistic fine-tunes
sketch: experiments/procedural_rl/figures/sketch_rivals.png
rehearsal: cluster only: every job is under 20 min on a cluster L40 in the debug QoS (ctl ~7, train ~6 and read ~8 min per task), so no pod is used
