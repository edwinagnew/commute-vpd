cell: P2.87
rival_a: the log-odds re-gating share reads the task's prior skill rather than how the update was built predicts an update trained only inside a fixed random subspace, once it gains at least 1 log-odds held out, reads shares near the fine-tunes' reuse values (0.75-1.10) on most of Fig known's 8 reuse tasks
rival_b: the share picks out structure that trained full fine-tunes find and a random-subspace update cannot reach predicts the random-subspace update at gain >= 1 reads shares well under the fine-tunes' on most reuse tasks, near the new-side values (0.12-0.42) or near 0
rung: premise
smallest_gap: 0.1
spread: 0.0314
n: 3
design: unpaired
unit: random-subspace updates per reuse task (3 seeds, each its own random subspace and training seed), 8 reuse tasks, 24 updates, each read once like the fine-tunes
spread_source: experiments/share_stability/results/run.json
spread_of: proxy: E161's batch sd of the log-odds share over five 64-row batches of one fine-tune (.stage1_step10.sd_r = 0.0314), the same reading in the same units; a subspace update's spread across seeds is not measured before this run
floor: MIN_GAIN, gain >= 1 log-odds held out (one-sided): both rivals expect every update that reaches it read; a task whose subspace updates never reach gain 1 inside the training budget is reported as "no comparable null" under either rival, with its largest gain
stop: none for the line; values per task go beside Fig known in App. I as the share's gain-reaching random-subspace reference on the reuse side
sketch: experiments/subspace_share/figures/sketch.png
rehearsal: cluster only: debug QoS jobs at $0, each under 30 min; a pod only if debug cannot start within 5 min, after a cluster rehearsal at full read sizes on one task
descriptive: user 2026-11-26 ~20:18 in whip-lead's chat ("if its missing data, run it yourself for anything indivudally under $100 and/or 6 hours") and ~21:05 ("yes, open all four"); readthrough fill, values per task beside the fine-tunes
