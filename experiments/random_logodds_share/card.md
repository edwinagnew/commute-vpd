cell: P2.82
rival_a: the log-odds re-gating share mostly measures how much of any output change on the trained answer flows through Gamma's gates predicts random changes scaled to a one log-odds gain read shares near the fine-tunes' new-side values (0.12-0.42) or above on most of Fig known's 16 tasks, a task's draws agreeing in sign
rival_b: the share separates structure that trained updates have and random changes lack predicts random changes scaled to a one log-odds gain read shares near 0, well under every new side's 0.12, or erratic in sign across a task's draws
rung: premise
smallest_gap: 0.1
spread: 0.0314
n: 9
design: unpaired
unit: random draws per task (3 fine-tune seeds x 3 Gaussian directions, E228's seeds), each scaled to a gain of 1.00-1.10 log-odds and read once; 16 tasks, 48 fine-tunes, 144 draws
spread_source: experiments/share_stability/results/run.json
spread_of: proxy: E161's batch sd of the log-odds share over five 64-row batches of one fine-tune (.stage1_step10.sd_r = 0.0314), the same reading in the same units; a random draw's spread at gain 1 is not measured before this run and may be far wider (its gain is 1, the fine-tunes' 1.6-23)
floor: MIN_GAIN, gain >= 1 log-odds (one-sided): every draw is scaled into the band 1.003-1.097, so both rivals expect every matched draw read; a draw that never reaches gain 1 up to 100 times its fine-tune's size is reported with its largest gain and no share, under either rival
stop: none for the line; the result goes beside Fig known in App. I as the log-odds form's random reference (the MUST 4 sentence of 5.2), reported as values per task
sketch: experiments/random_logodds_share/figures/sketch.png
rehearsal: cluster only: debug QoS jobs at $0, each under 30 min
descriptive: user 2026-11-25 in whip-lead's chat: "run option 1"; random baseline reported as values beside each fine-tune
