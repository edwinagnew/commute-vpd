cell: P2.44
rival_a: the 67M share is a share of a rise in the trained answer, which a random change of the same output size does not produce through the original subcomponents predicts on every fine-tune no scored random draw at or above the fine-tune's own share minus the band, the draws either reading low or not scored (gain under 1 log-odds)
rival_b: as on Qwen (P3.8: random changes 0.992-1.009 against fine-tunes 0.990-0.9998), re-gating of the original subcomponents carries any output change predicts on at least one fine-tune a scored random draw at or above the fine-tune's own share minus the band
rung: premise
smallest_gap: 0.06
spread: 0.0314
n: 5
design: unpaired
unit: random draws (Gaussian directions, seeds) per fine-tune, each scaled to the fine-tune's KL on its read rows and read once
spread_source: experiments/share_stability/results/run.json
spread_of: proxy: E161's batch sd of the re-gating share over five 64-row batches of one fine-tune (key .stage1_step10.sd_r = 0.0314), the same share in the same units; this run measures the band from split halves of the fine-tunes' rows in its control stage and the draws' own spread
floor: MIN_GAIN 1 log-odds (commute.local_faith): a draw whose gain is under it is not scored; rival a expects most draws under it at the every-position match (E211's norm-matched Gaussians gained at most 0.075, E46's KL-matched ones at most 0.49), rival b expects draws over it that read near the fine-tune
stop: a random draw reaches a fine-tune's share on the 67M (rival b): the share's reading of any update is not evidence of reuse on the 67M either, and the part 2 comparison reports it beside each share; if every fine-tune's draws are unscored, the control is silent on the 67M and is not rerun at other sizes without a new cell
sketch: experiments/random_kl_share/figures/sketch.png
rehearsal: cluster only: debug L40 jobs of forward passes (a control job of about 5 min, then one job per fine-tune of about 3-8 min, no pod, nothing trained)
