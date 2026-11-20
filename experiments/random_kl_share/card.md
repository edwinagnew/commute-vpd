cell: P2.44
rival_a: on the 67M the KL-form share (E57's, the form that failed on Qwen) credits re-gating only for output change the original subcomponents carry when Gamma re-gates, which a random change of the same output size does not have predicts no fine-tune of the nine built pairs and four twins has a random draw at or above its own KL-form share minus the band (rule K1 'rival a: low'), and the structured control reads low on every fine-tune
rival_b: as on Qwen (P3.8: random changes 0.992-1.009 against fine-tunes 0.990-0.9998), a change mostly outside the alive span leaves the split no better than re-gating, so the share sits near 1 for any change predicts more than half of the fine-tunes have random draws reaching their own KL-form share (K1 'rival b: reaches'), while the structured control still reads low
rung: premise
smallest_gap: 0.06
spread: 0.0314
n: 3
design: unpaired
unit: random draws (Gaussian directions, seeds) per fine-tune, each scaled to the fine-tune's KL on its read rows and read once; 88 fine-tunes
spread_source: experiments/share_stability/results/run.json
spread_of: proxy: E161's batch sd of the log-odds share over five 64-row batches of one fine-tune (key .stage1_step10.sd_r = 0.0314), a share in the same [0, 1] units; the KL-form share's own band is measured in this run from split halves of every fine-tune's rows (pooled rms) and the rules are read against that
floor: none for the KL-form share (defined for any change with K00 - K11 > 0); MIN_GAIN 1 log-odds for r, the secondary reading: rival a and b both expect most random draws under it, so P1 'silent: refused'
stop: K1 'rival b: reaches' with the structured control passing: the 67M share, like Qwen's, reads random changes as it reads fine-tunes, and part 2 reports it as not evidence of reuse; a control 'miss' stops the reading of K1 until its cause is found
sketch: experiments/random_kl_share/figures/sketch.png
rehearsal: E228 rehearse: one short RunPod H100 pod, label commute-E228-rehearse-kl, at full read sizes (the real 67M, its decomposition, the real rows): the zero check plus two fine-tunes (states 0 88: e62/copy_s0, 200 rows; states 14 88: e53/lowercase_s0, 500 rows of 127 tokens), each with its three draws and the control, measuring seconds per pass, per match and per split, peak GPU and host memory
