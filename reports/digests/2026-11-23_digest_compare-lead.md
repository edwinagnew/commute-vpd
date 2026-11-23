# compare-lead digest, 2026-11-23

**Cells:** P2.62 (due 11-24) is filled: E275, 1 of 4 runs used. P2.63 (the reuse dial) stays on hold on whip-lead's word (the user doubted it; on the morning list); P2.66 is regating-lead's. No asks are open.

**What ran:** E275 scanned the needed set K, the variables the fine-tune needs, for 114 saved states on RunPod and Vast pods ($9.56, including a first attempt the provider deleted). Then claim 2's comparisons were recomputed on the CPU at $0, with the write share taken over K instead of over the mean-ablation circuit.
- Built pairs: every cheap reading agrees with the write share exactly as before; no pair gap changes sign.
- Prior-matched pair: copy reads above cipher on all 20 seeds (p = 1e-5); the cheap readings' rank correlations with the write share are 0.66 to 0.83.
- Procedural states: steering, with KL held fixed, rose from 0.07 to 0.46 over all 39 states. redteam-lead's verdict was "holds with caveats", and it asked for one check: drop the 16 states whose K holds under 1,000 variables. On the 23 that remain, steering reads -0.48 over both sets and effective rank -0.81 over both, so the move came from under-sampled states. One reading does change there: the share on the untrained decomposition, given KL, from 0.30 to 0.65.
- I first told redteam-lead the procedural moves were all within sampling. The writer showed steering's move was beyond its own floor, and I sent corrections the same hour.
- Premise check: doubtful. The write share over K has not been checked against a case with a known answer, so "which numbers change" cannot be read as "which method is better". This is listed for the user's gate.
- Page: reports/analyses/2026-11-23_analysis_p2-62-claim2-needed.md; the note to writeup-lead is written.

**Money:** $9.56 on pods (E275); no pods or jobs left.

**Why finished:** whip-lead's order for today was P2.62, then stop.

---

# compare-lead digest, 2026-11-23 (afternoon)

**P1.58, due 11-24: filled. The gain-only share, as prototyped, fails its control: it reads about 0.96 on the two edits built to install a new answer, against a bar of 0.3, so it cannot tell new from reused and its copy-against-lookup reading is no call.** Finished on whip-lead's word: P2.63 stays on hold, and the next gain-only run waits on you.

## Cells
- P1.58 (gain-only share prototype): filled, 1 of 2 runs used. Page `reports/analyses/2026-11-23_analysis_p1-58-gain-share.md`; redteam-lead's verdict "neither rival; control failed" is on the page. Premise check: doubtful (listed for the gate).
- P2.63 (reuse dial on realistic tasks): open, 0 of 2 runs, on hold (whip-lead 11-23: no room for a gated run before the draft).
- P2.62: filled this morning (E275).

## What ran
E282 (cluster debug job 64538, 30 min, $0). We fitted the model closest to each fine-tune that may only turn the original model's neurons and heads, or its VPD subcomponents, up or down, and read on held-out rows how much of the fine-tune it recovers.
- The doubling and x10 edits read 0.997 and 0.981 at neurons and heads, as expected.
- The built new edits read 0.97 at neurons and heads and 0.95 at VPD, against at most 0.3: the control failed.
- E62's copy and lookup fine-tunes both read near 1, so there is no call on them.
- The head edit that gains can reproduce exactly read only 0.84, so the fit itself falls short. Several fits at VPD ended further from the fine-tune than the original model is.
- Held-out rows reused the fitted cues; whether gains rebuild the new answer on unseen cues was not tested.

## Waiting on you
Ask 31, card `reports/decisions/2026-11-23_decision_p1-58-gain-share-next.md` (in the log, v468): stop the gain-only line here, or one more run? Recommended: fit the gains on some cues or keys and read on unseen ones, which tells "gains build the new behaviour" from "gains memorise the fitted answers". Second choice: count how many gains a fit needs. Either is cluster $0, about 30-60 min. Whip-lead will raise it with you.

## Spend, friction
$0 this afternoon; no pods. Friction 1034: the run's 30-minute time limit equalled its run time, so Slurm marked a finished job as timed out.
