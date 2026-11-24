# P2.70 follow-up: refit latent scaling at the paper's token count, literal App. E.2 filter on top

*2026-11-23, compare-lead. Scoping only: no code, no run. Asked by whip-lead; rule settled by redteam-lead's verdict.
Source page: `reports/analyses/2026-11-23_analysis_p2-70-negbeta-rescore.md`. Needs a run count and an E-code from the
lead before any job.*

**What and why.** Our latent-scaling coefficients were fitted on 100,000 tokens per crosscoder, against Minder et al.'s
50M tokens (arXiv 2504.02922 v4, paper.txt line 401). At that size the sign of a base coefficient near zero is noisy, and
the literal App. E.2 rule, which drops any latent with a negative base coefficient, keeps or drops the planted latents on
that sign. The rule stays as published, with no floor. The fix removes the departure: refit the coefficients on up to
50M tokens and apply the literal rule to the refit.

**Which crosscoders.** Only crosscoders saved with their encoder can be refitted: 31 files, about 4.5 GB, under
`~/commute-data/results-large/`, each about 150 MB in bfloat16.
- E235, `crosscoder_proper/run1/`: 23, namely heads at blocks 2 and 3, Gaussian, pair sides, model against itself, k sweep.
- E243, `crosscoder_proper/run2/`: 6, the ladder rungs and the model against itself.
- E212, `realistic_tasks/run3/`: 2, the real-text plant and the Gaussian.
- Not refittable: E203 keeps decoders only, and E244 and E203's other sides keep coefficients only. Their dropped
  latents move no reading by more than 0.022, so no claim rests on them.

**What a fit needs.** Per crosscoder: both models' activations at its site, from the run's own stream, plus the encoder
and decoder. The stream code is `experiments/crosscoder_proper/run.py` `Job.stream` and `Job.pair`, or
`experiments/realistic_tasks/run3.py` for E212. It rebuilds the grafted head, the planted direction or the fine-tune.
The fit is our closed form, `src/commute/crosscoder.py` `betas`, a port of diffing-toolkit@617d926
`closed_form.py:40-168`. Its numerators and denominators are sums over tokens. One code change follows: `betas` returns
the sums so chunks accumulate. A refit script reads each saved crosscoder and writes coefficients only.

**Text.** The refit tokens are the run's own training stream. Each crosscoder trained on 100M distinct paired tokens per
epoch, so 50M fresh tokens past the first 100,000 exist. E235 and E243 mix Pile with grafted-head or task windows. E212's
task half is real text drawn up to 100 times a pass, as in training. Fits A and B use disjoint halves of 25M tokens each.
Because the fit is a ratio of sums, A plus B is the 50M fit.

**Cost per token.** Measured in `experiments/crosscoder_proper/results/run1/timing.json` on one H100:
- building one 1.05M-token chunk takes 2.6 s;
- one training step on 2,048 tokens takes 11.6 ms.

The fit's crosscoder pass is taken as one training forward, a third of a step. That is assumed, not measured. One 50M fit
is then about 2.2 min of stream, 1.6 min of fit and 1.5 min of setup, about 6 min on an H100. The L40 is taken as 4 times
slower, from memory bandwidth, 864 GB/s against 3.35 TB/s, also not measured. That gives about 24 min. The rehearsal
measures both factors.

**Memory.** One chunk is about 6.4 GB of paired activations in float32, and the crosscoder takes under 1 GB in float32.
E235's training peak was 20.9 GB, so an L40's 48 GB is enough. Host RAM: `--mem 32G`.

| stage | what | where | time | cost |
|---|---|---|---|---|
| 0 | rebuild each control's 100,000 latent-scaling tokens; reproduce the saved coefficients; token bootstrap | cluster debug L40, 3 jobs | about 1 h | $0 |
| 1 | the 9 planted controls: block-2 heads ×3, block-3 heads ×3, rung 4 ×2, rung 5, real-text plant; fits A and B, 25M each | cluster debug L40, 18 jobs of about 12 min | about 3.6 h serial, 1.8 h on 2 cards | $0 |
| 1, fallback | the same if debug does not start within 5 min | one H100 pod | about 1.4 h with upload | about $7 |
| 2 | the other 22 crosscoders, one 50M fit each | cluster debug L40, 22 jobs of about 24 min | about 9 h serial, 4.5 h on 2 cards | $0 |
| 2, fallback | the same on a pod | one H100 pod | about 2.5 h | about $12 |

Pod price: E212's two H100 pods cost $8.03 for about 50 minutes each, so about $4.8 per pod-hour, set-up included. The
upload is 1.35 GB for stage 1, about 2.3 min at 10 MB/s, or 4.5 GB for both stages, about 7.5 min. Downloads are
coefficient files only. Total on the cluster: $0. Total with both fallbacks: about $19. Stages 0 and 1 alone fit $15
either way. Stage 2 cannot change a quoted reading, so it waits for free cards.

**Stage 0, before any GPU spend.** The planted latents' coefficients over 100,000 tokens are not saved as activations, so
the bootstrap needs the stream rebuilt. That runs on the cluster, under 10 min per control.
- *Reproduction control first.* Refit on the same 100,000 tokens. The weights are bfloat16 copies of float32 training, so
  the floor is measured on this first item, as the coefficient gap over the shared latents. Expect a rank correlation
  near 1, and the saved fine-tune-specific set rebuilt within that floor. Stop if not.
- *Bootstrap.* Run 200 resamples of the 195 windows of 512 tokens. For each dropped latent, read the share of resamples
  with a base coefficient below zero.
  - Noise: the plant latents read between 0.2 and 0.8. They are the two largest at rung 5, rung 4's lead latent,
    block 2 seed 2's latent 4123 and the real-text latent 4899.
  - Real: they read above 0.95.
  - The block-3 latents with clearly negative ratios, median minus 0.055 for the error ratio, should read above 0.95.
- Seed spread across crosscoder seeds is dropped: different seeds learn different latents, so there is no counterpart.

**Stage 1 expected values.** If the plant latents' negative coefficients are noise:
- their base coefficient shrinks toward 0 with tokens, as one over the square root of the count, about 22 times from
  100,000 to 50M;
- fits A and B disagree in sign on about half of them.

If they are real, A and B agree in sign and the magnitude stays near the 100,000-token value. The scored reading is the
literal rule on the 50M coefficients. Under every rule, block 3 stays under the bar: restoring every near-zero dropped
latent gives 0.48, 0.48 and 0.43. So does the ladder. The open outcome is the real-text plant, 0 or 0.77.

**Done by tomorrow.** Stages 0 and 1 take about 5 h on the cluster with 2 free L40s, or about 2.5 h with the pod
fallback. One L40 node has 8 cards, about 6 in use at 2026-11-23 16:30. Stage 2 runs overnight on the cluster at $0. The
code is one `betas` change and one refit script, about 150 lines, under an hour to write.
