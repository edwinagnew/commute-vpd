# Do fine-tunes that imitate the four built pairs read like the built pairs?

*2026-11-09, cell P1.4 of the table of work, run only after R1 passes. Nothing here has run and no verdict is given.*

**In plain words.** For each of R1's four pairs, three seeds of an ordinary fine-tune of all weights of the 67M are trained to reproduce that pair's output. Each fine-tune is then read with R1's two searches and its reversion and setter mismatch. A fine-tune that matches a pair's outputs should have the pair's found shares. So this run asks whether the two numbers that see the fine-tune's own route, the reversion and the mismatch, differ from the built pair's.

## What is run

Twelve fine-tunes: the pairs Always on, Cue, XOR and Table, and seeds 0, 1, 2. The seed sets the batch draws only. Each starts from the 67M and trains every weight, not only the 24 decomposed matrices, because nothing here reads a decomposition. The loss is KL from the pair's model to the fine-tune at the last position, on R1's fitting windows for that pair. The pair's model is the 67M with its trained piece, rebuilt from R1's `pieces.pt`.

**Recipe and its source.** Adam at a peak rate of 3e-5, betas 0.9 and 0.999, no decay, gradient clip 1, 300 steps: the defaults of `finetune.ArmAConfig`. Two departures. A linear warm-up over the first 30 steps and a cosine decay after it come from `qwen_sft.lr_at`, because a fine-tune without a warm-up takes one huge first step, the lesson of 2026-10-28. The batch is 1,024 windows, one position each, which is R1's search batch; arm A reads about 1,300 positions per step.

Each fine-tune is read as R1 reads a pair, by the code moved to `commute.variable_read`: G at k = 4, four sites, three starts; H at five sites; reversion and setter mismatch; all on R1's read half. Also its distance to the pair's model and its loss curve. The gap is the mean total variation between the fine-tune and the 67M, with R1's floor of 0.02.

Filed after R1 reads, by the rule on the plan page: the cut-offs and the predictions, in `results/predictions.json`. The run refuses to start without that file committed. Its keys: `thresholds.floor_gap`, `first_steps_share`, `xor_kl_ratio`, `gap_replay_tol`, `replay_tol`; `cutoffs.G`, `cutoffs.H` for the figure.

## Expected values

| Reading | At the control, a fine-tune of zero steps, which is the 67M itself | At the do-nothing reference | Fine-tune that imitates |
|---|---|---|---|
| Gap | Float floor, under 0.02, so no reading | Zero | Near the pair's gap: 0.147, 0.072, 0.073, 0.554 in R1 |
| Distance to the pair | The pair's gap | The pair's gap | Far below that gap |
| G and H | Not read | Not read | Filed after R1 |

The control runs first and alone, and also checks the rebuilt pairs read R1's gaps and that the moved code repeats R1's G of 0.9225 and H of 0.976 on Always on at its best site.

## Early checks, in the code

- Stage 0 stops the job if the zero-step fine-tune is off the float floor, or reads above the 0.02 floor, or the moved code misses R1's numbers.
- After 10 training steps, the distance from the 67M in KL must stay under the filed share of the pair's own KL from the 67M. I propose 0.25, the first-steps lesson: the first steps cannot do most of the fine-tune. A miss stops the job.
- XOR learnability, after step 300, about one minute in: the KL from the pair's model to the fine-tune must be at most the filed share of the KL from the pair to the 67M with half the XOR's push. I propose 0.9. Half a push is the best a model that reads no XOR can do, and the 67M itself sits at about two times that reference. A miss makes the row "no reading" and the job moves on.

## Time and memory per stage, from R1's logs

| Stage | Time on an L40 | Memory |
|---|---|---|
| Job set-up | 25 s, model and rows | Host 6 GB peak, as R1 |
| Control, once | About 6 min: four gap reads at 17 s, one G at 115 s, one H at 140 s | Card about 35 GB at the `final` site |
| Fine-tune | About 1 min for 300 steps, estimated from R1's search step of 0.24 s | Adds 0.3 GB of weights and Adam moments |
| Reads | 25 s | |
| G, 12 searches | 23 min at 115 s each | As R1 |
| H, ridge plus five sites | 10 min: ridge 40 s, sites 100 to 137 s | As R1, about 35 GB of the card's 44 GB |

One item takes about 35 minutes on an L40, or a few minutes if the XOR is not learned. Twelve items are about 7 GPU hours in the worst case. In debug jobs of 105 minutes, three items fit per job, so four jobs at most; two cards in parallel finish in about 4 hours, four cards in about 2. The whole run fits under 6 hours of wall clock, so no cut is proposed. On an H100, no timing for this code exists; a guess is about a quarter faster, since its double-precision solves are faster and its forwards similar.

Cost: $0 on the cluster's debug queue, on L40 cards. The saved weights are 268 MB per item, 3.2 GB in all, in the gitignored `results/large/`.

## Not run

Nothing above has run, on any card or at any size; the code has not been run on a GPU or loaded a model. The training time is an estimate from a step of the search, and all three filed bars are my proposals, not measured.
